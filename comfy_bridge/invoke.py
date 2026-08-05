"""Call ComfyUI's shipped nodes directly (§7).

Replaces the machinery in ``execution.py`` that generated code would otherwise
have to reimplement inline. Four things matter, each learned the hard way while
getting a real MiniMax H3 generation to run:

1. **V1 vs V3 dispatch.** ``cls.FUNCTION`` is an instance method on V1 nodes and
   ``EXECUTE_NORMALIZED`` / ``EXECUTE_NORMALIZED_ASYNC`` on V3 ones
   (``comfy_api/latest/_io.py:1929``). ComfyUI normalises V3 returns for us, so
   the shim only unwraps ``NodeOutput.result``.
2. **V3 needs PREPARE_CLASS_CLONE** (C31). The raw class has ``hidden = None``,
   so any node reading ``cls.hidden.*`` raises AttributeError.
3. **Nothing is async** (C18). The 243 ``async def`` across comfy_extras are
   helpers; no shipped node's entrypoint is a coroutine function. The shim is
   synchronous throughout and *refuses* a coroutine rather than driving it, so
   if upstream ever ships one it fails loudly instead of passing a coroutine
   object downstream as though it were a tensor.
4. **No autograd** (C29). ``execution.py:751`` wraps the whole prompt in
   ``torch.inference_mode()``. Without an equivalent, every call retains its
   graph and VRAM climbs until it OOMs — measured at +7.2 GB per VAE tile.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable

from .errors import NodeExecutionError, UnsupportedNodeError

log = logging.getLogger("comfy_bridge.invoke")

__all__ = ["invoke", "node_class", "list_io_guard", "add_observer"]

_V3_PREFIX = "EXECUTE_NORMALIZED"

#: (callback, cuda_sync) for everything watching node calls. See add_observer.
_OBSERVERS: list[tuple[Callable[[str, float], None], bool]] = []
_CUDA_SYNC = False


def add_observer(
    fn: Callable[[str, float], None], *, cuda_sync: bool = False
) -> Callable[[], None]:
    """Watch every node call; returns a callable that removes the observer.

    ``invoke()`` is the single chokepoint every generated graph passes through,
    which makes it the one place per-node cost can be measured without touching
    generated code. The callback receives ``(class_type, seconds)`` after a
    successful call; exceptions propagate to the caller unobserved.

    ``cuda_sync=True`` makes *all* observed calls synchronise the current CUDA
    device before the clock stops. Without it the number is launch time, not
    execution time, and a node that only enqueues work looks free. With it, some
    overlap the real graph would have had is serialised away. Both are wrong in
    different directions; sync is the right default for attributing cost to
    nodes, and off is right when measuring end-to-end wall clock.
    """
    entry = (fn, cuda_sync)
    _OBSERVERS.append(entry)
    _refresh_sync()

    def remove() -> None:
        try:
            _OBSERVERS.remove(entry)
        except ValueError:
            return
        _refresh_sync()

    return remove


def _refresh_sync() -> None:
    global _CUDA_SYNC
    _CUDA_SYNC = any(sync for _, sync in _OBSERVERS)


def _notify(class_type: str, seconds: float) -> None:
    for fn, _ in list(_OBSERVERS):
        try:
            fn(class_type, seconds)
        except Exception:  # an instrument must never break the thing it measures
            log.exception("invoke observer raised; continuing")


def node_class(class_type: str):
    """Look up a shipped node class by its ComfyUI class_type."""
    from .bootstrap import get_runtime

    runtime = get_runtime()
    try:
        return runtime.nodes[class_type]
    except KeyError:
        raise UnsupportedNodeError(
            f"no shipped node named {class_type!r}. Custom nodes are out of "
            f"scope (D2); {len(runtime.nodes)} shipped nodes are available."
        ) from None


def _uses_list_io(value: Any) -> bool:
    """True only if list mapping is actually switched on.

    Careful with truthiness here: every V3 node exposes ``OUTPUT_IS_LIST`` as a
    per-output sequence, typically ``[False]``. A non-empty list is truthy, so a
    bare ``if value:`` reports that essentially every shipped node uses list I/O.
    Only the *contents* matter.
    """
    if isinstance(value, (list, tuple)):
        return any(bool(v) for v in value)
    return bool(value)


def list_io_guard(cls: Any, class_type: str) -> None:
    """Refuse nodes needing list mapping (C17).

    No shipped node switches these on, so the whole ``_map_node_over_list``
    branch of execution.py is unnecessary — but fail loudly rather than be
    silently wrong if that stops being true upstream.
    """
    for attr in ("INPUT_IS_LIST", "OUTPUT_IS_LIST"):
        value = getattr(cls, attr, False)
        if _uses_list_io(value):
            raise UnsupportedNodeError(
                f"{class_type} declares {attr}={value!r}. comfy_bridge does not "
                "implement list mapping (C17 measured no shipped node "
                "using it). This node needs the full execution.py path."
            )


def _is_execution_blocker(value: Any) -> bool:
    """True for comfy_execution.graph_utils.ExecutionBlocker instances.

    Resolved by name rather than isinstance so this stays a cheap check that
    does not depend on import order; there is exactly one class with this name
    in the checkout (``comfy_execution/graph_utils.py:140``).
    """
    return type(value).__name__ == "ExecutionBlocker"


def invoke(class_type: str, node_id: str | None = None, /, **kwargs) -> tuple:
    """Call a shipped node and return its outputs as a tuple.

    Both leading arguments are positional-only so that node inputs literally
    named ``class_type`` or ``node_id`` cannot collide with them.

        (model, clip, vae) = invoke("CheckpointLoaderSimple", ckpt_name="v1-5.safetensors")
        (latent,) = invoke("KSampler", "3", model=model, ...)

    ``node_id`` is the originating workflow node id, which generated code passes
    through so a failure names *which* CLIPTextEncode of six broke rather than
    just the class. It defaults to the class_type when omitted.

    Runs under ``torch.no_grad()``. Generated modules additionally carry
    ``@torch.inference_mode()`` on ``run_graph()``; this is the inner guard so a
    caller poking a single node is safe too.
    """
    import torch

    cls = node_class(class_type)
    list_io_guard(cls, class_type)
    where = node_id if node_id is not None else class_type

    fn_name = cls.FUNCTION
    is_v3 = fn_name.startswith(_V3_PREFIX)

    if is_v3:
        # Attaches a HiddenHolder and keeps per-call state off the shared class.
        cls = cls.PREPARE_CLASS_CLONE(None)
        target = getattr(cls, fn_name)
    else:
        target = getattr(cls(), fn_name)

    observed = bool(_OBSERVERS)
    started = time.perf_counter() if observed else 0.0
    try:
        with torch.no_grad():
            out = target(**kwargs)
    except Exception as exc:
        raise NodeExecutionError(where, class_type, exc) from exc

    if inspect.isawaitable(out):
        # C18 measured zero async entrypoints across all 591 shipped nodes, so
        # the shim is synchronous. If upstream adds one, refuse it here — the
        # alternative is a coroutine object flowing downstream as if it were a
        # tensor and failing somewhere unrelated.
        out.close()
        raise UnsupportedNodeError(
            f"{class_type} returned a coroutine. comfy_bridge is synchronous "
            "(C18 measured no shipped node with an async entrypoint); if "
            "upstream has added one, invoke() needs an await path again."
        )

    if observed:
        if _CUDA_SYNC and torch.cuda.is_available():
            torch.cuda.synchronize()
        _notify(class_type, time.perf_counter() - started)

    if not is_v3:
        result = out if isinstance(out, tuple) else (out,)
        # V1 nodes have no NodeOutput to carry a blocker, so it arrives inline.
        # No shipped node does this, but a locally-authored V1 node can.
        for value in result:
            if _is_execution_blocker(value):
                raise UnsupportedNodeError(
                    f"{class_type} returned an ExecutionBlocker. Node-driven "
                    "control flow is not expressible in generated code (§9)."
                )
        return result

    if getattr(out, "block_execution", None) is not None:
        raise UnsupportedNodeError(
            f"{class_type} returned an ExecutionBlocker "
            f"({out.block_execution!r}). Node-driven control flow is not "
            "expressible in generated code (§9)."
        )
    if getattr(out, "expand", None) is not None:
        raise UnsupportedNodeError(
            f"{class_type} returned an expanded subgraph. Dynamic graph "
            "expansion cannot be statically generated (§9, C20)."
        )
    return tuple(out.result or ())
