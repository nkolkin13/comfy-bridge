"""Call ComfyUI's shipped nodes directly (spec §7).

Replaces the machinery in ``execution.py`` that generated code would otherwise
have to reimplement inline. Four things matter, each learned the hard way while
getting a real MiniMax H3 generation to run:

1. **V1 vs V3 dispatch.** ``cls.FUNCTION`` is an instance method on V1 nodes and
   ``EXECUTE_NORMALIZED`` / ``EXECUTE_NORMALIZED_ASYNC`` on V3 ones
   (``comfy_api/latest/_io.py:1929``). ComfyUI normalises V3 returns for us, so
   the shim only unwraps ``NodeOutput.result``.
2. **V3 needs PREPARE_CLASS_CLONE** (C31). The raw class has ``hidden = None``,
   so any node reading ``cls.hidden.*`` raises AttributeError.
3. **Async is common, not exceptional** (C18) — 243 ``async def`` across
   comfy_extras. Coroutines are driven on the bridge's private loop so callers
   stay synchronous.
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
            f"scope (spec D2); {len(runtime.nodes)} shipped nodes are available."
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
    """Refuse nodes needing list mapping (spec C17).

    No shipped node switches these on, so the whole ``_map_node_over_list``
    branch of execution.py is unnecessary — but fail loudly rather than be
    silently wrong if that stops being true upstream.
    """
    for attr in ("INPUT_IS_LIST", "OUTPUT_IS_LIST"):
        value = getattr(cls, attr, False)
        if _uses_list_io(value):
            raise UnsupportedNodeError(
                f"{class_type} declares {attr}={value!r}. comfy_bridge does not "
                "implement list mapping (spec C17 measured no shipped node "
                "using it). This node needs the full execution.py path."
            )


def invoke(class_type: str, /, **kwargs) -> tuple:
    """Call a shipped node and return its outputs as a tuple.

    Positional-only first argument so a node input literally named
    ``class_type`` cannot collide with it.

        (model, clip, vae) = invoke("CheckpointLoaderSimple", ckpt_name="v1-5.safetensors")

    Runs under ``torch.no_grad()``. Generated modules additionally carry
    ``@torch.inference_mode()`` on ``run_graph()``; this is the inner guard so a
    caller poking a single node is safe too.
    """
    import torch

    from .bootstrap import get_runtime

    runtime = get_runtime()
    cls = node_class(class_type)
    list_io_guard(cls, class_type)

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
            if inspect.isawaitable(out):
                out = runtime.loop.run_until_complete(out)
    except Exception as exc:
        raise NodeExecutionError(class_type, class_type, exc) from exc

    if observed:
        if _CUDA_SYNC and torch.cuda.is_available():
            torch.cuda.synchronize()
        _notify(class_type, time.perf_counter() - started)

    if not is_v3:
        return out if isinstance(out, tuple) else (out,)

    if getattr(out, "block_execution", None) is not None:
        raise UnsupportedNodeError(
            f"{class_type} returned an ExecutionBlocker "
            f"({out.block_execution!r}). Node-driven control flow is not "
            "expressible in generated code (spec §9)."
        )
    if getattr(out, "expand", None) is not None:
        raise UnsupportedNodeError(
            f"{class_type} returned an expanded subgraph. Dynamic graph "
            "expansion cannot be statically generated (spec §9, C20)."
        )
    return tuple(out.result or ())
