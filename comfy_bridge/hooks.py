"""Inference hooks that go through ComfyUI's own extension API (spec D13).

The headline finding from surveying the checkout: **most inference-level
acceleration needs no patching at all.** ComfyUI ships a wrapper/callback system
(``comfy/patcher_extension.py``) that reaches every level a zero-knowledge method
cares about, and ``ModelPatcher`` exposes it. Going through it instead of
:func:`comfy_bridge.patch_attr` buys three things that matter for benchmarking:

* **Per-clone scope.** ``model.clone()`` is cheap and shares weights, so a
  baseline and an accelerated variant can coexist as two objects in one process.
  A module-global patch cannot do that.
* **Automatic teardown.** Drop the clone and the hook is gone; there is no
  process state to leak into the next measurement.
* **Upstream keeps it working.** These are the seams ComfyUI's own features
  (context windows, hooks, TeaCache-alikes) are built on.

Where the wrappers attach, outermost first:

===================== ============================================ ============
Wrapper               Fires                                         Site
===================== ============================================ ============
``OUTER_SAMPLE``      once per sampling call                        samplers.py:1331
``SAMPLER_SAMPLE``    once, inside the KSampler                     samplers.py:1233
``PREDICT_NOISE``     once per denoise step                         samplers.py:1214
``CALC_COND_BATCH``   once per cond/uncond batch (2x/step with CFG) samplers.py:217
``APPLY_MODEL``       once per model call — every architecture      model_base.py:199
``DIFFUSION_MODEL``   the transformer forward — per architecture    e.g. ldm/minimax/model.py:502
===================== ============================================ ============

``APPLY_MODEL`` is the safe universal choice; ``DIFFUSION_MODEL`` is per-model
but is the one that brackets exactly the transformer, so it is the honest place
to measure and the natural place to hang step-caching. MiniMax H3 supports it.

A wrapper is called as ``wrapper(executor, *args, **kwargs)`` and must call
``executor(*args, **kwargs)`` to continue the chain
(``patcher_extension.py:114``). Skipping that call is how a caching method
returns a stale result without running the model — which is the whole point.

Below the sampler, two more seams exist and are also patch-free:

* :func:`override_attention` intercepts every attention call through
  ``transformer_options["optimized_attention_override"]``
  (``ldm/modules/attention.py:158``) — the hook for sliding-window or quantized
  attention, with the original passed in so you can fall back per call.
* :func:`patch_object` reaches any attribute of the loaded model by dotted path
  via ``ModelPatcher.add_object_patch``, backed up at load and restored at unload
  (``model_patcher.py:1107-1161``) — the hook for swapping a ``Linear`` for an
  INT8 one, or a block's ``forward``.

Everything here imports ComfyUI lazily, so this module is safe to import before
``start()``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .errors import ExtensionError

log = logging.getLogger("comfy_bridge.hooks")

__all__ = [
    "APPLY_MODEL",
    "CALC_COND_BATCH",
    "DIFFUSION_MODEL",
    "OUTER_SAMPLE",
    "PREDICT_NOISE",
    "SAMPLER_SAMPLE",
    "add_callback",
    "add_wrapper",
    "attention_functions",
    "override_attention",
    "patch_object",
    "register_attention",
]

# Mirrored from comfy/patcher_extension.py so callers need not import comfy.
OUTER_SAMPLE = "outer_sample"
SAMPLER_SAMPLE = "sampler_sample"
PREDICT_NOISE = "predict_noise"
CALC_COND_BATCH = "calc_cond_batch"
APPLY_MODEL = "apply_model"
DIFFUSION_MODEL = "diffusion_model"

_WRAPPER_TYPES = frozenset(
    {
        OUTER_SAMPLE,
        SAMPLER_SAMPLE,
        PREDICT_NOISE,
        CALC_COND_BATCH,
        APPLY_MODEL,
        DIFFUSION_MODEL,
    }
)


def _as_patcher(model: Any, what: str) -> Any:
    """Accept a ModelPatcher, or anything wrapping one (comfy.sd.VAE, CLIP)."""
    if hasattr(model, "add_wrapper") and hasattr(model, "clone"):
        return model
    inner = getattr(model, "patcher", None)
    if inner is not None and hasattr(inner, "add_wrapper"):
        return inner
    raise ExtensionError(
        f"{what} needs a ModelPatcher (or an object with .patcher), got "
        f"{type(model).__name__}"
    )


def add_wrapper(
    model: Any,
    wrapper_type: str,
    fn: Callable[..., Any],
    *,
    key: str | None = None,
    clone: bool = True,
) -> Any:
    """Attach ``fn`` at ``wrapper_type`` and return the model to sample with.

    Clones by default, so the model you passed in is untouched and stays usable
    as the benchmark baseline::

        def timed(executor, *args, **kwargs):
            t0 = time.perf_counter()
            out = executor(*args, **kwargs)
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - t0)
            return out

        fast = hooks.add_wrapper(model, hooks.DIFFUSION_MODEL, timed)

    ``key`` groups wrappers so ``ModelPatcher.remove_wrappers_with_key`` can drop
    them again; without one they can only be removed by discarding the clone.

    The wrappers reach the sampler because ``sampler_helpers.py:224`` merges
    ``ModelPatcher.wrappers`` into ``model_options`` at prepare-sampling time —
    setting them after this point in a run has no effect.
    """
    if wrapper_type not in _WRAPPER_TYPES:
        raise ExtensionError(
            f"unknown wrapper type {wrapper_type!r}; expected one of "
            f"{sorted(_WRAPPER_TYPES)}"
        )
    if not callable(fn):
        raise ExtensionError(f"wrapper must be callable, got {type(fn).__name__}")

    patcher = _as_patcher(model, "add_wrapper")
    target = patcher.clone() if clone else patcher
    if key is None:
        target.add_wrapper(wrapper_type, fn)
    else:
        target.add_wrapper_with_key(wrapper_type, key, fn)
    return target


def add_callback(
    model: Any,
    call_type: str,
    fn: Callable[..., Any],
    *,
    key: str | None = None,
    clone: bool = True,
) -> Any:
    """Attach a ModelPatcher lifecycle callback (``CallbacksMP`` in upstream).

    Unlike wrappers these observe rather than intercept — ``on_load_after``,
    ``on_cleanup`` and friends. Useful for measuring load/offload cost, which on
    a 24 GB card is a real share of wall-clock for video models.
    """
    patcher = _as_patcher(model, "add_callback")
    target = patcher.clone() if clone else patcher
    if key is None:
        target.add_callback(call_type, fn)
    else:
        target.add_callback_with_key(call_type, key, fn)
    return target


def override_attention(
    model: Any, fn: Callable[..., Any], *, clone: bool = True
) -> Any:
    """Route every attention call through ``fn``.

    ``fn(original, q, k, v, heads, **kwargs)`` — the dispatched implementation
    arrives first, so falling back is ``return original(q, k, v, heads, **kwargs)``
    and a sliding-window variant can decide per call based on sequence length
    alone, with no calibration.

    Applies to every attention site the model routes through ``wrap_attn``
    (``ldm/modules/attention.py:147``), which is all of them in shipped
    architectures.
    """
    if not callable(fn):
        raise ExtensionError(f"override must be callable, got {type(fn).__name__}")
    patcher = _as_patcher(model, "override_attention")
    target = patcher.clone() if clone else patcher
    # deepcopy_list_dict on clone() means this dict is ours, not the parent's
    # (model_patcher.py:453).
    transformer_options = target.model_options.setdefault("transformer_options", {})
    transformer_options["optimized_attention_override"] = fn
    return target


def patch_object(model: Any, path: str, obj: Any, *, clone: bool = True) -> Any:
    """Replace an attribute of the loaded model by dotted path.

    The sanctioned architecture patch::

        fast = hooks.patch_object(model, "diffusion_model.blocks.0.attn.forward", mine)

    ComfyUI applies these at load and restores the originals at unload
    (``model_patcher.py:1107-1161``), so the weights on disk and the shared
    module tree both stay clean. Paths resolve through ``comfy.utils.set_attr``,
    so any attribute reachable by attribute access works, including a whole
    ``nn.Module`` or a bound method.

    Note this mutates the *shared* underlying model while loaded — the clone
    scopes which patcher applies it, not which memory it touches. Two clones with
    conflicting object patches must not be resident at once.
    """
    if not isinstance(path, str) or not path:
        raise ExtensionError(f"path must be a non-empty dotted string, got {path!r}")
    patcher = _as_patcher(model, "patch_object")
    target = patcher.clone() if clone else patcher
    target.add_object_patch(path, obj)
    return target


def register_attention(name: str, fn: Callable[..., Any]) -> None:
    """Add a named attention implementation to ComfyUI's registry.

    Upstream refuses to replace an existing name and only logs a warning
    (``ldm/modules/attention.py:53-58``), so this raises instead — silently
    benchmarking the wrong kernel is the failure mode worth preventing.
    """
    from comfy.ldm.modules.attention import (
        REGISTERED_ATTENTION_FUNCTIONS,
        register_attention_function,
    )

    if name in REGISTERED_ATTENTION_FUNCTIONS:
        raise ExtensionError(
            f"attention function {name!r} is already registered "
            f"({REGISTERED_ATTENTION_FUNCTIONS[name]!r}); upstream will not "
            "replace it. Pick another name."
        )
    register_attention_function(name, fn)


def attention_functions() -> dict[str, Callable[..., Any]]:
    """Attention implementations currently registered, by name."""
    from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS

    return dict(REGISTERED_ATTENTION_FUNCTIONS)
