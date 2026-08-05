"""Manual model/VRAM control for generated scripts (§5.3).

Dropping ``PromptExecutor`` also drops ``prompt_worker``'s housekeeping — its
timed ``gc.collect()``, cache resets and ``unload_all_models()``. Straight-line
generated code does not need that on a timer, but it does need the operations
available, so they are exposed here rather than automated.

The subtlety this module exists to hide: **ComfyUI has two different wrappers**
and only one of them is a ``ModelPatcher``. Loader nodes return

    UNETLoader / CLIPLoader  -> ModelPatcher (or ModelPatcherDynamic)
    VAELoader                -> comfy.sd.VAE, whose patcher is at `.patcher`

A naive ``hasattr(m, "model")`` check therefore silently skips every VAE — which
is exactly the bug the first draft of the MiniMax example shipped with.
"""

from __future__ import annotations

import gc
from typing import Any

from .errors import ExtensionError

__all__ = ["free_memory", "load_to_gpu", "offload", "as_patcher", "require_patcher"]


def as_patcher(obj: Any):
    """Return the ModelPatcher behind a loader output, or None.

    Accepts a ModelPatcher directly, a ``comfy.sd.VAE`` (``.patcher``), or a CLIP
    wrapper (also ``.patcher``). This is the single resolver for the whole
    package — :mod:`comfy_bridge.hooks` imports it rather than keeping its own,
    because two implementations of this check that disagreed about precedence is
    exactly how the VAE bug below gets reintroduced.

    ``.patcher`` is tried first and it is safe to do so: ModelPatcher has no
    ``.patcher`` attribute of its own (``comfy/model_patcher.py``), while VAE
    (``comfy/sd.py:1018``) and CLIP (``comfy/sd.py:263``) both set one.

    Two duck-types follow, because the two call sites this merged care about
    different halves of ModelPatcher and a real one satisfies both: the memory
    functions want the weights (``model`` + ``model_size``), the hooks want the
    extension seams (``add_wrapper`` + ``clone``).
    """
    patcher = getattr(obj, "patcher", None)
    if patcher is not None:
        return patcher
    # ModelPatcher itself exposes .model; VAE does not.
    if hasattr(obj, "model") and hasattr(obj, "model_size"):
        return obj
    if hasattr(obj, "add_wrapper") and hasattr(obj, "clone"):
        return obj
    return None


def require_patcher(obj: Any, what: str):
    """Like :func:`as_patcher`, but raises instead of returning None.

    For call sites where a non-model is a programming error rather than
    something to skip over.
    """
    patcher = as_patcher(obj)
    if patcher is None:
        raise ExtensionError(
            f"{what} needs a ModelPatcher, or something wrapping one such as a "
            f"VAE or CLIP (.patcher) — got {type(obj).__name__}"
        )
    return patcher


def offload(*models: Any, collect: bool = True) -> int:
    """Evict specific models from VRAM. Returns how many were unloaded.

    Drop your Python references afterwards too — an offloaded ModelPatcher keeps
    its weights in *host* RAM, which is the binding constraint when a 32B text
    encoder shares a 31GB machine with a diffusion model.

        images = invoke("VAEDecode", samples=lat, vae=vae)
        offload(vae)
        del vae

    ``None`` is skipped, so ``offload(maybe_vae)`` is fine. Anything else that
    is not a model raises: an offload that quietly did nothing would show up
    later as unexplained VRAM, which is far harder to diagnose than a traceback.
    """
    import comfy.model_management as mm

    unloaded = 0
    for obj in models:
        if obj is None:
            continue
        patcher = require_patcher(obj, "offload()")
        mm.unload_model_and_clones(patcher)
        unloaded += 1

    if collect:
        gc.collect()
        mm.soft_empty_cache()
    return unloaded


def free_memory(*, unload_models: bool = True) -> None:
    """Blunt reclaim: unload everything and empty the caches."""
    import comfy.model_management as mm

    if unload_models:
        mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()


def load_to_gpu(*models: Any, memory_required: int = 0) -> None:
    """Pre-load models onto the compute device.

    Rarely needed — nodes load on demand — but useful to front-load a stall so it
    happens at a predictable point rather than mid-graph.
    """
    import comfy.model_management as mm

    patchers = [
        require_patcher(m, "load_to_gpu()") for m in models if m is not None
    ]
    if patchers:
        mm.load_models_gpu(patchers, memory_required=memory_required)
