"""Manual model/VRAM control for generated scripts (spec §5.3).

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

__all__ = ["free_memory", "load_to_gpu", "offload", "as_patcher"]


def as_patcher(obj: Any):
    """Return the ModelPatcher behind a loader output, or None.

    Accepts a ModelPatcher directly, a ``comfy.sd.VAE`` (``.patcher``), or a CLIP
    wrapper (also ``.patcher``).
    """
    patcher = getattr(obj, "patcher", None)
    if patcher is not None:
        return patcher
    # ModelPatcher itself exposes .model; VAE does not.
    if hasattr(obj, "model") and hasattr(obj, "model_size"):
        return obj
    return None


def offload(*models: Any, collect: bool = True) -> int:
    """Evict specific models from VRAM. Returns how many were unloaded.

    Drop your Python references afterwards too — an offloaded ModelPatcher keeps
    its weights in *host* RAM, which is the binding constraint when a 32B text
    encoder shares a 31GB machine with a diffusion model.

        images = invoke("VAEDecode", samples=lat, vae=vae)
        offload(vae)
        del vae
    """
    import comfy.model_management as mm

    unloaded = 0
    for obj in models:
        if obj is None:
            continue
        patcher = as_patcher(obj)
        if patcher is None:
            continue
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

    patchers = [p for p in (as_patcher(m) for m in models) if p is not None]
    if patchers:
        mm.load_models_gpu(patchers, memory_required=memory_required)
