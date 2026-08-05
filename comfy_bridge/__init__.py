"""comfy_bridge — run ComfyUI's shipped nodes in-process, without the web UI.

Importing this package is cheap and side-effect free: it does not import torch
or ComfyUI. Everything happens in :func:`start`, which must be called before the
process imports torch if you intend to pin a device.

    import comfy_bridge

    rt = comfy_bridge.start(device=0)
    print(len(rt.nodes), "nodes available")
"""

from __future__ import annotations

from .bootstrap import (
    DEFAULT_COMFY_ROOT,
    FOOTPRINT,
    FOOTPRINT_REAL_SERVER,
    VALIDATED_COMFY_VERSION,
    VRAM_MODES,
    Runtime,
    get_runtime,
    is_started,
    start,
)
from . import bench, extend, hooks
from .errors import (
    BootstrapError,
    CodegenError,
    ComfyBridgeError,
    ExtensionError,
    NamespaceError,
    NodeExecutionError,
    UnsupportedNodeError,
)
from .extend import (
    Patch,
    PatchSet,
    active_patches,
    patch_attr,
    register_node,
    registered_nodes,
    revert_all_patches,
    unregister_node,
)
from .invoke import add_observer, invoke, node_class
from .memory import free_memory, load_to_gpu, offload

__all__ = [
    "DEFAULT_COMFY_ROOT",
    "FOOTPRINT",
    "FOOTPRINT_REAL_SERVER",
    "VALIDATED_COMFY_VERSION",
    "VRAM_MODES",
    "BootstrapError",
    "CodegenError",
    "ComfyBridgeError",
    "ExtensionError",
    "NamespaceError",
    "NodeExecutionError",
    "Patch",
    "PatchSet",
    "Runtime",
    "UnsupportedNodeError",
    "active_patches",
    "add_observer",
    "bench",
    "extend",
    "free_memory",
    "get_runtime",
    "hooks",
    "invoke",
    "is_started",
    "load_to_gpu",
    "node_class",
    "offload",
    "patch_attr",
    "register_node",
    "registered_nodes",
    "revert_all_patches",
    "start",
    "unregister_node",
]

__version__ = "0.1.0"
