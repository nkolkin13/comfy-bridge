"""Exception hierarchy for comfy_bridge (§8)."""

from __future__ import annotations


class ComfyBridgeError(Exception):
    """Base for every error raised by this package."""


class BootstrapError(ComfyBridgeError):
    """Bad ComfyUI root, wrong import order, or a conflicting second start()."""


class NamespaceError(ComfyBridgeError):
    """The sys.modules footprint guard tripped (§5.1 step 11)."""

    def __init__(self, unexpected: set[str], missing: set[str]) -> None:
        parts = []
        if unexpected:
            parts.append(f"unexpected top-level ComfyUI modules: {sorted(unexpected)}")
        if missing:
            parts.append(f"expected but absent: {sorted(missing)}")
        super().__init__(
            "ComfyUI namespace footprint changed — "
            + "; ".join(parts)
            + ". This usually means the ComfyUI checkout moved to a version that "
            "imports a new root-level module. Review before widening FOOTPRINT."
        )
        self.unexpected = unexpected
        self.missing = missing


class ExtensionError(ComfyBridgeError):
    """A locally-authored node or patch could not be registered or applied."""


class CodegenError(ComfyBridgeError):
    """A workflow could not be translated to Python."""


class UnsupportedNodeError(CodegenError):
    """A node uses a feature static generation cannot express (§9)."""


class NodeExecutionError(ComfyBridgeError):
    """A node raised during execution."""

    def __init__(self, node_id: str, class_type: str, original: BaseException) -> None:
        super().__init__(f"node {node_id} ({class_type}) failed: {original!r}")
        self.node_id = node_id
        self.class_type = class_type
        self.original = original
