"""A stand-in for ComfyUI's ``server`` module.

ComfyUI's real ``server.py`` exists to run an aiohttp web application. Under
codegen we never serve anything, but three shipped modules still import it:

    comfy_extras/nodes_images.py:18        from server import PromptServer
    comfy_extras/nodes_gaussian_splat.py:20  from server import PromptServer
    comfy_api/latest/__init__.py:31        from server import PromptServer  (deferred)

Between them the entire surface they touch is:

    PromptServer.instance.send_progress_text(text, node_id)      x2
    PromptServer.instance.node_replace_manager.register(...)     x1

Importing the real module to satisfy that costs six top-level names —
``server``, ``app``, ``api_server``, ``middleware``, ``utils`` and
``comfyui_version`` — plus aiohttp and the whole web stack. Registering this
stub in ``sys.modules['server']`` before ComfyUI loads satisfies all three
importers and keeps those names out of the host namespace entirely.

``comfy.utils.ProgressBar`` is unaffected either way: it dispatches through the
module-level ``PROGRESS_BAR_HOOK`` in comfy/utils.py, which only main.py's
``hijack_progress`` ever installs. We never call it.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Callable

log = logging.getLogger("comfy_bridge")

MODULE_NAME = "server"


class _NodeReplaceManager:
    """Satisfies comfy_api/latest/__init__.py:32.

    Node replacement is a frontend affordance — it tells the UI to swap one node
    for another. Under codegen the graph is already fixed, so registrations are
    recorded and ignored.
    """

    def __init__(self) -> None:
        self.registered: list[Any] = []

    def register(self, node_replace: Any) -> None:
        self.registered.append(node_replace)


class PromptServer:
    """Minimal stand-in. Only the attributes ComfyUI actually reaches for."""

    instance: "PromptServer | None" = None

    def __init__(self) -> None:
        PromptServer.instance = self
        self.node_replace_manager = _NodeReplaceManager()
        self.progress_callback: Callable[[str, Any, Any], None] | None = None
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.last_prompt_id: str | None = None

    # --- the surface the node layer uses -------------------------------------

    def send_progress_text(self, text: str, node_id: Any = None) -> None:
        self._emit("progress_text", {"text": text, "node_id": node_id}, None)

    def send_sync(self, event: str, data: Any, sid: Any = None) -> None:
        self._emit(event, data, sid)

    # --- everything else is a deliberate no-op -------------------------------

    def queue_updated(self) -> None:
        pass

    def _emit(self, event: str, data: Any, sid: Any) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event, data, sid)
        except Exception:  # a user callback must never break a graph
            log.exception("progress_callback raised; continuing")


class BinaryEventTypes:
    """Mirrors protocol.BinaryEventTypes so `server` importers stay satisfied."""

    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    TEXT = 3


def install() -> PromptServer:
    """Register the stub as ``sys.modules['server']`` and return the instance.

    Must run before ComfyUI imports anything that does ``from server import
    PromptServer``, i.e. before ``init_extra_nodes``.
    """
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None and getattr(existing, "__comfy_bridge_stub__", False):
        return existing.PromptServer.instance  # type: ignore[union-attr]
    if existing is not None:
        # The real ComfyUI server module already loaded; stubbing now would give
        # two different PromptServer classes to different importers.
        raise RuntimeError(
            "a 'server' module is already imported; comfy_bridge cannot install "
            "its stub. Call start() before importing ComfyUI."
        )

    module = types.ModuleType(MODULE_NAME)
    module.__comfy_bridge_stub__ = True  # type: ignore[attr-defined]
    module.PromptServer = PromptServer  # type: ignore[attr-defined]
    module.BinaryEventTypes = BinaryEventTypes  # type: ignore[attr-defined]
    sys.modules[MODULE_NAME] = module
    return PromptServer()
