"""M0 tests — the cheap guards from §11.

These are the tests that make an upstream ComfyUI bump fail loudly at CI rather
than silently at runtime.
"""

from __future__ import annotations

import socket
import subprocess
import sys

import pytest

import comfy_bridge


@pytest.fixture(scope="session")
def runtime():
    """Start ComfyUI once for the whole session — start() is process-global.

    dynamic_vram=True because torch in this env is 2.6, below the >= 2.8 gate
    upstream applies unless DynamicVRAM is explicitly requested (main.py:252).
    """
    return comfy_bridge.start(dynamic_vram=True)


def test_import_does_not_load_torch():
    """Spec C4: device env vars only work if torch is not yet imported.

    Runs in a subprocess because the session fixture imports torch.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, comfy_bridge; "
            "assert 'torch' not in sys.modules, 'comfy_bridge imported torch'; "
            "assert 'nodes' not in sys.modules, 'comfy_bridge imported ComfyUI'; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_start_loads_shipped_nodes(runtime):
    assert len(runtime.nodes) > 500, f"only {len(runtime.nodes)} nodes loaded"
    # A V1 node, a V3 node, and a sink node should all be present.
    for expected in ("CheckpointLoaderSimple", "KSampler", "SaveImage", "VAEDecode"):
        assert expected in runtime.nodes, f"{expected} missing from NODE_CLASS_MAPPINGS"


def test_no_custom_nodes_loaded(runtime):
    """Spec D2 — custom nodes are permanently out of scope."""
    import folder_paths

    custom_dirs = folder_paths.get_folder_paths("custom_nodes")
    loaded = {m for m in sys.modules if any(str(d) in str(m) for d in custom_dirs)}
    assert not loaded, f"custom node modules were imported: {sorted(loaded)}"


def test_dynamic_vram_is_wired(runtime):
    """main.py does DynamicVRAM setup in two places; both must have run.

    control.init() alone leaves aimdo_enabled False and the legacy patcher in
    place, which OOMs mid-sample on large models. Regression guard for the exact
    bug that made the MiniMax H3 example fail.
    """
    import comfy.memory_management
    import comfy.model_patcher

    if not runtime.dynamic_vram:
        pytest.skip("DynamicVRAM not active for this runtime configuration")

    assert comfy.memory_management.aimdo_enabled is True
    assert comfy.model_patcher.CoreModelPatcher is comfy.model_patcher.ModelPatcherDynamic


def test_footprint_is_exact(runtime):
    """Spec §11 — the namespace guard. Widening FOOTPRINT must be deliberate."""
    assert runtime.footprint == comfy_bridge.FOOTPRINT, (
        f"unexpected: {sorted(runtime.footprint - comfy_bridge.FOOTPRINT)}, "
        f"missing: {sorted(comfy_bridge.FOOTPRINT - runtime.footprint)}"
    )


def test_polluting_names_stay_absent(runtime):
    """The collisions that actually motivated this design.

    `tests` is the one that would break pytest collection in a host project;
    `sample`/`sd`/`options`/`float` would arrive only via the sys.path[0] insert
    at nodes.py:2334, which lives inside init_external_custom_nodes and
    therefore never runs for us (C9).
    """
    for name in ("tests", "sample", "sd", "options", "float", "ops", "conds"):
        module = sys.modules.get(name)
        if module is None:
            continue
        origin = getattr(module, "__file__", "") or ""
        assert not origin.startswith(str(runtime.root)), (
            f"{name!r} now resolves to ComfyUI ({origin}) — the sys.path[0] "
            "insert at nodes.py:2334 may have fired"
        )


def test_sys_path_is_clean(runtime):
    """Spec §5.1 step 10 — the path entry must not survive start()."""
    assert str(runtime.root) not in sys.path
    assert not any(str(runtime.root) == p for p in sys.path)


def test_no_port_bound(runtime):
    """Spec C6 — PromptServer is constructed but run() is never called."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        assert sock.connect_ex(("127.0.0.1", 8188)) != 0, "something is listening on 8188"


def test_prompt_server_instance_exists(runtime):
    """Spec C7 — V3 nodes reach for PromptServer.instance directly."""
    import server

    assert getattr(server, "__comfy_bridge_stub__", False), "expected the stub server"
    assert server.PromptServer.instance is runtime.server


def test_stub_satisfies_the_real_importers(runtime):
    """The three shipped modules that do `from server import PromptServer`."""
    import server

    instance = server.PromptServer.instance
    # comfy_extras/nodes_images.py:597 and nodes_gaussian_splat.py:1156
    instance.send_progress_text("hello", "node-1")

    # comfy_api/latest/__init__.py:32 — shipped nodes really do register
    # replacements during init_extra_nodes, so this path is live, not theoretical.
    manager = instance.node_replace_manager
    assert manager.registered, "expected shipped nodes to have registered replacements"
    before = len(manager.registered)
    manager.register(object())
    assert len(manager.registered) == before + 1


def test_progress_is_routed_to_callback(runtime):
    """Spec C12 — the real send_sync queues onto a loop that never runs."""
    events = []
    runtime.progress_callback = lambda event, data, sid: events.append(event)
    try:
        runtime.server.send_progress_text("x", "n1")
        runtime.server.send_sync("progress", {"value": 1, "max": 2}, "sid")
        assert events == ["progress_text", "progress"]
        assert not runtime.loop.is_running()
    finally:
        runtime.progress_callback = None


def test_web_stack_never_imported(runtime):
    """The point of the stub — no aiohttp, no `utils`, no `app`."""
    assert "aiohttp" not in sys.modules
    for name in ("utils", "app", "api_server", "middleware", "execution"):
        assert name not in sys.modules, f"{name} was imported despite the stub server"


def test_deferred_imports_resolve_after_path_removal(runtime):
    """Spec C11 — function-scope imports must hit the sys.modules cache."""
    from comfy_api.latest import ComfyAPI_latest  # noqa: F401
    from comfy_execution import caching  # noqa: F401

    # comfy_execution/progress.py:10 needs protocol; the deferred sites in
    # caching.py:303 (execution) and asset_enrichment.py:18 (app) are never
    # reached by codegen — asset_enrichment returns early unless --enable-assets.
    assert "protocol" in sys.modules
    assert "folder_paths" in sys.modules


def test_double_start_same_config_returns_same_runtime(runtime):
    assert comfy_bridge.start(dynamic_vram=True) is runtime


def test_double_start_different_config_raises(runtime):
    with pytest.raises(comfy_bridge.BootstrapError, match="different"):
        comfy_bridge.start(dynamic_vram=True, vram_mode="lowvram")


def test_bad_root_raises():
    from comfy_bridge.bootstrap import _resolve_root

    with pytest.raises(comfy_bridge.BootstrapError, match="does not look like"):
        _resolve_root("/tmp")

    with pytest.raises(comfy_bridge.BootstrapError, match="does not exist"):
        _resolve_root("/tmp/definitely-not-a-comfyui-checkout")
