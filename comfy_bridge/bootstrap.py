"""Ordering-critical ComfyUI startup (§5.1).

This is the only module in the package that touches ``sys.path`` or imports
ComfyUI. Every step below is ordered against a constraint recorded in CLAUDE.md;
reordering them will fail in ways that are quiet rather than loud.

Deliberately imports no third-party modules at module scope — in particular not
torch, because the device environment variables in step 1 only take effect if
they are set before torch is first imported (C4).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import _stub_server
from .errors import BootstrapError, NamespaceError

log = logging.getLogger("comfy_bridge")

DEFAULT_COMFY_ROOT = "/home/nick/Projects/ComfyUI"

#: ComfyUI version this package was last validated against (§10).
VALIDATED_COMFY_VERSION = "0.30.0"

#: The only top-level ComfyUI modules Tier 2.5 is allowed to introduce (C10).
#:
#: Measured empirically by running start(), not by static grep. The original
#: 13-name survey covered nodes.py, comfy_extras/, comfy_execution/ and
#: comfy_api/ but not server.py — which Tier 2.5 requires for
#: PromptServer.instance, and which imports api_server, middleware and
#: comfyui_version directly plus utils transitively via app.* (see
#: app/custom_node_manager.py:9, app/frontend_management.py:19). ``tests`` stays
#: absent, which was the collision that actually mattered.
#: Default footprint, with the stub server (see _stub_server.py). Stubbing drops
#: server/app/api_server/middleware/utils/comfyui_version — and with them
#: ``execution`` and ``protocol``, whose only importers were server.py and the
#: deferred site at comfy_execution/caching.py:303 that codegen never reaches.
FOOTPRINT = frozenset(
    {
        "comfy",
        "comfy_api",
        "comfy_config",
        "comfy_execution",
        "comfy_extras",
        "cuda_malloc",
        "folder_paths",
        "latent_preview",
        "node_helpers",
        "nodes",
        "protocol",  # comfy_execution/progress.py:10 imports it independently
    }
)

#: Footprint when start(use_real_server=True) imports ComfyUI's real server.py.
FOOTPRINT_REAL_SERVER = FOOTPRINT | {
    "api_server",
    "app",
    "comfyui_version",
    "execution",
    "middleware",
    "server",
    "utils",
}

#: Footprint members that legitimately may not appear.
OPTIONAL_FOOTPRINT = frozenset({"cuda_malloc"})

#: Top-level names the bridge itself puts into sys.modules, as opposed to ones
#: ComfyUI introduces. The stub server (D9) is not a file under the checkout, so
#: _comfy_top_level_modules cannot see it and the footprint guard would otherwise
#: be blind to the one name this package injects. It stays installed for the
#: process lifetime: comfy_api/latest/__init__.py:31 holds a *deferred*
#: ``from server import PromptServer``, and by the time it fires sys.path no
#: longer contains the checkout, so removing the stub would turn that into an
#: ImportError with no fallback.
INJECTED = frozenset({"server"})

VRAM_MODES = ("normal", "highvram", "lowvram", "novram", "cpu", "gpu-only")

# Files that must exist for a directory to plausibly be a ComfyUI checkout.
_ROOT_MARKERS = ("nodes.py", "folder_paths.py", "execution.py", "comfy")


@dataclass
class Runtime:
    """Handle to a started ComfyUI. Created by :func:`start`."""

    root: Path
    loop: Any
    server: Any
    nodes: dict[str, Any] = field(repr=False, default_factory=dict)
    display_names: dict[str, str] = field(repr=False, default_factory=dict)
    footprint: frozenset[str] = frozenset()
    #: Top-level names this package injected rather than ComfyUI (see INJECTED).
    injected: frozenset[str] = frozenset()
    dynamic_vram: bool = False
    progress_callback: Callable[[str, Any, Any], None] | None = None

    def __len__(self) -> int:
        return len(self.nodes)


@dataclass
class _Loaded:
    """What the load phase produced. At most one of these exists per process.

    Recorded *before* the footprint guard runs, so that a start() which loads
    ComfyUI successfully and then trips the guard can be retried with different
    validation settings without re-running any of the load steps — several of
    which (comfy-aimdo's two-stage init, init_extra_nodes) are not idempotent,
    and one of which (the device env vars) can never be applied twice because
    torch is imported in between.
    """

    root: Path
    loop: Any
    server: Any
    nodes_module: Any
    holder: dict[str, Runtime]
    dynamic_vram: bool
    config: dict[str, Any]


_RUNTIME: Runtime | None = None
_START_CONFIG: dict[str, Any] | None = None
_LOADED: _Loaded | None = None


def _resolve_root(explicit: str | os.PathLike[str] | None) -> Path:
    raw = explicit or os.environ.get("COMFY_ROOT") or DEFAULT_COMFY_ROOT
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise BootstrapError(f"ComfyUI root does not exist: {root}")
    missing = [m for m in _ROOT_MARKERS if not (root / m).exists()]
    if missing:
        raise BootstrapError(
            f"{root} does not look like a ComfyUI checkout (missing: {', '.join(missing)})"
        )
    return root


def _check_version(root: Path) -> None:
    """Warn — never fail — when the checkout has moved (§10)."""
    version_file = root / "comfyui_version.py"
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError:
        log.warning("could not read %s; skipping version check", version_file)
        return
    match = re.search(r"__version__\s*=\s*[\"']([^\"']+)[\"']", text)
    if not match:
        log.warning("could not parse a version from %s", version_file)
        return
    found = match.group(1)
    if found != VALIDATED_COMFY_VERSION:
        log.warning(
            "ComfyUI %s differs from the validated %s — regenerate golden files "
            "and review the diff before trusting output.",
            found,
            VALIDATED_COMFY_VERSION,
        )


def _set_device_env(device: int | str | None) -> None:
    """Step 1 — must precede any torch import (C4, mirrors main.py:83-93)."""
    if device is None:
        return
    if "torch" in sys.modules:
        raise BootstrapError(
            "torch was already imported before start(device=...) — the device "
            "environment variables can no longer take effect. Import comfy_bridge "
            "and call start() before anything that pulls in torch."
        )
    value = str(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    os.environ["HIP_VISIBLE_DEVICES"] = value


def _init_dynamic_vram() -> bool:
    """Initialise comfy-aimdo dynamic VRAM (mirrors main.py:58-70).

    This is the single most important thing main.py does that is not part of
    importing ComfyUI. Without it, ``comfy.memory_management.aimdo_enabled``
    stays False, model_management falls back to estimate-based loading, and
    large models OOM mid-sample instead of paging — which is exactly the
    difference between the web UI working and a bare embed failing.

    Must run after args are configured and before comfy.model_management is
    imported. Returns whether dynamic VRAM was enabled.
    """
    from comfy.cli_args import args, enables_dynamic_vram

    if not enables_dynamic_vram():
        log.info("dynamic VRAM disabled by configuration")
        return False

    import comfy_aimdo.control

    headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024**3)
    # Version-tolerant call chain, copied from main.py — comfy-aimdo 0.4.9,
    # 0.4.10 and 0.4.11 accept different keyword sets.
    try:
        comfy_aimdo.control.init(
            simple_vram_headroom=headroom,
            nvml_pressure=not args.disable_nvml_pressure,
        )
    except TypeError:
        try:
            comfy_aimdo.control.init(simple_vram_headroom=headroom)
        except TypeError:
            comfy_aimdo.control.init()
    log.info("dynamic VRAM enabled (comfy-aimdo)")
    return True


def _enable_dynamic_vram_devices(vram_headroom: float) -> bool:
    """Second half of dynamic VRAM setup (mirrors main.py:251-282).

    ``control.init()`` alone is not enough. This block runs *after*
    comfy.model_management is importable and does the part that actually
    matters: init_devices(), swapping CoreModelPatcher for ModelPatcherDynamic,
    and setting comfy.memory_management.aimdo_enabled = True. Without it the
    legacy estimate-based patcher stays in place and large models OOM mid-sample
    rather than paging.

    Note the torch gate: upstream requires torch >= 2.8 unless dynamic VRAM was
    explicitly requested. On older torch, pass dynamic_vram=True to force it —
    which is what the ComfyUI web UI is doing when it works on this box.
    """
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    import comfy_aimdo.control
    from comfy.cli_args import args, enables_dynamic_vram

    forced = bool(args.enable_dynamic_vram)
    eligible = forced or (
        enables_dynamic_vram()
        and comfy.model_management.is_nvidia()
        and not comfy.model_management.is_wsl()
    )
    if not eligible:
        return False

    if not forced and comfy.model_management.torch_version_numeric < (2, 8):
        log.warning(
            "DynamicVRAM needs torch >= 2.8 (found %s) and was not explicitly "
            "requested — falling back to the legacy ModelPatcher. VRAM estimates "
            "will be unreliable and large models may OOM mid-sample. Pass "
            "dynamic_vram=True to force it.",
            ".".join(map(str, comfy.model_management.torch_version_numeric)),
        )
        return False

    headroom = int(vram_headroom * 1024**3)
    devices = comfy.model_management.get_all_torch_devices()
    try:
        initialized = comfy_aimdo.control.init_devices(
            (d.index, headroom) for d in devices
        )
    except TypeError:
        # comfy-aimdo 0.4.9 protocol.
        initialized = comfy_aimdo.control.init_devices(d.index for d in devices)

    if not initialized:
        log.warning(
            "No working comfy-aimdo install detected; DynamicVRAM disabled and "
            "falling back to the legacy ModelPatcher."
        )
        return False

    try:
        comfy_aimdo.control.set_log_warning()
    except AttributeError:
        pass

    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    log.info("DynamicVRAM support detected and enabled")
    return True


def _configure_args(
    vram_mode: str,
    deterministic: bool,
    dynamic_vram: bool | None,
    reserve_vram: float | None,
    disable_nvml_pressure: bool,
) -> None:
    """Step 3 — mutate cli_args.args before model_management imports it (C5)."""
    if vram_mode not in VRAM_MODES:
        raise BootstrapError(f"vram_mode must be one of {VRAM_MODES}, got {vram_mode!r}")

    # comfy.options.args_parsing stays False, so cli_args uses parse_args([]) and
    # never touches our host process's sys.argv (comfy/cli_args.py:275-278).
    from comfy.cli_args import args

    args.disable_all_custom_nodes = True
    args.disable_api_nodes = True
    args.disable_manager_ui = True
    args.deterministic = deterministic

    for mode in ("highvram", "lowvram", "novram", "cpu"):
        setattr(args, mode, vram_mode == mode)
    args.gpu_only = vram_mode == "gpu-only"

    if deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Dynamic VRAM. Default (None) reproduces upstream: on unless a vram_mode
    # that is incompatible with it was selected — see enables_dynamic_vram().
    args.reserve_vram = reserve_vram
    args.disable_nvml_pressure = disable_nvml_pressure
    args.enable_dynamic_vram = dynamic_vram is True
    args.disable_dynamic_vram = dynamic_vram is False


def _configure_folder_paths(
    root: Path,
    models_dir: str | os.PathLike[str] | None,
    output_dir: str | os.PathLike[str] | None,
    temp_dir: str | os.PathLike[str] | None,
    input_dir: str | os.PathLike[str] | None,
    extra_model_paths: dict[str, str] | None,
) -> None:
    """Step 5.

    ``extra_model_paths`` is a ``{folder_name: path}`` mapping rather than a
    YAML file on purpose: ComfyUI's own loader lives in ``utils.extra_config``,
    and importing it would put ``utils`` — the worst of the generic top-level
    names — into the host namespace, breaking the C10 footprint.
    """
    import folder_paths

    if output_dir is not None:
        folder_paths.set_output_directory(str(Path(output_dir).expanduser().resolve()))
    if temp_dir is not None:
        folder_paths.set_temp_directory(str(Path(temp_dir).expanduser().resolve()))
    if input_dir is not None:
        folder_paths.set_input_directory(str(Path(input_dir).expanduser().resolve()))

    if models_dir is not None:
        base = Path(models_dir).expanduser().resolve()
        if not base.is_dir():
            raise BootstrapError(f"models_dir does not exist: {base}")
        for child in sorted(p for p in base.iterdir() if p.is_dir()):
            folder_paths.add_model_folder_path(child.name, str(child), is_default=True)

    for folder_name, path in (extra_model_paths or {}).items():
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise BootstrapError(
                f"extra_model_paths[{folder_name!r}] does not exist: {resolved}"
            )
        folder_paths.add_model_folder_path(folder_name, str(resolved))


def _install_progress_hook(server: Any, runtime_holder: dict[str, Runtime]) -> None:
    """Step 7 — replace send_sync (C12).

    PromptServer.send_sync schedules onto the event loop via call_soon_threadsafe
    (server.py:1392-1394). Our loop never runs, so those callbacks would pile up
    forever. Samplers push progress through comfy.utils.ProgressBar and V3 nodes
    call send_progress_text, so this fires on every sampling step — it is not a
    rare path.
    """

    def send_sync(event: str, data: Any, sid: Any = None) -> None:
        runtime = runtime_holder.get("runtime")
        callback = runtime.progress_callback if runtime else None
        if callback is not None:
            try:
                callback(event, data, sid)
            except Exception:  # never let a user callback break a graph
                log.exception("progress_callback raised; continuing")

    server.send_sync = send_sync


def _install_stub_progress_hook(server: Any, runtime_holder: dict[str, Runtime]) -> None:
    """Route the stub's events at the Runtime's progress_callback."""

    def forward(event: str, data: Any, sid: Any = None) -> None:
        runtime = runtime_holder.get("runtime")
        callback = runtime.progress_callback if runtime else None
        if callback is not None:
            callback(event, data, sid)

    server.progress_callback = forward


def _comfy_top_level_modules(root: Path) -> set[str]:
    """Top-level modules currently in sys.modules that live under the checkout.

    ComfyUI registers each comfy_extras module under its absolute file path
    rather than a dotted name (init_builtin_extra_nodes imports them by
    location), so sys.modules ends up with ~129 keys like
    ``/home/nick/Projects/ComfyUI/comfy_extras/nodes_ace``. Those are not
    importable names and cannot collide with anything, so they are excluded —
    only real identifiers count toward the footprint.
    """
    found: set[str] = set()
    root_str = str(root)
    for name, module in list(sys.modules.items()):
        if "." in name or module is None or not name.isidentifier():
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            paths = getattr(module, "__path__", None) or []
            origin = next(iter(paths), None)
        if origin and str(origin).startswith(root_str):
            found.add(name)
    return found


def _load_comfyui(config: dict[str, Any]) -> _Loaded:
    """Steps 1-10 of §5.1: everything that mutates the process.

    Runs at most once per process. Raises BootstrapError on any failure, having
    removed the sys.path entry first.
    """
    use_real_server = config["use_real_server"]

    # 1. device env vars, before torch exists
    _set_device_env(config["device"])

    # 2. path insert
    root = _resolve_root(config["comfy_root"])
    _check_version(root)
    root_str = str(root)
    sys.path.insert(0, root_str)

    try:
        # 3. args, before model_management reads them
        _configure_args(
            config["vram_mode"],
            config["deterministic"],
            config["dynamic_vram"],
            config["reserve_vram"],
            config["disable_nvml_pressure"],
        )

        # 3b. dynamic VRAM (main.py:58-70) — must precede model_management
        dynamic_vram_active = _init_dynamic_vram()

        # 4. cuda_malloc, mirroring main.py's conditional application
        from comfy.cli_args import args as _args

        if not _args.cpu and not _args.disable_cuda_malloc:
            if "torch" in sys.modules:
                # cuda_malloc sets PYTORCH_CUDA_ALLOC_CONF at import. torch parses
                # that once, at ITS import — changing it afterwards makes the
                # first CUDA init abort with "Allocator backend parsed at runtime
                # != allocator backend parsed at load time". main.py never hits
                # this because it imports cuda_malloc first; a library caller
                # (or a generated module with `import torch` at the top) can't
                # make that guarantee, so skip rather than crash.
                log.debug("torch already imported; skipping cuda_malloc")
            else:
                try:
                    import cuda_malloc  # noqa: F401
                except Exception as exc:  # non-fatal, as upstream treats it
                    log.debug("cuda_malloc unavailable: %r", exc)

        # 5. folder paths
        _configure_folder_paths(
            root,
            config["models_dir"],
            config["output_dir"],
            config["temp_dir"],
            config["input_dir"],
            config["extra_model_paths"],
        )

        # 5b. the half of dynamic VRAM that needs model_management loaded
        #     (main.py:251-282). This is what actually sets aimdo_enabled.
        if dynamic_vram_active:
            dynamic_vram_active = _enable_dynamic_vram_devices(config["vram_headroom"])

        # 6. PromptServer. The stub satisfies the only three importers
        #    (nodes_images, nodes_gaussian_splat, comfy_api.latest) without
        #    dragging in server/app/api_server/middleware/utils/comfyui_version
        #    or aiohttp. The real one is available as an escape hatch.
        holder: dict[str, Runtime] = {}
        loop = asyncio.new_event_loop()
        if use_real_server:
            import server as comfy_server

            prompt_server = comfy_server.PromptServer(loop)
            # 7. neutralise send_sync — the real one queues onto a loop that
            #    never runs (C12).
            _install_progress_hook(prompt_server, holder)
        else:
            prompt_server = _stub_server.install()
            _install_stub_progress_hook(prompt_server, holder)

        # 8. shipped nodes only — no custom nodes, no API nodes (C8/C9)
        import nodes as comfy_nodes

        loop.run_until_complete(
            comfy_nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
        )

        # 9. eagerly import the footprint so the deferred imports at
        #    comfy_api/latest/__init__.py:31, comfy_execution/caching.py:303 and
        #    comfy_execution/asset_enrichment.py:18-19 resolve from sys.modules
        #    once the path entry is gone (C11).
        expected = FOOTPRINT_REAL_SERVER if use_real_server else FOOTPRINT
        for name in sorted(expected):
            if name == "cuda_malloc" and "cuda_malloc" not in sys.modules:
                continue  # skipped on CPU-only or --disable-cuda-malloc
            __import__(name)

    except BootstrapError:
        _remove_path(root_str)
        raise
    except Exception as exc:
        _remove_path(root_str)
        raise BootstrapError(f"ComfyUI failed to start from {root}: {exc!r}") from exc

    # 10. drop the path entry; package submodules resolve via __path__ from here
    _remove_path(root_str)

    return _Loaded(
        root=root,
        loop=loop,
        server=prompt_server,
        nodes_module=comfy_nodes,
        holder=holder,
        dynamic_vram=dynamic_vram_active,
        config=config,
    )


def _check_footprint(root: Path, use_real_server: bool) -> frozenset[str]:
    """Step 11 — assert the namespace footprint, returning what was introduced."""
    expected = FOOTPRINT_REAL_SERVER if use_real_server else FOOTPRINT
    introduced = _comfy_top_level_modules(root)
    unexpected = introduced - expected
    # cuda_malloc is skipped on CPU, with --disable-cuda-malloc, or when torch
    # was already imported (see step 4), so its absence is not a regression.
    missing = expected - introduced - OPTIONAL_FOOTPRINT

    # The stub is not a file under the checkout, so the scan above cannot see
    # it. Check it explicitly rather than leave the one name we inject
    # unguarded — if something replaced sys.modules['server'] after startup,
    # the three importers that bound PromptServer at module scope are now
    # pointing at a different class than anything importing it later.
    if not use_real_server:
        installed = sys.modules.get(_stub_server.MODULE_NAME)
        if installed is None:
            missing = missing | {_stub_server.MODULE_NAME}
        elif not getattr(installed, "__comfy_bridge_stub__", False):
            unexpected = unexpected | {_stub_server.MODULE_NAME}

    if unexpected or missing:
        raise NamespaceError(unexpected, missing)
    return frozenset(introduced)


def start(
    comfy_root: str | os.PathLike[str] | None = None,
    *,
    device: int | str | None = None,
    vram_mode: str = "normal",
    models_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    temp_dir: str | os.PathLike[str] | None = None,
    input_dir: str | os.PathLike[str] | None = None,
    extra_model_paths: dict[str, str] | None = None,
    deterministic: bool = False,
    dynamic_vram: bool | None = None,
    reserve_vram: float | None = None,
    vram_headroom: float = 0.0,
    disable_nvml_pressure: bool = False,
    progress_callback: Callable[[str, Any, Any], None] | None = None,
    use_real_server: bool = False,
    enforce_footprint: bool = True,
) -> Runtime:
    """Start ComfyUI in-process with shipped nodes only.

    Process-global and idempotent: calling it again with the same configuration
    returns the existing Runtime, and calling it with a different one raises.
    ComfyUI keeps module-level state (NODE_CLASS_MAPPINGS, PromptServer.instance,
    model_management's device state) that cannot be meaningfully re-initialised.

    Runs in two phases. The **load** phase (§5.1 steps 1-10) mutates the process
    and happens at most once. The **validate** phase (step 11, the footprint
    guard) is re-runnable, so a call that loaded ComfyUI and then tripped the
    guard can be retried with ``enforce_footprint=False`` — which is the advice
    NamespaceError gives, and which only works because the retry skips loading.
    """
    global _RUNTIME, _START_CONFIG, _LOADED

    config = {
        "comfy_root": str(comfy_root) if comfy_root else None,
        "device": device,
        "vram_mode": vram_mode,
        "models_dir": str(models_dir) if models_dir else None,
        "output_dir": str(output_dir) if output_dir else None,
        "temp_dir": str(temp_dir) if temp_dir else None,
        "input_dir": str(input_dir) if input_dir else None,
        "extra_model_paths": dict(extra_model_paths or {}),
        "deterministic": deterministic,
        "dynamic_vram": dynamic_vram,
        "reserve_vram": reserve_vram,
        "vram_headroom": vram_headroom,
        "disable_nvml_pressure": disable_nvml_pressure,
        "use_real_server": use_real_server,
    }

    if _RUNTIME is not None:
        if config != _START_CONFIG:
            raise BootstrapError(
                "comfy_bridge.start() was already called with a different "
                "configuration. ComfyUI holds process-global state, so only one "
                "configuration per process is supported."
            )
        _RUNTIME.progress_callback = progress_callback
        return _RUNTIME

    if _LOADED is not None:
        # A previous start() completed the load phase and then failed — almost
        # always at the footprint guard, which is exactly when you want to retry
        # with enforce_footprint=False. Reuse what was loaded: re-running the
        # load phase would re-init comfy-aimdo and, if device= was given, fail on
        # the torch-already-imported check against torch that *we* imported.
        if config != _LOADED.config:
            raise BootstrapError(
                "ComfyUI is already loaded in this process from an earlier "
                "start() that did not complete, and it was loaded with a "
                "different configuration. Load-time settings (device, "
                "vram_mode, comfy_root, ...) are baked in once torch and "
                "model_management are imported and cannot be changed without a "
                "new process. Retry with the original arguments, or restart."
            )
        loaded = _LOADED
    else:
        loaded = _load_comfyui(config)
        _LOADED = loaded

    root = loaded.root

    # 11. footprint guard
    if enforce_footprint:
        introduced = _check_footprint(root, use_real_server)
    else:
        introduced = frozenset(_comfy_top_level_modules(root))

    comfy_nodes = loaded.nodes_module
    runtime = Runtime(
        root=root,
        loop=loaded.loop,
        server=loaded.server,
        nodes=dict(comfy_nodes.NODE_CLASS_MAPPINGS),
        display_names=dict(comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS),
        footprint=introduced,
        injected=frozenset() if use_real_server else INJECTED,
        dynamic_vram=loaded.dynamic_vram,
        progress_callback=progress_callback,
    )
    loaded.holder["runtime"] = runtime
    _RUNTIME = runtime
    _START_CONFIG = config
    log.info(
        "ComfyUI started from %s — %d nodes, %d new top-level modules, no port bound",
        root,
        len(runtime.nodes),
        len(introduced),
    )
    return runtime


def _remove_path(root_str: str) -> None:
    while root_str in sys.path:
        sys.path.remove(root_str)


def get_runtime() -> Runtime:
    """Return the started Runtime, or raise if start() has not been called."""
    if _RUNTIME is None:
        raise BootstrapError("comfy_bridge.start() has not been called yet")
    return _RUNTIME


def is_started() -> bool:
    return _RUNTIME is not None
