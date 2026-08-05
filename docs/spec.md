# ComfyUI Bridge — Plan & Spec

**Status:** draft v3 (revised after review)
**Date:** 2026-08-03
**Target ComfyUI:** v0.30.0 (`14b05228`) at `/home/nick/Projects/ComfyUI`

> **Changes from v1:** the runtime workflow executor is gone. The deliverable is
> now a **codegen tool** that translates workflow JSON into readable Python, plus
> the minimal runtime needed to make that Python importable and callable. The HTTP
> backend, the backend abstraction, `PromptExecutor`, `validate_prompt`, the
> `BridgeOutput` sink node, disk read-back, and result caching are all dropped.
>
> **Changes from v2:** no vendoring. The bridge points at the existing local
> ComfyUI checkout by path. The git submodule, `VendorVersionError`, and the
> vendoring milestone are dropped (D8).

---

## 1. Goal

Two components:

1. **`comfy-codegen`** — a CLI that reads an API-format workflow JSON and emits a
   standalone, readable Python module: one function per node, plus a `run_graph()`
   that wires them together.
2. **`comfy_bridge`** — the runtime the generated module imports: it makes
   ComfyUI's shipped nodes available in-process, without the web UI, and without
   ComfyUI's loose top-level modules leaking into the host namespace.

**In scope**
- Translate API-format workflow JSON → Python.
- Load ComfyUI's shipped node set in-process; own the order-sensitive startup.
- Contain the namespace footprint to a documented, test-enforced set.

**Non-goals**
- Running workflow JSON directly at runtime. *(Dropped in v2 — codegen replaces it.)*
- Custom / third-party nodes. **Permanently out of scope** (§3, Decision D2).
- Any HTTP surface, in either direction. *(HTTP backend deferred to a follow-up.)*
- Concurrency. One graph at a time per process (Decision D3).
- Modifying the ComfyUI checkout. The bridge reads it; it never patches it.

---

## 2. Decisions taken

| # | Decision | Consequence |
|---|---|---|
| D1 | Project drives ComfyUI; ComfyUI never calls our code | No custom-node package needed |
| D2 | Third-party custom nodes out of scope **forever** | Tier 2.5's guarantees hold permanently (§3) |
| D3 | One graph at a time per process | No locking, no worker pool |
| D4 | In-memory outputs required | **Free** under codegen — node functions return tensors directly (§4) |
| D5 | Upstream tracked rarely, major versions only | Golden-workflow suite kept small |
| D6 | Node functions return the node's **outputs** | §6.2 |
| D7 | **No result caching** | Each `run_graph()` re-executes everything, including checkpoint loads (§6.5) |
| D8 | **No vendoring** — use the existing local checkout | No submodule, no `_vendor/`; `comfy_root` is a configured path (§10) |
| D9 | **Stub `PromptServer`** instead of importing the real `server.py` | Footprint 18 → 11; no aiohttp, no `utils`/`app`/`api_server`/`middleware`. Real one still available via `start(use_real_server=True)` (§5.2) |
| D10 | **Keep `model_management`; do not default to `gpu_only`** | It is non-optional (C23), and gpu-only disables offload (C24) — on a 24GB 3090 with a 32B text encoder that OOMs rather than degrades |
| D11 | **Extending ComfyUI's behaviour is in scope**, provided the checkout is never polluted | Hard invariant: no monkey-patching ComfyUI modules or objects, no writing into the checkout — test-enforced. `vae_chunked.py` was the motivating case but proved unnecessary (C30) and now sits in `examples/` as a fallback rather than in the package. |
| D12 | **API-format input only** | No subgraph flattening, no UI-format parsing. Codegen rejects UI-format with a clear message. Drops scope item A. |
| D13 | **Locally-authored nodes and patches are in scope; D11's "no mutation" narrows to "no *unmanaged* mutation"** | Acceleration work needs to replace module globals, so the invariant becomes: mutate only through `patch_attr`, which records and reverses; leave nothing applied at rest — test-enforced via `active_patches()`. Prefer ComfyUI's own ModelPatcher wrapper API (`comfy_bridge.hooks`), which is per-clone and needs no patching at all. Local nodes register into the Runtime's *copy* of the node table, so `nodes.NODE_CLASS_MAPPINGS` stays untouched and codegen picks them up for free. D2 is unchanged — this is for code we write. See `docs/extending.md`. |

---

## 3. Constraints established during investigation

| # | Finding | Evidence |
|---|---|---|
| C1 | ComfyUI is **not** pip-installable. No `[build-system]`, no package config, no entry points. | `pyproject.toml:1-11` |
| C2 | `comfy/`, `comfy_execution/`, `comfy_extras/` have no `__init__.py`; core modules are loose files at the repo root. | repo layout |
| C3 | ComfyUI imports its own modules **absolutely**. Relocating files changes nothing — absolute imports resolve via `sys.path`, not the containing package (PEP 328). | repo-wide grep |
| C4 | Device env vars must be set **before torch is imported**. | `main.py:83-93` |
| C5 | `comfy.cli_args.args` comes from `parser.parse_args([])` when `args_parsing` is False (the default); `comfy.model_management` reads it at import time. | `comfy/cli_args.py:275-278` |
| C6 | `PromptServer.__init__` sets `PromptServer.instance` and does **not** bind a port. Binding happens only in `run()`. | `server.py:216` |
| C7 | Some V3 nodes call `PromptServer.instance` directly, so a real instance must exist. | `comfy_extras/nodes_images.py:597` |
| C8 | `init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)` is a first-class supported configuration. | `nodes.py:2541` |
| C9 | The `sys.path.insert(0, .../comfy)` lives **inside** `init_external_custom_nodes` — skipping custom nodes avoids it entirely. | `nodes.py:2323, 2334` |
| C10 | ~~13 names.~~ → measured at **18** with the real `server.py` → **11** with the stub (D9). The original survey omitted `server.py`, which imports `api_server`, `middleware`, `comfyui_version` directly and `utils` via `app.*`. Stubbing `server` drops all of those plus `execution`. Final set: `comfy`, `comfy_api`, `comfy_config`, `comfy_execution`, `comfy_extras`, `cuda_malloc`, `folder_paths`, `latent_preview`, `node_helpers`, `nodes`, `protocol`. | measured by `start()`; `server.py:44,59,63`, `app/custom_node_manager.py:9` |
| C22 | The node layer's **entire** use of `PromptServer` is `send_progress_text` (2 sites) and `node_replace_manager.register` (1 site). Only 3 shipped modules import `server` at all. `comfy.utils.ProgressBar` is independent — it dispatches via the module-level `PROGRESS_BAR_HOOK`, installed only by `main.py`'s `hijack_progress`, which we never call. | `comfy_extras/nodes_images.py:597`, `nodes_gaussian_splat.py:1156`, `comfy_api/latest/__init__.py:32`, `comfy/utils.py:1273` |
| C23 | `comfy.model_management` is **not** optional: 82 files under `comfy/` reference it, 97 references in `comfy/sd.py` alone. It is the device/dtype policy layer, not a service. | grep |
| C24 | Loader nodes take **no device argument** — `load_checkpoint(self, ckpt_name)`, `load_unet(self, unet_name, weight_dtype)`. Placement comes entirely from `model_management`. `args.gpu_only` makes `text_encoder_offload_device()`, `intermediate_device()` and `vae_offload_device()` return the GPU instead of CPU, i.e. it *disables* offload. | `nodes.py:627,977`, `comfy/model_management.py:1174-1245` |
| C25 | `comfy_execution/asset_enrichment.py` returns early unless `--enable-assets`, so its deferred `app` import never fires under codegen. | `comfy_execution/asset_enrichment.py:14-19` |
| **C26** | **DynamicVRAM setup happens in `main.py` in TWO places, and both are required.** `comfy_aimdo.control.init()` early (58-70), then after `comfy.model_management` is importable: `control.init_devices(...)`, `CoreModelPatcher = ModelPatcherDynamic`, `memory_management.aimdo_enabled = True` (251-282). Doing only the first leaves the legacy estimate-based patcher in place and large models **OOM mid-sample**. | `main.py:58-70, 251-282` |
| **C27** | Upstream gates DynamicVRAM behind **torch >= 2.8** unless `--enable-dynamic-vram` is passed explicitly. This env has torch 2.6, so it must be forced. | `main.py:252` |
| **C28** | `main.py` sets several env vars the bridge must mirror: `CUBLAS_WORKSPACE_CONFIG` when deterministic, `ASCEND_RT_VISIBLE_DEVICES`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`, `OCL_SET_SVM_SIZE` on ROCm. | `main.py:74-102` |
| **C29** | **`execution.py:751` wraps the entire prompt execution in `torch.inference_mode()`. Generated code MUST do the same.** Without it every node call retains its autograd graph and successive calls chain together: measured +7.2 GB of retained GPU memory per VAE tile, climbing 0.01 → 7.21 → 14.42 → 21.62 GB → OOM. With it, the same decode peaks at **0.11 GB**. This is the highest-consequence single line lost by dropping `PromptExecutor`. | `execution.py:751`, `comfy/samplers.py:471`; measured |
| **C30** | ~~External tiling is required for the MiniMax H3 video VAE.~~ **WRONG — retracted after M2.** Two facts stand: ComfyUI's decode estimate is 8-30x low (`estimate_decode_memory` predicts 0.22-0.33 GB against 1.8-7.2 GB actual), and `MiniMaxH3VideoVAE.decode_tiled` ignores ComfyUI's `tile_x/tile_y/overlap` (it is `return self.decode(z)`). But the conclusion drawn from them did not: every OOM measurement predated C29's `inference_mode`. The VAE tiles internally *within one* `decode()` call, so with autograd live it retained every tile's activations. Under `inference_mode` the stock `VAEDecode` node decodes 243 frames at 864x480 on a 24GB card. | `comfy/sd.py:954-963`, `comfy/ldm/minimax/vae.py:678`; measured end-to-end |
| **C38** | **Autograd retention was the single root cause behind every memory failure in this project** — the sampling OOM, the decode OOM, and the per-tile climb in the external decoder. Peak went from OOM-at-22.5 GB to 0.11 GB. Before attributing a ComfyUI memory failure to model size or an upstream bug, confirm the call is under `inference_mode`. | measured |
| **C31** | **V3 nodes must be called on `cls.PREPARE_CLASS_CLONE(v3_data)`, not the raw class.** The raw class has `hidden = None`, so any node reading `cls.hidden.*` raises `AttributeError` (e.g. `SaveVideo` → `cls.hidden.extra_pnginfo`). The clone attaches a `HiddenHolder`; passing `None` yields all-None fields, which is right for codegen, and it keeps per-call state off the shared class. 4 shipped node files read `cls.hidden.*`. | `comfy_api/latest/_io.py:1973-1980`; measured |
| **C32** | **`DynamicCombo` inputs take a dict keyed by input id, not the widget string.** `SaveVideo(codec=...)` wants `{"codec": "auto"}`, or `{"codec": "h264", "encoding": {"encoding": "re-encode", "crf": 23.0}}`; passing `"auto"` raises `TypeError: string indices must be integers`. UI `widgets_values` store the bare string, so **codegen must consult the schema and re-nest it**. 13 shipped node files declare `DynamicCombo.Input`. | `comfy_extras/nodes_video.py:122`, `comfy_api/latest/_io.py:1165` |
| **C33** | A `LATENT` is not always a tensor. MiniMax H3 generates video+audio jointly, so the sampler returns a `NestedTensor` pair: `VAEDecode` takes `unbind()[0]`, `vae_decode_audio` takes `unbind()[-1]`. Anything consuming latents must check `is_nested`. | `comfy/nested_tensor.py`, `nodes.py:331-333`, `comfy_extras/nodes_audio.py:100-101` |
| **C34** | **The API export flattens subgraphs for you**, emitting composite node ids like `105:24` (parent:inner). This is what makes D12 cheap — no flattening code needed — but ids are not valid Python identifiers, so naming must sanitise them. | measured on `video_minimax_h3_t2v_api.json` |
| **C35** | **Autogrow/dynamic-slot inputs arrive as dotted names.** `ComfyMathExpression` receives `"values.a"`, while `execute(expression, values: dict)` wants `values={"a": ...}`. `values.a=` is a syntax error as a kwarg, so codegen must nest dotted inputs and rename the parameter (`values_a`). Same family of problem as C32. | `comfy_extras/nodes_math.py`; `COMFY_AUTOGROW_V3` |
| **C36** | Node return names are types — `FLOAT`, `INT`, `BOOLEAN` — which snake_case straight onto `float`, `int`, `bool`. Generated variables must not shadow builtins; PEP 8 trailing underscore is the fix. | measured |
| **C37** | **`cuda_malloc` must not be imported once torch is loaded.** It sets `PYTORCH_CUDA_ALLOC_CONF`, which torch parses at *its* import; changing it later aborts the first CUDA init with "Allocator backend parsed at runtime != allocator backend parsed at load time". `main.py` is safe only because it imports cuda_malloc first. A library caller cannot guarantee that, so the bridge skips it when `torch` is already in `sys.modules`. | measured; `cuda_malloc.py`, `main.py:101` |
| C21 | ComfyUI registers each `comfy_extras` module in `sys.modules` under its **absolute file path** (~129 keys like `/home/nick/.../comfy_extras/nodes_ace`), not a dotted name. These are not importable identifiers and cannot collide; the footprint guard filters them with `str.isidentifier()`. | measured by `start()` |
| C11 | Only 4 deferred (function-scope) imports of root modules exist across `comfy_extras`/`comfy_api`/`comfy_execution`. | `comfy_api/latest/__init__.py:31`, `comfy_execution/caching.py:303`, `comfy_execution/asset_enrichment.py:18-19` |
| C12 | `send_sync` schedules via `call_soon_threadsafe`. With no loop running, callbacks accumulate unboundedly. | `server.py:1392-1394` |
| C13 | `history_result` carries UI dicts pointing at files already on disk — not tensors. **This is why codegen wins**: calling nodes directly returns tensors. | `execution.py:826-832` |
| C14 | `comfy_api/` is for nodes calling *into* ComfyUI, `STABLE = False`. Not a driving API. | `comfy_api/latest/__init__.py` |
| **C16** | **V3 is now the dominant node convention: 446 `define_schema` nodes vs a shrinking V1 set. V3 uses `@classmethod def execute(cls, ...) -> IO.NodeOutput`.** | `comfy_extras/nodes_images.py:48` |
| **C17** | ~~Zero shipped nodes use list I/O.~~ **WRONG — corrected at M1: 27 do.** The original grep looked for literal `INPUT_IS_LIST` assignments, which V3 nodes never make — they declare it via `Schema(is_input_list=...)`, so the attribute only exists at runtime. Affected: `RebatchLatents`, `RebatchImages`, `SplitImageToTileList`, `ImageMergeTileList`, `CreateList`, `ImageGrid`, and the whole training-dataset family. These are **unsupported** (§9); `invoke()` raises `UnsupportedNodeError`. | measured over `NODE_CLASS_MAPPINGS` |
| **C17a** | Truthiness trap: every V3 node exposes `OUTPUT_IS_LIST` as a per-output sequence, usually `[False]`. A non-empty list is truthy, so `if cls.OUTPUT_IS_LIST:` reports ~450 false positives. Guards must test the *contents*. | measured |
| **C18** | ~~243 async nodes.~~ **WRONG — corrected at M1: zero.** The 243 `async def` are helper functions inside `comfy_extras`; no shipped node's `execute` is a coroutine, so `FUNCTION` never resolves to `EXECUTE_NORMALIZED_ASYNC`. `invoke()` keeps its await path as cheap insurance, and a test fires if upstream adds one. | measured; `comfy_api/latest/_io.py:1927` |
| **C19** | Output-ness is declared two ways: `OUTPUT_NODE = True` (7, V1) and `is_output_node=True` (31, V3). | grep |
| **C20** | Dynamic graph expansion exists but is rare (3 sites). Cannot be statically generated (§9). | `execution.py`, `comfy_extras/` |

Both C17 and C18 were wrong in v2-v3 and were only caught by running code at M1.
Both had been derived from source greps rather than from the loaded registry —
worth remembering for any future constraint about node behaviour: **grep the
source to find things, measure the registry to make claims.**

---

## 4. Why codegen instead of a runtime executor

| | Runtime executor (v1) | Codegen (v2) |
|---|---|---|
| Outputs | UI dicts → files on disk (C13); needed a custom sink node to get tensors | Node functions return tensors **directly** |
| Readability | Workflow is opaque JSON at runtime | Generated Python is diffable, greppable, breakpoint-able |
| Machinery needed | `PromptExecutor`, `validate_prompt`, `PromptQueue`, `history_result` | A ~100-line `invoke()` shim |
| Editing a graph | Patch JSON by dotted path | Edit Python |
| Type support | None | Generated signatures + docstrings; IDE completion works |

The cost is §9's limitations — dynamic expansion and lazy evaluation don't
survive static generation. Given D2 (no custom nodes), both are edge cases.

---

## 5. Architecture

Two sibling packages, not one — codegen is a build-time tool, the bridge is a
runtime dependency of the code it emits.

```
/home/nick/Projects/comfy-bridge/          # runtime
  pyproject.toml
  comfy_bridge/
    __init__.py          # public API: start(), Runtime, errors           [M0 done]
    bootstrap.py         # ordering-critical startup (§5.1).              [M0 done]
                         # Only module that touches sys.path.
    errors.py            # exception hierarchy (§8)                       [M0 done]
    invoke.py            # V1/V3/async node-calling shim (§7)             [M1]
    nodes.py             # re-exports node classes under our namespace    [M1]
  tests/
    test_bootstrap.py    # the §11 cheap guards                           [M0 done]

/home/nick/Projects/comfy-codegen/          # build-time tool
  pyproject.toml
  comfy_codegen/
    cli.py               # `comfy-codegen workflow.json -o graph.py`      [M2]
    parse.py             # API-format JSON -> internal graph IR           [M2]
    emit.py              # IR -> Python source                            [M2]
    naming.py            # class_type + id -> stable identifiers          [M2]
  tests/
```

Generated modules import only `comfy_bridge`, never ComfyUI names directly —
that is what keeps the 18-name footprint out of user code.

**Invariant:** `bootstrap.py` is the only module that touches `sys.path` or
imports ComfyUI. `comfy_bridge/nodes.py` re-exports node classes so that
*generated files never import ComfyUI names directly* — user code sees only
`comfy_bridge.*`, which is the whole point of the namespace work.

### 5.1 Bootstrap sequence

Unchanged from v1 except step 9 (the `BridgeOutput` registration) is gone.

1. **Set device env vars** — `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`,
   `ONEAPI_DEVICE_SELECTOR`, before any torch import (C4). The bridge must not
   import torch at package-import time — enforced by a test.
2. **Insert the configured ComfyUI root at `sys.path[0]`.**
3. **Import and configure `comfy.cli_args.args`** before `comfy.model_management`
   is imported (C5).
3b. **`comfy_aimdo.control.init()`** — first half of DynamicVRAM (C26,
   `main.py:58-70`), before `model_management` is imported.
4. **Apply `cuda_malloc`** if enabled, mirroring `main.py`.
5. **Configure `folder_paths`** — base/models/output/temp dirs, extra model paths.
5b. **`control.init_devices()` + `CoreModelPatcher = ModelPatcherDynamic` +
   `aimdo_enabled = True`** — second half of DynamicVRAM (C26,
   `main.py:251-282`). **Omitting this is a silent correctness bug**: startup
   succeeds, small graphs run, and large models OOM mid-sample. Requires
   `dynamic_vram=True` on torch < 2.8 (C27).
6. **Install the stub `server` module** (D9) and construct its `PromptServer`,
   setting `PromptServer.instance` (C7). Must happen before `init_extra_nodes`,
   since the three importers bind the name at module scope. No port binds — there
   is no web application at all.
7. **Point the stub's callback** at the Runtime's `progress_callback`. With the
   real server this step instead overrides `send_sync`, which would otherwise
   queue onto a loop that never runs (C12).
8. **`await init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)`** (C8)
   via `run_until_complete` on the private loop.
9. **Eagerly import the 13 footprint names**, satisfying the deferred imports (C11).
10. **Remove the `sys.path` entry.** Package submodules still resolve via each
    package's absolute `__path__`.
11. **Assert the `sys.modules` delta** equals the expected set (§11).

`comfy_bridge.start()` is idempotent and process-global; a second call with
different settings raises `BootstrapError` rather than silently reusing the first
configuration. ComfyUI holds global state (`NODE_CLASS_MAPPINGS`,
`PromptServer.instance`, `model_management` device state), so this is not
negotiable.

### 5.2 The server stub (D9)

`comfy_bridge/_stub_server.py` registers a fake `server` module in `sys.modules`
before ComfyUI loads. It implements exactly the surface in C22 — nothing else.

Measured effect: footprint 18 → 11, `aiohttp` never imported, `utils`/`app`/
`api_server`/`middleware`/`execution` never imported, `start()` ≈ 3.8s.

Verified live, not theoretical: shipped nodes register 8 real `NodeReplace`
objects against the stub during `init_extra_nodes`, so the code path is
exercised on every startup.

`start(use_real_server=True)` restores upstream behaviour and widens the guard to
`FOOTPRINT_REAL_SERVER`. Kept as an escape hatch in case a future node needs more
of the real class than C22 covers.

### 5.3 Model & VRAM management (D10)

`comfy.model_management` **stays**. It is not an optional service that codegen
can route around — it is the device and dtype policy layer, referenced by 82
files under `comfy/` (C23). Every load path calls into it, and loader nodes
expose no device parameter (C24), so there is no "load onto device X" API to use
in its place.

What codegen *does* remove is the orchestration around it: `prompt_worker`'s
timed `gc.collect()`, cache resets, and `unload_all_models()` (spec v1 §8). None
of that is needed — generated scripts are straight-line code, and D7 means there
is no result cache to reset.

Manual control is therefore exposed rather than automated, re-exported through
`comfy_bridge` so generated modules never import ComfyUI directly:

```python
comfy_bridge.free_memory()                   # unload_all_models + soft_empty_cache
comfy_bridge.load_to_gpu(model)              # model_management.load_models_gpu([model])
```

**`gpu_only` is not the default.** It reads as the simple choice, but it *disables*
CPU offload — `text_encoder_offload_device()`, `intermediate_device()` and
`vae_offload_device()` all return the GPU (C24). On the target 24GB 3090, with a
32B text encoder plus a diffusion model plus VAE resident simultaneously, that
OOMs rather than degrading. `NORMAL_VRAM`'s offloading is precisely what makes
those models fit. It stays available as `vram_mode="gpu-only"` for small graphs.

---

## 6. Codegen

### 6.1 CLI

```
comfy-codegen workflow.json -o graph.py [--module-name graph] [--no-docstrings]
```

Input is **API format** (the UI's "Save (API Format)" export), not the UI
workflow JSON. The tool rejects UI-format input with a clear message, since the
two are easy to confuse and the failure would otherwise be cryptic.

### 6.2 Node functions (D6)

One function per node. **Link inputs become required positional parameters;
widget values become keyword parameters defaulted to the workflow's values.**

```python
def load_checkpoint(ckpt_name="v1-5-pruned-emaonly.safetensors"):
    """CheckpointLoaderSimple (node 4) -> (MODEL, CLIP, VAE)"""
    return invoke(CheckpointLoaderSimple, ckpt_name=ckpt_name)


def ksampler(model, positive, negative, latent_image,
             seed=42, steps=20, cfg=8.0,
             sampler_name="euler", scheduler="normal", denoise=1.0):
    """KSampler (node 3) -> (LATENT,)"""
    return invoke(KSampler, model=model, positive=positive,
                  negative=negative, latent_image=latent_image,
                  seed=seed, steps=steps, cfg=cfg,
                  sampler_name=sampler_name, scheduler=scheduler,
                  denoise=denoise)
```

Docstrings carry the originating node id and the `RETURN_TYPES`, so the generated
file stays traceable back to the workflow.

### 6.3 `run_graph()`

**`run_graph()` is emitted with `@torch.inference_mode()`** (C29). This is not an
optimisation — it replaces `execution.py:751`, and omitting it makes every
generated module leak activation memory until it OOMs. The emitter must never
produce a `run_graph()` without it, and `invoke()` additionally wraps each call
in `torch.no_grad()` so that calling a single node function directly is also
safe. One consequence worth documenting for callers: outputs are inference-mode
tensors, so anything feeding them into an autograd context must `.clone()` first.


Emitted at the bottom. Constructs and calls nodes in topological order, pipelining
outputs to downstream inputs:

```python
def run_graph():
    model, clip, vae = load_checkpoint()
    (pos,) = clip_text_encode(clip, text="a red bicycle")
    (neg,) = clip_text_encode_1(clip, text="")
    (lat,) = empty_latent_image()
    (lat_1,) = ksampler(model, pos, neg, lat)
    (img,) = vae_decode(vae, lat_1)
    return (save_image(img),)
```

**Return value:** a tuple of the outputs of every node that has incoming edges but
no outgoing edges — the graph's sinks.

*Note on sink detection:* this rule is structural, and it does not always coincide
with ComfyUI's own notion of an output node (`OUTPUT_NODE` / `is_output_node`,
C19). A `PreviewImage` with nothing downstream is a sink by both rules; an output
node used mid-graph is a sink by ComfyUI's rule but not ours. The generated
module will therefore also emit `OUTPUT_NODES = [...]` as a module constant, so
the divergence is visible rather than silent. Sinks with no incoming edges
(isolated nodes) are excluded per the stated rule.

### 6.4 Naming

`class_type` → `snake_case`, de-duplicated with a numeric suffix in node-id order
(`clip_text_encode`, `clip_text_encode_1`). Stable across regeneration as long as
node ids are stable, so regenerated files diff cleanly. Python keywords and
identifier collisions are suffixed.

### 6.5 No caching (D7)

Generated functions are pure calls with no memoization, so each `run_graph()`
re-executes everything including checkpoint loads. This keeps generated code
stateless and obvious. Because loaders are ordinary functions, a caller who wants
to amortize can hoist by hand:

```python
models = load_checkpoint()          # once
for prompt in prompts:
    ...                             # reuse `models`
```

This is documented in the generated file's header comment rather than solved in
code.

---

## 7. The `invoke()` shim

The one piece of real complexity. `execution.py` does non-trivial work to call a
node, and generated code must not reimplement it inline. `invoke(node_cls, **kwargs)`
centralises it:

- **V1 vs V3 dispatch** (C16) — V1: instantiate, call `getattr(inst, cls.FUNCTION)`,
  returns a tuple. V3: **call `cls.PREPARE_CLASS_CLONE(None)` first** (C31), then
  `clone.EXECUTE_NORMALIZED(...)`, and unwrap the returned `NodeOutput` via
  `.result`. Skipping the clone works until a node reads `cls.hidden.*`, then
  raises `AttributeError`.
- **DynamicCombo arguments** (C32) — the emitter must read the schema and emit a
  nested dict, not the widget string. Simplest correct default: for a
  `DynamicCombo` input whose UI value is `"x"`, emit `{"<input_id>": "x"}` plus
  any sub-inputs that option declares.
- **V3 hidden inputs** — `unique_id`, `prompt`, `dynprompt`, `extra_pnginfo` and
  the auth-token fields are injected via ComfyUI's own
  `_io.get_finalized_class_inputs`, with synthetic values (a stable fake node id,
  empty prompt dict). Nodes that genuinely need graph introspection will not work;
  that's accepted and listed in §9.
- **Async** (C18) — `inspect.iscoroutinefunction` → driven to completion on the
  bridge's private event loop, so `run_graph()` stays synchronous.
- **No list mapping** (C17) — deliberately omitted, but this is now a real
  limitation rather than a free win: 27 shipped nodes use it and `invoke()`
  refuses them with `UnsupportedNodeError`. The guard tests attribute *contents*
  (C17a). Implementing `_map_node_over_list` is the fix if any of those nodes
  turn out to be needed.

The shim mirrors `execution.py:159-430`. It is the piece most likely to break on
an upstream bump, which is what §11's golden tests are for.

---

## 8. Errors

```
ComfyBridgeError
├── BootstrapError          # bad root, wrong order, double start()
├── NamespaceError          # sys.modules delta guard tripped
├── CodegenError            # unparseable workflow, UI-format input, unsupported node
│   └── UnsupportedNodeError# dynamic expansion, lazy inputs, list I/O (§9)
└── NodeExecutionError      # a node raised; carries node id, class_type, original traceback
```

`NodeExecutionError` must name the originating node id and `class_type` — with
codegen this is easier than in v1, since the Python traceback already points at
the generated function.

---

## 9. Known limitations

Consequences of static generation. All are detected at codegen time and raise
`UnsupportedNodeError` rather than producing wrong code:

1. **Dynamic graph expansion** (C20) — nodes returning an expanded subgraph at
   runtime cannot be statically unrolled. Rare (3 sites), and D2 removes the
   custom nodes that use it most.
2. **Lazy input evaluation** — ComfyUI can skip evaluating inputs a node declares
   lazy. Generated code is eager, so a lazy-input graph does more work than the
   engine would. Correct, but slower.
3. **Node-driven control flow** — anything depending on `ExecutionBlocker` or
   partial-execution targets.
4. **Graph-introspecting nodes** — those genuinely needing `dynprompt` or the real
   `prompt` dict get synthetic values (§7).
5. **No caching** (D7) — by choice, not limitation, but worth stating alongside.

---

## 10. ComfyUI checkout (D8)

The bridge points at an existing local ComfyUI checkout — no vendoring, no
submodule, no copy.

- `comfy_root` is resolved in order: the `start()` argument, then the
  `COMFY_ROOT` environment variable, then the default
  `/home/nick/Projects/ComfyUI`.
- That checkout is also the model store (83GB under `models/`, gitignored) and
  stays independently runnable, which is what the golden tests compare against.
- `bootstrap.py` records the ComfyUI commit it last validated against and logs a
  warning — not an error — when the checkout has moved. Given D5 (rare upgrades),
  a nagging log is proportionate; hard-failing on every upstream commit is not.
- ComfyUI's `requirements.txt` is an install step for the target conda env
  (`gygax`), not a dependency of this package.
- On upgrade: pull ComfyUI, run the suite, regenerate golden files, review the
  diff. **Regenerating the golden graphs and diffing the emitted Python is the
  best upstream-drift detector** — signature changes show up as source diffs.

---

## 11. Test plan

| Test | Guards against |
|---|---|
| **`sys.modules` delta** — after `start()`, new top-level ComfyUI modules equal exactly the 11 in C10 (18 with `use_real_server=True`) | Silent namespace regression |
| **Polluting names absent** — `tests`, `sample`, `sd`, `options`, `float`, `ops`, `conds` must not resolve into the checkout | The `nodes.py:2334` `sys.path[0]` insert firing |
| **DynamicVRAM wired** — `aimdo_enabled is True` and `CoreModelPatcher is ModelPatcherDynamic` after `start(dynamic_vram=True)` | Doing only half of C26 — startup succeeds and large models OOM mid-sample |
| **No-torch-at-import** — `import comfy_bridge` must not import torch | Breaking C4 ordering |
| **`sys.path` clean** — no bridge entries remain after `start()` | Path leakage |
| **No port bound** — nothing listens after `start()` | Accidental `run()` |
| **Deferred-import probe** — exercise the 4 C11 sites after path removal | Path removal breaking lazy imports |
| **List-I/O set pinned** — the 27 list-I/O nodes stay 27, and `invoke()` refuses them | The unsupported set growing silently on an upstream bump |
| **No async entrypoints** — no shipped node has a coroutine `execute` | C18 flipping, making invoke()'s await path load-bearing and untested |
| **No ComfyUI mutation** — `PROGRESS_BAR_HOOK` unset, node registry is a copy | Violating the D11 no-pollution invariant |
| **Codegen golden files** — fixture workflows → committed `.py`, byte-compared | Emitter drift, upstream signature changes |
| **Generated-code execution** — golden graphs run on CPU, output hashes compared | Node behaviour drift, shim bugs |
| **V1/V3/async coverage** — one of each through `invoke()` | Shim dispatch bugs |
| **Round-trip** — every shipped node type codegens without `UnsupportedNodeError` (or is on a known-unsupported list) | §9 growing silently |

The first six are cheap and run every commit. The golden-file tests are the ones
that make an upstream bump fail loudly.

---

## 12. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| # | Deliverable | Exit criterion | Status |
|---|---|---|---|
| M0 | Bootstrap | `start()` loads all shipped nodes, binds no port, 11-name delta holds | **done** — 591 nodes, 16 tests green |
| M1 | `invoke()` shim | V1, V3, and async nodes all callable; list-I/O guard in place | **prototyped, not packaged** — see below |
| M2 | Codegen: parse + emit | one workflow → runnable Python, output matches upstream | **done** — 18 tests; MiniMax H3 workflow generates and runs |
| M3 | Codegen: coverage | sink detection, naming, `OUTPUT_NODES`, `UnsupportedNodeError` paths; round-trip test green | **done** — round-trip sweeps all 591 shipped nodes: 564 codegen, 27 list-I/O unsupported, 0 unexplained failures |
| M4 | Hardening | error hierarchy, generated-file header docs, golden suite green | not started |

**M1 status.** A working `invoke()` exists inside
`examples/minimax_h3_t2v.py` and has been exercised end-to-end against a real
MiniMax H3 generation. It is *not* yet `comfy_bridge/invoke.py` and has no tests.
Promoting it is the immediate next step; the semantics are already settled by
C16/C29/C31.

**Scope added since v3, discovered by running the example.** None of these were
in the original plan:

| # | Item | Why |
|---|---|---|
| ~~A~~ | ~~Subgraph flattening~~ | **Dropped by D12** — API-format only. Note the reference workflow `video_minimax_h3_t2v.json` is UI-format, so it must be re-exported via *Save (API Format)* before codegen can consume it. |
| B | **DynamicCombo re-nesting** | C32 — UI `widgets_values` hold `"auto"`, the node wants `{"codec": "auto"}`. 13 shipped node files affected. |
| C | **Nested latents** | C33 — `LATENT` may be a `NestedTensor` pair; which half depends on the consumer. |
| ~~D~~ | ~~`vae_chunked.py`~~ | **Resolved** — moved to `examples/`; C30 showed it was never needed. |
| ~~E~~ | ~~`offload()` correctness~~ | **Resolved** — `comfy_bridge.memory.as_patcher()` handles `VAE` (`.patcher`) and `ModelPatcher` alike; tested. |

**Deferred to follow-up:** HTTP backend (Tier 1) for callers who want process
isolation. The API contract is unaffected — generated modules would import a
different `invoke()`.
