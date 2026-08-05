# comfy-bridge — working notes

Run ComfyUI's shipped nodes in-process, without the web UI. This file is the
operative record: the decisions that constrain how code here is written, and the
measured facts that make those constraints non-negotiable. Code comments cite
these by id (`D13`, `C29`, `§5.1`) — keep the ids stable when editing.

**Target ComfyUI:** v0.30.0, at `COMFY_ROOT` (default in `bootstrap.py`).
The bridge reads that checkout and never writes to it.

## 1. Shape of the project

Two components. Only the first lives here.

- **`comfy_bridge`** (this repo) — the runtime: makes ComfyUI's shipped nodes
  available in-process, and owns the order-sensitive startup.
- **`comfy-codegen`** (separate repo) — a build-time CLI translating an
  API-format workflow JSON into a readable Python module that imports
  `comfy_bridge`.

Generated modules import only `comfy_bridge`, never ComfyUI names directly.
That is what keeps the namespace footprint out of user code.

**Out of scope permanently:** third-party custom nodes (D2), any HTTP surface,
concurrency (one graph at a time per process, D3), and running workflow JSON at
runtime — codegen replaced the runtime executor.

## 2. Decisions

| # | Decision | Consequence |
|---|---|---|
| D1 | This project drives ComfyUI; ComfyUI never calls our code | No custom-node package needed |
| D2 | Third-party custom nodes out of scope **forever** | The footprint guarantees hold permanently |
| D3 | One graph at a time per process | No locking, no worker pool |
| D5 | Upstream tracked rarely, major versions only | Golden-workflow suite stays small |
| D6 | Node functions return the node's **outputs** | Tensors directly, not UI dicts |
| D7 | **No result caching** | Each `run_graph()` re-executes everything, including checkpoint loads |
| D8 | **No vendoring** — use the existing local checkout | `comfy_root` is a configured path (§10) |
| D9 | **Stub `PromptServer`** instead of importing the real `server.py` | Footprint 18 → 11; no aiohttp. Real one still available via `start(use_real_server=True)` |
| D10 | **Keep `model_management`; do not default to `gpu_only`** | It is non-optional (C23), and gpu-only *disables* offload (C24) |
| D11 | Extending ComfyUI is in scope, provided the checkout is never polluted | No writing into the checkout — test-enforced |
| D12 | **API-format input only** | Codegen rejects UI-format with a clear message |
| D13 | Locally-authored nodes and patches are in scope; D11's "no mutation" narrows to "no *unmanaged* mutation" | Mutate only through `patch_attr`, which records and reverses; leave nothing applied at rest — test-enforced via `active_patches()`. Prefer ComfyUI's own ModelPatcher wrapper API (`comfy_bridge.hooks`), which is per-clone and needs no patching. Local nodes register into the Runtime's *copy* of the node table. See `docs/extending.md`. |

## 3. Measured constraints

Every row below was measured by running code, not by reading source. Where the
two disagreed, running code won — twice, on C17 and C18, both of which were
wrong in earlier drafts. **Grep the source to find things; measure the registry
to make claims.**

### The one that matters most

| # | Finding |
|---|---|
| **C29** | `execution.py:751` wraps prompt execution in `torch.inference_mode()`. **Generated code MUST do the same.** Without it every node call retains its autograd graph and successive calls chain: measured +7.2 GB retained per VAE tile, climbing 0.01 → 7.21 → 14.42 → 21.62 GB → OOM. With it the same decode peaks at **0.11 GB**. |
| **C38** | Autograd retention was the single root cause behind **every** memory failure in this project — the sampling OOM, the decode OOM, the per-tile climb in an external decoder. Before attributing a ComfyUI memory failure to model size or an upstream bug, confirm the call is under `inference_mode`. |

`run_graph()` carries `@torch.inference_mode()`; `invoke()` adds `torch.no_grad()`
underneath so a caller poking a single node is safe too. Consequence for callers:
outputs are inference-mode tensors, so anything feeding them into an autograd
context must `.clone()` first.

### Startup ordering

| # | Finding |
|---|---|
| C4 | Device env vars must be set **before torch is imported** (`main.py:83-93`). |
| C5 | `comfy.cli_args.args` comes from `parse_args([])` when `args_parsing` is False; `comfy.model_management` reads it at import time. |
| C6 | `PromptServer.__init__` sets `PromptServer.instance` and does **not** bind a port. Binding happens only in `run()`. |
| C7 | Some V3 nodes reach `PromptServer.instance` directly, so a real instance must exist (`comfy_extras/nodes_images.py:597`). |
| C8 | `init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)` is a supported configuration (`nodes.py:2541`). |
| C9 | The `sys.path.insert(0, .../comfy)` lives **inside** `init_external_custom_nodes` — skipping custom nodes avoids it entirely. |
| C12 | `send_sync` schedules via `call_soon_threadsafe`. With no loop running, callbacks accumulate unboundedly. |
| **C26** | **DynamicVRAM setup happens in `main.py` in TWO places, and both are required.** `comfy_aimdo.control.init()` early (58-70), then after `model_management` is importable: `control.init_devices(...)`, `CoreModelPatcher = ModelPatcherDynamic`, `aimdo_enabled = True` (251-282). Doing only the first leaves the legacy estimate-based patcher in place and large models **OOM mid-sample** — startup succeeds, small graphs run, so this fails silently. |
| **C27** | Upstream gates DynamicVRAM behind torch >= 2.8 unless `--enable-dynamic-vram` is passed. This env has torch 2.6, so it must be forced. |
| **C37** | **`cuda_malloc` must not be imported once torch is loaded.** It sets `PYTORCH_CUDA_ALLOC_CONF`, which torch parses at *its* import; changing it later aborts the first CUDA init. The bridge skips it when torch is already in `sys.modules`. |

### Namespace

| # | Finding |
|---|---|
| C1 | ComfyUI is **not** pip-installable — no build-system, no package config, no entry points. |
| C2 | `comfy/`, `comfy_execution/`, `comfy_extras/` have no `__init__.py`; core modules are loose files at the repo root. |
| C3 | ComfyUI imports its own modules **absolutely**. Relocating files changes nothing (PEP 328). |
| C10 | Footprint measured at **18** with the real `server.py`, **11** with the stub. Final set: `comfy`, `comfy_api`, `comfy_config`, `comfy_execution`, `comfy_extras`, `cuda_malloc`, `folder_paths`, `latent_preview`, `node_helpers`, `nodes`, `protocol`. |
| C11 | Only 4 deferred (function-scope) imports of root modules exist across `comfy_extras`/`comfy_api`/`comfy_execution`. One of them — `comfy_api/latest/__init__.py:31`'s `from server import PromptServer` — is why the stub `server` module can never be uninstalled after startup. |
| C21 | ComfyUI registers each `comfy_extras` module in `sys.modules` under its **absolute file path** (~129 keys), not a dotted name. The footprint guard filters them with `str.isidentifier()`. |
| C22 | The node layer's **entire** use of `PromptServer` is `send_progress_text` (2 sites) and `node_replace_manager.register` (1 site). `comfy.utils.ProgressBar` is independent — it dispatches via `PROGRESS_BAR_HOOK`, installed only by `main.py`'s `hijack_progress`, which we never call. |

### Node calling

| # | Finding |
|---|---|
| **C16** | V3 is the dominant convention: 446 `define_schema` nodes vs a shrinking V1 set. V3 uses `@classmethod def execute(cls, ...) -> IO.NodeOutput`. |
| **C17** | **27 shipped nodes use list I/O.** V3 nodes declare it via `Schema(is_input_list=...)`, so the attribute only exists at runtime and a source grep misses it. These are unsupported (§9); `invoke()` raises `UnsupportedNodeError`. |
| **C17a** | Truthiness trap: every V3 node exposes `OUTPUT_IS_LIST` as a per-output sequence, usually `[False]`. A non-empty list is truthy, so `if cls.OUTPUT_IS_LIST:` reports ~450 false positives. Guards must test the *contents*. |
| **C18** | **Zero shipped nodes have an async entrypoint.** The 243 `async def` in `comfy_extras` are helpers. `invoke()` is synchronous and *refuses* a coroutine rather than awaiting one, so an upstream change fails loudly. A test pins this. |
| **C19** | Output-ness is declared two ways: `OUTPUT_NODE = True` (7, V1) and `is_output_node=True` (31, V3). |
| **C20** | Dynamic graph expansion exists but is rare (3 sites). Cannot be statically generated (§9). |
| **C31** | **V3 nodes must be called on `cls.PREPARE_CLASS_CLONE(v3_data)`, not the raw class.** The raw class has `hidden = None`, so any node reading `cls.hidden.*` raises `AttributeError`. Passing `None` yields all-None fields, which is right for codegen, and keeps per-call state off the shared class. |
| **C32** | **`DynamicCombo` inputs take a dict keyed by input id, not the widget string.** `SaveVideo(codec=...)` wants `{"codec": "auto"}`; passing `"auto"` raises `TypeError: string indices must be integers`. UI `widgets_values` store the bare string, so **codegen must consult the schema and re-nest it**. |
| **C33** | A `LATENT` is not always a tensor. MiniMax H3 generates video+audio jointly, so the sampler returns a `NestedTensor` pair: `VAEDecode` takes `unbind()[0]`, `vae_decode_audio` takes `unbind()[-1]`. Anything consuming latents must check `is_nested`. |
| **C34** | **The API export flattens subgraphs for you**, emitting composite node ids like `105:24`. Ids are not valid Python identifiers, so naming must sanitise them. |
| **C35** | **Autogrow/dynamic-slot inputs arrive as dotted names** — `"values.a"` where `execute(expression, values: dict)` wants `values={"a": ...}`. Codegen must nest dotted inputs and rename the parameter. |
| **C36** | Node return names are types — `FLOAT`, `INT`, `BOOLEAN` — which snake_case onto `float`, `int`, `bool`. Generated variables must not shadow builtins; a trailing underscore is the fix. |

### Memory

| # | Finding |
|---|---|
| C13 | `history_result` carries UI dicts pointing at files on disk, not tensors. **This is why codegen wins** — calling nodes directly returns tensors. |
| C23 | `comfy.model_management` is **not** optional: 82 files under `comfy/` reference it. It is the device/dtype policy layer, not a service. |
| C24 | Loader nodes take **no device argument**. Placement comes entirely from `model_management`. `args.gpu_only` makes offload devices return the GPU, i.e. it *disables* offload. |
| **C30** | ComfyUI's decode estimate is 8-30x low, and `MiniMaxH3VideoVAE.decode_tiled` ignores `tile_x/tile_y/overlap` (it is `return self.decode(z)`). Neither matters in practice: under `inference_mode` the stock `VAEDecode` handles 243 frames at 864x480 on a 24GB card. External tiling is not required. |

## 4. Architecture

```
comfy_bridge/
  __init__.py      # public API
  bootstrap.py     # ordering-critical startup (§5.1). The ONLY module that
                   # touches sys.path or imports ComfyUI.
  _stub_server.py  # the D9 stub PromptServer
  invoke.py        # V1/V3 node-calling shim (§7)
  errors.py        # exception hierarchy (§8)
  extend.py        # local nodes + reversible patches (D13)
  hooks.py         # ModelPatcher wrapper API — patch-free extension (D13)
  memory.py        # manual VRAM control (§5.3)
  bench.py         # per-node timing and driver-level VRAM sampling
```

### §5.1 Bootstrap sequence

Runs in two phases. **Load** (steps 1-10) mutates the process and happens at most
once — several steps are not idempotent, and the device env vars can never be
applied twice because torch is imported in between. **Validate** (step 11) is
re-runnable, so a `start()` that loads and then trips the guard can be retried
with `enforce_footprint=False`.

1. **Set device env vars** before any torch import (C4). The package must not
   import torch at import time — test-enforced.
2. **Insert the configured ComfyUI root at `sys.path[0]`.**
3. **Import and configure `comfy.cli_args.args`** before `comfy.model_management`
   is imported (C5).
   3b. **`comfy_aimdo.control.init()`** — first half of DynamicVRAM (C26).
4. **Apply `cuda_malloc`** if enabled and torch is not yet loaded (C37).
5. **Configure `folder_paths`** — base/models/output/temp dirs.
   5b. **`control.init_devices()` + `CoreModelPatcher = ModelPatcherDynamic` +
   `aimdo_enabled = True`** — second half of DynamicVRAM (C26). **Omitting this
   is a silent correctness bug.**
6. **Install the stub `server` module** (D9) and construct its `PromptServer`
   (C7). Must precede `init_extra_nodes`. No port binds.
7. **Point the stub's callback** at the Runtime's `progress_callback`. With the
   real server this overrides `send_sync` instead (C12).
8. **`await init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)`** (C8).
9. **Eagerly import the footprint names**, satisfying the deferred imports (C11).
10. **Remove the `sys.path` entry.**
11. **Assert the footprint** equals the expected set, plus that `sys.modules['server']`
    is still our stub — it is not a file under the checkout, so the module scan
    cannot see it.

`start()` is idempotent and process-global; a second call with a different
configuration raises `BootstrapError`. ComfyUI holds global state
(`NODE_CLASS_MAPPINGS`, `PromptServer.instance`, `model_management` device
state), so this is not negotiable.

### §5.2 The server stub (D9)

`_stub_server.py` registers a fake `server` module in `sys.modules` before
ComfyUI loads, implementing exactly the surface in C22. Measured: footprint
18 → 11, `aiohttp` never imported, `start()` ≈ 3.8s. Shipped nodes register 8
real `NodeReplace` objects against it on every startup, so the path is exercised.

It stays installed for the process lifetime (C11). `start(use_real_server=True)`
restores upstream behaviour and widens the guard to `FOOTPRINT_REAL_SERVER`.

### §5.3 Model & VRAM management (D10)

`comfy.model_management` stays — it is the device and dtype policy layer (C23),
and loader nodes expose no device parameter (C24). What codegen removes is the
orchestration around it: `prompt_worker`'s timed `gc.collect()`, cache resets and
`unload_all_models()`. Manual control is exposed rather than automated:
`free_memory()`, `load_to_gpu()`, `offload()`.

**`gpu_only` is not the default.** It reads as the simple choice but *disables*
CPU offload (C24). On a 24GB 3090 with a 32B text encoder plus a diffusion model
plus VAE resident, that OOMs rather than degrading.

### §7 The `invoke()` shim

The one piece of real complexity — `execution.py` does non-trivial work to call a
node, and generated code must not reimplement it inline.

- **V1 vs V3 dispatch** (C16). V1: instantiate, call `getattr(inst, cls.FUNCTION)`.
  V3: `cls.PREPARE_CLASS_CLONE(None)` first (C31), then `EXECUTE_NORMALIZED`,
  unwrapping `NodeOutput.result`.
- **Synchronous throughout** (C18) — a coroutine return is refused, not awaited.
- **No list mapping** (C17) — `invoke()` refuses those 27 nodes. Implementing
  `_map_node_over_list` is the fix if one is ever needed.
- **`node_id`** is positional-only alongside `class_type`, so a node input of
  either name cannot collide with it. Generated code passes the workflow node id
  so failures name *which* node broke.

### §8 Errors

```
ComfyBridgeError
├── BootstrapError          # bad root, wrong order, double start()
├── NamespaceError          # footprint guard tripped
├── ExtensionError          # local node or patch rejected
├── CodegenError            # unparseable workflow, UI-format input
│   └── UnsupportedNodeError# dynamic expansion, lazy inputs, list I/O (§9)
└── NodeExecutionError      # a node raised; carries node_id, class_type, original
```

### §9 Known limitations

Consequences of static generation. All are detected and raise
`UnsupportedNodeError` rather than producing wrong code:

1. **Dynamic graph expansion** (C20).
2. **Lazy input evaluation** — generated code is eager. Correct, but slower.
3. **Node-driven control flow** — `ExecutionBlocker`, partial-execution targets.
4. **Graph-introspecting nodes** — those needing `dynprompt` or a real `prompt`
   dict get synthetic values.
5. **List I/O** (C17) — the 27 nodes above.

### §10 The ComfyUI checkout (D8)

- `comfy_root` resolves in order: the `start()` argument, the `COMFY_ROOT`
  environment variable, then `DEFAULT_COMFY_ROOT` in `bootstrap.py`.
- That checkout is also the model store and stays independently runnable.
- `bootstrap.py` pins `VALIDATED_COMFY_VERSION` and **warns rather than fails**
  when the checkout moves. Given D5, a nagging log is proportionate.
- On upgrade: pull ComfyUI, run the suite, regenerate golden files, review the
  diff, bump `VALIDATED_COMFY_VERSION`.

## 5. Rules for changing this code

- **`bootstrap.py` is the only module that may touch `sys.path` or import
  ComfyUI at module scope.** Everything else imports ComfyUI lazily, inside
  functions, so `import comfy_bridge` stays free of torch.
- **Never monkey-patch ComfyUI directly.** Use `patch_attr` (recorded,
  reversible) or `comfy_bridge.hooks` (per-clone, no patching). `active_patches()`
  must be empty at rest — a test asserts it.
- **Patches revert newest-first.** `Patch.revert()` refuses to revert out of
  order, because restoring an older value while a newer patch is live leaves the
  process mutated with nothing recording it. Use `revert_all_patches()`.
- **Never write into the ComfyUI checkout.**
- **New public API goes in `__init__.py`'s `__all__`.**
- **One resolver per concept.** `memory.as_patcher` / `require_patcher` is the
  only place that works out what a ModelPatcher is; `hooks` delegates to it.
  Two implementations that disagreed is how the "silently skips every VAE" bug
  got written the first time.

## 6. Testing

```bash
pytest                                  # 69 tests, ~5s
COMFY_BRIDGE_ALLOW_BUSY_GPU=1 pytest    # override the busy-GPU refusal
```

The suite starts ComfyUI and allocates real VRAM, so `conftest.py` refuses to run
while the GPU is busy — a skip would report green while covering nothing. A
pre-push hook runs it in the `gygax` conda env, warns on any test over 1s and
fails on any over 5s; `git push --no-verify` bypasses it.

What the guards are for:

| Test | Guards against |
|---|---|
| `sys.modules` delta | Silent namespace regression |
| Polluting names absent (`tests`, `sample`, `sd`, `options`, `ops`) | The `nodes.py:2334` `sys.path[0]` insert firing |
| DynamicVRAM wired | Doing only half of C26 |
| No torch at import | Breaking C4 ordering |
| `sys.path` clean, no port bound | Path leakage, accidental `run()` |
| Deferred-import probe | Path removal breaking lazy imports (C11) |
| List-I/O set pinned at 27 | The unsupported set growing silently |
| No async entrypoints | C18 flipping |
| No ComfyUI mutation at rest | Violating D13 |

Still missing: the golden-file and CPU-execution suites (M4) — see
`backlog/m4-golden-suite.md`. Those are the only tests that would catch ComfyUI
changing underneath us.

## 7. Status

M0–M3 complete: bootstrap, `invoke()`, codegen parse/emit/coverage. The
round-trip sweep covers all 591 shipped nodes: 564 codegen, 27 list-I/O
unsupported, 0 unexplained failures. M4 (golden suite) is deferred.
