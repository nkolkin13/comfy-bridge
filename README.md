# comfy-bridge

Run ComfyUI's shipped nodes in-process, as ordinary Python functions — no web UI,
no HTTP, no workflow JSON at runtime.

```python
import comfy_bridge

rt = comfy_bridge.start(device=0)
print(len(rt.nodes), "nodes available")

model, clip, vae = comfy_bridge.invoke(
    "CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors"
)
```

`comfy_bridge` is the runtime half of a two-part project. The other half,
`comfy-codegen`, translates an API-format workflow JSON into a readable Python
module that imports this package. Generated modules only ever see
`comfy_bridge.*` names — keeping ComfyUI's loose top-level modules out of the
host namespace is a large part of what this package is for.

## What it does

- **Owns the order-sensitive startup.** Device env vars before torch, CLI args
  before `model_management`, DynamicVRAM wired in both of the two places
  `main.py` wires it, node loading with custom and API nodes off.
- **Contains the namespace footprint.** Starting ComfyUI normally introduces 18
  top-level modules. A stub `PromptServer` (implementing exactly the three call
  sites the node layer actually uses) brings that to 11, skips `aiohttp`
  entirely, and is asserted on every `start()`.
- **Calls nodes uniformly.** `invoke()` handles V1 vs V3 dispatch, V3 class
  clones and hidden inputs, and async node functions, and refuses the node types
  that cannot work under static generation.
- **Never mutates the ComfyUI checkout.** It is read, never patched or written
  to. Local extensions go through `patch_attr` (recorded and reversible) or
  ComfyUI's own ModelPatcher wrapper API — both test-enforced to leave nothing
  applied at rest.

Binds no port. Starts in roughly 3.8s.

## Requirements

- Python ≥ 3.10
- A local ComfyUI checkout, validated against **v0.30.0**
- Whatever env satisfies ComfyUI's own `requirements.txt` (torch etc.)

`comfy-bridge` declares no runtime dependencies on purpose — pinning torch here
would fight with the environment that actually has to satisfy ComfyUI.

## Install

```bash
pip install -e '.[dev]'
git config core.hooksPath .githooks   # enable the pre-push test hook
```

The hook path is local git config, so it has to be set once per clone.

The ComfyUI checkout is located in this order:

1. the `comfy_root` argument to `start()`
2. the `COMFY_ROOT` environment variable
3. `DEFAULT_COMFY_ROOT` in `comfy_bridge/bootstrap.py`

The default is a local path from the machine this was developed on; set
`COMFY_ROOT` or pass `comfy_root=` rather than relying on it.

## Usage

`start()` is process-global and idempotent. ComfyUI keeps module-level state
that cannot be meaningfully re-initialised, so a second call with a *different*
configuration raises `BootstrapError` instead of silently reusing the first.

```python
rt = comfy_bridge.start(
    comfy_root="/path/to/ComfyUI",
    device=0,
    vram_mode="normal",       # "gpu-only" disables CPU offload — see below
    output_dir="/path/to/outputs",
)
```

`vram_mode="normal"` is the default deliberately. `gpu-only` reads as the simple
choice but *disables* offload: text encoder, intermediate and VAE devices all
become the GPU, which OOMs rather than degrading once a large text encoder,
diffusion model and VAE need to be resident at once.

### Memory

```python
comfy_bridge.free_memory()          # unload everything + empty cache
comfy_bridge.load_to_gpu(model)
comfy_bridge.offload(vae)           # accepts VAE or ModelPatcher
```

### Extending

Local nodes and patches are supported; polluting the checkout is not. See
[`docs/extending.md`](docs/extending.md).

```python
@comfy_bridge.register_node          # into the Runtime's copy of the node
class MyNode: ...                    # table, not ComfyUI's

with comfy_bridge.patch_attr(some_module, "ATTR", value):
    ...                              # reverted on exit
```

### Benchmarking

`comfy_bridge.bench` provides per-node timing, device-wide VRAM sampling and
before/after comparison:

```python
with comfy_bridge.bench.profile("baseline") as report:
    run_graph()
print(report.table())
```

## The one thing to know

Autograd retention was the root cause of **every** memory failure in this
project — the sampling OOM, the decode OOM, and a per-tile climb in a
hand-written decoder. Peak went from OOM-at-22.5 GB to 0.11 GB once the calls
ran under `torch.inference_mode()`.

ComfyUI gets this from `execution.py`, which wraps whole-prompt execution. Since
this project replaces the executor, generated `run_graph()` functions carry
`@torch.inference_mode()` and `invoke()` adds `torch.no_grad()` underneath. One
consequence for callers: outputs are inference-mode tensors, so anything feeding
them into an autograd context must `.clone()` first.

Before blaming a ComfyUI memory failure on model size or an upstream bug, check
the call is under `inference_mode`.

## Tests

```bash
pytest
```

69 tests, about five seconds. The suite starts ComfyUI and allocates real VRAM,
so it refuses to run while the GPU is busy — a skip would report green while
covering nothing. Override with `COMFY_BRIDGE_ALLOW_BUSY_GPU=1` when you know it
is safe.

A pre-push hook runs the suite in the `gygax` conda env, warns on any test
slower than 1s and fails on any slower than 5s. `git push --no-verify` bypasses
it.

## Documentation

| | |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | The operative record — decisions, measured constraints, bootstrap ordering, and the rules for changing this code. Read this first. |
| [`docs/extending.md`](docs/extending.md) | Adding local nodes and patches without polluting the checkout. |
| [`backlog/`](backlog/) | Deferred work, one file per item. |

Two constraints were wrong in earlier drafts and were only caught by running
code, which is worth generalising: **grep the source to find things, measure the
registry to make claims.**

## Status

| Milestone | State |
|---|---|
| M0 — bootstrap | done |
| M1 — `invoke()` shim + memory control | done |
| M2 — codegen parse + emit | done, verified end-to-end on MiniMax H3 |
| M3 — codegen coverage | done — 564/591 nodes codegen, 27 unsupported, 0 unexplained |
| M4 — golden suite | **deferred** → [`backlog/m4-golden-suite.md`](backlog/m4-golden-suite.md) |

Third-party custom nodes are permanently out of scope.

## Backlog

Deferred work, one file per item. Read `CLAUDE.md` before picking anything up —
several items exist *because* of a specific constraint, and the reasoning is
what a one-line summary loses.

| File | What | Priority |
|---|---|---|
| [`m4-golden-suite.md`](backlog/m4-golden-suite.md) | Golden-file + CPU execution tests | highest — this is what makes an upstream bump safe |
| [`list-mapping-support.md`](backlog/list-mapping-support.md) | The 27 nodes `invoke()` refuses | on demand — only if you need one |
| [`http-backend.md`](backlog/http-backend.md) | Out-of-process backend | low — deferred by design |
| [`generated-code-mutable-defaults.md`](backlog/generated-code-mutable-defaults.md) | `codec={'codec': 'auto'}` smell | cosmetic |
| [`output-directory-and-d11.md`](backlog/output-directory-and-d11.md) | Outputs land in the ComfyUI checkout | needs a decision, not code |

## License

Not yet licensed. All rights reserved.
