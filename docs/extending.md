# Extending the bridge: custom nodes, architecture patches, inference patches

Written for the acceleration work (Step 0). Everything here is for **code we
write**. Third-party custom nodes remain out of scope forever (D2).

The headline: **most of what acceleration work needs already exists**, because
ComfyUI ships a wrapper/callback system that reaches every level a zero-knowledge
method cares about. Reach for a patch only when there is no hook.

Pick the lowest-numbered mechanism that works:

| # | Mechanism | Scope | Reversible | Use for |
|---|-----------|-------|------------|---------|
| 1 | `hooks.add_wrapper` | one model clone | drop the clone | step caching, timing, skipping model calls |
| 2 | `hooks.override_attention` | one model clone | drop the clone | sliding-window / quantized attention |
| 3 | `hooks.patch_object` | one model clone, applied at load | automatic at unload | swapping a `Linear`, a block's `forward` |
| 4 | `register_node` | the Runtime's node table | `unregister_node` | anything that must appear in a workflow |
| 5 | `patch_attr` / `PatchSet` | process-global | explicit, recorded | module globals with no hook |

Mechanisms 1–3 are the sanctioned upstream extension API — nothing is mutated,
so a baseline and a variant can coexist in one process. That property is what
makes A/B benchmarking trustworthy, and it is why the table is ordered this way.

## Import ordering

Modules that `import comfy.*` must be imported **after** `comfy_bridge.start()`.
Before that, ComfyUI is not on `sys.path` at all — and after `start()` returns,
the path entry is removed again, so only names already in `sys.modules` resolve
(C11).

Two ways to stay safe:

```python
# a) import inside the function, as comfy_bridge itself does
def my_attention(original, q, k, v, heads, **kwargs):
    from comfy.ldm.modules.attention import attention_pytorch
    ...

# b) import the module only after start()
comfy_bridge.start(dynamic_vram=True)
from myproject.accel import int8_linear     # safe from here
```

Our own extension modules live outside the checkout, so they never count toward
the C10 `sys.modules` footprint and the namespace guard is unaffected.

## 1–2. Wrappers and attention

A wrapper is called as `wrapper(executor, *args, **kwargs)` and must call
`executor(*args, **kwargs)` to continue the chain. *Not* calling it is how a
caching method returns a stale result without running the model.

Where they attach, outermost first — see `comfy_bridge/hooks.py` for the table
with upstream line numbers. `APPLY_MODEL` is universal; `DIFFUSION_MODEL`
brackets exactly the transformer forward and is the honest place to measure.

```python
from comfy_bridge import hooks

cache = {}

def reuse_if_unchanged(executor, x, timestep, **kwargs):
    """Sketch: skip the forward when the input barely moved."""
    key = int(timestep.flatten()[0])
    prev = cache.get("last_input")
    if prev is not None and (x - prev).abs().mean() < 0.01:
        return cache["last_output"]
    out = executor(x, timestep, **kwargs)
    cache["last_input"], cache["last_output"] = x, out
    return out

fast = hooks.add_wrapper(model, hooks.DIFFUSION_MODEL, reuse_if_unchanged)
latent = ksampler(model=fast, ...)      # `model` is still the clean baseline
```

Attention is intercepted the same way, with the dispatched implementation passed
in first so you can fall back per call:

```python
def windowed(original, q, k, v, heads, **kwargs):
    if q.shape[1] < 4096:                        # geometry only — no calibration
        return original(q, k, v, heads, **kwargs)
    return my_windowed_kernel(q, k, v, heads, **kwargs)

fast = hooks.override_attention(model, windowed)
```

Wrappers must be attached **before** sampling starts: `sampler_helpers.py:224`
merges `ModelPatcher.wrappers` into `model_options` at prepare-sampling time, and
changes after that point have no effect.

## 3. Architecture patches

`hooks.patch_object` reaches any attribute of the loaded model by dotted path.
ComfyUI applies these at load and restores the originals at unload
(`model_patcher.py:1107-1161`), so the shared module tree comes back clean:

```python
fast = hooks.patch_object(model, "diffusion_model.blocks.0.attn.forward", mine)
```

Caveat: the clone scopes *which patcher applies the patch*, not which memory it
touches. Two clones with conflicting object patches must not be resident at once.

## 4. Our own nodes

Only needed when the thing must appear in a workflow JSON and flow through
comfy-codegen. For code called from a generated module, a plain function is
simpler and does not need registering at all.

```python
@comfy_bridge.register_node(class_type="Int8Linear")
class Int8Linear:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "threshold": ("FLOAT", {"default": 6.0})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"

    def run(self, model, threshold):
        return (hooks.patch_object(model, "diffusion_model", quantized(...)),)
```

The contract is checked at registration, not three hours into a run:
`INPUT_TYPES()`, `RETURN_TYPES`, `FUNCTION` naming a real attribute, and
`RETURN_NAMES` matching in length. List I/O is refused, same as for shipped nodes
(C17).

Registration writes to the Runtime's node table, which is a copy — ComfyUI's own
`NODE_CLASS_MAPPINGS` is never touched. comfy-codegen resolves classes through
`node_class()`, so a registered node appears in generated code with no codegen
change. Shadowing a shipped node requires `replace=True` and is restored by
`unregister_node`.

## 5. Patching module globals

The escape hatch when there is no hook. Every patch is recorded and reversible,
and a benchmark harness should assert `active_patches() == ()` between
configurations:

```python
import comfy.ldm.modules.attention as attention

with comfy_bridge.patch_attr(attention, "optimized_attention", my_kernel):
    run_graph()
# original restored exactly, including "the attribute did not exist before"
```

`PatchSet` groups several into one atomic toggle, rolling back if any member
fails:

```python
int8 = comfy_bridge.PatchSet("int8-linear")
int8.add(comfy.ops, "disable_weight_init", MyOps)
int8.add(attention, "optimized_attention", my_attention)
with int8:
    run_graph()
```

## Benchmarking

Three levels, deliberately separate — mixing them is how a 2x becomes a rounding
error.

```python
from comfy_bridge import bench

with bench.profile("baseline") as base:
    run_graph()

timings = bench.time_model_calls(model)          # per transformer forward
latent = ksampler(model=timings.model, ...)
print(timings.summary())

print(bench.compare(base, variant))
```

`profile()` hooks `invoke()`, so it attributes time per node with **no change to
generated code**. Everything synchronises CUDA by default; without that a timer
measures how fast Python queued the work. The cost is that per-node times will
not sum exactly to wall clock — `Report.unattributed_s` reports the gap rather
than hiding it.

On VRAM: `torch.cuda.max_memory_allocated` only sees torch's caching allocator,
and this install runs DynamicVRAM (comfy-aimdo), which pages weights and may
allocate outside it. `Report` also carries the driver-level figure from
`mem_get_info` — the one that actually predicts an OOM. When they disagree,
believe the device.

## Known sharp edge: inference tensors

Generated `run_graph()` carries `@torch.inference_mode()` (C29) — without
it, VRAM climbs until it OOMs, so it is not optional. But tensors created under
inference mode are *inference tensors*, and custom kernels that stash a tensor
for reuse across calls — exactly what step-caching does — can trip errors when
one escapes the region, or when `torch.compile` / CUDA-graph capture is involved.

If that happens, swap the decorator for `@torch.no_grad()` in the generated
module. It gives the same memory behaviour for our purposes with none of the
inference-tensor semantics. `invoke()` already uses `no_grad` internally; the
outer mode wins when the two nest.
