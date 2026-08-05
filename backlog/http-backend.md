# HTTP backend (out-of-process)

**Status:** deferred by design, not oversight
**Spec:** §3 approach selection (Tier 1), §12 "deferred to follow-up"

## Idea

Run ComfyUI as a normal server (`python main.py`) in its own environment and have
generated modules talk to it over HTTP/WebSocket instead of importing it. The
generated code's shape doesn't change — only which `invoke()` it imports.

## What it would buy

The three arguments that originally motivated process isolation, of which only
one still stands unambiguously:

1. **Namespace isolation** — largely solved already. The stub `PromptServer` (D9)
   got the footprint down to 11 top-level modules, and `tests` — the collision
   that would actually break a host project's pytest run — never appears.
2. **Dependency isolation** — was overstated. `requirements.txt` has five `==`
   pins, all Comfy-org's own packages that nothing else wants.
3. **Crash isolation and remote/multi-GPU** — still genuinely only available
   out-of-process. A segfault in a CUDA kernel takes down the host process today.

## What it would cost

- Serialising tensors over the wire, versus the current in-memory returns (D4)
- Losing the direct `comfy.model_management` control that `memory.py` exposes,
  since offload decisions would live server-side
- Output round-trips through disk — `history_result` carries UI dicts, not
  tensors (C13). This is the thing codegen was chosen to avoid.

## Sketch

Reintroduce the backend abstraction from spec v1 §4.2 (dropped in v2):

```python
class Backend(Protocol):
    def invoke(self, class_type: str, /, **kwargs) -> tuple: ...
```

`comfy_bridge.invoke` becomes a thin dispatch over the configured backend. The
HTTP one can't map node-by-node onto `/prompt` — that endpoint takes a whole
graph — so it would need to either batch a generated module's calls into one
prompt submission, or run a small RPC shim inside the server process. The second
is more faithful and more work.

## Prerequisite

Do M4 first. Spec §11 lists a **backend parity** test — same workflow through
both backends, same outputs — and that is only meaningful once golden outputs
exist to compare against.
