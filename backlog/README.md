# Backlog

Deferred work for `comfy-bridge` and `comfy-codegen`. One file per item.

The authoritative design record is [`../docs/spec.md`](../docs/spec.md) — decisions
(D1–D12), measured constraints (C1–C38), and milestones. Read it before picking
anything up here; several items exist *because* of a specific constraint, and the
spec explains the reasoning that a one-line summary loses.

## Status as of 2026-08-03

| Milestone | State |
|---|---|
| M0 — bootstrap | done |
| M1 — `invoke()` shim + memory control | done |
| M2 — codegen parse + emit | done, verified end-to-end on MiniMax H3 |
| M3 — codegen coverage | done — 564/591 nodes codegen, 27 unsupported, 0 unexplained |
| M4 — golden suite | **deferred** → [`m4-golden-suite.md`](m4-golden-suite.md) |

52 tests green (30 bridge, 22 codegen).

## Items

| File | What | Priority |
|---|---|---|
| [`m4-golden-suite.md`](m4-golden-suite.md) | Golden-file + CPU execution tests | highest — this is what makes an upstream bump safe |
| [`list-mapping-support.md`](list-mapping-support.md) | The 27 nodes `invoke()` refuses | on demand — only if you need one |
| [`http-backend.md`](http-backend.md) | Out-of-process backend (Tier 1) | low — deferred by design |
| [`generated-code-mutable-defaults.md`](generated-code-mutable-defaults.md) | `codec={'codec': 'auto'}` smell | cosmetic |
| [`output-directory-and-d11.md`](output-directory-and-d11.md) | Outputs land in the ComfyUI checkout | needs a decision, not code |

## The one thing to remember

**C38.** Autograd retention was the root cause of every memory failure in this
project — the sampling OOM, the decode OOM, and the per-tile climb in a
hand-written decoder. Peak went from OOM-at-22.5 GB to 0.11 GB once calls ran
under `inference_mode`. Before attributing any ComfyUI memory failure to model
size or an upstream bug, check the call is under `torch.inference_mode()`.
Generated `run_graph()` carries it; `invoke()` adds `no_grad` underneath.
