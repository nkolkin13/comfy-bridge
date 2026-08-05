# M4 — golden suite

**Status:** deferred 2026-08-03
**Spec:** §11 test plan, §12 milestone M4

## Why it matters

Everything else in the test suite checks that the code is self-consistent. This
is the only thing that would catch **ComfyUI changing underneath us** — a node
gaining an input, a return type being renamed, sampling numerics shifting.

Per D5 upstream is tracked rarely and deliberately, so the upgrade ritual is:
pull ComfyUI → run the suite → regenerate golden files → review the diff. Without
golden files that ritual has nothing to review, and an upstream bump lands
silently.

## Two tests

### 1. Golden-file (cheap, every commit)

Commit the emitted `.py` for each fixture workflow; regenerate and byte-compare.

Determinism is already proven (`test_emission_is_deterministic`), so this adds
the *pinning*: a changed signature upstream shows up as a readable source diff
rather than a behaviour change nobody noticed.

Fixture available now: `comfy-codegen/tests/fixtures/video_minimax_h3_t2v_api.json`.

### 2. Generated-code execution (expensive, on demand)

Run a generated graph on CPU and compare output hashes.

**The blocker is fixture weight.** The MiniMax H3 workflow needs ~17 minutes of
sampling on a 3090 and four large models — unusable for CI. Options, roughly in
order of preference:

- **A stub-node graph.** ComfyUI ships testing nodes under
  `tests/execution/testing_nodes/` in the checkout. Fastest and dependency-free,
  but exercises the emitter rather than real model behaviour.
- **A small real graph** (SD1.5 512x512, 4 steps, CPU). Genuine coverage of the
  node layer; needs a checkpoint present, which the current `models/` does not
  have — it holds Krea2/MiniMax/Qwen only.
- **Mark it `@pytest.mark.slow`** and run the MiniMax graph manually before an
  upgrade. Honest, zero fixture work, but only as reliable as the person
  remembering.

Recommendation: stub-node graph for CI, plus a documented manual run of the
MiniMax workflow as part of the upgrade ritual.

## Definition of done

- [ ] Golden `.py` committed for at least one fixture, byte-compared in CI
- [ ] A CPU-executable fixture graph with hashed outputs
- [ ] Upgrade runbook in `docs/` — pull, test, regenerate, review, bump
      `VALIDATED_COMFY_VERSION` in `bootstrap.py`

## Watch out

`bootstrap.py` pins `VALIDATED_COMFY_VERSION = "0.30.0"` and *warns* rather than
fails when the checkout moves (spec §10). That warning is currently the only
upstream-drift signal. Golden files are what turn it into something actionable.
