# List-mapping support (the 27 refused nodes)

**Status:** deferred — implement only if you need one of these nodes
**Spec:** C17, C17a, §9

## What's refused

`invoke()` raises `UnsupportedNodeError` for any node with `INPUT_IS_LIST` or
`OUTPUT_IS_LIST` switched on. 27 shipped nodes qualify, in two families:

- **Batching / tiling** — `RebatchLatents`, `RebatchImages`,
  `SplitImageToTileList`, `ImageMergeTileList`, `ImageGrid`, `CreateList`,
  `MergeImageLists`, `MergeTextLists`, `SeedVR2TemporalChunk`,
  `SeedVR2TemporalMerge`, `WanDancerPadKeyframesList`
- **Training datasets** — `TrainLoraNode`, `LoadImageDataSetFromFolder`,
  `MakeTrainingDataset`, `ShuffleDataset`, `ResolutionBucket`, and siblings

Full list is pinned in `comfy-codegen/tests/test_roundtrip.py`.

## Correcting the record

CLAUDE.md originally claimed **zero** shipped nodes used list I/O (C17), which
made skipping `_map_node_over_list` look free. That was wrong, and it's worth
understanding why so the mistake isn't repeated:

- The original survey grepped for literal `INPUT_IS_LIST = ...` assignments. V3
  nodes never write those — they declare it via `Schema(is_input_list=...)`, so
  the attribute only materialises at runtime.
- **Grep the source to find things; measure the registry to make claims.**

There's also a truthiness trap (C17a): every V3 node exposes `OUTPUT_IS_LIST` as
a per-output sequence, usually `[False]`. A non-empty list is truthy, so
`if cls.OUTPUT_IS_LIST:` reports ~450 false positives. `_uses_list_io()` in
`comfy_bridge/invoke.py` tests the *contents*; keep it that way.

## What implementing it involves

Port the mapping branch of `execution.py` (`_map_node_over_list`, around
`execution.py:245-330`):

- `INPUT_IS_LIST` — the node wants every input as a list; call it once with all
  values wrapped
- otherwise — broadcast: find the longest list input, call the node once per
  index, zip scalars across
- `OUTPUT_IS_LIST` — per-output flags decide whether each result is spliced or
  nested when results are recombined

Codegen also needs to decide how a list-valued edge appears in generated Python.
Right now every edge is a single value; list edges would break the
one-variable-per-output model in `emit.py`.

## Definition of done

- [ ] `invoke()` handles both flags, matching `execution.py` semantics
- [ ] Round-trip expectations updated (`EXPECTED_OK` 564 → 591,
      `EXPECTED_UNSUPPORTED` 27 → 0)
- [ ] A generated graph using `RebatchImages` runs and matches upstream output
