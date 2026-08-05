# Outputs land inside the ComfyUI checkout

**Status:** needs a decision, not code
**Spec:** D11 (no pollution), §5.1 step 5

## The situation

`folder_paths` defaults the output directory to `<comfy_root>/output`, so running
a generated graph writes `.mp4`/`.png` files into the ComfyUI checkout —
`/home/nick/Projects/ComfyUI/output/video/MiniMax_H3_00005_.mp4`, for instance.

D11 says extending ComfyUI's behaviour is fine "provided the checkout is never
polluted". The hard invariants that phrase was written for are enforced and
tested: no monkey-patching ComfyUI modules, no writing into the *source* tree.
Whether writing to `output/` counts is a judgement call that hasn't been made.

## The case for leaving it

- `output/` is gitignored and exists precisely to receive generated files
- Results land where the ComfyUI web UI would also put them, so the two installs
  stay interchangeable and outputs show up in the UI's gallery
- The 83GB of models already live in that checkout; it is the data root, not just
  a source tree

## The case for redirecting

- A bridge-driven run and a UI-driven run become indistinguishable after the fact
- The checkout stops being a clean read-only dependency, which was part of the
  original argument for pointing at it by path rather than vendoring (D8)
- Sharing a `git`-managed directory between "code we don't touch" and "output we
  generate constantly" invites accidents

## If redirecting

Already supported, no code needed:

```python
comfy_bridge.start(output_dir="/home/nick/Projects/comfy-outputs")
```

`temp_dir` and `input_dir` take the same treatment. Note `start()` is a
process-global singleton, so this has to be decided at startup — a generated
module's `__main__` block calls bare `start()` today, and `emit.py` would want a
matching comment if redirect becomes the convention.

## Definition of done

- [ ] Decide
- [ ] If redirecting: pick a location, update `examples/minimax_h3_t2v.py` and
      `emit.py`'s `__main__` block, note it in the spec under D11
- [ ] If not: add a sentence to D11 clarifying that `output/`, `temp/` and
      `input/` are data directories and explicitly outside the invariant
