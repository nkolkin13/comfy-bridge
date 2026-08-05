# Generated code: mutable default arguments

**Status:** cosmetic
**Spec:** §6.2, C32

## The smell

DynamicCombo inputs are re-nested into dicts (C32), so they emit as dict
defaults:

```python
def save_video(
    video,
    filename_prefix='video/MiniMax_H3',
    format='auto',
    codec={'codec': 'auto'},      # <- mutable default
):
```

Same applies to autogrow inputs that carry literal values (C35).

## Why it hasn't been fixed

It is harmless *here*: nothing mutates the dict, and each call passes it straight
to `invoke()`. But generated code is meant to be read and edited, and a reader
who mutates `codec` in one call site would silently change every subsequent call
— exactly the classic Python trap, in a file that otherwise looks safe.

## Options

**A — leave it, document it.** One line in the module header noting the defaults
are structured values that shouldn't be mutated in place. Zero code, keeps the
signature readable.

**B — `None` sentinel.**

```python
def save_video(video, filename_prefix='video/MiniMax_H3', format='auto', codec=None):
    if codec is None:
        codec = {'codec': 'auto'}
```

Correct, but adds a branch per structured input and pushes the actual default out
of the signature — where it is most useful to a reader.

**C — emit a module-level constant.**

```python
SAVE_VIDEO_CODEC = {'codec': 'auto'}

def save_video(video, ..., codec=SAVE_VIDEO_CODEC):
```

Keeps the signature clean and makes the value greppable, but doesn't actually fix
mutability — just relocates it.

**D — freeze it.** A small immutable mapping type in `comfy_bridge`. Genuinely
correct; costs the generated file an import and a less obvious literal.

Leaning **A**, on the grounds that the design's whole premise is that generated
code is ordinary readable Python and every guard added to it works against that.

## Definition of done

- [ ] Pick one
- [ ] If A: header note in `emit.py`'s `_HEADER`
- [ ] Otherwise: implement in `emit.py`, update the golden files
