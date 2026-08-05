"""Local extensions: our own nodes, and reversible patches to ComfyUI (D13).

D11 said "extending ComfyUI is fine; mutating it is not", and M0's test asserted
that literally — nothing in ``comfy.*`` may be reassigned. Optimization work
breaks that assumption: swapping an attention implementation or an ops class is
exactly a module-level reassignment, and there is no way to reach some of those
sites through ComfyUI's own extension API.

So the invariant is narrowed rather than dropped. Mutation is allowed, but only
through :func:`patch_attr`, which means every mutation is:

* **recorded** — :func:`active_patches` lists everything currently applied, so a
  test can assert the process is clean between benchmark runs;
* **reversible** — the original value (or its absence) is restored exactly, which
  is what makes A/B measurement in a single process trustworthy;
* **in memory only** — the checkout on disk is never written to.

Prefer :mod:`comfy_bridge.hooks` when a hook exists. ComfyUI's ModelPatcher
wrapper API covers most inference-level work per-model and per-clone, which is
strictly better than a process-global patch. Reach for ``patch_attr`` when the
thing you need to replace is a module global.

Custom nodes are registered into the Runtime's node table, which is a *copy* of
``nodes.NODE_CLASS_MAPPINGS`` — so registering ours cannot corrupt ComfyUI's, and
comfy-codegen picks them up for free because it resolves classes through
``node_class()`` (see comfy_codegen/parse.py:121). Third-party custom nodes
remain out of scope forever (D2); this is for code we write.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Iterable

from .errors import ExtensionError

log = logging.getLogger("comfy_bridge.extend")

__all__ = [
    "Patch",
    "PatchSet",
    "active_patches",
    "patch_attr",
    "register_node",
    "registered_nodes",
    "revert_all_patches",
    "unregister_node",
]


# --- our own nodes ---------------------------------------------------------

#: class_type -> class, for everything registered through this module.
_REGISTERED: dict[str, Any] = {}

#: class_type -> (shipped class, its display name) that a registration shadowed,
#: so unregister_node can put both back. The display name has to be kept here
#: too: register_node overwrites runtime.display_names, and ComfyUI's own name
#: ("Empty Latent Image") is not recoverable from the class afterwards.
_SHADOWED: dict[str, tuple[Any, str | None]] = {}


def _validate_node_class(cls: Any, class_type: str) -> None:
    """Check the contract both invoke() and comfy-codegen rely on.

    Fail at registration rather than three hours into a benchmark run. The
    required surface is small because we only support the V1 node shape plus the
    V3 shape ComfyUI's own nodes use.
    """
    from .invoke import list_io_guard

    if not isinstance(cls, type):
        raise ExtensionError(f"{class_type}: expected a class, got {type(cls).__name__}")

    input_types = getattr(cls, "INPUT_TYPES", None)
    if not callable(input_types):
        raise ExtensionError(
            f"{class_type}: needs an INPUT_TYPES() classmethod returning "
            '{"required": {...}, "optional": {...}} — comfy-codegen calls it to '
            "build the generated function signature (comfy_codegen/parse.py:102)."
        )
    try:
        declared = input_types()
    except Exception as exc:
        raise ExtensionError(f"{class_type}: INPUT_TYPES() raised {exc!r}") from exc
    if not isinstance(declared, dict):
        raise ExtensionError(
            f"{class_type}: INPUT_TYPES() returned {type(declared).__name__}, want dict"
        )

    return_types = getattr(cls, "RETURN_TYPES", None)
    if not isinstance(return_types, (tuple, list)):
        raise ExtensionError(
            f"{class_type}: needs RETURN_TYPES as a tuple, got {return_types!r}. "
            "Use () for an output node that returns nothing."
        )

    fn_name = getattr(cls, "FUNCTION", None)
    if not isinstance(fn_name, str):
        raise ExtensionError(f"{class_type}: needs FUNCTION set to a method name")
    if not hasattr(cls, fn_name):
        raise ExtensionError(f"{class_type}: FUNCTION={fn_name!r} but no such attribute")
    if fn_name.startswith("EXECUTE_NORMALIZED") and not hasattr(
        cls, "PREPARE_CLASS_CLONE"
    ):
        raise ExtensionError(
            f"{class_type}: declares a V3 entrypoint but has no PREPARE_CLASS_CLONE. "
            "Subclass comfy_api.latest.ComfyNode rather than imitating its surface "
            "(C31), or use the V1 shape."
        )

    names = getattr(cls, "RETURN_NAMES", None)
    if names is not None and len(names) != len(return_types):
        raise ExtensionError(
            f"{class_type}: RETURN_NAMES has {len(names)} entries but RETURN_TYPES "
            f"has {len(return_types)}"
        )

    # Same refusal invoke() would make later, surfaced now (C17).
    list_io_guard(cls, class_type)


def register_node(
    cls: Any = None,
    /,
    class_type: str | None = None,
    *,
    display_name: str | None = None,
    replace: bool = False,
) -> Any:
    """Make a locally-authored node visible to ``invoke()`` and comfy-codegen.

    Usable bare or parameterised as a decorator::

        @register_node
        class BenchmarkTimer: ...

        @register_node(class_type="Int8Linear", replace=False)
        class Int8LinearPatch: ...

    ``start()`` must have run first — the Runtime owns the node table. Pass
    ``replace=True`` to deliberately shadow a shipped node; the original is
    remembered and :func:`unregister_node` puts it back.
    """
    if cls is None:
        return functools.partial(
            register_node,
            class_type=class_type,
            display_name=display_name,
            replace=replace,
        )

    from .bootstrap import get_runtime

    runtime = get_runtime()
    name = class_type or cls.__name__
    _validate_node_class(cls, name)

    existing = runtime.nodes.get(name)
    if existing is not None and existing is not cls:
        if not replace:
            kind = "another local node" if name in _REGISTERED else "a shipped node"
            raise ExtensionError(
                f"{name!r} is already registered by {kind} ({existing!r}). Pass "
                "replace=True if shadowing it is the intent."
            )
        if name not in _SHADOWED and name not in _REGISTERED:
            _SHADOWED[name] = (existing, runtime.display_names.get(name))
            log.warning("local node %r now shadows the shipped %r", name, existing)

    runtime.nodes[name] = cls
    runtime.display_names[name] = display_name or name
    _REGISTERED[name] = cls
    return cls


def unregister_node(class_type: str) -> None:
    """Remove a locally-registered node, restoring any shipped node it shadowed."""
    from .bootstrap import get_runtime

    runtime = get_runtime()
    if class_type not in _REGISTERED:
        raise ExtensionError(f"{class_type!r} was not registered by register_node()")
    del _REGISTERED[class_type]
    shadowed = _SHADOWED.pop(class_type, None)
    if shadowed is None:
        runtime.nodes.pop(class_type, None)
        runtime.display_names.pop(class_type, None)
        return
    original, display_name = shadowed
    runtime.nodes[class_type] = original
    if display_name is None:
        runtime.display_names.pop(class_type, None)
    else:
        runtime.display_names[class_type] = display_name


def registered_nodes() -> dict[str, Any]:
    """The locally-authored nodes currently registered, class_type -> class."""
    return dict(_REGISTERED)


# --- reversible patches ----------------------------------------------------


class _Missing:
    """Marker for 'the attribute did not exist before we patched it'."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()

#: Every Patch currently applied, in application order.
_ACTIVE: list[Patch] = []


class Patch:
    """One reversible attribute replacement.

    Constructed unapplied, so it works as a context manager *or* as a long-lived
    handle::

        with patch_attr(attention, "optimized_attention", mine):
            ...                       # scoped

        p = patch_attr(attention, "optimized_attention", mine)
        p.apply()
        ...                           # spans several benchmark runs
        p.revert()
    """

    __slots__ = ("target", "attr", "value", "name", "_original", "_applied")

    def __init__(
        self, target: Any, attr: str, value: Any, *, name: str | None = None
    ) -> None:
        if not isinstance(attr, str):
            raise ExtensionError(f"attr must be a string, got {attr!r}")
        self.target = target
        self.attr = attr
        self.value = value
        self.name = name or f"{getattr(target, '__name__', target)}.{attr}"
        self._original: Any = _MISSING
        self._applied = False

    @property
    def applied(self) -> bool:
        return self._applied

    def apply(self) -> Patch:
        if self._applied:
            return self
        self._original = getattr(self.target, self.attr, _MISSING)
        setattr(self.target, self.attr, self.value)
        self._applied = True
        _ACTIVE.append(self)
        log.debug("patched %s", self.name)
        return self

    def revert(self, *, force: bool = False) -> None:
        """Restore the original value, or remove the attribute if there was none.

        **Refuses to revert out of order.** If something replaced the attribute
        after we did, this patch no longer knows the current value's provenance:
        restoring ``_original`` would discard the newer patch *and* leave that
        patch's value applied while :func:`active_patches` reports the process
        clean. That combination is how a benchmark sweep produces numbers nobody
        can reproduce, so it raises instead.

            a = patch_attr(mod, "x", "A").apply()
            b = patch_attr(mod, "x", "B").apply()
            a.revert()          # ExtensionError — b is still on top
            revert_all_patches()  # correct: newest first

        Use :func:`revert_all_patches` (or a :class:`PatchSet`) to unwind a stack
        in the right order. ``force=True`` reverts anyway, for the case where the
        foreign change is known and intended.
        """
        if not self._applied:
            return
        current = getattr(self.target, self.attr, _MISSING)
        if current is not self.value and not force:
            raise ExtensionError(
                f"{self.name} changed since it was patched (found {current!r}, "
                f"expected {self.value!r}) — most likely another patch is stacked "
                "on top of it. Reverting now would discard that patch and leave "
                "its value applied with nothing recording it. Revert newest-first "
                "(revert_all_patches() does this), or pass force=True if "
                "discarding the newer value is what you mean."
            )
        if self._original is _MISSING:
            try:
                delattr(self.target, self.attr)
            except AttributeError:
                pass
        else:
            setattr(self.target, self.attr, self._original)
        self._applied = False
        self._original = _MISSING
        try:
            _ACTIVE.remove(self)
        except ValueError:
            pass
        log.debug("reverted %s", self.name)

    def __enter__(self) -> Patch:
        return self.apply()

    def __exit__(self, *exc: object) -> None:
        self.revert()

    def __repr__(self) -> str:
        state = "applied" if self._applied else "pending"
        return f"<Patch {self.name} ({state})>"


def patch_attr(target: Any, attr: str, value: Any, *, name: str | None = None) -> Patch:
    """Build an unapplied :class:`Patch`. Entering it or calling ``apply()`` arms it.

    The one sanctioned way to mutate ComfyUI in memory. Module globals are the
    intended target — ``comfy.ldm.modules.attention.optimized_attention``,
    ``comfy.ops.disable_weight_init.Linear.forward``, and friends.
    """
    return Patch(target, attr, value, name=name)


class PatchSet:
    """A named group of patches applied and reverted as a unit.

    An optimization usually is not one replacement — an INT8 linear path touches
    the ops class, its forward, and possibly the attention dispatch. Grouping
    them means a benchmark can toggle the whole thing by name, and a partial
    failure rolls back rather than leaving a half-patched process behind.
    """

    def __init__(self, name: str, patches: Iterable[Patch] = ()) -> None:
        self.name = name
        self.patches: list[Patch] = list(patches)

    def add(self, target: Any, attr: str, value: Any) -> Patch:
        patch = Patch(target, attr, value, name=f"{self.name}:{attr}")
        self.patches.append(patch)
        return patch

    def apply(self) -> PatchSet:
        done: list[Patch] = []
        try:
            for patch in self.patches:
                patch.apply()
                done.append(patch)
        except Exception:
            # Rollback must not fail: a revert raising here would mask the error
            # that caused the rollback and leave the set half-applied.
            for patch in reversed(done):
                patch.revert(force=True)
            raise
        return self

    def revert(self) -> None:
        for patch in reversed(self.patches):
            patch.revert()

    @property
    def applied(self) -> bool:
        return bool(self.patches) and all(p.applied for p in self.patches)

    def __enter__(self) -> PatchSet:
        return self.apply()

    def __exit__(self, *exc: object) -> None:
        self.revert()

    def __repr__(self) -> str:
        return f"<PatchSet {self.name!r} ({len(self.patches)} patches)>"


def active_patches() -> tuple[Patch, ...]:
    """Everything currently applied, in application order.

    A benchmark harness should assert this is empty between configurations.
    """
    return tuple(_ACTIVE)


def revert_all_patches() -> int:
    """Revert every applied patch, newest first. Returns how many were undone.

    This is the sanctioned way to get back to a clean process, and newest-first
    is the order that makes a stack of patches on one attribute unwind correctly.
    Because reaching a clean state is the postcondition callers depend on — test
    teardown, the gap between two benchmark configurations — it forces through
    any value that drifted, logging each one rather than stopping half-unwound.
    Drift here means something mutated an attribute outside the ledger, which is
    worth seeing but not worth leaving the process half-patched over.
    """
    count = 0
    for patch in reversed(list(_ACTIVE)):
        current = getattr(patch.target, patch.attr, _MISSING)
        if current is not patch.value:
            log.warning(
                "%s drifted before revert_all_patches (found %r); restoring the "
                "pre-patch value anyway",
                patch.name,
                current,
            )
        patch.revert(force=True)
        count += 1
    return count
