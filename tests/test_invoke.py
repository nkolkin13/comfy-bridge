"""M1 tests — the invoke() shim and manual memory control (spec §7, §5.3)."""

from __future__ import annotations

import pytest

import comfy_bridge


@pytest.fixture(scope="session")
def runtime():
    return comfy_bridge.start(dynamic_vram=True)


# --- dispatch ------------------------------------------------------------


def test_v1_node(runtime):
    """V1: instance method returning a bare tuple."""
    cls = comfy_bridge.node_class("EmptyLatentImage")
    assert not cls.FUNCTION.startswith("EXECUTE_NORMALIZED"), "expected a V1 node"
    (latent,) = comfy_bridge.invoke(
        "EmptyLatentImage", width=64, height=64, batch_size=1
    )
    assert latent["samples"].shape == (1, 4, 8, 8)


def test_v3_node(runtime):
    """V3: classmethod returning NodeOutput, unwrapped via .result."""
    cls = comfy_bridge.node_class("KSamplerSelect")
    assert cls.FUNCTION.startswith("EXECUTE_NORMALIZED"), "expected a V3 node"
    (sampler,) = comfy_bridge.invoke("KSamplerSelect", sampler_name="euler")
    assert sampler is not None


def test_v3_hidden_inputs_are_populated(runtime):
    """Spec C31 — the raw class has hidden=None and would AttributeError."""
    raw = comfy_bridge.node_class("SaveVideo")
    assert getattr(raw, "hidden", None) is None
    clone = raw.PREPARE_CLASS_CLONE(None)
    assert clone.hidden is not None
    assert clone.hidden.extra_pnginfo is None  # present, not raising


def test_no_shipped_node_has_an_async_entrypoint(runtime):
    """Spec C18 corrected — measured zero, not 243.

    The original 243 counted every `async def` in comfy_extras, but those are
    helpers; no shipped node's `execute` is a coroutine function, so FUNCTION
    never resolves to EXECUTE_NORMALIZED_ASYNC (comfy_api/latest/_io.py:1927).
    invoke() keeps its await path anyway — two lines of cheap insurance for when
    upstream adds one. This test tells us when that happens.
    """
    import inspect

    async_nodes = [
        name
        for name, cls in runtime.nodes.items()
        if inspect.iscoroutinefunction(getattr(cls, "execute", None))
    ]
    assert not async_nodes, (
        f"shipped async nodes now exist ({async_nodes[:5]}) — invoke()'s await "
        "path is now load-bearing and needs a real execution test"
    )


def test_invoke_awaits_coroutines():
    """Cover the await path directly, since no shipped node exercises it."""
    from comfy_bridge.bootstrap import get_runtime

    async def coro():
        return ("awaited",)

    assert get_runtime().loop.run_until_complete(coro()) == ("awaited",)


def test_unknown_node_raises(runtime):
    with pytest.raises(comfy_bridge.UnsupportedNodeError, match="no shipped node"):
        comfy_bridge.invoke("ThisNodeDoesNotExist")


def test_node_error_carries_class_type(runtime):
    """Spec §8 — failures must name the node, not just surface a traceback."""
    with pytest.raises(comfy_bridge.NodeExecutionError) as excinfo:
        comfy_bridge.invoke("EmptyLatentImage", width="not-an-int", height=64, batch_size=1)
    assert excinfo.value.class_type == "EmptyLatentImage"
    assert excinfo.value.original is not None


# --- guards --------------------------------------------------------------


#: Shipped nodes that genuinely use list mapping (spec C17, corrected).
#: invoke() refuses these; they are a documented §9 limitation, not a bug.
#: Pinned so the set changing on an upstream bump is visible rather than silent.
KNOWN_LIST_IO_NODES = 27


def test_list_io_nodes_are_a_known_bounded_set(runtime):
    """Spec C17 corrected — 27 shipped nodes DO use list I/O, not zero.

    The original survey grepped for literal `INPUT_IS_LIST` assignments, which
    V3 nodes never make: they declare it through Schema(is_input_list=...), so
    the attribute only materialises at runtime.
    """
    from comfy_bridge.invoke import _uses_list_io

    offenders = sorted(
        name
        for name, cls in runtime.nodes.items()
        if _uses_list_io(getattr(cls, "INPUT_IS_LIST", False))
        or _uses_list_io(getattr(cls, "OUTPUT_IS_LIST", False))
    )
    assert len(offenders) == KNOWN_LIST_IO_NODES, (
        f"list-I/O node count moved from {KNOWN_LIST_IO_NODES} to "
        f"{len(offenders)}: {offenders}"
    )
    # Representative members, spanning both the batching and dataset families.
    assert "RebatchLatents" in offenders
    assert "CreateList" in offenders


def test_invoke_refuses_list_io_nodes(runtime):
    """The guard must fire rather than silently produce wrong results."""
    with pytest.raises(comfy_bridge.UnsupportedNodeError, match="list mapping"):
        comfy_bridge.invoke("RebatchLatents", latents={"samples": None}, batch_size=1)


def test_list_io_guard_detects_a_real_offender():
    """The guard must still fire when a node genuinely switches list I/O on."""
    from comfy_bridge.invoke import _uses_list_io

    assert not _uses_list_io([False])       # the V3 default
    assert not _uses_list_io(False)
    assert _uses_list_io([False, True])     # a real one
    assert _uses_list_io(True)


def test_invoke_disables_autograd(runtime):
    """Spec C29 — the single highest-consequence line from execution.py:751."""
    import torch

    assert torch.is_grad_enabled(), "precondition: grad on outside invoke()"
    (latent,) = comfy_bridge.invoke(
        "EmptyLatentImage", width=64, height=64, batch_size=1
    )
    assert not latent["samples"].requires_grad


# --- memory (spec §5.3) --------------------------------------------------


def test_as_patcher_handles_both_wrapper_shapes(runtime):
    """Scope item E — comfy.sd.VAE has no .model, so a naive check skips it."""
    from comfy_bridge.memory import as_patcher

    (latent,) = comfy_bridge.invoke(
        "EmptyLatentImage", width=64, height=64, batch_size=1
    )
    assert as_patcher(latent) is None  # not a model at all
    assert as_patcher(None) is None


def test_offload_ignores_non_models(runtime):
    assert comfy_bridge.offload(None, "not a model", 42) == 0


# --- the D11 invariant ---------------------------------------------------


def test_bridge_does_not_patch_comfyui(runtime):
    """Spec D11/D13 — the bridge itself mutates nothing at rest.

    Optimization work needs module-level patches, so the invariant is no longer
    "never mutate" but "mutate only through comfy_bridge.patch_attr, and leave
    nothing applied when idle". test_extend.py covers the managed path; this
    checks the resting state.
    """
    import comfy.sd
    import comfy.utils
    import nodes

    # These are the attributes a tempting "just monkey-patch it" fix would hit.
    assert comfy.utils.PROGRESS_BAR_HOOK is None, "something installed a progress hook"
    assert nodes.NODE_CLASS_MAPPINGS is not None
    # Our node registry is a copy, so mutating it cannot corrupt ComfyUI's.
    assert runtime.nodes is not nodes.NODE_CLASS_MAPPINGS
    assert comfy_bridge.active_patches() == (), "a patch was left applied"
