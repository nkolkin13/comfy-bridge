"""M5 tests — locally-authored nodes, reversible patches, instrumentation.

The point of these is the containment story, not the features: a patch must
always come back off, and a local node must never reach ComfyUI's own tables.
"""

from __future__ import annotations

import time
import types

import pytest

import comfy_bridge
from comfy_bridge import bench, hooks
from comfy_bridge.errors import ExtensionError, UnsupportedNodeError


@pytest.fixture(scope="session")
def runtime():
    return comfy_bridge.start(dynamic_vram=True)


@pytest.fixture(autouse=True)
def no_leaked_patches():
    """Every test leaves the process exactly as it found it."""
    yield
    leaked = comfy_bridge.active_patches()
    comfy_bridge.revert_all_patches()
    assert leaked == (), f"test leaked patches: {leaked}"


class DoubleFloat:
    """Minimal V1-shaped local node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("FLOAT", {"default": 1.0})}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("doubled",)
    FUNCTION = "run"
    CATEGORY = "bench"

    def run(self, value):
        return (value * 2.0,)


# --- registering our own nodes ---------------------------------------------


def test_local_node_is_invokable(runtime):
    comfy_bridge.register_node(DoubleFloat, class_type="TestDoubleFloat")
    try:
        assert comfy_bridge.invoke("TestDoubleFloat", value=21.0) == (42.0,)
    finally:
        comfy_bridge.unregister_node("TestDoubleFloat")


def test_registration_does_not_touch_comfyui(runtime):
    import nodes

    comfy_bridge.register_node(DoubleFloat, class_type="TestDoubleFloat")
    try:
        assert "TestDoubleFloat" in runtime.nodes
        assert "TestDoubleFloat" not in nodes.NODE_CLASS_MAPPINGS
    finally:
        comfy_bridge.unregister_node("TestDoubleFloat")
    assert "TestDoubleFloat" not in runtime.nodes


def test_register_node_works_as_a_decorator(runtime):
    @comfy_bridge.register_node(class_type="TestDecorated")
    class Decorated(DoubleFloat):
        pass

    try:
        assert runtime.nodes["TestDecorated"] is Decorated
        assert "TestDecorated" in comfy_bridge.registered_nodes()
    finally:
        comfy_bridge.unregister_node("TestDecorated")


def test_shadowing_a_shipped_node_needs_replace_and_restores(runtime):
    original = runtime.nodes["CLIPTextEncode"]

    with pytest.raises(ExtensionError, match="already registered"):
        comfy_bridge.register_node(DoubleFloat, class_type="CLIPTextEncode")

    comfy_bridge.register_node(DoubleFloat, class_type="CLIPTextEncode", replace=True)
    assert runtime.nodes["CLIPTextEncode"] is DoubleFloat
    comfy_bridge.unregister_node("CLIPTextEncode")
    assert runtime.nodes["CLIPTextEncode"] is original


@pytest.mark.parametrize(
    "attrs, match",
    [
        ({"RETURN_TYPES": ("FLOAT",), "FUNCTION": "run"}, "INPUT_TYPES"),
        ({"INPUT_TYPES": classmethod(lambda cls: {}), "FUNCTION": "run"}, "RETURN_TYPES"),
        (
            {"INPUT_TYPES": classmethod(lambda cls: {}), "RETURN_TYPES": ("FLOAT",)},
            "FUNCTION",
        ),
        (
            {
                "INPUT_TYPES": classmethod(lambda cls: {}),
                "RETURN_TYPES": ("FLOAT",),
                "FUNCTION": "nope",
            },
            "no such attribute",
        ),
        (
            {
                "INPUT_TYPES": classmethod(lambda cls: {}),
                "RETURN_TYPES": ("FLOAT",),
                "RETURN_NAMES": ("a", "b"),
                "FUNCTION": "run",
            },
            "RETURN_NAMES",
        ),
    ],
)
def test_bad_node_classes_are_refused_at_registration(runtime, attrs, match):
    attrs = dict(attrs)
    attrs.setdefault("run", lambda self: ())
    cls = type("Broken", (), attrs)
    with pytest.raises(ExtensionError, match=match):
        comfy_bridge.register_node(cls, class_type="TestBroken")


def test_list_io_node_is_refused_at_registration(runtime):
    cls = type(
        "Listy",
        (DoubleFloat,),
        {"INPUT_IS_LIST": True},
    )
    with pytest.raises(UnsupportedNodeError, match="INPUT_IS_LIST"):
        comfy_bridge.register_node(cls, class_type="TestListy")


# --- reversible patches ----------------------------------------------------


def test_patch_applies_and_reverts():
    module = types.SimpleNamespace(value=1)
    patch = comfy_bridge.patch_attr(module, "value", 2)

    assert module.value == 1, "patch_attr must not apply on construction"
    with patch:
        assert module.value == 2
        assert patch in comfy_bridge.active_patches()
    assert module.value == 1
    assert patch not in comfy_bridge.active_patches()


def test_patch_removes_an_attribute_that_did_not_exist():
    module = types.SimpleNamespace()
    with comfy_bridge.patch_attr(module, "added", 1):
        assert module.added == 1
    assert not hasattr(module, "added")


def test_patch_reverts_when_the_block_raises():
    module = types.SimpleNamespace(value=1)
    with pytest.raises(RuntimeError):
        with comfy_bridge.patch_attr(module, "value", 2):
            raise RuntimeError("boom")
    assert module.value == 1


def test_patch_set_rolls_back_a_partial_failure():
    good = types.SimpleNamespace(value=1)

    class Frozen:
        __slots__ = ()

    frozen = Frozen()
    patches = comfy_bridge.PatchSet("half-broken")
    patches.add(good, "value", 2)
    patches.add(frozen, "value", 3)  # __slots__ makes setattr raise

    with pytest.raises(AttributeError):
        patches.apply()
    assert good.value == 1, "first patch was not rolled back"
    assert comfy_bridge.active_patches() == ()


def test_revert_all_patches_returns_count():
    module = types.SimpleNamespace(a=1, b=2)
    comfy_bridge.patch_attr(module, "a", 10).apply()
    comfy_bridge.patch_attr(module, "b", 20).apply()
    assert comfy_bridge.revert_all_patches() == 2
    assert (module.a, module.b) == (1, 2)


def test_patching_comfyui_is_contained(runtime):
    """The real use case: swap a comfy module global, put it back exactly."""
    import comfy.ldm.modules.attention as attention

    original = attention.optimized_attention

    def stub(*args, **kwargs):  # pragma: no cover - never called
        raise AssertionError

    with comfy_bridge.patch_attr(attention, "optimized_attention", stub):
        assert attention.optimized_attention is stub
    assert attention.optimized_attention is original


# --- instrumentation -------------------------------------------------------


def test_profile_attributes_time_to_nodes(runtime):
    comfy_bridge.register_node(DoubleFloat, class_type="TestDoubleFloat")
    try:
        with bench.profile("unit") as report:
            comfy_bridge.invoke("TestDoubleFloat", value=1.0)
            comfy_bridge.invoke("TestDoubleFloat", value=2.0)
    finally:
        comfy_bridge.unregister_node("TestDoubleFloat")

    stat = report.nodes["TestDoubleFloat"]
    assert stat.calls == 2
    assert report.wall_s >= report.node_s >= 0.0
    assert "TestDoubleFloat" in report.table()


def test_observer_is_removed_after_profile(runtime):
    # comfy_bridge.invoke is the function, not the module — bind the list itself.
    from comfy_bridge.invoke import _OBSERVERS

    before = len(_OBSERVERS)
    with bench.profile():
        pass
    assert len(_OBSERVERS) == before


def test_observer_exceptions_do_not_break_invoke(runtime):
    comfy_bridge.register_node(DoubleFloat, class_type="TestDoubleFloat")
    remove = comfy_bridge.add_observer(lambda *a: 1 / 0)
    try:
        assert comfy_bridge.invoke("TestDoubleFloat", value=1.0) == (2.0,)
    finally:
        remove()
        comfy_bridge.unregister_node("TestDoubleFloat")


def test_device_memory_shape():
    used, total = bench.device_memory()
    assert 0 <= used <= total


def test_timed_measures_something():
    with bench.timed("noop") as timing:
        pass
    assert timing.seconds >= 0.0


# --- model hooks -----------------------------------------------------------


class _FakePatcher:
    """Enough of the ModelPatcher surface for the hook plumbing."""

    def __init__(self):
        self.wrappers = {}
        self.object_patches = {}
        self.model_options = {"transformer_options": {}}

    def clone(self):
        other = _FakePatcher()
        other.wrappers = {k: {k2: list(v2) for k2, v2 in v.items()} for k, v in self.wrappers.items()}
        other.object_patches = dict(self.object_patches)
        other.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        return other

    def add_wrapper(self, wrapper_type, wrapper):
        self.add_wrapper_with_key(wrapper_type, None, wrapper)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj


def test_add_wrapper_clones_and_leaves_the_original_clean():
    model = _FakePatcher()
    wrapped = hooks.add_wrapper(model, hooks.APPLY_MODEL, lambda ex, *a, **k: ex(*a, **k))
    assert wrapped is not model
    assert model.wrappers == {}
    assert wrapped.wrappers[hooks.APPLY_MODEL]


def test_add_wrapper_rejects_an_unknown_type():
    with pytest.raises(ExtensionError, match="unknown wrapper type"):
        hooks.add_wrapper(_FakePatcher(), "not_a_wrapper", lambda ex: ex())


def test_hooks_reject_a_non_patcher():
    with pytest.raises(ExtensionError, match="ModelPatcher"):
        hooks.add_wrapper(object(), hooks.APPLY_MODEL, lambda ex: ex())


def test_override_attention_writes_to_the_clone_only():
    model = _FakePatcher()
    fn = lambda original, *a, **k: original(*a, **k)  # noqa: E731
    patched = hooks.override_attention(model, fn)
    assert patched.model_options["transformer_options"]["optimized_attention_override"] is fn
    assert "optimized_attention_override" not in model.model_options["transformer_options"]


def test_patch_object_records_a_dotted_path():
    model = _FakePatcher()
    sentinel = object()
    patched = hooks.patch_object(model, "diffusion_model.blocks.0.attn", sentinel)
    assert patched.object_patches["diffusion_model.blocks.0.attn"] is sentinel
    assert model.object_patches == {}


def test_time_model_calls_records_each_forward():
    from comfy.patcher_extension import WrapperExecutor

    timings = bench.time_model_calls(_FakePatcher(), cuda_sync=False)
    wrappers = timings.model.wrappers[hooks.DIFFUSION_MODEL]["comfy_bridge.bench"]

    def forward(x):
        return x + 1

    for value in (1, 2, 3):
        assert WrapperExecutor.new_executor(forward, wrappers).execute(value) == value + 1

    assert timings.count == 3
    assert timings.total_s >= 0.0
    assert "3 calls" in timings.summary()


def test_wrapper_types_match_upstream(runtime):
    """Fail loudly if upstream renames a wrapper point (spec §10)."""
    from comfy.patcher_extension import WrappersMP

    assert hooks.OUTER_SAMPLE == WrappersMP.OUTER_SAMPLE
    assert hooks.SAMPLER_SAMPLE == WrappersMP.SAMPLER_SAMPLE
    assert hooks.PREDICT_NOISE == WrappersMP.PREDICT_NOISE
    assert hooks.CALC_COND_BATCH == WrappersMP.CALC_COND_BATCH
    assert hooks.APPLY_MODEL == WrappersMP.APPLY_MODEL
    assert hooks.DIFFUSION_MODEL == WrappersMP.DIFFUSION_MODEL


def test_attention_override_hook_still_exists(runtime):
    """The seam override_attention relies on (attention.py:158)."""
    import inspect

    import comfy.ldm.modules.attention as attention

    source = inspect.getsource(attention.wrap_attn)
    assert "optimized_attention_override" in source


# --- VRAM sampling ---------------------------------------------------------


def test_device_sampler_reports_a_peak(runtime):
    """torch's allocator counter cannot see DynamicVRAM; the driver can."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    sampler = bench.DeviceSampler(interval=0.01).start()
    block = torch.empty(int(256e6 // 4), device="cuda")  # ~256 MB
    torch.cuda.synchronize()
    time.sleep(0.1)
    peak = sampler.stop()
    del block

    assert sampler.samples > 1, "the polling thread never ran"
    assert peak > 0
    assert peak <= sampler.total_bytes


def test_device_sampler_takes_a_reading_even_if_stopped_immediately(runtime):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    sampler = bench.DeviceSampler(interval=10.0).start()
    peak = sampler.stop()
    assert sampler.samples >= 1 and peak > 0, "a short block must still report"


def test_profile_prefers_the_driver_peak(runtime):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    with bench.profile("vram", vram_interval=0.01) as report:
        block = torch.empty(int(256e6 // 4), device="cuda")
        torch.cuda.synchronize()
        time.sleep(0.05)
        del block

    assert report.device_samples > 0
    assert report.peak_vram_bytes == report.device_peak_bytes
    # The driver sees everything on the card, so it can never be the smaller of
    # the two. This is the inversion that made the old column meaningless.
    assert report.device_peak_bytes >= report.peak_allocated_bytes
    assert "peak VRAM" in report.table()


def test_profile_falls_back_when_sampling_is_disabled(runtime):
    with bench.profile("no-sampling", sample_vram=False) as report:
        pass
    assert report.device_samples == 0
    assert report.peak_vram_bytes == report.peak_allocated_bytes
