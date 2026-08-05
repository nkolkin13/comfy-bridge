"""Refuse to run the suite while the GPU is busy.

These tests start ComfyUI and allocate real VRAM. A long benchmark on the same
card is the normal state of this project, and a test run that competes with it
does damage in both directions: it can OOM the benchmark, and because a
DeviceSampler reading is device-wide, anything the tests allocate lands in the
benchmark's peak-VRAM figure and quietly corrupts the result.

So this fails fast rather than skipping. A skip would let the suite report green
while silently covering nothing, and the decision — pause the benchmark, or come
back later — belongs to whoever is at the keyboard, not to a heuristic.

Override with ``COMFY_BRIDGE_ALLOW_BUSY_GPU=1`` when you genuinely mean it.
"""

from __future__ import annotations

import pytest

from comfy_bridge.bench import gpu_utilisation

#: Above this GPU utilisation, assume something real is running.
BUSY_THRESHOLD_PERCENT = 30

OVERRIDE_ENV = "COMFY_BRIDGE_ALLOW_BUSY_GPU"


def busy_gpu_message(used: int, override: str = OVERRIDE_ENV) -> str:
    return (
        f"GPU is {used}% utilised — something is already running on it.\n"
        "\n"
        "This suite starts ComfyUI and allocates real VRAM. Running it now could\n"
        "OOM whatever is in flight, and its allocations would be counted into\n"
        "that run's peak-VRAM measurement.\n"
        "\n"
        "Wait until the card is idle and run the suite then, or if you know it is\n"
        "safe:\n"
        f"    {override}=1 pytest tests/\n"
    )


def check_gpu_is_idle(override_env: str = OVERRIDE_ENV) -> None:
    """Raise UsageError when the card is in use. Shared by both test suites."""
    import os

    if os.environ.get(override_env) == "1":
        return
    used = gpu_utilisation()
    # Unknown means no nvidia-smi — a CPU-only box or CI. Do not block there.
    if used is None or used <= BUSY_THRESHOLD_PERCENT:
        return
    raise pytest.UsageError(busy_gpu_message(used, override_env))


def pytest_configure(config: pytest.Config) -> None:
    check_gpu_is_idle()
