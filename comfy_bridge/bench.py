"""Measurement primitives for the acceleration work.

Three levels, because acceleration claims get made at all three and mixing them
up is how a 2x turns into a rounding error:

* **Whole run** — :func:`profile` wraps a ``run_graph()`` call and reports wall
  clock plus a per-node breakdown, using the :func:`comfy_bridge.add_observer`
  hook in ``invoke()``. No change to generated code.
* **Per denoise step** — :func:`time_model_calls` hangs a wrapper on the model
  and collects one duration per transformer forward. This is the number an
  attention or quantization change should move, isolated from loading, VAE
  decode, and encode.
* **Ad hoc** — :func:`timed` for anything else.

**On timing CUDA.** Every function here synchronises by default. Kernel launches
are asynchronous, so an unsynchronised timer measures how fast Python queued the
work — which for a 3090 running video diffusion is off by orders of magnitude.
The cost is that some genuine overlap is serialised away, so a per-node
breakdown will not sum exactly to the wall clock. That gap is real and worth
reporting rather than hiding.

**On VRAM.** ``torch.cuda.max_memory_allocated`` only sees allocations through
torch's caching allocator. This install runs DynamicVRAM (comfy-aimdo), which
pages weights and may allocate outside it, so :class:`Report` also carries the
device-level figure from ``torch.cuda.mem_get_info`` — the one that actually
predicts an OOM. When the two disagree, believe the device.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import hooks
from .invoke import add_observer

log = logging.getLogger("comfy_bridge.bench")

__all__ = [
    "DeviceSampler",
    "NodeStat",
    "Report",
    "StepTimings",
    "compare",
    "device_memory",
    "profile",
    "sync",
    "time_model_calls",
    "timed",
]


def sync(device: Any = None) -> None:
    """Block until queued CUDA work on ``device`` finishes. No-op on CPU."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def gpu_utilisation(index: int = 0) -> int | None:
    """Percent utilisation of a GPU, or None if it cannot be determined.

    Shells out to ``nvidia-smi`` rather than using torch, because the question is
    "is anything at all using this card" — including other processes, which
    torch cannot see. Returns None when there is no nvidia-smi, so callers can
    treat unknown as idle rather than blocking on CPU-only machines.

    Used by the test suites to refuse to run against a busy card: they allocate
    real VRAM, and since :class:`DeviceSampler` reads device-wide totals, a
    concurrent test run would land inside a benchmark's peak-VRAM figure.
    """
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    try:
        return int(lines[0].strip())
    except (IndexError, ValueError):
        return None


def device_memory(device: Any = None) -> tuple[int, int]:
    """``(used_bytes, total_bytes)`` for the CUDA device, from the driver.

    Counts everything on the card — torch's allocator, comfy-aimdo's, other
    processes — which is what determines whether the next allocation OOMs.
    Returns ``(0, 0)`` when there is no CUDA device.
    """
    import torch

    if not torch.cuda.is_available():
        return (0, 0)
    free, total = torch.cuda.mem_get_info(device)
    return (total - free, total)


class DeviceSampler:
    """Polls driver-level VRAM on a thread to find the true peak.

    ``torch.cuda.max_memory_allocated`` is the obvious way to report peak VRAM
    and it is wrong here. It only counts allocations made through torch's
    caching allocator, and DynamicVRAM (comfy-aimdo) installs its own CUDA
    allocator hooks — on a MiniMax H3 run it reported 2.27 GB on a card that was
    obviously holding far more. ``mem_get_info`` asks the driver, so it sees
    every allocation on the device.

    Sampling on a thread rather than at node boundaries matters because peak
    VRAM happens *inside* a denoise step, not between them. The cost is that a
    spike shorter than ``interval`` can be missed entirely — this is a floor on
    the true peak, never an overestimate, and the sample count is reported so
    that limitation stays visible.

    Counts other processes on the card too. That is the honest number for "will
    this OOM", which is the question peak VRAM is asked to answer, but it does
    mean a browser on the same GPU inflates it.
    """

    def __init__(self, device: Any = None, interval: float = 0.05) -> None:
        self.device = device
        self.interval = interval
        self.peak_bytes = 0
        self.total_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        import torch

        while not self._stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info(self.device)
            except Exception as exc:  # never let the instrument kill the run
                log.debug("device sampling stopped: %r", exc)
                return
            self.peak_bytes = max(self.peak_bytes, total - free)
            self.total_bytes = total
            self.samples += 1
            self._stop.wait(self.interval)

    def start(self) -> DeviceSampler:
        import torch

        if not torch.cuda.is_available():
            return self
        # Take one reading synchronously, so a block shorter than the poll
        # interval still reports something rather than zero.
        free, total = torch.cuda.mem_get_info(self.device)
        self.peak_bytes = total - free
        self.total_bytes = total
        self.samples = 1

        self._thread = threading.Thread(
            target=self._poll, name="comfy_bridge.bench.vram", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> int:
        """Stop polling and return the peak in bytes."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.peak_bytes

    def __enter__(self) -> DeviceSampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


@dataclass
class NodeStat:
    """Aggregated cost of every call to one node class during a profile."""

    class_type: str
    calls: int = 0
    total_s: float = 0.0

    @property
    def mean_s(self) -> float:
        return self.total_s / self.calls if self.calls else 0.0


@dataclass
class Report:
    """What :func:`profile` collected. Populated on exit from the context."""

    label: str = ""
    wall_s: float = 0.0
    nodes: dict[str, NodeStat] = field(default_factory=dict)
    #: Peak of torch's caching allocator during the profile. Under DynamicVRAM
    #: this UNDER-REPORTS badly — prefer device_peak_bytes.
    peak_allocated_bytes: int = 0
    #: Peak reserved by torch's allocator — the closer proxy for fragmentation.
    peak_reserved_bytes: int = 0
    #: True peak device usage from the driver, sampled on a thread. This is the
    #: number that predicts an OOM.
    device_peak_bytes: int = 0
    #: How many times the sampler read the device; 0 means it never ran.
    device_samples: int = 0
    #: Device-level usage at entry and exit, from the driver.
    device_used_start: int = 0
    device_used_end: int = 0
    device_total: int = 0

    @property
    def peak_vram_bytes(self) -> int:
        """Best available peak: the driver's if sampled, else torch's."""
        return self.device_peak_bytes or self.peak_allocated_bytes

    @property
    def node_s(self) -> float:
        """Sum of attributed per-node time."""
        return sum(stat.total_s for stat in self.nodes.values())

    @property
    def unattributed_s(self) -> float:
        """Wall clock not inside any node call — plus timer/sync skew."""
        return self.wall_s - self.node_s

    def slowest(self, n: int = 10) -> list[NodeStat]:
        return sorted(self.nodes.values(), key=lambda s: s.total_s, reverse=True)[:n]

    def table(self, n: int = 10) -> str:
        gb = 1024**3
        head = self.label or "profile"
        lines = [
            f"{head}: {self.wall_s:.2f}s wall, "
            f"{self.node_s:.2f}s in {sum(s.calls for s in self.nodes.values())} node "
            f"calls, {self.unattributed_s:.2f}s elsewhere",
        ]
        if self.device_samples:
            lines.append(
                f"  peak VRAM {self.device_peak_bytes / gb:.2f} GB of "
                f"{self.device_total / gb:.2f} GB "
                f"({self.device_samples:,} device samples)"
            )
        lines.append(
            f"  torch allocator: peak {self.peak_allocated_bytes / gb:.2f} GB, "
            f"reserved {self.peak_reserved_bytes / gb:.2f} GB "
            f"(excludes DynamicVRAM's own allocations)"
        )
        if self.device_total:
            lines.append(
                f"  device {self.device_used_start / gb:.2f} -> "
                f"{self.device_used_end / gb:.2f} GB at exit"
            )
        for stat in self.slowest(n):
            share = 100 * stat.total_s / self.wall_s if self.wall_s else 0.0
            lines.append(
                f"  {stat.total_s:8.2f}s  {share:5.1f}%  {stat.calls:4d}x  "
                f"{stat.class_type}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


@contextlib.contextmanager
def profile(
    label: str = "",
    *,
    cuda_sync: bool = True,
    device: Any = None,
    sample_vram: bool = True,
    vram_interval: float = 0.05,
) -> Iterator[Report]:
    """Measure a block, attributing time to the nodes ``invoke()`` sees.

        with bench.profile("baseline") as report:
            run_graph()
        print(report.table())

    The Report object is yielded immediately and filled in on exit, so it can be
    stashed before the block runs. Resets torch's peak-memory counters on entry;
    that is process-global, so nested profiles will report the inner peak in
    both.

    ``sample_vram`` starts a :class:`DeviceSampler` thread, because torch's own
    peak counters cannot see DynamicVRAM's allocations and under-report by a
    large factor. Turn it off only if the polling thread is itself under
    suspicion.
    """
    import torch

    report = Report(label=label)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    report.device_used_start, report.device_total = device_memory(device)

    def observe(class_type: str, seconds: float) -> None:
        stat = report.nodes.get(class_type)
        if stat is None:
            stat = report.nodes[class_type] = NodeStat(class_type)
        stat.calls += 1
        stat.total_s += seconds

    remove = add_observer(observe, cuda_sync=cuda_sync)
    sampler = (
        DeviceSampler(device, interval=vram_interval).start()
        if (cuda and sample_vram)
        else None
    )
    started = time.perf_counter()
    try:
        yield report
    finally:
        if cuda_sync:
            sync(device)
        report.wall_s = time.perf_counter() - started
        remove()
        if sampler is not None:
            report.device_peak_bytes = sampler.stop()
            report.device_samples = sampler.samples
            report.device_total = sampler.total_bytes or report.device_total
        if cuda:
            report.peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
            report.peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
        report.device_used_end, _ = device_memory(device)


@dataclass
class Timing:
    """Result of a :func:`timed` block; ``seconds`` is set on exit."""

    label: str = ""
    seconds: float = 0.0

    def __str__(self) -> str:
        return f"{self.label or 'block'}: {self.seconds:.3f}s"


@contextlib.contextmanager
def timed(label: str = "", *, cuda_sync: bool = True) -> Iterator[Timing]:
    """Time an arbitrary block, synchronising CUDA first by default."""
    result = Timing(label=label)
    if cuda_sync:
        sync()
    started = time.perf_counter()
    try:
        yield result
    finally:
        if cuda_sync:
            sync()
        result.seconds = time.perf_counter() - started


@dataclass
class StepTimings:
    """A model clone that records how long each of its forwards took.

    ``model`` is what you sample with; ``times`` fills up as it runs.
    """

    model: Any
    times: list[float] = field(default_factory=list)
    at: str = hooks.DIFFUSION_MODEL

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def total_s(self) -> float:
        return sum(self.times)

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.times else 0.0

    def summary(self) -> str:
        if not self.times:
            return f"{self.at}: no calls recorded"
        return (
            f"{self.at}: {self.count} calls, {self.total_s:.2f}s total, "
            f"{self.mean_s * 1000:.1f}ms mean, "
            f"first {self.times[0] * 1000:.1f}ms, last {self.times[-1] * 1000:.1f}ms"
        )

    def __str__(self) -> str:
        return self.summary()


def time_model_calls(
    model: Any, *, at: str = hooks.DIFFUSION_MODEL, cuda_sync: bool = True
) -> StepTimings:
    """Clone ``model`` with a timing wrapper and return both clone and timings.

        timings = bench.time_model_calls(model)
        latent = ksampler(model=timings.model, ...)
        print(timings.summary())

    Defaults to ``DIFFUSION_MODEL`` — the transformer forward, which brackets
    exactly the compute an attention or quantization change should affect. Not
    every architecture exposes it; ``at=hooks.APPLY_MODEL`` is universal but also
    includes conditioning prep. With CFG the model runs twice per denoise step,
    so expect ``count`` to be roughly twice the step count.

    The first entry is usually a large outlier — lazy CUDA context setup, kernel
    autotuning and weight paging land there. Compare medians, or drop it.
    """
    timings = StepTimings(model=None, at=at)

    def timer(executor: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        out = executor(*args, **kwargs)
        if cuda_sync:
            sync()
        timings.times.append(time.perf_counter() - started)
        return out

    timings.model = hooks.add_wrapper(model, at, timer, key="comfy_bridge.bench")
    return timings


def compare(baseline: Report, variant: Report) -> str:
    """Format a baseline-vs-variant diff, per node and overall.

    Speedups above 1.0 mean the variant is faster. Nodes present in only one run
    are listed rather than silently dropped, because an optimization that
    replaces a node changes the set.
    """
    gb = 1024**3
    lines = []
    a_name = baseline.label or "baseline"
    b_name = variant.label or "variant"
    ratio = baseline.wall_s / variant.wall_s if variant.wall_s else float("inf")
    lines.append(
        f"{a_name} {baseline.wall_s:.2f}s -> {b_name} {variant.wall_s:.2f}s "
        f"({ratio:.2f}x)"
    )
    lines.append(
        f"  peak VRAM {baseline.peak_vram_bytes / gb:.2f} -> "
        f"{variant.peak_vram_bytes / gb:.2f} GB"
    )

    shared = sorted(set(baseline.nodes) & set(variant.nodes))
    for class_type in sorted(
        shared,
        key=lambda c: baseline.nodes[c].total_s - variant.nodes[c].total_s,
        reverse=True,
    ):
        before = baseline.nodes[class_type].total_s
        after = variant.nodes[class_type].total_s
        if before < 0.01 and after < 0.01:
            continue
        node_ratio = before / after if after else float("inf")
        lines.append(
            f"  {before:7.2f}s -> {after:7.2f}s  ({node_ratio:5.2f}x)  {class_type}"
        )

    for name, only in (
        (a_name, sorted(set(baseline.nodes) - set(variant.nodes))),
        (b_name, sorted(set(variant.nodes) - set(baseline.nodes))),
    ):
        if only:
            lines.append(f"  only in {name}: {', '.join(only)}")
    return "\n".join(lines)
