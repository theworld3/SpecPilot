"""Hardware and system configuration.

Defaults target a single RTX 4090 (24 GiB, PCIe 4.0 x16) serving
Llama-3.1-8B in FP16 -- the reference setup used for every number in
``benchmarks/results/``.

Every latency constant here is a *model parameter*, not a magic number.
``scripts/calibrate.py`` measures them on real hardware and writes a JSON
override, so simulator output can be validated against the real engine
instead of being taken on faith.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .types import ModelSpec, Tier

__all__ = ["HardwareConfig", "SystemConfig", "load_config"]

GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(frozen=True)
class HardwareConfig:
    """Measured (or datasheet) hardware characteristics.

    Bandwidths are *effective* numbers, i.e. what a pinned-memory async copy
    actually achieves, not the theoretical peak printed on the box.
    """

    gpu_name: str = "RTX 4090"
    gpu_kv_capacity_bytes: int = 14 * GIB
    """KV pool size after weights + activations. 24 GiB - ~16 GiB(fp16 8B model
    + activations) leaves roughly 6-8 GiB in practice; 14 GiB models an A6000
    or a quantised 8B. Override per machine."""

    cpu_kv_capacity_bytes: int = 32 * GIB
    """Pinned host memory reserved for KV. Not the machine's total RAM:
    pinning locks pages, and allocating more than ~1/4 of system memory
    starves the page cache and the dataloader."""
    disk_kv_capacity_bytes: int = 512 * GIB

    h2d_bandwidth_bps: float = 24.0 * GIB
    """Host->device, pinned memory, PCIe 4.0 x16. Measured ~24 GB/s."""
    d2h_bandwidth_bps: float = 22.0 * GIB
    disk_read_bandwidth_bps: float = 3.0 * GIB
    """NVMe Gen4 sequential read."""

    prefill_tokens_per_s: float = 11000.0
    """Single-stream prefill throughput, 8B FP16 on a 4090.

    Roughly 165 TFLOP/s effective at 2*P*N FLOPs per token. Conservative:
    long contexts do better thanks to attention batching, which makes the
    recompute-avoidance case here *under*-stated rather than inflated."""

    decode_step_s: float = 0.028
    """Per-request per-token decode latency inside a moderate batch.

    Not the headline "700 tok/s" number: that is aggregate throughput across
    the batch. From one request's point of view a 4090 running 8B FP16 at
    batch 8 delivers ~35 tok/s. Using the aggregate figure here would shrink
    decode time ~8x and silently exaggerate how much KV policy matters."""

    transfer_launch_overhead_s: float = 0.00015
    """Fixed cost per transfer, dominated by kernel launch + sync."""

    def bandwidth_to_gpu(self, tier: Tier) -> float:
        if tier is Tier.CPU:
            return self.h2d_bandwidth_bps
        if tier is Tier.DISK:
            return self.disk_read_bandwidth_bps
        raise ValueError(f"no transfer path to GPU from {tier!r}")

    def capacity(self, tier: Tier) -> int:
        return {
            Tier.GPU: self.gpu_kv_capacity_bytes,
            Tier.CPU: self.cpu_kv_capacity_bytes,
            Tier.DISK: self.disk_kv_capacity_bytes,
        }[tier]


@dataclass(frozen=True)
class SystemConfig:
    """Engine-level knobs."""

    model: ModelSpec = field(default_factory=ModelSpec)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    block_size_tokens: int = 16
    """Tokens per KV block. Matches vLLM's default."""

    max_batch_size: int = 16
    """Maximum number of programs concurrently in the running set."""

    enable_cpu_tier: bool = True
    enable_disk_tier: bool = False
    """Disk is off by default: at 3 GB/s a 4 GiB context takes 1.3 s to load,
    which is the same order as recompute. It only pays off for very long
    contexts -- see docs/design.md."""

    fp8_offload: bool = True
    """Quantise KV to FP8 when demoting GPU->CPU. Halves transfer time at the
    cost of a small accuracy delta (measured in benchmarks/run_accuracy.py)."""

    @property
    def block_bytes(self) -> int:
        return self.model.bytes_for(self.block_size_tokens)

    @property
    def gpu_blocks(self) -> int:
        return self.hardware.gpu_kv_capacity_bytes // self.block_bytes

    @property
    def cpu_blocks(self) -> int:
        return self.hardware.cpu_kv_capacity_bytes // self.block_bytes

    def offload_bytes(self, raw_bytes: int) -> int:
        """Bytes actually moved when demoting to CPU."""
        return raw_bytes // 2 if self.fp8_offload else raw_bytes

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[str | Path] = None) -> SystemConfig:
    """Load a config, optionally overriding fields from a JSON file.

    The JSON is a partial override, e.g.::

        {"hardware": {"gpu_kv_capacity_bytes": 8589934592},
         "max_batch_size": 8}
    """
    cfg = SystemConfig()
    if path is None:
        return cfg

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    hw = HardwareConfig(**{**asdict(cfg.hardware), **data.pop("hardware", {})})
    model = ModelSpec(**{**asdict(cfg.model), **data.pop("model", {})})
    return SystemConfig(model=model, hardware=hw, **data)
