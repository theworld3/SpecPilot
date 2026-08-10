"""Storage backends: where KV bytes actually live.

``NullBackend`` is used by the simulator -- it records sizes and does no I/O.
``PinnedHostBackend`` is the real thing: a pre-allocated pinned host buffer
plus a dedicated CUDA copy stream, so demotions overlap with compute instead
of stalling it.

Pinned memory is not an optimisation detail here, it is the whole point:
a pageable ``cudaMemcpy`` on 4 GiB of KV runs at roughly half the bandwidth
*and* synchronises the device, which would turn every eviction into a
pipeline bubble.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

__all__ = ["StorageBackend", "NullBackend", "PinnedHostBackend"]


class StorageBackend(abc.ABC):
    """Moves block payloads between tiers."""

    @abc.abstractmethod
    def store(self, block_id: int, payload: Any) -> None:
        """GPU -> host (or disk)."""

    @abc.abstractmethod
    def load(self, block_id: int, out: Optional[Any] = None) -> Any:
        """Host (or disk) -> GPU."""

    @abc.abstractmethod
    def discard(self, block_id: int) -> None: ...

    def synchronize(self) -> None:
        """Block until outstanding transfers complete."""

    def transfer(self, blocks: Any, src: object, dst: object) -> None:
        """Move a set of blocks between tiers.

        The default is a no-op: the discrete-event simulator and the unit
        tests do not move real tensors. A real connector overrides this to
        issue (or schedule) the actual H2D/D2H copy. ``src``/``dst`` are
        :class:`~agentcache.types.Tier` values.
        """

    @property
    def pcie_backlog_s(self) -> float:
        """How long the copy engine is already booked for, in seconds.

        Used by the cost model to price offload contention. Defaults to 0
        (idle); a real backend tracks outstanding copy time here.
        """
        return 0.0


class NullBackend(StorageBackend):
    """Metadata-only backend for simulation and unit tests."""

    def __init__(self) -> None:
        self.stored: Dict[int, int] = {}

    def store(self, block_id: int, payload: Any) -> None:
        self.stored[block_id] = getattr(payload, "nbytes", 0)

    def load(self, block_id: int, out: Optional[Any] = None) -> Any:
        return self.stored.get(block_id)

    def discard(self, block_id: int) -> None:
        self.stored.pop(block_id, None)


class PinnedHostBackend(StorageBackend):
    """Pinned host buffer + dedicated copy stream.

    Notes on correctness
    --------------------
    * Every ``store`` is issued on ``self._stream`` and followed by an event
      record. The block must not be reused on the GPU until that event has
      fired, which the manager enforces via :meth:`wait`.
    * FP8 packing (when enabled) happens on the GPU *before* the copy, so we
      move half the bytes over PCIe. The pack kernel lives in
      ``agentcache.kernels.gather_pack``.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        import torch  # local import: keep the core importable without torch

        self._torch = torch
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype)
        elem_size = torch.empty((), dtype=self._dtype).element_size()
        self._pool = torch.empty(
            capacity_bytes // elem_size, dtype=self._dtype, pin_memory=True
        )
        self._stream = torch.cuda.Stream(device=self._device)
        self._offsets: Dict[int, tuple[int, int, tuple[int, ...]]] = {}
        self._events: Dict[int, Any] = {}
        self._cursor = 0

    # -- allocation -----------------------------------------------------
    def _reserve(self, numel: int, shape) -> tuple[int, int]:
        if self._cursor + numel > self._pool.numel():
            raise MemoryError("pinned host pool exhausted")
        start = self._cursor
        self._cursor += numel
        return start, numel

    # -- API ------------------------------------------------------------
    def store(self, block_id: int, payload: Any) -> None:
        torch = self._torch
        flat = payload.reshape(-1)
        start, numel = self._reserve(flat.numel(), tuple(payload.shape))
        self._offsets[block_id] = (start, numel, tuple(payload.shape))
        with torch.cuda.stream(self._stream):
            self._pool[start : start + numel].copy_(flat, non_blocking=True)
        event = torch.cuda.Event()
        event.record(self._stream)
        self._events[block_id] = event

    def load(self, block_id: int, out: Optional[Any] = None) -> Any:
        torch = self._torch
        start, numel, shape = self._offsets[block_id]
        src = self._pool[start : start + numel]
        with torch.cuda.stream(self._stream):
            if out is None:
                out = torch.empty(numel, dtype=self._dtype, device=self._device)
            out.reshape(-1).copy_(src, non_blocking=True)
        event = torch.cuda.Event()
        event.record(self._stream)
        self._events[block_id] = event
        return out.reshape(shape)

    def wait(self, block_id: int) -> None:
        event = self._events.get(block_id)
        if event is not None:
            event.synchronize()

    def discard(self, block_id: int) -> None:
        self._offsets.pop(block_id, None)
        self._events.pop(block_id, None)

    def synchronize(self) -> None:
        self._stream.synchronize()

    @property
    def stream(self):
        return self._stream
