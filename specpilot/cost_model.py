"""Measured, piecewise cost model for draft/verify steps."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CostEstimate:
    batch_size: int
    k: int
    draft_ms: float
    verify_ms: float
    sample_ms: float = 0.0
    scheduler_ms: float = 0.0
    context_bucket: str = "default"
    graph_mode: str = "full"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.k < 0:
            raise ValueError("k must be non-negative")
        components = (self.draft_ms, self.verify_ms, self.sample_ms, self.scheduler_ms)
        if any(not np.isfinite(x) or x < 0.0 for x in components):
            raise ValueError("cost components must be finite and non-negative")
        if self.total_ms <= 0.0:
            raise ValueError("total step cost must be positive")

    @property
    def total_ms(self) -> float:
        return self.draft_ms + self.verify_ms + self.sample_ms + self.scheduler_ms


class ProfileCostModel:
    """Interpolate a measured ``(batch, K, context, graph)`` profile.

    Interpolation is only performed along batch size for a fixed K and fixed
    runtime bucket. Extrapolation clamps by default; production users should
    profile through ``max_num_seqs`` or select ``out_of_range='error'``.
    """

    def __init__(
        self,
        rows: Iterable[CostEstimate],
        *,
        metadata: dict[str, Any] | None = None,
        out_of_range: str = "clamp",
    ) -> None:
        if out_of_range not in {"clamp", "error"}:
            raise ValueError("out_of_range must be 'clamp' or 'error'")
        self.out_of_range = out_of_range
        self.metadata = dict(metadata or {})
        self._groups: dict[tuple[int, str, str], list[CostEstimate]] = {}
        for row in rows:
            key = (row.k, row.context_bucket, row.graph_mode)
            self._groups.setdefault(key, []).append(row)
        if not self._groups:
            raise ValueError("at least one profile row is required")
        for key, group in self._groups.items():
            group.sort(key=lambda row: row.batch_size)
            batches = [row.batch_size for row in group]
            if len(batches) != len(set(batches)):
                raise ValueError(f"duplicate profile row in bucket {key}")

    @property
    def k_values(self) -> tuple[int, ...]:
        return tuple(sorted({key[0] for key in self._groups}))

    @property
    def max_k(self) -> int:
        return max(self.k_values)

    def supports(self, k: int, context_bucket: str = "default", graph_mode: str = "full") -> bool:
        return (k, context_bucket, graph_mode) in self._groups

    def estimate(
        self,
        batch_size: int,
        k: int,
        *,
        context_bucket: str = "default",
        graph_mode: str = "full",
    ) -> CostEstimate:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        key = (k, context_bucket, graph_mode)
        if key not in self._groups:
            raise KeyError(
                f"no profile for k={k}, context_bucket={context_bucket!r}, "
                f"graph_mode={graph_mode!r}"
            )
        group = self._groups[key]
        batches = np.asarray([row.batch_size for row in group], dtype=np.float64)
        if self.out_of_range == "error" and not batches[0] <= batch_size <= batches[-1]:
            raise KeyError(
                f"batch_size={batch_size} is outside profiled range "
                f"[{int(batches[0])}, {int(batches[-1])}]"
            )
        if batch_size <= batches[0]:
            return self._with_batch(group[0], batch_size)
        if batch_size >= batches[-1]:
            return self._with_batch(group[-1], batch_size)

        upper = int(np.searchsorted(batches, batch_size, side="right"))
        lo, hi = group[upper - 1], group[upper]
        weight = (batch_size - lo.batch_size) / (hi.batch_size - lo.batch_size)
        return CostEstimate(
            batch_size=batch_size,
            k=k,
            draft_ms=self._lerp(lo.draft_ms, hi.draft_ms, weight),
            verify_ms=self._lerp(lo.verify_ms, hi.verify_ms, weight),
            sample_ms=self._lerp(lo.sample_ms, hi.sample_ms, weight),
            scheduler_ms=self._lerp(lo.scheduler_ms, hi.scheduler_ms, weight),
            context_bucket=context_bucket,
            graph_mode=graph_mode,
        )

    @staticmethod
    def _lerp(left: float, right: float, weight: float) -> float:
        return float(left + (right - left) * weight)

    @staticmethod
    def _with_batch(row: CostEstimate, batch_size: int) -> CostEstimate:
        values = asdict(row)
        values["batch_size"] = batch_size
        return CostEstimate(**values)

    def to_dict(self) -> dict[str, Any]:
        rows = [row for group in self._groups.values() for row in group]
        rows.sort(key=lambda row: (row.context_bucket, row.graph_mode, row.k, row.batch_size))
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": self.metadata,
            "rows": [asdict(row) for row in rows],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, out_of_range: str = "clamp") -> ProfileCostModel:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported cost profile schema_version={payload.get('schema_version')!r}"
            )
        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError("profile rows must be a list")
        try:
            rows = [CostEstimate(**row) for row in raw_rows]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid cost profile row: {exc}") from exc
        return cls(rows, metadata=payload.get("metadata", {}), out_of_range=out_of_range)

    @classmethod
    def load(cls, path: str | Path, *, out_of_range: str = "clamp") -> ProfileCostModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("profile root must be a JSON object")
        return cls.from_dict(payload, out_of_range=out_of_range)
