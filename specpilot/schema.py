"""Versioned JSONL schemas for replayable experiments."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TraceStep:
    """One counterfactual-ready speculative decoding round.

    ``accepted_depths`` stores the maximum accepted prefix for every request at
    K_max. A replay at K clips each value to K, so every policy sees the same
    underlying workload.
    """

    step: int
    accepted_depths: tuple[int, ...]
    context_bucket: str = "default"
    graph_mode: str = "full"
    timestamp_ms: float | None = None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not self.accepted_depths:
            raise ValueError("accepted_depths must not be empty")
        if any(depth < 0 for depth in self.accepted_depths):
            raise ValueError("accepted depths must be non-negative")

    @property
    def batch_size(self) -> int:
        return len(self.accepted_depths)

    def accepted_at(self, k: int) -> tuple[int, ...]:
        if k < 0:
            raise ValueError("k must be non-negative")
        return tuple(min(k, depth) for depth in self.accepted_depths)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["schema_version"] = TRACE_SCHEMA_VERSION
        record["accepted_depths"] = list(self.accepted_depths)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TraceStep:
        if record.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported trace schema version")
        values = dict(record)
        values.pop("schema_version")
        values["accepted_depths"] = tuple(int(x) for x in values["accepted_depths"])
        return cls(**values)


def read_trace(path: str | Path) -> Iterator[TraceStep]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                yield TraceStep.from_record(record)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid trace record at line {line_number}: {exc}") from exc


def write_trace(path: str | Path, steps: Iterable[TraceStep]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for step in steps:
            handle.write(json.dumps(step.to_record(), separators=(",", ":")) + "\n")
