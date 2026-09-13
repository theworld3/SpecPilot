"""Convert repeated phase telemetry into a median B x K cost profile.

Input JSONL records are emitted by the vLLM integration hook and must contain:
``batch_size``, ``k``, ``draft_ms``, ``verify_ms``, ``sample_ms``, and
``scheduler_ms``. Warm-up filtering happens at collection time; this command
only aggregates completed samples and records their provenance.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from specpilot.cost_model import CostEstimate, ProfileCostModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="phase telemetry JSONL")
    parser.add_argument("--output", required=True, help="cost profile JSON")
    parser.add_argument("--manifest", required=True, help="run manifest JSON")
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()

    groups: dict[tuple[int, int, str, str], list[dict[str, float]]] = defaultdict(list)
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                key = (
                    int(record["batch_size"]),
                    int(record["k"]),
                    str(record.get("context_bucket", "default")),
                    str(record.get("graph_mode", "full")),
                )
                groups[key].append(record)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid telemetry record at line {line_number}") from exc

    rows: list[CostEstimate] = []
    fields = ("draft_ms", "verify_ms", "sample_ms", "scheduler_ms")
    for (batch, k, context, graph), records in sorted(groups.items()):
        if len(records) < args.min_samples:
            raise ValueError(
                f"B={batch}, K={k}, {context}/{graph} has {len(records)} samples; "
                f"need {args.min_samples}"
            )
        medians = {
            field: float(np.median([float(row[field]) for row in records])) for field in fields
        }
        rows.append(
            CostEstimate(
                batch_size=batch,
                k=k,
                context_bucket=context,
                graph_mode=graph,
                **medians,
            )
        )

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["synthetic"] = False
    manifest["aggregation"] = "median"
    manifest["samples_per_bucket"] = {
        f"B{key[0]}_K{key[1]}_{key[2]}_{key[3]}": len(records)
        for key, records in sorted(groups.items())
    }
    ProfileCostModel(rows, metadata=manifest, out_of_range="error").save(args.output)
    print(f"wrote {len(rows)} profile rows to {args.output}")


if __name__ == "__main__":
    main()
