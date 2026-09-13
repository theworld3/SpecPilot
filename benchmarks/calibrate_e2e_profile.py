"""Build an approximate B x K profile from repeated online benchmark runs.

This path is convenient for a first shadow-mode calibration. It estimates one
verification-round cost as median TPOT times mean acceptance length. A profile
built from CUDA-event phase telemetry with ``profile_cost.py`` is preferred for
final results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from specpilot.cost_model import CostEstimate, ProfileCostModel


def _metric_delta(payload: dict[str, object], suffix: str) -> float:
    metrics = payload.get("spec_decode_metric_deltas", {})
    matches = [float(value) for name, value in metrics.items() if name.endswith(suffix)]
    return sum(matches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grouped: dict[tuple[int, int], list[float]] = {}
    for path in args.inputs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = payload["config"]
        k = config.get("k")
        if k is None:
            raise ValueError(f"{path} has no static --k metadata")
        batch = int(config["concurrency"])
        tpot_ms = payload["summary"]["tpot_ms"]["p50"]
        if tpot_ms is None:
            raise ValueError(f"{path} has no TPOT samples")
        if int(k) == 0:
            mean_acceptance_length = 1.0
        else:
            drafts = _metric_delta(payload, "spec_decode_num_drafts_total")
            accepted = _metric_delta(payload, "spec_decode_num_accepted_tokens_total")
            if drafts <= 0.0:
                raise ValueError(f"{path} has no speculative metrics; use --collect-metrics")
            mean_acceptance_length = 1.0 + accepted / drafts
        step_ms = float(tpot_ms) * mean_acceptance_length
        grouped.setdefault((batch, int(k)), []).append(step_ms)

    rows = [
        CostEstimate(
            batch_size=batch,
            k=k,
            draft_ms=0.0,
            verify_ms=float(np.median(samples)),
            sample_ms=0.0,
            scheduler_ms=0.0,
        )
        for (batch, k), samples in sorted(grouped.items())
    ]
    metadata = {
        "synthetic": False,
        "approximate": True,
        "method": "median_tpot_x_mean_acceptance_length",
        "warning": "Use CUDA-event phase telemetry for final performance claims.",
        "source_runs": args.inputs,
    }
    ProfileCostModel(rows, metadata=metadata, out_of_range="error").save(args.output)
    print(f"wrote {len(rows)} approximate profile rows to {args.output}")


if __name__ == "__main__":
    main()
