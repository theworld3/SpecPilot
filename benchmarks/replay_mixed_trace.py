"""Generate and replay the deterministic mixed synthetic workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specpilot.controller import ControllerConfig, GoodputController
from specpilot.simulator import relative_change, replay_dynamic, sweep_static
from specpilot.synthetic import generate_mixed_trace, make_synthetic_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--output", default="results/raw/synthetic_replay.json")
    args = parser.parse_args()

    profile = make_synthetic_profile(k_max=args.k_max)
    trace = generate_mixed_trace(steps=args.steps, seed=args.seed, k_max=args.k_max)
    static_results = sweep_static(trace, profile, args.k_max)
    best = max(static_results, key=lambda result: result.output_tps)
    dynamic = replay_dynamic(
        trace,
        GoodputController(
            profile,
            ControllerConfig(k_max=args.k_max, explore_interval=64),
        ),
    )
    payload = {
        "synthetic": True,
        "warning": "This is a simulator result, not a GPU measurement.",
        "seed": args.seed,
        "steps": args.steps,
        "static": [result.summary_dict() for result in static_results],
        "specpilot": dynamic.summary_dict(),
        "specpilot_vs_best_static_output_tps_pct": relative_change(
            dynamic.output_tps, best.output_tps
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
