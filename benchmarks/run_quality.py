"""Fast statistical smoke test for the reference rejection sampler."""

from __future__ import annotations

import argparse
import json

import numpy as np

from specpilot.sampling import speculative_sample_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-tvd", type=float, default=0.01)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    target = np.asarray((0.05, 0.15, 0.30, 0.50))
    draft = np.asarray((0.40, 0.10, 0.10, 0.40))
    rng = np.random.default_rng(args.seed)
    counts = np.zeros(target.size, dtype=np.int64)
    accepted = 0
    for _ in range(args.samples):
        result = speculative_sample_one(target, draft, rng)
        counts[result.token] += 1
        accepted += result.accepted
    empirical = counts / args.samples
    tvd = 0.5 * float(np.abs(empirical - target).sum())
    payload = {
        "samples": args.samples,
        "seed": args.seed,
        "target": target.tolist(),
        "empirical": empirical.tolist(),
        "tvd": tvd,
        "acceptance_rate": accepted / args.samples,
        "threshold": args.max_tvd,
        "passed": tvd <= args.max_tvd,
    }
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
