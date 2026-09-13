"""Aggregate repeated server runs with percentile and bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, repeats: int
) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    samples = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")

    grouped: dict[str, list[dict[str, object]]] = {}
    for path in args.inputs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("synthetic") is not False:
            raise ValueError(f"{path} is not marked as a real benchmark")
        label = str(payload["config"]["label"])
        grouped.setdefault(label, []).append(payload)

    rng = np.random.default_rng(args.seed)
    output: dict[str, object] = {"bootstrap_repeats": args.bootstrap, "labels": {}}
    for label, runs in sorted(grouped.items()):
        metrics: dict[str, object] = {"repeats": len(runs)}
        for field in ("output_throughput_tps", "request_throughput_rps"):
            values = np.asarray([run["summary"][field] for run in runs], dtype=np.float64)
            low, high = _bootstrap_mean(values, rng, args.bootstrap)
            metrics[field] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "bootstrap_95_ci": [low, high],
            }
        output["labels"][label] = metrics

    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
