"""Generate the on-simulator policy comparison used in the README / docs.

Pure stdlib (no GPU). Produces a stable table and writes it to
``benchmarks/results/sim_comparison.json`` so the numbers in the README are
reproducible rather than hand-typed.

Run::

    PYTHONPATH=src python benchmarks/run_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow `python benchmarks/run_ablation.py` without PYTHONPATH=src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentcache.config import HardwareConfig, SystemConfig
from agentcache.sim import generate_programs, run_comparison, speedup_table

GIB = 1024 ** 3
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "sim_comparison.json")

POLICIES = (
    "lru",
    "lru-offload",
    "agentcache",
    "agentcache-block",
    "agentcache-no-queue",
    "oracle",
)


def main() -> None:
    cfg = SystemConfig(
        hardware=HardwareConfig(
            gpu_kv_capacity_bytes=8 * GIB,
            cpu_kv_capacity_bytes=8 * GIB,
        ),
        max_batch_size=8,
    )
    programs = generate_programs(
        "code-agent", num_programs=32, arrival_rate_qps=0.5, seed=1
    )
    results = run_comparison(programs, cfg, POLICIES)
    table = speedup_table(results, baseline="lru-offload")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            {
                "config": {
                    "gpu_kv_gib": 8,
                    "cpu_kv_gib": 8,
                    "max_batch_size": 8,
                    "profile": "code-agent",
                    "num_programs": 32,
                    "arrival_qps": 0.5,
                    "seed": 1,
                },
                "table": table,
            },
            f,
            indent=2,
        )

    # Pretty print
    cols = [
        "policy",
        "TTFT p50 (ms)",
        "TTFT p95 (ms)",
        "cache hit",
        "TTFT speedup",
        "tput speedup",
        "recompute tok",
    ]
    width = {}
    for c in cols:
        w = len(c)
        for r in table:
            w = max(w, len(f"{r[c]:.3f}") if isinstance(r[c], float) else len(str(r[c])))
        width[c] = w
    print("  ".join(c.ljust(width[c]) for c in cols))
    print("  ".join("-" * width[c] for c in cols))
    for r in table:
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                row.append(f"{v:.3f}".ljust(width[c]))
            else:
                row.append(str(v).ljust(width[c]))
        print("  ".join(row))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
