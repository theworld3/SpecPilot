# Experiment report

No GPU result is claimed in this repository snapshot.

The CPU demo is synthetic and exists only to test controller behavior. Real
numbers must be rebuilt from `results/raw/` and linked to a completed manifest.
Use at least three independent repeats per workload and compare against every
static K in the same environment.

## Required result table

| Workload | Concurrency | Policy | Output tok/s | P95 TPOT | Mean AL | Repeats |
|---|---:|---|---:|---:|---:|---:|
| _pending_ | _pending_ | no-spec | | | | |
| _pending_ | _pending_ | best static K | | | | |
| _pending_ | _pending_ | batch LUT | | | | |
| _pending_ | _pending_ | SpecPilot | | | | |

## Negative results

Record high-load regressions, profile misses, graph fallbacks, and workload
segments where the controller does not beat the best static baseline.

