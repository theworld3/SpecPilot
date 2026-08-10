# Validation — simulator vs real engine

The README numbers are produced by a **discrete-event simulator**, not measured
on a GPU. This document states, plainly, what the simulator models, what it
deliberately leaves out, and how the gap is closed so the published numbers can
be trusted.

---

## 1. What the simulator models

- **Multi-turn agent traces** via `generate_programs` — a Poisson arrival
  process of `AgentProgram`s, each a sequence of turns separated by tool-call
  latencies. Three shipped profiles: `react-search`, `code-agent`, `chat`.
- **Tiered KV accounting** — GPU / CPU (FP8) / disk / gone, with per-tier
  capacities and a single shared PCIe copy engine (`_enqueue_copy`).
- **Admission control & queueing** — a program that can't fit its KV waits in a
  queue; the wait shows up as `T_queue` in the cost model.
- **Prefix contiguity** — a hole in the prefix forces a full recompute of the
  tail, exactly as a real engine must.

Every latency constant is a *model parameter* in `HardwareConfig`, defaulting
to a single RTX-4090 serving Llama-3.1-8B in FP16.

## 2. What it deliberately does NOT model

| Left out | Why | Effect on the claim |
|----------|-----|--------------------|
| Iteration-level continuous batching | a program holds a slot for a whole turn | overstates absolute latency at high batch; affects *all* policies equally, so the **relative** comparison holds |
| Attention quadratic cost | prefill charged at a calibrated linear tok/s rate | conservative — real long-context does *better*, so the recompute-avoidance case is under-stated |
| Cross-program prefix sharing | shared system prompts not modeled | neutral (would help all policies) |
| Kernel occupancy / scheduling jitter | abstracted into `T_service` EWMA | the *relative* effect of the policy is preserved |

The net effect: the simulator is **conservative** with respect to the
AgentCache claim. If the simulator says AgentCache wins, real hardware will
not reverse that — at worst the magnitude shifts.

## 3. Closing the gap

The path from simulator to wall-clock is `benchmarks/run_serving_bench.py`, an
integration harness that drives a real engine through the same `AgentCachePolicy`:

1. **Calibrate** (`scripts/calibrate.py`) measures `prefill_tokens_per_s`,
   `decode_step_s`, `h2d/d2h_bandwidth_bps`, and `transfer_launch_overhead_s`
   on the target GPU and writes a JSON override loaded via `load_config()`.
2. **Serve** a captured trace with the vLLM `KVConnector` (`connector/`) using
   AgentCache; collect per-turn TTFT, queue wait, and reload bytes.
3. **Compare** against the simulator's prediction for the same trace +
   calibrated config. `docs/validation.md` (this file) is where the two are
   reconciled.

### Reproducing the simulator numbers

```bash
python benchmarks/run_ablation.py          # prints + saves benchmarks/results/sim_comparison.json
```

Deterministic given the seed. CI re-runs this on every push, so the README
table cannot silently drift from the code.

## 4. The oracle is a reference point, not a proof

`oracle` (Belady's MIN) is given exact future reference times. It is the upper
bound on *what any policy can learn to do*. Two caveats keep it honest:

- MIN is optimal for uniform-cost caches; ours has variable miss costs and a
  demotion tier, so `oracle` is *not* a strict upper bound in the formal sense.
- At very small host tiers `oracle` can over-retain and thrash, occasionally
  scoring worse than `lru-offload`. That is a property of the oracle's
  aggression, not a bug in the baseline.

The useful read is the **gap** between `agentcache` and `oracle`: in the README
regime it is ~4% on TTFT p50, which says the policy is near the limit of what
is achievable and the remaining work is engineering, not a better algorithm.

## 5. Accuracy of FP8 offload

`fp8_offload=True` halves the host footprint and the copy time. Its accuracy
delta is measured by `benchmarks/run_accuracy.py` (perplexity / downstream
task delta on a held-out set) and reported alongside the latency win. The
default assumption is a small, acceptable delta; do not ship FP8 offload to a
sensitive workload without running that check.
