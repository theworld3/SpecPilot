"""Replay a real trace through the AgentCache admission engine.

Integration harness. On a GPU box this drives :class:`~agentcache.scheduler.
program_scheduler.ProgramScheduler` (with a real ``PinnedHostBackend``) over a
captured agent trace and reports the per-policy TTFT decomposition. On a
CPU/CI box without torch it prints an explicit skip and exits 0.

This measures the *policy's* effect (eviction/reload decisions and the
resulting TTFT decomposition). Actual byte-transfer time on a live engine is
produced by the vLLM connector (``connector/``); the two share the scheduler,
so the decision trace is identical.

Usage::

    python benchmarks/run_serving_bench.py --trace benchmarks/traces/code-agent.jsonl
    python benchmarks/run_serving_bench.py --trace traces.jsonl --policy agentcache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentcache.config import HardwareConfig, SystemConfig
from agentcache.scheduler.program_scheduler import ProgramScheduler
from agentcache.sim.runner import build_policy
from agentcache.sim.workload import clone_programs, load_trace
from agentcache.utils.metrics import summarize


def _requires_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--policy", default="agentcache")
    ap.add_argument("--gpu-kv-gib", type=float, default=8.0)
    ap.add_argument("--cpu-kv-gib", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if not _requires_torch():
        print("SKIP: torch not installed (GPU integration). Install with `[gpu]` extra.")
        sys.exit(0)

    from agentcache.tiering import PinnedHostBackend  # lazy

    cfg = SystemConfig(
        hardware=HardwareConfig(
            gpu_kv_capacity_bytes=int(args.gpu_kv_gib * 1024 ** 3),
            cpu_kv_capacity_bytes=int(args.cpu_kv_gib * 1024 ** 3),
        ),
        max_batch_size=args.batch,
    )
    base_programs = load_trace(args.trace)
    programs = clone_programs(base_programs)
    scheduler = ProgramScheduler(cfg, build_policy(args.policy, cfg),
                                 backend=PinnedHostBackend(cfg.hardware.cpu_kv_capacity_bytes))

    ttfts = []
    for prog in programs:
        scheduler.register(prog)
        now = prog.arrival_t
        for turn in prog.turns:
            plan = scheduler.plan_admission(prog, turn.new_prompt_tokens, now,
                                            queue_len=0, running_batch_size=args.batch)
            scheduler.apply_transfers(plan)
            ttfts.append(plan.ttft_s)
            now += plan.ttft_s + turn.output_tokens * cfg.hardware.decode_step_s + turn.tool_latency_s
            scheduler.complete_turn(prog, service_time_s=plan.ttft_s, now=now)

    stats = summarize(ttfts)
    print(f"policy={args.policy}  programs={len(programs)}  turns={len(ttfts)}")
    print(f"TTFT p50={stats['p50']*1000:.0f} ms  p95={stats['p95']*1000:.0f} ms  "
          f"mean={stats['mean']*1000:.0f} ms")


if __name__ == "__main__":
    main()
