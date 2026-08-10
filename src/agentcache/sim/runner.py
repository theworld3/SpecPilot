"""Multi-policy comparison harness.

Guarantees the benchmark is fair:

* every policy gets a **fresh deep copy** of the trace (programs carry
  mutable run state);
* every policy gets a **fresh cache manager** (no warm start);
* the config is identical except for the policy under test.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from ..config import SystemConfig
from ..policy.agentcache import AgentCachePolicy
from ..policy.base import EvictionPolicy
from ..policy.lru import LRUOffloadPolicy, LRUPolicy
from ..policy.oracle import OraclePolicy
from ..types import AgentProgram
from .simulator import SimulationResult, Simulator
from .workload import clone_programs

__all__ = ["build_policy", "run_comparison", "speedup_table"]

PolicyFactory = Callable[[SystemConfig], EvictionPolicy]

_BUILDERS: Dict[str, PolicyFactory] = {
    "lru": lambda cfg: LRUPolicy(cfg),
    "lru-offload": lambda cfg: LRUOffloadPolicy(cfg),
    "agentcache": lambda cfg: AgentCachePolicy(cfg),
    "agentcache-no-queue": lambda cfg: AgentCachePolicy(cfg, enable_queue_term=False),
    "agentcache-block": lambda cfg: AgentCachePolicy(cfg, program_granular=False),
    "oracle": lambda cfg: OraclePolicy(cfg),
}


def build_policy(name: str, config: SystemConfig) -> EvictionPolicy:
    if name not in _BUILDERS:
        raise KeyError(f"unknown policy {name!r}; available: {sorted(_BUILDERS)}")
    policy = _BUILDERS[name](config)
    # Ablation variants share a class; keep the reported name distinct.
    policy.name = name  # type: ignore[misc]
    return policy


def run_comparison(
    programs: Sequence[AgentProgram],
    config: SystemConfig,
    policies: Sequence[str] = ("lru", "lru-offload", "agentcache", "oracle"),
    *,
    decode_contention: float = 0.35,
) -> List[SimulationResult]:
    results: List[SimulationResult] = []
    for name in policies:
        sim = Simulator(
            config,
            build_policy(name, config),
            clone_programs(programs),
            decode_contention=decode_contention,
        )
        results.append(sim.run())
    return results


def speedup_table(
    results: Sequence[SimulationResult], baseline: str = "lru-offload"
) -> List[Dict[str, object]]:
    """Normalise every result against ``baseline``.

    Reporting raw numbers only is how people accidentally claim 10x. The
    speedup is always stated against the *strong* baseline (LRU + CPU
    offload), never against plain LRU.
    """
    base = next((r for r in results if r.policy == baseline), None)
    rows: List[Dict[str, object]] = []
    for r in results:
        row: Dict[str, object] = {
            "policy": r.policy,
            "TTFT p50 (ms)": r.ttft["p50"] * 1000,
            "TTFT p95 (ms)": r.ttft["p95"] * 1000,
            "e2e p95 (s)": r.program_latency["p95"],
            "throughput (tok/s)": r.throughput_tok_s,
            "cache hit": r.cache_hit_rate,
            "recompute tok": r.recomputed_tokens,
        }
        if base is not None and base.ttft["p50"] > 0:
            row["TTFT speedup"] = base.ttft["p50"] / r.ttft["p50"] if r.ttft["p50"] else 0.0
            row["tput speedup"] = (
                r.throughput_tok_s / base.throughput_tok_s
                if base.throughput_tok_s
                else 0.0
            )
        rows.append(row)
    return rows
