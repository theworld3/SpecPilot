"""Belady-style oracle: the upper bound on what any policy can achieve.

This is the single most useful experiment in the whole repo and it should
be run *before* writing any kernel code. If the oracle only beats
``lru-offload`` by a few percent, the entire project premise is wrong and
you should pick a different problem. If the gap is large, everything after
this point is engineering rather than gambling.

The oracle is given the one thing no real policy has: the exact time each
block will next be referenced. It evicts the block referenced furthest in
the future (Belady's MIN), and drops -- rather than demotes -- anything that
is never referenced again, so it also never wastes host bandwidth.

It is *not* a strict upper bound in the formal sense: MIN is optimal for
uniform-cost caches, while ours has variable miss costs and a demotion
tier. It is a strong, cheap-to-compute reference point, and the gap between
it and ``agentcache`` is the honest measure of headroom left.
"""

from __future__ import annotations

import math
from typing import Callable, List, Mapping, Optional

from ..config import SystemConfig
from ..types import AgentProgram, BlockMeta, EvictionContext, EvictionDecision, Tier
from .base import EvictionPolicy, register_policy

__all__ = ["OraclePolicy"]

NextRefFn = Callable[[BlockMeta, float], float]


@register_policy
class OraclePolicy(EvictionPolicy):
    """Requires :meth:`bind` to be called with future-knowledge before use."""

    name = "oracle"

    def __init__(
        self,
        config: SystemConfig,
        *,
        next_ref_fn: Optional[NextRefFn] = None,
        cpu_full_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.config = config
        self._next_ref_fn = next_ref_fn
        self._cpu_full_fn = cpu_full_fn or (lambda: False)

    def bind(self, next_ref_fn: NextRefFn) -> OraclePolicy:
        self._next_ref_fn = next_ref_fn
        return self

    def select_victims(
        self,
        candidates: List[BlockMeta],
        programs: Mapping[str, AgentProgram],
        ctx: EvictionContext,
    ) -> List[EvictionDecision]:
        if self._next_ref_fn is None:
            raise RuntimeError(
                "OraclePolicy.bind(next_ref_fn) must be called by the simulator"
            )

        # Furthest next reference first; never-referenced blocks first of all.
        ranked = sorted(
            candidates,
            key=lambda b: (-self._next_ref_fn(b, ctx.now), b.block_id),
        )

        decisions: List[EvictionDecision] = []
        freed = 0
        for block in ranked:
            if freed >= ctx.bytes_needed:
                break
            next_t = self._next_ref_fn(block, ctx.now)
            if math.isinf(next_t):
                target = Tier.GONE  # never needed again: do not waste bandwidth
            elif self.config.enable_cpu_tier and not self._cpu_full_fn():
                target = Tier.CPU
            else:
                target = Tier.GONE
            freed += self.config.model.bytes_for(block.num_tokens)
            decisions.append(EvictionDecision(block.block_id, target, score=next_t))
        return decisions
