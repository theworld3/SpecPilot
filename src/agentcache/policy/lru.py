"""Baseline policies: what production engines do today.

``lru``
    Pure LRU drop. This is vLLM's default prefix-cache eviction and
    SGLang's RadixAttention LRU. Evicted blocks are gone; the next
    reference pays a full prefill.

``lru-offload``
    LRU order, but victims are demoted to a pinned-host-memory tier before
    being dropped. This is the strong baseline -- it is what vLLM's CPU
    ``KVConnector`` and LMCache give you. Beating *this* is the bar; beating
    plain LRU is not interesting.

Both are deliberately implemented against the same interface as AgentCache
so the only variable in the benchmark is the decision rule.
"""

from __future__ import annotations

from typing import List, Mapping

from ..config import SystemConfig
from ..types import AgentProgram, BlockMeta, EvictionContext, EvictionDecision, Tier
from .base import EvictionPolicy, register_policy

__all__ = ["LRUPolicy", "LRUOffloadPolicy"]


@register_policy
class LRUPolicy(EvictionPolicy):
    """Least-recently-used, drop on evict."""

    name = "lru"

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def select_victims(
        self,
        candidates: List[BlockMeta],
        programs: Mapping[str, AgentProgram],
        ctx: EvictionContext,
    ) -> List[EvictionDecision]:
        victims = sorted(candidates, key=lambda b: (b.last_access_t, b.block_id))
        decisions: List[EvictionDecision] = []
        freed = 0
        for block in victims:
            if freed >= ctx.bytes_needed:
                break
            freed += self.config.model.bytes_for(block.num_tokens)
            decisions.append(
                EvictionDecision(block.block_id, Tier.GONE, score=block.last_access_t)
            )
        return decisions


@register_policy
class LRUOffloadPolicy(EvictionPolicy):
    """LRU order with a host-memory tier. The competitive baseline."""

    name = "lru-offload"

    def __init__(self, config: SystemConfig, cpu_full_fn=None) -> None:
        self.config = config
        self._cpu_full_fn = cpu_full_fn or (lambda: False)

    def select_victims(
        self,
        candidates: List[BlockMeta],
        programs: Mapping[str, AgentProgram],
        ctx: EvictionContext,
    ) -> List[EvictionDecision]:
        victims = sorted(candidates, key=lambda b: (b.last_access_t, b.block_id))
        decisions: List[EvictionDecision] = []
        freed = 0
        for block in victims:
            if freed >= ctx.bytes_needed:
                break
            freed += self.config.model.bytes_for(block.num_tokens)
            target = Tier.GONE
            if self.config.enable_cpu_tier and not self._cpu_full_fn():
                target = Tier.CPU
            decisions.append(
                EvictionDecision(block.block_id, target, score=block.last_access_t)
            )
        return decisions
