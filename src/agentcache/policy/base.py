"""Eviction policy interface and registry.

All policies see exactly the same information at exactly the same decision
points, which is what makes the benchmark comparison fair. The only
exception is :class:`~agentcache.policy.oracle.OraclePolicy`, which is
explicitly handed future knowledge to compute an upper bound.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Mapping, Type

from ..types import AgentProgram, BlockMeta, EvictionContext, EvictionDecision

__all__ = ["EvictionPolicy", "register_policy", "get_policy", "available_policies"]


class EvictionPolicy(abc.ABC):
    """Decides which blocks leave the GPU, and where they go."""

    name: str = "base"

    @abc.abstractmethod
    def select_victims(
        self,
        candidates: List[BlockMeta],
        programs: Mapping[str, AgentProgram],
        ctx: EvictionContext,
    ) -> List[EvictionDecision]:
        """Free at least ``ctx.bytes_needed`` bytes on the GPU.

        Args:
            candidates: GPU-resident, unpinned blocks, in arbitrary order.
            programs: Program table, for state lookups.
            ctx: Live scheduler signals.

        Returns:
            Decisions in the order they should be applied. The caller stops
            once enough space is freed, so returning extra decisions is safe.
        """

    def on_hit(self, block: BlockMeta, now: float) -> None:
        """Hook: block was accessed while GPU-resident."""
        block.touch(now)

    def on_service_complete(self, duration_s: float) -> None:
        """Hook: a program released its batch slot after ``duration_s``."""

    def reset(self) -> None:
        """Clear per-run state so one instance can serve several traces."""


_REGISTRY: Dict[str, Type[EvictionPolicy]] = {}


def register_policy(cls: Type[EvictionPolicy]) -> Type[EvictionPolicy]:
    """Class decorator adding a policy to the CLI registry."""
    _REGISTRY[cls.name] = cls
    return cls


def get_policy(name: str) -> Type[EvictionPolicy]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown policy {name!r}; available: {available_policies()}")
    return _REGISTRY[name]


def available_policies() -> List[str]:
    return sorted(_REGISTRY)
