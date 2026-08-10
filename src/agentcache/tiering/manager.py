"""Tiered KV cache manager: GPU / CPU / disk bookkeeping.

Deliberately free of any PyTorch dependency. The manager owns *metadata and
decisions*; actually moving bytes is delegated to a
:class:`~agentcache.tiering.backend.StorageBackend`. The simulator injects a
null backend, the real runtime injects a pinned-memory CUDA backend, and both
exercise the identical policy and accounting code -- so a bug found in
simulation is a bug fixed on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from ..config import SystemConfig
from ..policy.base import EvictionPolicy
from ..types import (
    AgentProgram,
    BlockMeta,
    EvictionContext,
    EvictionDecision,
    Tier,
)

__all__ = ["LookupResult", "TieredCacheManager"]


@dataclass
class LookupResult:
    """What a program found when it came back from a tool call."""

    gpu_blocks: List[BlockMeta] = field(default_factory=list)
    reload_blocks: List[BlockMeta] = field(default_factory=list)
    """Resident on CPU/disk: need a transfer."""
    missing_tokens: int = 0
    """Tokens whose KV is gone entirely: need a prefill."""
    reload_bytes: int = 0
    covered_blocks: int = 0
    """Length of the usable prefix, in blocks. Everything from this index on
    must be discarded and rebuilt even if it happens to still be resident --
    prefill cannot skip a hole."""

    @property
    def hit_tokens(self) -> int:
        return sum(b.num_tokens for b in self.gpu_blocks) + sum(
            b.num_tokens for b in self.reload_blocks
        )


class TieredCacheManager:
    """Allocates, evicts and promotes KV blocks across tiers."""

    def __init__(self, config: SystemConfig, policy: EvictionPolicy) -> None:
        self.config = config
        self.policy = policy
        self.blocks: Dict[int, BlockMeta] = {}
        self._by_program: Dict[str, List[BlockMeta]] = {}
        self._tier_bytes: Dict[Tier, int] = {t: 0 for t in Tier}
        self._tier_index: Dict[Tier, Dict[int, BlockMeta]] = {t: {} for t in Tier}
        """Per-tier index. Without it every eviction scans the whole block
        table, which degenerates to O(n^2 log n) once the host tier fills up
        and starts needing evictions of its own."""
        self._next_block_id = 0

        # --- counters, surfaced in the benchmark report ---
        self.stat_gpu_hit_tokens = 0
        self.stat_reload_tokens = 0
        self.stat_recompute_tokens = 0
        self.stat_evicted_blocks = 0
        self.stat_demoted_blocks = 0
        self.stat_dropped_blocks = 0
        self.stat_bytes_to_cpu = 0
        self.stat_bytes_from_cpu = 0
        self.stat_cpu_dropped_blocks = 0

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------
    def _block_bytes(self, block: BlockMeta) -> int:
        return self.config.model.bytes_for(block.num_tokens)

    def tier_bytes(self, tier: Tier) -> int:
        return self._tier_bytes[tier]

    @property
    def gpu_bytes_used(self) -> int:
        return self._tier_bytes[Tier.GPU]

    @property
    def gpu_bytes_free(self) -> int:
        return self.config.hardware.gpu_kv_capacity_bytes - self.gpu_bytes_used

    def cpu_full(self) -> bool:
        if not self.config.enable_cpu_tier:
            return True
        return self._tier_bytes[Tier.CPU] >= self.config.hardware.cpu_kv_capacity_bytes

    def disk_full(self) -> bool:
        if not self.config.enable_disk_tier:
            return True
        return self._tier_bytes[Tier.DISK] >= self.config.hardware.disk_kv_capacity_bytes

    def stored_bytes(self, block: BlockMeta, tier: Optional[Tier] = None) -> int:
        """Footprint of a block in a given tier (FP8 halves the host copy)."""
        tier = tier if tier is not None else block.tier
        raw = self._block_bytes(block)
        if tier is Tier.CPU:
            return self.config.offload_bytes(raw)
        if tier is Tier.GONE:
            return 0
        return raw

    def _set_tier(self, block: BlockMeta, tier: Tier) -> None:
        self._tier_bytes[block.tier] -= self.stored_bytes(block, block.tier)
        self._tier_index[block.tier].pop(block.block_id, None)
        block.tier = tier
        self._tier_bytes[tier] += self.stored_bytes(block, tier)
        self._tier_index[tier][block.block_id] = block

    # ------------------------------------------------------------------
    # Lookup / allocation
    # ------------------------------------------------------------------
    def program_blocks(self, program_id: str) -> List[BlockMeta]:
        return self._by_program.get(program_id, [])

    def lookup(self, program_id: str, num_tokens: int, now: float) -> LookupResult:
        """Classify the first ``num_tokens`` of a program's context by tier.

        Prefill is prefix-contiguous, so the result is truncated at the first
        hole: everything after a ``GONE`` block must be recomputed regardless
        of whether it happens to still be resident. Modelling this honestly
        is what stops the simulator from over-crediting partial hits.
        """
        result = LookupResult()
        blocks = sorted(self.program_blocks(program_id), key=lambda b: b.seq_index)
        covered = 0

        for expected_index, block in enumerate(blocks):
            # Dropped blocks are removed outright, so a gap in seq_index *is*
            # the hole. Checking indices rather than relying on a sentinel
            # keeps the two representations from drifting apart.
            if block.seq_index != expected_index:
                break
            if covered >= num_tokens or block.tier is Tier.GONE:
                break
            if block.tier is Tier.GPU:
                result.gpu_blocks.append(block)
                block.touch(now)
            else:
                result.reload_blocks.append(block)
                result.reload_bytes += self.stored_bytes(block, block.tier)
            covered += block.num_tokens
            result.covered_blocks += 1

        result.missing_tokens = max(0, num_tokens - covered)
        self.stat_gpu_hit_tokens += sum(b.num_tokens for b in result.gpu_blocks)
        self.stat_reload_tokens += sum(b.num_tokens for b in result.reload_blocks)
        self.stat_recompute_tokens += result.missing_tokens
        return result

    def truncate(self, program_id: str, keep_blocks: int) -> int:
        """Drop everything from block index ``keep_blocks`` onwards.

        Called when a program comes back and finds a hole in its prefix. The
        tail is unusable (prefill is prefix-contiguous), so keeping it around
        would leak both memory and, worse, make ``lookup`` credit hits that
        can never be redeemed.

        Returns the number of blocks discarded.
        """
        blocks = sorted(self.program_blocks(program_id), key=lambda b: b.seq_index)
        if keep_blocks >= len(blocks):
            return 0
        keep, drop = blocks[:keep_blocks], blocks[keep_blocks:]
        for block in drop:
            self._tier_bytes[block.tier] -= self.stored_bytes(block, block.tier)
            self._tier_index[block.tier].pop(block.block_id, None)
            self.blocks.pop(block.block_id, None)
        self._by_program[program_id] = keep
        return len(drop)

    def ensure_space(
        self,
        needed_bytes: int,
        ctx: EvictionContext,
        programs: Mapping[str, AgentProgram],
        protected: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, List[EvictionDecision]]:
        """Free ``needed_bytes`` on the GPU.

        Returns ``(succeeded, decisions)``. The decisions are handed back so
        the caller can charge their transfer cost to the copy engine.

        Failure is a normal outcome, not an error: it means the admission
        controller must leave this program queued until a running turn
        finishes. Modelling that back-pressure is the whole reason capacity
        pressure surfaces as queueing delay.
        """
        if needed_bytes <= self.gpu_bytes_free:
            return True, []
        decisions = self.evict(ctx, programs, protected=protected)
        return needed_bytes <= self.gpu_bytes_free, decisions

    def allocate(
        self,
        program: AgentProgram,
        num_tokens: int,
        now: float,
    ) -> List[BlockMeta]:
        """Append GPU blocks for ``num_tokens``.

        Space must already have been secured via :meth:`ensure_space`; this
        method does no eviction so that allocation can never silently evict
        the caller's own context.
        """
        block_size = self.config.block_size_tokens
        start_index = len(self.program_blocks(program.program_id))
        needed_blocks = (num_tokens + block_size - 1) // block_size

        new_blocks: List[BlockMeta] = []
        remaining = num_tokens
        for i in range(needed_blocks):
            n = min(block_size, remaining)
            remaining -= n
            block = BlockMeta(
                block_id=self._next_block_id,
                program_id=program.program_id,
                seq_index=start_index + i,
                num_tokens=n,
                tier=Tier.GPU,
                last_access_t=now,
                created_t=now,
            )
            self._next_block_id += 1
            self.blocks[block.block_id] = block
            self._by_program.setdefault(program.program_id, []).append(block)
            self._tier_bytes[Tier.GPU] += self._block_bytes(block)
            self._tier_index[Tier.GPU][block.block_id] = block
            new_blocks.append(block)

        return new_blocks

    # ------------------------------------------------------------------
    # Eviction / promotion
    # ------------------------------------------------------------------
    def gpu_candidates(self, protected: Optional[Iterable[str]] = None) -> List[BlockMeta]:
        """GPU-resident blocks eligible for eviction.

        ``protected`` holds the programs currently occupying a batch slot
        plus the one being admitted. Evicting them would model a preemption
        that no engine performs mid-turn, and would let a policy "win" by
        stealing memory from work in flight.
        """
        blocked = set(protected or ())
        return [
            b
            for b in self._tier_index[Tier.GPU].values()
            if not b.pinned and b.program_id not in blocked
        ]

    def _make_room_in_cpu(
        self,
        bytes_needed: int,
        ctx: EvictionContext,
        programs: Mapping[str, AgentProgram],
        protected: Optional[Iterable[str]] = None,
    ) -> bool:
        """Evict from the host tier so a GPU demotion can land.

        The host tier is not a landfill. It is a second cache with its own
        capacity, and deciding *what to keep there* is a second instance of
        the same problem -- one that offload-everything designs answer with
        "whatever arrived first", which is strictly worse than answering it
        with the same cost model.
        """
        hw = self.config.hardware
        free = hw.cpu_kv_capacity_bytes - self._tier_bytes[Tier.CPU]
        if free >= bytes_needed:
            return True

        candidates = list(self._tier_index[Tier.CPU].values())
        if protected is not None:
            blocked = set(protected)
            candidates = [b for b in candidates if b.program_id not in blocked]
        if not candidates:
            return False

        sub_ctx = replace(ctx, bytes_needed=bytes_needed - free)
        decisions = self.policy.select_victims(candidates, programs, sub_ctx)

        freed = 0
        for decision in decisions:
            if freed >= sub_ctx.bytes_needed:
                break
            block = self.blocks.get(decision.block_id)
            if block is None or block.tier is not Tier.CPU:
                continue
            freed += self.stored_bytes(block, Tier.CPU)
            self.stat_cpu_dropped_blocks += 1
            self._set_tier(block, Tier.GONE)
            self._forget(block)

        return (hw.cpu_kv_capacity_bytes - self._tier_bytes[Tier.CPU]) >= bytes_needed

    def _reserve_cpu_for(
        self,
        decisions: List[EvictionDecision],
        ctx: EvictionContext,
        programs: Mapping[str, AgentProgram],
        protected: Optional[Iterable[str]] = None,
    ) -> None:
        """Free host capacity once for a whole eviction wave.

        Doing it per victim is also correct but quadratic -- enough to make
        the simulator run out of memory on a 30-program trace, which is how
        this function came to exist.
        """
        wanted = 0
        freed = 0
        for decision in decisions:
            if freed >= ctx.bytes_needed:
                break
            block = self.blocks.get(decision.block_id)
            if block is None or block.tier is not Tier.GPU:
                continue
            freed += self._block_bytes(block)
            if decision.target_tier is Tier.CPU:
                wanted += self.stored_bytes(block, Tier.CPU)
        if wanted > 0:
            self._make_room_in_cpu(wanted, ctx, programs, protected)

    def evict(
        self,
        ctx: EvictionContext,
        programs: Mapping[str, AgentProgram],
        protected: Optional[Iterable[str]] = None,
    ) -> List[EvictionDecision]:
        candidates = self.gpu_candidates(protected)
        decisions = self.policy.select_victims(candidates, programs, ctx)
        self._reserve_cpu_for(decisions, ctx, programs, protected)

        freed = 0
        applied: List[EvictionDecision] = []
        for decision in decisions:
            if freed >= ctx.bytes_needed:
                break
            block = self.blocks.get(decision.block_id)
            if block is None or block.tier is not Tier.GPU:
                continue
            target = decision.target_tier
            if target is Tier.CPU and self.cpu_full():
                target = Tier.GONE
            if target is Tier.DISK and self.disk_full():
                target = Tier.GONE

            freed += self._block_bytes(block)
            self.stat_evicted_blocks += 1
            if target is Tier.GONE:
                self.stat_dropped_blocks += 1
            else:
                self.stat_demoted_blocks += 1
                if target is Tier.CPU:
                    self.stat_bytes_to_cpu += self.stored_bytes(block, Tier.CPU)

            self._set_tier(block, target)
            applied.append(EvictionDecision(block.block_id, target, decision.score))
            if target is Tier.GONE:
                self._forget(block)

        return applied

    def _forget(self, block: BlockMeta) -> None:
        """Physically remove a dropped block.

        A ``GONE`` block holds no data, so keeping the object around would
        only slow down candidate scans and risk it being counted as a hit.
        Deletion leaves a gap in ``seq_index``, which is exactly how
        :meth:`lookup` detects the hole.
        """
        self.blocks.pop(block.block_id, None)
        self._tier_index[block.tier].pop(block.block_id, None)
        siblings = self._by_program.get(block.program_id)
        if siblings is not None:
            self._by_program[block.program_id] = [
                b for b in siblings if b.block_id != block.block_id
            ]

    def promote(self, blocks: Iterable[BlockMeta], now: float) -> int:
        """Bring blocks back to GPU. Returns bytes transferred."""
        moved = 0
        for block in blocks:
            if block.tier is Tier.GPU or block.tier is Tier.GONE:
                continue
            moved += self.stored_bytes(block, block.tier)
            if block.tier is Tier.CPU:
                self.stat_bytes_from_cpu += self.stored_bytes(block, Tier.CPU)
            self._set_tier(block, Tier.GPU)
            block.touch(now)
        return moved

    def release_program(self, program_id: str) -> None:
        """Program finished: reclaim everything it owns."""
        for block in self._by_program.pop(program_id, []):
            self._tier_bytes[block.tier] -= self.stored_bytes(block, block.tier)
            self._tier_index[block.tier].pop(block.block_id, None)
            self.blocks.pop(block.block_id, None)

    def snapshot(self) -> Dict[str, int]:
        return {
            "gpu_bytes": self._tier_bytes[Tier.GPU],
            "cpu_bytes": self._tier_bytes[Tier.CPU],
            "disk_bytes": self._tier_bytes[Tier.DISK],
            "num_blocks": len(self.blocks),
        }
