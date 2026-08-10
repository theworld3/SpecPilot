"""The AgentCache policy: agent-aware, cost-driven KV retention."""

from __future__ import annotations

from typing import Callable, List, Mapping, Optional

from ..config import SystemConfig
from ..types import (
    AgentProgram,
    BlockMeta,
    EvictionContext,
    EvictionDecision,
    ProgramState,
    Tier,
)
from .base import EvictionPolicy, register_policy
from .cost_model import QueueDelayModel, RetentionCostModel, ReturnProbabilityModel

__all__ = ["AgentCachePolicy"]


@register_policy
class AgentCachePolicy(EvictionPolicy):
    """Evict by ascending expected-saving-per-byte, demote instead of drop.

    Three things it does that LRU-offload cannot:

    1. **Reads program state.** A program parked on a tool call is protected
       even though its blocks are the oldest in the pool; a finished program
       is dropped immediately even though it was just touched.
    2. **Prices the queue.** Under contention the cost of losing a batch slot
       is folded into the retention value, so the policy gets *more*
       conservative exactly when contention makes recovery expensive.
    3. **Chooses a tier per block** instead of blanket-offloading everything,
       so scarce host bandwidth is spent on the blocks that actually earn it.

    Tail protection
    ---------------
    Within a program the *suffix* blocks are the ones a returning turn
    attends to first and the ones a partial recompute cannot skip (prefill
    is prefix-contiguous: you cannot rebuild block 40 without blocks 0..39).
    So when two blocks tie on value we evict the *earlier* one -- keeping a
    contiguous prefix is worthless if the tail is gone, whereas the reverse
    is a valid partial hit. ``prefer_suffix`` toggles this for ablation.
    """

    name = "agentcache"

    def __init__(
        self,
        config: SystemConfig,
        *,
        cost_model: Optional[RetentionCostModel] = None,
        cpu_full_fn: Optional[Callable[[], bool]] = None,
        disk_full_fn: Optional[Callable[[], bool]] = None,
        prefer_suffix: bool = True,
        enable_queue_term: bool = True,
        program_granular: bool = True,
    ) -> None:
        self.config = config
        self.cost_model = cost_model or RetentionCostModel(
            config=config,
            queue_model=QueueDelayModel(),
            return_model=ReturnProbabilityModel(),
            enable_queue_term=enable_queue_term,
        )
        self._cpu_full_fn = cpu_full_fn or (lambda: False)
        self._disk_full_fn = disk_full_fn or (lambda: False)
        self.prefer_suffix = prefer_suffix
        self.program_granular = program_granular
        """Evict whole programs rather than individual blocks. Partial
        eviction of a program is usually worthless -- see docstring."""

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.cost_model.queue_model = QueueDelayModel()

    def on_service_complete(self, duration_s: float) -> None:
        self.cost_model.queue_model.observe_service_time(duration_s)

    # ------------------------------------------------------------------
    def _score(
        self,
        block: BlockMeta,
        programs: Mapping[str, AgentProgram],
        ctx: EvictionContext,
    ) -> float:
        program = programs.get(block.program_id)
        if program is None:
            return 0.0
        blocked_since = 0.0
        if program.state is ProgramState.TOOL_CALL and program.admitted_t is not None:
            blocked_since = max(0.0, ctx.now - block.last_access_t)
        return self.cost_model.retention_value(
            block,
            state=program.state,
            ctx=ctx,
            blocked_since_s=blocked_since,
            will_return=program.will_return,
        )

    def select_victims(
        self,
        candidates: List[BlockMeta],
        programs: Mapping[str, AgentProgram],
        ctx: EvictionContext,
    ) -> List[EvictionDecision]:
        if not candidates:
            return []

        scored = [(self._score(b, programs, ctx), b) for b in candidates]

        if self.program_granular:
            order = self._order_program_granular(scored)
        else:
            order = self._order_block_granular(scored)

        decisions: List[EvictionDecision] = []
        freed = 0
        for score, block in order:
            if freed >= ctx.bytes_needed:
                break
            target = self._placement(block, programs, ctx)
            freed += self.config.model.bytes_for(block.num_tokens)
            decisions.append(EvictionDecision(block.block_id, target, score=score))
        return decisions

    def _placement(
        self,
        block: BlockMeta,
        programs: Mapping[str, AgentProgram],
        ctx: EvictionContext,
    ) -> Tier:
        """Decide where a victim goes -- including 'nowhere'.

        Dropping a low-value block is an active choice, not a failure: it
        keeps the copy engine free for the contexts that will actually be
        asked for. This is the main reason AgentCache moves far fewer bytes
        over PCIe than an offload-everything baseline while still serving
        more hits.
        """
        program = programs.get(block.program_id)
        state = program.state if program else ProgramState.FINISHED
        will_return = program.will_return if program else False
        blocked_since = max(0.0, ctx.now - block.last_access_t)

        worth_it = self.cost_model.should_offload(
            block,
            state=state,
            ctx=ctx,
            blocked_since_s=blocked_since,
            will_return=will_return,
        )
        if not worth_it:
            return Tier.GONE
        return self.cost_model.choose_target_tier(
            block, cpu_full=self._cpu_full_fn(), disk_full=self._disk_full_fn()
        )

    # ------------------------------------------------------------------
    def _seq_key(self, block: BlockMeta) -> int:
        """Sort key that puts the block we want to evict first.

        ``prefer_suffix`` means *keep* the suffix, so the prefix (low
        ``seq_index``) is offered up first.
        """
        return block.seq_index if self.prefer_suffix else -block.seq_index

    def _order_block_granular(self, scored):
        # Ascending value; within a program, evict the prefix first.
        return sorted(
            scored,
            key=lambda sb: (sb[0], self._seq_key(sb[1]), sb[1].block_id),
        )

    def _order_program_granular(self, scored):
        """Group by program, rank programs by their *mean* block value.

        Evicting half a program leaves a prefix that still cannot serve the
        next turn without a recompute of the missing tail, so we pay the
        eviction cost without collecting the space benefit. Ranking whole
        programs avoids that pathology; within a chosen program we still
        drop the prefix first so a partial hit remains possible.
        """
        by_program: dict[str, list] = {}
        for score, block in scored:
            by_program.setdefault(block.program_id, []).append((score, block))

        def program_key(item):
            pid, blocks = item
            mean_value = sum(s for s, _ in blocks) / len(blocks)
            return (mean_value, pid)

        order = []
        for _pid, blocks in sorted(by_program.items(), key=program_key):
            blocks.sort(key=lambda sb: self._seq_key(sb[1]))
            order.extend(blocks)
        return order
