"""The AgentCache retention cost model.

Why a cost model at all
-----------------------
Every production engine today evicts KV cache with LRU (optionally with a
CPU offload tier in front of the drop). LRU answers one question: *which
block was used least recently?* For agent workloads that is the wrong
question, because a program blocked on a tool call looks exactly like a dead
program -- it has not touched its blocks for seconds -- yet it is guaranteed
to come back with a near-identical prefix.

The right question is: **if I drop this block, what will it cost me later?**

Three cost terms
----------------
Dropping a block that gets re-referenced costs:

1. ``T_recompute`` -- prefill FLOPs to rebuild the KV.
2. ``-T_reload``   -- what we would have paid anyway if we demoted it to CPU
                      instead of dropping it. Only the *difference* matters.
3. ``T_queue``     -- **the term everyone misses.** A program whose cache was
                      dropped loses its slot and must re-enter the admission
                      queue. Under load the wait, not the recompute, dominates.
                      InferCept (ICML'24) models 1 and 2 but not 3; on a
                      20-turn trajectory the omitted term compounds into
                      seconds of added end-to-end latency, and it is invisible
                      to offline profiling because it only exists under
                      contention.

The retention value of a block is the expected saving per byte::

    value(b) = P_return(b) * (T_recompute + T_queue - T_reload) / bytes(b)

Blocks are evicted in ascending ``value``. Dividing by size makes this a
classic cost-benefit knapsack ordering (GreedyDual-Size), so a huge block
must justify its footprint rather than win on raw savings alone.

Estimating ``P_return``
-----------------------
This is where agent semantics beat generic heuristics. We do not need a
learned model: the engine *knows* whether a program is mid-trajectory.

    - blocked on a tool call, more turns remaining -> ~0.95
    - decaying with how long the tool has been running (long tool call =>
      more likely to be a timeout / abandoned session)
    - finished -> 0.0

The decay constant is the only tunable, and ``benchmarks/run_ablation.py``
sweeps it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..config import SystemConfig
from ..types import BlockMeta, EvictionContext, ProgramState, Tier

__all__ = ["QueueDelayModel", "ReturnProbabilityModel", "RetentionCostModel"]


@dataclass
class QueueDelayModel:
    """Estimates the scheduling penalty paid by a program that loses its KV.

    A dropped program must re-run prefill *and* re-acquire a batch slot. The
    slot wait is modelled as a simple M/D/c-flavoured approximation:

        T_queue = max(0, queue_len - free_slots) / max_batch * T_service

    where ``T_service`` is how long an average admitted program holds its
    slot. When the engine is under-utilised (free slots available) the term
    correctly collapses to zero -- which is why offline profiling on an idle
    box never sees it.
    """

    avg_service_time_s: float = 0.35
    """EWMA of observed slot-hold time, updated online by the scheduler."""
    ewma_alpha: float = 0.1

    def observe_service_time(self, duration_s: float) -> None:
        self.avg_service_time_s = (
            (1 - self.ewma_alpha) * self.avg_service_time_s
            + self.ewma_alpha * duration_s
        )

    def estimate(self, ctx: EvictionContext) -> float:
        free_slots = max(0, ctx.max_batch_size - ctx.running_batch_size)
        backlog = max(0, ctx.queue_len - free_slots)
        if backlog == 0:
            return 0.0
        # Each "round" of the batch drains at most max_batch_size programs.
        rounds = backlog / max(1, ctx.max_batch_size)
        return rounds * self.avg_service_time_s


@dataclass
class ReturnProbabilityModel:
    """P(this block is referenced again) from agent program semantics."""

    base_tool_call_p: float = 0.95
    """A program blocked on a tool call almost always comes back."""
    tool_timeout_s: float = 30.0
    """Beyond this, treat the program as likely abandoned."""
    decay_halflife_s: float = 12.0
    running_p: float = 0.99
    """Blocks of a running program are about to be attended to."""

    def estimate(
        self,
        state: ProgramState,
        *,
        blocked_since_s: float = 0.0,
        will_return: bool = True,
    ) -> float:
        if state is ProgramState.FINISHED or not will_return:
            return 0.0
        if state is ProgramState.RUNNING:
            return self.running_p
        if state is ProgramState.TOOL_CALL:
            if blocked_since_s >= self.tool_timeout_s:
                return 0.05
            decay = math.pow(0.5, blocked_since_s / self.decay_halflife_s)
            return self.base_tool_call_p * decay
        if state is ProgramState.WAITING:
            return 0.9
        return 0.5


@dataclass
class RetentionCostModel:
    """Scores blocks for eviction and picks a demotion target tier."""

    config: SystemConfig
    queue_model: QueueDelayModel = None  # type: ignore[assignment]
    return_model: ReturnProbabilityModel = None  # type: ignore[assignment]
    enable_queue_term: bool = True
    """Ablation switch: turning this off reduces the model to an
    InferCept-style recompute-vs-reload comparison."""
    congestion_scale_s: float = 0.5
    """PCIe backlog at which an offload is priced at 2x. Swept in the
    ablation; results are flat between 0.2 s and 1.0 s."""

    def __post_init__(self) -> None:
        if self.queue_model is None:
            self.queue_model = QueueDelayModel()
        if self.return_model is None:
            self.return_model = ReturnProbabilityModel()

    # ------------------------------------------------------------------
    # Cost terms
    # ------------------------------------------------------------------
    def recompute_time(self, num_tokens: int) -> float:
        """Prefill cost to rebuild ``num_tokens`` of KV.

        Note this is charged at the *sequence* level in reality (attention is
        quadratic), but blocks are re-materialised as part of a contiguous
        prefill, so a linear per-token rate calibrated at the working context
        length is the right first-order model. ``scripts/calibrate.py``
        fits the rate per context bucket.
        """
        return num_tokens / self.config.hardware.prefill_tokens_per_s

    def reload_time(self, num_bytes: int, from_tier: Tier) -> float:
        if from_tier is Tier.GPU:
            return 0.0
        if from_tier is Tier.GONE:
            return math.inf
        moved = self.config.offload_bytes(num_bytes) if from_tier is Tier.CPU else num_bytes
        hw = self.config.hardware
        return hw.transfer_launch_overhead_s + moved / hw.bandwidth_to_gpu(from_tier)

    def queue_penalty(self, ctx: EvictionContext) -> float:
        if not self.enable_queue_term:
            return 0.0
        return self.queue_model.estimate(ctx)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def retention_value(
        self,
        block: BlockMeta,
        *,
        state: ProgramState,
        ctx: EvictionContext,
        blocked_since_s: float = 0.0,
        will_return: bool = True,
        target_tier: Optional[Tier] = None,
    ) -> float:
        """Expected time saved per byte by keeping ``block`` on GPU.

        Lower value == better eviction candidate. Pinned blocks return
        ``inf`` so they sort to the end unconditionally.
        """
        if block.pinned:
            return math.inf

        p = self.return_model.estimate(
            state, blocked_since_s=blocked_since_s, will_return=will_return
        )
        if p <= 0.0:
            return 0.0

        num_bytes = self.config.model.bytes_for(block.num_tokens)
        target = target_tier or self._demotion_target(num_bytes)

        t_recompute = self.recompute_time(block.num_tokens)
        t_reload = self.reload_time(num_bytes, target)
        t_queue = self.queue_penalty(ctx)

        # If reload is somehow slower than recompute (tiny blocks on a slow
        # disk), the saving floors at zero -- never negative, dropping is then
        # simply free.
        saving = max(0.0, t_recompute + t_queue - t_reload)
        return p * saving / num_bytes

    def _demotion_target(self, num_bytes: int) -> Tier:
        """Cheapest tier that is still faster than recomputing."""
        cfg = self.config
        if cfg.enable_cpu_tier:
            return Tier.CPU
        if cfg.enable_disk_tier:
            return Tier.DISK
        return Tier.GONE

    def should_offload(
        self,
        block: BlockMeta,
        *,
        state: ProgramState,
        ctx: EvictionContext,
        blocked_since_s: float = 0.0,
        will_return: bool = True,
    ) -> bool:
        """Is this block worth spending PCIe bandwidth on?

        The question every offload-everything design skips. Writing a block
        to host memory costs link time *now* and delays reloads that are on
        somebody's critical path. So we compare:

            benefit = P_return * (T_recompute + T_queue - T_reload)
            cost    = T_write * (1 + backlog / congestion_scale)

        When the link is idle the cost term collapses and we offload freely.
        When it is congested the policy becomes selective automatically --
        no threshold tuning, no manual "offload rate limit" knob.
        """
        p = self.return_model.estimate(
            state, blocked_since_s=blocked_since_s, will_return=will_return
        )
        if p <= 0.0:
            return False

        num_bytes = self.config.model.bytes_for(block.num_tokens)
        moved = self.config.offload_bytes(num_bytes)
        hw = self.config.hardware

        t_write = hw.transfer_launch_overhead_s + moved / hw.d2h_bandwidth_bps
        benefit = p * max(
            0.0,
            self.recompute_time(block.num_tokens)
            + self.queue_penalty(ctx)
            - self.reload_time(num_bytes, Tier.CPU),
        )
        cost = t_write * (1.0 + ctx.pcie_backlog_s / self.congestion_scale_s)
        return benefit > cost

    def choose_target_tier(self, block: BlockMeta, *, cpu_full: bool, disk_full: bool) -> Tier:
        """Placement decision for a block that has been selected for eviction."""
        cfg = self.config
        num_bytes = cfg.model.bytes_for(block.num_tokens)
        t_recompute = self.recompute_time(block.num_tokens)

        if cfg.enable_cpu_tier and not cpu_full:
            return Tier.CPU
        if cfg.enable_disk_tier and not disk_full:
            # Only worth it if reading back genuinely beats recomputing.
            if self.reload_time(num_bytes, Tier.DISK) < t_recompute:
                return Tier.DISK
        return Tier.GONE
