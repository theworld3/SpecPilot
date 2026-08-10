"""The real-time admission/eviction engine shared by the simulator and the
serving connector.

The discrete-event :class:`~agentcache.sim.simulator.Simulator` owns the
*clock*; this class owns the *decisions*. Both drive the same
:class:`~agentcache.tiering.manager.TieredCacheManager` and the same policy,
so a decision rule validated here is a decision rule validated on hardware.

Concretely, a serving engine (or the simulator) calls, per turn:

    plan = scheduler.plan_admission(program, new_tokens, now, queue_len, running)
    scheduler.apply_transfers(plan)          # optional: move bytes via backend
    ... serve the prefill ...
    scheduler.complete_turn(program, now)    # advance state, feed the queue model

The class is pure stdlib + the manager, so it is unit-testable with a
``NullBackend`` and needs no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config import SystemConfig
from ..policy.base import EvictionPolicy
from ..policy.cost_model import QueueDelayModel
from ..tiering.backend import NullBackend, StorageBackend
from ..tiering.manager import LookupResult, TieredCacheManager
from ..types import AgentProgram, BlockMeta, EvictionContext, EvictionDecision, ProgramState, Tier


@dataclass
class AdmissionPlan:
    """What the scheduler decided for one turn's admission."""

    program_id: str
    context_before: int
    new_tokens: int
    missing_tokens: int
    reload_blocks: List[BlockMeta] = field(default_factory=list)
    reload_bytes: int = 0
    evict_decisions: List[EvictionDecision] = field(default_factory=list)
    # Best-effort TTFT decomposition (seconds). The real engine replaces
    # reload_s with a measured transfer time; queue_s with the observed wait.
    queue_wait_s: float = 0.0
    reload_s: float = 0.0
    prefill_s: float = 0.0

    @property
    def ttft_s(self) -> float:
        return self.queue_wait_s + self.reload_s + self.prefill_s


class ProgramScheduler:
    """Per-request KV admission controller."""

    def __init__(
        self,
        config: SystemConfig,
        policy: EvictionPolicy,
        backend: Optional[StorageBackend] = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.manager = TieredCacheManager(config, policy)
        self.backend = backend or NullBackend()
        self.queue_model = QueueDelayModel()
        self._programs: dict[str, AgentProgram] = {}

    # ------------------------------------------------------------------
    def register(self, program: AgentProgram) -> None:
        """Tell the scheduler a program exists (called on first arrival)."""
        self._programs[program.program_id] = program
        program.state = ProgramState.WAITING

    def _ctx(self, bytes_needed: int, queue_len: int, running_batch: int, now: float) -> EvictionContext:
        return EvictionContext(
            now=now,
            bytes_needed=bytes_needed,
            queue_len=queue_len,
            running_batch_size=running_batch,
            max_batch_size=self.config.max_batch_size,
            gpu_bytes_used=self.manager.gpu_bytes_used,
            gpu_bytes_capacity=self.config.hardware.gpu_kv_capacity_bytes,
            pcie_backlog_s=self.backend.pcie_backlog_s,
        )

    # ------------------------------------------------------------------
    def plan_admission(
        self,
        program: AgentProgram,
        new_tokens: int,
        now: float,
        queue_len: int,
        running_batch_size: int,
    ) -> AdmissionPlan:
        """Decide evictions / reloads / prefill for the next turn.

        Applies eviction metadata immediately (so the GPU footprint is freed),
        but does not move bytes -- call :meth:`apply_transfers` for that.
        """
        pid = program.program_id
        context_before = program.prompt_tokens_so_far

        lookup: LookupResult = self.manager.lookup(pid, context_before, now)
        self.manager.truncate(pid, lookup.covered_blocks)

        prefill_tokens = lookup.missing_tokens + new_tokens
        reload_raw = sum(self.config.model.bytes_for(b.num_tokens) for b in lookup.reload_blocks)
        needed_bytes = reload_raw + self.config.model.bytes_for(prefill_tokens)

        capacity = self.config.hardware.gpu_kv_capacity_bytes
        working_set = needed_bytes + self.config.model.bytes_for(
            sum(b.num_tokens for b in lookup.gpu_blocks)
        )
        if working_set > capacity:
            raise RuntimeError(
                f"program {pid} needs {working_set / 2 ** 30:.2f} GiB of KV but the "
                f"pool is only {capacity / 2 ** 30:.2f} GiB"
            )

        protected = {pid}
        evict_decisions: List[EvictionDecision] = []
        if needed_bytes > self.manager.gpu_bytes_free:
            ctx = self._ctx(needed_bytes - self.manager.gpu_bytes_free, queue_len, running_batch_size, now)
            ok, decisions = self.manager.ensure_space(needed_bytes, ctx, self._programs, protected)
            evict_decisions = decisions
            if not ok:
                raise RuntimeError(
                    f"program {pid} cannot be admitted: not enough GPU KV even after eviction"
                )

        # Reload estimate: the real backend measures the actual transfer time;
        # here we estimate from the configured PCIe bandwidth so the TTFT
        # decomposition is meaningful even under the NullBackend.
        reload_bytes = lookup.reload_bytes
        reload_s = 0.0
        if reload_bytes:
            moved = self.config.offload_bytes(reload_bytes)
            reload_s = (
                self.config.hardware.transfer_launch_overhead_s
                + moved / self.config.hardware.bandwidth_to_gpu(Tier.CPU)
            )
        self.manager.promote(lookup.reload_blocks, now)

        # Materialise the missing prefix + new tokens.
        program.recomputed_tokens += lookup.missing_tokens
        if prefill_tokens > 0:
            self.manager.allocate(program, prefill_tokens, now)

        queue_wait_s = self.queue_model.estimate(
            self._ctx(needed_bytes, queue_len, running_batch_size, now)
        )
        prefill_s = prefill_tokens / self.config.hardware.prefill_tokens_per_s

        program.prompt_tokens_so_far = context_before + new_tokens
        program.state = ProgramState.RUNNING

        return AdmissionPlan(
            program_id=pid,
            context_before=context_before,
            new_tokens=new_tokens,
            missing_tokens=lookup.missing_tokens,
            reload_blocks=list(lookup.reload_blocks),
            reload_bytes=reload_bytes,
            evict_decisions=evict_decisions,
            queue_wait_s=queue_wait_s,
            reload_s=reload_s,
            prefill_s=prefill_s,
        )

    def apply_transfers(self, plan: AdmissionPlan) -> None:
        """Move bytes for reloads (a real backend performs the H2D copy)."""
        if plan.reload_bytes:
            self.backend.transfer(plan.reload_blocks, src=Tier.CPU, dst=Tier.GPU)

    def complete_turn(self, program: AgentProgram, service_time_s: float, now: float) -> None:
        """Advance program state after a turn finishes serving.

        Feeds the observed slot-hold time to the queue model so the
        queueing term stays calibrated to the real engine.
        """
        self.queue_model.observe_service_time(service_time_s)
        program.turn_index += 1
        if program.turn_index >= program.num_turns:
            program.state = ProgramState.FINISHED
        else:
            program.state = ProgramState.TOOL_CALL
