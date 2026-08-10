"""Discrete-event simulator for multi-turn agent serving.

Why a simulator
---------------
Three reasons, in order of importance:

1. **It answers the go/no-go question in minutes, not weeks.** Running the
   oracle policy against ``lru-offload`` tells you the headroom that exists
   before you write a single CUDA kernel.
2. **It isolates the policy.** On a real engine, a change in eviction order
   perturbs batching, which perturbs kernel occupancy, which perturbs
   everything. Here the decision rule is the only variable.
3. **It runs anywhere.** No GPU, no PyTorch, no model weights -- so the
   results in this repo are reproducible by anyone in ~10 seconds.

What it deliberately does *not* model
-------------------------------------
* Iteration-level continuous batching: a program holds a slot for a whole
  turn. This overstates absolute latency at high batch sizes but affects all
  policies identically, so the *relative* comparison holds.
* Attention quadratic cost: prefill is charged at a calibrated linear
  tokens/s rate.
* Cross-program prefix sharing (shared system prompts).

Every one of these makes the simulator *conservative* with respect to the
AgentCache claim, except the last, which is neutral. Real-hardware numbers
from ``benchmarks/run_serving_bench.py`` are the ground truth; the simulator
is a design tool, and ``docs/validation.md`` tracks the gap between them.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Sequence

from ..config import SystemConfig
from ..policy.base import EvictionPolicy
from ..policy.oracle import OraclePolicy
from ..tiering.manager import TieredCacheManager
from ..types import (
    AgentProgram,
    BlockMeta,
    EvictionContext,
    ProgramState,
    Tier,
)
from ..utils.metrics import summarize

__all__ = ["SimulationResult", "Simulator"]


class _Event(IntEnum):
    ARRIVAL = 0
    TURN_COMPLETE = 1
    TOOL_DONE = 2


@dataclass
class SimulationResult:
    policy: str
    num_programs: int
    num_turns: int
    makespan_s: float
    ttft: Dict[str, float]
    program_latency: Dict[str, float]
    queue_wait: Dict[str, float]
    output_tokens: int
    recomputed_tokens: int
    reloaded_tokens: int
    gpu_hit_tokens: int
    bytes_to_cpu: int
    bytes_from_cpu: int
    dropped_blocks: int
    demoted_blocks: int
    # TTFT decomposition -- means, in seconds. Reporting only the total
    # invites the reader to assume the win came from the cache when it
    # actually came from queueing (or vice versa).
    mean_queue_wait_s: float = 0.0
    mean_reload_s: float = 0.0
    mean_prefill_s: float = 0.0

    @property
    def throughput_tok_s(self) -> float:
        return self.output_tokens / self.makespan_s if self.makespan_s else 0.0

    @property
    def gpu_hit_rate(self) -> float:
        total = self.gpu_hit_tokens + self.reloaded_tokens + self.recomputed_tokens
        return self.gpu_hit_tokens / total if total else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """GPU hits + successful reloads: anything that avoided a prefill."""
        total = self.gpu_hit_tokens + self.reloaded_tokens + self.recomputed_tokens
        return (self.gpu_hit_tokens + self.reloaded_tokens) / total if total else 0.0

    def as_row(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "TTFT p50 (ms)": self.ttft["p50"] * 1000,
            "TTFT p95 (ms)": self.ttft["p95"] * 1000,
            "e2e p50 (s)": self.program_latency["p50"],
            "e2e p95 (s)": self.program_latency["p95"],
            "throughput (tok/s)": self.throughput_tok_s,
            "cache hit": self.cache_hit_rate,
            "recompute tok": self.recomputed_tokens,
            "H2D (GiB)": self.bytes_from_cpu / (1024 ** 3),
        }


class Simulator:
    """Event-driven serving simulation over a set of agent programs."""

    def __init__(
        self,
        config: SystemConfig,
        policy: EvictionPolicy,
        programs: Sequence[AgentProgram],
        *,
        decode_contention: float = 0.35,
    ) -> None:
        self.config = config
        self.policy = policy
        self.programs: Dict[str, AgentProgram] = {p.program_id: p for p in programs}
        self.cache = TieredCacheManager(config, policy)
        self.decode_contention = decode_contention
        """How much a full batch slows per-token decode. 0 = perfect scaling."""

        self._now = 0.0
        self._seq = 0
        self._events: List[tuple] = []
        self._waiting: List[tuple[float, str]] = []
        self._running: set[str] = set()
        self._ready_t: Dict[str, float] = {}
        self._tool_done_t: Dict[str, float] = {}
        self._ttft: List[float] = []
        self._queue_wait: List[float] = []
        self._reload_times: List[float] = []
        self._prefill_times: List[float] = []
        self._output_tokens = 0
        self._copy_free_t = 0.0
        """When the copy engine finishes its current backlog.

        Modelled as a *single* shared channel rather than full-duplex PCIe.
        That matches how vLLM's KV connector actually works today -- one
        CUDA copy stream handling both directions -- and it is the
        conservative choice: it prices speculative offload honestly instead
        of pretending writes are free."""
        self._stat_offload_wait_s = 0.0

        self._wire_policy()

    # ------------------------------------------------------------------
    def _wire_policy(self) -> None:
        """Inject live capacity callbacks so policies see real tier pressure."""
        if hasattr(self.policy, "_cpu_full_fn"):
            self.policy._cpu_full_fn = self.cache.cpu_full
        if hasattr(self.policy, "_disk_full_fn"):
            self.policy._disk_full_fn = self.cache.disk_full
        if isinstance(self.policy, OraclePolicy):
            self.policy.bind(self._next_reference_t)

    def _next_reference_t(self, block: BlockMeta, now: float) -> float:
        """Future knowledge handed to the oracle only.

        A ``TOOL_CALL`` program's next reference is exactly when its tool
        returns -- a time the trace knows and no real policy does.
        """
        program = self.programs.get(block.program_id)
        if program is None or program.state is ProgramState.FINISHED:
            return math.inf
        if program.state is ProgramState.TOOL_CALL:
            return self._tool_done_t.get(program.program_id, math.inf)
        return now  # waiting or running: needed immediately

    # ------------------------------------------------------------------
    def _push(self, t: float, kind: _Event, program_id: str) -> None:
        self._seq += 1
        heapq.heappush(self._events, (t, self._seq, int(kind), program_id))

    def _ctx_factory(self, bytes_needed: int) -> EvictionContext:
        return EvictionContext(
            now=self._now,
            bytes_needed=bytes_needed,
            queue_len=len(self._waiting),
            running_batch_size=len(self._running),
            max_batch_size=self.config.max_batch_size,
            gpu_bytes_used=self.cache.gpu_bytes_used,
            gpu_bytes_capacity=self.config.hardware.gpu_kv_capacity_bytes,
            pcie_backlog_s=max(0.0, self._copy_free_t - self._now),
        )

    # ------------------------------------------------------------------
    # Copy engine
    # ------------------------------------------------------------------
    def _enqueue_copy(self, num_bytes: int, bandwidth_bps: float) -> float:
        """Book the copy engine. Returns completion time.

        Transfers queue behind one another, so a burst of speculative
        offloads directly delays the next reload -- which is precisely the
        coupling that makes 'just offload everything' a worse strategy than
        it looks on paper.
        """
        hw = self.config.hardware
        start = max(self._now, self._copy_free_t)
        duration = hw.transfer_launch_overhead_s + num_bytes / bandwidth_bps
        self._copy_free_t = start + duration
        return self._copy_free_t

    def _charge_offloads(self, decisions) -> None:
        """Demotions are asynchronous but still consume link time."""
        for decision in decisions:
            if decision.target_tier is Tier.GONE:
                continue
            block = self.cache.blocks.get(decision.block_id)
            if block is None:
                continue
            moved = self.cache.stored_bytes(block, decision.target_tier)
            self._enqueue_copy(moved, self.config.hardware.d2h_bandwidth_bps)

    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        for program in self.programs.values():
            self._push(program.arrival_t, _Event.ARRIVAL, program.program_id)

        while self._events:
            t, _seq, kind, pid = heapq.heappop(self._events)
            self._now = max(self._now, t)
            event = _Event(kind)

            if event is _Event.ARRIVAL:
                self._enqueue(pid, self._now)
            elif event is _Event.TOOL_DONE:
                self.programs[pid].state = ProgramState.WAITING
                self._enqueue(pid, self._now)
            elif event is _Event.TURN_COMPLETE:
                self._on_turn_complete(pid)

            self._try_admit()

        return self._collect()

    # ------------------------------------------------------------------
    def _enqueue(self, program_id: str, now: float) -> None:
        self._ready_t[program_id] = now
        heapq.heappush(self._waiting, (now, program_id))
        self.programs[program_id].state = ProgramState.WAITING

    def _try_admit(self) -> None:
        """FCFS admission.

        Head-of-line blocking is intentional: it is what vLLM's default
        scheduler does, and skipping ahead would quietly hide the memory
        pressure that this project is about. ``scheduler/program_scheduler.py``
        explores the alternative.
        """
        while self._waiting and len(self._running) < self.config.max_batch_size:
            _ready, pid = self._waiting[0]
            if self._admit(pid):
                heapq.heappop(self._waiting)
                continue
            if not self._running:
                raise RuntimeError(
                    f"deadlock: {pid} cannot be admitted with an empty running set. "
                    "This means eviction could not free enough space even with the "
                    "whole pool available -- check gpu_kv_capacity_bytes."
                )
            break

    def _admit(self, program_id: str) -> bool:
        """Try to start the next turn of ``program_id``.

        Returns ``False`` if the GPU cannot hold this turn's working set right
        now, leaving the program queued. That back-pressure *is* the queueing
        delay the cost model tries to price, so it must be modelled rather
        than assumed away.
        """
        program = self.programs[program_id]
        turn = program.current_turn
        now = self._now

        context_before = program.prompt_tokens_so_far
        new_tokens = turn.new_prompt_tokens

        # 1. What survived since the last turn? Discard anything past the
        #    first hole -- prefill cannot skip it.
        lookup = self.cache.lookup(program_id, context_before, now)
        self.cache.truncate(program_id, lookup.covered_blocks)

        prefill_tokens = lookup.missing_tokens + new_tokens
        reload_raw_bytes = sum(
            self.config.model.bytes_for(b.num_tokens) for b in lookup.reload_blocks
        )
        needed_bytes = reload_raw_bytes + self.config.model.bytes_for(prefill_tokens)

        capacity = self.config.hardware.gpu_kv_capacity_bytes
        working_set = needed_bytes + self.config.model.bytes_for(
            sum(b.num_tokens for b in lookup.gpu_blocks)
        )
        if working_set > capacity:
            raise RuntimeError(
                f"program {program_id} needs {working_set / 2 ** 30:.2f} GiB of KV but "
                f"the pool is only {capacity / 2 ** 30:.2f} GiB. Shorten the trace or "
                f"raise hardware.gpu_kv_capacity_bytes."
            )

        # 2. Secure GPU space for both the reload and the prefill.
        protected = self._running | {program_id}
        if needed_bytes > self.cache.gpu_bytes_free:
            ctx = self._ctx_factory(needed_bytes - self.cache.gpu_bytes_free)
            ok, decisions = self.cache.ensure_space(
                needed_bytes, ctx, self.programs, protected
            )
            self._charge_offloads(decisions)
            if not ok:
                return False  # stay queued; a finishing turn will free space

        # 3. Reload whatever is on the host tier. Reloads queue behind any
        #    offloads already booked on the copy engine, so a policy that
        #    over-offloads pays for it right here.
        reload_time = 0.0
        if lookup.reload_blocks:
            done_t = now
            for block in lookup.reload_blocks:
                nbytes = self.cache.stored_bytes(block, block.tier)
                done_t = self._enqueue_copy(
                    nbytes, self.config.hardware.bandwidth_to_gpu(block.tier)
                )
            reload_time = max(0.0, done_t - now)
            self.cache.promote(lookup.reload_blocks, now)
            program.reloaded_bytes += lookup.reload_bytes

        # 4. Materialise the missing prefix plus the new tokens.
        program.recomputed_tokens += lookup.missing_tokens
        if prefill_tokens > 0:
            self.cache.allocate(program, prefill_tokens, now)

        # 5. Timing.
        prefill_time = prefill_tokens / self.config.hardware.prefill_tokens_per_s
        queue_wait = now - self._ready_t.get(program_id, now)
        ttft = queue_wait + reload_time + prefill_time

        contention = 1.0 + self.decode_contention * (
            len(self._running) / max(1, self.config.max_batch_size)
        )
        decode_time = turn.output_tokens * self.config.hardware.decode_step_s * contention

        self._ttft.append(ttft)
        self._queue_wait.append(queue_wait)
        self._reload_times.append(reload_time)
        self._prefill_times.append(prefill_time)
        program.ttft_per_turn.append(ttft)
        if program.admitted_t is None:
            program.admitted_t = now

        program.state = ProgramState.RUNNING
        # Recomputed tokens rebuild existing context -- they do not extend it.
        # (Conflating the two makes the context grow geometrically and the
        # simulation blow up; a good reason to assert on it in tests.)
        program.prompt_tokens_so_far = context_before + new_tokens
        self._output_tokens += turn.output_tokens
        self._running.add(program_id)

        service_time = reload_time + prefill_time + decode_time
        self._push(now + service_time, _Event.TURN_COMPLETE, program_id)
        return True

    def _on_turn_complete(self, program_id: str) -> None:
        program = self.programs[program_id]
        turn = program.current_turn
        now = self._now

        self._running.discard(program_id)
        if program.admitted_t is not None:
            self.policy.on_service_complete(
                max(1e-6, now - self._ready_t.get(program_id, now))
            )

        # Generated tokens also occupy KV and carry into the next turn.
        # Space for them was already accounted for during admission in a real
        # engine (vLLM reserves incrementally); here we make room on demand
        # and allow eviction of *other* programs if the batch grew.
        if turn.output_tokens > 0:
            needed = self.config.model.bytes_for(turn.output_tokens)
            if needed > self.cache.gpu_bytes_free:
                ctx = self._ctx_factory(needed - self.cache.gpu_bytes_free)
                _ok, decisions = self.cache.ensure_space(
                    needed, ctx, self.programs, self._running | {program_id}
                )
                self._charge_offloads(decisions)
            self.cache.allocate(program, turn.output_tokens, now)
            program.prompt_tokens_so_far += turn.output_tokens

        if program.has_next_turn:
            program.state = ProgramState.TOOL_CALL
            program.turn_index += 1
            done_t = now + turn.tool_latency_s
            self._tool_done_t[program_id] = done_t
            self._push(done_t, _Event.TOOL_DONE, program_id)
        else:
            program.state = ProgramState.FINISHED
            program.finished_t = now
            # Deliberately do NOT free the blocks. A real engine keeps
            # finished sequences in the prefix cache in case a later request
            # shares the prefix, and lets the eviction policy reclaim them.
            # That is what makes dead context a policy problem: LRU will
            # happily spend PCIe bandwidth offloading a session that ended,
            # while AgentCache scores it at zero and drops it immediately.

    # ------------------------------------------------------------------
    def _collect(self) -> SimulationResult:
        latencies = [
            p.finished_t - p.arrival_t
            for p in self.programs.values()
            if p.finished_t is not None
        ]
        makespan = max(
            (p.finished_t for p in self.programs.values() if p.finished_t), default=0.0
        )
        return SimulationResult(
            policy=self.policy.name,
            num_programs=len(self.programs),
            num_turns=sum(p.num_turns for p in self.programs.values()),
            makespan_s=makespan,
            ttft=summarize(self._ttft),
            program_latency=summarize(latencies),
            queue_wait=summarize(self._queue_wait),
            output_tokens=self._output_tokens,
            recomputed_tokens=self.cache.stat_recompute_tokens,
            reloaded_tokens=self.cache.stat_reload_tokens,
            gpu_hit_tokens=self.cache.stat_gpu_hit_tokens,
            bytes_to_cpu=self.cache.stat_bytes_to_cpu,
            bytes_from_cpu=self.cache.stat_bytes_from_cpu,
            dropped_blocks=self.cache.stat_dropped_blocks,
            demoted_blocks=self.cache.stat_demoted_blocks,
            mean_queue_wait_s=_mean(self._queue_wait),
            mean_reload_s=_mean(self._reload_times),
            mean_prefill_s=_mean(self._prefill_times),
        )


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
