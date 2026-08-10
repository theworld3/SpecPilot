"""Tests for the shared ProgramScheduler admission engine."""
from __future__ import annotations

from agentcache.config import HardwareConfig, SystemConfig
from agentcache.policy.agentcache import AgentCachePolicy
from agentcache.scheduler.program_scheduler import ProgramScheduler
from agentcache.types import ProgramState, Turn


def _cfg(gpu_gib=8, cpu_gib=8):
    return SystemConfig(
        hardware=HardwareConfig(
            gpu_kv_capacity_bytes=gpu_gib * 1024 ** 3,
            cpu_kv_capacity_bytes=cpu_gib * 1024 ** 3,
        ),
        max_batch_size=8,
    )


def test_scheduler_admits_first_turn_with_prefill():
    sched = ProgramScheduler(_cfg(), AgentCachePolicy(_cfg()))
    prog = __import__("agentcache.types", fromlist=["AgentProgram"]).AgentProgram(
        "p1", arrival_t=0.0, turns=[Turn(2000, 50, 1.0), Turn(500, 50, 0.0)]
    )
    sched.register(prog)
    plan = sched.plan_admission(prog, 2000, now=0.0, queue_len=0, running_batch_size=0)
    # Nothing resident yet -> the whole prompt is *new* prefill, not a
    # recompute of dropped KV (missing == 0), and it gets allocated.
    assert plan.missing_tokens == 0
    assert plan.reload_blocks == []  # nothing to reload from host
    assert plan.new_tokens == 2000
    assert len(sched.manager.program_blocks("p1")) > 0
    assert plan.ttft_s > 0
    # After the turn, the program parks on its tool call and will return.
    sched.complete_turn(prog, plan.ttft_s, now=1.0)
    assert prog.state is ProgramState.TOOL_CALL
    assert prog.will_return is True


def test_scheduler_returns_context_on_subsequent_turn():
    sched = ProgramScheduler(_cfg(), AgentCachePolicy(_cfg()))
    prog = __import__("agentcache.types", fromlist=["AgentProgram"]).AgentProgram(
        "p1", arrival_t=0.0, turns=[Turn(2000, 50, 1.0), Turn(500, 50, 0.0)]
    )
    sched.register(prog)
    p1 = sched.plan_admission(prog, 2000, now=0.0, queue_len=0, running_batch_size=0)
    sched.complete_turn(prog, p1.ttft_s, now=1.0)
    # Second turn: the 2000-token prefix is still on GPU, so it is a hit.
    p2 = sched.plan_admission(prog, 500, now=2.0, queue_len=0, running_batch_size=0)
    assert p2.missing_tokens == 0
    assert len(sched.manager.program_blocks("p1")) > 0


def test_scheduler_predicts_queue_penalty_under_contention():
    cfg = _cfg()
    sched = ProgramScheduler(cfg, AgentCachePolicy(cfg))
    prog = __import__("agentcache.types", fromlist=["AgentProgram"]).AgentProgram(
        "p1", arrival_t=0.0, turns=[Turn(2000, 50)]
    )
    sched.register(prog)
    idle = sched.plan_admission(prog, 2000, now=0.0, queue_len=0, running_batch_size=0)
    busy = sched.plan_admission(prog, 2000, now=0.0, queue_len=12, running_batch_size=8)
    # Under contention the queue term adds wait time to the plan.
    assert busy.queue_wait_s >= idle.queue_wait_s
