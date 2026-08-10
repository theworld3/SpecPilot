"""Tests for the AgentCache retention cost model.

These encode the *claims* the README makes, so a regression in the model
shows up here before it shows up as a suspicious benchmark number.
"""
from __future__ import annotations

from agentcache.config import HardwareConfig, SystemConfig
from agentcache.policy.cost_model import (
    QueueDelayModel,
    RetentionCostModel,
    ReturnProbabilityModel,
)
from agentcache.types import ProgramState, Tier
from tests.conftest import make_block, make_ctx


# ----------------------------------------------------------------------
# ReturnProbabilityModel
# ----------------------------------------------------------------------
def test_return_prob_finished_is_zero():
    m = ReturnProbabilityModel()
    assert m.estimate(ProgramState.FINISHED) == 0.0


def test_return_prob_finished_via_will_return_flag():
    m = ReturnProbabilityModel()
    assert m.estimate(ProgramState.RUNNING, will_return=False) == 0.0


def test_return_prob_tool_call_high_when_fresh():
    m = ReturnProbabilityModel()
    assert m.estimate(ProgramState.TOOL_CALL, blocked_since_s=0.0) == 0.95


def test_return_prob_tool_call_decays_then_times_out():
    m = ReturnProbabilityModel()
    near = m.estimate(ProgramState.TOOL_CALL, blocked_since_s=12.0)
    assert 0.0 < near < 0.95
    assert m.estimate(ProgramState.TOOL_CALL, blocked_since_s=30.0) == 0.05


# ----------------------------------------------------------------------
# QueueDelayModel
# ----------------------------------------------------------------------
def test_queue_penalty_collapses_when_idle():
    q = QueueDelayModel()
    ctx = make_ctx(queue_len=0, running_batch_size=0, max_batch_size=8)
    assert q.estimate(ctx) == 0.0


def test_queue_penalty_grows_under_backlog():
    q = QueueDelayModel(avg_service_time_s=0.35)
    ctx = make_ctx(queue_len=10, running_batch_size=0, max_batch_size=8)
    # backlog = max(0, 10 - 8) = 2; rounds = 2/8 = 0.25 -> 0.25 * 0.35
    assert abs(q.estimate(ctx) - 0.0875) < 1e-9


# ----------------------------------------------------------------------
# RetentionCostModel
# ----------------------------------------------------------------------
def _model(enable_queue_term: bool = True) -> RetentionCostModel:
    return RetentionCostModel(
        config=SystemConfig(hardware=HardwareConfig(prefill_tokens_per_s=10000.0)),
        enable_queue_term=enable_queue_term,
    )


def test_retention_value_pinned_is_infinite():
    cm = _model()
    b = make_block(0, "p", 0, pinned=True)
    ctx = make_ctx()
    val = cm.retention_value(
        b, state=ProgramState.TOOL_CALL, ctx=ctx, will_return=True
    )
    assert val == float("inf")


def test_retention_value_zero_when_will_not_return():
    cm = _model()
    b = make_block(0, "p", 0)
    ctx = make_ctx()
    val = cm.retention_value(b, state=ProgramState.FINISHED, ctx=ctx, will_return=False)
    assert val == 0.0


def test_retention_value_higher_for_likely_return():
    cm = _model()
    ctx = make_ctx()
    # A block that will return is worth more than one that won't.
    returning = cm.retention_value(
        make_block(0, "p", 0), state=ProgramState.TOOL_CALL, ctx=ctx, will_return=True
    )
    gone = cm.retention_value(
        make_block(1, "p", 1), state=ProgramState.FINISHED, ctx=ctx, will_return=False
    )
    assert returning > gone


def test_queue_term_increases_value_under_contention():
    cm_off = _model(enable_queue_term=False)
    cm_on = _model(enable_queue_term=True)
    ctx = make_ctx(queue_len=10, running_batch_size=0, max_batch_size=8)
    b = make_block(0, "p", 0)
    v_off = cm_off.retention_value(
        b, state=ProgramState.TOOL_CALL, ctx=ctx, will_return=True
    )
    v_on = cm_on.retention_value(
        b, state=ProgramState.TOOL_CALL, ctx=ctx, will_return=True
    )
    assert v_on > v_off  # contention raises the value of keeping the block


def test_should_offload_true_when_link_idle():
    cm = _model()
    ctx = make_ctx(pcie_backlog_s=0.0)
    b = make_block(0, "p", 0)
    assert cm.should_offload(b, state=ProgramState.TOOL_CALL, ctx=ctx, will_return=True)


def test_should_offload_false_when_link_congested():
    cm = _model()
    # A small block with a deeply-backlogged link: write cost dominates.
    ctx = make_ctx(pcie_backlog_s=10.0)
    b = make_block(0, "p", 0, num_tokens=16)
    assert not cm.should_offload(
        b, state=ProgramState.TOOL_CALL, ctx=ctx, will_return=True
    )


def test_choose_target_tier_prefers_cpu():
    cm = _model()
    b = make_block(0, "p", 0)
    assert cm.choose_target_tier(b, cpu_full=False, disk_full=False) is Tier.CPU
    assert cm.choose_target_tier(b, cpu_full=True, disk_full=False) is Tier.GONE
