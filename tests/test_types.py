"""Tests for core data types and the will_return semantics."""
from __future__ import annotations

from agentcache.config import SystemConfig
from agentcache.types import AgentProgram, BlockMeta, ModelSpec, ProgramState, Tier, Turn


def test_model_bytes_per_token_default_8b():
    # 32 layers * 8 kv heads * 128 head_dim * 2 bytes * 2 (K and V)
    m = ModelSpec()
    assert m.bytes_per_token == 32 * 8 * 128 * 2 * 2
    assert m.bytes_for(1) == m.bytes_per_token
    assert m.bytes_for(16) == 16 * m.bytes_per_token


def test_tier_ordering_and_resident():
    assert Tier.GPU < Tier.CPU < Tier.DISK < Tier.GONE
    assert Tier.GPU.is_resident and Tier.CPU.is_resident
    assert not Tier.GONE.is_resident


def test_will_return_finished_is_false():
    prog = AgentProgram("p", arrival_t=0.0, turns=[Turn(10, 5), Turn(10, 5)])
    prog.state = ProgramState.FINISHED
    assert prog.will_return is False


def test_will_return_true_for_tool_call_on_last_turn():
    # A program parked on the tool call for its FINAL turn still returns:
    # turn_index has advanced, but state is TOOL_CALL, not FINISHED.
    prog = AgentProgram("p", arrival_t=0.0, turns=[Turn(10, 5), Turn(10, 5)])
    prog.state = ProgramState.TOOL_CALL
    prog.turn_index = 1  # about to run the last turn
    assert prog.will_return is True
    # But has_next_turn would be False here -- this is exactly the trap.
    assert prog.has_next_turn is False


def test_block_touch_updates_access_time():
    b = BlockMeta(0, "p", 0, 16)
    b.touch(3.5)
    assert b.last_access_t == 3.5


def test_default_config_is_sane():
    cfg = SystemConfig()
    assert cfg.hardware.gpu_kv_capacity_bytes > 0
    assert cfg.block_bytes == cfg.model.bytes_for(cfg.block_size_tokens)
