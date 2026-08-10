"""Shared fixtures for the AgentCache test-suite.

Keeps policy / manager tests terse: they mostly need a config, a few blocks,
and an eviction context, all built the same way.
"""
from __future__ import annotations

import pytest

from agentcache.config import HardwareConfig, SystemConfig
from agentcache.types import (
    AgentProgram,
    BlockMeta,
    EvictionContext,
    ProgramState,
    Tier,
    Turn,
)


@pytest.fixture
def config() -> SystemConfig:
    """Small, fast config for unit tests (no disk tier)."""
    return SystemConfig(
        hardware=HardwareConfig(
            gpu_kv_capacity_bytes=2 * 1024 ** 3,
            cpu_kv_capacity_bytes=4 * 1024 ** 3,
            prefill_tokens_per_s=10000.0,
        ),
        block_size_tokens=16,
        max_batch_size=8,
        enable_cpu_tier=True,
        enable_disk_tier=False,
        fp8_offload=True,
    )


def make_block(
    block_id: int,
    program_id: str,
    seq_index: int,
    num_tokens: int = 16,
    *,
    tier: Tier = Tier.GPU,
    last_access_t: float = 0.0,
    pinned: bool = False,
) -> BlockMeta:
    return BlockMeta(
        block_id=block_id,
        program_id=program_id,
        seq_index=seq_index,
        num_tokens=num_tokens,
        tier=tier,
        last_access_t=last_access_t,
        pinned=pinned,
    )


def make_program(
    program_id: str,
    *,
    state: ProgramState = ProgramState.TOOL_CALL,
    turn_index: int = 0,
    num_turns: int = 2,
    blocked_for: float = 0.0,
    arrival_t: float = 0.0,
) -> AgentProgram:
    """A 2-turn program parked on a tool call by default (will_return True)."""
    turns = [
        Turn(new_prompt_tokens=100, output_tokens=50, tool_latency_s=1.0)
        for _ in range(num_turns)
    ]
    prog = AgentProgram(program_id=program_id, arrival_t=arrival_t, turns=turns)
    prog.state = state
    prog.turn_index = turn_index
    if blocked_for and state is ProgramState.TOOL_CALL:
        # admitted_t in the past => blocked_since = now - admitted_t
        prog.admitted_t = -blocked_for
    return prog


def make_ctx(
    *,
    now: float = 10.0,
    bytes_needed: int = 16 * 1024,
    queue_len: int = 0,
    running_batch_size: int = 0,
    max_batch_size: int = 8,
    gpu_bytes_used: int = 0,
    gpu_bytes_capacity: int = 2 * 1024 ** 3,
    pcie_backlog_s: float = 0.0,
) -> EvictionContext:
    return EvictionContext(
        now=now,
        bytes_needed=bytes_needed,
        queue_len=queue_len,
        running_batch_size=running_batch_size,
        max_batch_size=max_batch_size,
        gpu_bytes_used=gpu_bytes_used,
        gpu_bytes_capacity=gpu_bytes_capacity,
        pcie_backlog_s=pcie_backlog_s,
    )
