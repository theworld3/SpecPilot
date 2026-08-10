"""Tests for the eviction policies (LRU baselines + AgentCache + Oracle)."""
from __future__ import annotations

from agentcache.policy.agentcache import AgentCachePolicy
from agentcache.policy.lru import LRUOffloadPolicy, LRUPolicy
from agentcache.policy.oracle import OraclePolicy
from agentcache.types import ProgramState, Tier
from tests.conftest import make_block, make_ctx, make_program


# ----------------------------------------------------------------------
# LRU baselines
# ----------------------------------------------------------------------
def test_lru_evicts_oldest_first(config):
    lru = LRUPolicy(config)
    b_old = make_block(0, "p", 0, last_access_t=1.0)
    b_new = make_block(1, "p", 1, last_access_t=5.0)
    ctx = make_ctx(bytes_needed=config.block_bytes)
    decisions = lru.select_victims([b_old, b_new], {"p": make_program("p")}, ctx)
    assert [d.block_id for d in decisions] == [0]
    assert all(d.target_tier is Tier.GONE for d in decisions)


def test_lru_offload_demotes_to_cpu(config):
    pol = LRUOffloadPolicy(config)
    ctx = make_ctx(bytes_needed=config.block_bytes)
    b = make_block(0, "p", 0, last_access_t=1.0)
    decisions = pol.select_victims([b], {"p": make_program("p")}, ctx)
    assert decisions[0].target_tier is Tier.CPU


def test_lru_orders_by_recency(config):
    lru = LRUPolicy(config)
    old = make_block(0, "p", 0, last_access_t=1.0)
    new = make_block(1, "p", 1, last_access_t=5.0)
    ctx = make_ctx(bytes_needed=config.block_bytes)
    decisions = lru.select_victims([new, old], {"p": make_program("p")}, ctx)
    # oldest first
    assert [d.block_id for d in decisions] == [0]


# ----------------------------------------------------------------------
# AgentCache
# ----------------------------------------------------------------------
def test_agentcache_protects_tool_call_contexts(config):
    pol = AgentCachePolicy(config)
    finished = make_program("fin", state=ProgramState.FINISHED, turn_index=1)
    returning = make_program("ret", state=ProgramState.TOOL_CALL, turn_index=0)

    b_fin = make_block(0, "fin", 0, last_access_t=1.0)
    b_ret = make_block(1, "ret", 0, last_access_t=1.0)

    ctx = make_ctx(bytes_needed=config.block_bytes)
    programs = {"fin": finished, "ret": returning}
    decisions = pol.select_victims([b_fin, b_ret], programs, ctx)

    # The finished program is the cheapest to evict; the one parked on a tool
    # call is protected. So only b_fin is chosen.
    assert [d.block_id for d in decisions] == [0]
    # Finished context is dropped (no future reference); returning one is
    # offloaded to CPU only if the link is idle (it is here).
    assert decisions[0].target_tier is Tier.GONE


def test_agentcache_offloads_returning_block_when_idle(config):
    pol = AgentCachePolicy(config)
    returning = make_program("ret", state=ProgramState.TOOL_CALL, turn_index=0)
    b_ret = make_block(1, "ret", 0, last_access_t=1.0)

    ctx = make_ctx(bytes_needed=config.block_bytes)
    decisions = pol.select_victims([b_ret], {"ret": returning}, ctx)
    assert decisions[0].target_tier is Tier.CPU


def test_agentcache_drops_when_link_congested(config):
    pol = AgentCachePolicy(config)
    returning = make_program("ret", state=ProgramState.TOOL_CALL, turn_index=0)
    b_ret = make_block(1, "ret", 0, num_tokens=16)
    # Huge PCIe backlog: offload is not worth the write cost.
    ctx = make_ctx(bytes_needed=config.block_bytes, pcie_backlog_s=10.0)
    decisions = pol.select_victims([b_ret], {"ret": returning}, ctx)
    assert decisions[0].target_tier is Tier.GONE


def test_agentcache_ablation_queue_term_changes_value(config):
    on = AgentCachePolicy(config, enable_queue_term=True)
    off = AgentCachePolicy(config, enable_queue_term=False)
    returning = make_program("ret", state=ProgramState.TOOL_CALL, turn_index=0)
    b = make_block(0, "ret", 0)
    ctx = make_ctx(queue_len=10, running_batch_size=0, max_batch_size=8)
    v_on = on._score(b, {"ret": returning}, ctx)
    v_off = off._score(b, {"ret": returning}, ctx)
    assert v_on > v_off


# ----------------------------------------------------------------------
# Oracle
# ----------------------------------------------------------------------
def test_oracle_requires_future_knowledge(config):
    pol = OraclePolicy(config)
    # __init__ should not blow up; select_victims before bind must.
    import pytest

    with pytest.raises(RuntimeError):
        pol.select_victims([], {}, make_ctx())


def test_oracle_keeps_soonest_referenced_block(config):
    pol = OraclePolicy(config)
    a = make_block(0, "pa", 0, last_access_t=1.0)
    b = make_block(1, "pb", 0, last_access_t=1.0)

    def next_ref(block, now=10.0):
        # block a (id 0) is referenced again at t=20, block b (id 1) at t=200
        return 20.0 if block.block_id == 0 else 200.0

    pol.bind(next_ref_fn=next_ref)
    ctx = make_ctx(bytes_needed=config.block_bytes)
    decisions = pol.select_victims([a, b], {}, ctx)
    # The block referenced furthest in the future (b) is evicted first.
    assert decisions[0].block_id == 1
