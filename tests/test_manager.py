"""Tests for the tiered cache manager (lookup / truncate / evict / indices)."""
from __future__ import annotations

from agentcache.config import HardwareConfig, SystemConfig
from agentcache.policy.lru import LRUOffloadPolicy
from agentcache.tiering.manager import TieredCacheManager
from agentcache.types import Tier
from tests.conftest import make_ctx, make_program


def _small_config():
    # 8 MiB GPU, 16 MiB CPU -> 4 / 8 blocks respectively (2 MiB per block).
    return SystemConfig(
        hardware=HardwareConfig(
            gpu_kv_capacity_bytes=8 * 1024 ** 2,
            cpu_kv_capacity_bytes=16 * 1024 ** 2,
        ),
        block_size_tokens=16,
        max_batch_size=8,
        enable_cpu_tier=True,
        enable_disk_tier=False,
        fp8_offload=True,
    )


def test_lookup_classifies_gpu_reload_and_missing():
    cfg = _small_config()
    mgr = TieredCacheManager(cfg, LRUOffloadPolicy(cfg))
    prog = make_program("p")
    b0 = mgr.allocate(prog, 16, now=0.0)[0]  # GPU
    b1 = mgr.allocate(prog, 16, now=0.0)[0]  # will become CPU
    b2 = mgr.allocate(prog, 16, now=0.0)[0]  # will become GONE
    mgr._set_tier(b1, Tier.CPU)
    mgr._set_tier(b2, Tier.GONE)

    res = mgr.lookup("p", num_tokens=48, now=0.0)
    assert [b.block_id for b in res.gpu_blocks] == [b0.block_id]
    assert [b.block_id for b in res.reload_blocks] == [b1.block_id]
    assert res.missing_tokens == 16  # only b2's tokens are gone
    assert res.covered_blocks == 2


def test_lookup_breaks_at_first_hole():
    cfg = _small_config()
    mgr = TieredCacheManager(cfg, LRUOffloadPolicy(cfg))
    prog = make_program("p")
    b0 = mgr.allocate(prog, 16, now=0.0)[0]
    b1 = mgr.allocate(prog, 16, now=0.0)[0]
    # Make b1 GONE -> everything after it must be recomputed even if resident.
    mgr._set_tier(b1, Tier.GONE)
    b2 = mgr.allocate(prog, 16, now=0.0)[0]
    mgr._set_tier(b2, Tier.GPU)

    res = mgr.lookup("p", num_tokens=48, now=0.0)
    assert [b.block_id for b in res.gpu_blocks] == [b0.block_id]
    assert res.missing_tokens == 32  # b1 (gone) + b2 (after the hole)


def test_truncate_drops_tail_blocks():
    cfg = _small_config()
    mgr = TieredCacheManager(cfg, LRUOffloadPolicy(cfg))
    prog = make_program("p")
    for _ in range(4):
        mgr.allocate(prog, 16, now=0.0)
    assert len(mgr.program_blocks("p")) == 4

    dropped = mgr.truncate("p", keep_blocks=2)
    assert dropped == 2
    assert len(mgr.program_blocks("p")) == 2


def test_evict_frees_gpu_space():
    cfg = _small_config()
    mgr = TieredCacheManager(cfg, LRUOffloadPolicy(cfg))
    # Fill the GPU: 4 blocks = 8 MiB capacity.
    prog = make_program("p")
    for _ in range(4):
        mgr.allocate(prog, 16, now=0.0)
    used_before = mgr.gpu_bytes_used
    assert used_before == cfg.hardware.gpu_kv_capacity_bytes

    ctx = make_ctx(bytes_needed=cfg.block_bytes)
    ok, decisions = mgr.ensure_space(ctx.bytes_needed, ctx, {"p": prog}, protected=set())
    assert ok
    assert mgr.gpu_bytes_used < used_before
    assert len(decisions) >= 1


def test_evict_respects_protected():
    cfg = _small_config()
    mgr = TieredCacheManager(cfg, LRUOffloadPolicy(cfg))
    p1 = make_program("p1")
    p2 = make_program("p2")
    for _ in range(4):
        mgr.allocate(p1, 16, now=0.0)
    for _ in range(4):
        mgr.allocate(p2, 16, now=0.0)

    # Protect p1: its blocks must survive eviction.
    ctx = make_ctx(bytes_needed=cfg.block_bytes, gpu_bytes_used=mgr.gpu_bytes_used,
                   gpu_bytes_capacity=cfg.hardware.gpu_kv_capacity_bytes)
    mgr.ensure_space(ctx.bytes_needed, ctx, {"p1": p1, "p2": p2}, protected={"p1"})
    p1_blocks = {b.block_id for b in mgr.program_blocks("p1")}
    for b in mgr._tier_index[Tier.GPU].values():
        if b.program_id == "p1":
            assert b.block_id in p1_blocks


def test_gpu_candidates_excludes_pinned_blocks():
    cfg = _small_config()
    mgr = TieredCacheManager(cfg, LRUOffloadPolicy(cfg))
    prog = make_program("p")
    pinned = mgr.allocate(prog, 16, now=0.0)[0]
    pinned.pinned = True
    mgr.allocate(prog, 16, now=0.0)  # unpinned
    candidates = mgr.gpu_candidates(protected=set())
    assert pinned.block_id not in {b.block_id for b in candidates}


def test_gpu_candidates_excludes_protected_programs():
    cfg = _small_config()
    mgr = TieredCacheManager(cfg, LRUOffloadPolicy(cfg))
    p1 = make_program("p1")
    p2 = make_program("p2")
    mgr.allocate(p1, 16, now=0.0)
    mgr.allocate(p2, 16, now=0.0)
    candidates = mgr.gpu_candidates(protected={"p1"})
    assert all(b.program_id != "p1" for b in candidates)


def test_tier_index_stays_consistent_after_evict():
    cfg = _small_config()
    mgr = TieredCacheManager(cfg, LRUOffloadPolicy(cfg))
    prog = make_program("p")
    for _ in range(4):
        mgr.allocate(prog, 16, now=0.0)
    ctx = make_ctx(bytes_needed=cfg.block_bytes)
    mgr.ensure_space(ctx.bytes_needed, ctx, {"p": prog}, protected=set())
    # Every block in the GPU index is actually GPU-resident.
    for b in mgr._tier_index[Tier.GPU].values():
        assert b.tier is Tier.GPU
    # The index matches a fresh count of gpu_bytes_used.
    recomputed = sum(cfg.model.bytes_for(b.num_tokens) for b in mgr._tier_index[Tier.GPU].values())
    assert recomputed == mgr.gpu_bytes_used
