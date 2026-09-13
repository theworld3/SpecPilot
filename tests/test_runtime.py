from __future__ import annotations

import pytest

from specpilot.runtime import KVLedger


def test_rejected_target_tail_is_not_logically_visible() -> None:
    ledger = KVLedger()
    handle = ledger.add_request("a", slot=0, prompt_tokens=15)
    ledger.speculative_write(handle, k=5)
    ledger.commit(handle, accepted_drafts=1)
    assert ledger.logical_tokens(handle) == 17
    assert ledger.prefix_hash_length(handle) == 17
    target_physical, _ = ledger.physical_tokens(handle)
    assert target_physical == 21


def test_eos_truncation_wins_over_bonus_accounting() -> None:
    ledger = KVLedger()
    handle = ledger.add_request("eos", slot=1, prompt_tokens=4)
    ledger.speculative_write(handle, k=3)
    ledger.commit(handle, accepted_drafts=3, emitted_tokens=2)
    assert ledger.logical_tokens(handle) == 6


def test_zero_k_round_commits_one_target_token() -> None:
    ledger = KVLedger()
    handle = ledger.add_request("zero", slot=0, prompt_tokens=2)
    ledger.speculative_write(handle, k=0)
    ledger.commit(handle, accepted_drafts=0)
    assert ledger.logical_tokens(handle) == 3


def test_batch_reorder_invalidates_stale_handles() -> None:
    ledger = KVLedger()
    old_a = ledger.add_request("a", slot=0, prompt_tokens=8)
    old_b = ledger.add_request("b", slot=1, prompt_tokens=8)
    new = ledger.reorder({"a": 1, "b": 0})
    with pytest.raises(RuntimeError, match="stale"):
        ledger.logical_tokens(old_a)
    assert ledger.logical_tokens(new["a"]) == 8
    assert ledger.logical_tokens(new["b"]) == 8
    with pytest.raises(RuntimeError, match="stale"):
        ledger.logical_tokens(old_b)


def test_slot_reuse_invalidates_removed_request() -> None:
    ledger = KVLedger()
    old = ledger.add_request("old", slot=3, prompt_tokens=1)
    ledger.remove_request(old)
    new = ledger.add_request("new", slot=3, prompt_tokens=2)
    with pytest.raises(KeyError):
        ledger.logical_tokens(old)
    assert new.generation > old.generation


def test_runtime_validation() -> None:
    ledger = KVLedger()
    handle = ledger.add_request("a", 0, 1)
    with pytest.raises(ValueError, match="already"):
        ledger.add_request("a", 1, 1)
    with pytest.raises(ValueError, match="occupied"):
        ledger.add_request("b", 0, 1)
    with pytest.raises(ValueError, match="prompt"):
        KVLedger().add_request("x", 0, -1)
    with pytest.raises(ValueError, match="negative"):
        ledger.speculative_write(handle, -1)
    with pytest.raises(RuntimeError, match="no pending"):
        ledger.commit(handle, accepted_drafts=0)
    ledger.speculative_write(handle, 1)
    with pytest.raises(RuntimeError, match="already"):
        ledger.speculative_write(handle, 1)
    with pytest.raises(ValueError, match="outside"):
        ledger.commit(handle, accepted_drafts=2)
    with pytest.raises(ValueError, match="truncation"):
        ledger.commit(handle, accepted_drafts=1, emitted_tokens=3)


def test_invalid_reorder_mapping() -> None:
    ledger = KVLedger()
    ledger.add_request("a", 0, 1)
    ledger.add_request("b", 1, 1)
    with pytest.raises(ValueError, match="every"):
        ledger.reorder({"a": 1})
    with pytest.raises(ValueError, match="duplicate"):
        ledger.reorder({"a": 2, "b": 2})
