from __future__ import annotations

import numpy as np
import pytest

from specpilot.sampling import speculative_sample_one, verify_greedy


def test_rejection_sampling_matches_target_distribution() -> None:
    target = np.asarray((0.05, 0.15, 0.30, 0.50))
    draft = np.asarray((0.40, 0.10, 0.10, 0.40))
    rng = np.random.default_rng(12345)
    counts = np.zeros(4, dtype=np.int64)
    for _ in range(50_000):
        counts[speculative_sample_one(target, draft, rng).token] += 1
    empirical = counts / counts.sum()
    tvd = 0.5 * np.abs(empirical - target).sum()
    assert tvd < 0.01


def test_equal_distributions_always_accept() -> None:
    rng = np.random.default_rng(9)
    probs = [0.2, 0.3, 0.5]
    assert all(speculative_sample_one(probs, probs, rng).accepted for _ in range(1_000))


def test_greedy_mismatch_and_bonus() -> None:
    assert verify_greedy([1, 2, 3], [1, 9, 8, 7]) == (1, 9)
    assert verify_greedy([1, 2, 3], [1, 2, 3, 7]) == (1, 2, 3, 7)
    with pytest.raises(ValueError):
        verify_greedy([1], [1])


@pytest.mark.parametrize(
    "target,draft",
    [
        ([], [1.0]),
        ([1.0], []),
        ([-1.0, 2.0], [0.5, 0.5]),
        ([0.0], [1.0]),
        ([1.0], [0.5, 0.5]),
    ],
)
def test_invalid_distributions(target, draft) -> None:
    with pytest.raises(ValueError):
        speculative_sample_one(target, draft, np.random.default_rng(1))
