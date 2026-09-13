from __future__ import annotations

import numpy as np
import pytest

from specpilot.acceptance import SurvivalEstimator


def test_only_observed_positions_are_updated() -> None:
    estimator = SurvivalEstimator(4, beta=1.0, prior=(0.8, 0.7, 0.6, 0.5))
    estimator.observe([2, 1, 0, 2], used_k=2)
    np.testing.assert_allclose(estimator.survival, (0.75, 0.5, 0.5, 0.5))
    assert estimator.observations_per_position.tolist() == [4, 4, 0, 0]


def test_variable_proposal_lengths_do_not_count_missing_as_rejected() -> None:
    estimator = SurvivalEstimator(3, beta=1.0, prior=0.5)
    estimator.observe_variable(
        accepted_per_request=[3, 1, 0],
        proposed_per_request=[3, 1, 0],
    )
    np.testing.assert_allclose(estimator.survival, (1.0, 1.0, 1.0))
    assert estimator.observations_per_position.tolist() == [2, 1, 1]


def test_monotonic_projection_and_expected_length() -> None:
    estimator = SurvivalEstimator(3, prior=(0.2, 0.9, 0.4))
    np.testing.assert_allclose(estimator.survival, (0.2, 0.2, 0.2))
    assert estimator.expected_tokens_per_request(2) == pytest.approx(1.4)


@pytest.mark.parametrize("used_k,accepted", [(-1, [0]), (4, [0]), (1, [-1]), (1, [2])])
def test_invalid_observation_rejected(used_k: int, accepted: list[int]) -> None:
    estimator = SurvivalEstimator(3)
    with pytest.raises(ValueError):
        estimator.observe(accepted, used_k)


def test_snapshot_roundtrip() -> None:
    source = SurvivalEstimator(2, beta=0.5)
    source.observe([0, 1], 1)
    target = SurvivalEstimator(2, beta=0.5, prior=0.1)
    target.load_snapshot(source.snapshot())
    np.testing.assert_allclose(target.survival, source.survival)
    assert target.snapshot() == source.snapshot()


@pytest.mark.parametrize(
    "args",
    [
        {"k_max": -1},
        {"k_max": 2, "beta": 0.0},
        {"k_max": 2, "prior": (0.1,)},
        {"k_max": 2, "prior": 1.1},
    ],
)
def test_invalid_constructor(args) -> None:
    with pytest.raises(ValueError):
        SurvivalEstimator(**args)


def test_invalid_expected_length_and_empty_observation() -> None:
    estimator = SurvivalEstimator(2)
    with pytest.raises(ValueError):
        estimator.expected_tokens_per_request(3)
    with pytest.raises(ValueError):
        estimator.observe([], 0)


@pytest.mark.parametrize(
    "accepted,proposed",
    [([], []), ([0], [0, 1]), ([0], [3]), ([1], [0])],
)
def test_invalid_variable_observation(accepted, proposed) -> None:
    with pytest.raises(ValueError):
        SurvivalEstimator(2).observe_variable(accepted, proposed)


def test_invalid_snapshot_rejected() -> None:
    snapshot = SurvivalEstimator(2).snapshot()
    with pytest.raises(ValueError, match="k_max"):
        SurvivalEstimator(1).load_snapshot(snapshot)
