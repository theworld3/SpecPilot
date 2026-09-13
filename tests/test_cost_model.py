from __future__ import annotations

import json

import pytest

from specpilot.cost_model import CostEstimate, ProfileCostModel


def test_linear_interpolation(simple_profile: ProfileCostModel) -> None:
    left = simple_profile.estimate(1, 1)
    right = simple_profile.estimate(8, 1)
    middle = simple_profile.estimate(4, 1)
    weight = 3 / 7
    assert middle.draft_ms == pytest.approx(left.draft_ms)
    assert middle.verify_ms == pytest.approx(
        left.verify_ms + (right.verify_ms - left.verify_ms) * weight
    )


def test_clamp_does_not_change_measured_cost(simple_profile: ProfileCostModel) -> None:
    boundary = simple_profile.estimate(8, 2)
    extrapolated = simple_profile.estimate(64, 2)
    assert extrapolated.batch_size == 64
    assert extrapolated.total_ms == pytest.approx(boundary.total_ms)


def test_out_of_range_error(simple_profile: ProfileCostModel) -> None:
    strict = ProfileCostModel.from_dict(simple_profile.to_dict(), out_of_range="error")
    with pytest.raises(KeyError):
        strict.estimate(64, 0)


def test_roundtrip_json(tmp_path, simple_profile: ProfileCostModel) -> None:
    path = tmp_path / "profile.json"
    simple_profile.save(path)
    restored = ProfileCostModel.load(path)
    assert restored.to_dict() == simple_profile.to_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_duplicate_and_invalid_rows_rejected() -> None:
    row = CostEstimate(1, 0, 0.0, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        ProfileCostModel([row, row])
    with pytest.raises(ValueError):
        CostEstimate(1, 0, 0.0, 0.0)


def test_missing_runtime_bucket(simple_profile: ProfileCostModel) -> None:
    with pytest.raises(KeyError):
        simple_profile.estimate(1, 0, context_bucket="long")


def test_invalid_model_and_schema(simple_profile: ProfileCostModel) -> None:
    with pytest.raises(ValueError, match="out_of_range"):
        ProfileCostModel([CostEstimate(1, 0, 0.0, 1.0)], out_of_range="linear")
    with pytest.raises(ValueError, match="at least"):
        ProfileCostModel([])
    with pytest.raises(ValueError, match="schema_version"):
        ProfileCostModel.from_dict({"schema_version": 2, "rows": []})
    with pytest.raises(ValueError, match="rows"):
        ProfileCostModel.from_dict({"schema_version": 1, "rows": {}})
    with pytest.raises(ValueError, match="invalid cost"):
        ProfileCostModel.from_dict({"schema_version": 1, "rows": [{"batch_size": 0}]})
    with pytest.raises(ValueError):
        simple_profile.estimate(0, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0, "k": 0, "draft_ms": 0.0, "verify_ms": 1.0},
        {"batch_size": 1, "k": -1, "draft_ms": 0.0, "verify_ms": 1.0},
        {"batch_size": 1, "k": 0, "draft_ms": -1.0, "verify_ms": 1.0},
    ],
)
def test_invalid_cost_estimate(kwargs) -> None:
    with pytest.raises(ValueError):
        CostEstimate(**kwargs)
