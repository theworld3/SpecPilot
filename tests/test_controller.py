from __future__ import annotations

import pytest

from specpilot.controller import ControllerConfig, GoodputController
from specpilot.cost_model import CostEstimate, ProfileCostModel


def _profile(costs: tuple[float, ...]) -> ProfileCostModel:
    return ProfileCostModel([CostEstimate(1, k, 0.0, cost) for k, cost in enumerate(costs)])


def test_selects_goodput_argmax() -> None:
    controller = GoodputController(
        _profile((1.0, 1.2, 2.5)),
        ControllerConfig(
            k_max=2,
            survival_prior=1.0,
            default_k=0,
            warmup_steps=0,
            min_gain=0.0,
            fallback_margin=0.0,
            min_dwell_steps=0,
            explore_interval=0,
        ),
    )
    decision = controller.choose(1)
    assert decision.chosen_k == 1
    assert decision.reason == "argmax"
    assert decision.chosen_score == pytest.approx(2 / 0.0012)


def test_safety_fallback_beats_warmup_default() -> None:
    controller = GoodputController(
        _profile((1.0, 3.0)),
        ControllerConfig(k_max=1, default_k=1, warmup_steps=10, explore_interval=0),
    )
    decision = controller.choose(1)
    assert decision.chosen_k == 0
    assert decision.reason == "safety_fallback"


def test_hysteresis_rejects_small_gain() -> None:
    controller = GoodputController(
        _profile((1.0, 1.96)),
        ControllerConfig(
            k_max=1,
            survival_prior=1.0,
            default_k=0,
            warmup_steps=0,
            min_gain=0.03,
            fallback_margin=0.0,
            min_dwell_steps=0,
            explore_interval=0,
        ),
    )
    decision = controller.choose(1)
    assert decision.unconstrained_k == 1
    assert decision.chosen_k == 0
    assert decision.reason == "hysteresis"


def test_min_dwell_blocks_non_safety_switch() -> None:
    controller = GoodputController(
        _profile((1.0, 1.3)),
        ControllerConfig(
            k_max=1,
            survival_prior=1.0,
            default_k=0,
            warmup_steps=0,
            min_gain=0.0,
            fallback_margin=0.0,
            min_dwell_steps=3,
            explore_interval=0,
        ),
    )
    assert controller.choose(1).chosen_k == 1
    controller.current_k = 0
    controller.steps_since_switch = 1
    decision = controller.choose(1)
    assert decision.chosen_k == 0
    assert decision.reason == "min_dwell"


def test_shadow_decision_does_not_mutate_state(simple_profile: ProfileCostModel) -> None:
    controller = GoodputController(
        simple_profile,
        ControllerConfig(k_max=2, warmup_steps=0, min_dwell_steps=0, explore_interval=0),
    )
    before = controller.state_dict()
    decision = controller.choose(4, commit=False)
    assert not decision.state_committed
    assert controller.state_dict() == before


def test_state_roundtrip(simple_profile: ProfileCostModel) -> None:
    config = ControllerConfig(k_max=2, explore_interval=0)
    source = GoodputController(simple_profile, config)
    source.observe([0, 1], 1)
    source.choose(1)
    target = GoodputController(simple_profile, config)
    target.load_state_dict(source.state_dict())
    assert target.state_dict() == source.state_dict()


def test_slo_validation(simple_profile: ProfileCostModel) -> None:
    controller = GoodputController(simple_profile, ControllerConfig(k_max=2))
    with pytest.raises(ValueError, match="slo_ms"):
        controller.choose(1, slo_ms=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k_max": -1},
        {"k_max": 1, "default_k": 2},
        {"ewma_beta": 0.0},
        {"survival_prior": 1.1},
        {"warmup_steps": -1},
        {"explore_interval": -1},
        {"min_gain": -0.1},
        {"exploration_floor": 1.1},
    ],
)
def test_invalid_controller_config(kwargs) -> None:
    with pytest.raises(ValueError):
        ControllerConfig(**kwargs)


def test_controller_requires_baseline_and_profile_depth() -> None:
    no_baseline = ProfileCostModel([CostEstimate(1, 1, 0.0, 1.0)])
    with pytest.raises(ValueError, match="K=0"):
        GoodputController(no_baseline, ControllerConfig(k_max=1))
    with pytest.raises(ValueError, match="exceeds"):
        GoodputController(_profile((1.0,)), ControllerConfig(k_max=1))


def test_slo_penalty_and_exploration() -> None:
    controller = GoodputController(
        _profile((1.0, 1.2)),
        ControllerConfig(
            k_max=1,
            survival_prior=1.0,
            default_k=1,
            warmup_steps=0,
            min_gain=0.0,
            fallback_margin=0.0,
            min_dwell_steps=0,
            explore_interval=1,
            exploration_floor=0.0,
            slo_penalty_lambda=1000.0,
        ),
    )
    decision = controller.choose(1, slo_ms=1.0)
    assert decision.reason == "exploration"
    assert decision.chosen_k == 0


def test_invalid_controller_state(simple_profile: ProfileCostModel) -> None:
    controller = GoodputController(simple_profile, ControllerConfig(k_max=2))
    state = controller.state_dict()
    state["schema_version"] = 99
    with pytest.raises(ValueError, match="schema"):
        controller.load_state_dict(state)
    state = controller.state_dict()
    state["config"]["min_gain"] = 999
    with pytest.raises(ValueError, match="configuration"):
        controller.load_state_dict(state)
