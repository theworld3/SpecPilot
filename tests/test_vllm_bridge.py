from __future__ import annotations

import pytest

from specpilot.synthetic import make_synthetic_profile
from specpilot.vllm_bridge import SchedulerControllerBridge


def test_bridge_loads_profile_and_supports_shadow(tmp_path) -> None:
    path = tmp_path / "profile.json"
    make_synthetic_profile(k_max=2, batch_sizes=(1, 2)).save(path)
    bridge = SchedulerControllerBridge.from_config(
        {
            "policy": "shadow_goodput",
            "profile_path": str(path),
            "static_k": 2,
            "warmup_steps": 0,
            "min_dwell_steps": 0,
            "explore_interval": 0,
        },
        k_max=2,
    )
    assert bridge.choose(1) == 2
    assert bridge.last_decision is not None
    assert bridge.last_decision.active_k == 2
    bridge.observe(accepted_per_request=[1, 0], proposed_per_request=[1, 0])
    assert bridge.controller.estimator.observed_batches == 1


def test_bridge_rejects_unknown_config(tmp_path) -> None:
    path = tmp_path / "profile.json"
    make_synthetic_profile(k_max=1, batch_sizes=(1,)).save(path)
    with pytest.raises(ValueError, match="unknown"):
        SchedulerControllerBridge.from_config({"profile_path": str(path), "typo": 1}, k_max=1)


def test_bridge_config_validation(tmp_path) -> None:
    path = tmp_path / "profile.json"
    make_synthetic_profile(k_max=1, batch_sizes=(1,)).save(path)
    valid = SchedulerControllerBridge.from_config({"profile_path": str(path)}, k_max=1)
    with pytest.raises(ValueError, match="mode"):
        SchedulerControllerBridge(valid.controller, mode="bad")
    with pytest.raises(ValueError, match="static_k"):
        SchedulerControllerBridge(valid.controller, static_k=2)


def test_bridge_missing_profile(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SchedulerControllerBridge.from_config({}, k_max=1)
    with pytest.raises(ValueError, match="not found"):
        SchedulerControllerBridge.from_config({"profile_path": str(tmp_path / "missing")}, k_max=1)
