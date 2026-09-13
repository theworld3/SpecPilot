from __future__ import annotations

import pytest

from specpilot.controller import ControllerConfig, GoodputController
from specpilot.schema import TraceStep, read_trace, write_trace
from specpilot.simulator import (
    best_static,
    relative_change,
    replay,
    replay_batch_lut,
    replay_dynamic,
    replay_static,
    sweep_static,
)
from specpilot.synthetic import generate_mixed_trace, make_synthetic_profile


def test_static_replay_accounting() -> None:
    profile = make_synthetic_profile(k_max=2, batch_sizes=(1, 2))
    trace = (
        TraceStep(0, (2, 0)),
        TraceStep(1, (1,)),
    )
    result = replay_static(trace, profile, 2)
    assert result.requests == 3
    assert result.accepted_drafts == 3
    assert result.emitted_tokens == 6
    assert result.drafted_tokens == 6
    assert result.mean_acceptance_length == pytest.approx(2.0)
    assert result.draft_acceptance_rate == pytest.approx(0.5)


def test_dynamic_replay_is_reproducible() -> None:
    profile = make_synthetic_profile()
    trace = generate_mixed_trace(steps=80, seed=3)
    config = ControllerConfig(k_max=5, explore_interval=0)
    first = replay_dynamic(trace, GoodputController(profile, config))
    second = replay_dynamic(trace, GoodputController(profile, config))
    assert first.summary_dict() == second.summary_dict()
    assert first.records == second.records


def test_best_static_is_member_of_sweep() -> None:
    profile = make_synthetic_profile(k_max=3)
    trace = generate_mixed_trace(steps=30, k_max=3)
    results = sweep_static(trace, profile)
    best = best_static(trace, profile)
    assert best in results
    assert best.output_tps == max(item.output_tps for item in results)


def test_trace_jsonl_roundtrip_and_error(tmp_path) -> None:
    trace = (TraceStep(0, (0, 2)), TraceStep(1, (1,), timestamp_ms=3.5))
    path = tmp_path / "trace.jsonl"
    write_trace(path, trace)
    assert tuple(read_trace(path)) == trace
    path.write_text('{"schema_version":999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        tuple(read_trace(path))


def test_batch_lut_and_policy_validation() -> None:
    profile = make_synthetic_profile(k_max=1, batch_sizes=(1, 2))
    trace = (TraceStep(0, (1,)), TraceStep(1, (1, 0)))
    result = replay_batch_lut(trace, profile, {1: 1, 2: 0})
    assert [record.chosen_k for record in result.records] == [1, 0]
    with pytest.raises(ValueError, match="must not be empty"):
        replay_batch_lut(trace, profile, {})
    with pytest.raises(ValueError, match="negative"):
        replay(trace, profile, lambda _: (-1, "bad"), policy_name="bad")
    with pytest.raises(ValueError, match="no steps"):
        replay((), profile, lambda _: (0, "empty"), policy_name="empty")


def test_relative_change_and_trace_validation() -> None:
    assert relative_change(110.0, 100.0) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="non-zero"):
        relative_change(1.0, 0.0)
    with pytest.raises(ValueError):
        TraceStep(-1, (0,))
    with pytest.raises(ValueError):
        TraceStep(0, ())
    with pytest.raises(ValueError):
        TraceStep(0, (-1,))
    with pytest.raises(ValueError):
        TraceStep(0, (0,)).accepted_at(-1)


def test_synthetic_validation() -> None:
    with pytest.raises(ValueError):
        generate_mixed_trace(steps=0)
    with pytest.raises(ValueError):
        generate_mixed_trace(k_max=0)
    trace = generate_mixed_trace(steps=3, k_max=7)
    assert max(max(step.accepted_depths) for step in trace) <= 7
