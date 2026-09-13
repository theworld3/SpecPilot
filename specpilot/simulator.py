"""Counterfactual trace replay for static, LUT, and online policies."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np

from specpilot.controller import GoodputController
from specpilot.cost_model import ProfileCostModel
from specpilot.schema import TraceStep


@dataclass(frozen=True)
class StepResult:
    step: int
    batch_size: int
    chosen_k: int
    accepted_drafts: int
    emitted_tokens: int
    step_ms: float
    context_bucket: str
    reason: str


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    steps: int
    requests: int
    emitted_tokens: int
    accepted_drafts: int
    drafted_tokens: int
    total_ms: float
    output_tps: float
    request_throughput_rps: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    mean_acceptance_length: float
    draft_acceptance_rate: float
    chosen_k_histogram: dict[int, int]
    records: tuple[StepResult, ...]

    def summary_dict(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("records")
        return values


Policy = Callable[[TraceStep], tuple[int, str]]
Observer = Callable[[TraceStep, int, tuple[int, ...]], None]


def replay(
    steps: Iterable[TraceStep],
    cost_model: ProfileCostModel,
    policy: Policy,
    *,
    policy_name: str,
    observer: Observer | None = None,
) -> ReplayResult:
    records: list[StepResult] = []
    per_request_tpot: list[float] = []
    total_requests = 0
    total_emitted = 0
    total_accepted = 0
    total_drafted = 0
    total_ms = 0.0
    spec_requests = 0
    histogram: Counter[int] = Counter()

    for trace_step in steps:
        k, reason = policy(trace_step)
        if k < 0:
            raise ValueError("policy returned a negative K")
        accepted = trace_step.accepted_at(k)
        estimate = cost_model.estimate(
            trace_step.batch_size,
            k,
            context_bucket=trace_step.context_bucket,
            graph_mode=trace_step.graph_mode,
        )
        emitted_per_request = tuple(value + 1 for value in accepted)
        emitted = sum(emitted_per_request)
        accepted_count = sum(accepted)
        drafted = trace_step.batch_size * k

        records.append(
            StepResult(
                step=trace_step.step,
                batch_size=trace_step.batch_size,
                chosen_k=k,
                accepted_drafts=accepted_count,
                emitted_tokens=emitted,
                step_ms=estimate.total_ms,
                context_bucket=trace_step.context_bucket,
                reason=reason,
            )
        )
        per_request_tpot.extend(estimate.total_ms / count for count in emitted_per_request)
        total_requests += trace_step.batch_size
        total_emitted += emitted
        total_accepted += accepted_count
        total_drafted += drafted
        total_ms += estimate.total_ms
        histogram[k] += 1
        if k > 0:
            spec_requests += trace_step.batch_size
        if observer is not None:
            observer(trace_step, k, accepted)

    if not records:
        raise ValueError("trace contains no steps")
    seconds = total_ms / 1000.0
    mean_acceptance = 1.0 + total_accepted / spec_requests if spec_requests else 1.0
    acceptance_rate = total_accepted / total_drafted if total_drafted else 0.0
    return ReplayResult(
        policy=policy_name,
        steps=len(records),
        requests=total_requests,
        emitted_tokens=total_emitted,
        accepted_drafts=total_accepted,
        drafted_tokens=total_drafted,
        total_ms=total_ms,
        output_tps=total_emitted / seconds,
        request_throughput_rps=total_requests / seconds,
        tpot_p50_ms=float(np.percentile(per_request_tpot, 50)),
        tpot_p95_ms=float(np.percentile(per_request_tpot, 95)),
        tpot_p99_ms=float(np.percentile(per_request_tpot, 99)),
        mean_acceptance_length=mean_acceptance,
        draft_acceptance_rate=acceptance_rate,
        chosen_k_histogram=dict(sorted(histogram.items())),
        records=tuple(records),
    )


def replay_static(steps: Iterable[TraceStep], cost_model: ProfileCostModel, k: int) -> ReplayResult:
    return replay(steps, cost_model, lambda _: (k, "static"), policy_name=f"static_k{k}")


def replay_batch_lut(
    steps: Iterable[TraceStep],
    cost_model: ProfileCostModel,
    lut: Mapping[int, int],
) -> ReplayResult:
    if not lut:
        raise ValueError("batch LUT must not be empty")
    thresholds = sorted(lut)

    def choose(step: TraceStep) -> tuple[int, str]:
        eligible = [threshold for threshold in thresholds if step.batch_size <= threshold]
        threshold = eligible[0] if eligible else thresholds[-1]
        return lut[threshold], "batch_lut"

    return replay(steps, cost_model, choose, policy_name="batch_lut")


def replay_dynamic(
    steps: Iterable[TraceStep],
    controller: GoodputController,
    *,
    slo_ms: float | None = None,
) -> ReplayResult:
    def choose(step: TraceStep) -> tuple[int, str]:
        decision = controller.choose(
            step.batch_size,
            context_bucket=step.context_bucket,
            graph_mode=step.graph_mode,
            slo_ms=slo_ms,
        )
        return decision.chosen_k, decision.reason

    def observe(_: TraceStep, k: int, accepted: tuple[int, ...]) -> None:
        controller.observe(accepted, k)

    return replay(steps, controller.cost_model, choose, policy_name="specpilot", observer=observe)


def sweep_static(
    steps: Iterable[TraceStep], cost_model: ProfileCostModel, k_max: int | None = None
) -> tuple[ReplayResult, ...]:
    materialized = tuple(steps)
    upper = cost_model.max_k if k_max is None else k_max
    return tuple(replay_static(materialized, cost_model, k) for k in range(upper + 1))


def best_static(
    steps: Iterable[TraceStep], cost_model: ProfileCostModel, k_max: int | None = None
) -> ReplayResult:
    results = sweep_static(steps, cost_model, k_max)
    return max(results, key=lambda item: item.output_tps)


def relative_change(value: float, baseline: float) -> float:
    if baseline == 0.0:
        raise ValueError("baseline must be non-zero")
    return (value / baseline - 1.0) * 100.0
