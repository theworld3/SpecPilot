"""Deterministic synthetic fixtures for CI and controller development.

Nothing generated here is a hardware benchmark. Files produced by these
helpers carry ``synthetic=true`` in their metadata.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from specpilot.cost_model import CostEstimate, ProfileCostModel
from specpilot.schema import TraceStep


def make_synthetic_profile(
    *,
    k_max: int = 5,
    batch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    context_buckets: Sequence[str] = ("default",),
) -> ProfileCostModel:
    rows: list[CostEstimate] = []
    for context in context_buckets:
        context_scale = {"short": 0.85, "default": 1.0, "long": 1.35}.get(context, 1.0)
        for batch in batch_sizes:
            for k in range(k_max + 1):
                query_tokens = batch * (k + 1)
                # A launch-dominated base plus a convex saturation term. This
                # creates a realistic regime where long drafts help at low B
                # but lose to K=0 at high B.
                draft_ms = k * (0.055 + 0.0030 * batch) * context_scale
                verify_ms = (
                    0.62 + 0.0085 * query_tokens + 0.00024 * query_tokens**2
                ) * context_scale
                sample_ms = 0.035 + 0.0015 * batch * (1.0 + 0.12 * k)
                scheduler_ms = 0.022 + 0.0008 * batch
                rows.append(
                    CostEstimate(
                        batch_size=batch,
                        k=k,
                        draft_ms=draft_ms,
                        verify_ms=verify_ms,
                        sample_ms=sample_ms,
                        scheduler_ms=scheduler_ms,
                        context_bucket=context,
                        graph_mode="full",
                    )
                )
    return ProfileCostModel(
        rows,
        metadata={
            "synthetic": True,
            "warning": "CI fixture only; do not report as GPU measurement",
        },
    )


def generate_mixed_trace(
    *,
    steps: int = 240,
    k_max: int = 5,
    seed: int = 7,
    batch_sizes: Sequence[int] = (1, 4, 8, 16, 32, 64),
) -> tuple[TraceStep, ...]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    rng = np.random.default_rng(seed)
    phases = (
        np.asarray((0.93, 0.84, 0.74, 0.64, 0.54), dtype=np.float64),
        np.asarray((0.62, 0.39, 0.24, 0.13, 0.07), dtype=np.float64),
        np.asarray((0.88, 0.78, 0.67, 0.56, 0.46), dtype=np.float64),
    )
    phases = tuple(_resize_survival(values, k_max) for values in phases)
    trace: list[TraceStep] = []
    for index in range(steps):
        phase = phases[min(index * len(phases) // steps, len(phases) - 1)]
        # Alternate load envelopes as well as language difficulty.
        load_phase = (index // max(1, steps // 6)) % 3
        candidates = batch_sizes[:3] if load_phase == 0 else batch_sizes[2:]
        batch = int(rng.choice(candidates))
        draws = rng.random(batch)
        depths = tuple(int(np.sum(draw <= phase)) for draw in draws)
        trace.append(TraceStep(step=index, accepted_depths=depths))
    return tuple(trace)


def _resize_survival(values: np.ndarray, k_max: int) -> np.ndarray:
    if k_max <= values.size:
        return values[:k_max]
    tail = [float(values[-1]) * 0.75 ** (index + 1) for index in range(k_max - values.size)]
    return np.concatenate((values, np.asarray(tail)))
