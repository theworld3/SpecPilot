"""Reference speculative-sampling routines for statistical tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _probabilities(values: np.ndarray | list[float], name: str) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D distribution")
    if np.any(probs < 0.0) or not np.all(np.isfinite(probs)):
        raise ValueError(f"{name} contains invalid probability mass")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive probability mass")
    return probs / total


@dataclass(frozen=True)
class SampleResult:
    token: int
    draft_token: int
    accepted: bool


def speculative_sample_one(
    target: np.ndarray | list[float],
    draft: np.ndarray | list[float],
    rng: np.random.Generator,
) -> SampleResult:
    """Sample one token using lossless rejection sampling."""

    p = _probabilities(target, "target")
    q = _probabilities(draft, "draft")
    if p.shape != q.shape:
        raise ValueError("target and draft vocabularies differ")
    draft_token = int(rng.choice(p.size, p=q))
    acceptance = min(1.0, float(p[draft_token] / q[draft_token]))
    if float(rng.random()) <= acceptance:
        return SampleResult(draft_token, draft_token, True)

    residual = np.maximum(p - q, 0.0)
    residual_total = float(residual.sum())
    if residual_total <= np.finfo(np.float64).eps:
        # Only reachable through floating-point roundoff: an exact p=q draw is
        # accepted with probability one.
        residual = p
    else:
        residual /= residual_total
    token = int(rng.choice(p.size, p=residual))
    return SampleResult(token, draft_token, False)


def verify_greedy(
    draft_tokens: list[int] | tuple[int, ...],
    target_argmax: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    """Greedy linear verification, including the target bonus token.

    ``target_argmax`` must contain K+1 values: one for each proposed position
    and the all-accepted bonus position.
    """

    if len(target_argmax) != len(draft_tokens) + 1:
        raise ValueError("target_argmax must contain K+1 tokens")
    output: list[int] = []
    for index, draft_token in enumerate(draft_tokens):
        target_token = target_argmax[index]
        if draft_token != target_token:
            output.append(target_token)
            return tuple(output)
        output.append(draft_token)
    output.append(target_argmax[-1])
    return tuple(output)
