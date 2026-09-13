"""Goodput-aware online speculative-depth controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from specpilot.acceptance import SurvivalEstimator, SurvivalSnapshot
from specpilot.cost_model import CostEstimate, ProfileCostModel


@dataclass(frozen=True)
class ControllerConfig:
    k_max: int = 5
    ewma_beta: float = 0.10
    survival_prior: float = 0.70
    default_k: int = 1
    warmup_steps: int = 4
    min_gain: float = 0.03
    fallback_margin: float = 0.02
    min_dwell_steps: int = 4
    explore_interval: int = 64
    exploration_floor: float = 0.95
    slo_penalty_lambda: float = 0.0

    def __post_init__(self) -> None:
        if self.k_max < 0:
            raise ValueError("k_max must be non-negative")
        if not 0 <= self.default_k <= self.k_max:
            raise ValueError("default_k must be in [0, k_max]")
        if not 0.0 < self.ewma_beta <= 1.0:
            raise ValueError("ewma_beta must be in (0, 1]")
        if not 0.0 <= self.survival_prior <= 1.0:
            raise ValueError("survival_prior must be in [0, 1]")
        if self.warmup_steps < 0 or self.min_dwell_steps < 0:
            raise ValueError("step counts must be non-negative")
        if self.explore_interval < 0:
            raise ValueError("explore_interval must be non-negative")
        if min(self.min_gain, self.fallback_margin, self.slo_penalty_lambda) < 0.0:
            raise ValueError("gain, fallback, and SLO penalty values must be non-negative")
        if not 0.0 <= self.exploration_floor <= 1.0:
            raise ValueError("exploration_floor must be in [0, 1]")


@dataclass(frozen=True)
class CandidateScore:
    k: int
    emitted_tokens: float
    cost: CostEstimate
    raw_goodput_tps: float
    score: float


@dataclass(frozen=True)
class Decision:
    chosen_k: int
    unconstrained_k: int
    reason: str
    scores: tuple[CandidateScore, ...]
    survival: tuple[float, ...]
    observed_batches: int
    state_committed: bool

    @property
    def chosen_score(self) -> float:
        return next(item.score for item in self.scores if item.k == self.chosen_k)


class GoodputController:
    """Choose one batch-uniform K for the next verification round.

    The controller is CPU-only and intentionally consumes completed host-side
    statistics. A runtime adapter is responsible for asynchronous D2H and for
    mapping observations by request identity before calling :meth:`observe`.
    """

    def __init__(self, cost_model: ProfileCostModel, config: ControllerConfig | None = None):
        self.cost_model = cost_model
        self.config = config or ControllerConfig(k_max=cost_model.max_k)
        if 0 not in cost_model.k_values:
            raise ValueError("the cost profile must contain a K=0 baseline")
        if self.config.k_max > cost_model.max_k:
            raise ValueError("controller k_max exceeds the cost profile")
        self.estimator = SurvivalEstimator(
            self.config.k_max,
            beta=self.config.ewma_beta,
            prior=self.config.survival_prior,
        )
        self.current_k = self.config.default_k
        self.steps_since_switch = self.config.min_dwell_steps
        self.decision_steps = 0
        self._explore_up = True

    def observe(self, accepted_per_request: list[int] | tuple[int, ...], used_k: int) -> None:
        self.estimator.observe(accepted_per_request, used_k)

    def observe_variable(
        self,
        accepted_per_request: list[int] | tuple[int, ...],
        proposed_per_request: list[int] | tuple[int, ...],
    ) -> None:
        self.estimator.observe_variable(accepted_per_request, proposed_per_request)

    def _score(
        self,
        batch_size: int,
        k: int,
        *,
        context_bucket: str,
        graph_mode: str,
        slo_ms: float | None,
    ) -> CandidateScore:
        cost = self.cost_model.estimate(
            batch_size,
            k,
            context_bucket=context_bucket,
            graph_mode=graph_mode,
        )
        emitted = batch_size * self.estimator.expected_tokens_per_request(k)
        raw_goodput = emitted / (cost.total_ms / 1000.0)
        score = raw_goodput
        if slo_ms is not None:
            if slo_ms <= 0.0:
                raise ValueError("slo_ms must be positive")
            overrun_ms = max(0.0, cost.total_ms - slo_ms)
            score -= self.config.slo_penalty_lambda * overrun_ms**2
        return CandidateScore(
            k=k,
            emitted_tokens=emitted,
            cost=cost,
            raw_goodput_tps=raw_goodput,
            score=score,
        )

    def score_candidates(
        self,
        batch_size: int,
        *,
        context_bucket: str = "default",
        graph_mode: str = "full",
        slo_ms: float | None = None,
    ) -> tuple[CandidateScore, ...]:
        scores: list[CandidateScore] = []
        for k in range(self.config.k_max + 1):
            if self.cost_model.supports(k, context_bucket, graph_mode):
                scores.append(
                    self._score(
                        batch_size,
                        k,
                        context_bucket=context_bucket,
                        graph_mode=graph_mode,
                        slo_ms=slo_ms,
                    )
                )
        if not scores or scores[0].k != 0:
            raise KeyError("selected runtime bucket has no K=0 baseline")
        return tuple(scores)

    def choose(
        self,
        batch_size: int,
        *,
        context_bucket: str = "default",
        graph_mode: str = "full",
        slo_ms: float | None = None,
        commit: bool = True,
    ) -> Decision:
        """Choose K; set ``commit=False`` for a side-effect-free shadow decision."""

        scores = self.score_candidates(
            batch_size,
            context_bucket=context_bucket,
            graph_mode=graph_mode,
            slo_ms=slo_ms,
        )
        by_k = {item.k: item for item in scores}
        baseline = by_k[0]
        best = max(scores, key=lambda item: (item.score, -item.k))
        unconstrained_k = best.k
        candidate = best.k
        reason = "argmax"

        if self.estimator.observed_batches < self.config.warmup_steps:
            candidate = self.config.default_k if self.config.default_k in by_k else 0
            reason = "warmup"

        # Safety fallback bypasses dwell/hysteresis: predicted non-positive
        # speculative benefit should never be prolonged by a stability guard.
        candidate_score = by_k[candidate]
        safety_fallback = candidate != 0 and (
            candidate_score.score < baseline.score * (1.0 + self.config.fallback_margin)
        )
        if safety_fallback:
            candidate = 0
            reason = "safety_fallback"

        current = self.current_k if self.current_k in by_k else 0
        if candidate != current and not safety_fallback:
            if self.steps_since_switch < self.config.min_dwell_steps:
                candidate = current
                reason = "min_dwell"
            elif by_k[candidate].score <= by_k[current].score * (1.0 + self.config.min_gain):
                candidate = current
                reason = "hysteresis"

        if (
            commit
            and self.config.explore_interval > 0
            and self.estimator.observed_batches >= self.config.warmup_steps
            and (self.decision_steps + 1) % self.config.explore_interval == 0
        ):
            explored = self._exploration_candidate(candidate, by_k, baseline)
            if explored != candidate:
                candidate = explored
                reason = "exploration"

        if commit:
            if candidate != self.current_k:
                self.current_k = candidate
                self.steps_since_switch = 0
            else:
                self.steps_since_switch += 1
            self.decision_steps += 1

        return Decision(
            chosen_k=candidate,
            unconstrained_k=unconstrained_k,
            reason=reason,
            scores=scores,
            survival=tuple(float(x) for x in self.estimator.survival),
            observed_batches=self.estimator.observed_batches,
            state_committed=commit,
        )

    def _exploration_candidate(
        self,
        candidate: int,
        by_k: dict[int, CandidateScore],
        baseline: CandidateScore,
    ) -> int:
        directions = (1, -1) if self._explore_up else (-1, 1)
        self._explore_up = not self._explore_up
        for direction in directions:
            neighbor = candidate + direction
            if neighbor not in by_k:
                continue
            score = by_k[neighbor].score
            # Exploration is the deliberate exception to the normal safety
            # margin. It may spend one bounded round below K=0 so a controller
            # parked at K=0 can detect a workload recovery. The floor below
            # limits that predicted loss.
            if neighbor != 0 and score < baseline.score * self.config.exploration_floor:
                continue
            if score >= by_k[candidate].score * self.config.exploration_floor:
                return neighbor
        return candidate

    def state_dict(self) -> dict[str, Any]:
        snapshot = self.estimator.snapshot()
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "current_k": self.current_k,
            "steps_since_switch": self.steps_since_switch,
            "decision_steps": self.decision_steps,
            "explore_up": self._explore_up,
            "survival": asdict(snapshot),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported controller state schema")
        if state.get("config") != asdict(self.config):
            raise ValueError("controller state configuration mismatch")
        current_k = int(state["current_k"])
        if not 0 <= current_k <= self.config.k_max:
            raise ValueError("invalid current_k in controller state")
        raw_snapshot = state["survival"]
        snapshot = SurvivalSnapshot(
            values=tuple(raw_snapshot["values"]),
            observations_per_position=tuple(raw_snapshot["observations_per_position"]),
            observed_batches=int(raw_snapshot["observed_batches"]),
            observed_requests=int(raw_snapshot["observed_requests"]),
        )
        self.estimator.load_snapshot(snapshot)
        self.current_k = current_k
        self.steps_since_switch = int(state["steps_since_switch"])
        self.decision_steps = int(state["decision_steps"])
        self._explore_up = bool(state["explore_up"])
