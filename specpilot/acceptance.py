"""Online, position-aware acceptance estimation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurvivalSnapshot:
    """Immutable estimator state used in telemetry and checkpoints."""

    values: tuple[float, ...]
    observations_per_position: tuple[int, ...]
    observed_batches: int
    observed_requests: int


class SurvivalEstimator:
    """EWMA estimator for ``s_j = P(A >= j)``.

    Only positions proposed in a round are updated. This is important: if a
    round uses K=2, positions 3..K_max are unobserved rather than rejected.
    """

    def __init__(
        self,
        k_max: int,
        beta: float = 0.1,
        prior: float | Iterable[float] = 0.7,
    ) -> None:
        if k_max < 0:
            raise ValueError("k_max must be non-negative")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        self.k_max = int(k_max)
        self.beta = float(beta)
        if np.isscalar(prior):
            values = np.full(self.k_max, float(prior), dtype=np.float64)
        else:
            values = np.asarray(tuple(prior), dtype=np.float64)
            if values.shape != (self.k_max,):
                raise ValueError(f"prior must contain exactly {self.k_max} values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("prior survival probabilities must be in [0, 1]")
        self._survival = self._project(values)
        self._observations = np.zeros(self.k_max, dtype=np.int64)
        self.observed_batches = 0
        self.observed_requests = 0

    @staticmethod
    def _project(values: np.ndarray) -> np.ndarray:
        """Clip probabilities and enforce ``s_1 >= ... >= s_K``."""

        clipped = np.clip(values, 0.0, 1.0)
        return np.minimum.accumulate(clipped)

    @property
    def survival(self) -> np.ndarray:
        """Return a copy so callers cannot mutate estimator state."""

        return self._survival.copy()

    @property
    def observations_per_position(self) -> np.ndarray:
        return self._observations.copy()

    def expected_tokens_per_request(self, k: int) -> float:
        if not 0 <= k <= self.k_max:
            raise ValueError(f"k must be in [0, {self.k_max}]")
        return 1.0 + float(self._survival[:k].sum())

    def observe(self, accepted_per_request: Iterable[int], used_k: int) -> None:
        """Update from one completed verification round.

        ``accepted_per_request`` is keyed to request identity by the runtime
        adapter before it reaches this method; slot indices are deliberately
        not part of the API.
        """

        if not 0 <= used_k <= self.k_max:
            raise ValueError(f"used_k must be in [0, {self.k_max}]")
        accepted = np.asarray(tuple(accepted_per_request), dtype=np.int64)
        if accepted.ndim != 1 or accepted.size == 0:
            raise ValueError("accepted_per_request must be a non-empty 1-D sequence")
        if np.any(accepted < 0) or np.any(accepted > used_k):
            raise ValueError("accepted counts must be between zero and used_k")

        proposed = np.full(accepted.shape, used_k, dtype=np.int64)
        self._observe_arrays(accepted, proposed)

    def observe_variable(
        self,
        accepted_per_request: Iterable[int],
        proposed_per_request: Iterable[int],
    ) -> None:
        """Update a round where grammar/EOS trimming made proposal lengths ragged."""

        accepted = np.asarray(tuple(accepted_per_request), dtype=np.int64)
        proposed = np.asarray(tuple(proposed_per_request), dtype=np.int64)
        if accepted.ndim != 1 or accepted.size == 0 or accepted.shape != proposed.shape:
            raise ValueError("accepted and proposed counts must be equal, non-empty 1-D sequences")
        if np.any(proposed < 0) or np.any(proposed > self.k_max):
            raise ValueError("proposed counts must be between zero and k_max")
        if np.any(accepted < 0) or np.any(accepted > proposed):
            raise ValueError("accepted counts must be between zero and proposed counts")
        self._observe_arrays(accepted, proposed)

    def _observe_arrays(self, accepted: np.ndarray, proposed: np.ndarray) -> None:
        for pos in range(self.k_max):
            observed_mask = proposed >= pos + 1
            denominator = int(np.count_nonzero(observed_mask))
            if denominator == 0:
                continue
            observation = float(np.mean(accepted[observed_mask] >= pos + 1))
            self._survival[pos] = (1.0 - self.beta) * self._survival[pos] + self.beta * observation
            self._observations[pos] += denominator

        self._survival = self._project(self._survival)
        self.observed_batches += 1
        self.observed_requests += int(accepted.size)

    def snapshot(self) -> SurvivalSnapshot:
        return SurvivalSnapshot(
            values=tuple(float(x) for x in self._survival),
            observations_per_position=tuple(int(x) for x in self._observations),
            observed_batches=self.observed_batches,
            observed_requests=self.observed_requests,
        )

    def load_snapshot(self, snapshot: SurvivalSnapshot) -> None:
        if len(snapshot.values) != self.k_max:
            raise ValueError("snapshot k_max does not match estimator")
        values = np.asarray(snapshot.values, dtype=np.float64)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("snapshot contains invalid probabilities")
        observations = np.asarray(snapshot.observations_per_position, dtype=np.int64)
        if observations.shape != (self.k_max,) or np.any(observations < 0):
            raise ValueError("snapshot contains invalid observation counts")
        self._survival = self._project(values)
        self._observations = observations.copy()
        self.observed_batches = int(snapshot.observed_batches)
        self.observed_requests = int(snapshot.observed_requests)
