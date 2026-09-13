"""Narrow scheduler-side bridge for the pinned vLLM prototype patch.

The bridge has no torch dependency and performs no device synchronization. It
expects accepted/proposed counts already present on the scheduler host after a
completed model-runner step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from specpilot.controller import ControllerConfig, Decision, GoodputController
from specpilot.cost_model import ProfileCostModel


@dataclass(frozen=True)
class BridgeDecision:
    active_k: int
    shadow_k: int
    reason: str


class SchedulerControllerBridge:
    """Translate a small vLLM config mapping to :class:`GoodputController`."""

    _CONFIG_FIELDS: ClassVar[set[str]] = {
        "ewma_beta",
        "survival_prior",
        "default_k",
        "warmup_steps",
        "min_gain",
        "fallback_margin",
        "min_dwell_steps",
        "explore_interval",
        "exploration_floor",
        "slo_penalty_lambda",
    }

    def __init__(
        self,
        controller: GoodputController,
        *,
        mode: str = "online_goodput",
        static_k: int | None = None,
    ) -> None:
        if mode not in {"online_goodput", "shadow_goodput"}:
            raise ValueError("mode must be online_goodput or shadow_goodput")
        self.controller = controller
        self.mode = mode
        self.static_k = controller.config.k_max if static_k is None else static_k
        if not 0 <= self.static_k <= controller.config.k_max:
            raise ValueError("static_k must be in [0, k_max]")
        self.last_decision: BridgeDecision | None = None

    @classmethod
    def from_config(
        cls,
        dynamic_depth: dict[str, Any],
        *,
        k_max: int,
    ) -> SchedulerControllerBridge:
        unknown = (
            set(dynamic_depth)
            - cls._CONFIG_FIELDS
            - {
                "policy",
                "profile_path",
                "static_k",
            }
        )
        if unknown:
            raise ValueError(f"unknown dynamic_depth keys: {sorted(unknown)}")
        profile_path = dynamic_depth.get("profile_path")
        if not isinstance(profile_path, str) or not profile_path:
            raise ValueError("dynamic_depth.profile_path must be a non-empty path")
        if not Path(profile_path).is_file():
            raise ValueError(f"cost profile not found: {profile_path}")
        overrides = {
            key: value for key, value in dynamic_depth.items() if key in cls._CONFIG_FIELDS
        }
        config = ControllerConfig(k_max=k_max, **overrides)
        model = ProfileCostModel.load(profile_path, out_of_range="error")
        controller = GoodputController(model, config)
        return cls(
            controller,
            mode=str(dynamic_depth.get("policy", "online_goodput")),
            static_k=int(dynamic_depth.get("static_k", k_max)),
        )

    def choose(
        self,
        batch_size: int,
        *,
        context_bucket: str = "default",
        graph_mode: str = "full",
        slo_ms: float | None = None,
    ) -> int:
        decision: Decision = self.controller.choose(
            batch_size,
            context_bucket=context_bucket,
            graph_mode=graph_mode,
            slo_ms=slo_ms,
        )
        active = decision.chosen_k if self.mode == "online_goodput" else self.static_k
        self.last_decision = BridgeDecision(active, decision.chosen_k, decision.reason)
        return active

    def observe(
        self,
        *,
        accepted_per_request: list[int],
        proposed_per_request: list[int],
    ) -> None:
        if not accepted_per_request:
            return
        self.controller.observe_variable(accepted_per_request, proposed_per_request)
