from __future__ import annotations

import pytest

from specpilot.cost_model import CostEstimate, ProfileCostModel


@pytest.fixture
def simple_profile() -> ProfileCostModel:
    rows = []
    for batch in (1, 8):
        rows.extend(
            (
                CostEstimate(batch, 0, 0.0, 1.0 + 0.1 * batch),
                CostEstimate(batch, 1, 0.2, 1.2 + 0.1 * batch),
                CostEstimate(batch, 2, 0.4, 1.5 + 0.1 * batch),
            )
        )
    return ProfileCostModel(rows, metadata={"fixture": True})
