"""Scan the deployable static-K baselines on one replay trace."""

from __future__ import annotations

import argparse
import json

from specpilot.cost_model import ProfileCostModel
from specpilot.schema import read_trace
from specpilot.simulator import sweep_static


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--k-max", type=int)
    args = parser.parse_args()
    profile = ProfileCostModel.load(args.profile)
    results = sweep_static(tuple(read_trace(args.trace)), profile, args.k_max)
    print(json.dumps([result.summary_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
