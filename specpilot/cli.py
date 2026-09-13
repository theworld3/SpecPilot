"""Command-line entry points for reproducible CPU workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specpilot.controller import ControllerConfig, GoodputController
from specpilot.cost_model import ProfileCostModel
from specpilot.schema import read_trace, write_trace
from specpilot.simulator import best_static, relative_change, replay_dynamic, sweep_static
from specpilot.synthetic import generate_mixed_trace, make_synthetic_profile


def _json_dump(payload: object, path: str | None = None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        Path(path).write_text(rendered, encoding="utf-8")


def command_demo(args: argparse.Namespace) -> None:
    profile = make_synthetic_profile(k_max=args.k_max)
    trace = generate_mixed_trace(steps=args.steps, k_max=args.k_max, seed=args.seed)
    static = sweep_static(trace, profile, args.k_max)
    oracle = max(static, key=lambda result: result.output_tps)
    controller = GoodputController(
        profile,
        ControllerConfig(
            k_max=args.k_max,
            warmup_steps=4,
            explore_interval=64,
        ),
    )
    dynamic = replay_dynamic(trace, controller)
    _json_dump(
        {
            "synthetic": True,
            "warning": "simulation output; not a GPU benchmark",
            "static": [result.summary_dict() for result in static],
            "best_static": oracle.summary_dict(),
            "specpilot": dynamic.summary_dict(),
            "specpilot_vs_best_static_output_tps_pct": relative_change(
                dynamic.output_tps, oracle.output_tps
            ),
        },
        args.output,
    )


def command_generate_profile(args: argparse.Namespace) -> None:
    make_synthetic_profile(k_max=args.k_max).save(args.output)
    print(f"wrote synthetic profile to {args.output}")


def command_generate_trace(args: argparse.Namespace) -> None:
    trace = generate_mixed_trace(steps=args.steps, k_max=args.k_max, seed=args.seed)
    write_trace(args.output, trace)
    print(f"wrote {len(trace)} synthetic trace steps to {args.output}")


def command_replay(args: argparse.Namespace) -> None:
    profile = ProfileCostModel.load(args.profile, out_of_range=args.out_of_range)
    trace = tuple(read_trace(args.trace))
    k_max = profile.max_k if args.k_max is None else args.k_max
    config = ControllerConfig(
        k_max=k_max,
        ewma_beta=args.beta,
        default_k=min(args.default_k, k_max),
        min_gain=args.min_gain,
        fallback_margin=args.fallback_margin,
        min_dwell_steps=args.min_dwell,
        explore_interval=args.explore_interval,
    )
    dynamic = replay_dynamic(trace, GoodputController(profile, config), slo_ms=args.slo_ms)
    oracle = best_static(trace, profile, k_max)
    _json_dump(
        {
            "profile_metadata": profile.metadata,
            "best_static": oracle.summary_dict(),
            "specpilot": dynamic.summary_dict(),
            "specpilot_vs_best_static_output_tps_pct": relative_change(
                dynamic.output_tps, oracle.output_tps
            ),
        },
        args.output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specpilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a deterministic CPU simulation")
    demo.add_argument("--steps", type=int, default=240)
    demo.add_argument("--k-max", type=int, default=5)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output")
    demo.set_defaults(func=command_demo)

    profile = subparsers.add_parser(
        "generate-profile", help="write a clearly marked synthetic cost profile"
    )
    profile.add_argument("--k-max", type=int, default=5)
    profile.add_argument("--output", required=True)
    profile.set_defaults(func=command_generate_profile)

    trace = subparsers.add_parser("generate-trace", help="write a synthetic mixed trace")
    trace.add_argument("--steps", type=int, default=240)
    trace.add_argument("--k-max", type=int, default=5)
    trace.add_argument("--seed", type=int, default=7)
    trace.add_argument("--output", required=True)
    trace.set_defaults(func=command_generate_trace)

    replay_parser = subparsers.add_parser("replay", help="replay a JSONL trace")
    replay_parser.add_argument("--profile", required=True)
    replay_parser.add_argument("--trace", required=True)
    replay_parser.add_argument("--output")
    replay_parser.add_argument("--k-max", type=int)
    replay_parser.add_argument("--beta", type=float, default=0.1)
    replay_parser.add_argument("--default-k", type=int, default=1)
    replay_parser.add_argument("--min-gain", type=float, default=0.03)
    replay_parser.add_argument("--fallback-margin", type=float, default=0.02)
    replay_parser.add_argument("--min-dwell", type=int, default=4)
    replay_parser.add_argument("--explore-interval", type=int, default=64)
    replay_parser.add_argument("--slo-ms", type=float)
    replay_parser.add_argument("--out-of-range", choices=("clamp", "error"), default="clamp")
    replay_parser.set_defaults(func=command_replay)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
