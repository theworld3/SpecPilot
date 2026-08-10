"""Convert real agent logs into AgentCache traces.

AgentCache traces are engine-agnostic JSONL (see ``agentcache/sim/workload.py``):
one line per multi-turn program, each turn carrying only *shapes* -- token
counts and tool latency -- never the prompt text. That is enough to drive the
cache simulation, and it means a trace can be shared without leaking data.

This script accepts two input shapes:

1. **LangSmith-style export** -- a JSONL where each line is a run tree
   (``{"id", "name", "inputs", "outputs", "inputs_tokens", "outputs_tokens",
   "start_time", "end_time", "tool_calls": [...]}``). We flatten each run into
   one program: every model->tool->model segment is a turn; token counts come
   from the run's reported usage (falling back to a chars/4 heuristic), tool
   latency from the gap between the model finishing and the next call.

2. **Intermediate schema** -- already in our shape, passed straight through.

With ``--demo`` it synthesises a tiny sample so the CLI is testable with no
external data.

Usage::

    python scripts/trace_from_langsmith.py --langsmith export.jsonl --out traces/from_langsmith.jsonl
    python scripts/trace_from_langsmith.py --demo --out traces/demo.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Make the package importable when run as `python scripts/<script>.py`
# without setting PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentcache.sim.workload import save_trace
from agentcache.types import AgentProgram, Turn


def _chars_to_tokens(text: str) -> int:
    # Cheap, dependency-free heuristic (~4 chars/token for English code/text).
    return max(1, len(text or "") // 4)


def _from_langsmith_line(obj: dict) -> AgentProgram:
    """Flatten one LangSmith run into one AgentProgram."""
    turns: list[Turn] = []
    steps = obj.get("tool_calls") or []
    # If the export already carries token counts, trust them; else estimate.
    for i, step in enumerate(steps):
        prompt = step.get("inputs") or obj.get("inputs") or ""
        output = step.get("outputs") or ""
        new_prompt_tokens = step.get("inputs_tokens") or _chars_to_tokens(str(prompt))
        output_tokens = step.get("outputs_tokens") or _chars_to_tokens(str(output))
        # Tool latency: gap between this step's start and the next step's start.
        lat = 0.0
        if i + 1 < len(steps) and step.get("start_time") and steps[i + 1].get("start_time"):
            lat = max(0.0, steps[i + 1]["start_time"] - step["start_time"])
        turns.append(Turn(new_prompt_tokens, output_tokens, tool_latency_s=lat))
    if not turns:
        # Single-turn program: model produced a final answer, no tool call.
        turns.append(Turn(_chars_to_tokens(str(obj.get("inputs", ""))),
                          _chars_to_tokens(str(obj.get("outputs", ""))),
                          tool_latency_s=0.0))
    return AgentProgram(program_id=obj.get("id", "prog"), arrival_t=0.0, turns=turns)


def convert_langsmith(path: str | Path) -> list[AgentProgram]:
    programs: list[AgentProgram] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                programs.append(_from_langsmith_line(json.loads(line)))
    return programs


def _demo() -> list[AgentProgram]:
    rng = random.Random(0)
    progs = []
    for i in range(4):
        n = rng.randint(3, 7)
        turns = [
            Turn(rng.randint(400, 1200), rng.randint(40, 200),
                 tool_latency_s=0.0 if j == n - 1 else rng.uniform(0.2, 1.5))
            for j in range(n)
        ]
        progs.append(AgentProgram(f"demo-{i}", arrival_t=0.0, turns=turns))
    return progs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--langsmith", help="path to a LangSmith JSONL export")
    src.add_argument("--demo", action="store_true", help="synthesise a sample")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    programs = _demo() if args.demo else convert_langsmith(args.langsmith)
    save_trace(programs, args.out)
    total = sum(len(p.turns) for p in programs)
    print(f"converted {len(programs)} programs ({total} turns) -> {args.out}")


if __name__ == "__main__":
    main()
