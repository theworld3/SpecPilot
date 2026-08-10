"""Agent trace format, synthetic generators and JSONL I/O.

An *agent trace* is a list of multi-turn programs. One JSONL line per
program::

    {"program_id": "react-0",
     "arrival_t": 0.0,
     "workload": "react-search",
     "turns": [{"new_prompt_tokens": 1800, "output_tokens": 96,
                "tool_latency_s": 1.4}, ...]}

The format is deliberately engine-agnostic and contains no text, only
shapes -- so traces can be shared without leaking prompts, and replayed
against vLLM, SGLang or this simulator alike. ``benchmarks/traces/`` ships
three synthetic profiles; ``scripts/trace_from_langsmith.py`` converts real
agent logs into the same schema.

Why synthetic traces are acceptable here
----------------------------------------
The claim under test is about *cache reference patterns*, which are fully
determined by turn count, context growth and tool latency. Those three are
directly measurable from any real agent framework and are what the
generators parameterise. Token *content* is irrelevant to eviction
decisions. What synthetic data cannot tell you is accuracy impact -- that
is measured separately on real models in ``benchmarks/run_accuracy.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..types import AgentProgram, Turn

__all__ = [
    "WorkloadProfile",
    "PROFILES",
    "generate_programs",
    "save_trace",
    "load_trace",
]


@dataclass(frozen=True)
class WorkloadProfile:
    """Statistical shape of one class of agent workload."""

    name: str
    description: str
    turns_range: tuple[int, int]
    first_prompt_range: tuple[int, int]
    """System prompt + task description, tokens."""
    tool_output_range: tuple[int, int]
    """Tokens appended by each tool result."""
    output_range: tuple[int, int]
    """Model-generated tokens per turn."""
    tool_latency_range: tuple[float, float]

    def sample(self, rng: random.Random, program_id: str, arrival_t: float) -> AgentProgram:
        num_turns = rng.randint(*self.turns_range)
        turns: List[Turn] = []
        for i in range(num_turns):
            if i == 0:
                new_tokens = rng.randint(*self.first_prompt_range)
            else:
                new_tokens = rng.randint(*self.tool_output_range)
            is_last = i == num_turns - 1
            turns.append(
                Turn(
                    new_prompt_tokens=new_tokens,
                    output_tokens=rng.randint(*self.output_range),
                    tool_latency_s=0.0
                    if is_last
                    else rng.uniform(*self.tool_latency_range),
                )
            )
        return AgentProgram(program_id=program_id, arrival_t=arrival_t, turns=turns)


PROFILES: Dict[str, WorkloadProfile] = {
    "react-search": WorkloadProfile(
        name="react-search",
        description=(
            "ReAct-style web research agent. Many short turns, a search "
            "result appended each turn, tool latency dominated by network."
        ),
        turns_range=(6, 14),
        first_prompt_range=(1200, 2600),
        tool_output_range=(500, 1600),
        output_range=(48, 160),
        tool_latency_range=(0.6, 3.0),
    ),
    "code-agent": WorkloadProfile(
        name="code-agent",
        description=(
            "Repo-scale coding agent. Large initial context (files + tree), "
            "moderate turns, fast local tools (tests, grep, build). Output is "
            "short because most turns emit a tool call, not prose -- which is "
            "exactly why prefill/reload dominates here and not decode."
        ),
        turns_range=(4, 8),
        first_prompt_range=(4000, 12000),
        tool_output_range=(400, 1500),
        output_range=(60, 200),
        tool_latency_range=(0.15, 1.2),
    ),
    "chat": WorkloadProfile(
        name="chat",
        description=(
            "Plain multi-turn chat, no tool calls. Control group: AgentCache "
            "must not regress this."
        ),
        turns_range=(2, 6),
        first_prompt_range=(300, 1200),
        tool_output_range=(80, 300),
        output_range=(100, 400),
        tool_latency_range=(0.0, 0.05),
    ),
}


def generate_programs(
    profile: str | WorkloadProfile,
    *,
    num_programs: int = 64,
    arrival_rate_qps: float = 2.0,
    seed: int = 0,
    id_prefix: Optional[str] = None,
) -> List[AgentProgram]:
    """Sample programs with Poisson arrivals.

    ``arrival_rate_qps`` is the knob that creates contention. Sweeping it is
    how the benchmark shows that the advantage of AgentCache *grows* with
    load -- the regime where the queue-delay term matters.
    """
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    rng = random.Random(seed)
    prefix = id_prefix or prof.name

    programs: List[AgentProgram] = []
    t = 0.0
    for i in range(num_programs):
        t += rng.expovariate(arrival_rate_qps) if arrival_rate_qps > 0 else 0.0
        programs.append(prof.sample(rng, f"{prefix}-{i}", round(t, 6)))
    return programs


def mixed_workload(
    weights: Dict[str, float],
    *,
    num_programs: int = 64,
    arrival_rate_qps: float = 2.0,
    seed: int = 0,
) -> List[AgentProgram]:
    """Interleave several profiles on a single shared arrival process."""
    rng = random.Random(seed)
    names = list(weights)
    ws = [weights[n] for n in names]

    programs: List[AgentProgram] = []
    t = 0.0
    for i in range(num_programs):
        t += rng.expovariate(arrival_rate_qps) if arrival_rate_qps > 0 else 0.0
        name = rng.choices(names, weights=ws, k=1)[0]
        programs.append(PROFILES[name].sample(rng, f"{name}-{i}", round(t, 6)))
    return programs


# ----------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------
def save_trace(programs: Sequence[AgentProgram], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in programs:
            fh.write(
                json.dumps(
                    {
                        "program_id": p.program_id,
                        "arrival_t": p.arrival_t,
                        "turns": [
                            {
                                "new_prompt_tokens": t.new_prompt_tokens,
                                "output_tokens": t.output_tokens,
                                "tool_latency_s": round(t.tool_latency_s, 4),
                            }
                            for t in p.turns
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_trace(path: str | Path) -> List[AgentProgram]:
    programs: List[AgentProgram] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            programs.append(
                AgentProgram(
                    program_id=obj["program_id"],
                    arrival_t=float(obj["arrival_t"]),
                    turns=[Turn(**t) for t in obj["turns"]],
                )
            )
    return programs


def clone_programs(programs: Iterable[AgentProgram]) -> List[AgentProgram]:
    """Fresh copies so the same trace can be replayed under several policies.

    Programs carry mutable run state (``turn_index``, counters), so reusing
    the objects across policies would silently compare a warm run against a
    cold one -- an easy way to publish a wrong speedup.
    """
    return [
        AgentProgram(
            program_id=p.program_id,
            arrival_t=p.arrival_t,
            turns=[Turn(t.new_prompt_tokens, t.output_tokens, t.tool_latency_s) for t in p.turns],
        )
        for p in programs
    ]
