"""Core data types shared across AgentCache.

This module is intentionally dependency-free (stdlib only) so that the
simulation and policy layers can be imported and tested on any machine,
including CI runners without a GPU or PyTorch installation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "Tier",
    "ProgramState",
    "ModelSpec",
    "BlockMeta",
    "Turn",
    "AgentProgram",
    "EvictionContext",
    "EvictionDecision",
]


class Tier(enum.IntEnum):
    """Storage tier of a KV block, ordered by access latency (fastest first)."""

    GPU = 0
    CPU = 1
    DISK = 2
    GONE = 3
    """Not materialised anywhere: the only way back is a full recompute."""

    @property
    def is_resident(self) -> bool:
        return self is not Tier.GONE


class ProgramState(enum.IntEnum):
    """Lifecycle state of an agent program (a multi-turn conversation)."""

    WAITING = 0
    """Queued for admission into the running batch."""
    RUNNING = 1
    """Occupying a slot in the running batch (prefill or decode)."""
    TOOL_CALL = 2
    """Turn finished; the external tool is executing. The program *will* return."""
    FINISHED = 3


@dataclass(frozen=True)
class ModelSpec:
    """Minimal model description needed to size the KV cache.

    The default is Llama-3.1-8B with GQA (8 KV heads), which gives
    ``128 KiB`` of KV per token in FP16 -- the number the whole project
    economics are built on.
    """

    name: str = "llama-3.1-8b"
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2

    @property
    def bytes_per_token(self) -> int:
        """K and V, all layers, one token."""
        return self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes * 2

    def bytes_for(self, num_tokens: int) -> int:
        return num_tokens * self.bytes_per_token


@dataclass
class BlockMeta:
    """Metadata for one KV cache block.

    A block is the unit of allocation, eviction and transfer. We never model
    the tensor payload in the simulator -- only its size, location and the
    access pattern that drives policy decisions.
    """

    block_id: int
    program_id: str
    seq_index: int
    """Position of this block inside the program's token sequence."""
    num_tokens: int
    tier: Tier = Tier.GPU
    last_access_t: float = 0.0
    created_t: float = 0.0
    pinned: bool = False
    """Pinned blocks are never evicted (e.g. currently being attended to)."""
    prefix_hash: Optional[str] = None
    """Content hash of ``[0, seq_index]``, used for cross-program prefix sharing."""

    def touch(self, now: float) -> None:
        self.last_access_t = now


@dataclass
class Turn:
    """One agent turn: model generates, then an external tool runs."""

    new_prompt_tokens: int
    """Tokens appended before this turn (tool output from the previous turn)."""
    output_tokens: int
    """Tokens the model generates this turn."""
    tool_latency_s: float = 0.0
    """Wall-clock time the external tool takes after this turn. 0 = last turn."""


@dataclass
class AgentProgram:
    """A multi-turn agent trajectory -- the scheduling unit of AgentCache.

    The key property that motivates this project: between turns the program
    is *not* finished. It is blocked on a tool call for ``tool_latency_s``
    seconds and will come back with a prefix that is almost entirely
    identical to what it just used.
    """

    program_id: str
    arrival_t: float
    turns: List[Turn]

    state: ProgramState = ProgramState.WAITING
    turn_index: int = 0
    prompt_tokens_so_far: int = 0
    """Length of the KV context the program currently owns."""

    # --- populated by the simulator ---
    admitted_t: Optional[float] = None
    finished_t: Optional[float] = None
    ttft_per_turn: List[float] = field(default_factory=list)
    recomputed_tokens: int = 0
    reloaded_bytes: int = 0

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def current_turn(self) -> Turn:
        return self.turns[self.turn_index]

    @property
    def has_next_turn(self) -> bool:
        return self.turn_index + 1 < self.num_turns

    @property
    def will_return(self) -> bool:
        """Whether this program's KV will be referenced again.

        Careful: this is *not* ``has_next_turn``. Once a program enters
        ``TOOL_CALL`` its ``turn_index`` has already advanced to the turn it
        is about to run, so a program parked on the tool call for its final
        turn still returns. Conflating the two makes a policy evict exactly
        the contexts that are one tool call away from being needed -- a bug
        worth keeping this comment for.
        """
        return self.state is not ProgramState.FINISHED


@dataclass
class EvictionContext:
    """Runtime signals handed to a policy at eviction time.

    ``queue_len`` and ``running_batch_size`` are what make the AgentCache
    cost model different from a pure hit-rate heuristic: they let the policy
    reason about *scheduling* consequences, not just recompute FLOPs.
    """

    now: float
    bytes_needed: int
    queue_len: int
    running_batch_size: int
    max_batch_size: int
    gpu_bytes_used: int
    gpu_bytes_capacity: int
    pcie_backlog_s: float = 0.0
    """How long the copy engine is already booked for.

    Host offload is not free: the PCIe link is a single shared resource, and
    every speculative demotion delays the reload of a context that is
    actually needed now. A policy that ignores this happily saturates the
    link with blocks nobody will ask for again."""


@dataclass
class EvictionDecision:
    """Where a victim block should go."""

    block_id: int
    target_tier: Tier
    score: float = 0.0
