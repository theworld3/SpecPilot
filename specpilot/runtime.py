"""A small executable model of KV commit and batch-reorder invariants.

This is not a replacement for vLLM's KV manager. It gives unit tests a state
machine that catches stale-slot observations and rejected-prefix visibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestHandle:
    request_id: str
    slot: int
    generation: int


@dataclass
class _RequestState:
    handle: RequestHandle
    logical_tokens: int
    target_physical_tokens: int
    draft_physical_tokens: int
    pending_k: int = 0
    has_pending_round: bool = False


class KVLedger:
    """Track logical versus physically written speculative KV tokens."""

    def __init__(self) -> None:
        self._requests: dict[str, _RequestState] = {}
        self._slot_generation: dict[int, int] = {}

    def add_request(self, request_id: str, slot: int, prompt_tokens: int) -> RequestHandle:
        if request_id in self._requests:
            raise ValueError(f"request {request_id!r} already exists")
        if any(state.handle.slot == slot for state in self._requests.values()):
            raise ValueError(f"slot {slot} is occupied")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        generation = self._slot_generation.get(slot, 0) + 1
        self._slot_generation[slot] = generation
        handle = RequestHandle(request_id, slot, generation)
        self._requests[request_id] = _RequestState(
            handle=handle,
            logical_tokens=prompt_tokens,
            target_physical_tokens=prompt_tokens,
            draft_physical_tokens=prompt_tokens,
        )
        return handle

    def speculative_write(self, handle: RequestHandle, k: int) -> None:
        state = self._resolve(handle)
        if k < 0:
            raise ValueError("k must be non-negative")
        if state.has_pending_round:
            raise RuntimeError("request already has a pending speculative round")
        state.pending_k = k
        state.has_pending_round = True
        state.target_physical_tokens = max(
            state.target_physical_tokens, state.logical_tokens + k + 1
        )
        state.draft_physical_tokens = max(state.draft_physical_tokens, state.logical_tokens + k)

    def commit(
        self,
        handle: RequestHandle,
        *,
        accepted_drafts: int,
        emitted_tokens: int | None = None,
    ) -> None:
        state = self._resolve(handle)
        if not state.has_pending_round:
            raise RuntimeError("request has no pending speculative round")
        k = state.pending_k
        if not 0 <= accepted_drafts <= k:
            raise ValueError("accepted_drafts is outside the pending K")
        committed = accepted_drafts + 1 if emitted_tokens is None else emitted_tokens
        if not 0 <= committed <= accepted_drafts + 1:
            raise ValueError("EOS/stop truncation emitted an invalid token count")
        state.logical_tokens += committed
        # Physical tails may remain allocated, but neither logical length nor a
        # prefix hash can see them. The drafter's next visible base is the same
        # committed target prefix.
        state.target_physical_tokens = max(state.target_physical_tokens, state.logical_tokens)
        state.draft_physical_tokens = max(state.draft_physical_tokens, state.logical_tokens)
        state.pending_k = 0
        state.has_pending_round = False

    def remove_request(self, handle: RequestHandle) -> None:
        self._resolve(handle)
        del self._requests[handle.request_id]

    def reorder(self, request_to_slot: dict[str, int]) -> dict[str, RequestHandle]:
        if set(request_to_slot) != set(self._requests):
            raise ValueError("reorder mapping must contain every live request exactly once")
        if len(set(request_to_slot.values())) != len(request_to_slot):
            raise ValueError("reorder targets contain duplicate slots")
        handles: dict[str, RequestHandle] = {}
        for request_id, slot in request_to_slot.items():
            state = self._requests[request_id]
            generation = self._slot_generation.get(slot, 0) + 1
            self._slot_generation[slot] = generation
            state.handle = RequestHandle(request_id, slot, generation)
            handles[request_id] = state.handle
        return handles

    def logical_tokens(self, handle: RequestHandle) -> int:
        return self._resolve(handle).logical_tokens

    def physical_tokens(self, handle: RequestHandle) -> tuple[int, int]:
        state = self._resolve(handle)
        return state.target_physical_tokens, state.draft_physical_tokens

    def prefix_hash_length(self, handle: RequestHandle) -> int:
        return self._resolve(handle).logical_tokens

    def _resolve(self, handle: RequestHandle) -> _RequestState:
        state = self._requests.get(handle.request_id)
        if state is None:
            raise KeyError(f"unknown request {handle.request_id!r}")
        if state.handle != handle:
            raise RuntimeError("stale request handle after slot reuse or batch reorder")
        return state
