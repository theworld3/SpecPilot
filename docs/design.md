# AgentCache — Design

This document explains *why* AgentCache is built the way it is. It is the
long-form companion to the README and the source comments; read it before
changing any of the cost-model math.

---

## 1. The problem with LRU for agents

Every production LLM engine evicts KV cache with LRU (vLLM prefix cache,
SGLang RadixAttention). LRU answers one question: *which block was used least
recently?* For a **chat** workload that is roughly the right question, because
"used recently" correlates well with "will be used again soon".

For an **agent** workload it is the wrong question, for a structural reason:

> Between turns, an agent program is *blocked on a tool call*. It is not
> finished. Its KV blocks are the **oldest in the pool** (nothing has touched
> them for seconds), yet the program is **guaranteed to return** with a prefix
> that is almost entirely identical to what it just used.

So LRU will happily evict exactly the contexts that are about to be needed.
The stronger baseline, `lru-offload`, demotes those blocks to a host-memory
tier instead of dropping them — which helps, but still orders eviction by
recency and still offloads *everything*, saturating the PCIe link with bytes
nobody will ask for again.

### What an agent program actually is

```python
AgentProgram {
    state: WAITING | RUNNING | TOOL_CALL | FINISHED
    turns: [Turn(new_prompt_tokens, output_tokens, tool_latency_s), ...]
}
```

The key property is `will_return`:

```python
@property
def will_return(self) -> bool:
    return self.state is not ProgramState.FINISHED
```

**This is deliberately not `has_next_turn`.** Once a program enters
`TOOL_CALL`, its `turn_index` has *already advanced* to the turn it is about to
run. A program parked on the tool call for its **final** turn still returns —
`has_next_turn` would be `False` there, and a policy using it would evict the
contexts that are one tool call away from being needed. Conflating the two is
a real bug, and it is worth keeping the comment in `types.py`.

---

## 2. The cost model

The right question is: *if I drop block `b`, what will it cost me later?*

```
value(b) = P_return(b) · ( T_recompute + T_queue − T_reload ) / bytes(b)
```

Evict in ascending `value`. Dividing by `bytes(b)` makes this a
GreedyDual-Size ordering (classic cost-benefit knapsack): a huge block must
justify its footprint rather than winning on raw savings alone.

### Term 1 + 2 — recompute vs reload (InferCept-style)

Dropping a block that gets re-referenced costs `T_recompute` (prefill to
rebuild it). But if we had demoted it to CPU/disk instead, we'd have paid
`T_reload` anyway — so only the *difference* matters. This is what InferCept
(ICML'24) captures.

### Term 3 — the queueing term (the novel part)

A dropped program loses its KV **and its batch slot**. To serve its next turn
it must re-run prefill *and re-acquire a slot in the running batch*. The slot
wait is modelled as an M/D/c-flavoured approximation:

```
T_queue = max(0, queue_len − free_slots) / max_batch · T_service
```

`T_service` is the EWMA of observed slot-hold time. **When the engine is
under-utilised (`free_slots > 0`) the term collapses to zero** — which is
exactly why offline profiling on an idle box never sees it, and why a policy
that ignores it looks fine until you put it under load.

Folding `T_queue` into the value means the policy gets *more conservative
exactly when contention makes recovery expensive* — it keeps the blocks whose
owners would otherwise pay the queue penalty.

### Estimating `P_return`

No learned model needed: the engine *knows* the program state.

| State | `P_return` |
|-------|-----------|
| `FINISHED` | 0.0 |
| `RUNNING` | 0.99 |
| `TOOL_CALL` | 0.95, decaying with how long the tool has run (timeout → 0.05) |
| `WAITING` | 0.90 |

The decay constant is the only tunable, and `benchmarks/run_ablation.py`
sweeps it. The intuition for the decay: a tool call that has been running a
long time is increasingly likely to be a timeout or an abandoned session.

### Should we even offload?

Offloading a block to host memory costs link time *now* and delays the reloads
that are on someone's critical path. So `should_offload` compares:

```
benefit = P_return · (T_recompute + T_queue − T_reload)
cost    = T_write · (1 + pcie_backlog / congestion_scale)
```

When the PCIe link is idle the cost term collapses and we offload freely. When
it is congested the policy becomes selective **automatically** — no threshold
tuning, no manual "offload rate limit" knob. This is the single biggest reason
AgentCache moves far fewer wasted bytes than an offload-everything baseline.

---

## 3. Policy structure

`AgentCachePolicy.select_victims` does three things LRU-offload cannot:

1. **Reads program state** — protects a `TOOL_CALL` context even though its
   blocks are the oldest in the pool; drops a `FINISHED` context immediately
   even though it was just touched.
2. **Prices the queue** — folds `T_queue` into the retention value.
3. **Chooses a tier per block** — demote to CPU if it earns the bandwidth,
   else drop.

### Program-granular eviction

Within a program, evicting *half* the blocks leaves a prefix that still can't
serve the next turn without a recompute of the missing tail — you pay the
eviction cost without collecting the space benefit. So AgentCache ranks
**whole programs** by their mean block value and evicts the cheapest program
first. *Within* a chosen program it still drops the **prefix** first
(`prefer_suffix`), because a returning turn attends to the suffix first and a
partial hit on the tail is better than none.

The `agentcache-block` ablation (block-granular) in the README shows this
single decision is worth ~17 points of hit rate.

---

## 4. Tiered store & the shared copy engine

The `TieredCacheManager` owns *metadata and decisions*; moving bytes is
delegated to a `StorageBackend`. The simulator injects a `NullBackend`
(metadata only); the real runtime injects a `PinnedHostBackend` (pinned host
pool + dedicated CUDA stream, FP8 pack before copy). Both exercise the
identical policy and accounting code.

A subtle but important modelling choice: the PCIe link is a **single shared
copy engine**. Every offload and every reload queues behind it via
`_enqueue_copy`. This is what makes the `pcie_backlog_s` signal in
`EvictionContext` meaningful — a policy that over-offloads pays for it at
reload time, which is exactly when a block is on the critical path.

The host tier is **not a landfill**. It has its own capacity and its own
eviction (driven by the same cost model). `lru-offload`'s "whatever arrived
first" host policy is strictly worse than answering that with the cost model.

---

## 5. Simulation methodology

The discrete-event simulator (`sim/`) exists for one reason: answer the
go/no-go question in minutes, not weeks. See `docs/validation.md` for what it
does and does not model, and how the numbers here are validated against a real
engine.

Three invariants the simulator is built to preserve:

- **Fairness.** Every policy gets a fresh deep copy of the trace and a fresh
  cache manager. The only variable is the decision rule.
- **Prefix contiguity.** `lookup` truncates at the first hole. A `GONE` block
  mid-sequence forces a recompute of everything after it, even if those blocks
  happen to still be resident. Not modelling this over-credits partial hits.
- **Honest baselines.** Speedups are always stated against `lru-offload`, the
  thing production actually ships, never against plain `lru`.

---

## 6. Limitations (read before you cite)

- The advantage over the strong baseline is **real but modest (~10–15% on
  TTFT p50)** and regime-dependent. With a large host tier and light load,
  offload-everything catches up on hit rate.
- The simulator uses a linear prefill-time model and holds a batch slot for a
  whole turn; both are conservative and affect all policies identically, so the
  *relative* comparison holds but absolute latencies are not wall-clock.
- `oracle` is a strong reference point, not a formal MIN (variable miss costs +
  a demotion tier break the optimality assumption). It is the honest measure of
  headroom, not a proof of optimality.
