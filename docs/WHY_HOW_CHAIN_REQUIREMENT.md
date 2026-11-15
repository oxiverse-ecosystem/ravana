# Making the causal chain reachable for `why`/`how`

**Status: ROOT CAUSE CONFIRMED (two independent agents + my own measurement).
The fix is NOT shipped** — three attempts produced wrong output and were
reverted. This note records the requirement so the next attempt does not
repeat them.

## The confirmed root cause

`enumerate_chains` is called from exactly ONE place: inside
`_simulate_counterfactual` (`response_gen.py:6395`). That method has exactly two
callers (`:8328`, `:8486`), and **both are behind**

```python
if self._is_conditional_query(ctx.raw_input) and _HAS_HYPOTHETICAL:
```

A plain `why does X` / `how does X work` is not a conditional, so the assembler
is unreachable. Measured: **0 calls across a 17-turn probe** for every why/how
query. The graph is not the problem — 129 causal edges, 23 provenance-trusted,
34 forward-assertable 2–4 hop chains.

## What a correct fix must satisfy

The chain must read **cause → effect** regardless of walk direction. Three
attempts failed on exactly this:

| attempt | output |
|---|---|
| hoist + walk-order hops | `climate leads to heat, and heat leads to sun` (arrows reversed) |
| + reverse hop for backward | `light leads to photosynthesis, and photosynthesis leads to light` (round trip) |
| + reverse `labels` for backward | still a round trip |

### The requirement

A backward walk visits `effect ← cause`. To present cause → effect you must
**reverse the whole hop list**, not flip individual hops. Concretely:

1. Walk backward with `get_incoming`, collecting hops in walk order
   (`h_from = walk node`, `h_to = neighbour`).
2. Reverse the collected hop list.
3. Each hop's `from_label`/`to_label` must then describe the TRUE relation:
   after reversal, hop *i* runs `labels[i] -> labels[i+1]` in causal order.

Flipping hops individually (attempt 2) breaks the chain because a hop's
endpoints are shared with its neighbour. Reversing only the labels (attempt 3)
reverses a list whose hops were already walk-ordered, giving a round trip.

### A second, independent problem

`light` has BOTH an incoming edge from `sun` and an outgoing edge to
`photosynthesis`. A backward walk from `photosynthesis` reaches `light`, then
`get_incoming(light)` offers `sun` — but the graph also contains the reverse
edge, so a walk can ping-pong. The simple-path check must test **both**
endpoints of every hop (the shipped check tests only `to_label`).

## What is already shipped

`8e2f200` fixed the direction flag: backward walks now call `get_incoming`
instead of `get_outgoing`. Verified:

```
enumerate_chains([zz_rainfall], direction="backward")
  ['zz_rainfall', 'zz_condense', 'zz_evaporate']             assert adeq=1.000
  ['zz_rainfall', 'zz_condense', 'zz_evaporate', 'zz_water']  assert adeq=1.000
```

That is correct for a chain assembled from a **tree-shaped** subgraph. It is
not correct for the real graph, which has reciprocal edges.

## Suggested approach

Do not patch the walker again. Build the chain in **causal order from the
start**: walk backward collecting `(cause, effect)` pairs, then emit them
reversed. Alternatively, restrict the walk to a DAG by dropping reciprocal
edges. Either way, the acceptance test is:

```
why does climate change  ->  "sun leads to heat, and heat leads to climate"
why does it rain        ->  a real causal chain, or an honest hedge
```

and it must never emit a chain containing the same node twice.

## Also required

`gated_explanation_chain` is absent from `strategy_scores`
(`engine_generation.py:688-703`), so a correct chain scores the 0.3 default and
the quality monitor queues a good answer for "learning". Any shipped fix must
add it (suggested 0.75).
