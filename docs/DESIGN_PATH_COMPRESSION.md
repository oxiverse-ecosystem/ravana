# Path compression: why it cannot fire, and what should replace it

**Status:** design note. No code change shipped with this document — the
finding is a logical impossibility in the current wiring, and the fix is a
design decision, not a bug fix.

**Finding owner:** audit of the continual-learning path, 2026-11-08.

---

## 1. The claim

`GraphPlasticity._compress_paths` reports `paths_compressed > 0` on a live run,
but it **never creates a shortcut edge**. Every unit it reports is a
no-op-ish strengthening of a shortcut that something else had already created.
The mechanism it is named for — *a repeatedly-traversed inference chain
collapses into a direct edge* — is unreachable.

## 2. The mechanism, precisely

Two functions are involved.

`ConceptGraph.record_path(source_id, target_id)` (`ravana_ml/graph.py:2750`)
increments a counter keyed on the **pair**.

`ConceptGraph.compress_paths(chains, min_chain_score)` (`ravana_ml/graph.py:2976`)
does one of two things per pair:

| state of the pair | action |
|---|---|
| no edge exists | **create** a shortcut edge (`shortcut=True`, `relation_type="inferred"`) |
| an edge exists and `edge.shortcut` | strengthen it slightly |
| an edge exists and not `shortcut` | **nothing** |

So creation requires the recorded pair to have **no** edge.

## 3. What the live path actually records

The only production caller is `chain_walker.py:1244`:

```python
for _a, _b, _e in _traversed:
    ...
    _gp.record_successful_path(self.graph, _a, _b)
```

`_traversed` is built in the same loop, and every `(_a, _b, _e)` triple comes
from iterating `graph.get_outgoing(...)` — i.e. **`_e` is an edge that already
exists between `_a` and `_b`.** The pair is recorded *because* the edge was
traversed.

That is the whole bug: the recorder only ever sees pairs that are already
edges, and the compressor only ever creates a shortcut for pairs that are not.

## 4. Measured

Live 14-turn run (`dim=64`, `seed=11`), then inspect the compressible set:

```
paths recorded: 1082
recorded pairs that ALREADY have an edge: 61/61   (100%)
compressible chains: 61
  would CREATE a shortcut: 0
  would STRENGTHEN an existing shortcut: 8
  existing non-shortcut edge -> SKIPPED: 53
```

The 8 "compressed" units are strengthenings of shortcuts that
`nn/rlm.py:1798` / `nn/rlm.py:2948` created during its own encode path — not
chains this mechanism collapsed. The other 53 are silently skipped.

Control experiment on a clean graph confirms both halves:

```
# A -> B -> C, no A -> C. Record the CHAIN ENDPOINTS (A, C):
compressible: [(0, 2, 0.45)]
compress: 1
A->C after: {'shortcut': True, 'w': 0.225, 'rel': 'inferred'}   <- works

# Same graph, record only the DIRECT edge actually traversed (A, B):
compressible: [(0, 1, 0.45)]
compress: 0
A->B shortcut? False                                            <- dead
```

So the graph primitive is correct and the recorder is wired to the wrong
quantity.

## 5. Why this matters (the neuroscience)

The intended mechanism is real and well-grounded:

* **Proceduralisation / knowledge compilation.** Anderson's ACT-R (1993):
  production rules begin as declarative knowledge retrieved and interpreted
  step by step; with practice the same transition is compiled into a single
  production. The behavioural signature is exactly what the code claims —
  a multi-step inference that used to require a walk becomes one step.
* **Hippocampal replay → neocortical consolidation.** Kumaran, Hassabis &
  McClelland (2016): replay does not merely strengthen, it *extracts* structure
  that was only implicit in the episodes. A shortcut edge is the graph form of
  that extraction.
* **Two timescales.** The fast store (hippocampal buffer) holds the episode;
  the slow store (ConceptGraph) should end up holding the compiled rule. The
  current code moves items from the first to the second but never compiles.

The gap is not that the theory is missing; it is that the recorder captures
*edges traversed* where the mechanism needs *chains completed*.

## 6. What the fix has to decide

Three defensible designs. They differ in what "a chain" is and in how much
they risk fabricating structure.

### Option A — record chain endpoints at the inference sites (minimal, faithful)

`infer_chain` already returns `(target_id, chain_score, path)` triples, and
`path` is the full node sequence. The two production call sites
(`engine_generation.py:1713` and `:1766`) walk those paths and discard them.
Recording `(path[0], path[-1])` when a chain of length ≥ 2 is actually used
would give the compressor exactly the pairs it can act on.

*Pros:* smallest change; the recorded pair is a real, used inference; no new
fabrication risk beyond what inference already asserts.
*Cons:* only fires on the graph-override path, which is gated off by default
(`graph_override=False`), so it would stay nearly inert.

### Option B — compress over the spread walk's transitive closure

`_spread_and_collect` already follows multi-hop paths. Record the
`(seed, collected)` pair when a collected node was reached in ≥ 2 hops, with
the hop count as the score. This fires on every turn, not just on the
override path.

*Pros:* always on; the "practice makes the route automatic" story is literally
true here — the walk is the practice.
*Cons:* spread reaches nodes through activation, not through a verified
relation chain, so a shortcut would assert a relation the graph never claimed.
This is the fabrication risk, and it is why the relation_type must stay
`"inferred"` and the confidence must stay low.

### Option C — leave it dead and delete the machinery

If neither A nor B is wanted, `record_path`, `get_compressible_paths`,
`compress_paths` and the `_compress_paths` stage should be removed rather than
left reporting a number that means nothing. A metric that reads 8 and means 0
is worse than an absent metric.

## 7. Recommendation

**Option A now, Option B behind a flag.** A is a two-line change at the two
`infer_chain` call sites and makes the existing primitive honest. B is the
version that actually delivers the ACT-R story on every turn, but it needs the
fabrication guard designed (relation_type `"inferred"`, low confidence,
`min_usage` high enough that a single walk never compiles) before it is turned
on. Until B is guarded, A alone converts a lying metric into a truthful one.

### 7.1 Status of Option A as implemented

Option A is implemented at both `infer_chain` call sites
(`engine_generation.py`: the `graph_override` branch and the `fallback_to_graph`
branch). **It is correct but not yet load-bearing**, and that should be stated
plainly rather than glossed:

* the `graph_override` branch is gated off by default
  (`graph_override: bool = False`), so it does not run in production;
* the `fallback_to_graph` branch fires only when the HRR store has *nothing*
  for a (head, verb) pair. On a live 14-turn run the HRR store answered every
  probe put to it (`cat/is_a -> carnivore`, `water/is_a -> liquid`), so this
  branch did not fire either.

A live re-run after the change is therefore unchanged, and that is the honest
result:

```
chains=60  would_CREATE=0  would_STRENGTHEN=8  skipped=52
paths_recorded: 1071  paths_compressed: 8
```

The metric is not lying any more — the 8 units are now genuinely shortcut
strengthenings, and the creation path is reachable in principle — but creation
still does not occur on a default run. Making it occur requires either turning
on `graph_override`, or Option B, or a third option not analysed here: record
chain endpoints from the spread walk when the walk's hop count is retained
(`_spread_and_collect` currently discards it). Closing that last gap is the
remaining work; it is a design choice about how much fabrication risk is
acceptable, not an implementation detail.

## 8. Verification that would accompany the fix

* A unit test asserting that after a used 2-hop chain, a shortcut edge exists
  between the chain's endpoints and `relation_type == "inferred"`.
* A unit test asserting that recording a **directly traversed** edge does
  *not* create a shortcut (the current behaviour, pinned so the distinction
  stays visible).
* The live-run probe from §4 re-run: `would CREATE a shortcut` must be > 0.

The first two are shipped (`tests/unit/test_path_compression.py`, 6 tests).
The third is **not** satisfied on a default run — see §7.1.
