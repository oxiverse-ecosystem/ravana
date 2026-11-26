# FINDING — WFA stall after PR #549: causal-edge relation-vector init is the new hotspot

Scratch-only investigation on `invest/wfa-remaining-stall-profile` (worktree
`ravana_wfa_restall`, branched from fixed `main` at `c537d21`, the PR #549
activation-index fix). **No production file, gate or threshold was edited.**
No production PR is proposed by this investigation.

## Instrument validation

- `ps -ef | grep -E 'measure_learn5_wfa|uv run|python'` returned no engine
  process before the boot. E2C had exited with `WFA_E2C_EXIT=124` (log birth
  19:36:31, modify 20:36:32 — a 60-minute timeout).
- Reused the **same three turns and the same script** as PR #549:
  `scratch/profile_ground_query_wfa.py`, copied verbatim to
  `scratch/profile_remaining_stall.py` (only an added provenance print).
  Run with `PYTHONHASHSEED=0 RAVANA_OFFLINE=1 uv run python ...` and the
  harness's own `mfwf.install_frozen_patches()` (network and `save` blocked).
- Provenance printed at runtime: harness and engine both resolved from the
  worktree; boot `nodes=2167`, `glove_words=400001` — the **same store shape**
  as PR #549 (`nodes=2167`, `glove_words=400001`, old boot 2.63 s / new 3.39 s).
- All three prompt strings are byte-identical to PR #549 (lengths 220 / 2349 /
  2388). `scratch/profile_*` in the main repo was left untouched.

## cProfile, sorted by cumulative — fixed main vs PR #549

| turn | prompt chars | wall before | wall after | `_ground_query` before→after | `_activate_from_input` before→after |
|---|---:|---:|---:|---:|---:|
| inclusion_short | 220 | 0.623 s | 0.844 s | 1 / 0.0243 → 1 / 0.0196 | 0 → 0 |
| exclusion_long_context | 2349 | 0.407 s | 0.259 s | 1 / 0.0154 → 1 / 0.0083 | 0 → 0 |
| forced_choice_provenance | 2388 | **13.596 s** | **3.978 s** | 3 / 0.0247 → 3 / 0.0173 | **4 / 10.1019 → 4 / 0.1149** |

The activation-index fix removed **9.99 s** of the 13.60 s stalled turn
(88× faster `_activate_from_input`). It did not remove the stall: 3.978 s
remain, a 3.42× improvement, not a clearance. Total profiled calls fell from
19,542,301 to 2,504,008.

### New dominant hotspot (forced-choice turn, cumulative)

| cum s | % of 3.978 s | ncalls | function |
|---:|---:|---:|---|
| 2.227 | 56 % | 1 | `chain_walker.py:1250 _extract_and_store_causal_relations` |
| 1.894 | 48 % | **135,291** | `chain_walker.py:1387 _create_causal_edge` |
| 1.768 | 44 % | 4,352 | `ravana_ml/graph.py:1694 add_edge` |
| 1.666 | 42 % | 4,352 | `ravana_ml/graph.py:193 ConceptEdge.__init__` |
| **1.618** | **41 %** | **4,352** | `ravana_ml/graph.py:381 ConceptEdge._init_relation_vector` |
| 0.638 | 16 % | 4,352 | `numpy _ufunc_config.inner` (inside the above) |
| 0.626 | 16 % | 3 | `user_model.py:1718 mine_personal_facts` |
| 0.518 | 13 % | 2 | `chain_walker.py:295 _auto_expand_concepts` |
| 0.230 | 6 % | 4,968 | `chain_walker.py:595 _word_causal_score` |

PR #549 already flagged `_extract_and_store_causal_relations` (1.738 s,
135,291 `_create_causal_edge` calls) as a "separate issue". With the
activation loop fixed it is now the **dominant** cost. The single largest leaf
is `_init_relation_vector`: every one of the 4,352 new `ConceptEdge`s rebuilds
a `RandomState(43)`, draws `randn(16)`, and normalises — even though the
function is pure in `(relation_type, dim)` and all 4,352 calls use
`("causal", 16)`. `np.inner`/`randn` accounts for 0.638 s of its 1.618 s.

The 135,291 `_create_causal_edge` calls come from URL-derived tokens in the
eight "trusted sources" URLs: a causal connective score ≥0.30 splits the
sentence and the content-word cross-product is attempted. Most calls return
early, but 4,352 actually insert edges, taking the graph from 2,167 to 2,276
nodes and leaving 22,658 edges after three turns.

## Scratch A/B — memoise the deterministic relation-vector template

`scratch/ab_relation_vector.py` monkeypatches the **unmodified** production
`ConceptEdge._init_relation_vector` at runtime with a scratch function that
builds one template per `(relation_type, dim)` and returns `template.copy()`
on every call. A fresh copy matters: Hebbian code mutates
`edge.relation_vector` in place (e.g. `graph.py:2294`), so sharing the array
would change later state. No production file is touched; the patch is confined
to the probe process.

- **Functional equivalence:** 104 pairs (13 relation labels × 8 dims) — exact
  `np.array_equal` plus dtype/shape/writable/distinct-object checks, 0 failures.
  Mutating one returned copy does not affect its sibling.
- **Instrument check:** the candidate was called **4,352** times during the
  three-turn arm, exactly matching the profile's `_init_relation_vector` count.
- **Microbenchmark** (4,352 `("causal", 16)` calls, median of 3): 1.208 s →
  0.0047 s, **258×**.
- **End-to-end A/B** (two engines, same store, same seeds, one process;
  baseline arm first, candidate second):

| turn | baseline wall | candidate wall |
|---|---:|---:|
| inclusion_short | 0.200 s | 0.055 s |
| exclusion_long_context | 0.223 s | 0.173 s |
| forced_choice_provenance | 2.353 s | **1.021 s** (2.3×) |

  **Identical outputs:** responses (SHA-256 equal), `_last_grounding_method`,
  2,276 nodes, 22,658 edges, 24,420 belief assertions, 8 activations, and an
  SHA-256 of every edge's `relation_vector` bytes plus its weight, confidence,
  relation type, edge type, predicate id and non-time provenance metadata.
  Diff count **0**.

`scratch/accumulation_probe.py` (12 distinct forced-choice items per arm)
confirmed identical node/edge counts at every turn. First turn 2.51 s →
1.13 s; steady state ~1 s/turn in both arms; edges grew 22,658 → 22,878. The
hotspot is a large **per-turn** cost, not a compounding leak in this sample.

## Recommendation

1. **Safe, output-identical optimisation to land:** cache
   `ConceptEdge._init_relation_vector` by `(relation_type, dim)` and return a
   fresh `copy()` per edge. This removes ~1.6 s of the 3.98 s URL-heavy turn
   and cannot change scores because the emitted vectors are byte-identical.
   Keep it behind the normal code-review path; this worktree makes no
   production edit.
2. **Separate follow-up, not part of this fix:** the 135,291
   `_create_causal_edge` calls / 4,352 URL-token edges are the real amplifier.
   Filtering URL tokens before causal extraction would change the graph and
   therefore WFA outputs; it needs its own A/B with the WFA fronts and gates
   unchanged, not a silent scope creep into this perf patch.
3. Do not touch `_ground_query`, WFA gates, thresholds or scoring.

## Limitations

- The three-turn profile does not reproduce the exact parent E2C turn that
  was in flight at the timeout; the `exclusion_long_context` turn in this
  sample never entered the causal-extraction path (0.259 s). The hotspot, its
  call counts and the behaviour-preserving fix are measured directly.
- E2C emitted 1,940 `ground_query` lines in 60 minutes (~1.85 s per printed
  line) versus ~1 s for a whole URL-heavy turn in this focused probe and the
  harness's own double `process_turn` audit path; the remaining harness-level
  cost is outside these three turns and is not attributed here.
- No WFA front score or gate was re-measured: outputs are byte-identical on
  the measured turns, so no score change is expected, but that is an inference
  from equality, not a new front run.
