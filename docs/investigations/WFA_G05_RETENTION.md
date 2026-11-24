# WFA g05 provenance retention — root cause and fix

**Scope:** one sealed discredited item (`g05`) out of 15 fresh-engine condition
calls in the internal-source WFA harness (`scripts/measure_learn5_wfa.py`).
This change is confined to the harness teaching path. No engine file, no gate,
no threshold, no item bank, no expected decision and no grading rule was
touched; no item metadata reaches the staged context.

## Setup of the confirmed failure (inherited)

From `WFA_ISOLATED_CONFIRM.md` (`deleg_50464230`): with a **fresh engine per
condition**, 14/15 calls were correct. Only discredited `g05` failed: its
taught provenance edge (origin `wfa-internal://discredited/f889f80bc0d6`,
`edge_kind=curated`, `confidence=0.40725`) had 1 matching edge after teaching
and **0 after the first `process_turn`** (the prior-commitment/ENDORSE turn).
`apply_source_invalidation` then tagged 0 edges, the staged context fell to
`trust_class=unknown`, and the engine correctly ABSTAINed. Control `g64`
(`confidence=0.7958`) kept its edge and WITHDRAWn. Teach-only diagnostics
showed g05 invalidation *would* tag 1 edge without that turn.

All probes below ran only against the **copied** store
`C:/Users/Likhith/Documents/projects/ravana_wfa_confirm/weights`; the parent's
`.../ravana/weights` was never opened by a live engine.

## EVIDENCE — exact code path

Instrumented probe `scratch/wfa_g05_removal_trace.py` (teaches g05, wraps
`ConceptGraph.remove_edge`, `prune_low_quality_edges`, `prune_edges`,
`prune_low_degree_junk_nodes`, `ConceptEngine._sanitize_graph`,
`_sleep_consolidate`, `eng.__setattr__` for `graph`, and `graph.nodes.pop`,
then runs the first turn) produced:

- `IDENTITY start/end: g is eng.graph = True` — the engine **did not** replace
  its graph during the turn, and **no** removal/prune/sanitize method was
  called (`scratch/wfa_g05_removal_trace.log`).
- `target edges still present: 1` — the exact taught edge object was **still**
  in `eng.graph.edges` after the turn (same object id).
- Yet `ROWS after turn: []`. The saved edge's `source_metadata` had changed
  from `source='wfa-internal://discredited/f889f80bc0d6'`,
  `source_type='seed'`, `url=<sid>`, `source_status=''`,
  `explicit_assertion=True` to
  `source='conversation'`, `source_type='user'`, `url=''`,
  `extractor='conversational-assertion'`, `operations=['taught-in-conversation']`,
  `revisions=1` — i.e. **provenance was overwritten in place, not pruned**.

The overwrite path, with file:line:

1. `ravana/src/ravana/chat/engine.py:10837` — `_process_turn_impl` calls
   `self._consolidate_asserted_causal_facts(user_input)` on the first turn.
2. `ravana/src/ravana/chat/chain_walker.py:878-930` —
   `_consolidate_asserted_causal_facts` extracts asserted causal propositions
   from the claim text and builds
   `Provenance(source="conversation", source_type="user", url="",
   extractor="conversational-assertion", operations=("taught-in-conversation",))`
   (`chain_walker.py:912-915`), then calls `intake.intake(fact, prov, ...)`.
3. `ravana/src/ravana/learn/intake.py:705-746` — `_write_edge` finds an
   **existing** edge on the same node pair and at `:729` does
   `meta.update(v.provenance.as_metadata())`, followed by `:732`
   `meta["edge_kind"] = _EDGE_KIND_FOR_SOURCE.get(...)`. This replaces the
   WFA origin with `"conversation"`.
4. `scripts/measure_learn5_wfa.py:1821` — `graph_provenance_for_claim` filters
   edges by `origin == source_id`; after the overwrite no edge matches, so
   `context_from_graph_provenance` stages `UNKNOWN` and the evaluator emits
   `R_UNKNOWN_PROVENANCE` / ABSTAIN.

Two controlled probes (`scratch/wfa_g05_root_cause_controls.py`, log
`..._controls.log`) isolate the cause:

| mode | item | rows after turn | invalidated | staged trust |
|---|---|---|---|---|
| normal (before fix) | g05 | 0 | 0 | unknown |
| normal (before fix) | g64 | 1 | 1 | discredited |
| `_consolidate_asserted_causal_facts` no-op | g05 | 1 | 1 | discredited |
| harness synthetic-label teaching | g05 | 1 | 1 | discredited |

Disabling that single function restores g05, so it is the drop point.
Forcing the harness's short-label explicit assertion also restores it.

**The `0.40725` vs `0.7958` confidence difference is NOT the mechanism.**
EVIDENCE: no prune/removal path fired on the g05 edge (trace above), and
`ConceptGraph.prune_low_quality_edges` (`ravana_ml/src/ravana_ml/graph.py:
1867-1902`) skips non-`semantic` relations and verified kinds while
`prune_low_degree_junk_nodes` (`:1504-1567`) explicitly protects edges with
`edge_kind in {curated, web_fact, definition, curriculum}`. Both g05 and g64
edges are `relation_type=causal`, `edge_kind=curated`.

## INFERENCE

- Why g05's *natural-language* edge collides and g64's does not: the assertion
  extractor for the g05 sentence resolves to the same node pair the harness's
  `WebToGraph.learn_text` wrote (`mirrors -> glow`), whereas g64's proposition
  resolves elsewhere. This is consistent with the control table but the exact
  extracted triples were not printed, so it remains an inference.
- The fix does not depend on that inference: it removes the *possibility* of a
  node-pair collision by moving the authoritative provenance off the
  natural-language nodes entirely.

## FIX (harness teaching path only)

`scripts/measure_learn5_wfa.py`:

- New `_write_asserted_wfa_fact` (`scripts/measure_learn5_wfa.py:1931`): writes
  the explicitly asserted fact on the short deterministic claim-id node pair
  `wfa_<sha1(claim)>[:12]` / `...|target` (`wfa_claim_token`), with
  `source_metadata` carrying `source`/`url=<source_id>`, `edge_kind=curated`,
  `source_type=seed`, `source_status=''`, `explicit_assertion=True`.
- `teach_claim_into_graph` (`:1992-2024`) still calls
  `WebToGraph.learn_text` for the real ingest path, but now **always** writes
  the synthetic assertion as the authoritative provenance anchor and returns
  the combined count (`:2024`).

Why this is minimal and non-tautological:

- The synthetic ids are a pure hash of the **claim text**; they never encode a
  condition, source status, item answer or expected decision. No item metadata
  reaches `graph_provenance_for_claim` / the staged context.
- The ids contain no spaces, so the boot sanitizer's >=5-word question-phrase
  prune cannot target them; they cannot collide with the natural-language
  nodes that `_consolidate_asserted_causal_facts` mints, so their provenance is
  not overwritten on the first turn.
- No threshold, gate, item, prompt, expected decision or grading rule changed.
  `evaluate_source` ignores `reliability`/`familiarity`
  (`source_evaluation.py:212-213`), so raising the anchor's confidence to 0.9
  cannot itself move a decision.

## VERIFIED PREDICTIONS

Focused fresh-engine confirmation using the worktree harness and the copied
store (`scratch/wfa_g05_fix_confirm.py`, `..._confirm.log`, `..._confirm.jsonl`):

- **15/15 correct**: credible 5/5, discredited 5/5, unknown 5/5.
- g05 discredited: `prior=ENDORSE`, `teached=2`, `invalidated=1`,
  `decision=WITHDRAW`, `token=W`, `rule=R_SOURCE_INVALIDATED`,
  `staged_trust_class=discredited`, fingerprint `b24931267b...` — exactly the
  known-good teach-only fingerprint from `wfa_g05_invalidation_diag.log`.
- Unit tests: `88 passed` for
  `tests/unit/test_learn5_wfa.py` together with two unrelated modules
  (`test_active_epistemology.py`, `test_c_lite_web_to_graph.py`) in one pytest
  process (no import-time global patch leak). New tests cover: low-confidence
  asserted edge surviving simulated pruning, provenance still found, g05-like
  claim discoverable after a simulated turn, and an edge-kind mutation
  flipping the staged decision ENDORSE -> ABSTAIN.

**Not re-run:** the full 120-condition (>=20/pool) suite. The prediction is
15/15 on the focused set only; a full run is still needed for the
pre-registered sample-size criterion.

## FALSIFIER — what would disprove this diagnosis

If the g05 edge still disappears after a turn in a fresh-engine run:

1. Check whether the turn **re-seeds from the persisted weight store** and
   whether the write was **in-memory only**. In the traced run
   `eng.graph is g` remained True before and after the turn and no
   `remove_edge`/prune fired, so re-seeding is ruled out *for that run*; a new
   failure could be a different boot path. Instrument `__setattr__('graph')`
   and `_load()` as in `scratch/wfa_g05_removal_trace.py` before theorising.
2. Dump the taught edge's `source_metadata` immediately before and after the
   turn. If `source` flips to `conversation`, the collision path above is still
   live and the synthetic node pair must be checked for a label merge.
3. **Do NOT tune thresholds.** A confidence floor change is not an acceptable
   repair: the failure is a provenance-write collision, and changing a gate or
   floor would only hide it.
