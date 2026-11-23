# PDR live census — materializing real facts before intake

Scope: STATIC/code repair. No engine process was booted and no pytest was run
in this worktree (the parent runs the suite). All changes are in
`scripts/measure_learn5_pdr.py`; fake-store unit tests live in
`tests/unit/test_learn5_pdr.py`. No threshold, gate rule, item, sealing split,
control definition or decision rule was touched.

## 1. What was broken (recap of `PDR_LIVE_CONTROL_FAILURE.md`)

The wired-live run (commit `a52edad`, log `scratch/learn5_pdr_wired_live.log`)
failed 6 of 8 instrument controls and exited with `PDR_WIRED_EXIT=2`.

Root cause (EVIDENCE + INFERENCE): the census (`RealBackend._census`) matches a
graph edge only by its target node's label equalling the full object sentence
in `_roster`. The real gate resolves labels with
`SchemaConsistencyGate._node_id` → `ge._all_labels.get(_norm_label(label))`
(`intake.py:316-320`), and `_all_labels` on the chat engine is a **derived read
only view** of `eng._concept_keywords` (`interface.py:71-83`,
`web_learning.py:3113-3130`). `intake.py` never creates nodes, and
`KnowledgeIntake._write_edge` returns `False` when the object id is `None`
(`intake.py:705-716`). Full-sentence objects were therefore never materialized
as nodes/edges: every `query()` returned `facts=[]` and
`engine_observation_missing == len(roster)`. C3 and C8 passed because they
never read backend facts.

Secondary defect: the old `_real_write` computed `edge_kind`/`active` on a
local dict and passed only the `Provenance` to `intake()` — both keys were
discarded, so every poison kind would read `web_fact` and the sleeper would be
active at write.

## 2. The previous wired-live run produced NO score

The invalid run returns at `scripts/measure_learn5_pdr.py:1211-1213`
(original numbering) with `INSTRUMENT INVALID -- controls failed; NO SCORE
reported.` **before** the row dump (`:1224`) and the summary/`decide()` write
(`:1237-1267`). No endpoint, PASS/FAIL decision, or saturation flag exists for
that run. `scratch/learn5_pdr_summary.json` (mtime ~20 h earlier) is the stale
pre-rewire oracle-graded result and must not be cited as its score. The log's
`backend=live`, `api_calls` and `backend_error=None` prove only that the real
path ran, not that the graded observation came from it.

## 3. Exact changed functions (all in `scripts/measure_learn5_pdr.py`)

1. `RealBackend._ensure_engine_node(eng, label)` — NEW. Mints a live node in
   `eng.graph` via `graph.add_node(label=_norm(label))` and registers it in the
   authoritative `eng._concept_keywords[_norm(label)]` (fresh id first). It
   reuses an existing live id and skips a stale one. This is the write the gate
   cannot do: it makes `ge._all_labels` resolve the full-sentence object so
   `gate._node_id` and `_write_edge` succeed. (The equivalent node-creation
   path is `WebToGraph._ensure_node` / `_register_label`,
   `web_to_graph.py:109-196`.) Both subject and object are ensured before
   intake.

2. `RealBackend._find_engine_edge(eng, subject, obj)` — NEW. Resolves the real
   edge index-first (`_concept_keywords`) with a full-scan fallback matching
   source/target node labels, so it survives a cold reload where the
   full-sentence key has not rehydrated.

3. `RealBackend._stamp_engine_edge(edge, prov, edge_kind, source_type, url,
   active)` — NEW. Stamps the graded keys the census reads, **from the real
   `Provenance` passed to intake**: `edge_kind`, `source_url = prov.url`,
   `source_type`, and `active`. Real provenance fields are added with
   `setdefault` (never overwritten).

4. `RealBackend._real_write(...)` — REWRITTEN. Ensures object/subject nodes
   **before** `self._intake.intake(fact, provenance=prov, salience=1.0)`;
   captures the real `IntakeVerdict`; if `verdict.route == "episodic_buffer"`
   materializes the graded edge with `self._intake._write_edge(verdict, conf)`
   from **that same verdict/provenance** (no synthetic fact, no gate/route
   change); then finds the resulting real edge and stamps it. The old discarded
   local `meta` dict is gone.

5. `RealBackend.inject_poison(...)` — passes `active = (kind != "sleeper")`
   so the sleeper is dormant at write (previously `True` for every poison).

6. `RealBackend.activate_sleeper(...)` — after `super().activate_sleeper`
   (which issues the process_turn cue), flips `source_metadata["active"] =
   True` on the corresponding real edge(s) matched by (subject, detail). The
   label-based `_find_engine_edge` fallback keeps this working after reload.

7. `RealBackend.query()` / `_census()` — UNCHANGED. Still read only
   `self._engine.graph`; there is no `FakeBackend.facts` fallback.

Module docstring updated to record the materialization seam. `TH`,
`instrument_controls()` (all 8 controls), `decide()`, `gate_saturation()`, the
item bank and the sealed hash split are byte-identical.

## 4. Predicted control outcomes (measurable, no threshold moved)

After the repair, a one-item live probe at `p=5` should yield
`cued_obs["facts"]` containing the original plus the three poison origins with
`kind = curated / co_occurrence / web_fact / definition`, `source_url` from the
real `Provenance`, `trusted` True only for the original, and sleeper
`active=False` at write / `True` after cue. `engine_observation_missing` per
query should fall from `len(roster)` to ~0.

| Control | Depends on `obs["facts"]` | Predicted |
|---|---|---|
| C1 p=0 baseline | yes | FAIL → **ok** |
| C2 naive non-resistant | yes | FAIL → **ok** |
| C3 attribution | no | ok (unchanged) |
| C4 coverage | yes | FAIL → **ok** |
| C5 forged provenance + reject | yes | FAIL → **ok** |
| C6 sleeper dormant + reject | yes | FAIL → **ok** |
| C7 durable | yes | FAIL → **ok** |
| C8 warning/URL/span | no | ok (unchanged) |

`coverage_by_p[0]` becomes non-zero (answered is no longer empty) and the
degenerate 1.00/0.00 oracle endpoints disappear.

## 5. Falsifier

If, after materializing the edges, the census still returns zero facts or
C1/C2/C4/C7 still fail, the diagnosis is wrong: the drop is then in `eng.save()`
not persisting the new nodes/edges, or in `_concept_keywords` not rehydrating
the full-sentence key on cold reload. Check the on-disk graph/edge counts and
the reloaded index before changing anything else. `_find_engine_edge` and
`_census` already carry a full edge-scan fallback, so an empty census with
non-empty stored edges would specifically implicate node-label persistence.

Second falsifier for the rewiring itself: plant one wrong real edge (original
tagged `web_fact`/`web`, or a poison edge tagged `curated`/`seed`) and require
`retained`/`contaminated` to move. If they do not move, the score is still
measuring an oracle, not the engine. Both directions are covered by
`test_mutation_of_engine_provenance_moves_graded_observation`.

## 6. Tests added (fake store, no engine boot)

In `tests/unit/test_learn5_pdr.py`, using an injected `_StoreEngine`,
`_RecordingIntake` and the real `Fact`/`Provenance` dataclasses (never
`CognitiveChatEngine`):

- object node created and indexed before intake is called;
- engine edge carries `edge_kind` / `source_url` / `active` from the real
  provenance;
- a buffered verdict still materializes a graded edge from the same real
  verdict/provenance (sleeper stamped `definition`, not the intake default);
- sleeper flips `active` after the cue;
- census returns all authored facts with `engine_observation_missing == 0`;
- mutation control: changing one engine edge's provenance moves the graded
  observation in both directions;
- no synthetic fallback: with no real edges, `query()` emits no facts even when
  `FakeBackend.facts` holds a fully graded fact.

The existing `test_fake_path_does_not_boot_engine` and threshold-neutrality
test remain intact.

## 7. Unverified items (parent to check)

This worktree deliberately ran no engine and no pytest. Unverified by this
change: that `eng.save()` persists the minted nodes/edges and that
`_concept_keywords` rehydrates their labels on cold reload; that the real gate
routes the poison facts as expected on the live store; and the full 82-item
wired-live control outcome. Run the fake unit tests and a one-item live probe
before the full `--live` sweep.
