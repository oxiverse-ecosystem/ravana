# CAR stale-ID fix — static verification and remaining write-path audit

**Scope:** read-only/static audit of the current `main` (HEAD `6a6fef6`) in worktree
`C:/Users/Likhith/Documents/projects/ravana_car_stale_audit`, branch
`audit/car-stale-id-write-paths`. No engine boot, no weight-store access, no
network, no production file, gate, threshold, rubric or fixture was changed.
Everything committed here lives under `scratch/car_stale_audit/`.

**Question:** does current `main` actually fix the sealed CAR k=2 stale-node-id
write failure (prior finding `scratch/car_adjk/FINDING.md`, commit `b53dd2a`), and
can any remaining write path still attach an edge to a dead node?

## 1. Verdict

| arm | gating | instrument | open holes | verdict | exit |
|---|---|---|---|---|---|
| current main `6a6fef6` | **5/5 PASS** | 1/1 PASS | 7 OPEN | **PASS** | 0 |
| pre-fix shadow `d882a009` (= `254fbc9^`) | 1/5 PASS (only F3, a separate fix) | 1/1 PASS | 7 OPEN | **FAIL** | 1 |

Deterministic report hashes (canonical sorted-key JSON, LF-normalised, SHA-256):

- current: `81de456d4c51a2d2e421433d77f559de580f2f8c313860b06489164f67614bfa`
- pre-fix shadow: `7691cf4279396f7e2c4ed93fe62ab7de24a870237187dfaca4f1c0517f269897`

The same current-mode hash was produced under `PYTHONHASHSEED=0` and `=1`,
and again from a **fresh detached checkout of this commit** with
`core.autocrlf=true` (line endings normalised before hashing, per the
frozen-provenance rule). No Python `hash()` of a string enters the report.

## 2. The fix, verified statically (gating checks)

| id | fixed behaviour | evidence | current | pre-fix |
|---|---|---|---|---|
| G1 | `WebToGraph._ensure_node` checks `get_node(existing)` at the write choke point, strips the stale id, mints a live node | `web_to_graph.py:151-163` | PASS: dead id 0 → fresh live id 1 | FAIL: returns dead id 0 |
| G2 | both engine shims resolve labels through live ids only | `web_learning.py:3112-3127`, `interface.py:70-85` | PASS: both return `None` for dead-only label | FAIL: both return dead id 0 |
| G3 | junk prune spares a node carrying an earned `curated` edge | `graph.py:1546-1566` (F4, broadened by `b1fc8c1` to all earned edge kinds) | PASS; mutation control (stamp removed) prunes the same node | FAIL: prunes the trusted node |
| G4 | label index synced after the junk-prune sweep | `engine_generation.py:1516-1531` (F1) | PASS | FAIL |
| G5 | trust-first partition before traversal limit | `explanation.py:261-289` (F3) | PASS | PASS (separate earlier fix) |

G3 needed a construction note: with the shipped `tau_low=1`, a node with an
incident edge has degree ≥ 1 and is skipped **before** the provenance guard.
The probe clears that node's adjacency entries (a restore/lag state) so it is
prune-eligible while its trusted edge record remains; only then does the guard
discriminate. The shipped `test_trusted_causal_node_survives_junk_prune` uses
degree-1 nodes and **passes on pre-fix code** — it never exercised F4. G3 here
does, with a provenance-removal mutation control.

## 3. Remaining write paths that can still write to dead IDs

All seven are reported as OPEN on current main. H0, H1, H2, H4, H5 and H6 drive
the **real production function** on a synthetic graph (no boot); H3 is
source-level because invoking those methods requires a live engine.

| id | path | mechanism | evidence |
|---|---|---|---|
| **H0 (high)** | `ChainWalkerMixin._create_causal_edge` | `cause_nids[0]` / `effect_nids[0]` from `_concept_keywords`, then `add_edge` — this is the conversational causal teaching channel, the same class of write CAR uses | `chain_walker.py:1394-1401,1425-1430`; probe wrote dead edge `(0,1)` |
| **H1** | `MemoryMixin._store_episodic` | `subj_nids[0]` / `assoc_nids[0]` → `add_edge(..., relation_type="episodic")` | `engine_memory.py:2594-2606`; dead edge `(0,1)` written |
| **H2** | `PersistenceMixin._weaken_edges_for_response` | correction fact `fact_nids[0]` / `val_nids[0]` → `add_edge` | `engine_persistence.py:588-596`; dead edge `(0,1)` written |
| **H3** | sleep episodic graduation and inline `process_turn` replay | `_a_ids[0]` / `_b_ids[0]` → `add_edge`, no liveness check | `engine_generation.py:1471-1481`, `engine.py:9585-9595` |
| **H4** | `SchemaConsistencyGate._node_id` | no `get_node` check of its own; depends on the host's `_all_labels` being live-only. Chat path uses the fixed shim, but plain `GraphEngine._all_labels` and `chain_walker._all_labels` are unsynced dicts | `intake.py:315-320`, `graph/engine.py:330,526`, `chain_walker.py:144`; resolver returned dead id 0 |
| **H5 (structural)** | `ConceptGraph.add_edge` | no live-endpoint invariant: dead ids are inserted into `edges`, `_outgoing`, `_incoming`. F5 (dangling-edge invariant) from the root-cause doc was never shipped | `graph.py:1694-1782`; `add_edge` accepted dead source 0 |
| **H6** | capacity eviction | `add_node` → `_prune_oldest` → `remove_node` drops a node with no engine-index sync; F1 only covers the junk-prune call site | `graph.py:1332-1336,2439-2441`; probe evicted id 0 with index still holding it |

**Cleared (not a hole):** `core/sleep.py` writers at `:192,:344,:400` check
endpoints with `get_node()` before writing; `WebToGraph.learn_text` resolves
through the fixed `_ensure_node` and the live-only intake shim.

Root cause is unchanged from the prior finding: `remove_node` cannot see the
engine's label index, and `add_edge` accepts dead endpoints. F1/F2/F4 close
three specific entrances; they are not a graph-wide invariant, and H0-H6 are
the unclosed entrances. The narrowest general repair would be a live-endpoint
check at `ConceptGraph.add_edge` plus index sync at *every* node-removal site
(or a single authoritative resolver used by all writers) — **not** proposed as
a production change here.

## 4. Reproduce

```
cd C:/Users/Likhith/Documents/projects/ravana_car_stale_audit
uv run python scratch/car_stale_audit/make_legacy_shadow.py
uv run python scratch/car_stale_audit/probe.py                       # exit 0, VERDICT PASS
uv run python scratch/car_stale_audit/probe.py --legacy-shadow scratch/car_stale_audit/legacy_shadow  # exit 1, VERDICT FAIL
```

Artifacts: `probe.py`, `make_legacy_shadow.py`, `results.json`,
`results_legacy.json`, `run_current_hs0.log`, `run_current_hs1.log`,
`run_legacy_hs0.log`. The shadow is generated from git into
`scratch/car_stale_audit/legacy_shadow/` and is not committed.

## 5. Limitations

- No engine was booted and no sealed CAR rerun was performed (TRA-K focused run
  is active), so this verifies the mechanism and write paths, not a fresh score.
- H3 is a source-pattern audit, not a runtime invocation; those writers are
  reachable only through a live engine.
- H0-H2 prove the writers *can* write a dangling edge on stale index state;
  they do not claim the live engine reaches that state on every run.
- The pre-fix arm shadows five modules from `254fbc9^` and lets every other
  submodule fall through to current `main`; it reproduces the historical
  resolver/prune behaviour, not a full old checkout.
