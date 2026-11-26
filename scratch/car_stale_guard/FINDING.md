# CAR stale-endpoint write guard — boot-free A/B prototype

**Scope:** scratch-only. Worktree
`C:/Users/Likhith/Documents/projects/ravana_car_stale_guard`, branch
`invest/car-stale-write-guard`, base `origin/main` `66480ab`. No engine boot, no
weight-store access (verified: both weight files absent before and after), no
network, no production file, gate, threshold, rubric or fixture edited. Everything
committed here lives under `scratch/car_stale_guard/`.

**Context:** `scratch/car_stale_audit/REPORT.md` (worktree
`ravana_car_stale_audit`, branch `audit/car-stale-id-write-paths`) reports seven
open dead-endpoint write paths on current `main`; H0
`ChainWalkerMixin._create_causal_edge` is highest severity because it is the
conversational causal teaching channel — the same class of write CAR uses. PR
#544 was still OPEN when this was run, so the report was read from that worktree.

## Verdict

| arm | instrument | falsifiers | result | exit |
|---|---|---|---|---|
| all arms in one process, identical fresh state per arm | 1/1 | 3/3 PASS | **PASS** | 0 |

Canonical report SHA-256 (LF-normalised, sorted-key JSON, commit identity excluded
so the measurement hash is stable across commits) `e22b79102cb791bf40d7c6e7f7f748161e352e744e0b6d447341ea6876feffb3`,
identical under `PYTHONHASHSEED=0` and `=1`.

**A production fix is justified** as a narrow write-path repair, **not** as a
claim about the CAR score. Evidence and limits below.

## Exact failing scenario (H0 baseline)

State: a fresh `ConceptGraph(dim=8)` with concepts `impact` (id 0) and `bruise`
(id 1), both removed by `remove_node()` to reproduce the prune/eviction state in
which the engine's `_concept_keywords` index still holds the dead ids
(`impact -> [0]`, `bruise -> [1]`). No boot. Call the UNMODIFIED production
writer directly with the same surface as a teaching turn:

```python
ChainWalkerMixin._create_causal_edge(host, "impact", "bruise",
                                     source_text="impact causes bruise.")
# host._concept_keywords = {"impact": [0], "bruise": [1]}
```

Production output (`chain_walker.py:1394-1401,1425-1433`):

- `ConceptEdge(0, 1, weight=0.75, relation_type="causal", confidence=0.7)` is
  inserted into `graph.edges`, `_outgoing[0]` and `_incoming[1]`.
- `get_node(0)` and `get_node(1)` are both `None` — both endpoints are dead.
- The belief `("impact", "causal", "bruise")` is appended to
  `_belief_assertions` anyway.

So the engine records a fully confident, provenance-looking causal belief that
no chain walker can ever traverse, because enumeration walks live nodes only.
This is the stale-endpoint failure mode: the write succeeds, the knowledge does
not exist.

## A/B arms and outcomes

- **A — baseline:** production `_create_causal_edge` unchanged.
- **B — candidate LOCAL guard (scratch):** before calling the same production
  method, sanitise the resolver index to live ids only — drop dead ids from
  `_concept_keywords` (the F2 pattern already shipped at
  `web_to_graph.py:151-163`). The production method already returns `False` on
  an empty id list, so no production logic was copied or edited.
- **C — candidate CHOKE-POINT guard (scratch monkeypatch):** `ConceptGraph.add_edge`
  rejects a dead source or target, returning an UNINSERTED dummy `ConceptEdge`
  (same fail-closed shape as the production self-loop guard at `graph.py:1707-1718`).

| check | A baseline | B local guard | C add_edge guard |
|---|---|---|---|
| H0 stale index, new edge | **OPEN** — dangling `(0,1)`, false belief | **CLOSED** — index emptied, no edge, no belief | **CLOSED** edge blocked **but false belief still recorded** |
| H0 all-live index | WRITTEN | WRITTEN (no regression) | — |
| H0 mixed index `[dead, live]` | OPEN — blindly picks `[0]` | **WRITTEN on the live ids** (resolves, does not merely abstain) | — |
| H0 stale index, pre-existing dangling edge `0.1/semantic` | STRENGTHENED to `0.4/causal/0.3` | **UNCHANGED** (`0.1/semantic`) | **STRENGTHENED to `0.4/causal/0.3`** |
| H5 direct `add_edge(dead, live)` | OPEN | — | CLOSED (dummy returned, not inserted) |
| H1 episodic writer, stale index | OPEN | — | CLOSED |
| H5/H1 live endpoints | — | — | WRITTEN (no regression) |

## Key findings

1. **The candidate local guard closes both H0 branches with no measured
   regression.** It neutralises the new-edge branch and the strengthen branch
   because both start from the same stale index resolution. In the mixed-index
   control it keeps the live id and still writes the causal edge — the guard is
   a live-only resolver, not a blanket veto.
2. **An `add_edge` guard alone is NOT sufficient.** `_create_causal_edge` first
   calls `get_edge`; on a hit it mutates the existing edge in place without
   calling `add_edge`. Arm C therefore still strengthened a dangling edge from
   `0.1/semantic` to `0.4/causal`, and still appended the false belief in the
   new-edge case. A graph-level invariant at `add_edge` is necessary for H1/H2/H5
   but cannot protect H0's existing-edge branch. Both guards are needed.
3. **The belief assertion is a second, independent write.** It fires after the
   edge write regardless of whether an edge was inserted. A production fix at
   the edge choke point alone leaves the belief store claiming a fact the graph
   never received.
4. **The guard does not clean up history.** In the pre-planted-edge case the
   old dangling edge remains in `graph.edges`; the local guard only stops new
   writes and further strengthening. Old dangling edges need a separate prune
   sweep; that sweep was not in scope here.

## Production-fix recommendation

Justified, narrow and gate-preserving:

1. In `ChainWalkerMixin._create_causal_edge`, resolve `cause_word`/`effect_word`
   through `_concept_keywords` to **live ids only** (`get_node` check at the
   write choke point), removing stale ids from the index; return `False` when no
   live id remains. This is the shipped F2 pattern, moved into this writer.
2. Add a live-endpoint invariant at `ConceptGraph.add_edge` for the other open
   entrances (H1, H2, H5), following the existing self-loop dummy-edge shape.
3. Gate `_belief_assertions` on the edge actually being written/admitted, so a
   blocked edge cannot register a believed fact.
4. Sync the label index at every node-removal site, including capacity eviction
   (H6, `graph.py:1332-1336,2439-2441`), for a graph-wide invariant.

Do **not** open a production PR from this scratch result alone. Per the standing
RAVANA rule, a production change needs a live A/B after the active CAR rerun
finishes, with no gate or threshold relaxed; the boot-free probe establishes the
mechanism and the insufficiency of the choke-point-only fix, not a score change.

## Reproduce

```
cd C:/Users/Likhith/Documents/projects/ravana_car_stale_guard
C:/Users/Likhith/Documents/projects/ravana_car_stale_audit/.venv/Scripts/python.exe \
    scratch/car_stale_guard/ab_probe.py     # exit 0, VERDICT PASS
```

Artifacts: `ab_probe.py`, `results.json`, `run_hs0.log`, `run_hs1.log`.

## Limitations

- No engine boot and no sealed CAR rerun (a CAR rerun is active in the parent
  repo); the claim is about the write mechanism, not the benchmark score.
- H3 (sleep graduation / inline replay) remains source-level only from the
  parent audit; this probe did not invoke live-engine-only writers.
- Arm B is a scratch pre-write sanitisation of the host index, not an edited
  production method; it simulates the F2-style resolver faithfully but a
  production implementation still needs its own unit test.
- The venv used to execute the probe is the audit worktree's; the import-root
  assertion in the probe proves modules came from this worktree's source tree.
