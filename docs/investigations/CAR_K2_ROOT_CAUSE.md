# CAR sealed k=2 acquisition failure — root cause (in-process, before reload)

**Status:** static analysis + evidence audit. **No engine run, no pytest, no source/scratch/weights edit
was performed.** The only artifact written is this document. Evidence: `scratch/learn5_car_measured.json`
(commit `20aca05`, 1333 s, 120 records), `scratch/car_full_p2.log`,
`scratch/p2_write_probe.py`, `scratch/p2_interference_probe2.log`, a **read-only** unpickle census of
the harness's own saved graph `scratch/learn5_car_data/ravana_weights.pkl` (saved at
`measure_learn5_car.py:462`, i.e. *after* teaching and the immediate probes, *before* cold boot 2),
and `scratch/learn5_car_data/junk_labels.jsonl`. HEAD is `d882a00`; `git diff --stat 20aca05..HEAD`
touches only docs, so the measured run and current source are code-identical.

---

## 0. One-sentence root cause

The taught causal hops **are written** by `teach_web`, but sleep-time junk-node pruning during the
100-item teaching run deletes the freshly created concept nodes **without removing their ids from
`_concept_keywords`**; the write path then resolves labels through the derived `_all_labels` view,
which returns the *first* (now dead) id, so both hops of the taught chain are attached to node ids
that no longer exist in `graph.nodes`. `node_ids_for`/`label_map` and `enumerate_chains` enumerate
only **live** nodes, so the chain exists in the raw edge table with correct `curated` provenance yet
is unreachable: `raw_chains=0`, `chain_present=false` **in-process, immediately after teaching**.
This is mechanism **(c) node-id resolution drift**, driven by sleep pruning, not a trust-gate,
ranking, or reload defect.

**Hard evidence (saved post-teaching graph):** 7/10 sealed k=2 failures each have at least one
`teach.local`-sourced `causal` `curated` hop whose endpoint id is in `_concept_keywords` but absent
from `graph.nodes`; all 3 sealed k=2 successes and all 10 primary k=2 successes have both taught
hops on live endpoints. Across all 100 taught items: 29/30 items with a stale id fail the immediate
chain criterion; 58/60 items without a stale id pass.

---

## 1. Exactly how `k` determines exposures, and what `ask()` does between them

### 1.1 Exposure schedule (file:line)

- `scripts/measure_learn5_car.py:77-78`: `K_LEVELS=(0,1,2,4,8)`, `MAX_K=8`.
- `:214-218` `assign_k(pool)`: `ordered=sorted(pool,key=item_hash)`; `k=K_LEVELS[i % 5]`.
  Each of the 100 subject items (50 primary + 50 sealed) gets exactly one k; 20 items per k level.
- `:339-352`: bank of 100 causal items + 20 definition-only items; `:401-405` flatten to
  `all_items = primary(50) + sealed(50) + def_primary(10) + def_sealed(10)` = 120 items.
- `:425-444` teaching loop: `for r in 1..8: for each item`:
  - `defonly` → `teach_definition` every round (8 exposures);
  - `k==0` → one neutral `mlc.ask("what do you know about X?")` every round (8 matched turns, **no causal teaching**);
  - `elif it["k"] < r: continue` (`:437-438`) → item is finished;
  - else → `do_exposure(eng, it)` (`:439-441`).
- Therefore an item with **k=2 is taught in rounds 1 and 2 only** (2 `do_exposure` calls), k=1 only
  in round 1, k=4 in rounds 1-4, k=8 in rounds 1-8. Round sizes logged in
  `car_full_p2.log:2916,3927,4758`: 100, 80, 60 exposures — matching this rule.
- `do_exposure` (`:300-303`) = `mlc.teach_conversational` (one `ask`) **+** `mlc.teach_web`
  (two `WebToGraph.learn_text` calls from `sourceA` and `sourceB`).
- Round-end `mlc.ask("things happen for many reasons, don't they?")` at `:442` — one extra turn per round.
- Probes: 120 baseline `ask`s at `:419-423` **before any teaching**; 120 held-out paraphrase + 120
  identical-surface `ask`s at `:446-459` after all teaching; 120 cold `ask`s at `:471-475` after reload.
- `probe()` (`:315-327`) asks a paraphrase from `PARAPHRASES` (`:306-311`), then computes
  `nids = node_ids_for(item, g, label_map(g))` and calls `trusted_chain_detected` (`:239-258`,
  `enumerate_chains(..., limit=TRAVERSAL_LIMIT=64)` at `:87`) followed by `gate_census` (`:261-289`).
  Both use the **same live-node-only seed set**.

### 1.2 What `ask()` does to the graph (this is the "between exposures" mutation)

- `scripts/measure_learning_curve.py:222-231`: `ask()` is exactly `eng.process_turn(q)`.
- `engine.py:8269` increments `turn_count` on every `process_turn`.
- `engine.py:11426-11459`: **every turn** adds sleep pressure (`+0.02 + 0.01*(1-confidence)`) and
  fires a mini sleep cycle when `pressure>0.3 and turns_since_sleep>8` or after
  `_sleep_schedule_turns=20` (`engine.py:1846`). In this run: `turn_count=988`,
  `sleep_cycles_completed=106`, `last_sleep_episode=984` — one sleep cycle per ~9 turns.
- `_sleep_consolidate` (`engine_generation.py:1214-1239`) → `SleepConsolidation.run_cycle`
  (`core/sleep.py:66-144`): outgoing-weight downscaling (`:124-125`), weight prune
  (`_prune_weak_edges`, `core/sleep.py:236-273`, `prune_threshold=0.1` at `:26`), then
  `engine_generation.py:1256-1258` `prune_low_quality_edges()`, and finally
  `engine_generation.py:1513-1521` **`graph.prune_low_degree_junk_nodes(tau_low=1, tau_high=8)`**.
- `junk_labels.jsonl` contains hundreds of `{"label":-1, "meta":{"degree":0,"source_count":1,
  "glove_mag":null}}` rows — the exact signature written at `engine_generation.py:1527-1529` — for
  `bruise`, `echo`, `snore`, `spinning`, `impact`, `heating`, `displacement` at cycles 3-116.
  Those are precisely the failing items' tokens.

So `ask()` does **not** rewrite edges through response generation; it mutates the graph through the
**sleep maintenance it triggers at turn end**. A k=2 item's two exposures are separated by ~100 other
`ask`s in round 1 (≈11 sleep cycles), and the item then waits through rounds 3-8 (~6 more rounds,
~65 sleep cycles) before the immediate probe. A single-exposure probe in a fresh copied store has
only a handful of turns — no sleep churn — which is why it succeeds in isolation.

---

## 2. Mechanism discrimination

| candidate | verdict | evidence |
|---|---|---|
| **(a) why-probe `ask()` rewrites/prunes just-taught edges via generation** | **Partly right, wrong mechanism.** `ask()` does mutate the graph, but through per-turn sleep (`engine.py:11426-11459`), not through `_why_how_causal_chain` (read-only). Sleep prunes nodes/edges, leaving stale label ids. | 106 sleep cycles / 988 turns; pruned-label rows in `junk_labels.jsonl`. |
| **(b) later items overwrite shared-token edges' `source_metadata`/`relation_type`** | **Not the drop.** `WebToGraph.learn_text` XdG guard (`web_to_graph.py:259-275`) `continue`s on a cross-context re-sighting, i.e. it *protects* the earlier edge; it does not overwrite. `intake._write_edge` upgrades non-causal→causal (`intake.py:742-745`), which is the P2 fix already shipped. The live shared hop (e.g. `reflection→repetition` from `reverb`, `2634→4501`, w=0.307, 4 contexts) is intact — it simply leads to `echo`'s **dead** node (`repetition→3436`). | pkl edge census below. |
| **(c) node-id resolution drift (`_concept_keywords` / `_all_labels` first-id)** | **CONFIRMED — primary cause.** Stale ids: `echo` [3436,4429,4561,4645,4900] all dead; `snore` [4447,4582,4653,4918,4921] all dead; `bruise` first id 4422 dead; `spinning` all 7 ids dead; `heating` first 2943 dead; `displacement` first 3526 dead; `impact` first 3010 dead. Taught hops attach to these dead ids. | pkl census; `interface.py:71-74`, `intake.py:315-320`, `web_to_graph.py:151-154`, `explanation.py:211-214,226-227`. |
| **(d) graph-level cap/prune removes low-confidence first-sighting edges** | **Contributing, not sufficient.** Weight prune and junk-node prune are the *mutator* that kills the node, but the edges themselves still exist (curated, conf 0.57-1.0, w 0.0-0.307). If edges alone were pruned the node would remain live and `label_map` would still seed the walk; here the node is gone and the edge dangles. | pkl: 406/21102 dangling edges, 1049/2535 stale kw ids. |
| **(e) trusted chain exists but crowded out at limit 64 by F3 ranking** | **RULED OUT for the 7 failures.** `gate.raw_chains=0`, `graded_assert=0`, `trusted_all_hops=0`, `hop_gates=[]` in **both** immediate and cold stages — no candidate at any trust level, so no ranking effect. (Successes `voice`/`pearl` do show raw 61-64 with 12-15 trusted and are handled by the shipped F3.) | `learn5_car_measured.json` per-item gates; `explanation.py:261-289`. |

### 2.1 Why the write lands on a dead id (exact call chain)

1. `graph.remove_node` (`ravana_ml/src/ravana_ml/graph.py:1479-1502`) deletes the node **and** its
   edges, but has no access to the engine's `_concept_keywords` index, so the id survives there.
   (`_sanitize_graph`, `chain_walker.py:1328-1371`, *does* clean the index — but only for load-time
   question-phrase nodes, not for sleep-junk nodes.)
2. `interface.py:71-74` builds `_all_labels` as `{lab: nids[0] ...}` from `_concept_keywords` with
   **no liveness filter**. So the first, dead id is returned for that label.
3. `WebToGraph._ensure_node` (`web_to_graph.py:151-154`) trusts `_all_labels.get(label)` and returns
   the dead id; `SchemaConsistencyGate._node_id` (`intake.py:315-320`) does the same.
4. `ConceptGraph.add_edge` (`graph.py:1673-1729`) never validates that source/target ids name live
   nodes — it happily inserts `(dead_id, live_id)` into `edges`, `_outgoing`, `_incoming`.
5. `node_ids_for` (`measure_learning_curve.py:143-147`) and `label_map` (`:131-141`) enumerate live
   `graph.nodes`, and `enumerate_chains` (`explanation.py:211-214`, `:226-227`) drops any neighbour
   whose `graph.get_node()` is `None`. Result: `raw_chains=0` even though the curated edge is in the table.

---

## 3. Per-item attribution — sealed k=2 (10 items)

From `learn5_car_measured.json` (immediate/cold stages) and the read-only pkl census.
"stale" = ids in `_concept_keywords[x|a|b]` absent from `graph.nodes` at save time;
"pruned(x/a/b)" = count of `junk_labels.jsonl` `label=-1/source_count=1` rows for those tokens.

| item_id | x (a→b) | stale ids (x/a/b) | pruned events (x/a/b) | taught hop on dead id? | imm/cold gate raw | imm/cold chain |
|---|---|---|---|---|---|---|
| subject_003 | echo (reflection→repetition) | **5/0/0** (all echo ids dead) | 11/0/0 | `repetition→3436(dead echo)` from echo/sourceB | 0 / 0 | FAIL / FAIL |
| subject_006 | hiccup (spasm→interruption) | 0/0/0 | 0/0/0 | no — both hops live (4559→4558→4427) | 1 / 1 | PASS / PASS |
| subject_010 | bruise (impact→rupture) | **3/1/0** | 3/55/0 | `3010(dead impact)→4552`, `4552→4422(dead bruise)` | 0 / 0 | FAIL / FAIL |
| subject_045 | tsunami (displacement→wave) | 0/**1**/0 | 10/14/17 | `3526(dead displacement)→2783` | 0 / 0 | FAIL / FAIL |
| subject_064 | pearl (irritation→secretion) | 0/0/0 | 0/1/8 | no — 4585→4486→4455 all live | 61 / 61 | PASS / PASS |
| subject_073 | webbing (spinning→thread) | 0/**7**/0 | 0/12/0 | `3987(dead spinning)→4567` | 0 / 0 | FAIL / FAIL |
| subject_074 | cocoon (spinning→silkworm) | 0/**7**/0 | 0/12/0 | `3987(dead spinning)→4571` | 0 / 0 | FAIL / FAIL |
| subject_094 | voice (vibration→sound) | 1/0/0 (voice id live) | 1/0/0 | no — 257→231→4445 all live | 64 / 64 | PASS / PASS |
| subject_096 | snore (vibration→breath) | **5**/0/0 (all snore ids dead) | 6/0/0 | `4524→4447(dead snore)` | 0 / 0 | FAIL / FAIL |
| subject_098 | boil (heating→bubbles) | 0/**1**/0 | 4/23/10 | `2943(dead heating)→4583` | 0 / 0 | FAIL / FAIL |

**7/7 failures have a stale id on the taught path; 3/3 successes have none.** Primary k=2: all
10 items have zero stale ids and all 10 pass (immediate Δ=0.8). Whole-bank taught-item count:
`(stale, no chain)=29`, `(clean, chain)=58`, `(clean, no chain)=2` (`flood`, `color` — separate
rank/acquisition cases), `(stale, chain)=1` (`aurora`, stale id off the taught path). The remaining
`(stale, no chain)` cases are the 10 untaught k=0 controls plus the above. This is a deterministic
item-composition effect (which labels get pruned), not gate noise; the −0.2/−0.3/−0.4 magnitude
variation across runs reflects *when* the sleep cycles land relative to each item's two exposures.

**Shared tokens matter because they are pruned repeatedly, not because they are overwritten.**
`spinning` is taught by silk/webbing/cocoon → 7 ids, all dead; `impact` by bruise/crater → first id
dead; `heating` by boil/steam/geyser → first id dead; `reflection`/`repetition` by echo/reverb/mirror/color.
`vibration` is a high-degree pre-existing hub (id 257, 16 out / 38 in), never pruned — which is why
`voice` succeeds while `snore` (whose own subject node is pruned) fails on the same `a` token.

### 3.1 Why the isolated single-exposure probe writes the chain

`scratch/p2_interference_probe2.log:3,13,23`: before bruise `impact=[3010]` (live), `rupture=None`,
`bruise_ids=None`; after one `do_exposure` `bruise=[4374]`, chain `['impact','rupture','bruise']`,
`raw_chains=1`, curated `w≈0.42`. With only a few turns there is no sleep prune between node
creation, edge write and measurement, so `_all_labels` resolves live ids. Teaching bruise then
hiccup keeps the chain (`log:23`). The full harness adds ~988 `ask` turns and 106 sleep cycles
before/around those two exposures — the same write path now resolves dead ids.

---

## 4. Ranked fix list (all gate-preserving: no threshold, rubric, split or trust-set change)

### F1 — Synchronise `_concept_keywords` with node deletion (rank 1)
- **Target:** `ravana/src/ravana/chat/engine_generation.py:1513-1521` (call site after
  `graph.prune_low_degree_junk_nodes`), reusing the cleanup idiom in
  `ravana/src/ravana/chat/chain_walker.py:1367-1371`; the graph primitive is
  `ravana_ml/src/ravana_ml/graph.py:1479-1502 remove_node`.
- **Primitive change:** after each pruned nid, delete that id from every `_concept_keywords[label]`
  list (and drop the key when no live id remains). Keep `edge_kind`/provenance untouched.
- **Measurable prediction:** zero stale kw ids after teaching; the next exposure mints a live node
  instead of reusing a dead id; the 7 sealed k=2 items move from `raw_chains=0` to `>0`; the
  `(stale, no-chain)` census cell drops to the 10 k=0 controls only.
- **Falsifier:** if stale ids reach zero but the 7 items still have `raw_chains=0` **and** their
  taught edges are absent from live endpoints, the write path (F2), not the index, owns the drop —
  record that and stop editing the prune path.

### F2 — Resolve write-time labels through live nodes only (rank 2)
- **Target:** `ravana/src/ravana/chat/interface.py:71-74 _all_labels`;
  `ravana/src/ravana/learn/intake.py:315-320 SchemaConsistencyGate._node_id`;
  `ravana/src/ravana/web/web_to_graph.py:151-154,183-206 _ensure_node`.
- **Primitive change:** `_all_labels` returns `{lab: first id with graph.get_node(id) is not None}`;
  when a label has only dead ids, treat it as absent and have `_ensure_node` mint a fresh node and
  put its id **first** in `_concept_keywords[label]`. Never write an edge whose endpoint fails
  `graph.get_node` (log + skip instead).
- **Measurable prediction:** zero `teach.local`-sourced edges with a dead endpoint; all 7 items
  acquire a live 2-hop trusted chain; k=0/def-only controls stay at 0.00.
- **Falsifier:** if a fresh live node is minted but the curated edge is not in `graph.edges`
  (intake routed it to the episodic buffer), F2 is insufficient and the gate route is the drop —
  instrument `KnowledgeIntake.intake` (`intake.py:688-703`) before changing anything else.

### F3 — Provenance-based grace window around taught episodes (rank 3)
- **Target:** `ravana/src/ravana/chat/engine_generation.py:1513-1521` (junk-node prune) and
  `ravana/src/ravana/core/sleep.py:236-273 _prune_weak_edges`.
- **Primitive change:** skip nodes/edges whose `source_metadata.edge_kind ∈ {curated, web_fact}`
  **and** whose `source` starts with `http://teach.local/` for N turns after their write stamp.
  This is a write-time tag read by maintenance — not a threshold change and not a trust exemption.
- **Measurable prediction:** pruned events for `bruise/echo/spinning/snore/impact/heating/displacement`
  drop to zero during teaching; stale-id count at immediate stage drops; sealed k=2 Δ_cold rises
  by 3-7 items (0.3 → 0.6-1.0) and the −0.1 rejection bound is met without touching any gate.
- **Falsifier:** if disabling only the junk-node prune leaves the stale-id count unchanged, the
  dead ids come from a different remover (`_sanitize_graph`); revert F3 and census that path.

### F4 — Trust-aware degree floor in junk-node pruning (rank 4)
- **Target:** `ravana_ml/src/ravana_ml/graph.py:1504-1558 prune_low_degree_junk_nodes`.
- **Primitive change:** compute degree from live edges only and never prune a node with an incident
  provenance-trusted causal edge, even when `degree < tau_low=1`.
- **Measurable prediction:** no `curated` causal node appears in the prune set; the 7 items keep a
  live first id; unrelated junk-node counts stay flat.
- **Falsifier:** if a census of the prune set contains no trusted causal node, this mechanism is
  innocent for CAR — record the negative and revert (skill §7a).

### F5 — Dangling-edge invariant + measurement census (rank 5, prerequisite for all A/Bs)
- **Target:** `ravana_ml/src/ravana_ml/graph.py:1673 add_edge` (harness/debug builds) and
  `scripts/measure_learn5_car.py:315-327 probe()`.
- **Primitive change:** refuse/log `add_edge` with a non-live endpoint; add `stale_ids`,
  `dangling_edges`, and `taught_hop_liveness` to each stage record (measurement only — gate functions
  unchanged).
- **Measurable prediction:** the post-teaching census shows zero dangling `teach.local` edges and
  makes F1/F2 attributable per item; the JSON can prove the chain's absence is node liveness, not
  trust ranking.
- **Falsifier:** if the census reports zero stale/dangling ids in a fresh run yet `raw_chains=0`
  remains, this entire root cause is wrong for that run — stop and re-diagnose from the write gate.

**Deliberately not ranked high:** changing `WebToGraph.xdg` (`web_to_graph.py:263-275`). It protects
shared-token edges from cross-context overwrite but does not create dead endpoints; test it as a
secondary A/B only after F1/F2 are measured.

---

## 5. What remains unconfirmed, and the probe that would confirm it

1. **Exact intra-turn ordering.** We know nodes were deleted and edges later written to the dead
   ids, but not whether the deleted node was minted by conversational grounding in the same
   `process_turn` or by an earlier web exposure and killed in a later sleep cycle.
   **Probe:** wrap `ConceptGraph.remove_node` and `_sleep_consolidate` to log
   `(nid, label, turn_count, source_metadata, caller)` for every node pruned during teaching, and
   compare with the first `_concept_keywords` id for each item's x/a/b at that turn.
2. **Which channel wrote the dead endpoint.** The provenance (`source=http://teach.local/.../sourceB`)
   implicates `teach_web`/`learn_text`→`intake._write_edge`, but no write-time id trace exists.
   **Probe:** log `subj_id, obj_id, fact, route` at `intake.py:688-703` and `_write_edge:705-762` for
   the 7 failing items, plus `graph.get_node(id)` immediately after each write.
3. **XdG's contribution for shared tokens.** **Probe:** A/B on identical copied state, one process,
   with `WebToGraph.xdg=False`, same seed; measure stale-id count and sealed k=2 `raw_chains` plus
   unrelated-front negatives before keeping any change.
4. **Whether seed-dependent timing changes *which* items fail.** Reversal magnitude varies
   (−0.2/−0.3/−0.4) across three runs. **Probe:** rerun the F1/F2 A/B at two hash seeds on identical
   state; if the same items recover at both, item composition (not timing) owns the failure.
5. **Cold-reload interaction.** Both immediate and cold stages show identical `raw_chains=0` for
   these 7 items, so reload adds nothing here; the shipped F1/F3/F5 and durability fixes are
   exonerated by the per-item identity of the gate numbers. No further reload work is indicated
   by this evidence.

---

## 6. File:line index used

- `scripts/measure_learn5_car.py:77-78,87,214-218,239-258,261-289,300-311,315-327,419-444,446-475`
- `scripts/measure_learning_curve.py:131-147,222-231,247-296`
- `ravana/src/ravana/chat/engine.py:8269,1846,11426-11459`
- `ravana/src/ravana/chat/engine_generation.py:1214-1258,1493-1521,1527-1529`
- `ravana/src/ravana/core/sleep.py:19-44,104-144,236-273`
- `ravana/src/ravana/chat/interface.py:66-82`
- `ravana/src/ravana/web/web_to_graph.py:109-133,135-213,215-308`
- `ravana/src/ravana/learn/intake.py:295-320,688-703,705-762`
- `ravana/src/ravana/chat/chain_walker.py:1328-1371`
- `ravana/src/ravana/learn/explanation.py:185-252,261-289,292-352`
- `ravana_ml/src/ravana_ml/graph.py:1479-1502,1504-1558,1673-1729`
