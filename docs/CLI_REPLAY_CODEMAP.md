# CLI replay codemap — why sleep ON ≡ sleep OFF in RAVANA-LEARN-5 CLI

Scope: line-evidence map only. No code or other docs were modified. Investigated on
`c11e2fe` (clean tree apart from another agent's untracked doc). Repro artifacts are
`scratch/probe_replay_codemap.py`, `scratch/probe_replay_grade.py` and their `.log`
files; measured numbers are from `scratch/learn5_cli_measured.json` (commit `c47ac09`,
the measurement the task refers to).

## 1. Verdict in one sentence

The CLI "sleep replay" never changes a field the trust gate uses to make its decision:
it adds `+0.02` weight / `+0.01` confidence per cycle to edges that the gate has already
accepted or already rejected on **provenance**, while the only destructive work in the
same cycles (REM counterfactual edges and relation-type flips) runs identically in both
arms; the other half of the manipulation (`HippocampalReplay`) is a throwaway object
that never touches the graded graph.

Measured: PRIMARY `ret_on = ret_off = 0.700` (7/10), SEALED `ret_on = ret_off = 0.833`
(5/6), `on_minus_off = 0.0` in both pools; A-only 0.900/0.833 and concurrent 0.875/1.000
show the interference effect is real, so the instrument is not blind — only the replay
manipulation is inert. Replay control passes (real edge 0.20 → 0.22, noop unchanged),
so the pipeline is executing; it is executing the wrong mutation.

## 2. The two replay implementations, and which one the CLI runs

| Implementation | State it mutates | Called by CLI harness? |
|---|---|---|
| `SleepConsolidation.run_cycle` — `ravana/src/ravana/core/sleep.py:66-222` | `ConceptGraph` edges: `weight`, `confidence`, `relation_type`, `stability`, node vectors; prunes/adds edges | Yes, but via a **new** instance (`scripts/measure_learn5_cli.py:402`), not `eng.sleep_engine` |
| `HippocampalReplay.sleep_cycle` — `ravana/src/ravana/learn/consolidation.py:94-105` | only its own `self.consolidated` dict (`run_nrem_cycle` :55-61, `run_interleaved_cycle` :63-87, `prune` :89-92). `grep graph` in this file returns **zero** matches | Yes, via a **new** instance (`measure_learn5_cli.py:389,407`), never `eng.hippocampal_replay` |

The engine's real sleep path is `CognitiveChatEngine._sleep_consolidate`
(`ravana/src/ravana/chat/engine_generation.py:1214-1339`): it calls
`self.sleep_engine.run_cycle` (:1226-1238), `self.hippocampal_replay.sleep_cycle`
(:1296-1304), and then **Phase 3b** drains `self.hippocampal_buffer` candidates into
durable graph edges through the live schema-consistency gate with their real
provenance (:1305-1339). The CLI harness never calls `_sleep_consolidate`; Phase 3b —
the one step that actually writes replayed knowledge back into the graph — is bypassed.

## 3. Probe results (real tool output)

### Probe 1 — one A item (`fog/chill/condensation`) taught A then B, ON vs OFF
`scratch/probe_replay_codemap.log`, PYTHONHASHSEED=0, uv run, identical cold boots.

- Before sleep (both arms): `present=True`, `structural=False`; graded chain labels
  `[cooling, condensation, fog]`. The taught `chill→condensation` edge exists but a
  higher-weight pre-existing `cooling→condensation` curriculum edge (w 0.78, conf 0.756)
  outranks it; the retention metric grades **presence**, not the A chain.
- ON after 3 cycles: taught edges `chill→condensation` w **0.50→0.56**, conf
  **0.5469→0.5769**; `condensation→fog` w **0.25→0.31**, conf **0.4238→0.4538**;
  `source_metadata` and `relation_type` byte-identical; `edges_strengthened=2/cycle`.
- OFF after 3 cycles: both edges **unchanged**; `edges_strengthened=0/cycle`.
- Both arms: `present=True`, `structural=False`, `verdict="assert"`, adequacy 1.0,
  `graded_assert=4`, `trusted_all_hops=4`, `rejected_by_trust≈47-48` — same decision.
- Functions called, in both arms (3× each): `SleepConsolidation.run_cycle`,
  `HippocampalReplay.sleep_cycle`, `HippocampalReplay.run_nrem_cycle`. The only
  difference in outcome is the 2 strengthened edges (ON) vs 0 (OFF).
- Both arms: `edges_pruned=136/1/3`, `rem_counterfactual_edges=38/36/36`,
  `rem_flipped_edges=12/13/14`, `hippocampal total_consolidated=1` — identical.

### Probe 2 — grade components before/after sleep for four A items
`scratch/probe_replay_grade.log`; expected `a→b→x` chain extracted explicitly then graded.

| item | before sleep | after sleep ON | after sleep OFF |
|---|---|---|---|
| fog | adequacy 0.745, assert, both hops `curated`, min-conf 0.245 | expected chain gone from top list (junk `frost happens fog` ranks first); graded chain still present → no metric change | same |
| frost | adequacy 1.0, assert | adequacy 1.0, assert; conf 0.728→0.759 / 0.901→0.933, w 0.50→0.423 | adequacy 1.0, assert; conf unchanged, w 0.50→0.361 |
| shadow | adequacy 1.0, assert | expected chain gone from top list (junk `malleability causes shadow`) | same |
| thunder | **decline** — `shock→expansion` has `edge_kind="inferred"`, `prov=false` | still decline; w 0.361→0.345, conf 0.74→0.771; provenance unchanged | still decline; w/conf unchanged |

Direct conclusion from the table: where the chain is accepted, replay's confidence
bump lands on an already-saturated adequacy of 1.0; where it is rejected, the reason is
a provenance field or a missing edge that replay never writes.

### Measured loss forensics (`scratch/learn5_cli_measured.json`, PRIMARY_on)
For the three A items lost after B+sleep, the post-sleep gate census is:

- `A_015 frost`: `raw_chains=1`, `trusted_all_hops=1`, **`graded_assert=0`** — the
  trusted chain exists but fails the grade; ON and OFF are identical.
- `A_003 shadow`: `raw_chains=0`, `trusted_all_hops=0` — no chain to grade at all.
- `A_024 thunder`: `raw_chains=0`, `trusted_all_hops=0` — same.
- In `PRIMARY_a_only` (no B) A_015 and A_003 are retained, so B is the interference
  source; replay cannot undo it because it neither restores deleted edges nor
  re-provenances an `inferred` hop.

## 4. Ranked root causes with file/line evidence

**1. Replay writes none of the fields the trust gate decides on.** The only replay
mutation (`sleep.py:118-122`) is `edge.weight += 0.02`, `edge.confidence += 0.01`,
`hebbian_update(...)`. The decision variables are `edge.source_metadata.edge_kind` /
`edge.provenance` (`learn/explanation.py:268-299,302-316`), `relation_type`
(`explanation.py:128-138,209-210`), and `confidence` only as one saturated term of
`grade_chain` (`explanation.py:331-366`). `assertable_chain` fails closed on one
untrusted hop (:319-328) regardless of weight/confidence, and `grade_chain` forcibly
converts `assert`→`decline` when provenance fails (:360-362). Evidence: probe 2
thunder (inferred hop stays decline in both arms) and frost (adequacy 1.0 before the
bump). Sleep never stamps `source_metadata` anywhere in `sleep.py:104-222`.

**2. Replay content is derived from the already-damaged graph, so lost A edges are
never replayed.** `a_chain_triples` (`measure_learn5_cli.py:346-362`) collects only
edges for which `g.get_edge(s,t) is not None` at sleep time (after B), and
`pad_replay` (:365-373) pads the rest with missing-edge no-ops. STAGE 1 then skips
those no-ops (`sleep.py:114-116`). Net: the +0.02/+0.01 applies only to A edges that
survived B; the items that failed (post `raw_chains=0`) get no replay at all.
Three cycles give at most +0.06 weight/+0.03 confidence, far below the damage.

**3. The `HippocampalReplay` half of the manipulation is disconnected.**
`consolidation.py:55-105` updates only `self.consolidated`; there is no `graph`
reference in the file. The harness creates its own `hr_on`/`hr_off`
(`measure_learn5_cli.py:389-394`) and calls `hr.sleep_cycle` (:407); the engine's
`self.hippocampal_replay` and the Phase 3b neocortical drain
(`engine_generation.py:1296-1339`) are never reached. Its `total_consolidated=1` in
both arms is a private dict entry, not a graded edge.

**4. The same sleep cycle destroys chains identically in both arms.** REM sabotage
(`sleep.py:156-160`, implementation :275-426) creates ~36-38 counterfactual edges per
cycle (:307-349) and flips ~12-14 existing `relation_type` values per cycle
(:351-366, `edge.relation_type = new_type` at :364). A flipped causal hop is invisible
to `enumerate_chains` (`explanation.py:209-210`). The harness pins `time.time` for both
arms (`measure_learn5_cli.py:398`), so the REM RNG is identical — this is a common
destructive force replay is not configured to protect against. The unbounded junk also
explains `rejected_by_trust` of 47-64 in probe 1 and the A-only gate census:
`enumerate_chains(limit=64)` (`measure_learn5_car.py:87,247-248`) is filled with
untrusted cross-product edges that crowd real chains out of the traversal, exactly the
trap in the ravana-debugging skill.

**5. The retention metric is a binary presence check that is already saturated.**
`chain_result` (`measure_learn5_cli.py:313-328`) and `rate` (:501-511) count
`chain["present"]` = first enumerated chain that is both grade-`assert` and
`assertable`; `structural` (the exact a→b→x match) is recorded but never used for
retention. Probe 1 shows ON changes ranking (pre `[cooling,condensation,fog]` → post
`[chill,condensation,fog]` in the full run) and structural correctness, while
presence stays 1 → the headline number cannot move.

**6. "Episodic → semantic consolidation" is a counter, not a write.**
`sleep.py:162-170` loops the episodic buffer; the only body for a concept is
`if cid in graph.nodes: pass`, followed by `episodic_consolidated += 1`. With
`episodic_buffer=[]` in the harness (`measure_learn5_cli.py:403`) this stage is doubly
empty. No replayed fact is written to the graph here.

## 5. Requested checks

- **Env flag / baby mode gate:** none. `sleep.py` and `consolidation.py` contain no
  `os.environ`/`getenv`; repo-wide grep for `RAVANA_SLEEP`, `DISABLE_SLEEP`,
  `SLEEP_ENABLED`, `RAVANA_REPLAY` returns nothing. `RAVANA_OFFLINE=1`
  (`measure_learn5_cli.py:71`) only suppresses network. `baby_mode=True`
  (`engine.py:911,948`; `interface.py:347`) affects grounding responses
  (`engine.py:10775,11054,11403`), not replay. The engine's automatic sleep trigger
  is pressure/schedule based (`engine.py:11354-11388`, `_sleep_schedule_turns=20` at
  :1846) and is not what the harness calls. ON/OFF is purely the harness argument at
  `measure_learn5_cli.py:376-408`.
- **Same engine instance holding A knowledge:** partially. `run_sleep` receives the
  same `eng` and passes `eng.graph` into `SleepConsolidation.run_cycle`
  (`measure_learn5_cli.py:402-405`), so STAGE 1 does touch A's real edges (proved by
  the +0.02 deltas). But the `SleepConsolidation` and `HippocampalReplay` objects are
  fresh throwaways; `eng.sleep_engine`, `eng.hippocampal_replay`,
  `eng.hippocampal_buffer` and `_sleep_consolidate` are never used. The engine that
  holds A knowledge is only used as an edge container.
- **Does replay write provenance the trust gate accepts:** no. Accepted provenance is
  stamped at write time — `web_to_graph.py:288-293` sets `edge_kind="web_fact"`, and
  conversational teaching stamps `edge_kind="curated"` (probe 1 `source_metadata`).
  Sleep only reads `source_metadata` in the prune gate (`sleep.py:264-268`), never
  writes it. An `inferred` hop stays `inferred` and `grade_chain` declines it.
- **Compute parity is genuine:** padding uses missing-edge triples
  (`measure_learn5_cli.py:365-373`) so step counts are equal; measured wall clock
  0.313 s vs 0.309 s (PRIMARY), 0.2395 s vs 0.2426 s (SEALED). This is not an
  instrument failure; the mutation is inert by construction.

## 6. Narrowest correct change

Make replay act on the fields the trust gate reads, using provenance **already earned
at write time**, and make its content survive the interference that creates the loss:

1. `ravana/src/ravana/core/sleep.py:104-122` — STAGE 1 must reconsolidate the replayed
   edge's *identity*, not only its magnitude: carry the experience's recorded
   `source_metadata` / `relation_type`, restore them on the replayed edge (this repairs
   a REM-flipped causal hop), and flag replayed edges so the REM flip
   (`sleep.py:352-366`) and the two prune passes (`sleep.py:127-144`) skip them for
   that cycle. Keep the existing `+0.02/+0.01` strengthening. This changes
   `edge_kind`, `relation_type` and `confidence` — the exact inputs of
   `explanation.py:209-210,302-316,331-366` — with the edge's own provenance, not a
   fabricated or blanket stamp.
2. `scripts/measure_learn5_cli.py:346-362,376-414` — build replay triples from the
   engine's episodic/replay store, not only from `g.get_edge` on the post-B graph, and
   run them through the engine's own `sleep_engine`/`hippocampal_replay` so the
   existing gate-preserving Phase 3b drain (`engine_generation.py:1305-1339`) writes
   restored facts with their real provenance. Keep equal steps/cycles and pinned time
   for the OFF arm. This is what allows an A edge that B deleted (A_003/A_024,
   `raw_chains=0`) to be replayed at all.

This does **not** lower `THETA_ASSERT` (0.70, `explanation.py:47`), does not relax
`assertable_chain`, does not stamp `web_fact`/`curated` onto an `inferred` hop, does not
change the metric, and does not add compute to one arm only. An `inferred` hop that
never had trusted provenance must still decline — that is the gate working, not a bug
to be patched around.

Explicitly rejected alternatives: bump weight/confidence harder (leaves provenance
untouched and just games `score_chain`); exempt the ON arm from REM/prune (unmatched
compute and it hides interference rather than consolidating through it); count prose
paraphrases as retention (the skill's regex trap); change `ret` to `structural` alone
(measures a different property and does not create learning).
