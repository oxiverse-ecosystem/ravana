# TRA experiment 1 — typed-role write/read feasibility on the real engine

**Status: experiment only. DO NOT IMPLEMENT a production change from this result.**
No file under `ravana/src/`, `ravana_ml/src/`, `scripts/` or `tests/` was edited.
The engine imported was the **unmodified production tree** at
`C:/Users/Likhith/Documents/projects/ravana` (main `dbed80d`); the worktree
`C:/Users/Likhith/Documents/projects/ravana_tra_exp` on
`experiment/tra-write-read` holds only this doc and the three `scratch/`
artifacts. All engine writes went to a **copied store** at
`C:/Users/Likhith/AppData/Local/hermes/cache/scratch/tra_write_read_data`;
production `weights/` was read from but never written to. Module resolution was
asserted at runtime: `engine.py`, `schema_induction.py`, `chain_walker.py` and
`response_gen.py` all resolved under the production tree, not the worktree.

This is the pre-registered "scratch experiment before production implementation"
step for TRA (`TRA_BRAIN_DESIGN.md` §0, `TRA_BENCHMARK_SPEC.md` §7). It is a
**bounded feasibility probe** (2 boots, 10 teaching turns), *not* the 400-family
TRA benchmark run and **not** a LEARN-5 component claim.

## What was run

```
cd C:/Users/Likhith/Documents/projects/ravana_tra_exp
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 PYTHONPATH=<production paths> \
  C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/tra_write_read_experiment.py > scratch/tra_write_read_run.log 2>&1
```

`EXIT=0`. Boot 1 = 2.3 s (load 2167-word store), boot 2 = 2.7 s, `eng.save()`
wrote 95,397 KB to the **scratch** store. Ten `process_turn` teaching assertions
used the exact natural-language forms from `TRA_BENCHMARK_SPEC.md` §1/§2: two
`is a` type assertions (`mist is a weather phenomenon.`, `election is a civic
event.`), two `part of`, two `causes`, two `precedes`/`comes before`, two
`resembles`/`is similar to`. No item-bank label, `world_model`, `type_sets`,
`gold_slot` or grader hook was supplied to the engine or the grader.

## Instrument validation (before reading any number)

Two probe defects were found and fixed; their original outputs are retracted,
not interpreted.

| Defect | First (invalid) reading | Corrected probe | Effect on conclusion |
|---|---|---|---|
| `is_a_phrase_present` used substring `"is a" in p` | `true` — matched `is a component of` / `is akin to` | exact phrase equality | **false** — no standalone `is a` type phrase exists in the write map |
| slot-label census scanned all `source_metadata` values with substrings | 52 "slot" hits — one was a URL containing `effects` matching token `effect` | exact match in a dedicated type-slot metadata key only | **0** slot labels before and after teaching |
| Flag-on pick `effect` was read as a typed-slot answer without a control | "typed-slot selection expressible" | option permutation + unrelated-subject + disjoint-option controls | see below — the pick is GloVe cosine, not a taught slot |

The scratch-only read probe was checked against the real helper signatures
(`si.traces_from_edges`, `si.induce_schema_hypotheses`,
`rg.schema_association_scores_from_graph`, `si.rank_candidates_for_unseen_subject`)
and reproduced the production flag-on picks from the scorer inputs (below).

## Per-stage counts (final valid run)

| Stage | Count |
|---|---|
| Teaching turns (`process_turn`) | **10** |
| Engine boots | **2** (load 2167-word store, then cold reload) |
| Edges / trusted edges before teaching | 18,984 / 170 |
| `is_a` edges before → after teaching → after reload | **169 → 169 → 169** (zero written) |
| Type-relation edges (`is_a`,`has_property`,`property`,`part_of`,`located_in`) before → after → reload | 238 → 240 → 240 (+2 `part_of`) |
| Trusted edges before → after → reload | 170 → 178 → 178 |
| New provenance-trusted edges from teaching | **8** |
| Cold-reload survivors | **8 / 8** (0 lost) |
| Type-slot labels stored on any edge | **0 before, 0 after** |
| Typed-slot forced-choice, **default production path** | **unparseable** — hedge, no decision |
| Typed-slot forced-choice, `RAVANA_SCHEMA=1` (non-default opt-in) | parseable pick, but see controls |
| Scratch-only probe: traces → hypotheses → keys | 60 → 3 → `cause`, `is_a`, `property` (relations, **never slots**) |

The 8 new trusted edges were exactly:
`drizzle→causal→mist`, `storm→causal→blizzard`, `wheel→part_of→vehicle`,
`petal→part_of→flower`, `dawn→temporal→sunrise`, `winter→temporal→spring`,
`mist→analogical→fog`, `wheel→analogical→gear`. Each carried
`edge_kind='curated'`, `source='conversation'`, `source_type='user'`,
`extractor='conversational-assertion'`, `asserted_relation=<FAMILY>`; all 8
survived `save()` + cold reload by exact (label, relation, label) identity.

## EVIDENCE

**Write side**

1. The 10 teaching turns produced **8** new provenance-trusted relation edges and
   **zero** new `is_a` edges. The two type assertions (`mist is a weather
   phenomenon.`, `election is a civic event.`) wrote nothing trusted; `is_a`
   stayed at 169 before/after/reload (`scratch/tra_write_read_results.json`,
   stages `baseline` / `post_teach_in_process` / `post_cold_reload`).
2. The write path has no `is a` assertion handler. `process_turn` calls only
   `_consolidate_asserted_relations` (`engine.py:10858-10865`). Its phrase map
   `ChainWalkerMixin._ASSERTED_RELATION_PHRASES` (`chain_walker.py:946-955`)
   contains PART_OF (`is part of`, `belongs to`, `is a component of`, `contains`,
   `includes`, `is composed of`), PRECEDES (`precedes`, `comes before`, …) and
   RESEMBLES (`resembles`, `is similar to`, `is akin to`) — no standalone `is a`
   / `is a kind of`. `_asserted_relation_propositions` (`chain_walker.py:989-1029`)
   therefore emits no `is_a` fact for the two type assertions. The runtime
   `write_surface.is_a_phrase_present` was `false`.
3. No edge, before or after teaching, carries a type-slot label in a dedicated
   metadata key (`slot_label_evidence_count = 0` at all three censuses). The
   stored object of `part_of` is just the word `vehicle`/`flower`; nothing marks
   it as the `whole` slot, and `effect`/`whole`/`later`/`similar` are never
   stamped as types of an asserted word.
4. Intake itself *could* write `is_a` (`intake.py:262-288` delegates to
   `WebToGraph.RELATION_TO_EDGE`, which maps `is_a → is_a`), but no conversational
   caller constructs such a `Fact` with user `Provenance`. The write failure is
   upstream of the gate, not a trust-threshold rejection: the unchanged gate
   admitted all 8 facts it was given.

**Read side**

5. Default production answer path: the typed-slot query
   `what is true of thaw? Options: effect, whole, later, similar.` emitted
   `"true thaw — that's a fun image! …"` — **no parseable decision**. The same
   hedge occurred for an object-candidate version. Per TRA falsifier **F4**
   ("forced single-token slot choice with echoed text stripped yields ~0
   parseable decisions → no decision routine exists"), this is a genuine
   negative, not a prompt-tuning opportunity.
6. `RAVANA_SCHEMA=1` (a non-default, supported env opt-in) makes
   `response_gen._schema_candidate` (`response_gen.py:7689-7734`) emit `'effect'`.
   The production hook builds traces from **all** trusted edges, induces
   hypotheses and ranks by `schema_association_scores_from_graph` — graph/GloVe
   cosine. It is **not** the default path.
7. The flag-on pick is permutation-stable (`effect` for orders
   `effect,whole,later,similar`, `similar,later,whole,effect`, `later,effect,similar,whole`),
   so it is not a first-option artifact. But the scratch-only reproduction of
   the hook's exact scorer inputs shows why: cosine(subject, type-label word)
   for `thaw` was `effect=0.323, whole=0.098, later=0.144, similar=0.132`
   against an instance floor of `0.032`; `effect` is the strict cosine winner.
   For the unrelated subject `zeppelin` the scores were
   `effect=-0.010, whole=0.203, later=0.267, similar=0.173`, and the production
   query picked **`later`** — exactly the cosine prediction. All four type
   labels are pre-existing graph nodes, not taught slots.
8. The disjoint-option control (`apple, banana, chair, table`) emitted
   **`banana`**, not `None`. TRA gate **C10** requires "a disjoint control must
   return `None`; fail ⇒ INSTRUMENT INVALID". The flag-on path therefore
   **fails C10** — it fails open by picking the highest lexical association
   even when no taught type can apply.
9. The scratch-only probe on the **real recovered edges** formed 60 traces and 3
   hypotheses keyed `cause` (support 35), `is_a` (20), `property` (5). The
   `is_a` hypothesis comes from pre-existing store facts (e.g.
   `phenomenon → "also called evaporative cooling"`, `edge_kind=web_fact`),
   **not** from the teaching turns. No hypothesis is keyed by an `effect/whole/
   later/similar` slot, and `SchemaHypothesis` has no type/slot field at all
   (`schema_induction.py:224-242, 275-296`). `rank_candidates_for_unseen_subject`
   (`schema_induction.py:354-423`) returns an *object* token by cosine, never a
   type label.
10. Cold reload is not the gap: all 8 new trusted edges survived; node count
    held at 1318 and edge count at 19,000 across `save()` + reload.

## INFERENCE (not established by the above)

- **I1** — The 2167→1318 node drop observed during the 10 teaching turns is
  consistent with junk maintenance pruning, but its cause was not isolated here.
  It did not delete any of the 8 trusted edges, so it is recorded as an
  observation, not a diagnosis.
- **I2** — The `is_a` hypothesis (support 20) reflects pre-existing authored/web
  facts in the copied store; a store with a different history would give a
  different support count. This experiment does not generalise to other stores.
- **I3** — Whether a future `is_a` write path could reuse `intake.intake`
  unchanged is plausible from E3 but unmeasured. No production code was added
  to test it, per the user rule.
- **I4** — The `RAVANA_SCHEMA=1` pick would be a confounded "success" if read as
  typed-slot selection: it is a lexical association *over type-label words*,
  the exact failure mode `TRA_BRAIN_DESIGN.md` §2 M3 warns against ("a type is a
  position in a relational structure, whereas GloVe proximity measures lexical
  affinity"). This is an interpretation of E7/E8, not a new measurement.

## Comparison with the TRA pre-registered gates

| Gate / rule | Observed here | Verdict |
|---|---|---|
| TRA §4.1.1–4.1.2 natural assertion written by unchanged intake with real provenance | Holds for CAUSE/PART_OF/PRECEDES/RESEMBLES; **fails for `is a` type assertions** (0 edges) | **FAIL for TRA teaching content** |
| TRA §4.1.3 recoverable after save + cold reload by exact identity/provenance | 8/8 trusted edges survive | PASS (for relation edges) |
| TRA §4.1.4 query selects the type from what is stored, no grader hook | Default path cannot decide; flag-on path decides by GloVe cosine over pre-existing label nodes | **FAIL** |
| **F4** forced single-token slot choice yields ~0 parseable decisions | 0/1 default, 3/3 flag-on but cosine-driven | F4 fires on the default path (F10-adjacent: mechanism absent) |
| **F7** cold-reload loss > 0.10 | 0/8 lost | does not fire — persistence is fine |
| **C9** cold-reload retention ≥ 0.80 | 8/8 = 1.00 | PASS |
| **C10** type-contrast sanity; disjoint control must return `None` | disjoint control returned `banana` | **INSTRUMENT INVALID under flag-on** |
| **C12** provenance integrity on taught edges | real `edge_kind`/`source`/`source_type`, no fabricated URL | PASS |
| **P1–P3, P5–P8, C1–C8, C11, C13** k-growth/foil/negative/deferral/full-bank gates | not run (feasibility probe, n=1 subject, 10 turns) | NOT ASSESSED — do not infer PASS |

## Recommendation

**DO NOT IMPLEMENT.** Do not open a production PR for TRA from this experiment.

The honest result is a **negative for "can the current unmodified engine acquire
typed role slots from conversational assertions and read them back"**:

- the engine writes provenance-trusted **relation** edges from natural assertions
  and retains them perfectly across cold reload (so storage and persistence are
  *not* the missing mechanism);
- it **cannot write the `is_a` type assertions** that TRA's teaching content
  requires — no such phrase exists in `chain_walker.py:946-955`;
- the default answer path **cannot express typed-slot selection at all**;
- the only opt-in path that emits a type label does so by **GloVe cosine over
  pre-existing type-label nodes**, fails the disjoint-option control (C10), and
  therefore carries no evidence of an acquired typed slot.

This is consistent with the three prior RSI scratch results: the gap is a
**knowledge/acquisition-and-representation gap**, upstream of ranking and
upstream of persistence. What is missing is (a) a write that stamps a
domain-general type label into provenance-trusted storage from an `is a`
assertion, and (b) a read whose candidate score comes from that stored slot
rather than from cosine to the label word. Until both exist in a scratch arm and
pass C10 with the disjoint control returning `None`, TRA cannot be scored.

**Next work item (still scratch-only):** design the narrow `is_a` assertion write
and a slot-keyed read as an in-process scratch monkeypatch on a copied store,
re-run this same experiment plus C10 (disjoint ⇒ `None`) and the mutation
controls (M-A/M-B) from `TRA_BENCHMARK_SPEC.md` §3, and only then consider a
production PR. No threshold, gate or item is to be relaxed to turn the current
result into a PASS.

## Honest caveats

- n = 1 typed-slot query subject (`thaw`) plus one unrelated control subject
  (`zeppelin`); 10 teaching turns and 2 boots. This is a mechanism probe, not
  the ≥200-family TRA verdict, and gives no Wilson CI or k-curve.
- The copied store is one snapshot of `weights/`; a deployment shipping
  `data/conceptnet/ont.pkl` would change the typed-edge census (`RSI_TYPED_ROLE_DESIGN.md` E7).
- The flag-on arm is a supported production env flag, not a monkeypatch; the
  only monkeypatch in the script is the clearly labelled scratch-only adapter
  that feeds real edges to the pure helpers (`scratch/tra_write_read_experiment.py`,
  `scratch_only_read_probe`). No production file was changed.
- The node-count drop during teaching is reported, not explained.

## Files

- `scratch/tra_write_read_experiment.py` — bounded 2-boot experiment script
- `scratch/tra_write_read_results.json` — full per-stage censuses, query rows, controls
- `scratch/tra_write_read_run.log` — captured stdout/stderr with `EXIT=0`
- this document
