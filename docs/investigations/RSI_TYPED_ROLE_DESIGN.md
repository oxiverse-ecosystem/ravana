# RAVANA typed-role substrate audit and third scratch experiment plan

**Status: audit + design only.** No engine process was booted, no `process_turn`
called, no pytest run, no gate/threshold/item bank/benchmark edited, and no file
under `ravana/src/`, `ravana_ml/`, `scripts/` or `tests/` changed. The only
artifact is this document, written into worktree
`C:/Users/Likhith/Documents/projects/ravana_rsi_types` on branch
`experiment/rsi-typed-roles-design` from main `39a0288`.

Source-scope note: `RSI_SYMBOLIC_ROLE_EXPERIMENT.md` and
`scratch/rsi_focus_live_rows.json` are not on `main` (they live on
`docs/rsi-symbolic-role-experiment` at `961963f`); they were read from the main
checkout. The read-only store census below used
`C:/Users/Likhith/Documents/projects/ravana/weights/ravana_weights.db`
(the store the recorded focused-live runs used) via a `mode=ro` SQLite
connection; the new worktree has no `weights/` directory.

**Question.** Experiments 1 and 2 both say DO NOT IMPLEMENT. GloVe role centres
regressed near (`RSI_ROLE_BINDING_EXPERIMENT.md:85-105`); symbolic relation +
r2-token signatures left PART_OF/PRECEDES at 0/4 and leaked foils
(`RSI_SYMBOLIC_ROLE_EXPERIMENT.md:86-118, 162-177`). Both trace the failure to
the same fact: gold candidates are deliberately unseen demonstration tokens and
`relation_type` is constant inside a case, so neither associative vectors nor
token signatures carry transfer signal. The missing mechanism is a **typed,
domain-general role slot**. Does RAVANA already have a substrate for it?

---

## 1. EVIDENCE vs INFERENCE: what type-like information exists

### 1a. EVIDENCE — the code has a typed-edge *vocabulary*, and the RSI path does not read it

| # | Observed fact | File:line |
|---|---|---|
| E1 | Graph edges carry a `relation_type` string; defaults include `semantic/causal/temporal/analogical/...`, and the table has no type-dimension for a role slot. | `ravana_ml/src/ravana_ml/graph.py:166-218`, `:1694-1750` |
| E2 | `ConceptNodeType` declares six node types (`concrete/abstract/relational/physical/propositional/episodic`) but production never assigns a non-default value: the only `node_type=` write is DB serialisation/deserialisation. | `graph.py:48-55`; `ravana/src/ravana/storage/db.py:192,232` |
| E3 | The intake/web write path maps `is_a`, `has_property`, `part_of`, `located_in` to distinct `relation_type`s; OpenIE maps copulas (`is/are/was/be/means/...`) to `is_a`. Typed edges are writeable. | `learn/intake.py:262-288`; `web/web_to_graph.py:53-62`; `web/openie.py:40-48` |
| E4 | A ConceptNet injection module exists and emits `isa/has_property/capable_of/used_for/part_of`; it requires pre-existing graph nodes on both ends and skips feature targets absent from the label index. | `ontology/graph_typing.py:45-54`, `:118-196`, `:154-170`, `:179-181` |
| E5 | The injection is boot-wired on BOTH load and cold-start paths. | `chat/engine.py:2209`, `:2287`; `chat/engine_graph.py:1022-1064` |
| E6 | It is gated on `data/conceptnet/ont.pkl`; the loader returns `None` when absent, `_cn_ontology` is then `None` and `use_conceptnet_primary` is auto-disabled. | `engine_graph.py:820-842`; `engine.py:2152-2165` |
| E7 | `data/conceptnet/ont.pkl` does **not** exist in this checkout or anywhere under `C:/Users/Likhith/Documents/projects` (`.gitignore:29` excludes the ~1.3 GB corpora). Therefore `_typed_edges_bootstrap()` returns 0 on every boot. | `.gitignore:29`; filesystem search; `engine_graph.py:1035-1038` |
| E8 | Hand-authored type-like seeds exist but only for 15 unrelated concepts (`sky, blue, cat, dog, music, water, sun, gravity, tree, earth, star, light, plant, animal, human`) with `is_a`/`property`/`causal` edges and prose definitions. No RSI word appears. | `data/common_facts.json:1-122`; `engine_generation.py:2763-2818`; `engine_graph.py:966-1004` |
| E9 | The RSI answer path is relation-string + vector only. `TrustedTrace` = `(declaration, stamp, trace_id)`; `SchemaHypothesis` = `(relation, support, instances, assertable=False)`; `induce_schema_hypotheses` groups **solely** by `declaration.relation`. No type/category/slot field exists. | `chat/schema_induction.py:224-242`, `:275-296`, `:316-351`, `:354-423` |
| E10 | The graph adapter converts edges to `EdgeView(source, relation_type, target, edge_kind, edge_source)` and discards all node-level type information; the scorer computes only cosine of stored/GloVe vectors. | `chat/response_gen.py:196-218`, `:221-285`, `:7689-7734` |
| E11 | The RSI hook is a pure function of those scores; it is default-OFF (`RAVANA_SCHEMA`) and defers to a directly taught edge. | `schema_induction.py:62-93`; `response_gen.py:7697-7716`, `:7755-7768` |

### 1b. EVIDENCE — read-only census of the persisted graph (the store used by the recorded runs)

| # | Observed fact | How obtained |
|---|---|---|
| E12 | 2,167 nodes, 18,984 edges. Relation types: `semantic` 14,760; `temporal` 1,190; `causal` 1,028; `contrastive` 638; `analogical` 569; `inferred` 375; `episodic` 184; **`is_a` 169**; `located_in` 47; `has_property` 12; `property` 5; `part_of` 5; `contextual` 2. No `capable_of`, no `used_for`, no `isa`. | `SELECT relation_type, COUNT(*) ... GROUP BY` on `ravana_weights.db`, read-only |
| E13 | **All 2,167 nodes have `node_type='concrete'`** — the six-type vocabulary in E2 is inert. | `SELECT node_type, COUNT(*) ... GROUP BY` |
| E14 | Of 30 RSI words checked, only 10 exist as nodes: `climate, cloud, storm, rainbow, role, dew, glow, custom, team, law`. None of the near/far gold tokens (`mist, thaw, tempest, typhoon, ceremony, contract, guild, chill, hail, blizzard`) is a node. | `SELECT id,label FROM concepts WHERE lower(label) IN (...)` |
| E15 | Exactly two `is_a` edges touch those 10 nodes: `dew → "moisture forms result condensation"` (multi-word junk phrase, weight 0.023) and `glow → <node id 1668>` (dangling: no row in `concepts`). **Zero usable typed edges on the RSI vocabulary.** | join `edges`→`concepts`, filtered to typed relations |
| E16 | 588 definitions are mirrored in `metadata['definitions']`; only 6 of the 30 RSI words have one: `rainbow, cloud, role, team, dew, climate`. Of those, `cloud` is the **wrong sense** ("a managed service for storing unstructured data"), `dew` is "used as a symbol", and the critical gold tokens have none. | read-only `pickle.loads` of the `metadata` row |
| E17 | `definition_proposition.py:1-45` reports 324 stored definitions with 98/324 (30%) copula-first recoverable to `is_a`; that is a text-parsing capability over definitions that mostly are not RSI words. | source docstring + E16 |

**Key EVIDENCE finding:** RAVANA has the *vocabulary and plumbing* for typed
edges, but in the measured store the type substrate is effectively empty where
RSI needs it — 169/18,984 typed edges (0.9%), all nodes `concrete`, zero usable
typed edges on the RSI vocabulary — and the RSI hypothesis path never reads a
type edge at all (E9–E11). The ConceptNet path that would populate it is
disabled by a missing data file in this checkout (E6–E7).

### 1c. INFERENCE — explicitly not established by the above

* **I1 (inference):** *If* `ont.pkl` were restored, `graph_typing.py` could add
  typed edges only for labels already in the graph with a matching parent also
  in the graph (`graph_typing.py:154-170`). E14 says 20/30 RSI words are not
  nodes, so coverage would remain partial. Not measured; do not claim.
* **I2 (inference):** ConceptNet's anchor roots are `living / physical_object /
  time / abstract / event / social` (`ontology/conceptnet.py:100-131`,
  `category_of` at `:281-311`; `ontology/derived.py:56-90,226-254`). Those are
  coarse *ontological* categories, not role slots like `effect/whole/later/
  similar`. Even if populated, they are constant within a candidate set (all
  near candidates share the demo domain; all far candidates share the far
  domain — see §2), so they would reproduce experiment 2's constant-feature
  failure. Plausible, unverified.
* **I3 (inference):** `DerivedOntology.category_of` requires caller-supplied
  `category_seeds` (`derived.py:226-254`); it is a geometric classifier, not a
  stocked ontology. It has the same GloVe-centroid weakness experiment 1
  falsified.
* **I4 (inference):** The only reason to expect typed edges on web words is
  OpenIE copula extraction (E3); whether the web loop actually ran on any RSI
  word is unknown, and E15 suggests it did not produce usable types.

---

## 2. Does the item bank carry type labels the engine never sees?

### 2a. EVIDENCE

| # | Observed fact | File:line |
|---|---|---|
| B1 | The bank defines 6 disjoint 24-word domain vocabularies: `weather, body, machine, social, geometry, plant`; overlap is asserted absent. | `scripts/measure_learn5_rsi.py:137-169` |
| B2 | Each item stores `domain=dname` (demo pool) and `far_domain=fname`, deterministically `+3` domains away; `validate_items` asserts they differ. | `:243-246`, `:295-318`, `:372-373`; `tests/unit/test_learn5_rsi.py:84` |
| B3 | Each item stores `world_model[(rel, subject)] = {targets}`, built from the 8 demo pairs plus the exact/near gold, the far gold and the negative counterexample target. `validate_items` checks every gold against it. | `:281-284`, `:317`, `:343-350` |
| B4 | `RealBackend.present()` sends the engine **only** demo assertion sentences and the forced-choice question. No `domain`, `far_domain`, `world_model` or type label is ever sent. | `:542-590` |
| B5 | The engine-graded arm explicitly grades parsed `process_turn` output with "no world-model shortcut, no demo triple". | `:665-668` |
| B6 | The recorded row `input` keeps only `demonstrations, query_subject, candidates, domain, foil_relation`; `far_domain` and `world_model` are dropped before serialisation. | `:712-726`; census of all 384 rows in `scratch/rsi_focus_live_rows.json` (`domain`: weather 320 / social 64; no other type-like key) |
| B7 | Within a case candidates come from one pool: `exact_cands` from `OBJECTS[dname]`, `far_cands` from `OBJECTS[fname]`. The domain label is therefore **constant across the four candidates**. | `:286-293` |
| B8 | `world_model[(rel, query_subject)]` contains exactly the gold target for near/far; it is the answer key. It is consulted only by the fake/chance/naive/counterexample oracles. | `:281-284`, `:723`, `:663-710` |

### 2b. Legitimacy: teaching material or contamination?

* **`domain` / `far_domain`: legitimate only as demo-time teaching material; inert as a ranker.** They are authored pool tags fixed before any run (B1–B2), so teaching them during demonstrations ("these are weather words") is legitimate *teaching material*, not scaffolding — it is the same class as the authored triples, and no control is bypassed. But (a) the engine never receives them today (B4), and (b) by B7 the label is constant within the candidate set, and by B5/B6 a query-time tag would be test-set metadata rather than something induced from the demonstrations. Feeding `far_domain` at query time as a side channel is therefore **test-set leakage even though it does not identify the gold**; feeding `domain` in demos is the only defensible form. An experiment that uses these labels must mark them ORACLE / NOT DEPLOYABLE because the engine never sees them.
* **`world_model`: contamination, without qualification.** By B8 it is the answer key keyed by exactly the query `(relation, subject)`. Exposing it to a matcher measures answer lookup, not type binding. It may appear only as a clearly-labelled oracle ceiling (arm A2 below) and must never be reported as capability evidence or used to justify an implementation.

**Conclusion:** the item bank contains a coarse `domain`/`far_domain` tag, but it is constant across candidates and therefore cannot supply the type contrast that experiment 2 found missing. The only type-bearing label that *would* separate gold from foils is `world_model`, and that is the answer key. No legitimate item-bank label supplies the typed role slot.

---

## 3. Third scratch experiment — design, runnable on recorded cases only

**Harness constraints (same as experiments 1 and 2):** one process, no
`CognitiveChatEngine`, no `process_turn`, no store writes, no network. Replay the
same 16 cases (4 families × {near, far, foil, negative}, k=8, items `rsi_000`
CAUSE, `rsi_001` PART_OF, `rsi_002` PRECEDES, `rsi_005` RESEMBLES) from
`scratch/rsi_focus_live_rows.json` with the same selection/dedup rule. Gold
scores a pick **after** ranking. Feed every arm to the *same* unmodified
`rank_candidates_for_unseen_subject` through the already-used one-argument
`role_scores=None` monkeypatch; keep the production congruence floor and a
strict `min_margin=0`. Report both `prodfloor` and `symfloor` treatments as
experiment 2 did. Assert `ravana.chat.schema_induction.__file__` and
`response_gen.__file__` resolve to the worktree, not an editable install.

**Instrument validation before any score (F7):** (i) recomputed vector-only
baseline must reproduce experiment 1/2's recorded baseline on 16/16 cases;
(ii) a synthetic, clearly-labelled control in which the supplied type label
truly separates two candidates (e.g. `car` typed as `vehicle`, `petal` as `plant`)
must flip a pick. If either fails: `INSTRUMENT INVALID — no score`.

### Arms

| Arm | What supplies the candidate score | Status |
|---|---|---|
| **A0** | Recomputed production vector baseline (`schema_association_scores_from_graph`). | Control; must reproduce 16/16. |
| **A1 — ORACLE-DOMAIN** | Regenerate `build_items()` deterministically (`PYTHONHASHSEED=0`) and re-join `domain`/`far_domain` onto each recorded row by `(id, condition)`. Tag each demonstration r2 token with the item's `domain`; score a candidate 1 if its condition-domain equals the demo-domain (near) or the item's `far_domain` (far), else 0. | **ORACLE / NOT DEPLOYABLE** — these labels never reach the engine (B4, B6). |
| **A2 — ORACLE-ANSWER-KEY** | Score 1 iff `(relation, subject, candidate)` is in the regenerated `world_model`, else 0. | **CONTAMINATION / CEILING ONLY** — the gold answer key (B8). Reports the upper bound, never a capability. |
| **A3 — GRAPH-DERIVED TYPE (the only deployable candidate)** | Read-only copy of `weights/ravana_weights.db`. Build `type_map[word]` from (a) real `is_a` edges, excluding multi-word junk targets and dangling endpoints per E15, and (b) `is_a` heads recovered from `metadata['definitions']` using the existing copula logic in `learn/definition_proposition.py:125-163`. Score each candidate by Jaccard overlap of its type set with the union of demonstration-r2 type sets. | Deployable *in principle*: uses only information already in the graph. |

**Coverage gate (measure before scoring A3):** count RSI words with ≥1 usable
type from E12/E15/E16. Static census predicts ~0–2 typed words and 6 definition
words (one wrong-sense). If coverage < 5 usable words, A3 must be reported as
**INFORMATIVE NULL — substrate absent**, not as a failed matcher; no tuning of
the scorer is permitted to manufacture coverage.

**Foil/negative discipline:** A0 and A3 must keep foil 0/4 (reject) and negative
4/4 (reject). A1/A2 are diagnostics; any foil change is recorded, not repaired
by touching a gate. Do not lower floors, widen margins, edit items or relabel a
cosine as a type. n=1 per family/condition stays hypothesis-level.

---

## 4. Exact predictions and falsifiers

| ID | Pre-registered prediction | Result that **justifies** a type-extraction path | Result that **proves the item bank lacks the information** / falsifies |
|---|---|---|---|
| D1 | A1 candidate scores are constant within each case (max−min = 0 on all 16 rows, by B7), so the strict ranker emits `None`; PART_OF + PRECEDES near+far stays **0/4**. | A1 > 0 on PART_OF/PRECEDES with foils and negatives held → domain tags do carry a type contrast; investigate a demo-time tag path (still oracle until the engine can see the tags). | Constant scores + 0/4, with A0 reproduced and F7 passed → the bank's `domain`/`far_domain` labels carry **no candidate-discriminating type signal**. This is the expected outcome. |
| D2 | A3 type coverage < 5 usable RSI words; A3 fails closed on nearly all cases. | ≥20 RSI words with usable types **and** PART_OF/PRECEDES > 0 at k=8 **and** foils/negatives held **and** CAUSE/RESEMBLES unchanged **and** flag-off byte-identical **and** a single-label mutation moves the metric → implement a narrow type-extraction path. | Coverage ~0 (as E15/E16 predict), or coverage present but scores do not separate gold while A0 does → no deployable substrate; record the count and what the knowledge source would need to contain. |
| D3 | A2 = 4/4 on near+far while A1/A3 = 0/4. | A1 or A3 approaches A2 **without** `world_model` → a legitimate type signal exists. | A2 alone succeeds → the only type-bearing label in the bank is the answer key; exposing it is contamination. Retract any A2 number presented as capability. |
| D4 | Neither A1 nor A3 fixes PART_OF/PRECEDES, reproducing experiments 1 and 2. | Any non-oracle arm moves both families above zero with all controls held → the "typed slot" hypothesis gains its first positive evidence. | All non-oracle arms stay 0 → the gap is not in the ranking signature family; consistent with Gentner [3] and Tse [6]: transfer needs a pre-existing typed schema, and neither GloVe centroid, token signature, nor domain tag constitutes one. |
| D5 | No arm is implemented in production. | The full D2 conjunction above. | Any gain obtained by relaxing a trust gate, floor, margin, benchmark item or gold, or by consuming `world_model`, is a **false-memory analogue** [12,13] — stop and revert. |
| D6 | F7 synthetic control passes and A0 reproduces 16/16. | — | Otherwise `INSTRUMENT INVALID — no score`; do not interpret any arm. |

**Implementation threshold (single sentence):** build a type-extraction path
only if arm A3, with **no oracle labels**, raises PART_OF and PRECEDES above
zero at k=8 while foils/negatives stay perfect, CAUSE/RESEMBLES do not regress,
flag-off is byte-identical, and a type-label mutation flips the result.
Anything short of this is a documented negative.

---

## 5. Brain grounding (citations already verified in `RSI_ROLE_BINDING_DESIGN.md`)

No new citation was researched or added here. All numbers below are the
verified ledger entries in `docs/investigations/RSI_ROLE_BINDING_DESIGN.md:86-129`,
which resolved each source against publisher/PubMed/PMC metadata.

* **Gentner 1983**, *Cognitive Science* 7(2):155–170, doi:10.1207/s15516709cog0702_3
  ([3] in the ledger): structure mapping transfers relations, not object
  attributes, and mapping depends on representation syntax rather than domain
  content. Motivates a role slot defined by relation type, not by word identity.
* **Gick & Holyoak 1983**, *Cognitive Psychology* 15(1):1–38,
  doi:10.1016/0010-0285(83)90002-6 ([4]): two prior analogs, not one, yielded a
  schema that supported transfer — the reason `min_support=2` is retained.
* **Tse et al. 2007**, *Science* 316(5821):76–82, doi:10.1126/science.1135935
  ([6]): rapid one-trial assimilation depends on an *already established*
  neocortical schema. This is the direct brain-grounded reason a missing typed
  substrate cannot be substituted by a better ranker, and why D1/D2 predict
  failure until the substrate exists.
* **McClelland, McNaughton & O'Reilly 1995**, *Psychological Review*
  102(3):419–457, doi:10.1037/0033-295X.102.3.419 ([7]): fast episode storage
  versus slow regularity extraction — why demonstrations stay immutable edges
  and any schema stays an unassertable per-call view.
* **Bunge et al. 2005**, *Cerebral Cortex* 15(2):239–249,
  doi:10.1093/cercor/bhh126 ([9]): retrieval and relational integration are
  separable — why type-coverage (retrieval) is reported separately from
  ranking (integration), per the A3 coverage gate.
* **Webb, Turney & Dennis 2016**, *Neuropsychologia* 93(Pt A):61–75,
  doi:10.1016/j.neuropsychologia.2016.09.023 ([12]) and **Johnson, Hashtroudi &
  Lindsay 1993**, *Psychological Bulletin* 114(1):3–28,
  doi:10.1037/0033-2909.114.1.3 ([13]): schema congruence raises false alarms and
  source attribution is fallible — why `world_model` must stay an oracle and
  congruence must never assert or write.

Ledger note from that document still applies: the brief's
`10.1126/science.1135935` and `10.1207/s15516709cog0702_3` are the verified
DOIs for Tse 2007 and Gentner 1983 respectively; no unverified DOI is repeated
here.

---

## 6. Honest caveats

* This is a **static audit and a design**. Experiments 1 and 2 were not rerun;
  the focused-live rows were not regenerated; no engine, pytest or store write
  was used. The store census is of one persisted snapshot and is reported as
  such, not as a universal property of every possible RAVANA store.
* `data/conceptnet/ont.pkl` is absent here; a deployment that ships it would
  change the typed-edge census. This document does not claim what that census
  would be (I1).
* The item-bank join in A1/A2 requires regenerating `build_items()` from the
  same deterministic seed. If the regenerated `(id, condition)` keys do not
  reproduce the recorded gold labels, the join is invalid and that arm must be
  reported as `INSTRUMENT INVALID`, not scored.
* Falsifiers D1–D6 are predictions, not measured outcomes. No result in this
  document licenses a production change.

## Sources (in-repo, read for this audit)

* `docs/investigations/RSI_ROLE_BINDING_DESIGN.md` (verified citation ledger)
* `docs/investigations/RSI_ROLE_BINDING_EXPERIMENT.md` (experiment 1, DO NOT IMPLEMENT)
* `docs/investigations/RSI_SYMBOLIC_ROLE_EXPERIMENT.md` (experiment 2, DO NOT IMPLEMENT)
* `scratch/rsi_focus_live_rows.json` (384 recorded rows)
* `scripts/measure_learn5_rsi.py`, `tests/unit/test_learn5_rsi.py` (item bank)
* `ravana/src/ravana/chat/{schema_induction,response_gen,engine,engine_graph,engine_generation,brain_regions}.py`
* `ravana/src/ravana/ontology/{graph_typing,conceptnet,derived}.py`
* `ravana/src/ravana/learn/{intake,definition_proposition}.py`; `web/{openie,web_to_graph}.py`
* `ravana_ml/src/ravana_ml/graph.py`; `data/common_facts.json`; `.gitignore`
