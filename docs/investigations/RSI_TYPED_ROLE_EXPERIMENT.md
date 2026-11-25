# RSI typed role slots — scratch experiment 3 (oracle upper bound vs graph-derived types)

**Status: experiment only. DO NOT IMPLEMENT.** No file under `ravana/src/`,
`ravana_ml/src/`, `scripts/` or `tests/` was edited. No `CognitiveChatEngine`
was constructed, no `process_turn` called, no live engine booted, no store
written and no network used. The only production-behaviour change is an
in-process monkeypatch in the scratch script that adds one defaulted
`role_scores=None` parameter to the existing ranker; the congruence floor and
strict-margin gate remain the unmodified main code. Gold is used **only** to
score a pick after ranking.

Worktree: `C:/Users/Likhith/Documents/projects/ravana_rsi_types_exp` on
`experiment/rsi-typed-roles` from main `3028b4e`. The script asserted that
`ravana.chat.schema_induction.__file__` and `ravana.chat.response_gen.__file__`
resolve to this worktree, not an editable install.

## What was run

```
cd C:/Users/Likhith/Documents/projects/ravana_rsi_types_exp
PYTHONHASHSEED=0 PYTHONPATH="<worktree>/ravana/src;<worktree>/ravana_ml/src;<worktree>" \
  C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/typed_role_experiment.py > scratch/typed_role_run.log 2>&1
```

`EXIT=0`. Inputs (all read-only; the worktree has no `weights/` or GloVe data):

| Input | Use |
|---|---|
| `scratch/rsi_focus_live_rows.json` (384 rows) | same recorded focused-live items as experiments 1–2 |
| `scratch/role_binding_symbolic_results.json` | exp-2 comparator for the baseline reproduction check |
| `scratch/rsi_teach_form_probe2.log` | recorded trusted CAUSE census (8 engine edges) |
| `data/ravana_glove_cache.npz` | GloVe-64, used **only** for the A0 baseline floor |
| `weights/ravana_weights.db` (`mode=ro`) | read-only store census and A3 type map |
| `scripts/measure_learn5_rsi.py` (worktree) | regenerate `build_items()` for the A1/A2 join |

16 cases were replayed: 4 families × {near, far, foil, negative}, k=8, one
item per family per condition (`rsi_000` CAUSE, `rsi_001` PART_OF, `rsi_002`
PRECEDES, `rsi_005` RESEMBLES), same selection/dedup rule as experiments 1–2.
Both floor treatments are reported: `*_prodfloor` keeps the production GloVe
congruence floor (min demo-pair cosine), `*_symfloor` rescales it to the
arm's own typed congruence floor. Strict margin `min_margin=0` unchanged.

## Instrument validation (before any score) — F7

| Check | Result |
|---|---|
| Synthetic typed control: `car` typed `vehicle`, `petal` typed `plant` | scores `car=1.0, petal=0.0`, pick `car` ✅ |
| Disjoint control: no candidate carries the demo type | pick `None` ✅ |
| `passes` | `true` |
| A0 recomputed vector baseline matches exp-2 on 16/16 | **16/16** ✅ |
| Item-bank `(id, condition)` join failures | **0** ✅ (subject, candidates, gold, demos, domain all reproduced) |
| `F7_OK` | `true` |

The instrument is live: the typed scorer can move a production pick and fails
closed when the type is absent. The nulls below are informative, not wiring
failures.

## Arms — ORACLE / DEPLOYABLE separation

| Arm | Supplies candidate score | Status |
|---|---|---|
| **A0** | production vector baseline (`schema_association_scores_from_graph`) | Control; reproduced 16/16 |
| **A1 ORACLE-DOMAIN** | regenerated item-bank `domain`/`far_domain` tag | **ORACLE / NOT DEPLOYABLE** — the engine never receives these labels (item bank B4/B6); constant within a case |
| **A2 ORACLE-ANSWER-KEY** | `world_model[(rel, subject)]` membership | **CONTAMINATION / CEILING ONLY** — this is the gold answer key; never capability evidence |
| **A3 GRAPH-DERIVED** | Jaccard of stored `is_a`/definition types | **the only deployable candidate** — uses only information in the read-only store |

## Result — per-arm correct counts (foil/negative correct = reject)

Totals over 4 families per condition (n=4 each):

| Arm | near | far | foil (reject) | negative (reject) |
|---|---|---|---|---|
| A0 production baseline | **2/4** | **0/4** | 0/4 | 4/4 |
| A1 oracle-domain — prodfloor | 0/4 | 0/4 | 4/4 | 4/4 |
| A1 oracle-domain — symfloor | 0/4 | 0/4 | 4/4 | 4/4 |
| A2 oracle-answer-key — prodfloor | 4/4 | 4/4 | 4/4 | 4/4 |
| A2 oracle-answer-key — symfloor | 4/4 | 4/4 | 4/4 | 4/4 |
| A3 graph-derived — prodfloor | 0/4 | 0/4 | 4/4 | 4/4 |
| A3 graph-derived — symfloor | 0/4 | 0/4 | 4/4 | 4/4 |

Key contrasts (correct/n):

| Contrast | A0 | A1 oracle-domain | A2 oracle-answer-key | A3 graph-derived |
|---|---|---|---|---|
| PART_OF + PRECEDES near+far | 0/4 | **0/4** | **4/4** | **0/4** |
| far across all 4 families | 0/4 | **0/4** | **4/4** | **0/4** |
| near across all 4 families | 2/4 | 0/4 | 4/4 | 0/4 |

Foil/negative notes: A0 admits 4/4 foils (the known baseline leak). A1/A3
"4/4 foil" is **fail-closed rejection by a constant-zero or constant-tie
score**, not a type matcher success; negative rejection is a gate property
(negated demos admit no hypothesis). A2's 4/4 everywhere is the answer-key
ceiling, not a mechanism.

### Per-case near/far picks (gold in parentheses)

| Family | Cond | Subject | Gold | A0 | A1 oracle-domain | A2 oracle-answer-key | A3 graph-derived |
|---|---|---|---|---|---|---|---|
| CAUSE | near | thaw | mist | mist ✅ | None ❌ | mist ✅ | None ❌ |
| CAUSE | far | role | ceremony | election ❌ | None ❌ | ceremony ✅ | None ❌ |
| PART_OF | near | thunder | tempest | chill ❌ | None ❌ | tempest ✅ | None ❌ |
| PART_OF | far | custom | contract | ceremony ❌ | None ❌ | contract ✅ | None ❌ |
| PRECEDES | near | cloud | typhoon | glow ❌ | None ❌ | typhoon ✅ | None ❌ |
| PRECEDES | far | election | ceremony | law ❌ | None ❌ | ceremony ✅ | None ❌ |
| RESEMBLES | near | drizzle | chill | chill ✅ | None ❌ | chill ✅ | None ❌ |
| RESEMBLES | far | market | guild | election ❌ | None ❌ | guild ✅ | None ❌ |

## Why the arms behaved this way (instrument inputs/outputs)

* **A1 scores are constant within a case.** Every candidate in a case comes
  from one object pool, so all four candidates score `1.0` (e.g. CAUSE far:
  `ceremony=council=election=team=1.0`). The unmodified strict ranker sees a
  four-way tie and emits `None`. This reproduces D1: the `domain`/`far_domain`
  labels carry **no candidate-discriminating type signal**.
* **A2 scores are exactly the answer key.** For CAUSE far, `ceremony=1.0` and
  the other three `0.0`; the ranker picks the gold. This proves the *only*
  label in the bank that separates gold from foils is `world_model` — i.e. the
  answer itself. A2 is reported only as a ceiling and is retracted as
  capability evidence.
* **A3 scores are `0.0` on all four candidates of all 16 cases.** The
  demonstration type union contains tokens like `layer`, `multicolored`,
  `managed`, `pattern`, `sound` (copula heads pulled out of definitions), and
  no candidate shares them, so every Jaccard is zero and the ranker fails
  closed.

### A3 substrate census (read-only store)

| Observation | Measured value |
|---|---|
| Nodes / node types | 2,167 / all `concrete` |
| Relation types | `semantic` 14,760; `temporal` 1,190; `causal` 1,028; `contrastive` 638; `analogical` 569; `inferred` 375; `episodic` 184; **`is_a` 169**; `located_in` 47; `has_property` 12; `property` 5; `part_of` 5; `contextual` 2 |
| Usable `is_a` edges after excluding junk | **20** of 169 (138 multi-word junk targets, 11 dangling endpoints rejected) |
| Usable `is_a` edges touching the RSI vocabulary | **0** |
| Definitions mirrored | 588; **405** copula-first `is_a` heads recoverable |
| RSI object words that are store nodes | 32 of 144 |
| RSI case words with any usable type | **7**: `climate→pattern`, `cloud→managed`, `frost→layer`, `rainbow→multicolored`, `role→behaviors`, `team→group`, `thunder→sound` (all from definitions, **zero** from `is_a` edges) |
| Gold candidates with any usable type | **0 of 7** (`mist, tempest, typhoon, ceremony, contract, guild, chill`) |

**Coverage-gate nuance, reported honestly.** The design's pre-registered
numeric gate was "< 5 usable case words ⇒ INFORMATIVE NULL — substrate
absent". Measured coverage is **7 ≥ 5**, so the gate did not fire: this is
**not** reported as a substrate-absent null. It is a stronger negative: type-
like information exists for 7 case words, but (i) it consists of definition-head
artifacts and coarse attributes, not role slots (`effect/whole/later/similar`);
(ii) it is absent on **every** gold transfer candidate; and (iii) A3 still
scores 0/4 on near and far under both floors. No scorer tuning was attempted
to manufacture a signal.

## Answers to the three questions

**(a) Do oracle typed slots fix PART_OF/PRECEDES and far=0?**
Only the answer-key oracle does. **A1 oracle-domain does NOT**: PART_OF/PRECEDES
near+far stays **0/4** and far across all families stays **0/4**, because the
domain tag is constant inside a case and the strict ranker emits `None`.
**A2 oracle-answer-key** reaches **4/4** on both contrasts — but it consumes
`world_model[(rel, subject)]`, the gold mapping, so it is a contamination
ceiling and **not a deployable mechanism**. The practical reading: the
information that would fix these cases is precisely the gold target, not a
coarse domain category.

**(b) Does any graph-derived type signal exist that could justify an ontology path?**
**No.** A3, the only arm using information actually in the store, stays at
**0/4** on PART_OF/PRECEDES near+far and far overall, with all scores `0.0`,
foils/negatives held, and both floor treatments. D2's implementation threshold
(>0 on PART_OF/PRECEDES, no regression, flag-off equivalence, and a mutation
that flips the metric) is **not met**. The recovered types are definition-head
artifacts and none of the gold candidates carries a type. Restoring
`data/conceptnet/ont.pkl` is not a demonstrated fix: `graph_typing.py` can only
type labels already present as graph nodes, and 112 of 144 RSI words are not
nodes.

**(c) What is the smallest missing knowledge source if the answer is no?**
A **per-word typed role-slot label for the held-out vocabulary, stored in the
engine graph and reachable from the RSI path** — concretely, a world-model
mapping `(relation, word) → {words in the same role}` typed as
`effect / whole / earlier / similar`, ingested as provenance-trusted `is_a`-or-
role edges whose endpoints are single-word nodes. The bank's `domain` tag is
too coarse (constant within a case); `world_model` is the answer key and is
contamination. The gap is therefore a **knowledge/acquisition gap, not a
ranking gap**: no ranker change, floor change, margin change or item edit can
supply a category that is nowhere in the store. This matches Tse et al. (2007):
one-trial assimilation needs an already established schema; a better matcher
cannot substitute for the missing typed substrate.

## Recommendation

**DO NOT IMPLEMENT.** No ontology path, no typed-edge extraction path and no
production edit. Three scratch experiments now agree: GloVe role centres
regressed near (experiment 1), symbolic token/subword signatures left
PART_OF/PRECEDES at 0/4 and leaked foils (experiment 2), and typed slots add
nothing except the gold answer key (this experiment). The only arm that fixes
the gap is the contamination ceiling, which must never be shipped. Keep the
unmodified production baseline.

The next work item is upstream of ranking: acquire a typed role-slot source
for the unseen vocabulary (an external world model / restored ConceptNet with
single-word endpoints), verify in scratch that it puts a usable type on the
gold candidates, and only then re-run A3 with the same gates. Until that
source exists, the honest result is 0/4.

## Brain grounding

No new citation was researched. The verified ledger in
`docs/investigations/RSI_ROLE_BINDING_DESIGN.md:86-129` is used:
Gentner 1983 (*Cognitive Science* 7(2):155–170) — structure mapping transfers
relations, not object attributes; Gick & Holyoak 1983 (*Cognitive Psychology*
15(1):1–38) — two converged analogs support a schema (`min_support=2` retained);
**Tse et al. 2007 (*Science* 316(5821):76–82)** — rapid assimilation requires
an already established neocortical schema, the direct reason a missing typed
substrate cannot be replaced by a better ranker; McClelland, McNaughton &
O'Reilly 1995 (*Psychological Review* 102(3):419–457) — fast episodes vs slow
regularities, why demonstrations stay immutable and schemas stay unassertable;
Webb, Turney & Dennis 2016 (*Neuropsychologia* 93(Pt A):61–75) and Johnson,
Hashtroudi & Lindsay 1993 (*Psychological Bulletin* 114(1):3–28) — schema
congruence raises false alarms, why `world_model` stays an oracle.

## Honest caveats

* Static, recorded cases only: n=1 per family/condition (16 total),
  hypothesis-level evidence, not a suite verdict.
* The store census is one persisted snapshot (`weights/ravana_weights.db`), not
  a universal property of every RAVANA store. `data/conceptnet/ont.pkl` is
  absent here; a deployment shipping it would change the typed-edge census.
* A1/A2 depend on regenerating `build_items()` with `PYTHONHASHSEED=0`; the
  `(id, condition)` join was verified with zero failures, so the labels are
  the item bank's own, not invented.
* F7 passed, so the zeros are informative. No gate, threshold, floor, margin,
  item or gold was changed to obtain any number.

## Files

* `scratch/typed_role_experiment.py` — script (scratch only)
* `scratch/typed_role_results.json` — full per-case scores, floors, picks,
  store census, join check and the 16/16 reproduction check
* `scratch/typed_role_run.log` — captured stdout
* this document
