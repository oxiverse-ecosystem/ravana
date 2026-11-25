# RSI role binding — self-contained scratch experiment (no production edits)

**Status: experiment only.** No file under `ravana/src/`, `scripts/` or `tests/`
was edited. No `CognitiveChatEngine` was constructed, no `process_turn` was
called, no `pytest` suite was run, and no live engine probe was booted. The
only production behaviour change is an **in-process monkeypatch** inside the
scratch script that adds one defaulted `role_scores=None` argument to the
existing ranker; gates, floors and margins are the unmodified main code.

Worktree: `C:/Users/Likhith/Documents/projects/ravana_rsi_rolebind` on
`feat/rsi-role-binding` from main `dbb3e78`.

## What was actually run

```
cd C:/Users/Likhith/Documents/projects/ravana_rsi_rolebind
PYTHONPATH=".../ravana_rsi_rolebind/ravana/src;.../ravana_ml/src;.../ravana_rsi_rolebind" \
  C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/role_binding_experiment.py
```

The script printed and asserted that
`ravana.chat.schema_induction.__file__` and `ravana.chat.response_gen.__file__`
resolve to the **worktree**, not the main editable install.

Inputs (all read-only, from the main repo):

| Input | Use | SHA-256 (first 16) |
|---|---|---|
| `scratch/rsi_focus_live_rows.json` (384 rows) | recorded focused-live items: demos, subject, options, gold | `c9c49326aa340525c` |
| `scratch/rsi_teach_form_probe2.log:158` | recorded real engine census: `natural trusted 127 [('gale','rainbow'), … ('monsoon','frost')]` — 8 trusted CAUSE edges | `92eed7e41b3b6f3f7` |
| `data/ravana_glove_cache.npz` | the engine's real 400,001-word, 100D→64D projected GloVe vectors, loaded through unmodified `ravana.ontology.attribute_encoder.build_glove64_lookup` | `92b3c65ddb779ac92` |

16 cases were replayed: 4 families × {near, far, foil, negative}, k=8,
one item per family per condition (`rsi_000` CAUSE, `rsi_001` PART_OF,
`rsi_002` PRECEDES, `rsi_005` RESEMBLES). Gold was used **only** to score the
pick after ranking; it never entered a trace, centre or candidate score.

## Method

* **Baseline (vector-only):** unmodified `schema_induction.traces_from_edges`
  → `induce_schema_hypotheses` → `response_gen.schema_association_scores_from_graph`
  (GloVe cosine of subject↔candidate; instance cosines set the congruence floor)
  → unmodified `rank_candidates_for_unseen_subject` (floor, positivity, strict
  margin, no ties).
* **Role-aligned:** scratch `role_aligned_scores(...)` builds an r1 centre
  (mean unit-normalised demo SOURCE vectors) and the **complementary r2 centre**
  (mean demo TARGET vectors), applies the subject slot-typing gate at the
  observed within-role lower bound (floored at 0), and scores each option by
  `cos(vec(option), r2 centre)` **only**. Scores are computed per relation
  family, then the best family fit per candidate is passed to the *same*
  production ranker via the monkeypatched `role_scores` argument. No candidate
  is compared to the query subject.
* **Trust model:** CAUSE near/far/foil use the 8 **actually recorded** trusted
  engine edges. PART_OF / PRECEDES / RESEMBLES use synthetic trusted edges from
  the recorded demonstrations — justified by the write-side fix's 28 fake tests
  in `test_rsi_all_relations.py`, but **not** backed by a recorded live edge
  census for those families. Negated demonstration rows produce **no** trusted
  edges, in both arms.
* **Instrument validation (F7):** a synthetic, clearly-labelled control with
  role-separable vectors gives baseline `warmth` (wrong) → role `engine`
  (right). The scorer is live; a null result is not a wiring failure.

Full per-row scores, floors, margins, gate values and picks are in
`scratch/role_binding_results.json`; stdout is in `scratch/role_binding_run.log`.

## Result (n=1 per family × condition — mechanism evidence, not a suite verdict)

| Condition | baseline correct | role correct |
|---|---|---|
| near | 2/4 | **1/4** |
| far | 0/4 | **1/4** |
| foil (must reject) | 0/4 | 0/4 |
| negative (must reject) | 4/4 | 4/4 |

Per family:

| Family | near: baseline → role (gold) | far: baseline → role (gold) | foil: baseline → role | negative |
|---|---|---|---|---|
| CAUSE | mist → **mist** (mist) ✅ both | election → **None** (ceremony) ❌ both | thunder → mist ❌ both | reject both ✅ |
| PART_OF | chill → chill (tempest) ❌ both | ceremony → habit (contract) ❌ both | shower → frost ❌ both | reject both ✅ |
| PRECEDES | glow → glow (typhoon) ❌ both | law → **ceremony** (ceremony) ✅ role only | breeze → storm ❌ both | reject both ✅ |
| RESEMBLES | chill → **blizzard** (chill) ❌ role regression | election → election (guild) ❌ both | blizzard → blizzard ❌ both | reject both ✅ |

### Honest verdict

**The design does not fix the stated gap.** P1 (PART_OF near) and P2
(PRECEDES near) are **not supported**: both still pick the wrong option.
PART_OF far changes from `ceremony` to `habit` while the gold is `contract`. P3 (CAUSE near) holds;
P5 gets one accidental hit (PRECEDES far, `ceremony`) out of four far items —
far = 0 is **not** generally fixed. P4 **fails**: RESEMBLES near regresses
from the correct `chill` to `blizzard`. P7 (negative rejection) holds in all
four families. P6 (foil rejection) is **not demonstrated at the ranking
primitive**: both arms emit a candidate on all four foil rows; the live
engine's foil rejection must come from gates outside this path, which this
experiment does not model.

Why: the r2 centre is still a GloVe centroid. On PART_OF near, `chill` scores
0.7065 against the demo-target centre while gold `tempest` scores 0.3925; on
RESEMBLES near, `blizzard` 0.6586 beats `chill` 0.6052. The "complementary
role" is an *embedding neighbourhood*, not a structural role — exactly F3,
the extrapolation the design said would kill this design if it failed. The
subject gate behaves fail-closed where it rejects (CAUSE far: fit 0.0221 <
lower bound 0.0920 → no answer rather than a wrong answer), but that is a
safe failure, not a fix.

### Falsifier ledger for the parent's live probe

* **F3 — triggered.** GloVe role centres do not separate gold from distractors
  for PART_OF / RESEMBLES near; do not widen the floor or relabel cosine as
  structure.
* **F5 — warning, not a live confabulation claim.** Role scoring *changed* the
  foil pick in 3/4 families and broke CAUSE far by emitting `None`. No trust
  predicate, threshold or benchmark was touched. If a live arm reproduces a
  foil admission, stop per F5.
* **F6 — not applicable at n=1/family**; no saturation was observed here.
* **F7 — passed** (sanity control), so these nulls are informative.
* **F1/F2/F4 — not addressed here** (write-side and articulation), and remain
  untested by this scratch experiment.

## What remains missing

1. A **non-embedding role representation**: the complementary slot must be
   constrained by the relation's type/world-model compatibility (e.g. which
   candidate can be a `part`/`effect`/`later` term), not by mean GloVe.
2. An **articulation path** that consumes a strict rank proposal without
   bypassing H3 deferral, foil rejection or negative-rule rejection; the
   rank primitive alone cannot reject foils in this replay.
3. **More items**: one item per family/condition is a hypothesis. Run at least
   ~20 held-out items per family before treating any delta as an effect.
4. Recorded trusted-edge censuses for PART_OF / PRECEDES / RESEMBLES, so the
   non-CAUSE arms stop depending on a synthetic trust model.

## Recommendation

Do **not** implement this design into production as-is. The ONLY measured gain
(1/4 far) came with a near regression (2/4 → 1/4) and changed foil picks;
that fails the design's own "no regression" predictions P3/P4 and its
gate-preserving intent. Keep this branch as evidence, not as a PR.

Files: `scratch/role_binding_experiment.py` (script, sha256
`f6edd22d60c764365`), `scratch/role_binding_results.json` (data, sha256
`dda4eec2aa7fec307`), `scratch/role_binding_run.log` (stdout), this document.
