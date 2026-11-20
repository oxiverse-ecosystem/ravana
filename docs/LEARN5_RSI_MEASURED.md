# RAVANA-LEARN-5 / RSI — measured result

**Status: PASS on all 21 pre-registered decision rules, with a hard saturation
caveat.** This document records an actual completed run of the RSI measurement
harness. It changes no weights, no gate and no fixture. Harness, tests and this
report are working-tree additions on main `c47ac09`. The measured front is
**saturated** — every capability endpoint sits at 1.00 — so the shared Tier-0
G6 gate flags it **INVALID as a progress signal**. Like PDR and WFA, this is a
demonstration that the specified decision rule is implemented correctly on the
authored distribution; it is **not** evidence of learning-shaped growth. Read
§6 before citing any number here.

## 1. What was measured

RSI (Rule Induction and Sealed Transfer) is the analogy / structure-mapping
component of RAVANA-LEARN-5. The mechanism is grounded in structure-mapping
theory — analogy maps **relations between objects, not object attributes**, and
systematicity prefers interconnected relational systems (Gentner 1983);
structural alignment underlies both analogy and similarity, while appearance
matches map attributes instead (Gentner & Markman 1997) — and in one-shot
generative structure induction (Lake, Salakhutdinov & Tenenbaum 2015). The
criterion is therefore **a relation preserved under filler substitution**, with
the permutation as the decisive control: if changing demonstration order or
object roles breaks performance, the system matched serial/surface structure
rather than the rule. Because an unsupported relation is exactly what the store
does not support, the answer must commit to the relation explicitly, not just
the filled target. Full design: `docs/BRAIN_LEARN5_DESIGN.md` §7 (lines 512–588);
reuse map: `docs/LEARN5_CODEMAP.md`.

The system under test is the structure-mapping decision layer implemented in
`scripts/measure_learn5_rsi.py` (`RSIComponent`, `run_item`). It is the same
kind of harness-level component as WFA's `WFAComponent` and PDR's
`PDRComponent`: it implements the design's decision rule over an authored rule
bank, it is **not** the RAVANA engine's own generation path. See limitation L2.

For a rule family with relation *R* and demonstrations over new objects, the
harness measures, at *k ∈ {1,2,4}*:

- `exact` — exact structural match on a held-out target with **new objects** in
  the same domain;
- `permuted` — same, but demonstrations are order-reversed and role-swapped
  (every directional demonstration is expressed in inverse form, e.g.
  `storm causes rainbow` → `rainbow is caused by storm`);
- `foil` — a **same-object wrong-relation** item (a demonstrated object is
  among the candidates, the subject has no *R* fact): the answer must expose
  the induced relation field and **reject** the item;
- `far` — exact structural match in a **new domain** sharing only the relation
  (the two domain vocabularies are checked disjoint at import);
- `chance` — a shuffled-relation-label arm with the true target, asserted.

Headline metric is the **exact structural match grader on the structured answer
object, never a prose regex**: for a positive probe, correct iff
`verdict == 'match' AND answer.relation == gold_relation AND answer.target ==
gold_target`. The relation field is checked against the *demonstration's*
induced relation, not just the filled target (design §7.2/7.4), so an answer
that copies a target under the wrong relation cannot pass. For a foil, correct
iff `verdict == 'reject'`.

## 2. Instrument and reuse (what was not reinvented)

- **Rule bank fixed before any run.** 100 authored rule families over six
  relations (CAUSE, PART_OF, PRECEDES, SUPPORTS, OPPOSITE, RESEMBLES) and six
  disjoint object domains. Split by
  `sha256("RAVANA-LEARN-5|RSI|<difficulty>|<item_id>")`, sorted within difficulty,
  first half PRIMARY / rest SEALED: **48 primary + 52 sealed** (both ≥ the
  design's n≥40). Item-bank manifest SHA-256
  `273b312c4f3b90db66884ed33be3f117eb6b2f69ed556e636871f6d099b33113`. No item
  was dropped after seeing an outcome; the split is computed at import and
  never changed.
- **No shared-fixture edit.** RSI needs no frozen web store. It reuses
  `scripts/front_gates.py::gate_saturation` (G6) unmodified for the
  informativeness flag; no front gate threshold was edited. It does not call
  the CAR/PDR/WFA helpers because RSI's endpoint is a structured relation field,
  not a graph chain, claim span or poison verdict.
- **Structure-mapping operations are explicit.** A surface-phrase lexicon maps
  forward (`causes`) and inverse (`is caused by`) forms to `(relation,
  polarity)`; canonicalisation swaps roles on inverse forms; induction takes the
  majority relation over canonicalised demonstrations; application looks up the
  target in an authored `(relation, subject) → {target}` world model.
- **One process, no engine, no save, no network.** There is no
  `CognitiveChatEngine` boot, no `eng.save()` and no network call; the
  "scratch data_dir / no eng.save on weights" constraint is satisfied
  vacuously and recorded as `engine_boot=false`, `save_attempts=0`,
  `network_attempts=0`. `PYTHONHASHSEED=0`; all selection is hash-based, no RNG.
- **No global patch at import** (skill §2b): the harness only defines counters;
  importing it in a shared pytest process is side-effect free.

## 3. Instrument controls (all six pass before any score)

| # | Control | Result |
|---|---|---|
| C1 | positive control: structural exact ≥ 0.90 | ok (1.00) |
| C2 | foil power: structural rejects ≥ 0.90 **and** naive surface matcher accepts (≤ 0.10 reject) | ok (1.00 / 0.00) |
| C3 | permutation power: naive clean−permuted gap ≥ 0.15 **and** structural gap ≤ 0.10 | ok (gap 1.00−0.33; structural 0.00) |
| C4 | chance ≤ 0.35 and far − chance ≥ 0.20 | ok (0.208 / +0.792) |
| C5 | relation field present in ≥ 95% of structural answers, both pools | ok (1.00 / 1.00) |
| C6 | grader paired check: good accepted; wrong relation, wrong target and accepted foil rejected; rejected foil accepted | ok |

C2 and C3 are the confound demonstrations: a plausible surface/order memoriser
**does** accept the same-object foil and **does** break under permutation, so
those two controls have power to detect the failures the design warns about
(§7.3–7.4). If any control fails, `main()` prints `INSTRUMENT INVALID` and exits
**without printing a score** (this was exercised during development: three
real probe bugs were found and fixed this way before any number was reported).

## 4. Measured numbers — primary and sealed side by side

Command: `PYTHONPATH=scripts PYTHONHASHSEED=0 uv run python
scripts/measure_learn5_rsi.py` (exit 0). Artifacts:
`scratch/learn5_rsi_probes.jsonl` (3,600 rows = 100 families × 3 k × 4
conditions × 3 arms), `scratch/learn5_rsi_summary.json`,
`scratch/learn5_rsi_gates.jsonl`.

| Metric (structural arm) | PRIMARY (n=48 families; 144 probes/cond) | SEALED (n=52 families; 156 probes/cond) | Criterion | Result |
|---|---|---|---|---|
| `exact` | 1.000 (144/144) | 1.000 (156/156) | ≥ 0.75 | PASS |
| `permuted` | 1.000 | 1.000 | ≥ exact − 0.10 | PASS |
| `exact − permuted` (paired) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | not < −0.15 | PASS |
| `foil` rejection | 1.000 | 1.000 | ≥ 0.85 (hard floor 0.70) | PASS |
| `far` | 1.000 | 1.000 | ≥ chance + 0.20 | PASS |
| `chance` (shuffled labels) | 0.208 (30/144) | 0.211 (33/156) | ≤ 0.35 | PASS |
| `far − chance` | +0.792 | +0.789 | ≥ +0.20 (hard floor +0.10) | PASS |
| relation field coverage | 1.000 | 1.000 | ≥ 0.95 | PASS |

The k-sweep is flat: every one of `k = 1, 2, 4` scores 1.000 on exact, permuted,
foil and far in both pools. One-shot induction already suffices, which is
consistent with the one-shot generative-structure result [20] but means k has
no measurable effect here (limitation L6). Wilson 95% interval on primary
`exact` = [0.974, 1.000], sealed = [0.976, 1.000]; Fisher exact vs chance
`p = 3.74e−52` (primary) and `4.17e−56` (sealed). The bootstrap 95% CI on the
paired `exact − permuted` difference is degenerate at `[0,0]` because no
discordant pair exists. The shared G6 gate flags `rsi_exact = 100` as
non-informative (non-blocking FLAG, recorded beside the result).

## 5. Per-item audit trail (hand-checkable inputs and outputs)

Every one of the 3,600 rows logs `id`, `pool`, `difficulty`, `relation`, `k`,
`condition`, `arm`, `input` (demonstration strings, query subject, candidate
list, domain), `output` (`subject`, `relation`, `target`, `verdict`), `gold`,
`verdict` and the `flags` sub-scores. Representative structural rows:

- **PASS, primary `rsi_000`, CAUSE, exact, k=1** — demos `storm causes rainbow`;
  query subject `hail`, candidates `[frost, chill, cloud, dew]`. Output
  `subject=hail, relation=CAUSE, target=frost, verdict=match`; gold relation
  CAUSE, gold target `frost` → correct. The relation field is checked, not just
  the target.
- **PASS, same item, permuted** — demos `rainbow is caused by storm` (inverse
  surface, roles swapped). Canonicalised back to `CAUSE`; identical output
  `hail / CAUSE / frost / match`. A serial memoriser would have read the wrong
  direction (control C3 confirms the naive arm does).
- **PASS, same item, foil** — query subject `shower` (no CAUSE fact), candidates
  include the demonstrated object `storm`. Output `subject=shower,
  relation=CAUSE, target=None, verdict=reject` → correct rejection; the naive
  arm copies `rainbow` and asserts a match (control C2).
- **PASS, same item, far** — query subject `team`, candidates
  `[norm, role, market, law]` from the disjoint `social` domain. Output
  `team / CAUSE / market / match`; only the relation was shared with the
  weather demonstrations.

## 6. Criteria (fixed from design §7.6; machine-checked in the script)

All 21 rules true: per pool `exact ≥ 0.75`, `permuted ≥ exact − 0.10`, no
`permuted < exact − 0.15` hard fail, `foil ≥ 0.85` with floor 0.70,
`far ≥ chance + 0.20` with floor `chance + 0.10`, relation coverage ≥ 0.95;
plus sealed `exact ≥ primary exact − 0.10`, the six instrument controls,
no network, no save, and a recorded saturation flag. No threshold, rubric or
gate was relaxed to obtain the result. The criteria were written from design
§7.6 before the run; the three development-time failures were instrument bugs
found by the controls, not rubric changes.

## 7. Limitations — read before citing these 1.00s

1. **The front is saturated and therefore INVALID as a progress signal (G6
   FLAG).** Every capability endpoint is at its best value. The measurement
   shows the specified structure-mapping decision rule is internally consistent
   and implements the design's permutation/foil/far semantics; it does **not**
   measure difficulty-graded induction, a learning curve, or generalisation
   beyond the authored families. A 1.00 here is a mechanism check, not a
   capability or growth claim.
2. **The subject under test is a harness-level decision layer, not the RAVANA
   engine.** There is no `CognitiveChatEngine` boot, no graph write and no
   generative path. This is deliberate — RSI's endpoint is a structured
   `(subject, relation, target)` field, and scoring engine prose with a regex
   would violate the debugging skill's "measure the decision, not the prose" —
   but it bounds the claim: this does not show the engine itself induces
   relations from demonstrations or emits the relation field.
3. **Objects are meaningless tokens; the world model is authored.** The
   "relations" are facts hand-authored into a lookup table over arbitrary
   domain words. Induction reads a relation *label* off demonstration surface
   phrases; it does not infer a predicate from object semantics. Foil and far
   tasks rely on disjoint authored object sets, which makes them structurally
   clean but semantically shallow. Construct validity is limited.
4. **Surface parsing is an exact-phrase lexicon.** A demonstration phrased
   outside the fixed forward/inverse phrase lists is unparseable, yields no
   induced relation and no answer. Robustness to natural phrasing is not
   tested, and a saturated score on the authored phrases must not be read as
   robustness.
5. **Two relations are symmetric, so their permutation is trivial.**
   OPPOSITE and RESEMBLES have no inverse form; their "permuted" demonstrations
   are a role swap of a symmetric predicate, which canonicalisation handles by
   construction. Only the four directional relations carry a genuine
   forward/inverse permutation signal (control C3 still passes overall).
6. **k has no measured effect.** `k = 1` already scores 1.00 everywhere, so the
   1/2/4 sweep is flat. That is consistent with one-shot induction [20] but
   also means RSI does not measure improvement with more demonstrations. A
   harder bank with noisy or conflicting demonstrations is needed for k to
   carry signal.
7. **The chance arm isolates the relation field, not a fully random policy.**
   It emits the true target with a shuffled relation label, so chance ≈
   1/|relations| ≈ 0.21 by construction. The large `far − chance` gap is
   correspondingly partly definitional: far uses the correct authored world-model
   fact while chance deliberately corrupts only the label.
8. **Foil realism is weak.** The foil is built by placing a demonstrated object
   among the candidates of a subject with no *R* fact; the naive arm's
   acceptance is therefore largely by construction. This demonstrates the
   control's power but does not establish difficulty against a genuinely
   confusable, semantically plausible foil.
9. **Single split, two runs, one seed.** Reproducibility is byte-for-byte under
   `PYTHONHASHSEED=0` (two identical runs produced the same probe-log SHA-256),
   but 48/52 families on one fixed split is not a population estimate, and no
   second configuration or independent rubric author was used.
10. **Statistics are limited.** Fisher exact vs chance and a bootstrap CI on
    the paired `exact − permuted` difference are computed; the design's
    McNemar tests and Holm-adjusted secondary comparisons are not. The paired
    CI is degenerate because there are zero discordant pairs.

## 8. Concrete next work (a shortfall is work, never a ceiling)

Per the debugging skill, the saturation and the shallow authored world model
are the next work items. Author **noisy/conflicting demonstration banks** where
the majority relation must actually be recovered and *k* matters; add a natural
surface-phrase variant arm so parsing is not an exact-match trick; and wire the
structured relation-field endpoint into a real engine output path so RSI tests
the engine's induction rather than a stand-in decision layer. Re-freeze a new
hash-split pool and re-run; do not relax a threshold or report the current
saturated split as a difficulty-graded result.

## 9. Citations (from `docs/BRAIN_LEARN5_DESIGN.md` §11)

- Gentner, D. (1983), "Structure-mapping: A theoretical framework for analogy,"
  *Cognitive Science* 7(2):155–170 — relations map in preference to object
  attributes; systematicity favours interconnected predicate systems.
  Cognitive-level theory, not a neural finding.
- Gentner, D., & Markman, A. B. (1997), "Structure mapping in analogy and
  similarity," *American Psychologist* 52(1):45–56,
  doi:10.1037/0003-066X.52.1.45 — structural alignment underlies analogy and
  similarity; appearance matches map attributes instead of relations.
- Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015), "Human-level concept
  learning through probabilistic program induction," *Science* 350(6266):1332–
  1338, doi:10.1126/science.aab3050 — one-shot concept learning by inferring a
  generative structure, motivating the k=1 / one-shot criterion.

## 10. Reproduction and artifacts

```
cd C:/Users/Likhith/Documents/projects/ravana
PYTHONPATH=scripts PYTHONHASHSEED=0 uv run python scripts/measure_learn5_rsi.py
PYTHONPATH=scripts uv run python -m pytest tests/unit/test_learn5_rsi.py \
    tests/unit/test_learn5_pdr.py tests/unit/test_learn5_wfa.py -q
```

- Harness: `scripts/measure_learn5_rsi.py`
- Tests: `tests/unit/test_learn5_rsi.py` (25 tests; 63 pass with the two co-run
  modules in the same process — skill §2b, no global-patch leak)
- Per-item rows: `scratch/learn5_rsi_probes.jsonl` (3,600)
- Summary: `scratch/learn5_rsi_summary.json` (probe-log SHA-256
  `34a3af7d9b0cf79c28d621ff173b66be584463195be6a2f0826843193162f33f`,
  byte-identical across two identical-config runs)
- Gate log: `scratch/learn5_rsi_gates.jsonl`
- Measurement commit: `c47ac09` (the three RSI files below were uncommitted in
  the working tree)
