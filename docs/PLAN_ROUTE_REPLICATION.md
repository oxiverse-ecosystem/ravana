# Replicating PR #364's plan-route claim at n>=40

**Status:** replicated at n=133 (84 paired). **The 21x claim does not hold.**

Reproduce (single process, no `eng.save()`, persisted state at commit `86ae48c`):

```
RAVANA_OFFLINE=1 uv run python scripts/replicate_plan_route.py \
    --out docs/plan_route_replication_data.json
RAVANA_OFFLINE=1 uv run python scripts/measure_plan_narrative.py \
    --max-subjects 40 --out <absolute-native-path>/plan_narr40.json
```

Fixed seed: Python `random.seed(42)`, `np.random.seed(42)`, engine `seed=42`
(`build_engine("persisted")`). Residual sampling noise is reported below.

## Claim under test

PR #364 (`8389178`, "measured: unsupported assertions to 0") headline:

| claim | value |
|---|---|
| plan route fires on | 23/41 subjects |
| plan route unsupported | 0.087 |
| fallback unsupported | 1.833 |
| reduction | **21x** |
| supporting arm table (n=40) | baseline on_topic 0.1947 / unsup 1.975 / frame 0.791; wired plan n=7 unsup 0.167 vs fallback 0.438 |

We test whether a >=40-subject, fixed-seed run reproduces the direction **and
magnitude** of the unsupported reduction, and whether on-topic and grammatical
quality are preserved.

## Method

One cold engine boot. For each discovered subject both arms run through the
**same** entry point the original instruments used,
`eng._generate_with_situation_model(ctx)`, with a freshly built context each
time (`build_ctx` resets the situation model):

- **FALLBACK**: `eng._plan_narrative_enabled = False`
- **PLAN**: `eng._plan_narrative_enabled = True`; a subject counts as *fired*
  when the returned strategy is `situation_model_plan`.

Raw text, strategy and all sub-scores are stored for every subject in
`docs/plan_route_replication_data.json` (133 rows). No engine state was written.
A second identical run is the determinism check.

## Metric definitions (the original functions, unchanged)

Imported directly from `scripts/eval_decoder_groundedness.py` and
`scripts/prototype_linearization.py`:

1. **`score_unsupported_assertions`** (:459) — for each token in `RELATION_WORDS`
   (:147), find the nearest content word within 5 tokens (skipping
   prepositions/stop words/other relation words), forward first then backward.
   The `(relation, argument)` pair is *unsupported* iff the argument is not a
   graph neighbour of the subject, not in its stored definition, and not a
   constituent of a multi-word neighbour. Reported number is the **mean
   unsupported-pair count per subject**, not a proportion.
2. **`score_on_topic`** (:407) — fraction of content words (`_content_words`,
   :337) that are a neighbour, a definition word, or a constituent of either;
   plus mean GloVe cosine to the subject vector.
3. **`_frame_score`** (:53 in `prototype_linearization.py`) — `verb` = a hit
   against a hardcoded `_REL_RE` predicate list; `func_share` = share of tokens
   in the function-word set; `frame = verb * min(1, func_share / max(3, n/3))`.
   This is the grammatical/surface proxy from the original prototype.
4. Coverage/strategy: the `(text, strategy)` tuple returned by the decoder path;
   `salad` = `_is_word_salad` from `score_fluency` (:538).

## Results

### Subject pool (exact evaluation order, n=133)

`discover_subjects(eng)` returns 133 subjects (nominated seeds + all graph
labels with degree >=2 + 3 unknown controls). 111/133 have a stored definition.

```
gravity, ocean, water, light, time, music, colour, language, air, although,
analysis, animal, animals, another, archived, being, blue, called, carnivore,
cat, changes, climate, clouds, common, comparison, conditions, data,
definition, development, different, dog, early, earth, edit, education,
environment, environmental, even, example, explained, fall, first, fish, form,
found, full, good, health, heat, history, however, human, important, including,
information, intentforge, knowledge, known, land, learn, learning, life, long,
made, mammal, management, many, mathematics, meaning, mind, much, natural,
nature, often, only, out, oxiverse, oxygen, pdf, people, perceive, person, pet,
philosophy, photosynthesis, physics, plant, plants, practical, psychology, rain,
rains, rather, ravana, reason, research, right, science, scientific, scientists,
sea, sky, social, sound, species, star, studies, study, sunlight, system,
systems, term, theory, there, thought, three, tree, two, type, understanding,
university, used, using, vibrate, vibration, view, way, well, world, years,
zzzunknown, qxzblorp, notarealword
```

The first 40 of these are the pool used for the n=40 arm table below. The
machine-readable authoritative list is `subjects` in the evidence JSON.

### Primary paired A/B — subjects where BOTH arms produced text (n=84)

| arm | n | unsupported (mean count) | on-topic | frame | verb | salad |
|---|---|---|---|---|---|---|
| FALLBACK (flag OFF) | 84 | **0.5357** | **0.4795** | **0.7976** | 68/84 | 0/84 |
| PLAN (flag ON) | 84 | **0.3571** | 0.4098 | 0.6587 | 56/84 | 0/84 |
| paired delta (plan−fb) | | **−0.1786** | −0.0697 | −0.1389 | | |

Reduction factor = **1.50x** (fallback 0.5357 / plan 0.3571). On-topic and
frame both **regress**.

Coverage over all 133 subjects (not paired): fallback produced 90/133, plan
produced 110/133, plan route *fired* on **70–71/133**. The plan arm's extra
26 outputs are a coverage gain, not a paired quality comparison.

Determinism re-run (identical seed, n=84 both-produced): fallback
unsup 0.5357 / on-topic 0.4804 / frame 0.7976; plan unsup **0.3333** /
on-topic 0.4001 / frame 0.6587; factor 1.61x; fired 70/133. Run-to-run noise
is ~0.02 unsupported assertions per subject; the conclusion is unchanged.

### Where the plan route actually fired (paired subset)

| arm | n | unsupported | on-topic | frame |
|---|---|---|---|---|
| FALLBACK (paired, fired subjects) | 48 | 0.7292 | 0.4489 | 0.8542 |
| PLAN (paired, fired subjects) | 48 | 0.3542 | 0.3611 | 0.5833 |

Factor 2.06x on the subjects the route was designed for. Still not 21x, and
on-topic/frame still regress.

Headline-style *disjoint* split (plan arm on the 71 fired subjects vs fallback
arm on the 42 non-fired subjects): plan 0.3803 vs fallback 0.3095 — a
**0.81x** ratio, i.e. the plan subset is *worse* on this raw metric. No cut of
this data gives 0.087 vs 1.833 / 21x.

### Original arm-table instrument (`measure_plan_narrative.py`, n=40)

| arm | n | on-topic | unsupported | frame |
|---|---|---|---|---|
| baseline | 40 | 0.4830 | 2.600 | 0.974 |
| plan_narr | 26 | 0.4500 | 0.308 | 0.718 |

Paired on n=26: `d_on_topic = −0.0681`, `d_unsupported = −2.308` (8.4x on the
arm table). The direction replicates; the PR's exact numbers do not:
baseline unsup is 2.600 (PR: 1.975), plan_narr is 0.308 (PR: 0.000) with
coverage 26/40 (PR: 7/40), because later fixes changed the reachable set. The
sign of the on-topic delta also flipped (+0.076 in PR, −0.068 here).

### Sensitivity to the instrument's known false positives

Dropping `(relation, argument)` pairs where the relation word is the hedge
`appears` (21 flags across the run; see validation): paired-all fallback
0.5238 vs plan 0.2024 = **2.59x**. Excluding hedges strengthens the plan
advantage but still leaves ~2.6x, not 21x.

## Metric validation (per `ravana-debugging/references/probe-validation.md`)

Raw input/output for a hand-labelled 10-subject sample is printed by
`replicate_plan_route.py` and stored in the evidence JSON. Hand-checking the
flagged pairs against the property *"asserts a relation to a concept the graph
does not support"*:

| subject / arm | raw output | metric flags | hand label |
|---|---|---|---|
| gravity / fallback | "...relates to an earth. ... an orbit. ... a star." | none | correct: earth/orbit/star are neighbours |
| gravity / plan | "It appears that it is connected to an earth." | none | correct |
| ocean / fallback | "Ocean feeds into a sea." | none | correct (but `frame=0` — see below) |
| ocean / plan | "It appears that it feeds into a sea." | `appears:feeds` | **false positive** — `appears` is the hedge, not a predicate; the real argument `sea` is a neighbour |
| water / plan | "It appears that it is a liquid." | none | correct |
| light / fallback | "...connects to a sun. ... holds photosynthesis. ... similar to a star." | none | correct |
| time / fallback | "It is different from a well. ... It broadly is opposite to years." | `is:different`, `is:broadly` | **false positives** — multiword predicate `is different from`; the real arguments are neighbours |
| time / plan | "It appears that it is similar to way." | none | correct |
| air / plan | "It appears that they spark a sound." | `appears:spark` | **false positive** (same hedge) |
| analysis / fallback | long stored definition | `drives:initial`, `is:gathered` | **false positives** — modifier/participle picked as the argument |
| animal / plan | "It appears that it drives plants." | none | correct |

Two confirmed probe flaws:

1. `appears` (and other hedges) are in `RELATION_WORDS` and get paired with the
   first content word after "It appears that ...", usually the clause's real
   verb. This manufactures unsupported pairs, mostly on the plan arm (whose
   outputs nearly always begin with the hedge).
2. For multiword predicates ("is different from X", "paves the way to X",
   "gives rise to X") the metric pairs the relation with a glue word
   (`different`, `way`) instead of X. All 7 residual unsupported subjects in
   the n=40 plan_narr arm are this class (e.g. "gravity is different from an
   earth" flagged as `is:different`).

`_frame_score` also has a hardcoded verb list: grammatical plan clauses such
as "It appears that it feeds into a sea." score `frame=0.0`, `verb=False`
because `feeds` is absent from `_REL_RE`. The lower plan frame score is
therefore partly an instrument artifact (shorter single clauses also carry
fewer function words), but the plan outputs are visibly shorter and less
informative; on-topic and overlap-word counts fall too.

Both flaws are in the *same* functions the PR used, so the comparison is
apples-to-apples with the original claim; they do not manufacture the ~1.5x
effect, and correcting them does not approach 21x.

## Verdict

**PR #364's headline does not replicate at n>=40.** With a fixed seed on the
current tree (commit `86ae48c`):

- Unsupported assertions fall from 0.5357 to 0.3571 on the paired eval set —
  **1.50x**, not 21x; ~2.1x on the subset where the route fires and ~2.6x
  under the most probe-favourable correction.
- On-topic drops (0.4795 -> 0.4098) and the structural frame score drops
  (0.7976 -> 0.6587); the original positive on-topic delta is not reproduced.
- The specific headline numbers (0.087 vs 1.833, 23/41) are not reproducible
  from this state or any subset examined.

The defensible residual claim is narrow: *on subjects where a committed plan
exists, the plan route produces short, grammatical, mostly graph-supported
clauses with roughly half the unsupported assertions of the fallback* — a
precision route with a real but modest effect and a coverage/on-topic cost.
That is not the 21x result PR #364 was held back to replicate, so the claim is
recorded as **not replicated**.

## Files

- `scripts/replicate_plan_route.py` — new, read-only single-process A/B instrument.
- `docs/plan_route_replication_data.json` — full raw per-subject evidence (133 rows).
- `docs/PLAN_ROUTE_REPLICATION.md` — this report.
