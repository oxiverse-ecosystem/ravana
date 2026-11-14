# Seven capability fronts: baseline and programme

**Status:** baseline measured by `scripts/measure_fronts.py`. This document is
the plan of record for the seven fronts, with the honest starting scores.

## Why a harness

Every previous score in this repo was asserted. The first version of this
harness scored **100/100 on three fronts** while the live answers were
deflective:

```
Q: what do you think about artificial intelligence
A: i'm still forming a view on artificial intelligence. i don't have a fixed
   stance on artificial intelligence yet — what's your take?
```

The probes matched on "answer is longer than 40 chars and contains a verb".
That is not a capability measure. The harness now carries a `DEFLECT` pattern
listing the actual hedging register, and each probe prints its evidence so a
score can be audited.

**Lesson (third occurrence):** an unvalidated proxy scores high. Same as
relation typing (75.5% → 0/32), the determiner frame (0.787 → 0.730), and the
noun-suffix table (0.803 → 0.608).

## The seven fronts

| # | front | how it is scored |
|---|---|---|
| 1 | language understanding / conversation | coherent non-degenerate replies |
| 2 | grounded factual answers | subject + predicate, not a deflection |
| 3 | opinions / stance | substantive + evaluative, not a deflection |
| 4 | thinking / reasoning | multi-step causal structure |
| 5 | language learning (online) | intake + memory + plasticity + web wired |
| 6 | speaking / surface realization | grammatical, non-degenerate surface form |
| 7 | code generation | emits a real function AND has learned functions |

## Corrected baseline (tightened probes, `fronts2.txt`)

```
100/100  language understanding / conversation
100/100  grounded factual answers        <- STRUCTURAL only; see caveat
 67/100  opinions / stance
 33/100  thinking / reasoning
  0/100  speaking / surface realization
  0/100  code generation
 40/100  language learning (online)
 49/100  MEAN
```

The tightened probes moved three scores: opinions 100 → 67, speaking 100 → 0,
and the mean 68 → 49. Each change was a probe that had been passing defective
output:

- **speaking 100 → 0**: both probes returned "the closest i get to X is Y and
  maybe Z" — a degenerate association template. The old probe only checked
  for ≥4 words and a verb.
- **opinions 100 → 67**: "i'm still forming a view ... what's your take?" now
  counts as deflection.
- **grounded stays 100 but is STRUCTURAL, not truthful.** `what causes rain`
  still returns "Rain leads to heat, and that in turn leads to climate" — a
  wrong causal claim. No regex can judge truth, so this score measures
  subject+predicate structure and the truth defect is tracked separately as
  graph defect #4. **Do not read 100/100 here as "answers are correct".**

## Baseline (first run, pre-tightening) — superseded

```
100/100  language understanding / conversation
100/100  grounded factual answers
100/100  opinions / stance
 33/100  thinking / reasoning
100/100  speaking / surface realization
  0/100  code generation
 40/100  language learning (online)
 68/100  MEAN
```

## What the live probes actually show

```
hello                    -> hello. what's on your mind?                    [good]
what is gravity          -> Gravity is the force by which a planet ...      [good]
tell me about yourself   -> i'm ravana, a cognitive architecture ...        [good]
what did i say my name was -> not that i recall — you haven't told me ...  [good]
what causes rain         -> Rain leads to heat, and that in turn leads to
                            climate.                                       [WRONG]
what do you think about AI -> i'm still forming a view ... what's your take? [DEFLECT]
why do you think it rains -> i'm drawn to rains. it sits close to people ... [AFFECT, not causal]
explain how a battery works -> i don't have a clean definition for battery,
                              but it is tied to plasma and cells            [WEAK]
write a function that adds -> i wish i could write that, but i don't have a
                              way to act on your system                     [REFUSAL]
```

Four concrete defects, each with a distinct cause:

1. **`rain → heat → climate`** — a wrong causal claim, confidently emitted. The
   graph has a bad edge; the grounding metric measures graph-support, not
   truth, so it cannot see this.
2. **Opinion deflection** — the stance machinery exists (belief store, user
   model, 75 capability docs) but the live path asks the user instead of
   committing.
3. **Explanation is affective association** — `engine_reasoning.py` is 180 KB
   and is not producing causal chains.
4. **Code generation refuses** — `_maybe_code_learn_or_generate` is wired, but
   the learner has 0 functions, and the refusal is a *capability* refusal
   ("I can't act on your system") rather than a knowledge refusal.

## Programme

Ordered by (measured deficit × certainty of the fix), not by size:

| step | front | change | why first |
|---|---|---|---|
| 1 | code generation | populate the learner; separate "I haven't learned this" from "I can't do this" | wired and empty; cheapest large gain |
| 2 | opinions | commit a stance from the belief store instead of deferring | machinery exists, one path change |
| 3 | thinking | build a 2-4 step causal chain from typed edges, with an adequacy check | `engine_reasoning.py` exists but is unused for this |
| 4 | grounded | graph-truth check on causal edges | the metric cannot see a wrong edge |
| 5 | language learning | verify intake actually writes from conversation and web | modules import but wiring unverified |

Each step: research → implement → measure with `measure_fronts.py` → PR.

## Research dispatched

- `docs/BRAIN_OPINION_FORMATION.md` — attitude construction, stance commitment,
  attitude certainty/ambivalence, calibrated expression.
- `docs/BRAIN_EXPLANATION_AND_REASONING.md` — causal chain construction, the
  multiple-demand network, adequacy monitoring, the self-explanation effect.

## Honest limits

- The scores are judgements backed by printed evidence, not metrics. They are
  only useful for showing *movement* between runs.
- Probe sets are small (3-4 questions per front). A score can move on one
  question.
- `DEFLECT` is a hand-written pattern; it will miss novel hedging. Every score
  is auditable because the answers are printed.
