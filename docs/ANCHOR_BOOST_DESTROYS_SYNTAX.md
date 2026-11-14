# The anchor boost selects the right words and destroys the sentence

Date: 2026-11-14
Instrument: `scripts/measure_anchor_boost_strength.py` (production path,
persisted engine), `scripts/probe_anchor_boost.py`

## The finding

The anchor boost works exactly as its calibration said it would -- and the
calibration was measuring the wrong thing.

Swept on the PRODUCTION path (`_generate_with_situation_model`, the call the
chat engine actually makes), 4 subjects, 6 strengths:

```
gravity (38 neighbours, 12 definition words)
  boost= 0.0: <no output>
  boost= 2.0: on=4/4  (1.00) cos=0.427 unsup=0  "Objects other earth planet centre one body."
  boost= 4.0: on=6/6  (1.00) cos=0.488 unsup=0  "Objects other body earth centre gravity gives force."
  boost= 6.0: on=8/8  (1.00) cos=0.474 unsup=0  "Earth centre other body objects planet force gives moon gravity."
  boost= 9.0: on=13/13 (1.00) cos=0.463 unsup=0 "Other earth body objects gives centre planet force gravity moon..."
  boost=12.0: on=18/18 (1.00) cos=0.417 unsup=0 "Earth other objects body planet gives centre force gravity moon..."

time (165 neighbours, 43 definition words)
  boost= 0.0: on=1/3 (0.33) cos=0.454 unsup=1  "Other philosophy energy such used there one."
  boost= 2.0: on=6/6 (1.00) cos=0.660 unsup=0  "Used action time often three change forms one."
  boost= 9.0: on=23/23 (1.00) cos=0.547 unsup=0 "Component used forms often change time three present experience..."
```

**Every strength from 2.0 up gives on-topic 1.00 and unsupported 0.** The
subject words appear. The graph-supported words appear. Nothing ungrounded
appears.

And every one of them is a **word list, not a sentence**:

```
"Other earth body objects gives centre planet force gravity moon other orbit
 weight toward keeps objects body earth."
```

Nineteen content words, zero function words, no determiner, no preposition, no
grammatical frame. It is a bag of the definition's nouns and verbs in
near-random order.

## Why the calibration missed it

The 9.0 knee was fitted in `scripts/train.py` by sweeping the boost and scoring
"the fraction of generated content words that come from the anchor
description". The quality function it feeds (`_grounded_generation_quality`,
`scripts/train.py:559`) is:

```
coherence = in-vocab ratio * distinct-1 * on-topic
```

`distinct-1` is `len(set(words)) / len(words)` -- the type-token ratio. It
measures **word diversity**, not grammaticality. A word list and a sentence
both score 1.0. So the sweep's "distinct-1 0.996, repetition 0.000" row means
"no word repeats", and the sweep had no term that could penalize the loss of
syntax.

The production path makes it worse in a way the harness cannot see: it passes
`cerebellar_ngram` (a fluency prior) and a situation-derived temperature, and
it applies the boost at **every step** (`neural_decoder.py:525`). The training
harness's `_gen_one` passes the boost only through the fallback branch, and
its decoder call does not use the production temperature schedule.

## What this means

The boost is not a nudge; at any strength that moves the output it is a
**replacement** of the decoder's distribution with a uniform draw over the
anchor's content words. That is why on-topic saturates at 1.00 and stays there
while the word count grows -- the model is not elaborating, it is sampling
from the boosted set.

This is the same conclusion `docs/BOOST_STRENGTH_NOT_THE_LEVER.md` reached from
the other direction: a bias on the logit vector is the wrong level of
intervention. That document found the boost plateaus at 2.5 -> 5.0; this
measurement finds that past the plateau the output stops being a sentence
entirely.

## The brain-faithful reading

`docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md` §2.3: the message layer binds
concepts to **thematic roles** and the formulator then linearizes them. A
boost over content words supplies the CONCEPTS but supplies no ROLES and no
LINEARIZATION ORDER. The decoder's own linearization is being overridden, so
what remains is the concept set with no syntactic plan -- which is precisely
the "fluent empty speech" profile, one level down: not even fluent.

The mechanism the evidence supports is not "raise the anchor words' logits" but
"**condition the linearization on the retrieved proposition**": the retrieved
definition is a sequence with roles already assigned, and the decoder should
elaborate FROM it rather than re-select words from it.

## Reproduce

```
uv run python scripts/measure_anchor_boost_strength.py gravity ocean species time
uv run python scripts/probe_anchor_boost.py gravity ocean species time
```

## Related

- `docs/ANCHOR_TOKEN_BOOST_MEASURED.md` -- the original sweep (the metric that
  could not see syntax)
- `docs/BOOST_STRENGTH_NOT_THE_LEVER.md` -- the plateau from the CPG side
- `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md` §2.3, §5.2 -- the message layer
- `docs/CONDITIONING_COLLAPSES_TO_A_POINT.md` -- why the conditioning path
  cannot supply the linearization either
