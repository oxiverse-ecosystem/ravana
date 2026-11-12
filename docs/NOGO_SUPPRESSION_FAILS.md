# Suppression (NoGo) makes output worse — a negative result

**Status:** measured. The NoGo pathway was implemented, tested, wired, and
**rejected on evidence.**
**Consequence:** CPG stays boost-only. Suppression is available
(`suppress > 0`) but defaults to off.

## What was built

The Go pathway (`token_boost`) raises admissible tokens. The NoGo pathway
(`token_suppress`) lowers inadmissible content words. Both were implemented in
`NeuralDecoder.generate` and wired through CPG, with a `suppress_margin` so
near-misses are spared.

The motivation was sound: CPG's first measurement moved on-topic 4.5× but left
**unsupported assertions flat** (1.13 → 1.19). Boosting alone does not
penalize unsupported content, so suppression was the obvious next increment.

## The measurement

Same engine, same queries, three configurations:

```
BOOST-ONLY (suppress=0)
  why do we dream      "It appears that dream carries through life. Connection begins with a spark o..."
  tell me about trust  "It seems to me that trust: it holds respect freedom. Trust is earned through..."
  what is time         "Time is the continuous progression of existence that occurs in an apparently..."

NOGO-2 (suppress=2.0)
  why do we dream      "i might be reaching, but Archived history learning one like life own such."
  tell me about trust  "i might be reaching, but There learning such must management like one own."
  what is time         "Time is the continuous progression of existence that occurs in an apparently..."

NOGO-4 (suppress=4.0)
  why do we dream      "i might be reaching, but Archived there learning like one psychology such ow..."
  tell me about trust  "It seems to me that trust sparks respect freedom. Trust is earned through co..."
```

**Suppression produces word salad.** At `suppress=2.0` both queries collapse
into "Archived history learning one like life own such" — grammatically
structured, semantically empty.

## Why it fails

The NoGo pathway suppresses content words that are not committed targets. But
most content words in a sentence are **not** proposition targets — they are
supporting vocabulary: "history", "learning", "management", "one", "such".

Suppressing them does not steer the decoder toward the committed proposition.
It removes the words the decoder would have used to *build* a sentence, and the
decoder fills the gap with whatever survives — which is arbitrary.

This is the same failure mode as the `_is_word_salad` check: the instrument
can detect the result but the mechanism that produced it was wrong.

## The lesson

**Boosting is a prior; suppression is a constraint.** A prior can be ignored
when it conflicts with the learned distribution. A hard constraint on a
language model's vocabulary does not redirect generation — it degrades it,
because the model has no trained way to satisfy it.

The brain's NoGo pathway does not work by deleting words from the formulator's
vocabulary either. It inhibits **competing plans**, not lexical items. CPG's
suppression was applied at the wrong level of abstraction.

## What this leaves

CPG remains boost-only, which is the configuration that measurably worked:

```
                    CPG OFF    CPG ON    change
on-topic fraction    0.0508    0.2270    4.5x
supported assertions 0.0107    0.1101    10.3x
unsupported          1.1337    1.1927    ~flat
```

The unsupported count is still the open problem, and suppression is not the
answer. The next candidate is to make the boost strong enough to actually
compete with the learned distribution, rather than adding a second pathway —
or to reduce the decoder's reliance on unsupported relations at the point where
they are selected, not where words are.

## Related

- `docs/CPG_MEASURED_EFFECT.md` — the boost-only result
- `docs/GROUNDING_MECHANISM.md` — the mechanism
- PR #323, #324 — the CPG module and wiring
