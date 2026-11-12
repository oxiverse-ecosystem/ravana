# The decoder's penalties are 56x its logit range

**Status:** measured and real -- but **NOT the root cause**. See the
correction at the top.

## Correction first

I wrote this doc claiming the penalty scale was the root cause of the decoder's
word salad. **That is wrong.** With the penalties patched to `-= 0.0` the output
is essentially unchanged:

```
gravity  penalties  : sharma therapies axiomatization mohammedan pending dishonor ganges dwi
gravity  NO penalties: sharma antioxidative array scientifically dismiss massage explor
```

The penalty scale is a genuine defect -- the constants are tuned for a
different weight scale -- but it is not why the decoder produces salad. The
actual cause is the training corpus: see
`docs/DECODER_TRAINED_ON_APHORISMS.md`.

I am keeping this doc because the measurement is correct and the fix is worth
making. It just is not the lever I thought it was.

## The measurement

One generation step, `gravity` conditioning, no CPG:

```
logits: n=15000  min=-0.11  max=0.16  mean=0.00  std=0.02
logit RANGE (max-min) = 0.27
```

The penalties applied during generation:

```
repetition penalty on last token   -15.0
bigram repetition                  -10.0
trigram repetition                 -15.0
exploration bonus per unseen word  +2.0
```

**A −15 penalty is 56× the entire natural logit range.**

## Why this breaks generation

The decoder's learned signal lives in a band 0.27 wide. The penalties are
three orders of magnitude larger in effect. So the argmax is not chosen by
what the model learned -- it is chosen by whichever penalty happens to apply:

- the last token is suppressed by −15, so the model can never repeat anything
- every unseen content word gets +2.0, so the model is pushed toward rare
  words it has never used in context
- the result is a sequence of unrelated rare words

That is exactly the observed output:

```
gravity  ->  sharma therapies axiomatization mohammedan pending dishonor ganges dwi
ocean    ->  gothein aspires eel compulsion exoplanets regulations spelling sulliva
```

Proper nouns, obscure nouns, no syntax. The exploration bonus is doing
precisely what it says: exploring rare content words.

## The decisive test

Teacher-forced CE on the decoder's own output vs a real sentence:

```
=== gravity ===
  generated : sharma therapies axiomatization mohammedan pending dishonor ganges dwi
  CE on its OWN output: 3.650
  CE on 'gravity is a concept': 1.323

=== ocean ===
  generated : gothein aspires eel compulsion exoplanets regulations spelling sulliva
  CE on its OWN output: 6.980
  CE on 'ocean is a concept': 1.039
```

The model assigns **3.65–6.98** to its own generations and **1.04–1.32** to a
real sentence. It does not recognize its own output as language.

This is not undertraining. `_avg_cross_entropy` is **1.59** (perplexity 4.9)
and real sentences score 0.7–2.8. The model is well-fit; generation is broken
by the penalty scale.

## Why it was never seen

Training uses teacher forcing with the same `combined` vector but **no
penalties** -- they are generation-only. So training and generation optimize
different objectives. The model learns a distribution in a 0.27-wide band;
generation then perturbs it with ±15 offsets.

`train_on_sentence` reports CE 1.59 and everyone concluded the decoder was
working. It is -- just not at generation time.

## The fix

Scale the penalties to the logit range. Either:

1. **Normalize the logits** before applying penalties (z-score or min-max
   across the vocab), so a "−15" means the same thing regardless of weight
   scale. This is the robust fix -- it survives retraining.
2. **Rescale the constants** to the observed range: −15 → −0.15, −10 →
   −0.10, +2.0 → +0.02. Cheap, but breaks again the next time the weights
   are retrained.

Option 1 is correct. The penalties are a *policy* on top of a *distribution*;
they should be expressed in the distribution's own units.

## Also worth noting

`output_proj` weight norm is 125.8 with `absmax=0.334` across 15000×256
weights. That is a very flat distribution -- consistent with a Hebbian update
at `lr=0.0002` that never grew large weights. The flat distribution is why the
logit range is so small.

## Related

- `docs/DECODER_OFF_TOPIC_REGARDLESS_OF_CPG.md` -- CPG cannot help because
  the decoder's output is penalty-driven, not model-driven
- `docs/WSM_CONDITIONER_DEAD_AT_GENERATION.md` -- the conditioner wiring,
  necessary but not sufficient
- `docs/WSM_IS_NOISE.md`, `docs/BOOST_STRENGTH_NOT_THE_LEVER.md`
- `docs/ESCALATION_INERT.md` -- escalation is inert for the same reason
