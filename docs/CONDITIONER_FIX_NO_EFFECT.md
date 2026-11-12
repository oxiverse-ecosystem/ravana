# The conditioner fix did not move the baseline

**Result:** negative. PR #318 fixed a real bug; the measured grounding numbers
did not improve.
**Implication:** the off-topic problem has a different cause.

## The numbers

Same instrument (`scripts/eval_decoder_groundedness.py`), same persisted state,
before and after PR #318:

```
                    n     on_topic   unsupported   cosine
before (baseline)  440     0.1501       1.6955      0.3422
after  (fix)       440     0.1325       1.5886      0.3378
```

**Every axis moved slightly the wrong way**, and by less than run-to-run
variance. The fix is real — `_linggen_W_sm` now attaches at boot with shape
(75, 65), verified — but it does not change decoder output.

The baby engine moved the same way (0.5021 → 0.4911), confirming this is noise,
not a regression caused by the fix.

## What this rules out

I hypothesised that the off-topic problem was caused by the conditioner not
being loaded: the grounding pass fits `W_sm`, and if the decoder never receives
it, conditioning would be inert.

**Ruled out.** The projection is now attached and the decoder is still off-topic
85% of the time.

## What is actually happening

Three possibilities remain, and they are not mutually exclusive:

**1. The conditioning signal is too weak to steer the decoder.**
`_build_conditioned_bos` blends the sensorimotor vector 0.3 into an 11-D tail
of a 75-D vector, then renormalises. The subject's influence on the initial
state is small, and the decoder's 200-pass CE training has learned to ignore it.

**2. `W_sm` was fitted on too little data to be useful.**
`train_decoder_grounded` fits a ridge regression 65→75 on the grounded
description pairs. If that is a few hundred pairs, the projection may be
near-identity or near-noise — correctly fitted, but carrying little signal.

**3. The decoder's CE training has simply overwritten the conditioning.**
400,000 CE presentations against 15,000 grounding presentations. Even a
correctly-attached projection can be ignored by a network trained to predict
the next word of general prose.

## How to tell them apart

The decisive measurement is **whether conditioning changes the output at all**:

- Take one subject. Generate with the real conditioning vector and with a
  shuffled/zero vector. If the outputs are near-identical, the decoder is
  ignoring conditioning entirely → cause 1 or 3.
- If they differ substantially but both are off-topic, the projection carries
  signal but not the right signal → cause 2.

That is a small, cheap experiment and it should come before any further
training-side work.

## Honest status

The bug was worth fixing on its own terms — a fitted artefact that production
never loaded is a defect regardless of whether it moves a metric. But I
presented it as likely to improve grounding, and it did not.

The baseline stands: **on-topic 0.133, unsupported 1.59, cosine 0.338.**

## Related

- `docs/DECODER_GROUNDEDNESS_BASELINE.md` — the instrument and the baseline
- `docs/GROUNDING_MECHANISM.md` — Committed-Proposition Gating, the proposed
  runtime mechanism (unchanged by this result)
- PR #318 (`7682a87`) — the conditioner-at-boot fix
