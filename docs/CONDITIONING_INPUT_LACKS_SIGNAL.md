# The Binder vector does not contain the target: R² plateaus at 0.22

**Status:** measured. This locates the problem in the *input*, not the model.
**Consequence:** no model change to `W_sm` will help. The conditioning signal
itself lacks the information.

## The decisive test

`WSM_R2_MEASUREMENT.md` left one open question: is the 65→75 mapping weak
because a single ridge layer is too simple, or because the Binder vector does
not contain the information?

Fit a 2-layer MLP (tanh) on the same 250 pairs, 80/20 split, 3000 Adam steps:

```
linear ridge   R²: 0.21838
MLP h=64       R²: 0.19162
MLP h=128      R²: 0.19675
MLP h=256      R²: 0.21939
```

**The MLP does no better than ridge.** At h=256 it matches it (0.2194 vs
0.2184); at smaller widths it is worse. Adding nonlinearity buys nothing.

## What this means

The Binder vector — `_combined_attr_encoder.attribute_vector(glove64(concept))`
— carries about **22% of the variance** of the 75-D dual-code embedding, and
**no model can extract more**. The remaining 78% is simply not present in the
input.

This is a statement about information, not capacity. A 256-unit hidden layer
with 3000 steps can fit far more than 22% of a 75-dim target if the information
were there.

## The chain of claims, corrected each time

```
WRONG:  grounding pass is outweighed by CE           (ratio real, irrelevant)
WRONG:  conditioner not loaded at boot               (real bug, no effect)
WRONG:  W_sm is noise                                (norm misread)
HALF:   W_sm has R^2 = 0.22, weak but real           (correct, but implied
        the model was the limit)
RIGHT:  the Binder vector does not contain the
        target. R^2 plateaus at 0.22 across model
        classes, so the INPUT is the problem.
```

## Where the missing 78% lives

The target is `_embed_75d(concept)` — the concept's own dual-code embedding.
The input is a 65-D attribute vector derived from the GloVe vector.

So the question becomes: **what is in the 75-D embedding that is not in the
65-D attribute vector?** Candidates:

1. **The sensorimotor tail.** The 75-D embedding is `[GloVe | Lancaster-11]`.
   If the Lancaster channels carry most of the variance, a Binder vector with
   no sensorimotor content cannot predict them — and should not be expected to.
2. **Graph-derived structure.** If `_embed_75d` incorporates graph context
   (neighbours, edges), that is not recoverable from a single word's GloVe
   vector by construction.
3. **The attribute encoder may be lossy.** `_combined_attr_encoder` compresses
   a 64-D GloVe vector into 65 attributes; the compression may discard exactly
   the dimensions the target needs.

Hypothesis 1 is testable immediately and is the most likely: **the target
contains 11 sensorimotor dimensions the input has no access to.** A projection
cannot invent them, so R² is capped.

## The fix

Not a better model. The conditioning signal needs to carry information the
input currently lacks:

* **Include the sensorimotor vector in the input**, or
* **Remove the sensorimotor tail from the target**, or
* **Condition on something that actually predicts the target** — e.g. the
  concept's graph neighbourhood, which is where its meaning lives.

The third is the most brain-faithful: a concept's semantic identity is its
relations, not its surface attributes.

## Related

- `docs/WSM_R2_MEASUREMENT.md` — the ridge baseline (R² 0.218)
- `docs/WSM_IS_NOISE.md` — superseded
- `docs/DECODER_GROUNDEDNESS_BASELINE.md` — the downstream symptom
- `docs/GROUNDING_MECHANISM.md` — the runtime mechanism, which bypasses this
  entirely by committing to graph propositions rather than projecting
  attributes

---

## Update: the sensorimotor-tail hypothesis is disproved

The most testable candidate from the list above was: the target is
`[GloVe | Lancaster-11]`, so 11 of 75 dimensions are sensorimotor and the
Binder input has no sensorimotor content — capping R².

Tested by fitting the same ridge to each part separately:

```
R² full target (75-D)      0.21838
R² body only (first 64-D)  0.22842
R² tail only (last 11-D)   0.18254

target variance  body 0.009431   tail 0.015377
tail norm mean 1.0000   body norm mean 1.0000
```

**The tail is no harder to predict than the body** (0.183 vs 0.228, and the
difference is within noise). So the missing 78% is *not* the sensorimotor
channels.

Also note both parts are unit-norm (1.0000) — the target is normalised, so
neither block dominates by scale.

## What remains

The two surviving candidates:

1. **The attribute encoder is lossy.** `_combined_attr_encoder` compresses a
   64-D GloVe vector into 65 attributes. The compression may discard exactly
   the dimensions the target needs — in which case the fix is to condition on
   the raw GloVe vector, or on richer graph context.
2. **The target contains graph-derived structure.** If `_embed_75d`
   incorporates neighbours or edges, that is not recoverable from a single
   word's surface vector by construction.

Both point the same way: **condition on the concept's graph neighbourhood,
not its surface attributes.** That is also the most brain-faithful option — a
concept's semantic identity is its relations.

And it is exactly what `docs/GROUNDING_MECHANISM.md` proposes independently:
Committed-Proposition Gating commits to graph propositions rather than
projecting attributes. Two independent lines of evidence now converge on the
same design.
