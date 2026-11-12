# `W_sm` has R² = 0.22: weak, not noise

**Status:** measured. This partially corrects `WSM_IS_NOISE.md`.
**Conclusion:** the projection carries *some* signal — about 22% of the
dual-code embedding's variance is linearly predictable from the Binder vector.
That is weak but not nothing, and it changes what the fix should be.

## The measurement

Rebuilt the exact `(X, Y)` the fit uses — 250 grounded description pairs,
`X` = 65-D Binder vector, `Y` = 75-D dual-code embedding — and evaluated a
ridge fit (λ=1e-3) on an 80/20 split:

```
grounded pairs in file: 250
usable pairs:           250
X shape (250, 65)   Y shape (250, 75)

held-out R²   0.21838
held-out RMSE 0.08974
target variance (mean per-dim) 0.01030
```

## Correcting myself, again

`WSM_IS_NOISE.md` concluded the projection was "indistinguishable from noise"
from its Frobenius norm (2.44 vs 1.25 for a random matrix of the same scale).

**That inference was too strong.** A diffuse matrix is not necessarily a
useless one — a ridge solution on 250 correlated inputs is spread across many
dimensions by construction. The norm alone cannot distinguish "noise" from
"weak but real".

The held-out R² is the right instrument, and it says **0.218**.

## What R² = 0.22 actually means

* **Not noise.** 22% of the target variance is linearly recoverable. A random
  projection would score ≈ 0.
* **Not good enough.** 78% of the signal is missing. Conditioning on a
  one-fifth-accurate target cannot steer generation reliably.
* **Consistent with everything observed.** The decoder is off-topic 85% of the
  time, and loading `W_sm` correctly changed nothing — because a 0.22-quality
  signal is too weak to move a decoder trained on 400,000 general-prose
  presentations.

## Why the fit is weak

Two candidate reasons, and they need different fixes:

**1. The mapping may not be linear.** A single ridge layer from a 65-D
Binder vector to a 75-D dual-code embedding is a strong assumption. The
brain's semantic→form mapping is not a linear projection. A 2-layer MLP would
tell us immediately: if its held-out R² jumps, linearity was the limit.

**2. 250 pairs is thin** for a 65→75 mapping (4,875 parameters). More grounded
pairs would help any model.

The decisive test is the MLP: if R² stays ~0.22 with a nonlinear model, the
Binder vector simply does not contain the information, and the *input* is the
problem — not the model.

## Where this leaves the plan

```
WRONG:  grounding pass is outweighed by CE           (ratio real, irrelevant)
WRONG:  conditioner not loaded at boot               (real bug, no effect)
WRONG:  W_sm is noise                                (norm misread)
RIGHT:  W_sm has R^2 = 0.22 -- weak but real, and
        too weak to steer a decoder trained on
        400k general-prose presentations
```

The fix is now well-scoped: **improve the conditioning signal** (nonlinear
model, more pairs, or a richer input), then re-measure the groundedness
baseline. Training balance is still not the lever.

## Related

- `docs/WSM_IS_NOISE.md` — superseded on the "noise" claim
- `docs/CONDITIONER_FIX_NO_EFFECT.md` — the negative result
- `docs/DECODER_GROUNDEDNESS_BASELINE.md` — the instrument and baseline
- `docs/GROUNDING_MECHANISM.md` — the runtime mechanism, which does not depend
  on `W_sm` at all
