# `W_sm` is a near-null projection — the grounding pass learned nothing

**Status:** measured. This is the root cause of the off-topic decoder.
**Consequence:** the grounding pass is not "outweighed" — it is **empty**. No
amount of training balance will fix it until the projection itself carries
signal.

## The measurement

`data/linggen_wsm.npz`, loaded onto the decoder at boot (PR #318):

```
shape          (75, 65)
mean |weight|  0.01822
max  |weight|  0.28301
frobenius      2.44087
diag mean     -0.00068
diag |mean|    0.01594
row-norm mean  0.27376
col-norm mean  0.19653

reference: random N(0, 0.018) 325-element matrix -> frobenius 1.25061
```

**The fitted projection is indistinguishable from noise.** Its Frobenius norm
(2.44) is barely above a random matrix of the same scale (1.25), the diagonal
is flat (mean -0.0007), and no row or column carries meaningful weight.

A ridge regression that has found real structure produces a low-rank,
concentrated matrix. This one is diffuse — the signature of a fit on too little
data, or on targets that carry no linear relationship to the inputs.

## What this means

`train_decoder_grounded` fits `W_sm` to map a 65-D Binder vector to a 75-D
dual-code embedding, then trains the decoder to reproduce KB descriptions
conditioned on that projection.

If `W_sm ≈ noise`, the conditioning signal is noise. The decoder is being
trained to condition on something that carries no information about the
concept — so it learns to ignore conditioning, which is exactly what the
groundedness baseline shows:

```
on_topic 0.133   unsupported 1.59   cosine 0.338
```

**This also explains why PR #318 had no effect.** Loading a near-null
projection correctly changes nothing.

## Correcting my earlier diagnosis

I previously wrote (`GROUNDING_PASS_TOO_WEAK.md`) that the grounding signal was
**outweighed** — 15,000 grounding presentations against 400,000 CE
presentations, ratio 1:27.

That was wrong. The ratio is real but irrelevant: the signal itself is empty.
Rebalancing would just give the decoder more presentations of noise.

## What has to happen

The fix is upstream of training balance:

1. **Diagnose the fit.** How many pairs did `W_sm` fit on? What is the
   held-out R² of the 65→75 mapping? If R² ≈ 0, the Binder vector does not
   linearly predict the dual-code embedding — and a ridge fit will always
   return noise.
2. **If the mapping is not linear, change the model.** A 2-layer MLP, or fit on
   a different target. The brain's semantic→form mapping is not a linear
   projection either.
3. **Only then re-balance passes.** Once `W_sm` carries signal, the 1:27 ratio
   becomes worth addressing.

## The honest sequence

```
WRONG:  grounding pass exists but is outweighed by CE        (my first claim)
WRONG:  the conditioner was not loaded at boot               (PR #318 -- real bug, no effect)
RIGHT:  the conditioner is loaded and is ~noise, so the
        grounding pass never had a signal to deliver
```

Each step was a real finding. The first two were not the cause.

## Related

- `docs/DECODER_GROUNDEDNESS_BASELINE.md` — the instrument and baseline
- `docs/CONDITIONER_FIX_NO_EFFECT.md` — the negative result that forced this
- `docs/GROUNDING_PASS_TOO_WEAK.md` — superseded by this file
- `docs/GROUNDING_MECHANISM.md` — the runtime mechanism (still valid; it does
  not depend on `W_sm`)
- PR #318 (`7682a87`) — the boot-load fix, kept: loading a fitted artefact is
  correct regardless of its quality
