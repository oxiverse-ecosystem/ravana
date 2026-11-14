# The conditioning signal collapses to a single point

Date: 2026-11-14
Instrument: `scripts/diagnose_wsm_collapse.py`, `scripts/measure_conditioning_effect.py`

## The finding

The decoder's conditioning path is:

```
subject -> GloVe(64-D) -> CombinedAttributeEncoder -> signature(65-D)
        -> W_sm(75x65) -> conditioned BOS(75-D) -> GRU
```

Measured over 12 corpus subjects:

```
sky      |sig|=11.419  sig[:4]=[3.134 3.134 3.134 3.134]
blue     |sig|=11.504  sig[:4]=[3.176 3.176 3.176 3.176]
cat      |sig|=11.591  sig[:4]=[3.101 3.101 3.101 3.101]
gravity  |sig|=10.977  sig[:4]=[2.915 2.915 2.915 2.915]
...
```

- **The first dimensions are IDENTICAL within each subject** -- the signature is
  near-constant across its own dimensions.
- **Pairwise cosine of the projected vectors: mean 0.9990** (min 0.9965).
- **`W_sm` effective rank: 9 of 65** (singular values max 1.845, min 0.0).

So `gravity` and `ocean` produce conditioning vectors with cosine **0.9996**.
The subject information does not reach the GRU.

## Consequence

```
what is like    -> i might be reaching, but Revolution one such used often there.
what is being   -> Type energy other system such one used there often.
```

Same skeleton regardless of subject. The decoder learned the register of
encyclopedic prose without a subject signal to bind it to.

## Where it breaks

`CombinedAttributeEncoder.attribute_vector` (attribute_encoder.py:280) takes a
64-D GloVe vector and produces the 65-D signature. The Lancaster probe supplies
most dimensions (`val > av[j]` keeps the larger), and Lancaster norms are known
to be similar across unrelated words -- every noun scores moderately on
"visual", "physical", "animated", etc. The signature therefore encodes
**typicality**, not identity.

`W_sm` was fitted to map these signatures to conditioned BOS vectors. With
effective rank 9, it cannot separate subjects that the signature already
barely separates.

## What this is NOT

- Not the readout mismatch (that was measured as noise and reverted).
- Not the attention head (fixed in #344).
- Not the negative sampling (fixed in #345).
- Not the grounding gate (fixed in #351/#358/#359).

It is upstream of all of those: **the conditioning signal itself carries
essentially no subject information**, so no amount of decoder-side correction
can make the output subject-relevant.

## Candidate fixes (not attempted)

1. **Condition on the GloVe vector directly**, bypassing the attribute encoder.
   The 64-D GloVe vector separates subjects well (that is what the grounding
   gate uses for its similarity checks). `W_sm` would need re-fitting from
   signature -> GloVe, which `train.py --mode linggen` already does.
2. **Re-fit `W_sm` on a signature that includes an identity-bearing component**
   (e.g. concatenate the raw GloVe vector, or use a higher-rank projection).
3. **Add a subject-token constraint at generation time** -- the codebase already
   has `token_boost` / `content_word_ids`, which is the cheapest intervention
   and needs no retraining.

Option 3 is the only one that does not require re-fitting, and it is the natural
next step.

## Reproduce

```
uv run python scripts/diagnose_wsm_collapse.py 12
uv run python scripts/measure_conditioning_effect.py gravity ocean
```
