# The decoder emits a fixed filler register, not subject-conditioned text

Date: 2026-11-14
Instrument: `scripts/diagnose_decoder_salad.py` (40 subjects with NO stored
definition -- the case where production actually consults the decoder)

## The finding

Two different subjects, same frame:

```
what is like    -> i might be reaching, but Revolution one such used often there.
what is being   -> Type energy other system such one used there often.
```

The outputs share the skeleton `one such used often there`. The decoder is not
conditioning on the subject; it emits a **fixed filler register** with minor
variation.

## Token distribution over 40 novel subjects

```
such 24, one 23, used 22, other 20, there 20, unfolds 16, often 16,
this 13, process 13, energy 13, and 12, like 12, appears 9, through 7,
system 7, that 7, birth 6, species 6, begins 5, believe 5, new 5, ...
```

- **top-10 tokens are 39.4% of all tokens**
- **133 distinct tokens / 457 total** (type-token ratio 0.29)

The top tokens are function words and vague generics (`such`, `one`, `used`,
`other`, `there`, `often`, `this`, `and`, `like`). The only content words that
appear at all (`process`, `energy`, `species`, `birth`) are the SAME regardless
of subject.

## What this means

The decoder learned the **register** of encyclopedic prose -- the connective
tissue -- without learning to bind it to a subject. That is exactly the profile
of the failure the codebase's own comments call "fluent, grammatical, empty
speech": the surface statistics of definitions with the semantic content absent.

This is the fallback path for subjects RAVANA has no stored definition for, so
it is where "factual" breaks. For subjects WITH a definition, production emits
the stored text and scores 99/100; the decoder never runs.

## Relation to the prior audit

`docs/CODEBASE_DECODER_AUDIT.md` found the training/inference readout mismatch
(training reads `h`, inference reads `h_star`). An auxiliary loss to fix it was
attempted and **measured as noise over 8 seeds**, so it was reverted
(`docs/FULL_RETRAIN_VS_PARTIAL.md` records the measurement).

The evidence here suggests the mismatch is real but not sufficient: even with
the readout aligned, a model trained on 512 short definitions may simply not
have enough signal to learn subject-conditioned generation. The conditioning
path exists (`initial_emb` / `persistent_emb` via the fitted `W_sm`), but the
emitted tokens do not track it.

## Reproduce

```
uv run python scripts/diagnose_decoder_salad.py --limit 40
```
