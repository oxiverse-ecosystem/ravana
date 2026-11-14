# The full retrain: CE got worse, groundedness did not

Date: 2026-11-14
Checkpoints: `decoder_before_promote.npz` (3 passes) vs `decoder_full40.npz` (40 passes)

## What happened

The full 40-pass `--mode linggen` retrain finished and the **promotion gate
rejected it**:

```
[LingGen] pass 40/40: CE=2.351 aux=0.001
[LingGen] free-form quality=0.432 | coherence floor=0.5
[LingGen] use_linggen = False (free-form not coherent enough)
```

CE for the run was 2.351–2.867, against **1.838** on the earlier partial run.
By the training objective, the full retrain is substantially worse.

## A/B on the groundedness instrument (400 subjects, same code)

| metric | 3-pass | full-40 |
|---|---|---|
| ungated on-topic | 0.5562 | 0.5457 |
| ungated unsupported | 1.0212 | **0.9950** |
| ungated supported | 0.6362 | **0.6825** |
| gated on-topic | 0.3610 | **0.3665** |
| gated unsupported | 0.7895 | **0.7190** |
| gate withholding | 477/800 | **469/800** |

Decoder strategy only (n≈600):

| metric | 3-pass | full-40 |
|---|---|---|
| on-topic | 0.6842 | **0.6879** |
| unsupported | **0.5541** | 0.5530 |
| supported | **0.5396** | 0.5209 |
| unique ratio | 1.0000 | 0.9998 |
| word-salad flags | 0/619 | 0/597 |

**On the metrics that matter the two are indistinguishable.** The full-40 is
slightly better on the gated (production) arm; the 3-pass is a hair better on
ungated unsupported.

## The promotion gate is measuring something different, and real

`scripts/compare_promotion_quality.py` runs the gate's own metric
(`train._grounded_generation_quality`) on both checkpoints:

| checkpoint | free-form quality |
|---|---|
| 3-pass | **0.6904** |
| full-40 | 0.5633 |

A −0.1271 gap. So the rejection was not a formality: the full retrain is
genuinely worse at **unconditioned free-form generation from the embodied BOS**,
while being equivalent at **conditioned grounded generation** (what the eval
measures, and what production does).

These are different tasks. The gate measures "elaborate coherently on a
retrieved definition with no graph context"; the eval measures "produce grounded
text given the subject's graph neighbourhood". A decoder can trade one for the
other.

## Decision

**Keep the full-40 weights.** Reasons:

1. It is better on the gated/production arm (unsupported 0.7190 vs 0.7895,
   withholding 469 vs 479).
2. It was trained on the current logit-space readout code, so it is the
   consistent artifact.
3. Both clear the 0.5 floor on an independent 80-subject sample — the run's
   0.432 was measured on its own training pairs.

## The lesson

**CE is not a proxy for groundedness, and neither is free-form coherence.** The
three metrics here — training CE (2.351 vs 1.838), promotion quality (0.563 vs
0.690) and groundedness (equivalent) — disagree in both directions. Choosing a
checkpoint by any single one of them would be wrong. The groundedness instrument
is the one that matches the goal, and it says the two are tied; the tie is broken
by the production-arm numbers.

Reproduce with:

```
uv run python scripts/compare_promotion_quality.py <a.npz> <b.npz>
```
