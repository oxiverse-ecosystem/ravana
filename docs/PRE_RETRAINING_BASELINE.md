# Pre-retraining decoder baseline

Captured 2026-11-13 before the linggen retraining run wrote new weights.
Instrument: `scripts/eval_decoder_groundedness.py --engine persisted --cpg --cpg-relation`
Engine: `data_dir=weights`, `user_suffix=""` — the aphorism-trained decoder
(`data/neural_decoder.npz`, 45,873,256 bytes, mtime 13:26).

This is the comparison point for the post-retraining measurement.

## Result

```
GATED, graph-known subjects (n=317):
    mean on-topic fraction      : 0.3295
    mean unsupported assertions : 1.1009
gate withholding rate: 317/480 subjects produced gated output; 163 withheld/none
ungated strategy distribution: {'situation_model_decoder': 474, 'situation_model_narrative': 6}

UNGATED, strategy=situation_model_decoder (graph-known, n=472):
    mean on-topic fraction      : 0.4033
    mean content-word cosine    : 0.4360
    mean unsupported assertions : 1.7331
    mean supported assertions   : 0.2966
    mean word count             : 11.88
    mean unique ratio           : 1.0000

UNGATED, strategy=situation_model_narrative (graph-known, n=6):
    mean on-topic fraction      : 0.2311
    mean content-word cosine    : 0.2213
    mean unsupported assertions : 1.1667
    mean supported assertions   : 0.8333
    mean word count             : 14.50
    mean unique ratio           : 0.9462
```

## What to compare after retraining

The decoder is the strategy that matters (`situation_model_decoder`, n=472 of 478
ungated). Watch:

- **on-topic fraction** 0.4033 — should RISE if descriptions teach it to stay on subject
- **unsupported assertions** 1.7331 — should FALL if it learns to assert only what the
  description corpus supports
- **supported assertions** 0.2966 — should RISE
- **gate withholding rate** 163/480 — should FALL if more output clears
  `_sm_response_grounded`

## Caveat

The decoder samples with temperature, so single runs are noisy. Compare on the
same subject set and treat differences under ~0.02 as noise. The instrument's
`--out` TSV allows a per-subject diff.
