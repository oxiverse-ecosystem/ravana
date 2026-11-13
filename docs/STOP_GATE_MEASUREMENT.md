# The stop gate fixes termination, not selection

**Measurement date:** 2026-11-13
**PR:** #340 (`fix/eos-absorbing-sink`)
**Instrument:** `scripts/eval_decoder_groundedness.py --engine persisted --cpg --cpg-relation`

## What was measured

The Go/NoGo stop gate (masking `<bos>/<eos>/<pad>/<unk>` out of the lexical
softmax and deciding termination on a separate annealed `cos(h, eos)`
criterion) was turned ON by default and the instrument re-run against the
same 480-row subject set as the post-retraining baseline.

## Result: no measurable grounding change

Paired comparison on 480 rows (`scripts/compare_groundedness_runs.py`):

| metric | mean delta | improved | worsened | same |
|---|---|---|---|---|
| on-topic fraction | +0.0224 | 168 | 148 | 164 |
| unsupported assertions | −0.0792 | 129 | 134 | 217 |
| supported assertions | +0.0813 | 120 | 117 | 243 |
| content-word cosine | +0.0039 | 227 | 230 | 23 |
| word count | +0.6687 | — | — | 480 |

Every delta is within the instrument's own run-to-run noise. The per-row
swings are large (±0.94 on on-topic for individual subjects) because the
decoder samples with temperature — re-running the same configuration
produces different outputs. The effect size is smaller than that variance.

Aggregate metrics, post-retraining vs stop-gate-on:

| metric | post-retrain | stop gate on |
|---|---|---|
| ungated decoder on-topic | 0.4505 | 0.4734 |
| ungated decoder unsupported | 1.1837 | 1.2035 |
| ungated decoder supported | 0.5481 | 0.5436 |
| ungated decoder words | 8.93 | 8.96 |
| gated on-topic | 0.3616 | 0.3683 |
| gated unsupported | 0.9338 | 0.9617 |

## What the gate does change (and why it is still worth keeping)

The gate's effect is on **termination style**, not grounding. Direct probe
of the settle generator on the same subject:

```
gate OFF:  gravity -> gravity overshadowed novita millennia burdens aid
                      narrow traits prairie differentiating met disruptions
                      summaries commodity fitrihidajati one is the
gate ON:   gravity -> gravity waste elissa energy lyotard favor mechanik
                      paradoxically eggs
```

The function-word tail (`one is the`, `may computer is the`,
`framework or paradigm antarctic khz`) is gone. That is the intended
effect: `<eos>`/`<bos>` no longer compete as lexical items, so the
softmax is over words only.

## Why the grounding did not improve

The gate addresses **which tokens compete**, not **which token is
correct**. The readout still cannot pick the right content word, because
the GRU hidden state does not encode next-word content — it encodes
sentence position (`docs/HIDDEN_STATE_COLLAPSES_TO_EOS.md`). Removing
`<eos>` from the competition removes the *symptom* (the collapse onto an
ending direction) but the underlying representation is unchanged, so the
argmax is simply the next-most-degenerate direction.

This is the same pattern as the escalation path
(`docs/ESCALATION_INERT.md`): a mechanism that is correct in principle
and inert in measurement because the condition it acts on is not the
binding constraint.

## Conclusion

Keep the gate — it is a real defect fix (control tokens should not be
lexical competitors) and it makes output visibly cleaner. Do not claim it
improves grounding; the measurement does not support that.

The binding constraint remains the representation: the hidden state must
come to encode *what word comes next*, not *where we are in the sentence*.
That is the subject of `docs/READOUT_CANNOT_PREDICT_CONTENT.md` and the
parallel research/audit tasks.
