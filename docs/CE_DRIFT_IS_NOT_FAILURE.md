# Why the training CE drifts upward without meaning failure

Observed on the 2026-11-13 linggen run (`proc_6ba2c8bfbbb1`):

```
pass  4/40: CE=1.970 aux=0.010
pass  8/40: CE=2.027 aux=0.001
pass 12/40: CE=2.051 aux=0.006
pass 16/40: CE=2.062 aux=0.006
```

Monotone rise. This is NOT evidence the decoder is getting worse.

## What `_avg_cross_entropy` actually is

`ravana_ml/src/ravana_ml/nn/neural_decoder.py:1163`:

```python
alpha = 0.05
self._avg_cross_entropy = (1 - alpha) * self._avg_cross_entropy + alpha * avg_ce
```

An **EMA over per-sentence training CE**, computed on the same batches the
decoder just updated on. It is a training-time diagnostic, not a held-out
validation metric.

## Why it can rise while generation improves

1. **Each pass samples a different subset.** `scripts/train.py:367` uses
   `rng2 = np.random.RandomState(1000 + i)` and `rng2.choice(len(sentences), size=min(pp, len(sentences)))`.
   With only 522 sentences and `pp=200`, each pass trains on ~38% of the corpus.
   The CE therefore reflects *which* sentences were drawn, not only learning.

2. **`freeze_core=True`** (`scripts/train.py:867`). The recurrent core is frozen;
   only the readout / conditioning path trains. The readout can shift in a way
   that raises per-token CE on the training batches while improving the
   free-form generation the promotion gate actually measures.

3. **The metric is on-training-batch.** A decoder that is exploring (higher
   entropy output) scores worse on teacher-forced CE while producing better
   free-form text.

## The gate that matters

`scripts/train.py:414-418`:

```python
quality = _grounded_generation_quality(engine, nd, cond_sent, W_sm, attr_enc, glove_fn)
COHERENCE_FLOOR = 0.5
use_linggen = bool(quality >= COHERENCE_FLOOR)
```

Free-form generation quality from the embodied BOS, floor 0.5. A previous run
scored **0.661**. CE is not part of this gate.

## Conclusion

Do not abort the run on a rising CE. Judge by:
- `_grounded_generation_quality` at pass 40 (must be >= 0.5),
- then the post-retraining `scripts/eval_decoder_groundedness.py` comparison
  against `docs/PRE_RETRAINING_BASELINE.md`.

If the coherence gate fails at pass 40, the run correctly stays fail-closed
(`use_linggen=False`) and the persisted decoder is untouched.
