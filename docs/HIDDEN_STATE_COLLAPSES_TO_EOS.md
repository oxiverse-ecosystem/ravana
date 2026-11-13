# The decoder's hidden state points at `<eos>` — measured

## Summary

The decoder produces fluent, grammatical, empty speech because its recurrent
hidden state has learned **"the sentence is over"** as its default. After
reading a real sentence, `h` has cosine **+0.97** with the `<eos>` readout row
and **+0.94** with `<bos>`. The readout is not broken; it is faithfully
reporting a hidden state that encodes end-of-utterance.

This is a *training objective* artefact, not a bug in the settle loop, the
conditioning, or the CPG gate.

## Measurements

All on the persisted engine (`data_dir=weights`, `user_suffix=""`), vocab
17,021, hidden 256, embed 75.

### 1. The readout cannot recover its own target

`_readout(_gru_step(emb(word)))` — i.e. feed a word, read out, does the word
come back?

| word | self-recovery rank |
|---|---|
| gravity | 10,975 |
| force | 922 |
| planet | 1,862 |
| ocean | 9 |
| water | 182 |
| music | 159 |

Over all 17,021 words: median rank **7,266**, mean 7,608, top-1 rate
**0.0001**, top-100 rate **0.0085**.

### 2. The readout's argmax is end-of-sentence

Next-word rank on teacher-forced real trajectories:

| sentence | ranks | median | top-1 |
|---|---|---|---|
| Gravity is the force by which a planet draws objects toward its centre | 11, 0, 945, 30, 18, 5, 61, 11361, 176, 4484, 53, 1720 | 57 | 1/12 |
| Water is a transparent tasteless odourless liquid essential for life | 7, 1, 4190, 4401, 4244, 1001, 334, 12, 29 | 334 | 0/9 |
| Music is sound organized in time | 6, 1284, 820, 7, 70 | 70 | 0/5 |

Argmax after each word of the gravity sentence:

```
after gravity    -> of(0.10), on(0.08), between(0.08), system(0.07), environment(0.07)
after is         -> the(4.32), a(4.31), <bos>(4.24), <eos>(4.07), an(3.98)
after the        -> <bos>(6.22), original(5.80), same(5.18), <eos>(5.08), world(4.88)
after force      -> <bos>(6.87), <eos>(6.51), of(5.26), is(4.97), to(3.90)
after which      -> <bos>(7.42), <eos>(7.40), is(5.51), of(5.25), the(5.24)
after centre     -> <eos>(7.38), <bos>(7.31), is(5.61), of(5.48), to(5.15)
```

The logits are not flat — they are *confidently* end-of-sentence.

### 3. The hidden state points at `<eos>` — the actual cause

`cos(h, eos_row_direction)`:

| input | cos(h, eos) | cos(h, bos) |
|---|---|---|
| full gravity sentence | **+0.9710** | +0.9431 |
| full water sentence | **+0.9805** | +0.9402 |
| full music sentence | **+0.9706** | +0.9483 |
| single word `science` | +0.8330 | +0.8658 |
| single word `time` | +0.7662 | +0.7393 |
| single word `philosophy` | +0.7120 | +0.6727 |
| single word `water` | +0.7117 | +0.7020 |
| single word `music` | +0.6284 | +0.5453 |
| single word `love` | +0.6013 | +0.5398 |
| single word `ocean` | +0.4787 | +0.4674 |
| single word `gravity` | +0.1093 | +0.1056 |

Even a *single* content word pushes `h` toward the end-of-sentence direction.
`gravity` is the lone exception.

### 4. Ruled out: the bias and the row norms

- `output_proj.bias` is effectively **zero**: min -0.032, max +0.030, mean 0.000,
  std 0.003. Every named token (`<eos>`, `<bos>`, `is`, `the`, `gravity`) sits
  at exactly +0.000. The bias is not driving this.
- Row norms: `<bos>` 1.678 and `<eos>` 1.644 are the two largest rows in the
  matrix (mean 0.915); function words are all above mean (`is` 1.388, `the`
  1.335, `of` 1.327) and content words at or below it (`gravity` 0.999).
- But row-normalising the readout does **not** fix the ranking — EOS/BOS and
  `and/of/to/is` still top the list. `corr(row_norm, logit)` over the vocab is
  only **0.04**. The logits are driven by the *direction of `h`*, not by row
  norms.

## Why this happened

`scripts/train.py` trains with `freeze_core=True` and teacher-forced CE on
`data/corpora/teen_seeds.txt` — 14,037 lines of aphorisms and book blurbs.
Every training sentence ends. The readout is trained only on hidden states the
GRU actually visits in that corpus, and the state that follows *any* input in
a corpus of short complete sentences is the end-of-sentence state.

The settle loop then does exactly what it was designed to do:

1. `h_star = _gru_step(proj(target_emb), h)` — an end-of-sentence state.
2. `h += settle_lr * (h_star - h)` for 8 steps at lr 0.25 -> `h` is 90% of the
   way to `h_star`.
3. `_pick(_readout(h))` — the readout says `<eos>`/`<bos>` with logit 7-15.
4. `_pick` rejects `<...>` tokens and falls through to the next-best
   non-special token: `is`, `of`, `to`, `the`.

That is the function-word tail. It is not a bug in `_pick`; `_pick` is
correctly refusing to emit `<eos>`.

## What this does and does not explain

**Explains:** function-word tails ("is the of a to or in and that as an for
by is"), short outputs (8.93 words post-retrain vs 11.88 pre), the readout's
inability to recover a target word, and why token boosts of 2.5-8.0 barely move
the output (the boost is applied to logits that are already 7-15 nats below
the EOS direction).

**Does not explain:** why the training quality gate
(`_grounded_generation_quality`, floor 0.5) passed at 0.641. The gate samples
free-form output and scores coherence; a model that emits function-word salad
can still score above a low floor if the scorer is lenient. The gate is
measuring something other than groundedness.

## Consequence for next steps

The fix is not in the settle loop, the conditioning, or the gate. It is in
what the readout is trained on:

- Train on data where content words are the *prediction targets* in
  mid-sentence positions, not only at sentence ends.
- Or train the readout without `freeze_core`, so the recurrent state itself
  learns to carry content rather than collapse to an end-of-sentence attractor.
- Or add an explicit end-of-utterance decision that is *separate* from the
  content readout, so "stop" is not the default direction of `h`.

The brain-faithful framing: the cortical language loop does not treat
end-of-utterance as the attractor of the hidden state. It maintains a
continuous "continue / stop" signal (basal-ganglia Go/NoGo) that is *modulatory*
on the language-production loop, not the default fixed point of it. A decoder
whose hidden state collapses to "stop" has the NoGo pathway stuck on.
