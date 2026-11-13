# Why the decoder cannot say "gravity": the readout never learned content

## The chain of causation

The decoder emits fluent function-word salad because of a four-link chain. Each
link was measured on the persisted engine (`data_dir=weights`, `user_suffix=""`,
vocab 17,021, hidden 256, embed 75).

### Link 1 — the training corpus is 96% aphorisms

```
data/corpora/teen_seeds.txt          14,037 lines
data/corpora/grounded_descriptions.txt  512 lines
```

`scripts/train.py:98` uses `teen_seeds.txt` as the primary corpus. Its content:

```
A bird does not sing because it has an answer. It sings because it has a song.
A clever person turns great troubles into little ones, and little ones into none at all.
A cloud is water vapour in the atmosphere that has condensed into very small water droplets...
```

Every line is a complete, self-contained, short sentence. Every line **ends**.

### Link 2 — `freeze_core=True` trains only the readout

`scripts/train.py:867` passes `freeze_core=True`. Per the docstring at
`ravana_ml/src/ravana_ml/nn/neural_decoder.py:713`:

> `freeze_core`: If True, only update word_embedding and output_proj.
> Skips GRU, condition_proj, and attention updates.

So the GRU is frozen at its pre-trained state and only the readout learns.

### Link 3 — the GRU is effectively dead

Persisted GRU weight statistics versus a freshly initialised decoder:

| matrix | persisted mean\|W\| | persisted frobenius | fresh frobenius |
|---|---|---|---|
| W_z | 0.00286 | 1.480 | 22.66 |
| W_r | 0.00210 | 1.031 | 22.60 |
| W_h | 0.00621 | 2.835 | 22.63 |

The persisted GRU is **15× smaller** than a fresh initialisation, with all
biases exactly 0.0. It was not trained; it was shrunk (weight decay / free-energy
regularisation on an under-trained core). `condition_proj` (frob 21.77) is at a
normal scale; `W_h_bias` (0.424) is near-zero by design.

### Link 4 — the readout's hidden state points at `<eos>`

`cos(h, eos_row_direction)`:

| input | cos(h, eos) | cos(h, bos) |
|---|---|---|
| full gravity sentence | **+0.9710** | +0.9431 |
| full water sentence | **+0.9805** | +0.9402 |
| full music sentence | **+0.9706** | +0.9483 |
| single word `science` | +0.8330 | +0.8658 |
| single word `time` | +0.7662 | +0.7393 |
| single word `water` | +0.7117 | +0.7020 |
| single word `gravity` | +0.1093 | +0.1056 |

After reading a whole sentence, the hidden state points almost exactly at the
`<eos>` row. Even a single content word pushes it there.

### Contributing factor — embedding norms are wildly imbalanced

`word_embedding` row norms:

```
min 0.923   max 48.696   mean 1.038   median 1.001   std 0.531
p50 = 1.001   p99 = 1.527
```

99% of tokens sit at unit norm, but a handful of function words escaped:

| token | norm |
|---|---|
| `<bos>` | 48.696 |
| `the` | 30.432 |
| `to` | 16.354 |
| `a` | 16.103 |
| `of` | 14.565 |
| `in` | 11.237 |
| `and` | 10.336 |
| `is` | 9.463 |

Function words average **3.679** vs content words **1.365** (2.69×); the worst
offenders are 10–48×. After `condition_proj`: `the` → 86.9, `is` → 41.2,
`gravity` → 2.5. The GRU input is dominated by function words by two orders of
magnitude.

### The readout cannot produce its own input word

`_readout(_gru_step(emb(word)))` — feed a word, read out, does the word return?

| word | self-recovery rank |
|---|---|
| gravity | 10,975 |
| force | 922 |
| planet | 1,862 |
| ocean | 9 |
| water | 182 |
| music | 159 |

Over all 17,021 words: median rank **7,266**, top-1 rate **0.0001**, top-100
rate **0.0085**.

## What was ruled out

- **`output_proj.bias`**: effectively zero (min -0.032, max +0.030, std 0.003).
  Every named token sits at exactly +0.000. Not the cause.
- **Row norms**: `<bos>`/`<eos>` have the largest rows (1.678/1.644 vs mean
  0.915), but row-normalising the readout does not change the ranking, and
  `corr(row_norm, logit)` over the vocab is **0.04**. Not the cause.
- **Embedding normalisation**: row-normalising all embeddings drops
  `cos(h,eos)` from 0.971 to **0.841** and collapses the logit spread from 7.38
  to 1.20 — a real improvement in confidence, but the ranking is still
  `<bos>/<eos>/of/is/to/or/and`, and `gravity` self-rank stays 10,971.
  A contributor, not the cause.
- **The settle loop**: it does exactly what it was designed to do. `_pick`
  correctly refuses to emit `<eos>`; the fall-through to `is/of/to/the` is the
  correct behaviour given a readout that ranks them highest.
- **The CPG gate**: it constrains which propositions may be committed; it cannot
  make the readout produce a word the readout cannot rank.

## The actual defect

The readout was trained, with teacher forcing, to predict the next token of
short self-contained aphorisms. In such a corpus the *easiest* thing to predict
is the end of the sentence, and the state that follows any input is the
end-of-sentence state. The readout learned that. It never learned to predict a
content word in a mid-sentence position, because the corpus rarely requires it
and `freeze_core=True` prevented the recurrent state from adapting.

`gravity self-rank 10,975` is the cleanest statement of the defect: the readout
literally cannot produce the word `gravity` from the embedding of `gravity`.

## Why the training gate passed anyway

`_grounded_generation_quality` (floor 0.5) scored **0.641**. The gate samples
free-form output and scores coherence; a model emitting function-word salad
scores above a low floor because the scorer is not measuring groundedness. The
gate is measuring something other than what it claims.

## The fix

The objective must require content prediction in mid-sentence positions:

1. **Train on descriptions, not aphorisms.** `grounded_descriptions.txt` has
   real mid-sentence content ("A cat is a small domesticated carnivorous mammal
   (Felis catus), kept by humans as a pet"). Use it as the primary corpus.
2. **Unfreeze the core** (at least `condition_proj` and `W_h_bias`) so the
   recurrent state can learn to carry content rather than collapsing to an
   end-of-utterance attractor.
3. **Normalise the embeddings** (or clamp outliers) so function words cannot
   dominate the GRU input by 30×.
4. **Separate "stop" from "what next".** The brain does not make
   end-of-utterance the attractor of the language hidden state; it maintains a
   continuous Go/NoGo signal that *modulates* production. A decoder whose `h`
   collapses to "stop" has the NoGo pathway stuck on.

Until the readout can rank a content word above `<eos>` when the concept is
active, no amount of gating, conditioning, or boosting will produce grounded
prose.
