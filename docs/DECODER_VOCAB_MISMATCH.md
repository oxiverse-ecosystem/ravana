# The trained decoder is discarded at runtime

**Status:** verified by tracing the live engine and the training script.
**Consequence:** the 500,000-example training run has no effect on the running
system. The decoder cannot produce language, and no amount of further training
will change that until this is fixed.

## The measurement

```
training:   "Vocab: 15000 words"   CE=1.570  top-1=0.621  top-5=0.790
runtime:    word_to_idx: 217
            output_proj out_features: 217
            word_embedding num_embeddings: 217
            MAX_DECODER_VOCAB_SIZE: 15000
            graph nodes: 228
```

The decoder is trained on a **15,000-word** vocabulary. At runtime it is given
**217 words** -- function words plus graph concept labels, derived from 228 graph
nodes.

Asked to generate with that vocabulary, it emits one word and stops:

```
raw tokens: [35, 1]
words (1): if
```

"if" is a function word. It is nearly all the runtime vocabulary can express.

## Two independent defects

### 1. No persistence: the trained weights are never saved

`scripts/train.py` trains the decoder in memory. There is no `torch.save`, no
`np.savez`, no artifact of any kind -- `ls data/ | grep -i decoder` is empty.

Every training run's weights vanish when the process exits. The 500,000-example
run that reached CE 1.570 left nothing behind.

### 2. The runtime rebuilds the vocabulary from scratch

`_build_decoder_vocab` (`response_gen.py:255`) has two branches:

- **Vocab dicts already populated** (line 264): reuse them -- but still
  construct a **fresh `NeuralDecoder` with random weights**, then copy in only
  the *embedding* rows for words that have a GloVe vector. The GRU, the output
  projection, and every other learned parameter are new and random.
- **Vocab dicts empty** (line 291): build from function words + graph concept
  labels, capped at `MAX_DECODER_VOCAB_SIZE`.

Either way the trained parameters are discarded. The first branch looks like a
resume path but is not one -- it restores embeddings and nothing else.

## Why this explains every symptom

| symptom | cause |
|---|---|
| decoder emits "if" and stops | 217-word runtime vocab, mostly function words |
| `situation_model_decoder` never observed | the decoder branch cannot produce 5+ tokens, so `len(generated) >= 5` fails and control falls to `situation_model_syntax` |
| `use_linggen` appears inert | the flag gates a path that cannot succeed |
| every "decoder" reply was a template | the decoder never completed a generation |

The gate numbers (decoder_ce 3.8228 vs template_ce 3.7699) were never measuring
the trained decoder. They were measuring the template path twice.

## What has to change

1. **Persist the trained decoder** -- weights plus the vocabulary it was trained
   with, written to `data/` by `scripts/train.py`.
2. **Load it at runtime** -- `_build_decoder_vocab` must restore the saved
   vocabulary and the saved parameters, not rebuild a 217-word concept vocab.
3. **Re-measure** -- only then does `measure_linggen_ce.py` observe the decoder
   that was actually trained.

Until (1) and (2) land, `use_linggen` cannot be promoted for any reason, and
further training is wasted: the next run will also be discarded.

## Note on the vocabulary design

A 217-word concept vocabulary is not inherently wrong -- a decoder that speaks
in graph concepts and is then realised into surface form is a legitimate
architecture (and is what `situation_model_syntax` does). But that is a
*different decoder* from the one trained on 15,000 words of corpus text. The two
designs must not be mixed: either train the concept-vocabulary decoder, or
persist and load the corpus decoder. Currently neither happens.
