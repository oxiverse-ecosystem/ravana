# The decoder vocabulary covers 61% of common words -- 93% of descriptions are unusable

**Status:** measured. This is the real blocker on retraining the decoder, and
it supersedes "the corpus is too small" as the framing.

## The measurement

```
decoder vocab: 15000 words
Gutenberg top-5000 words covered: 3032 (61%)
```

Per-word coverage is 61%. Training requires **every** word of a sentence to be
in the vocabulary, so:

```
sentence length  5 -> P(all in vocab) = 8.4%
sentence length 10 -> P(all in vocab) = 0.7%
sentence length 22 -> P(all in vocab) = 0.02%
```

Measured on the actual corpus:

```
wikipedia    rows=512  fully-in-vocab= 37 ( 7%)
local_books  rows=200  fully-in-vocab= 22 (11%)
```

**93% of the harvested descriptions cannot be trained on.**

## Why this was invisible

The corpus work (#332, #333, #335) fixed the *pipeline* and grew the corpus
from 250 to 512 clean pairs. That is real progress, but it does not help if
the decoder cannot represent the sentences.

The vocabulary was built from `teen_seeds.txt` (the aphorism corpus). It
contains `aail`, `aails`, `aaron` -- but not `transparent`, `colourless` or
`treatment`:

```
common words tested: 41  MISSING: 3
missing: ['transparent', 'colourless', 'treatment']
```

A 15k slice of a 14k-line corpus is necessarily narrow.

## What this means

Retraining on the current corpus would train on ~37 sentences out of 512. That
is below `_MIN_PAIRS = 8`... it passes, but it is a fraction of the intended
data, and it biases toward short sentences.

The fix is the vocabulary, not the corpus:

1. **Rebuild the vocab from the description corpus.** `train.py`'s
   `load_corpus` already does this -- it adds corpus words to
   `_decoder_word_to_idx` before freezing. The question is whether the
   *persisted* vocab (`data/neural_decoder_vocab.json`) is rebuilt when the
   corpus changes, or only when the seed corpus changes.
2. **Or lower the per-sentence requirement** -- train on the in-vocab subset
   of each description rather than requiring the whole sentence.

Option 1 is correct. A vocabulary fitted to aphorisms cannot represent
descriptions.

## The two harvests, compared

```
               pairs   quality                          trainable
wikipedia       512    real definitions                 7%
local_books     200    usage sentences, fragments       11%
```

The local-books harvest is **20s for 200 concepts** vs ~14 min for 1000
Wikipedia concepts -- 40x faster. But its output is dialogue and chapter
headings:

```
psychology  "That's a simple point in psychology
life        I almost wish I hadn't gone down that rabbit-hole—and yet—and yet—
learning    "And so these three little sisters—they were learning to draw, you know—"
human       CHAPTER IV Human nature is so well disposed towards those who are in interesting
moral       I can't tell you just now what the moral of that is, but I shall remember it in
```

`_mode_linggen` calls `harvest_grounded_corpus(force=True)`, which **overwrites**
the Wikipedia file with these. That is a regression in quality for a 40x speed
gain -- and since only ~10% of either is trainable, the speed gain buys
nothing.

**Recommendation:** keep the Wikipedia corpus, and fix the vocabulary.

## Related

- #331 -- the root cause (trained on aphorisms)
- #332, #333, #335 -- the harvest pipeline fixes
- `docs/DECODER_TRAINED_ON_APHORISMS.md`
- `docs/HARVEST_RESTORED.md`
