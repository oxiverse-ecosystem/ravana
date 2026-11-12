# The decoder already has a grounding pass. It is not enough.

**Status:** traced. This reframes the "pure CE" diagnosis.
**Consequence:** the fix is not "add grounding to training" — grounding is
already there and the decoder still produces fluent nonsense.

## The existing mechanism

`scripts/train.py:187` — `train_decoder_grounded()`:

```
Reads data/corpora/grounded_descriptions.txt (concept<TAB>human description)
For each pair:
  target 75-D dual-code embedding = engine._embed_75d(concept)
  conditioning signal = 65-D Binder vector = _combined_attr_encoder
                        .attribute_vector(glove64(concept))
Fits LingGenConditioner (ridge W_sm: 65->75), persists data/linggen_wsm.npz,
then trains the decoder to reproduce each description given its embodied
signature.
```

This is a real grounding mechanism: the decoder is trained to produce text
**conditioned on the concept's own embodied signature**, from curated KB
definitions. It runs before the main CE pass (`load_corpus`, line ~120).

**And it ran.** Both artefacts exist:

```
data/corpora/grounded_descriptions.txt   32,638 bytes   Nov  9 07:47
data/linggen_wsm.npz                       41,868 bytes   Nov  9 10:05
```

## So why is the output still ungrounded?

The grounding pass is **30 passes at 500 sentences** (`n_passes=30, pp=500`),
then the main CE pass runs **200 passes at 2000 sentences** on the seed corpus
(`n_passes=200, pp=2000`).

That is **40× more CE training than grounding training**, and the CE corpus is
`teen_seeds.txt` — general prose with no concept-conditioning requirement.

So the decoder learns:
1. to paraphrase KB definitions when strongly conditioned (30 passes), then
2. to produce fluent general English (200 passes, 40× the data)

Step 2 washes out step 1. The measured output is the result:
*"It paves the way for brain. It is opposite to data."*

## The precise diagnosis

Not "no grounding term". The grounding signal is **outweighed**:

```
grounding:  30 passes x 500 sentences  =  15,000 sentence-presentations
CE:        200 passes x 2000 sentences = 400,000 sentence-presentations
```

Ratio **1 : 27**. And the two objectives are different in kind — one teaches
"say what is true of this concept", the other teaches "produce fluent English".

A gradient update does not distinguish them; the larger, more frequent signal
dominates.

## What this changes

The obvious fix ("add a grounding term to the objective") is already
implemented. The real questions are:

1. **Balance.** Is the grounding pass simply too small relative to CE? Would
   more grounding passes, or interleaving them, hold the constraint?
2. **Order.** Grounding runs first, then is overwritten. Would running it
   *after* CE (a fine-tune) persist the constraint?
3. **Conditioning strength.** Is the conditioning signal actually reaching the
   decoder at generation time? `_build_conditioned_bos` blends it 0.3 into the
   sensorimotor tail — is that enough to steer 200 passes of CE?
4. **Objective.** CE on general prose actively teaches the decoder to say
   things that are *not* about the subject. A grounding-aware objective would
   need to penalise unsupported assertions, not just reward fluency.

Question 3 is the cheapest to test and the most likely culprit: if the
conditioning signal is too weak at generation time, no amount of training
balance will help.

## Measurement first

`scripts/eval_decoder_groundedness.py` (being built) will tell us which of
these it is: if the decoder is on-topic but asserts false relations, it is an
objective problem; if it is off-topic entirely, it is a conditioning problem.

## Related

- `docs/SESSION_GOAL_grounded_decoder.md` — the session goal
- `docs/SWITCH_IS_OFF.md` — the measured failure
- `scripts/train.py:187` — `train_decoder_grounded`
- `scripts/train.py:86` — `load_corpus`, where grounding runs before CE
