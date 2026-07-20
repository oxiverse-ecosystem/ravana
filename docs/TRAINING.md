# Training

All training lives in **`scripts/train.py`** (one file, four modes). The old
`iterative_train.py`, `train_decoder_phase2.py`, and `_linggen_train_big.py`
scripts were merged into it or removed.

## Prerequisites

- `data/corpora/teen_seeds.txt` — the seed English corpus (~296 sentences). It
  is gitignored; if missing, regenerate with
  `python scripts/gather_teen_seeds.py`.
- `data/attribute_encoder.npz` and the GloVe→64-D cache (downloaded/built once).
- `data/lancaster_encoder.npz` — wide-coverage sensorimotor probe (optional;
  build with `python scripts/train_lancaster_probe.py`).

## Modes

```
python scripts/train.py --mode <phase2|full|test|linggen> [options]
```

| Mode | What it does | Time |
|------|--------------|------|
| `phase2` | Heavy decoder training on `teen_seeds.txt` + curiosity-driven web learning + consolidation. Saves weights. | ~1 h |
| `full` | Same single-phase pipeline as `phase2` (the multi-cycle approach was removed because it caused catastrophic forgetting). | ~3–5 h |
| `test` | Quick diagnostic: trains on 50 sentences, generates a few responses, saves. | seconds |
| `linggen` | Offline LingGen P6 promotion: harvests grounded corpus from local Gutenberg books, trains the sensorimotor decoder (`W_sm` 65→75), and promotes `use_linggen` **only if** free-form generation clears the coherence floor. Writes `data/linggen_train_report.txt`. | minutes |

Common flags: `--dim` (graph dim, default 64), `--seed`, `--reset` (delete saved
weights), `--no-web` (skip web learning), `--web-topics N`.

## What "training" means here

There is **no offline pretraining step required to chat**. The decoder trains
online:

1. `load_corpus()` expands the decoder vocab from `teen_seeds.txt`, then runs
   LingGen grounded training if a harvest exists (no-op otherwise).
2. `train_seed_corpus()` does sampled-softmax passes with early stopping on
   cross-entropy (honest CE — the self-conditioning cheat was removed).
3. Web learning (`engine.learn_from_web`) harvests real sentences and writes
   typed edges + decoder training examples.
4. Consolidation pass + `engine.save()`.

`nd.sleep_cycle()` runs consolidation between passes — this is the free-energy
"instead of `optimizer.step()`" loop from `ravana_ml`.

## LingGen promotion gate (fail-closed)

`train_decoder_grounded()` fits `LingGenConditioner` (`W_sm`) on
(binder-65, embed-75) pairs, trains the decoder on grounded descriptions with
sensorimotor conditioning, then measures **free-form generation quality** =
in-vocab ratio × distinct-1 × on-topic cosine. If `quality >= 0.5` the engine
sets `use_linggen = True`; otherwise it stays `False` and generation falls back
to the template/realize path — it never emits ungrounded gibberish.

## Continuous (autonomous) learning

`python scripts/ravana_learn.py` runs the same `CognitiveChatEngine` with no
chat: the curiosity drive (`engine._get_curiosity_scores`) selects what to learn
from prediction error, information gaps, contradiction pairs, and novelty.
Flags: `--cycles N` (0 = infinite), `--delay SECONDS`, `--no-curiosity`.
Ctrl+C saves weights.
