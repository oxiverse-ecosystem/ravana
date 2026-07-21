# Tutorial 02: Neural Decoder Training

**Part of a 7-tutorial progression. Run Tutorial 01 first.**

## What you'll learn

- Load a previously-saved `CognitiveChatEngine`
- Access the `NeuralDecoder` — the engine's language generation model
- Train the decoder on a few sentences
- Interpret cross-entropy, top-1, and top-5 accuracy
- Understand **how the decoder is different from an LLM**

---

## Prerequisites

```bash
python tutorials/01-chat-basics/run.py   # create the saved state first
```

---

## Run it

```bash
python tutorials/02-decoder-training/run.py
```

---

## Deep dive: NeuralDecoder architecture

**File:** `ravana_ml/src/ravana_ml/nn/neural_decoder.py`

### What it is

The `NeuralDecoder` is a **64-D GRU** (Gated Recurrent Unit) — a recurrent
neural network that predicts the next word in a sequence, conditioned on a
**concept embedding** from the graph walk.

### Architecture diagram

```
     ┌─────────────────────────────────────────────┐
     │  Graph walk output (concept activations)     │
     │  → averaged into 64-D conditioning vector    │
     └────────────────┬────────────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────────────┐
     │  DecoderConfig                                │
     │  • embed_dim = 64                             │
     │  • hidden_dim = 64                            │
     │  • vocab_size = ~100-500                      │
     │  • num_layers = 1                             │
     │  • dropout = 0.0 (no dropout at this scale)   │
     └────────────────┬────────────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────────────┐
     │  GRU Cell (64 → 64)                          │
     │  Takes: previous hidden + word embedding     │
     │  Produces: new hidden state                  │
     └────────────────┬────────────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────────────┐
     │  Output projection (64 → vocab_size)         │
     │  → Linear layer without bias                │
     └────────────────┬────────────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────────────┐
     │  Softmax → word probabilities                │
     │  → Sample or argmax for generation           │
     └─────────────────────────────────────────────┘
```

### How it's different from an LLM

| Property | NeuralDecoder | GPT-style LLM |
|----------|--------------|---------------|
| **Parameters** | ~50K (64×64 GRU + 64×vocab projection) | 7B–175B |
| **Vocabulary** | ~100-500 words (from seed corpus) | ~50K tokens (BPE) |
| **Conditioning** | Concept graph embedding | Full context window |
| **Training** | Online (single-pass, no offline corpus) | Offline pre-training on trillions of tokens |
| **Optimizer** | Sleep consolidation (no gradient descent) | AdamW |
| **Generation** | Short phrases (5-15 words) | Arbitrary length |
| **Hardware** | CPU, single thread | GPU cluster |

### The module framework (not PyTorch)

**File:** `ravana_ml/src/ravana_ml/nn/module.py`

The decoder uses RAVANA's **native tensor framework** (`ravana_ml.tensor`),
not PyTorch. The framework provides:

- `Module` — base class with `.forward()`, `.parameters()`, `.save()`, `.load()`
- `Linear` — dense layer (numpy-backed, no autograd)
- `GRUCell` — GRU recurrence
- `Embedding` — lookup table
- `LayerNorm`, `Dropout` — standard regularizers

The tensor framework is designed to make the API familiar (it looks like
`torch.nn`) but operates on plain numpy arrays. There is no autograd graph,
no CUDA, and no GPU requirement.

---

## Deep dive: how training works

### The training signal

The decoder uses **cross-entropy loss** — standard for language models:

```
CE = -Σ P_true(word) * log(P_pred(word))
```

But there's a critical difference from standard ML: **there is no optimizer.**

In standard ML:
```
loss = CE(pred, target)
loss.backward()        ← computes gradients via autograd
optimizer.step()       ← updates weights via SGD/Adam
```

In RAVANA:
```
loss = CE(pred, target)
weight_update = hebbian_coactivation + free_energy_minimization
sleep_cycle()          ← consolidates changes
```

The `sleep_cycle()` method replaces `optimizer.step()`. It doesn't compute
gradients — it reorganizes the decoder's internal representations based on
prediction error patterns detected during training.

### Sampled softmax

The decoder uses **sampled softmax** during training — instead of computing
softmax over the full vocabulary (which is expensive), it samples a subset
of negative words for each positive word. This makes training faster without
significant quality loss at small vocabulary sizes.

### Early stopping

**File:** `scripts/train.py` → `train_seed_corpus()`

The training loop monitors cross-entropy and stops when:
1. CE stops decreasing (plateau detection)
2. A minimum number of passes is completed
3. The decoder has seen enough examples to generalize

### Plasticity

The decoder has a `reset_plasticity(stability=0.5)` method. This controls
how much the decoder is willing to change during training:

- **stability=0.0** → fully plastic (forgets old knowledge quickly)
- **stability=1.0** → fully rigid (resists all change)
- **stability=0.5** → balanced (default)

Resetting plasticity clears accumulated "momentum" from previous training,
giving the decoder a clean slate for new examples.

---

## Interpreting the metrics

After training, you see:

```
trained_on_examples=21119 avg_ce=3.287 top1=0.351 top5=0.568
```

### Cross-entropy (CE)

**What it is:** The average negative log-probability assigned to the correct word.

- **CE = 0.0** → perfect prediction (always assigns probability 1.0 to the right word)
- **CE = 3.0** → typical for a small decoder (equivalent to ~1/20 chance of being right)
- **CE = 6.0+** → random chance (decoder hasn't learned anything yet)

Lower is always better. CE tends to increase (worsen) as vocabulary grows because
there are more words to choose from.

### Top-1 accuracy

**What it is:** How often the decoder's highest-probability word is the correct one.

- **0.35** → 35% of the time, the #1 prediction is correct
- Random baseline: 1/vocab_size ≈ 0.002 (for 500 words)

### Top-5 accuracy

**What it is:** How often the correct word is in the decoder's top-5 predictions.

- **0.57** → 57% of the time, the correct word is among the top-5
- This is always higher than top-1 (easier to be in top-5 than #1)

### Why these numbers matter

The decoder is the **final word-selection mechanism** for any generated response.
Higher accuracy means:
- Less word-salad in responses
- More relevant vocabulary choices
- Better grounding in the concept graph

---

## What the code does step by step

```python
# 1. Load engine — uses the state saved by Tutorial 01
engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
```

The engine constructor automatically calls `_load()` which restores the graph,
decoder weights, GRACE module states, and vocabulary from the pickle file.

```python
# 2. Access the neural decoder
nd = engine.neural_decoder
```

The decoder is stored as `engine.neural_decoder` — a `NeuralDecoder` instance.

```python
# 3. Train on sentences
nd.train_on_sentence(
    text.split(),              # tokenized sentence: ["trust", "is", "the", "basis", ...]
    engine._decoder_word_to_embed,  # word → embedding mapping
    engine._decoder_word_to_idx,     # word → index mapping
)
```

`train_on_sentence()` does:
1. For each word position, predict the next word using the GRU
2. Compute cross-entropy between prediction and actual next word
3. Apply Hebbian-like weight update (no backprop!)
4. Update running statistics (avg CE, top-1, top-5)

```python
# 4. Read metrics
getattr(nd, "_total_training_examples")    # how many words trained on
getattr(nd, "_avg_cross_entropy")       # average CE (lower = better)
getattr(nd, "_avg_top1_acc")           # top-1 accuracy (higher = better)
getattr(nd, "_avg_top5_acc")           # top-5 accuracy (higher = better)
```

These are **running averages** — they reflect the decoder's performance over
all training examples seen so far (including pre-load state).

---

## Expected output (annotated)

```
Loading engine from data/ravana_weights.pkl ...
  [GloVe] Loaded 400000 projected vectors from cache (100D -> 64D)
  └── GloVe is shared with the graph — no separate decoder embedding

  Initial: 21079 examples, CE=3.332, top1=0.296
  └── The decoder has already seen 21079 words from previous runs

  trained_on_examples=21099 avg_ce=3.287 top1=0.351 top5=0.568
  └── After 20 new words, CE dropped from 3.33 to 3.29 (-1.4%)
  └── Top-1 improved from 29.6% to 35.1% (+5.5%)

  [OK] State saved - ready for Tutorial 03
```

After just 3 sentences (20 words), you typically see:
- **CE decrease** of 1-5%
- **Top-1 increase** of 2-8 percentage points
- **Top-5 increase** of 3-10 percentage points

---

## Key source files reference

| Component | File (relative to repo root) |
|-----------|------------------------------|
| NeuralDecoder class | `ravana_ml/src/ravana_ml/nn/neural_decoder.py` (line 67) |
| GRU cell | `ravana_ml/src/ravana_ml/nn/module.py` (line 393, `GRUCell`) |
| Module base class | `ravana_ml/src/ravana_ml/nn/module.py` (line 7) |
| Tensor framework | `ravana_ml/src/ravana_ml/tensor.py` |
| Training loop | `scripts/train.py` → `train_seed_corpus()` |
| Sleep consolidation | `ravana_ml/src/ravana_ml/nn/rlm_v2_sleep.py` |

---

## Design philosophy notes

1. **Sleep replaces gradient descent.** The decoder doesn't use backpropagation.
   Learning emerges from the sleep cycle's analysis-compression-integration stages.
2. **Online learning.** The decoder learns from every user interaction. There's
   no separation between "training" and "inference" phases.
3. **Fail-closed generation.** If the decoder's confidence is too low, the
   engine falls back to template responses rather than emitting gibberish.
4. **Vocabulary is shared with the graph.** The decoder doesn't have its own
   embedding table — it uses the same GloVe-projected vectors as the concept graph.

---

## Next tutorial

[**Tutorial 03: Concept Graph**](../03-graph/) — build and inspect a concept graph
manually. Understand the storage layer that the decoder reads from.
