# Getting Started with RAVANA

> **Quickstart guide** — from installation to your first cognitive model in 5 minutes.

---

## Installation

### Requirements

- **Python 3.10+**
- **NumPy ≥ 1.20** (only hard dependency)
- Optional: `tiktoken` for BPE tokenization

### Install

```bash
# Primary (Codeberg)
git clone https://codeberg.org/oxiverse/ravana.git
# Mirror (GitHub)
git clone https://github.com/oxiverse-ecosystem/ravana.git
cd ravana

# Install unified package (NumPy only)
pip install -e ravana/

# Optional: BPE tokenizer support
pip install tiktoken
```

### Verify Installation

```python
import ravana as torch
import numpy as np

# Test tensor operations
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor: {x}")
print(f"Shape: {x.shape}")

# Test module
model = torch.nn.Linear(3, 2)
y = model(x)
print(f"Linear output: {y}")

# Test RLMv2
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer

tokenizer = WordTokenizer()
tokenizer.encode("hello world")
model = RLM(vocab_size=tokenizer.vocab_size, embed_dim=64, concept_dim=64, n_concepts=50)
print("RLMv2 initialized ✓")

# Test CognitiveFramework
from ravana.cognitive import CognitiveFramework
fw = CognitiveFramework()
state = fw.initialize()
print("CognitiveFramework initialized ✓")
```

---

## Your First Models

### 1. PyTorch-Style ML (No Backprop!)

```python
import ravana as torch

# Create tensors (StateTensor with cognitive metadata)
x = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.5, 0.3])

# Build model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 16),
    torch.nn.LayerNorm(16),
    torch.nn.Linear(16, 2)
)

# Forward pass
y = model(x)
print(f"Prediction: {y.data}")

# Learn WITHOUT backpropagation!
model.accumulate_free_energy(y - target)
model.sleep_cycle()  # ← replaces optimizer.step()

print(f"After sleep: {model(x).data}")
```

**Key difference:** `accumulate_free_energy()` + `sleep_cycle()` replaces `loss.backward()` + `optimizer.step()`.

### 2. RLMv2 — Language Model (Triple Decomposition)

```python
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

# Setup tokenizer
tokenizer = WordTokenizer()
tokenizer.encode("heat causes expansion")  # builds vocab

# Create model
model = RLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    concept_dim=64,
    n_concepts=100,
    sleep_interval=50,        # sleep every 50 learn steps
    gate_concept_creation=True,
)

# Learn a fact: "heat causes expansion"
input_ids = np.array(tokenizer.encode("heat causes"), dtype=np.int64)
target_ids = np.array(tokenizer.encode("expansion"), dtype=np.int64)

model.learn(input_ids, target_ids)
print(f"After learning: top predictions = {model.get_top_predictions(input_ids, k=5)}")

# Consolidate with sleep
model.sleep_cycle()
print(f"After sleep: top predictions = {model.get_top_predictions(input_ids, k=5)}")
```

### 3. Cognitive Framework — Full Cognitive Cycle

```python
from ravana.cognitive import CognitiveFramework
import numpy as np

fw = CognitiveFramework()
state = fw.initialize()

# Training data: (input_vector, target_vector)
# In practice, these come from tokenizing text and embedding
training_data = [
    (np.random.randn(64).astype(np.float32), np.random.randn(64).astype(np.float32))
    for _ in range(100)
]

for episode, (input_vec, target_vec) in enumerate(training_data):
    # 1. PERCEIVE: input → active concepts
    concepts = fw.perceive(state, input_vec)
    
    # 2. PREDICT: spreading activation → predictions
    predictions = fw.predict(state, concepts)
    
    # 3. LEARN: pressure-based update
    state = fw.learn(state, predictions, target_vec, episode)
    
    # 4. SLEEP (periodic): consolidation
    if episode % 20 == 0:
        state = fw.sleep(state)
        print(f"Episode {episode}: dissonance={state.dissonance:.3f}, identity={state.identity:.3f}")

# Inference (no state change)
result = fw.infer(state, test_input_vec)
print(f"Predicted concepts: {result['predicted_concepts']}")
print(f"Coherence: {result['coherence']:.3f}")
```

---

## Core Concepts Quick Reference

| Concept | Traditional ML | RAVANA |
|---------|---------------|--------|
| **Learning signal** | Loss gradient | Free energy (pressure) |
| **Update rule** | `w ← w - η∇L` | Hebbian plasticity + sleep |
| **Memory** | Weights / RAG | ConceptGraph + episodic/semantic |
| **Consolidation** | Batch norm / EMA | SWS + REM sleep cycles |
| **Regulation** | Weight decay | GRACE Governor (hard constraints) |
| **Representation** | Dense vectors | Typed concept graph |

---

## Next Steps

| Goal | Read |
|------|------|
| Understand architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Deep dive ML framework | [ML_FRAMEWORK.md](ML_FRAMEWORK.md) |
| Deep dive cognitive core | [COGNITIVE_CORE.md](COGNITIVE_CORE.md) |
| Unified package API | [UNIFIED_PACKAGE.md](UNIFIED_PACKAGE.md) |
| Core theory | [CONCEPTS.md](CONCEPTS.md) |
| Run experiments | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Build custom models | [TUTORIALS.md](TUTORIALS.md) |
| Contribute | [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) |

---

## Common Issues

### `ModuleNotFoundError: No module named 'ravana_ml'`

```bash
pip install -e ravana/  # installs ravana_ml as part of unified package
# OR
pip install -e .        # from root, installs all three layers
```

### `ImportError: cannot import name 'RLM' from 'ravana.nn'`

RLMv2 is exported as `RLM`:
```python
from ravana.nn import RLM  # this is RLMv2 (triple decomposition)
# Old RLM v1: from ravana_ml.nn import RLM as RLMv1
```

### GloVe embeddings not found (RLMv2)

```bash
# Download GloVe 6B 100D
mkdir -p data/glove
wget -O data/glove/glove.6B.100d.txt http://nlp.stanford.edu/data/glove.6B.100d.txt
# Or 50D fallback
wget -O data/glove/glove.6B.50d.txt http://nlp.stanford.edu/data/glove.6B.50d.txt
```

### Slow performance

- Use `WordTokenizer` (5× faster than char-level for cognitive experiments)
- Reduce `n_concepts` and `concept_dim` for quick experiments
- Disable sleep during rapid prototyping: `sleep_interval=999999`

---

## Minimal Working Example

```python
# minimal_ravana.py
import numpy as np
import ravana as torch
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer

# 1. Tokenizer
tok = WordTokenizer()
tok.encode("fire burns wood fire creates heat")

# 2. Model
model = RLM(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32, n_concepts=50)

# 3. Learn facts
facts = [
    ("fire burns", "wood"),
    ("fire creates", "heat"),
    ("water extinguishes", "fire"),
]

for premise, conclusion in facts:
    inp = np.array(tok.encode(premise), dtype=np.int64)
    tgt = np.array(tok.encode(conclusion), dtype=np.int64)
    model.learn(inp, tgt)

# 4. Sleep to consolidate
model.sleep_cycle()

# 5. Test
for premise, _ in facts:
    inp = np.array(tok.encode(premise), dtype=np.int64)
    logits = model.forward(inp)
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"{premise} → {[tok.decode([t]) for t in top5]}")
```

Run:
```bash
python minimal_ravana.py
```

Expected output (approximate):
```
fire burns → ['wood', 'heat', 'fire', 'creates', 'water']
fire creates → ['heat', 'wood', 'fire', 'burns', 'extinguishes']
water extinguishes → ['fire', 'heat', 'wood', 'burns', 'creates']
```

---

## Environment Variables

```bash
# Optional: custom GloVe path
export RAVANA_GLOVE_PATH=/path/to/glove.6B.100d.txt

# Optional: cache directory
export RAVANA_CACHE_DIR=/path/to/cache
```

---

## Support

- **Documentation**: This `docs/` directory
- **Experiments**: `experiments/` folder
- **Tests**: `tests/` and `ravana-v2/tests/`
- **Issues**: GitHub Issues