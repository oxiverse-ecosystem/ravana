# RAVANA Model Checkpoints

Two conversational model checkpoints are provided:

## 1. Standard Conversational Model (GRACE 27-phase)
**File**: `standard_conversational_model.pkl`

- **Architecture**: GRACE 27-phase cognitive architecture (Governor, Identity, Sleep, Emotion, World Model, Meaning, Human Memory, Dialogue)
- **Training**: 10,000 steps with lifelong learning
- **Graph**: 384 concepts, 20,416 edges
- **Dimensions**: 32D embeddings
- **Vocab**: 256 tokens (byte-level)
- **Sleep cycles**: 836 completed
- **Best for**: Long-term conversation, memory, learning, personality consistency

### Usage
```python
import pickle
from ravana_grace.core.governor import Governor

with open('standard_conversational_model.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# Restore governor state
governor = Governor(config=checkpoint['config'])
governor.load_state(checkpoint['state_dict'])
governor.graph = checkpoint['graph']
governor.binding_map = checkpoint['binding_map']

# Chat
response = governor.process_turn("hello, how are you?")
print(response)
```

## 2. RLMv2 Conversational Model (Modular Triple-Based)
**File**: `rlmv2_conversational_model.pkl`

- **Architecture**: RLMv2 (RLMv2 = GraphMixin + EncoderMixin + RPMixin + VerbMixin + SleepMixin)
- **Training**: 3 epochs on friendly conversation patterns
- **Graph**: 31 concepts, 36 edges
- **Dimensions**: 32D embeddings, 64D hidden, 32D latent
- **Vocab**: 62 words (conversation-focused)
- **Features**: Verb-offset arithmetic, entity adapters, spreading activation
- **Best for**: Quick inference, structured reasoning, cross-domain transfer

### Usage
```python
import pickle
import numpy as np
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer

with open('rlmv2_conversational_model.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# Restore tokenizer
tokenizer = WordTokenizer()
tokenizer.word_to_id = checkpoint['tokenizer_vocab']
tokenizer.id_to_word = {v: k for k, v in tokenizer.word_to_id.items()}
tokenizer._next_id = max(tokenizer.word_to_id.values()) + 1

# Restore model
config = checkpoint['config']
model = RLMv2(
    vocab_size=config['vocab_size'],
    embed_dim=config['embed_dim'],
    concept_dim=config['concept_dim'],
    n_concepts=config['n_concepts'],
    latent_dim=config['latent_dim'],
    hidden_dim=config['hidden_dim'],
)
model.load_state_dict(checkpoint['state_dict'])
model.graph = checkpoint['graph']
model.binding_map = checkpoint['binding_map']
model._tokenizer_val = tokenizer

# Inference
ctx = np.array([tokenizer.encode("hello")[:-1]], dtype=np.int64)
logits = model.forward(ctx)
pred_id = int(np.argmax(logits.data))
print(tokenizer.decode([pred_id]))
```

## Package Installation

```bash
pip install ravana-ml ravana-grace ravana-chat
```

All three packages published to PyPI (v0.2.0):
- ravana-ml: ML framework (tensors, modules, ConceptGraph, RLMv1/RLMv2)
- ravana-grace: 27-phase cognitive architecture
- ravana-chat: Decoder-first chat with continuous web learning