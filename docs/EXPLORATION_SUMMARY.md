# RAVANA Cognitive Framework Exploration

## What is RAVANA?

A **pressure-driven cognitive ML framework** where learning emerges from self-organization, not gradient descent.

### Key Innovations:
- **No backpropagation** - uses Hebbian/Anti-Hebbian plasticity instead
- **Concept Graph** - typed edges (semantic, causal, temporal, analogical, contextual, inferred)
- **Sleep Consolidation** - SWS + REM cycles that reorganize knowledge
- **VAD Emotion Engine** - 3D affective state (Valence, Arousal, Dominance)
- **Governor** - constraint-based regulation (not SGD/Adam)

## Demo Results

### Concept Relationships (Semantic Similarity)
- `trust` → hope (0.69), hold (0.65), want (0.62), believe (0.62)
- `consciousness` → mind (0.67), imagination (0.61), life (0.61), moral (0.60)
- `gravity` → quantum (0.71), theory (0.63), earth (0.62)
- `love` → life (0.76), true (0.74), i (0.73), you (0.73)
- `learning` → knowledge (0.76), learn (0.74), mind (0.71)

### Cross-Domain Transfer (The Mind-Blowing Part!)
- **Before Sleep**: 12.5% Top-1 on held-out science facts, 0% on social facts
- **After Sleep**: 81.2% Top-1 on science, 80% Top-1 on social
- **Cross-domain**: Science verbs + Social subjects = 81% Top-1

This means RAVANA can learn "heat causes expansion" and generalize to "anger causes conflict"
by recognizing the shared "causes" relation pattern!

### How It Works
```
input "heat causes expansion"
    → decompose: (subject="heat", relation="causes", object="expansion")
    → find subject concept node in graph
    → spread activation from subject, filtered by relation type
    → score activated nodes against all token embeddings
    → return predictions
```

## What Makes This Special?

| Traditional ML | RAVANA |
|---------------|--------|
| Loss functions → gradients | Free energy → self-organization |
| Backpropagation | Hebbian co-activation |
| Weight decay | Identity regularization |
| Batch normalization | Sleep consolidation |
| Dense weight matrices | Concept graph (sparse) |
| GPU required | CPU-native! |

## The Governor Philosophy

> "No state modification without governor passage. No exceptions."

The governor acts like a central control system that regulates all cognitive changes through:
1. Hard constraints (absolute ceilings/floors)
2. Predictive dampening (slow before wall)
3. Boundary pressure (sigmoid soft resistance)
4. Center-seeking homeostasis

This replaces the need for learning rates, momentum, and weight decay!

## Try It Yourself

```python
from scripts.ravana_chat import CognitiveChatEngine

engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=False)
response = engine.process_turn('what is consciousness')
print(response)
# Output: "Consciousness are deeply connected with a debate. It brings about create. It is part of life."
```

## Files to Explore

- `ravana-v2/src/ravana_grace/core/governor.py` - The central regulator
- `ravana-v2/src/ravana_grace/core/emotion.py` - VAD emotion engine
- `ravana_ml/src/ravana_ml/graph.py` - Concept graph with typed edges
- `ravana_ml/src/ravana_ml/nn/rlm_v2.py` - Triple-decomposition neural net
- `scripts/ravana_chat.py` - Full chat interface
- `experiments/experiment_cross_domain.py` - Transfer learning demo