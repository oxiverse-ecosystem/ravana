# RLC — Recursive Learning Model

A CPU-native ML framework where learning emerges from pressure-driven self-organization, not gradient descent.

## Quick Start

```python
import rlc as torch

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])

# Build a model
model = torch.nn.Linear(3, 2)

# Forward pass
y = model(x)

# Learn (no backprop!)
model.accumulate_pressure(y - target)
model.sleep_cycle()  # instead of optimizer.step()
```

## Cognitive System

```python
from rlc.cognitive import CognitiveFramework

framework = CognitiveFramework()
state = framework.initialize()

for episode, data in enumerate(stream):
    concepts = framework.perceive(state, input_vec)
    predictions = framework.predict(state, concepts)
    state = framework.learn(state, predictions, outcomes, episode)
    if episode % 100 == 0:
        state = framework.sleep(state)
```

## Architecture

- `rlc.tensor` — RawTensor, StateTensor with cognitive fields
- `rlc.nn` — Module, Linear, Embedding, RLM (language model)
- `rlc.graph` — ConceptGraph with Hebbian plasticity
- `rlc.cognitive` — Governor, Emotion, Sleep, Meaning, Global Workspace, and 20+ modules
- `rlc.cognitive.CognitiveFramework` — Top-level API

## Why RLC?

| PyTorch | RLC |
|---------|-----|
| `loss.backward()` | `model.accumulate_pressure(error)` |
| `optimizer.step()` | `model.sleep_cycle()` |
| Gradient descent | Governor regulation |
| Backpropagation | Hebbian co-activation |
| GPU required | CPU native |
