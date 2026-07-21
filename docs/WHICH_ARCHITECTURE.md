# Two Architectures, One Codebase

RAVANA contains **two complementary cognitive architectures** under the same repository.
This document explains the difference and when to use each.

## Quick comparison

| | `ravana/` (Modular Chat) | `ravana-v2/` (GRACE Core) |
|---|---|---|
| **Role** | Decoder-first chat engine with web learning | Pressure-driven cognitive governor framework |
| **Entry point** | `scripts/ravana_chat.py` or `from ravana.chat.engine import CognitiveChatEngine` | `from ravana_grace.core.governor import Governor` |
| **Learning** | Continuous web learning, curiosity-driven | Dissonance-minimization through pressure accumulation |
| **Output** | Natural language (decoder → realizer) | Cognitive state vectors + regulation decisions |
| **User-facing** | Interactive chatbot, ready to run | Research framework, needs a harness |
| **Dependency** | Uses GRACE modules for emotion/identity/sleep but not the full Governor | Standalone — no chat dependency |
| **Where** | `ravana/src/ravana/` | `ravana-v2/src/ravana_grace/` |

## How they relate

```
User text
    │
    ▼
┌──────────────────────────────────────────────────┐
│  CognitiveChatEngine  (ravana/chat/engine.py)    │
│  •  Uses: VADEmotionEngine, IdentityEngine,      │
│     DualProcessController, GlobalWorkspace,       │
│     SleepConsolidation — all from ravana_grace   │
│  •  Adds: graph walking, decoder, web learning    │
└──────────────────────┬───────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────────┐     ┌─────────────────────────┐
│  GRACE Modules   │     │  Governor / MetaCog /   │
│  (individual)    │     │  BeliefReasoner / ...   │
│  ravana_grace    │     │  (20-phase orchestrated, A–P)│
└──────────────────┘     └─────────────────────────┘
```

The chat engine **imports individual GRACE modules** (emotion, identity, sleep, etc.) but does **not** use the full Governor pipeline. The GRACE core is designed as a standalone research framework where all 20 phases (A–P) operate together.

## When to use which

### Use `ravana/` (the chat engine) when you want to:

- **Run an interactive chatbot** — `python scripts/ravana_chat.py`
- **Have the system learn from the web** — web search + background learning
- **Generate natural language from a concept graph** — decoder + surface realizer
- **Experiment with brain-repair prepasses** — coherence gates, junk detection, empathy
- **Work with the RLMv2 language model** — triple decomposition, verb offset

### Use `ravana-v2/` (GRACE core) when you want to:

- **Study the cognitive governor** — pressure-based regulation, identity, resolution
- **Run k0–k3 agent experiments** — survival environments, classroom pilots
- **Develop new cognitive phases** — the 20-phase architecture (A–P) is extensible
- **Test dissonance-minimization learning** — pressure accumulation, sleep consolidation
- **Build your own agent loop** — `CognitiveFramework` class with perceive/predict/learn/infer

### Use both when you want to:

- **Replace individual GRACE modules in the chat engine** — e.g. swap sleep strategies
- **Add chat as a sensory modality to the GRACE agent** — feed user text through the governor
- **Benchmark the full stack** — `experiments/experiment_cross_domain.py` uses both

## Package boundaries (important)

| Package | Path | Import as | Installed via |
|---------|------|-----------|---------------|
| `ravana` (chat) | `ravana/src/ravana/` | `from ravana.chat import ...` | `pip install -e .` |
| `ravana_ml` (ML) | `ravana_ml/src/ravana_ml/` | `from ravana_ml import ...` | `pip install -e .` |
| `ravana_grace` (GRACE) | `ravana-v2/src/ravana_grace/` | `from ravana_grace.core import ...` | `pip install -e .` |

All three are imported together by `scripts/ravana_chat.py` via `sys.path` insertion.

## Example: importing from each

```python
# ── Chat engine (ravana/) ──
from ravana.chat.engine import CognitiveChatEngine
engine = CognitiveChatEngine(dim=64)
response = engine.process_turn("what is trust")

# ── ML substrate (ravana_ml/) ──
from ravana_ml.graph import ConceptGraph
g = ConceptGraph(dim=64)
from ravana_ml.nn.neural_decoder import NeuralDecoder

# ── GRACE modules (ravana_grace, used inside engine) ──
from ravana_grace.core.emotion import VADEmotionEngine, VADConfig
from ravana_grace.core.governor import Governor, GovernorConfig
```

## File location reference

All `ravana/` modules: `ravana/src/ravana/chat/`, `ravana/src/ravana/core/`, etc.
All `ravana_ml/` modules: `ravana_ml/src/ravana_ml/`
All `ravana_grace/` modules: `ravana-v2/src/ravana_grace/core/`
