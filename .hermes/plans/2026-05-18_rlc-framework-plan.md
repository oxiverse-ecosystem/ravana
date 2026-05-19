# RLC Framework Plan — From RAVANA to Production-Ready Framework

## Goal

Transform the RAVANA project into **RLC (Recursive Learning Model)** — a pip-installable, production-ready ML framework that is:
- A **PyTorch/TensorFlow alternative** (same API surface, different learning paradigm)
- An **LLM alternative** (the RLM module replaces transformers)
- **CPU-native** — no GPU required, no gradient descent, no backprop

## Current State Analysis

### What exists today:
- `ravana/` — Tensor system (RawTensor, StateTensor), nn modules (Linear, Embedding, LayerNorm, Dropout, RLM), ConceptGraph, propagation, pressure, plasticity, functional ops, world simulations, lab tools
- `ravana-v2/` — Cognitive core (Governor, Identity, Resolution, Emotion, Sleep, Dual-Process, Meaning, 20+ modules)
- No `setup.py` or `pyproject.toml` — not pip-installable
- No README at root level
- Two disconnected codebases (ravana/ and ravana-v2/)
- Test file exists but no test runner config

### What's strong:
- The tensor API is already PyTorch-compatible (`import ravana as torch` works)
- The RLM (Recursive Learning Model) is a real working language model — alternative to LLM
- The v2 cognitive core is comprehensive (28 modules)
- The learning paradigm (pressure-driven, no backprop) is genuinely novel

## Target Package Structure

```
rlc/
├── pyproject.toml              # pip install rlc
├── README.md                   # Comprehensive docs
├── LICENSE                     # MIT
├── rlc/                        # Main package
│   ├── __init__.py             # import rlc as torch
│   ├── tensor.py               # RawTensor, StateTensor, Parameter (existing)
│   ├── graph.py                # ConceptGraph (existing)
│   ├── propagation.py          # PropagationEngine (existing)
│   ├── pressure.py             # PressureAccumulator (existing)
│   ├── plasticity.py           # Hebbian, AntiHebbian, Structural (existing)
│   ├── optim.py                # NEW: PressureOptimizer, SleepOptimizer
│   ├── data.py                 # NEW: DataLoader, Dataset, BatchSampler
│   ├── serialize.py            # NEW: save/load models, checkpoints
│   ├── train.py                # NEW: Trainer, TrainingLoop, callbacks
│   ├── metrics.py              # NEW: CoherenceMetric, PressureMetric, etc.
│   ├── nn/
│   │   ├── __init__.py         # Re-export all modules
│   │   ├── module.py           # Module, Sequential, Linear, etc. (existing)
│   │   ├── functional.py       # relu, softmax, etc. (existing)
│   │   ├── rlm.py              # RLM - the language model (existing)
│   │   ├── attention.py        # NEW: ConceptAttention (graph-based attention)
│   │   ├── recurrent.py        # NEW: PressureRNN, PressureLSTM
│   │   └── losses.py           # NEW: PressureLoss, CoherenceLoss
│   ├── cognitive/
│   │   ├── __init__.py         # Re-export v2 cognitive modules
│   │   ├── governor.py         # (from ravana-v2/core/)
│   │   ├── identity.py
│   │   ├── resolution.py
│   │   ├── emotion.py
│   │   ├── sleep.py
│   │   ├── dual_process.py
│   │   ├── meaning.py
│   │   ├── empathy.py
│   │   ├── meta_cognition.py
│   │   ├── belief_reasoner.py
│   │   ├── active_epistemology.py
│   │   ├── hypothesis_generation.py
│   │   ├── occam_layer.py
│   │   ├── social_epistemology.py
│   │   ├── surgical_probes.py
│   │   ├── reality_friction.py
│   │   ├── meta2_cognition.py
│   │   ├── meta2_integration.py
│   │   ├── strategy.py
│   │   ├── strategy_learning.py
│   │   ├── intent.py
│   │   ├── planning.py
│   │   ├── environment.py
│   │   ├── predictive_world.py
│   │   ├── adaptation.py
│   │   ├── state.py
│   │   └── memory.py
│   ├── world/
│   │   ├── __init__.py         # (existing)
│   │   └── environments.py     # CausalSequenceWorld, etc.
│   └── lab/
│       ├── __init__.py         # (existing)
│       └── experiments.py      # ConceptLab, measurements
├── examples/
│   ├── quickstart.py           # 10-line getting started
│   ├── train_language_model.py # Train RLM on text
│   ├── cognitive_agent.py      # Full cognitive agent demo
│   └── benchmark_vs_pytorch.py # Compare paradigms
├── tests/
│   ├── test_tensor.py
│   ├── test_nn.py
│   ├── test_graph.py
│   ├── test_cognitive.py
│   └── test_integration.py
└── docs/
    ├── PHILOSOPHY.md           # Why pressure > gradients
    ├── API.md                  # Full API reference
    └── MIGRATION.md            # PyTorch → RLC migration guide
```

## Step-by-Step Implementation Plan

### Phase 1: Package Scaffolding (Foundation)
**Files to create/modify:**
1. `rlc/pyproject.toml` — Package metadata, dependencies (numpy only), entry points
2. `rlc/LICENSE` — MIT license
3. Rename `ravana/` → `rlc/` directory structure
4. Move `ravana-v2/core/` modules into `rlc/cognitive/`
5. Update all internal imports from `ravana` to `rlc`
6. Update `__init__.py` files for clean public API

### Phase 2: Missing Framework Pieces
**New modules to build:**
1. `rlc/optim.py` — PressureOptimizer (replaces `optimizer.step()`)
   - `PressureOptimizer(model, lr=0.1)` — wraps accumulate_pressure + sleep_cycle
   - `step(error)` — accumulates pressure, triggers sleep if threshold reached
   - `zero_pressure()` — resets pressure counters
2. `rlc/data.py` — Data loading utilities
   - `Dataset` base class
   - `DataLoader` with batching
   - `TextDataset` for language model training
3. `rlc/serialize.py` — Model persistence
   - `save_model(model, path)` — JSON-based state dict
   - `load_model(path)` → model
   - `Checkpoint` class for training resumption
4. `rlc/train.py` — Training infrastructure
   - `Trainer` class with callback system
   - `TrainingLoop` with metrics tracking
   - Progress reporting
5. `rlc/metrics.py` — Built-in metrics
   - `CoherenceMetric`, `PressureMetric`, `PredictionAccuracy`
6. `rlc/nn/attention.py` — Graph-based attention mechanism
7. `rlc/nn/recurrent.py` — PressureRNN, PressureLSTM
8. `rlc/nn/losses.py` — PressureLoss, CoherenceLoss

### Phase 3: API Polish (PyTorch Compatibility)
- Ensure `import rlc as torch` works seamlessly
- Add `rlc.device`, `rlc.no_grad()`, `rlc.save()`, `rlc.load()`
- Add `rlc.nn.Module` base class with `parameters()`, `state_dict()`, `load_state_dict()`
- Add `rlc.nn.functional` namespace
- Add type hints throughout
- Add docstrings to all public APIs

### Phase 4: Examples & Documentation
1. `README.md` — What is RLC, why it exists, quickstart, API overview
2. `examples/quickstart.py` — 10-line demo
3. `examples/train_language_model.py` — Full RLM training
4. `examples/cognitive_agent.py` — Cognitive agent with emotion, sleep, meaning
5. `docs/PHILOSOPHY.md` — Theoretical foundations
6. `docs/API.md` — Full API reference
7. `docs/MIGRATION.md` — PyTorch → RLC migration guide

### Phase 5: Testing & CI
1. Move existing tests into `tests/`
2. Add `pytest.ini` or `pyproject.toml [tool.pytest]`
3. Add test coverage for new modules
4. Add CI config (GitHub Actions)

## Key Design Decisions

### 1. Naming: RLC = Recursive Learning Model
- Package name: `rlc`
- Import: `import rlc` or `import rlc as torch`
- The "recursive" part refers to the pressure→sleep→consolidate cycle

### 2. Learning Paradigm Mapping
| PyTorch Concept | RLC Equivalent |
|----------------|----------------|
| `loss.backward()` | `model.accumulate_pressure(error)` |
| `optimizer.step()` | `model.sleep_cycle()` |
| `nn.Module` | `rlc.nn.Module` |
| `torch.tensor()` | `rlc.tensor()` |
| `nn.Linear` | `rlc.nn.Linear` |
| Gradient descent | Pressure-driven self-organization |
| Loss function | Cognitive dissonance |
| Optimizer (SGD/Adam) | Governor + Sleep Cycle |
| Backpropagation | Hebbian pressure propagation |

### 3. Dependencies
- **Only numpy** as hard dependency
- No torch, no tensorflow, no jax
- Optional: matplotlib for visualization

### 4. Cognitive Core Integration
The v2 cognitive modules (Governor, Emotion, Sleep, etc.) become `rlc.cognitive.*`:
- Available but not required for basic usage
- Users can opt-in to cognitive features
- The RLM uses them internally but exposes simple API

## Files Likely to Change

### Existing (rename/import updates):
- `ravana/__init__.py` → `rlc/__init__.py`
- `ravana/tensor.py` → `rlc/tensor.py`
- `ravana/graph.py` → `rlc/graph.py`
- `ravana/propagation.py` → `rlc/propagation.py`
- `ravana/pressure.py` → `rlc/pressure.py`
- `ravana/plasticity.py` → `rlc/plasticity.py`
- `ravana/nn/` → `rlc/nn/`
- `ravana/world/` → `rlc/world/`
- `ravana/lab/` → `rlc/lab/`
- `ravana-v2/core/*.py` → `rlc/cognitive/*.py`

### New files:
- `pyproject.toml`
- `LICENSE`
- `README.md`
- `rlc/optim.py`
- `rlc/data.py`
- `rlc/serialize.py`
- `rlc/train.py`
- `rlc/metrics.py`
- `rlc/nn/attention.py`
- `rlc/nn/recurrent.py`
- `rlc/nn/losses.py`
- `rlc/cognitive/__init__.py`
- `examples/*.py`
- `tests/*.py`
- `docs/*.md`

## Verification Steps

1. `pip install -e .` succeeds
2. `import rlc` works
3. `import rlc as torch; x = torch.tensor([1,2,3])` works
4. `rlc.nn.Linear(10, 5)` creates a module
5. `rlc.nn.RLM(...)` trains on a sequence
6. `rlc.cognitive.Governor()` instantiates
7. All existing tests pass under new structure
8. `python examples/quickstart.py` runs clean

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Breaking existing code during rename | Keep `ravana/` as symlink/alias for backward compat |
| Cognitive modules have complex deps | Make cognitive import lazy, not required for basic usage |
| Performance regression | Benchmark before/after, numpy ops unchanged |
| Too many changes at once | Phase by phase, each phase is independently useful |

## Open Questions

1. Should we keep `ravana/` as a backward-compat alias? (I'd say yes for now)
2. Version number — start at 0.1.0 or 1.0.0?
3. Should the cognitive modules be a separate sub-package (`rlc-cognitive`) or part of core?
4. PyPI package name availability for `rlc`?

---

*This plan transforms a research project into a distributable framework. Each phase delivers usable value.*
