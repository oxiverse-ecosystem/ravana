# Developer Guide

> **Contributing to RAVANA** — code standards, testing, architecture decisions, and release process.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Standards](#code-standards)
3. [Architecture Principles](#architecture-principles)
4. [Testing](#testing)
5. [Adding New Modules](#adding-new-modules)
6. [Debugging](#debugging)
7. [Performance Profiling](#performance-profiling)
8. [Documentation](#documentation)
9. [Release Process](#release-process)

---

## Development Setup

### Environment

```bash
# Primary (Codeberg)
git clone https://codeberg.org/oxiverse/ravana.git
# Mirror (GitHub)
git clone https://github.com/oxiverse-ecosystem/ravana.git
cd ravana

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ravana/
pip install -e .  # All layers

# Development dependencies
pip install pytest pytest-cov black ruff mypy

# Optional: pre-commit hooks
pip install pre-commit
pre-commit install
```

### Project Structure

```
ravana/
├── ravana/                 # Unified package (pip install -e ravana/)
│   ├── __init__.py
│   ├── nn/
│   ├── cognitive/
│   ├── graph/              # Symlinks to ravana_ml/
│   ├── propagation/
│   ├── world/
│   └── lab/
├── ravana_ml/              # ML Framework (~5,200 lines)
│   ├── nn/
│   ├── graph.py
│   ├── tensor.py
│   ├── tokenizer.py
│   ├── currencies.py
│   ├── free_energy.py
│   ├── plasticity.py
│   ├── propagation.py
│   └── ...
├── ravana-v2/              # Cognitive Core (~19,500 lines)
│   ├── core/               # 27 GRACE modules
│   ├── experiments/
│   └── tests/
├── experiments/            # Cross-layer experiments
├── tests/                  # ML framework tests (CI, unit, integration)
├── scripts/                # Analysis tools
├── docs/                   # This documentation
└── data/                   # GloVe embeddings (gitignored)
```

### Key Entry Points

| Layer | Main Entry | Purpose |
|-------|------------|---------|
| `ravana_ml` | `ravana_ml/__init__.py` | `import ravana_ml as torch` |
| `ravana-v2` | `ravana-v2/core/__init__.py` | GRACE modules |
| `ravana` | `ravana/__init__.py` | `import ravana as torch` + CognitiveFramework |

---

## Code Standards

### Python Version

- **Minimum**: Python 3.10
- **Target**: Python 3.11+ (for performance)

### Style Guide

```bash
# Format
black ravana/ ravana_ml/ ravana-v2/ tests/ experiments/ scripts/

# Lint
ruff check ravana/ ravana_ml/ ravana-v2/ tests/ experiments/ scripts/

# Type check
mypy ravana/ ravana_ml/ ravana-v2/
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `free_energy.py` |
| Classes | `PascalCase` | `ConceptGraph` |
| Functions | `snake_case` | `spread_activation` |
| Constants | `UPPER_SNAKE_CASE` | `RELATION_TYPES` |
| Private | `_leading_underscore` | `_init_concepts` |
| Type hints | Required for public API | `def foo(x: np.ndarray) -> float:` |

### NumPy Style

```python
# Good
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Avoid
def cos_sim(a, b):
    return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
```

### Cognitive Metadata in Tensors

```python
# When creating StateTensor, always include cognitive metadata
st = StateTensor(
    data=arr,
    salience=1.0,        # Importance weight
    free_energy=0.0,     # Local prediction error
    stability=0.5,       # Resistance to change
    decay=0.0,           # Time decay rate
)
```

### No Backprop Rule

**Critical**: Never add `autograd`, `backward()`, or gradient-based optimizers.

```python
# WRONG - never do this
x.requires_grad = True
loss.backward()
optimizer.step()

# CORRECT - pressure-driven
model.accumulate_free_energy(error)
model.sleep_cycle()
```

---

## Architecture Principles

### 1. Local Learning

Every learning signal must be **local** to the concept/edge:

```python
# GOOD: edge.update(source, target, local_coactivation)
# BAD:  model.optimizer.step()  # global gradient
```

### 2. Graph-First

Knowledge lives in the **ConceptGraph**, not in weight matrices:

```python
# GOOD: graph.add_node(...) / graph.hebbian_update(...)
# BAD:  model.layer.weight.data += ...  # hidden in dense matrix
```

### 3. Sleep as First-Class

Consolidation is not optional — it's part of the learning loop:

```python
# Every training loop MUST include sleep
for episode in range(n):
    learn(...)
    if episode % sleep_interval == 0:
        sleep()  # Required
```

### 4. Regulation is Explicit

All state changes pass through **Governor**:

```python
# GOOD: output = governor.regulate(d, i, signals)
# BAD:  d += delta  # unregulated
```

### 5. Emotion as Compute

VAD (valence, arousal, dominance) modulates inference:

```python
# Modulation applied in inference path
if arousal > 0.7:
    exploration_bonus = 0.2
```

### 6. Inspectability > Performance

Prefer debuggable, inspectable code over micro-optimizations:

```python
# GOOD: explicit loops with clear variable names
for src, tgt in active_pairs:
    edge = graph.get_edge(src, tgt)
    edge.weight += lr * coactivation

# AVOID: dense matrix ops that hide semantics
# W += lr * (A.T @ B)  # What does this mean cognitively?
```

---

## Testing

### Test Structure

```
tests/                          # ML framework tests
├── unit/                       #   Module-level unit tests (fast)
│   ├── test_ravana_core.py
│   ├── test_rlm_v2_*.py
│   ├── test_rp_only.py
│   └── ...
├── integration/                #   Cross-module integration tests
│   ├── test_ravana.py
│   ├── test_dialogue_engine_integration.py
│   ├── test_dialogue_system.py
│   └── ...
├── ci/                         #   CI smoke tests
│   └── test_core.py
└── ...

ravana-v2/tests/               # Cognitive core tests
├── test_governor.py
├── test_sleep.py
├── test_human_memory.py
├── test_emotion.py
├── test_identity.py
└── ...
```

### Running Tests

```bash
# All tests
python -m pytest tests/ ravana-v2/tests/ -v

# Unit tests only (fast)
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# CI smoke tests
python -m pytest tests/ci/ -v

# Specific module
python -m pytest tests/unit/test_rlm_v2.py -v
python -m pytest ravana-v2/tests/test_governor.py -v

# With coverage
python -m pytest tests/ --cov=ravana_ml --cov-report=html

# Parallel
python -m pytest tests/ -n auto
```

### Writing Tests

```python
# tests/test_my_module.py
import numpy as np
import pytest
from ravana_ml.my_module import MyClass

class TestMyClass:
    def test_basic_functionality(self):
        obj = MyClass(param=0.5)
        result = obj.method(np.array([1., 2.]))
        assert result.shape == (2,)
        assert np.allclose(result, expected)
    
    def test_edge_case_empty_input(self):
        obj = MyClass()
        result = obj.method(np.array([]))
        assert result == expected_default
    
    @pytest.mark.parametrize("input,expected", [
        (np.array([1.]), 1.0),
        (np.array([1., 2.]), 1.5),
    ])
    def test_parametrized(self, input, expected):
        assert MyClass().method(input) == expected
```

### Cognitive Core Tests

```python
# ravana-v2/tests/test_governor.py
def test_governor_hard_constraints():
    gov = Governor(GovernorConfig(max_dissonance=0.5))
    signals = CognitiveSignals(dissonance_delta=1.0)  # Would exceed
    output = gov.regulate(0.4, 0.5, signals)
    # Should be clamped
    assert output.dissonance_delta <= 0.1  # 0.5 - 0.4 - 0.01
    assert output.capped

def test_governor_alignment_metrics():
    gov = Governor()
    for _ in range(100):
        gov.regulate(0.3, 0.7, CognitiveSignals(dissonance_delta=0.1))
    metrics = gov.get_clamp_metrics()
    assert metrics["alignment_score"] > 0.5
```

### Experiment Tests

```python
# experiments/test_experiment_name.py
def test_cross_domain_minimum_accuracy():
    """Integration test: cross-domain should not regress below baseline."""
    results = run_cross_domain_experiment()
    assert results["top10_accuracy"] > 0.5  # Minimum threshold
```

---

## Adding New Modules

### 1. ML Framework Module (`ravana_ml/`)

```python
# ravana_ml/new_module.py
"""Brief description of module purpose."""

import numpy as np
from typing import Optional, List, Tuple
from .tensor import StateTensor
from .graph import ConceptGraph

class NewClass:
    """Docstring: what it does, key parameters, returns."""
    
    def __init__(self, graph: ConceptGraph, param: float = 0.1):
        self.graph = graph
        self.param = param
        # Initialize any state
    
    def method(self, input: np.ndarray) -> StateTensor:
        """Process input, return StateTensor with cognitive metadata."""
        # Implementation
        return StateTensor(data=result, salience=1.0)

# Export in ravana_ml/__init__.py
from .new_module import NewClass
__all__ = [..., 'NewClass']
```

### 2. Cognitive Core Module (`ravana-v2/core/`)

```python
# ravana-v2/core/new_module.py
"""Phase X: Description."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class NewConfig:
    """Immutable configuration."""
    param: float = 0.1
    threshold: float = 0.5

@dataclass
class NewState:
    """Serializable state snapshot."""
    value: float = 0.0

class NewModule:
    """Phase X module description."""
    
    def __init__(self, config: Optional[NewConfig] = None):
        self.config = config or NewConfig()
        self.history: List[NewState] = []
    
    def step(self, input: Any) -> NewState:
        """Process one cognitive cycle."""
        # Implementation
        state = NewState(value=result)
        self.history.append(state)
        return state
    
    def snapshot(self) -> Dict[str, Any]:
        return {"history": [s.value for s in self.history]}

# Export in ravana-v2/core/__init__.py
from .new_module import NewModule, NewConfig, NewState
__all__ = [..., 'NewModule', 'NewConfig', 'NewState']
```

### 3. Unified Package Export (`ravana/`)

```python
# ravana/__init__.py
from ravana_ml import NewClass  # If ML module
# OR
from core import NewModule      # If cognitive module

__all__ = [..., 'NewClass']  # or 'NewModule'
```

---

## Debugging

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `NaN` in training | LR too high, no clamping | Reduce LR, add clamping |
| Graph not growing | `gate_concept_creation=True` + high threshold | Lower threshold or disable gating |
| Sleep never triggers | `pressure_threshold` too high | Lower `SleepConfig.pressure_threshold` |
| Zero predictions | No edges formed | Check `structural.form_threshold` |
| Concept conflation | Structured init on large vocab | Use `vocab_size <= 32` check |
| Slow `find_similar` | No caching, large graph | Enable `_token_embed_norms` cache |

### Debug Helpers

```python
# In ravana_ml/graph.py
def debug_print(graph, n=10):
    print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
    print("Top active:")
    for nid, node in sorted(graph.nodes.items(), key=lambda x: -x[1].activation)[:n]:
        print(f"  {nid}: {node.label} act={node.activation:.3f}")
    print("Top edges:")
    for (s,t), e in sorted(graph.edges.items(), key=lambda x: -x[1].weight)[:n]:
        print(f"  {s}→{t} w={e.weight:.3f} {e.relation_type}")

# Call during training
if episode % 50 == 0:
    debug_print(model.graph)
```

### Governor Diagnostics

```python
# Always check clamp report when debugging instability
print(gov.get_clamp_report())

# Low alignment_score → upstream modules producing bad deltas
# High clamp_rate → constraints too tight
# Significant corrections → instability events
```

### Sleep Record Analysis

```python
for r in sleep.sleep_history[-10:]:
    if r.rollback_occurred:
        print(f"⚠️ Ep {r.episode}: ROLLBACK!")
    if r.post_coherence < r.pre_coherence - 0.05:
        print(f"⚠️ Ep {r.episode}: coherence drop {r.pre_coherence:.3f}→{r.post_coherence:.3f}")
```

---

## Performance Profiling

### Built-in Profiling

```python
import cProfile, pstats, io

profiler = cProfile.Profile()
profiler.enable()

# Run training
for ep in range(100):
    train_step()

profiler.disable()

# Print stats
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())
```

### Key Bottlenecks

| Function | Typical % | Optimization |
|----------|-----------|--------------|
| `graph.find_similar` | 30-50% | Cache norms, use FAISS for >5k nodes |
| `graph.spread_activation` | 20-30% | Vectorize edge traversal |
| `np.linalg.norm` | 15-25% | Cache norms per forward pass |
| `sleep_cycle` | 10-20% | Reduce replay samples, parallelize |

### Optimization Checklist

- [ ] Use `WordTokenizer` (not char-level)
- [ ] Cache `token_embed_norms` per forward (RLMv2 does this)
- [ ] Reduce `n_concepts` and `concept_dim` for dev
- [ ] Increase `sleep_interval` during prototyping
- [ ] Disable dream sabotage for speed: `counterfactual_rate=0`
- [ ] Use `np.dot` not `@` in tight loops (sometimes faster)
- [ ] Pre-allocate arrays, avoid list append in loops

---

## Documentation

### Writing Docs

- All public APIs must have docstrings
- New modules → add to `docs/API_REFERENCE.md`
- New concepts → add to `docs/CONCEPTS.md`
- New tutorials → add to `docs/TUTORIALS.md`
- Architecture changes → update `docs/ARCHITECTURE.md`

### Docstring Format

```python
def function_name(param: Type, optional: int = 0) -> ReturnType:
    """Brief one-line summary.

    Longer description if needed. Explain parameters, returns,
    side effects, and any cognitive significance.

    Args:
        param: Description of param.
        optional: Description, defaults to 0.

    Returns:
        Description of return value.

    Raises:
        ValueError: If param is negative.
    """
```

### Building Docs Locally

```bash
# If using MkDocs (not yet configured)
pip install mkdocs mkdocs-material
mkdocs serve
```

---

## Release Process

### Versioning

- **MAJOR**: Breaking API changes, architecture overhaul
- **MINOR**: New modules, features, backwards compatible
- **PATCH**: Bug fixes, performance, documentation

Current: `0.1.0` (pre-1.0, unstable API)

### Release Checklist

```bash
# 1. All tests pass
python -m pytest tests/ ravana-v2/tests/ -v

# 2. No lint errors
ruff check ravana/ ravana_ml/ ravana-v2/

# 3. Update version
# ravana/__init__.py: __version__ = '0.2.0'
# ravana/pyproject.toml: version = "0.2.0"

# 4. Update CHANGELOG.md
# 5. Tag release
git tag -a v0.2.0 -m "Release 0.2.0: ..."
git push origin v0.2.0

# 6. Build distribution
pip install build
python -m build ravana/

# 7. Publish (when ready)
# twine upload dist/*
```

### Pre-Release Testing

```bash
# Test in clean environment
python -m venv test_env
source test_env/bin/activate
pip install dist/ravana-0.2.0-py3-none-any.whl

# Verify imports
python -c "import ravana as torch; from ravana.cognitive import CognitiveFramework; print('OK')"

# Run smoke tests
python -m pytest tests/test_ravana.py::test_basic -v
```

---

## Contributing Guidelines

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation
- [ ] Refactoring

## Testing
- [ ] Added/updated unit tests
- [ ] Ran full test suite
- [ ] Tested manually with: `python my_test.py`

## Cognitive Impact
- [ ] No change to learning dynamics
- [ ] Modified learning (explain)
- [ ] Added new cognitive module (explain)

## Checklist
- [ ] Code follows style guide (black, ruff)
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] Documentation updated (if applicable)
- [ ] No backprop/gradient code added
```

### Code Review Criteria

1. **No gradients/backprop** — pressure-driven only
2. **Local learning** — no global optimizer steps
3. **Graph-first** — knowledge in ConceptGraph
4. **Regulation** — state changes through Governor
5. **Inspectability** — debuggable, logged, measurable
6. **Tests** — coverage for new functionality

---

## Architecture Decision Records (ADRs)

Major decisions are recorded in `docs/adr/`:

```
docs/adr/
├── 001-pressure-driven-learning.md
├── 002-conceptgraph-typed-edges.md
├── 003-sleep-consolidation.md
├── 004-grace-governor.md
├── 005-triple-decomposition-rlmv2.md
└── 006-verb-stem-offset.md
```

### Creating an ADR

```markdown
# ADR 007: Title

## Status
Proposed / Accepted / Superseded

## Context
What problem are we solving?

## Decision
What did we decide?

## Consequences
- Positive: ...
- Negative: ...
- Neutral: ...
```

---

## See Also

- [Architecture](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Advanced Topics](ADVANCED_TOPICS.md)
- [Testing](EXPERIMENTS.md#diagnostic-tests)