# Running Experiments

> **Complete guide to running, configuring, and understanding RAVANA experiments.**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Experiment Categories](#experiment-categories)
3. [Cross-Domain Transfer](#cross-domain-transfer)
4. [Phase 4 Integrated](#phase-4-integrated)
5. [RLMv2 Benchmarks](#rlmv2-benchmarks)
6. [Cognitive Architecture Experiments](#cognitive-architecture-experiments)
7. [Diagnostic Tests](#diagnostic-tests)
8. [Custom Experiments](#custom-experiments)
9. [Results & Analysis](#results--analysis)

---

## Quick Start

```bash
# From repository root
cd /path/to/ravana

# Run all ML framework tests
python -m pytest tests/ -v

# Run cognitive core tests
python -m pytest ravana-v2/tests/ -v

# Run specific experiment
python experiments/experiment_cross_domain.py

# Run with output
python experiments/experiment_cross_domain.py 2>&1 | tee results/cross_domain_$(date +%Y%m%d_%H%M%S).log
```

---

## Experiment Categories

### Modular Package Experiments (NEW — `ravana/`)

| Category | Location | Purpose |
|----------|----------|---------|
| **Ablation Study** | `scripts/run_ablation.py` | Systematic config ablation (VAD, RLM, Beliefs, Curiosity, Modes) |
| **Per-Triple Evaluation** | `scripts/triple_eval.py` | Detailed per-triple diagnostics (PE, confidence, source, edge health) |
| **Chat Quality** | `experiments/experiment_chat_quality.py` | Dialogue coherence (modular) |
| **Background Learner** | `experiments/experiment_background_learner.py` | Continuous web learning |
| **Curiosity** | `experiments/experiment_curiosity.py` | Intrinsic motivation (Phase 18) |
| **Chat Ablation** | `scripts/run_ablation.py --quick` | Fast baseline + single flag removals |

### GRACE / ML Framework Experiments

| Category | Location | Purpose |
|----------|----------|---------|
| **Cross-Domain** | `experiments/experiment_cross_domain.py` | Zero-shot generalization across domains |
| **Phase 4 Integrated** | `experiments/experiment_phase4_integrated.py` | Triplet margin + wake-sleep cycle |
| **RLMv2 Benchmarks** | `experiments/experiment_triple_benchmark_v6.py` | Triple decomposition accuracy |
| **Component Interaction** | `experiments/experiment_component_interaction.py` | Ablation of architectural components |
| **Domain Specialization** | `experiments/experiment_domain_specialization.py` | Multi-domain learning |
| **Performance** | `experiments/experiment_performance.py` | Speed/memory profiling |
| **Robustness** | `experiments/experiment_robustness.py` | Noise, adversarial, OOD |
| **Human Eval** | `experiments/experiment_human_eval.py` | Qualitative assessment |
| **Persistence** | `experiments/experiment_persistence.py` | Save/load, checkpoint integrity |
| **Scaling Laws** | `experiments/experiment_scaling_laws.py` | Performance vs model size |
| **Ablation** | `experiments/experiment_ablation.py` | Component necessity |
| **Sleep/Memory** | `experiments/experiment_sleep_memory.py` | Consolidation effects |
| **Longitudinal** | `experiments/experiment_longitudinal.py` | Long-term training dynamics |
| **Cross-Domain (v2)** | `experiments/experiment_cross_domain.py` | Cognitive core transfer |

---

## Modular Package Experiments (NEW — `ravana/`)

### Ablation Study — `scripts/run_ablation.py`

Systematically runs all combinations of ablation flags and compares results.

```bash
# Full factorial (4 flags × 3 modes = 48 configs, 3 runs each)
python scripts/run_ablation.py --queries "hello|what is trust|explain oxiverse|bye" --runs 3

# Quick mode (baseline + 4 single removals + 2 mode variants = 7 configs)
python scripts/run_ablation.py --quick --modes

# Custom queries
python scripts/run_ablation.py --queries "what causes rain|how does learning work|tell me about sleep" --runs 2
```

**Ablation flags tested:**

| Flag | Component | What it removes |
|------|-----------|-----------------|
| `--no-vad` | VADEmotionEngine | VAD emotion modulation of inference |
| `--no-rlm` | RLMv2 (RelationPredictor + PropagationEngine) | Triple verification pathway |
| `--no-beliefs` | BeliefStore | Multi-user belief tracking & merging |
| `--no-curiosity` | WebLearner curiosity drive | Autonomous background learning |

**Mode variants:** `stochastic` (default) | `deterministic` | `exploratory`

**Output:** `ablation_results.json` + markdown comparison table with:
- Success rate per config
- Average latency per query
- Total time
- Error modes

### Per-Triple Evaluation Harness — `scripts/triple_eval.py`

Provides detailed per-triple diagnostics instead of averaged metrics.

```bash
# Default 10 triples (semantic, causal, contrastive) against trained engine
python scripts/triple_eval.py --data-dir ./data --output triple_eval_report.json

# Custom triples file
python scripts/triple_eval.py --triples-file my_triples.json --output report.json
```

**Per-triple metrics:**

| Metric | Description |
|--------|-------------|
| `relation_type` | semantic / causal / contrastive / possessive / temporal / analogical |
| `prediction_error` | How well the model predicts the object (1 - prob) |
| `confidence` | Model's confidence in the prediction |
| `top1_accuracy` | Exact match (rank = 1) |
| `top5_accuracy` | In top 5 predictions |
| `rank` | Position of correct object in predictions |
| `edge_weight` | Graph edge weight (if exists) |
| `edge_confidence` | Graph edge confidence |
| `edge_type` | Graph edge relation type |
| `edge_prediction_free_energy` | Edge-level free energy (if available) |
| `source` | Where learned: `seed` \| `web` \| `user` \| `sleep` \| `inference` \| `unknown` |
| `learned_turn` | Turn number when concept was learned |
| `predicted_object` | Model's top-1 prediction |
| `predicted_prob` | Probability of top-1 prediction |
| `all_candidates` | Top-5 (token, prob) pairs |

**Aggregated diagnostics generated:**
- **Overall**: top-1, top-5, avg PE, avg confidence, avg rank
- **By relation type**: per-type counts, accuracies, PE, confidence, edge stats
- **By source**: per-source counts, accuracies, PE
- **Edge health**: avg weight, confidence, PE, type distribution

**Example output:**
```
========================================
PER-TRIPLE EVALUATION REPORT
========================================
Timestamp: 2026-06-14T15:05:12.137630
Total triples evaluated: 10

OVERALL METRICS:
  Top-1 Accuracy:    0.0%
  Top-5 Accuracy:    0.0%
  Avg Prediction PE: 0.0000
  Avg Confidence:    0.0000
  Avg Rank:          0.0

BY RELATION TYPE:
  semantic          : n=  5  Top1=  0.0%  Top5=  0.0%  PE=0.0000  Conf=0.0000  EdgeW=0.000
  causal            : n=  4  Top1=  0.0%  Top5=  0.0%  PE=0.0000  Conf=0.0000  EdgeW=0.000
  contrastive       : n=  1  Top1=  0.0%  Top5=  0.0%  PE=0.0000  Conf=0.0000  EdgeW=0.000

BY SOURCE:
  unknown           : n= 10  Top1=  0.0%  PE=0.0000
```

---

## Cross-Domain Transfer

### Experiment: `experiment_cross_domain.py`

Tests **zero-shot generalization** from trained domains to held-out domains.

```bash
python experiments/experiment_cross_domain.py
```

**Protocol:**
1. Train on Domain A (e.g., physics: "heat causes expansion")
2. Test on Domain B (e.g., social: "kindness causes friendship") — zero-shot
3. Measure top-1 / top-10 accuracy

**Key Results (from paper):**
| Condition | Top-1 | Top-10 |
|-----------|-------|--------|
| Optimized config | 95% | 100% |
| Neutral, zero-shot | 10% | — |

**Configurable:**
```python
# In experiment_cross_domain.py
config = ExperimentConfig(
    train_domains=["physics", "chemistry"],
    test_domains=["social", "biology"],
    n_triples_per_domain=50,
    epochs=500,
    model_config={...},
)
```

### Experiment: `experiment_triple_benchmark_v6.py`

RLMv2 triple decomposition benchmark (47 triples, 500 epochs).

```bash
python experiments/experiment_triple_benchmark_v6.py
```

**Results:**
| Benchmark | Result |
|-----------|--------|
| Overall top-10 | 80.9% |
| Cross-domain causal top-10 | 75% |
| RP-only verb-offset cross-domain top-10 | 6.7% |

---

## Phase 4 Integrated

### Experiment: `experiment_phase4_integrated.py`

Combines **triplet margin loss** + **wake-sleep cycle** for contrastive representation learning.

```bash
python experiments/experiment_phase4_integrated.py
```

**Architecture:**
```
Wake:  Contrastive learning (InfoNCE) on hidden states
       ↓
Sleep: Replay + consolidation + contrastive replay
       ↓
Repeat
```

**Key metrics:**
- Within-domain top-1: 100% (was 0%)
- Contrastive alignment: COSINE similarity separation

---

## RLMv2 Benchmarks

### Diagnostic Tests (Moved to `tests/`)

```bash
# RP-only (analogy path only, no spreading activation)
python tests/test_rp_only.py

# RP contrastive (with negative sampling)
python tests/test_rp_contrastive.py

# Structural transfer (graph-aware encoder alignment)
python tests/test_structural_transfer.py
```

**What they test:**
- `test_rp_only.py`: Pure verb-stem offset predictor accuracy
- `test_rp_contrastive.py`: Contrastive regularization effect
- `test_structural_transfer.py`: Graph-aware encoder alignment (33% → 100% traversal)

---

## Cognitive Architecture Experiments

### GRACE Phase Runner

```bash
cd ravana-v2
python experiments/runner.py           # Run all phases
python experiments/phases/run_phase_b.py  # Specific phase
```

**Phases tested:**
- Phase A: Governor regulation stability
- Phase B: Adaptation from clamp events
- Phase C: Strategy mode selection
- Phase D: Intent formation
- Phase E: Non-stationary environment survival
- Phase F: World model prediction
- Phase G: VoI-driven action selection
- Phase J: Hypothesis generation
- Phase K: VAD emotion dynamics
- Phase L: Sleep consolidation
- Phase M: Meaning/intrinsic motivation
- Phase N: Global workspace broadcast
- Phase O: Human memory persistence

### Custom Phase Config

```python
# ravana-v2/experiments/phases/config.py
PhaseConfig(
    governor=GovernorConfig(...),
    environment=EnvironmentConfig(...),
    sleep=SleepConfig(...),
    # ...
)
```

---

## Diagnostic Tests

### Unit Tests (ML Framework)

```bash
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_ravana.py -v
python -m pytest tests/test_rlm_v2.py -v
python -m pytest tests/test_dialogue_engine_integration.py -v
python -m pytest tests/test_dialogue_system.py -v
```

### Cognitive Core Unit Tests

```bash
python -m pytest ravana-v2/tests/ -v
```

Key test files:
- `test_governor.py` — Regulation modes, clamp diagnostics
- `test_sleep.py` — Sleep stages, dream sabotage, rollback
- `test_human_memory.py` — Episodic/semantic, decay, consolidation
- `test_emotion.py` — VAD dynamics, modulation
- `test_identity.py` — Momentum, recovery bias
- `test_global_workspace.py` — Competition, broadcast
- `test_dual_process.py` — System 1/2 routing

---

## Custom Experiments

### Template

```python
# my_experiment.py
import numpy as np
from ravana.cognitive import CognitiveFramework, FrameworkConfig
from ravana_ml.tokenizer import WordTokenizer

def run_experiment():
    # 1. Setup
    tok = WordTokenizer()
    # ... build training data ...
    
    config = FrameworkConfig(
        concept_dim=64,
        max_concepts=5000,
        # ... custom config ...
    )
    fw = CognitiveFramework(config)
    state = fw.initialize()
    
    # 2. Train
    for episode, (inp, tgt) in enumerate(training_data):
        concepts = fw.perceive(state, inp)
        preds = fw.predict(state, concepts)
        state = fw.learn(state, preds, tgt, episode)
        if episode % 100 == 0:
            state = fw.sleep(state)
    
    # 3. Evaluate
    results = evaluate(fw, state, test_data)
    return results

if __name__ == "__main__":
    results = run_experiment()
    print(results)
```

### Using RLMv2 Directly

```python
# rlm_experiment.py
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

tok = WordTokenizer()
# ... encode facts ...

model = RLM(
    vocab_size=tok.vocab_size,
    embed_dim=64,
    concept_dim=64,
    n_concepts=200,
    sleep_interval=50,
)

for premise, conclusion in facts:
    inp = np.array(tok.encode(premise), dtype=np.int64)
    tgt = np.array(tok.encode(conclusion), dtype=np.int64)
    model.learn(inp, tgt)
    if model._step_counter % 50 == 0:
        model.sleep_cycle()

# Test
for premise, _ in facts:
    inp = np.array(tok.encode(premise), dtype=np.int64)
    logits = model.forward(inp)
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"{premise} → {[tok.decode([t]) for t in top5]}")
```

---

## Results & Analysis

### Output Locations

```
results/
├── benchmarks/           # JSON benchmark results
├── diagnostics/          # Per-triple diagnostic CSVs
├── plots/                # Activation dynamics, coherence trajectories
├── logs/                 # Full experiment logs
└── checkpoints/          # Model checkpoints
```

### Key Metrics to Track

| Metric | Meaning | Target |
|--------|---------|--------|
| `top1_accuracy` | Exact match rate | >80% within-domain |
| `top10_accuracy` | In top 10 predictions | >95% within-domain |
| `cross_domain_top10` | Zero-shot transfer | >70% |
| `forgetting_rate` | Retention after domain switch | 0% (eliminated) |
| `sleep_cycle_time` | Consolidation speed | <300ms |
| `coherence` | Graph activation coherence | >0.7 |
| `dissonance` | Prediction error pressure | ~0.3 (target) |
| `identity` | Self-concept stability | >0.7 |

### Analysis Tools

```python
from ravana.lab import (
    analyze_concept_graph,
    plot_activation_dynamics,
    compute_coherence_trajectory,
    visualize_sleep_cycle,
    diagnose_learning,
)

# Graph structure
stats = analyze_concept_graph(model.graph)

# Activation over time
plot_activation_dynamics(model.graph, save_path="results/plots/activation.png")

# Coherence trajectory
traj = compute_coherence_trajectory(fw, episodes=1000)

# Sleep diagnostics
visualize_sleep_cycle(fw.state_manager.sleep, save_path="results/plots/sleep.png")

# Learning diagnosis
report = diagnose_learning(model, test_data)
```

### Comparing Runs

```bash
# Compare two experiment runs
python scripts/compare_benchmarks.py results/benchmarks/run1.json results/benchmarks/run2.json

# Output:
# Metric                    Run 1     Run 2     Delta
# top10_accuracy            0.809     0.852     +0.043
# cross_domain_top10        0.750     0.812     +0.062
# forgetting_rate           0.000     0.000     0.000
# sleep_cycle_ms            255       240       -15
```

---

## Experiment Configuration Reference

### ML Framework Experiments

```python
# Common config pattern
class ExperimentConfig:
    # Model
    vocab_size: int
    embed_dim: int = 64
    concept_dim: int = 64
    n_concepts: int = 100
    n_hidden: int = 128
    n_layers: int = 3
    
    # Training
    epochs: int = 500
    sleep_interval: int = 50
    replay_buffer_max: int = 500
    replay_n_samples: int = 20
    
    # Data
    train_domains: List[str]
    test_domains: List[str]
    n_triples_per_domain: int = 50
    
    # Ablation flags
    anchor_relation_vectors: bool = True
    gate_concept_creation: bool = True
    adaptive_downscale: bool = True
    use_glove: bool = True
    use_verb_offset: bool = True
```

### Cognitive Core Experiments

```python
class CognitiveExperimentConfig:
    # Framework
    framework_config: FrameworkConfig
    
    # Environment (Phase E)
    environment_config: EnvironmentConfig
    
    # Episodes
    n_episodes: int = 10000
    eval_interval: int = 1000
    
    # Metrics
    track_dissonance: bool = True
    track_identity: bool = True
    track_clamp_diagnostics: bool = True
    track_sleep_records: bool = True
```

---

## Reproducing Paper Results

### Key Results from Paper

| Experiment | Command | Expected |
|------------|---------|----------|
| Within-domain top-1 | `python experiments/experiment_triple_benchmark_v6.py` | 100% |
| RLMv2 overall top-10 | `python experiments/experiment_triple_benchmark_v6.py` | 80.9% |
| Cross-domain causal top-10 | `python experiments/experiment_cross_domain.py` | 75% |
| RP-only verb-offset top-10 | `python tests/test_rp_only.py` | 6.7% |
| Catastrophic forgetting | `python experiments/experiment_cross_domain.py` | 0% |
| Graph-aware alignment | `python tests/test_structural_transfer.py` | 100% traversal |
| Sleep optimization | (internal benchmark) | 255ms (2.6× speedup) |

### Hardware Requirements

- **CPU**: Any modern x86_64 (NumPy uses BLAS)
- **RAM**: 4GB+ for large graphs (10k concepts)
- **Disk**: ~1GB for GloVe embeddings + checkpoints
- **GPU**: NOT required (CPU-only by design)

### Random Seeds

```python
# For reproducibility
np.random.seed(42)
# Note: Some stochasticity in graph operations, sleep sabotage
# Set seeds in each module for full determinism
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: ravana_ml` | `pip install -e ravana/` |
| `GloVe not found` | Download to `data/glove/glove.6B.100d.txt` |
| Slow experiments | Reduce `n_concepts`, `concept_dim`, increase `sleep_interval` |
| OOM on large graphs | Reduce `max_nodes` in `ConceptGraph` |
| Sleep never triggers | Lower `pressure_threshold` in `SleepConfig` |
| NaN in training | Check learning rates, add gradient clipping |

---

## See Also

- [Getting Started](GETTING_STARTED.md)
- [Architecture](ARCHITECTURE.md)
- [ML Framework](ML_FRAMEWORK.md)
- [Cognitive Core](COGNITIVE_CORE.md)
- [Advanced Topics](ADVANCED_TOPICS.md)