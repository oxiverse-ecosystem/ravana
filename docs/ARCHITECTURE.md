# RAVANA — Cognitive Architecture & ML Framework

> **A self-stabilizing, self-expanding epistemic system.**
> CPU-native cognitive ML framework — no GPU required.
>
> **Date**: 2026-06-01
> **Status**: Active Development (v2 GRACE + Cognitive Modules + RLMv2 + Phase 2 NN Bridge)
> **Author**: Likhith + Zo Agent

---

## What RAVANA Is

RAVANA is a **self-consistent learning paradigm** built on pressure-driven self-organization rather than loss function optimization. Unlike PyTorch/TensorFlow which minimize objective functions via gradient descent, RAVANA learns through **cognitive dissonance minimization**: prediction errors create internal pressure, and the system self-organizes to reduce that pressure.

Think: differentiable physics engine for cognition, not a differentiable function approximator.

---

## Version Lineage

| Version | Status | Core Innovation | Location |
|---------|--------|----------------|----------|
| v1 | Concept | Initial cognitive architecture | N/A |
| v2 | **Active** | GRACE + VAD Emotion + Sleep + Dual-Process + Meaning | `ravana-v2/core/` |
| v4 | Reference | Sensorimotor + Pressure Physics + Sleep cognition | `ARCHITECTURE.md` (prior) |

> v2 is the active codebase with integrated cognitive modules. v4 is a reference design document for future evolution.

---

## Core Design Philosophy

### Learning Through Physics, Not Optimization

Transformers/PyTorch minimize loss functions. RAVANA is a dynamical self-stabilizing system.

```
Traditional ML:         loss = f(prediction, target) → gradient → update weights
RAVANA:                 prediction error → pressure → self-organization → equilibrium
```

### Hybrid Discrete + Continuous

- **Discrete**: Symbolic concept IDs, prediction edges, mode selections
- **Continuous**: State vectors, pressure fields, identity momentum

### Key Primitives

1. **Pressure** = learning signal (replaces loss function)
2. **Coherence** = objective metric (replaces accuracy)
3. **Governor** = optimizer (replaces gradient descent)
4. **Identity** = parameter regularization (replaces weight decay)
5. **Sleep** = consolidation phase (replaces batch normalization)
6. **VAD Emotion** = affective state (adds valence/arousal/dominance to every concept)
7. **Meaning** = intrinsic motivation (coherence gain × effort cost)

---

## Theoretical Foundations

RAVANA integrates multiple cognitive science frameworks into a unified pressure-driven system:

### Kahneman's Dual-Process Theory (System 1 / System 2)

| System | Characteristics | RAVANA Implementation |
|--------|----------------|----------------------|
| System 1 | Fast, automatic, intuitive | Concept activation spread, Hebbian completion |
| System 2 | Slow, deliberate, analytical | MCTS planning, belief reasoning, argument construction |

**Override logic**: System 2 engages when System 1 confidence is low, novelty is high, or stakes are high.
Cognitive cycles (~200-300ms each) form the atomic unit of processing.

### Mayer-Salovey Four-Branch Emotional Intelligence

| Branch | Function | RAVANA Implementation |
|--------|----------|----------------------|
| Perception | Identify emotions in self/others | VAD state tracking + Gaussian Process empathy |
| Use | Harness emotion for thinking | Emotion-scaled GW bidding, empathetic action bonuses |
| Understanding | Comprehend emotional nuance | Causal emotion models, differentiation calibration |
| Regulation | Manage emotional responses | Reappraisal-focused regulation (reframing, not suppression) |

### Festinger's Cognitive Dissonance (Lehr et al. 2025 extension)

Dissonance is computed as belief-action distance, not just scalar error:

```
D = Σ|belief_i - action_j| × confidence_i × VAD_weight_k
    + identity_violation_penalty × commitment_strength
```

High dissonance forces: reappraisal, behavioral correction, belief change, or commitment decay.
The system cannot maintain indefinite dissonance — forced into coherence.

### Behavioral Economics & Bias Mitigation

| Bias | RAVANA Mitigation |
|------|-------------------|
| Confirmation bias | 20% counterfactual reversals in dreams |
| Overconfidence | Dual-confidence system (mean + volatility decay) |
| Status quo bias | Commitment Identity Link requires verification |
| Anchoring | Explicit priors with adaptation tracking |
| Sunk cost | Forward-looking planning ignores past costs |

### Four Pressures for Emergence

1. **Global Falsification** — prediction error lowers confidence, raises volatility, deprioritizes weak beliefs
2. **Dissonance-Driven Self-Correction** — belief-action mismatch forces coherence
3. **Structured Dream Sabotage** — 20% counterfactual reversals prevent overfitting
4. **Meaning as Staked Coherence** — costly coherence gain drives genuine growth

Wisdom emerges not from programming, but from optimization under coherence pressures.

---

## GRACE Architecture (v2 Codebase)

The `ravana-v2/core/` directory implements the GRACE architecture across ordered development phases:

```
Governance      → Governor, ClampDiagnostics         (Phase A)
Reflection      → ResolutionEngine, IdentityEngine    (Phase A)
Adaptation      → PolicyTweakLayer                    (Phase B)
Constraint      → GovernorConfig, ClampEvents         (Phase A)
Exploration     → StrategyLayer, StrategyLearning     (Phase C)
```

### Phase A: Core Regulation (`governor.py`, `resolution.py`, `identity.py`, `state.py`)

**The Central Governor** — all state changes must pass through it.

```python
# No state modification without governor passage
regulated = governor.regulate(current_dissonance, current_identity, signals)
```

Three-layer regulation:
1. **Hard Constraints**: Absolute ceilings/floors (e.g., `max_dissonance=0.95`)
2. **Soft Boundary**: Sigmoid pressure curve near limits (air, not brick wall)
3. **Center-Seeking**: Homeostatic pull toward target dissonance

**The Identity Engine** — momentum-based self-concept with recovery bias:
- Identity grows from successful resolution, decays from stagnation
- Failure penalty is structural (-0.08 per failed prediction)
- Resolution bonus matches penalty (+0.08 per success)
- Streak multiplier rewards sustained correctness

**The Resolution Engine** — continuous partial credit accumulation:
- Not binary success/failure — accumulated partial credit toward wisdom
- Threshold crossing generates wisdom events
- Difficulty-based scaling on credit

### Phase B: Adaptation (`adaptation.py`)

Lightweight policy learning from clamp events:
- Simple 5D state → 2D output linear policy
- Learns to avoid needing correction (not just stay in bounds)
- Reward = exploration bonus - clamp_penalty * correction_magnitude
- Momentum-based gradient updates

### Phase C: Strategy Learning (`strategy.py`, `strategy_learning.py`)

Four deliberate exploration modes selected via soft sigmoid scoring:

| Mode | When | Behavior |
|------|------|----------|
| EXPLORE_AGGRESSIVE | Low D, room to grow | High delta, high noise |
| EXPLORE_SAFE | Near boundary, not crisis | Moderate delta, low noise |
| STABILIZE | High identity, low variance | Low delta, preserve gains |
| RECOVER | Crisis detected | Force D reduction |

Strategy learning evaluates mode effectiveness over time and shifts preferences.

### Phase D: Intent & Planning (`intent.py`, `planning.py`)

Dynamic objectives that evolve based on outcomes:
- Four system objectives: EXPLORE, STABILIZE, OPTIMIZE_IDENTITY, MINIMIZE_CLAMPS
- Objective weights self-adjust based on satisfaction trends
- Micro-planner simulates forward trajectories before mode selection

### Phase E: Non-Stationary Environment (`environment.py`)

A world that changes in ways RAVANA must discover:
- Boundary shifts, noise drift, goal flips, hidden difficulty cycles
- RAVANA is NOT told when changes happen — must detect through consequences

### Phase F: Learned Predictive World Model (`predictive_world.py`)

Simple neural network that predicts next state:
- Input: `[D, I, clamp_rate, trend, stability, mode]`
- Hidden: 12 units, Output: 3 (predicts D, I, clamp)
- Adaptive surprise threshold (2x baseline)
- Belief inertia resists false world patterns

### Phase F.5: Belief Reasoner (`belief_reasoner.py`)

Multiple competing hypotheses about the world:
- Not single belief, but distribution over hypotheses
- Confidence decays without confirmation (built-in skepticism)
- Structural consistency checks ("would past make sense?")
- Hypothesis spawning when evidence contradicts all active beliefs

### Phase G: Active Epistemology (`active_epistemology.py`)

From passive reasoner to intentional discoverer:
- Value of Information (VoI) calculation per action
- Hypothesis-driven action selection
- Intentionally probes to maximize hypothesis separation

### Phase G.5: Surgical Probing (`surgical_probes.py`)

KL-divergence driven probe selection:
- Designs experiments that maximally separate competing hypotheses
- Predicts outcomes under each hypothesis for each probe type
- Selects probe with highest expected information gain

### Phase H: Social Epistemology (`social_epistemology.py`)

Multi-agent belief conflict resolution:
- Trust scoring by epistemic reliability
- Consensus formation via trust-weighted averaging
- Adversarial testing (deliberate misleading agents)
- Deception detection (low honesty over time)

### Phase I: Meta-Cognition (`meta_cognition.py`)

Self-awareness of the epistemic process:
- Monitors probe effectiveness, confidence calibration, systematic bias
- Recommends epistemic mode: CAUTIOUS, EXPLORATORY, RECOVERY, CONFIDENT
- Prevents confident-fool, systematic-misinterpreter, frozen-thinker failure modes

### Phase I+: Reality Friction (`reality_friction.py`)

From laboratory epistemics to adversarial reality:
- Partial observability (never fully visible state)
- Delayed ground truth (feedback arrives late, partial, or never)
- Noisy signals (observation = truth + adversarial noise)
- Hidden variables (causal factors RAVANA cannot observe)

### Phase I²: Meta²-Cognition (`meta2_cognition.py`, `meta2_integration.py`)

The system questioning its own epistemic method:
- Hypothesis space audits (is the true model in my space?)
- Bias detection (am I structurally blind?)
- Epistemic epiphanies (radical method revisions)
- Auto-expands hypothesis space when systematic failure detected

### Phase J: Hypothesis Generation (`hypothesis_generation.py`)

Constraint-guided hypothesis generation when probing plateaus:
- Incremental complexity: parametric → structural → causal
- Occam penalty prevents overfitting
- Lifecycle management: confidence, complexity, survival score, pruning

### Phase J.1: Occam Layer (`occam_layer.py`)

Explicit complexity penalties for epistemic discipline:
- `score = explanatory_power - lambda * complexity * evidence_factor`
- Penalty grows with evidence (prevent overfitting with more data)
- Pruning of low-scoring hypotheses

### Phase K: VAD Emotion Engine (`emotion.py`)

3D affective state (Valence, Arousal, Dominance) with differential equation dynamics:

```python
dV/dt = ηv(stimulus_valence - V) - λv * V
dA/dt = ηa(stimulus_arousal + 0.3 * uncertainty - A) - λa(A - baseline)
dD/dt = ηd(stimulus_dominance - D) - λd * D
```

- Euler integration for real-time update
- VAD tags on all concepts and memories
- Emotion-cognition coupling (VAD affects mode selection, GW bidding)
- Anticipation-driven emotion via MCTS forward simulation
- Supports dissonance weighting (VAD_k scales belief-action gap)

### Phase K.5: Empathy Engine (`empathy.py`)

Theory of Mind via Gaussian Process regression:
- Infers others' VAD states from behavioral cues (text sentiment, speech rate, etc.)
- Perspective-taking: simulates others' belief states under own model
- Empathy-driven modulation: own decisions weighted by inferred other-state
- Trust-weighted multi-agent emotion tracking

### Phase L: Sleep & Dream Consolidation (`sleep.py`)

Periodic consolidation phase triggered by accumulated pressure > threshold:

**4-Stage Sleep:**
1. **Topology Analysis** — Identify high-pressure zones, unstable prediction edges
2. **Pattern Compression** — Find frequent concept clusters, strengthen intra-cluster edges
3. **Contradiction Resolution** — For each active contradiction, rewire (not delete) weakest edge
4. **Integration** — Merge consistent clusters, update confidence, rollback if coherence drops

**Dream Sabotage** (anti-overfitting):
- 20% counterfactual outcome reversals ("what if failure was success?")
- 10% emotional valence flipping
- 1.5x failure experience oversampling
- Symbolic recombination of concept graph edges

**Safeguards:**
- Tier-0 identity protection (core self-concept never perturbed)
- Abort-on-instability rollback (coherence drop > threshold → restore snapshot)
- Controlled perturbation radius (≤2 hops from contradiction, max 0.05 per edge)

### Phase L.5: Dual-Process Controller (`dual_process.py`)

Explicit System 1 / System 2 architecture:

```
System 1 (fast):    Hebbian concept activation, pattern completion  →  1-2 GW cycles
System 2 (slow):    MCTS planning, belief reasoning, argument construction  →  4-10 GW cycles
```

- **Override logic**: System 2 engages when confidence < threshold, novelty high, or stakes high
- **Cognitive load tracking**: limits System 2 invocation under pressure
- **Fluency heuristic**: skip System 2 when System 1 is fluent and confident
- **Metacognitive control**: dual-process itself is subject to meta-cognition oversight

### Phase M: Meaning Engine (`meaning.py`)

Intrinsic motivation via costly coherence gain:

```
M = w1(-ΔD_future) + w2(Δidentity_coherence) + w3(Δpredictive_power)
    × (1 + κ × effort_cost)
```

- Meaning accumulates as tracked metric (parallel to wisdom)
- Meaning-staking: committing to hypotheses imposes identity cost if wrong
- Meaning-driven curiosity: pursue actions with high expected M
- Meta-RL shaped by M learns: pursue coherence gains, resolve conflicts, value integrity

### Phase N: Global Workspace Integration (`global_workspace.py`)

Soft attention bidding system for module coordination:

```
bid_i = emotion_intensity_i × novelty_i × goal_relevance_i
        × mean_conf_i × exp(-α × volatility_conf_i)
```

- All modules send signals with computed bids
- Softmax(bids) → top-K (K=3-5) signals broadcast to all modules
- Low-confidence or volatile signals naturally deprioritized
- Winning signals influence all downstream processing

### Memory System (`memory.py`)

Human-like memory architecture:
- **Episodic**: Time-stamped traces, salience-weighted capacity management
- **Semantic**: Bayesian knowledge graph for long-term norms
- **Working**: Global workspace buffer (capacity-limited attention)

---

## Training Loop

The training pipeline (`training/pipeline.py`) orchestrates cognitive episodes:

```python
for episode in range(total_episodes):
    difficulty = compute_difficulty(episode)
    correctness = simulate_outcome(difficulty)
    
    # Governor-gated cognitive step
    step_record = manager.step(
        correctness=correctness,
        difficulty=difficulty
    )
    
    # All state changes pass through Governor
    # Governor applies: hard constraints → predictive dampening
    #   → boundary pressure → center-seeking → mode regulation → final clamp
```

No backprop. No gradient descent. The Governor is the optimizer.

---

## Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Dissonance (D) | Current cognitive tension | 0.0 (none) - 1.0 (crisis) |
| Identity (I) | Self-concept strength | 0.0 (fragile) - 1.0 (rigid) |
| Wisdom | Accumulated partial credit threshold crossings | Epistemic growth |
| Coherence | Weighted mean of confidence over active concepts | System health |
| Meaning (M) | Coherence gain × effort cost | Intrinsic motivation |
| Clamp Rate | Governor corrections / upstream suggestions | Constitutional alignment |
| Prediction Accuracy | Successful / total predictions | World model quality |
| Valence (V) | VAD emotional valence | -1.0 (negative) - 1.0 (positive) |
| Arousal (A) | VAD emotional arousal | 0.0 (calm) - 1.0 (excited) |
| Dominance (Dm) | VAD emotional dominance | 0.0 (submissive) - 1.0 (dominant) |
| Dream Pressure | Accumulated consolidation need | Triggers sleep when > threshold |

---

## Quick Start

```bash
# Run unit tests (recommended first step)
python -m pytest ravana-v2/core/ -v

# Run cognitive architecture experiments
python experiments/runner.py

# Run RLMv2 unit tests
python -m pytest tests/ -v
```

---

## What RAVANA Is NOT

- Not an LLM — no transformer, no token prediction
- Not symbolic AI — no logic rules, no truth tables
- Not a reward-based RL — no external reward signal
- Not a neural network trainer — no gradient descent, no backprop, no GPU needed
- Not PyTorch/TensorFlow — no loss functions, no optimizers, no autograd

## What RAVANA IS

- A pressure-driven self-organizing cognitive system
- A CPU-native ML framework where learning emerges from prediction failures
- Where memory IS the model (semantic weights = parameters)
- Where sleep is thermodynamic necessity (consolidation phase)
- Where identity is structural, not programmed (self-concept as regularizer)
- Where emotions (VAD) shape cognition and learning
- Where dual-process (System 1 / System 2) enables both fast intuition and slow deliberation
- Where dreams prevent dogmatism through structured sabotage
- Where meaning emerges from costly coherence gain

---

## RLMv2 — Triple Decomposition Architecture

A clean-room rewrite replacing character-level GRU with brain-inspired triple decomposition. Input text is decomposed into (subject, relation_type, object) triples, which activate concepts via spreading activation over the concept graph. 1,247 lines. 11/11 unit tests passing.

**Key innovation**: Instead of character-level sequence modeling, RLMv2 parses knowledge into structured triples and performs graph-based reasoning — closer to how biological neural circuits encode relational knowledge.

**Current RLMv2 v6 benchmark**: 80.9% top-10 on 47-triple benchmark (500 epochs, standard config). Relation vector separation: 0.551.

---

## Phase 2: NN Bridge + Composed Reasoning

Pre-trained sentence transformer (MiniLM-L6-v2, 384-dim) provides semantic embeddings for novel term bridging. Unknown terms are mapped to nearest known concepts via cosine similarity (no dimensionality projection needed).

**Composed Reasoning Pipeline:**
- Independent traversals per candidate concept
- Depth decay factor (0.7x per hop)
- Reverse edge inheritance (if A→B, infer B can relate back to A)
- Bridge-as-candidate (bridged concepts compete with direct matches)

**Best Results** (experiment_reverse_inheritance.py / experiment_final_bridge.py, verified 2026-06-03):

| Metric | Value |
|--------|-------|
| Bridge accuracy | 67% (8/12 terms) |
| Query success | 95% (21/22) |
| Object hit rate | 94% (29/31) |

Only failure: matcha (MiniLM embedding similarity 0.32 — below threshold).

**Progression over iterations:**
```
42% bridge / 45% query → 67% bridge / 59% query → 67% bridge / 68% query → 67% bridge / 95% query
```

---

## New Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `episode_injector.py` | 276 | Synthetic Episode Injector for structured knowledge injection |
| `relation_ontology.py` | 231 | Multi-level relation hierarchy (Family > Sub-family > Predicate) |
| `word_tokenizer.py` | 46 | Word-level tokenizer for RLMv2 |
| `ravana-v2/core/embedder.py` | 188 | LearnedEmbedder (character n-gram + random projection) |

---

## Updated Line Counts (2026-06-02)

| Component | Lines | Files |
|-----------|-------|-------|
| `ravana_ml/` | 4,383 | 16 |
| `ravana-v2/core/` | 10,162 | 27 |
| `ravana/` package | 855 | 10 |
| **Source total** | **15,400** | **53** |
| **Full project (all Python)** | **~40,700** | **170** |

---

## Reference: v4 Sensorimotor Design (Future Direction)

The v4 architecture (documented in prior `ARCHITECTURE.md`) extends these principles to sensorimotor grounding: a 2D organism in a 10x10 grid world with food, heat, walls — learning through physics-based pressure accumulation, sleep-triggered topology reorganization, and coherence recovery.

Key v4 concepts for future evolution:
- **Concept Nodes** as tiny dynamical systems with state vectors, attractor fields
- **4-Stage Sleep**: Episodic Replay → Pattern Compression → Contradiction Simulation → Topology Stabilization
- **Identity Anchors**: Very-low-plasticity attractors preserving self-continuity
- **Pressure Physics**: The core primitive — contradiction → pressure → sleep → reorganization → coherence improvement

---

> *"Cognition as pressure-driven self-organization."*
> *"The model learns through physics, not optimization."*
>
> This is the center of gravity. Everything else is implementation detail.
