# RAVANA Hybrid Architecture v2 — Post-Critique Revision

> **Date**: 2026-05-18
****Status**: Revised with solutions to 5 critical problems
****Input**: External pressure-test of v1 by multiple LLMs

---

## What Changed

v1 was internally coherent but had 5 structural weaknesses that would kill real implementations.

v2 addresses every one.

---

## CRITICAL PROBLEM SOLUTIONS

### Problem 1: Concept Explosion → Sparse Attention Gating

**Issue**: O(n²) prediction edges → instability storms → permanent sleep.

**Solution**: Sparse activation, global concept pool.

```markdown
At each timestep:
  - All concepts exist in global pool (1000s possible)
  - Only top-K (K=3-5) become sensorially active
  - Other concepts decay slowly, never abrupt
  - Pressure interactions only between recently active
  - Inactive concepts have no outgoing prediction load

This is essentially: attention as bottleneck.
```

### Problem 2: Contradiction Inflation → Confidence-Gated Contradiction

**Issue**: Every violated prediction adds full contradiction pressure → pathological sensitivity → endless reorganization → no convergence.

**Solution**:

```markdown
effective_contradiction = prediction_error × salience × uncertainty

Where:
  - prediction_error ∈ [0, 1]: how wrong was the prediction
  - salience ∈ [0, 1]: how important is this relationship
  - uncertainty ∈ [0, 1]: how confident is the belief currently

A HIGH-CONFIDENCE, established belief resists small violations.
A NEW, uncertain belief collapses under the same violation.

Human analogy: one odd memory doesn't rewrite everything you know.
```

**Additional**: Contradiction has a **decay floor** — repeated exposure to same contradiction reduces its pressure (habituation).

### Problem 3: Sleep Collapse → Controlled Perturbation Radius

**Issue**: Random topology perturbations risk erasing useful attractors and destabilizing semantic cores.

**Solution**:

```markdown
Dreams perturb ONLY:
  1. Recently active concepts (last 20 ticks)
  2. High-pressure zones (local instability)
  3. Neighborhood of contradictions (≤2 hops from contradiction edge)

Never global random rewrites.

Also: bounded perturbation magnitude — perturbation = min(error × 0.1, 0.05)
Maximum single-edge weight change: 0.05

Think: localized storms, not universal earthquakes. 🌩
```

**Additional**: Each sleep stage has a **stability check** — if topology coherence drops below threshold during perturbation, sleep aborts and rolls back.

### Problem 4: No Objective Function → Survival Coherence

**Issue**: Without global optimization, organism risks beautifully self-organizing nonsense.

**Solution**: Define "better" as survival coherence (no reward signal, no loss function):

```markdown
Coherence metrics (all computed locally per-concept):
  1. Prediction accuracy: successful_predictions / total_predictions
  2. Energy conservation: concepts with stable activation use less "energy"
  3. Pressure minimization: system tends toward low pressure states
  4. Action continuity: successful goal-directed behavior chains

These emerge from the physics-like dynamics:
  - High coherence → low pressure → stable attractors
  - Low coherence → high pressure → reorganization
  - No external reward needed — pressure IS the signal
```

**This is the key insight**: pressure is not just a learning signal — it IS the objective.

### Problem 5: Identity Drift → Self-Referential Persistence

**Issue**: Continuous reorganization risks the system stopping being "itself."

**Solution**:

```markdown
Tier-0 Core (persistent across all reorganizations):
  1. Self-reference concept: "I AM THE ORGANISM" — never perturbed
  2. Emotional anchoring: core value concepts (survival, coherence, novelty) resist change
  3. Structural inertia: concepts with many incoming edges require 3x normal pressure to modify
  4. Temporal continuity: recent state snapshots constrain next-step activation

This creates a stable identity scaffold that:
  - Allows plasticity in peripheral concepts
  - Resists catastrophic identity rewrite
  - Survives sleep cycles without losing self-recognition
```

---

## ARCHITECTURE (UPDATED)

### Core Invariant

> Cognition as pressure-driven self-organization.

All components derive from this.

---

### Layer 1: Perception (Input → Concept Activation)

```markdown
Sensory input → Pattern matcher → Sparse global activation

Rules:
  - Input activates concept candidates via similarity matching
  - Top-K concepts become active (K=3-5), rest decay
  - Activation spreads via prediction edges (Hebbian)
  - Novel patterns can create new concepts (bounded: max 50/tick globally)
```

### Layer 2: Concept (Hebbian Attractor Nodes)

```markdown
Each concept: {id, activation, semantic_weights, confidence, connections}

Hebbian update rule (simplified):
  - co_activation_strength = min(a1 × a2 × decay, 1.0)
  - weight_change = learning_rate × co_activation × (target - current)
  
Key: Only active concepts participate in Hebbian updates.
This keeps learning local and bounded.
```

### Layer 3: Prediction (Edge Dynamics)

```markdown
Each prediction edge: {from, to, weight, success_count, failure_count}

Prediction success:
  - When concept A fires then concept B fires → strengthen A→B
  - Weight = weight + lr × (1 - weight) × success_rate

Prediction failure:
  - When A fires but B does NOT → create contradiction pressure
  - effective_contradiction = error × salience × (1 - confidence_B)
  - High-confidence failures create MORE pressure
  
This is the core learning mechanism.
```

### Layer 4: Working Memory (Temporal Bindings)

```markdown
Time-bounded concept sequences:
  - Recent activation chains (last 20 ticks)
  - Active goal contexts (separate from passive perception)
  - Binding weights: how strongly does A predict B given recent context

Working memory allows:
  - Sequential reasoning (not just parallel association)
  - Goal maintenance across multiple timesteps
  - Contextual prediction (A predicts B only in context X)
```

### Layer 5: Consciousness (Global Workspace / Attention)

```markdown
Not all concepts enter global workspace.

Selection pressure for consciousness:
  - High unresolved tension (contradiction pressure)
  - Novel patterns
  - Goal-relevant concepts
  
Cconscious acts as the integration layer for cross-domain reasoning.
```

### Layer 6: Sleep (Reconciliation Engine)

```markdown
Sleep triggers when accumulated_pressure > threshold.

Four stages (all with controlled perturbation radius):

Stage 1 — Topology Analysis:
  - Find high-pressure zones (contradiction clusters)
  - Only these zones are candidates for perturbation
  - Identify unstable prediction edges

Stage 2 — Pattern Compression:
  - Find frequently co-activated concept clusters
  - These represent learned patterns worth keeping stable
  - Strengthen internal edges within clusters

Stage 3 — Contradiction Resolution:
  - For each active contradiction: find weakest edge in chain
  - Attempt to rewire (not delete) — redirect prediction
  - If resolution creates new contradictions → abort this resolution
  
Stage 4 — Integration:
  - Merge any new consistent clusters
  - Update confidence scores based on post-sleep stability
  - If coherence drops below pre-sleep - threshold → rollback all changes
```

### Tier-0: Core Identity (Never Perturbed)

```markdown
These concepts/features are structurally protected from sleep:

1. Self-reference: the "I" concept
2. Core value anchors: survival, coherence, novelty
3. High-degree concepts (incoming_edges > threshold, e.g. 10+)
4. Recent consciousness entries (last 5 significant thoughts)

Rule: Any concept with structural_inertia > 2.0 survives sleep unchanged.
```

---

## LEARNING RULES (REFINED)

### Wake Phase

```markdown
Every timestep:
  1. Perception → Sparse activation (top-K concepts only)
  2. Active concepts → Hebbian co-activation updates
  3. Prediction edges → Success/failure tracking
  4. Successful prediction → strengthen edge
  5. Failed prediction → contradiction pressure
  6. Working memory → Update temporal bindings
  7. Pressure accumulation → slow pressure_buildup += contradiction × 0.1

Contradiction gating:
  - effective = error × salience × (1 - confidence)
  - High confidence → hard to contradict
  - Established beliefs are sticky
```

### Sleep Phase

```markdown
Trigger: accumulated_pressure > 0.2

Boundaries:
  - Only perturb concepts in high_pressure_zones (≤2 hops from contradiction)
  - Perturbation magnitude: max 0.05 per edge
  - 4-stage process with abort-on-instability

Coherence check:
  - If post-sleep coherence < pre-sleep coherence - 0.05 → rollback
  - Rollback: restore pre-sleep weights from snapshot
```

---

## METRICS

```markdown
Coherence: weighted_mean(concept.confidence × activation) over active concepts

Prediction accuracy: successful_predictions / (successful + failed)

Pressure: accumulated + fresh_components

Convergence: rate_of_change(coherence) approaching zero

Identity stability: coherence of core concepts across sleep cycles
```

---

## IMPLEMENTATION PRIORITY

```markdown
Phase 1 (now): Sparse attention gating
  - Add K-active concept bottleneck to existing prototype
  - Test: does system stabilize vs explode with 100 concepts?

Phase 2 (next): Confidence-gated contradiction
  - Add uncertainty × salience filtering
  - Test: do established beliefs resist noise?

Phase 3: Bounded sleep with rollback
  - Implement controlled perturbation + coherence check
  - Test: does sleep improve without destroying useful attractors?

Phase 4: Tier-0 identity protection
  - Mark protected concepts
  - Test: does organism maintain identity across 1000+ sleep cycles?

Phase 5: Hebbian plasticity refinement
  - Implement full Hebbian update with bounded learning rate
  - Test: does system learn useful predictions from sequential inputs?

Phase 6: Working memory temporal bindings
  - Add goal-context mechanism
  - Test: can system maintain goal across 50+ timesteps with interference?
```

---

## WHAT THIS ARCHITECTURE IS NOT

- Not an LLM — no transformer, no token prediction
- Not symbolic AI — no logic rules, no truth tables
- Not a reward-based RL — no external reward signal
- Not a neural network — no gradient descent, no backprop

## WHAT THIS ARCHITECTURE IS

- A pressure-driven self-organizing system
- Where learning emerges from prediction failures
- Where memory IS the model (semantic weights)
- Where sleep is thermodynamic necessity
- Where identity is structural, not programmed

---

*"Cognition as pressure-driven self-organization."*

This is the center of gravity. Everything else is implementation detail.