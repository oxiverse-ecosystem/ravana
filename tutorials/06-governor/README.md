# Tutorial 06: Governor Basics

**Part of a 7-tutorial progression. Layer 3 of 3: cognitive internals.**

## What you'll learn

- Import GRACE governor modules directly (no chat engine needed)
- Understand the **20-phase GRACE architecture** (A–P) and how it structures cognition
- Instantiate and configure each module: VAD emotion, identity, meaning,
  dual-process, global workspace, and metacognition
- See how the chat engine uses these modules internally

---

## Run it

```bash
python tutorials/06-governor/run.py
```

---

## Deep dive: the GRACE cognitive architecture

**GRACE** = Governance, Reflection, Adaptation, Constraint, Exploration

It's a 20-phase cognitive architecture (phases A through P) that implements
pressure-driven self-organization. Each phase addresses a specific cognitive
function, building on the phases before it.

### Phase map

```
A: Governor Regulation     ← Core constraint satisfaction
B: Adaptation              ← Policy tweaking
C: Strategy Learning       ← Mode selection
D: Intent & Planning       ← Goal generation
E: Non-Stationary Env      ← Changing world
F: Predictive World Model  ← Anticipation
F.5: Belief Reasoning      ← Multi-hypothesis
G: Active Epistemology     ← Information seeking
G.5: Surgical Probes       ← Hypothesis testing
H-I: [reserved]
J: Hypothesis Generation   ← Creative exploration
J.1: Occam Layer           ← Simplicity bias
K: VAD Emotion             ← Affect dynamics
K.5: Empathy               ← Theory of mind
L: Sleep & Dreams          ← Consolidation
L.5: Dual-Process          ← Fast/slow reasoning
M: Meaning                 ← Intrinsic motivation
N: Global Workspace        ← Conscious broadcast
O: Human Memory            ← Episodic/semantic
P: Dialogue                ← Conversation
```

The chat engine (`CognitiveChatEngine`) uses Phases K-P (the "higher" cognitive
functions). The full governor (Phases A-J.1) is designed for standalone research
in the k0 agent framework.

### How these modules connect

```
                 ┌─────────────────┐
                 │  VAD Emotion    │◄──── User input affects mood
                 │  (Phase K)      │
                 └────────┬────────┘
                          │ emotion modulates cognition
                          ▼
                 ┌─────────────────┐
                 │  Dual-Process   │◄──── Decides fast vs slow reasoning
                 │  (Phase L.5)    │
                 └────────┬────────┘
                          │ route decision
                          ▼
              ┌──────────────────────┐
              │   Global Workspace    │◄──── Attention + broadcast
              │   (Phase N)           │
              └────────┬─────────────┘
                       │ broadcasts to all modules
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │Identity  │ │ Meaning  │ │ MetaCog  │
   │(Phase A) │ │(Phase M) │ │(Phase J)│
   └──────────┘ └──────────┘ └──────────┘
   stabilizes   drives      monitors
   self-concept  curiosity   own reasoning
```

---

## Module by module deep dive

### 1. VAD Emotion Engine (`VADEmotionEngine`)

**File:** `ravana-v2/src/ravana_grace/core/emotion.py`

**What it does:** Models affect as a 3D continuous space using differential equations.

#### The 3D VAD space

| Dimension | Range | Meaning | Example states |
|-----------|-------|---------|----------------|
| **Valence** | -1 to +1 | Pleasantness | +0.8 = happy, -0.8 = sad |
| **Arousal** | 0 to 1 | Alertness | 0.9 = excited, 0.1 = sleepy |
| **Dominance** | 0 to 1 | Control | 0.8 = confident, 0.2 = submissive |

#### The differential equations

Emotion evolves continuously via Euler integration:

```
dV/dt = η_v × (stimulus_v - V) - λ_v × V
dA/dt = η_a × (stimulus_a + 0.3 × uncertainty - A) - λ_a × (A - baseline)
dD/dt = η_d × (stimulus_d - D) - λ_d × D
```

Where:
- `η` (eta): learning rate — how fast emotion responds to stimuli
- `λ` (lambda): decay rate — how fast emotion returns to baseline
- `stimulus`: the emotional impact of the current event
- `uncertainty`: amplifies arousal (anxiety effect)

#### Reappraisal regulation

The engine supports **reappraisal** — reframing a stimulus to change its
emotional impact (a cognitive emotion regulation strategy):

```python
emotion.update(stimulus_valence=-0.5, reappraisal_reframe="opportunity")
# reappraisal adds +0.3 to valence, shifting it to -0.2
```

This is implemented as a lookup table of reframe strategies:
```python
reframe_valence_shift = {
    "opportunity": 0.3,   # "this is a chance to learn"
    "learning": 0.2,      # "what can I learn from this?"
    "threat": -0.2,       # "this is dangerous"
    "loss": -0.3,         # "this is a permanent loss"
}
```

#### How the chat engine uses it

After every user turn, the engine:
1. Detects user emotion from text (sentiment analysis via GloVe)
2. Updates its own VAD state via emotional contagion (`mirror_engine`)
3. Uses VAD to modulate response: high arousal → more diverse associations,
   positive valence → more confident answers

### 2. Identity Engine (`IdentityEngine`)

**File:** `ravana-v2/src/ravana_grace/core/identity.py`

**What it does:** Maintains a self-concept that prevents catastrophic forgetting.
Identity is the computational analog of the **vmPFC self-model** (D'Argembeau 2013).

#### Dynamics

```
I_t = I_{t-1} + ΔI_governor + bonus - penalty
```

Where:
- `ΔI_governor`: governor-approved identity change
- `bonus`: +0.08 per successful resolution (streak-multiplied up to 2×)
- `penalty`: -0.08 per failed prediction (fixed, not canceled by governor floor)
- `recovery_bias`: growth boost when I < 0.5
- `stability_damping`: shrink delta when I > 0.85

#### The failure penalty mechanism

One of the most carefully designed parts of the system. When the engine makes
a wrong prediction, identity takes a -0.08 hit. The penalty is applied BEFORE
the governor's floor constraint, so a recovery boost can't cancel it. This design:

1. **Ensures failures always hurt** — the penalty is non-negotiable
2. **Prevents catastrophic collapse** — the governor floors provide a safety net
3. **Creates hysteresis** — it's harder to rebuild identity after failure than
   to maintain it after success

#### How the chat engine uses identity

Identity strength modulates:
- **Confidence in answers** — higher identity → more assertive language
- **Curiosity** — higher identity → more exploration (safe to try new things)
- **Learning rate** — higher identity → slower change (resists new knowledge
  that contradicts core beliefs)

### 3. Meaning Engine (`MeaningEngine`)

**File:** `ravana-v2/src/ravana_grace/core/meaning.py`

**What it does:** Computes meaning as "costly coherence gain" — the system
finds actions meaningful when they reduce dissonance at the cost of effort.

#### Meaning formula

```
M = w₁ × (-ΔD_future) + w₂ × (Δidentity_coherence) + w₃ × (Δpredictive_power)
    × (1 + κ × effort_cost)
```

Where:
- `ΔD_future`: expected reduction in dissonance (negative delta = good)
- `Δidentity_coherence`: improvement in self-consistency
- `Δpredictive_power`: improvement in prediction accuracy
- `effort_cost`: how much cognitive effort was expended (0-1)
- `κ` (kappa): effort scaling factor
- The effort multiplier amplifies meaning: hard-won insights are more meaningful

#### Meaning-staking

The engine can **stake meaning** on beliefs:

```python
meaning.stake_meaning("gravity_pulls", 0.5)
```

If the belief is later falsified, the staked meaning is lost (-0.25 penalty).
This creates an **identity cost for holding false beliefs** — a form of
epistemic integrity.

#### How the chat engine uses meaning

Meaning drives **curiosity**: the engine pursues actions with high expected
meaning gain. This is how the system decides what to learn — not from a
fixed curriculum, but from expected dissonance reduction.

### 4. Dual-Process Controller (`DualProcessController`)

**File:** `ravana-v2/src/ravana_grace/core/dual_process.py`

**What it does:** Routes cognition between System 1 (fast/intuitive) and
System 2 (slow/deliberative), plus a RECOLLECT route for exact graph retrieval.

#### Three routes

| Route | Speed | Cost | When used |
|-------|-------|------|-----------|
| `SYSTEM1_FAST` | ~5 ms | Low | High confidence, low novelty, familiar topics |
| `RECOLLECT` | ~10 ms | Medium | HRR near-tie + graph has exact edge |
| `SYSTEM2_SLOW` | ~100 ms | High | Low confidence, high novelty, high stakes |

#### Decision criteria

```python
if confidence < S2_confidence_threshold:
    → engage System 2 (need careful reasoning)
if novelty > S2_novelty_threshold:
    → engage System 2 (unfamiliar situation)
if stakes > S2_stakes_threshold:
    → engage System 2 (important decision)
if too_many_consecutive_S2:
    → force System 1 (prevent cognitive burnout)
```

#### The RECOLLECT route (Yonelinas dual-process)

Standard dual-process theory (Kahneman) has only S1/S2. RAVANA adds a third
route — **recollection** (Yonelinas 2002) — which is exact graph retrieval
triggered by the ACC conflict monitor (Botvinick 2001):

```python
# When HRR is uncertain (top-1 vs top-2 gap < 0.06) AND graph has exact edge:
conflict_monitor(top1_conf, top2_conf, graph_has_edge=True)
→ RECOLLECT route: use graph edge instead of HRR composition
```

#### How the chat engine uses dual-process

Every `process_turn()` call runs through the dual-process controller to decide
whether to answer quickly (System 1: graph walk + decoder) or engage deeper
reasoning (System 2: belief reasoning, hypothetical simulation).

### 5. Global Workspace (`GlobalWorkspace`)

**File:** `ravana-v2/src/ravana_grace/core/global_workspace.py`

**What it does:** Implements the **Global Workspace Theory** of consciousness
(Baars 1988, 2002) — modules compete for access to a limited-capacity workspace;
the winner broadcasts to all modules.

#### Competition mechanism

1. **Bid submission**: modules submit content with urgency scores
   ```python
   gw.submit_bid(source="emotion", payload={"feeling": "anger"}, urgency=0.8)
   ```
2. **Competition**: bids compete via noisy tournament (adds Gaussian noise to
   prevent deterministic selection)
3. **Broadcast**: the winning bid is added to the workspace buffer and made
   available to all modules
4. **Buffer decay**: old broadcasts decay in influence via exponential decay

#### Config

```python
GWConfig(
    capacity=7,                # 7 ± 2 items (Miller's law)
    broadcast_threshold=0.3,   # minimum urgency to broadcast
    decay_rate=0.1,            # how fast broadcasts fade
    competition_noise=0.05,    # stochasticity in selection
)
```

The capacity of 7 items is a direct reference to Miller's (1956) "magic number
seven" for working memory capacity.

#### How the chat engine uses the global workspace

The GW serves as the **sole communication channel** between modules. Emotion
broadcasts to the decoder (affecting tone). The decoder broadcasts to the
realizer (affecting word choice). Beliefs broadcast to identity (affecting
confidence). Modules don't call each other directly — they communicate through
the workspace.

### 6. MetaCognition (`MetaCognition`)

**File:** `ravana-v2/src/ravana_grace/core/meta_cognition.py`

**What it does:** Thinks about thinking. Monitors reasoning quality, detects
bias, calibrates confidence, and decides when to abstain.

#### Components

| Component | What it does |
|-----------|-------------|
| `ReasoningQualityTracker` | Tracks accuracy, consistency, and coherence of past reasoning |
| `ConfidenceCalibrator` | Adjusts confidence scores to match actual accuracy (calibration curve) |
| `BiasDetector` | Detects confirmation bias, availability bias, anchoring |
| `EpistemicMode` | Controls exploration vs exploitation |

#### Confidence calibration

Raw confidence from the decoder is poorly calibrated (it tends to be overconfident
or underconfident). The `ConfidenceCalibrator` adjusts it using a running window
of (confidence, accuracy) pairs:

```python
# If the model says 0.8 confident but is only right 60% of the time,
# the calibrator maps 0.8 → 0.6
```

#### Epistemic modes

| Mode | Behavior |
|------|----------|
| `cautious` | High abstention threshold, hedged answers |
| `exploratory` | Low curiosity threshold, diverse associations |
| `confident` | Low abstention threshold, assertive answers |
| `recovery` | Post-failure mode, conservative |
| `curious` | High exploration, high novelty-seeking |

#### How the chat engine uses metacognition

Before every response, the metacognition layer:
1. Checks if confidence is high enough to answer
2. If confidence is too low, returns an honest abstention ("I don't know")
3. After the response, logs accuracy for calibration improvement

---

## Expected output (annotated)

```
=== GRACE Governor Modules Demo ===

  VAD Emotion:      V=0.02  A=0.30  D=0.50
  └── After stimulus valence=0.6, arousal=0.3, dominance=0.7
  └── State is near-baseline due to rapid decay (λ=0.1)

  Identity:         strength=0.44
  └── Started at 0.25, gained 0.19 from resolution_delta + success bonus

  Meaning:          raw=0.250  effective=0.312
  └── raw meaning: 0.250 (dissonance reduction + identity gain + predictive power)
  └── effective: 0.312 (raw × 1.25 effort multiplier for effort=0.5)

  Dual-Process:     route=system1_fast
  └── With confidence=0.3 and novelty=0.6, S2 was triggered... but hysteresis
      kept it S1 (system1_hysteresis=0.1, random roll favored S1)

  Global Workspace: winner=belief, buffer_size=1
  └── "belief" module won with urgency 0.7 (bid on trust concept)
  └── Buffer now has 1 item (the winning broadcast)

  MetaCognition:    epistemic_modes=['cautious', 'exploratory', 'recovery', 'confident']
  └── First 4 epistemic modes from the EpistemicMode enum

  [OK] All 6 GRACE modules instantiated successfully
```

---

## Key source files reference

| Module | File (relative to repo root) |
|--------|------------------------------|
| VAD Emotion | `ravana-v2/src/ravana_grace/core/emotion.py` |
| Identity | `ravana-v2/src/ravana_grace/core/identity.py` |
| Meaning | `ravana-v2/src/ravana_grace/core/meaning.py` |
| Dual-Process | `ravana-v2/src/ravana_grace/core/dual_process.py` |
| Global Workspace | `ravana-v2/src/ravana_grace/core/global_workspace.py` |
| MetaCognition | `ravana-v2/src/ravana_grace/core/meta_cognition.py` |
| Governor (full pipeline) | `ravana-v2/src/ravana_grace/core/governor.py` |
| GRACE module index | `ravana-v2/src/ravana_grace/core/__init__.py` |

---

## Design philosophy notes

1. **Emotion is not a label — it's a differential equation.** VAD emotion
   is a continuous dynamical system, not a set of discrete categories. This
   allows smooth transitions and context-sensitive responses.
2. **Identity is a structural self-concept, not a scalar.** The identity engine
   tracks momentum, stability, and trend — not just a single strength value.
   Catastrophic forgetting is prevented by this structural inertia.
3. **Consciousness is a workspace, not a stream.** The Global Workspace Theory
   frames consciousness as a competitive broadcast system, not a continuous
   stream of experience.
4. **Metacognition is the abstention mechanism.** The system knows when it
   doesn't know. This is the computational foundation of honest AI.

---

## Next tutorial

[**Tutorial 07: RLMv2**](../07-rlm/) — the final tutorial: explore the relation
learner model that decomposes sentences into triples.
