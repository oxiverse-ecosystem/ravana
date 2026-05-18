# RAVANA v2 — GRACE Architecture

**Governance · Reflection · Adaptation · Constraint · Exploration**

A proto-homeostatic cognitive system with fully bounded dynamics.

---

## 🧭 Phase A Complete: Stable Physics

RAVANA v2 now operates as a **closed-loop regulated system** with four layers of control:

| Layer | Function | Mechanism |
|-------|----------|-----------|
| **Predictive** | Foresight | Look-ahead dampening based on horizon projection |
| **Boundary** | Soft resistance | Sigmoid pressure curve (air, not brick wall) |
| **Center** | Homeostasis | Anti-overshoot pull toward target dissonance |
| **Hard Stop** | Absolute limits | Constraints that cannot be breached |
| **Constitution** | Identity enforcement | Final clamp that overrides all downstream |

### The Identity Clamp — Keystone Innovation

The system now has **constitutional enforcement**: no behavioral layer can override the identity bounds. This closes the loophole where perfect regulation could be bypassed downstream.

> "Predictive dampening = foresight 👁️ | Constraints = law 🚧 | Identity clamp = constitution 📜"

### Current Metrics (Healthy Baseline)

- **Dissonance range**: 0.18–0.84 (healthy exploration, not hugging extremes)
- **Identity range**: 0.11–0.94 (plasticity without collapse)
- **Constraint hits**: 8/100 (curious but disciplined)
- **Mode switches**: 31 (responsive, not stuck in loops)

---

## 🧠 Phase B: Adaptive Intelligence

**Core insight**: Clamp events aren't failures — they're **teachable moments**.

Every time the constitution overrides the controller, the system learns *how not to need correction*.

### The Adaptation Engine

```
Raw Signals → Policy Tweak Layer → Governor → Clamp Check → Learn
```

**Design constraints**:
- **Lightweight**: ~100 lines core logic
- **Reversible**: Can disable instantly without breaking safety
- **Measurable**: Clear before/after comparison

### Learning Signal

```python
reward = exploration_bonus - clamp_penalty * correction_magnitude
```

- **Dual objective**: Explore healthy dissonance while avoiding constitutional violation
- **Pre-clamp avoidance**: Learn to stay away from boundary, not just bounce off it

### Key Metrics

| Metric | Interpretation |
|--------|----------------|
| `alignment_score` | How often upstream thinks correctly |
| `clamp_rate` | Friction with constitutional reality |
| `final_clamp_clamps` | 🚨 Canary metric (should trend to ~0) |
| `mean_tweak_magnitude` | How much adaptation is intervening |
| `mean_recent_reward` | Learning progress indicator |

---

## Clamp Diagnostics (Constitutional Court)

Every correction is now logged as a **ClampEvent**:

```python
episode, variable, before, after, correction, layer, reason
```

**Reports**:
- `get_clamp_report()` — Human-readable summary
- `results/clamp_events.json` — Full event log for analysis

**Phase B usage**: Feed clamp events to adaptation layer as negative reward.

---

## Architecture

```
core/
  governor.py        — Central regulation (first-class citizen)
  identity.py        — Identity dynamics with momentum
  resolution.py      — Conflict resolution engine
  state.py           — State manager (wires components)
  adaptation.py      — 🆕 Phase B: Learning from corrections

probes/
  constraint_stress.py     — Monitor constraint system
  exploration_pressure.py  — Track exploration drive
  learning_signal.py       — Extract learning indicators

training/
  pipeline.py        — Phase A training orchestration

experiments/runs/run_training.py      — Phase A entry point
experiments/phases/run_phase_b.py       — 🆕 Phase B entry point (adaptive)
```

---

## Quick Start

**Phase A** (stable physics):
```bash
python experiments/runs/run_training.py
```

**Phase B** (adaptive intelligence):
```bash
python experiments/phases/run_phase_b.py
```

---

## 🧪 Phase B.1: Experimental Validation

**Problem**: How do we know intelligence emerged, not cowardice?

**Solution**: Three-way comparison with intelligence dashboard.

### Experimental Protocol

```bash
# Run all three experiments
python experiments/runner.py

# Generate intelligence dashboard
python experiments/visualize.py
```

| Experiment | Purpose | What It Tests |
|------------|---------|---------------|
| **A: Baseline** | No adaptation | Raw clamp metrics |
| **B: Adaptive** | Normal learning rate | Does adaptation help? |
| **C: Stress** | High learning rate | Is it learning or drifting? |

### Key Upgrades from Your Analysis

**Nonlinear Penalty Shaping**:
```python
penalty = tanh(correction * 2.0)  # Saturate extreme cases
```
- Prevents overreaction to catastrophes
- Preserves sensitivity to subtle patterns

**Positive Signal**:
```python
reward += dissonance_utilization_bonus  # For healthy exploration
```
- Seeks good, not just avoids bad
- Prevents "minimally violating, not maximally intelligent"

### 🎯 The Intelligence Dashboard

```
┌─────────────────────────────────────────────┐
│  🧠 INTELLIGENCE SIGNATURE                  │
│                                             │
│        Exploration (std) ↑                  │
│                🧠 INTELLIGENT               │
│       (low clamp, high exploration)         │
│                ⚖️ CAUTIOUS                  │
│       (low clamp, low exploration)          │
│  ─────────────────────────────────→         │
│  Clamp Rate →                               │
└─────────────────────────────────────────────┘
```

**🚨 Red Flag: Cowardice Detection**

If you see:
- Clamp rate ↓
- Dissonance range ↓

**Verdict**: You built a coward, not intelligence.

**🧠 Success: Disciplined Curiosity**
- Clamp rate ↓
- Dissonance range → (or ↑)
- Final clamps → ~0

**Verdict**: Intelligence with constitutional awareness.

---

## License

MIT — Built for the RAVANA-AGI-Research initiative.
