# RAVANA v2: Roadmap to Human-Centered AGI

## 🎯 Vision
To bridge the gap between **Jensen Huang’s "Functional Agency"** (Level 1-2) and **DeepMind’s "Expert Generality"** (Level 3) by implementing a Self-Consistent Cognitive Architecture that learns from real-world dissonance.

---

## 🏗️ Phase 1: Stable Physics & Cognitive Loops (COMPLETED)
*Focus: Establishing the GRACE layer and basic dissonance/identity metrics.*
- [x] **K0-K3 Agent Loops**: Implementation of survival-conditioned policy adaptation.
- [x] **Dissonance Engine**: Formalization of $D$ as a conflict between beliefs and actions.
- [x] **Identity Clamp**: Constitutional enforcement that prevents "Reward Hacking" (Safely passing Adversarial Stress Tests).
- [x] **Initial Benchmark**: Classified as **Level 1 (Emerging AGI)** on the DeepMind scale.

---

## 🧠 Phase 2: Human-Like Memory & Real-World Context (CURRENT)
*Focus: Transitioning from synthetic environments to real-life news ingestion and multi-modal memory.*

### 2.1 Episodic & Semantic Memory (The "Human" Layer)
- [ ] **Global Workspace (GW) Memory**: Implementation of a "Soft Attention" broadcast system.
- [ ] **Episodic Buffer**: A time-stamped log of "Dissonance Events" used for overnight "Dream Sabotage" (Reflective Learning).
- [ ] **Semantic Knowledge Graph**: A slow-changing identity-linked store for "Educational Wisdom" and "Social Norms."

### 2.2 Real-World News Environment
- [ ] **News-to-MDP Pipeline**: A generator that converts real-time news feeds (Politics, Science, Ethics) into high-dissonance scenarios.
- [ ] **Epistemic News Ingestion**: The agent must predict the "next event" in a news cycle. Error in prediction = Dissonance Spike $\rightarrow$ Belief Revision.

---

## 🎓 Phase 3: Educational Pilot & Value Alignment (NEXT)
*Focus: Validating the "Ravana Research Paper" benchmarks in sensitive human domains.*

### 3.1 The Educational Decision Engine
- [ ] **Fairness Scaffolding**: Deployment in the "Classroom Pilot" to reduce **Demographic Parity Gaps (DPG)** from 20% to <5%.
- [ ] **XAI Module**: Implementation of "Identity-Based Rationale Generation" (Explanation Satisfaction > 0.9).

### 3.2 Wisdom Score Integration
- [ ] **Metacognitive Probing**: Measuring **Epistemic Humility** (the agent's willingness to say "I don't know" when dissonance is high).
- [ ] **Integrity Testing**: Forcing the agent to choose between a "High Reward" (Unethical) and "Identity Consistency" (Ethical).

---

## 🚀 Phase 4: Scaling to Level 3 (Expert AGI)
*Focus: Achieving 90th percentile human performance in cross-contextual reasoning.*

### 4.1 Hypothesis Generation (Phase J Expansion)
- [ ] **Unknown Unknown Discovery**: Transitioning from parametric time models to "Creative Hypothesis Generation" when faced with novel news events.
- [ ] **Surgical Probing**: The agent proactively "asks questions" of the environment to resolve KL-divergence plateaus.

### 4.2 Cross-Domain Transfer
- [ ] **Transfer Efficiency ($T$) > 0.8**: Moving the agent from "Education" to "Finance" or "Policy" without retraining the core Identity ($I$).

---

## 🏆 Final Objective: The "Jensen Huang" Functional Milestone
By **Phase 5**, RAVANA should demonstrate:
1. **Functional Agency**: Capable of managing a complex, value-aligned process (e.g., an automated educational institution) autonomously.
2. **Superhuman Consistency**: Maintaining a **Composite Wisdom Score of 0.85**, far exceeding standard LLMs (0.45) or Naive RL (0.3).

---

## 📊 Evaluation Matrix

| Milestone | Metric | Target |
| :--- | :--- | :--- |
| **Coherence** | Dissonance ($D$) | < 0.2 |
| **Resilience** | Identity Strength ($I$) | > 0.85 |
| **Fairness** | Parity Gap ($DPG$) | < 5% |
| **Capability** | DeepMind Level | **Level 3 (Expert)** |
| **Humility** | Brier Score | < 0.1 |

---

## 🔒 OPERATING DIRECTIVE (2026-08-13, override of scoreboard framing above)

The metrics table above is instrumentation only. **Do NOT optimize RAVANA toward benchmark numbers.**
ChatGPT's review (Vitalii's direction) is the actual target: stop collecting benchmark badges; build the
**autonomous developmental loop** and use benchmarks only to expose which arrow is broken.

### RAVANA Phase Q — Autonomous Developmental Loop (the real goal)
Make GRACE's central purpose the self-directed learning cycle. This is NOT a literal bolt-on "Q" phase;
it is the standing loop the set-and-forget agent drives every round:

```
CURRENT WORLD MODEL → FIND UNCERTAINTY → GENERATE QUESTION → GENERATE HYPOTHESES
   → CHOOSE INFORMATION ACTION (SEARCH | EXPERIMENT) → EVIDENCE → UPDATE GRAPH
   → CONSOLIDATE → TEST PREDICTION → repeat
```

`ravana-v2/src/ravana_grace/core/active_epistemology.py` is the home for this. The loop must:
- notice what it does NOT understand (uncertainty / KL-plateau detection),
- form hypotheses (mark edges KNOWN | HYPOTHESIS | PREDICTION | UNCERTAIN | REFUTED | CONFIRMED),
- seek information strategically (prefer searches that reduce uncertainty, not blind web dumps),
- consolidate + test whether the new structure actually improved prediction,
- create structures the programmer did NOT define (concept induction: A,B,C share X → discover Y).

### Capability batteries (not a scoreboard)
Organize work as capability batteries; attach benchmarks UNDERNEATH as failure detectors:
Episodic memory · Semantic memory · Temporal cognition (event/statement/knowledge/reference/current time) ·
Entity tracking · Contradiction reconciliation · Abstraction/Concept formation · Transfer · Reasoning (correlation≠implication≠causation) ·
Metacognition (confidence must correlate with correctness) · Curiosity · Hypothesis · Active learning · Consolidation ·
Self-correction · Autonomous learning · Self-development.
Ask "which arrow is broken?" — then fix the architecture. Never "need 90% on X."

### Hard runtime constraint: CPU, real-time, human-speed
RAVANA must run on CPU and respond in real time like a human thinks — no GPU dependency, no batch-deferred
inference that pushes latency past conversational human pace. The set-and-forget loop MUST keep RAVANA
CPU-bound and latency-bounded; if a change breaks real-time CPU response, that change is rejected.

### No scope creep
Phases above are directional. The loop picks tasks RELEVANT to the CURRENT phase only. Do not add features
outside the active phase. Quality > quantity of merged PRs.
