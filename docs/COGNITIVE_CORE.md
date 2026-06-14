# RAVANA Cognitive Core (`ravana-v2/core/`)

> **Complete reference for the GRACE cognitive architecture** — 27 modules implementing Governance, Reflection, Adaptation, Constraint, Exploration (Phases A–P).

---

## Table of Contents

1. [Overview](#overview)
2. [Phase Architecture](#phase-architecture)
3. [Phase A: Core Regulation](#phase-a-core-regulation)
4. [Phase B: Adaptation](#phase-b-adaptation)
5. [Phase C: Strategy Learning](#phase-c-strategy-learning)
6. [Phase D: Intent & Planning](#phase-d-intent--planning)
7. [Phase E: Non-Stationary Environment](#phase-e-non-stationary-environment)
8. [Phase F: Predictive World Model](#phase-f-predictive-world-model)
9. [Phase F.5: Belief Reasoning](#phase-f5-belief-reasoning)
10. [Phase G: Active Epistemology](#phase-g-active-epistemology)
11. [Phase J: Hypothesis & Occam Layer](#phase-j-hypothesis--occam-layer)
12. [Phase K: Emotion & Empathy](#phase-k-emotion--empathy)
13. [Phase L: Sleep & Dual Process](#phase-l-sleep--dual-process)
14. [Phase M: Meaning Engine](#phase-m-meaning-engine)
15. [Phase N: Global Workspace](#phase-n-global-workspace)
16. [Phase O: Human Memory](#phase-o-human-memory)
17. [Phase P: Dialogue System](#phase-p-dialogue-system)
18. [State Management](#state-management)
19. [Integration Patterns](#integration-patterns)

---

## Overview

`ravana-v2/core/` implements the **GRACE architecture** — a self-stabilizing, self-expanding epistemic system with active experimentation and explicit Occam discipline.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GRACE COGNITIVE CORE                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase A: GOVERNANCE                                               │  │
│  │  Governor (hard constraints) → Identity → Resolution → StateMgr  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase B: ADAPTATION                                               │  │
│  │  PolicyTweakLayer ← ClampEvents from Governor                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase C: STRATEGY → Phase D: INTENT → Phase D.5: PLANNING        │  │
│  │  4 exploration modes → Dynamic objectives → Micro-planner        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase E: ENV → Phase F: WORLD MODEL → Phase F.5: BELIEF          │  │
│  │  Non-stationary → Neural world model → Competing hypotheses      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase G: ACTIVE EPISTEMOLOGY → Phase G.5: SURGICAL PROBES        │  │
│  │  Value of Information → Targeted interventions                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase J: HYPOTHESIS → Phase J.1: OCCAM LAYER                     │  │
│  │  Hypothesis generation → Complexity penalization                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase K: EMOTION → Phase K.5: EMPATHY                            │  │
│  │  VAD differential equations → Other-mind modeling                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase L: SLEEP → Phase L.5: DUAL PROCESS                         │  │
│  │  4-stage consolidation + dream sabotage → System 1/2 routing     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase M: MEANING → Phase N: GW → Phase O: MEMORY                 │  │
│  │  Intrinsic motivation → Consciousness bottleneck → Persistent    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Phase P: DIALOGUE                                                │  │
│  │  Context tracking → Conversational repair                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Import

```python
# All modules available from:
from core import (
    # Phase A
    Governor, GovernorConfig, RegulationMode, ClampDiagnostics,
    ResolutionEngine, ResolutionMemory,
    IdentityEngine, IdentityState,
    StateManager, CognitiveState,
    # Phase B
    PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig,
    # Phase C
    StrategyLayer, StrategyConfig, ExplorationMode, ModeSelection, BehavioralContext,
    StrategyLearningLayer, ModeOutcome, LearningConfig, StrategyWithLearning,
    # Phase D
    IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective,
    # Phase D.5
    MicroPlanner, PlanningConfig, SimulatedFuture,
    # Phase E
    NonStationaryEnvironment, EnvironmentConfig, HiddenDynamics, WorldState,
    # Phase F
    LearnedWorldModel, WorldModelConfig, PredictedState, AnomalyEvent, FalseWorldTester,
    # Phase F.5
    BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent,
    # Phase G
    ActiveEpistemology, VoIConfig, InformationGainMethod, HypothesisDrivenActionSelector,
    # Phase G.5
    SurgicalProbeSelector, SurgicalProbeConfig, ProbeType, ProbeExperiment, SurgicalProbing,
    # Phase J
    HypothesisGenerator, GenerationConfig, HypothesisType, GeneratedHypothesis,
    # Phase J.1
    OccamLayer, OccamConfig, HypothesisScore, DisciplinedBeliefSystem,
    # Phase K
    VADEmotionEngine, VADConfig, VADState,
    # Phase K.5
    EmpathyEngine, EmpathyConfig, OtherMind,
    # Phase L
    SleepConsolidation, SleepConfig, SleepStage, SleepRecord, DreamPerturbationType,
    # Phase L.5
    DualProcessController, DualProcessConfig, ProcessingRoute, RouteDecision,
    # Phase M
    MeaningEngine, MeaningConfig, MeaningRecord,
    # Phase N
    GlobalWorkspace, GWConfig, GWContent,
    # Phase O
    HumanMemoryEngine, HumanMemoryConfig, HumanMemoryRecord,
    # Phase P
    DialogueContext, ActiveSubgraph, Triple, DialogueState,
    ConversationalRepair, RepairEvent, CorrectionType,
)
```

---

## Phase A: Core Regulation

### Governor — Central Control

```python
from core.governor import Governor, GovernorConfig, RegulationMode, CognitiveSignals

config = GovernorConfig(
    # Hard constraints
    max_dissonance=0.95,
    min_dissonance=0.15,
    target_dissonance=0.30,
    center_target=0.50,
    max_identity=0.95,
    soft_limit=0.70,
    boundary_k=12.0,
    min_pressure=0.2,
    min_identity=0.10,
    
    # Regulation
    dissonance_target=0.35,
    identity_target=0.85,
    exploration_threshold=0.25,
    resolution_threshold=0.60,
    use_smoothed_dissonance=True,
    smoothing_alpha=0.2,
    
    # Recovery
    recovery_boost=0.15,
    crisis_threshold=0.90,
    
    # Plateau detection
    plateau_window=50,
    plateau_tolerance=0.02,
    
    # Diagnostics
    clamp_alert_threshold=0.05,
)

gov = Governor(config)

# Every state change MUST pass through governor
signals = CognitiveSignals(
    dissonance_delta=0.1,
    identity_delta=0.05,
    exploration_drive=0.2,
    resolution_potential=0.3,
    trend=0.02,              # dD/dt
    predicted_dissonance=0.35,
    horizon=3,
    source="state_step",
    confidence=0.8,
)

output = gov.regulate(
    current_dissonance=0.3,
    current_identity=0.7,
    signals=signals,
    episode=100,
)

# Output
output.dissonance_delta    # regulated
output.identity_delta      # regulated
output.mode                # RegulationMode
output.dampened            # bool
output.boosted             # bool
output.capped              # bool
output.reason              # str

# Diagnostics
report = gov.get_clamp_report()
metrics = gov.get_clamp_metrics()
health = gov.get_health_metrics()
```

**Four Regulatory Layers (in order):**

1. **Hard Constraints** — Absolute ceilings/floors (non-negotiable)
2. **Predictive Dampening** — Look-ahead: `dd *= 1 - max(0, predicted - target) * k`
3. **Boundary Pressure** — Air resistance: `dd *= (1 - pressure)^k` near limits
4. **Center-Seeking Force** — Homeostatic pull: `dd += -k * (current - target)`

### Identity Engine

```python
from core.identity import IdentityEngine, IdentityState

identity = IdentityEngine(initial_strength=0.5)

# Step with outcome
new_state = identity.step(
    correctness=True,      # prediction correct?
    difficulty=0.5,        # task difficulty [0,1]
    stakes=0.0,            # consequence magnitude
)

# State
new_state.strength         # current identity strength [0,1]
new_state.momentum         # momentum (recovery bias)
new_state.history          # list of past strengths
new_state.wisdom           # cumulative resolution credit
```

**Momentum-based recovery**: After failure, identity doesn't immediately drop — momentum provides recovery bias.

### Resolution Engine

```python
from core.resolution import ResolutionEngine, ResolutionMemory

resolution = ResolutionEngine()

# Called by StateManager on outcome
result = resolution.process(
    correctness=True,
    difficulty=0.5,
    identity_strength=0.7,
    dissonance=0.3,
)

result.wisdom_delta        # wisdom increment
result.identity_delta      # identity increment (resolution bonus)
result.partial_credit      # continuous credit [0,1]
```

### State Manager — Orchestrator

```python
from core.state import StateManager, CognitiveState

# Wires all modules together
mgr = StateManager(
    governor=gov,
    resolution_engine=resolution,
    identity_engine=identity,
    emotion_engine=emotion,
    sleep_engine=sleep,
    dual_process=dual_process,
    meaning_engine=meaning,
    global_workspace=gw,
    human_memory=memory,
)

# Single step with all modules
step_result = mgr.step(
    correctness=True,
    difficulty=0.5,
    novelty=0.2,
    stakes=0.0,
    effort=0.3,
)

# State snapshot
cognitive_state = mgr.state
cognitive_state.dissonance
cognitive_state.identity
cognitive_state.wisdom
cognitive_state.valence, cognitive_state.arousal, cognitive_state.dominance
cognitive_state.mode
cognitive_state.processing_route
cognitive_state.sleep_cycles
```

---

## Phase B: Adaptation

```python
from core.adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig

config = AdaptationConfig(
    learning_rate=0.01,
    adaptation_window=100,
    min_clamp_rate_for_adapt=0.1,
)

policy = PolicyTweakLayer(config)
bridge = AdaptiveGovernorBridge(governor, policy)

# Learns from clamp events (when governor overrides)
# Adapts governor parameters to reduce future clamps
```

---

## Phase C: Strategy Learning

```python
from core.strategy import StrategyLayer, StrategyConfig, ExplorationMode, ModeSelection, BehavioralContext
from core.strategy_learning import StrategyLearningLayer, ModeOutcome, LearningConfig, StrategyWithLearning

# Base strategy (4 modes)
config = StrategyConfig(
    mode_persistence=3,        # min episodes per mode
    exploration_bonus=0.1,
    recovery_threshold=0.2,
)

strategy = StrategyLayer(config)

context = BehavioralContext(
    dissonance=0.3,
    identity=0.7,
    uncertainty=0.4,
    recent_performance=0.8,
)

selection = strategy.select_mode(context)
selection.mode      # ExplorationMode.AGGRESSIVE / SAFE / STABILIZE / RECOVER
selection.confidence

# Meta-learning over outcomes
learning = StrategyLearningLayer(LearningConfig())
learning.record_outcome(
    ModeOutcome(
        mode=ExplorationMode.AGGRESSIVE,
        dissonance_before=0.3,
        dissonance_after=0.25,
        identity_change=0.02,
        success=True,
    )
)
```

**Exploration Modes:**

| Mode | Behavior | When Active |
|------|----------|-------------|
| `AGGRESSIVE` | High learning rate, broad exploration | Low dissonance, high identity |
| `SAFE` | Conservative, exploit known | Moderate dissonance |
| `STABILIZE` | Minimize change, consolidate | High dissonance, resolving |
| `RECOVER` | Emergency restoration | Crisis (dissonance > 0.9) |

---

## Phase D: Intent & Planning

### Intent Engine

```python
from core.intent import IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective

config = IntentConfig(
    objective_horizon=10,
    intent_decay=0.05,
    min_intent_strength=0.1,
)

intent = IntentEngine(config)

# Dynamic objectives from outcomes
objective = intent.form_objective(
    current_state=cognitive_state,
    recent_outcomes=outcomes,
)

objective.description      # str
objective.priority         # float [0,1]
objective.horizon          # episodes
objective.success_criteria # callable
```

### Micro-Planner

```python
from core.planning import MicroPlanner, PlanningConfig, SimulatedFuture

config = PlanningConfig(
    max_depth=3,
    branching_factor=4,
    simulation_budget=20,
)

planner = MicroPlanner(config)

future = planner.plan(
    current_state=cognitive_state,
    objective=objective,
    world_model=world_model,
)

future.actions        # list of planned actions
future.expected_value # estimated outcome
future.risk           # uncertainty
```

---

## Phase E: Non-Stationary Environment

```python
from core.environment import NonStationaryEnvironment, EnvironmentConfig, HiddenDynamics, WorldState

config = EnvironmentConfig(
    boundary_shift_rate=0.01,
    noise_drift=0.001,
    goal_flip_probability=0.001,
)

env = NonStationaryEnvironment(config)

# Step environment
world_state = env.step(action)

world_state.observation       # np.ndarray
world_state.reward            # float
world_state.done              # bool
world_state.hidden_dynamics   # HiddenDynamics (true state)

# Hidden dynamics (non-stationary)
hidden = world_state.hidden_dynamics
hidden.current_boundary
hidden.current_noise
hidden.current_goal
```

---

## Phase F: Predictive World Model

```python
from core.predictive_world import LearnedWorldModel, WorldModelConfig, PredictedState, AnomalyEvent, FalseWorldTester

config = WorldModelConfig(
    latent_dim=64,
    hidden_dim=128,
    prediction_horizon=5,
    surprise_threshold=0.3,
)

world_model = LearnedWorldModel(config)

# Predict
predicted = world_model.predict(state, action)

predicted.next_state      # np.ndarray
predicted.reward          # float
predicted.uncertainty     # float
predicted.surprise        # float (prediction error)

# Update from experience
world_model.update(state, action, next_state, reward)

# Anomaly detection
anomaly = world_model.detect_anomaly(observation, predicted)

# False world testing (counterfactual)
tester = FalseWorldTester(world_model)
counterfactual = tester.test(state, alt_action)
```

---

## Phase F.5: Belief Reasoning

```python
from core.belief_reasoner import BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent

config = BeliefConfig(
    max_hypotheses=10,
    confidence_decay=0.01,
    evidence_threshold=0.1,
)

reasoner = BeliefReasoner(config)

# Add hypothesis
hyp = Hypothesis(
    id="h1",
    content="heat causes expansion",
    confidence=0.7,
    uncertainty=0.3,
)
reasoner.add_hypothesis(hyp)

# Update with evidence
event = EvidenceEvent(
    hypothesis_id="h1",
    evidence="observed expansion after heat",
    strength=0.8,
    valence=1.0,  # supporting
)
reasoner.update(event)

# Get current beliefs
beliefs = reasoner.get_beliefs()
for b in beliefs:
    print(f"{b.content}: conf={b.confidence:.2f}, unc={b.uncertainty:.2f}")
```

---

## Phase G: Active Epistemology

```python
from core.active_epistemology import ActiveEpistemology, VoIConfig, InformationGainMethod, HypothesisDrivenActionSelector

config = VoIConfig(
    method=InformationGainMethod.MUTUAL_INFORMATION,
    exploration_weight=0.3,
    computation_budget=100,
)

epistemology = ActiveEpistemology(config)

# Compute Value of Information for actions
voi_scores = epistemology.compute_voi(
    hypotheses=beliefs,
    possible_actions=actions,
    world_model=world_model,
)

# Select action maximizing VoI
selector = HypothesisDrivenActionSelector()
action = selector.select(voi_scores, beliefs)
```

---

## Phase G.5: Surgical Probes

```python
from core.surgical_probes import SurgicalProbeSelector, SurgicalProbeConfig, ProbeType, ProbeExperiment, SurgicalProbing

config = SurgicalProbeConfig(
    max_probes_per_cycle=3,
    probe_budget=10,
    significance_threshold=0.05,
)

prober = SurgicalProbing(config)

# Design targeted intervention
experiment = prober.design_probe(
    target_hypothesis="causal link between heat and expansion",
    probe_type=ProbeType.INTERVENTIONAL,
    context=current_state,
)

# Execute and record
result = prober.execute(experiment, environment)
```

---

## Phase J: Hypothesis & Occam Layer

### Hypothesis Generator

```python
from core.hypothesis_generation import HypothesisGenerator, GenerationConfig, HypothesisType, GeneratedHypothesis

config = GenerationConfig(
    max_hypotheses=20,
    novelty_threshold=0.3,
    coherence_requirement=0.5,
)

generator = HypothesisGenerator(config)

hypotheses = generator.generate(
    anomalies=anomalies,
    current_beliefs=beliefs,
    world_model=world_model,
)

for h in hypotheses:
    print(f"{h.type}: {h.content} (novelty={h.novelty:.2f})")
```

### Occam Layer (Hypothesis Discipline)

```python
from core.occam_layer import OccamLayer, OccamConfig, HypothesisScore, DisciplinedBeliefSystem

config = OccamConfig(
    complexity_penalty=0.1,
    max_hypothesis_complexity=5,
    parsimony_weight=0.3,
)

occam = OccamLayer(config)

# Score hypotheses by accuracy - complexity penalty
scores = occam.score_hypotheses(hypotheses, evidence)

for score in scores:
    print(f"{score.hypothesis.content}: acc={score.accuracy:.2f}, "
          f"complexity={score.complexity:.2f}, final={score.final_score:.2f}")

# Disciplined belief system
belief_system = DisciplinedBeliefSystem(occam)
belief_system.update(hypotheses, evidence)
```

---

## Phase K: Emotion & Empathy

### VAD Emotion Engine

```python
from core.emotion import VADEmotionEngine, VADConfig, VADState

config = VADConfig(
    # Differential equation parameters
    gamma_valence=0.1,
    gamma_arousal=0.15,
    gamma_dominance=0.08,
    beta_valence=0.3,
    beta_arousal=0.4,
    beta_dominance=0.2,
    
    # Modulation thresholds
    high_arousal_threshold=0.7,
    positive_valence_threshold=0.5,
    high_dominance_threshold=0.6,
)

emotion = VADEmotionEngine(config)

# Step with prediction error
vad_state = emotion.step(
    prediction_error=0.3,
    correctness=True,
    control=0.7,
    helplessness=0.1,
)

# State
vad_state.valence      # [-1, 1] negative → positive
vad_state.arousal      # [0, 1] calm → excited
vad_state.dominance    # [0, 1] submissive → dominant
vad_state.emotional_label  # e.g., "excited/confident"

# Modulation of inference
modulation = emotion.get_inference_modulation(vad_state)
# Returns dict with exploration_bonus, prediction_confidence, concept_activation_scale
```

### Empathy Engine

```python
from core.empathy import EmpathyEngine, EmpathyConfig, OtherMind

config = EmpathyConfig(
    projection_strength=0.5,
    self_other_overlap=0.3,
    compassion_threshold=0.6,
)

empathy = EmpathyEngine(config)

# Model another mind
other = OtherMind(
    beliefs={"heat_causes_expansion": 0.8},
    emotional_state=VADState(valence=-0.3, arousal=0.7, dominance=0.4),
    goals=["understand_physics"],
)

# Simulate their reaction
simulated = empathy.simulate(
    self_state=cognitive_state,
    other=other,
    situation="heat applied to metal",
)

simulated.predicted_reaction
simulated.empathic_concern
simulated.perspective_taking
```

---

## Phase L: Sleep & Dual Process

### Sleep Consolidation

```python
from core.sleep import SleepConsolidation, SleepConfig, SleepStage, SleepRecord, DreamPerturbationType

config = SleepConfig(
    pressure_threshold=0.2,
    min_pressure_for_sleep=0.05,
    topology_analysis_cycles=5,
    pattern_compression_cycles=8,
    contradiction_resolution_cycles=10,
    integration_cycles=5,
    counterfactual_rate=0.20,    # 20% reversal
    emotional_flip_rate=0.10,    # 10% valence flip
    failure_oversample_factor=1.5,
    max_perturbation_hops=2,
    max_edge_weight_change=0.05,
    abstraction_min_cluster_size=3,
    abstraction_max_cluster_size=8,
    abstraction_coactivation_threshold=0.5,
    abstraction_max_level=5,
    coherence_drop_threshold=0.05,
    tier_0_identifiers=["self_reference", "identity_core", "survival_pressure", "coherence_drive"],
)

sleep = SleepConsolidation(config)

# Accumulate pressure from cognitive events
sleep.accumulate_pressure(delta=0.1)

# Check trigger
if sleep.should_sleep():
    record = sleep.execute_sleep_cycle(
        episode=100,
        state_snapshot=state_snapshot,
        beliefs=beliefs,
        hypotheses=hypotheses,
        episodic_memories=memories,
        emotion_engine=emotion,
        coherence_fn=lambda s: 1.0 - s["dissonance"],
        graph=concept_graph,
    )

record.pre_coherence
record.post_coherence
record.perturbations_applied
record.rollback_occurred
record.sabotages_applied
record.pressure_before
record.pressure_after
record.details
```

**Sleep Stages:**

| Stage | Function |
|-------|----------|
| `TOPOLOGY_ANALYSIS` | Identify high-pressure zones (dissonance, belief instability) |
| `PATTERN_COMPRESSION` | Strengthen co-activation clusters in pressure zones |
| `ABSTRACTION_COMPRESSION` | Hierarchical merging of frequently co-activated concepts |
| `CONTRADICTION_RESOLUTION` | Rewire weakest edges in pressure zones |
| `MODEL_UPDATE` | Hippocampal replay → Hebbian strengthening, vector drift, pruning |
| `INTEGRATION` | Merge, stabilize, rollback if coherence drops >5% |

**Dream Sabotage (REM):**
- `COUNTERFACTUAL_REVERSAL` (20%): Reverse causal direction
- `EMOTIONAL_FLIP` (10%): Flip valence of memory
- `FAILURE_OVERSAMPLE` (1.5×): Replay failures more often
- `SYMBOLIC_RECOMBINATION`: Combine unrelated concepts

### Dual Process Controller

```python
from core.dual_process import DualProcessController, DualProcessConfig, ProcessingRoute, RouteDecision

config = DualProcessConfig(
    system1_threshold=0.3,      # dissonance below → fast
    system2_threshold=0.6,      # dissonance above → slow
    arbitration_window=5,       # episodes before re-eval
)

dual = DualProcessController(config)

decision = dual.route(
    dissonance=0.4,
    identity=0.7,
    novelty=0.5,
    stakes=0.2,
    time_pressure=0.1,
)

decision.route        # ProcessingRoute.SYSTEM1_FAST or SYSTEM2_SLOW
decision.confidence
decision.reasoning
```

---

## Phase M: Meaning Engine

```python
from core.meaning import MeaningEngine, MeaningConfig, MeaningRecord

config = MeaningConfig(
    w_dissonance=1.0,      # weight for -D
    w_identity=0.5,        # weight for I
    w_prediction=0.3,      # weight for prediction accuracy
    kappa_effort=0.2,      # effort multiplier
    meaning_decay=0.01,
)

meaning = MeaningEngine(config)

record = meaning.evaluate(
    dissonance=0.3,
    identity=0.7,
    prediction_accuracy=0.8,
    effort=0.4,
)

record.meaning_value     # M = w1(-D) + w2(I) + w3(pred) × (1 + κ×effort)
record.dissonance_component
record.identity_component
record.prediction_component
record.effort_component
```

---

## Phase N: Global Workspace

```python
from core.global_workspace import GlobalWorkspace, GWConfig, GWContent

config = GWConfig(
    capacity=4,              # max concurrent contents
    broadcast_threshold=0.6, # activation for broadcast
    competition_strength=0.5,
    decay_rate=0.1,
)

gw = GlobalWorkspace(config)

# Submit content for competition
content = GWContent(
    source="perception",
    activation=0.8,
    payload={"concepts": [1, 5, 10], "vector": np.randn(64)},
    metadata={"timestamp": 100, "modality": "text"},
)

gw.submit(content)

# Broadcast (winning content)
broadcast = gw.broadcast()
if broadcast:
    print(f"Conscious content: {broadcast.source}")
    # All modules receive broadcast
```

---

## Phase O: Human Memory Engine

```python
from core.human_memory import HumanMemoryEngine, HumanMemoryConfig, HumanMemoryRecord

config = HumanMemoryConfig(
    episodic_capacity=5000,
    semantic_capacity=10000,
    working_memory_capacity=7,
    ebbinghaus_half_life=3600,      # seconds
    interference_factor=0.1,
    consolidation_threshold=0.7,
    retrieval_k=10,
)

memory = HumanMemoryEngine(config)

# Store episodic memory
mem_id = memory.store(
    content="heat causes expansion",
    importance=0.8,
    tags="causal,physics",
    emotional_valence=0.2,
    context={"experiment": "thermal"},
)

# Recall
results = memory.recall(
    query="heat",
    limit=5,
    min_confidence=0.3,
)

for r in results:
    print(f"{r.content}: conf={r.confidence:.2f}, valence={r.emotional_valence:.2f}")

# Sleep operations
memory.sleep_replay(state_snapshot)
memory.apply_decay()          # Ebbinghaus forgetting
memory.consolidate()          # Episodic → Semantic

# Bridge to ConceptGraph
bridge_result = memory.bridge_to_graph(concept_graph, lr=0.02)
bridge_result["edges_created"]
bridge_result["edges_strengthened"]
```

**Memory Types:**

| Type | Capacity | Decay | Consolidation |
|------|----------|-------|---------------|
| Working | 7 items | Immediate | — |
| Episodic | 5000 | Ebbinghaus (half-life 1hr) | During sleep |
| Semantic | 10000 | Very slow | Continuous |

---

## Phase P: Dialogue System

```python
from core.dialogue_context import DialogueContext, ActiveSubgraph, Triple, DialogueState
from core.conversational_repair import ConversationalRepair, RepairEvent, CorrectionType

# Track conversation
dialogue = DialogueContext()

# Add utterance
triple = Triple(
    subject="user",
    relation="stated",
    object="heat causes expansion",
    confidence=0.9,
    source="user",
)
dialogue.add_triple(triple)

# Build active subgraph
subgraph = dialogue.build_subgraph(concept_graph, focus="heat")

# Conversational repair
repair = ConversationalRepair()

# Handle correction
event = repair.process_correction(
    user_input="no, heat causes melting not expansion",
    current_beliefs=beliefs,
    dialogue_context=dialogue,
)

event.correction_type        # CorrectionType.CONTRADICTION / REFINEMENT / EXTENSION
event.target_triple
event.proposed_correction
event.confidence
```

---

## State Management

### CognitiveState (Immutable Snapshot)

```python
from core.state import CognitiveState

state = CognitiveState(
    dissonance=0.3,
    identity=0.7,
    wisdom=0.15,
    cycle=100,
    episode=100,
    valence=0.1,
    arousal=0.3,
    dominance=0.6,
    mode="normal",
    processing_route="system1_fast",
    sleep_cycles=3,
)

# Serializable
snapshot = state.snapshot()
# {"dissonance": 0.3, "identity": 0.7, ...}
```

---

## Integration Patterns

### With ML Framework (CognitiveFramework)

```python
from ravana.cognitive import CognitiveFramework, FrameworkConfig

config = FrameworkConfig(
    concept_dim=64,
    max_concepts=10000,
    k_active=5,
    governor_config=GovernorConfig(),
    emotion_config=VADConfig(),
    sleep_config=SleepConfig(),
    meaning_config=MeaningConfig(),
    dual_process_config=DualProcessConfig(),
    gw_config=GWConfig(),
    human_memory_config=HumanMemoryConfig(),
    hebbian_lr=0.03,
    anti_hebbian_lr=0.02,
    propagation_steps=3,
    propagation_decay=0.5,
    initial_identity=0.5,
)

fw = CognitiveFramework(config)
state = fw.initialize()

# Full cognitive cycle
concepts = fw.perceive(state, input_vec)
predictions = fw.predict(state, concepts)
state = fw.learn(state, predictions, target_vec, episode)
if episode % 100 == 0:
    state = fw.sleep(state)

# Inference
result = fw.infer(state, test_vec)

# Query semantic memory
query_result = fw.query(state, concept_id)
```

### Standalone Module Usage

```python
# Use any module independently
gov = Governor(GovernorConfig())
identity = IdentityEngine(initial_strength=0.5)
emotion = VADEmotionEngine(VADConfig())
sleep = SleepConsolidation(SleepConfig())
memory = HumanMemoryEngine(HumanMemoryConfig())

# Wire manually
mgr = StateManager(
    governor=gov,
    identity_engine=identity,
    emotion_engine=emotion,
    sleep_engine=sleep,
    human_memory=memory,
    # ... others optional
)
```

---

## Configuration Reference

### FrameworkConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `concept_dim` | 64 | Concept vector dim |
| `max_concepts` | 10000 | Max concept nodes |
| `k_active` | 5 | Top-k active concepts |
| `hebbian_lr` | 0.03 | Hebbian learning rate |
| `anti_hebbian_lr` | 0.02 | Anti-Hebbian rate |
| `propagation_steps` | 3 | Spread iterations |
| `propagation_decay` | 0.5 | Per-step decay |
| `initial_identity` | 0.5 | Starting identity |

### Module Configs (all optional, have defaults)

- `GovernorConfig` — regulation bounds, smoothing, recovery
- `VADConfig` — emotion ODE parameters
- `SleepConfig` — sleep triggers, dream sabotage rates
- `MeaningConfig` — intrinsic motivation weights
- `DualProcessConfig` — System 1/2 thresholds
- `GWConfig` — workspace capacity, broadcast threshold
- `HumanMemoryConfig` — capacities, decay rates

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [ML Framework](ML_FRAMEWORK.md)
- [Unified Package](UNIFIED_PACKAGE.md)
- [Core Concepts](CONCEPTS.md)
- [API Reference](API_REFERENCE.md)