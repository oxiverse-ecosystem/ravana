# RAVANA Architecture v2: Cognitive Mechanisms for System 2 Reasoning

**Version:** 2.0  
**Date:** 2026-06-27  
**Status:** Design Specification  

---

## Executive Summary

The current RAVANA system (v1) implements a spreading-activation concept graph with Hebbian plasticity, a Global Workspace, dual-process controller, and free-energy-driven language generation. However, systematic testing reveals **12 critical failure modes** that prevent genuine reasoning.

| # | Failure Mode | Root Cause |
|---|-------------|------------|
| 1 | No working memory - facts lost between turns | No episodic buffer; human_memory_engine has slots but no binding |
| 2 | No transitive/relational reasoning | No structure-mapping; graph edges are associative not relational |
| 3 | Concept drift - middle sentence jumps | No top-down PFC bias; spreading activation unconstrained |
| 4 | Morphology errors ("haves", "causes compare") | No agreement controller; cerebellar n-gram + BG gate missing |
| 5 | No physical/common-sense world model | No qualitative physics; object-file representation absent |
| 6 | No negation/difference operators | No inhibitory control; complement sets missing |
| 7 | No nested proposition handling | No meta-representation; recursive belief modeling absent |
| 8 | No lexical antonym knowledge | Contradiction map exists but no opponent-process representation |
| 9 | No comparison reasoning (abstract equality) | No relational operators (EQUAL, GREATER, LESS, OPPOSITE) |
| 10 | No pragmatic inference/theory of mind | No recursive belief modeling (TPJ/mPFC) |
| 11 | No analogical completion | No Structure Mapping Engine (Gentner) |
| 12 | No abstract reflection | No DMN simulation; meaning_engine lacks self-reference |

This document specifies **8 cognitive modules** addressing all 12 failures, with neural basis, computational mechanism, integration points, data structures, API signatures, and test cases.


---

## 1. Working Memory System (Addresses Failures #1, #7)

### Neural Basis
- **Phonological Loop**: Left Broca area (BA 44/45) + left supramarginal gyrus (BA 40) - verbal rehearsal
- **Visuospatial Sketchpad**: Right parietal cortex (BA 7/40) - spatial/mental imagery
- **Episodic Buffer**: Bilateral hippocampus + medial PFC - binds multimodal info into episodes
- **Central Executive**: Dorsolateral PFC (BA 9/46) - attention control, task-set maintenance, gating

### Computational Mechanism
Baddeleys Multicomponent Model with capacity 7+/-2 slots (teen = 5):
WorkingMemory = {
    phonological_loop: RingBuffer(capacity=5),      // verbal traces
    visuospatial_sketchpad: SpatialMap(capacity=5), // object files
    episodic_buffer: EpisodeBuffer(capacity=7),     // bound episodes
    central_executive: ExecutiveController()        // gating, updating
}

Key Mechanisms:
- PFC Gating via BasalGangliaGate: Direct/indirect pathway competition selects which representations enter WM
- Temporal Context Binding: Hippocampal time cells bind items to serial position (Howard & Kahana, 2002)
- Maintenance via Rehearsal: Phonological loop refreshes verbal items; sketchpad refreshes spatial
- Interference Resolution: Executive inhibits irrelevant items (right IFG)

### RAVANA Integration Point
Existing: HumanMemoryEngine.working_slots (capacity 7), PrefrontalWorkspace.capacity (5)
New Module: ravana/cognitive/working_memory.py


---

## 1. Working Memory System (Addresses Failures #1, #7)

### Neural Basis
- **Phonological Loop**: Left Broca area (BA 44/45) + left supramarginal gyrus (BA 40) - verbal rehearsal
- **Visuospatial Sketchpad**: Right parietal cortex (BA 7/40) - spatial/mental imagery
- **Episodic Buffer**: Bilateral hippocampus + medial PFC - binds multimodal info into episodes
- **Central Executive**: Dorsolateral PFC (BA 9/46) - attention control, task-set maintenance, gating

### Computational Mechanism
Baddeleys Multicomponent Model with capacity 7+/-2 slots (teen = 5):
WorkingMemory = {
    phonological_loop: RingBuffer(capacity=5),
    visuospatial_sketchpad: SpatialMap(capacity=5),
    episodic_buffer: EpisodeBuffer(capacity=7),
    central_executive: ExecutiveController()
}

Key Mechanisms:
- PFC Gating via BasalGangliaGate: Direct/indirect pathway competition selects which representations enter WM
- Temporal Context Binding: Hippocampal time cells bind items to serial position (Howard & Kahana, 2002)
- Maintenance via Rehearsal: Phonological loop refreshes verbal items; sketchpad refreshes spatial
- Interference Resolution: Executive inhibits irrelevant items (right IFG)

### RAVANA Integration Point
Existing: HumanMemoryEngine.working_slots (capacity 7), PrefrontalWorkspace.capacity (5)
New Module: ravana/cognitive/working_memory.py


---

## 2. Relational Reasoning Module (Addresses Failures #2, #8, #9, #11)

### Neural Basis
- **Hippocampus-PFC Circuit**: Relational binding (hippocampus) + rule application (dlPFC)
- **Rostrolateral PFC (BA 10)**: Analogical reasoning, structure mapping
- **Parietal Cortex (BA 7/40)**: Spatial/relational representation
- **Basal Ganglia**: Sequence control for multi-step inference

### Computational Mechanism
Structure Mapping Engine (SME) (Gentner, 1983; Forbus et al., 2017):
1. Representation: Entities + Attributes + Relations (predicate calculus)
2. Mapping: Find structural alignment between base and target
3. Candidate Inferences: Project unmapped base structure to target
4. Evaluation: Structural consistency score

Transitive Inference via ordered representations:
A > B, B > C  =>  A > C  (hippocampal-PFC ordered encoding)

Vector Symbolic Architecture (VSA) for relation binding:
- Binding: A * R * B (circular convolution)
- Unbinding: A * R * B approx B (inverse)
- Comparison: cosine similarity of bound vectors

Comparison Operators (first-class relations):
EQUAL(x, y)       := similarity(x, y) > theta
GREATER(x, y)     := magnitude(x) > magnitude(y)  (on dimension d)
LESS(x, y)        := GREATER(y, x)
OPPOSITE(x, y)    := x in antonym_set(y)
SIMILAR(x, y, d)  := projection_d(x) approx projection_d(y)

### RAVANA Integration Point
Existing: ConceptGraph edges have relation_type (causal, contrastive, semantic, etc.)
New Module: ravana/cognitive/relational_reasoning.py


---

## 3. Negation/Inhibition System (Addresses Failure #6)

### Neural Basis
- **Right Inferior Frontal Gyrus (rIFG, BA 44/45)**: Response inhibition, no-go signals
- **Pre-SMA**: Conflict monitoring, action suppression
- **Basal Ganglia Indirect Pathway**: Global inhibition via STN-GPe-SNr
- **Anterior Cingulate (ACC)**: Conflict detection recruits inhibition

### Computational Mechanism
Opponent-Process Representation (Huron, 2006):
Concept(x) = [excitatory_weights, inhibitory_weights]
Antonym(x) = inhibitory_weights activated as positive pattern

Complement Sets in Concept Graph:
- Each concept has explicit antonyms set (bidirectional)
- Negation = activation of antonym set + suppression of target
- NOT(x) = activate(antonyms(x)) + inhibit(x)

Inhibitory Control Gating:
if conflict_detected(prefrontal, anterior_cingulate):
    basal_ganglia.indirect_pathway.activate()

### RAVANA Integration Point
Existing: _contradiction_map in GraphEngine, BasalGangliaGate in language module
New Module: ravana/cognitive/negation_inhibition.py


---

## 4. Physical World Model / Intuitive Physics (Addresses Failure #5)

### Neural Basis
- **Posterior Parietal Cortex (BA 7/39/40)**: Object files, spatial representation
- **Premotor Cortex (BA 6)**: Action simulation, affordances
- **Cerebellum**: Forward models, prediction of physical dynamics
- **Ventral Visual Stream (LOC)**: Object identity, properties

### Computational Mechanism
Object-File Representation (Kahneman & Treisman, 1992):
ObjectFile = {
    id: obj_3,
    properties: {mass: 0.5, temperature: 273, state: solid, material: water, volume: 1.0},
    location: (x, y, z),
    affordances: [graspable, pourable, freezable],
    history: []
}

Qualitative Physics Engine (Forbus, 1984):
- Processes: Continuous changes (heating, flowing, melting)
- Quantities: Amount, temperature, pressure (ordinal scales)
- Causal Laws: heat(solid, >melting_point) -> liquid; contain(liquid, container) -> supported

Causal Mechanism Simulation (Pearl/Woodward):
do(X = x) -> simulate -> P(Y | do(X))

### RAVANA Integration Point
Existing: ConceptGraph nodes have level (0=concrete, 1=abstract), abstraction_degree
New Module: ravana/cognitive/world_model.py


---

## 5. Theory of Mind / Pragmatic Inference (Addresses Failure #10)

### Neural Basis
- **Temporoparietal Junction (TPJ, BA 39/40)**: Belief attribution, false belief tasks
- **Medial PFC (mPFC, BA 10/32)**: Self-referential, trait inference
- **Precuneus/PCC**: Episodic simulation, perspective taking
- **Superior Temporal Sulcus (STS)**: Biological motion, intention detection

### Computational Mechanism
Recursive Belief Modeling (level-k reasoning):
B^0_i = prior beliefs
B^1_i(phi) = P(phi | B^0_i)
B^2_i(phi) = P(phi | B^1_j)  // what I think you think
B^k_i(phi) = P(phi | B^{k-1}_j)  // k-level recursion

Gricean Maxims Implementation (soft constraints):
- Quantity: info(content) in [required, required + epsilon]
- Quality: belief(speaker, content) > 0.7
- Relation: relevance(content, context) > threshold
- Manner: complexity(content) < threshold

Scalar Implicature Computation:
some + not all -> not all  (via Horn scale <some, most, all>)
or + not and -> not both   (via scale <or, and>)

### RAVANA Integration Point
Existing: BeliefStore (multi-user), HumanMemoryEngine (episodic), GlobalWorkspace
New Module: ravana/cognitive/theory_of_mind.py


---

## 6. Morphological/Syntactic Control (Addresses Failure #4)

### Neural Basis
- **Cerebellum (Crus I/II)**: Internal models for sequence timing, n-gram prediction
- **Basal Ganglia (Putamen/Caudate)**: Gating of motor/cognitive sequences, agreement selection
- **Left IFG (BA 44/45)**: Syntactic unification, morphological computation
- **Left Posterior Temporal**: Lexical access, lemma retrieval

### Computational Mechanism
Cerebellar n-gram + Basal Ganglia Gating:
- Cerebellum learns n-gram statistics over verb-form sequences
- Basal Ganglia direct pathway: selects winning agreement form
- Indirect pathway: suppresses competing forms
- VerbLexicon integration: Each verb root carries agreement features [+/-sg, +/-pl, +/-1st, +/-2nd, +/-3rd]

Agreement Features:
VerbRoot {
    form: create,
    features: {
        3sg_present: creates,
        3pl_present: create,
        past: created,
        participle: created,
        gerund: creating
    }
}

Gating Rule:
IF subject_features == [3rd, sg] AND tense == present:
    BG_gate.select(creates)  # direct pathway
    BG_gate.suppress(create) # indirect pathway

### RAVANA Integration Point
Existing: VerbLexicon (morphemic seeds + Hebbian weights), CerebellarNgram, BasalGangliaGate
Enhancement: ravana/language/morphological_controller.py


---

## 7. Concept Drift Control (Addresses Failure #3)

### Neural Basis
- **Dorsolateral PFC (dlPFC, BA 9/46)**: Top-down bias, task-set maintenance
- **Anterior Cingulate (ACC)**: Conflict monitoring, detects drift
- **Locus Coeruleus (NE)**: Precision weighting (gain control)
- **Parietal Cortex**: Attention-gated spreading activation

### Computational Mechanism
Attention-Gated Spreading Activation:
activation_t+1(i) = sum_j W_ij * activation_t(j) * gate(i, task_set)
gate(i, task_set) = sigmoid(beta * similarity(concept_i, task_set))

PFC Top-Down Bias (Task-Set Maintenance):
task_vector = encode(task_description)
bias(i) = cosine(task_vector, concept_vector_i)
effective_activation(i) = activation(i) * (1 + lambda * bias(i))

Free Energy Precision Weighting (Friston, 2009):
precision = 1 / variance
high_precision -> narrow attentional focus (exploit)
low_precision -> broad exploration (explore)

Drift Detection:
drift_score = 1 - cosine(current_activation_profile, task_activation_profile)
if drift_score > threshold: trigger_PFC_reorient()

### RAVANA Integration Point
Existing: PrefrontalWorkspace (task-set via QTYPE_PRIMARY_RELATION), PropagationEngine, FreeEnergyAccumulator
New Module: ravana/cognitive/concept_drift_control.py


---

## 8. Abstract Reflection / Meta-Cognition (Addresses Failure #12)

### Neural Basis
- **Default Mode Network (DMN)**: mPFC, PCC/precuneus, angular gyrus - self-referential, autobiographical
- **Rostrolateral PFC (BA 10)**: Meta-reasoning, thinking about thinking
- **Anterior Insula**: Interoceptive awareness, uncertainty feeling
- **Hippocampus**: Episodic simulation, scene construction

### Computational Mechanism
DMN Simulation (Andrews-Hanna et al., 2014):
- Core DMN: Self-referential processing, autobiographical memory
- DMN-MTL subsystem: Episodic simulation, scene construction
- DMN-dMPFC subsystem: Mentalizing, social cognition

Self-Referential Processing:
SelfModel = {
    beliefs: BeliefStore,
    identity_narrative: List[Episode],
    epistemic_status: Dict[Proposition, Confidence],
    metacognitive_judgments: List[MetaJudgment]
}

MeaningEngine Integration: Abstraction via hierarchical compression:
Level 0: Concrete episodes
Level 1: Patterns across episodes (schemas)
Level 2: Abstract principles (wisdom)
Level 3: Meta-principles (values, identity)

Meta-Cognitive Judgments:
MetaJudgment = {
    target: Proposition,
    judgment_type: confidence | reliability | source | control,
    value: float,
    basis: internal | external | inferential
}

### RAVANA Integration Point
Existing: MetaCognition (bias detection, calibration), MeaningEngine (meaning accumulation), IdentityEngine, SleepConsolidation (abstraction compression)
New Module: ravana/cognitive/abstract_reflection.py

---

## Integration Architecture

### Dual-Process Controller (Enhanced)

```python
class DualProcessControllerV2:
    """
    System 1: Fast spreading activation + cerebellar n-gram + BG gating
    System 2: Slow relational reasoning + working memory + world model + ToM
    
    Routing decision based on:
    - Confidence (low -> System 2)
    - Novelty (high -> System 2)  
    - Stakes (high -> System 2)
    - Relational complexity (transitive, analogical, nested -> System 2)
    - Negation/inhibition needed -> System 2
    """
    
    def route(self, input: str, context: CognitiveContext) -> RouteDecision:
        if context.relational_complexity > 1:
            return Route.SYSTEM2_SLOW
        if context.requires_negation:
            return Route.SYSTEM2_SLOW
        if context.requires_world_simulation:
            return Route.SYSTEM2_SLOW
        if context.requires_tom:
            return Route.SYSTEM2_SLOW
        return self.base_controller.decide_route(...)
```

### Global Workspace (Enhanced)

```python
class GlobalWorkspaceV2:
    """
    Broadcasts to ALL modules:
    - WorkingMemoryEngine (maintains active trace)
    - RelationalReasoningEngine (computes relations)
    - WorldModelEngine (simulates physics)
    - TheoryOfMindEngine (models beliefs)
    - NegationInhibitionEngine (resolves conflict)
    - ConceptDriftController (monitors drift)
    - MorphologicalController (ensures agreement)
    - AbstractReflectionEngine (DMN mode)
    
    Capacity: 7+/-2 items (Miller)
    Competition: urgency * valence * precision_weight
    """
```

### Sleep Consolidation (Enhanced)

```python
class SleepConsolidationV2:
    """
    4-Stage + Memory-Graph Bridging:
    1. Topology Analysis: Graph communities, contradiction hotspots
    2. Compression: Abstraction hierarchy (Level 0->1->2->3)
    3. Contradiction Resolution: BeliefStore.reconcile()
    4. Integration: Episodic -> Semantic (hippocampal replay)
    
    NEW: Memory-Graph Bridge
    - HumanMemoryEngine.consolidate() -> ConceptGraph edges
    - WorkingMemory episodes -> hippocampal index
    - AbstractReflection principles -> MeaningEngine accumulation
    """
```

### Cognitive Framework V2 (Top-Level API)

```python
class CognitiveFrameworkV2:
    def __init__(self, config: FrameworkConfigV2):
        # Core (v1)
        self.graph = ConceptGraph(...)
        self.propagation = PropagationEngine(self.graph)
        self.hebbian = HebbianPlasticity(...)
        
        # NEW: Cognitive modules
        self.working_memory = WorkingMemoryEngine(capacity=5)
        self.relational_reasoning = RelationalReasoningEngine(self.graph, self.working_memory)
        self.negation_inhibition = NegationInhibitionEngine(self.graph, basal_ganglia)
        self.world_model = WorldModelEngine(self.graph)
        self.theory_of_mind = TheoryOfMindEngine(belief_store, self.working_memory)
        self.morphological_control = MorphologicalController(verb_lexicon, cerebellar, bg)
        self.concept_drift = ConceptDriftController(self.graph, propagation, pfc, free_energy)
        self.abstract_reflection = AbstractReflectionEngine(meaning, identity, sleep, memory, meta)
        
        # Integration
        self.dual_process = DualProcessControllerV2(...)
        self.global_workspace = GlobalWorkspaceV2(...)
        self.sleep = SleepConsolidationV2(...)
        
    def perceive(self, input_vec) -> List[int]:
        for nid in active_nids:
            self.working_memory.write(nid, "verbal", source="perception")
        return active_nids
    
    def reason(self, state, query: str) -> ReasoningResult:
        route = self.dual_process.route(query, context)
        if route == Route.SYSTEM2_SLOW:
            return self._system2_reason(query)
        return self._system1_reason(query)
    
    def _system2_reason(self, query: str) -> ReasoningResult:
        task = self.prefrontal_workspace.analyze_query(query)
        self.concept_drift.set_task_set(task)
        
        concepts = self.working_memory.gate(..., task)
        relations = self.relational_reasoning.chain_reason(...)
        
        if task.requires_physics:
            sim = self.world_model.simulate(...)
        if task.requires_tom:
            tom = self.theory_of_mind.predict_utterance_implicature(...)
        if task.requires_negation:
            self.negation_inhibition.apply_negation(...)
        
        response = self.surface_realizer.realize(..., 
            morphological_controller=self.morphological_control)
        
        if task.abstraction_level > 1:
            insight = self.abstract_reflection.reflect_on(...)
        
        return ReasoningResult(...)
```
