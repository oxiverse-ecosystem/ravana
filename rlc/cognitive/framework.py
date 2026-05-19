"""
RLC — Cognitive Framework API

The top-level user interface that wires together:
- ravana/ (ConceptGraph, PropagationEngine, pressure, plasticity)
- ravana-v2/core/ (Governor, Identity, Emotion, Sleep, Meaning, GlobalWorkspace)

Usage:
    from rlc.cognitive import CognitiveFramework

    framework = CognitiveFramework()
    state = framework.initialize()

    for episode, (input_vec, target_vec) in enumerate(data):
        concepts = framework.perceive(state, input_vec)
        predictions = framework.predict(state, concepts)
        state = framework.learn(state, predictions, target_vec, episode)
        if episode % 100 == 0:
            state = framework.sleep(state)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time

# ravana/ ML framework
from ravana.graph import ConceptGraph
from ravana.propagation import PropagationEngine
from ravana.plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity
from ravana.pressure import PressureAccumulator

# ravana-v2/ cognitive core (imported via rlc.cognitive)
from core.governor import Governor, GovernorConfig
from core.identity import IdentityEngine
from core.resolution import ResolutionEngine
from core.state import StateManager, CognitiveState
from core.emotion import VADEmotionEngine, VADConfig, VADState
from core.sleep import SleepConsolidation, SleepConfig
from core.meaning import MeaningEngine, MeaningConfig
from core.dual_process import DualProcessController, DualProcessConfig
from core.global_workspace import GlobalWorkspace, GWConfig
from core.human_memory import HumanMemoryEngine, HumanMemoryConfig


@dataclass
class FrameworkConfig:
    """Configuration for the Cognitive Framework."""
    # Concept graph
    concept_dim: int = 64
    max_concepts: int = 10000
    k_active: int = 5

    # Cognitive modules (optional configs)
    governor_config: Optional[GovernorConfig] = None
    emotion_config: Optional[VADConfig] = None
    sleep_config: Optional[SleepConfig] = None
    meaning_config: Optional[MeaningConfig] = None
    dual_process_config: Optional[DualProcessConfig] = None
    gw_config: Optional[GWConfig] = None
    human_memory_config: Optional[HumanMemoryConfig] = None

    # Learning
    hebbian_lr: float = 0.03
    anti_hebbian_lr: float = 0.02
    propagation_steps: int = 3
    propagation_decay: float = 0.5

    # Identity
    initial_identity: float = 0.5


@dataclass
class FrameworkState:
    """
    Immutable snapshot of the full cognitive system.

    Returned by every framework method that modifies state.
    """
    # Core metrics
    dissonance: float = 0.5
    identity: float = 0.5
    wisdom: float = 0.0
    meaning: float = 0.0
    cycle: int = 0
    episode: int = 0

    # Emotion
    vad: Optional[VADState] = None
    emotional_label: str = "neutral/relaxed"

    # Pressure
    total_pressure: float = 0.0

    # Processing
    mode: str = "normal"
    processing_route: str = "system1_fast"

    # Sleep
    sleep_cycles: int = 0

    # GW
    gw_broadcast_source: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        """Serializable snapshot."""
        return {
            "dissonance": self.dissonance,
            "identity": self.identity,
            "wisdom": self.wisdom,
            "meaning": self.meaning,
            "cycle": self.cycle,
            "episode": self.episode,
            "emotional_label": self.emotional_label,
            "total_pressure": self.total_pressure,
            "mode": self.mode,
            "processing_route": self.processing_route,
            "sleep_cycles": self.sleep_cycles,
        }


class CognitiveFramework:
    """
    Top-level API for the RAVANA cognitive system.

    Wires together the ML framework (ConceptGraph, pressure, plasticity)
    with the cognitive core (Governor, Emotion, Sleep, Meaning, GW).

    The framework operates in cycles:
    1. perceive() — input → concept activation
    2. predict() — Hebbian activation spread → predictions
    3. learn() — pressure-based governor update
    4. sleep() — 4-stage consolidation
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        self._initialized = False

    def initialize(self) -> FrameworkState:
        """
        Create initial cognitive state, wire all modules.

        Returns:
            Initial FrameworkState
        """
        c = self.config

        # === ML Framework ===
        self.graph = ConceptGraph(dim=c.concept_dim, max_nodes=c.max_concepts)
        self.propagation = PropagationEngine(self.graph)
        self.hebbian = HebbianPlasticity(self.graph, lr=c.hebbian_lr)
        self.anti_hebbian = AntiHebbianPlasticity(self.graph, lr=c.anti_hebbian_lr)
        self.structural = StructuralPlasticity(self.graph)
        self.graph_pressure = PressureAccumulator(self.graph)

        # === Cognitive Core ===
        self.governor = Governor(c.governor_config)
        self.identity = IdentityEngine(initial_strength=c.initial_identity)
        self.resolution = ResolutionEngine()

        # Optional engines
        self.emotion_engine = VADEmotionEngine(c.emotion_config) if c.emotion_config is not None else VADEmotionEngine()
        self.sleep_engine = SleepConsolidation(c.sleep_config) if c.sleep_config is not None else SleepConsolidation()
        self.meaning_engine = MeaningEngine(c.meaning_config) if c.meaning_config is not None else MeaningEngine()
        self.dual_process_engine = DualProcessController(c.dual_process_config) if c.dual_process_config is not None else DualProcessController()
        self.gw_engine = GlobalWorkspace(c.gw_config) if c.gw_config is not None else GlobalWorkspace()
        self.human_memory_engine = HumanMemoryEngine(c.human_memory_config) if c.human_memory_config is not None else HumanMemoryEngine()

        # === State Manager (orchestrator) ===
        self.state_manager = StateManager(
            governor=self.governor,
            resolution_engine=self.resolution,
            identity_engine=self.identity,
            emotion_engine=self.emotion_engine,
            sleep_engine=self.sleep_engine,
            dual_process=self.dual_process_engine,
            meaning_engine=self.meaning_engine,
            global_workspace=self.gw_engine,
            human_memory=self.human_memory_engine,
        )

        self._initialized = True
        self._step_count = 0

        return self._build_framework_state()

    def perceive(self, state: FrameworkState, input_vec: np.ndarray) -> List[int]:
        """
        Map input to active concepts via similarity + activation spreading.

        Args:
            state: Current cognitive state (unused internally, kept for API consistency)
            input_vec: Input vector (concept_dim,)

        Returns:
            List of active concept node IDs
        """
        self._ensure_initialized()

        # Bind input to graph — find similar concepts and activate them
        active_nids = self.graph.bind_input(input_vec, k=self.config.k_active)

        # Spread activation through edges
        self.graph.spread_activation(
            steps=self.config.propagation_steps,
            k_active=self.config.k_active + 2,  # allow slight expansion
            decay=self.config.propagation_decay,
        )

        # Get the most active nodes after spreading
        active_nodes = sorted(
            self.graph.nodes.values(),
            key=lambda n: n.activation,
            reverse=True,
        )[:self.config.k_active + 2]

        return [n.id for n in active_nodes if n.activation > 0.1]

    def predict(self, state: FrameworkState, active_concepts: List[int]) -> np.ndarray:
        """
        Hebbian activation spread → concept predictions → vector output.

        Args:
            state: Current cognitive state
            active_concepts: Concept IDs from perceive()

        Returns:
            Predicted vector (concept_dim,) — weighted average of predicted concept vectors
        """
        self._ensure_initialized()

        if not active_concepts:
            return np.zeros(self.config.concept_dim)

        # Get predictions via edge traversal
        predicted_nids = self.propagation.get_prediction(
            active_concepts,
            top_k=self.config.k_active,
        )

        if not predicted_nids:
            # Fallback: use active concepts
            predicted_nids = active_concepts[:3]

        # Build prediction vector from predicted concepts
        return self.propagation.get_activation_vector(predicted_nids)

    def learn(
        self,
        state: FrameworkState,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        episode: int = 0,
        difficulty: float = 0.5,
        effort: float = 0.0,
    ) -> FrameworkState:
        """
        Pressure-based update: governor regulates, identity stabilizes,
        emotion tags, meaning evaluates.

        Args:
            state: Current cognitive state
            predictions: Predicted vector from predict()
            outcomes: Actual outcome vector
            episode: Current episode number
            difficulty: Task difficulty (0-1)
            effort: Cognitive effort invested (0-1)

        Returns:
            New immutable FrameworkState
        """
        self._ensure_initialized()

        # Compute prediction error
        error_vec = outcomes - predictions
        error_magnitude = float(np.linalg.norm(error_vec))
        error_norm = min(1.0, error_magnitude / (np.linalg.norm(outcomes) + 1e-8))

        # Determine correctness (low error = correct)
        correctness = error_norm < 0.3

        # === Graph-level learning ===
        # Apply prediction error to graph
        if not correctness:
            # Find which concepts were predicted vs actual
            predicted_nids = self.propagation.get_prediction(
                list(self.graph.nodes.keys())[:self.config.k_active],
                top_k=3,
            )
            actual_nids = self.graph.bind_input(outcomes, k=3)

            # Strengthen correct edges, weaken incorrect ones
            for pred_nid in predicted_nids:
                for actual_nid in actual_nids:
                    if pred_nid != actual_nid:
                        self.hebbian.update(pred_nid, actual_nid)

            # Apply pressure from error
            self.graph_pressure.accumulate_semantic(error_norm, salience=0.5)

        # === Cognitive cycle via StateManager ===
        step_result = self.state_manager.step(
            correctness=correctness,
            difficulty=difficulty,
            novelty=error_norm,  # high error = high novelty
            stakes=0.0,
            effort=effort,
        )

        self._step_count += 1

        return self._build_framework_state()

    def sleep(self, state: FrameworkState) -> FrameworkState:
        """
        4-stage consolidation: topology analysis, compression,
        contradiction resolution, integration.

        Args:
            state: Current cognitive state

        Returns:
            New FrameworkState after consolidation
        """
        self._ensure_initialized()

        # === Graph-level sleep ===
        # Reconcile contradiction hotspots
        reconciled = self.graph.reconcile_contradictions()

        # Structural plasticity: prune weak edges, form new co-activation edges
        pruned, formed = self.structural.step()

        # Normalize edge weights
        self._normalize_outgoing_weights(budget=3.0)

        # === Cognitive-level sleep (with graph for abstraction compression) ===
        if self.sleep_engine_available():
            sleep_record = self.state_manager.sleep.execute_sleep_cycle(
                episode=self._step_count,
                state_snapshot=self.state_manager.state.snapshot(),
                episodic_memories=self.state_manager.memory.episodic.traces,
                emotion_engine=self.emotion_engine,
                coherence_fn=lambda s: 1.0 - s.get("dissonance", 0.5),
                graph=self.graph,
            )

        # Decay graph pressure
        self.graph_pressure.decay(rate=0.3)

        # Human memory: decay + consolidation
        self.human_memory_engine.apply_decay()
        self.human_memory_engine.consolidate()

        return self._build_framework_state()

    def infer(self, state: FrameworkState, input_vec: np.ndarray) -> Dict[str, Any]:
        """
        Forward-only inference (no state change).

        Args:
            state: Current cognitive state
            input_vec: Input vector

        Returns:
            {concepts, confidences, coherence, dissonance, predictions}
        """
        self._ensure_initialized()

        # Save activation state
        saved_activations = {nid: n.activation for nid, n in self.graph.nodes.items()}

        # Perceive
        active_nids = self.graph.bind_input(input_vec, k=self.config.k_active)
        self.graph.spread_activation(
            steps=self.config.propagation_steps,
            k_active=self.config.k_active,
            decay=self.config.propagation_decay,
        )

        # Predict
        predicted_nids = self.propagation.get_prediction(active_nids, top_k=3)
        pred_vector = self.propagation.get_activation_vector(predicted_nids)

        # Coherence
        coherence = self.propagation.measure_coherence(active_nids)

        # Confidences
        confidences = [
            self.graph.nodes[nid].confidence
            for nid in active_nids
            if nid in self.graph.nodes
        ]

        # Restore activation state
        for nid, act in saved_activations.items():
            if nid in self.graph.nodes:
                self.graph.nodes[nid].activation = act

        return {
            "concepts": active_nids,
            "predicted_concepts": predicted_nids,
            "predictions": pred_vector,
            "confidences": confidences,
            "coherence": coherence,
            "dissonance": state.dissonance,
        }

    def query(self, state: FrameworkState, concept_id: int) -> Dict[str, Any]:
        """
        Semantic memory retrieval. Returns graph neighborhood.

        Args:
            state: Current cognitive state
            concept_id: ID of the concept to query

        Returns:
            {concept, neighbors, edges, pressure}
        """
        self._ensure_initialized()

        node = self.graph.get_node(concept_id)
        if node is None:
            return {"error": f"Concept {concept_id} not found"}

        # Get neighbors
        neighbors = []
        edges = []
        for (src, tgt), edge in self.graph.edges.items():
            if src == concept_id:
                target_node = self.graph.get_node(tgt)
                if target_node:
                    neighbors.append({
                        "id": tgt,
                        "label": target_node.label,
                        "activation": target_node.activation,
                        "confidence": target_node.confidence,
                    })
                    edges.append({
                        "source": src,
                        "target": tgt,
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                    })
            elif tgt == concept_id:
                source_node = self.graph.get_node(src)
                if source_node:
                    neighbors.append({
                        "id": src,
                        "label": source_node.label,
                        "activation": source_node.activation,
                        "confidence": source_node.confidence,
                    })

        return {
            "concept": {
                "id": concept_id,
                "label": node.label,
                "activation": node.activation,
                "salience": node.salience,
                "pressure": node.pressure,
                "stability": node.stability,
                "confidence": node.confidence,
                "level": node.level,
                "parent": node.parent,
                "children": list(node.children),
                "abstraction_degree": node.abstraction_degree,
            },
            "neighbors": neighbors,
            "edges": edges,
            "neighbor_count": len(neighbors),
        }

    def diagnose(self, state: FrameworkState) -> Dict[str, Any]:
        """
        Full cognitive dashboard.

        Returns:
            Comprehensive system status
        """
        self._ensure_initialized()

        diag = {
            "state": state.snapshot(),
            "graph": {
                "node_count": len(self.graph.nodes),
                "edge_count": len(self.graph.edges),
                "total_pressure": self.graph.total_pressure,
                "contradiction_hotspots": len(self.graph.contradiction_hotspots),
            },
            "abstraction": self.graph.get_abstraction_stats(),
            "cognitive": self.state_manager.get_status(),
            "pressure": self.graph_pressure.report(),
            "step_count": self._step_count,
        }

        return diag

    # --- Helpers ---

    def _ensure_initialized(self):
        if not self._initialized:
            raise RuntimeError("Call framework.initialize() first")

    def _build_framework_state(self) -> FrameworkState:
        """Build immutable FrameworkState from current internal state."""
        sm = self.state_manager
        cs = sm.state

        vad_state = self.emotion_engine.state if self.emotion_engine else None
        label = self.emotion_engine.get_emotional_label() if self.emotion_engine else "neutral"

        gw_source = None
        if self.gw_engine and self.gw_engine._buffer:
            gw_source = self.gw_engine._buffer[0].source

        return FrameworkState(
            dissonance=cs.dissonance,
            identity=cs.identity,
            wisdom=cs.accumulated_wisdom,
            meaning=cs.accumulated_meaning,
            cycle=cs.cycle,
            episode=cs.episode,
            vad=vad_state,
            emotional_label=label,
            total_pressure=self.graph.total_pressure,
            mode=cs.last_update_reason,
            processing_route=cs.processing_route,
            sleep_cycles=cs.sleep_cycles_completed,
            gw_broadcast_source=gw_source,
        )

    def _normalize_outgoing_weights(self, budget: float = 3.0):
        """Cap total outgoing weight per node."""
        # Group edges by source
        source_weights: Dict[int, List[Tuple[int, float]]] = {}
        for (src, tgt), edge in self.graph.edges.items():
            if src not in source_weights:
                source_weights[src] = []
            source_weights[src].append((tgt, edge.weight))

        # Normalize each source's outgoing edges
        for src, targets in source_weights.items():
            total = sum(w for _, w in targets)
            if total > budget:
                scale = budget / total
                for tgt, _ in targets:
                    edge = self.graph.get_edge(src, tgt)
                    if edge:
                        edge.weight *= scale
                        if edge.weight < 0.005:
                            self.graph.remove_edge(src, tgt)

    def sleep_engine_available(self) -> bool:
        """Check if sleep engine is wired up."""
        return (
            self.state_manager.sleep is not None
            and hasattr(self.state_manager.sleep, 'execute_sleep_cycle')
        )
