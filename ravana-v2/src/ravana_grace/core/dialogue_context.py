"""
RAVANA v2 — DIALOGUE CONTEXT MANAGEMENT
Working memory layer with salience decay and spreading activation integration.

PRINCIPLE: Dialogue is a time-indexed process where recent information
has higher salience but gracefully decays as the conversation progresses.

Subsystem 1 from the architectural plan:
- ActiveSubgraph: A live overlay on ConceptGraph tracking high-salience edges
- DialogueContext: Main interface managing user state, turn tracking, context
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from collections import defaultdict


# ─── Triple Type ─────────────────────────────────────────────────────────────

@dataclass
class Triple:
    """A (subject, relation, object) triple representing a parsed fact/belief."""
    subject: str
    relation: str
    relation_type: str  # 'causal', 'semantic', 'temporal', etc.
    object: str
    confidence: float = 0.5
    source_agent: str = 'system'
    epistemic_status: str = 'fact'  # 'fact', 'belief', 'experience', 'hypothesis'
    timestamp: float = 0.0

    def __hash__(self):
        return hash((self.subject, self.relation, self.object))


# ─── ActiveSubgraph ──────────────────────────────────────────────────────────

class ActiveSubgraph:
    """
    A live, temporary overlay on the ConceptGraph tracking only high-salience edges.

    Salience decays exponentially (half-life ~1 minute of dialogue).
    Edges below the activation threshold are pruned to free compute.

    This is NOT stored in the ConceptGraph itself — it's a separate, temporary overlay
    that keeps the global graph stable.
    """

    def __init__(
        self,
        decay_rate: float = 0.92,
        activation_threshold: float = 0.01,
        max_spreading_depth: int = 3,
        spreading_decay_per_hop: float = 0.7,
    ):
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.max_spreading_depth = max_spreading_depth
        self.spreading_decay_per_hop = spreading_decay_per_hop

        # Salience overlay: edge_key (subject, relation, object) -> salience score
        self._salience: Dict[Tuple[str, str, str], float] = {}

        # Turn timestamps: edge_key -> when last activated
        self._turn_timestamps: Dict[Tuple[str, str, str], float] = {}

        # Active concept nodes (by label/name, since we're at the semantic level)
        self._active_concepts: Dict[str, float] = {}  # concept_label -> activation

        # Turn counter
        self._turn_count: int = 0

    def inject(self, triples: List[Triple], salience: float = 1.0):
        """Inject parsed triples into the active subgraph with high salience."""
        now = time.time()
        for triple in triples:
            key = (triple.subject, triple.relation, triple.object)
            self._salience[key] = salience
            self._turn_timestamps[key] = now

            # Activate the subject and object concepts
            self._active_concepts[triple.subject] = self._active_concepts.get(
                triple.subject, 0.0
            ) + salience
            self._active_concepts[triple.object] = self._active_concepts.get(
                triple.object, 0.0
            ) + salience * 0.8  # slightly less for the object

        self._turn_count += 1

    def decay(self):
        """Apply exponential decay to all salience scores and prune below threshold."""
        to_remove = []
        for key, salience in self._salience.items():
            new_salience = salience * self.decay_rate
            if new_salience < self.activation_threshold:
                to_remove.append(key)
            else:
                self._salience[key] = new_salience

        # Prune sub-threshold edges
        for key in to_remove:
            del self._salience[key]
            self._turn_timestamps.pop(key, None)

        # Also decay concept activations
        to_remove_concepts = []
        for label, activation in self._active_concepts.items():
            new_activation = activation * self.decay_rate
            if new_activation < self.activation_threshold:
                to_remove_concepts.append(label)
            else:
                self._active_concepts[label] = new_activation

        for label in to_remove_concepts:
            del self._active_concepts[label]

    def propagate_activation(
        self,
        graph: Any,  # ConceptGraph from ravana_ml
        user_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run spreading activation using the salience-weighted active subgraph.

        This provides the "context activations" used by the generation pipeline.

        Args:
            graph: A ConceptGraph instance for finding concept nodes
            user_id: If provided, use agent-specific edge weights

        Returns:
            Dict mapping concept label -> activation score
        """
        # Start with current active concepts as seeds
        activations: Dict[str, float] = dict(self._active_concepts)

        # BFS spreading from most active concepts
        # Sort by activation descending
        sorted_concepts = sorted(
            activations.items(), key=lambda x: x[1], reverse=True
        )

        for concept_label, base_activation in sorted_concepts[:10]:
            # Find this concept in the graph
            concept_node = self._find_concept_in_graph(graph, concept_label)
            if concept_node is None:
                continue

            # BFS up to max_spreading_depth hops
            visited = {concept_label}
            frontier = [(concept_node, 1.0, 0)]  # (node, weight, depth)

            while frontier:
                current_node, current_weight, depth = frontier.pop(0)
                if depth >= self.max_spreading_depth:
                    continue

                # Get outgoing edges (agent-aware if user_id provided)
                outgoing = self._get_outgoing_edges(
                    graph, current_node, user_id
                )

                for neighbor_label, edge_weight in outgoing:
                    if neighbor_label in visited:
                        continue
                    visited.add(neighbor_label)

                    # Activation contribution = base * weight * decay_per_hop^depth
                    hop_weight = (
                        base_activation
                        * edge_weight
                        * (self.spreading_decay_per_hop ** depth)
                    )
                    activations[neighbor_label] = (
                        activations.get(neighbor_label, 0.0) + hop_weight
                    )

                    # Add to frontier for next hop
                    neighbor_node = self._find_concept_in_graph(
                        graph, neighbor_label
                    )
                    if neighbor_node is not None:
                        frontier.append(
                            (neighbor_node, edge_weight, depth + 1)
                        )

        return activations

    def get_active_edges(self) -> List[Tuple[str, str, str, float]]:
        """Get all active edges with their salience scores."""
        return [
            (subj, rel, obj, sal)
            for (subj, rel, obj), sal in self._salience.items()
        ]

    def get_active_concepts(self, threshold: float = 0.01) -> Dict[str, float]:
        """Get active concepts above threshold."""
        return {
            label: act
            for label, act in self._active_concepts.items()
            if act >= threshold
        }

    def reset(self):
        """Clear all salience and activation data (for sleep consolidation)."""
        self._salience.clear()
        self._turn_timestamps.clear()
        self._active_concepts.clear()
        self._turn_count = 0

    def _find_concept_in_graph(self, graph: Any, label: str) -> Any:
        """Find a concept node by label in the ConceptGraph."""
        if graph is None:
            return None
        for nid, node in graph.nodes.items():
            if node.label == label or node.label == label.lower():
                return node
        return None

    def _get_outgoing_edges(
        self,
        graph: Any,
        node: Any,
        user_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Get outgoing edges from a node, using agent-aware weights."""
        if graph is None:
            return []

        outgoing = graph.get_outgoing(node.id)
        results = []
        for target_id, edge in outgoing:
            target_node = graph.get_node(target_id)
            if target_node is None:
                continue

            # Use agent-specific weight if available
            if user_id is not None and hasattr(edge, 'agent_weights'):
                agent_key = f"user_{user_id}"
                if agent_key in edge.agent_weights:
                    weight = edge.agent_weights[agent_key]
                else:
                    weight = edge.weight
            else:
                weight = edge.weight

            results.append((target_node.label, weight))

        return results

    def __len__(self):
        return len(self._salience)


# ─── DialogueContext ─────────────────────────────────────────────────────────

@dataclass
class DialogueState:
    """Serializable snapshot of dialogue context."""
    user_id: str
    turn_count: int
    active_concepts: Dict[str, float]
    active_edge_count: int
    last_output: str
    last_output_triples: List[Triple]


class DialogueContext:
    """
    Main interface for dialogue working memory management.

    Handles:
    - Injection of parsed triples into the active subgraph
    - Salience decay and pruning
    - Spreading activation integration
    - Turn tracking and state persistence
    """

    def __init__(
        self,
        user_id: str = "default",
        decay_rate: float = 0.92,
        activation_threshold: float = 0.01,
        max_spreading_depth: int = 3,
        spreading_decay_per_hop: float = 0.7,
    ):
        self.user_id = user_id
        self.turn_count = 0

        # Working memory
        self.active_subgraph = ActiveSubgraph(
            decay_rate=decay_rate,
            activation_threshold=activation_threshold,
            max_spreading_depth=max_spreading_depth,
            spreading_decay_per_hop=spreading_decay_per_hop,
        )

        # Last turn tracking for self-correction
        self.last_output: str = ""
        self.last_output_triples: List[Triple] = []
        self.last_activations: Dict[str, float] = {}

        # Conversational memory
        self._conversation_history: List[Dict[str, Any]] = []

        # User-specific episodic clusters (persistent across sessions)
        self._user_beliefs: Dict[str, Dict[str, Any]] = {}  # belief_key -> metadata

    def process_turn(
        self,
        user_input: str,
        triples: List[Triple],
        graph: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Process one conversation turn.

        Steps:
        1. Inject parsed triples with high salience
        2. Decay existing salience
        3. Run spreading activation
        4. Save state for next turn

        Args:
            user_input: Raw user input string
            triples: Parsed triples from the input
            graph: Optional ConceptGraph for spreading activation

        Returns:
            Dict of activated concept -> activation score
        """
        self.turn_count += 1

        # Step 1: Inject triples
        self.active_subgraph.inject(triples, salience=1.0)

        # Step 2: Decay
        self.active_subgraph.decay()

        # Step 3: Spread activation
        activations = self.active_subgraph.propagate_activation(
            graph, user_id=self.user_id
        )
        self.last_activations = activations

        # Step 4: Record history
        self._conversation_history.append({
            "turn": self.turn_count,
            "user_input": user_input,
            "triples": triples,
            "activations": dict(activations),
            "timestamp": time.time(),
        })

        # Keep history bounded
        if len(self._conversation_history) > 100:
            self._conversation_history = self._conversation_history[-100:]

        return activations

    def record_output(self, output: str, triples: List[Triple]):
        """Record the system's output for self-correction."""
        self.last_output = output
        self.last_output_triples = triples

    def get_context(self) -> Dict[str, Any]:
        """Get current context for generation / debugging."""
        return {
            "user_id": self.user_id,
            "turn_count": self.turn_count,
            "active_concepts": self.active_subgraph.get_active_concepts(),
            "active_edges": self.active_subgraph.get_active_edges(),
            "last_output": self.last_output,
            "history_length": len(self._conversation_history),
        }

    def get_active_concepts(self, threshold: float = 0.01) -> Dict[str, float]:
        """Get currently active concepts above threshold."""
        return self.active_subgraph.get_active_concepts(threshold)

    def get_state(self) -> DialogueState:
        """Get serializable state snapshot."""
        return DialogueState(
            user_id=self.user_id,
            turn_count=self.turn_count,
            active_concepts=self.active_subgraph.get_active_concepts(),
            active_edge_count=len(self.active_subgraph),
            last_output=self.last_output,
            last_output_triples=self.last_output_triples,
        )

    def store_user_belief(self, key: str, metadata: Dict[str, Any]):
        """Store a user-specific belief for long-term persistence."""
        metadata["stored_at"] = time.time()
        if key in self._user_beliefs:
            # Update existing: increment strength
            old = self._user_beliefs[key]
            old["strength"] = min(1.0, old.get("strength", 0.5) + 0.1)
            old["access_count"] = old.get("access_count", 0) + 1
            old["last_accessed"] = time.time()
            old.update(metadata)
        else:
            metadata["strength"] = 0.5
            metadata["access_count"] = 1
            metadata["last_accessed"] = time.time()
            self._user_beliefs[key] = metadata

    def get_user_belief(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a stored user-specific belief."""
        return self._user_beliefs.get(key)

    def get_all_user_beliefs(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored user-specific beliefs."""
        return dict(self._user_beliefs)

    def clear_for_sleep(self):
        """Clear working memory for sleep consolidation (preserves user beliefs)."""
        self.active_subgraph.reset()
        self.last_output = ""
        self.last_output_triples = []
        self.last_activations = {}

    def __repr__(self):
        return (
            f"<DialogueContext user={self.user_id} turns={self.turn_count} "
            f"active_edges={len(self.active_subgraph)}>"
        )
