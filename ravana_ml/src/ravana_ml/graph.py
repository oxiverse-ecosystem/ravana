import numpy as np
import time
from typing import Any, Optional, List, Tuple, Dict, Set
from .tensor import StateTensor, RawTensor, tensor
from collections import defaultdict

try:
    from scipy.sparse import csr_matrix
    _HAS_SCIPY_SPARSE = True
except ImportError:
    _HAS_SCIPY_SPARSE = False


class ConceptNodeType:
    """Node type categories for semantic organization."""
    CONCRETE = "concrete"        # tangible entities (objects, persons, places)
    ABSTRACT = "abstract"        # abstract concepts (justice, freedom, numbers)
    RELATIONAL = "relational"    # relations/verbs (cause, part-of, similar-to)
    PHYSICAL = "physical"        # physical properties/mechanisms (mass, velocity, heat)
    PROPOSITIONAL = "propositional"  # truth-bearing statements (facts, beliefs)
    EPISODIC = "episodic"        # autobiographical events (memories, experiences)


class ConceptNode:
    def __init__(self, node_id: int, vector: np.ndarray, label: str = "",
                 node_type: str = ConceptNodeType.CONCRETE):
        self.id = node_id
        self.vector = vector.copy()          # active_vector — fast plastic representation
        self.core_vector = vector.copy()     # identity anchor — slowly changing, drift-resistant
        self.genesis_vector = vector.copy()  # original vector for drift tracking
        self.label = label or f"c{node_id}"
        self.activation = 0.0
        self.salience = 0.3
        self.prediction_free_energy = 0.0
        self.stability = 0.5
        self.confidence = 0.1
        self.timestamp = time.time()
        self.contradiction_count = 0
        self.fatigue = 0.0

        # Node type for semantic organization
        self.node_type: str = node_type  # one of ConceptNodeType values

        # Temporal free energy tracking (prediction error dynamics)
        # Renamed from "pressure" to "free_energy" for consistency with FreeEnergyAccumulator
        self.contradiction_free_energy: float = 0.0  # accumulated free energy from prediction errors
        self.free_energy_history: List[float] = []    # last 20 prediction error magnitudes
        self.free_energy_gradient: float = 0.0        # rate of change (positive = escalating)

        # Hierarchical abstraction fields
        self.level: int = 0  # 0 = leaf, higher = more abstract
        self.parent: Optional[int] = None  # parent concept ID
        self.children: Set[int] = set()  # child concept IDs
        self.abstraction_degree: float = 0.0  # 0.0 = raw, 1.0 = fully compressed

        # Temporal tracking
        self.last_activated: float = 0.0  # most recent activation timestamp
        self.activation_history: List[float] = []  # last 100 activation timestamps
        self.temporal_context: Optional[np.ndarray] = None  # context at last activation

    @property
    def effective_activation(self) -> float:
        """Activation scaled down by fatigue.
        
        Prevents fatigued concepts from staying active or dominating.
        """
        return self.activation * (1.0 - getattr(self, 'fatigue', 0.0))

    @property
    def drift_magnitude(self) -> float:
        """L2 distance between current vector and genesis vector.

        High drift means the concept has shifted significantly from its
        original meaning — it may be time to split or re-evaluate.
        """
        diff = self.vector - self.genesis_vector
        return float(np.linalg.norm(diff))

    def age(self) -> float:
        return time.time() - self.timestamp

    def decay(self, rate=0.01):
        self.activation *= (1.0 - rate * self.age())
        self.timestamp = time.time()

    def record_activation(self, context_vector: Optional[np.ndarray] = None):
        """Record an activation event in temporal history."""
        now = time.time()
        self.last_activated = now
        self.activation_history.append(now)
        # Keep rolling window of 100
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
        if context_vector is not None:
            self.temporal_context = context_vector.copy()

    def recency_score(self, decay_rate: float = 0.1) -> float:
        """How recently was this concept activated? Exponential decay from last activation.

        Returns 1.0 if just activated, approaches 0.0 over time.
        """
        if self.last_activated is None or self.last_activated <= 0:
            return 0.0
        elapsed = time.time() - self.last_activated
        return float(np.exp(-decay_rate * elapsed / 3600.0))  # decay per hour

    def frequency_score(self, window_seconds: float = 86400.0) -> float:
        """What fraction of activations happened in the recent window?

        Returns 0.0 if never activated, up to 1.0 if all activations are recent.
        """
        if not self.activation_history:
            return 0.0
        cutoff = time.time() - window_seconds
        recent = sum(1 for t in self.activation_history if t > cutoff)
        return recent / len(self.activation_history)

    @property
    def plasticity(self):
        return 1.0 - self.stability

    def __repr__(self):
        hierarchy = f" L{self.level}" if self.level > 0 else ""
        children = f" [{len(self.children)}ch]" if self.children else ""
        return (f"<Node {self.id} '{self.label}' act={self.activation:.3f} "
                f"conf={self.confidence:.3f} stab={self.stability:.3f}"
                f"{hierarchy}{children}>")


class ConceptEdge:
    # Default weights for each relation type (can be overridden in add_edge)
    RELATION_DEFAULT_WEIGHTS = {
        "semantic": 0.5,         # general association
        "causal": 0.7,           # cause-effect
        "temporal": 0.6,         # before/after
        "analogical": 0.7,       # structure mapping
        "contextual": 0.4,       # context-dependent
        "inferred": 0.3,         # derived via reasoning
        "negation": 0.8,         # inhibitory: A NOT B (weight=0.8, edge_type="inhibitory")
        "antonym": 0.9,          # opposite mapping: A ↔ opposite B
        "transitive": 0.8,       # ordered chain: A > B > C → A > C
        "physical_cause": 0.6,   # mechanism simulation: ice + heat → water
        "comparison": 0.7,       # dimensional equality: 1kg feather = 1kg steel
        "pragmatic": 0.5,        # implicature: "I ate breakfast" → ¬specific food
    }

    # Default edge types for relation types
    RELATION_DEFAULT_EDGE_TYPES = {
        "negation": "inhibitory",
        "antonym": "excitatory",
        "transitive": "excitatory",
        "physical_cause": "excitatory",
        "comparison": "excitatory",
        "pragmatic": "excitatory",
    }

    def __init__(self, source: int, target: int, weight: float = 0.5,
                 shortcut: bool = False, edge_type: str = "excitatory",
                 relation_type: str = "semantic", relation_dim: int = 16,
                 confidence: float = 0.5):
        self.source = source
        self.target = target
        self._weight = max(0.0, min(1.0, weight))
        self._confidence = confidence
        self.prediction_free_energy = 0.0
        self.stability = 0.3
        self.timestamp = time.time()
        self.prediction_count = 0
        self.forward_pred_count = 0   # A→B successful predictions
        self.backward_pred_count = 0  # B→A successful predictions
        self.shortcut = shortcut  # context→target edges are exempt from competition
        self._edge_type = edge_type  # "excitatory" or "inhibitory"
        self.relation_type = relation_type  # "semantic", "causal", "temporal", "analogical", "contextual", "inferred"
        # Predicate token: the verb token ID that created/primarily trains this edge.
        # Enables verb-level discrimination within the same relation type.
        # E.g., heat→expansion has predicate "causes", heat→ice has predicate "melts".
        # -1 means unset (edges created by non-token mechanisms).
        self.predicate_token_id: int = -1
        # Relation embedding: learned vector encoding the relational pattern
        # Initialized from relation_type seed, updated by Hebbian learning
        self.relation_vector = self._init_relation_vector(relation_type, relation_dim)
        # EWC (Elastic Weight Consolidation) fields
        self.fisher_importance: float = 0.0  # per-edge importance for old tasks
        self.old_weight: float = 0.5         # weight snapshot at domain boundary
        # Bayesian posterior: Beta(alpha, beta) over edge weight
        # alpha = 1 + successes, beta = 1 + failures — starts as uniform Beta(1,1)
        self._posterior_alpha: float = 1.0 + weight * 10.0  # prior from initial weight
        self._posterior_beta: float = 1.0 + (1.0 - weight) * 10.0
        # Cached norm for relation_vector (invalidated when RV changes)
        self._rv_norm_cache: Optional[float] = None
        self.parent_graph = None
        # Multi-Agent Edge Weights (dialogue system)
        self.agent_weights: Dict[str, float] = {}
        self.source_metadata: Dict[str, Any] = {
            'source_agent': 'system',
            'epistemic_status': 'fact',
            'is_user_statement': False,
            'is_user_experience': False,
            'is_inferred': False,
            'correction_history': [],
        }

    def __getstate__(self):
        """Lightweight pickle: exclude heavy unused fields (agent_weights, source_metadata)."""
        state = self.__dict__.copy()
        state.pop('agent_weights', None)
        state.pop('source_metadata', None)
        state.pop('parent_graph', None)
        return state

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, val):
        self._weight = max(0.0, min(1.0, val))
        self._on_change()

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, val):
        self._confidence = val
        self._on_change()

    @property
    def edge_type(self):
        return self._edge_type

    @edge_type.setter
    def edge_type(self, val):
        self._edge_type = val
        self._on_change()

    @property
    def posterior_alpha(self):
        return self._posterior_alpha

    @posterior_alpha.setter
    def posterior_alpha(self, val):
        self._posterior_alpha = val
        self._on_change()

    @property
    def posterior_beta(self):
        return self._posterior_beta

    @posterior_beta.setter
    def posterior_beta(self, val):
        self._posterior_beta = val
        self._on_change()

    def _on_change(self):
        p_graph = getattr(self, "parent_graph", None)
        if p_graph is not None:
            p_graph._adj_dirty = True
            p_graph.version = getattr(p_graph, "version", 0) + 1

    @staticmethod
    def _init_relation_vector(relation_type: str, dim: int) -> np.ndarray:
        """Initialize relation vector from type label with deterministic seeding.

        The first few dimensions act like type anchors so the initial relation
        families are genuinely separable instead of merely pseudo-random.
        Small deterministic noise keeps vectors from collapsing into one-hot
        aliases while preserving a strong type signature.
        """
        type_order = {
            "semantic": 0,
            "causal": 1,
            "temporal": 2,
            "analogical": 3,
            "contextual": 4,
            "inferred": 5,
            # Phase 6: new cognitive reasoning edge types
            "negation": 6,        # inhibitory: A NOT B
            "antonym": 7,         # opposite mapping: A ↔ opposite B
            "transitive": 8,      # ordered chain: A > B > C → A > C
            "physical_cause": 9,  # mechanism simulation: ice + heat → water
            "comparison": 10,     # dimensional equality: 1kg feather = 1kg steel
            "pragmatic": 11,      # implicature: "I ate breakfast" → ¬specific food
        }
        type_idx = type_order.get(relation_type, 0)
        vec = np.zeros(dim, dtype=np.float32)
        if dim > 0:
            vec[type_idx % dim] = 1.0

        if dim > 1:
            rng = np.random.RandomState(type_idx + 42)
            noise = rng.randn(dim).astype(np.float32) * (0.02 if dim >= 6 else 0.01)
            noise[type_idx % dim] = 0.0
            vec += noise

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @property
    def effective_weight(self) -> float:
        """Weight with sign applied for inhibitory edges."""
        if self.edge_type == "inhibitory":
            return -self.weight
        return self.weight

    @property
    def posterior_mean(self) -> float:
        """Bayesian posterior mean E[w] = alpha / (alpha + beta)."""
        return self.posterior_alpha / (self.posterior_alpha + self.posterior_beta + 1e-10)

    def get_weight_for_agent(self, agent_id: str) -> float:
        """
        Get the effective edge weight for a specific agent.
        Hierarchy: agent-specific > global.
        Args:
            agent_id: Agent identifier (e.g., 'user_likhith' or 'global')
        Returns:
            Effective weight for this agent
        """
        if not hasattr(self, 'agent_weights'):
            return self.weight
        if agent_id in self.agent_weights:
            return self.agent_weights[agent_id]
        return self.weight

    def update_weight_for_agent(self, agent_id: str, delta: float):
        """
        Update the edge weight for a specific agent.
        Only modifies user-specific weight, not global weight.
        Args:
            agent_id: Agent identifier (e.g., 'user_likhith')
            delta: Amount to add (negative = penalize, positive = boost)
        """
        if not hasattr(self, 'agent_weights'):
            return
        current = self.agent_weights.get(agent_id, self.weight)
        new_weight = max(0.0, min(1.0, current + delta))
        self.agent_weights[agent_id] = new_weight


    @property
    def posterior_uncertainty(self) -> float:
        """Posterior variance — high = uncertain about this edge's strength."""
        a, b = self.posterior_alpha, self.posterior_beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1) + 1e-10)

    @property
    def plasticity(self):
        return 1.0 - self.stability

    def __setstate__(self, state):
        """Handle deserialization of older checkpoints missing newer attributes."""
        self.__dict__.update(state)
        if 'weight' in state and not hasattr(self, '_weight'):
            self._weight = state['weight']
        if 'confidence' in state and not hasattr(self, '_confidence'):
            self._confidence = state['confidence']
        if 'edge_type' in state and not hasattr(self, '_edge_type'):
            self._edge_type = state['edge_type']
        if 'posterior_alpha' in state and not hasattr(self, '_posterior_alpha'):
            self._posterior_alpha = state['posterior_alpha']
        if 'posterior_beta' in state and not hasattr(self, '_posterior_beta'):
            self._posterior_beta = state['posterior_beta']

        defaults = {
            'predicate_token_id': -1,
            'fisher_importance': 0.0,
            'old_weight': 0.5,
            '_posterior_alpha': 1.0,
            '_posterior_beta': 1.0,
            'parent_graph': None,
        }
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)
        # relation_vector may also be missing from very old checkpoints
        if not hasattr(self, 'relation_vector') or self.relation_vector is None:
            self.relation_vector = ConceptEdge._init_relation_vector(
                getattr(self, 'relation_type', 'semantic'), 16)

    def __repr__(self):
        inh = " [I]" if self.edge_type == "inhibitory" else ""
        rt = f" {self.relation_type}" if self.relation_type != "semantic" else ""
        return f"<Edge {self.source}->{self.target} w={self.weight:.3f} conf={self.confidence:.3f}{'[S]' if self.shortcut else ''}{inh}{rt}>"


class ConceptBinding:
    """Probabilistic mapping between tokens, concepts, and memories.

    A single binding links a token (or memory) to a concept node with
    confidence, source tracking, and reinforcement history. Supports
    the unified semantic namespace: language ↔ conceptual ↔ autobiographical.

    Bindings can drift, split, merge, and decay — they are not static
    dictionary entries but living semantic links.
    """

    __slots__ = ('token_id', 'concept_id', 'confidence', 'source',
                 'reinforcement_count', 'last_used', 'created_at',
                 'decay_score', 'ambiguity')

    def __init__(self, token_id: int, concept_id: int,
                 confidence: float = 0.5, source: str = "learned"):
        self.token_id = token_id
        self.concept_id = concept_id
        self.confidence = max(0.0, min(1.0, confidence))
        self.source = source  # "learned", "memory", "manual", "inferred"
        self.reinforcement_count = 0
        self.last_used = time.time()
        self.created_at = time.time()
        self.decay_score = 0.0
        self.ambiguity = 0.0  # 0 = unambiguous, 1 = highly ambiguous

    def reinforce(self, amount: float = 0.05):
        """Strengthen this binding through use."""
        self.confidence = min(1.0, self.confidence + amount)
        self.reinforcement_count += 1
        self.last_used = time.time()
        self.decay_score = max(0.0, self.decay_score - 0.1)
        self.ambiguity = max(0.0, self.ambiguity - 0.02)

    def decay(self, rate: float = 0.01):
        """Apply time-based decay to this binding."""
        age = time.time() - self.last_used
        self.decay_score += rate * (age / 3600.0)  # decay per hour
        if self.decay_score > 5.0:
            self.confidence *= 0.95  # weaken on heavy decay

    @property
    def strength(self) -> float:
        """Effective binding strength = confidence * (1 - decay/10)."""
        return self.confidence * max(0.0, 1.0 - self.decay_score / 10.0)

    def __repr__(self):
        return (f"<Binding tok={self.token_id}→con={self.concept_id} "
                f"conf={self.confidence:.3f} str={self.strength:.3f} "
                f"src={self.source} rein={self.reinforcement_count}>")


class ConceptBindingMap:
    """Manages the full token ↔ concept ↔ memory binding space.

    Supports:
    - Multiple bindings per token (ambiguous meanings)
    - Confidence-weighted lookup
    - Ambiguity detection (multiple high-confidence bindings)
    - Decay and reinforcement
    - Split/merge when concepts evolve
    """

    def __init__(self):
        # token_id -> list of bindings (sorted by confidence)
        self._by_token: Dict[int, List[ConceptBinding]] = {}
        # concept_id -> list of bindings
        self._by_concept: Dict[int, List[ConceptBinding]] = {}
        # (token_id, concept_id) -> binding (fast lookup)
        self._index: Dict[Tuple[int, int], ConceptBinding] = {}

    def bind(self, token_id: int, concept_id: int,
             confidence: float = 0.5, source: str = "learned") -> ConceptBinding:
        """Create or reinforce a binding."""
        key = (token_id, concept_id)
        if key in self._index:
            binding = self._index[key]
            binding.reinforce(confidence * 0.1)
            return binding

        binding = ConceptBinding(token_id, concept_id, confidence, source)
        self._index[key] = binding
        self._by_token.setdefault(token_id, []).append(binding)
        self._by_concept.setdefault(concept_id, []).append(binding)
        # Sort by confidence descending
        self._by_token[token_id].sort(key=lambda b: -b.confidence)
        self._by_concept[concept_id].sort(key=lambda b: -b.confidence)
        return binding

    def get_concepts(self, token_id: int, min_confidence: float = 0.1) -> List[ConceptBinding]:
        """Get all concept bindings for a token, sorted by confidence."""
        bindings = self._by_token.get(token_id, [])
        return [b for b in bindings if b.strength >= min_confidence]

    def get_tokens(self, concept_id: int, min_confidence: float = 0.1) -> List[ConceptBinding]:
        """Get all token bindings for a concept."""
        bindings = self._by_concept.get(concept_id, [])
        return [b for b in bindings if b.strength >= min_confidence]

    def best_concept(self, token_id: int) -> Optional[int]:
        """Get the strongest concept for a token, or None."""
        bindings = self.get_concepts(token_id)
        return bindings[0].concept_id if bindings else None

    def best_token(self, concept_id: int) -> Optional[int]:
        """Get the strongest token for a concept, or None."""
        bindings = self.get_tokens(concept_id)
        return bindings[0].token_id if bindings else None

    def is_ambiguous(self, token_id: int, threshold: float = 0.3) -> bool:
        """Check if a token has multiple strong concept bindings."""
        bindings = self.get_concepts(token_id, min_confidence=threshold)
        return len(bindings) > 1

    def ambiguity_score(self, token_id: int) -> float:
        """How ambiguous is this token? 0 = unambiguous, 1 = highly ambiguous."""
        bindings = self.get_concepts(token_id, min_confidence=0.1)
        if len(bindings) <= 1:
            return 0.0
        # Entropy-like measure over binding strengths
        strengths = [b.strength for b in bindings]
        total = sum(strengths)
        if total == 0:
            return 0.0
        probs = [s / total for s in strengths]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        max_entropy = np.log2(len(bindings))
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    def decay_all(self, rate: float = 0.01):
        """Apply decay to all bindings."""
        for binding in self._index.values():
            binding.decay(rate)

    def split_bindings(self, parent_concept_id: int, child_a_id: int, child_b_id: int,
                       criterion_fn) -> None:
        """Redistribute bindings from a parent concept to its split children.

        Args:
            parent_concept_id: The concept that was split
            child_a_id, child_b_id: The two new child concepts
            criterion_fn: Callable that takes a vector and returns True for child_a, False for child_b
        """
        bindings = self._by_concept.get(parent_concept_id, [])
        for binding in bindings:
            # Create new bindings for each child
            token_id = binding.token_id
            # The criterion uses the binding's context or the token's associated vector
            # Since we don't store vectors in bindings, we use confidence as a proxy
            # and let the caller provide the criterion
            self.bind(token_id, child_a_id, confidence=binding.confidence * 0.5, source="split")
            self.bind(token_id, child_b_id, confidence=binding.confidence * 0.5, source="split")

    def disambiguate(self, token_id: int, context_vector: np.ndarray,
                     graph: 'ConceptGraph', suppression_rate: float = 0.1) -> Optional[int]:
        """Resolve ambiguity by suppressing competing bindings based on context.

        When a token maps to multiple concepts (e.g., "python" → snake/programming),
        the current context determines which meaning wins. Losing bindings have
        their confidence decayed faster — like semantic suppression in the brain.

        Args:
            token_id: The ambiguous token
            context_vector: Current context (e.g., surrounding concept activations)
            graph: ConceptGraph to look up concept vectors
            suppression_rate: How much to suppress losing bindings

        Returns:
            The winning concept_id, or None if no bindings
        """
        bindings = self.get_concepts(token_id, min_confidence=0.1)
        if len(bindings) <= 1:
            return bindings[0].concept_id if bindings else None

        # Score each binding by context similarity
        scored = []
        for b in bindings:
            node = graph.get_node(b.concept_id)
            if node is None:
                scored.append((b, -1.0))
                continue
            sim = np.dot(context_vector, node.vector) / (
                np.linalg.norm(context_vector) * np.linalg.norm(node.vector) + 1e-15
            )
            scored.append((b, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]

        # Suppress losing bindings
        for binding, sim in scored[1:]:
            # More similar to context = less suppression (they're plausible alternatives)
            suppression = suppression_rate * (1.0 - max(0.0, sim))
            binding.confidence = max(0.0, binding.confidence - suppression)
            binding.ambiguity = min(1.0, binding.ambiguity + 0.05)

        # Reinforce winner
        winner.reinforce(0.02)
        winner.ambiguity = max(0.0, winner.ambiguity - 0.05)

        # Re-sort bindings by confidence since values changed
        if token_id in self._by_token:
            self._by_token[token_id].sort(key=lambda b: b.confidence, reverse=True)

        return winner.concept_id

    def prune(self, min_strength: float = 0.05):
        """Remove bindings that have decayed below threshold."""
        to_remove = [(tok, con) for (tok, con), b in self._index.items()
                     if b.strength < min_strength]
        for tok, con in to_remove:
            key = (tok, con)
            binding = self._index.pop(key)
            if tok in self._by_token:
                self._by_token[tok] = [b for b in self._by_token[tok] if b is not binding]
            if con in self._by_concept:
                self._by_concept[con] = [b for b in self._by_concept[con] if b is not binding]
        return len(to_remove)

    def __len__(self):
        return len(self._index)

    def __contains__(self, key: Tuple[int, int]):
        return key in self._index


class CognitiveRegulator:
    """Multi-timescale regulation with damping and hysteresis.

    Prevents overcorrection instability by operating at three timescales:
    - Fast: immediate inhibition boost/relaxation (per-step)
    - Medium: sleep scheduling urgency (per-sleep-cycle)
    - Slow: topology reshaping decisions (every N cycles)

    Each controller has:
    - Damping: exponential smoothing to prevent overshooting
    - Hysteresis: dead zone to prevent oscillation between phases
    - Cooldown: minimum steps between adjustments
    """

    def __init__(self):
        # Current state
        self.current_phase: str = "exploratory"
        self.phase_confidence: float = 0.0

        # Fast controller state (inhibition)
        self._fast_inhibition_boost: float = 0.0
        self._fast_cooldown: int = 0
        self._fast_damping: float = 0.7  # exponential smoothing factor

        # Medium controller state (sleep urgency)
        self._medium_sleep_urgency: float = 0.0
        self._medium_cooldown: int = 0
        self._medium_damping: float = 0.8

        # Slow controller state (topology)
        self._slow_plasticity_boost: float = 0.0
        self._slow_cooldown: int = 0
        self._slow_damping: float = 0.9

        # Hysteresis: phase must be consistent for N steps before switching
        self._phase_buffer: List[str] = []
        self._phase_buffer_size: int = 3
        self._hysteresis_dead_zone: float = 0.1  # ignore phase changes with confidence < this

        # Counters
        self._step: int = 0
        self._adjustments_made: int = 0
        self._oscillation_count: int = 0  # tracks rapid phase flips
        self._last_phases: List[str] = []  # recent phase history for oscillation detection

    def update(self, phase_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a phase classification and return damped adjustments.

        Args:
            phase_info: output from ConceptGraph.classify_phase()

        Returns:
            adjustments dict with damped values for each timescale
        """
        self._step += 1
        new_phase = phase_info["phase"]
        confidence = phase_info["confidence"]
        recs = phase_info.get("recommendations", {})

        # Hysteresis: only switch phase if confident and consistent
        self._phase_buffer.append(new_phase)
        if len(self._phase_buffer) > self._phase_buffer_size:
            self._phase_buffer = self._phase_buffer[-self._phase_buffer_size:]

        # Check if phase is stable (all recent readings agree)
        if len(self._phase_buffer) == self._phase_buffer_size:
            if all(p == new_phase for p in self._phase_buffer):
                if confidence > self._hysteresis_dead_zone:
                    # Phase transition confirmed
                    if new_phase != self.current_phase:
                        self._last_phases.append(new_phase)
                        if len(self._last_phases) > 10:
                            self._last_phases = self._last_phases[-10:]
                    self.current_phase = new_phase
                    self.phase_confidence = confidence

        # Oscillation detection: if phase flips rapidly, increase damping
        if len(self._last_phases) >= 4:
            recent = self._last_phases[-4:]
            unique_phases = len(set(recent))
            if unique_phases >= 3:
                self._oscillation_count += 1
                # Increase damping to dampen oscillations
                self._fast_damping = min(0.95, self._fast_damping + 0.05)
                self._medium_damping = min(0.95, self._medium_damping + 0.03)
            else:
                # Relax damping slowly
                self._fast_damping = max(0.5, self._fast_damping - 0.01)
                self._medium_damping = max(0.6, self._medium_damping - 0.005)

        # ── Fast controller: inhibition boost ──
        adjustments = {}
        if self._fast_cooldown <= 0:
            target = recs.get("inhibition_boost", 0.0)
            # Damped update: approach target smoothly
            diff = target - self._fast_inhibition_boost
            if abs(diff) > 0.01:  # dead zone
                self._fast_inhibition_boost += diff * (1.0 - self._fast_damping)
                self._fast_cooldown = 2  # minimum steps between adjustments
                self._adjustments_made += 1
        else:
            self._fast_cooldown -= 1
        adjustments["inhibition_boost"] = self._fast_inhibition_boost

        # ── Medium controller: sleep urgency ──
        if self._medium_cooldown <= 0:
            target = recs.get("sleep_urgency", 0.0)
            diff = target - self._medium_sleep_urgency
            if abs(diff) > 0.05:
                self._medium_sleep_urgency += diff * (1.0 - self._medium_damping)
                self._medium_cooldown = 5
                self._adjustments_made += 1
        else:
            self._medium_cooldown -= 1
        adjustments["sleep_urgency"] = self._medium_sleep_urgency

        # ── Slow controller: plasticity boost ──
        if self._slow_cooldown <= 0:
            target = recs.get("plasticity_boost", 0.0)
            diff = target - self._slow_plasticity_boost
            if abs(diff) > 0.02:
                self._slow_plasticity_boost += diff * (1.0 - self._slow_damping)
                self._slow_cooldown = 10
                self._adjustments_made += 1
        else:
            self._slow_cooldown -= 1
        adjustments["plasticity_boost"] = self._slow_plasticity_boost

        adjustments["contradiction_focus"] = recs.get("contradiction_focus", False)
        adjustments["phase"] = self.current_phase
        adjustments["phase_confidence"] = self.phase_confidence
        adjustments["oscillation_count"] = self._oscillation_count

        return adjustments

    def status(self) -> Dict[str, Any]:
        """Current regulator status for diagnostics."""
        return {
            "phase": self.current_phase,
            "confidence": self.phase_confidence,
            "inhibition_boost": self._fast_inhibition_boost,
            "sleep_urgency": self._medium_sleep_urgency,
            "plasticity_boost": self._slow_plasticity_boost,
            "oscillation_count": self._oscillation_count,
            "fast_damping": self._fast_damping,
            "medium_damping": self._medium_damping,
            "slow_damping": self._slow_damping,
            "adjustments_made": self._adjustments_made,
        }

    def meta_adapt(self, overshoot: float = 0.0, recovery_speed: float = 0.0,
                   oscillation_rate: float = 0.0) -> Dict[str, Any]:
        """Adapt regulator parameters based on recovery feedback.

        The regulator learns from its own interventions: if overshoot is high,
        increase damping; if recovery is too slow, decrease damping; if
        oscillation is high, widen dead zones and increase hysteresis.

        This is the meta-learning loop: the regulator adjusts its own
        governance policies based on observed outcomes.

        Args:
            overshoot: how far the system went past baseline during recovery (0-1+)
            recovery_speed: how fast recovery happened (0=never, 1=instant)
            oscillation_rate: fraction of recent steps with phase oscillation (0-1)

        Returns:
            dict of parameter changes made
        """
        changes = {}

        # Meta-loss: weighted combination of failure modes
        # High overshoot = damping too low
        # Low speed = damping too high (or dead zone too wide)
        # High oscillation = damping too low or hysteresis too tight
        overshoot_penalty = min(1.0, overshoot * 2.0)  # normalize: 0.5 overshoot = penalty 1.0
        slowness_penalty = max(0.0, 1.0 - recovery_speed * 2.0)  # speed 0.5 = penalty 0
        oscillation_penalty = min(1.0, oscillation_rate * 5.0)  # rate 0.2 = penalty 1.0

        # ── Damping adaptation ──
        # If overshoot is high, increase damping (more conservative)
        # If recovery is too slow, decrease damping (more aggressive)
        if overshoot_penalty > 0.3:
            # Overshoot detected: increase damping
            adjustment = 0.02 * overshoot_penalty
            self._fast_damping = min(0.95, self._fast_damping + adjustment)
            self._medium_damping = min(0.95, self._medium_damping + adjustment * 0.7)
            self._slow_damping = min(0.95, self._slow_damping + adjustment * 0.5)
            changes["damping_direction"] = "increase"
        elif slowness_penalty > 0.3 and overshoot_penalty < 0.1:
            # Too slow, no overshoot: decrease damping
            adjustment = 0.01 * slowness_penalty
            self._fast_damping = max(0.5, self._fast_damping - adjustment)
            self._medium_damping = max(0.6, self._medium_damping - adjustment * 0.7)
            self._slow_damping = max(0.7, self._slow_damping - adjustment * 0.5)
            changes["damping_direction"] = "decrease"

        # ── Dead zone adaptation ──
        # If oscillating, widen dead zones to reduce sensitivity
        if oscillation_penalty > 0.3:
            widen = 0.01 * oscillation_penalty
            self._hysteresis_dead_zone = min(0.3, self._hysteresis_dead_zone + widen)
            changes["dead_zone"] = "widen"
        elif oscillation_penalty < 0.1 and self._hysteresis_dead_zone > 0.1:
            # Stable: can afford tighter dead zones
            self._hysteresis_dead_zone = max(0.05, self._hysteresis_dead_zone - 0.005)
            changes["dead_zone"] = "tighten"

        # ── Cooldown adaptation ──
        # If oscillating, increase cooldowns to slow down reactions
        if oscillation_penalty > 0.5:
            self._fast_cooldown = max(self._fast_cooldown, 3)
            self._medium_cooldown = max(self._medium_cooldown, 7)
            changes["cooldown"] = "increase"

        # Track meta-learning state
        if not hasattr(self, '_meta_history'):
            self._meta_history = []
        self._meta_history.append({
            "step": self._step,
            "overshoot": overshoot,
            "recovery_speed": recovery_speed,
            "oscillation_rate": oscillation_rate,
            "fast_damping": self._fast_damping,
            "medium_damping": self._medium_damping,
        })
        if len(self._meta_history) > 100:
            self._meta_history = self._meta_history[-100:]

        return changes


class GeometryHistory:
    """Circular buffer of geometry metric snapshots with trend detection.

    Records semantic geometry metrics over time to detect phase transitions,
    drift, and emergent dynamics. Each snapshot is a dict of metric values
    tagged with a step counter and optional event label.
    """

    def __init__(self, max_snapshots: int = 200):
        self.max_snapshots = max_snapshots
        self.snapshots: List[Dict[str, Any]] = []
        self.step: int = 0

    def record(self, metrics: Dict[str, Any], event: str = ""):
        """Record a geometry snapshot."""
        snapshot = {
            "step": self.step,
            "event": event,
            "timestamp": time.time(),
            **{k: v for k, v in metrics.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)},
        }
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        self.step += 1

    def get_series(self, metric_name: str) -> List[float]:
        """Extract a time series for a specific metric."""
        return [s.get(metric_name, 0.0) for s in self.snapshots if metric_name in s]

    def detect_trend(self, metric_name: str, window: int = 20) -> Dict[str, float]:
        """Detect trend in a metric over the last `window` snapshots.

        Returns:
        - mean: mean value over window
        - slope: linear trend (positive=rising, negative=falling)
        - volatility: std of first differences
        - anomaly: current value's z-score relative to window
        """
        series = self.get_series(metric_name)
        if len(series) < 3:
            return {"mean": 0.0, "slope": 0.0, "volatility": 0.0, "anomaly": 0.0}

        arr = np.array(series[-window:])
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        # Linear trend
        x = np.arange(len(arr))
        slope = float(np.polyfit(x, arr, 1)[0]) if len(arr) > 1 else 0.0

        # Volatility
        diffs = np.diff(arr)
        volatility = float(np.std(diffs)) if len(diffs) > 0 else 0.0

        # Anomaly: z-score of last value
        anomaly = float((arr[-1] - mean) / (std + 1e-15)) if std > 0 else 0.0

        return {"mean": mean, "slope": slope, "volatility": volatility, "anomaly": anomaly}

    def detect_phase_transition(self, window: int = 30) -> List[str]:
        """Detect potential phase transitions in recent history.

        Returns list of warnings about notable changes.
        """
        warnings = []
        key_metrics = ["graph_entropy", "relation_separation", "inference_specificity_mean",
                       "contradiction_density", "neighbor_preservation", "attractor_stability"]

        for metric in key_metrics:
            trend = self.detect_trend(metric, window)
            if abs(trend["anomaly"]) > 2.0:
                direction = "spiking" if trend["anomaly"] > 0 else "dropping"
                warnings.append(f"{metric} {direction} (z={trend['anomaly']:.1f})")
            if abs(trend["slope"]) > 0.01 and trend["volatility"] < abs(trend["slope"]):
                direction = "rising" if trend["slope"] > 0 else "falling"
                warnings.append(f"{metric} steadily {direction} (slope={trend['slope']:+.4f})")

        return warnings

    def compute_recovery_elasticity(self, metric_name: str,
                                     baseline_value: float,
                                     perturbation_value: float,
                                     recovery_start: int = 0) -> Dict[str, float]:
        """Compute recovery elasticity for a metric after perturbation.

        elasticity = recovery_speed * recovery_completeness / (1 + overshoot)

        Args:
            metric_name: which metric to analyze
            baseline_value: the stable value before perturbation
            perturbation_value: the degraded value after perturbation
            recovery_start: index in snapshots where recovery began

        Returns:
            - speed: 1/tau where tau is steps to reach 90% recovery (0 if never)
            - completeness: 1 - |final - baseline| / |perturbation - baseline|
            - overshoot: max deviation beyond baseline during recovery
            - elasticity: combined metric (speed * completeness / (1 + overshoot))
            - tau: steps to 90% recovery (inf if never)
        """
        series = self.get_series(metric_name)
        if len(series) < 2 or perturbation_value == baseline_value:
            return {"speed": 0.0, "completeness": 0.0, "overshoot": 0.0,
                    "elasticity": 0.0, "tau": float("inf")}

        recovery = series[recovery_start:]
        if not recovery:
            return {"speed": 0.0, "completeness": 0.0, "overshoot": 0.0,
                    "elasticity": 0.0, "tau": float("inf")}

        # Distance to recover
        total_distance = abs(perturbation_value - baseline_value)

        # Speed: how fast does it approach baseline?
        # Find first time it reaches 90% recovery
        tau = float("inf")
        for i, val in enumerate(recovery):
            remaining = abs(val - baseline_value)
            if remaining < total_distance * 0.1:
                tau = float(i + 1)
                break
        speed = 1.0 / tau if tau < float("inf") else 0.0

        # Completeness: how close is final value to baseline?
        final_val = recovery[-1]
        completeness = 1.0 - abs(final_val - baseline_value) / total_distance
        completeness = max(0.0, min(1.0, completeness))

        # Overshoot: max deviation beyond baseline during recovery
        overshoot = 0.0
        for val in recovery:
            deviation = abs(val - baseline_value)
            if deviation > total_distance:  # overshot past baseline
                overshoot = max(overshoot, (deviation - total_distance) / total_distance)

        # Combined elasticity
        elasticity = speed * completeness / (1.0 + overshoot)

        return {
            "speed": speed,
            "completeness": completeness,
            "overshoot": overshoot,
            "elasticity": elasticity,
            "tau": tau,
        }

    def summary(self) -> Dict[str, Any]:
        """Summary of all tracked metrics over history."""
        if not self.snapshots:
            return {"snapshots": 0}
        result = {"snapshots": len(self.snapshots), "steps": self.step}
        key_metrics = ["graph_entropy", "relation_separation", "inference_specificity_mean",
                       "contradiction_density", "neighbor_preservation", "attractor_stability",
                       "clustering_coefficient", "branching_factor"]
        for metric in key_metrics:
            series = self.get_series(metric)
            if series:
                result[metric] = {
                    "mean": float(np.mean(series)),
                    "std": float(np.std(series)),
                    "min": float(np.min(series)),
                    "max": float(np.max(series)),
                    "last": series[-1],
                }
        return result


class ConceptGraph:
    def __init__(self, dim: int = 64, max_nodes: int = 10000,
                 anchor_relation_vectors: bool = True,
                 adaptive_downscale: bool = True):
        self.dim = dim
        self.max_nodes = max_nodes
        self._anchor_relation_vectors = anchor_relation_vectors
        self._adaptive_downscale = adaptive_downscale
        self.nodes: Dict[int, ConceptNode] = {}
        self.edges: Dict[Tuple[int, int], ConceptEdge] = {}
        # Thread-safety: the web learner / background thread mutate the graph
        # concurrently with turn-time reasoning that iterates these dicts.
        # All mutators below take this lock; read loops in callers snapshot
        # via list(self.graph.nodes.items()) so they never iterate a live dict
        # the writer can resize (which raises
        # "dictionary changed size during iteration").
        import threading as _threading
        self._lock = _threading.RLock()
        self.version = 0

        # Scalability: optional FAISS index for O(log N) similarity search
        self._faiss_index = None
        self._use_faiss = False
        try:
            import faiss
            self._use_faiss = True
            self._faiss = faiss
        except ImportError:
            pass

        # Scalability: incremental consolidation tracking
        self._activated_since_sleep: Set[int] = set()

        # Scalability: pre-bucketed edges by relation type for contrastive sampling
        self._edges_by_relation_type: Dict[str, List[Tuple[Tuple[int,int], ConceptEdge]]] = defaultdict(list)
        self.next_id = 0
        self.total_free_energy = 0.0
        self.contradiction_hotspots: Set[int] = set()

        # Temporal context: a drifting vector that represents "when" we are
        # Slowly shifts toward the centroid of currently active concepts
        self.temporal_context: np.ndarray = np.zeros(dim, dtype=np.float32)
        self.temporal_context_drift_rate: float = 0.05

        # ── Performance indices ──
        # Adjacency list: source_id -> [(target_id, edge)]
        self._outgoing: Dict[int, List[Tuple[int, ConceptEdge]]] = defaultdict(list)
        # Reverse adjacency: target_id -> [(source_id, edge)]
        self._incoming: Dict[int, List[Tuple[int, ConceptEdge]]] = defaultdict(list)
        # Active node tracking: avoids O(N) scan in propagation
        self._active_nodes: Set[int] = set()
        # Vectorized similarity matrix (lazy-built, incremental updates)
        self._vector_matrix_normed: Optional[np.ndarray] = None
        self._node_id_order: List[int] = []
        self._vectors_dirty: bool = True
        self._dirty_nodes: Set[int] = set()  # nodes needing matrix row update
        # Sparse adjacency matrix for bulk spread_activation (Phase 3a)
        self._adj_sparse = None  # scipy.sparse CSR, lazy-built
        self._adj_dirty = True
        self._sparse_threshold = 200  # use sparse path when nodes > this

        # ── Relational inference tracking ──
        # Successful inference paths: (source, target) -> usage_count
        self._successful_paths: Dict[Tuple[int, int], int] = defaultdict(int)
        # Relation embedding dimensionality (must match ConceptEdge default)
        self._relation_dim: int = dim
        # Inference sparsity tracking
        self._inference_log: List[Dict[str, float]] = []  # last 50 inference runs
        # Semantic curvature: neighbor snapshots for topology deformation tracking
        self._neighbor_snapshot: Dict[int, Set[int]] = {}  # node_id -> set of k-nearest neighbor IDs
        self._curvature_history: List[float] = []  # preservation scores over time
        # Long-horizon geometry tracking
        self._geometry_history = GeometryHistory(max_snapshots=200)
        # Multi-timescale cognitive regulator
        self._regulator = CognitiveRegulator()

        # ── Vector stability: max L2 change per adjust_vector call ──
        # Prevents concept vectors from jumping too fast, which causes
        # _nearest_concept() to return different IDs and orphan edges.
        self.max_step_delta: float = 0.01

        # ── Diagnostics caching (Phase 3 optimization) ──
        # graph_diagnostics is expensive; cache result and reuse within a window
        self._diagnostics_cache: Optional[Dict[str, Any]] = None
        self._diagnostics_cache_step: int = -999
        # compute_curvature/basin_depth reuse the cached vector matrix
        self._cached_norms: Optional[np.ndarray] = None

    def __getstate__(self):
        """Exclude non-picklable / recreated-at-load state (the thread lock)."""
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        """Handle deserialization of older checkpoints missing newer attributes."""
        self.__dict__.update(state)
        # Initialize attributes added after checkpoints were created
        defaults = {
            '_dirty_nodes': set(),
            '_use_faiss': False,
            '_faiss_index': None,
            '_cached_norms': None,
            '_vector_matrix_normed': None,
            '_node_id_order': [],
            '_vectors_dirty': True,
            '_adj_sparse': None,
            '_adj_dirty': True,
            '_sparse_threshold': 200,
            '_diagnostics_cache': None,
            '_diagnostics_cache_step': -999,
            '_activated_since_sleep': set(),
            '_edges_by_relation_type': defaultdict(list),
            '_outgoing': defaultdict(list),
            '_incoming': defaultdict(list),
            '_active_nodes': set(),
            '_successful_paths': defaultdict(int),
            '_inference_log': [],
            '_neighbor_snapshot': {},
            '_curvature_history': [],
            'max_step_delta': 0.01,
            'version': 0,
        }
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)
        # Locks are never pickled — recreate the internal thread-safety lock
        # on every load so saved state is immediately reentrant.
        import threading as _threading
        self._lock = _threading.RLock()
        # Re-detect FAISS availability
        try:
            import faiss
            self._use_faiss = True
            self._faiss = faiss
        except ImportError:
            pass
        # Ensure parent_graph is set on all edges
        for edge in self.edges.values():
            edge.parent_graph = self

    # ── node management ──

    def add_node(self, vector: Optional[np.ndarray] = None, label: str = "") -> ConceptNode:
        with self._lock:
            if len(self.nodes) >= self.max_nodes:
                self._prune_oldest()
            nid = self.next_id
            self.next_id += 1
            v = vector.copy() if vector is not None else np.random.randn(self.dim).astype(np.float32) * 0.1
            node = ConceptNode(nid, v, label)
            self.nodes[nid] = node
            self._vectors_dirty = True
            self._adj_dirty = True
            self.version = getattr(self, "version", 0) + 1
            return node

    def get_node(self, nid: int) -> Optional[ConceptNode]:
        return self.nodes.get(nid)

    def remove_node(self, nid: int):
        with self._lock:
            if nid in self.nodes:
                node = self.nodes[nid]
                # Unlink from parent
                if node.parent is not None and node.parent in self.nodes:
                    self.nodes[node.parent].children.discard(nid)
                # Orphan children (move them up one level)
                for child_id in node.children:
                    child = self.nodes.get(child_id)
                    if child:
                        child.parent = node.parent
                        if node.parent is not None and node.parent in self.nodes:
                            self.nodes[node.parent].children.add(child_id)
                # Remove connected edges and clean adjacency indices
                edges_to_remove = [k for k in self.edges if k[0] == nid or k[1] == nid]
                for (s, t) in edges_to_remove:
                    self.remove_edge(s, t)
                del self.nodes[nid]
                self._outgoing.pop(nid, None)
                self._incoming.pop(nid, None)
                self._active_nodes.discard(nid)
                self._adj_dirty = True
                self.version = getattr(self, "version", 0) + 1

    # ── adjacency helpers ──

    def get_outgoing(self, nid: int) -> List[Tuple[int, ConceptEdge]]:
        """Get outgoing edges for a node in O(degree) time."""
        return self._outgoing.get(nid, [])

    def get_incoming(self, nid: int) -> List[Tuple[int, ConceptEdge]]:
        """Get incoming edges for a node in O(degree) time."""
        return self._incoming.get(nid, [])

    def _rebuild_vector_matrix(self):
        """Rebuild the normalized vector matrix for fast cosine similarity.

        Incremental: only updates rows for dirty nodes when possible.
        Full rebuild only when matrix doesn't exist, node set changed,
        or dirty set exceeds 50% of nodes.
        """
        ids = sorted(self.nodes.keys())
        need_full = (
            not self.nodes
            or self._vector_matrix_normed is None
            or set(ids) != set(self._node_id_order)
            or len(self._dirty_nodes) > len(ids) * 0.5
        )
        if not self.nodes:
            self._vector_matrix_normed = None
            self._node_id_order = []
            self._vectors_dirty = False
            self._dirty_nodes.clear()
            self._cached_norms = None
            self._faiss_index = None
            return
        if need_full:
            self._node_id_order = ids
            vecs = np.stack([self.nodes[i].vector for i in ids]).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self._vector_matrix_normed = vecs / (norms + 1e-15)
            self._cached_norms = norms.squeeze(1)
        else:
            # Incremental: update only dirty rows
            id_to_row = {nid: row for row, nid in enumerate(self._node_id_order)}
            for nid in self._dirty_nodes:
                row = id_to_row.get(nid)
                if row is not None and nid in self.nodes:
                    vec = self.nodes[nid].vector.astype(np.float32)
                    norm = np.linalg.norm(vec)
                    self._vector_matrix_normed[row] = vec / (norm + 1e-15)
                    if self._cached_norms is not None:
                        self._cached_norms[row] = norm
        self._vectors_dirty = False
        self._dirty_nodes.clear()

        # Build FAISS index for similarity search
        if self._use_faiss and len(ids) >= 64:
            vecs_normed = self._vector_matrix_normed.copy()
            if len(ids) >= 1000:
                # HNSW for O(log N) approximate NN at scale
                index = self._faiss.IndexHNSWFlat(self.dim, 32)
                index.hnsw.efSearch = 64
            else:
                # Exact search for small graphs
                index = self._faiss.IndexFlatIP(self.dim)
            index.add(vecs_normed)
            self._faiss_index = index
        else:
            self._faiss_index = None

    def get_activation_vector(self) -> Tuple[List[int], np.ndarray]:
        """Get ordered activation vector aligned with node_id_order."""
        acts = np.zeros(len(self._node_id_order), dtype=np.float32)
        for i, nid in enumerate(self._node_id_order):
            acts[i] = self.nodes[nid].activation
        return self._node_id_order, acts

    def _rebuild_sparse_adj(self):
        """Build scipy.sparse CSR adjacency matrix for bulk spread_activation.

        Only used when graph exceeds _sparse_threshold nodes.
        Edge weights include posterior_mean, confidence, and relation boost.
        """
        if len(self.edges) == 0:
            self._adj_sparse = None
            self._adj_dirty = False
            return
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            self._adj_sparse = None
            self._adj_dirty = False
            return
        n = max(max(self.nodes.keys(), default=0),
                max((s for s, t in self.edges), default=0),
                max((t for s, t in self.edges), default=0)) + 1
        if n == 0:
            self._adj_sparse = None
            self._adj_dirty = False
            return
        rows, cols, data = [], [], []
        for (src, tgt), edge in self.edges.items():
            if src not in self.nodes or tgt not in self.nodes:
                continue
            eff_w = edge.posterior_mean if hasattr(edge, 'posterior_mean') else edge.weight
            w = eff_w * edge.confidence
            if edge.edge_type == "inhibitory":
                w = -w
            rows.append(src)
            cols.append(tgt)
            data.append(w)
        self._adj_sparse = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
        self._adj_dirty = False

    # ── edge management ──

    def add_edge(self, source: int, target: int, weight: Optional[float] = None,
                 shortcut: bool = False, edge_type: str = "excitatory",
                 relation_type: str = "semantic",
                 confidence: Optional[float] = None) -> ConceptEdge:
        with self._lock:
            # Guard: never wire a concept to itself. A self-loop (source == target)
            # has no directional/relational content and is a degenerate edge that
            # pollutes spreading activation and chain walking. Several call sites
            # (web learning, auto-expand, domain seeding) can reach src==tgt when a
            # concept label coincides with a relation target; this is the single
            # choke point that rejects all of them. Discovered via the self-
            # referential phrasing bug ("X causes X" / "time is similar to time").
            if source == target:
                existing = self.edges.get((source, target))
                if existing is not None:
                    return existing
                # Create a dummy edge object only so callers that dereference
                # add_edge(...).weight/.confidence don't crash; it is never
                # inserted into the adjacency structures, so it never propagates.
                return ConceptEdge(source, target, weight or 0.0,
                                    shortcut=shortcut, edge_type=edge_type,
                                    relation_type=relation_type,
                                    relation_dim=self._relation_dim,
                                    confidence=confidence or 0.5)
            key = (source, target)
            if key in self.edges:
                edge = self.edges[key]
                edge.parent_graph = self
                if weight is not None:
                    edge.weight = max(0.0, min(1.0, weight))
                if shortcut:
                    edge.shortcut = True
                if edge_type == "inhibitory":
                    edge.edge_type = "inhibitory"
                if relation_type != "semantic":
                    edge.relation_type = relation_type
                if confidence is not None:
                    edge.confidence = confidence
                return edge
            if confidence is None:
                confidence = 0.5
            # Use relation-type-specific defaults if weight/edge_type not explicitly provided
            default_weight = ConceptEdge.RELATION_DEFAULT_WEIGHTS.get(relation_type, 0.5)
            default_edge_type = ConceptEdge.RELATION_DEFAULT_EDGE_TYPES.get(relation_type, "excitatory")
            # Only use defaults if caller didn't override (weight != 0.5 or edge_type != "excitatory")
            if weight is None:
                weight = default_weight
            if edge_type == "excitatory":
                edge_type = default_edge_type
            edge = ConceptEdge(source, target, weight, shortcut=shortcut,
                              edge_type=edge_type, relation_type=relation_type,
                              relation_dim=self._relation_dim, confidence=confidence)
            edge.parent_graph = self
            # Ablation: randomize relation vector if type-anchoring is disabled
            if not self._anchor_relation_vectors:
                edge.relation_vector = np.random.randn(self._relation_dim).astype(np.float32)
                edge.relation_vector /= (np.linalg.norm(edge.relation_vector) + 1e-15)
            self.edges[key] = edge
            # Maintain adjacency indices
            self._outgoing[source].append((target, edge))
            self._incoming[target].append((source, edge))
            # Maintain relation-type bucket index for scalable contrastive sampling
            self._edges_by_relation_type[relation_type].append((key, edge))
            self._adj_dirty = True
            self.version = getattr(self, "version", 0) + 1
            return edge

    def get_edge(self, source: int, target: int) -> Optional[ConceptEdge]:
        return self.edges.get((source, target))

    def remove_edge(self, source: int, target: int):
        with self._lock:
            edge = self.edges.pop((source, target), None)
            if edge is not None:
                self._outgoing[source] = [(t, e) for t, e in self._outgoing.get(source, []) if t != target]
                self._incoming[target] = [(s, e) for s, e in self._incoming.get(target, []) if s != source]
                self._adj_dirty = True
                self.version = getattr(self, "version", 0) + 1

    def prune_low_quality_edges(self, C1: float = 0.35, K: int = 1, U: float = 0.3,
                                enabled: bool = True) -> int:
        """Synaptic-homeostasis prune of orphan / noisy semantic edges.

        See ravana.graph.engine.GraphEngine.prune_low_quality_edges for the full
        brain-basis and predicate (pattern separation + source monitoring +
        offline pruning). Implemented here on ConceptGraph so it is reachable
        from both the GraphEngine wrapper and CognitiveChatEngine._sleep_consolidate
        (which operates on self.graph directly).

        Removes an edge iff ALL of:
          - relation_type == "semantic"
          - NOT a verified fact (source_metadata.edge_kind != "web_fact")
          - prediction_count < K
          - AND (edge_kind in {"co_occurrence","auto_expand"} OR confidence < C1)
        """
        if not enabled:
            return 0
        NOISE_KINDS = ("co_occurrence", "auto_expand")
        removed = []
        for (src, tgt), edge in list(self.edges.items()):
            if getattr(edge, "relation_type", "semantic") != "semantic":
                continue
            meta = getattr(edge, "source_metadata", None) or {}
            kind = meta.get("edge_kind")
            if kind == "web_fact":
                continue
            pc = getattr(edge, "prediction_count", 0) or 0
            if pc >= K:
                continue
            conf = getattr(edge, "confidence", 0.0) or 0.0
            if kind in NOISE_KINDS or conf < C1:
                removed.append((src, tgt))
        for src, tgt in removed:
            self.remove_edge(src, tgt)
        return len(removed)


    def activate(self, nid: int, amount: float = 1.0, context_vector: Optional[np.ndarray] = None):
        with self._lock:
            node = self.nodes.get(nid)
            if node:
                node.activation = min(3.0, node.activation + amount)
                node.record_activation(context_vector)
                self._active_nodes.add(nid)
                self._activated_since_sleep.add(nid)  # track for incremental consolidation

    def update_temporal_context(self):
        """Drift temporal context toward the centroid of currently active concepts.

        Called after each cognitive step. The temporal context slowly shifts to
        reflect the current "era" of processing — enabling time-sensitive retrieval.
        """
        active_list = [self.nodes[nid] for nid in self._active_nodes
                       if nid in self.nodes and self.nodes[nid].activation > 0.1]
        if not active_list:
            return
        centroid = np.mean([n.vector * n.activation for n in active_list], axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        # Exponential moving average
        self.temporal_context = (
            (1.0 - self.temporal_context_drift_rate) * self.temporal_context +
            self.temporal_context_drift_rate * centroid
        )
        norm = np.linalg.norm(self.temporal_context)
        if norm > 0:
            self.temporal_context /= norm

    def temporal_context_similarity(self, node: ConceptNode) -> float:
        """How similar is this node's stored context to the current temporal context?

        Enables encoding specificity: memories are easier to recall in the
        context where they were encoded (Tulving, 1973).
        """
        if node.temporal_context is None:
            return 0.0
        sim = np.dot(self.temporal_context, node.temporal_context) / (
            np.linalg.norm(self.temporal_context) * np.linalg.norm(node.temporal_context) + 1e-15
        )
        return max(0.0, sim)

    def spread_activation(self, steps: int = 3, k_active: int = 7, decay: float = 0.5,
                          relation_type: Optional[str] = None):
        for _ in range(steps):
            new_activations: Dict[int, float] = {}

            # Sparse bulk path for large graphs (Phase 3a)
            # Handles base propagation; relation boost still per-edge
            # Skip sparse path when relation_type filter is set (no per-edge info)
            if relation_type is None:
                if self._adj_dirty:
                    self._rebuild_sparse_adj()
                if (self._adj_sparse is not None
                        and len(self.nodes) > self._sparse_threshold):
                    # Build activation vector for all nodes
                    n = self._adj_sparse.shape[0]
                    act_vec = np.zeros(n, dtype=np.float32)
                    for nid in self._active_nodes:
                        node = self.nodes.get(nid)
                        if node is not None and node.activation > 0.01:
                            act_vec[nid] = node.activation
                    # Bulk propagation: sparse.T @ act_vec gives incoming activation
                    bulk_prop = self._adj_sparse.T @ act_vec  # (n,)
                    # Apply fan effect vectorized
                    in_degrees = np.array(self._adj_sparse.sum(axis=0)).flatten()
                    fan_factors = 1.0 / (np.sqrt(np.maximum(0.0, in_degrees)) + 1.0)
                    bulk_prop *= fan_factors * decay
                    # Accumulate
                    for nid in self._active_nodes:
                        if bulk_prop[nid] != 0:
                            new_activations[nid] = new_activations.get(nid, 0.0) + float(bulk_prop[nid])

            # Fine-grained path: handles relation vector gate + precision weighting
            # (only when sparse path is not active — avoids double-counting)
            else:
                for nid in list(self._active_nodes):
                    node = self.nodes.get(nid)
                    if node is None or node.activation <= 0.01:
                        continue
                    # Pre-compute source norm ONCE per node (not per edge)
                    src_vec = node.vector
                    src_norm = float(np.linalg.norm(src_vec))
                    src_normed = src_vec / src_norm if src_norm > 0 else None
                    # O(degree) neighbor lookup via adjacency list
                    for target_id, edge in self._outgoing.get(nid, []):
                        # Relation type filter: only spread along matching edges
                        if relation_type is not None and edge.relation_type != relation_type:
                            continue
                        # Precision weighting: edge.confidence modulates signal strength
                        # Relation vector gate: RV alignment with source concept boosts flow
                        # Cache rv norm on edge (invalidated in learn() when RV changes)
                        rv = edge.relation_vector
                        rv_norm_cached = edge._rv_norm_cache
                        if rv_norm_cached is None:
                            rv_norm_cached = float(np.linalg.norm(rv))
                            edge._rv_norm_cache = rv_norm_cached
                        if src_normed is not None and rv_norm_cached > 0:
                            rel_boost = 1.0 + 0.3 * float(np.dot(rv[:len(src_vec)], src_normed))
                        else:
                            rel_boost = 1.0
                        # Use Bayesian posterior mean for activation spreading
                        # Falls back to raw weight if posterior not initialized
                        eff_weight = edge.posterior_mean if hasattr(edge, 'posterior_mean') else edge.weight
                        # Uncertainty penalty: high-uncertainty edges transmit less signal
                        if hasattr(edge, 'posterior_uncertainty'):
                            uncertainty = edge.posterior_uncertainty
                            precision_gate = 1.0 / (1.0 + uncertainty * 50.0)  # [0.3, 1.0] range
                        else:
                            precision_gate = 1.0
                        act = node.activation * eff_weight * edge.confidence * decay * rel_boost * precision_gate
                        if edge.edge_type == "inhibitory":
                            act = -act
                        # Fan effect: normalize by in-degree (cached per-step)
                        in_deg = len(self._incoming.get(target_id, []))
                        fan_factor = 1.0 / (in_deg ** 0.5 + 1.0)
                        act *= fan_factor
                        new_activations[target_id] = new_activations.get(target_id, 0.0) + act

            for nid, act in new_activations.items():
                if nid in self.nodes:
                    self.nodes[nid].activation = max(0.0, min(3.0, self.nodes[nid].activation + act))
                    if self.nodes[nid].activation > 0.01:
                        self._active_nodes.add(nid)

            # Hierarchical upward propagation: children activate parents
            self._propagate_upward(decay=0.3)
            self._soft_lateral_inhibition(k_active)

    def _propagate_upward(self, decay: float = 0.3):
        """Propagate activation from children to their parent concepts."""
        parent_activations: Dict[int, float] = {}
        for nid in list(self._active_nodes):
            node = self.nodes.get(nid)
            if node is not None and node.activation > 0.01 and node.parent is not None:
                parent_activations[node.parent] = (
                    parent_activations.get(node.parent, 0.0)
                    + node.activation * decay
                )
        for parent_id, act in parent_activations.items():
            if parent_id in self.nodes:
                self.nodes[parent_id].activation = min(
                    1.0, self.nodes[parent_id].activation + act
                )
                if self.nodes[parent_id].activation > 0.01:
                    self._active_nodes.add(parent_id)

    def _top_k_activation(self, k: int):
        """Legacy hard winner-take-all — prefer _soft_lateral_inhibition."""
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.activation, reverse=True)
        for node in sorted_nodes[k:]:
            node.activation = 0.0

    def _soft_lateral_inhibition(self, k: int, inhibition_strength: float = 0.5):
        """Soft lateral inhibition: each active concept suppresses others proportionally to similarity.

        Unlike hard winner-take-all, this preserves near-winners — you can be
        partially aware of alternative meanings. Closer to biological cortical dynamics.
        """
        active_list = [(nid, self.nodes[nid]) for nid in self._active_nodes
                       if nid in self.nodes and self.nodes[nid].activation > 0.01]
        if len(active_list) <= 1:
            return

        # Sort by activation descending, keep top-k as "winners"
        active_list.sort(key=lambda x: x[1].activation, reverse=True)
        winners = active_list[:k]

        # Vectorized pairwise similarity via numpy matmul
        ids = [a[0] for a in active_list]
        vecs = np.stack([a[1].vector for a in active_list]).astype(np.float32)  # (A, D)
        acts = np.array([a[1].activation for a in active_list], dtype=np.float32)  # (A,)

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_normed = vecs / (norms + 1e-15)

        # Pairwise cosine similarity matrix (A, A)
        sim_matrix = vecs_normed @ vecs_normed.T
        np.fill_diagonal(sim_matrix, 0.0)
        sim_matrix = np.maximum(sim_matrix, 0.0)  # only positive similarity suppresses

        # Suppression = sim_matrix @ acts (vectorized)
        suppression = sim_matrix @ acts * inhibition_strength  # (A,)

        # Apply suppression
        for i, nid in enumerate(ids):
            node = self.nodes[nid]
            node.activation = max(0.0, node.activation / (1.0 + suppression[i]))

        # Ensure at least top-k have meaningful activation (rescue near-winners)
        for nid, node in winners:
            if node.activation < 0.01:
                node.activation = 0.01

    # ── similarity search ──

    def find_similar(self, vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if self._vectors_dirty or self._vector_matrix_normed is None:
            self._rebuild_vector_matrix()
        if self._vector_matrix_normed is None or len(self._node_id_order) == 0:
            return []
        k = min(k, len(self._node_id_order))

        # FAISS path: O(log N) approximate NN
        if self._use_faiss and self._faiss_index is not None and len(self._node_id_order) >= 64:
            vec = vector.reshape(1, -1).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            sims, idxs = self._faiss_index.search(vec, k)
            results = []
            for sim, idx in zip(sims[0], idxs[0]):
                if idx >= 0:
                    results.append((self._node_id_order[idx], float(sim)))
            return results

        # Brute-force path: O(N·D) matrix multiply
        vec_norm = vector / (np.linalg.norm(vector) + 1e-15)
        sims = self._vector_matrix_normed @ vec_norm.astype(np.float32)  # (N,)
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
        return [(self._node_id_order[i], float(sims[i])) for i in top_k_idx]

    def bind_input(self, vector: np.ndarray, k: int = 5) -> List[int]:
        matches = self.find_similar(vector, k)
        for nid, sim in matches:
            self.activate(nid, sim)
        return [nid for nid, _ in matches]

    # ── free energy ──

    def apply_free_energy(self, nid: int, amount: float):
        node = self.nodes.get(nid)
        if node:
            node.prediction_free_energy += amount * node.salience * (1.0 - node.confidence)
            node.prediction_free_energy = min(100.0, node.prediction_free_energy)
            self.total_free_energy += amount
            if node.prediction_free_energy > 2.0:  # lowered from 5.0 to make splitting reachable
                self.contradiction_hotspots.add(nid)

    def apply_prediction_error(self, predicted_nids: List[int], actual_vector: np.ndarray):
        for nid in predicted_nids:
            node = self.nodes.get(nid)
            if node is None:
                continue
            sim = np.dot(node.vector, actual_vector) / (np.linalg.norm(node.vector) * np.linalg.norm(actual_vector) + 1e-15)
            error = max(0.0, 1.0 - sim)

            # Temporal free energy tracking (pressure accumulation dynamics)
            node.free_energy_history.append(error)
            if len(node.free_energy_history) > 20:
                node.free_energy_history = node.free_energy_history[-20:]
            # Free energy gradient: is error getting worse?
            if len(node.free_energy_history) >= 3:
                recent = np.mean(node.free_energy_history[-3:])
                older = np.mean(node.free_energy_history[:-3]) if len(node.free_energy_history) > 3 else recent
                node.free_energy_gradient = recent - older
            # Accumulate free energy with decay
            node.contradiction_free_energy = 0.9 * node.contradiction_free_energy + error

            if error > 0.3:
                node.contradiction_count += 1
            # Escalating errors produce larger free energy (temporal amplification)
            escalation = 1.0 + max(0.0, node.free_energy_gradient) * 2.0
            self.apply_free_energy(nid, error * escalation)

            # High free energy increases plasticity (frequently-wrong concepts become more learnable)
            if node.contradiction_free_energy > 3.0:
                node.stability = max(0.1, node.stability - 0.05)

    def adjust_vector(self, nid: int, delta: np.ndarray, lr: float = 0.1,
                      _max_delta: Optional[float] = None):
        node = self.nodes.get(nid)
        if node is None:
            return
        # Active vector: fast plastic update
        step = delta * lr * node.plasticity
        # Clamp per-step magnitude to prevent concept ID oscillation
        max_d = _max_delta if _max_delta is not None else self.max_step_delta
        if max_d and max_d > 0:
            step_norm = np.linalg.norm(step)
            if step_norm > max_d:
                step = step * (max_d / step_norm)
        node.vector += step
        norm = np.linalg.norm(node.vector)
        if norm > 0:
            node.vector /= norm
        # Core vector: slow consolidation update (0.05x rate)
        # This preserves identity while allowing gradual drift
        node.core_vector += delta * lr * 0.05
        core_norm = np.linalg.norm(node.core_vector)
        if core_norm > 0:
            node.core_vector /= core_norm
        self._vectors_dirty = True
        self._dirty_nodes.add(nid)

    def get_or_create_edge(self, source: int, target: int, weight: Optional[float] = None,
                           shortcut: bool = False, edge_type: str = "excitatory",
                           relation_type: str = "semantic") -> ConceptEdge:
        key = (source, target)
        if key in self.edges:
            edge = self.edges[key]
            if weight is not None and weight != edge.weight:
                edge.weight = max(0.0, min(1.0, weight))
            if shortcut:
                edge.shortcut = True
            if edge_type == "inhibitory":
                edge.edge_type = "inhibitory"
            if relation_type != "semantic":
                edge.relation_type = relation_type
            return edge
        if weight is None:
            weight = 0.3
        return self.add_edge(source, target, weight, shortcut=shortcut,
                           edge_type=edge_type, relation_type=relation_type)

    # ── plasticity ──

    def hebbian_update(self, source_nid: int, target_nid: int, coactivation: float, lr: float = 0.01):
        edge = self.get_edge(source_nid, target_nid)
        if edge is None:
            if coactivation > 0.3:
                self.add_edge(source_nid, target_nid, coactivation * 0.5)
            return
        source = self.nodes.get(source_nid)
        target = self.nodes.get(target_nid)
        if source is None or target is None:
            return
        pred_error = 1.0 - edge.confidence
        # Surprise-driven learning rate: high-confidence errors produce bigger updates
        # (like the brain's error-related negativity signal)
        # Surprise = how wrong we were * how confident we were about it
        surprise = abs(pred_error) * edge.confidence
        effective_lr = lr * (1.0 + surprise * 5.0)
        delta = effective_lr * source.activation * target.activation * pred_error * source.salience * target.plasticity
        # EWC penalty: protect important weights from drifting away from old-task optimum
        ewc_penalty = 0.1 * edge.fisher_importance * (edge.weight - edge.old_weight)
        delta -= ewc_penalty
        if edge.edge_type == "inhibitory":
            edge.weight = min(1.0, edge.weight + delta)
        else:
            edge.weight = max(0.0, min(1.0, edge.weight + delta))
        edge.confidence = min(1.0, edge.confidence + abs(delta) * 0.1)
        edge.prediction_count += 1
        edge.stability = min(1.0, edge.stability + 0.001)

        # Bayesian posterior update: Beta(alpha, beta) via prediction outcomes
        # Correct prediction (low error) → increase alpha (evidence for edge)
        # Incorrect prediction (high error) → increase beta (evidence against)
        evidence = coactivation * source.salience
        if pred_error < 0.5:
            edge.posterior_alpha += evidence  # successful prediction
        else:
            edge.posterior_beta += evidence   # failed prediction

        # Relation vector learning: push-pull dynamics (rate-limited — expensive)
        # Only update relation vectors every 5th call per edge to reduce overhead
        edge._hebbian_count = getattr(edge, '_hebbian_count', 0) + 1
        if edge._hebbian_count % 5 != 0:
            return

        # PULL: drift toward Hebbian signal (source * target correlation)
        # PUSH: repel from other relation types to maintain cluster separation
        if source.vector is not None and target.vector is not None:
            # Hebbian signal: elementwise product projected into relation space
            rv_len = len(edge.relation_vector)
            hebbian_signal = source.vector[:rv_len] * target.vector[:rv_len]
            hebbian_signal = hebbian_signal / (np.sqrt(np.sum(hebbian_signal * hebbian_signal)) + 1e-15)
            # Pull toward own signal
            edge.relation_vector += effective_lr * 0.1 * (hebbian_signal - edge.relation_vector)

            # Contrastive push: explicit negative sampling from different relation types
            # Use pre-bucketed index for O(bucket_size) instead of O(E) linear scan
            sample_edges = list(self.edges.values())[:200]
            if self._edges_by_relation_type:
                negatives = []
                for rtype, bucket in list(self._edges_by_relation_type.items()):
                    if rtype != edge.relation_type:
                        for item in bucket[:20]:
                            e = item[1] if isinstance(item, tuple) and hasattr(item[1], 'relation_vector') else item
                            negatives.append(e)
            else:
                negatives = [e for e in sample_edges
                             if e.relation_type != edge.relation_type and e is not edge]
            # Sample up to 3 negatives for explicit push
            if negatives:
                np.random.shuffle(negatives)
                for neg in negatives[:3]:
                    neg_rv = neg.relation_vector / (np.sqrt(np.sum(neg.relation_vector * neg.relation_vector)) + 1e-15)
                    # Push away from negative sample (stronger than centroid approach)
                    repel_strength = effective_lr * 0.05
                    edge.relation_vector -= repel_strength * neg_rv

                # Also pull toward centroid of same type (cluster cohesion)
                if self._edges_by_relation_type:
                    bucket = self._edges_by_relation_type.get(edge.relation_type, [])
                    same_type = [e.relation_vector for _, e in bucket
                                 if e is not edge]
                else:
                    same_type = [e.relation_vector for e in sample_edges
                                 if e.relation_type == edge.relation_type and e is not edge]
                if same_type:
                    same_centroid = np.mean(same_type, axis=0)
                    same_centroid = same_centroid / (np.sqrt(np.sum(same_centroid * same_centroid)) + 1e-15)
                    edge.relation_vector += effective_lr * 0.03 * same_centroid

            rv_norm_sq = np.sum(edge.relation_vector * edge.relation_vector)
            if rv_norm_sq > 0:
                edge.relation_vector /= np.sqrt(rv_norm_sq)

    def anti_hebbian_update(self, source_nid: int, target_nid: int, lr: float = 0.01):
        edge = self.get_edge(source_nid, target_nid)
        if edge is None:
            return
        source = self.nodes.get(source_nid)
        if source is None:
            return
        delta = -lr * source.activation * edge.confidence
        edge.weight = max(0.0, min(1.0, edge.weight + delta))
        edge.confidence = max(0.0, edge.confidence - 0.05)
        edge.stability = max(0.0, edge.stability - 0.01)
        # When excitatory edge dies from persistent mismatch, convert to inhibitory
        # instead of deleting — the mismatch itself is information
        if edge.confidence < 0.01 and edge.edge_type == "excitatory":
            edge.edge_type = "inhibitory"
            edge.weight = 0.1  # small initial inhibitory weight
            edge.confidence = 0.1
            edge.stability = 0.1

    def form_inhibitory_edges(self, contradiction_threshold: int = 3):
        """Form inhibitory edges between concepts with persistent contradictions.

        When a source concept has strong edges to multiple targets that
        can't all be true simultaneously, form inhibitory edges between
        the competing targets — like semantic suppression in the brain.

        Two pathways:
        1. Weak edges: convert low-confidence excitatory edges to inhibitory
        2. Strong contradictions: form inhibitory edges between competing targets
           when source has very high contradiction count
        """
        formed = 0
        for nid in list(self.contradiction_hotspots):
            node = self.nodes.get(nid)
            if node is None:
                continue
            # Trigger on: cumulative count OR high pressure OR escalating gradient
            pressure_trigger = (node.contradiction_free_energy > 2.0 and node.free_energy_gradient > 0.0)
            if node.contradiction_count < contradiction_threshold and not pressure_trigger:
                continue

            # Collect competing targets from this source (O(degree) via adjacency list)
            competing_targets = []
            for tgt, edge in self._outgoing.get(nid, []):
                if edge.edge_type == "excitatory":
                    target = self.nodes.get(tgt)
                    if target:
                        competing_targets.append((tgt, edge, target))

            # Pathway 1: Convert weak edges to inhibitory (original logic)
            for tgt, edge, target in competing_targets:
                if target.activation > 0.1 and edge.confidence < 0.3:
                    edge.edge_type = "inhibitory"
                    edge.weight = 0.2
                    edge.confidence = 0.2
                    formed += 1

            # Pathway 2: Strong contradictions — form inhibitory edges
            # between competing targets when source has high contradiction pressure
            # Adaptive threshold: more contradictions = more aggressive
            if node.contradiction_count >= contradiction_threshold * 3:
                # Find pairs of competing targets and form mutual inhibition
                for i, (tgt_a, edge_a, target_a) in enumerate(competing_targets):
                    for tgt_b, edge_b, target_b in competing_targets[i + 1:]:
                        # Don't inhibit if they're already related
                        existing_ab = self.get_edge(tgt_a, tgt_b)
                        existing_ba = self.get_edge(tgt_b, tgt_a)
                        if existing_ab is None and existing_ba is None:
                            # Form bidirectional inhibitory edges between targets
                            self.add_edge(tgt_a, tgt_b, 0.2, edge_type="inhibitory")
                            self.add_edge(tgt_b, tgt_a, 0.2, edge_type="inhibitory")
                            formed += 2

            # Pathway 3: Adaptive confidence — weaken strongly-held contradictions
            # When contradiction count is very high, dampen edge confidence
            if node.contradiction_count >= contradiction_threshold * 5:
                for tgt, edge, target in competing_targets:
                    if edge.confidence > 0.3:
                        # Dampen confidence proportional to contradiction severity
                        dampen = min(0.1, node.contradiction_count * 0.001)
                        edge.confidence = max(0.1, edge.confidence - dampen)

        return formed

    # ── structural plasticity ──

    def prune_edges(self, threshold: float = 0.05):
        to_remove = [k for k, e in self.edges.items() if e.confidence < threshold]
        for (s, t) in to_remove:
            self.remove_edge(s, t)
        return len(to_remove)

    def form_edges(self, coactivation_threshold: float = 0.5):
        formed = 0
        active_list = [self.nodes[nid] for nid in self._active_nodes
                       if nid in self.nodes and self.nodes[nid].activation > 0.1]
        for i, a in enumerate(active_list):
            for b in active_list[i + 1:]:
                coact = a.activation * b.activation
                if coact > coactivation_threshold and self.get_edge(a.id, b.id) is None:
                    self.add_edge(a.id, b.id, coact * 0.3)
                    formed += 1
        return formed

    def _prune_oldest(self):
        oldest = min(self.nodes.values(), key=lambda n: n.timestamp)
        self.remove_node(oldest.id)

    # ── topology-aware structural importance ──

    def compute_edge_structural_importance(self, sample_size: int = 20) -> Dict[Tuple[int, int], float]:
        """Compute structural importance scores for all edges.

        Uses a hybrid metric:
        1. Endpoint degree centrality: edges connecting high-degree hubs are important
        2. Usage frequency: edges used in successful inference paths are important
        3. Bridge approximation: edges on many shortest paths (sampled BFS) are critical

        Returns dict mapping (src, tgt) -> importance score (0-1 normalized).
        """
        if not self.edges:
            return {}

        # 1. Degree centrality per node
        degree = {}
        for nid in self.nodes:
            out_deg = len(self._outgoing.get(nid, []))
            in_deg = len(self._incoming.get(nid, []))
            degree[nid] = out_deg + in_deg

        max_degree = max(degree.values()) if degree else 1

        # 2. Edge usage from successful paths
        edge_usage = {}
        for (src, tgt), count in self._successful_paths.items():
            # Count how many times this edge appears in successful inference
            edge_usage[(src, tgt)] = count

        # 3. Sampled BFS for bridge approximation
        # Run BFS from random sample of nodes, count edge participation in shortest paths
        edge_path_count: Dict[Tuple[int, int], int] = defaultdict(int)
        node_ids = list(self.nodes.keys())
        if len(node_ids) > 1:
            rng = np.random.RandomState(42)
            sources = list(rng.choice(node_ids, min(sample_size, len(node_ids)), replace=False))

            for src in sources:
                # BFS from src
                visited = {src}
                queue = [src]
                parent_map: Dict[int, List[int]] = {src: []}
                while queue:
                    current = queue.pop(0)
                    for tgt, _ in self._outgoing.get(current, []):
                        if tgt not in self.nodes:
                            continue
                        if tgt not in visited:
                            visited.add(tgt)
                            queue.append(tgt)
                            parent_map[tgt] = [current]
                        elif tgt in parent_map:
                            parent_map[tgt].append(current)

                # Count edge participation in all shortest paths
                for tgt in visited:
                    for parent in parent_map.get(tgt, []):
                        edge_path_count[(parent, tgt)] += 1

        # Normalize path counts
        max_path = max(edge_path_count.values()) if edge_path_count else 1

        # Combine into structural importance score
        importance = {}
        for key, edge in self.edges.items():
            src, tgt = key
            # Degree component: geometric mean of endpoint degrees
            d_src = degree.get(src, 0) / max_degree
            d_tgt = degree.get(tgt, 0) / max_degree
            degree_score = (d_src * d_tgt) ** 0.5

            # Path component: how often on shortest paths
            path_score = edge_path_count.get(key, 0) / max_path

            # Usage component
            usage_score = min(1.0, edge_usage.get(key, 0) / 10.0)

            # Hybrid: weighted combination
            score = 0.3 * degree_score + 0.4 * path_score + 0.3 * usage_score
            importance[key] = score

        return importance

    # ── concept splitting ──

    def should_split(self, nid: int, contradiction_threshold: int = 5,
                     drift_threshold: float = 0.5, entropy_threshold: float = 0.7) -> bool:
        """Check if a concept has accumulated enough internal contradiction to split.

        A concept should split when ANY of these signals is strong enough:
        - High contradiction count (prediction errors accumulating)
        - High drift from original meaning (semantic shift)
        - High edge entropy (edges point to diverse, unrelated targets)
        - High contradiction pressure with positive gradient (escalating)

        Uses OR logic but with raised thresholds to prevent runaway graph growth.
        Thresholds tuned: contradiction >= 5 (was 2), drift >= 0.5 (was 0.3),
        entropy >= 0.7 (was 0.5), pressure > 3.0 (was 2.0).
        """
        node = self.nodes.get(nid)
        if node is None:
            return False

        # Signal 1: Contradiction count
        if node.contradiction_count >= contradiction_threshold:
            return True

        # Signal 2: Drift from genesis
        if node.drift_magnitude >= drift_threshold:
            return True

        # Signal 3: Edge entropy — diverse targets suggest polysemy
        outgoing = [t for t, e in self._outgoing.get(nid, []) if e.edge_type == "excitatory"]
        if len(outgoing) >= 2:
            targets = [self.nodes[t] for t in outgoing if t in self.nodes]
            if len(targets) >= 2:
                dists = []
                for i, a in enumerate(targets):
                    for b in targets[i + 1:]:
                        sim = np.dot(a.vector, b.vector) / (
                            np.linalg.norm(a.vector) * np.linalg.norm(b.vector) + 1e-15
                        )
                        dists.append(1.0 - sim)
                if dists:
                    mean_dist = np.mean(dists)
                    if mean_dist > entropy_threshold:
                        return True

        # Signal 4: Escalating contradiction pressure
        if hasattr(node, 'contradiction_free_energy') and hasattr(node, 'free_energy_gradient'):
            if node.contradiction_free_energy > 3.0 and node.free_energy_gradient > 0.0:
                return True

        return False

    def split_concept(self, nid: int, binding_map: Optional['ConceptBindingMap'] = None) -> Tuple[int, int]:
        """Split a concept into two competing sub-concepts.

        Creates two children from the parent, distributes edges based on
        vector alignment, forms an inhibitory edge between them, and
        optionally redistributes bindings.

        Returns:
            (child_a_id, child_b_id)
        """
        parent = self.nodes.get(nid)
        if parent is None:
            raise ValueError(f"Node {nid} not found")

        # Find the two most divergent neighbors to seed the split
        outgoing = [(t, e) for t, e in self._outgoing.get(nid, [])
                    if e.edge_type == "excitatory"]
        if len(outgoing) < 2:
            # Not enough structure to split meaningfully — create orthogonal children
            vec_a = parent.vector.copy()
            vec_b = -parent.vector.copy()
        else:
            # Find the two most dissimilar target vectors
            targets = [(t, self.nodes[t]) for t, _ in outgoing if t in self.nodes]
            max_dist = 0.0
            best_pair = (targets[0][0], targets[1][0]) if len(targets) >= 2 else (None, None)
            for i, (t1, n1) in enumerate(targets):
                for t2, n2 in targets[i + 1:]:
                    dist = np.linalg.norm(n1.vector - n2.vector)
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (t1, t2)

            # Seed children toward the two divergent targets
            n_a = self.nodes.get(best_pair[0])
            n_b = self.nodes.get(best_pair[1])
            if n_a is not None and n_b is not None:
                vec_a = 0.5 * parent.vector + 0.5 * n_a.vector
                vec_b = 0.5 * parent.vector + 0.5 * n_b.vector
            else:
                vec_a = parent.vector.copy()
                vec_b = -parent.vector.copy()

        # Normalize
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a > 0:
            vec_a /= norm_a
        if norm_b > 0:
            vec_b /= norm_b

        # Create children
        child_a = self.add_node(vec_a, label=f"{parent.label}_a")
        child_b = self.add_node(vec_b, label=f"{parent.label}_b")

        # Copy parent properties to children
        for child in [child_a, child_b]:
            child.level = parent.level  # same level as parent
            child.salience = parent.salience
            child.confidence = parent.confidence * 0.5  # reduced — they're new interpretations
            child.stability = parent.stability * 0.5

        # Make parent abstract
        parent.children = {child_a.id, child_b.id}
        parent.abstraction_degree = 0.5
        parent.level += 1

        # Distribute edges: align each edge to the closer child
        for target_id, edge in list(self._outgoing.get(nid, [])):
            target_node = self.nodes.get(target_id)
            if target_node is None:
                continue

            # Which child is closer to the target?
            sim_a = np.dot(vec_a, target_node.vector) / (
                np.linalg.norm(vec_a) * np.linalg.norm(target_node.vector) + 1e-15
            )
            sim_b = np.dot(vec_b, target_node.vector) / (
                np.linalg.norm(vec_b) * np.linalg.norm(target_node.vector) + 1e-15
            )

            chosen_child = child_a if sim_a >= sim_b else child_b
            self.add_edge(chosen_child.id, target_id, edge.weight,
                         shortcut=edge.shortcut, edge_type=edge.edge_type)

        # Inhibitory edge between children (they're competing interpretations)
        self.add_edge(child_a.id, child_b.id, 0.3, edge_type="inhibitory")
        self.add_edge(child_b.id, child_a.id, 0.3, edge_type="inhibitory")

        # Redistribute bindings if binding map provided
        if binding_map is not None:
            binding_map.split_bindings(nid, child_a.id, child_b.id,
                                       lambda vec: np.dot(vec, vec_a) > np.dot(vec, vec_b))

        return child_a.id, child_b.id

    def homeostatic_downscale(self, protection_threshold: float = 0.8,
                               downscale_factor: float = 0.8,
                               structural_protection: float = 0.5) -> Tuple[float, float]:
        """Global synaptic homeostasis — adaptive downscale that preserves learned associations.

        Unlike uniform downscale, this uses per-edge adaptive factors:
        - High-confidence, frequently-used edges: barely weakened (0.95x)
        - Low-confidence, rarely-used edges: strongly weakened (0.6x)
        - Formula: factor = 0.6 + 0.35 * min(1.0, confidence * prediction_count / 10)

        Also: post-downscale renormalization prevents concept orphaning
        (nodes losing all meaningful connections).

        Args:
            protection_threshold: edges with stability above this are fully protected
            downscale_factor: base downscale factor (overridden by adaptive per-edge)
            structural_protection: edges with structural importance above this
                                   are also protected (0 = disabled)

        Returns:
            (total_weight_before, total_weight_after)
        """
        total_before = sum(e.weight for e in self.edges.values())

        # Compute structural importance (reuse cache from regulate() if fresh)
        si = {}
        if structural_protection > 0 and len(self.edges) > 0:
            if self._si_cache_is_fresh():
                si = self._structural_importance_cache
            else:
                si = self.compute_edge_structural_importance()
                self._update_si_cache(si)

        # First pass: downscale with protection
        for key, edge in self.edges.items():
            # Fully protected edges: skip downscale
            if edge.stability >= protection_threshold:
                continue
            if si.get(key, 0.0) >= structural_protection:
                continue

            if self._adaptive_downscale:
                # Adaptive factor: confident, frequently-used edges resist downscale
                # Floor raised from 0.6 to 0.85 — prevents catastrophic forgetting
                usage = min(1.0, edge.confidence * edge.prediction_count / 10.0)
                factor = 0.85 + 0.14 * usage  # range: 0.85-0.99
            else:
                # Uniform downscale: all edges equally weakened (ablation mode)
                factor = downscale_factor  # 0.8
            edge.weight *= factor

        # Post-downscale renormalization: prevent concept orphaning
        # If a node's mean outgoing weight drops below 0.1, restore top-3 edges
        node_max_edges: Dict[int, List[Tuple[int, float]]] = {}
        for (src, tgt), edge in self.edges.items():
            if src not in node_max_edges:
                node_max_edges[src] = []
            node_max_edges[src].append((tgt, edge.weight))

        for src, targets in node_max_edges.items():
            if not targets:
                continue
            mean_w = np.mean([w for _, w in targets])
            if mean_w < 0.1:
                # Restore top-3 edges to 0.2 minimum
                targets.sort(key=lambda x: x[1], reverse=True)
                for tgt, _ in targets[:3]:
                    edge = self.get_edge(src, tgt)
                    if edge and edge.weight < 0.2:
                        edge.weight = 0.2

        total_after = sum(e.weight for e in self.edges.values())
        return total_before, total_after

    # ── structural importance cache (shared across homeostasis + regulate) ──

    def _si_cache_is_fresh(self) -> bool:
        """Check if the structural importance cache is still fresh.

        Fresh = edge count hasn't changed significantly since last computation.
        Both homeostatic_downscale() and regulate() share this cache to avoid
        redundant O(sample_size × E) BFS computation in the same sleep cycle.
        """
        if not hasattr(self, '_structural_importance_cache') or \
           not hasattr(self, '_si_cache_n_edges'):
            return False
        # Fresh if edge count hasn't drifted more than 5% since last computation
        return abs(len(self.edges) - self._si_cache_n_edges) <= max(5, len(self.edges) * 0.05)

    def _update_si_cache(self, si: Dict[Tuple[int, int], float]):
        """Update the structural importance cache with current edge count snapshot."""
        self._structural_importance_cache = si
        self._si_cache_n_edges = len(self.edges)

    # ── sleep support ──

    def get_hotspots(self, threshold: float = 3.0) -> List[int]:
        return [nid for nid in self.contradiction_hotspots
                if nid in self.nodes and self.nodes[nid].prediction_free_energy > threshold]

    def clear_hotspots(self):
        self.contradiction_hotspots.clear()

    def reconcile_contradictions(self):
        # Clear diagnostics cache — graph state has changed
        if hasattr(self, '_diag_cache'):
            del self._diag_cache
        reconciled = 0
        for nid in list(self.contradiction_hotspots):
            node = self.nodes.get(nid)
            if node is None:
                continue
            neighbors = self._incoming.get(nid, [])
            if not neighbors:
                continue
            conflicting_weights = [e.weight for _, e in neighbors]
            mean_w = np.mean(conflicting_weights)
            if node.contradiction_count > 3:
                node.stability = max(0.0, node.stability - 0.1)
                node.prediction_free_energy *= 0.5
                # DON'T reset contradiction_count here — let it accumulate for split detection
                # Only reset after a successful split (in split_concept or _sleep_sws)
                node.contradiction_free_energy *= 0.5
                reconciled += 1
            else:
                node.prediction_free_energy = max(0.0, node.prediction_free_energy - 1.0)
                # Gentle free energy decay for non-reconciled nodes
                node.contradiction_free_energy *= 0.9
        # Don't clear all hotspots — persist unresolved ones across cycles
        # Only remove nodes that are fully resolved (low free energy)
        self.contradiction_hotspots = {
            nid for nid in self.contradiction_hotspots
            if nid in self.nodes and self.nodes[nid].prediction_free_energy > 1.0
        }
        return reconciled

    # ── hierarchy traversal ──

    def get_children(self, nid: int) -> List[int]:
        """Get direct children of a concept."""
        node = self.nodes.get(nid)
        if node is None:
            return []
        return list(node.children)

    def get_leaves(self, nid: int) -> List[int]:
        """Get all leaf descendants of a concept (recursive)."""
        node = self.nodes.get(nid)
        if node is None:
            return []
        if not node.children:
            return [nid]
        leaves = []
        for child_id in node.children:
            leaves.extend(self.get_leaves(child_id))
        return leaves

    def get_ancestors(self, nid: int) -> List[int]:
        """Get all ancestors from node to root."""
        ancestors = []
        current = self.nodes.get(nid)
        while current and current.parent is not None:
            ancestors.append(current.parent)
            current = self.nodes.get(current.parent)
        return ancestors

    def get_level(self, nid: int) -> int:
        """Get abstraction level of a concept."""
        node = self.nodes.get(nid)
        return node.level if node else 0

    def get_siblings(self, nid: int) -> List[int]:
        """Get siblings (same parent) of a concept."""
        node = self.nodes.get(nid)
        if node is None or node.parent is None:
            return []
        parent = self.nodes.get(node.parent)
        if parent is None:
            return []
        return [c for c in parent.children if c != nid]

    # ── vector consolidation ──

    def consolidate_vectors(self, rate: float = 0.1, incremental: bool = True):
        """Sleep consolidation: merge active_vector toward core_vector.

        This is the SWS analogue — during sleep, fast-changing active
        representations consolidate into the stable identity anchor.
        rate: how much active_vector moves toward core_vector (0.0-1.0)
        incremental: if True, only consolidate nodes activated since last sleep
                     (O(active_N) instead of O(N))
        """
        if not self.nodes:
            return

        # Incremental mode: only consolidate recently activated nodes
        if incremental and self._activated_since_sleep:
            ids = sorted(self._activated_since_sleep)
            self._activated_since_sleep.clear()
        else:
            ids = sorted(self.nodes.keys())
            self._activated_since_sleep.clear()

        if not ids:
            return

        # Vectorized: batch selected nodes
        active_vecs = np.stack([self.nodes[nid].vector for nid in ids]).astype(np.float32)
        core_vecs = np.stack([self.nodes[nid].core_vector for nid in ids]).astype(np.float32)
        # Move active toward core (consolidation)
        merged = active_vecs * (1.0 - rate) + core_vecs * rate
        norms = np.linalg.norm(merged, axis=1, keepdims=True)
        merged = merged / (norms + 1e-15)
        # Core vector also drifts slightly toward active (learning)
        core_merged = core_vecs * (1.0 - rate * 0.1) + merged * (rate * 0.1)
        core_norms = np.linalg.norm(core_merged, axis=1, keepdims=True)
        core_merged = core_merged / (core_norms + 1e-15)
        # Write back
        for i, nid in enumerate(ids):
            self.nodes[nid].vector = merged[i]
            self.nodes[nid].core_vector = core_merged[i]
        self._vectors_dirty = True

    # ── relational inference engine ──

    def record_path(self, source_id: int, target_id: int):
        """Record a successful inference path for later compression."""
        self._successful_paths[(source_id, target_id)] += 1

    def get_compressible_paths(self, min_usage: int = 2) -> List[Tuple[int, int, float]]:
        """Get paths that have been used enough times to compress.

        Returns list of (source, target, score) for compression.
        """
        results = []
        for (src, tgt), count in self._successful_paths.items():
            if count >= min_usage:
                # Score based on usage frequency, capped at 1.0
                score = min(1.0, count * 0.15)
                results.append((src, tgt, score))
        return results

    def infer_chain(self, start_id: int, max_hops: int = 3,
                    confidence_threshold: float = 0.15,
                    min_weight: float = 0.25,
                    k: int = 5,
                    frontier_budget: int = 5) -> List[Tuple[int, float, List[int]]]:
        """Sparse multi-hop inference with activation budgets and winner-take-most.

        Key constraints to prevent percolation (semantic fog):
        - frontier_budget: max nodes per hop (winner-take-most)
        - min_weight: ignore weak edges
        - confidence_threshold: stop propagating low-confidence chains
        - entropy penalty: penalize paths that branch too broadly
        - coherence gate: each hop must maintain semantic alignment with start

        Returns ranked list of (target_id, chain_score, path) tuples.
        """
        if start_id not in self.nodes:
            return []

        # Use core_vector for identity resolution (stable anchor)
        start_vec = self.nodes[start_id].core_vector
        start_norm = start_vec / (np.linalg.norm(start_vec) + 1e-15)

        # Relation context: average relation_vector of start node's outgoing edges
        # Used to score paths with consistent relation types
        relation_context = None
        start_edges = self._outgoing.get(start_id, [])
        if start_edges:
            rvectors = [e.relation_vector for _, e in start_edges if e.edge_type != "inhibitory"]
            if rvectors:
                relation_context = np.mean(rvectors, axis=0)
                relation_context = relation_context / (np.linalg.norm(relation_context) + 1e-15)

        # BFS with score tracking and activation budget
        frontier: List[Tuple[int, float, float, List[int]]] = [
            (start_id, 1.0, 1.0, [start_id])
        ]
        visited: Dict[int, float] = {start_id: 1.0}
        results: List[Tuple[int, float, List[int]]] = []
        edges_traversed = 0  # energy cost tracking

        for hop in range(max_hops):
            candidates: List[Tuple[int, float, float, List[int]]] = []

            for nid, score, conf, path in frontier:
                outgoing = self._outgoing.get(nid, [])
                # Count how many edges this node fans out to (entropy signal)
                fanout = len(outgoing)

                for target_id, edge in outgoing:
                    edges_traversed += 1
                    if target_id in path:
                        continue  # no cycles
                    if edge.edge_type == "inhibitory":
                        continue  # skip inhibitory edges
                    if edge.weight < min_weight:
                        continue  # skip weak edges

                    # Chain scoring with confidence decay
                    hop_score = score * edge.weight * edge.confidence
                    hop_conf = conf * edge.confidence

                    # Relation consistency bonus: paths with consistent relation types score higher
                    if relation_context is not None:
                        rv = edge.relation_vector
                        rv_norm = rv / (np.linalg.norm(rv) + 1e-15)
                        relation_sim = float(np.dot(relation_context, rv_norm))
                        hop_score *= (1.0 + 0.3 * max(0.0, relation_sim))

                    # Confidence decay accelerates with hop depth
                    hop_score *= (0.85 ** hop)  # gentler decay (was 0.7, too aggressive)

                    # Contradiction penalty
                    if target_id in self.contradiction_hotspots:
                        hop_score *= 0.3
                    if nid in self.contradiction_hotspots:
                        hop_score *= 0.5

                    # Entropy penalty: penalize high-fanout nodes (semantic fog)
                    if fanout > 3:
                        entropy_penalty = 1.0 / (1.0 + 0.1 * (fanout - 3))
                        hop_score *= entropy_penalty

                    # Coherence gate: each hop must stay semantically close to start (core_vector)
                    if target_id not in self.nodes:
                        continue
                    target_vec = self.nodes[target_id].core_vector
                    target_norm_vec = target_vec / (np.linalg.norm(target_vec) + 1e-15)
                    coherence = float(np.dot(start_norm, target_norm_vec))
                    if coherence < -0.2:
                        continue  # reject semantically opposed paths
                    hop_score *= (1.0 + 0.3 * max(0.0, coherence))

                    if hop_conf < confidence_threshold:
                        continue

                    new_path = path + [target_id]

                    # Record intermediate results (hops > 0)
                    if hop > 0 and target_id != start_id:
                        results.append((target_id, hop_score, new_path))

                    # Track candidate for winner-take-most
                    if target_id not in visited or visited[target_id] < hop_score:
                        visited[target_id] = hop_score
                        candidates.append((target_id, hop_score, hop_conf, new_path))

            # Winner-take-most: keep only top frontier_budget nodes
            candidates.sort(key=lambda x: x[1], reverse=True)
            frontier = candidates[:frontier_budget]

        # Deduplicate: keep best score per target
        best: Dict[int, Tuple[float, List[int]]] = {}
        for target_id, score, path in results:
            if target_id not in best or score > best[target_id][0]:
                best[target_id] = (score, path)

        ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)
        final_results = [(tid, sc, p) for tid, (sc, p) in ranked[:k]]

        # Log inference sparsity metrics
        if final_results:
            scores = [sc for _, sc, _ in final_results]
            total_score = sum(scores)
            winner_score = scores[0]
            specificity = winner_score / total_score if total_score > 0 else 1.0
            # Energy cost: total activation mass of all visited nodes
            visited_activation = sum(
                self.nodes[nid].activation for nid in visited if nid in self.nodes
            )
            self._inference_log.append({
                "n_results": len(final_results),
                "specificity": specificity,
                "winner_score": winner_score,
                "mean_score": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "energy_cost": edges_traversed,
                "activation_mass": visited_activation,
            })
            if len(self._inference_log) > 50:
                self._inference_log = self._inference_log[-50:]

        return final_results

    def compress_paths(self, successful_chains: List[Tuple[int, int, float]],
                       min_chain_score: float = 0.2):
        """Compress successful inference chains into shortcut edges.

        successful_chains: list of (source_id, target_id, chain_score)
        Creates shortcut edge: source → target with weight = chain_score.
        """
        compressed = 0
        for source_id, target_id, chain_score in successful_chains:
            if chain_score < min_chain_score:
                continue
            if source_id == target_id:
                continue
            existing = self.get_edge(source_id, target_id)
            if existing and existing.shortcut:
                # Strengthen existing shortcut
                existing.weight = min(1.0, existing.weight + chain_score * 0.1)
                existing.confidence = min(1.0, existing.confidence + 0.05)
                compressed += 1
            elif existing is None:
                # Create new shortcut
                self.add_edge(source_id, target_id, weight=chain_score * 0.5,
                             shortcut=True, relation_type="inferred")
                compressed += 1
        return compressed

    def find_analogy(self, source_id: int, target_domain_ids: List[int],
                     k: int = 3) -> List[Tuple[int, float]]:
        """Find analogical matches by structural pattern similarity.

        Extracts the local relation pattern around source_id (edge types,
        weight distribution, target similarity) and compares against each
        candidate in target_domain_ids. Returns best structural matches.
        """
        if source_id not in self.nodes:
            return []

        source_pattern = self._extract_relation_pattern(source_id)
        if not source_pattern:
            return []

        matches: List[Tuple[int, float]] = []
        for tid in target_domain_ids:
            if tid == source_id:
                continue
            target_pattern = self._extract_relation_pattern(tid)
            if not target_pattern:
                continue
            similarity = self._pattern_similarity(source_pattern, target_pattern)
            matches.append((tid, similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:k]

    def _extract_relation_pattern(self, nid: int) -> Dict[str, Any]:
        """Extract local relation pattern around a node for analogy matching.

        Uses relation vectors (learned embeddings) rather than type labels
        for structural comparison. This enables true analogical transfer:
        "planet orbits star" ≈ "electron orbits nucleus" because the
        relational *shape* matches even though surface concepts differ.
        """
        outgoing = self._outgoing.get(nid, [])
        incoming = self._incoming.get(nid, [])
        if not outgoing and not incoming:
            return {}

        out_weights: List[float] = []
        in_weights: List[float] = []
        out_relation_vecs: List[np.ndarray] = []
        in_relation_vecs: List[np.ndarray] = []
        target_vectors: List[np.ndarray] = []

        for tgt, edge in outgoing:
            out_weights.append(edge.weight)
            out_relation_vecs.append(edge.relation_vector)
            if tgt in self.nodes:
                target_vectors.append(self.nodes[tgt].vector)

        for src, edge in incoming:
            in_weights.append(edge.weight)
            in_relation_vecs.append(edge.relation_vector)

        # Aggregate relation vectors: weighted mean of outgoing relation embeddings
        if out_relation_vecs:
            out_rel_agg = np.mean(out_relation_vecs, axis=0)
        else:
            out_rel_agg = np.zeros(self._relation_dim, dtype=np.float32)

        if in_relation_vecs:
            in_rel_agg = np.mean(in_relation_vecs, axis=0)
        else:
            in_rel_agg = np.zeros(self._relation_dim, dtype=np.float32)

        return {
            "out_degree": len(outgoing),
            "in_degree": len(incoming),
            "out_weight_mean": float(np.mean(out_weights)) if out_weights else 0.0,
            "in_weight_mean": float(np.mean(in_weights)) if in_weights else 0.0,
            "out_weight_std": float(np.std(out_weights)) if out_weights else 0.0,
            "out_relation_vec": out_rel_agg,
            "in_relation_vec": in_rel_agg,
            "target_centroid": np.mean(target_vectors, axis=0) if target_vectors else None,
        }

    def _pattern_similarity(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> float:
        """Compute structural similarity between two relation patterns.

        Uses relation vector similarity (learned embeddings) for the core
        comparison, plus structural features (degree, weight distribution).
        This enables analogical transfer across different surface concepts.
        """
        if not p1 or not p2:
            return 0.0

        score = 0.0
        total = 0.0

        # 1. Degree similarity (structural topology)
        max_deg = max(p1["out_degree"] + p1["in_degree"],
                      p2["out_degree"] + p2["in_degree"], 1)
        deg_sim = 1.0 - abs((p1["out_degree"] + p1["in_degree"]) -
                            (p2["out_degree"] + p2["in_degree"])) / max_deg
        score += deg_sim * 1.0
        total += 1.0

        # 2. Relation vector similarity (the key innovation)
        # Compare outgoing relation patterns
        out_sim = float(np.dot(p1["out_relation_vec"], p2["out_relation_vec"]) /
                       (np.linalg.norm(p1["out_relation_vec"]) *
                        np.linalg.norm(p2["out_relation_vec"]) + 1e-15))
        score += max(0.0, out_sim) * 2.0  # double weight — this is the core signal
        total += 2.0

        # Compare incoming relation patterns
        in_sim = float(np.dot(p1["in_relation_vec"], p2["in_relation_vec"]) /
                      (np.linalg.norm(p1["in_relation_vec"]) *
                       np.linalg.norm(p2["in_relation_vec"]) + 1e-15))
        score += max(0.0, in_sim) * 1.0
        total += 1.0

        # 3. Weight distribution similarity
        w1 = p1["out_weight_mean"]
        w2 = p2["out_weight_mean"]
        w_sim = 1.0 - abs(w1 - w2) / max(w1 + w2, 1e-15)
        score += w_sim * 0.5
        total += 0.5

        return score / total if total > 0 else 0.0

    # ── hierarchical abstraction ──

    def merge_concepts(
        self,
        child_ids: List[int],
        label: str = "",
        abstraction_degree: float = 0.5,
    ) -> Optional[int]:
        """
        Create a parent concept by merging child concepts.

        The parent's vector is the centroid of its children. Edges from/to
        children are aggregated to the parent. Children retain their edges
        but gain a parent pointer.

        Args:
            child_ids: Concept IDs to merge (must be >= 2)
            label: Label for the new parent concept
            abstraction_degree: How compressed this abstraction is (0-1)

        Returns:
            Parent concept ID, or None if merge is not possible
        """
        # Validate
        valid_children = [cid for cid in child_ids if cid in self.nodes]
        if len(valid_children) < 2:
            return None

        # Don't merge if any child already has a parent at the same level
        # (prevent double-merging)
        for cid in valid_children:
            child = self.nodes[cid]
            if child.parent is not None:
                # Check if the parent is also being merged — that's fine
                if child.parent not in valid_children:
                    return None

        child_nodes = [self.nodes[cid] for cid in valid_children]

        # Compute parent level: max child level + 1
        max_child_level = max(n.level for n in child_nodes)

        # Compute parent vector: centroid of children
        child_vectors = np.array([n.vector for n in child_nodes])
        parent_vector = np.mean(child_vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(parent_vector)
        if norm > 0:
            parent_vector /= norm

        # Create parent node
        parent_label = label or f"abs_{'_'.join(str(c) for c in valid_children[:3])}"
        parent = self.add_node(parent_vector, parent_label)
        parent.level = max_child_level + 1
        parent.abstraction_degree = abstraction_degree
        parent.children = set(valid_children)
        parent.salience = np.mean([n.salience for n in child_nodes])
        parent.confidence = np.mean([n.confidence for n in child_nodes])
        parent.stability = np.mean([n.stability for n in child_nodes])

        # Link children to parent
        for child in child_nodes:
            child.parent = parent.id

        # Aggregate edges: collect all external edges from/to children
        # and create weighted edges from/to the parent
        self._aggregate_child_edges(parent.id, valid_children)

        return parent.id

    def _aggregate_child_edges(self, parent_id: int, child_ids: List[int]):
        """
        Aggregate external edges from children to the parent node.

        For each external target connected to any child, create an edge
        from parent to that target with weight = mean child edge weight.
        Similarly for incoming edges.
        """
        child_set = set(child_ids)

        # Outgoing: child → external target
        outgoing: Dict[int, List[float]] = defaultdict(list)
        # Incoming: external source → child
        incoming: Dict[int, List[float]] = defaultdict(list)

        for child_id in child_ids:
            for tgt, edge in self._outgoing.get(child_id, []):
                if tgt not in child_set:
                    outgoing[tgt].append(edge.weight)
            for src, edge in self._incoming.get(child_id, []):
                if src not in child_set:
                    incoming[src].append(edge.weight)

        # Create aggregated edges to parent
        for target_id, weights in outgoing.items():
            mean_weight = float(np.mean(weights))
            self.get_or_create_edge(parent_id, target_id, mean_weight)

        for source_id, weights in incoming.items():
            mean_weight = float(np.mean(weights))
            self.get_or_create_edge(source_id, parent_id, mean_weight)

    def find_coactivated_clusters(
        self,
        coactivation_threshold: float = 0.5,
        min_cluster_size: int = 3,
        max_cluster_size: int = 8,
    ) -> List[List[int]]:
        """
        Find clusters of frequently co-activated leaf concepts.

        Uses a simple greedy approach: for each highly active node,
        find its co-activated neighbors and group them.

        Returns:
            List of concept ID lists (each list is a merge candidate)
        """
        active_leaves = [
            n for n in self.nodes.values()
            if n.activation > 0.1 and n.level == 0 and n.parent is None
        ]

        if len(active_leaves) < min_cluster_size:
            return []

        # Build co-activation adjacency from edges
        coactive_pairs: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for node in active_leaves:
            for target_id, edge in self._outgoing.get(node.id, []):
                target = self.nodes.get(target_id)
                if target and target.level == 0 and target.parent is None:
                    coact = node.activation * target.activation * edge.weight
                    if coact > coactivation_threshold:
                        coactive_pairs[node.id].append((target_id, coact))

        # Greedy clustering: seed from highest-activity nodes
        clusters = []
        used = set()
        sorted_nodes = sorted(active_leaves, key=lambda n: n.activation, reverse=True)

        for seed in sorted_nodes:
            if seed.id in used:
                continue

            cluster = [seed.id]
            used.add(seed.id)

            # Add co-activated neighbors
            neighbors = coactive_pairs.get(seed.id, [])
            neighbors.sort(key=lambda x: x[1], reverse=True)

            for neighbor_id, _ in neighbors:
                if len(cluster) >= max_cluster_size:
                    break
                if neighbor_id not in used:
                    # Check that neighbor is co-activated with most of the cluster
                    sim_count = 0
                    for existing_id in cluster:
                        for nid, _ in coactive_pairs.get(existing_id, []):
                            if nid == neighbor_id:
                                sim_count += 1
                                break
                    if sim_count >= max(1, len(cluster) - 1):
                        cluster.append(neighbor_id)
                        used.add(neighbor_id)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def compute_compression_ratio(self) -> float:
        """
        Compute the abstraction compression ratio.

        Returns the fraction of nodes that are abstract (level > 0).
        """
        if not self.nodes:
            return 0.0
        abstract_count = sum(1 for n in self.nodes.values() if n.level > 0)
        return abstract_count / len(self.nodes)

    def get_abstraction_stats(self) -> Dict[str, Any]:
        """Get full abstraction statistics."""
        levels = [n.level for n in self.nodes.values()]
        abstract_nodes = [n for n in self.nodes.values() if n.level > 0]
        return {
            "total_nodes": len(self.nodes),
            "leaf_nodes": sum(1 for l in levels if l == 0),
            "abstract_nodes": len(abstract_nodes),
            "max_level": max(levels) if levels else 0,
            "compression_ratio": self.compute_compression_ratio(),
            "mean_abstraction_degree": (
                float(np.mean([n.abstraction_degree for n in abstract_nodes]))
                if abstract_nodes else 0.0
            ),
        }

    # ── semantic geometry diagnostics ──

    def graph_diagnostics(self, lightweight: bool = False) -> Dict[str, Any]:
        """Compute semantic geometry metrics for observability.

        Returns a dict of metrics describing the shape of the semantic field.
        Cached: returns cached result if called multiple times per cycle.
        """
        # Cache: return cached result if called multiple times in same cycle
        cache_key = 'lightweight' if lightweight else 'full'
        if hasattr(self, '_diag_cache') and self._diag_cache_key == cache_key:
            return self._diag_cache
        # Branching factor, edge weight stats, shortcut ratio, path degeneracy,
        # inference specificity, relation separation, attractor stability.
        metrics: Dict[str, Any] = {}

        n_nodes = len(self.nodes)
        n_edges = len(self.edges)

        if n_nodes == 0:
            return {"empty": True}

        # 1. Graph entropy: Shannon entropy of activation distribution
        activations = np.array([n.activation for n in self.nodes.values()])
        active_mask = activations > 0.001
        if active_mask.any():
            probs = activations[active_mask]
            probs = probs / (probs.sum() + 1e-15)
            # Shannon entropy normalized by log(N) for comparability
            raw_entropy = -np.sum(probs * np.log(probs + 1e-15))
            max_entropy = np.log(max(active_mask.sum(), 2))
            metrics["graph_entropy"] = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0
            metrics["active_count"] = int(active_mask.sum())
        else:
            metrics["graph_entropy"] = 0.0
            metrics["active_count"] = 0

        # 2. Activation spread
        metrics["activation_mean"] = float(np.mean(activations))
        metrics["activation_std"] = float(np.std(activations))
        metrics["activation_max"] = float(np.max(activations))

        # 3. Clustering coefficient (global, sampled for speed)
        # For each node, check if its neighbors are connected to each other
        sample_nodes = list(self.nodes.keys())
        if len(sample_nodes) > 200:
            rng = np.random.RandomState(42)
            sample_nodes = list(rng.choice(sample_nodes, 200, replace=False))

        triangles = 0
        triplets = 0
        for nid in sample_nodes:
            neighbors = set()
            for tgt, _ in self._outgoing.get(nid, []):
                if tgt in self.nodes:
                    neighbors.add(tgt)
            for src, _ in self._incoming.get(nid, []):
                if src in self.nodes:
                    neighbors.add(src)
            neighbors_list = list(neighbors)
            k = len(neighbors_list)
            if k < 2:
                continue
            # Count connected triples and triangles
            for i in range(k):
                for j in range(i + 1, k):
                    triplets += 1
                    if self.get_edge(neighbors_list[i], neighbors_list[j]) is not None:
                        triangles += 1
                    elif self.get_edge(neighbors_list[j], neighbors_list[i]) is not None:
                        triangles += 1

        metrics["clustering_coefficient"] = float(triangles / triplets) if triplets > 0 else 0.0

        # 4. Contradiction density
        if n_edges > 0:
            inhibitory_count = sum(1 for e in self.edges.values() if e.edge_type == "inhibitory")
            metrics["contradiction_density"] = float(inhibitory_count / n_edges)
            metrics["inhibitory_count"] = inhibitory_count
        else:
            metrics["contradiction_density"] = 0.0
            metrics["inhibitory_count"] = 0

        # 5. Relation cluster separation
        # Compare mean intra-type cosine distance vs inter-type cosine distance
        relation_groups: Dict[str, List[np.ndarray]] = {}
        for edge in self.edges.values():
            rt = edge.relation_type
            if rt not in relation_groups:
                relation_groups[rt] = []
            relation_groups[rt].append(edge.relation_vector)

        if len(relation_groups) >= 2:
            # Intra-type: mean pairwise cosine similarity within each group
            intra_sims = []
            for rt, vecs in relation_groups.items():
                if len(vecs) < 2:
                    continue
                # Sample if too many
                if len(vecs) > 50:
                    vecs = vecs[:50]
                mat = np.array(vecs)
                norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-15
                mat_normed = mat / norms
                sim_matrix = mat_normed @ mat_normed.T
                # Mean of upper triangle
                upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                if len(upper) > 0:
                    intra_sims.append(float(np.mean(upper)))

            # Inter-type: mean cosine similarity between different types
            inter_sims = []
            type_list = list(relation_groups.keys())
            for i in range(len(type_list)):
                for j in range(i + 1, len(type_list)):
                    vecs_i = relation_groups[type_list[i]][:20]
                    vecs_j = relation_groups[type_list[j]][:20]
                    mat_i = np.array(vecs_i)
                    mat_j = np.array(vecs_j)
                    norms_i = np.linalg.norm(mat_i, axis=1, keepdims=True) + 1e-15
                    norms_j = np.linalg.norm(mat_j, axis=1, keepdims=True) + 1e-15
                    cross_sim = (mat_i / norms_i) @ (mat_j / norms_j).T
                    inter_sims.append(float(np.mean(cross_sim)))

            mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
            mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
            # Separation = intra - inter (positive means clusters are distinct)
            metrics["relation_intra_similarity"] = mean_intra
            metrics["relation_inter_similarity"] = mean_inter
            metrics["relation_separation"] = mean_intra - mean_inter
        else:
            metrics["relation_intra_similarity"] = 0.0
            metrics["relation_inter_similarity"] = 0.0
            metrics["relation_separation"] = 0.0

        # 6. Attractor stability + core/active alignment (vectorized batch)
        # Instead of per-node loops with np.linalg.norm, batch all at once
        all_core = np.stack([n.core_vector for n in self.nodes.values()]).astype(np.float32)
        all_genesis = np.stack([n.genesis_vector for n in self.nodes.values()]).astype(np.float32)
        all_active = np.stack([n.vector for n in self.nodes.values()]).astype(np.float32)
        core_norms = np.linalg.norm(all_core, axis=1) + 1e-15
        genesis_norms = np.linalg.norm(all_genesis, axis=1) + 1e-15
        active_norms = np.linalg.norm(all_active, axis=1) + 1e-15
        core_genesis_sims = np.sum(all_core * all_genesis, axis=1) / (core_norms * genesis_norms)
        core_active_sims = np.sum(all_core * all_active, axis=1) / (core_norms * active_norms)
        metrics["attractor_stability"] = float(np.mean(core_genesis_sims))
        metrics["core_active_alignment"] = float(np.mean(core_active_sims))

        # 7. Branching factor: mean out-degree
        out_degrees = [len(self._outgoing.get(nid, [])) for nid in self.nodes]
        metrics["branching_factor"] = float(np.mean(out_degrees)) if out_degrees else 0.0
        metrics["branching_max"] = int(np.max(out_degrees)) if out_degrees else 0

        # 8. Edge weight stats
        weights = np.array([e.weight for e in self.edges.values()])
        if len(weights) > 0:
            metrics["edge_weight_mean"] = float(np.mean(weights))
            metrics["edge_weight_std"] = float(np.std(weights))
            metrics["edge_confidence_mean"] = float(np.mean([e.confidence for e in self.edges.values()]))
        else:
            metrics["edge_weight_mean"] = 0.0
            metrics["edge_weight_std"] = 0.0
            metrics["edge_confidence_mean"] = 0.0

        # 9. Shortcut ratio
        shortcut_count = sum(1 for e in self.edges.values() if e.shortcut)
        metrics["shortcut_ratio"] = float(shortcut_count / n_edges) if n_edges > 0 else 0.0
        metrics["shortcut_count"] = shortcut_count

        # 10. Path degeneracy: source-target pairs with multiple recorded paths
        if self._successful_paths:
            pair_counts = list(self._successful_paths.values())
            metrics["path_degeneracy_mean"] = float(np.mean(pair_counts))
            metrics["path_degeneracy_max"] = int(np.max(pair_counts))
            metrics["tracked_paths"] = len(self._successful_paths)
        else:
            metrics["path_degeneracy_mean"] = 0.0
            metrics["path_degeneracy_max"] = 0
            metrics["tracked_paths"] = 0

        # 11. Relation type distribution
        type_counts: Dict[str, int] = {}
        for edge in self.edges.values():
            type_counts[edge.relation_type] = type_counts.get(edge.relation_type, 0) + 1
        metrics["relation_type_distribution"] = type_counts

        # 12. Semantic curvature (neighbor preservation) — EXPENSIVE, skip in lightweight mode
        if not lightweight:
            preservation = self.compute_curvature()
            curvature_trend = self.curvature_trend()
            metrics["neighbor_preservation"] = preservation
            metrics["curvature_trend"] = curvature_trend["trend"]
            metrics["curvature_volatility"] = curvature_trend["volatility"]

            # 12b. Basin depth: perturbation resistance of concept neighborhoods
            if n_nodes >= 15:  # need enough nodes for meaningful k-NN
                basin = self.compute_basin_depth(k=min(10, n_nodes // 3), n_samples=min(30, n_nodes))
                metrics["basin_depth_mean"] = basin["basin_depth_mean"]
                metrics["basin_depth_min"] = basin["basin_depth_min"]
                metrics["shallow_fraction"] = basin["shallow_fraction"]
            else:
                metrics["basin_depth_mean"] = 0.0
                metrics["basin_depth_min"] = 0.0
                metrics["shallow_fraction"] = 0.0
        else:
            metrics["neighbor_preservation"] = 1.0
            metrics["curvature_trend"] = 0.0
            metrics["curvature_volatility"] = 0.0
            metrics["basin_depth_mean"] = 0.0
            metrics["basin_depth_min"] = 0.0
            metrics["shallow_fraction"] = 0.0

        # 13. Inference sparsity & energy cost (from logged inference runs)
        if self._inference_log:
            specs = [l["specificity"] for l in self._inference_log]
            n_results = [l["n_results"] for l in self._inference_log]
            energy = [l["energy_cost"] for l in self._inference_log]
            activation = [l["activation_mass"] for l in self._inference_log]
            metrics["inference_specificity_mean"] = float(np.mean(specs))
            metrics["inference_specificity_last"] = specs[-1]
            metrics["inference_branching_mean"] = float(np.mean(n_results))
            metrics["inference_count"] = len(self._inference_log)
            metrics["energy_cost_mean"] = float(np.mean(energy))
            metrics["energy_cost_last"] = energy[-1]
            metrics["activation_mass_mean"] = float(np.mean(activation))
        else:
            metrics["inference_specificity_mean"] = 0.0
            metrics["inference_specificity_last"] = 0.0
            metrics["inference_branching_mean"] = 0.0
            metrics["inference_count"] = 0
            metrics["energy_cost_mean"] = 0.0
            metrics["energy_cost_last"] = 0.0
            metrics["activation_mass_mean"] = 0.0

        # Cache result for this cycle (cleared by reconcile_contradictions)
        self._diag_cache = metrics
        self._diag_cache_key = cache_key
        return metrics

    # ── semantic curvature ──

    def compute_curvature(self, k: int = 10) -> float:
        """Compute semantic curvature: how much have concept neighborhoods changed?

        Compares current k-nearest neighbors (by vector similarity) to the
        last snapshot. Returns mean Jaccard similarity across all nodes.
        1.0 = perfectly stable, 0.0 = complete neighborhood churn.

        High curvature = semantic instability / topology deformation.
        Low curvature = stable semantic geometry.
        """
        if not self.nodes:
            return 1.0

        # Reuse the cached normalized vector matrix instead of rebuilding
        if self._vectors_dirty or self._vector_matrix_normed is None:
            self._rebuild_vector_matrix()
        if self._vector_matrix_normed is None:
            return 1.0

        # Sample nodes if graph is large
        node_ids = list(self.nodes.keys())
        max_sample = 500
        if len(node_ids) > max_sample:
            sample_idx = np.random.choice(len(node_ids), max_sample, replace=False)
            node_ids = [node_ids[i] for i in sample_idx]
            # Build small similarity matrix from sampled nodes only
            id_to_idx = {nid: i for i, nid in enumerate(self._node_id_order)}
            sample_global_idx = [id_to_idx[nid] for nid in node_ids if nid in id_to_idx]
            if len(sample_global_idx) < 2:
                return 1.0
            vecs_normed = self._vector_matrix_normed[sample_global_idx]
            sim_matrix = vecs_normed @ vecs_normed.T
        else:
            # Use full matrix for small graphs
            sim_matrix = self._vector_matrix_normed @ self._vector_matrix_normed.T

        np.fill_diagonal(sim_matrix, -1.0)  # exclude self

        # Use argpartition instead of full argsort for top-k (O(N) vs O(N log N))
        k_actual = min(k, sim_matrix.shape[1] - 1)
        current_neighbors: Dict[int, Set[int]] = {}
        for i, nid in enumerate(node_ids):
            top_k_idx = np.argpartition(sim_matrix[i], -k_actual)[-k_actual:]
            current_neighbors[nid] = {node_ids[j] for j in top_k_idx}

        # Compare to previous snapshot
        if self._neighbor_snapshot:
            jaccard_scores = []
            for nid, current_set in current_neighbors.items():
                if nid in self._neighbor_snapshot:
                    prev_set = self._neighbor_snapshot[nid]
                    intersection = len(current_set & prev_set)
                    union = len(current_set | prev_set)
                    jaccard = intersection / union if union > 0 else 1.0
                    jaccard_scores.append(jaccard)
            preservation = float(np.mean(jaccard_scores)) if jaccard_scores else 1.0
        else:
            preservation = 1.0  # no previous snapshot = no drift

        # Update snapshot
        self._neighbor_snapshot = current_neighbors
        self._curvature_history.append(preservation)
        if len(self._curvature_history) > 100:
            self._curvature_history = self._curvature_history[-100:]

        return preservation

    def curvature_trend(self) -> Dict[str, float]:
        """Analyze curvature history for trends.

        Returns:
        - mean_preservation: average neighbor preservation over history
        - trend: slope of preservation over time (negative = destabilizing)
        - volatility: std of preservation changes
        """
        if len(self._curvature_history) < 2:
            return {"mean_preservation": 1.0, "trend": 0.0, "volatility": 0.0}

        arr = np.array(self._curvature_history)
        # Linear trend
        x = np.arange(len(arr))
        slope = float(np.polyfit(x, arr, 1)[0])
        # Volatility: std of first differences
        diffs = np.diff(arr)
        volatility = float(np.std(diffs))

        return {
            "mean_preservation": float(np.mean(arr)),
            "trend": slope,  # negative = neighborhoods destabilizing
            "volatility": volatility,
        }

    def compute_basin_depth(self, k: int = 10, n_samples: int = 50,
                            jaccard_threshold: float = 0.5,
                            max_noise: float = 2.0,
                            n_steps: int = 8) -> Dict[str, float]:
        """Measure how much perturbation concepts can absorb before losing neighborhood identity.

        For each sampled node, captures its k-nearest neighbors, then applies
        graduated Gaussian noise to find the perturbation magnitude where the
        node's neighborhood (Jaccard similarity) drops below the threshold.

        This measures basin depth: deep basins resist perturbation, shallow ones
        don't. A node with basin_depth=0.5 means you need noise magnitude 0.5
        to push it out of its semantic basin.

        Args:
            k: number of nearest neighbors to track
            n_samples: number of nodes to sample (0 = all)
            jaccard_threshold: neighborhood retention threshold for basin boundary
            max_noise: maximum noise magnitude to test
            n_steps: number of noise levels to test (binary search resolution)

        Returns:
            dict with basin_depth_mean, basin_depth_min, basin_depth_std,
            shallow_fraction (fraction of nodes with depth < 0.3)
        """
        if len(self.nodes) < k + 1:
            return {"basin_depth_mean": 0.0, "basin_depth_min": 0.0,
                    "basin_depth_std": 0.0, "shallow_fraction": 0.0}

        # Reuse cached vector matrix instead of rebuilding
        if self._vectors_dirty or self._vector_matrix_normed is None:
            self._rebuild_vector_matrix()
        if self._vector_matrix_normed is None:
            return {"basin_depth_mean": 0.0, "basin_depth_min": 0.0,
                    "basin_depth_std": 0.0, "shallow_fraction": 0.0}

        all_ids = self._node_id_order
        all_normed = self._vector_matrix_normed
        id_to_idx = {nid: i for i, nid in enumerate(all_ids)}

        # Sample nodes
        if n_samples > 0 and len(all_ids) > n_samples:
            rng = np.random.RandomState(42)
            sample_ids = list(rng.choice(all_ids, n_samples, replace=False))
        else:
            sample_ids = all_ids

        # Capture baseline neighborhoods using matrix slice (avoid full N×N rebuild)
        sample_idx = np.array([id_to_idx[nid] for nid in sample_ids])
        baseline_sim = all_normed[sample_idx] @ all_normed.T  # (S, N)
        np.fill_diagonal(baseline_sim, -1.0)  # but only for self-sim; fine since we use global idx

        baseline_neighbors = {}
        for si, nid in enumerate(sample_ids):
            top_k_idx = np.argpartition(baseline_sim[si], -k)[-k:]
            baseline_neighbors[nid] = set(all_ids[j] for j in top_k_idx)

        # Noise levels to test (geometric progression for better resolution)
        noise_levels = np.linspace(0.0, max_noise, n_steps + 1)[1:]  # skip 0

        # For each node, search for basin boundary
        # Use precomputed all_normed directly instead of rebuilding
        depths = []
        for nid in sample_ids:
            node = self.nodes[nid]
            baseline_set = baseline_neighbors[nid]

            # Search: find smallest noise where Jaccard < threshold
            depth = max_noise  # default: never escaped

            for noise_mag in noise_levels:
                # Apply noise to a copy of the vector
                noise = np.random.randn(*node.vector.shape).astype(np.float32) * noise_mag
                perturbed = node.vector + noise
                perturbed_norm = np.linalg.norm(perturbed)
                if perturbed_norm > 0:
                    perturbed = perturbed / perturbed_norm

                # Find k-NN of perturbed vector using cached matrix
                sims = all_normed @ perturbed
                top_k_idx = np.argpartition(sims, -k)[-k:]
                perturbed_set = set(all_ids[j] for j in top_k_idx)

                # Jaccard similarity
                intersection = len(baseline_set & perturbed_set)
                union = len(baseline_set | perturbed_set)
                jaccard = intersection / union if union > 0 else 1.0

                if jaccard < jaccard_threshold:
                    depth = noise_mag
                    break

            depths.append(depth)

        depths_arr = np.array(depths)
        shallow_mask = depths_arr < 0.3

        return {
            "basin_depth_mean": float(np.mean(depths_arr)),
            "basin_depth_min": float(np.min(depths_arr)),
            "basin_depth_std": float(np.std(depths_arr)),
            "shallow_fraction": float(np.mean(shallow_mask)),
        }

    def project_relation_manifold(self, n_components: int = 2) -> Dict[str, Any]:
        """Project relation vectors into low-dimensional space via PCA.

        Returns:
        - coordinates: dict mapping relation_type -> list of (x, y, ...) tuples
        - centroid_drift: dict mapping relation_type -> centroid vector
        - explained_variance: PCA explained variance ratio
        - separation_score: mean inter-centroid distance / mean intra-centroid distance
        """
        if not self.edges:
            return {"coordinates": {}, "centroid_drift": {}, "explained_variance": [], "separation_score": 0.0}

        # Group edges by relation type
        groups: Dict[str, List[np.ndarray]] = {}
        for edge in self.edges.values():
            rt = edge.relation_type
            if rt not in groups:
                groups[rt] = []
            groups[rt].append(edge.relation_vector.copy())

        # Collect all vectors for PCA
        all_vecs = []
        labels = []
        for rt, vecs in groups.items():
            for v in vecs:
                all_vecs.append(v)
                labels.append(rt)

        if len(all_vecs) < 2:
            return {"coordinates": {}, "centroid_drift": {}, "explained_variance": [], "separation_score": 0.0}

        mat = np.array(all_vecs)
        mean = np.mean(mat, axis=0)
        centered = mat - mean

        # SVD-based PCA
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            components = Vt[:n_components]
            projected = centered @ components.T
            total_var = np.sum(S ** 2)
            explained = (S[:n_components] ** 2) / total_var if total_var > 0 else np.zeros(n_components)
        except np.linalg.LinAlgError:
            return {"coordinates": {}, "centroid_drift": {}, "explained_variance": [], "separation_score": 0.0}

        # Group coordinates by type
        coordinates: Dict[str, List[Tuple[float, ...]]] = {}
        idx = 0
        for rt, vecs in groups.items():
            n = len(vecs)
            coords = [tuple(projected[idx + i]) for i in range(n)]
            coordinates[rt] = coords
            idx += n

        # Compute centroids
        centroid_drift = {}
        for rt, coords in coordinates.items():
            centroid_drift[rt] = tuple(np.mean(coords, axis=0))

        # Separation score: inter-centroid / intra-centroid distance
        centroids = list(centroid_drift.values())
        if len(centroids) >= 2:
            inter_dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    inter_dists.append(np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j])))
            mean_inter = np.mean(inter_dists)

            intra_dists = []
            for rt, coords in coordinates.items():
                cent = np.array(centroid_drift[rt])
                for c in coords:
                    intra_dists.append(np.linalg.norm(np.array(c) - cent))
            mean_intra = np.mean(intra_dists) if intra_dists else 1e-15

            separation_score = float(mean_inter / (mean_intra + 1e-15))
        else:
            separation_score = 0.0

        return {
            "coordinates": coordinates,
            "centroid_drift": centroid_drift,
            "explained_variance": explained.tolist(),
            "separation_score": separation_score,
        }

    def record_geometry_snapshot(self, event: str = "", lightweight: bool = False):
        """Record current geometry metrics to history for long-horizon tracking.

        Call this after learning steps, sleep cycles, or contradiction events
        to build a time series of semantic geometry evolution.
        """
        metrics = self.graph_diagnostics(lightweight=lightweight)
        self._geometry_history.record(metrics, event)
        return metrics

    def geometry_history_summary(self) -> Dict[str, Any]:
        """Get summary of geometry evolution over time."""
        return self._geometry_history.summary()

    def geometry_trends(self, window: int = 20) -> Dict[str, Dict[str, float]]:
        """Get trends for all key metrics over recent history."""
        key_metrics = ["graph_entropy", "relation_separation", "inference_specificity_mean",
                       "contradiction_density", "neighbor_preservation", "attractor_stability"]
        return {m: self._geometry_history.detect_trend(m, window) for m in key_metrics}

    def detect_phase_transitions(self, window: int = 30) -> List[str]:
        """Detect potential phase transitions in recent geometry history."""
        return self._geometry_history.detect_phase_transition(window)

    # ── cognitive phase state & adaptive regulation ──

    # Phase definitions: (name, entropy_range, specificity_range, separation_range)
    # Ranges are (low, high) thresholds
    PHASE_THRESHOLDS = {
        "focused":    {"entropy": (0.0, 0.4), "specificity": (0.7, 1.0), "separation": (0.0, 1.0)},
        "exploratory": {"entropy": (0.3, 0.7), "specificity": (0.3, 0.7), "separation": (0.0, 1.0)},
        "diffuse":    {"entropy": (0.6, 1.0), "specificity": (0.0, 0.4), "separation": (0.0, 1.0)},
        "rigid":      {"entropy": (0.0, 0.3), "specificity": (0.8, 1.0), "separation": (0.7, 1.0)},
        "crisis":     {"contradiction_density": 0.15},  # threshold for crisis mode
    }

    # Global invariant anchors — hard limits that NEVER change regardless of history.
    # Prevents boiling frog failure where adaptive thresholds drift to normalize pathology.
    # If a metric breaches these, the phase is forced even if adaptive thresholds say it's fine.
    GLOBAL_INVARIANTS = {
        "entropy_max": 0.95,           # above this, always diffuse
        "contradiction_max": 0.25,     # above this, always crisis
        "preservation_min": 0.2,       # below this, always crisis (topology collapsing)
        "separation_max": 0.98,        # above this, always rigid (over-crystallized)
        "entropy_min_for_focused": 0.05, # below this, even "focused" is suspicious (dead system)
    }

    def calibrate_thresholds(self, min_snapshots: int = 20) -> Dict[str, Any]:
        """Compute adaptive phase thresholds from geometry history.

        After enough history is collected, phase boundaries shift to match
        the system's own operating range. This prevents mature systems from
        being misclassified by static thresholds.

        Returns calibrated thresholds dict, or empty dict if insufficient data.
        """
        history = self._geometry_history
        if len(history.snapshots) < min_snapshots:
            return {}

        key_metrics = {
            "entropy": "graph_entropy",
            "specificity": "inference_specificity_mean",
            "separation": "relation_separation",
            "contradiction": "contradiction_density",
        }

        stats = {}
        for key, metric_name in key_metrics.items():
            series = history.get_series(metric_name)
            if len(series) < min_snapshots:
                return {}
            arr = np.array(series)
            stats[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
            }

        # Adaptive thresholds: phase boundaries relative to system's own history
        e = stats["entropy"]
        s = stats["specificity"]
        sep = stats["separation"]
        c = stats["contradiction"]

        # Use percentile-based boundaries for robustness
        calibrated = {
            "focused": {
                "entropy": (0.0, e["mean"] - 0.5 * max(e["std"], 0.05)),
                "specificity": (s["mean"] + 0.5 * max(s["std"], 0.05), 1.0),
            },
            "exploratory": {
                "entropy": (e["mean"] - 0.5 * max(e["std"], 0.05), e["mean"] + 0.5 * max(e["std"], 0.05)),
                "specificity": (s["mean"] - 0.5 * max(s["std"], 0.05), s["mean"] + 0.5 * max(s["std"], 0.05)),
            },
            "diffuse": {
                "entropy": (e["mean"] + 1.5 * max(e["std"], 0.05), 1.0),
            },
            "rigid": {
                "entropy": (0.0, e["p10"]),
                "separation": (sep["mean"] + max(sep["std"], 0.1), 1.0),
            },
            "crisis": {
                "contradiction_density": c["mean"] + 2.0 * max(c["std"], 0.02),
            },
            "_stats": stats,
        }

        self._calibrated_thresholds = calibrated
        return calibrated

    def classify_phase(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify the current cognitive phase based on geometry metrics.

        Returns a dict with:
        - phase: "focused", "exploratory", "diffuse", "rigid", "crisis"
        - confidence: how clearly the system matches the phase (0-1)
        - recommendations: dict of parameter adjustments
        - metrics_used: the metric values that determined the phase
        """
        if metrics is None:
            metrics = self.graph_diagnostics()
        if metrics.get("empty"):
            return {"phase": "empty", "confidence": 1.0, "recommendations": {}, "metrics_used": {}}

        entropy = metrics.get("graph_entropy", 0.5)
        # Use inference specificity if available, otherwise derive from topology
        if metrics.get("inference_count", 0) > 0:
            specificity = metrics.get("inference_specificity_mean", 0.5)
        else:
            # Topology-based proxy: high entropy = low specificity
            specificity = 1.0 - entropy
        separation = metrics.get("relation_separation", 0.0)
        contradiction = metrics.get("contradiction_density", 0.0)

        # Use calibrated thresholds if available, otherwise static defaults
        cal = getattr(self, "_calibrated_thresholds", None)
        if not cal:
            cal = self.calibrate_thresholds()

        # Crisis threshold: calibrated or static
        crisis_thresh = (cal.get("crisis", {}).get("contradiction_density", 0.15)
                         if cal else self.PHASE_THRESHOLDS["crisis"]["contradiction_density"])

        # Crisis takes priority — contradiction overload
        if contradiction > crisis_thresh:
            return {
                "phase": "crisis",
                "confidence": min(1.0, contradiction / 0.3),
                "recommendations": {
                    "inhibition_boost": 0.3,       # strengthen inhibitory edges
                    "plasticity_boost": 0.0,       # don't explore during crisis
                    "sleep_urgency": 0.8,          # sleep to consolidate contradictions
                    "contradiction_focus": True,   # prioritize contradiction resolution
                },
                "metrics_used": {"entropy": entropy, "specificity": specificity,
                                 "separation": separation, "contradiction_density": contradiction},
            }

        # ── Global invariant overrides ──
        # These hard limits prevent boiling frog failure where adaptive
        # thresholds drift to normalize pathology over time.
        inv = self.GLOBAL_INVARIANTS

        # Hard crisis: topology collapsing (preservation below minimum)
        preservation = metrics.get("neighbor_preservation", 1.0)
        if preservation < inv["preservation_min"]:
            return {
                "phase": "crisis",
                "confidence": min(1.0, (inv["preservation_min"] - preservation) / inv["preservation_min"]),
                "recommendations": {
                    "inhibition_boost": 0.2,
                    "plasticity_boost": 0.0,
                    "sleep_urgency": 1.0,
                    "contradiction_focus": False,
                },
                "metrics_used": {"entropy": entropy, "specificity": specificity,
                                 "separation": separation, "contradiction_density": contradiction,
                                 "neighbor_preservation": preservation,
                                 "_global_override": "preservation_min"},
            }

        # Hard diffuse: entropy exceeds absolute ceiling
        if entropy > inv["entropy_max"]:
            return {
                "phase": "diffuse",
                "confidence": min(1.0, (entropy - inv["entropy_max"]) / (1.0 - inv["entropy_max"]) + 0.5),
                "recommendations": {
                    "inhibition_boost": 0.4,
                    "plasticity_boost": 0.0,
                    "sleep_urgency": 0.7,
                    "contradiction_focus": False,
                },
                "metrics_used": {"entropy": entropy, "specificity": specificity,
                                 "separation": separation, "contradiction_density": contradiction,
                                 "_global_override": "entropy_max"},
            }

        # Hard rigid: separation exceeds absolute ceiling
        if separation > inv["separation_max"]:
            return {
                "phase": "rigid",
                "confidence": min(1.0, (separation - inv["separation_max"]) / (1.0 - inv["separation_max"]) + 0.5),
                "recommendations": {
                    "inhibition_boost": 0.0,
                    "plasticity_boost": 0.4,
                    "sleep_urgency": 0.3,
                    "contradiction_focus": False,
                },
                "metrics_used": {"entropy": entropy, "specificity": specificity,
                                 "separation": separation, "contradiction_density": contradiction,
                                 "_global_override": "separation_max"},
            }

        # Score each phase by how well metrics fit
        # Use calibrated thresholds when available
        scores = {}

        if cal and "_stats" in cal:
            # Calibrated scoring: thresholds derived from system's own history
            e_hi = cal["diffuse"]["entropy"][0]
            e_lo = cal["focused"]["entropy"][1]
            s_lo = cal["focused"]["specificity"][0]

            # Focused: entropy below calibrated low, specificity above calibrated high
            if entropy < e_lo and specificity > s_lo:
                scores["focused"] = (1.0 - entropy / e_lo) * specificity

            # Exploratory: near the system's historical mean
            e_stats = cal["_stats"]["entropy"]
            e_dist = abs(entropy - e_stats["mean"]) / (e_stats["std"] + 1e-15)
            if e_dist < 1.0:
                scores["exploratory"] = 1.0 - e_dist

            # Diffuse: entropy above calibrated high
            if entropy > e_hi:
                scores["diffuse"] = (entropy - e_hi) / (1.0 - e_hi + 1e-15) + 0.3

            # Rigid: entropy very low + separation very high (relative to history)
            rigid_sep_hi = cal.get("rigid", {}).get("separation", (0.7, 1.0))[0]
            rigid_e_hi = cal.get("rigid", {}).get("entropy", (0.0, 0.3))[1]
            if entropy < rigid_e_hi and separation > rigid_sep_hi:
                scores["rigid"] = (1.0 - entropy / rigid_e_hi) * separation
        else:
            # Static fallback thresholds
            # Focused: low entropy, high specificity
            if entropy < 0.4 and specificity > 0.7:
                scores["focused"] = (1.0 - entropy) * specificity

            # Exploratory: medium entropy, medium specificity
            if 0.3 < entropy < 0.7 and 0.3 < specificity < 0.7:
                ent_score = 1.0 - abs(entropy - 0.5) * 2
                spec_score = 1.0 - abs(specificity - 0.5) * 2
                scores["exploratory"] = ent_score * spec_score

            # Diffuse: high entropy alone (>0.8) is sufficient
            if entropy > 0.8:
                scores["diffuse"] = entropy * (1.0 - specificity + 0.3)
            elif entropy > 0.6 and specificity < 0.4:
                scores["diffuse"] = entropy * (1.0 - specificity)

            # Rigid: low entropy, very high specificity, high separation
            if entropy < 0.3 and specificity > 0.8 and separation > 0.7:
                scores["rigid"] = (1.0 - entropy) * specificity * separation

        if not scores:
            # Default to exploratory if no clear match
            return {
                "phase": "exploratory",
                "confidence": 0.3,
                "recommendations": {
                    "inhibition_boost": 0.0,
                    "plasticity_boost": 0.0,
                    "sleep_urgency": 0.2,
                    "contradiction_focus": False,
                },
                "metrics_used": {"entropy": entropy, "specificity": specificity,
                                 "separation": separation, "contradiction_density": contradiction},
            }

        phase = max(scores, key=scores.get)
        confidence = min(1.0, scores[phase])

        # Generate recommendations based on phase
        recommendations = {
            "inhibition_boost": 0.0,
            "plasticity_boost": 0.0,
            "sleep_urgency": 0.0,
            "contradiction_focus": False,
        }

        if phase == "diffuse":
            # Semantic fog — increase inhibition to sharpen
            recommendations["inhibition_boost"] = min(0.5, (entropy - 0.6) * 2)
            recommendations["sleep_urgency"] = 0.5
        elif phase == "rigid":
            # Over-separated — increase plasticity to explore
            recommendations["plasticity_boost"] = min(0.5, (separation - 0.7) * 2)
            recommendations["sleep_urgency"] = 0.3
        elif phase == "exploratory":
            # Balanced — mild sleep to consolidate discoveries
            recommendations["sleep_urgency"] = 0.2
        elif phase == "focused":
            # Good state — light maintenance
            recommendations["sleep_urgency"] = 0.1

        return {
            "phase": phase,
            "confidence": confidence,
            "recommendations": recommendations,
            "metrics_used": {"entropy": entropy, "specificity": specificity,
                             "separation": separation, "contradiction_density": contradiction},
        }

    def regulate(self) -> Dict[str, Any]:
        """Run the full regulation pipeline with multi-timescale damping.

        1. Compute geometry diagnostics
        2. Classify cognitive phase
        3. Apply damped regulation via CognitiveRegulator
        4. Apply graph-level effects (inhibition boost)
        5. Meta-adapt: feed recovery feedback to regulator for self-tuning

        Returns: phase info + damped adjustments + regulator status + meta changes
        """
        metrics = self.graph_diagnostics(lightweight=True)
        phase_info = self.classify_phase(metrics)
        adjustments = self._regulator.update(phase_info)

        # Apply inhibition boost to inhibitory edges
        if adjustments.get("inhibition_boost", 0) > 0.01:
            boost = adjustments["inhibition_boost"]
            for edge in self.edges.values():
                if edge.edge_type == "inhibitory":
                    edge.weight = min(1.0, edge.weight + boost * 0.05)
                    edge.confidence = min(1.0, edge.confidence + boost * 0.02)

        # Entropy-driven pruning: when entropy is high, prune weakest edges
        # Topology-aware: protect structurally important edges (bridges, hubs, used paths)
        if metrics.get("graph_entropy", 0) > 0.8 and len(self.edges) > 10:
            prune_threshold = 0.15  # prune edges below this weight
            # Compute structural importance (cached, shared with homeostatic_downscale)
            if not self._si_cache_is_fresh():
                si = self.compute_edge_structural_importance()
                self._update_si_cache(si)
            si = self._structural_importance_cache

            # Only prune edges with low structural importance
            si_threshold = 0.3  # protect edges above this importance
            prunable = [(k, e) for k, e in self.edges.items()
                        if e.weight < prune_threshold
                        and not e.shortcut
                        and e.edge_type != "inhibitory"
                        and si.get(k, 0.0) < si_threshold]
            # Sort by structural importance (prune least important first)
            prunable.sort(key=lambda x: si.get(x[0], 0.0))
            # Prune up to 10% of weak edges per regulation cycle
            max_prune = max(1, len(prunable) // 10)
            for (src, tgt), _ in prunable[:max_prune]:
                self.remove_edge(src, tgt)

        # ── Meta-adaptive regulation: learn from intervention outcomes ──
        meta_changes = {}
        if not hasattr(self, '_prev_entropy'):
            self._prev_entropy = metrics.get("graph_entropy", 0.5)
            self._prev_phase = phase_info["phase"]
            self._overshoot_count = 0
            self._regulation_steps = 0

        self._regulation_steps += 1
        current_entropy = metrics.get("graph_entropy", 0.5)
        prev_entropy = self._prev_entropy

        # Detect local overshoot: did regulation cause entropy to swing past target?
        # Target entropy ~0.5 (balanced). Overshoot = crossed target and went too far.
        target_entropy = 0.5
        overshoot = 0.0
        if prev_entropy > target_entropy and current_entropy < target_entropy:
            # Was above target, now below = possible overshoot
            overshoot = abs(current_entropy - target_entropy)
        elif prev_entropy < target_entropy and current_entropy > target_entropy:
            # Was below target, now above = possible overshoot
            overshoot = abs(current_entropy - target_entropy)

        # Track oscillation: phase changes that reverse previous changes
        if phase_info["phase"] != self._prev_phase:
            self._overshoot_count += 1

        oscillation_rate = self._overshoot_count / max(1, self._regulation_steps)

        # Estimate recovery speed from GeometryHistory if available
        recovery_speed = 0.5  # default: neutral
        if hasattr(self, '_geometry_history') and len(self._geometry_history.snapshots) > 5:
            try:
                entropy_series = self._geometry_history.get_series("graph_entropy")
                if len(entropy_series) >= 5:
                    recent = entropy_series[-5:]
                    # If entropy is stabilizing (low variance), recovery is fast
                    variance = float(np.var(recent))
                    recovery_speed = max(0.0, min(1.0, 1.0 - variance * 10.0))
            except (KeyError, IndexError):
                pass

        # Feed feedback to regulator every 5 steps (not every step — avoid overfitting)
        if self._regulation_steps % 5 == 0:
            meta_changes = self._regulator.meta_adapt(
                overshoot=overshoot,
                recovery_speed=recovery_speed,
                oscillation_rate=oscillation_rate,
            )
            # Decay oscillation counter
            self._overshoot_count = max(0, self._overshoot_count - 1)

        self._prev_entropy = current_entropy
        self._prev_phase = phase_info["phase"]

        return {
            "phase": phase_info,
            "adjustments": adjustments,
            "regulator": self._regulator.status(),
            "meta_changes": meta_changes,
        }

    def adaptive_regulation(self) -> Dict[str, Any]:
        """Apply adaptive regulation based on current cognitive phase.

        Modifies graph parameters to steer toward healthy operating regime.
        Returns the phase classification and adjustments made.
        """
        phase_info = self.classify_phase()
        recs = phase_info["recommendations"]
        adjustments = {}

        # Apply inhibition boost: strengthen all inhibitory edges
        if recs.get("inhibition_boost", 0) > 0:
            boost = recs["inhibition_boost"]
            for edge in self.edges.values():
                if edge.edge_type == "inhibitory":
                    edge.weight = min(1.0, edge.weight + boost * 0.1)
                    edge.confidence = min(1.0, edge.confidence + boost * 0.05)
            adjustments["inhibition_boosted"] = boost

        # Apply plasticity boost: increase edge learning rates temporarily
        # (tracked via a flag that hebbian_update can read)
        if recs.get("plasticity_boost", 0) > 0:
            adjustments["plasticity_boosted"] = recs["plasticity_boost"]

        # Sleep urgency: return for the caller (RLM) to act on
        adjustments["sleep_urgency"] = recs.get("sleep_urgency", 0.0)
        adjustments["contradiction_focus"] = recs.get("contradiction_focus", False)

        phase_info["adjustments"] = adjustments
        return phase_info

    def geometry_report(self) -> str:
        """Human-readable semantic geometry report with phase classification."""
        m = self.graph_diagnostics()
        if m.get("empty"):
            return "Graph is empty."

        phase_info = self.classify_phase(m)
        cal = getattr(self, "_calibrated_thresholds", None)

        lines = [
            "═══ Semantic Geometry Report ═══",
            f"Nodes: {len(self.nodes)}  Edges: {len(self.edges)}",
            "",
            "── Cognitive Phase ──",
            f"  Phase:         {phase_info['phase'].upper()}  (confidence={phase_info['confidence']:.2f})",
            f"  Entropy:       {m['graph_entropy']:.3f}  Specificity: {m.get('inference_specificity_mean', 0):.3f}",
            f"  Separation:    {m['relation_separation']:.3f}  Contradiction: {m['contradiction_density']:.3f}",
            "",
            "── Thresholds ──",
            f"  Mode:        {'CALIBRATED' if cal and '_stats' in cal else 'STATIC'}  ({len(self._geometry_history.snapshots)} history snapshots)",
            f"  Invariants:  entropy_max={self.GLOBAL_INVARIANTS['entropy_max']}  contradiction_max={self.GLOBAL_INVARIANTS['contradiction_max']}  preservation_min={self.GLOBAL_INVARIANTS['preservation_min']}",
            "",
            "── Regulator ──",
            f"  Inhibition boost:  {self._regulator._fast_inhibition_boost:.3f}  (damping={self._regulator._fast_damping:.2f})",
            f"  Sleep urgency:     {self._regulator._medium_sleep_urgency:.3f}  (damping={self._regulator._medium_damping:.2f})",
            f"  Plasticity boost:  {self._regulator._slow_plasticity_boost:.3f}",
            f"  Oscillations:      {self._regulator._oscillation_count}  adjustments={self._regulator._adjustments_made}",
            "",
            "── Activation Field ──",
            f"  Active nodes:      {m['active_count']}",
            f"  Graph entropy:     {m['graph_entropy']:.3f}  (0=focused, 1=diffuse)",
            f"  Activation mean:   {m['activation_mean']:.4f}  std={m['activation_std']:.4f}  max={m['activation_max']:.4f}",
            "",
            "── Topology ──",
            f"  Clustering coeff:  {m['clustering_coefficient']:.3f}",
            f"  Branching factor:  {m['branching_factor']:.1f}  max={m['branching_max']}",
            f"  Edge weight mean:  {m['edge_weight_mean']:.3f}  std={m['edge_weight_std']:.3f}",
            f"  Edge confidence:   {m['edge_confidence_mean']:.3f}",
            "",
            "── Inhibition & Contradiction ──",
            f"  Contradiction density: {m['contradiction_density']:.3f}  ({m['inhibitory_count']} inhibitory edges)",
            "",
            "── Relation Geometry ──",
            f"  Intra-type sim:    {m['relation_intra_similarity']:.3f}",
            f"  Inter-type sim:    {m['relation_inter_similarity']:.3f}",
            f"  Separation:        {m['relation_separation']:.3f}  (positive = distinct clusters)",
            f"  Types: {m['relation_type_distribution']}",
            "",
            "── Identity Stability ──",
            f"  Attractor stability:   {m['attractor_stability']:.3f}  (core vs genesis)",
            f"  Core-active alignment: {m['core_active_alignment']:.3f}  (fast vs slow vector)",
            f"  Neighbor preservation: {m['neighbor_preservation']:.3f}  (1.0=stable, 0.0=churn)",
            f"  Curvature trend:       {m['curvature_trend']:+.4f}  (negative=destabilizing)",
            "",
            "── Path Structure ──",
            f"  Shortcut ratio:    {m['shortcut_ratio']:.3f}  ({m['shortcut_count']} shortcuts)",
            f"  Tracked paths:     {m['tracked_paths']}",
            f"  Path degeneracy:   mean={m['path_degeneracy_mean']:.1f}  max={m['path_degeneracy_max']}",
            "",
            "── Inference Sparsity ──",
            f"  Runs logged:       {m['inference_count']}",
            f"  Specificity:       mean={m['inference_specificity_mean']:.3f}  last={m['inference_specificity_last']:.3f}  (1.0=winner-take-all)",
            f"  Inference branching: mean={m['inference_branching_mean']:.1f} results/run",
            f"  Energy cost:       mean={m['energy_cost_mean']:.0f} edges  last={m['energy_cost_last']}  (fewer=efficient)",
            f"  Activation mass:   mean={m['activation_mass_mean']:.3f}  (cognitive fuel burned)",
        ]
        return "\n".join(lines)

    # ── state ──

    def reset_activation(self):
        for nid in self._active_nodes:
            if nid in self.nodes:
                self.nodes[nid].activation = 0.0
        self._active_nodes.clear()

    def __repr__(self):
        return (f"<ConceptGraph nodes={len(self.nodes)} edges={len(self.edges)} "
                f"dim={self.dim} free_energy={self.total_free_energy:.2f}>")
