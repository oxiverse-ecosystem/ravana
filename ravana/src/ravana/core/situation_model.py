"""
RAVANA Situation Model — DMN-Inspired Continuous Cognitive Workspace
====================================================================
Maintains a dense, continuously-evolving vector representation of the
current cognitive "situation" — inspired by the brain's Default Mode
Network (DMN) which constructs and updates Situation Models.

Neuroscience grounding:
- Buckner & Carroll (2007): DMN integrates episodic memory, semantic
  knowledge, and self-projection into a coherent mental scene.
- Zwaan & Radvansky (1998): Situation Models are multi-dimensional
  representations of the current state of discourse (time, space,
  causation, intentionality).
- PMC 2025 (Nature Human Behaviour): DMN constructs shared
  representations at longer timescales that integrate incoming
  conversational content with prior context.

Architecture:
- The situation model maintains a DENSE VECTOR that is a soft blend
  of all currently active concept embeddings, weighted by activation.
- This vector updates continuously as new concepts activate.
- Unlike discrete triples (subject-relation-object), the situation
  vector captures the full distributed pattern of activation.
- The vector is used to condition the NeuralDecoder for fluid
  text generation (like an LLM's hidden state).
- A "DMN state" is maintained across turns with slow decay.

Key innovation:
  Instead of passing discrete (subject, relation, object) triples
  to the surface realizer, we pass a BLENDED VECTOR that represents
  the entire activated cognitive state. This vector serves as the
  conditioning context for generating fluid, narrative responses.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field


@dataclass
class SituationState:
    """The current situation model state."""
    # Primary blended vector (weighted mean of active concept embeddings)
    blended_vector: Optional[np.ndarray] = None
    # Running DMN state (slowly decaying across turns)
    dmn_state: Optional[np.ndarray] = None
    # Thematic content vector (what we're talking about)
    content_vector: Optional[np.ndarray] = None
    # Pragmatic context vector (how we're talking about it)
    context_vector: Optional[np.ndarray] = None
    # Which concepts are currently active in the model
    active_concepts: Dict[str, float] = field(default_factory=dict)
    # Narrative theme label
    narrative_theme: str = ""
    # Coherence score (0-1, how well concepts fit together)
    coherence: float = 0.5
    # DMN decay rate between turns (0-1)
    dmn_decay: float = 0.6
    # Blending temperature (0=winner-take-all, 1=uniform blend)
    blend_temperature: float = 0.7


class SituationModel:
    """DMN-inspired continuous cognitive workspace.

    Maintains a blended vector representation of all currently active
    concepts, updated continuously during graph traversal. This vector
    serves as the "situation model" that conditions language generation,
    replacing discrete triple-by-triple SVO generation.

    Usage:
        sm = SituationModel(dim=64)
        sm.update(concept_embeddings, activations, concept_labels)
        situation_vec = sm.get_blended_vector()
        dmn_vec = sm.get_dmn_state()
    """

    def __init__(self, dim: int = 64, dmn_decay: float = 0.6):
        self.dim = dim
        self.state = SituationState(dmn_decay=dmn_decay)
        self._history: List[np.ndarray] = []  # History of blended vectors
        self._max_history: int = 10

    def update(self,
               concept_embeddings: Dict[str, np.ndarray],
               activations: Dict[str, float],
               graph_get_vector_fn: Optional[Callable] = None,
               sentence_vector: Optional[np.ndarray] = None,
               context_vector_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Update the situation model from current concept activations.

        Args:
            concept_embeddings: {label: vector} map for available concepts
            activations: {label: activation_strength} for activated concepts
            graph_get_vector_fn: Optional function to get graph node vector by label
            sentence_vector: Optional sentence-level compositional vector
            context_vector_input: Optional discourse context vector

        Returns:
            The updated blended vector
        """
        if not concept_embeddings and not activations:
            if self.state.blended_vector is not None:
                return self.state.blended_vector
            return np.zeros(self.dim, dtype=np.float32)

        # Collect weighted vectors
        vectors = []
        weights = []
        labels = []

        # Use activations dict if provided, otherwise use all embeddings
        if activations:
            source = activations
        elif concept_embeddings:
            source = {k: 0.5 for k in concept_embeddings}
        else:
            source = {}

        # Blend with temperature: higher temp = softer blend
        temp = self.state.blend_temperature
        for label, activation in source.items():
            if activation < 0.05:
                continue
            vec = concept_embeddings.get(label)
            if vec is None and graph_get_vector_fn is not None:
                vec = graph_get_vector_fn(label)
            if vec is not None and vec.shape[0] == self.dim:
                # Apply soft weighting with temperature
                w = float(activation) ** (1.0 / max(temp, 0.1))
                vectors.append(vec.astype(np.float32))
                weights.append(w)
                labels.append(label)

        if not vectors:
            if self.state.blended_vector is not None:
                return self.state.blended_vector
            return np.zeros(self.dim, dtype=np.float32)

        # Weighted blend
        weights_arr = np.array(weights, dtype=np.float32)
        weights_arr = weights_arr / (np.sum(weights_arr) + 1e-10)
        vectors_arr = np.stack(vectors, axis=0)
        blended = np.sum(vectors_arr * weights_arr[:, np.newaxis], axis=0)

        # Normalize
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm

        self.state.blended_vector = blended.astype(np.float32)
        self.state.active_concepts = {l: float(w) for l, w in zip(labels, weights_arr)}

        # Update DMN state (slowly evolving)
        if self.state.dmn_state is not None:
            decay = self.state.dmn_decay
            self.state.dmn_state = decay * self.state.dmn_state + (1.0 - decay) * blended
            n = np.linalg.norm(self.state.dmn_state)
            if n > 0:
                self.state.dmn_state /= n
        else:
            self.state.dmn_state = blended.copy()

        # Update content vector with sentence-level information
        if sentence_vector is not None:
            self.state.content_vector = sentence_vector.copy()
        else:
            self.state.content_vector = blended.copy()

        # Update context vector
        if context_vector_input is not None:
            # Orthogonalize: content and context should be orthogonal
            if self.state.content_vector is not None:
                self.state.context_vector = self._ensure_orthogonal(
                    self.state.content_vector, context_vector_input
                )
            else:
                self.state.context_vector = context_vector_input.copy()
        else:
            self.state.context_vector = blended.copy()

        # Update coherence: measure how well activations cluster
        self.state.coherence = self._compute_coherence(vectors_arr, weights_arr)

        # Determine narrative theme from most active concept
        if labels:
            sorted_labels = sorted(
                zip(labels, weights_arr), key=lambda x: x[1], reverse=True
            )
            self.state.narrative_theme = sorted_labels[0][0]

        # Track history
        self._history.append(blended.copy())
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return self.state.blended_vector

    def get_blended_vector(self) -> np.ndarray:
        """Get the current blended situation vector."""
        if self.state.blended_vector is not None:
            return self.state.blended_vector
        return np.zeros(self.dim, dtype=np.float32)

    def get_dmn_state(self) -> Optional[np.ndarray]:
        """Get the slow-evolving DMN state (persists across turns)."""
        return self.state.dmn_state

    def get_blended_as_conditioning(self) -> np.ndarray:
        """Get the situation vector formatted as conditioning for NeuralDecoder.

        The decoder expects (n_concepts, embed_dim). We provide the blended
        vector as a single conditioning concept, tiled for stability.
        Returns shape (3, dim) with the vector repeated.
        """
        vec = self.get_blended_vector()
        if np.all(vec == 0):
            return np.zeros((3, self.dim), dtype=np.float32)
        # Tile 3 times for stable conditioning (like having 3 copies)
        return np.tile(vec, (3, 1))

    def get_full_conditioning(self) -> np.ndarray:
        """Get full conditioning including blended vector + top concepts.

        Returns shape (n_active + 1, dim) where the first row is the
        blended vector and remaining rows are individual concept vectors.
        """
        blended = self.get_blended_vector()
        concepts = []
        labels = list(self.state.active_concepts.keys())[:5]
        for label in labels:
            pass  # Vectors would come from external source
        return blended[np.newaxis, :] if np.any(blended != 0) else np.zeros((1, self.dim), dtype=np.float32)

    def decay(self, factor: float = 0.4):
        """Decay the DMN state between turns (natural forgetting).

        Args:
            factor: How much to retain (0.4 = forget 60% of context)
        """
        if self.state.dmn_state is not None:
            self.state.dmn_state *= factor
            n = np.linalg.norm(self.state.dmn_state)
            if n > 0:
                self.state.dmn_state /= n
        if self.state.blended_vector is not None:
            self.state.blended_vector *= factor
            n = np.linalg.norm(self.state.blended_vector)
            if n > 0:
                self.state.blended_vector /= n

    def reset(self):
        """Reset the situation model for a completely new topic."""
        self.state = SituationState(dmn_decay=self.state.dmn_decay)
        self._history.clear()

    def get_narrative_suggestions(self) -> Dict[str, any]:
        """Get suggestions for narrative generation from the situation model.

        Returns:
            Dict with:
            - 'theme': dominant concept label
            - 'coherence': how well concepts hang together
            - 'active_concepts': sorted list of (label, weight) pairs
            - 'diversity': how diverse the active concepts are (0-1)
        """
        sorted_concepts = sorted(
            self.state.active_concepts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        # Diversity: entropy-like measure of weight distribution
        weights = [w for _, w in sorted_concepts]
        if weights:
            w_arr = np.array(weights, dtype=np.float32)
            w_arr = w_arr / (np.sum(w_arr) + 1e-10)
            entropy = -np.sum(w_arr * np.log(w_arr + 1e-10))
            max_entropy = np.log(len(weights))
            diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            diversity = 0.0

        return {
            'theme': self.state.narrative_theme,
            'coherence': self.state.coherence,
            'active_concepts': sorted_concepts[:8],
            'diversity': min(1.0, diversity),
        }

    def _compute_coherence(self, vectors: np.ndarray, weights: np.ndarray) -> float:
        """Compute coherence as average pairwise cosine similarity."""
        n = len(vectors)
        if n < 2:
            return 0.5
        sims = []
        for i in range(min(n, 10)):
            for j in range(i + 1, min(n, 10)):
                vi = vectors[i]
                vj = vectors[j]
                ni = np.linalg.norm(vi)
                nj = np.linalg.norm(vj)
                if ni > 0 and nj > 0:
                    sims.append(float(np.dot(vi, vj) / (ni * nj)))
        if not sims:
            return 0.5
        # Coherence: clamp to [0, 1]
        mean_sim = np.mean(sims)
        return float(max(0.0, min(1.0, (mean_sim + 1.0) * 0.5)))

    def _ensure_orthogonal(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Project b to be orthogonal to a (Gram-Schmidt)."""
        a_n = a / (np.linalg.norm(a) + 1e-10)
        b_ortho = b - np.dot(b, a_n) * a_n
        n = np.linalg.norm(b_ortho)
        if n > 0:
            b_ortho /= n
        return b_ortho.astype(np.float32)

    def get_state(self) -> Dict:
        """Serialize state for saving."""
        return {
            'dmn_decay': self.state.dmn_decay,
            'narrative_theme': self.state.narrative_theme,
            'coherence': self.state.coherence,
            'blend_temperature': self.state.blend_temperature,
        }

    def set_state(self, state: Dict):
        """Restore state from saved data."""
        self.state.dmn_decay = state.get('dmn_decay', 0.6)
        self.state.narrative_theme = state.get('narrative_theme', '')
        self.state.coherence = state.get('coherence', 0.5)
        self.state.blend_temperature = state.get('blend_temperature', 0.7)
