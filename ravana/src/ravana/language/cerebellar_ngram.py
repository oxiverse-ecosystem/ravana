"""
RAVANA Cerebellar N-gram — Sparse Sequence Learning
====================================================
Learns which concept sequences produce grammatical utterances through
Hebbian co-activation. Models the cerebellum's role in sequence timing
(Ito 2008, Doya 2000).

KEY DESIGN DECISIONS:
- SPARSE storage only (Dict[str, Dict[str, float]]), NOT a dense 10000x10000 matrix
  which would be 400MB. The brain stores sparse synaptic weights for observed
  sequences, not all possible word pairs.
- Bigram (concept_i → concept_j) and trigram (concept_i, concept_j → concept_k)
  transition probabilities.
- Function word prediction: learns which grammatical words ("is", "are", "the")
  tend to appear between specific concept pairs.
- Hebbian update: strengthen transitions that lead to successful utterances.
- Predictions feed into the BasalGangliaGate as an additional modulator signal.
- Integrates with the existing _cerebellar_ngram dict infrastructure.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CerebellarState:
    """Snapshot of cerebellar learning for diagnostics."""
    total_bigram_entries: int = 0
    total_trigram_entries: int = 0
    total_function_word_entries: int = 0
    avg_confidence: float = 0.0
    most_learned_transitions: List[Tuple[str, str, float]] = field(default_factory=list)


class CerebellarNgram:
    """Sparse n-gram model for concept transition learning.

    Stores only observed transitions. Uses Hebbian-style updates:
    weight += learning_rate * activation_i * activation_j
    """

    def __init__(self, decay_rate: float = 0.01, learning_rate: float = 0.05):
        # Sparse bigram: concept_from -> {concept_to -> weight}
        self.bigram: Dict[str, Dict[str, float]] = {}

        # Sparse trigram: (concept_i, concept_j) -> {concept_k -> weight}
        self.trigram: Dict[Tuple[str, str], Dict[str, float]] = {}

        # Function word predictions: (concept_from, concept_to) -> {func_word -> prob}
        # e.g., ("trust", "knowledge") -> {"is": 0.8, "are": 0.1, ...}
        self.function_word_probs: Dict[Tuple[str, str], Dict[str, float]] = {}

        # Depth tracking: how many hops typically follow from a concept
        self.depth: Dict[str, float] = {}

        # Parameters
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate

        # POS-based agreement patterns (initialized from _concept_pos)
        # e.g., singular noun → "is", plural noun → "are"
        self._pos_agreement: Dict[str, str] = {}

    # ─── Initialization ───

    def seed_from_pos(self, concept_pos: Dict[str, str]):
        """Pre-populate basic agreement patterns from concept POS tags."""
        self._pos_agreement = concept_pos.copy()

    # ─── Learning ───

    def learn_chain(self, chain_labels: List[str], successful: bool = True,
                    chain_hops: Optional[List[Tuple[str, str]]] = None):
        """Learn from a completed chain walk.

        Hebbian update: if successful, strengthen bigram/trigram weights.
        If unsuccessful (user corrective feedback), weaken them.

        Args:
            chain_labels: List of concept labels in sequence (including connectors)
            successful: Whether the utterance was well-received
            chain_hops: Optional list of (from, to) concept pairs (excluding connectors)
        """
        # Issue 6: Cerebellar n-gram diversity — track rejected transitions too.
        # When successful=False, the transition strength between template phrases
        # decreases, making them less likely to be chosen in the future.
        lr = self.learning_rate if successful else -self.learning_rate * 0.5

        # Learn from chain hops (concept-to-concept transitions without connectors)
        if chain_hops:
            self._learn_hops(chain_hops, lr)

        # Learn from raw chain labels (includes connectors like "is", "connect")
        self._learn_label_sequence(chain_labels, lr)

        # Decay all weights slightly to prevent unbounded growth
        self._apply_decay()

    def _learn_hops(self, hops: List[Tuple[str, str]], lr: float):
        """Learn from (from_label, to_label) concept hops."""
        for i, (from_lbl, to_lbl) in enumerate(hops):
            fl, tl = from_lbl.lower(), to_lbl.lower()

            # Bigram update
            if fl not in self.bigram:
                self.bigram[fl] = {}
            current = self.bigram[fl].get(tl, 0.0)
            self.bigram[fl][tl] = max(0.0, min(1.0, current + lr))

            # Depth tracking
            remaining = len(hops) - i - 1
            current_depth = self.depth.get(fl, 0.0)
            self.depth[fl] = current_depth * 0.7 + remaining * 0.3

            # Trigram update (need 2+ hops)
            if i > 0:
                prev_from, prev_to = hops[i - 1]
                trigram_key = (prev_from.lower(), fl)
                if trigram_key not in self.trigram:
                    self.trigram[trigram_key] = {}
                curr_tri = self.trigram[trigram_key].get(tl, 0.0)
                self.trigram[trigram_key][tl] = max(0.0, min(1.0, curr_tri + lr))

    def _learn_label_sequence(self, labels: List[str], lr: float):
        """Learn from raw label sequence (including connector words like 'is')."""
        for i in range(len(labels) - 1):
            c_i = labels[i].lower()
            c_j = labels[i + 1].lower()

            # Bigram on raw labels
            if c_i not in self.bigram:
                self.bigram[c_i] = {}
            current = self.bigram[c_i].get(c_j, 0.0)
            self.bigram[c_i][c_j] = max(0.0, min(1.0, current + lr * 0.3))

            # Trigram
            if i > 0:
                c_prev = labels[i - 1].lower()
                tri_key = (c_prev, c_i)
                if tri_key not in self.trigram:
                    self.trigram[tri_key] = {}
                curr_tri = self.trigram[tri_key].get(c_j, 0.0)
                self.trigram[tri_key][c_j] = max(0.0, min(1.0, curr_tri + lr * 0.3))

    def learn_function_word(self, from_concept: str, to_concept: str,
                            function_word: str, increment: float = 0.1):
        """Learn that a function word (e.g., 'is', 'are', 'the') appeared
        between two concepts."""
        key = (from_concept.lower(), to_concept.lower())
        if key not in self.function_word_probs:
            self.function_word_probs[key] = {}
        current = self.function_word_probs[key].get(function_word.lower(), 0.0)
        self.function_word_probs[key][function_word.lower()] = min(1.0, current + increment)

    def _apply_decay(self):
        """Apply gentle decay to all weights (prevents unbounded growth)."""
        decay = self.decay_rate

        # Decay bigram
        for from_lbl in list(self.bigram.keys()):
            for to_lbl in list(self.bigram[from_lbl].keys()):
                self.bigram[from_lbl][to_lbl] = max(0.0, self.bigram[from_lbl][to_lbl] - decay)
                if self.bigram[from_lbl][to_lbl] <= 0:
                    del self.bigram[from_lbl][to_lbl]
            if not self.bigram[from_lbl]:
                del self.bigram[from_lbl]

        # Decay trigram (sample only for performance)
        for tri_key in list(self.trigram.keys()):
            for to_lbl in list(self.trigram[tri_key].keys()):
                self.trigram[tri_key][to_lbl] = max(0.0, self.trigram[tri_key][to_lbl] - decay)
                if self.trigram[tri_key][to_lbl] <= 0:
                    del self.trigram[tri_key][to_lbl]
            if not self.trigram[tri_key]:
                del self.trigram[tri_key]

    # ─── Prediction ───

    def predict_next(self, current: str,
                     previous: Optional[str] = None,
                     pos: Optional[str] = None,
                     top_k: int = 5) -> Dict[str, float]:
        """Predict the next concept given current context.

        Combines bigram and trigram predictions with learned weights.
        Returns dict of {concept: score} sorted by likelihood.

        Args:
            current: Current concept label
            previous: Previous concept (for trigram)
            pos: Part-of-speech constraint (only return concepts with this POS)
            top_k: Maximum number of predictions

        Returns:
            Dict of {concept_label: score} sorted descending
        """
        cl = current.lower()
        scores: Dict[str, float] = {}

        # Bigram predictions
        if cl in self.bigram:
            for target, weight in self.bigram[cl].items():
                scores[target] = scores.get(target, 0.0) + weight * 0.6  # bigram weight

        # Trigram predictions (if previous is known)
        if previous and (previous.lower(), cl) in self.trigram:
            tri_key = (previous.lower(), cl)
            for target, weight in self.trigram[tri_key].items():
                scores[target] = scores.get(target, 0.0) + weight * 0.4  # trigram weight

        # Apply POS constraint if provided
        if pos:
            # We can't filter by POS here since we don't have access to _concept_pos
            # This will be applied externally
            pass

        # Sort by score descending
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return sorted_scores

    def predict_function_word(self, from_concept: str, to_concept: str) -> Optional[str]:
        """Predict the most likely function word between two concepts.

        If no learned pattern exists, returns None (caller should use default).
        """
        key = (from_concept.lower(), to_concept.lower())
        if key in self.function_word_probs:
            best_word = max(self.function_word_probs[key], key=self.function_word_probs[key].get)
            return best_word
        return None

    def get_transition_strength(self, from_concept: str, to_concept: str) -> float:
        """Get the cerebellar prediction strength for a transition.

        Used by the BasalGangliaGate to bias concept selection toward
        grammatically-proven transitions.
        """
        fl, tl = from_concept.lower(), to_concept.lower()
        strength = 0.0
        if fl in self.bigram and tl in self.bigram[fl]:
            strength += self.bigram[fl][tl] * 0.6
        # Check any trigram key that starts with fl
        for (pf, cf), targets in self.trigram.items():
            if cf == fl and tl in targets:
                strength += targets[tl] * 0.4
                break
        return strength

    def get_expected_depth(self, label: str) -> float:
        """Get expected number of remaining hops from this concept."""
        return self.depth.get(label.lower(), 0.0)

    # ─── State Management ───

    def get_state(self) -> Dict:
        """Get full state for serialization."""
        # Convert trigram tuple keys to string for serialization
        serialized_trigram = {}
        for (a, b), targets in self.trigram.items():
            key_str = f"{a}|{b}"
            serialized_trigram[key_str] = targets

        serialized_fw = {}
        for (a, b), probs in self.function_word_probs.items():
            key_str = f"{a}|{b}"
            serialized_fw[key_str] = probs

        return {
            'bigram': self.bigram,
            'trigram': serialized_trigram,
            'function_word_probs': serialized_fw,
            'depth': self.depth,
        }

    def set_state(self, state: Dict):
        """Restore state from serialized data."""
        self.bigram = state.get('bigram', {})

        # Deserialize trigram string keys back to tuples
        self.trigram = {}
        for key_str, targets in state.get('trigram', {}).items():
            parts = key_str.split('|')
            if len(parts) == 2:
                self.trigram[(parts[0], parts[1])] = targets

        # Deserialize function word keys
        self.function_word_probs = {}
        for key_str, probs in state.get('function_word_probs', {}).items():
            parts = key_str.split('|')
            if len(parts) == 2:
                self.function_word_probs[(parts[0], parts[1])] = probs

        self.depth = state.get('depth', {})

    def get_stats(self) -> CerebellarState:
        """Get diagnostic statistics."""
        total_bigram = sum(len(targets) for targets in self.bigram.values())
        total_trigram = sum(len(targets) for targets in self.trigram.values())
        total_fw = sum(len(probs) for probs in self.function_word_probs.values())

        # Find top transitions
        all_transitions = []
        for fl, targets in self.bigram.items():
            for tl, weight in sorted(targets.items(), key=lambda x: x[1], reverse=True)[:3]:
                all_transitions.append((fl, tl, weight))
        top = sorted(all_transitions, key=lambda x: x[2], reverse=True)[:10]

        avg_conf = np.mean([t[2] for t in all_transitions]) if all_transitions else 0.0

        return CerebellarState(
            total_bigram_entries=total_bigram,
            total_trigram_entries=total_trigram,
            total_function_word_entries=total_fw,
            avg_confidence=avg_conf,
            most_learned_transitions=top,
        )
