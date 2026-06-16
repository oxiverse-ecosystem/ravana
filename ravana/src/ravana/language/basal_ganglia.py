"""
RAVANA Basal Ganglia Gate — Biologically-Plausible Concept Selection
=====================================================================
Replaces temperature-based softmax selection with Go/NoGo competitive gating
inspired by the GODIVA model (Guenther 2016) and cortico-basal ganglia-thalamocortical (CBGTC) loops.

KEY DESIGN INSIGHT:
The 15+ existing modulators in _walk_chain (VAD emotion, activation fatigue,
novelty, PFC gating, contradiction penalty, subject proximity, etc.) are NOT
eliminated. They are REFRAMED as inputs that set the Basal Ganglia Gate's
parameters — Go threshold, NoGo lateral inhibition strength, and dopamine tone.
This prevents the "modulator collision" problem where additive bonuses fight each other.

ARCHITECTURE:
  Input Signals (15 modulators) → BG Gate Parameters → Competitive Selection
       
  Direct Pathway (Go):   go_score = edge_weight × confidence × (1 + dopamine_tone)
                         Raises activation for the candidate concept.
                         
  Indirect Pathway (NoGo): no_go_score = lateral_inhibition × sum(competitor_scores)
                           Suppresses competing candidates.
                           
  Dynamic Threshold:      effective_go_threshold = base_threshold × (1 - arousal × 0.3)
                          Modified by: novelty, exploration_drive, prediction_error
                          
  Winner Selection:       Candidate with highest go_score ABOVE threshold wins.
                          If none above threshold → lowest-temperature softmax fallback.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class BasalGangliaGate:
    """Go/NoGo gating mechanism for concept selection in chain walking.

    The gate replaces the final temperature-based softmax with a competitive
    selection process. All upstream modulators (VAD, fatigue, novelty, etc.)
    set the gate's operating parameters rather than multiplicatively fighting
    each other.
    """

    def __init__(self,
                 base_go_threshold: float = 0.25,
                 base_no_go_strength: float = 0.4,
                 dopamine_tone: float = 0.5,
                 lateral_inhibition_range: float = 0.6,
                 min_candidates_for_gating: int = 2):
        self.base_go_threshold = base_go_threshold
        self.base_no_go_strength = base_no_go_strength
        self.dopamine_tone = dopamine_tone
        self.lateral_inhibition_range = lateral_inhibition_range
        self.min_candidates_for_gating = min_candidates_for_gating

        # Gate state (updated per-hop)
        self.current_arousal: float = 0.3
        self.current_novelty: float = 0.0
        self.current_exploration_drive: float = 0.0
        self.current_prediction_error: float = 0.0
        self.current_identity_strength: float = 0.5
        self.current_fatigue_level: float = 0.0
        self.current_prefrontal_boost: float = 0.0
        self.current_thalamic_salience: float = 0.0
        self.current_subject_proximity_bonus: float = 0.0
        self.current_contradiction_penalty: float = 0.6

        # Statistics
        self.total_gate_hits: int = 0
        self.total_softmax_fallbacks: int = 0
        self.go_threshold_history: List[float] = []

    # ─── Parameter Setting: Each modulator sets a specific gate parameter ───

    def set_arousal(self, arousal: float):
        """Arousal lowers the Go threshold → more exploratory selection.
        Low arousal → high threshold → conservative (predictable).
        High arousal → low threshold → creative (exploratory)."""
        self.current_arousal = np.clip(arousal, 0.0, 1.0)

    def set_novelty(self, novelty: float):
        """Novelty lowers Go threshold → favors unexplored paths."""
        self.current_novelty = np.clip(novelty, 0.0, 1.0)

    def set_exploration_drive(self, drive: float):
        """Exploration drive reduces NoGo inhibition → allows more candidates through."""
        self.current_exploration_drive = np.clip(drive, 0.0, 1.0)

    def set_prediction_error(self, error: float):
        """High prediction error → raise Go threshold (be conservative when uncertain).
        Low prediction error → lower Go threshold (explore when things are going well)."""
        self.current_prediction_error = np.clip(error, 0.0, 1.0)

    def set_identity_strength(self, strength: float):
        """High identity strength → raise Go threshold (stable personality → conservative speech).
        Low identity strength → lower Go threshold (exploratory, trying on personas)."""
        self.current_identity_strength = np.clip(strength, 0.0, 1.0)

    def set_fatigue(self, fatigue_level: float):
        """Activation fatigue raises NoGo strength for recently-traversed concepts."""
        self.current_fatigue_level = np.clip(fatigue_level, 0.0, 1.0)

    def set_prefrontal_boost(self, boost: float):
        """PFC gating: boost from working memory content (keeps conversation on-topic)."""
        self.current_prefrontal_boost = np.clip(boost, 0.0, 1.0)

    def set_thalamic_salience(self, salience: float):
        """Thalamic gating: salience threshold for broadcasting to cortex."""
        self.current_thalamic_salience = np.clip(salience, 0.0, 1.0)

    def set_subject_proximity_bonus(self, bonus: float):
        """Subject proximity: bonus for concepts close to the original topic."""
        self.current_subject_proximity_bonus = np.clip(bonus, 0.0, 1.0)

    def set_contradiction_penalty(self, penalty: float):
        """Contradiction penalty: suppress concepts that contradict beliefs."""
        self.current_contradiction_penalty = np.clip(penalty, 0.0, 1.0)

    def set_dopamine_tone(self, tone: float):
        """Dopamine modulation: high tone → high Go threshold (conservative).
        Low tone → low Go threshold (exploratory, novelty-seeking).
        Models tonic dopamine's effect on BG direct pathway."""
        self.dopamine_tone = np.clip(tone, 0.0, 1.0)

    def set_all_from_modulators(self, modulator_dict: Dict[str, float]):
        """Batch-set all gate parameters from a dictionary of modulator signals.
        
        Expected keys (all optional):
            arousal, novelty, exploration_drive, prediction_error, identity_strength,
            fatigue_level, prefrontal_boost, thalamic_salience, subject_proximity_bonus,
            contradiction_penalty, dopamine_tone
        """
        for key, value in modulator_dict.items():
            setter = getattr(self, f"set_{key}", None)
            if setter:
                setter(value)

    # ─── Core Gating Logic ───

    def compute_effective_go_threshold(self) -> float:
        """Compute the dynamic Go threshold from all modulator inputs.

        The threshold determines how strong a candidate must be to pass the gate.
        Lower threshold = more concepts get through (creative).
        Higher threshold = fewer concepts (conservative).

        Modulator mapping:
          - Arousal:       lowers threshold  (aroused → creative speech)
          - Novelty:       lowers threshold  (novel situation → explore)
          - Pred error:    raises threshold  (uncertain → be conservative)
          - Identity str:  raises threshold  (strong self → stable speech)
        """
        threshold = self.base_go_threshold

        # Arousal: high arousal = low threshold (more exploratory)
        threshold *= (1.0 - self.current_arousal * 0.3)

        # Novelty: high novelty = lower threshold
        threshold *= (1.0 - self.current_novelty * 0.2)

        # Prediction error: high error = higher threshold (conservative when uncertain)
        threshold *= (1.0 + self.current_prediction_error * 0.4)

        # Identity strength: strong identity = higher threshold (stable)
        threshold *= (1.0 + self.current_identity_strength * 0.15)

        # Exploration drive: high drive = lower threshold
        threshold *= (1.0 - self.current_exploration_drive * 0.25)

        # Dopamine tone: high dopamine = higher threshold (satisfied → conservative)
        threshold *= (1.0 + self.dopamine_tone * 0.2)

        # Clamp
        effective = np.clip(threshold, 0.05, 0.95)
        self.go_threshold_history.append(effective)
        return effective

    def compute_effective_no_go_strength(self) -> float:
        """Compute the dynamic NoGo lateral inhibition strength.

        NoGo strength determines how strongly competing concepts suppress
        each other. Strong NoGo = single dominant winner. Weak NoGo = multiple
        candidates can coexist.

        Modulator mapping:
          - Fatigue:       raises NoGo (recently used concepts suppressed)
          - Exploration:   lowers NoGo (allow diverse candidates)
          - Prefrontal:    raises NoGo for non-PFC concepts
          - Thalamic:      raises NoGo for low-salience concepts
        """
        strength = self.base_no_go_strength

        # Fatigue: tired concepts get extra NoGo suppression
        strength *= (1.0 + self.current_fatigue_level * 0.5)

        # Exploration: want diversity → reduce NoGo
        strength *= (1.0 - self.current_exploration_drive * 0.3)

        # Prefrontal boost: PFC content gets less NoGo
        strength *= (1.0 - self.current_prefrontal_boost * 0.4)

        # Thalamic salience: low-salience concepts get more NoGo
        strength *= (1.0 + (1.0 - self.current_thalamic_salience) * 0.3)

        return np.clip(strength, 0.05, 0.95)

    def select_concept(self,
                       candidates: List[Tuple[str, float, float, str]],
                       rng: Optional[np.random.RandomState] = None
                       ) -> Tuple[str, str, float]:
        """Select a concept via Go/NoGo competitive gating.

        Args:
            candidates: List of (label, score, confidence, relation_type) tuples.
                        The 'score' is the raw edge weight × confidence (pre-modulation).
            rng: Optional RandomState for fallback softmax.

        Returns:
            (selected_label, selected_relation_type, go_score)
            If no candidate passes the gate, returns ("", "", 0.0)
        """
        if not candidates:
            return ("", "", 0.0)

        if len(candidates) < self.min_candidates_for_gating:
            # Too few candidates: just pick the best one directly
            best = max(candidates, key=lambda c: c[1])
            return (best[0], best[3], best[1])

        # Step 1: Compute GO scores (Direct Pathway)
        # go_score = raw_score × confidence × (1 + dopamine_tone)
        go_scores = []
        for label, raw_score, confidence, rel_type in candidates:
            go_mod = 1.0 + self.dopamine_tone * 0.3
            go_score = raw_score * confidence * go_mod

            # Apply subject proximity bonus
            go_score *= (1.0 + self.current_subject_proximity_bonus * 0.3)

            # Apply contradiction penalty
            go_score *= (1.0 - self.current_contradiction_penalty * 0.3)

            # Apply PFC boost
            go_score *= (1.0 + self.current_prefrontal_boost * 0.2)

            go_scores.append((label, go_score, confidence, rel_type))

        # Step 2: Apply NoGo lateral inhibition (Indirect Pathway)
        # Each candidate suppresses all others proportionally to their GO scores
        effective_no_go = self.compute_effective_no_go_strength()
        n_candidates = len(go_scores)

        for i in range(n_candidates):
            label_i, go_i, conf_i, rel_i = go_scores[i]
            total_inhibition = 0.0
            for j in range(n_candidates):
                if i == j:
                    continue
                _, go_j, _, _ = go_scores[j]
                # Lateral inhibition: competitor's GO × NoGo strength
                total_inhibition += go_j * effective_no_go

            # Apply inhibition (normalized by number of competitors)
            if n_candidates > 1:
                inhibition = total_inhibition / (n_candidates - 1)
                # Clamp inhibition so we never go below 0
                net_score = max(0.0, go_i - inhibition * self.lateral_inhibition_range * 0.5)
                go_scores[i] = (label_i, net_score, conf_i, rel_i)

        # Step 3: Dynamic threshold gating
        effective_threshold = self.compute_effective_go_threshold()

        # Find candidates above threshold
        above_threshold = [(l, s, c, r) for l, s, c, r in go_scores if s >= effective_threshold]

        if above_threshold:
            # Pick winner: highest GO score among above-threshold candidates
            winner = max(above_threshold, key=lambda x: x[1])
            self.total_gate_hits += 1
            return (winner[0], winner[3], winner[1])

        # Step 4: Fallback — no candidate above threshold
        # Use lowest-temperature softmax for last-resort selection
        self.total_softmax_fallbacks += 1

        if rng is None:
            rng = np.random.RandomState()

        labels = [c[0] for c in go_scores]
        scores = np.array([c[1] for c in go_scores], dtype=np.float64)

        # Near-deterministic: temperature extremely low (0.01)
        scores = scores - scores.max()
        exp_scores = np.exp(scores / 0.01)
        probs = exp_scores / exp_scores.sum()

        try:
            idx = rng.choice(len(labels), p=probs)
        except Exception:
            idx = int(np.argmax(scores))

        return (labels[idx], go_scores[idx][3], float(scores[idx]))

    def get_stats(self) -> Dict[str, Any]:
        """Return gate diagnostics."""
        return {
            "gate_hits": self.total_gate_hits,
            "softmax_fallbacks": self.total_softmax_fallbacks,
            "current_go_threshold": self.compute_effective_go_threshold(),
            "current_no_go_strength": self.compute_effective_no_go_strength(),
            "dopamine_tone": self.dopamine_tone,
            "avg_go_threshold": float(np.mean(self.go_threshold_history[-50:])) if self.go_threshold_history else 0.0,
            "modulator_state": {
                "arousal": self.current_arousal,
                "novelty": self.current_novelty,
                "prediction_error": self.current_prediction_error,
                "identity_strength": self.current_identity_strength,
                "fatigue_level": self.current_fatigue_level,
                "exploration_drive": self.current_exploration_drive,
            },
        }
