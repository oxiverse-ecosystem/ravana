"""
RAVANA v2 — PHASE J.1: Occam Layer
Hypothesis discipline through explicit complexity penalties.

PRINCIPLE: A disciplined scientist prefers simple explanations that explain enough,
not complex explanations that explain everything.

Without this: RAVANA becomes a conspiracy theorist (fitting noise with elaborate stories)
With this: RAVANA becomes a disciplined scientist (trading off fit vs complexity)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import deque


@dataclass
class OccamConfig:
    """Configuration for Occam pressure system."""
    # Complexity penalty coefficient (λ) - higher = more skeptical
    lambda_penalty: float = 0.3
    
    # Belief budget: max number of active hypotheses
    max_hypotheses: int = 5
    
    # Min evidence needed before complexity matters
    min_evidence_before_penalty: int = 10
    
    # Pruning threshold: remove hypotheses below this score
    prune_threshold: float = -0.5
    
    # Grace period: don't prune young hypotheses (episodes)
    min_age_to_prune: int = 20
    
    # Overfitting detection: rapid complexity increase = danger
    complexity_jump_threshold: float = 0.2


@dataclass
class HypothesisScore:
    """Complete scoring of a hypothesis."""
    hypothesis_id: str
    
    # Components
    explanatory_power: float = 0.0  # How well it fits observations
    complexity: float = 0.0  # Structural complexity (0-1)
    stability: float = 0.0  # Consistency over time (0-1)
    evidence_count: int = 0  # How much data supports it
    age: int = 0  # Episodes since creation
    
    # Computed score
    raw_score: float = 0.0  # Before penalty
    occam_score: float = 0.0  # After complexity penalty
    
    # Diagnostics
    penalty_applied: float = 0.0
    reason: str = ""


class OccamLayer:
    """
    🔒 Discipline layer: prevents epistemic overfitting.
    
    Core mechanism:
    score = explanatory_power - λ * complexity * (evidence_factor)
    
    where evidence_factor increases with data (penalty grows as evidence mounts)
    """
    
    def __init__(self, config: Optional[OccamConfig] = None):
        self.config = config or OccamConfig()
        self.score_history: Dict[str, List[float]] = {}
        self.pruned_hypotheses: List[str] = []
        self.overfitting_alerts: deque = deque(maxlen=50)
        
    def score_hypothesis(
        self,
        hypothesis: Any,  # Hypothesis object or dict
        explanatory_power: float,
        evidence_count: int,
        age: int
    ) -> HypothesisScore:
        """
        Score a hypothesis with full Occam penalty.
        Handles both Hypothesis objects and dicts.
        """
        """
        Score a hypothesis with full Occam penalty.
        
        Key insight: penalty grows with evidence.
        - Early: low penalty (exploration encouraged)
        - Late: high penalty (overfitting punished)
        """
        # Extract complexity (handle both objects and dicts)
        if isinstance(hypothesis, dict):
            complexity = hypothesis.get('complexity_score', 0.5)
        else:
            complexity = getattr(hypothesis, 'complexity_score', 0.5)
        
        # Calculate stability from history
        if isinstance(hypothesis, dict):
            hyp_id = str(hypothesis.get('id', 'unknown'))
        else:
            hyp_id = str(getattr(hypothesis, 'id', 'unknown'))
        if hyp_id not in self.score_history:
            self.score_history[hyp_id] = []
        
        # Track score history
        self.score_history[hyp_id].append(explanatory_power)
        if len(self.score_history[hyp_id]) > 20:
            self.score_history[hyp_id].pop(0)
        
        # Stability = low variance in explanatory power
        if len(self.score_history[hyp_id]) >= 5:
            recent = self.score_history[hyp_id][-5:]
            stability = 1.0 - min(1.0, np.std(recent))
        else:
            stability = 0.5  # Neutral until enough data
        
        # Raw score (explanatory power weighted by stability)
        raw_score = explanatory_power * (0.7 + 0.3 * stability)
        
        # Evidence factor: penalty grows with evidence
        # This is key: we punish complexity MORE as we see more data
        if evidence_count < self.config.min_evidence_before_penalty:
            evidence_factor = 0.0  # No penalty during exploration
        else:
            # Penalty ramps up from 0 to 1 as evidence accumulates
            evidence_factor = min(1.0, (evidence_count - self.config.min_evidence_before_penalty) / 20.0)
        
        # 🔴 OCCAM PENALTY: complexity * λ * evidence_factor
        penalty = complexity * self.config.lambda_penalty * evidence_factor
        
        # Final score
        occam_score = raw_score - penalty
        
        return HypothesisScore(
            hypothesis_id=hyp_id,
            explanatory_power=explanatory_power,
            complexity=complexity,
            stability=stability,
            evidence_count=evidence_count,
            age=age,
            raw_score=raw_score,
            occam_score=occam_score,
            penalty_applied=penalty,
            reason=f"power={explanatory_power:.3f} - complexity={complexity:.3f}*λ={self.config.lambda_penalty}*evidence={evidence_factor:.2f}"
        )
    
    def select_best_hypothesis(
        self,
        scored_hypotheses: List[HypothesisScore]
    ) -> Optional[HypothesisScore]:
        """
        Select best hypothesis after Occam scoring.
        
        Returns None if all hypotheses are pruned.
        """
        if not scored_hypotheses:
            return None
        
        # Sort by Occam score (descending)
        sorted_scores = sorted(scored_hypotheses, key=lambda x: x.occam_score, reverse=True)
        
        # Return best
        return sorted_scores[0]
    
    def identify_pruning_candidates(
        self,
        scored_hypotheses: List[HypothesisScore]
    ) -> List[str]:
        """
        Identify hypotheses that should be pruned.
        
        Criteria:
        - Score below threshold
        - Old enough (grace period passed)
        - Better alternatives exist
        """
        candidates = []
        
        if not scored_hypotheses:
            return candidates
        
        # Find best score for comparison
        best_score = max(h.occam_score for h in scored_hypotheses)
        
        for score in scored_hypotheses:
            # Don't prune young hypotheses
            if score.age < self.config.min_age_to_prune:
                continue
            
            # Prune if score too low
            if score.occam_score < self.config.prune_threshold:
                candidates.append(score.hypothesis_id)
                self.pruned_hypotheses.append(score.hypothesis_id)
                continue
            
            # Prune if far behind best (and not just starting out)
            if score.evidence_count > 15:
                score_gap = best_score - score.occam_score
                if score_gap > 0.3:  # 30% worse than best
                    candidates.append(score.hypothesis_id)
                    self.pruned_hypotheses.append(score.hypothesis_id)
        
        return candidates
    
    def detect_overfitting(
        self,
        hypothesis: Any,
        recent_scores: List[float]
    ) -> bool:
        """
        Detect if a hypothesis is overfitting.
        """
        if len(recent_scores) < 10:
            return False
        
        # High variance = unstable = likely overfitting
        recent_variance = np.var(recent_scores[-10:])
        if recent_variance > 0.1:  # High oscillation
            if isinstance(hypothesis, dict):
                hyp_id = hypothesis.get('id', 'unknown')
            else:
                hyp_id = getattr(hypothesis, 'id', 'unknown')
            self.overfitting_alerts.append({
                'hypothesis_id': hyp_id,
                'variance': recent_variance,
                'type': 'oscillation',
                'timestamp': len(self.overfitting_alerts)
            })
            return True
        
        return False
    
    def get_discipline_status(self) -> Dict[str, Any]:
        """Full status of Occam discipline system."""
        return {
            'lambda_penalty': self.config.lambda_penalty,
            'max_hypotheses': self.config.max_hypotheses,
            'pruned_count': len(self.pruned_hypotheses),
            'recent_prunings': self.pruned_hypotheses[-5:] if self.pruned_hypotheses else [],
            'overfitting_alerts': len(self.overfitting_alerts),
            'recent_alerts': list(self.overfitting_alerts)[-3:] if self.overfitting_alerts else []
        }


class DisciplinedBeliefSystem:
    """
    🔬 Complete belief system with Occam discipline.
    
    Wraps BeliefReasoner + HypothesisGenerator with explicit complexity control.
    """
    
    def __init__(
        self,
        belief_reasoner: Any,
        hypothesis_generator: Any,
        occam_config: Optional[OccamConfig] = None
    ):
        self.belief = belief_reasoner
        self.generator = hypothesis_generator
        self.occam = OccamLayer(occam_config)
        
        # Hypothesis tracking
        self.hypothesis_scores: Dict[str, HypothesisScore] = {}
        self.pruned_ids: set = set()
        
    def score_all_hypotheses(self, episode: int) -> List[HypothesisScore]:
        """Score all active hypotheses with Occam penalty."""
        scores = []
        
        for hyp in self.belief.hypotheses:
            hyp_id = str(getattr(hyp, 'id', 'unknown'))
            
            # Skip pruned hypotheses
            if hyp_id in self.pruned_ids:
                continue
            
            # Calculate explanatory power
            # (simplified: inverse of prediction error)
            boundary_error = abs(
                getattr(hyp, 'boundary_estimate', 0.5) - 
                self.belief.current_belief
            )
            explanatory_power = max(0, 1.0 - 2 * boundary_error)
            
            # Score with Occam penalty
            score = self.occam.score_hypothesis(
                hypothesis=hyp,
                explanatory_power=explanatory_power,
                evidence_count=getattr(hyp, 'evidence_count', 0),
                age=episode - getattr(hyp, 'birth_episode', 0)
            )
            
            self.hypothesis_scores[hyp_id] = score
            scores.append(score)
        
        return scores
    
    def select_best(self, scores: List[HypothesisScore]) -> Optional[HypothesisScore]:
        """Select best hypothesis, applying pruning."""
        # Identify pruning candidates
        to_prune = self.occam.identify_pruning_candidates(scores)
        self.pruned_ids.update(to_prune)
        
        # Filter out pruned
        active_scores = [s for s in scores if s.hypothesis_id not in self.pruned_ids]
        
        # Select best
        return self.occam.select_best_hypothesis(active_scores)
    
    def should_generate_new(self, scores: List[HypothesisScore], episode: int) -> bool:
        """
        Determine if we should generate new hypotheses.
        
        Only generate if:
        - Current best is unsatisfactory
        - We have budget for new hypotheses
        - Evidence suggests model inadequacy
        """
        # Check budget
        active_count = len([s for s in scores if s.hypothesis_id not in self.pruned_ids])
        if active_count >= self.occam.config.max_hypotheses:
            return False  # At capacity
        
        # Check if current best is inadequate
        if not scores:
            return True  # No hypotheses at all
        
        best = max(scores, key=lambda x: x.occam_score)
        
        # Generate if best score is poor and we've tried enough
        if best.occam_score < 0.3 and best.evidence_count > 15:
            return True
        
        # Generate if uncertainty is high despite good hypotheses
        if self.belief.current_uncertainty > 0.4 and episode % 50 == 0:
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Full disciplined belief system status."""
        scores = list(self.hypothesis_scores.values())
        
        return {
            'total_hypotheses': len(self.belief.hypotheses),
            'active_hypotheses': len([s for s in scores if s.hypothesis_id not in self.pruned_ids]),
            'pruned_hypotheses': len(self.pruned_ids),
            'best_score': max(s.occam_score for s in scores) if scores else 0.0,
            'avg_complexity': np.mean([s.complexity for s in scores]) if scores else 0.0,
            'occam_discipline': self.occam.get_discipline_status()
        }
