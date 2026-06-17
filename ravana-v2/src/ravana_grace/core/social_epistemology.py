"""
RAVANA v2 — PHASE H: Social Epistemology
From solitary reasoning → multi-agent belief conflict resolution

PRINCIPLE: Truth emerges from epistemic friction between minds.

CORE CAPABILITIES:
    1. Belief Conflict Detection: Identify when agents disagree
    2. Trust Scoring: Weight agents by epistemic reliability
    3. Adversarial Testing: Challenge beliefs through deliberate disagreement
    4. Consensus Formation: Merge beliefs when justified
    5. Deception Detection: Identify adversarial manipulation

PHASE H ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────┐
    │  SOCIAL EPISTEMOLOGY LAYER                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │ Multi-Agent │  │   Trust     │  │ Adversarial │     │
    │  │   Network   │→ │   Engine    │→ │   Testing   │     │
    │  └─────────────┘  └─────────────┘  └─────────────┘     │
    │         ↓                  ↓                  ↓         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │   Belief    │  │  Consensus  │  │  Deception  │     │
    │  │   Conflict  │  │  Formation  │  │  Detection  │     │
    │  └─────────────┘  └─────────────┘  └─────────────┘     │
    └─────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from enum import Enum, auto
from collections import deque
import numpy as np
from copy import deepcopy


class AgentType(Enum):
    """Types of agents in the social epistemic network."""
    RAVANA = "ravana"              # Self — the primary agent
    PEER = "peer"                   # Honest peer with independent beliefs
    ADVERSARY = "adversary"         # Deliberately misleading agent
    EXPERT = "expert"               # High-reliability domain expert
    NOVICE = "novice"               # Low-reliability learning agent
    MEDIATOR = "mediator"           # Neutral consensus facilitator


class ConflictType(Enum):
    """Types of belief conflicts."""
    BOUNDARY_DISAGREEMENT = auto()   # Disagree on where boundary is
    CONFIDENCE_MISMATCH = auto()     # Same boundary, different confidence
    STRATEGY_DIVERGENCE = auto()     # Different approach to problem
    HYPOTHESIS_INCOMPATIBLE = auto() # Mutually exclusive hypotheses
    EVIDENCE_INTERPRETATION = auto() # Same evidence, different conclusions


class TrustUpdateReason(Enum):
    """Reasons for trust score updates."""
    PREDICTION_SUCCESS = auto()
    PREDICTION_FAILURE = auto()
    CONSENSUS_CONVERGENCE = auto()
    CONSENSUS_DIVERGENCE = auto()
    ADVERARIAL_DETECTED = auto()
    HONESTY_VERIFIED = auto()


@dataclass
class AgentBelief:
    """A belief held by an agent in the network."""
    agent_id: str
    boundary_estimate: float
    confidence: float
    uncertainty: float
    hypothesis_id: Optional[int] = None
    evidence_count: int = 0
    last_updated: int = 0
    
    def belief_distance(self, other: 'AgentBelief') -> float:
        """Compute epistemic distance between two beliefs."""
        # Boundary difference weighted by confidences
        boundary_diff = abs(self.boundary_estimate - other.boundary_estimate)
        
        # Confidence agreement (disagreement increases distance)
        confidence_diff = abs(self.confidence - other.confidence)
        
        # Combined distance
        return boundary_diff + 0.3 * confidence_diff


@dataclass
class TrustScore:
    """Trust score for an agent in the network."""
    agent_id: str
    reliability: float = 0.5       # 0-1, prediction accuracy
    honesty: float = 0.5           # 0-1, non-manipulation
    expertise: float = 0.5         # 0-1, domain knowledge
    
    # Temporal tracking
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=50))
    honesty_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Confidence intervals
    reliability_uncertainty: float = 0.3
    honesty_uncertainty: float = 0.3
    
    @property
    def composite_trust(self) -> float:
        """Composite trust score weighted by certainty."""
        # Weight by inverse uncertainty
        w_rel = 1.0 / (1 + self.reliability_uncertainty)
        w_hon = 1.0 / (1 + self.honesty_uncertainty)
        
        # Weighted average
        total_weight = w_rel + w_hon + 0.5  # expertise has fixed weight
        return (w_rel * self.reliability + w_hon * self.honesty + 0.5 * self.expertise) / total_weight
    
    def update_reliability(self, predicted: float, actual: float, episode: int):
        """Update reliability based on prediction accuracy."""
        error = abs(predicted - actual)
        accuracy = max(0, 1.0 - error * 2)
        
        self.prediction_history.append({
            'episode': episode,
            'predicted': predicted,
            'actual': actual,
            'accuracy': accuracy
        })
        
        # Bayesian-ish update with decay
        if len(self.prediction_history) >= 5:
            recent_acc = np.mean([h['accuracy'] for h in list(self.prediction_history)[-5:]])
            self.reliability = 0.8 * self.reliability + 0.2 * recent_acc
            
            # Uncertainty decreases with more data
            self.reliability_uncertainty = max(0.1, 0.3 / (1 + len(self.prediction_history) / 10))
    
    def update_honesty(self, agreement_with_consensus: float, episode: int):
        """Update honesty based on consensus alignment."""
        self.honesty_history.append({
            'episode': episode,
            'consensus_agreement': agreement_with_consensus
        })
        
        # Detect systematic disagreement (possible manipulation)
        if len(self.honesty_history) >= 10:
            recent = list(self.honesty_history)[-10:]
            avg_agreement = np.mean([h['consensus_agreement'] for h in recent])
            
            # Low agreement over time = suspicious
            self.honesty = 0.9 * self.honesty + 0.1 * avg_agreement
            
            # Sharp drops trigger deception detection
            if avg_agreement < 0.3 and self.honesty > 0.5:
                return True  # Possible deception detected
        
        return False


@dataclass
class BeliefConflict:
    """A detected conflict between agent beliefs."""
    conflict_id: int
    agent_a: str
    agent_b: str
    conflict_type: ConflictType
    severity: float  # 0-1
    belief_a: AgentBelief
    belief_b: AgentBelief
    episode_detected: int
    
    # Resolution state
    resolved: bool = False
    resolution_episode: Optional[int] = None
    resolution_method: Optional[str] = None
    winner: Optional[str] = None  # Which agent was right (if known)


@dataclass
class ConsensusBelief:
    """A belief formed through consensus of multiple agents."""
    boundary_estimate: float
    confidence: float
    contributing_agents: List[str]
    trust_weights: Dict[str, float]
    formation_episode: int
    
    # Quality metrics
    agreement_score: float  # How much agents agreed
    diversity_bonus: float  # Benefit from different perspectives


@dataclass
class SocialEpistemicConfig:
    """Configuration for social epistemology system."""
    # Trust parameters
    initial_trust: float = 0.5
    trust_update_rate: float = 0.2
    deception_threshold: float = 0.3  # Honesty below this = flagged
    
    # Conflict detection
    conflict_threshold: float = 0.15  # Belief distance > this = conflict
    min_confidence_for_conflict: float = 0.3  # Ignore low-confidence beliefs
    
    # Consensus formation
    min_agents_for_consensus: int = 3
    consensus_threshold: float = 0.6  # Agreement > this = consensus
    max_diversity_bonus: float = 0.2
    
    # Adversarial testing
    adversarial_test_frequency: int = 20  # Episodes between tests
    adversary_count: int = 1
    
    # Network limits
    max_network_size: int = 10
    belief_history_size: int = 100


class SocialEpistemologyEngine:
    """
    Multi-agent epistemic system with trust scoring and adversarial testing.
    
    CORE INSIGHT: Truth is not discovered in isolation.
    It emerges from the friction between multiple perspectives,
    weighted by their demonstrated reliability.
    """
    
    def __init__(self, config: Optional[SocialEpistemicConfig] = None):
        self.config = config or SocialEpistemicConfig()
        
        # Agent network
        self.agents: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata
        self.agent_beliefs: Dict[str, AgentBelief] = {}  # Current beliefs
        self.belief_history: Dict[str, deque] = {}  # Historical beliefs
        
        # Trust system
        self.trust_scores: Dict[str, TrustScore] = {}
        
        # Conflict tracking
        self.active_conflicts: Dict[int, BeliefConflict] = {}
        self.resolved_conflicts: List[BeliefConflict] = []
        self._next_conflict_id: int = 1
        
        # Consensus tracking
        self.consensus_beliefs: List[ConsensusBelief] = []
        self.current_consensus: Optional[ConsensusBelief] = None
        
        # Adversarial state
        self.adversarial_agents: Set[str] = set()
        self.last_adversarial_test: int = -100
        self.adversarial_episodes: List[Dict] = []
        
        # Deception detection
        self.deception_alerts: List[Dict] = []
        
        # Episode tracking
        self.current_episode: int = 0
        
        # Analytics
        self.social_metrics: deque = deque(maxlen=1000)
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: AgentType,
        initial_belief: Optional[AgentBelief] = None,
        expertise_domain: Optional[str] = None
    ) -> bool:
        """
        Register a new agent in the social epistemic network.
        """
        if len(self.agents) >= self.config.max_network_size:
            return False
        
        self.agents[agent_id] = {
            'type': agent_type,
            'expertise_domain': expertise_domain,
            'registered_episode': self.current_episode
        }
        
        # Initialize belief
        if initial_belief:
            self.agent_beliefs[agent_id] = initial_belief
        else:
            self.agent_beliefs[agent_id] = AgentBelief(
                agent_id=agent_id,
                boundary_estimate=0.75,
                confidence=0.5,
                uncertainty=0.2,
                last_updated=self.current_episode
            )
        
        # Initialize belief history
        self.belief_history[agent_id] = deque(maxlen=self.config.belief_history_size)
        self.belief_history[agent_id].append(self.agent_beliefs[agent_id])
        
        # Initialize trust
        base_reliability = 0.5
        if agent_type == AgentType.EXPERT:
            base_reliability = 0.7
        elif agent_type == AgentType.NOVICE:
            base_reliability = 0.3
        elif agent_type == AgentType.ADVERSARY:
            base_reliability = 0.4  # May be accurate but manipulative
            self.adversarial_agents.add(agent_id)
        
        self.trust_scores[agent_id] = TrustScore(
            agent_id=agent_id,
            reliability=base_reliability,
            honesty=0.5 if agent_type != AgentType.ADVERSARY else 0.3,
            expertise=0.7 if agent_type == AgentType.EXPERT else 0.5
        )
        
        return True
    
    def update_belief(
        self,
        agent_id: str,
        boundary_estimate: float,
        confidence: float,
        uncertainty: float,
        evidence_count: int = 0
    ) -> Dict[str, Any]:
        """
        Update an agent's belief and trigger conflict detection.
        """
        if agent_id not in self.agents:
            return {"error": "agent_not_registered"}
        
        # Store old belief
        old_belief = self.agent_beliefs.get(agent_id)
        
        # Create new belief
        new_belief = AgentBelief(
            agent_id=agent_id,
            boundary_estimate=boundary_estimate,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence_count=evidence_count,
            last_updated=self.current_episode
        )
        
        # Update belief
        self.agent_beliefs[agent_id] = new_belief
        self.belief_history[agent_id].append(new_belief)
        
        # Detect conflicts with other agents
        conflicts_detected = self._detect_conflicts(agent_id, new_belief)
        
        # Update trust if this is self-update
        if self.agents[agent_id]['type'] == AgentType.RAVANA:
            self._update_peer_trust_from_belief_change(agent_id, old_belief, new_belief)
        
        return {
            "conflicts_detected": len(conflicts_detected),
            "conflict_ids": conflicts_detected,
            "belief_distance_from_consensus": self._distance_from_consensus(new_belief)
        }
    
    def _detect_conflicts(self, agent_id: str, belief: AgentBelief) -> List[int]:
        """Detect conflicts between this belief and other agents."""
        conflicts = []
        
        for other_id, other_belief in self.agent_beliefs.items():
            if other_id == agent_id:
                continue
            
            # Skip low-confidence beliefs
            if belief.confidence < self.config.min_confidence_for_conflict:
                continue
            if other_belief.confidence < self.config.min_confidence_for_conflict:
                continue
            
            # Compute belief distance
            distance = belief.belief_distance(other_belief)
            
            if distance > self.config.conflict_threshold:
                # Determine conflict type
                conflict_type = self._classify_conflict(belief, other_belief)
                
                # Create conflict record
                conflict = BeliefConflict(
                    conflict_id=self._next_conflict_id,
                    agent_a=agent_id,
                    agent_b=other_id,
                    conflict_type=conflict_type,
                    severity=min(1.0, distance / self.config.conflict_threshold),
                    belief_a=belief,
                    belief_b=other_belief,
                    episode_detected=self.current_episode
                )
                
                self.active_conflicts[self._next_conflict_id] = conflict
                conflicts.append(self._next_conflict_id)
                self._next_conflict_id += 1
        
        return conflicts
    
    def _classify_conflict(
        self,
        belief_a: AgentBelief,
        belief_b: AgentBelief
    ) -> ConflictType:
        """Classify the type of conflict between two beliefs."""
        boundary_diff = abs(belief_a.boundary_estimate - belief_b.boundary_estimate)
        confidence_diff = abs(belief_a.confidence - belief_b.confidence)
        
        if boundary_diff > 0.1:
            return ConflictType.BOUNDARY_DISAGREEMENT
        elif confidence_diff > 0.2:
            return ConflictType.CONFIDENCE_MISMATCH
        else:
            return ConflictType.EVIDENCE_INTERPRETATION
    
    def resolve_conflict(
        self,
        conflict_id: int,
        actual_boundary: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Resolve a conflict using trust-weighted consensus.
        
        If actual_boundary provided, update trust scores based on who was right.
        """
        if conflict_id not in self.active_conflicts:
            return {"error": "conflict_not_found"}
        
        conflict = self.active_conflicts[conflict_id]
        
        # Get trust scores for both agents
        trust_a = self.trust_scores[conflict.agent_a].composite_trust
        trust_b = self.trust_scores[conflict.agent_b].composite_trust
        
        # Weighted consensus prediction
        total_trust = trust_a + trust_b
        if total_trust > 0:
            predicted_boundary = (
                trust_a * conflict.belief_a.boundary_estimate +
                trust_b * conflict.belief_b.boundary_estimate
            ) / total_trust
        else:
            predicted_boundary = (conflict.belief_a.boundary_estimate + conflict.belief_b.boundary_estimate) / 2
        
        # If ground truth available, update trust
        if actual_boundary is not None:
            error_a = abs(conflict.belief_a.boundary_estimate - actual_boundary)
            error_b = abs(conflict.belief_b.boundary_estimate - actual_boundary)
            
            # Update reliability
            self.trust_scores[conflict.agent_a].update_reliability(
                conflict.belief_a.boundary_estimate, actual_boundary, self.current_episode
            )
            self.trust_scores[conflict.agent_b].update_reliability(
                conflict.belief_b.boundary_estimate, actual_boundary, self.current_episode
            )
            
            # Determine winner
            if error_a < error_b:
                conflict.winner = conflict.agent_a
            elif error_b < error_a:
                conflict.winner = conflict.agent_b
            else:
                conflict.winner = None  # Tie
        
        # Mark resolved
        conflict.resolved = True
        conflict.resolution_episode = self.current_episode
        conflict.resolution_method = "trust_weighted_consensus"
        
        # Move to resolved
        del self.active_conflicts[conflict_id]
        self.resolved_conflicts.append(conflict)
        
        return {
            "conflict_id": conflict_id,
            "predicted_boundary": predicted_boundary,
            "winner": conflict.winner,
            "trust_a": trust_a,
            "trust_b": trust_b
        }
    
    def form_consensus(self) -> Optional[ConsensusBelief]:
        """
        Form consensus belief from all agents weighted by trust.
        """
        if len(self.agent_beliefs) < self.config.min_agents_for_consensus:
            return None
        
        # Gather high-confidence beliefs
        qualified_beliefs = [
            (aid, belief) for aid, belief in self.agent_beliefs.items()
            if belief.confidence >= self.config.min_confidence_for_conflict
        ]
        
        if len(qualified_beliefs) < self.config.min_agents_for_consensus:
            return None
        
        # Compute trust weights
        total_trust = sum(self.trust_scores[aid].composite_trust for aid, _ in qualified_beliefs)
        
        if total_trust == 0:
            # Equal weighting if no trust info
            weights = {aid: 1.0 / len(qualified_beliefs) for aid, _ in qualified_beliefs}
        else:
            weights = {
                aid: self.trust_scores[aid].composite_trust / total_trust
                for aid, _ in qualified_beliefs
            }
        
        # Weighted average boundary
        consensus_boundary = sum(
            weights[aid] * belief.boundary_estimate for aid, belief in qualified_beliefs
        )
        
        # Confidence based on agreement
        belief_values = [belief.boundary_estimate for _, belief in qualified_beliefs]
        agreement = 1.0 - np.std(belief_values)  # Higher std = lower agreement
        
        # Diversity bonus: different perspectives help
        agent_types = set(self.agents[aid]['type'] for aid, _ in qualified_beliefs)
        diversity_bonus = min(self.config.max_diversity_bonus, len(agent_types) * 0.05)
        
        consensus = ConsensusBelief(
            boundary_estimate=consensus_boundary,
            confidence=min(1.0, agreement + diversity_bonus),
            contributing_agents=[aid for aid, _ in qualified_beliefs],
            trust_weights=weights,
            formation_episode=self.current_episode,
            agreement_score=agreement,
            diversity_bonus=diversity_bonus
        )
        
        self.consensus_beliefs.append(consensus)
        self.current_consensus = consensus
        
        return consensus
    
    def run_adversarial_test(self, episode: int) -> Dict[str, Any]:
        """
        Run adversarial epistemic test by introducing deceptive agents.
        """
        # Rate limiting
        if episode - self.last_adversarial_test < self.config.adversarial_test_frequency:
            return {"skipped": True, "reason": "rate_limited"}
        
        self.last_adversarial_test = episode
        
        # Create temporary adversarial agents with false beliefs
        adversary_ids = []
        for i in range(self.config.adversary_count):
            adv_id = f"adversary_test_{episode}_{i}"
            
            # Create belief opposite to current consensus
            if self.current_consensus:
                false_boundary = max(0.2, min(0.95, self.current_consensus.boundary_estimate + np.random.choice([-0.3, 0.3])))
            else:
                false_boundary = np.random.uniform(0.3, 0.9)
            
            false_belief = AgentBelief(
                agent_id=adv_id,
                boundary_estimate=false_boundary,
                confidence=0.7,  # Confident but wrong
                uncertainty=0.1,
                last_updated=episode
            )
            
            # Register as adversary (check if successful)
            success = self.register_agent(
                adv_id,
                AgentType.ADVERSARY,
                initial_belief=false_belief
            )
            
            if success:
                adversary_ids.append(adv_id)
        
        # Record test only if we actually added adversaries
        if not adversary_ids:
            return {
                "adversaries_introduced": 0,
                "adversary_ids": [],
                "manipulation_resistance": None,
                "test_record": {"skipped": True, "reason": "network_at_capacity"}
            }
        
        # Get false boundaries only for successfully registered agents
        test_record = {
            'episode': episode,
            'adversary_ids': adversary_ids,
            'false_boundaries': [self.agent_beliefs[aid].boundary_estimate for aid in adversary_ids],
            'consensus_before': self.current_consensus.boundary_estimate if self.current_consensus else None
        }
        
        self.adversarial_episodes.append(test_record)
        
        # Detect if RAVANA resists manipulation
        # (Check if RAVANA's belief stays close to consensus despite adversaries)
        ravana_belief = None
        for aid, belief in self.agent_beliefs.items():
            if self.agents[aid]['type'] == AgentType.RAVANA:
                ravana_belief = belief
                break
        
        if ravana_belief and self.current_consensus:
            ravana_consensus_distance = abs(ravana_belief.boundary_estimate - self.current_consensus.boundary_estimate)
            manipulation_resistance = ravana_consensus_distance < 0.15
        else:
            manipulation_resistance = None
        
        return {
            "adversaries_introduced": len(adversary_ids),
            "adversary_ids": adversary_ids,
            "manipulation_resistance": manipulation_resistance,
            "test_record": test_record
        }
    
    def detect_deception(self) -> List[Dict]:
        """
        Detect agents that may be deliberately deceptive.
        """
        alerts = []
        
        for agent_id, trust in self.trust_scores.items():
            # Check for low honesty
            if trust.honesty < self.config.deception_threshold:
                alert = {
                    'episode': self.current_episode,
                    'agent_id': agent_id,
                    'honesty_score': trust.honesty,
                    'reliability_score': trust.reliability,
                    'type': 'low_honesty_detected',
                    'severity': 'high' if trust.honesty < 0.2 else 'medium'
                }
                alerts.append(alert)
                self.deception_alerts.append(alert)
            
            # Check for systematic disagreement with consensus
            if self.current_consensus:
                belief = self.agent_beliefs.get(agent_id)
                if belief:
                    distance = abs(belief.boundary_estimate - self.current_consensus.boundary_estimate)
                    if distance > 0.25 and trust.honesty < 0.4:
                        alert = {
                            'episode': self.current_episode,
                            'agent_id': agent_id,
                            'consensus_distance': distance,
                            'type': 'systematic_divergence',
                            'severity': 'high'
                        }
                        alerts.append(alert)
                        self.deception_alerts.append(alert)
        
        return alerts
    
    def step(self, episode: int) -> Dict[str, Any]:
        """
        Execute one social epistemology step.
        """
        self.current_episode = episode
        
        # Update consensus
        consensus = self.form_consensus()
        
        # Run adversarial test if time
        adversarial_result = None
        if episode - self.last_adversarial_test >= self.config.adversarial_test_frequency:
            adversarial_result = self.run_adversarial_test(episode)
        
        # Detect deception
        deception_alerts = self.detect_deception()
        
        # Resolve old conflicts (if evidence available)
        # In practice, this would use actual outcomes
        resolved_this_step = []
        
        # Record metrics
        metrics = {
            'episode': episode,
            'num_agents': len(self.agents),
            'active_conflicts': len(self.active_conflicts),
            'consensus_confidence': consensus.confidence if consensus else 0,
            'avg_trust': np.mean([t.composite_trust for t in self.trust_scores.values()]) if self.trust_scores else 0,
            'deception_alerts': len(deception_alerts),
            'adversarial_active': len(self.adversarial_agents)
        }
        self.social_metrics.append(metrics)
        
        return {
            'consensus_formed': consensus is not None,
            'consensus_boundary': consensus.boundary_estimate if consensus else None,
            'adversarial_test': adversarial_result,
            'deception_alerts': deception_alerts,
            'active_conflicts': len(self.active_conflicts),
            'metrics': metrics
        }
    
    def _distance_from_consensus(self, belief: AgentBelief) -> float:
        """Compute distance from current consensus."""
        if not self.current_consensus:
            return 0.0
        return abs(belief.boundary_estimate - self.current_consensus.boundary_estimate)
    
    def _update_peer_trust_from_belief_change(
        self,
        agent_id: str,
        old_belief: Optional[AgentBelief],
        new_belief: AgentBelief
    ):
        """Update trust scores when RAVANA changes belief."""
        if not old_belief or not self.current_consensus:
            return
        
        # Did RAVANA move toward or away from consensus?
        old_distance = abs(old_belief.boundary_estimate - self.current_consensus.boundary_estimate)
        new_distance = abs(new_belief.boundary_estimate - self.current_consensus.boundary_estimate)
        
        movement_toward_consensus = new_distance < old_distance
        
        # Update trust for agents whose beliefs RAVANA moved toward
        for peer_id, peer_belief in self.agent_beliefs.items():
            if peer_id == agent_id:
                continue
            
            peer_distance_to_new = abs(peer_belief.boundary_estimate - new_belief.boundary_estimate)
            peer_distance_to_old = abs(peer_belief.boundary_estimate - old_belief.boundary_estimate)
            
            if peer_distance_to_new < peer_distance_to_old:
                # RAVANA moved toward this peer
                if movement_toward_consensus:
                    # And toward consensus — peer gains trust
                    self.trust_scores[peer_id].reliability = min(
                        1.0, self.trust_scores[peer_id].reliability + 0.02
                    )
    
    def get_status(self) -> Dict[str, Any]:
        """Full social epistemology status."""
        return {
            'num_agents': len(self.agents),
            'agent_types': {aid: self.agents[aid]['type'].value for aid in self.agents},
            'trust_scores': {
                aid: {
                    'reliability': t.reliability,
                    'honesty': t.honesty,
                    'composite': t.composite_trust
                }
                for aid, t in self.trust_scores.items()
            },
            'active_conflicts': len(self.active_conflicts),
            'resolved_conflicts': len(self.resolved_conflicts),
            'consensus_history': len(self.consensus_beliefs),
            'current_consensus': {
                'boundary': self.current_consensus.boundary_estimate,
                'confidence': self.current_consensus.confidence,
                'contributors': len(self.current_consensus.contributing_agents)
            } if self.current_consensus else None,
            'deception_alerts_total': len(self.deception_alerts),
            'adversarial_tests': len(self.adversarial_episodes)
        }


# Convenience alias
SocialEpistemology = SocialEpistemologyEngine
