"""
RAVANA v2 — PHASE I²: Meta²-Cognition
From "what do I believe?" → "how do I generate beliefs?"

PRINCIPLE: The system must be able to question its own epistemic method.

META² QUESTIONS:
    1. Hypothesis Space Critique: Is the true hypothesis in my space?
    2. Bias Detection: Am I structurally blind to certain truths?
    3. Probe Strategy Evaluation: Are my probes effective?
    4. Identity-Protected Beliefs: Is my identity hiding truth?

PHASE I² ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────┐
    │  META²-COGNITION LAYER                                  │
    │                                                         │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │ Hypothesis Space Auditor                        │   │
    │  │ • Tracks hypothesis generation failure          │   │
    │  │ • Detects when space is inadequate              │   │
    │  └─────────────────────────────────────────────────┘   │
    │                                                         │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │ Bias Detector                                   │   │
    │  │ • Occam bias (too simple)                       │   │
    │  │ • Complexity bias (too complex)                 │   │
    │  │ • Exploration bias (under-exploration)          │   │
    │  └─────────────────────────────────────────────────┘   │
    │                                                         │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │ Probe Strategy Critic                           │   │
    │  │ • Evaluates probe effectiveness                 │   │
    │  │ • Detects probe failure patterns                │   │
    │  └─────────────────────────────────────────────────┘   │
    │                                                         │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │ Identity-Belief Coupling Detector               │   │
    │  │ • Detects when identity protects wrong beliefs  │   │
    │  │ • Flags epistemic closure due to identity       │   │
    │  └─────────────────────────────────────────────────┘   │
    │                                                         │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │ Epiphany Generator                              │   │
    │  │ • Triggers when method failure detected         │   │
    │  │ • Proposes radical method revisions             │   │
    │  └─────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘

KEY INSIGHT:
    This is NOT about generating better hypotheses.
    This is about asking: "Should I be generating hypotheses differently?"

EPISTEMIC EPIPHANY:
    When failure rate exceeds threshold AND hypothesis space audit fails:
    → Trigger epiphany: "My way of thinking is wrong"
    → Propose structural revision to epistemic method
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum, auto
from collections import deque
import numpy as np


class EpistemicCritiqueType(Enum):
    """Types of epistemic self-critique."""
    HYPOTHESIS_SPACE_INADEQUATE = auto()  # True hypothesis not representable
    OCCAM_BIAS = auto()                   # Preference for simplicity hurts
    COMPLEXITY_BIAS = auto()              # Preference for complexity hurts
    PROBE_STRATEGY_FAILURE = auto()       # Probes don't separate hypotheses
    IDENTITY_PROTECTED_BELIEF = auto()    # Identity blocking truth
    EXPLORATION_DEFICIT = auto()          # Not exploring enough
    METHOD_EPIPHANY = auto()              # Fundamental method revision needed


@dataclass
class Meta2Config:
    """Configuration for Meta²-Cognition."""
    # Failure detection thresholds
    sustained_failure_window: int = 30        # Episodes of failure
    failure_rate_threshold: float = 0.4         # 40% failure = problem
    
    # Hypothesis space audit
    min_hypotheses_for_audit: int = 3
    space_inadequacy_threshold: float = 0.7     # Confidence below this = inadequate
    
    # Bias detection
    occam_penalty_threshold: float = 0.2        # Too much simplicity preference
    complexity_penalty_threshold: float = 0.3     # Too much complexity preference
    
    # Probe strategy
    probe_ineffectiveness_threshold: float = 0.3  # KL gain below this = ineffective
    min_probes_for_evaluation: int = 10
    
    # Identity-belief coupling
    identity_belief_coupling_threshold: float = 0.6  # Correlation above this = coupling
    
    # Epiphany trigger
    epiphany_failure_threshold: float = 0.5     # Failures needed for epiphany
    epiphany_confidence_min: float = 0.7        # Must be confident critique is right


@dataclass
class HypothesisSpaceAudit:
    """Audit of hypothesis space adequacy."""
    timestamp: int
    hypothesis_types_present: Set[str]
    max_confidence_achieved: float
    unexplained_variance: float
    space_adequate: bool
    recommendation: Optional[str] = None


@dataclass
class BiasAssessment:
    """Assessment of epistemic biases."""
    timestamp: int
    occam_bias_score: float           # 0-1, higher = more simplicity preference
    complexity_bias_score: float        # 0-1, higher = more complexity preference
    exploration_deficit_score: float   # 0-1, higher = under-exploring
    dominant_bias: Optional[str] = None
    bias_impact_estimate: float = 0.0  # How much bias is hurting performance


@dataclass
class ProbeStrategyEvaluation:
    """Evaluation of probing strategy effectiveness."""
    timestamp: int
    avg_kl_gain: float
    hypothesis_separation_rate: float
    probe_effectiveness_score: float   # 0-1
    ineffective_probe_types: List[str]
    recommended_probe_redesign: Optional[str] = None


@dataclass
class IdentityBeliefCoupling:
    """Detection of identity protecting beliefs."""
    timestamp: int
    coupling_strength: float           # 0-1, correlation between identity and belief
    belief_identity_correlation: float
    protected_beliefs: List[str]       # Which beliefs are identity-protected
    dissonance_suppression_detected: bool
    recommendation: str = ""


@dataclass
class EpistemicEpiphany:
    """
    A radical realization that the epistemic method itself needs revision.
    
    This is the output of Meta²-Cognition when it detects fundamental
    failures in how the system thinks.
    """
    timestamp: int
    trigger: EpistemicCritiqueType
    confidence: float
    
    # The realization
    realization: str                    # What was realized
    current_method_limitation: str      # What's wrong with current approach
    proposed_method_revision: str      # How to change thinking
    
    # Specific recommendations
    hypothesis_space_expansion: Optional[List[str]] = None
    bias_corrections: Optional[List[str]] = None
    probe_strategy_redesign: Optional[str] = None
    identity_decoupling_recommended: bool = False
    
    # Validation
    expected_improvement: float = 0.0   # Expected performance gain
    risk_of_revision: float = 0.0       # Risk of changing method


class Meta2CognitionEngine:
    """
    Meta²-Cognition: The system questioning its own epistemic method.
    
    This engine monitors:
    - Whether the hypothesis space can represent truth
    - Whether biases are systematically distorting reasoning
    - Whether probes are effectively separating hypotheses
    - Whether identity is protecting wrong beliefs
    
    When failures accumulate, it generates epiphanies:
    radical realizations that the method of thinking must change.
    """
    
    def __init__(self, config: Optional[Meta2Config] = None):
        self.config = config or Meta2Config()
        
        # Audit history
        self.space_audits: deque = deque(maxlen=100)
        self.bias_assessments: deque = deque(maxlen=100)
        self.probe_evaluations: deque = deque(maxlen=100)
        self.coupling_detections: deque = deque(maxlen=100)
        
        # Epiphany history
        self.epiphanies: List[EpistemicEpiphany] = []
        self.epiphany_count: int = 0
        
        # Failure tracking for epiphany triggering
        self.failure_history: deque = deque(maxlen=50)
        self.recent_critiques: List[EpistemicCritiqueType] = []
        
        # Current state
        self.last_epiphany_episode: int = -1000
        self.current_audit: Optional[HypothesisSpaceAudit] = None
        self.current_bias: Optional[BiasAssessment] = None
        self.current_probe_eval: Optional[ProbeStrategyEvaluation] = None
        self.current_coupling: Optional[IdentityBeliefCoupling] = None
    
    def audit_hypothesis_space(
        self,
        hypotheses: List[Any],
        belief_history: List[float],
        episode: int
    ) -> HypothesisSpaceAudit:
        """
        Audit whether hypothesis space can represent the truth.
        
        Key insight: If all hypotheses have low confidence despite much data,
        the space itself may be inadequate.
        """
        if len(hypotheses) < self.config.min_hypotheses_for_audit:
            return HypothesisSpaceAudit(
                timestamp=episode,
                hypothesis_types_present=set(),
                max_confidence_achieved=0.0,
                unexplained_variance=1.0,
                space_adequate=True  # Not enough data to judge
            )
        
        # Extract hypothesis types
        hypothesis_types = set()
        confidences = []
        
        for h in hypotheses:
            if hasattr(h, 'hypothesis_type'):
                if isinstance(h.hypothesis_type, str):
                    hypothesis_types.add(h.hypothesis_type)
                else:
                    hypothesis_types.add(h.hypothesis_type.name)
            confidences.append(getattr(h, 'confidence', 0.5))
        
        # Compute unexplained variance in belief history
        if len(belief_history) > 10:
            recent_beliefs = belief_history[-20:]
            belief_variance = np.var(recent_beliefs)
            # High variance despite many hypotheses = space inadequate
            unexplained = belief_variance
        else:
            unexplained = 0.5  # Not enough history
        
        max_confidence = max(confidences) if confidences else 0.0
        
        # Space is inadequate if max confidence low despite many hypotheses
        space_adequate = (
            max_confidence > self.config.space_inadequacy_threshold
            or len(hypotheses) < 3  # Too early to tell
        )
        
        recommendation = None
        if not space_adequate:
            # Recommend expanding hypothesis space
            present = hypothesis_types
            all_types = {'PARAMETRIC_TIME', 'PARAMETRIC_STATE', 'STRUCTURAL_DUAL', 
                        'STRUCTURAL_ASYMMETRIC', 'CAUSAL_CORRELATE', 'CAUSAL_MECHANISM'}
            missing = all_types - present
            if missing:
                recommendation = f"Expand hypothesis space to include: {', '.join(missing)}"
        
        audit = HypothesisSpaceAudit(
            timestamp=episode,
            hypothesis_types_present=hypothesis_types,
            max_confidence_achieved=max_confidence,
            unexplained_variance=unexplained,
            space_adequate=space_adequate,
            recommendation=recommendation
        )
        
        self.space_audits.append(audit)
        self.current_audit = audit
        
        return audit
    
    def detect_biases(
        self,
        hypotheses: List[Any],
        exploration_history: List[bool],
        episode: int
    ) -> BiasAssessment:
        """
        Detect systematic biases in epistemic method.
        """
        # Analyze hypothesis type distribution
        type_counts = {}
        for h in hypotheses:
            htype = getattr(h, 'hypothesis_type', 'unknown')
            if not isinstance(htype, str):
                htype = htype.name if hasattr(htype, 'name') else str(htype)
            type_counts[htype] = type_counts.get(htype, 0) + 1
        
        # Occam bias: too many simple (parametric) hypotheses
        parametric_count = sum(1 for t in type_counts if 'PARAMETRIC' in t)
        structural_count = sum(1 for t in type_counts if 'STRUCTURAL' in t)
        causal_count = sum(1 for t in type_counts if 'CAUSAL' in t)
        
        total = len(hypotheses)
        if total > 0:
            occam_score = parametric_count / total
            complexity_score = (structural_count + causal_count) / total
        else:
            occam_score = 0.5
            complexity_score = 0.5
        
        # Exploration deficit: not enough exploration attempts
        if len(exploration_history) > 20:
            recent_explore = exploration_history[-20:]
            explore_rate = sum(recent_explore) / len(recent_explore)
            exploration_deficit = 1.0 - explore_rate
        else:
            exploration_deficit = 0.5
        
        # Determine dominant bias
        scores = {
            'occam': occam_score,
            'complexity': complexity_score,
            'exploration_deficit': exploration_deficit
        }
        dominant = max(scores, key=scores.get)
        
        # Estimate impact (simplified)
        impact = max(0, occam_score - 0.5) + max(0, complexity_score - 0.5)
        
        assessment = BiasAssessment(
            timestamp=episode,
            occam_bias_score=occam_score,
            complexity_bias_score=complexity_score,
            exploration_deficit_score=exploration_deficit,
            dominant_bias=dominant if scores[dominant] > 0.6 else None,
            bias_impact_estimate=impact
        )
        
        self.bias_assessments.append(assessment)
        self.current_bias = assessment
        
        return assessment
    
    def evaluate_probe_strategy(
        self,
        probe_history: List[Dict],
        hypothesis_separations: List[float],
        episode: int
    ) -> ProbeStrategyEvaluation:
        """
        Evaluate whether current probing strategy is effective.
        """
        if len(probe_history) < self.config.min_probes_for_evaluation:
            return ProbeStrategyEvaluation(
                timestamp=episode,
                avg_kl_gain=0.0,
                hypothesis_separation_rate=0.0,
                probe_effectiveness_score=0.5,
                ineffective_probe_types=[]
            )
        
        # Calculate average KL gain from recent probes
        recent_kl = [p.get('kl_gain', 0) for p in probe_history[-20:]]
        avg_kl = np.mean(recent_kl) if recent_kl else 0.0
        
        # Calculate separation rate
        if len(hypothesis_separations) > 5:
            recent_sep = hypothesis_separations[-10:]
            separation_rate = sum(1 for s in recent_sep if s > 0.1) / len(recent_sep)
        else:
            separation_rate = 0.5
        
        # Overall effectiveness
        effectiveness = (avg_kl / 0.5) * 0.5 + separation_rate * 0.5
        effectiveness = np.clip(effectiveness, 0, 1)
        
        # Identify ineffective probe types
        ineffective = []
        probe_type_performance = {}
        for p in probe_history[-30:]:
            ptype = p.get('probe_type', 'unknown')
            kl = p.get('kl_gain', 0)
            if ptype not in probe_type_performance:
                probe_type_performance[ptype] = []
            probe_type_performance[ptype].append(kl)
        
        for ptype, kls in probe_type_performance.items():
            if len(kls) >= 3 and np.mean(kls) < self.config.probe_ineffectiveness_threshold:
                ineffective.append(ptype)
        
        redesign = None
        if effectiveness < 0.4:
            redesign = "Consider radical probe strategy redesign: new probe types, different perturbation patterns"
        
        evaluation = ProbeStrategyEvaluation(
            timestamp=episode,
            avg_kl_gain=avg_kl,
            hypothesis_separation_rate=separation_rate,
            probe_effectiveness_score=effectiveness,
            ineffective_probe_types=ineffective,
            recommended_probe_redesign=redesign
        )
        
        self.probe_evaluations.append(evaluation)
        self.current_probe_eval = evaluation
        
        return evaluation
    
    def detect_identity_belief_coupling(
        self,
        identity_history: List[float],
        belief_history: List[float],
        dissonance_history: List[float],
        episode: int
    ) -> IdentityBeliefCoupling:
        """
        Detect if identity is protecting beliefs from dissonance.
        
        Key signal: High correlation between identity and beliefs,
        combined with dissonance suppression.
        """
        min_history = 20
        if len(identity_history) < min_history or len(belief_history) < min_history:
            return IdentityBeliefCoupling(
                timestamp=episode,
                coupling_strength=0.0,
                belief_identity_correlation=0.0,
                protected_beliefs=[],
                dissonance_suppression_detected=False
            )
        
        # Compute correlation between identity and beliefs
        recent_i = np.array(identity_history[-20:])
        recent_b = np.array(belief_history[-20:])
        
        # Normalize
        recent_i = (recent_i - np.mean(recent_i)) / (np.std(recent_i) + 0.001)
        recent_b = (recent_b - np.mean(recent_b)) / (np.std(recent_b) + 0.001)
        
        correlation = np.abs(np.corrcoef(recent_i, recent_b)[0, 1])
        coupling = correlation
        
        # Check for dissonance suppression
        if len(dissonance_history) >= 10:
            recent_d = dissonance_history[-10:]
            # Low dissonance despite belief changes = suppression
            belief_change = np.std(recent_b[-10:])
            dissonance_level = np.mean(recent_d)
            suppression = belief_change > 0.1 and dissonance_level < 0.3
        else:
            suppression = False
        
        protected = []
        if coupling > 0.5:
            protected = ["identity-correlated_boundary_estimate"]
        
        recommendation = ""
        if coupling > self.config.identity_belief_coupling_threshold:
            recommendation = "Decouple identity from specific beliefs. Identity should be about epistemic method, not conclusions."
        
        coupling_detect = IdentityBeliefCoupling(
            timestamp=episode,
            coupling_strength=coupling,
            belief_identity_correlation=correlation,
            protected_beliefs=protected,
            dissonance_suppression_detected=suppression,
            recommendation=recommendation
        )
        
        self.coupling_detections.append(coupling_detect)
        self.current_coupling = coupling_detect
        
        return coupling_detect
    
    def generate_epiphany(
        self,
        critique_type: EpistemicCritiqueType,
        episode: int
    ) -> Optional[EpistemicEpiphany]:
        """
        Generate an epiphany about fundamental epistemic method failure.
        
        This is the core output of Meta²-Cognition: the realization that
        the way of thinking itself must change.
        """
        # Rate limit epiphanies
        if episode - self.last_epiphany_episode < 50:
            return None
        
        # Gather evidence
        confidence = 0.5
        realization = ""
        limitation = ""
        revision = ""
        
        space_expansion = None
        bias_corrections = None
        probe_redesign = None
        decouple = False
        
        if critique_type == EpistemicCritiqueType.HYPOTHESIS_SPACE_INADEQUATE:
            confidence = 0.8 if self.current_audit else 0.5
            realization = "My hypothesis space cannot represent the true model"
            limitation = "I am only considering hypothesis types that exclude the truth"
            revision = "Expand hypothesis space to include more complex structural and causal models"
            if self.current_audit:
                all_types = {'PARAMETRIC_TIME', 'PARAMETRIC_STATE', 'STRUCTURAL_DUAL', 
                            'STRUCTURAL_ASYMMETRIC', 'CAUSAL_CORRELATE', 'CAUSAL_MECHANISM'}
                present = self.current_audit.hypothesis_types_present
                missing = list(all_types - present)
                space_expansion = missing
        
        elif critique_type == EpistemicCritiqueType.OCCAM_BIAS:
            confidence = 0.75 if self.current_bias else 0.5
            realization = "My preference for simple models is blinding me to truth"
            limitation = "The true model may be complex, but I keep choosing simple ones"
            revision = "Reduce Occam penalty and give complex hypotheses fair consideration"
            bias_corrections = ["Reduce simplicity bias", "Increase complexity tolerance"]
        
        elif critique_type == EpistemicCritiqueType.PROBE_STRATEGY_FAILURE:
            confidence = 0.7 if self.current_probe_eval else 0.5
            realization = "My probes are not effectively distinguishing hypotheses"
            limitation = "Current probe design cannot generate evidence to separate competing models"
            revision = "Redesign probe strategy with novel perturbation types"
            if self.current_probe_eval:
                probe_redesign = self.current_probe_eval.recommended_probe_redesign
        
        elif critique_type == EpistemicCritiqueType.IDENTITY_PROTECTED_BELIEF:
            confidence = 0.75 if self.current_coupling else 0.5
            realization = "My identity is protecting beliefs from scrutiny"
            limitation = "I am suppressing dissonance to maintain identity-coherent beliefs"
            revision = "Decouple identity from specific conclusions; tie identity to epistemic virtue"
            decouple = True
        
        elif critique_type == EpistemicCritiqueType.METHOD_EPIPHANY:
            confidence = 0.85
            realization = "My entire epistemic method needs radical revision"
            limitation = "Multiple failures indicate fundamental approach limitation"
            revision = "Consider alternative epistemic frameworks entirely"
            space_expansion = ["All unexplored hypothesis types"]
            bias_corrections = ["Rebalance all epistemic biases"]
            probe_redesign = "Complete probe strategy overhaul"
            decouple = True
        
        # Only generate if confident enough
        if confidence < self.config.epiphany_confidence_min:
            return None
        
        epiphany = EpistemicEpiphany(
            timestamp=episode,
            trigger=critique_type,
            confidence=confidence,
            realization=realization,
            current_method_limitation=limitation,
            proposed_method_revision=revision,
            hypothesis_space_expansion=space_expansion,
            bias_corrections=bias_corrections,
            probe_strategy_redesign=probe_redesign,
            identity_decoupling_recommended=decouple,
            expected_improvement=0.3,
            risk_of_revision=0.2
        )
        
        self.epiphanies.append(epiphany)
        self.epiphany_count += 1
        self.last_epiphany_episode = episode
        
        return epiphany
    
    def step(
        self,
        episode: int,
        hypothesis_space: List[str],
        failure_rate: float,
        belief_history: List[float],
        hypothesis_generator: Any,
        surgical_prober: Any
    ) -> Dict[str, Any]:
        """
        Execute one Meta²-Cognition step.
        
        Returns epiphany if one is triggered.
        """
        # Track failure
        self.failure_history.append(failure_rate > self.config.failure_rate_threshold)
        
        # Run all audits and detections
        hypotheses = getattr(hypothesis_generator, 'hypotheses', {}).values() if hypothesis_generator else []
        
        audit = self.audit_hypothesis_space(
            list(hypotheses), belief_history, episode
        )
        
        bias = self.detect_biases(
            list(hypotheses), [], episode
        )
        
        probe_eval = self.evaluate_probe_strategy(
            [], [], episode
        )
        
        # Check for epiphany triggers
        sustained_failures = sum(self.failure_history) >= self.config.sustained_failure_window * 0.7
        
        epiphany = None
        critique_triggered = None
        
        if sustained_failures:
            # Check which critique applies
            if not audit.space_adequate:
                critique_triggered = EpistemicCritiqueType.HYPOTHESIS_SPACE_INADEQUATE
                epiphany = self.generate_epiphany(critique_triggered, episode)
            
            elif bias.dominant_bias and bias.bias_impact_estimate > 0.3:
                if bias.dominant_bias == 'occam':
                    critique_triggered = EpistemicCritiqueType.OCCAM_BIAS
                elif bias.dominant_bias == 'exploration_deficit':
                    critique_triggered = EpistemicCritiqueType.EXPLORATION_DEFICIT
                epiphany = self.generate_epiphany(critique_triggered, episode)
            
            elif probe_eval.probe_effectiveness_score < 0.4:
                critique_triggered = EpistemicCritiqueType.PROBE_STRATEGY_FAILURE
                epiphany = self.generate_epiphany(critique_triggered, episode)
        
        return {
            'audit': audit,
            'bias': bias,
            'probe_eval': probe_eval,
            'epiphany_triggered': epiphany is not None,
            'critique_issued': critique_triggered.name if critique_triggered else None,
            'epiphany': epiphany
        }
    
    def get_meta2_status(self) -> Dict[str, Any]:
        """Get full Meta²-Cognition status."""
        return {
            'epiphany_count': self.epiphany_count,
            'recent_epiphanies': [
                {
                    'episode': e.timestamp,
                    'realization': e.realization[:50] + '...',
                    'confidence': e.confidence
                }
                for e in self.epiphanies[-3:]
            ],
            'current_audit': {
                'space_adequate': self.current_audit.space_adequate if self.current_audit else None,
                'types_present': list(self.current_audit.hypothesis_types_present) if self.current_audit else []
            },
            'current_bias': {
                'dominant': self.current_bias.dominant_bias if self.current_bias else None,
                'impact': self.current_bias.bias_impact_estimate if self.current_bias else 0
            },
            'failure_rate': np.mean(self.failure_history) if self.failure_history else 0
        }


# Convenience alias
Meta2Cognition = Meta2CognitionEngine
