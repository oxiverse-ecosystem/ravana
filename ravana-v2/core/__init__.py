"""
RAVANA v2 Core Modules

A self-stabilizing, self-expanding epistemic system with active experimentation
and explicit Occam discipline.
"""

# Phase A: Core Regulation
from .governor import Governor, GovernorConfig, RegulationMode, ClampDiagnostics
from .resolution import ResolutionEngine, ResolutionMemory
from .identity import IdentityEngine, IdentityState
from .state import StateManager, CognitiveState

# Phase B: Adaptation
from .adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig

# Phase C: Strategy Learning
from .strategy import StrategyLayer, StrategyConfig, ExplorationMode, ModeSelection, BehavioralContext
from .strategy_learning import StrategyLearningLayer, ModeOutcome, LearningConfig, StrategyWithLearning

# Phase D: Intent
from .intent import IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective

# Phase D.5: Planning
from .planning import MicroPlanner, PlanningConfig, SimulatedFuture

# Phase E: Non-Stationary Environment
from .environment import NonStationaryEnvironment, EnvironmentConfig, HiddenDynamics, WorldState

# Phase F: Predictive World Model
from .predictive_world import LearnedWorldModel, WorldModelConfig, PredictedState, AnomalyEvent, FalseWorldTester

# Phase F.5: Belief Reasoning
from .belief_reasoner import BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent

# Phase G: Active Epistemology
from .active_epistemology import ActiveEpistemology, VoIConfig, InformationGainMethod, HypothesisDrivenActionSelector

# Phase G.5: Surgical Probing
from .surgical_probes import SurgicalProbeSelector, SurgicalProbeConfig, ProbeType, ProbeExperiment, SurgicalProbing

# Phase J: Hypothesis Generation
from .hypothesis_generation import HypothesisGenerator, GenerationConfig, HypothesisType, GeneratedHypothesis

# Phase J.1: Occam Layer (Hypothesis Discipline)
from .occam_layer import OccamLayer, OccamConfig, HypothesisScore, DisciplinedBeliefSystem
from .meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode, ProbeResult, ReasoningQualityTracker, ConfidenceCalibrator, BiasDetector

# Phase K: VAD Emotion
from .emotion import VADEmotionEngine, VADConfig, VADState

# Phase K.5: Empathy
from .empathy import EmpathyEngine, EmpathyConfig, OtherMind

# Phase L: Sleep & Dream Consolidation
from .sleep import SleepConsolidation, SleepConfig, SleepStage, SleepRecord, DreamPerturbationType

# Phase L.5: Dual-Process Controller
from .dual_process import DualProcessController, DualProcessConfig, ProcessingRoute, RouteDecision

# Phase M: Meaning Engine
from .meaning import MeaningEngine, MeaningConfig, MeaningRecord

# Phase N: Global Workspace
from .global_workspace import GlobalWorkspace, GWConfig, GWContent

__all__ = [
    # Phase A
    "Governor", "GovernorConfig", "RegulationMode", "ClampDiagnostics",
    "ResolutionEngine", "ResolutionMemory",
    "IdentityEngine", "IdentityState",
    "StateManager", "CognitiveState",
    # Phase B
    "PolicyTweakLayer", "AdaptiveGovernorBridge", "AdaptationConfig",
    # Phase C
    "StrategyLayer", "StrategyConfig", "ExplorationMode", "ModeSelection", "BehavioralContext",
    "StrategyLearningLayer", "ModeOutcome", "LearningConfig", "StrategyWithLearning",
    # Phase D
    "IntentEngine", "IntentConfig", "IntentAwareStrategy", "SystemObjective",
    # Phase D.5
    "MicroPlanner", "PlanningConfig", "SimulatedFuture",
    # Phase E
    "NonStationaryEnvironment", "EnvironmentConfig", "HiddenDynamics", "WorldState",
    # Phase F
    "LearnedWorldModel", "WorldModelConfig", "PredictedState", "AnomalyEvent", "FalseWorldTester",
    # Phase F.5
    "BeliefReasoner", "BeliefConfig", "Hypothesis", "EvidenceEvent",
    # Phase G
    "ActiveEpistemology", "VoIConfig", "InformationGainMethod", "HypothesisDrivenActionSelector",
    # Phase G.5
    "SurgicalProbeSelector", "SurgicalProbeConfig", "ProbeType", "ProbeExperiment", "SurgicalProbing",
    # Phase J
    "HypothesisGenerator", "GenerationConfig", "HypothesisType", "GeneratedHypothesis",
    # Phase J.1
    "OccamLayer", "OccamConfig", "HypothesisScore", "DisciplinedBeliefSystem",
    # Phase K
    "VADEmotionEngine", "VADConfig", "VADState",
    # Phase K.5
    "EmpathyEngine", "EmpathyConfig", "OtherMind",
    # Phase L
    "SleepConsolidation", "SleepConfig", "SleepStage", "SleepRecord", "DreamPerturbationType",
    # Phase L.5
    "DualProcessController", "DualProcessConfig", "ProcessingRoute", "RouteDecision",
    # Phase M
    "MeaningEngine", "MeaningConfig", "MeaningRecord",
    # Phase N
    "GlobalWorkspace", "GWConfig", "GWContent",
]
