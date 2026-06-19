"""
RAVANA Cognitive Core — GRACE Architecture

Governance, Reflection, Adaptation, Constraint, Exploration.

All cognitive modules from RAVANA v2, accessible via ravana.cognitive.
"""

from ravana_grace.core import (
    Governor, GovernorConfig, RegulationMode, ClampDiagnostics,
    ResolutionEngine, ResolutionMemory,
    IdentityEngine, IdentityState,
    StateManager, CognitiveState,
    PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig,
    StrategyLayer, StrategyConfig, ExplorationMode, ModeSelection, BehavioralContext,
    StrategyLearningLayer, ModeOutcome, LearningConfig, StrategyWithLearning,
    IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective,
    MicroPlanner, PlanningConfig, SimulatedFuture,
    NonStationaryEnvironment, EnvironmentConfig, HiddenDynamics, WorldState,
    LearnedWorldModel, WorldModelConfig, PredictedState, AnomalyEvent, FalseWorldTester,
    BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent,
    ActiveEpistemology, VoIConfig, InformationGainMethod, HypothesisDrivenActionSelector,
    SurgicalProbeSelector, SurgicalProbeConfig, ProbeType, ProbeExperiment, SurgicalProbing,
    HypothesisGenerator, GenerationConfig, HypothesisType, GeneratedHypothesis,
    OccamLayer, OccamConfig, HypothesisScore, DisciplinedBeliefSystem,
    MetaCognition, MetaCognitiveConfig, EpistemicMode, ProbeResult,
    ReasoningQualityTracker, ConfidenceCalibrator, BiasDetector,
    VADEmotionEngine, VADConfig, VADState,
    EmpathyEngine, EmpathyConfig, OtherMind,
    SleepConsolidation, SleepConfig, SleepStage, SleepRecord, DreamPerturbationType,
    DualProcessController, DualProcessConfig, ProcessingRoute, RouteDecision,
    MeaningEngine, MeaningConfig, MeaningRecord,
    GlobalWorkspace, GWConfig, GWContent,
    HumanMemoryEngine, HumanMemoryConfig, HumanMemoryRecord,
)

import ravana_grace.core.memory as memory
import ravana_grace.core.reality_friction as reality_friction
import ravana_grace.core.meta2_cognition as meta2_cognition
import ravana_grace.core.meta2_integration as meta2_integration
import ravana_grace.core.social_epistemology as social_epistemology

# Framework API — the top-level user interface
from .framework import CognitiveFramework, FrameworkConfig

__all__ = [
    "Governor", "GovernorConfig", "RegulationMode", "ClampDiagnostics",
    "ResolutionEngine", "ResolutionMemory",
    "IdentityEngine", "IdentityState",
    "StateManager", "CognitiveState",
    "PolicyTweakLayer", "AdaptiveGovernorBridge", "AdaptationConfig",
    "StrategyLayer", "StrategyConfig", "ExplorationMode", "ModeSelection",
    "StrategyLearningLayer", "ModeOutcome", "LearningConfig", "StrategyWithLearning",
    "IntentEngine", "IntentConfig", "IntentAwareStrategy", "SystemObjective",
    "MicroPlanner", "PlanningConfig", "SimulatedFuture",
    "NonStationaryEnvironment", "EnvironmentConfig", "HiddenDynamics", "WorldState",
    "LearnedWorldModel", "WorldModelConfig", "PredictedState", "AnomalyEvent", "FalseWorldTester",
    "BeliefReasoner", "BeliefConfig", "Hypothesis", "EvidenceEvent",
    "ActiveEpistemology", "VoIConfig", "InformationGainMethod", "HypothesisDrivenActionSelector",
    "SurgicalProbeSelector", "SurgicalProbeConfig", "ProbeType", "ProbeExperiment", "SurgicalProbing",
    "HypothesisGenerator", "GenerationConfig", "HypothesisType", "GeneratedHypothesis",
    "OccamLayer", "OccamConfig", "HypothesisScore", "DisciplinedBeliefSystem",
    "MetaCognition", "MetaCognitiveConfig", "EpistemicMode", "ProbeResult",
    "ReasoningQualityTracker", "ConfidenceCalibrator", "BiasDetector",
    "VADEmotionEngine", "VADConfig", "VADState",
    "EmpathyEngine", "EmpathyConfig", "OtherMind",
    "SleepConsolidation", "SleepConfig", "SleepStage", "SleepRecord", "DreamPerturbationType",
    "DualProcessController", "DualProcessConfig", "ProcessingRoute", "RouteDecision",
    "MeaningEngine", "MeaningConfig", "MeaningRecord",
    "GlobalWorkspace", "GWConfig", "GWContent",
    "HumanMemoryEngine", "HumanMemoryConfig", "HumanMemoryRecord",
    "CognitiveFramework", "FrameworkConfig",
]
