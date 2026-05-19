"""
RLC Cognitive Core — GRACE Architecture

Governance, Reflection, Adaptation, Constraint, Exploration.

All cognitive modules from RAVANA v2, accessible via rlc.cognitive.
"""

import sys
import importlib
from pathlib import Path

# ravana-v2 has a hyphen, so we can't import it directly.
# Add its parent to sys.path so we can import 'core' as a package.
_ravana_v2_root = str(Path(__file__).resolve().parent.parent.parent / "ravana-v2")
if _ravana_v2_root not in sys.path:
    sys.path.insert(0, _ravana_v2_root)

# Import the core package (uses relative imports internally, which work
# because 'core' is a proper package directory with __init__.py)
from core import (
    # Phase A: Core Regulation
    Governor, GovernorConfig, RegulationMode, ClampDiagnostics,
    ResolutionEngine, ResolutionMemory,
    IdentityEngine, IdentityState,
    StateManager, CognitiveState,

    # Phase B: Adaptation
    PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig,

    # Phase C: Strategy Learning
    StrategyLayer, StrategyConfig, ExplorationMode, ModeSelection, BehavioralContext,
    StrategyLearningLayer, ModeOutcome, LearningConfig, StrategyWithLearning,

    # Phase D: Intent
    IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective,

    # Phase D.5: Planning
    MicroPlanner, PlanningConfig, SimulatedFuture,

    # Phase E: Non-Stationary Environment
    NonStationaryEnvironment, EnvironmentConfig, HiddenDynamics, WorldState,

    # Phase F: Predictive World Model
    LearnedWorldModel, WorldModelConfig, PredictedState, AnomalyEvent, FalseWorldTester,

    # Phase F.5: Belief Reasoning
    BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent,

    # Phase G: Active Epistemology
    ActiveEpistemology, VoIConfig, InformationGainMethod, HypothesisDrivenActionSelector,

    # Phase G.5: Surgical Probing
    SurgicalProbeSelector, SurgicalProbeConfig, ProbeType, ProbeExperiment, SurgicalProbing,

    # Phase J: Hypothesis Generation
    HypothesisGenerator, GenerationConfig, HypothesisType, GeneratedHypothesis,

    # Phase J.1: Occam Layer
    OccamLayer, OccamConfig, HypothesisScore, DisciplinedBeliefSystem,

    # Phase K: Meta-Cognition
    MetaCognition, MetaCognitiveConfig, EpistemicMode, ProbeResult,
    ReasoningQualityTracker, ConfidenceCalibrator, BiasDetector,

    # Phase K.5: VAD Emotion
    VADEmotionEngine, VADConfig, VADState,

    # Phase K.5: Empathy
    EmpathyEngine, EmpathyConfig, OtherMind,

    # Phase L: Sleep & Dream
    SleepConsolidation, SleepConfig, SleepStage, SleepRecord, DreamPerturbationType,

    # Phase L.5: Dual-Process
    DualProcessController, DualProcessConfig, ProcessingRoute, RouteDecision,

    # Phase M: Meaning
    MeaningEngine, MeaningConfig, MeaningRecord,

    # Phase N: Global Workspace
    GlobalWorkspace, GWConfig, GWContent,
)

# Also make submodules accessible
from core import (
    memory,
    reality_friction,
    meta2_cognition,
    meta2_integration,
    social_epistemology,
)

# Framework API — the top-level user interface
from rlc.cognitive.framework import CognitiveFramework, FrameworkConfig

__all__ = [
    # Core regulation
    "Governor", "GovernorConfig", "RegulationMode", "ClampDiagnostics",
    "ResolutionEngine", "ResolutionMemory",
    "IdentityEngine", "IdentityState",
    "StateManager", "CognitiveState",
    # Adaptation
    "PolicyTweakLayer", "AdaptiveGovernorBridge", "AdaptationConfig",
    # Strategy
    "StrategyLayer", "StrategyConfig", "ExplorationMode", "ModeSelection",
    "StrategyLearningLayer", "ModeOutcome", "LearningConfig", "StrategyWithLearning",
    # Intent & Planning
    "IntentEngine", "IntentConfig", "IntentAwareStrategy", "SystemObjective",
    "MicroPlanner", "PlanningConfig", "SimulatedFuture",
    # Environment
    "NonStationaryEnvironment", "EnvironmentConfig", "HiddenDynamics", "WorldState",
    # World Model
    "LearnedWorldModel", "WorldModelConfig", "PredictedState", "AnomalyEvent", "FalseWorldTester",
    # Belief
    "BeliefReasoner", "BeliefConfig", "Hypothesis", "EvidenceEvent",
    # Epistemology
    "ActiveEpistemology", "VoIConfig", "InformationGainMethod", "HypothesisDrivenActionSelector",
    "SurgicalProbeSelector", "SurgicalProbeConfig", "ProbeType", "ProbeExperiment", "SurgicalProbing",
    # Hypothesis
    "HypothesisGenerator", "GenerationConfig", "HypothesisType", "GeneratedHypothesis",
    "OccamLayer", "OccamConfig", "HypothesisScore", "DisciplinedBeliefSystem",
    # Meta-cognition
    "MetaCognition", "MetaCognitiveConfig", "EpistemicMode", "ProbeResult",
    "ReasoningQualityTracker", "ConfidenceCalibrator", "BiasDetector",
    # Emotion & Empathy
    "VADEmotionEngine", "VADConfig", "VADState",
    "EmpathyEngine", "EmpathyConfig", "OtherMind",
    # Sleep & Dual-Process
    "SleepConsolidation", "SleepConfig", "SleepStage", "SleepRecord", "DreamPerturbationType",
    "DualProcessController", "DualProcessConfig", "ProcessingRoute", "RouteDecision",
    # Meaning & Global Workspace
    "MeaningEngine", "MeaningConfig", "MeaningRecord",
    "GlobalWorkspace", "GWConfig", "GWContent",
    # Framework
    "CognitiveFramework", "FrameworkConfig",
]
