"""
Cognitive Core - RAVANA's cognitive engines package.
Contains: emotion, identity, meaning, dual_process, global_workspace, meta_cognition, sleep, belief_store, mirror
"""
from .emotion import VADEmotionEngine, VADConfig
from .identity import IdentityEngine, IdentityState, IdentityConfig
from .meaning import MeaningEngine, MeaningConfig
from .dual_process import DualProcessController, DualProcessConfig, Route
from .global_workspace import GlobalWorkspace, GWConfig
from .meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from .sleep import SleepConsolidation, SleepConfig
from .belief_store import BeliefStore, UserBeliefProfile, BeliefConfig
from .mirror import EmotionalMirrorEngine, MirrorConfig, MirrorState, UserEmotionDetector

__all__ = [
    'VADEmotionEngine', 'VADConfig',
    'IdentityEngine', 'IdentityState', 'IdentityConfig',
    'MeaningEngine', 'MeaningConfig',
    'DualProcessController', 'DualProcessConfig', 'Route',
    'GlobalWorkspace', 'GWConfig',
    'MetaCognition', 'MetaCognitiveConfig', 'EpistemicMode',
    'SleepConsolidation', 'SleepConfig',
    'BeliefStore', 'UserBeliefProfile', 'BeliefConfig',
    'EmotionalMirrorEngine', 'MirrorConfig', 'MirrorState', 'UserEmotionDetector',
]