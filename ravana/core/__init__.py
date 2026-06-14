"""
Cognitive Core - RAVANA's cognitive engines package.
Contains: emotion, identity, meaning, dual_process, global_workspace, meta_cognition, sleep, belief_store
"""
from .emotion import VADEmotionEngine, VADConfig
from .identity import IdentityEngine, IdentityState
from .meaning import MeaningEngine, MeaningConfig
from .dual_process import DualProcessController, DualProcessConfig, Route
from .global_workspace import GlobalWorkspace, GWConfig
from .meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from .sleep import SleepConsolidation, SleepConfig
from .belief_store import BeliefStore, UserBeliefProfile, BeliefConfig

__all__ = [
    'VADEmotionEngine', 'VADConfig',
    'IdentityEngine', 'IdentityState',
    'MeaningEngine', 'MeaningConfig',
    'DualProcessController', 'DualProcessConfig', 'Route',
    'GlobalWorkspace', 'GWConfig',
    'MetaCognition', 'MetaCognitiveConfig', 'EpistemicMode',
    'SleepConsolidation', 'SleepConfig',
    'BeliefStore', 'UserBeliefProfile', 'BeliefConfig',
]