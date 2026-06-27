"""
Cognitive Core - RAVANA's cognitive engines package.
Contains: emotion, identity, meaning, dual_process, global_workspace, meta_cognition,
sleep, belief_store, mirror, hippocampal_buffer, proposition_parser, causal_schema,
implicature_detector, relation_memory, quantity_modifier
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
from .hippocampal_buffer import HippocampalBuffer, HippocampalConfig, FactTriple
from .proposition_parser import PropositionParser, Proposition
from .causal_schema import CausalSchemaLearner, CausalSchemaConfig, CausalSchema
from .implicature_detector import ImplicatureDetector, ImplicatureResult
from .relation_memory import RelationMemory, RelationMemoryConfig, ComparativeRelation
from .quantity_modifier import QuantityModifierSystem, QuantityModifier
from .analogy_engine import AnalogyEngine, AnalogyConfig, AnalogicalMapping, solve_analogy_query
from .abstraction_engine import AbstractionEngine, AbstractionConfig, AbstractionResult, AbstractionPerspective, analyze_abstract_concept

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
    'HippocampalBuffer', 'HippocampalConfig', 'FactTriple',
    'PropositionParser', 'Proposition',
    'CausalSchemaLearner', 'CausalSchemaConfig', 'CausalSchema',
    'ImplicatureDetector', 'ImplicatureResult',
    'RelationMemory', 'RelationMemoryConfig', 'ComparativeRelation',
    'QuantityModifierSystem', 'QuantityModifier',
]