"""
RLM Package - Decomposed triple-based cognitive architecture.
Contains: RelationPredictor, PropagationEngine, Plasticity
"""
from .relation_predictor import RelationPredictor, RELATION_TYPES, _KEYWORD_MAP
from .propagation import PropagationEngine
from .plasticity import Plasticity

__all__ = [
    'RelationPredictor',
    'PropagationEngine',
    'Plasticity',
    'RELATION_TYPES',
    '_KEYWORD_MAP',
]