"""Triplet Inference Operator — learned relational inference (CLS).

Every inference property (transitivity, symmetry, inversion,
composition, hierarchy) is a learned per-predicate statistic gated by
Wilson lower confidence bounds — no hardcoded relational rules, no
fixed behavior thresholds (see core.py / learning.py docstrings).
"""
from .core import (InferenceResult, RelationProfile, RelationalSchema,
                   Triple, wilson_lower)
from .canonical import canonical_predicate
from .memory import TripletMemory
from .learning import ProfileLearner
from .engine import TripletInferenceOperator
from .sleep import SleepSchemaExtractor
from .curiosity import InferenceCuriosityHook
from .abstention import AbstentionGate
from .seed import SEED_TRIPLES

__all__ = [
    "Triple", "RelationProfile", "RelationalSchema", "InferenceResult",
    "wilson_lower", "canonical_predicate", "TripletMemory",
    "ProfileLearner", "TripletInferenceOperator", "SleepSchemaExtractor",
    "InferenceCuriosityHook", "AbstentionGate", "SEED_TRIPLES",
]
