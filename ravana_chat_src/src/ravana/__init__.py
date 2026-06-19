"""RAVANA modular package -- decoder-first chat with continuous web learning."""

from .core import (
    VADEmotionEngine,
    IdentityEngine,
    MeaningEngine,
    GlobalWorkspace,
    MetaCognition,
    SleepConsolidation,
    BeliefStore,
)
from .chat import ChatInterface
from .web import WebLearner

__version__ = "0.1.1"
__all__ = [
    "VADEmotionEngine",
    "IdentityEngine",
    "MeaningEngine",
    "GlobalWorkspace",
    "MetaCognition",
    "SleepConsolidation",
    "BeliefStore",
    "ChatInterface",
    "WebLearner",
]