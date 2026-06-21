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

try:
    from importlib.metadata import version as _v
    __version__ = _v("ravana-chat")
except Exception:
    __version__ = "0.0.0"
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