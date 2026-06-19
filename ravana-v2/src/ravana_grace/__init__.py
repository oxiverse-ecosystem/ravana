"""RAVANA GRACE -- 27-phase cognitive architecture."""

from .core import (
    Governor,
    IdentityEngine as Identity,
    SleepConsolidation as Sleep,
    VADEmotionEngine as Emotion,
    LearnedWorldModel as WorldModel,
    MeaningEngine as Meaning,
    HumanMemoryEngine as HumanMemory,
)
from .dialogue import DialogueEngine

__version__ = "0.2.0"
__all__ = [
    "Governor",
    "Identity",
    "Sleep",
    "Emotion",
    "WorldModel",
    "Meaning",
    "HumanMemory",
    "DialogueEngine",
]