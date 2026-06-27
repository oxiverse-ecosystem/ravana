"""RAVANA modular package -- decoder-first chat with continuous web learning."""

import sys as _sys

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

# Re-export ravana_ml tensor/nn for drop-in compatibility
from ravana_ml import nn as nn
from ravana_ml.tensor import tensor, RawTensor, StateTensor
from ravana_ml.tensor import randn, eye, zeros, ones, from_numpy

Tensor = StateTensor

# Re-export ravana_ml submodules as ravana.* for backward compat
from ravana_ml import propagation as _propagation
from ravana_ml import free_energy as _free_energy
from ravana_ml import plasticity as _plasticity
_sys.modules['ravana.propagation'] = _propagation
_sys.modules['ravana.free_energy'] = _free_energy
_sys.modules['ravana.plasticity'] = _plasticity

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
    "nn",
    "tensor",
    "RawTensor",
    "StateTensor",
    "Tensor",
    "randn",
    "eye",
    "zeros",
    "ones",
    "from_numpy",
]