"""
RAVANA — Recursive Learning Model

A CPU-native ML framework where learning emerges from
pressure-driven self-organization, not gradient descent.

Usage:
    import ravana as torch
    x = torch.tensor([1, 2, 3])
    model = torch.nn.Linear(10, 10)
    y = model(x)
    model.accumulate_pressure(y - target)
    model.sleep_cycle()  # ← instead of optimizer.step()

    # Cognitive system
    from ravana.cognitive import CognitiveFramework
    framework = CognitiveFramework()

    # Recursive Learning Model (alternative to LLM)
    from ravana.nn import RLM
"""

import os
import sys

# ── Path setup ────────────────────────────────────────────────────────
# Extend __path__ to include ravana/src/ravana/ so that subpackages
# (core, language, nn, cognitive, chat, web, etc.) are found there.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_ravana = os.path.join(_this_dir, "src", "ravana")
if os.path.isdir(_src_ravana) and _src_ravana not in __path__:
    __path__.insert(0, _src_ravana)

# ── ML framework re-exports (always available) ──────────────────────
from ravana_ml import (
    RawTensor, StateTensor, Parameter,
    tensor, eye, arange, stack, cat, zeros, ones, randn, from_numpy,
    Device, device, cuda, cuda_is_available,
    is_tensor, no_grad, save, load,
)

# Tensor alias
Tensor = StateTensor

# Re-export submodules from ML framework
from ravana_ml import graph
from ravana_ml import propagation
from ravana_ml import free_energy
from ravana_ml import plasticity

# ── Full ravana submodules (from ravana/src/ravana/) ────────────────
# These become available because __path__ now includes ravana/src/ravana/.
try:
    from ravana import core
except Exception:
    core = None

try:
    from ravana import language
except Exception:
    language = None

try:
    from ravana import nn
except Exception:
    nn = None

try:
    from ravana import cognitive
except Exception:
    cognitive = None

try:
    from ravana import chat
except Exception:
    chat = None

try:
    from ravana import web
except Exception:
    web = None

try:
    from ravana import bootstrap
except Exception:
    bootstrap = None

try:
    from ravana import decoder
except Exception:
    decoder = None

try:
    from ravana import world
except Exception:
    world = None

try:
    from ravana import lab
except Exception:
    lab = None

__version__ = '0.3.2'

__all__ = [
    'RawTensor', 'StateTensor', 'Parameter', 'Tensor',
    'tensor', 'eye', 'arange', 'stack', 'cat', 'zeros', 'ones', 'randn',
    'from_numpy', 'nn', 'cognitive', 'graph', 'propagation', 'pressure',
    'plasticity', 'world', 'lab', 'device', 'cuda', 'cuda_is_available',
    'is_tensor', 'no_grad', 'save', 'load',
    'core', 'language', 'chat', 'web', 'bootstrap', 'decoder',
]
