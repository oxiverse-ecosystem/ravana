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

# ── ML framework re-exports (dynamic — ravana-ml is optional) ───────
try:
    from ravana_ml import (
        RawTensor, StateTensor, Parameter,
        tensor, eye, arange, stack, cat, zeros, ones, randn, from_numpy,
        Device, device, cuda, cuda_is_available,
        is_tensor, no_grad, save, load,
    )
    Tensor = StateTensor
    _HAVE_RAVANA_ML = True
except ImportError:
    RawTensor = None
    StateTensor = None
    Parameter = None
    tensor = None
    eye = None
    arange = None
    stack = None
    cat = None
    zeros = None
    ones = None
    randn = None
    from_numpy = None
    Device = None
    device = None
    cuda = None
    cuda_is_available = False
    is_tensor = None
    no_grad = None
    save = None
    load = None
    Tensor = None
    _HAVE_RAVANA_ML = False

# Re-export submodules from ML framework (dynamic)
try:
    from ravana_ml import graph as _graph_module
    graph = _graph_module
except ImportError:
    graph = None

try:
    from ravana_ml import propagation as _propagation_module
    propagation = _propagation_module
except ImportError:
    propagation = None

try:
    from ravana_ml import free_energy as _free_energy_module
    free_energy = _free_energy_module
except ImportError:
    free_energy = None

try:
    from ravana_ml import plasticity as _plasticity_module
    plasticity = _plasticity_module
except ImportError:
    plasticity = None

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

try:
    from importlib.metadata import version as _v
    __version__ = _v("ravana-cognitive")
except Exception:
    __version__ = '0.3.5'

__all__ = [
    'RawTensor', 'StateTensor', 'Parameter', 'Tensor',
    'tensor', 'eye', 'arange', 'stack', 'cat', 'zeros', 'ones', 'randn',
    'from_numpy', 'nn', 'cognitive', 'graph', 'propagation', 'pressure',
    'plasticity', 'world', 'lab', 'device', 'cuda', 'cuda_is_available',
    'is_tensor', 'no_grad', 'save', 'load',
    'core', 'language', 'chat', 'web', 'bootstrap', 'decoder',
]
