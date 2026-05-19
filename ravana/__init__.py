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

# Re-export everything from the ML framework
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
from ravana_ml import pressure
from ravana_ml import plasticity

# Ravana-specific submodules
from ravana import nn
from ravana import cognitive
from ravana import world
from ravana import lab

__version__ = '0.1.0'

__all__ = [
    'RawTensor', 'StateTensor', 'Parameter', 'Tensor',
    'tensor', 'eye', 'arange', 'stack', 'cat', 'zeros', 'ones', 'randn',
    'from_numpy', 'nn', 'cognitive', 'graph', 'propagation', 'pressure',
    'plasticity', 'world', 'lab', 'device', 'cuda', 'cuda_is_available',
    'is_tensor', 'no_grad', 'save', 'load',
]
