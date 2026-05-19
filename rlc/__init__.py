"""
RLC — Recursive Learning Model

A CPU-native ML framework where learning emerges from
pressure-driven self-organization, not gradient descent.

Usage:
    import rlc as torch
    x = torch.tensor([1, 2, 3])
    model = torch.nn.Linear(10, 10)
    y = model(x)
    model.accumulate_pressure(y - target)
    model.sleep_cycle()  # ← instead of optimizer.step()

    # Cognitive system
    from rlc.cognitive import CognitiveFramework
    framework = CognitiveFramework()
"""

# Re-export everything from ravana
from ravana import (
    RawTensor, StateTensor, Parameter,
    tensor, eye, arange, stack, cat, zeros, ones, randn, from_numpy,
    Device, device, cuda, cuda_is_available,
    is_tensor, no_grad, save, load,
)

# Tensor alias
Tensor = StateTensor

# Re-export submodules
from ravana import graph
from ravana import propagation
from ravana import pressure
from ravana import plasticity

# RLC-specific submodules
from rlc import nn
from rlc import cognitive
from rlc import world
from rlc import lab

__version__ = '0.1.0'

__all__ = [
    'RawTensor', 'StateTensor', 'Parameter', 'Tensor',
    'tensor', 'eye', 'arange', 'stack', 'cat', 'zeros', 'ones', 'randn',
    'from_numpy', 'nn', 'cognitive', 'graph', 'propagation', 'pressure',
    'plasticity', 'world', 'lab', 'device', 'cuda', 'cuda_is_available',
    'is_tensor', 'no_grad', 'save', 'load',
]
