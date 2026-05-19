"""
RAVANA — A CPU-native ML framework where learning emerges from
pressure-driven self-organization, not gradient descent.

Usage:
    import ravana as torch
    x = torch.tensor([1, 2, 3])
    model = torch.nn.Linear(10, 10)
    y = model(x)
    model.accumulate_pressure(y - target)
    model.sleep_cycle()  # ← instead of optimizer.step()
"""

from .tensor import (
    RawTensor, StateTensor, Parameter,
    tensor, eye, arange, stack, cat, zeros, ones, randn, from_numpy,
)

Tensor = StateTensor

from . import graph
from . import propagation
from . import pressure
from . import plasticity
from . import world
from . import nn

# ── device management (CPU-only, for API compatibility) ──

class Device:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"device('{self.name}')"
    def __eq__(self, other):
        return isinstance(other, Device) and self.name == other.name

device = Device('cpu')
cuda = Device('cuda')
cuda_is_available = False

def is_tensor(obj):
    return isinstance(obj, (RawTensor, StateTensor))

def no_grad():
    class _NoGrad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return _NoGrad()

def save(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model.state_dict(), f)

def load(model, path):
    import pickle
    with open(path, 'rb') as f:
        sd = pickle.load(f)
    model.load_state_dict(sd)
    return model

__version__ = '0.1.0'

__all__ = [
    'RawTensor', 'StateTensor', 'Parameter', 'Tensor',
    'tensor', 'eye', 'arange', 'stack', 'cat', 'zeros', 'ones', 'randn',
    'from_numpy', 'nn', 'graph', 'propagation', 'pressure', 'plasticity',
    'world', 'device', 'cuda', 'cuda_is_available', 'is_tensor',
    'no_grad', 'save', 'load',
]
