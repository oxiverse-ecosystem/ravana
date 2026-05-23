"""
RAVANA — A CPU-native ML framework where learning emerges from
free-energy-driven self-organization, not gradient descent.

Usage:
    import ravana as torch
    x = torch.tensor([1, 2, 3])
    model = torch.nn.Linear(10, 10)
    y = model(x)
    model.accumulate_free_energy(y - target)
    model.sleep_cycle()  # ← instead of optimizer.step()
"""

from .tensor import (
    RawTensor, StateTensor, Parameter,
    tensor, eye, arange, stack, cat, zeros, ones, randn, from_numpy,
)

Tensor = StateTensor

from . import graph
from . import propagation
from . import free_energy
from . import plasticity
from . import world
from . import nn
from . import currency

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

def save(obj, path):
    """Save a model or state dict to disk.

    For RLM models: saves complete checkpoint (weights + graph + scalars).
        Use .pkl extension for pickle, .zip for human-readable zip archive.
    For other modules: saves state dict with cognitive metadata.
    """
    import pickle
    if hasattr(obj, 'save') and callable(obj.save):
        if path.endswith('.zip') and hasattr(obj, 'save_zip'):
            obj.save_zip(path)
        else:
            obj.save(path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj.state_dict() if hasattr(obj, 'state_dict') else obj, f)

def load(path_or_model, path=None):
    """Load a model from disk.

    Usage:
        model = ravana.load("checkpoint.pkl")           # RLM pickle (auto-detect)
        model = ravana.load("checkpoint.zip")           # RLM zip (auto-detect)
        model = ravana.load(model, "checkpoint.pkl")    # any Module (state dict)
    """
    import pickle
    import zipfile
    if path is None:
        # First arg is the path — auto-detect format
        if path_or_model.endswith('.zip') or zipfile.is_zipfile(path_or_model):
            from .nn.rlm import RLM
            return RLM.load_zip(path_or_model)
        with open(path_or_model, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "config" in data and "graph" in data:
            from .nn.rlm import RLM
            return RLM.load(path_or_model)
        return data
    else:
        # First arg is a model, second is path
        with open(path, 'rb') as f:
            sd = pickle.load(f)
        path_or_model.load_state_dict(sd)
        return path_or_model

__version__ = '0.1.0'

__all__ = [
    'RawTensor', 'StateTensor', 'Parameter', 'Tensor',
    'tensor', 'eye', 'arange', 'stack', 'cat', 'zeros', 'ones', 'randn',
    'from_numpy', 'nn', 'graph', 'propagation', 'free_energy', 'plasticity',
    'world', 'device', 'cuda', 'cuda_is_available', 'is_tensor',
    'no_grad', 'save', 'load',
]
