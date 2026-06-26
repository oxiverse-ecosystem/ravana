"""Ravana free_energy module — re-export from ML framework (optional)."""

try:
    from ravana_ml.free_energy import *
except ImportError:
    pass
