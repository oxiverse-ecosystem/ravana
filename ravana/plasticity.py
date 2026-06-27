"""Ravana plasticity module — re-export from ML framework (optional)."""

try:
    from ravana_ml.plasticity import *
except ImportError:
    pass
