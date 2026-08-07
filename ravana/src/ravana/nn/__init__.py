"""Ravana neural network modules — free-energy-driven learning."""

try:
    from ravana_ml.nn import Module, Sequential, Linear, Embedding, LayerNorm, Dropout
    from ravana_ml.nn import RLM
    from ravana_ml.nn import functional
except ImportError:
    from ravana_ml._import_guard_shim import report_missing
    report_missing("ravana_ml.nn", "neural-network modules (RLM, layers, functional)", kind="internal")
    Module = None
    Sequential = None
    Linear = None
    Embedding = None
    LayerNorm = None
    Dropout = None
    RLM = None
    functional = None
