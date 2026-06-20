"""
EXPERIMENT 2: Metric Ablation Study
Tests if the Dissonance mechanism is necessary for stability and fairness.
"""
try:
    from .conftest import import_core
except ImportError:
    from conftest import import_core

Governor, ResolutionEngine, IdentityEngine, StateManager = import_core(
    "", "Governor", "ResolutionEngine", "IdentityEngine", "StateManager"
)