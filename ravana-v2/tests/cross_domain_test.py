"""
EXPERIMENT 3: Cross-Environment Generalization
Tests if an identity trained in Survival carries over to the Classroom.
"""
try:
    from .conftest import import_core
except ImportError:
    from conftest import import_core

Governor, ResolutionEngine, IdentityEngine, StateManager = import_core(
    "", "Governor", "ResolutionEngine", "IdentityEngine", "StateManager"
)