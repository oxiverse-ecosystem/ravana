"""
Tests for the RAVANA mode orchestrator.
"""

try:
    from .conftest import import_agent
except ImportError:
    from conftest import import_agent

# Dev:  from agent.mode_orchestrator import AgentMode, ModeOrchestrator
# Pkg:  from ravana_grace.agent.mode_orchestrator import …
AgentMode, ModeOrchestrator = import_agent("mode_orchestrator", "AgentMode", "ModeOrchestrator")