"""
RAVANA v2 — Dialogue System Package

A multi-turn conversational agent that learns from corrections,
maintains per-user models, and consolidates experience through sleep
— all without backpropagation.

Subsystems:
1. DialogueContext (working memory with salience decay)
2. ConversationalRepair (correction handling via local plasticity)
3. DialogueEngine (main orchestrator tying all subsystems together)
"""

from .dialogue_engine import DialogueEngine, DialogueEngineConfig, DialogueTurnRecord

__all__ = ["DialogueEngine", "DialogueEngineConfig", "DialogueTurnRecord"]
