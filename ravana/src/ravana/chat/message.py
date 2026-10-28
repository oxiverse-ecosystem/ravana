"""Message — the structured, non-linguistic intent handed to the emitter.

Per the Language Production Pivot (docs/LANGUAGE_PRODUCTION_PIVOT.md, M1/M5):
the arrow into the emitter is a ``Message``, not a string. A Message is the
output of conceptual preparation: who did what to whom, with what focus. It
contains no words and no authored text. Every downstream decision (lemma
selection, unification, settling, gating) reads from this structure.

Online-learnable by construction: a Message is built from the typed concept
graph at runtime; nothing in this module is authored per-topic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureBundle:
    """Feature set for a single lemma (M2): meaning + syntax, no surface form."""
    pos: str = ""                 # part of speech
    number: str = "sg"            # sg / pl
    tense: str = "present"        # present / past / future
    aspect: str = "simple"        # simple / progressive / perfect
    polarity: float = 0.0         # -1..+1 evaluative polarity


@dataclass
class Message:
    """Non-linguistic message produced by conceptual preparation (M1/M5).

    Attributes:
        predicate_edge: the typed relation edge id that grounds the predicate
            (from the typed concept graph). None if no grounded predicate.
        agent_node: concept-graph node id for the agent/subject. None if none.
        patient_node: concept-graph node id for the patient/object. None if none.
        modifiers: list of (role, node_id) modifier pairs (mood, time, place, ...).
        focus: which constituent carries the information focus.
    """
    predicate_edge: Optional[str] = None
    agent_node: Optional[str] = None
    patient_node: Optional[str] = None
    modifiers: List = field(default_factory=list)
    focus: str = "agent"

    def is_grounded(self) -> bool:
        """True iff the message carries at least one grounded graph node."""
        return bool(self.predicate_edge or self.agent_node or self.patient_node
                    or self.modifiers)

    def active_nodes(self) -> List[str]:
        """The sparse active set (M7): content nodes, capped at 4."""
        out = []
        if self.agent_node:
            out.append(self.agent_node)
        if self.patient_node:
            out.append(self.patient_node)
        for _, n in self.modifiers:
            if n and n not in out:
                out.append(n)
        return out[:4]
