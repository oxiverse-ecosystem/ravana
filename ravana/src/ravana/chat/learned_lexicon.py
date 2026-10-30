"""Learned lexicon store — words RAVANA learns at runtime.

A living lexicon: a word learned in one conversation must be usable later,
must participate in composition, and must be revisable when the user
corrects it. No frozen vocabulary, no retraining — entries are added
online from real disclosures and survive via get_state()/set_state().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LexiconEntry:
    word: str
    meaning: str = ""
    context: str = ""
    confidence: float = 0.6
    turn_number: int = 0
    rehearsal_count: int = 1
    source: str = "user_disclosure"
    superseded: bool = False


class LearnedLexiconStore:
    def __init__(self, decay_turns: int = 50):
        # word.lower() -> LexiconEntry
        self.entries: Dict[str, LexiconEntry] = {}
        self.turn_num: int = 0
        self._decay_turns = decay_turns

    def advance_turn(self):
        self.turn_num += 1

    def learn(self, word: str, meaning: str = "", context: str = "",
              confidence: float = 0.6, source: str = "user_disclosure") -> None:
        w = (word or "").strip().lower()
        if not w:
            return
        existing = self.entries.get(w)
        if existing is not None and not existing.superseded:
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.rehearsal_count += 1
            existing.turn_number = self.turn_num
            if meaning:
                existing.meaning = meaning
            if context:
                existing.context = context
            existing.superseded = False
            return
        self.entries[w] = LexiconEntry(
            word=w, meaning=meaning, context=context,
            confidence=confidence, turn_number=self.turn_num,
            rehearsal_count=1, source=source)

    def revise(self, word: str, new_meaning: str) -> None:
        w = (word or "").strip().lower()
        if w in self.entries:
            self.entries[w].meaning = new_meaning
            self.entries[w].confidence = min(1.0, self.entries[w].confidence + 0.2)
            self.entries[w].source = "correction"
            self.entries[w].superseded = False
        else:
            self.learn(w, new_meaning, source="correction")

    def get(self, word: str) -> Optional[LexiconEntry]:
        e = self.entries.get((word or "").strip().lower())
        if e is None or e.superseded:
            return None
        return e

    def known(self, word: str) -> bool:
        return self.get(word) is not None

    def get_state(self) -> Dict:
        return {
            "entries": {k: (e.word, e.meaning, e.context, e.confidence,
                            e.turn_number, e.rehearsal_count, e.source, e.superseded)
                        for k, e in self.entries.items()},
            "turn_num": self.turn_num,
        }

    def set_state(self, state: Dict) -> None:
        self.entries = {}
        for k, v in (state.get("entries") or {}).items():
            w, meaning, ctx, conf, tn, rc, src, sup = v
            self.entries[w.lower().strip()] = LexiconEntry(
                word=w, meaning=meaning, context=ctx, confidence=conf,
                turn_number=tn, rehearsal_count=rc, source=src, superseded=sup)
        self.turn_num = state.get("turn_num", 0)
