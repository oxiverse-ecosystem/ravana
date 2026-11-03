"""Trailing-modifier stripping for topic phrases (capability, online-growable).

Brain basis
-----------
Human topic extraction does not keep every word of an utterance: trailing
adverbials ("actually", "so far", "earlier", "yet") are discourse-level
markers, not part of the referent. Keeping them produced garbled stance
topics such as "actually defines person" (round 2026-11-02T0300Z chat
probe, TURN 20). This module strips them, and the modifier lexicon grows
online: a word observed to be a trailing modifier in a grounded extraction
is learned, so RAVANA's notion of "modifier" is shaped by experience rather
than a frozen list.

Design constraints honoured
--------------------------
* No retraining, no LLM.
* No hardcoded reply strings; only a seed *vocabulary* of modifier forms,
  which is online-expandable via ``learn``.
* Seed grows; removing one entry degrades gracefully.
* Cold start: empty learned set -> seed only.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Set


_SEED_MODIFIERS: Set[str] = {
    "actually", "really", "just", "still", "yet", "so", "far",
    "earlier", "before", "now", "today", "lately", "recently",
    "quite", "very", "pretty", "kind", "of", "a", "bit",
    "somehow", "maybe", "perhaps", "always", "never", "ever",
}


class ModifierLexicon:
    """Online-growable set of trailing-modifier words."""

    def __init__(self, seed: Iterable[str] = _SEED_MODIFIERS):
        self._seed: Set[str] = set(seed)
        self._learned: Set[str] = set()

    def seed(self) -> FrozenSet[str]:
        return frozenset(self._seed)

    def learned(self) -> FrozenSet[str]:
        return frozenset(self._learned)

    def is_modifier(self, word: str) -> bool:
        w = (word or "").strip().lower()
        return w in self._seed or w in self._learned

    def learn(self, *words: str) -> Set[str]:
        added: Set[str] = set()
        for w in words:
            w = (w or "").strip().lower()
            if w and w not in self._seed and w not in self._learned:
                self._learned.add(w)
                added.add(w)
        return added

    def to_state(self) -> dict:
        return {"learned": sorted(self._learned)}

    def load_state(self, state) -> None:
        if not state:
            return
        for w in (state.get("learned") or []):
            self._learned.add(str(w).lower())


_DEFAULT = ModifierLexicon()


def strip_trailing_modifiers(phrase: str, lexicon: ModifierLexicon = None) -> str:
    """Strip trailing modifier words from a topic phrase.

    Strips from the right while the last token is a known modifier. Stops
    when a non-modifier token is reached or the phrase would become empty.
    """
    if lexicon is None:
        lexicon = _DEFAULT
    toks = (phrase or "").split()
    # Strip leading modifiers (e.g. "actually defines person" -> "defines person").
    while len(toks) > 1 and lexicon.is_modifier(toks[0]):
        toks.pop(0)
    # Strip trailing modifiers.
    while len(toks) > 1 and lexicon.is_modifier(toks[-1]):
        toks.pop()
    return " ".join(toks)


def default_lexicon() -> ModifierLexicon:
    return _DEFAULT
