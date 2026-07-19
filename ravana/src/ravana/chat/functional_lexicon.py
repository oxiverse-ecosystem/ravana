"""Consolidated closed-class functional lexicon (Stage 5b-ii, de-hardcoding plan).

Collapses the duplicated inline functional lexicons that were scattered
across ``engine.py`` (`_generic` at 7729, `_FRAMING` at 10665, `_bare_moral` at
7661, `_INC/_DEC/_REM` at 7614-7618) into ONE data-driven source of truth.

These are *functional* (closed-class) primitives — quantity/negation/polarity
markers, moral-question cues, and generic question framing words. Per the
code's own comments they are legitimate to keep as curated seeds (the brain
tracks closed-class modification words for quantity/causal reasoning), but
they must live in a single ``data/*.json`` file, not three+ inline copies, so
there is one authority and they can be EER-fit / decayed centrally.

Mirrors the ``SnippetPEConfig`` / ``SaladClassifier`` pattern: ``load()`` fails
open to seed constants when the fit file is absent, so removing the inline
copies never regresses behavior.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Set

# This file lives at <repo>/ravana/src/ravana/chat/functional_lexicon.py.
# To reach <repo>/data go up 5 levels: chat -> ravana -> src -> ravana -> <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_FIT_PATH = os.path.join(_DATA_DIR, "functional_lexicon.json")

# Seed = the consolidated current inline sets (verified, day-one identical).
# Duplicates across _generic / _FRAMING / _bare_moral are unioned here.
_SEED: Dict[str, list] = {
    # Quantity / polarity primitives (closed-class modifiers).
    "polarity_increase": ["doubl", "tripl", "increase", "more", "stronger",
                          "higher", "grow", "add", "extra", "boost", "amplif",
                          "intensif"],
    "polarity_decrease": ["halv", "less", "weaker", "lower", "reduc", "shrink",
                          "diminish"],
    "polarity_remove": ["without", "gone", "disappear", "vanish", "remov",
                        "lost", "absent", "cease", "eliminat", "no longer",
                        "none"],
    # Moral / advice question cues.
    "moral_markers": ["okay", "ok", "moral", "should", "ethical", "fair",
                      "promise", "lie", "ever"],
    "moral_ambiguous": ["right", "wrong"],
    # Generic question-framing words (signal a malformed multi-word subject).
    "framing": ["ever", "okay", "ok", "break", "make", "really", "right",
               "wrong", "thing", "things", "actually", "question", "answer"],
}


class FunctionalLexicon:
    """Single source of truth for closed-class functional lexicons.

    Every accessor returns a ``set`` (seed-fallback per key) so a partial fit
    file never breaks a consumer.
    """

    def __init__(self, values: Optional[Dict[str, list]] = None) -> None:
        self._v: Dict[str, Set[str]] = {
            k: set(v) for k, v in _SEED.items()}
        if values:
            for k, v in values.items():
                if k in _SEED:
                    self._v[k] = set(v)

    @property
    def polarity_increase(self) -> Set[str]:
        return self._v.get("polarity_increase", set(_SEED["polarity_increase"]))

    @property
    def polarity_decrease(self) -> Set[str]:
        return self._v.get("polarity_decrease", set(_SEED["polarity_decrease"]))

    @property
    def polarity_remove(self) -> Set[str]:
        return self._v.get("polarity_remove", set(_SEED["polarity_remove"]))

    @property
    def moral_markers(self) -> Set[str]:
        return self._v.get("moral_markers", set(_SEED["moral_markers"]))

    @property
    def moral_ambiguous(self) -> Set[str]:
        return self._v.get("moral_ambiguous", set(_SEED["moral_ambiguous"]))

    @property
    def framing(self) -> Set[str]:
        return self._v.get("framing", set(_SEED["framing"]))

    @classmethod
    def load(cls) -> Optional["FunctionalLexicon"]:
        if not os.path.exists(_FIT_PATH):
            return None
        try:
            with open(_FIT_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return cls(values=d)
        except Exception:
            return None

    def save(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        out = {k: sorted(v) for k, v in self._v.items()}
        with open(_FIT_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    def to_dict(self) -> Dict[str, list]:
        return {k: sorted(v) for k, v in self._v.items()}


def default_lexicon() -> "FunctionalLexicon":
    """Loaded fit file if present, else the seed lexicon (fail-open)."""
    loaded = FunctionalLexicon.load()
    return loaded if loaded is not None else FunctionalLexicon()
