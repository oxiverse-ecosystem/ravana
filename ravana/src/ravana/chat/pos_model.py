"""Learned distributional POS classifier (Stage 5b-i, de-hardcoding plan).

Replaces the rule-based ``classify_word_pos`` + the hardcoded
``KNOWN_VERBS`` / ``KNOWN_ADJS`` / ``FUNCTION_WORDS`` lists (constants.json)
with a *distributional* POS model: a word's part of speech is the nearest
centroid in GloVe space among the verb / adjective / function / noun
prototypes. This is the brain-faithful mechanism — grammatical class is
distributed (words in similar syntactic slots share embedding neighborhoods);
the brain does not carry a function-word list (Zhang 2020, Nat. Commun.;
Rosch prototype categorization).

The centroids are SEEDed from the existing curated lists (so day-one behavior
matches the rule-based tagger on the seed vocabulary), then the ambiguity
margin is pinned at EER by ``experiments/measure_pos_model.py`` on a
POS-tagged corpus. The model persists only the centroids (no GloVe needed at
inference), mirroring ``SaladClassifier`` / ``SnippetPEConfig`` ``load()/save()``.

Flag-gated: the engine uses this only when ``use_learned_pos`` is ON; the
default path stays rule-based (``classify_word_pos``) so nothing regresses.

Brain basis: POS/grammatical class is a distributed code, not a membership
test in a frozen list.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# This file lives at <repo>/ravana/src/ravana/chat/pos_model.py.
# To reach <repo>/data go up 5 levels: chat -> ravana -> src -> ravana -> <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_FIT_PATH = os.path.join(_DATA_DIR, "pos_model.json")

_CLASSES = ("verb", "adj", "func", "noun")


class PosModel:
    """Nearest-centroid POS classifier over GloVe embeddings.

    Centroids are computed from seed word lists (or loaded from the fit file).
    ``classify`` returns the nearest class when the best-vs-second gap exceeds
    the EER-fit ambiguity margin; below the margin it falls back to 'noun'
    (the unmarked/default class) to avoid confident mis-tags.
    """

    def __init__(self, centroids: Optional[Dict[str, List[float]]] = None,
                 ambiguity_margin: float = 0.04) -> None:
        # centroids: {class: [dim]} vectors (unit-normalized at fit time)
        self._centroids = {c: np.asarray(v, dtype=float)
                           if v is not None else None
                           for c, v in (centroids or {}).items()}
        self.ambiguity_margin = float(ambiguity_margin)

    @classmethod
    def from_seed(cls, glove_fn, seed_words: Dict[str, List[str]],
                  ambiguity_margin: float = 0.04) -> "PosModel":
        """Build centroids as the mean (unit-normalized) GloVe vector of each
        seed list. Requires GloVe at build time; the result is persisted so
        inference needs no GloVe."""
        centroids: Dict[str, List[float]] = {}
        for clz in _CLASSES:
            vecs = []
            for w in seed_words.get(clz, []):
                v = glove_fn(w) if callable(glove_fn) else None
                if v is not None:
                    vecs.append(np.asarray(v, dtype=float))
            if vecs:
                m = np.mean(vecs, axis=0)
                n = np.linalg.norm(m)
                centroids[clz] = (m / n).tolist() if n > 0 else m.tolist()
        return cls(centroids=centroids, ambiguity_margin=ambiguity_margin)

    def classify(self, word: str, glove_fn=None) -> str:
        """Return the POS of ``word`` (verb/adj/func/noun).

        Falls back to 'noun' when GloVe is unavailable or the word is unknown,
        or when the nearest-centroid gap is below the ambiguity margin (uncertain).
        """
        if not self._centroids:
            return "noun"
        v = glove_fn(word) if callable(glove_fn) else None
        if v is None:
            return "noun"
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            return "noun"
        v = v / n
        # Nearest two classes by cosine similarity.
        sims = {}
        for clz, c in self._centroids.items():
            if c is None:
                continue
            cn = np.linalg.norm(c)
            sims[clz] = float(np.dot(v, c)) / cn if cn > 0 else 0.0
        if not sims:
            return "noun"
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        best_cls, best_sim = ranked[0]
        if len(ranked) > 1:
            second_sim = ranked[1][1]
            # Ambiguity: if the best and second are too close, the tag is
            # unreliable -> default to the unmarked 'noun' class (fail-soft,
            # never a confident mis-tag that could corrupt grounding/syntax).
            if (best_sim - second_sim) < self.ambiguity_margin:
                return "noun"
        return best_cls

    @classmethod
    def load(cls) -> Optional["PosModel"]:
        if not os.path.exists(_FIT_PATH):
            return None
        try:
            with open(_FIT_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return cls(centroids=d.get("centroids", {}),
                       ambiguity_margin=float(d.get("ambiguity_margin", 0.04)))
        except Exception:
            return None

    def save(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_FIT_PATH, "w", encoding="utf-8") as f:
            json.dump({"centroids": {k: (v.tolist() if hasattr(v, "tolist")
                                        else v) for k, v in self._centroids.items()},
                       "ambiguity_margin": self.ambiguity_margin}, f, indent=2)

    def to_dict(self) -> Dict[str, object]:
        return {"centroids": {k: (v.tolist() if hasattr(v, "tolist") else v)
                              for k, v in self._centroids.items()},
                "ambiguity_margin": self.ambiguity_margin}


def _seed_from_constants():
    """Pull the seed POS word lists from constants.json (the legacy source)."""
    import json as _json
    p = os.path.join(_DATA_DIR, "constants.json")
    with open(p, encoding="utf-8") as f:
        c = _json.load(f)
    return {
        "verb": c.get("known_verbs", []),
        "adj": c.get("known_adjs", []),
        # function_words carries mostly preps/determiners/conjunctions -> 'func'
        "func": c.get("function_words", []),
        # 'noun' has no explicit seed list; the model defaults to 'noun' when
        # uncertain, and content nouns naturally sit far from verb/adj/func
        # centroids, so an empty noun seed is fine.
        "noun": [],
    }
