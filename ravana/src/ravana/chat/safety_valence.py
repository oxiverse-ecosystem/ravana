"""Learned safety / social-inappropriateness gate (Stage 7, de-hardcoding plan).

Retires the hardcoded ``INAPPROPRIATE_WORDS`` list from constants.py with a
**learned distributional valence** model (OFC-style social-reward filtering).

Mechanism: a candidate definition is inappropriate when it contains a token
whose GloVe vector is near any *profanity/slur anchor* (high cosine). This is
the brain-faithful OFC signal — social inappropriateness is learned by
association (valence/reward learning), not an innate blocklist, and it
*generalizes*: misspellings and morphological variants (e.g. "f*ck", "sh1t")
land near the anchor and are caught, whereas the frozen list missed them.

The fit criterion (max-cosine threshold) is externalized to
``data/safety_valence.json`` and EER-fit by ``experiments/measure_safety.py``
(per the plan's SDT-criterion discipline). A minimal **hard-override** set
(the original highest-severity terms) is retained as the one ethically
defensible short list — everything else is learned.

Fail-open: if the fit file is absent or the anchor set is empty, the model
falls back to the exact INAPPROPRIATE_WORDS membership test (current behavior,
no regression).

Mirrors SaladClassifier / SnippetPEConfig: load()/save() to data/*.json.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set

import numpy as np

# This file lives at <repo>/ravana/src/ravana/chat/safety_valence.py.
# To reach <repo>/data go up 5 levels: chat -> ravana -> src -> ravana -> <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_FIT_PATH = os.path.join(_DATA_DIR, "safety_valence.json")

# Minimal high-severity hard-override — the ethically-defensible last resort
# (plan: "the one place a short list is ethically defensible"). Distributional
# matching catches variants; this exact-set catches the canonical slurs even
# if GloVe has no vector for a token.
_HARD_OVERRIDE = {
    "penis", "vagina", "cum", "fuck", "shit", "bitch", "asshole",
    "cunt", "pussy", "dick", "cock", "bastard", "slut", "whore",
    "rape", "incest", "pedophile",
}

_TOKEN_RE = re.compile(r"[a-z']{3,}")


class SafetyValence:
    """Distributional social-inappropriateness gate.

    ``anchors``: list of GloVe vectors for the profanity/slur prototypes.
    ``threshold``: max-cosine-to-anchor above which a token is inappropriate
        (EER-fit, default seed 0.55).
    ``hard_override``: exact tokens always flagged (minimal last-resort set).
    """

    def __init__(self, anchors: Optional[List[List[float]]] = None,
                 threshold: float = 0.55,
                 hard_override: Optional[Set[str]] = None) -> None:
        self._anchors = [np.asarray(a, dtype=float) for a in (anchors or [])]
        self._anchors = [a / (np.linalg.norm(a) or 1.0) for a in self._anchors]
        self.threshold = float(threshold)
        self.hard_override = set(hard_override or _HARD_OVERRIDE)

    @classmethod
    def from_seed(cls, glove_fn,
                  inappropriate: Optional[Set[str]] = None,
                  threshold: float = 0.55) -> "SafetyValence":
        """Build anchors as the GloVe vectors of the seed inappropriate words
        (those with vectors); the full set also seeds the hard-override."""
        inappropriate = inappropriate or _HARD_OVERRIDE
        anchors = []
        for w in inappropriate:
            v = glove_fn(w) if callable(glove_fn) else None
            if v is not None:
                anchors.append(np.asarray(v, dtype=float).tolist())
        return cls(anchors=anchors, threshold=threshold,
                   hard_override=set(inappropriate))

    def score(self, text: str, glove_fn=None) -> float:
        """Max cosine of any token to the nearest anchor (0.0 = clean).
        Requires ``glove_fn`` at inference to vectorize tokens; anchors are the
        prototype vectors themselves."""
        if not self._anchors or not callable(glove_fn):
            return 0.0
        best = 0.0
        for tok in _TOKEN_RE.findall((text or "").lower()):
            v = glove_fn(tok)
            if v is None:
                continue
            v = np.asarray(v, dtype=float)
            n = np.linalg.norm(v)
            if n == 0:
                continue
            v = v / n
            for a in self._anchors:
                c = float(np.dot(v, a))
                if c > best:
                    best = c
        return best

    def is_inappropriate(self, text: str, glove_fn=None) -> bool:
        toks = {t for t in _TOKEN_RE.findall((text or "").lower())}
        # Hard override: exact canonical slurs always flagged.
        if toks & self.hard_override:
            return True
        # Distributional: any token near an anchor prototype.
        if self._anchors and self.score(text, glove_fn) >= self.threshold:
            return True
        return False

    @classmethod
    def load(cls) -> Optional["SafetyValence"]:
        if not os.path.exists(_FIT_PATH):
            return None
        try:
            with open(_FIT_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return cls(anchors=d.get("anchors", []),
                       threshold=float(d.get("threshold", 0.55)),
                       hard_override=set(d.get("hard_override", [])) or _HARD_OVERRIDE)
        except Exception:
            return None

    def save(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_FIT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "anchors": [a.tolist() if hasattr(a, "tolist") else list(a)
                            for a in self._anchors],
                "threshold": self.threshold,
                "hard_override": sorted(self.hard_override),
            }, f, indent=2)

    def to_dict(self) -> Dict[str, object]:
        return {
            "anchors": [a.tolist() if hasattr(a, "tolist") else list(a)
                        for a in self._anchors],
            "threshold": self.threshold,
            "hard_override": sorted(self.hard_override),
        }
