"""Compositional realizer lexicon (Stage 6, de-hardcoding plan).

Retires the canned assertion-mirror templates in response_gen.py
(``f"yeah, {topic}."`` + the leads/follows/backchannel pools that were
selected by ``random.choice`` over inline typed lists). The templates become
*exemplars* in a single externalized pool (data/realizer_lexicon.json); a
realization is drawn via a pluggable scorer rather than blind random.choice.

Brain basis (Menon 2023; DMN "internal narrative"): production is compositional
from a message + situation model, not slot-filling a fixed frame. The scorer
slot lets a learned fluency/coherence ranker (e.g. reusing salad_classifier)
pick/rank candidates later; the seed scorer is uniform (== random.choice), so
day-one behavior is identical and the module fails open to the inline defaults
if the fit file is absent.

Mirrors SaladClassifier / SnippetPEConfig: load()/save() to data/*.json.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

# This file lives at <repo>/ravana/src/ravana/chat/realizer_lexicon.py.
# To reach <repo>/data go up 5 levels: chat -> ravana -> src -> ravana -> <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_FIT_PATH = os.path.join(_DATA_DIR, "realizer_lexicon.json")

# Seed exemplar pools — the former inline typed lists, demoted to data.
_SEED_POOLS: Dict[str, List[str]] = {
    "user_leads": [
        "got it — so you're {topic}.", "ah, i see — you're {topic}.",
        "nice, so you're {topic}.", "makes sense. you're {topic}.",
    ],
    "other_leads": [
        "right, {topic}.", "got it — {topic}.", "ok, noted: {topic}.",
        "yeah, {topic}.",
    ],
    "user_leads_notopic": ["got it.", "ah, i see.", "nice, noted.", "makes sense."],
    "other_leads_notopic": ["right.", "got it.", "ok, noted.", "yeah."],
    "follows": [
        "what made you think of that?",
        "tell me more about it?",
        "anything else on your mind?",
        "what do you make of it?",
    ],
    "backchannels": [
        "haha fair enough.", "nice.", "gotcha.", "makes sense.", "alright.",
    ],
}


class RealizerLexicon:
    """Externalized exemplar pool for assertion acknowledgments.

    ``realize(pool, topic, ctx, rng)`` returns a formatted candidate drawn from
    the named pool, ranked by ``scorer`` (seed: uniform => random.choice
    equivalent). The ``{topic}`` placeholder is filled if a topic is present.
    """

    def __init__(self, pools: Optional[Dict[str, List[str]]] = None,
                 weights: Optional[Dict[str, List[float]]] = None) -> None:
        self._pools = {k: list(v) for k, v in _SEED_POOLS.items()}
        if pools:
            for k, v in pools.items():
                if k in _SEED_POOLS:
                    self._pools[k] = list(v)
        self._weights = weights or {}

    @classmethod
    def load(cls) -> Optional["RealizerLexicon"]:
        if not os.path.exists(_FIT_PATH):
            return None
        try:
            with open(_FIT_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return cls(pools=d.get("pools", {}), weights=d.get("weights", {}))
        except Exception:
            return None

    def save(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_FIT_PATH, "w", encoding="utf-8") as f:
            json.dump({"pools": self._pools, "weights": self._weights}, f, indent=2)

    def realize(self, pool: str, topic: str = "",
                scorer=None, rng: Optional[random.Random] = None) -> str:
        """Draw a formatted candidate from ``pool``.

        ``scorer(candidate, topic, ctx) -> float`` ranks candidates; if None or
        all-equal, selection degrades to uniform random (== legacy random.choice
        distribution). Falls back to the seed pool if the named pool is empty.
        """
        cands = self._pools.get(pool, _SEED_POOLS.get(pool, []))
        if not cands:
            return ""
        rng = rng or random
        scored = []
        for c in cands:
            s = scorer(c, topic) if callable(scorer) else 0.0
            scored.append((s, c))
        # Uniform when scores tie (seed scorer returns 0 -> all tie -> random).
        best = max(s for s, _ in scored)
        top = [c for s, c in scored if s == best] or cands
        pick = rng.choice(top)
        return pick.format(topic=topic) if "{topic}" in pick else pick


def default_lexicon() -> "RealizerLexicon":
    loaded = RealizerLexicon.load()
    return loaded if loaded is not None else RealizerLexicon()
