"""Learned configuration for the snippet answer-plausibility forward model (M-C).

Externalizes the snippet-PE gate's typed numbers (Stage 5a of the de-hardcoding
plan) into a single fit file ``data/snippet_pe.json`` so they are refittable
(EER-pinned criterion, SDT ``c``) rather than hand-typed guesses. This mirrors
the established ``SaladClassifier`` pattern: a ``load()`` that fails open (returns
``None`` when the fit file is absent, so the engine keeps seed constants and is
never a regression source) and a ``save()``.

Seed values are the CURRENT engine constants (verified to beat the backstop on
the golden regression set) — externalizing them changes NO behavior on day one;
a fit harness (``experiments/measure_snippet_pe.py``) can later move them.

Fields
------
coverage_threshold   : max-token cosine to the subject head below which a snippet
                       fails topic coverage (was ``_cov_thr = 0.6``).
coverage_surprise    : PE returned when coverage fails (was hardcoded ``0.7``).
answer_type_surprise : PE returned on answer-type / speech-act mismatch
                       (was ``_answer_type_mismatch`` returns ``0.6``).
polarity_surprise    : PE on premise-polarity contradiction (was ``1.0``).
veto_midpoint        : combined-PE midpoint above which a candidate is withheld
                       (was ``_ANSWER_PE_VETO = 0.6``).
veto_slope           : reserved for a future continuous (synaptic_dynamics-style)
                       sigmoid veto; the current code uses the hard midpoint.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

# This file lives at <repo>/ravana/src/ravana/chat/snippet_pe_config.py.
# To reach <repo>/data go up 5 levels: chat -> ravana -> src -> ravana -> <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_FIT_PATH = os.path.join(_DATA_DIR, "snippet_pe.json")

# Seed = current engine constants (day-one behavior identical). These are the
# cold-start values; the fit harness moves them once enough labeled data exists.
_SEED: Dict[str, float] = {
    "coverage_threshold": 0.6,
    "coverage_surprise": 0.7,
    "answer_type_surprise": 0.6,
    "polarity_surprise": 1.0,
    "veto_midpoint": 0.6,
    "veto_slope": 8.0,
}


class SnippetPEConfig:
    """Fit parameters for the snippet answer-plausibility forward model.

    Constructed from a dict (loaded from ``data/snippet_pe.json`` or the seed).
    Every accessor falls back to the seed value when a key is missing, so a
    partial fit file never breaks the gate.
    """

    def __init__(self, values: Optional[Dict[str, float]] = None) -> None:
        self._v = dict(_SEED)
        if values:
            self._v.update({k: float(v) for k, v in values.items()
                            if k in _SEED})

    # ── accessors (seed-fallback) ──────────────────────────────────────────
    @property
    def coverage_threshold(self) -> float:
        return self._v.get("coverage_threshold", _SEED["coverage_threshold"])

    @property
    def coverage_surprise(self) -> float:
        return self._v.get("coverage_surprise", _SEED["coverage_surprise"])

    @property
    def answer_type_surprise(self) -> float:
        return self._v.get("answer_type_surprise", _SEED["answer_type_surprise"])

    @property
    def polarity_surprise(self) -> float:
        return self._v.get("polarity_surprise", _SEED["polarity_surprise"])

    @property
    def veto_midpoint(self) -> float:
        return self._v.get("veto_midpoint", _SEED["veto_midpoint"])

    @property
    def veto_slope(self) -> float:
        return self._v.get("veto_slope", _SEED["veto_slope"])

    # ── persistence (mirrors SaladClassifier.load/save) ─────────────────────
    @classmethod
    def load(cls) -> Optional["SnippetPEConfig"]:
        """Load from data/snippet_pe.json; return None (fail open) if absent."""
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
        with open(_FIT_PATH, "w", encoding="utf-8") as f:
            json.dump(self._v, f, indent=2)

    def to_dict(self) -> Dict[str, float]:
        return dict(self._v)


def default_config() -> "SnippetPEConfig":
    """Return the loaded fit config if present, else the seed config.

    Fails open: callers always get a usable config, either the fitted one or
    the seed constants (identical day-one behavior to the old inline numbers).
    """
    loaded = SnippetPEConfig.load()
    return loaded if loaded is not None else SnippetPEConfig()
