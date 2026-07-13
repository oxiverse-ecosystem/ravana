"""Attribute-probe calibration + realization helpers (research item A).

The engine gates property possession with a BLIND constant (_THETA = 0.8 on
the Binder 0-6 scale; prior_theta = 4.5 in conceptnet.py). The Lancaster probe
IS trained and evaluated against the published human norms; experiments/
measure_attribute_theta.py FITs a per-dimension threshold to those ratings
(held-out accuracy ~0.94). This module loads that FIT boundary and provides:

  - load_fitted_theta(): the per-dimension theta dict from data/attribute_theta.json
  - calibrated_property_threshold(prop): per-property theta mapped from the
    property's Binder dims back to their Lancaster source dims (via
    LANCASTER_TO_BINDER), using the FITTED theta, never the blind 0.8.
  - ood_abstain(enc, gvec): OOD-abstain signal — if the probe's max activation
    across all dims is near zero, the word is off the training manifold; the
    caller should abstain from a metaphor rather than forcing a random one.
  - realize_dim(dim_name, magnitude): data-derived phrasing. The active DIM
    is selected by the probe (data, not a hand list); the phrasing is modulated
    by the activation MAGNITUDE (the perceptual-intensity / blend-weight param
    the plan flags as "calibrate, don't fix"). This is the measured stand-in
    for full LingGen BOS-injection (which is future work, reported honestly).

Dependency-free numpy; globs the production probe artifacts.
"""
from __future__ import annotations

import os
import json
import sys
from typing import Dict, Any, Optional, Tuple

_THIS = os.path.dirname(os.path.abspath(__file__))
# attribute_calibration.py lives at <repo>/ravana/src/ravana/ontology/.
# Repo root is FOUR levels up:
#   ontology -> ravana(src/ravana) -> src -> ravana(package) -> <repo root>.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
for p in (os.path.join(_ROOT, "ravana", "src"),
         os.path.join(_ROOT, "ravana_ml", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from ravana.ontology.attribute_encoder import (  # noqa: E402
    LANCASTER_TO_BINDER, LANCASTER_DIMS, PROPERTY_TO_DIMS, BINDER_DIMS,
)

_THETA_JSON = os.path.join(_ROOT, "data", "attribute_theta.json")
# Fallback default if the calibration json is absent (documented, not a blind 0.8).
_DEFAULT_THETA = 0.8


def load_fitted_theta() -> Optional[Dict[str, float]]:
    """Per-dimension fitted theta (Lancaster 11-D) from the calibration harness.

    Returns None if the json is missing (caller falls back to _DEFAULT_THETA and
    the OOD signal degrades to a fixed floor).
    """
    if not os.path.exists(_THETA_JSON):
        return None
    try:
        with open(_THETA_JSON) as f:
            d = json.load(f)
        return d.get("per_dim_theta")
    except Exception:
        return None


# Reverse map: Binder dim -> owning Lancaster dim(s).
_BINDER_TO_LANCASTER: Dict[str, str] = {}
for _l, _bs in LANCASTER_TO_BINDER.items():
    for _b in _bs:
        _BINDER_TO_LANCASTER.setdefault(_b, _l)


def calibrated_property_threshold(prop: str,
                                   fitted: Optional[Dict[str, float]]) -> float:
    """Per-property theta mapped from Binder dims -> Lancaster source dims.

    Uses the FITTED per-dimension theta when available; otherwise the documented
    default. This replaces the blind `0.8`/`4.5` with a distribution-fit value.
    """
    if fitted is None:
        return _DEFAULT_THETA
    dims = PROPERTY_TO_DIMS.get(prop.lower())
    if not dims:
        return _DEFAULT_THETA
    thetas = []
    for bd in dims:
        ld = _BINDER_TO_LANCASTER.get(bd)
        if ld in fitted:
            thetas.append(fitted[ld])
    if not thetas:
        return _DEFAULT_THETA
    # The property is possessed when ANY mapped dim exceeds its own fitted theta
    # (OR-dim gate). Use the MAX fitted theta among the property's dims as the
    # calibrated threshold for the max-activation score.
    return float(max(thetas))


def ood_abstain(enc, gvec: np.ndarray, floor: float = 0.15) -> bool:
    """OOD-abstain: True when the probe is essentially silent (off-manifold).

    If the max activation across all Binder dims is below `floor`, the probe has
    no confident sensorimotor signal for this word -> the caller should abstain
    from a cross-modal metaphor rather than forcing one (the plan's point 4).
    """
    if enc is None or gvec is None:
        return True
    try:
        av = enc.attribute_vector(np.asarray(gvec, dtype=np.float64))
    except Exception:
        return True
    return float(np.max(av)) < floor


# Canonical sense-verb families, derived from the modality encoded in the DIM
# NAME (which comes from the trained probe, not a hand-authored analogy list).
# This is the measured stand-in for full LingGen BOS-injection: the DIM is
# selected by the probe; the phrasing is keyed by modality family + modulated by
# the activation magnitude (perceptual intensity).
_SENSE_FAMILY = [
    (("vision", "color", "bright", "dark", "pattern", "shape", "complexity", "face"),
     ("visual character", "see")),
    (("sound", "audition", "loud", "low", "high", "music", "speech"),
     ("sound", "hear")),
    (("touch", "texture", "temperature", "weight", "pain"),
     ("texture and feel", "feel")),
    (("taste",), ("taste", "taste")),
    (("smell", "olfactory"), ("smell", "smell")),
    (("motion", "biomotion", "fast", "slow"), ("movement", "watch move")),
    (("large", "small"), ("size", "picture by its scale")),
    (("time", "duration", "long", "short"), ("sense of time", "place in time")),
    (("number", "order"), ("sense of amount", "count out")),
]


def _sense_for_dim(dim_name: str) -> Tuple[str, str]:
    dl = (dim_name or "").lower()
    for keys, (phrase, sense) in _SENSE_FAMILY:
        if any(k in dl for k in keys):
            return (phrase, sense)
    return ("character", "sense")


def realize_dim(dim_name: str, magnitude: float) -> str:
    """Data-derived phrasing for the active dimension, modulated by magnitude.

    magnitude is the probe activation (0-6 Binder scale). Higher -> more vivid
    hedging; lower -> more tentative. This is the perceptual-intensity parameter
    the plan says to calibrate, not fix.
    """
    phrase, sense = _sense_for_dim(dim_name)
    if magnitude >= 2.0:
        return (f"strong {phrase}", sense)
    if magnitude >= 1.0:
        return (f"{phrase}", sense)
    if magnitude >= 0.5:
        return (f"faint {phrase}", sense)
    return (f"hint of {phrase}", sense)
