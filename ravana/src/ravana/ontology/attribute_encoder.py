"""
AttributeEncoder — a brain-aligned (Binder 2016) componential feature space.

The frontopolar gate previously failed to derive affordances from GloVe cosine
because distributional vectors do NOT separate modality/experiential attributes
(see GATE_DERIVATION_FINDINGS.md). The brain instead represents each concept as
a vector over ~65 *experiential attribute dimensions* (visual-color, weight,
taste, sound, temporal, social, emotional, ...). Possession of a property is a
direct read-off of the relevant dimension.

This module learns a linear probe (ridge regression) that maps a concept's
64-D distributional vector onto Binder's 65-D attribute space, trained on the
published human norms (535 words). The probe then generalizes to the full
open vocabulary, producing the per-concept `sensorimotor_vector` the graph DB
already provisions but never fills.

Dependency-free (numpy only). The learned weights persist to
data/attribute_encoder.npz so inference needs no retraining.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Binder 65 experiential attribute dimensions (cols 5:70 of word_ratings) ──
BINDER_DIMS: List[str] = [
    "Vision", "Bright", "Dark", "Color", "Pattern", "Large", "Small", "Motion",
    "Biomotion", "Fast", "Slow", "Shape", "Complexity", "Face", "Body", "Touch",
    "Temperature", "Texture", "Weight", "Pain", "Audition", "Loud", "Low", "High",
    "Sound", "Music", "Speech", "Taste", "Smell", "Head", "UpperLimb", "LowerLimb",
    "Practice", "Landmark", "Path", "Scene", "Near", "Toward", "Away", "Number",
    "Time", "Duration", "Long", "Short", "Caused", "Consequential", "Social",
    "Human", "Communication", "Self", "Cognition", "Benefit", "Harm", "Pleasant",
    "Unpleasant", "Happy", "Sad", "Angry", "Disgusted", "Fearful", "Surprised",
    "Drive", "Needs", "Attention", "Arousal",
]

# Gate properties (from engine.py _PROPERTY_CATEGORIES / _CATEGORY_AFFORDANCES)
# mapped onto Binder dimensions. A property is "possessed" when ANY of its
# mapped dims exceeds theta (OR-dim); the gate reads the attribute vector.
PROPERTY_TO_DIMS: Dict[str, List[str]] = {
    "color": ["Color", "Vision"],
    "colour": ["Color", "Vision"],
    "weight": ["Weight"],
    "weigh": ["Weight"],
    "weighs": ["Weight"],
    "mass": ["Weight"],
    "taste": ["Taste"],
    "smell": ["Smell"],
    "sound": ["Sound", "Audition"],
    "loudness": ["Loud", "Sound"],
    "size": ["Large", "Small"],
    "shape": ["Shape"],
    "texture": ["Texture", "Touch"],
    "temperature": ["Temperature"],
    "brightness": ["Bright", "Dark"],
    "duration": ["Duration", "Time"],
    "order": ["Number", "Time"],
    "cycle": ["Time", "Duration"],
}

# ── Lancaster Sensorimotor Norms (Lynott et al. 2019), 39,707 words × 11 dims ──
# The published human norms aggregate 6 perceptual modalities + 5 action
# effectors on a 0-5 strength scale. This is ~75× the coverage of the Binder
# 535-word probe, so a GloVe→Lancaster ridge generalizes the cross-modal probe
# to the open vocabulary instead of only the 535 training lemmas.
LANCASTER_DIMS: List[str] = [
    "Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual",
    "Foot_leg", "Hand_arm", "Head", "Mouth", "Torso",
]

# Each Lancaster dim subsumes a set of Binder 65-D dims. The wider-coverage
# Lancaster probe drives the EXISTING Binder-based gate (_PROP_TO_DIMS) and
# metaphor (_SENSORY_DIM_PHRASE) through this expansion, so no engine code
# changes: a Lancaster 11-D vector is projected into the 65-D Binder space
# (taking the Lancaster value whenever it carries more signal than Binder).
LANCASTER_TO_BINDER: Dict[str, List[str]] = {
    "Auditory": ["Audition", "Sound", "Loud", "Low", "High", "Music", "Speech"],
    "Gustatory": ["Taste"],
    "Haptic": ["Touch", "Texture", "Temperature", "Weight", "Pain"],
    "Interoceptive": ["Pleasant", "Unpleasant", "Happy", "Sad",
                      "Angry", "Disgusted", "Fearful", "Surprised", "Arousal"],
    "Olfactory": ["Smell"],
    "Visual": ["Vision", "Bright", "Dark", "Color", "Pattern", "Shape",
               "Large", "Small", "Complexity", "Face"],
    "Foot_leg": ["LowerLimb"],
    "Hand_arm": ["UpperLimb"],
    "Head": ["Head"],
    "Mouth": ["Mouth"],
    "Torso": ["Torso"],
}


class AttributeEncoder:
    """Linear probe: glove64 (n) -> binder attributes (65)."""

    def __init__(self, dims: List[str] = BINDER_DIMS, lam: float = 1.0):
        self.dims = list(dims)
        self.dim_index = {d: i for i, d in enumerate(self.dims)}
        self.lam = float(lam)
        self.W: Optional[np.ndarray] = None   # (65, 64)
        self.b: Optional[np.ndarray] = None   # (65,)
        self._mean_in: Optional[np.ndarray] = None
        self._std_in: Optional[np.ndarray] = None
        self._mean_out: Optional[np.ndarray] = None

    # ── training ──────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "AttributeEncoder":
        """X: (m, 64) glove vectors; Y: (m, 65) binder attributes (0-6)."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self._mean_in = X.mean(0)
        self._std_in = X.std(0) + 1e-8
        self._mean_out = Y.mean(0)
        Xz = (X - self._mean_in) / self._std_in
        Yc = Y - self._mean_out
        # ridge closed form: W = (X'X + lam*I)^-1 X'Y
        XtX = Xz.T @ Xz + self.lam * np.eye(Xz.shape[1])
        XtY = Xz.T @ Yc
        Wz = np.linalg.solve(XtX, XtY)        # (64, 65)
        self.W = Wz.T                          # (65, 64)
        self.b = self._mean_out.copy()         # predict = xz @ W.T + mean_out
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.W is None:
            raise RuntimeError("encoder not fitted")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        xz = (x - self._mean_in) / self._std_in
        return xz @ self.W.T + self._mean_out   # (k, 65)

    # ── gate helpers ──────────────────────────────────────────────────────
    def attribute_vector(self, vec64: np.ndarray) -> np.ndarray:
        """Predicted 65-D attribute vector for a concept's 64-D glove vec."""
        return self.predict(vec64).reshape(-1)

    def property_score(self, vec64: np.ndarray, prop: str) -> Optional[float]:
        """Max Binder-dimension activation for gate property `prop`."""
        dims = PROPERTY_TO_DIMS.get(prop.lower())
        if dims is None or self.W is None:
            return None
        av = self.attribute_vector(vec64)
        idxs = [self.dim_index[d] for d in dims if d in self.dim_index]
        if not idxs:
            return None
        return float(np.max(av[idxs]))

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        np.savez(path, W=self.W, b=self.b, mean_in=self._mean_in,
                 std_in=self._std_in, mean_out=self._mean_out,
                 dims=np.array(self.dims, dtype=object))

    @classmethod
    def load(cls, path: str) -> "AttributeEncoder":
        z = np.load(path, allow_pickle=True)
        enc = cls(dims=list(z["dims"]))
        enc.W = z["W"]
        enc.b = z["b"]
        enc._mean_in = z["mean_in"]
        enc._std_in = z["std_in"]
        enc._mean_out = z["mean_out"]
        return enc


# ── glove-64 loader (mirrors node storage: proj @ glove100) ─────────────────
def build_glove64_lookup(cache_npz: str) -> Tuple[Dict[str, np.ndarray], int]:
    d = np.load(cache_npz, allow_pickle=True)
    words = d["words"].tolist()
    vecs = d["vecs"].astype(np.float32)
    proj = d["proj"].astype(np.float32)        # (64, 100)
    lut: Dict[str, np.ndarray] = {}
    for w, v in zip(words, vecs):
        lut[str(w).lower()] = (proj @ v).astype(np.float32)
    return lut, proj.shape[0]


def train_from_binder(cache_npz: str, binder_xlsx: str,
                      out_path: str, lam: float = 1.0) -> "AttributeEncoder":
    """End-to-end: build glove64 lookup, load Binder norms, fit + save."""
    import pandas as pd
    lut, dim = build_glove64_lookup(cache_npz)
    xl = pd.ExcelFile(binder_xlsx)
    df = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
    attr_cols = BINDER_DIMS
    Xs, Ys = [], []
    for _, row in df.iterrows():
        w = str(row["Word"]).lower()
        if w not in lut:
            continue
        y = []
        for c in attr_cols:
            val = row[c]
            try:
                y.append(float(val))
            except (TypeError, ValueError):
                y.append(0.0)
        if any(np.isnan(v) for v in y):
            continue
        Xs.append(lut[w])
        Ys.append(y)
    X = np.stack(Xs, 0)
    Y = np.stack(Ys, 0)
    enc = AttributeEncoder(lam=lam).fit(X, Y)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    enc.save(out_path)
    return enc


class LancasterEncoder(AttributeEncoder):
    """Linear probe: glove64 (n) -> Lancaster 11-D sensorimotor norms.

    Same ridge formulation as the Binder probe, but trained on 39,707 words
    (Lynott et al. 2019) instead of 535, so it generalizes the cross-modal
    attribute read-out to the open vocabulary.
    """

    def __init__(self, dims: List[str] = None, lam: float = 1.0):
        super().__init__(dims=dims if dims is not None else LANCASTER_DIMS, lam=lam)


def train_from_lancaster(cache_npz: str, csv_path: str,
                         out_path: str, lam: float = 1.0) -> "LancasterEncoder":
    """End-to-end: build glove64 lookup, load Lancaster norms CSV, fit + save.

    The CSV carries 11 '<Dim>.mean' columns (Auditory..Torso); we regress the
    glove-64 vector onto those, mirroring train_from_binder.
    """
    import csv
    lut, dim = build_glove64_lookup(cache_npz)
    mean_cols = [d + ".mean" for d in LANCASTER_DIMS]
    Xs, Ys = [], []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            w = str(row.get("Word", "")).strip().lower()
            if not w or w not in lut:
                continue
            try:
                y = [float(row[c]) for c in mean_cols]
            except (TypeError, ValueError):
                continue
            if any(np.isnan(v) for v in y):
                continue
            Xs.append(lut[w])
            Ys.append(y)
    if not Xs:
        raise RuntimeError("no overlapping words between glove cache and Lancaster norms")
    X = np.stack(Xs, 0)
    Y = np.stack(Ys, 0)
    enc = LancasterEncoder(lam=lam).fit(X, Y)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    enc.save(out_path)
    return enc


class CombinedAttributeEncoder:
    """Primary = Lancaster (wide coverage, 39k words), fallback = Binder
    (fine 65-D, 535 words). Exposes the SAME 65-D Binder interface
    (``dims``, ``dim_index``, ``attribute_vector``, ``property_score``) so
    every engine/ontology call site is unchanged.

    Construction: prefer the Lancaster value on each expanded Binder dim; keep
    the Binder value otherwise. Binder's non-sensory dims (Social, Cognition,
    Time, Number, ...) are retained untouched because Lancaster cannot express
    them, and the gate/metaphor only ever read the sensory subset.
    """

    def __init__(self, binder: Optional["AttributeEncoder"] = None,
                 lancaster: Optional["LancasterEncoder"] = None):
        self.binder = binder
        self.lancaster = lancaster
        self.dims = list(BINDER_DIMS)
        self.dim_index = {d: i for i, d in enumerate(self.dims)}

    def attribute_vector(self, vec64: np.ndarray) -> np.ndarray:
        if self.binder is not None:
            av = self.binder.attribute_vector(vec64).astype(np.float64).copy()
        else:
            av = np.zeros(len(self.dims), dtype=np.float64)
        if self.lancaster is not None:
            lv = self.lancaster.attribute_vector(vec64)
            for i, ldim in enumerate(self.lancaster.dims):
                val = float(lv[i])
                for bdim in LANCASTER_TO_BINDER.get(ldim, []):
                    j = self.dim_index.get(bdim)
                    if j is not None and val > av[j]:
                        av[j] = val
        return av

    def property_score(self, vec64: np.ndarray, prop: str) -> Optional[float]:
        dims = PROPERTY_TO_DIMS.get(prop.lower())
        if dims is None:
            return None
        av = self.attribute_vector(vec64)
        idxs = [self.dim_index[d] for d in dims if d in self.dim_index]
        if not idxs:
            return None
        return float(np.max(av[idxs]))


def load_combined_encoder(binder_path: str,
                          lancaster_path: Optional[str] = None) -> Optional[CombinedAttributeEncoder]:
    """Load a CombinedAttributeEncoder: Lancaster primary (if present) with
    Binder as fallback. Returns None if neither probe artifact exists."""
    binder = None
    if binder_path and os.path.exists(binder_path):
        try:
            binder = AttributeEncoder.load(binder_path)
        except Exception:
            binder = None
    lancaster = None
    if lancaster_path and os.path.exists(lancaster_path):
        try:
            lancaster = LancasterEncoder.load(lancaster_path)
        except Exception:
            lancaster = None
    if binder is None and lancaster is None:
        return None
    return CombinedAttributeEncoder(binder=binder, lancaster=lancaster)
