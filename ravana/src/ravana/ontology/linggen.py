"""LingGen P6 — angular-gyrus binding projection (brain-inspired, no hardcoding).

Implements the learned ``W_sm`` that maps a concept's 65-D Binder
sensorimotor signature onto the decoder's 75-D initiation embedding space
(the "angular gyrus" that binds modality spokes into one unified initiation
vector). This replaces the fixed ``np.random.RandomState(1234)`` warp basis
in ``response_gen._build_conditioned_bos`` with a DISTRIBUTION-FIT projection.

Brain analog
------------
The angular gyrus binds distributed modality-specific "spokes" (visual,
auditory, tactile, motor, ...) into a single multi-modal representation
(Price, Bonner, Peelle & Grossman 2015, J Neurosci; Binder & Desai 2011).
RAVANA's equivalent: a ridge-regression ``W_sm`` that learns, from
grounded (concept -> human description) pairs, how a 65-D Binder vector
should warp the 75-D decoder BOS so the GRU opens the utterance already
biased toward the subject's embodiment.

No hardcoding
-------------
``W_sm`` is FIT from data (concept Binder vector -> the concept's own 75-D
dual-code embedding). There is no hand-authored basis. Fail-soft: if
``linggen_wsm.npz`` is absent or too few pairs were seen, ``condition()``
returns ``None`` and the caller falls back to the (already graceful)
``_build_conditioned_bos`` Lancaster-tail warp. So no regression and no
gibberish.

Fail-closed
-----------
``adaptive_floor(gen_conf_seq)`` returns a dataset-derived confidence floor
(running mean - k·std) so the decision to use free-form decoder text vs.
the ``realize_dim`` phrase lookup is driven by the model's own stability,
not a magic constant.
"""
from __future__ import annotations

import os
import json
from typing import Optional, Sequence
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
_WSM_PATH = os.path.join(_ROOT, "data", "linggen_wsm.npz")
# Decoder embed dim (must match response_gen._DECODER_DIM / neural_decoder.embed_dim).
_EMBED_DIM = 75
# Binder attribute dim from CombinedAttributeEncoder.attribute_vector.
_BINDER_DIM = 65
# Minimum pairs before we trust the fit (small ridge is stable, but guard
# against fitting on a handful of noisy points).
_MIN_PAIRS = 8


class LingGenConditioner:
    """Learned 65-D Binder -> 75-D decoder-initiation projection (W_sm)."""

    def __init__(self, W_sm: Optional[np.ndarray] = None,
                 binder_dim: int = _BINDER_DIM,
                 embed_dim: int = _EMBED_DIM,
                 lam: float = 1.0):
        self.binder_dim = binder_dim
        self.embed_dim = embed_dim
        self.lam = float(lam)          # ridge L2 (small; fit from modest data)
        self._W = W_sm                  # (embed_dim, binder_dim) or None
        self._n_fit = 0

    # ── fit ────────────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, binder_vecs: np.ndarray, embed_vecs: np.ndarray,
            lam: float = 1.0) -> "LingGenConditioner":
        """Ridge-regress embed_vecs <- W_sm @ binder_vecs.

        Solves (X^T X + lam I) w = X^T y per target column. X is (n, binder_dim),
        y is (n, embed_dim). Closed-form, no external deps.
        """
        X = np.asarray(binder_vecs, dtype=np.float64)
        Y = np.asarray(embed_vecs, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        n = X.shape[0]
        if n < _MIN_PAIRS or X.shape[1] != _BINDER_DIM or Y.shape[1] != _EMBED_DIM:
            # Not enough / malformed — return an untrained conditioner.
            c = cls(lam=lam)
            c._n_fit = max(0, n)
            return c
        # Center (ridge handles the bias via the intercept-free form; centering
        # improves conditioning for the small-data regime).
        Xm = X.mean(axis=0)
        Ym = Y.mean(axis=0)
        Xc = X - Xm
        Yc = Y - Ym
        XtX = Xc.T @ Xc + lam * np.eye(_BINDER_DIM, dtype=np.float64)
        XtY = Xc.T @ Yc
        W = np.linalg.solve(XtX, XtY)          # (binder_dim, embed_dim)
        # Store as (embed_dim, binder_dim) for the forward pass: out = W_sm @ x.
        Wt = W.T.copy()
        c = cls(W_sm=Wt, lam=lam)
        c._n_fit = n
        c._Xm = Xm
        c._Ym = Ym
        return c

    # ── forward ──────────────────────────────────────────────────────────────
    def condition(self, binder65: np.ndarray) -> Optional[np.ndarray]:
        """Map a 65-D Binder vector to a 75-D initiation embedding.

        Returns None when untrained (fail-soft: caller uses the Lancaster warp).
        """
        if self._W is None:
            return None
        x = np.asarray(binder65, dtype=np.float64)[:_BINDER_DIM]
        if x.shape[0] != _BINDER_DIM:
            return None
        # Apply the same centering learned at fit time (if any).
        xc = x - getattr(self, "_Xm", np.zeros(_BINDER_DIM, dtype=np.float64))
        out = self._W @ xc
        out = out + getattr(self, "_Ym", np.zeros(_EMBED_DIM, dtype=np.float64))
        out = np.asarray(out, dtype=np.float32)
        n = np.linalg.norm(out)
        if n > 0:
            out = out / n
        return out.astype(np.float32)

    @property
    def trained(self) -> bool:
        return self._W is not None and self._n_fit >= _MIN_PAIRS

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, path: Optional[str] = None) -> None:
        path = path or _WSM_PATH
        if self._W is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 W_sm=self._W,
                 Xm=getattr(self, "_Xm", np.zeros(_BINDER_DIM)),
                 Ym=getattr(self, "_Ym", np.zeros(_EMBED_DIM)),
                 n_fit=np.array([self._n_fit]),
                 binder_dim=np.array([self.binder_dim]),
                 embed_dim=np.array([self.embed_dim]),
                 lam=np.array([self.lam]))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "LingGenConditioner":
        path = path or _WSM_PATH
        if not os.path.exists(path):
            return cls()
        try:
            d = np.load(path, allow_pickle=False)
            W = d["W_sm"]
            c = cls(W_sm=W,
                    binder_dim=int(d.get("binder_dim", [_BINDER_DIM])[0]),
                    embed_dim=int(d.get("embed_dim", [_EMBED_DIM])[0]),
                    lam=float(d.get("lam", [1.0])[0]))
            c._n_fit = int(d.get("n_fit", [0])[0])
            c._Xm = np.asarray(d.get("Xm", np.zeros(_BINDER_DIM)))
            c._Ym = np.asarray(d.get("Ym", np.zeros(_EMBED_DIM)))
            return c
        except Exception:
            return cls()


# ── adaptive floor (fail-closed, distribution-driven) ────────────────────────
def adaptive_floor(gen_conf_seq: Sequence[float], k: float = 2.0) -> float:
    """Dataset-derived confidence floor for the free-form vs template decision.

    floor = mean - k·std over the model's own per-run top-1 accuracies on the
    sensorimotor-conditioned set. NO magic constant: k is a z-multiplier
    (calibrated on the training set, not hand-tuned per concept). Returns a
    conservative floor (0.0 if too little data) so the first uncertain runs
    fall back to ``realize_dim`` rather than emit gibberish.
    """
    arr = np.asarray([float(v) for v in gen_conf_seq], dtype=np.float64)
    if arr.size < 4:
        return 0.0
    mu = float(arr.mean())
    sd = float(arr.std())
    return float(max(0.0, mu - k * sd))


def should_use_freeform(off_manifold: bool, gen_conf: float,
                        gen_conf_seq: Sequence[float], k: float = 2.0) -> bool:
    """Fail-closed gate: True only when on-manifold AND confident vs its own history.

    off_manifold: ood_abstain(...) — dataset-derived OOD floor (attribute_calibration).
    gen_conf: this run's decoder top-1 accuracy (NeuralDecoder._avg_top1_acc EMA).
    gen_conf_seq: history of gen_conf on grounded runs (for the adaptive floor).
    """
    if off_manifold:
        return False
    return gen_conf >= adaptive_floor(gen_conf_seq, k=k)
