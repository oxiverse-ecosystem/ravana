"""
N3 — dual-code HRR space (additive, not a migration).
======================================================
Keeps 64D GloVe for intent/prototypes (proven 87-90% in A1/N1/N4/N2) and
ADDS a 2048-D HRR space alongside, ONLY for binding / analogy / resonator
decode. Brain-faithful dual-coding (Paivio; CLS multiple systems). Matches
Word2HyperVec (Ayar et al. 2024): a linear lift on top of embeddings, not a
vsa.py rewrite. This kills the cross-cutting-risk objection that deferred N3.

Why high-D: VSA noise ~ 1/d. At 64D, GloVe atoms are non-orthogonal so circular
convolution degenerates (that's exactly why the earlier A0 probe regressed).
Lifting to 2048-D makes bound vectors quasi-orthogonal, so role-filler unbinding
and analogy actually work. Validated in experiments/n3_dual_code_hrr.py:
  - clean single-fact recovery error 0.555 (64D) -> 0.489 (2048D)
  - same-relation structure similarity 0.776 (analogy works)

The 64D glove space is NEVER modified. This module is inert unless instantiated,
so it cannot regress the production intent pipeline.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from ravana.core.vsa import hrr_bind, hrr_unbind, cosine_sim
from ravana.ontology.attribute_encoder import build_glove64_lookup


DEFAULT_HRR_DIM = 2048


class DualCodeSpace:
    """Additive dual-code: 64D glove (intent) + lifted 2048D HRR (binding)."""

    def __init__(self, glove_cache_npz: str, hrr_dim: int = DEFAULT_HRR_DIM,
                 lift_seed: int = 0, whiten: bool = True, sparse_k: int = 0,
                 unitary_roles: bool = True):
        self._lut64, self._dim64 = build_glove64_lookup(glove_cache_npz)
        self.hrr_dim = hrr_dim
        self.whiten = whiten
        self.sparse_k = int(sparse_k)
        self.unitary_roles = unitary_roles
        # Word2HyperVec-style random linear lift (JL-preserving, scaled).
        rng = np.random.RandomState(lift_seed)
        self._lift = (rng.randn(hrr_dim, self._dim64).astype(np.float32)
                      / np.sqrt(self._dim64))
        # ZCA-sphering fit on an HRR-side COPY of the LUT (Limitation 1).
        # We transform on the fly only for HRR atoms; self._lut64 / atom64()
        # are NEVER mutated, so the proven 87-90% intent pipeline is untouched.
        self._whiten = None
        self._whiten_mean = None
        if whiten and self._lut64:
            try:
                vals = list(self._lut64.values())
                # Sample for covariance estimate (64-D -> a few 10k rows suffice).
                if len(vals) > 32768:
                    perm = np.random.RandomState(0).permutation(len(vals))[:32768]
                    mat = np.stack([vals[i] for i in perm]).astype(np.float64)
                else:
                    mat = np.stack(vals).astype(np.float64)
                self._whiten_mean = mat.mean(axis=0)
                centered = mat - self._whiten_mean
                cov = (centered.T @ centered) / (len(mat) - 1)
                eigval, eigvec = np.linalg.eigh(cov + 1e-6 * np.eye(self._dim64))
                eps = 1e-4
                d = np.diag(1.0 / np.sqrt(np.maximum(eigval, eps)))
                self._whiten = (eigvec @ d @ eigvec.T).astype(np.float32)
            except Exception:
                self._whiten = None
                self._whiten_mean = None
        # Cached high-D atoms
        self._lut_hrr: Dict[str, np.ndarray] = {}
        # Role vectors for binding (subject/verb/object/...)
        self._roles: Dict[str, np.ndarray] = {}

    # ── 64D intent space (untouched glove) ──
    def atom64(self, word: str) -> Optional[np.ndarray]:
        v = self._lut64.get(word.lower())
        if v is None:
            return None
        return v / (np.linalg.norm(v) + 1e-9)

    # ── 2048D binding space (lifted) ──
    def atom_hrr(self, word: str) -> np.ndarray:
        w = word.lower()
        cached = self._lut_hrr.get(w)
        if cached is not None:
            return cached
        v64 = self._lut64.get(w)
        if v64 is None:
            # OOV: deterministic random unit atom in HRR space
            r = np.random.RandomState(abs(hash(w)) % (2 ** 31)).randn(self.hrr_dim)
            r = r.astype(np.float32)
            nv = np.linalg.norm(r)
            r = r / nv if nv > 0 else r
            self._lut_hrr[w] = r
            return r
        # HRR-side decorrelation (Limitation 1): ZCA-whiten the 64D atom on the
        # fly, then JL-lift. The stored _lut64 / atom64() are untouched.
        v = np.asarray(v64, dtype=np.float32)
        if self._whiten is not None and self._whiten_mean is not None:
            v = self._whiten @ (v - self._whiten_mean)
        h = self._lift @ v
        if self.sparse_k and self.sparse_k > 0:
            # Sparse high-D code (Olshausen & Field 1996; Frady et al. 2022):
            # keep top-k magnitudes, ternarize the rest to +/-1/0. Cuts binding
            # crosstalk. Normalized after for binding stability.
            k = min(self.sparse_k, self.hrr_dim)
            idx = np.argpartition(np.abs(h), -k)[-k:]
            mask = np.zeros(self.hrr_dim, dtype=np.float32)
            mask[idx] = np.sign(h[idx])
            h = mask
        nh = np.linalg.norm(h)
        h = h / nh if nh > 0 else h
        self._lut_hrr[w] = h.astype(np.float32)
        return self._lut_hrr[w]

    def role(self, name: str) -> np.ndarray:
        r = self._roles.get(name)
        if r is not None:
            return r
        rng = np.random.RandomState(abs(hash("role:" + name)) % (2 ** 31))
        r = rng.randn(self.hrr_dim).astype(np.float32)
        if self.unitary_roles:
            # Force |FFT(role)| = 1 so HRR unbind is the EXACT inverse of bind
            # (Plate 2003). Without this, role fillers crosstalk on recovery.
            f = np.fft.rfft(r, n=self.hrr_dim)
            mag = np.abs(f)
            mag[mag == 0] = 1.0
            f = f / mag
            r = np.fft.irfft(f, n=self.hrr_dim).astype(np.float32)
        nr = np.linalg.norm(r)
        r = r / nr if nr > 0 else r
        self._roles[name] = r
        return r

    # ── binding primitives ──
    def bind_role(self, role: str, filler_word: str) -> np.ndarray:
        return hrr_bind(self.role(role), self.atom_hrr(filler_word))

    def bind_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return hrr_bind(a, b)

    def unbind_role(self, structure: np.ndarray, role: str) -> np.ndarray:
        return hrr_unbind(structure, self.role(role))

    def bundle(self, vectors: list) -> np.ndarray:
        if not vectors:
            return np.zeros(self.hrr_dim, dtype=np.float32)
        s = np.sum(vectors, axis=0)
        ns = np.linalg.norm(s)
        return (s / ns).astype(np.float32) if ns > 0 else s.astype(np.float32)

    # ── fact -> bound structure (uses C-lite relations) ──
    def encode_fact(self, subject: str, verb: str, obj: str) -> np.ndarray:
        return self.bundle([
            self.bind_role("subject", subject),
            self.bind_role("verb", verb),
            self.bind_role("object", obj),
        ])

    def recover_role_filler_with_conf(self, structure: np.ndarray, role: str,
                                       candidate_words: list) -> Tuple[Optional[str], float]:
        """Resonator-style decode: unbind role, return nearest candidate word in
        HRR space. Returns (best_word, best_cosine) — the cosine is the confidence
        proxy used by the reasoning benchmark (decode similarity ≈ recollection
        strength; Yonelinas)."""
        rec = self.unbind_role(structure, role)
        best, best_sim = None, -1.0
        for w in candidate_words:
            sim = cosine_sim(rec, self.atom_hrr(w))
            if sim > best_sim:
                best_sim, best = sim, w
        return best, float(best_sim)

    def recover_role_filler(self, structure: np.ndarray, role: str,
                            candidate_words: list) -> Optional[str]:
        """Resonator-style decode: unbind role, return nearest candidate word
        in HRR space (the 'resonator' of Eliasmith 2012 / Plate 2003)."""
        best, _ = self.recover_role_filler_with_conf(structure, role, candidate_words)
        return best

    def relation_similarity(self, fact_a: Tuple[str, str, str],
                            fact_b: Tuple[str, str, str]) -> float:
        """Analogy probe: same relational structure -> high similarity.
        Compare subject*verb bundles (the 'shape' of the fact)."""
        sa = self.bundle([self.bind_role("subject", fact_a[0]),
                          self.bind_role("verb", fact_a[1])])
        sb = self.bundle([self.bind_role("subject", fact_b[0]),
                          self.bind_role("verb", fact_b[1])])
        return cosine_sim(sa, sb)
