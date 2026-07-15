"""Attractor Memory Store — pattern completion for the 30-Watt generator.

Brain basis (see plan "The 30-Watt Proof of Concept"):
  - A PARTIAL concept cue reactivates the WHOLE coherent memory in one shot
    (Hopfield 1982; Khona & Fiete 2022, Nature Reviews Neuroscience 23:744).
    Not word-by-word sampling from a void.
  - Coherent speech is RETRIEVAL + local settling, not generation-from-scratch
    (Cell Patterns 2022: "Predictive coding is a consequence of energy
    efficiency").

What this module does
---------------------
  * Stores each KB definition as an HRR-bound structure:
        attractor = hrr_bind(concept_vec, trajectory_of_word_embeddings)
    where `trajectory_of_word_embeddings` is the ordered sequence of 75-D
    dual-code embeddings of the definition's words (the "retrieved coherent
    trajectory", the attractor the generator settles toward).
  * Content-addressable retrieval = pattern completion: given a (possibly
    partial / noisy) concept cue vector, recover the bound attractor via
    hrr_unbind, then read off the word-embedding trajectory.
  * Mirrors the hippocampal_buffer pattern-completion retrieval logic, but
    operates in the DISTRIBUTIONAL embedding space (not string triples), so
    the reactivated trace is directly consumable by the decoder settle loop.

No hardcoding: the store is built from engine._definitions (clean KB) + the
decoder's own word embeddings. Fail-closed: retrieve() returns None when
nothing is on-manifold (caller falls back to realize_dim / retrieved text).

This is the Phase 0 primitive the PredictiveCodingGenerator (Phase 2) consumes.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from ravana.core.vsa import hrr_bind, hrr_unbind, cosine_sim


class AttractorMemory:
    """Content-addressable HRR store of (concept -> word-embedding trajectory)."""

    def __init__(self, dim: int = 75, role_dim: int = 75):
        self.dim = dim
        # Single reusable role vector (the "definition" role) in the same space
        # as the word embeddings, used to bind concept_vec -> trajectory. We keep
        # it fixed (deterministic) so retrieval is reproducible without persisting
        # role tables; the binding is still content-addressable because concept_vec
        # carries the concept identity.
        rng = np.random.RandomState(20260630)
        r = rng.randn(role_dim).astype(np.float64)
        r /= np.linalg.norm(r)
        self._role = r
        # concept_vec (dim,) -> bound attractor (dim,)
        self._attractors: Dict[str, np.ndarray] = {}
        # concept_vec key -> ordered (word, embed) trajectory (for introspection)
        self._trajectories: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        # concept_vec key -> concept string (original label)
        self._labels: Dict[str, str] = {}

    # ── build ──────────────────────────────────────────────────────────────
    @classmethod
    def from_definitions(
        cls,
        definitions: Dict[str, str],
        word_to_embed: Dict[str, np.ndarray],
        concept_vec_fn,
        dim: int = 75,
        max_words: int = 24,
        stop: Optional[set] = None,
    ) -> "AttractorMemory":
        """Construct the store from a clean KB.

        Args:
            definitions: concept -> definition text (e.g. engine._definitions).
            word_to_embed: word -> (dim,) decoder dual-code embedding.
            concept_vec_fn: callable(str) -> Optional[np.ndarray]; maps the
                concept label to its embedding (graph node vec or _embed_75d).
                If it returns None for a concept, that concept is skipped.
            max_words: cap trajectory length (a definition is a bounded cue,
                not an open-ended generation).
            stop: words to drop from the trajectory (cheap function words whose
                embedding we don't need to anchor the attractor).
        """
        am = cls(dim=dim)
        stop = stop or set()
        for concept, text in definitions.items():
            cvec = concept_vec_fn(concept)
            if cvec is None:
                continue
            cvec = np.asarray(cvec, dtype=np.float64)[:dim]
            n = np.linalg.norm(cvec)
            if n == 0:
                continue
            cvec = cvec / n
            # Build the word-embedding trajectory (ordered, content words first).
            traj: List[Tuple[str, np.ndarray]] = []
            for w in text.replace("\n", " ").split():
                w = w.lower().strip(".,!?;:\"'()[]")
                if not w or w in stop:
                    continue
                e = word_to_embed.get(w)
                if e is None:
                    continue
                e = np.asarray(e, dtype=np.float64)[:dim]
                en = np.linalg.norm(e)
                if en == 0:
                    continue
                traj.append((w, e / en))
                if len(traj) >= max_words:
                    break
            if not traj:
                continue
            key = concept.lower()
            am._attractors[key] = am._bind(cvec, traj)
            am._trajectories[key] = traj
            am._labels[key] = concept
        return am

    # ── HRR binding of a trajectory ─────────────────────────────────────────
    def _bind(self, concept_vec: np.ndarray,
              traj: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """attractor = hrr_bind(concept_vec, perm(traj)) after permutation.

        We bind the concept to a SUPERPOSED+permuted trajectory so the attractor
        encodes BOTH which concept and the gist of its definition words. A pure
        superposition of trajectory words would be a bag-of-words; convolving with
        a fixed permutation (role channel) preserves a little of the ordering
        cue while keeping the structure recoverable by hrr_unbind + inverse
        permutation. This is the HRR "role-filler" idiom from core/vsa.py.
        """
        traj_vec = np.zeros(self.dim, dtype=np.float64)
        for _, e in traj:
            traj_vec = traj_vec + e
        if np.linalg.norm(traj_vec) > 0:
            traj_vec = traj_vec / np.linalg.norm(traj_vec)
        # Permute the trajectory channel so it is decodable separately from the
        # concept channel (role-filler separation). Retrieval un-rolls this.
        # The concept identity lives in the HRR binding (hrr_bind(concept_vec,
        # perm)), so unbind(cue) recovers perm(traj) only when cue ~= concept_vec
        # — genuine content-addressable pattern completion from a partial cue.
        perm = self._permute(traj_vec, shift=7)
        return hrr_bind(concept_vec, perm).astype(np.float64)

    def _unpermute(self, v: np.ndarray, shift: int = 7) -> np.ndarray:
        """Inverse of _permute (roll back) to recover the stored trajectory."""
        return np.roll(v, -shift).copy()

    @staticmethod
    def _permute(v: np.ndarray, shift: int) -> np.ndarray:
        return np.roll(v, shift).copy()

    # ── retrieval / pattern completion ──────────────────────────────────────
    def retrieve(self, concept_vec: np.ndarray,
                 threshold: float = 0.15) -> Optional[List[Tuple[str, np.ndarray]]]:
        """Pattern-completion retrieval: partial/noisy cue -> full trajectory.

        Unbinds the concept_vec from each stored attractor; the one whose
        recovered trajectory best matches (mean cosine to its own stored words)
        ABOVE `threshold` is returned. Because unbinding a near-correct cue
        still recovers a near-correct trajectory (HRR tolerance to noise), this
        is one-shot pattern completion from a partial cue — the brain-faithful
        alternative to autoregressive sampling from a void.
        """
        cvec = np.asarray(concept_vec, dtype=np.float64)[:self.dim]
        n = np.linalg.norm(cvec)
        if n == 0:
            return None
        cvec = cvec / n
        best_key, best_score = None, -1.0
        for key, attr in self._attractors.items():
            recovered = hrr_unbind(attr, cvec)
            recovered = self._unpermute(recovered)
            traj = self._trajectories[key]
            sims = []
            for _, e in traj:
                sims.append(cosine_sim(recovered, e))
            score = float(np.mean(sims)) if sims else 0.0
            if score > best_score:
                best_score, best_key = score, key
        if best_key is not None and best_score >= threshold:
            return self._trajectories[best_key]
        return None

    def recover_trajectory(self, concept_vec: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Always return the best-matching trajectory (even if below threshold).

        Used by the settle loop as the ATTRACTOR the GRU settles toward; the
        settled output is then gated by the threshold externally so we stay
        fail-closed.
        """
        cvec = np.asarray(concept_vec, dtype=np.float64)[:self.dim]
        n = np.linalg.norm(cvec)
        if n == 0 or not self._attractors:
            return []
        cvec = cvec / n
        best_key, best_score = None, -1.0
        for key, attr in self._attractors.items():
            recovered = hrr_unbind(attr, cvec)
            recovered = self._unpermute(recovered)
            traj = self._trajectories[key]
            sims = [cosine_sim(recovered, e) for _, e in traj]
            score = float(np.mean(sims)) if sims else 0.0
            if score > best_score:
                best_score, best_key = score, key
        return self._trajectories[best_key] if best_key else []

    def pattern_completion_score(self, concept_vec: np.ndarray) -> float:
        """Score (mean cosine) of the best-matching recovered trajectory.

        The brain-faithful confidence signal for the fail-closed gate: how
        well the cue activates a known attractor. A low score means the cue is
        off-manifold -> do NOT settle, fall back to realize_dim.
        """
        cvec = np.asarray(concept_vec, dtype=np.float64)[:self.dim]
        n = np.linalg.norm(cvec)
        if n == 0 or not self._attractors:
            return 0.0
        cvec = cvec / n
        best = -1.0
        for attr in self._attractors.values():
            recovered = hrr_unbind(attr, cvec)
            recovered = self._unpermute(recovered)
            traj = self._trajectories[next(k for k, a in self._attractors.items() if a is attr)]
            sims = [cosine_sim(recovered, e) for _, e in traj]
            best = max(best, float(np.mean(sims)) if sims else 0.0)
        return best

    def __len__(self) -> int:
        return len(self._attractors)

    def keys(self):
        return list(self._attractors.keys())
