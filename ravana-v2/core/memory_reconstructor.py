"""
MemoryReconstructor — human-like reconstructive recall.

Humans don't retrieve exact memory records. They reconstruct from fragments:
a partial cue activates related memories, which blend into a coherent
reconstruction. The result is shaped by what's stored AND by the retrieval
context.

This module implements that process:
1. Partial cue -> vector search for candidate memories
2. Candidates activate graph neighbors via spreading activation
3. Seed + neighbor context are blended into a reconstruction
4. A fidelity score tracks how much is direct vs inferred
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MemoryReconstructor:
    """Reconstructs full memories from partial cues using vector blending."""

    def __init__(self, vector_index: Any, memory_store: Any):
        """
        Args:
            vector_index: SharedVectorIndex instance.
            memory_store: HumanMemoryEngine instance.
        """
        self.index = vector_index
        self.store = memory_store

    def reconstruct(
        self,
        cue_vector: np.ndarray,
        cue_text: str = "",
        k: int = 5,
        blend_depth: int = 3,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Reconstruct memories from a partial cue.

        Algorithm:
        1. Vector search for top-k candidates from cue_vector
        2. If cue_text provided, boost candidates with text overlap
        3. For each candidate, spread activation through graph neighbors
        4. Blend candidate content + neighbor context into reconstruction
        5. Compute fidelity score (how much is direct vs inferred)

        Args:
            cue_vector: The partial cue as an embedding vector.
            cue_text: Optional text hint for hybrid scoring.
            k: Number of reconstructions to return.
            blend_depth: Max number of neighbors to blend per candidate.
            memory_type: Optional filter by memory type.

        Returns:
            List of reconstructed memory dicts, sorted by reconstruction score.
        """
        # Step 1: Vector search for initial candidates
        candidates = self.index.search(cue_vector, k=k * 3, min_score=0.05)
        if not candidates:
            return []

        # Step 2: Optional text boost (hybrid scoring)
        scored_candidates = []
        for mid, vec_score in candidates:
            mem = self.store._get(mid)
            if not mem:
                continue
            if memory_type and mem.get("memory_type") != memory_type:
                continue
            # Hybrid score: 0.7 cosine + 0.3 text overlap
            if cue_text:
                text_overlap = self._token_overlap(cue_text, mem.get("content", ""))
                score = vec_score * 0.7 + text_overlap * 0.3
            else:
                score = vec_score
            scored_candidates.append((mem, score))

        # Sort by hybrid score
        scored_candidates.sort(key=lambda x: -x[1])

        # Step 3: For each candidate, spread through graph and blend
        results = []
        seen_ids = set()
        for seed_mem, seed_score in scored_candidates:
            if len(results) >= k:
                break
            mid = seed_mem["id"]
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            # Graph spreading activation
            nid = self.store._node_id(mid)
            try:
                activated = self.store._spreading_activation([nid])
            except Exception:
                activated = []

            # Collect activated neighbors
            neighbors = []
            for node_id, act_score in activated:
                if node_id == nid or act_score < 0.05:
                    continue
                try:
                    neighbor_mid = int(node_id.split("-")[1])
                except (ValueError, IndexError):
                    continue
                neighbor = self.store._get(neighbor_mid)
                if neighbor:
                    neighbor["_act_score"] = act_score
                    neighbors.append(neighbor)

            # Sort neighbors by activation, take top blend_depth
            neighbors.sort(key=lambda x: -x.get("_act_score", 0))
            neighbors = neighbors[:blend_depth]

            # Step 4: Blend
            blended = self._blend_memories(
                seed_mem, neighbors,
                [n.get("_act_score", 0) for n in neighbors]
            )
            blended["match_score"] = round(seed_score, 4)
            blended["seed_id"] = mid

            # Step 5: Fidelity score
            neighbor_scores = [n.get("_act_score", 0) for n in neighbors]
            fidelity = self._compute_fidelity(
                seed_score, neighbor_scores,
                seed_mem.get("coherence", 1.0)
            )
            blended["reconstruction_fidelity"] = round(fidelity, 4)
            blended["neighbor_count"] = len(neighbors)
            blended["reconstructed"] = len(neighbors) > 0

            results.append(blended)

        # Sort by reconstruction score
        results.sort(key=lambda x: -(
            x.get("reconstruction_fidelity", 0) * x.get("match_score", 0)
        ))
        return results

    def _blend_memories(
        self,
        seed: Dict[str, Any],
        neighbors: List[Dict[str, Any]],
        activation_scores: List[float],
    ) -> Dict[str, Any]:
        """Weighted blend of memory attributes based on activation.

        The seed memory provides the base. Neighbor contributions are
        weighted by their activation score — strongly activated neighbors
        contribute more context.
        """
        result = dict(seed)

        if not neighbors:
            return result

        # Blend content: seed + weighted neighbor fragments
        seed_content = seed.get("content", "")
        neighbor_fragments = []
        all_tags = set(t.strip() for t in (seed.get("tags") or "").split(",") if t.strip())

        total_weight = sum(activation_scores) + 1e-15
        for neighbor, score in zip(neighbors, activation_scores):
            n_content = neighbor.get("content", "")
            if n_content:
                neighbor_fragments.append(n_content)
            for t in (neighbor.get("tags") or "").split(","):
                t = t.strip()
                if t:
                    all_tags.add(t)

        if neighbor_fragments:
            result["content"] = (
                seed_content
                + " [associated: " + "; ".join(neighbor_fragments[:3]) + "]"
            )

        # Blend importance and emotional by activation-weighted average
        if activation_scores:
            imp = seed.get("importance", 0.5)
            emo = seed.get("emotional", 0.5)
            for n, s in zip(neighbors, activation_scores):
                w = s / total_weight
                imp += n.get("importance", 0.5) * w * 0.3  # partial blend
                emo += n.get("emotional", 0.5) * w * 0.3
            result["blended_importance"] = min(1.0, imp)
            result["blended_emotional"] = min(1.0, emo)

        result["reconstructed_tags"] = ",".join(sorted(all_tags))
        return result

    @staticmethod
    def _compute_fidelity(
        seed_score: float,
        neighbor_scores: List[float],
        seed_coherence: float,
    ) -> float:
        """How much of the reconstruction is direct vs inferred.

        High fidelity = mostly from the seed memory (direct recall).
        Low fidelity = heavily reconstructed from neighbors (inferred).
        """
        if not neighbor_scores:
            return seed_score * seed_coherence

        direct_ratio = seed_score
        graph_ratio = min(1.0, sum(neighbor_scores) * 0.2)
        fidelity = direct_ratio * 0.6 + graph_ratio * 0.4
        # Coherence modulates fidelity — degraded memories have lower fidelity
        fidelity *= max(0.1, seed_coherence)
        return min(1.0, fidelity)

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        """Compute token overlap ratio between two texts."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = tokens_a & tokens_b
        return len(overlap) / max(len(tokens_a), len(tokens_b))
