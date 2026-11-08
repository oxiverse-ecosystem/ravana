"""Graph plasticity wiring: the ConceptGraph's own learning rules, reached
from the chat engine.

`ravana_ml.graph.ConceptGraph` already implements the brain's plasticity
mechanisms — and implements them well:

  * `hebbian_update`              surprise-driven Hebbian delta with an EWC
                                  penalty and a Beta posterior
  * `apply_prediction_error`      free-energy accumulation, contradiction
                                  counting, plasticity escalation
  * `form_inhibitory_edges`       lateral inhibition between competing targets
  * `compute_edge_structural_importance`  topology-aware importance (SI analogue)
  * `should_split` / `split_concept`      polysemy resolution by splitting
  * `homeostatic_downscale`       SHY-style synaptic homeostasis
  * `consolidate_vectors`         fast active vector -> slow core vector
  * `record_path` / `compress_paths`      inference chains -> shortcut edges
  * `reconcile_contradictions`    contradiction hotspots -> stability/FE decay

Every one of them was reachable only from `ravana_ml/nn/rlm.py`. None was
called from `ravana/src/ravana/chat/`, i.e. from the path that actually serves
a turn. `consolidate_vectors`, `compress_paths`, `record_path`,
`should_split`, `split_concept` and `apply_prediction_error` had ZERO callers
anywhere in the chat engine.

This module is the wiring. It adds no new learning rule — it calls the existing
ones from the right places, with the brain's gating:

  * **Per turn**, an edge that is actually traversed and whose weight poorly
    predicts the observed vector similarity is corrected by a bounded step,
    gated by salience. Predictable edges are not touched: they cost nothing
    (Friston 2010; Chang, Dell & Bock 2006 — plasticity is gated by prediction
    error, not by exposure).
  * **At sleep**, the structural pass runs: inhibitory edges, concept
    splitting (rate-limited), vector consolidation, contradiction
    reconciliation and path compression. Rate limits and a global concept
    budget keep it from runaway growth — the same guards `rlm.py` uses.

Grounding: McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis &
McClelland 2016; Abraham & Bear 1996 (metaplasticity — plasticity must be
gated, not unconditional); Rasch & Born 2013 (sleep consolidation); Lewis &
Durrant 2011 (overlapping replay builds structure); Zenke, Agnes & Gerstner
2017 (continual learning needs both fast and protected weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PlasticityConfig:
    """Knobs for the wiring. Defaults are deliberately conservative: the
    per-turn rule only fires on genuinely surprising edges and is capped."""

    # ── per-turn, surprise-gated ────────────────────────────────────────────
    edge_pe_learning: bool = True
    edge_pe_learning_rate: float = 0.02
    # Only edges whose squared weight/similarity mismatch exceeds this are
    # updated. Below it the prediction is good enough and learning is free.
    edge_pe_error_threshold: float = 0.05
    edge_pe_max_updates_per_turn: int = 8
    edge_pe_min_weight: float = 0.05

    # ── sleep-time structural pass ──────────────────────────────────────────
    form_inhibitory: bool = True
    inhibitory_contradiction_threshold: int = 3
    reconcile_contradictions: bool = True
    split_concepts: bool = True
    max_splits_per_cycle: int = 2
    # Global budget: stop splitting past this multiple of the concept count at
    # the start of the cycle (prevents runaway graph growth).
    split_max_concepts_factor: float = 1.5
    split_drift_threshold: float = 0.4
    split_contradiction_threshold: int = 4
    consolidate_vectors: bool = True
    consolidate_rate: float = 0.08
    # SHY-style synaptic homeostasis. `structural_protection > 0` makes the
    # graph compute its own topology-aware importance (the Synaptic
    # Intelligence analogue) and protect structurally critical edges.
    homeostatic_downscale: bool = True
    homeostatic_protection_threshold: float = 0.8
    homeostatic_downscale_factor: float = 0.8
    homeostatic_structural_protection: float = 0.5
    # Feed the computed structural importance into edge.fisher_importance so
    # hebbian_update's EWC penalty actually protects important synapses.
    si_to_ewc: bool = True
    compress_paths: bool = True
    compress_min_usage: int = 2
    compress_min_score: float = 0.15
    # ── forgetting (SHY-style down-selection) ────────────────────────────────
    # Wake potentiates, sleep down-selects: most synapses weaken, reactivated
    # ones are spared. Without this the graph could only strengthen, so
    # retention was not a variable -- every fact ever written was retained
    # forever and "desirable difficulty" had nothing to act on.
    decay_edges: bool = True
    # Fraction of confidence lost per elapsed turn before the rehearsal
    # halflife. Deliberately small: the point is graded forgetting, not
    # amnesia, and the rehearsal rescue is what does the real work.
    decay_base_rate: float = 0.02
    # Hours a freshly-rehearsed edge is protected for. Retrieval practice
    # stamps last_rehearsal, so a fact just reviewed is not decayed.
    decay_rehearsal_halflife: float = 6.0
    decay_min_confidence: float = 0.05


class GraphPlasticity:
    """Drive the ConceptGraph's own plasticity from the chat engine.

    Stateless apart from the per-turn update budget and the counters; safe to
    construct once and share.
    """

    def __init__(self, config: Optional[PlasticityConfig] = None):
        self.config = config or PlasticityConfig()
        self._updates_this_turn = 0
        self.stats: Dict[str, int] = field(default_factory=dict)  # type: ignore[assignment]
        self.stats = {
            "edge_updates": 0,
            "edge_updates_skipped_predictable": 0,
            "prediction_errors_applied": 0,
            "paths_recorded": 0,
            "paths_compressed": 0,
            "inhibitory_edges_formed": 0,
            "concepts_split": 0,
            "contradictions_reconciled": 0,
            "vectors_consolidated": 0,
            "edges_downscaled": 0,
            "si_edges_protected": 0,
            "edges_decayed": 0,
        }

    # ── per-turn budget ──────────────────────────────────────────────────────

    def reset_budget(self) -> None:
        """Reset the update budget.

        The budget bounds how many edge corrections one retrieval pass may
        apply, so a long conversation cannot accumulate unbounded change.
        """
        self._updates_this_turn = 0

    def begin_turn(self) -> None:
        self.reset_budget()

    def end_turn(self) -> None:
        self.reset_budget()

    # ── per-turn: prediction-error-driven edge correction ───────────────────

    def on_edge_traversed(self, graph, edge,
                          src_vec: np.ndarray, tgt_vec: np.ndarray,
                          salience: float = 1.0) -> float:
        """Correct one traversed edge's weight toward observed similarity.

        The edge's weight is its prediction of how related its endpoints are;
        the cosine of the endpoint vectors is the observation. The squared
        residual is the prediction error. The correction is a bounded gradient
        step, scaled by salience — so a predictable edge is left alone and a
        surprising one moves.

        Returns the prediction error (0.0 when nothing was updated).
        """
        cfg = self.config
        if not cfg.edge_pe_learning:
            return 0.0
        if self._updates_this_turn >= cfg.edge_pe_max_updates_per_turn:
            return 0.0
        if salience is None or salience <= 0.0:
            return 0.0
        try:
            w = float(edge.weight)
        except Exception:
            return 0.0

        sn = float(np.linalg.norm(src_vec)) or 1e-15
        tn = float(np.linalg.norm(tgt_vec)) or 1e-15
        actual = float(np.dot(src_vec, tgt_vec)) / (sn * tn)
        error = (w - actual) ** 2

        if error < cfg.edge_pe_error_threshold:
            # Predictable: no update, no cost. This is the 20 W property —
            # plasticity is gated by surprise, not by exposure.
            self.stats["edge_updates_skipped_predictable"] += 1
            return 0.0

        delta = -cfg.edge_pe_learning_rate * 2.0 * (w - actual) * float(salience)
        new_w = float(np.clip(w + delta, cfg.edge_pe_min_weight, 0.99))
        try:
            edge.weight = new_w
            edge.confidence = max(0.05, 1.0 - float(np.tanh(error * 3.0)))
            alpha = 0.1
            prev_fe = float(getattr(edge, "prediction_free_energy", 0.0) or 0.0)
            edge.prediction_free_energy = (1 - alpha) * prev_fe + alpha * error
            edge.prediction_count = int(getattr(edge, "prediction_count", 0)) + 1
        except Exception:
            return 0.0

        self._updates_this_turn += 1
        self.stats["edge_updates"] += 1
        return error

    def on_prediction_error(self, graph, predicted_nids: List[int],
                            actual_vector: np.ndarray) -> int:
        """Report a failed prediction to the graph.

        `apply_prediction_error` accumulates free energy on the missed nodes,
        escalates their contradiction count, and lowers their stability when
        pressure is high — which is exactly what makes a repeatedly-wrong
        concept easier to relearn (and eventually to split).
        """
        if not predicted_nids:
            return 0
        try:
            graph.apply_prediction_error(list(predicted_nids), actual_vector)
            self.stats["prediction_errors_applied"] += len(predicted_nids)
            return len(predicted_nids)
        except Exception:
            return 0

    def record_successful_path(self, graph, src_id: int, tgt_id: int) -> None:
        """Record a traversal that worked, for later path compression.

        Repeated successful inference chains get compressed into shortcut
        edges at sleep — the graph equivalent of "practice makes the route
        automatic".
        """
        if src_id is None or tgt_id is None or src_id == tgt_id:
            return
        try:
            graph.record_path(int(src_id), int(tgt_id))
            self.stats["paths_recorded"] += 1
        except Exception:
            pass

    # ── sleep-time structural pass ───────────────────────────────────────────

    def sleep_cycle(self, graph) -> Dict[str, int]:
        """Run the structural half of consolidation.

        Order matters and mirrors `ravana_ml/nn/rlm.py`:
          1. lateral inhibition between persistently contradictory targets
          2. concept splitting, rate-limited and budget-capped
          3. fast active vectors -> slow core vectors (two timescales)
          4. contradiction reconciliation (stability / free-energy decay)
          5. path compression (successful chains -> shortcut edges)
        """
        cfg = self.config
        out: Dict[str, int] = {}
        if graph is None:
            return out

        n_before = len(graph.nodes)

        if cfg.form_inhibitory:
            try:
                out["inhibitory_edges_formed"] = int(
                    graph.form_inhibitory_edges(
                        contradiction_threshold=cfg.inhibitory_contradiction_threshold))
            except Exception:
                out["inhibitory_edges_formed"] = 0
            self.stats["inhibitory_edges_formed"] += out["inhibitory_edges_formed"]

        if cfg.split_concepts:
            out["concepts_split"] = self._split_concepts(graph, n_before)
            self.stats["concepts_split"] += out["concepts_split"]

        if cfg.consolidate_vectors:
            try:
                before = self._mean_drift(graph)
                graph.consolidate_vectors(rate=cfg.consolidate_rate,
                                          incremental=True)
                after = self._mean_drift(graph)
                moved = int(round((before - after) * 1000))
                out["vectors_consolidated"] = max(0, moved)
            except Exception:
                out["vectors_consolidated"] = 0
            self.stats["vectors_consolidated"] += out["vectors_consolidated"]

        if cfg.homeostatic_downscale:
            out["edges_downscaled"], out["si_edges_protected"] = \
                self._homeostatic_downscale(graph)
            self.stats["edges_downscaled"] += out["edges_downscaled"]
            self.stats["si_edges_protected"] += out["si_edges_protected"]

        if cfg.reconcile_contradictions:
            try:
                out["contradictions_reconciled"] = int(
                    graph.reconcile_contradictions())
            except Exception:
                out["contradictions_reconciled"] = 0
            self.stats["contradictions_reconciled"] += out["contradictions_reconciled"]

        if cfg.compress_paths:
            out["paths_compressed"] = self._compress_paths(graph)
            self.stats["paths_compressed"] += out["paths_compressed"]

        # Forgetting runs LAST: it must not confound the retention measurement
        # above, and it must see the rehearsal stamps this cycle produced.
        if cfg.decay_edges:
            out["edges_decayed"] = self._decay_edges(graph)
            self.stats["edges_decayed"] += out["edges_decayed"]

        return out

    def _decay_edges(self, graph) -> int:
        """SHY-style down-selection: weaken what was not reactivated.

        Runs after the retention measurement so it cannot confound it, and
        after every rehearsal stamp this cycle produced.
        """
        cfg = self.config
        try:
            return int(graph.decay_edges(
                turns_elapsed=max(1, int(getattr(self, "_turns_since_sleep", 1) or 1)),
                base_rate=cfg.decay_base_rate,
                rehearsal_halflife=cfg.decay_rehearsal_halflife,
                min_confidence=cfg.decay_min_confidence))
        except Exception:
            return 0

    def _homeostatic_downscale(self, graph) -> Tuple[int, int]:
        """SHY-style homeostasis, plus the structural-importance -> EWC handoff.

        `homeostatic_downscale` computes topology-aware structural importance
        internally (with a cache) and protects structurally critical edges from
        the downscale. Afterwards that importance is pushed into
        `edge.fisher_importance`, so the EWC penalty inside `hebbian_update`
        actually protects the synapses the topology says matter. Without this
        handoff the importance score is computed and then discarded.

        Returns (edges_downscaled, si_edges_protected).
        """
        cfg = self.config
        weights_before = {k: float(e.weight) for k, e in graph.edges.items()}
        try:
            graph.homeostatic_downscale(
                protection_threshold=cfg.homeostatic_protection_threshold,
                downscale_factor=cfg.homeostatic_downscale_factor,
                structural_protection=cfg.homeostatic_structural_protection)
        except Exception:
            return 0, 0

        downscaled = sum(
            1 for k, e in graph.edges.items()
            if float(e.weight) < weights_before.get(k, 1.0) - 1e-9)

        protected = 0
        if cfg.si_to_ewc:
            si = getattr(graph, "_structural_importance_cache", None) or {}
            for (s, t), importance in si.items():
                edge = None
                try:
                    edge = graph.get_edge(s, t)
                except Exception:
                    edge = None
                if edge is None:
                    continue
                imp = float(importance)
                if imp > float(getattr(edge, "fisher_importance", 0.0) or 0.0):
                    try:
                        edge.fisher_importance = imp
                        protected += 1
                    except Exception:
                        pass
        return downscaled, protected

    def _split_concepts(self, graph, n_before: int) -> int:
        cfg = self.config
        max_total = int(max(n_before, 1) * cfg.split_max_concepts_factor)
        candidates = set(getattr(graph, "contradiction_hotspots", set()) or set())
        for nid, node in list(graph.nodes.items()):
            try:
                if (getattr(node, "drift_magnitude", 0.0) > cfg.split_drift_threshold
                        or getattr(node, "contradiction_count", 0)
                        >= cfg.split_contradiction_threshold):
                    candidates.add(nid)
            except Exception:
                continue

        splits = 0
        for nid in list(candidates):
            if splits >= cfg.max_splits_per_cycle:
                break
            if len(graph.nodes) >= max_total:
                break
            try:
                if graph.should_split(
                        nid,
                        contradiction_threshold=cfg.split_contradiction_threshold,
                        drift_threshold=cfg.split_drift_threshold):
                    graph.split_concept(nid)
                    splits += 1
            except Exception:
                continue
        return splits

    def _compress_paths(self, graph) -> int:
        cfg = self.config
        try:
            chains = graph.get_compressible_paths(min_usage=cfg.compress_min_usage)
            if not chains:
                return 0
            return int(graph.compress_paths(chains,
                                            min_chain_score=cfg.compress_min_score))
        except Exception:
            return 0

    @staticmethod
    def _mean_drift(graph) -> float:
        """Mean distance between active and core vectors (consolidation gap)."""
        try:
            drifts = [float(getattr(n, "drift_magnitude", 0.0) or 0.0)
                      for n in graph.nodes.values()]
        except Exception:
            return 0.0
        return float(np.mean(drifts)) if drifts else 0.0

    # ── serialisation ────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats)}

    def set_state(self, state: Dict[str, Any]) -> None:
        self.stats = dict(state.get("stats", self.stats) or self.stats)
