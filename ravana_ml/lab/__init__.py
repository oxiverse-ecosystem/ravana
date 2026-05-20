"""
Concept Physics Lab — controlled experiments for RLM cognition.

Measures internal geometry: free energy localization, attractor drift,
branch formation, sleep recovery dynamics.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .. import nn
from ..tensor import StateTensor
from ..graph import ConceptGraph


class ExperimentPhase:
    def __init__(self, name: str, transitions: List[Tuple[int, int]],
                 n_repeats: int = 20):
        self.name = name
        self.transitions = transitions
        self.n_repeats = n_repeats


class Probe:
    def __init__(self, input_id: int, expected: Optional[List[int]] = None,
                 label: str = ""):
        self.input_id = input_id
        self.expected = expected or []
        self.label = label or f"probe_{input_id}"


class Snapshot:
    def __init__(self, rlm, phase_name: str, label: str = ""):
        self.phase = phase_name
        self.label = label or phase_name
        self.conceptual_accuracy = float(rlm.conceptual_accuracy)
        self.total_free_energy = float(rlm.total_free_energy)
        self.n_edges = len(rlm.graph.edges)
        self.n_nodes = len(rlm.graph.nodes)
        self.free_energies = {
            'semantic': float(rlm.free_energy_engine.semantic_free_energy),
            'linguistic': float(rlm.free_energy_engine.linguistic_free_energy),
            'episodic': float(rlm.free_energy_engine.episodic_free_energy),
        }
        self.node_free_energies = {
            nid: float(node.prediction_free_energy) for nid, node in rlm.graph.nodes.items()
        }
        self.vectors = {
            nid: node.vector.copy() for nid, node in rlm.graph.nodes.items()
        }
        self.activations = {
            nid: float(node.activation) for nid, node in rlm.graph.nodes.items()
        }
        self.confidences = {
            nid: float(node.confidence) for nid, node in rlm.graph.nodes.items()
        }
        self.edges = {
            (s, t): {
                'weight': float(e.weight),
                'confidence': float(e.confidence),
                'stability': float(e.stability),
                'free_energy': float(e.prediction_free_energy),
            }
            for (s, t), e in rlm.graph.edges.items()
        }
        self._token_embeds = rlm.token_embed.weight.data.copy()


class ConceptLab:
    def __init__(self, rlm_config: dict, name: str = "experiment"):
        self.rlm = nn.RLM(**rlm_config)
        self.config = rlm_config
        self.name = name
        self.snapshots: List[Snapshot] = []
        self.phases_run: List[str] = []

    def run_phase(self, phase: ExperimentPhase, probe_every: int = 5):
        for _ in range(phase.n_repeats):
            for src, tgt in phase.transitions:
                inp = np.array([src], dtype=np.int64)
                nxt = np.array([tgt], dtype=np.int64)
                self.rlm.learn(inp, nxt)
        self.phases_run.append(phase.name)
        snap = Snapshot(self.rlm, phase.name,
                        f"after_{phase.name}")
        self.snapshots.append(snap)
        return snap

    def probe(self, input_id: int) -> dict:
        logits = self.rlm.forward(np.array([input_id], dtype=np.int64))
        pred_id = int(np.argmax(logits.data))
        probs = np.maximum(logits.data.flatten(), 1e-15)
        probs /= probs.sum()
        confidence = float(probs[pred_id])
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)) / np.log(len(probs)))
        return {
            'input': input_id,
            'predicted': pred_id,
            'confidence': confidence,
            'entropy': entropy,
            'predicted_concepts': self.rlm._last_predicted_concepts.copy(),
            'input_concepts': self.rlm._last_input_concepts.copy(),
        }

    def probe_with_context(self, context: List[int]) -> dict:
        logits = self.rlm.forward(np.array([context], dtype=np.int64))
        pred_id = int(np.argmax(logits.data))
        probs = np.maximum(logits.data.flatten(), 1e-15)
        probs /= probs.sum()
        return {
            'context': context,
            'predicted': pred_id,
            'confidence': float(probs[pred_id]),
            'predicted_concepts': self.rlm._last_predicted_concepts.copy(),
        }

    # ── Measurements ─────────────────────────────────────────────

    def free_energy_localization(self, snapshot_idx: int = -1) -> dict:
        snap = self.snapshots[snapshot_idx]
        pressures = np.array(list(snap.node_free_energies.values()))
        total = pressures.sum() + 1e-15
        probs = pressures / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
        max_entropy = float(np.log(len(pressures)))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        n_hotspots = int(np.sum(pressures > 0.1))
        return {
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': norm_entropy,
            'hotspots': n_hotspots,
            'total': float(total),
        }

    def attractor_drift(self, concept_id: int,
                        before_idx: int = 0,
                        after_idx: int = -1) -> dict:
        before = self.snapshots[before_idx]
        after = self.snapshots[after_idx]
        if concept_id not in after.vectors or concept_id not in before.vectors:
            return {'drift': None, 'exists': False}
        v_before = before.vectors[concept_id]
        v_after = after.vectors[concept_id]
        v_before_norm = v_before / (np.linalg.norm(v_before) + 1e-15)
        v_after_norm = v_after / (np.linalg.norm(v_after) + 1e-15)
        l2_dist = float(np.linalg.norm(v_after - v_before))
        cosine_sim = float(np.dot(v_before_norm, v_after_norm))
        return {
            'concept_id': concept_id,
            'l2_distance': l2_dist,
            'cosine_similarity': cosine_sim,
            'exists': True,
        }

    def attractor_preservation(self, concept_ids: Optional[List[int]] = None,
                               before_idx: int = 0,
                               after_idx: int = -1) -> dict:
        if concept_ids is None:
            concept_ids = sorted(self.snapshots[after_idx].vectors.keys())[:64]
        drifts = []
        for cid in concept_ids:
            d = self.attractor_drift(cid, before_idx, after_idx)
            if d['exists']:
                drifts.append(d['l2_distance'])
        return {
            'mean_drift': float(np.mean(drifts)) if drifts else None,
            'max_drift': float(np.max(drifts)) if drifts else None,
            'n_concepts': len(drifts),
        }

    def branch_detection(self, concept_id: int,
                         snapshot_idx: int = -1,
                         radius: float = 0.3) -> dict:
        snap = self.snapshots[snapshot_idx]
        if concept_id not in snap.vectors:
            return {'exists': False}
        center = snap.vectors[concept_id]
        center_norm = center / (np.linalg.norm(center) + 1e-15)
        neighbors = []
        for nid, vec in snap.vectors.items():
            if nid == concept_id:
                continue
            vec_norm = vec / (np.linalg.norm(vec) + 1e-15)
            sim = float(np.dot(center_norm, vec_norm))
            if sim > (1.0 - radius):
                neighbors.append({
                    'id': nid,
                    'similarity': sim,
                    'confidence': snap.confidences.get(nid, 0.0),
                    'activation': snap.activations.get(nid, 0.0),
                })
        neighbors.sort(key=lambda x: x['similarity'], reverse=True)
        return {
            'exists': True,
            'concept_id': concept_id,
            'n_neighbors': len(neighbors),
            'neighbors': neighbors[:20],
        }

    def sleep_efficiency(self) -> dict:
        free_energy_drops = []
        for i in range(len(self.snapshots) - 1):
            before = self.snapshots[i]
            after = self.snapshots[i + 1]
            drop = before.total_free_energy - after.total_free_energy
            free_energy_drops.append(drop)
        return {
            'mean_drop': float(np.mean(free_energy_drops)) if free_energy_drops else 0.0,
            'max_drop': float(np.max(free_energy_drops)) if free_energy_drops else 0.0,
            'n_intervals': len(free_energy_drops),
            'final_free_energy': self.snapshots[-1].total_free_energy,
        }

    def edge_topology_summary(self, snapshot_idx: int = -1) -> dict:
        snap = self.snapshots[snapshot_idx]
        weights = [e['weight'] for e in snap.edges.values()]
        confs = [e['confidence'] for e in snap.edges.values()]
        stabs = [e['stability'] for e in snap.edges.values()]
        return {
            'n_edges': len(snap.edges),
            'mean_weight': float(np.mean(weights)) if weights else 0.0,
            'mean_confidence': float(np.mean(confs)) if confs else 0.0,
            'mean_stability': float(np.mean(stabs)) if stabs else 0.0,
            'n_edges_weight_1': sum(1 for w in weights if w >= 0.99),
        }

    def edge_exists(self, src_concept: int, tgt_concept: int) -> bool:
        return self.rlm.graph.get_edge(src_concept, tgt_concept) is not None

    def token_concept_map(self, token_id: int) -> int:
        emb = self.rlm.token_embed(StateTensor(np.array([token_id]))).data[0]
        return self.rlm._nearest_concept(emb)

    def concept_token_map(self, concept_id: int) -> int:
        node = self.rlm.graph.get_node(concept_id)
        if node is None:
            return -1
        token_embeds = self.rlm.token_embed.weight.data
        token_norms = token_embeds / (np.linalg.norm(token_embeds, axis=1, keepdims=True) + 1e-15)
        vec_norm = node.vector / (np.linalg.norm(node.vector) + 1e-15)
        sims = token_norms @ vec_norm
        return int(np.argmax(sims))

    def report(self) -> str:
        lines = []
        lines.append(f"╔══ Concept Physics Lab Report: {self.name}")
        lines.append(f"║ RLM: {self.rlm}")
        lines.append(f"║ Phases run: {', '.join(self.phases_run)}")
        lines.append(f"║ Snapshots: {len(self.snapshots)}")
        lines.append(f"╠══")
        for i, snap in enumerate(self.snapshots):
            loc = self.free_energy_localization(i)
            lines.append(f"║ [{i}] {snap.label}")
            lines.append(f"║     accuracy={snap.conceptual_accuracy:.3f}  "
                         f"free_energy={snap.total_free_energy:.3f}  "
                         f"edges={snap.n_edges}")
            lines.append(f"║     free_energy: S={snap.free_energies['semantic']:.3f} "
                         f"L={snap.free_energies['linguistic']:.3f} "
                         f"E={snap.free_energies['episodic']:.3f}")
            lines.append(f"║     localization: entropy={loc['normalized_entropy']:.3f} "
                         f"hotspots={loc['hotspots']}")
        lines.append(f"╚══ End Report")
        return '\n'.join(lines)
