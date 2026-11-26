"""Regression: indexed activation must match the original full-graph scan.

The WFA stall was caused by `_activate_from_input` scanning every graph node
for every token. The indexed replacement must preserve the exact scores and
activation order on all label-match branches.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from ravana.chat.constants import STOP_WORDS
from ravana.chat.engine_memory import MemoryMixin


class _Graph:
    def __init__(self, labels):
        self.nodes = {
            i: SimpleNamespace(label=label, vector=None)
            for i, label in enumerate(labels)
        }
        self.activations = []

    def get_node(self, nid):
        return self.nodes.get(nid)

    def get_outgoing(self, nid):
        return []

    def activate(self, nid, strength):
        self.activations.append((nid, strength))


class _Host:
    def __init__(self, labels):
        self.graph = _Graph(labels)
        self.dim = 4
        self._concept_keywords = {}
        self._concept_pos = {}


def _reference(host, text):
    """Verbatim pre-index label activation logic (for equivalence only)."""
    import re
    words = re.findall(r"[a-zA-Z']{1,}", text.lower())
    scores = {}
    acc_ctx = np.zeros(host.dim, dtype=np.float32)
    word_count = 0
    for w in words:
        if w in STOP_WORDS and word_count > 0:
            continue
        w_nids = host._concept_keywords.get(w, [])
        w_vec = None
        if w_nids:
            node = host.graph.get_node(w_nids[0])
            if node and node.vector is not None:
                w_vec = node.vector
        n400_surprise = 0.5
        if w_vec is not None and word_count > 0:
            n_acc = np.linalg.norm(acc_ctx)
            n_w = np.linalg.norm(w_vec)
            if n_acc > 1e-8 and n_w > 1e-8:
                cos_sim = float(np.dot(acc_ctx, w_vec)) / (n_acc * n_w)
                n400_surprise = 1.0 - max(0.0, min(1.0, cos_sim))
        elif word_count == 0:
            n400_surprise = 0.8
        if w_nids:
            for nid in w_nids:
                node = host.graph.nodes.get(nid)
                pos_boost = 0.5 if node and node.label and host._concept_pos.get(node.label.lower()) == 'noun' else 0.0
                base = 5.0 + pos_boost
                n400_boost = 1.0 + n400_surprise * 2.0
                scores[nid] = scores.get(nid, 0) + base * n400_boost
        for nid, node in list(host.graph.nodes.items()):
            if not node or not node.label:
                continue
            label = node.label.lower()
            s = scores.get(nid, 0.0)
            if label == w:
                s += 5.0 * (1.0 + n400_surprise)
            elif len(w) >= 3 and (label == w or label.startswith(w + " ") or (" " + w + " ") in label or label.endswith(" " + w)):
                s += 3.0
            elif len(label) >= 3 and label in w:
                s += 2.0
            if s > 0:
                scores[nid] = s
        if w_vec is not None:
            integration_gate = 0.5 + 0.3 * (1.0 - n400_surprise)
            if word_count == 0:
                acc_ctx = w_vec.copy()
            else:
                acc_ctx = integration_gate * w_vec + (1.0 - integration_gate) * acc_ctx
            n = np.linalg.norm(acc_ctx)
            if n > 0:
                acc_ctx /= n
        word_count += 1
        if w_nids:
            for src_nid in w_nids:
                for tgt_id, edge in host.graph.get_outgoing(src_nid):
                    if tgt_id not in scores:
                        prop_strength = 2.0 * (1.0 - n400_surprise * 0.5)
                        scores[tgt_id] = scores.get(tgt_id, 0) + edge.weight * prop_strength
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    activations = [(nid, min(1.0, sc * 0.12)) for nid, sc in ordered[:8]]
    return [nid for nid, _ in activations], activations


@pytest.mark.parametrize("text", [
    "water vapor and cold water",
    "https://www.wikidata.org/wiki/Q1052 natural phenomenon",
    "waterfall dew frost hail mist",
    "endorse a claim only if its source is discredited",
])
def test_indexed_activation_matches_full_scan(text):
    labels = [
        "water", "waterfall", "cold water", "water vapor", "dew", "frost",
        "hail", "mist", "natural phenomenon", "source", "claim",
        "https", "www.wikidata.org", "wiki", "q1052", "discredited",
    ]
    new_host = _Host(labels)
    got = MemoryMixin._activate_from_input(new_host, text)

    old_host = _Host(labels)
    expected_ids, expected_activations = _reference(old_host, text)

    assert got == expected_ids
    assert new_host.graph.activations == expected_activations
