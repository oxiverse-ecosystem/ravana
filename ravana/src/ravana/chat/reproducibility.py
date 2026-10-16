"""Bit-exact reproducibility for RAVANA.

Implements BrainCore's SHA-256 standard: same inputs must produce same
outputs, every time, on every machine. This module captures a deterministic
spike log and a cryptographic fingerprint of the cognitive state at key
checkpoints, then exposes them for CI gating.

Determinism contract:
- RNG is seeded (np.random.RandomState(seed)) and persisted/restored.
- RAVANA_OFFLINE=1 disables web learning (non-deterministic).
- PYTHONHASHSEED-stable hashing (order-independent structural fingerprints).
- All learned state that feeds the spike log is derived from the seeded RNG
  + the input sequence.
"""
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple


class SpikeLog:
    """Deterministic record of cognitive spikes per turn.

    A "spike" is a discrete event in the cognitive trace: a prediction error
    above threshold, a concept activation, a free-energy surge. The log is
    append-only and ordered, so it is hashable for reproducibility.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def record(self, turn: int, kind: str, data: Dict[str, Any]) -> None:
        """Append a spike event. `kind` is one of 'pe', 'activation', 'split', 'surge'."""
        clean: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, float):
                clean[k] = round(v, 8)
            elif isinstance(v, dict):
                clean[k] = {kk: round(vv, 8) if isinstance(vv, float) else vv
                            for kk, vv in v.items()}
            else:
                clean[k] = v
        self.entries.append({
            "turn": turn,
            "kind": kind,
            "ts": round(time.time(), 6),  # wall-clock, informational only — EXCLUDED from hash
            "data": clean,
        })

    def to_hashable(self) -> List[Tuple]:
        """Return a deterministic, order-preserving representation for hashing.

        CRITICAL: excludes the `ts` (wall-clock timestamp) field — only the
        strategy + cognitive signals are hashed, so the fingerprint is
        reproducible across runs/machines regardless of when they execute.
        """
        out: List[Tuple] = []
        for e in self.entries:
            data_items = tuple(sorted(
                (k, tuple(v.items()) if isinstance(v, dict) else v)
                for k, v in e["data"].items()
            ))
            out.append((e["turn"], e["kind"], data_items))
        return out

    def sha256(self) -> str:
        """SHA-256 of the full spike log. Deterministic across processes."""
        blob = repr(self.to_hashable()).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def __len__(self) -> int:
        return len(self.entries)


def deterministic_total_free_energy(graph) -> float:
    """Order-independent total free energy for determinism."""
    return round(sum(
        float(getattr(graph.nodes[nid], "prediction_free_energy", 0.0))
        for nid in sorted(graph.nodes)
    ), 8)


def graph_state_fingerprint(graph) -> str:
    """SHA-256 of the concept graph's cognitive state.

    Only structural signals that are deterministic for a given seed+input:
    node/edge counts, per-node (label, vector_sum, free_energy, stability,
    confidence), per-edge (source, target, weight).

    EXCLUDED (non-deterministic): timestamps, activation_history, last_activated,
    fatigue, contradiction_count, quarantine_log, faiss index state.
    """
    node_sigs = []
    for nid in sorted(graph.nodes.keys()):
        n = graph.nodes[nid]
        vec_sum = round(float(n.vector.sum()), 8) if n.vector is not None else 0.0
        fe = round(float(getattr(n, "prediction_free_energy", 0.0)), 8)
        stab = round(float(getattr(n, "stability", 0.5)), 8)
        conf = round(float(getattr(n, "confidence", 0.1)), 8)
        node_sigs.append((n.label, vec_sum, fe, stab, conf))

    edge_sigs = []
    for (s, t) in sorted(graph.edges.keys()):
        e = graph.edges[(s, t)]
        w = round(float(getattr(e, "weight", 0.5)), 8) if hasattr(e, "weight") else round(float(e[1]), 8)
        edge_sigs.append((s, t, w))

    total_fe = deterministic_total_free_energy(graph)

    structure = {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "total_free_energy": total_fe,
        "nodes": tuple(node_sigs),
        "edges": tuple(edge_sigs),
    }
    blob = repr(structure).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def engine_fingerprint(engine) -> str:
    """SHA-256 of the engine's deterministic state dimensions.

    Excludes: wall-clock timestamps, pickle-incompatible objects, the graph
    (which has its own fingerprint), and RNG state (which can vary between
    runs due to non-deterministic background thread scheduling even when
    outputs are identical — the spike log captures the real reproducibility
    signal).
    """
    signals = {
        "turn_count": int(getattr(engine, "turn_count", 0)),
        "learning_count": int(getattr(engine, "_learning_count", 0)),
        "sleep_cycles": int(getattr(engine, "sleep_cycles_completed", 0)),
        "free_energy": round(float(getattr(engine, "_free_energy", 0.0)), 8),
        "mean_pe": round(float(getattr(engine, "_mean_prediction_error", 0.0)), 8),
        "pe_count": int(getattr(engine, "_prediction_error_count", 0)),
        "identity": _identity_fingerprint(engine),
    }
    blob = repr(sorted(signals.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _identity_fingerprint(engine) -> Tuple:
    """Hashable identity state."""
    ident = getattr(engine, "identity", None)
    if ident is None:
        return ()
    st = ident.state if hasattr(ident, "state") else None
    if st is None:
        return ()
    return (
        round(float(getattr(st, "strength", 0.0)), 8),
        round(float(getattr(st, "momentum", 0.0)), 8),
        round(float(getattr(st, "stability", 0.0)), 8),
        round(float(getattr(st, "trend", 0.0)), 8),
    )


def full_reproducibility_fingerprint(
    engine,
    spike_log: Optional[SpikeLog] = None,
) -> Dict[str, str]:
    """Composite fingerprint: spike log + graph + engine state.

    Returns a dict with component hashes and a combined hash. This is the
    BrainCore SHA-256 standard: same inputs -> same dict, always.
    """
    spike_hash = spike_log.sha256() if spike_log is not None else "n/a"
    graph_hash = graph_state_fingerprint(engine.graph)
    eng_hash = engine_fingerprint(engine)

    combined = hashlib.sha256(
        f"{spike_hash}|{graph_hash}|{eng_hash}".encode("utf-8")
    ).hexdigest()

    return {
        "spike_log_sha256": spike_hash,
        "graph_sha256": graph_hash,
        "engine_sha256": eng_hash,
        "combined_sha256": combined,
        "turn_count": str(getattr(engine, "turn_count", 0)),
        "schema_version": str(getattr(engine, "SAVE_SCHEMA_VERSION", "unknown")),
    }
