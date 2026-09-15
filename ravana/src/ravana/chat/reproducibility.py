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
        # Normalize floats to round to 8 decimals so the hash is stable across
        # platforms (IEEE 754 guarantees same result for same ops, but guard
        # against any representation noise in the repr).
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
            "ts": round(time.time(), 6),  # wall-clock, informational only
            "data": clean,
        })

    def to_hashable(self) -> List[Tuple]:
        """Return a deterministic, order-preserving representation for hashing."""
        out: List[Tuple] = []
        for e in self.entries:
            # Sort the data dict so key order doesn't matter per-entry, but
            # the entries themselves stay in append order.
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


def graph_state_fingerprint(graph) -> str:
    """SHA-256 of the concept graph's cognitive state.

    Includes: node count, edge count, total_free_energy, and per-node
    (label, free_energy, stability) — deterministic structural signals
    only, NOT activation timestamps or history buffers which depend on
    wall-clock time, and NOT vector values (which depend on global numpy
    RNG state at graph construction time, not the engine seed).
    """
    # Nodes: sorted by label for determinism
    node_sigs = []
    for nid in sorted(graph.nodes.keys()):
        n = graph.nodes[nid]
        fe = round(float(getattr(n, "prediction_free_energy", 0.0)), 8)
        stab = round(float(getattr(n, "stability", 0.5)), 8)
        conf = round(float(getattr(n, "confidence", 0.1)), 8)
        node_sigs.append((n.label, fe, stab, conf))

    # Edges: sorted by (source, target) for determinism
    edge_sigs = []
    for (s, t) in sorted(graph.edges.keys()):
        e = graph.edges[(s, t)]
        w = round(float(getattr(e, "weight", 0.5)), 8) if hasattr(e, "weight") else round(float(e[1]), 8)
        edge_sigs.append((s, t, w))

    total_fe = round(float(getattr(graph, "total_free_energy", 0.0)), 8)

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

    Excludes: wall-clock timestamps, pickle-incompatible objects, and the
    graph (which has its own fingerprint via graph_state_fingerprint).
    """
    signals = {
        "turn_count": int(getattr(engine, "turn_count", 0)),
        "learning_count": int(getattr(engine, "_learning_count", 0)),
        "sleep_cycles": int(getattr(engine, "sleep_cycles_completed", 0)),
        "free_energy": round(float(getattr(engine, "_free_energy", 0.0)), 8),
        "mean_pe": round(float(getattr(engine, "_mean_prediction_error", 0.0)), 8),
        "pe_count": int(getattr(engine, "_prediction_error_count", 0)),
        "rng_state_hash": hashlib.sha256(
            repr(engine.rng.get_state()).encode("utf-8")
        ).hexdigest()[:32],
        "td_error_history": tuple(
            round(float(x), 8) for x in getattr(engine, "_td_error_history", [])
        ),
        "identity": _identity_fingerprint(engine),
    }
    blob = repr(sorted(signals.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _identity_fingerprint(engine) -> Tuple:
    """Hashable identity state."""
    ident = getattr(engine, "identity", None)
    if ident is None:
        return ()
    # IdentityState is a dataclass — use getattr not dict.get
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
    BrainCore SHA-256 standard: same inputs → same dict, always.
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
