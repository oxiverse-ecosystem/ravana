"""BrainCore SHA-256 bit-exact reproducibility test.

Pins the determinism contract: same seed + same inputs (RAVANA_OFFLINE=1)
must produce the same spike-log + graph + engine fingerprint, every time,
on every machine. Two back-to-back runs in the SAME process must agree.

This is the CI gate that catches non-determinism regressions.
"""
import hashlib
import os
import sys

# CRITICAL: lock hash seed BEFORE importing numpy/ravana
os.environ["PYTHONHASHSEED"] = "0"
os.environ["RAVANA_OFFLINE"] = "1"

import pytest

pytestmark = pytest.mark.ci


# Insert paths so we can import the engine in-process
_PROJ = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
for _p in (
    _PROJ,
    os.path.join(_PROJ, "ravana", "src"),
    os.path.join(_PROJ, "ravana_ml", "src"),
    os.path.join(_PROJ, "ravana-v2", "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.reproducibility import (
    SpikeLog,
    full_reproducibility_fingerprint,
    graph_state_fingerprint,
    engine_fingerprint,
)


def _build_engine(seed: int = 42) -> CognitiveChatEngine:
    """Build a fresh engine with a fixed seed, isolated suffix."""
    eng = CognitiveChatEngine(
        dim=64, seed=seed, baby_mode=True, user_suffix="repro_test"
    )
    return eng


# A fixed probe sequence — enough to exercise PE, concept activation, identity
_PROBE_INPUTS = [
    "hello",
    "what is your name",
    "i like open source software",
    "do you think privacy is important",
    "my cat is named milo",
    "what is my cat's name",
    "the ocean is beautiful",
    "do you prefer the sea or the mountains",
]


def _run_probe_sequence(eng: CognitiveChatEngine) -> SpikeLog:
    """Run the probe sequence. The engine records spikes automatically."""
    for q in _PROBE_INPUTS:
        eng.process_turn(q)
    return eng.spike_log


def _clean_suffix_files(user_suffix: str = "repro_test"):
    """Remove any leftover save files so each run starts from a clean seed."""
    import glob
    proj = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    weights_dir = os.path.join(proj, "weights")
    patterns = [
        os.path.join(weights_dir, f"ravana_weights{user_suffix}*.pkl"),
        os.path.join(weights_dir, f"ravana_usermodel{user_suffix}*.pkl"),
        os.path.join(weights_dir, f"ravana_weights{user_suffix}*.db"),
        os.path.join(weights_dir, f"ravana_weights{user_suffix}*.sha"),
    ]
    for pat in patterns:
        for fpath in glob.glob(pat):
            try:
                os.remove(fpath)
            except OSError:
                pass


def _two_run_fingerprint(seed: int = 42):
    """Run the probe sequence twice and return both fingerprints."""
    results = []
    for _ in range(2):
        _clean_suffix_files("repro_test")
        eng = _build_engine(seed=seed)
        spike_log = _run_probe_sequence(eng)
        fp = full_reproducibility_fingerprint(eng, spike_log)
        results.append(fp)
    _clean_suffix_files("repro_test")
    return results[0], results[1]


class TestReproducibility:
    """BrainCore SHA-256 bit-exact reproducibility gate."""

    def setup_method(self):
        _clean_suffix_files("repro_test")

    def teardown_method(self):
        _clean_suffix_files("repro_test")

    def test_same_seed_same_fingerprint(self):
        """Two runs with the SAME seed must produce the SAME fingerprint."""
        fp_a, fp_b = _two_run_fingerprint(seed=42)
        assert fp_a["combined_sha256"] == fp_b["combined_sha256"], (
            f"combined_sha256 mismatch:\n  A: {fp_a}\n  B: {fp_b}"
        )
        assert fp_a["spike_log_sha256"] == fp_b["spike_log_sha256"]
        assert fp_a["graph_sha256"] == fp_b["graph_sha256"]
        assert fp_a["engine_sha256"] == fp_b["engine_sha256"]

    def test_different_seed_different_fingerprint(self):
        """Different seeds SHOULD produce different fingerprints."""
        eng_a = _build_engine(seed=42)
        spike_a = _run_probe_sequence(eng_a)
        fp_a = full_reproducibility_fingerprint(eng_a, spike_a)

        eng_b = _build_engine(seed=99)
        spike_b = _run_probe_sequence(eng_b)
        fp_b = full_reproducibility_fingerprint(eng_b, spike_b)

        assert fp_a["combined_sha256"] != fp_b["combined_sha256"], (
            "Different seeds should produce different fingerprints"
        )

    def test_spike_log_is_ordered(self):
        """Spike log entries must be in turn order (append-only)."""
        eng = _build_engine(seed=42)
        spike_log = _run_probe_sequence(eng)
        turns = [e["turn"] for e in spike_log.entries]
        assert turns == sorted(turns), "spike log entries must be in turn order"
        assert len(spike_log) == len(_PROBE_INPUTS)  # one spike per turn

    def test_fingerprint_includes_turn_count(self):
        """Fingerprint must carry the turn count for auditability."""
        eng = _build_engine(seed=42)
        spike_log = _run_probe_sequence(eng)
        fp = full_reproducibility_fingerprint(eng, spike_log)
        assert int(fp["turn_count"]) == len(_PROBE_INPUTS)

    def test_rng_state_persists(self):
        """Engine RNG state must be deterministic and persisted."""
        eng = _build_engine(seed=42)
        state_before = eng.rng.get_state()
        _run_probe_sequence(eng)
        state_after_run = eng.rng.get_state()
        assert repr(state_before) != repr(state_after_run)

        eng2 = _build_engine(seed=42)
        assert repr(eng2.rng.get_state()) == repr(state_before)

    def test_spike_log_sha_stable_across_repr(self):
        """The spike-log SHA must be identical whether computed before or after
        a repr round-trip (catches float repr noise)."""
        eng = _build_engine(seed=42)
        spike_log = _run_probe_sequence(eng)
        h1 = spike_log.sha256()
        entries_repr = repr(spike_log.to_hashable())
        h2 = hashlib.sha256(entries_repr.encode("utf-8")).hexdigest()
        assert h1 == h2
