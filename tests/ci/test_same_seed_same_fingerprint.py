#!/usr/bin/env python3
"""Cross-process determinism test: same seed → same fingerprint.

Runs the probe sequence in two ISOLATED subprocesses with identical seeds
and asserts the composite fingerprint matches bit-for-bit.

Uses time.time() mocking (via _repro_subprocess_helper) to ensure graph
timestamps, decay calculations, and FE values are reproducible across runs.
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))


def _run_subprocess_helper(seed: int) -> dict:
    """Run _repro_subprocess_helper.py in an isolated process and parse its JSON output."""
    import subprocess

    helper = os.path.join(_HERE, "_repro_subprocess_helper.py")
    helper = os.path.normpath(helper)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["RAVANA_OFFLINE"] = "1"

    result = subprocess.run(
        [sys.executable, helper, str(seed)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=_REPO_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess helper failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Parse the last line of stdout as JSON
    lines = result.stdout.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)

    raise RuntimeError(f"No JSON fingerprint found in subprocess output:\n{result.stdout}")


@pytest.mark.ci
def test_same_seed_same_fingerprint():
    """Two isolated runs with seed=42 must produce identical fingerprints."""
    fp1 = _run_subprocess_helper(seed=42)
    fp2 = _run_subprocess_helper(seed=42)

    assert fp1["combined_sha256"] == fp2["combined_sha256"], (
        f"Fingerprint mismatch:\n  run1: {fp1['combined_sha256']}\n  run2: {fp2['combined_sha256']}"
    )
    assert fp1["graph_sha256"] == fp2["graph_sha256"], "Graph fingerprint mismatch"
    assert fp1["engine_sha256"] == fp2["engine_sha256"], "Engine fingerprint mismatch"
    assert fp1["spike_log_sha256"] == fp2["spike_log_sha256"], "Spike log fingerprint mismatch"

    # Smoke: all hashes are non-empty hex strings
    for key in ("combined_sha256", "graph_sha256", "engine_sha256", "spike_log_sha256"):
        assert fp1[key] and len(fp1[key]) == 64, f"{key} is not a 64-char hex string: {fp1[key]}"

    # Turn count should match across runs
    assert fp1["turn_count"] == fp2["turn_count"], "Turn count mismatch"


@pytest.mark.ci
def test_different_seed_different_fingerprint():
    """Different seeds must produce different combined fingerprints."""
    fp1 = _run_subprocess_helper(seed=42)
    fp2 = _run_subprocess_helper(seed=99)

    assert fp1["combined_sha256"] != fp2["combined_sha256"], (
        "Different seeds produced the same combined fingerprint — "
        "determinism gate may be too weak"
    )
