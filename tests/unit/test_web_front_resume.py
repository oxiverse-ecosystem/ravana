"""Contract tests for the resumable web-front harnesses.

The live arm died at 195/434 probes because its log was opened with "w", losing
all completed work. Both harnesses now support --resume with a state_id key
(commit + weight-pkl mtime) so stale rows from a different state are ignored.
These tests check the contract without booting an engine or using the network.
"""
from __future__ import annotations

import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_both_harnesses_declare_resume_contract():
    for script in ("measure_fronts_web.py", "measure_fronts_web_frozen.py"):
        src = open(os.path.join(PROJ, "scripts", script), encoding="utf-8").read()
        assert 'RESUME_MODE = "--resume" in sys.argv' in src
        assert '"state_id": _STATE_ID' in src
        assert 'if _r.get("state_id") == _STATE_ID' in src
        assert ('("a" if RESUME_MODE else "w")' in src
                or '_log_mode = "a" if RESUME_MODE else "w"' in src)
        assert "_RESUME_ROWS.get((name, i))" in src


def test_resume_cache_filters_by_state_id(tmp_path):
    # Replicate the harness's cache-loading contract on a synthetic log: only
    # rows tagged with the CURRENT state_id are reusable; control rows and rows
    # from another code/weight state are excluded.
    log = tmp_path / "probes.jsonl"
    state = "abc123:1700000000"
    rows = [
        {"front": "grounded factual answers", "probe_index": 1,
         "state_id": state, "response": "a photon is light", "passed": True,
         "reason": "ok", "control": False},
        {"front": "grounded factual answers", "probe_index": 2,
         "state_id": "old999:1600000000", "response": "stale", "passed": False,
         "reason": "old", "control": False},
        {"front": "x", "probe_index": 3, "state_id": state,
         "response": "control", "passed": True, "reason": "c", "control": True},
    ]
    with open(log, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    cache = {}
    with open(log, encoding="utf-8") as rf:
        for line in rf:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("state_id") == state and not r.get("control"):
                cache[(r.get("front"), r.get("probe_index"))] = r
    assert list(cache) == [("grounded factual answers", 1)]
    assert cache[("grounded factual answers", 1)]["response"] == "a photon is light"


def test_live_harness_imports_with_gates_only():
    # --gates-only runs no engine and no network; it must import cleanly with
    # the resume machinery present (guard against module-level syntax/state bugs).
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(PROJ, "scripts", "measure_fronts_web.py"),
         "--gates-only"],
        cwd=PROJ, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": os.path.join(PROJ, "scripts"),
             "RAVANA_OFFLINE": "1", "PYTHONHASHSEED": "0"},
    )
    assert out.returncode in (0, 1), out.stderr[-500:]  # 1 = gate failure, not import error
    assert "Traceback" not in out.stderr
