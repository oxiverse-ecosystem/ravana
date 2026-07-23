#!/usr/bin/env python3
"""Phase 0 smoke test: (1) LoCoMo loader now yields real dialogue turns,
(2) interrogatives are no longer ingested as episodic facts (echo bug)."""
import os, sys, json, re
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, os.path.join(_proj_root, "ravana_ml", "src"))
os.environ["RAVANA_SILENT"] = "1"

# ── Test 1: loader shape ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(_proj_root, "scripts"))
import importlib
ev = importlib.import_module("evaluate_ravana")
cases = ev._load_locoMo(max_cases=15)
p0 = cases[0]["primer"]
assert len(p0) > 100, f"expected >100 primer turns, got {len(p0)}"
assert not any(t in ("speaker_a", "speaker_b", "session_1") for t in p0), \
    "primer still contains dict keys!"
assert p0[0].startswith("(Session 1, dated"), p0[0]
assert any(":" in t for t in p0[1:5]), "no speaker-prefixed turns"
# later cases of same dialogue must have empty primer
assert cases[1]["primer"] == [], "case 2 should not re-feed the primer"
print(f"TEST 1 PASS: primer[0]={p0[0]!r}, turns={len(p0)}, case2 primer empty")

# ── Test 2: echo-bug fix on a live engine ───────────────────────────────
engine = ev.restore_from_snapshot()
stmts = [
    "(Session 1, dated 1:00 pm on 8 May, 2023)",
    "I got my new car serviced last week and the GPS system stopped working right after.",
    "My sister Anna adopted a beagle puppy named Rusty.",
]
for s in stmts:
    engine.process_turn(s)
q1 = "What was the first issue I had with my new car after its first service?"
r1 = engine.process_turn(q1)
q2 = "Which vehicle did I take care of first in February, the bike or the car?"
r2 = engine.process_turn(q2)
q3 = "What breed is Anna's puppy?"
r3 = engine.process_turn(q3)
print("Q1:", r1[:140])
print("Q2:", r2[:140])
print("Q3:", r3[:140])
for tag, q, r in (("Q2", q1, r2), ("Q3", q2, r3)):
    lo = (r or "").lower()
    assert not (lo.startswith("you told me earlier") and
                any(w in lo for w in ("what was the first issue",
                                       "which vehicle did"))), \
        f"{tag} STILL echoes a previous question: {r[:120]}"
print("TEST 2 PASS: no previous-question echo")
