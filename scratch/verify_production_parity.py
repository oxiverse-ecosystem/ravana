#!/usr/bin/env python3
"""PARITY CHECK: replay all 800 frozen cases through the PRODUCTION module.

No engine, no store, no process_turn, no network. The production module must
reproduce every scratch pick byte-for-byte with the flag ON, and must emit
nothing with the flag OFF.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_k_items  # unmodified
from ravana.learn import lexical_superordinate as ls

assert "ravana/src" in ls.__file__.replace("\\", "/") or "ravana\\src" in ls.__file__
data = json.load(open(os.path.join(ROOT, "scratch", "lexical_superordinate_v2_results.json"), encoding="utf-8"))
bank = {it["item_id"]: it for it in tra_k_items.build_items()}
rows = data["rows"]
fallback = data["frozen"]["GLOBAL_Theta"]
print("production module:", ls.__file__)
print("flag default:", ls.enabled(), "wordnet available:", ls.wordnet_available())

# OFF arm: nothing may emit.
off_picks = [ls.pick(r["subject"], r["candidates"],
                     [(e["r1"], e["r2"]) for e in bank[r["item_id"]]["taught_exemplars"][:bank[r["item_id"]]["k"]]],
                     fallback) for r in rows]
assert all(p is None for p in off_picks), "flag-off emitted!"

os.environ[ls.ENV_FLAG] = "1"
on_picks, mismatches = [], []
for r in rows:
    it = bank[r["item_id"]]
    pairs = [(e["r1"], e["r2"]) for e in it["taught_exemplars"][:it["k"]]]
    p = ls.pick(r["subject"], r["candidates"], pairs, fallback)
    on_picks.append(p)
    if p != r["struct"]:
        mismatches.append((r["item_id"], r["subject"], p, r["struct"]))
print("rows", len(rows), "flag-off emits", sum(p is not None for p in off_picks),
      "flag-on emits", sum(p is not None for p in on_picks),
      "scratch emits", sum(r["struct"] is not None for r in rows),
      "mismatches", len(mismatches))
if mismatches:
    print("first mismatches:", mismatches[:10])
graded = [r for r, p in zip(rows, on_picks) if r["key"] is not None and p is not None]
correct = sum(1 for r, p in zip(rows, on_picks) if r["key"] is not None and p == r["key"])
print("production precision on graded emits:", round(correct / len(graded), 4),
      "n =", len(graded))
assert not mismatches, "production/scratch parity failed"
print("PARITY: PASS")
