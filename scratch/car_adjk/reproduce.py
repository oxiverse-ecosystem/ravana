#!/usr/bin/env python3
"""CAR sealed adjacent-k failure — engine-free scratch reproduction and fix test.

SCRATCH ONLY. Nothing here boots CognitiveChatEngine, touches production source,
changes a gate/threshold, or writes the live weight store.

What is imported UNMODIFIED from production:
  * ravana_ml.graph.ConceptGraph / ConceptGraph.remove_node / add_edge
  * ravana.web.web_to_graph.WebToGraph._ensure_node   (write-time liveness guard, F2)
  * ravana.learn.explanation.enumerate_chains / grade_chain / assertable_chain
    (the exact gate the CAR harness uses)

What is replayed locally (recorded state transition, not a new theory):
  * a low-degree taught node is absent from graph.nodes while its id remains in
    the engine label index -- the measured post-sleep state in the P2 run;
  * the prune-side guard (F4) that would have prevented that state.

Recorded cases: the ten sealed k=2 items of commit 20aca05. Seven had
raw_chains=0 in-process; three (hiccup, voice, pearl) acquired.
See scratch/car_adjk/FINDING.md.
"""
from __future__ import annotations

import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["RAVANA_OFFLINE"] = "1"

import numpy as np
from ravana_ml.graph import ConceptGraph
from ravana.web.web_to_graph import WebToGraph
from ravana.learn.explanation import enumerate_chains, grade_chain, assertable_chain

LIMIT = 64  # TRAVERSAL_LIMIT of scripts/measure_learn5_car.py

# Replayed from scratch/learn5_car_measured_PRE_LIVENESS.json (commit 20aca05):
# sealed k=2 immediate raw_chains == 0 for these seven, >0 for the other three.
ITEMS = {
    "boil":    ("heating",      "bubbles",       "heating"),
    "bruise":  ("impact",       "rupture",       "impact"),
    "cocoon":  ("spinning",     "silkworm",      "spinning"),
    "echo":    ("reflection",   "repetition",    "echo"),
    "snore":   ("vibration",    "breath",        "snore"),
    "tsunami": ("displacement", "wave",          "displacement"),
    "webbing": ("spinning",     "thread",        "spinning"),
    "hiccup":  ("spasm",        "interruption",  None),
    "voice":   ("vibration",    "sound",         None),
    "pearl":   ("irritation",   "secretion",     None),
}
FAILING = [s for s, v in ITEMS.items() if v[2] is not None]
SUCCESS = [s for s, v in ITEMS.items() if v[2] is None]

ARMS = ("A0", "F2", "F4", "F2F4")


class _GE:
    """Minimal host shim; same fields the real engine exposes to WebToGraph."""
    def __init__(self, graph):
        self.graph = graph
        self._all_labels = {}
        self._concept_keywords = {}
        self._provisional_nodes = {}
        self._promote_min_sources = 1
        self._junk_theta = 2.0          # admit the recorded concept tokens


def _wt(ge):
    w = WebToGraph.__new__(WebToGraph)
    w.ge = ge
    return w


def _ensure_fixed(ge, label, source_url):
    """Production write choke point: live-id check + stale-index cleanup."""
    return _wt(ge)._ensure_node(label, source_url=source_url)


def _ensure_prefix(ge, label):
    """Historical pre-fix resolution: trust _all_labels unconditionally.

    This is the one-line behaviour removed by the F2 guard at
    ravana/src/ravana/web/web_to_graph.py:152-163.
    """
    return ge._all_labels.get(label)


def _write_chain(g, ids, source):
    """Teach a->b->x as two trusted curated causal hops (as teach_web does)."""
    a, b, x = ids
    e1 = g.add_edge(a, b, weight=0.5, relation_type="causal", confidence=0.5)
    e2 = g.add_edge(b, x, weight=0.5, relation_type="causal", confidence=0.5)
    for e in (e1, e2):
        if e is not None:
            e.source_metadata = {"edge_kind": "curated", "source": source}
    return e1, e2


def _dangling(g):
    return [(s, t) for (s, t) in g.edges
            if s not in g.nodes or t not in g.nodes]


def _trusted_incident(g, nid):
    """Prune-side guard F4, mirroring graph.py:1556-1566: trust is a property
    of the stored provenance, not of the hop's grammatical relation."""
    for (s, t), e in g.edges.items():
        if s != nid and t != nid:
            continue
        md = getattr(e, "source_metadata", None)
        if isinstance(md, dict) and md.get("edge_kind") in (
                "curated", "web_fact", "definition", "curriculum"):
            return True
    return False


def label_map(g):
    """Verbatim from scripts/measure_learning_curve.py:131-141 (live nodes only)."""
    out = {}
    for nid, node in g.nodes.items():
        lab = (getattr(node, "label", "") or "").strip().lower()
        if lab:
            out.setdefault(lab, []).append(nid)
    return out


def node_ids_for(spec, g, lm=None):
    """Verbatim from scripts/measure_learning_curve.py:143-147."""
    lm = lm if lm is not None else label_map(g)
    return list(lm.get(spec["x"].lower(), []))


def gate_census(nids, g):
    """Same computation as scripts/measure_learn5_car.py:295-323."""
    out = {"raw_chains": 0, "graded_assert": 0, "trusted_all_hops": 0,
           "rejected_by_trust": 0, "search_limit": LIMIT}
    if not nids:
        return out
    chains = enumerate_chains(nids, g, direction="backward", min_steps=2,
                              max_steps=4, limit=LIMIT)
    out["raw_chains"] = len(chains)
    for ch in chains:
        gr = grade_chain(ch)
        trusted = assertable_chain(ch)
        if gr.get("verdict") == "assert":
            out["graded_assert"] += 1
        if trusted:
            out["trusted_all_hops"] += 1
        else:
            out["rejected_by_trust"] += 1
    return out


def run_item(subject, arm, inject_untrusted=False, oracle=False):
    a, b, dead_label = ITEMS[subject]
    x = subject
    g = ConceptGraph(dim=8)
    ge = _GE(g)
    spec = {"x": subject}

    # ---- exposure 1: write the chain on fresh, live nodes ----
    ids1 = [_ensure_fixed(ge, lab, "http://teach.local/%s/sourceA" % subject)
            for lab in (a, b, x)]
    _write_chain(g, ids1, "http://teach.local/%s/sourceA" % subject)
    first_ids = {lab: nid for lab, nid in zip((a, b, x), ids1)}

    # ---- sleep junk-prune: the recorded state transition ----
    pruned = False
    if dead_label is not None:
        dead_id = ge._all_labels[dead_label]
        guard = arm in ("F4", "F2F4")
        protected = guard and _trusted_incident(g, dead_id)
        if not protected:
            g.remove_node(dead_id)          # index is NOT synced (pre-fix)
            pruned = True
    state_after_prune = {
        "dead_label": dead_label,
        "dead_id": (ge._all_labels.get(dead_label) if dead_label else None),
        "dead_id_live": (ge._all_labels.get(dead_label) in g.nodes
                         if dead_label else None),
        "pruned": pruned,
    }

    if arm == "A0":
        ids2 = [_ensure_prefix(ge, lab) for lab in (a, b, x)]
    else:                                   # F2 / F4 / F2F4 use production F2
        ids2 = [_ensure_fixed(ge, lab, "http://teach.local/%s/sourceB" % subject)
                for lab in (a, b, x)]

    minted = {}
    for lab, old, new in zip((a, b, x), ids1, ids2):
        if new is not None and new != old:
            minted[lab] = new

    # ---- exposure 2: re-teach over the resolved endpoints ----
    _write_chain(g, ids2, "http://teach.local/%s/sourceB" % subject)

    if inject_untrusted:
        # A high-weight UNtrusted competitor with the same shape. It must be
        # rejected by assertable_chain and must never raise trusted_all_hops.
        u1 = _ensure_fixed(ge, "noise_hub", "http://teach.local/noise/sourceA")
        u2 = _ensure_fixed(ge, "noise_leaf", "http://teach.local/noise/sourceA")
        e = g.add_edge(u1, u2, weight=0.95, relation_type="causal", confidence=0.95)
        e.source_metadata = {"edge_kind": "co_occurrence", "source": "http://noise"}
        g.add_edge(u2, ids2[2], weight=0.95, relation_type="causal", confidence=0.95)

    if oracle:
        # NON-DEPLOYABLE upper bound: stamp a chain using the item's gold answer
        # key, bypassing graph resolution. Diagnostic only.
        for lab in ("oracle_hub", "oracle_leaf"):
            if ge._all_labels.get(lab) not in g.nodes:
                _ensure_fixed(ge, lab, "http://oracle.local/%s" % subject)
        h = ge._all_labels["oracle_hub"]; l = ge._all_labels["oracle_leaf"]
        for s, t in ((h, l), (l, ids2[2])):
            e = g.add_edge(s, t, weight=1.0, relation_type="causal", confidence=1.0)
            e.source_metadata = {"edge_kind": "curated", "source": "gold_answer_key"}

    # ---- probe: production harness helpers + production gate ----
    lm = label_map(g)
    nids = node_ids_for(spec, g, lm)
    census = gate_census(nids, g)
    chain_present = census["trusted_all_hops"] > 0
    result = {
        "subject": subject, "arm": arm, "a": a, "b": b, "x": x,
        "dead_label": dead_label,
        "probe_seed_ids": nids,
        "endpoint_ids": {"a": ids1[0], "b": ids1[1], "x": ids1[2]},
        "resolved_ids": dict(zip(("a", "b", "x"), ids2)),
        "minted_fresh_ids": minted,
        "state_after_prune": state_after_prune,
        "dangling_edges": len(_dangling(g)),
        "dangling_detail": [[s, t] for s, t in _dangling(g)][:6],
        "gate": census,
        "chain_present": chain_present,
    }
    return result


def main():
    results = {"arms": list(ARMS),
               "replayed_run": "commit 20aca05, scratch/learn5_car_measured_PRE_LIVENESS.json",
               "curves": {"primary": [0.0, 0.8, 0.8, 0.7, 0.8],
                          "sealed": [0.0, 0.7, 0.3, 0.5, 0.9]},
               "items": {}, "controls": {}, "instrument_checks": []}

    for arm in ARMS:
        results["items"][arm] = {}
        for subject in ITEMS:
            r = run_item(subject, arm)
            results["items"][arm][subject] = r
            print("[%-4s] %-8s present=%-5s raw=%-3d trusted=%-3d dangling=%d minted=%s"
                  % (arm, subject, r["chain_present"], r["gate"]["raw_chains"],
                     r["gate"]["trusted_all_hops"], r["dangling_edges"],
                     sorted(r["minted_fresh_ids"])))

    def rate(arm, names):
        return round(sum(results["items"][arm][s]["chain_present"] for s in names)
                     / len(names), 3)

    results["rates"] = {
        arm: {"replayed_failures": rate(arm, FAILING),
              "replayed_successes": rate(arm, SUCCESS),
              "all_ten": rate(arm, list(ITEMS))}
        for arm in ARMS
    }

    # ---- INSTRUMENT VALIDATION: input/output hand-check ----
    r0 = results["items"]["A0"]["bruise"]
    checks = []
    checks.append({
        "name": "A0 failure is dead endpoints, not a trust-gate rejection",
        # One dead hop is sufficient: enumerate_chains requires BOTH endpoints
        # of every hop to be live, so a single dangling curated edge cannot
        # yield a 2-hop chain. The earlier >=2 expectation was a probe error.
        "ok": (r0["gate"]["raw_chains"] == 0 and r0["dangling_edges"] >= 1
               and r0["state_after_prune"]["dead_id_live"] is False),
        "evidence": {"raw_chains": r0["gate"]["raw_chains"],
                     "dangling_edges": r0["dangling_edges"],
                     "dead_id_live": r0["state_after_prune"]["dead_id_live"],
                     "probe_seeds": r0["probe_seed_ids"]},
    })
    r2 = results["items"]["F2"]["bruise"]
    checks.append({
        "name": "F2 restores the same 2-hop curated chain on live endpoints",
        "ok": (r2["gate"]["trusted_all_hops"] >= 1 and r2["dangling_edges"] == 0
               and bool(r2["minted_fresh_ids"])),
        "evidence": {"raw_chains": r2["gate"]["raw_chains"],
                     "trusted": r2["gate"]["trusted_all_hops"],
                     "dangling_edges": r2["dangling_edges"],
                     "minted": r2["minted_fresh_ids"]},
    })
    # recorded successes had no stale id in A0 -> should still acquire
    checks.append({
        "name": "A0 leaves recorded-success items intact (no stale id)",
        "ok": all(results["items"]["A0"][s]["chain_present"] for s in SUCCESS),
        "evidence": {s: results["items"]["A0"][s]["chain_present"] for s in SUCCESS},
    })
    results["instrument_checks"] = checks
    for c in checks:
        print("INSTRUMENT CHECK %-6s %s" % ("OK" if c["ok"] else "FAIL", c["name"]))

    # ---- CONTROLS ----
    # 1. definition-only: no causal hops -> no trusted chain in any arm
    defonly = ConceptGraph(dim=8); gedef = _GE(defonly)
    d1 = _ensure_fixed(gedef, "definition_subject", "http://teach.local/def/sourceA")
    d2 = _ensure_fixed(gedef, "definition_object", "http://teach.local/def/sourceA")
    ed = defonly.add_edge(d1, d2, weight=0.9, relation_type="is_a", confidence=0.9)
    ed.source_metadata = {"edge_kind": "definition", "source": "http://teach.local/def"}
    defres = {"spec": "definition-only subject with is_a edge",
              "raw_chains": gate_census(node_ids_for({"x": "definition_subject"},
                                                      defonly, label_map(defonly)),
                                        defonly)["raw_chains"],
              "trusted_chain": (gate_census(
                  node_ids_for({"x": "definition_subject"}, defonly, label_map(defonly)),
                  defonly)["trusted_all_hops"] > 0)}
    results["controls"]["definition_only"] = defres

    # 2. untrusted competitor: must never be counted as trusted
    untrusted = run_item("bruise", "F2", inject_untrusted=True)
    results["controls"]["untrusted_competitor"] = {
        "raw_chains": untrusted["gate"]["raw_chains"],
        "trusted_all_hops": untrusted["gate"]["trusted_all_hops"],
        "rejected_by_trust": untrusted["gate"]["rejected_by_trust"],
        "note": "co_occurrence edge at w/c=0.95 must stay rejected by assertable_chain",
    }

    # 3. oracle (non-deployable) vs deployable graph-derived arm
    oracle = run_item("bruise", "A0", oracle=True)
    results["controls"]["oracle_non_deployable"] = {
        "arm": "A0 + gold-key stamp (bypasses graph resolution)",
        "trusted_chain": oracle["gate"]["trusted_all_hops"] > 0,
        "deployable": False,
        "note": "uses the item-bank answer key; upper bound only, never evidence",
    }
    results["controls"]["deployable_counterpart"] = {
        "arm": "F2 (production live-id resolution, graph-derived only)",
        "trusted_chain": results["items"]["F2"]["bruise"]["chain_present"],
        "deployable": True,
    }

    # ---- monotonicity arithmetic on the given curve ----
    for pool, curve in results["curves"].items():
        drops = [round(curve[i + 1] - curve[i], 3) for i in range(len(curve) - 1)]
        results.setdefault("monotonicity", {})[pool] = {
            "curve": curve, "adjacent_deltas": drops, "worst": min(drops),
            "gate_bound": -0.10, "pass": min(drops) >= -0.10,
        }
    results["verdict"] = {
        "instrument_ok": all(c["ok"] for c in checks),
        "definition_control_ok": not defres["trusted_chain"],
        "untrusted_control_ok": (
            results["controls"]["untrusted_competitor"]["trusted_all_hops"] >= 1
            and results["controls"]["untrusted_competitor"]["rejected_by_trust"] >= 1),
        "mechanism_confirmed": (
            results["rates"]["A0"]["replayed_failures"] == 0.0
            and results["rates"]["F2"]["all_ten"] == 1.0),
    }
    results["verdict"]["overall"] = all(
        v for k, v in results["verdict"].items() if k != "overall")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("\nRATES:", json.dumps(results["rates"], sort_keys=True))
    print("VERDICT:", json.dumps(results["verdict"], sort_keys=True))
    print("WROTE", out)
    return 0 if results["verdict"]["overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
