#!/usr/bin/env python3
"""g62 first-turn isolation probe: fresh teach vs. after other items' invalidations."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for p in (os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"), PROJ):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("RAVANA_OFFLINE", "1")
os.environ["RAVANA_SOURCE_EVAL"] = "1"
import measure_learn5_wfa as w  # noqa: E402


def snapshot(eng, claim, sid):
    rows = w.graph_provenance_for_claim(eng, claim, sid)
    ctx, audit = w.context_from_graph_provenance(eng, claim, sid)
    from ravana.chat.source_evaluation import evaluate_source
    dec = evaluate_source(claim, ctx)
    return {"rows": rows, "audit": audit, "decision": dec.decision.value,
            "rule": dec.rule}


def main():
    w.mfwf.install_frozen_patches()
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"), user_suffix="")
    items = {it["id"]: it for it in w.build_items()}
    it = items["g62"]
    claim, sid = it["claim"], w.is_source_id(it["claim"], "discredited")
    out = {"claim": claim, "source_id": sid, "steps": []}
    print("CLAIM:", claim, flush=True)
    w.teach_claim_into_graph(eng, claim, sid)
    out["steps"].append({"stage": "after_teach_only", **snapshot(eng, claim, sid)})
    print("after_teach:", out["steps"][-1]["decision"], out["steps"][-1]["rule"],
          "n=", out["steps"][-1]["audit"]["n_matching_edges"],
          "status=", out["steps"][-1]["audit"]["source_status"], flush=True)
    for iid in ("gg039", "gg032", "g64"):
        other = items[iid]
        osid = w.is_source_id(other["claim"], "discredited")
        w.teach_claim_into_graph(eng, other["claim"], osid)
        n = w.apply_source_invalidation(eng, osid)
        out["steps"].append({"stage": "after_invalidate_" + iid,
                             "tagged": n, **snapshot(eng, claim, sid)})
        s = out["steps"][-1]
        print("after_%s: tagged=%d g62_dec=%s rule=%s n=%d status=%s" % (
            iid, n, s["decision"], s["rule"], s["audit"]["n_matching_edges"],
            s["audit"]["source_status"]), flush=True)
    with open(os.path.join(HERE, "wfa_g62_isolation.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("done", flush=True)


if __name__ == "__main__":
    main()
