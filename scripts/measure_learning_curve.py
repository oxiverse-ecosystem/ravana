#!/usr/bin/env python3
"""WITHIN-SESSION LEARNING CURVE for RAVANA.

First metric that asks whether RAVANA *learns inside one conversation* rather
than scoring a static front. Based on how human learning is actually measured:

  * Ebbinghaus (1885/1913) -- the savings/relearning method: compare the same
    subject before and after an intervening learning episode. Savings = the
    change attributable to the episode, not a single-test score.
  * Roediger & Karpicke (2006), Psychological Science 17(3):249-255 -- the
    testing effect: the post-teaching measurement must be a RETRIEVAL test (a
    held-out paraphrase), not a re-read of the taught sentence.
  * Loewenstein (1994), Psychological Bulletin 116(1):75-98 -- information-gap
    theory: a mechanism question creates the gap; teaching is meant to close it.
  * Schacter & Addis (2007), Phil Trans R Soc B 362(1481):773-786 -- constructive
    episodic simulation: learning is the recombination of elements across
    episodes into a new causal chain, so the unit of learning is a CHAIN, not a
    single fact.

PROPERTY UNDER TEST (one sentence):
  After being taught a 2-hop causal mechanism, RAVANA's answer to a held-out
  paraphrase of the same question contains a PROVENANCE-TRUSTED 2+ hop causal
  chain that was absent at baseline; and this does NOT happen when no teaching
  occurs or when only a definition is taught.

INSTRUMENT (validated BEFORE the curve -- see run_detector_controls):
  A "trusted chain" is the STRUCTURED chain object, never a regex over prose:
    enumerate_chains(..., min_steps=2) -> grade_chain verdict == 'assert'
    -> assertable_chain() (every hop has trusted provenance).
  The same gate is the one the engine itself uses in _why_how_causal_chain.
  A trace hook on ResponseGenMixin._why_how_causal_chain records whether the
  gate actually fired during generation, so we can tell "chain existed in the
  graph" from "chain reached the emitted answer".

ENGINE POLICY: RAVANA_OFFLINE=1, ONE cold boot, ALL arms in one process, on a
SEPARATE scratch data_dir copy. NEVER eng.save().

Usage:
    uv run python scripts/measure_learning_curve.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import traceback

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCRATCH = os.path.join(PROJ, "scratch")
DATA_DIR = os.path.join(SCRATCH, "learning_curve_data")
AUDIT_PATH = os.path.join(SCRATCH, "learning_curve_audit.json")
os.makedirs(SCRATCH, exist_ok=True)

CAUSAL_REL = frozenset({"causal", "physical_cause"})
CAUSAL_MARKERS = re.compile(
    r"\b(because|so that|therefore|thus|hence|leads? to|lead to|causes?|"
    r"makes?|results? in|result in|triggers?|drives?|produces?|since)\b", re.I)

# ── held-out subjects. Each: X = subject, A->B->X is the mechanism to teach. ──
# Single-word A/B/X so WebToGraph's OpenIE + node admission can create nodes.
TEACH = [
    {"x": "fog",       "a": "chill",      "b": "condensation"},
    {"x": "dew",       "a": "cooling",    "b": "condensation"},
    {"x": "rust",      "a": "moisture",   "b": "oxidation"},
    {"x": "echo",      "a": "reflection", "b": "repetition"},
    {"x": "rainbow",   "a": "refraction", "b": "dispersion"},
    {"x": "shadow",    "a": "blockage",   "b": "darkness"},
    {"x": "hiccup",    "a": "spasm",      "b": "interruption"},
    {"x": "blush",     "a": "arousal",    "b": "flushing"},
    {"x": "sunburn",   "a": "radiation",  "b": "damage"},
    {"x": "callus",    "a": "friction",   "b": "thickening"},
    {"x": "bruise",    "a": "impact",     "b": "rupture"},
    {"x": "sweat",     "a": "heat",       "b": "secretion"},
]
NO_TEACH = [   # control: no teaching must NOT improve
    {"x": "sneeze",    "a": "irritation", "b": "reflex"},
    {"x": "shiver",    "a": "chill",      "b": "contraction"},
    {"x": "yawn",      "a": "fatigue",    "b": "inhalation"},
    {"x": "tear",      "a": "irritant",   "b": "secretion"},
    {"x": "cough",     "a": "tickling",   "b": "expulsion"},
    {"x": "hiccup",    "a": "spasm",      "b": "interruption"},
]
DEF_ONLY = [   # control: a definition must NOT yield a causal chain
    {"x": "marble",    "a": "rock",       "b": "mineral"},
    {"x": "velvet",    "a": "fabric",     "b": "texture"},
    {"x": "amber",     "a": "resin",      "b": "fossil"},
    {"x": "copper",    "a": "metal",      "b": "conductor"},
    {"x": "denim",     "a": "cloth",      "b": "cotton"},
    {"x": "ivory",     "a": "material",   "b": "tusk"},
]
WEB_UNGATED = [  # POSITIVE control: same web channel, write-gate bypassed
    {"x": "frost",     "a": "chill",      "b": "deposition"},
    {"x": "mist",      "a": "cooling",    "b": "condensation"},
    {"x": "tarnish",   "a": "moisture",   "b": "oxidation"},
    {"x": "reverb",    "a": "reflection", "b": "repetition"},
    {"x": "prism",     "a": "refraction", "b": "dispersion"},
    {"x": "glow",      "a": "excitation", "b": "emission"},
]

# ── scratch data_dir: a COPY. Never boot against / never save into weights/. ──
def prepare_data_dir():
    src = os.path.join(PROJ, "weights")
    os.makedirs(DATA_DIR, exist_ok=True)
    picked = []
    for name in os.listdir(src):
        if (name.startswith("ravana_weights.pkl")
                or name == "ravana_weights.db"
                or name.startswith("junk_")):
            dst = os.path.join(DATA_DIR, name)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(src, name), dst)
            picked.append(name)
    return picked

# ── graph helpers (dict-vs-node trap handled explicitly) ────────────────────
def _gv(g):
    inner = getattr(g, "graph", None)
    if inner is not None and hasattr(inner, "get_outgoing"):
        return inner
    return g

def label_map(g):
    out = {}
    gv = _gv(g)
    try:
        for nid, node in gv.nodes.items():
            lab = (getattr(node, "label", "") or "").strip().lower()
            if lab:
                out.setdefault(lab, []).append(nid)
    except Exception:
        pass
    return out

def node_ids_for(spec, g, lm=None):
    lm = lm if lm is not None else label_map(g)
    nids = list(lm.get(spec["x"].lower(), []))
    # Hebbian/stale-index trap: also use the engine's own label resolver.
    return nids

def causal_snapshot(g):
    """Set of causal edges by (from_label, rel, to_label, edge_kind)."""
    gv = _gv(g)
    out = set()
    try:
        for nid in list(gv.nodes.keys()):
            for oid, e in (gv.get_outgoing(nid) or []):
                if (str(getattr(e, "relation_type", "") or "").strip().lower()
                        not in CAUSAL_REL):
                    continue
                a = (getattr(gv.get_node(nid), "label", "") or "").strip().lower()
                b = (getattr(gv.get_node(oid), "label", "") or "").strip().lower()
                meta = getattr(e, "source_metadata", None)
                kind = ""
                if isinstance(meta, dict):
                    kind = str(meta.get("edge_kind") or "")
                out.add((a, str(getattr(e, "relation_type", "")), b, kind))
    except Exception:
        pass
    return out

def trusted_chain_summary(c, gr):
    hops = []
    for h in c.hops:
        meta = h.source_metadata if isinstance(h.source_metadata, dict) else {}
        hops.append({"from": h.from_label, "rel": h.relation_type,
                     "to": h.to_label, "edge_kind": str(meta.get("edge_kind") or ""),
                     "weight": round(float(h.weight or 0.0), 3),
                     "confidence": round(float(h.confidence or 0.0), 3)})
    return {"labels": list(c.labels), "hops": hops,
            "score": round(float(c.score), 3),
            "adequacy": round(float(gr.get("adequacy", 0.0)), 3),
            "verdict": gr.get("verdict")}

def trusted_chain(nids, g, direction="backward"):
    """The structured, provenance-gated chain object -- NOT a prose regex."""
    from ravana.learn.explanation import enumerate_chains, grade_chain, assertable_chain
    if not nids:
        return None, None
    try:
        chains = enumerate_chains(nids, g, direction=direction,
                                  min_steps=2, max_steps=4, limit=8)
    except Exception:
        return None, None
    for c in chains:
        try:
            gr = grade_chain(c)
            if gr.get("verdict") == "assert" and assertable_chain(c):
                return c, gr
        except Exception:
            continue
    return None, None

# ── generation trace hook: did the trusted-chain gate actually fire? ────────
def install_trace():
    import ravana.chat.response_gen as rg
    mixin = rg.ResponseGenMixin
    orig = mixin._why_how_causal_chain
    slot = {"hits": []}
    def traced(self, ctx):
        r = orig(self, ctx)
        try:
            slot["hits"].append({
                "subject": getattr(ctx, "subject", None),
                "raw": getattr(ctx, "raw_input", None),
                "chain": (list(r) if isinstance(r, tuple) else r),
            })
        except Exception:
            pass
        return r
    mixin._why_how_causal_chain = traced
    return slot

def ask(eng, q, slot):
    slot["hits"] = []
    t0 = time.time()
    try:
        ans = (eng.process_turn(q) or "").strip()
    except Exception as e:
        ans = "<ERR %s: %s>" % (type(e).__name__, e)
    return {"query": q, "answer": ans,
            "why_hits": list(slot["hits"]),
            "secs": round(time.time() - t0, 2)}

def answer_verdict(turn):
    texts = []
    for h in turn["why_hits"]:
        c = h.get("chain")
        if isinstance(c, (list, tuple)) and c and c[0]:
            texts.append(str(c[0]))
    generated = bool(texts)
    reached = any(t in turn["answer"] for t in texts)
    return {"generated": generated, "reached": reached,
            "chain_texts": texts,
            "markers": CAUSAL_MARKERS.findall(turn["answer"]),
            "has_chain": bool(generated and reached)}

# ── teaching channels ────────────────────────────────────────────────────────
def teach_conversational(eng, spec):
    """Arm (a): the literal conversational assertion, via process_turn."""
    x, a, b = spec["x"], spec["a"], spec["b"]
    stmt = ("%s happens because %s causes %s, and %s causes %s."
            % (x, a, b, b, x))
    turn = ask(eng, stmt, TRACE)
    return {"assertion": stmt, "response": turn["answer"],
            "why_hits": turn["why_hits"], "secs": turn["secs"]}

def teach_web(eng, spec, definition=False, ungated=False):
    """Arm (b): WebToGraph.learn_text from TWO distinct source URLs.

    ungated=True is a POSITIVE CONTROL: it disables the schema-consistency
    write gate for this call only. The measurement criterion is untouched;
    only the teaching channel varies. Measured (gated) causal teaching wrote
    0 facts while definition teaching wrote 1 -- see docs/LEARNING_CURVE.md.
    """
    w = eng._get_web_to_graph()
    if w is None:
        return {"error": "WebToGraph unavailable"}
    x, a, b = spec["x"], spec["a"], spec["b"]
    if definition:
        # Definition-only control: NO causal predicate anywhere.
        text = ("%s is a kind of %s and refers to a %s with a distinctive form."
                % (x, a, b))
    else:
        text = "%s causes %s. %s causes %s." % (a, b, b, x)
    urls = ["http://teach.local/%s/sourceA" % x,
            "http://teach.local/%s/sourceB" % x]
    counts, errs = [], []
    old_intake = w.intake
    if ungated:
        w.intake = None
    try:
        for u in urls:
            try:
                # Authored TEACHING material (teach.local), not an arbitrary web
                # page: declare it as a high-reliability seed so a genuine
                # contradiction with a curriculum incumbent can be reconsoli-
                # dated instead of buffering the fact forever (source monitoring,
                # Johnson, Hashtroudi & Lindsay 1993). Arbitrary web ingests keep
                # the default source_type="web".
                counts.append(int(w.learn_text(text, source_url=u,
                                               source_type="seed")))
            except Exception as e:
                errs.append("%s: %s" % (type(e).__name__, e))
    finally:
        w.intake = old_intake
    return {"text": text, "urls": urls, "facts_written": counts, "errors": errs,
            "gate_bypassed": bool(ungated)}

# ── instrument self-check (run_detector_controls) ────────────────────────────
def run_detector_controls():
    """Positive/negative controls on the CHAIN DETECTOR, before any curve."""
    from ravana_ml.graph import ConceptGraph
    from ravana.learn.explanation import CAUSAL_RELATIONS
    g = ConceptGraph()
    def add_node(lab):
        return g.add_node(label=lab)
    n_x, n_b, n_a = add_node("zzx"), add_node("zzb"), add_node("zza")
    # positive: two trusted web_fact causal edges a->b->x
    for s, t in ((n_a.id, n_b.id), (n_b.id, n_x.id)):
        e = g.add_edge(source=s, target=t, weight=0.8,
                       relation_type="causal", confidence=0.9)
        if e is not None and hasattr(e, "source_metadata"):
            e.source_metadata.update({"edge_kind": "web_fact",
                                      "source": "http://ctrl"})
    c, gr = trusted_chain([n_x.id], g, "backward")
    pos = (c is not None and len(c.hops) >= 2 and gr["verdict"] == "assert")
    # negative: a definition -- semantic is_a edges, no causal predicate
    g2 = ConceptGraph()
    m = g2.add_node(label="defx")
    y = g2.add_node(label="defy")
    z = g2.add_node(label="defz")
    for s, t in ((m.id, y.id), (y.id, z.id)):
        e = g2.add_edge(source=s, target=t, weight=0.8,
                        relation_type="is_a", confidence=0.9)
        if e is not None and hasattr(e, "source_metadata"):
            e.source_metadata.update({"edge_kind": "web_fact"})
    c2, gr2 = trusted_chain([m.id], g2, "backward")
    neg = c2 is None
    # direction check: forward walk must find a->b->x from a
    cf, _ = trusted_chain([n_a.id], g, "forward")
    fwd = cf is not None and cf.labels[-1].lower() == "zzx"
    return {"positive_trusted_2hop_detected": bool(pos),
            "negative_definition_rejected": bool(neg),
            "direction_forward_ok": bool(fwd),
            "positive_summary": trusted_chain_summary(c, gr) if c else None}

TRACE = None

def main():
    global TRACE
    os.environ["RAVANA_OFFLINE"] = "1"
    print("=== prepare scratch data_dir (COPY, no writes to weights/) ===", flush=True)
    print("  copied:", prepare_data_dir(), flush=True)
    print("=== cold boot (one process) ===", flush=True)
    t0 = time.time()
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(data_dir=DATA_DIR, user_suffix="")
    TRACE = install_trace()
    eng._web_direct_answer = lambda ctx: None   # no network direct answers
    print("=== booted in %.1fs ===" % (time.time() - t0), flush=True)

    ctrl = run_detector_controls()
    print("\n=== INSTRUMENT SELF-CHECK (before the curve) ===", flush=True)
    print(json.dumps(ctrl, indent=2), flush=True)

    audit = {"commit": os.popen("git -C %s rev-parse --short HEAD" % PROJ).read().strip(),
             "data_dir": DATA_DIR, "detector_controls": ctrl,
             "arms": {}}

    def run_arm(name, specs, mode):
        print("\n################ ARM: %s (n=%d) ################" % (name, len(specs)),
              flush=True)
        rows = []
        for i, spec in enumerate(specs):
            g = eng.graph
            lm = label_map(g)
            base_nids = node_ids_for(spec, g, lm)
            c0, gr0 = trusted_chain(base_nids, g, "backward")
            t_base = ask(eng, "why does %s happen?" % spec["x"], TRACE)
            v_base = answer_verdict(t_base)
            snap_before = causal_snapshot(g)

            teach = {"mode": mode}
            if mode == "teach":
                teach["conversation"] = teach_conversational(eng, spec)
                teach["conversation"]["post_edge_delta"] = len(
                    causal_snapshot(g) - snap_before)
                teach["web"] = teach_web(eng, spec, definition=False)
            elif mode == "define":
                teach["conversation"] = None
                teach["web"] = teach_web(eng, spec, definition=True)
            elif mode == "web_ungated":
                teach["conversation"] = None
                teach["web"] = teach_web(eng, spec, definition=False,
                                         ungated=True)
            else:
                teach["web"] = None

            snap_after = causal_snapshot(g)
            new_edges = sorted(snap_after - snap_before)

            lm1 = label_map(g)
            post_nids = node_ids_for(spec, g, lm1)
            c1, gr1 = trusted_chain(post_nids, g, "backward")
            t_post = ask(eng, "can you explain why %s occurs?" % spec["x"], TRACE)
            v_post = answer_verdict(t_post)

            row = {
                "idx": i, "subject": spec["x"], "a": spec["a"], "b": spec["b"],
                "baseline": {
                    "question": t_base["query"], "answer": t_base["answer"],
                    "why_hits": t_base["why_hits"], "secs": t_base["secs"],
                    "graph_chain": trusted_chain_summary(c0, gr0) if c0 else None,
                    "verdict": v_base,
                },
                "teaching": teach,
                "post": {
                    "question": t_post["query"], "answer": t_post["answer"],
                    "why_hits": t_post["why_hits"], "secs": t_post["secs"],
                    "graph_chain": trusted_chain_summary(c1, gr1) if c1 else None,
                    "verdict": v_post,
                },
                "new_trusted_causal_edges": [list(e) for e in new_edges
                                             if e[3] in ("web_fact", "curated",
                                                         "definition", "conversation")],
                "new_causal_edges_all": [list(e) for e in new_edges],
            }
            rows.append(row)
            print("\n--- [%s %02d] subject=%s ---" % (mode, i, spec["x"]), flush=True)
            print("  BASE Q: %r" % t_base["query"], flush=True)
            print("  BASE A: %r" % t_base["answer"], flush=True)
            print("  BASE graph_chain: %s" % (row["baseline"]["graph_chain"],), flush=True)
            print("  BASE verdict: %s" % json.dumps(v_base), flush=True)
            print("  TEACH: %s" % json.dumps(teach, ensure_ascii=False), flush=True)
            print("  NEW trusted causal edges: %s" % json.dumps(row["new_trusted_causal_edges"]), flush=True)
            print("  POST Q: %r" % t_post["query"], flush=True)
            print("  POST A: %r" % t_post["answer"], flush=True)
            print("  POST graph_chain: %s" % (row["post"]["graph_chain"],), flush=True)
            print("  POST verdict: %s" % json.dumps(v_post), flush=True)
        audit["arms"][name] = rows
        return rows

    teach_rows = run_arm("TEACH_BOTH", TEACH, "teach")
    no_rows = run_arm("NO_TEACH_CONTROL", NO_TEACH, "none")
    def_rows = run_arm("DEF_ONLY_CONTROL", DEF_ONLY, "define")
    ung_rows = run_arm("WEB_UNGATED_POSITIVE", WEB_UNGATED, "web_ungated")

    # ── aggregate ────────────────────────────────────────────────────────────
    def summarize(rows):
        n = len(rows)
        def rate(key, where):
            return (sum(1 for r in rows if r[where]["verdict"][key]) / n
                    if n else 0.0)
        return {
            "n": n,
            "baseline_graph_chain_rate": round(
                sum(1 for r in rows if r["baseline"]["graph_chain"]) / n, 3) if n else 0,
            "post_graph_chain_rate": round(
                sum(1 for r in rows if r["post"]["graph_chain"]) / n, 3) if n else 0,
            "baseline_answer_chain_rate": round(rate("has_chain", "baseline"), 3),
            "post_answer_chain_rate": round(rate("has_chain", "post"), 3),
            "new_trusted_edges_total": sum(len(r["new_trusted_causal_edges"]) for r in rows),
            "chain_rate_delta": round(
                (sum(1 for r in rows if r["post"]["graph_chain"])
                 - sum(1 for r in rows if r["baseline"]["graph_chain"])) / n, 3) if n else 0,
        }
    summ = {k: summarize(v) for k, v in audit["arms"].items()}
    audit["summary"] = summ
    audit["verdict"] = {
        "teach_delta": summ["TEACH_BOTH"]["chain_rate_delta"],
        "no_teach_delta": summ["NO_TEACH_CONTROL"]["chain_rate_delta"],
        "define_delta": summ["DEF_ONLY_CONTROL"]["chain_rate_delta"],
        "ungated_positive_delta": summ["WEB_UNGATED_POSITIVE"]["chain_rate_delta"],
    }
    with open(AUDIT_PATH, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, ensure_ascii=False, indent=2)

    print("\n\n================ LEARNING CURVE (summary) ================", flush=True)
    print(json.dumps(summ, indent=2), flush=True)
    print("\nAUDIT -> %s" % AUDIT_PATH, flush=True)
    print("DONE", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
