"""Frozen-state A/B for the default-OFF external-signal encoding gate.

Arms: OFF (flag=0), ON-signal (flag=1 + external re-ask), ON-no-signal (floor),
ON-wrong-tag (negative control). Each arm boots ONE engine sequentially on its
own isolated copy of the repaired weights; weights/ is never a save target.
Fixtures are the captured web snapshots; no live fetch is attempted.

Metric that counts: new provenance-stamped facts and ASSERTABLE chains on the
probe subjects (chain_hops is diagnostic only). Falsifiers are checked inline.
"""
import copy, hashlib, json, os, shutil, sys
from collections import Counter

P = "C:/Users/Likhith/Documents/projects/ravana"
for p in (P, P + "/scripts", P + "/ravana/src", P + "/ravana_ml/src"):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
import measure_learning_curve as mlc
from ravana.chat.engine import CognitiveChatEngine
from ravana.learn.explanation import enumerate_chains, grade_chain, assertable_chain

FIX = json.load(open(P + "/tests/fixtures/web_snapshots/web_fixtures.json", encoding="utf-8"))
QUERIES = FIX["queries"]
SUBJECTS = [q for q, v in QUERIES.items()
            if len(__import__("re").findall(r"\bcauses?\b",
                " ".join(h.get("content", "") for h in v.get("hits", [])), __import__("re").I)) >= 2]


def split(q):
    h = hashlib.sha256(("encoding_gate|medium|" + q).encode()).hexdigest()
    return "primary" if int(h[0], 16) % 2 == 0 else "sealed"


SUBJECTS = sorted(SUBJECTS, key=lambda q: hashlib.sha256(q.encode()).hexdigest())
SUBJECTS = SUBJECTS[:24]
assert len(SUBJECTS) == 24


def fresh_dir(name):
    d = P + "/scratch/encoding_gate_" + name
    if os.path.exists(d):
        shutil.rmtree(d)
    mlc.DATA_DIR = d
    mlc.prepare_data_dir()
    return d


def boot(d, flag):
    os.environ["RAVANA_CURIOSITY_GATE"] = flag
    e = CognitiveChatEngine(data_dir=d, user_suffix="")
    e._web_direct_answer = lambda ctx: None
    return e


def subject_from_query(q):
    # use the last content word with a graph node if possible; fixtures are
    # natural-language queries, so take the longest word present in the index
    w = q.lower().split()
    cands = [x.strip("?.,!") for x in w if len(x) >= 4]
    return cands[-1] if cands else q


def assertable_count(e, subject):
    g = e.graph
    lm = mlc.label_map(g)
    ids = lm.get(subject, [])
    if not ids:
        return 0
    out = 0
    for nid in ids[:3]:
        try:
            chs = enumerate_chains(g, nid, direction="backward", min_steps=2,
                                   max_steps=4, limit=64)
            for c in chs:
                if assertable_chain(c) and grade_chain(c).get("verdict") == "assert":
                    out += 1
        except Exception:
            pass
    return out


def run_arm(name, flag, signal, wrong_tag):
    d = fresh_dir(name)
    e = boot(d, flag)
    w = e._get_web_to_graph()
    adm = 0
    facts = 0
    written_ids = set()
    for q in SUBJECTS:
        subj = subject_from_query(q)
        # external signal: simulate a re-ask by recording the same content
        # words as the previous turn, then setting the current turn
        if signal:
            e._prev_query_words = e._curiosity_content_words(q)
            e.turn_count = int(getattr(e, "turn_count", 0)) + 1
            e._track_curiosity_signal(q)
        allowed = e._curiosity_gate_allows(subj)
        adm += bool(allowed)
        if not allowed:
            continue
        content = " ".join(h.get("content", "") for h in QUERIES[q].get("hits", []))
        try:
            n = w.learn_text(content[:6000], source_url="http://fixture.local/" + q.replace(" ", "_"),
                             source_type="seed", context="encoding_gate_" + name)
            facts += int(n)
            # Measure from the endpoints ACTUALLY written, not from a guess at
            # the query's subject label (fixture OpenIE writes phrase nodes).
            for (s, t), ed in e.graph.edges.items():
                md = getattr(ed, "source_metadata", None) or {}
                if "fixture.local" in str(md.get("source", "")):
                    written_ids.add(s)
                    written_ids.add(t)
                    if wrong_tag:
                        md["edge_kind"] = "unknown"
                        md["source"] = ""
        except Exception as ex:
            print("  write error", name, q, type(ex).__name__, ex, flush=True)
    # Assertable chains from every endpoint written in THIS arm.
    counts = 0
    for nid in list(written_ids)[:120]:
        try:
            chs = enumerate_chains(e.graph, nid, direction="backward",
                                   min_steps=2, max_steps=4, limit=64)
            for c in chs:
                if assertable_chain(c) and grade_chain(c).get("verdict") == "assert":
                    counts += 1
        except Exception:
            pass
    e.save()
    result = {"arm": name, "flag": flag, "signal": signal, "wrong_tag": wrong_tag,
              "reads_admitted": adm, "facts_written": facts,
              "written_endpoints": len(written_ids),
              "assertable_total": counts,
              "pool_split": {p: sum(1 for q in SUBJECTS if split(q) == p)
                             for p in ("primary", "sealed")}}
    del e, w
    return result


out = {"fixture": "tests/fixtures/web_snapshots/web_fixtures.json",
       "n_subjects": len(SUBJECTS),
       "candidate_queries_with_2plus_causal_tokens": len(SUBJECTS)}
for name, flag, signal, wrong in (
        ("flag_off", "0", True, False),
        ("on_no_signal", "1", False, False),
        ("on_signal", "1", True, False),
        ("on_wrong_tag", "1", True, True)):
    r = run_arm(name, flag, signal, wrong)
    out[name] = r
    print(name, json.dumps({k: v for k, v in r.items() if k != "assertable_subjects"}), flush=True)

out["falsifiers"] = {
    "floor_admits_zero": out["on_no_signal"]["reads_admitted"] == 0,
    "flag_off_admits_all": out["flag_off"]["reads_admitted"] == len(SUBJECTS),
    "on_bounded": out["on_signal"]["reads_admitted"] <= 3,
    "wrong_tag_no_gain": out["on_wrong_tag"]["assertable_total"] <= out["flag_off"]["assertable_total"],
}
out["verdict"] = "GAIN" if (out["on_signal"]["assertable_total"] >
                            out["flag_off"]["assertable_total"] and
                            all(out["falsifiers"].values())) else "NO_GAIN"
print("RESULT", json.dumps(out, indent=2), flush=True)
json.dump(out, open(P + "/scratch/encoding_gate_ab_result.json", "w", encoding="utf-8"), indent=2)
