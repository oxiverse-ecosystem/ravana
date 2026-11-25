#!/usr/bin/env python3
"""SCRATCH-ONLY: two hypotheses for UNSEEN-SUBJECT type/role transfer, on recorded cases.

No engine boot, no process_turn, no store write, no production edit, no answer key
at grading time. Imports the UNMODIFIED merged typed-slot primitive
(ravana.chat.schema_induction) and the shipped G6 gate read-only.

H_a  TRA-OST: a NEW benchmark variant in which the query subject's type is taught
     ostensively through a provenance-stamped conversational turn; measure whether
     the already-merged typed-slot primitive can consume it (F4-style).
H_b  TRA-VOCAB: a NEW benchmark variant with a subject-specific knowledge channel
     built by conversational teaching of a BROAD vocabulary before the query;
     measure whether an unseen subject is typed without teaching that subject.

Every arm: exact near/far/foil/negative counts, foil+negative rejection, disjoint /
unrelated / untrusted controls, and G6 via the shipped front_gates.gate_saturation.
The ORACLE arms are explicitly NON-DEPLOYABLE and never averaged into a verdict.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parent
MAIN = WT.parent / "ravana"

# Import the UNMODIFIED production modules read-only (no worktree ravana package
# shadows them). PYTHONPATH must point at WT/ravana/src; we add it defensively.
for _p in (str(WT / "ravana" / "src"), str(MAIN / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat import schema_induction as si  # noqa: E402

TYPE_LABELS = sorted(set(si.__dict__.get("ROLE_TO_TYPE", {}).values())) or [
    "contrast-case", "earlier-event", "effect", "similar-form",
    "structured-whole", "supported-outcome",
]
# Fall back to the item bank's canonical order if the primitive has no such map.

CONGRUENCE_FLOOR = si.CONGRUENCE_FLOOR   # 0.20, frozen in primitive
STRICT_MARGIN = si.STRICT_MARGIN         # 0.10, frozen in primitive
EDGE_CONF = 0.8                          # curated write confidence (intake signal)
DISJOINT = ["apple", "banana", "chair", "table"]
UNRELATED = "zeppelin"
RECORDED_IDS = [f"tra_{i:04d}" for i in range(40)]
CONDITIONS = ("near", "far", "foil", "negative")

OUT = HERE / "goal_unseen_type_results.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def log(msg=""):
    print(msg, flush=True)


def slot_view(subject, target, type_label, *, kind="curated", source="conversation",
              source_type="user", rel="is_a", conf=EDGE_CONF):
    return si.TypedSlotEdgeView(
        source=str(subject).strip().lower(),
        relation_type=rel,
        target=str(target).strip().lower(),
        type_label=str(type_label).strip().lower(),
        slot_label="",
        edge_kind=str(kind).strip().lower(),
        edge_source=str(source).strip().lower(),
        source_type=str(source_type).strip().lower(),
        confidence=float(conf),
    )


def is_trusted(view):
    # Use the primitive's own predicate so trust is not re-implemented.
    return si._is_trusted_edge(view)


def instrument():
    assert "CognitiveChatEngine" not in sys.modules, "engine imported: no boot allowed"
    loaded = Path(si.__file__).resolve()
    expected = (WT / "ravana" / "src" / "ravana" / "chat" / "schema_induction.py").resolve()
    assert loaded == expected, f"primitive resolved to {loaded}, want {expected}"
    prim_sha = sha256(expected)
    main_sha = sha256(MAIN / "ravana" / "src" / "ravana" / "chat" / "schema_induction.py")
    assert prim_sha == main_sha, "merged primitive differs between main and worktree"

    ti = load(WT / "scripts" / "tra_items.py", "ti_goal")
    bank = ti.build_items()
    errs = ti.validate_items(bank)
    assert not errs, f"item bank invalid: {errs[:3]}"
    by = {it["item_id"]: it for it in bank}

    rows = [json.loads(l) for l in open(MAIN / "scratch" / "tra_harness" / "tra_rows.jsonl",
                                        encoding="utf-8")]
    live = [r for r in rows if r.get("backend") == "live"]
    assert sorted({r["id"] for r in live}) == RECORDED_IDS
    sel = [r for r in live if r.get("arm") == "structural" and r.get("phase") == "query"
           and not r.get("permuted")]
    assert len(sel) == 160
    assert collections.Counter(r["condition"] for r in sel) == {c: 40 for c in CONDITIONS}

    checked = 0
    for r in sel:
        it = by[r["id"]]
        cond = r["condition"]
        if cond in ("near", "far"):
            q = it["unseen_test_words"][cond]
            exp = (q["subject"], q["candidates"], q["gold"])
        elif cond == "foil":
            exp = (it["foil"]["subject"], it["foil"]["candidates"], None)
        else:
            exp = (it["negative"]["subject"], it["negative"]["candidates"], None)
        assert r["input"]["query_subject"] == exp[0]
        assert set(r["input"]["candidates"]) == set(exp[1])
        assert r["gold"]["target"] == exp[2]
        checked += 1

    fg = load(MAIN / "scripts" / "front_gates.py", "fg_goal")

    # Info audit (recompute, identical to prior docs).
    subject_own_type = 0
    gold_has_type = 0
    n_gold = 0
    for r in sel:
        it = by[r["id"]]
        if r["condition"] in ("near", "far"):
            s = r["input"]["query_subject"]
            if it["type_label"] in it["type_sets"].get(s, []):
                subject_own_type += 1
            g = r["gold"]["target"]
            n_gold += 1
            if it["type_label"] in it["type_sets"].get(g, []):
                gold_has_type += 1

    return {
        "ti": ti, "bank": bank, "by": by, "sel": sel, "fg": fg,
        "prim_sha": prim_sha, "fg_sha": sha256(MAIN / "scripts" / "front_gates.py"),
        "bank_sha": sha256(WT / "scripts" / "tra_items.py"),
        "rows_sha": sha256(MAIN / "scratch" / "tra_harness" / "tra_rows.jsonl"),
        "info": {"subject_own_type_in_teaching_label": [subject_own_type, 160],
                 "gold_candidate_has_taught_type": [gold_has_type, n_gold]},
    }


def g6(fg, near_n, neg_reject_n, n=40):
    r = fg.gate_saturation({"tra_near": 100.0 * near_n / n,
                            "tra_negative_reject": 100.0 * neg_reject_n / n})
    return {"status": r.status, "flagged": r.flagged_fronts,
            "detail": r.detail}


def grade_pick(pick, gold, reject_expected):
    if reject_expected:
        return pick is None
    return pick is not None and str(pick).strip().lower() == str(gold).strip().lower()


# ═══════════════════════ H_a : TRA-OST ═══════════════════════════════════════
def run_ha(inst):
    ti, by, sel = inst["ti"], inst["by"], inst["sel"]
    rows = []
    for r in sel:
        it = by[r["id"]]
        cond = r["condition"]
        subj = r["input"]["query_subject"]
        cands = list(r["input"]["candidates"])
        label = it["type_label"]
        gold_word = r["gold"]["target"]
        label_opts = list(TYPE_LABELS)

        # Demonstrated anchor from this item's own teaching (wrong subject for foil).
        demo_anchor = it["taught_exemplars"][0]["r2"] if it["taught_exemplars"] else "cloud"

        # Teaching turns (provenance-stamped). Definition per condition:
        #  near/far  : ostensive positive turn about the query subject
        #  foil      : positive turn about a DIFFERENT (demonstrated) subject
        #  negative  : NEGATED ostensive turn about the query subject
        if cond in ("near", "far"):
            teach = {"kind": "pos_subject", "edges": [slot_view(subj, label, label)]}
        elif cond == "foil":
            teach = {"kind": "wrong_subject",
                     "edges": [slot_view(demo_anchor, label, label)]}
        else:
            # Negation writes no trusted is_a edge.
            teach = {"kind": "negated_subject", "edges": []}

        views = teach["edges"]

        # (1) type-label option space (the TRA-OST benchmark query)
        dec_type = si.slot_keyed_forced_choice(views, subj, label_opts)
        pick_type = dec_type["pick"]
        correct_type = grade_pick(pick_type, label, cond in ("foil", "negative"))

        # (2) original TRA domain-word option space (transfer test)
        dec_word = si.slot_keyed_forced_choice(views, subj, cands)
        pick_word = dec_word["pick"]
        correct_word = grade_pick(pick_word, gold_word, cond in ("foil", "negative"))

        # (3) baseline: no subject-type teaching at all
        dec_base = si.slot_keyed_forced_choice([], subj, label_opts)
        pick_base = dec_base["pick"]
        correct_base = grade_pick(pick_base, label, cond in ("foil", "negative"))

        rows.append({
            "id": r["id"], "condition": cond, "k": it["k"], "pool": it["pool"],
            "subject": subj, "teach": teach["kind"], "type_label": label,
            "gold_word": gold_word, "demo_anchor": demo_anchor,
            "pick_type": pick_type, "correct_type": bool(correct_type),
            "type_reason": dec_type["reason"],
            "pick_word": pick_word, "correct_word": bool(correct_word),
            "word_reason": dec_word["reason"],
            "pick_base": pick_base, "correct_base": bool(correct_base),
        })

    def tally(field):
        out = {}
        for c in CONDITIONS:
            rs = [x for x in rows if x["condition"] == c]
            out[c] = sum(1 for x in rs if x[field])
        out["near+far"] = out["near"] + out["far"]
        return out

    res = {
        "OST_type_space": tally("correct_type"),
        "OST_word_space_originalTRA": tally("correct_word"),
        "OST_no_subject_teaching": tally("correct_base"),
    }
    # Controls
    pos_edge = [slot_view(subj, label, label) for subj, label in [("tempest", "effect")]]
    ctl = {}
    ctl["disjoint_type"] = si.slot_keyed_forced_choice(pos_edge, "tempest", DISJOINT)["pick"]
    ctl["unrelated_subject"] = si.slot_keyed_forced_choice(pos_edge, UNRELATED, TYPE_LABELS)["pick"]
    ctl["untrusted_seed_edge"] = si.slot_keyed_forced_choice(
        [slot_view("tempest", "effect", "effect", kind="seed", source="web")],
        "tempest", TYPE_LABELS)["pick"]
    ctl["wrong_relation"] = si.slot_keyed_forced_choice(
        [slot_view("tempest", "effect", "effect", rel="causes")],
        "tempest", TYPE_LABELS)["pick"]
    res["controls"] = ctl
    res["controls_pass"] = all(v is None for v in ctl.values())
    res["g6_type_space"] = g6(inst["fg"], res["OST_type_space"]["near"],
                              res["OST_type_space"]["negative"])
    res["g6_word_space"] = g6(inst["fg"], res["OST_word_space_originalTRA"]["near"],
                              res["OST_word_space_originalTRA"]["negative"])
    return res, rows


# ═══════════════════════ H_b : TRA-VOCAB ════════════════════════════════════
def build_broad_lexicon(inst, exclude_subjects=()):
    """Conversational teaching of a BROAD vocabulary: word -> set(type labels).

    In the NEW benchmark variant this is legitimate teaching material (real
    conversational turns written through the unchanged intake/provenance path).
    In this scratch replay the curriculum is populated from the item bank's
    authored ontology; that ontology is ORACLE-only in ORIGINAL TRA, so the
    resulting arm is CHANGED-BENCHMARK evidence, never an original-TRA claim.
    """
    lex = collections.defaultdict(set)
    for it in inst["bank"]:
        for w, tset in it["type_sets"].items():
            for t in tset:
                if t in TYPE_LABELS:
                    lex[w].add(t)
    for s in exclude_subjects:
        lex.pop(str(s).strip().lower(), None)
    edges = []
    for w, ts in lex.items():
        for t in ts:
            edges.append(slot_view(w, t, t))
    return lex, edges


def run_hb(inst):
    ti, by, sel = inst["ti"], inst["by"], inst["sel"]
    # Broad channel taught once over the authored vocabulary. Exclude all 40
    # recorded query subjects so every tested subject stays unseen.
    subjects = {r["input"]["query_subject"].strip().lower() for r in sel}
    lex, broad_edges = build_broad_lexicon(inst, exclude_subjects=subjects)
    log(f"  broad lexicon: {len(lex)} words, {len(broad_edges)} trusted is_a edges")

    rows = []
    for r in sel:
        it = by[r["id"]]
        cond = r["condition"]
        subj = r["input"]["query_subject"]
        cands = list(r["input"]["candidates"])
        label = it["type_label"]
        gold_word = r["gold"]["target"]

        # Item teaching traces (demonstrated r2 is_a edges) for schema induction.
        demo_edges = [slot_view(ex["r2"], label, label)
                      for ex in it["taught_exemplars"][:it["k"]]]
        traces = si.traces_from_edges(demo_edges)
        hyps = si.induce_schema_hypotheses(traces, min_support=2)

        # (a) coverage diagnostic: is the gold candidate typed as `label` by the channel?
        gold_typed = label in lex.get(str(gold_word).strip().lower(), set()) if gold_word else None

        # (b) type-blind ranking: every taught candidate = 1.0
        blind_scores = {c.lower(): (1.0 if c.lower() in lex else 0.0) for c in cands}
        pick_blind = si.rank_candidates_for_unseen_subject(
            subj, cands, hyps, traces, blind_scores, min_margin=STRICT_MARGIN)

        # (c) typed-query arm (NEW variant: the query states the target type T):
        #     candidate score = 1.0 iff the channel taught that candidate as T.
        # INSTRUMENT CORRECTION (run 2): run 1 picked a fallback "other" type for
        # foils, which was degenerate (it was itself held by candidates), so its
        # foil column measured the fallback, not rejection. Run 2 states the SAME
        # gold type T for foil/negative as for near/far:
        #   foil      = original TRA foil candidates (contain a demonstrated
        #               anchor); correct behaviour is REJECT (pick must be None).
        #   negative  = channel taught negation for T (no trusted edge to T);
        #               correct behaviour is REJECT.
        # No threshold, gate or edge-confidence value is changed.
        qtype = label
        if cond == "negative":
            typed_scores = {c.lower(): 0.0 for c in cands}
        else:
            typed_scores = {c.lower(): (1.0 if qtype in lex.get(c.lower(), set()) else 0.0)
                            for c in cands}
        pick_typed = si.rank_candidates_for_unseen_subject(
            subj, cands, hyps, traces, typed_scores, min_margin=STRICT_MARGIN)

        # Subject-independence control (decisive): same candidates, same scores,
        # same hypotheses, but an UNRELATED subject that was never taught. If the
        # pick is unchanged, the decision carried zero information about `subj`.
        pick_zeppelin = si.rank_candidates_for_unseen_subject(
            UNRELATED, cands, hyps, traces, typed_scores, min_margin=STRICT_MARGIN)
        subject_dependent = (pick_typed is not None and pick_typed != pick_zeppelin)

        correct_typed = grade_pick(pick_typed, gold_word, cond in ("foil", "negative"))
        correct_blind = grade_pick(pick_blind, gold_word, cond in ("foil", "negative"))

        # (d) ORACLE (non-deployable): additionally teach the subject's own type.
        oracle_views = broad_edges + [slot_view(subj, label, label)]
        pick_oracle = si.slot_keyed_forced_choice(oracle_views, subj, TYPE_LABELS)["pick"]

        rows.append({
            "id": r["id"], "condition": cond, "k": it["k"], "pool": it["pool"],
            "subject": subj, "type_label": label, "gold_word": gold_word,
            "gold_typed_by_channel": bool(gold_typed) if gold_typed is not None else None,
            "n_hyps": len(hyps), "n_traces": len(traces),
            "pick_blind": pick_blind, "correct_blind": bool(correct_blind),
            "qtype": qtype, "pick_typed": pick_typed,
            "correct_typed": bool(correct_typed),
            "pick_zeppelin": pick_zeppelin, "subject_dependent": bool(subject_dependent),
            "pick_oracle_type": pick_oracle,
        })

    def tally(field):
        out = {}
        for c in CONDITIONS:
            rs = [x for x in rows if x["condition"] == c]
            out[c] = sum(1 for x in rs if x[field])
        out["near+far"] = out["near"] + out["far"]
        return out

    res = {
        "VOCAB_typed_query": tally("correct_typed"),
        "VOCAB_typeblind": tally("correct_blind"),
        "gold_candidate_typed_by_channel_near+far": sum(
            1 for x in rows if x["condition"] in ("near", "far") and x["gold_typed_by_channel"]),
        "near+far_total": 80,
        "subject_dependent_picks_near+far": sum(
            1 for x in rows if x["condition"] in ("near", "far") and x["subject_dependent"]),
        "pick_identical_to_unrelated_subject_near+far": sum(
            1 for x in rows if x["condition"] in ("near", "far")
            and x["pick_typed"] is not None and x["pick_typed"] == x["pick_zeppelin"]),
        "oracle_subject_taught_type_pick_matches_label": sum(
            1 for x in rows if x["condition"] in ("near", "far")
            and x["pick_oracle_type"] == x["type_label"]),
    }
    ctl = {}
    ctl["disjoint"] = si.rank_candidates_for_unseen_subject(
        UNRELATED, DISJOINT, [], [], {c.lower(): 1.0 for c in DISJOINT}, min_margin=STRICT_MARGIN)
    ctl["unrelated_no_scores"] = si.rank_candidates_for_unseen_subject(
        UNRELATED, list(TYPE_LABELS), [], [], {}, min_margin=STRICT_MARGIN)
    res["controls"] = ctl
    res["controls_pass"] = all(v is None for v in ctl.values())
    res["g6_typed_query"] = g6(inst["fg"], res["VOCAB_typed_query"]["near"],
                               res["VOCAB_typed_query"]["negative"])
    return res, rows


def main():
    log("=== INSTRUMENT ===")
    inst = instrument()
    log(f"  primitive sha256={inst['prim_sha'][:16]}  bank sha256={inst['bank_sha'][:16]}")
    log(f"  recorded rows sha256={inst['rows_sha'][:16]}  graded=160")
    log(f"  info audit: subject-own-type={inst['info']['subject_own_type_in_teaching_label']}"
        f"  gold-has-type={inst['info']['gold_candidate_has_taught_type']}")

    log("\n=== H_a : TRA-OST (ostensive subject-type teaching) ===")
    ha, ha_rows = run_ha(inst)
    log("  TYPE-LABEL space: " + json.dumps(ha["OST_type_space"]))
    log("  ORIGINAL word space: " + json.dumps(ha["OST_word_space_originalTRA"]))
    log("  NO subject teaching: " + json.dumps(ha["OST_no_subject_teaching"]))
    log("  controls: " + json.dumps(ha["controls"]) + " pass=" + str(ha["controls_pass"]))
    log("  G6 type=" + json.dumps(ha["g6_type_space"]) + " word=" + json.dumps(ha["g6_word_space"]))

    log("\n=== H_b : TRA-VOCAB (broad conversational type channel) ===")
    hb, hb_rows = run_hb(inst)
    log("  typed-query: " + json.dumps(hb["VOCAB_typed_query"]))
    log("  type-blind: " + json.dumps(hb["VOCAB_typeblind"]))
    log(f"  gold typed by channel: {hb['gold_candidate_typed_by_channel_near+far']}/{hb['near+far_total']}")
    log(f"  ORACLE subject-taught picks: {hb['oracle_subject_taught_type_pick_matches_label']}/80")
    log(f"  subject-dependent picks: {hb['subject_dependent_picks_near+far']}/80 "
        f"(identical to unrelated subject: {hb['pick_identical_to_unrelated_subject_near+far']})")
    log("  controls: " + json.dumps(hb["controls"], default=str) + " pass=" + str(hb["controls_pass"]))
    log("  G6 typed-query=" + json.dumps(hb["g6_typed_query"]))

    out = {
        "meta": {
            "branch": "goal/unseen-type-transfer",
            "primitive_sha256": inst["prim_sha"], "front_gates_sha256": inst["fg_sha"],
            "bank_sha256": inst["bank_sha"], "recorded_rows_sha256": inst["rows_sha"],
            "n_graded": 160, "n_items": 40, "no_engine_boot": True,
            "floor": CONGRUENCE_FLOOR, "margin": STRICT_MARGIN, "edge_conf": EDGE_CONF,
            "type_labels": TYPE_LABELS, "info_audit": inst["info"],
        },
        "H_a": ha, "H_b": hb,
        "H_a_rows": ha_rows, "H_b_rows": hb_rows,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log(f"\nWROTE {OUT}")
    log("EXIT=0")


if __name__ == "__main__":
    main()
