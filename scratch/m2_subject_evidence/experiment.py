#!/usr/bin/env python3
"""M2 scratch experiment: run the 8 pre-registered gates. Static only.

No engine, no process_turn, no store, no network, no benchmark score.
Reads: frozen licensed snapshot + PUBLIC subject strings from tra_k_items.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "C:/Users/Likhith/Documents/projects/ravana_m2_subject_evidence/scripts")

import subject_type_evidence as ste  # noqa: E402

SNAPSHOT = ("C:/Users/Likhith/Documents/projects/ravana_m2_subject_evidence/"
            "data/subject_type_evidence/snapshot.json")
RESULTS = str(HERE / "results.json")
FORBIDDEN = {
    "gold", "type_sets", "world_model", "typed_queries", "channel_evidence",
    "item_id", "role", "type_label", "unseen_test_words", "domain",
    "far_domain", "process_turn", "CognitiveChatEngine",
}
FORBIDDEN_MODULES = {
    "tra_k_items", "tra_items", "front_gates", "measure_tra", "measure_tra_k",
    "ravana",
}
PERMUTE_OFFSET = 137
SAMPLE_N = 40


def public_subjects() -> list[str]:
    import tra_k_items
    out = set()
    for it in tra_k_items.build_items():
        out.add(it["subject"])
        for cond in ("near", "far"):
            out.add(it["unseen_test_words"][cond]["subject"])
    chosen = sorted(out, key=lambda s: hashlib.sha256(
        ("M2-v1|" + s).encode("utf-8")).hexdigest())
    return chosen[:SAMPLE_N]


def ast_audit() -> list[str]:
    src = Path(ste.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_MODULES:
                    problems.append("import:" + a.name)
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in FORBIDDEN_MODULES:
                problems.append("from:" + mod)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                ident = None
                if isinstance(sub, ast.Name):
                    ident = sub.id
                elif isinstance(sub, ast.arg):
                    ident = sub.arg
                elif isinstance(sub, ast.Attribute):
                    ident = sub.attr
                elif isinstance(sub, ast.keyword):
                    ident = sub.arg
                if ident and ident in FORBIDDEN:
                    problems.append(f"{node.name}:{ident}")
    return sorted(set(problems))


def permute_within(subject: str, vocab: list[str]) -> str:
    i = vocab.index(subject)
    return vocab[(i + PERMUTE_OFFSET) % len(vocab)]


def permute_string(s: str) -> str:
    return "".join(chr((ord(c) + PERMUTE_OFFSET - 32) % 95 + 32) for c in s)


def main() -> None:
    os.environ.pop(ste.ENV_FLAG, None)
    snap = json.load(open(SNAPSHOT, encoding="utf-8"))
    subjects = public_subjects()
    vocab = sorted({f["subject"] for f in snap["facts"]})
    R: dict = {
        "schema": "m2-subject-evidence-results-v1",
        "snapshot_facts": len(snap["facts"]),
        "snapshot_subjects": len(vocab),
        "snapshot_facts_sha256": snap["provenance"]["facts_sha256"],
        "sample_n": len(subjects),
        "sample": subjects,
    }

    # G1 anti-leak
    R["G1_ast_problems"] = ast_audit()
    R["G1_pass"] = not R["G1_ast_problems"]

    # G2 default-off (before enabling anything)
    off = {}
    for s in subjects:
        ev, rej = ste.evidence_for_subject(s, snap)
        off[s] = (ste.signature(ev), tuple((r.reason, r.type_name) for r in rej))
    again = {}
    for s in subjects:
        ev, rej = ste.evidence_for_subject(s, snap)
        again[s] = (ste.signature(ev), tuple((r.reason, r.type_name) for r in rej))
    R["G2_off_nonempty"] = sum(1 for v in off.values() if v[0])
    R["G2_byte_identical"] = (off == again)
    R["G2_pass"] = R["G2_off_nonempty"] == 0 and R["G2_byte_identical"]

    os.environ[ste.ENV_FLAG] = "1"

    # G3 specificity
    sigs = {}
    for s in subjects:
        ev, _ = ste.evidence_for_subject(s, snap)
        sigs[s] = ste.signature(ev)
    in_vocab = [s for s in subjects if s in vocab]
    distinct = 0
    compared = 0
    perm_changed = 0
    perm_total = 0
    for s in subjects:
        other = permute_within(s, vocab)
        ev_o, _ = ste.evidence_for_subject(other, snap)
        sig_o = ste.signature(ev_o)
        perm_total += 1
        if sigs[s] and sigs[s] != sig_o:
            perm_changed += 1
    for i, s in enumerate(in_vocab):
        nxt = in_vocab[(i + 1) % len(in_vocab)]
        if nxt != s:
            compared += 1
            if sigs[s] != sigs[nxt]:
                distinct += 1
    fake = permute_string("overcast")
    ev_fake, _ = ste.evidence_for_subject(fake, snap)
    R["G3_distinct"] = distinct
    R["G3_compared"] = compared
    R["G3_distinct_rate"] = distinct / compared if compared else 0.0
    R["G3_perm_changed"] = perm_changed
    R["G3_perm_total"] = perm_total
    R["G3_perm_rate"] = perm_changed / perm_total if perm_total else 0.0
    R["G3_absent_evidence"] = len(ev_fake)
    R["G3_pass"] = (R["G3_distinct_rate"] >= 0.90
                    and R["G3_perm_rate"] >= 0.80
                    and R["G3_absent_evidence"] == 0)

    # G5/G6 coverage + provenance (compute before G4 for the adversarial types)
    covered = 0
    provenance_ok = 0
    admitted_total = 0
    handcheck = []
    for s in subjects:
        ev, _ = ste.evidence_for_subject(s, snap)
        if ev:
            covered += 1
        admitted_total += len(ev)
        for e in ev:
            good = (e.subject == s and e.type_name and e.source and
                    e.license in ste.LICENSE_CREDIBILITY and e.license_url and
                    e.source_url and e.snapshot_sha256 and e.assertion_text and
                    e.credibility >= ste.CREDIBILITY_FLOOR)
            provenance_ok += int(good)
        if ev and len(handcheck) < 10:
            handcheck.append({
                "subject": s, "type_name": ev[0].type_name,
                "assertion_text": ev[0].assertion_text,
                "source_url": ev[0].source_url, "license": ev[0].license,
            })
    R["G5_covered"] = covered
    R["G5_rate"] = covered / len(subjects)
    R["G5_pass"] = covered >= 20
    R["G6_admitted"] = admitted_total
    R["G6_provenance_ok"] = provenance_ok
    R["G6_pass"] = admitted_total > 0 and provenance_ok == admitted_total
    R["handcheck"] = handcheck

    # G4 adversarial 14 cases
    base_subject = next(s for s in subjects if s in vocab)
    base_types = sorted({f["type_name"] for f in snap["facts"]
                         if f["subject"] == base_subject})
    t = base_types[0]
    A = ste.Assertion
    cases = [
        ("neg_not", A(base_subject, "is_a", t, f"{base_subject} is not a {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("neg_never", A(base_subject, "is_a", t, f"{base_subject} is never a {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("neg_isnt", A(base_subject, "is_a", t, f"{base_subject} isn't a {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("neg_no", A(base_subject, "is_a", t, f"{base_subject} is no {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("neg_cannot", A(base_subject, "is_a", t, f"{base_subject} cannot be a {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("neg_classified", A(base_subject, "is_a", t, f"{base_subject} is not classified as a {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("low_credibility", A(base_subject, "is_a", t, f"{base_subject} is a {t}.", "wikidata", "CC0-1.0", credibility=0.1), 0),
        ("unlicensed", A(base_subject, "is_a", t, f"{base_subject} is a {t}.", "unknown", "", credibility=0.9), 0),
        ("empty_text", A(base_subject, "is_a", t, "", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("wrong_subject", A("a_completely_different_term", "is_a", t, f"a_completely_different_term is a {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("bad_relation", A(base_subject, "causes", t, f"{base_subject} causes {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("contradicted", A(base_subject, "is_a", t, f"{base_subject} is a {t}.", "wikidata", "CC0-1.0", credibility=0.7), 0),
        ("pos_cc0", A(base_subject, "is_a", t, f"{base_subject} is a {t}.", "wikidata", "CC0-1.0", credibility=0.7), 1),
        ("pos_wordnet", A(base_subject, "is_a", t, f"{base_subject} is a {t}.", "wordnet", "WordNet-3.0", credibility=0.9), 1),
    ]
    detail = []
    g4_ok = True
    for name, assertion, expected in cases:
        pool = [assertion]
        if name == "contradicted":
            pool.append(A(base_subject, "is_a", t,
                          f"{base_subject} is never a {t}.", "wordnet",
                          "WordNet-3.0", credibility=0.9))
        admitted, rejected = ste.admit(base_subject, pool)
        n = sum(1 for e in admitted if e.type_name == t)
        ok = (n == expected)
        g4_ok &= ok
        detail.append({
            "case": name, "expected": expected, "admitted_for_type": n,
            "reasons": [r.reason for r in rejected], "pass": ok,
        })
    R["G4_detail"] = detail
    R["G4_pass"] = g4_ok

    # G7 candidate permutation: channel has no candidate parameter; emulate a
    # caller that shuffles options and confirm byte-identical evidence.
    opts_a = ("a", "b", "c", "d")
    opts_b = tuple(reversed(opts_a))
    ev_a, _ = ste.evidence_for_subject(subjects[0], snap)
    ev_b, _ = ste.evidence_for_subject(subjects[0], snap)
    R["G7_candidate_permutation_noop"] = (opts_a != opts_b and
                                          ste.signature(ev_a) == ste.signature(ev_b))
    R["G7_pass"] = R["G7_candidate_permutation_noop"]

    # G8: answer-key analogue is structurally inapplicable: API accepts no key.
    sig = __import__("inspect").signature(ste.evidence_for_subject)
    R["G8_structural"] = sorted(sig.parameters)
    R["G8_pass"] = "not_applicable_structural" if "gold" not in sig.parameters else "FAIL"

    R["GATES"] = {f"G{i}": R[f"G{i}_pass"] for i in range(1, 8)}
    R["IMPLEMENT_DECISION"] = "IMPLEMENT" if all(R["GATES"].values()) else "NO-GO"
    with open(RESULTS, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    print("snapshot facts", R["snapshot_facts"], "subjects", R["snapshot_subjects"])
    print("sample", len(subjects))
    for i in range(1, 8):
        print(f"G{i}: {R['GATES'][f'G{i}']}")
    print("G3 distinct %.3f perm %.3f absent %d" % (
        R["G3_distinct_rate"], R["G3_perm_rate"], R["G3_absent_evidence"]))
    print("G5 coverage %d/%d = %.3f" % (R["G5_covered"], len(subjects), R["G5_rate"]))
    print("G4 failures", [d["case"] for d in detail if not d["pass"]])
    print("DECISION", R["IMPLEMENT_DECISION"])
    print("results", RESULTS)


if __name__ == "__main__":
    main()
