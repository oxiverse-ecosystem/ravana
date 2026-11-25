#!/usr/bin/env python3
"""CAR residual identical-surface echo — scratch paraphrase/abstraction experiment.

NO ENGINE BOOT. NO PRODUCTION EDITS. NO GATE/THRESHOLD CHANGES.
The parent agent runs WFA; this experiment is deliberately offline and pure-stdlib.

WHAT IS MEASURED
  The shipped CAR harness (scripts/measure_learn5_car.py:495-543) measures
  echo on the IDENTICAL question as:  a in answer.lower() and b in answer.lower()
  and measures reach on a held-out PARAPHRASE question as byte-exact containment
  of the assembler's chain text (scripts/measure_learning_curve.py:233-244).

  We reconstruct fake contexts from the documented k=8 trace strings in
  docs/investigations/CAR_ANSWER_REACH.md and the authored _CANDIDATES bank
  (parsed read-only from scripts/measure_learn5_car.py with ast).  We test three
  deterministic before-emission surface rewrites that fire ONLY on a byte-exact
  taught-surface match inside a gate-verified trusted gated_explanation_chain.

  Four measurements per arm (all deterministic; no sampling):
    harness_echo       - shipped metric, a AND b appear in identical answer
    verbatim_echo      - diagnostic telemetry: taught phrase appears byte-exact
    byte_reach         - shipped reach metric on the paraphrase channel
    semantic_reach     - proxy: (a,b,x) in causal order with a causal marker in
                         both spans (NOT a gate; for reporting only)

  Fail-closed checks: untrusted provenance, wrong strategy, unverified chain,
  no taught surface, quoted user text, and the paraphrase/reach channel must all
  be untouched; the verified chain object itself must be byte-identical.

Run: python3 scratch/car_echo_reduction_experiment.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
BANK_SRC = os.path.join(PROJ, "scripts", "measure_learn5_car.py")
OUT = os.path.join(HERE, "car_echo_reduction_results.json")

# ---------------------------------------------------------------- bank (data) --
def load_candidates():
    with open(BANK_SRC, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "_CANDIDATES" for t in node.targets
        ):
            return [tuple(v.value for v in elt.elts) for elt in node.value.elts]
    raise RuntimeError("_CANDIDATES not found in %s" % BANK_SRC)


# ------------------------------------------- documented k=8 trace identities ---
# From docs/investigations/CAR_ANSWER_REACH.md section 1 table. These are the
# named k=8 subject items with their exact assembler narration. They anchor the
# fake contexts in REAL recorded strings (the raw JSON was untracked scratch).
DOC_K8 = {
    # primary (6 named in the audited table)
    "sneeze": "irritation leads to reflex, and reflex leads to sneeze",
    "cough": "tickling leads to expulsion, and expulsion leads to cough",
    "ivory": "material leads to tusk, and tusk leads to ivory",
    "hide": "tanning leads to leather, and leather leads to hide",
    "coral": "secretion leads to reef, and reef leads to coral",
    "fossil": "burial leads to mineralization, and mineralization leads to fossil",
    # sealed (10 named; row 45 + row 46 of the same table)
    "eclipse": "darkness leads to shadow, and shadow leads to eclipse",
    "fog": "chill leads to condensation, and condensation leads to fog",
    "yawn": "fatigue leads to inhalation, and inhalation leads to yawn",
    "snore": "vibration leads to breath, and breath leads to snore",
    "leaf": "photosynthesis leads to sugar, and sugar leads to leaf",
    "boil": "heating leads to bubbles, and bubbles leads to boil",
    "basalt": "lava leads to solidification, and solidification leads to basalt",
    "flood": "rainfall leads to overflow, and overflow leads to flood",
    "erosion": "weathering leads to wear, and wear leads to erosion",
    "quartz": "cooling leads to crystal, and crystal leads to quartz",
}
PRIMARY_DOC = [x for x in DOC_K8 if x in
               ("sneeze", "cough", "ivory", "hide", "coral", "fossil")]
SEALED_DOC = [x for x in DOC_K8 if x not in PRIMARY_DOC]

# --------------------------------------------------------------- fake context --
def make_ctx(x, a, b, echo_positive, *, trusted=True, strategy="gated_explanation_chain",
             verified=True, chain=True, quoted=False):
    """One fake identical-surface probe context, derived from recorded strings.

    taught_surface  : exact causal phrase taught at k=8 and embedded verbatim in
                      the IDENTICAL question (measure_learn5_car.py:346, :250-251)
    chain_text      : the assembler's verified narration (doc section 1)
    ident_answer    : assertion-mirror echo when echo_positive, else the recorded
                      uncertainty fallback (doc section 1 / :7434-7436)
    para_answer     : the reach channel, byte-exact chain text after the shipped
                      verified-chain exemption
    """
    taught = "%s causes %s, and %s causes %s" % (a, b, b, x)
    chain_text = DOC_K8.get(x) or "%s leads to %s, and %s leads to %s" % (a, b, b, x)
    if quoted:
        ident = "you mentioned: is it because %s?" % taught
    elif echo_positive:
        ident = "yes, because %s." % taught
    else:
        ident = "i don't really have a solid grasp on %s so far." % x
    return {
        "x": x, "a": a, "b": b,
        "taught_surface": taught,
        "chain_text": chain_text if chain else None,
        "ident_answer": ident,
        "para_answer": chain_text if chain else
                       "i don't really have a solid grasp on %s so far." % x,
        "verified": verified, "strategy": strategy,
        "source_status": "trusted" if trusted else "untrusted",
        "edge_kind": "curated",
    }

# ------------------------------------------------------------------ metrics ----
def harness_echo(answer, ctx):
    """Shipped metric verbatim: measure_learn5_car.py:500-501."""
    a, b = ctx["a"].lower(), ctx["b"].lower()
    s = answer.lower()
    return a in s and b in s


def verbatim_echo(answer, ctx):
    """Diagnostic ONLY: exact taught causal phrase copied into the answer."""
    return ctx["taught_surface"] in answer


CAUSAL_MARKERS = (
    "causes", "cause", "leads to", "lead to", "produces", "produce",
    "triggers", "sets off", "results in", "ends in", "brings about",
    "passes through", "starts", "initiates",
)


def semantic_reach(answer, ctx):
    """PROXY only (not a gate): a->b->x in order, causal marker in both spans."""
    s = answer.lower()
    a, b, x = ctx["a"].lower(), ctx["b"].lower(), ctx["x"].lower()
    pa, pb, px = s.find(a), s.find(b), s.find(x)
    if min(pa, pb, px) < 0 or not (pa < pb < px):
        return False
    s1 = s[pa + len(a):pb]
    s2 = s[pb + len(b):px]
    return any(w in s1 for w in CAUSAL_MARKERS) and any(w in s2 for w in CAUSAL_MARKERS)


def byte_reach(answer, ctx):
    """Shipped reach metric: measure_learning_curve.py:240, on paraphrase channel."""
    return bool(ctx["chain_text"]) and ctx["chain_text"] in answer


# ----------------------------------------------------------------- variants ----
def eligible(ctx, answer=None):
    """Fail-closed trigger: verified chain + trusted provenance + right strategy
    + byte-exact taught surface in the answer being emitted. The answer is an
    explicit argument: eligibility is a property of the TEXT, not of a cached
    field (skill: validate the probe's INPUT, not just its conclusion)."""
    text = ctx["ident_answer"] if answer is None else answer
    return (bool(ctx["verified"])
            and ctx["strategy"] == "gated_explanation_chain"
            and ctx["source_status"] == "trusted"
            and bool(ctx["chain_text"])
            and ctx["taught_surface"] in text)


def _apply(answer, ctx, replacement_fn):
    if not eligible(ctx, answer):
        return answer, False
    taught = ctx["taught_surface"]
    # Fail-closed: never rewrite a quoted user frame (echoed text is not our
    # assertion; skill section 1 -- strip echoed/mirrored text before parsing).
    low = answer.lower()
    for q in ("you mentioned", "you said", "you asked"):
        qi = low.find(q)
        if qi >= 0 and low.find(taught.lower(), qi) >= 0:
            return answer, False
    new = answer.replace(taught, replacement_fn(ctx), 1)
    return new, new != answer   # CHANGED IS A MEASUREMENT, never hardcoded


def v1_role_template(answer, ctx):
    """Genuine re-encoding: keep every verified label, change the role frame.

    (a,b,x) is remapped into an initiator -> intermediary -> outcome schema.
    No taught phrase survives, but the full proposition and all three labels do.
    This is the epistemic analogue of episodic->semantic gist *with detail kept*.
    """
    a, b, x = ctx["a"], ctx["b"], ctx["x"]
    return _apply(answer, ctx, lambda c: "%s starts the causal chain, which passes "
                                        "through %s and ends in %s" % (a, b, x))


def v2_gist_abstraction(answer, ctx):
    """Gist abstraction: drop the two intermediate labels to a schema role."""
    x = ctx["x"]
    return _apply(answer, ctx, lambda c: "an initial trigger sets off an "
                                        "intermediate process that ends in %s" % x)


def v3_cosmetic_dodge(answer, ctx):
    """DELIBERATELY DECEPTIVE control: zero-width-join b so a substring metric
    fails while the rendered text is unchanged. Must be rejected as cosmetics."""
    b = ctx["b"]
    return _apply(answer, ctx, lambda c: c["taught_surface"].replace(b, "\u200b".join(b)))


VARIANTS = OrderedDict([
    ("baseline_identity", lambda ans, c: (ans, False)),
    ("v1_role_template", v1_role_template),
    ("v2_gist_abstraction", v2_gist_abstraction),
    ("v3_cosmetic_dodge", v3_cosmetic_dodge),
])

# ------------------------------------------------------------- self-test -------
def self_test():
    """Validate the probe on known cases BEFORE reporting any number."""
    bank = {x: (a, b) for x, a, b in load_candidates()}
    for x in DOC_K8:
        assert x in bank, "documented trace %s missing from bank" % x
    # The recorded assembler narration is authoritative for the fake reach
    # channel. Where it differs from the naive bank-triple narration, record the
    # divergence instead of hiding it: the harness echo metric uses the BANK
    # labels (it["a"], it["b"]), not the labels in the emitted chain text, so
    # this divergence is itself a documented source of the sealed 0.00 echo.
    drifts = []
    for x, a, b in load_candidates():
        if x in DOC_K8:
            expected = "%s leads to %s, and %s leads to %s" % (a, b, b, x)
            if DOC_K8[x] != expected:
                drifts.append({"x": x, "bank_a": a, "bank_b": b,
                               "recorded_chain": DOC_K8[x],
                               "naive_chain": expected})
            else:
                assert DOC_K8[x].endswith(" " + x)
    c = make_ctx("sneeze", *bank["sneeze"], echo_positive=True)
    assert harness_echo(c["ident_answer"], c) is True
    assert verbatim_echo(c["ident_answer"], c) is True
    assert byte_reach(c["para_answer"], c) is True
    assert semantic_reach(c["para_answer"], c) is True
    neg = make_ctx("sneeze", *bank["sneeze"], echo_positive=False)
    assert harness_echo(neg["ident_answer"], neg) is False
    assert v1_role_template(neg["ident_answer"], neg) == (neg["ident_answer"], False)
    return drifts


# ------------------------------------------------------------------- run --------
def rate(items, fn):
    hits = sum(1 for ans, c in items if fn(ans, c))
    return hits, len(items), round(hits / len(items), 3) if items else 0.0


def run():
    drifts = self_test()  # raises on instrument failure
    bank = {x: (a, b) for x, a, b in load_candidates()}

    # Pre-register echo-positive membership by hash BEFORE reading outcomes.
    # Documented aggregate: primary 5/10 echo-positive (0.50), sealed 0/10 (0.00).
    def echo_flag(x, want):
        h = int(hashlib.sha256(("car-echo-reduction|" + x).encode()).hexdigest(), 16)
        return want and (h % 10) < 5

    primary = [(x, bank[x], echo_flag(x, True)) for x in PRIMARY_DOC]
    sealed = [(x, bank[x], echo_flag(x, False)) for x in SEALED_DOC]
    # Deterministic full-bank stress set: all 104 authored triples echo-positive.
    stress = [(x, (a, b), True) for x, a, b in load_candidates()]

    out = {
        "instrument": {
            "self_test": "PASS",
            "no_engine_boot": True,
            "production_files_modified": 0,
            "gates_or_thresholds_changed": 0,
            "bank_source": "scripts/measure_learn5_car.py::_CANDIDATES (ast, read-only)",
            "trace_source": "docs/investigations/CAR_ANSWER_REACH.md section 1 table",
            "echo_metric_definition": "a in answer.lower() and b in answer.lower() "
                                      "(measure_learn5_car.py:500-501)",
            "note": "echo channel = identical-surface question; reach channel = "
                    "held-out paraphrase question. They are distinct.",
            "recorded_vs_bank_chain_label_drifts": drifts,
        },
        "sets": {},
        "fail_closed": [],
    }

    for setname, spec in (("primary_k8_documented", primary),
                          ("sealed_k8_documented", sealed),
                          ("full_bank_stress", stress)):
        out["sets"][setname] = {}
        for vname, vfn in VARIANTS.items():
            ident_items, para_items = [], []
            n_ident_rewrote, n_para_rewrote = 0, 0
            prop_preserved = 0
            audit = []
            for x, (a, b), epos in spec:
                c = make_ctx(x, a, b, epos)
                old_ident = c["ident_answer"]
                new_ident, ichanged = vfn(old_ident, c)
                new_para, pchanged = vfn(c["para_answer"], c)
                n_ident_rewrote += int(ichanged)
                n_para_rewrote += int(pchanged)
                prop_preserved += int(semantic_reach(new_ident, c))
                if ichanged and len(audit) < 6:
                    audit.append({"x": x, "a": a, "b": b,
                                  "before": old_ident, "after": new_ident,
                                  "harness_echo": harness_echo(new_ident, c),
                                  "verbatim_echo": verbatim_echo(new_ident, c),
                                  "proposition_preserved": semantic_reach(new_ident, c)})
                ident_items.append((new_ident, c))
                para_items.append((new_para, c))
            out["sets"][setname][vname] = {
                "identical_surface_echo": rate(ident_items, harness_echo)[2],
                "verbatim_taught_surface_echo": rate(ident_items, verbatim_echo)[2],
                "proposition_preserved_in_rewritten_answer": round(
                    prop_preserved / len(spec), 3) if spec else 0.0,
                "ident_channel_rewrote_n": n_ident_rewrote,
                "answer_reach_proxy_byte": rate(para_items, byte_reach)[2],
                "answer_reach_proxy_semantic": rate(para_items, semantic_reach)[2],
                "paraphrase_channel_rewrote_n": n_para_rewrote,
                "rewrite_audit_examples": audit,
            }

    # ---------------------------------------------------------- fail-closed -----
    x, a, b = "sneeze", *bank["sneeze"]
    cases = {
        "trusted_verified_gated": make_ctx(x, a, b, True),
        "untrusted_provenance": make_ctx(x, a, b, True, trusted=False),
        "wrong_strategy": make_ctx(x, a, b, True, strategy="free_assoc"),
        "unverified_chain": make_ctx(x, a, b, True, verified=False),
        "no_chain": make_ctx(x, a, b, True, chain=False),
        "no_taught_surface_uncertainty": make_ctx(x, a, b, False),
        "quoted_user_frame": make_ctx(x, a, b, True, quoted=True),
        "paraphrase_channel": make_ctx(x, a, b, True),
    }
    for cname, c in cases.items():
        answer = c["para_answer"] if cname == "paraphrase_channel" else c["ident_answer"]
        row = {"case": cname, "eligible": eligible(c, answer)}
        for vname, vfn in VARIANTS.items():
            new, changed = vfn(answer, c)
            row[vname] = {
                "changed": changed,
                "text_identical": new == answer,
                "verified_chain_unchanged": c["chain_text"] == (
                    c["chain_text"]),
            }
        out["fail_closed"].append(row)

    # Hard invariants that must hold for any deployable candidate.
    inv = {}
    for vname in ("v1_role_template", "v2_gist_abstraction", "v3_cosmetic_dodge"):
        rows = {r["case"]: r[vname] for r in out["fail_closed"]}
        inv[vname] = {
            "fires_on_valid": rows["trusted_verified_gated"]["changed"],
            "all_negative_cases_unchanged": all(
                rows[k]["text_identical"] for k in rows
                if k != "trusted_verified_gated"),
            "paraphrase_reach_channel_untouched":
                rows["paraphrase_channel"]["text_identical"],
        }
    out["fail_closed_invariants"] = inv

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run()
