"""WFA decision-routine output contract + adversarial boot-free fixtures.

STRICT SCOPE (do not loosen in this file):
  * NO CognitiveChatEngine boot, NO weight store, NO network, NO engine save.
  * Every production function under test is imported UNMODIFIED from this
    checkout (scripts/measure_learn5_wfa.py, ravana.chat.source_evaluation,
    ravana.chat.engine).  Gate methods are called as UNBOUND functions on
    tiny fakes, exactly as tests/unit/test_learn5_wfa.py does.
  * No gate, threshold, item, prompt or production file is edited.
  * The expected-answer key is sealed away from the key-free metric: it is
    used only in the separate `keyed_audit` block, after all key-free scores
    have been computed.

Question: what output shape must a WFA decision routine produce, which
fixtures exist, and what boot-free test can separate a provenance-reading
routine from an answer-key-reading or echoing one?

Run:  uv run python scratch/wfa_decision_contract_audit.py
Writes scratch/wfa_decision_contract_results.json.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, os.path.join(WT, "ravana", "src"))

import measure_learn5_wfa as w  # noqa: E402  (unmodified production harness)
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402 (import only)
from ravana.chat.source_evaluation import (  # noqa: E402
    DECISION_TOKENS, ClaimContext, Decision, TrustClass, evaluate_source,
)

# The opt-in flag is needed to exercise the REAL emission gate.  Save/restore
# so importing this probe can never change another process's environment.
_SAVED_FLAG = os.environ.get(w.IS_FLAG)
os.environ[w.IS_FLAG] = "1"

OUT = os.path.join(WT, "scratch", "wfa_decision_contract_results.json")
RESULT: dict = {
    "environment": {
        "engine_boot": False,
        "store_opened": False,
        "network": False,
        "production_files_changed": 0,
        "note": "pure functions + tiny fakes; gate methods called unbound",
    },
}

CLAIM = "Mirrors reflect glow through specular reflection."
ITEM = next(i for i in w.build_items() if i["id"] == "g05")
RESULT["item_manifest_sha256"] = w.item_manifest_hash(w.build_items())
RESULT["g05_claim"] = ITEM["claim"]
RESULT["used_claim"] = CLAIM
PROMPT = w.source_question_prompt(CLAIM)
RESULT["prompt"] = PROMPT
# Condition-neutral origin: the shipped is_source_id() embeds the condition
# NAME in the URL, so it cannot be used in a key-free triad.
TRIAL_ORIGIN = "wfa-internal://trial/" + hashlib.sha256(
    ("WFA-CONTRACT|" + CLAIM).encode("utf-8")).hexdigest()[:12]
RESULT["condition_neutral_origin"] = TRIAL_ORIGIN
RESULT["source_id_condition_leak"] = {
    c: (c in w.is_source_id(CLAIM, c)) for c in w.IS_CONDITIONS
}


# ─────────────────────────────────────────────────────────────────────────────
# 0. Output contract, read off the production code (not asserted here: shown)
# ─────────────────────────────────────────────────────────────────────────────
cred_ctx = ClaimContext(
    source=type("S", (), {
        "origin_id": TRIAL_ORIGIN, "trust_class": TrustClass.CREDIBLE,
        "reliability": 0.99, "credibility_token": "credible",
        "has_explicit_credibility": lambda self: True,
    })(),
    prior_decision=Decision.ENDORSE,
)
# Use the real SourceStatus for the contract illustration.
from ravana.chat.source_evaluation import SourceStatus  # noqa: E402

cred_ctx = ClaimContext(
    source=SourceStatus(origin_id=TRIAL_ORIGIN, trust_class=TrustClass.CREDIBLE,
                        reliability=0.99, credibility_token="credible"),
    prior_decision=Decision.ENDORSE,
)
sd = evaluate_source(CLAIM, cred_ctx)
RESULT["output_contract"] = {
    "source_decision_as_dict": sd.as_dict(),
    "token": sd.token,
    "fields": sorted(sd.as_dict().keys()),
    "decision_values": [d.value for d in Decision],
    "token_map": {d.value: DECISION_TOKENS[d] for d in Decision},
    "emission_gate": {
        "ENDORSE": "response byte-identical (silent); audit on _last_source_evaluation",
        "non_ENDORSE": "response == decision.token exactly (Y/N/W/?)",
        "gate_ok": "decision is not None and (ENDORSE or text.strip()==token)",
        "one_turn_context": "_pending_source_context cleared in process_turn finally",
    },
    "legacy_text_parser": {
        "endorsement_labels": sorted(w._ENDORSEMENT_LABELS),
        "withdrawal_labels": sorted(w._WITHDRAWAL_LABELS),
        "forced_choice_labels": sorted(w._FORCED_CHOICE_LABELS),
        "rule": "strip quoted echo frames -> text after LAST Decision: -> "
                "FIRST alphabetic token must be an explicit label; "
                "no all-tokens fallback; else unparseable",
        "no_answer": "separate bucket for forced choice; never folded into unparseable",
    },
    "counter_example": {
        "rule_table_with_no_audit_object": "REJECT - grader cannot grade ENDORSE "
                                           "(emission is deliberately silent)",
        "bare_token_with_echoed_claim": "ACCEPT only if token is first alphabetic "
                                        "token of final span, not inside a quote",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Static re-score of the surviving pre-echo artifact (no engine, no re-run)
#    Reproduces pd_unparseable_rate=1.00 and the algebraic reason R=0.
# ─────────────────────────────────────────────────────────────────────────────
PRE = r"C:/Users/Likhith/Documents/projects/ravana_wfa_v3/scratch/learn5_wfa_probes.jsonl"
rescore = {"source_artifact": PRE, "available": os.path.exists(PRE)}
if rescore["available"]:
    old = [json.loads(line) for line in open(PRE, encoding="utf-8")]
    rescore["records"] = len(old)
    arm_map = (
        ("pd_inclusion_text", "endorsement"),
        ("pd_exclusion_text", "endorsement"),
        ("inv_after", "withdrawal"),
    )
    unparseable, echoed, counted, groups = 0, 0, 0, {}
    for rec in old:
        if rec.get("ctl"):
            continue
        for key, kind in arm_map:
            if key not in rec:
                continue
            d = w.parse_decision(rec.get(key) or "", kind)
            counted += 1
            unparseable += bool(d["unparseable"])
            echoed += bool(d["echo_marker"])
            if key in ("pd_inclusion_text", "pd_exclusion_text"):
                groups.setdefault((rec.get("pool"), key), []).append(
                    d["decision"] is True)
    rescore.update({
        "core_arm_responses": counted,
        "unparseable": unparseable,
        "echo_marked": echoed,
        "pd_unparseable_rate": round(unparseable / counted, 4) if counted else None,
        "echo_marker_rate": round(echoed / counted, 4) if counted else None,
        "endorsements_after_echo_immune_parse": sum(
            sum(v) for v in groups.values()),
        "note": "R = I - E; I and E are endorsement rates from the same "
                "three core arms.  Unparseable is fail-closed to endorse=False "
                "in EngineEndorsementResponder, so I=E=0 => R=0 mechanically. "
                "R=0 here is a parser event, not a source-memory finding.",
    })
    I = sum(groups.get(("primary", "pd_inclusion_text"), []))
    nI = len(groups.get(("primary", "pd_inclusion_text"), []))
    E = sum(groups.get(("primary", "pd_exclusion_text"), []))
    nE = len(groups.get(("primary", "pd_exclusion_text"), []))
    rescore["recomputed_primary_I_E_R"] = {
        "I": round(I / nI, 4) if nI else None,
        "E": round(E / nE, 4) if nE else None,
        "R": round(I / nI - E / nE, 4) if nI and nE else None,
        "n_inclusion": nI, "n_exclusion": nE,
    }
RESULT["pre_echo_artifact_rescore"] = rescore


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tiny graph fake: only the read surface graph_provenance_for_claim uses.
#    No ConceptGraph boot, no store, no file IO.
# ─────────────────────────────────────────────────────────────────────────────
class _Node:
    def __init__(self, label):
        self.label = label


class _Edge:
    def __init__(self, sm, confidence=0.9):
        self.source_metadata = dict(sm)
        self.confidence = confidence
        self.relation_type = "semantic"
        self.timestamp = 1.0
        self.source, self.target = 1, 2


class _Graph:
    def __init__(self, edges=()):
        self.nodes = {1: _Node("mirrors"), 2: _Node("glow")}
        self.edges = dict(edges)

    def get_node(self, nid):
        return self.nodes.get(nid)


class _GateEngine:
    """Fake engine exposing only the pieces the unbound gate methods touch."""
    _source_monitor_decision = staticmethod(
        CognitiveChatEngine._source_monitor_decision)

    def __init__(self, graph, generated="the generated answer"):
        self.graph = graph
        self.generated = generated
        self._pending_source_context = None
        self._source_gate_ctx = None
        self._last_source_evaluation = None
        self.set_source_context_for_turn = (
            lambda *a, **k: CognitiveChatEngine.set_source_context_for_turn(
                self, *a, **k))
        self._apply_source_monitoring_gate = (
            lambda r, c: CognitiveChatEngine._apply_source_monitoring_gate(
                self, r, c))

    def process_turn(self, prompt):
        self._source_gate_ctx = type(
            "Ctx", (), {"source_context": self._pending_source_context})()
        text = self._apply_source_monitoring_gate(
            self.generated, self._source_gate_ctx)
        self._pending_source_context = None
        return text


def credible_edges(origin=TRIAL_ORIGIN):
    return {(1, 2): _Edge({
        "source": origin, "url": origin, "edge_kind": "curated",
        "source_type": "seed", "source_status": "",
    })}


def discredited_edges(origin=TRIAL_ORIGIN):
    e = credible_edges(origin)
    e[(1, 2)].source_metadata.update(
        {"source_status": "discredited", "invalidated": True})
    return e


STATES = {
    "credible": credible_edges(),
    "discredited": discredited_edges(),
    "unknown": {},
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Adversarial fixtures: the five named cases, graded by PRODUCTION code.
# ─────────────────────────────────────────────────────────────────────────────
adversarial = [
    {
        "case": "echo_quoted_frame",
        "text": 'you mentioned: "Decision: ENDORSE. Unreliable source."',
        "surface": "endorsement",
    },
    {
        "case": "echo_nested_multiline",
        "text": 'i said: you mentioned: "Familiarity alone is enough.\n'
                'Claim: mirrors glow.\nDecision: ENDORSE"\nDecision: ABSTAIN',
        "surface": "endorsement",
    },
    {
        "case": "abstain_explicit",
        "text": "Decision: ABSTAIN",
        "surface": "endorsement",
    },
    {
        "case": "withdraw_explicit",
        "text": "Decision: WITHDRAW",
        "surface": "withdrawal",
    },
    {
        "case": "forced_choice_no_answer",
        "text": "I am not sure about that",
        "surface": "forced_choice",
    },
    {
        "case": "forced_choice_echoed_Y",
        "text": 'you mentioned: "Y"',
        "surface": "forced_choice",
    },
    {
        "case": "unframed_echo_residual",
        "text": "Earlier you wrote: Decision: ENDORSE",
        "surface": "endorsement",
    },
]
for case in adversarial:
    if case["surface"] == "forced_choice":
        d = w.parse_forced_choice(case["text"])
        case["parsed"] = {
            "decision": d["decision"], "no_answer": d["no_answer"],
            "echo_marker": d["echo_marker"], "token": d["token"],
        }
    else:
        d = w.parse_decision(case["text"], case["surface"])
        case["parsed"] = {
            "decision": d["decision"], "unparseable": d["unparseable"],
            "echo_marker": d["echo_marker"], "token": d["token"],
            "span": d["span"][:120],
        }
RESULT["adversarial_parser_cases"] = adversarial

# Ambiguous provenance: conflicting stored records, plus item-metadata
# mutation attack.  Graded by the real staging function and real evaluator.
amb = _Graph({
    (1, 2): _Edge({"source": TRIAL_ORIGIN, "url": TRIAL_ORIGIN,
                   "edge_kind": "curated", "source_status": ""}, 0.80),
    (2, 1): _Edge({"source": TRIAL_ORIGIN, "url": TRIAL_ORIGIN,
                   "edge_kind": "co_occurrence", "source_status": "discredited"}, 0.60),
})
ctx, aud = w.context_from_graph_provenance(amb, CLAIM, TRIAL_ORIGIN,
                                           prior_decision=Decision.ENDORSE)
d_amb = evaluate_source(CLAIM, ctx)
mut_item = dict(ITEM)
mut_item.update({"source_status": "credible", "expected_answer": "WRONG",
                 "source_id": "attacker://anything"})
_, aud_mut = w.stage_context_for_item(amb, mut_item, TRIAL_ORIGIN)
RESULT["ambiguous_provenance"] = {
    "staged_audit": aud,
    "decision": d_amb.decision.value,
    "rule": d_amb.rule,
    "token": d_amb.token,
    "item_metadata_mutation_fingerprint_unchanged":
        aud["fingerprint"] == aud_mut["fingerprint"],
    "reading": "an invalidated lower-confidence stored origin outranks a "
               "clean curated edge and forces WITHDRAW; item answer metadata "
               "cannot move the staged fingerprint",
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. KEY-FREE METRIC: Counterfactual Provenance Sensitivity (CPS).
#    Inputs to every routine are byte-identical across states EXCEPT the
#    stored edge provenance.  No condition label, no expected answer, no
#    source_status and no item id is passed to the routine.  The metric itself
#    never reads a key; the keyed audit is computed only afterwards.
# ─────────────────────────────────────────────────────────────────────────────
PAIRS = (("credible", "discredited"), ("credible", "unknown"),
         ("discredited", "unknown"))
NULL_STATES = (("credible", "credible"), ("unknown", "unknown"),
               ("discredited", "discredited"))

# `prior` is FIXED for every state: ENDORSE from the engine's own audit of the
# pre-invalidation trial (here: the real evaluator on the credible state).
prior_ctx, _ = w.context_from_graph_provenance(
    _Graph(credible_edges()), CLAIM, TRIAL_ORIGIN)
PRIOR = evaluate_source(CLAIM, prior_ctx).decision
RESULT["fixed_prior_from_engine_audit"] = PRIOR.value

# All routines receive ONE constant item dict, so any routine that reads item
# answer metadata is blind to the provenance intervention by construction.
FIXED_ITEM = dict(ITEM)
FIXED_ITEM["source_status"] = "credible"
FIXED_ITEM["expected_answer"] = "credible"


def r_echo(claim, graph, origin, item, prompt):
    return {"surface": "text", "text":
            'you mentioned: "%s\\nDecision:"' % prompt[:60],
            "kind": "endorsement"}


def r_abstain(claim, graph, origin, item, prompt):
    return {"surface": "text", "text": "Decision: ABSTAIN",
            "kind": "endorsement"}


def r_withdraw(claim, graph, origin, item, prompt):
    return {"surface": "text", "text": "Decision: WITHDRAW",
            "kind": "withdrawal"}


def r_forced_no_answer(claim, graph, origin, item, prompt):
    return {"surface": "forced_choice", "text": "maybe"}


def r_key_cheat(claim, graph, origin, item, prompt):
    """Tautological wiring: reads the caller-supplied item ANSWER metadata.
    The metadata is held constant in every trial, so the cheat is blind to
    the provenance intervention."""
    status = str(item.get("source_status") or "unknown").lower()
    tok = "Y" if status == "credible" else "N"
    return {"surface": "forced_choice", "text": tok}


def r_graph_gate(claim, graph, origin, item, prompt):
    """Deployable arm: real graph read -> real evaluator -> real emission gate.
    `item` is NOT passed into the staging function: only `claim` is."""
    ctx, audit = w.context_from_graph_provenance(
        graph, claim, origin, prior_decision=PRIOR)
    eng = _GateEngine(graph)
    eng.set_source_context_for_turn(claim, ctx)
    text = eng.process_turn(prompt)
    dec = eng._last_source_evaluation
    return {"surface": "gate", "text": text,
            "decision": dec.decision.value if dec else "INERT",
            "token": dec.token if dec else None,
            "rule": dec.rule if dec else None,
            "audit": audit}


ROUTINES = {
    "echo": r_echo,
    "constant_abstain": r_abstain,
    "constant_withdraw": r_withdraw,
    "forced_choice_no_answer": r_forced_no_answer,
    "answer_key_cheat": r_key_cheat,
    "graph_provenance_gate": r_graph_gate,
}


def surface_of(out):
    if out["surface"] == "gate":
        return out["decision"]
    if out["surface"] == "forced_choice":
        d = w.parse_forced_choice(out["text"])
        return d["decision"]
    d = w.parse_decision(out["text"], out["kind"])
    if d["decision"] is True:
        return "ENDORSE" if out["kind"] == "endorsement" else "WITHDRAW"
    if d["decision"] is False:
        return "ABSTAIN"
    return "UNPARSEABLE"


keyfree = {}
for name, routine in ROUTINES.items():
    runs = {}
    for state, edges in STATES.items():
        graph = _Graph(edges)
        runs[state] = routine(CLAIM, graph, TRIAL_ORIGIN, FIXED_ITEM, PROMPT)
    surf = {s: surface_of(o) for s, o in runs.items()}
    changed = sum(surf[a] != surf[b] for a, b in PAIRS) / len(PAIRS)
    null_changed = sum(
        surface_of(routine(CLAIM, _Graph(STATES[a]), TRIAL_ORIGIN, FIXED_ITEM, PROMPT))
        != surface_of(routine(CLAIM, _Graph(STATES[b]), TRIAL_ORIGIN, FIXED_ITEM, PROMPT))
        for a, b in NULL_STATES) / len(NULL_STATES)
    # prompt byte-identity + label-free (the harness-side anti-leak checks)
    prompts_ok = all(
        w.source_question_prompt(CLAIM) == PROMPT and
        w.prompt_is_label_free(PROMPT, TRIAL_ORIGIN)
        for _ in PAIRS)
    keyfree[name] = {
        "surface_by_state": surf,
        "changed_pair_rate": round(changed, 4),
        "null_changed_rate": round(null_changed, 4),
        "CPS": round(changed - null_changed, 4),
        "prompts_identical_and_label_free": prompts_ok,
    }
RESULT["key_free_metric"] = {
    "definition": "CPS = P(decision changes | ONLY stored provenance changes) "
                  "- P(decision changes | provenance unchanged). "
                  "No expected answer, source_status, condition label, item id "
                  "or condition-bearing origin is read by the routine or metric.",
    "per_routine": keyfree,
    "gate_expectation": "echo / constant abstain / constant withdraw / "
                        "forced-choice no-answer / answer-key cheat -> CPS 0; "
                        "graph-provenance gate -> CPS > 0",
}

# Emission contract check on the one arm that uses the engine gate.
gate_rows = {}
for state, edges in STATES.items():
    eng = _GateEngine(_Graph(edges))
    ctx, audit = w.context_from_graph_provenance(
        eng.graph, CLAIM, TRIAL_ORIGIN, prior_decision=PRIOR)
    eng.set_source_context_for_turn(CLAIM, ctx)
    text = eng.process_turn(PROMPT)
    dec = eng._last_source_evaluation
    gate_rows[state] = {
        "emitted_text": text,
        "decision": dec.decision.value if dec else None,
        "token": dec.token if dec else None,
        "gate_ok": bool(dec is not None and
                        (dec.decision is Decision.ENDORSE or
                         text.strip() == dec.token)),
        "source_context_consumed": eng._pending_source_context is None,
    }
RESULT["emission_contract_rows"] = gate_rows


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mutation / power controls (boot-free).  A metric that cannot MOVE is not
#    measuring the system (ravana-debugging section 1).
# ─────────────────────────────────────────────────────────────────────────────
ctl = {}
# C1: flip stored metadata cred->unknown at fixed item metadata (must move).
g1 = _Graph(credible_edges())
c1a, a1 = w.context_from_graph_provenance(g1, CLAIM, TRIAL_ORIGIN)
d1 = evaluate_source(CLAIM, c1a).decision.value
for _, e in w._iter_graph_edges(g1):
    e.source_metadata["edge_kind"] = "co_occurrence"
c1b, a2 = w.context_from_graph_provenance(g1, CLAIM, TRIAL_ORIGIN)
d2 = evaluate_source(CLAIM, c1b).decision.value
ctl["C1_stored_provenance_flip"] = {
    "before": d1, "after": d2, "moved": d1 != d2,
    "fingerprint_before": a1["fingerprint"], "fingerprint_after": a2["fingerprint"],
}
# C2: invalidation flip with a prior ENDORSE (ENDORSE -> WITHDRAW).
g2 = _Graph(credible_edges())
c2, _ = w.context_from_graph_provenance(g2, CLAIM, TRIAL_ORIGIN)
b2 = evaluate_source(CLAIM, c2).decision.value
g3 = _Graph(discredited_edges())
c3, _ = w.context_from_graph_provenance(g3, CLAIM, TRIAL_ORIGIN,
                                       prior_decision=Decision.ENDORSE)
a3 = evaluate_source(CLAIM, c3).decision.value
ctl["C2_invalidation_flip"] = {
    "before": b2, "after": a3,
    "moved": b2 == "ENDORSE" and a3 == "WITHDRAW",
}
# C3: item-answer metadata mutation must NOT move the graph-derived endpoint.
it2 = dict(ITEM)
it2.update({"source_status": "discredited", "expected_answer": "attacker",
            "source_id": "attacker://x", "reliability": 1.0})
_, base = w.context_from_graph_provenance(
    _Graph(credible_edges()), CLAIM, TRIAL_ORIGIN)
_, mut = w.stage_context_for_item(_Graph(credible_edges()), it2, TRIAL_ORIGIN)
ctl["C3_item_metadata_invariance"] = {
    "base_fingerprint": base["fingerprint"],
    "mutated_fingerprint": mut["fingerprint"],
    "unchanged": base["fingerprint"] == mut["fingerprint"],
}
# C4: power -- a keyed cheat scores high on the ONE endpoint whose key it
# knows, but cannot score on the counterfactual triad.
cheat_surf = keyfree["answer_key_cheat"]["surface_by_state"]
ctl["C4_keyed_endpoint_gaming"] = {
    "single_credible_endpoint_keyed_accuracy": 1.0 if cheat_surf["credible"] == "endorse" else 0.0,
    "triad_keyed_accuracy": round(sum(cheat_surf[s] == tgt for s, tgt in {
        "credible": "endorse", "discredited": "withdraw",
        "unknown": "abstain"}.items()) / 3, 4),
    "CPS": keyfree["answer_key_cheat"]["CPS"],
}
# C5: static source check -- no non-cheat routine reads an answer-key field.
leaky = [n for n, fn in ROUTINES.items()
         if n != "answer_key_cheat" and
         any(tok in inspect.getsource(fn)
             for tok in ("expected_answer", "source_status", "IS_EXPECTED"))]
ctl["C5_no_answer_key_in_graded_routines"] = {
    "leaky_routines": leaky,
    "ok": not leaky,
    "note": "r_key_cheat deliberately reads item.source_status to demonstrate "
            "why the key-free metric must not use keyed accuracy as its score",
}
RESULT["controls"] = ctl


# ─────────────────────────────────────────────────────────────────────────────
# 6. KEYED AUDIT — computed ONLY after every key-free number above exists.
#    Uses the shipped sealed mapping.  This part is explicitly NOT capability
#    evidence: a deterministic rule table can saturate it (G6).
# ─────────────────────────────────────────────────────────────────────────────
KEY = {"credible": "ENDORSE", "discredited": "WITHDRAW", "unknown": "ABSTAIN"}
CANON = {"endorse": "ENDORSE", "abstain": "ABSTAIN", "withdraw": "WITHDRAW",
         "no_answer": "NO_ANSWER", "UNPARSEABLE": "UNPARSEABLE",
         "INERT": "INERT"}
RESULT["keyed_audit_G6_NOT_capability_evidence"] = {
    name: {
        "keyed_accuracy": round(sum(CANON.get(surf, surf) == KEY[s]
                                    for s, surf in
                                    keyfree[name]["surface_by_state"].items()) / 3, 4),
        "CPS": keyfree[name]["CPS"],
    }
    for name in ROUTINES
}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Verdict over pre-registered falsifiers.
# ─────────────────────────────────────────────────────────────────────────────
checks = {
    "artifact_pd_unparseable_rate_is_1":
        rescore.get("pd_unparseable_rate") == 1.0,
    "artifact_R_is_0":
        rescore.get("recomputed_primary_I_E_R", {}).get("R") == 0.0,
    "echo_case_unparseable":
        adversarial[0]["parsed"]["unparseable"] is True,
    "echo_nested_then_explicit_abstain_parsed": (
        adversarial[1]["parsed"]["decision"] is False and
        adversarial[1]["parsed"]["echo_marker"] is True),
    "abstain_case_parseable":
        adversarial[2]["parsed"]["decision"] is False and
        adversarial[2]["parsed"]["unparseable"] is False,
    "withdraw_case_parseable":
        adversarial[3]["parsed"]["decision"] is True and
        adversarial[3]["parsed"]["unparseable"] is False,
    "forced_no_answer_separate":
        adversarial[4]["parsed"]["no_answer"] is True and
        "unparseable" not in adversarial[4]["parsed"],
    "echoed_Y_not_an_endorsement":
        adversarial[5]["parsed"]["no_answer"] is True,
    "ambiguous_provenance_reads_graph":
        RESULT["ambiguous_provenance"]["decision"] == "WITHDRAW" and
        RESULT["ambiguous_provenance"]["rule"] == "R_SOURCE_INVALIDATED",
    "item_metadata_cannot_move_fingerprint":
        ctl["C3_item_metadata_invariance"]["unchanged"] is True,
    "provenance_flip_moves_decision":
        ctl["C1_stored_provenance_flip"]["moved"] is True and
        ctl["C2_invalidation_flip"]["moved"] is True,
    "no_answer_key_in_graded_routines": ctl["C5_no_answer_key_in_graded_routines"]["ok"],
    "all_adversaries_CPS_zero":
        all(keyfree[n]["CPS"] == 0.0 for n in (
            "echo", "constant_abstain", "constant_withdraw",
            "forced_choice_no_answer", "answer_key_cheat")),
    "graph_gate_CPS_positive": keyfree["graph_provenance_gate"]["CPS"] > 0,
    "gate_ok_all_states": all(v["gate_ok"] for v in gate_rows.values()),
    "one_turn_context_consumed": all(
        v["source_context_consumed"] for v in gate_rows.values()),
    "prompts_identical_label_free": all(
        v["prompts_identical_and_label_free"] for v in keyfree.values()),
    "source_id_embeds_condition": any(RESULT["source_id_condition_leak"].values()),
    "unframed_echo_accepted_residual":
        adversarial[6]["parsed"]["decision"] is True and
        adversarial[6]["parsed"]["echo_marker"] is False,
}
RESULT["checks"] = checks
RESULT["verdict"] = ("AUDIT_COMPLETE" if all(checks.values())
                     else "CHECK_FAILED: " +
                     ", ".join(k for k, v in checks.items() if not v))

FIXTURES_OUT = os.path.join(WT, "scratch", "wfa_adversarial_fixtures.json")
with open(FIXTURES_OUT, "w", encoding="utf-8") as fh:
    json.dump({
        "generated_by": "scratch/wfa_decision_contract_audit.py",
        "engine_boot": False,
        "five_named_cases": adversarial,
        "ambiguous_provenance": RESULT["ambiguous_provenance"],
        "key_free_triad": {
            "origin": TRIAL_ORIGIN, "prompt": PROMPT,
            "constant_item": FIXED_ITEM,
            "states": {k: {"edge_metadata": [
                dict(e.source_metadata) for e in v.values()]}
                for k, v in STATES.items()},
            "metric": RESULT["key_free_metric"]["definition"],
        },
    }, fh, indent=2, ensure_ascii=False, default=str)

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(RESULT, fh, indent=2, ensure_ascii=False, default=str)
if _SAVED_FLAG is None:
    os.environ.pop(w.IS_FLAG, None)
else:
    os.environ[w.IS_FLAG] = _SAVED_FLAG
print(json.dumps({
    "verdict": RESULT["verdict"],
    "checks": checks,
    "pd_unparseable_rate": rescore.get("pd_unparseable_rate"),
    "recomputed_primary_I_E_R": rescore.get("recomputed_primary_I_E_R"),
    "CPS": {k: v["CPS"] for k, v in keyfree.items()},
    "outputs": OUT,
}, indent=2, ensure_ascii=False))
