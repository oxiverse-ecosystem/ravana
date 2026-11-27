#!/usr/bin/env python3
"""M3 boot-free self-test. NO engine boot, NO store write, NO network.

Imports the unmodified production constants/predicates and the scratch M3
reader, builds synthetic PROVENANCE-BEARING evidence (an oracle-shaped
reachability check, NON-DEPLOYABLE), and verifies the reader contract plus the
READ/SHAM call-count control. Exits 0 only when every check passes.
"""
from __future__ import annotations

import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (HERE, os.path.join(REPO, "ravana_ml", "src"),
           os.path.join(REPO, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m3_graded_probe as m3  # noqa: E402
from ravana.chat.schema_induction import (  # noqa: E402
    CONGRUENCE_FLOOR, STRICT_MARGIN, _is_trusted_edge)
from ravana.chat.source_evaluation import Decision  # noqa: E402

PROV = {"edge_kind": "curated", "source": "conversation",
        "source_type": "user", "source_id": "wikidata-Q8074"}
OPTS = ("snow storm", "party", "intersection", "natural phenomenon")


def rec(label="snow storm", *, subject="blizzard", relation="is_a",
        confidence=0.8, provenance=None, negated=None, surface=None,
        stamp=None, kind="m2", source_status="credible",
        prior_decision="ENDORSE", invalidated=False, cue_present=True,
        familiarity=0.0, gold=None, **extra):
    out = {
        "record_id": "r_%s_%s" % (kind, str(label).replace(" ", "_")),
        "subject": subject, "relation": relation, "label": label,
        "provenance": dict(provenance or PROV), "confidence": confidence,
    }
    if negated is not None:
        out["negated"] = negated
    if surface is not None:
        out["surface"] = surface
    if stamp is not None:
        out["stamp"] = stamp
    if gold is not None:
        out["gold"] = gold  # decoy: reader must never read this
    if kind == "m1":
        out.update(source_status=source_status, prior_decision=prior_decision,
                   invalidated=invalidated, cue_present=cue_present,
                   familiarity=familiarity,
                   claim="%s is a %s." % (subject, label))
    out.update(extra)
    return out


def bundle(m1=(), m2=()):
    return m3.EvidenceBundle(tuple(m1), tuple(m2))


def pick(subject, evidence, *, trust=_is_trusted_edge, options=OPTS):
    d = m3.read_graded_probe(subject, options, evidence, trust=trust)
    return d


def run():
    checks = {}

    # 0. Boot-free: importing a production package (ravana.chat.__init__) loads
    #    the engine MODULE, so "no engine import" is not a valid criterion.
    #    The checkable property is that the reader never CONSTRUCTS an engine,
    #    boots a store, issues a turn or imports a harness.
    src = inspect.getsource(m3)
    banned_calls = ("CognitiveChatEngine(", "process_turn(", "urlopen",
                    "requests.", "sqlite3", "pickle.load", "measure_tra_k")
    checks["no_engine_boot_in_reader_source"] = not any(t in src for t in banned_calls)
    checks["no_harness_imported"] = not (
        {"measure_tra_k", "trak_live_ab_probe"} & set(sys.modules))

    # 1. Contract: no answer-key inputs anywhere in the public reader surface.
    reader_params = set(inspect.signature(m3.read_graded_probe).parameters)
    hook_params = set(inspect.signature(m3.graded_probe_pick).parameters)
    bundle_fields = set(m3.EvidenceBundle.__dataclass_fields__)
    banned = {"item", "gold", "world_model", "type_sets", "graph", "eng"}
    checks["no_answer_key_or_graph_inputs"] = not (
        (banned & reader_params) or (banned & hook_params) or
        (banned & bundle_fields))
    checks["no_oracle_fields_in_bundle"] = not ({"gold", "world_model", "type_sets"} & bundle_fields)

    # 2. Production constants and predicate are reused, not re-set.
    checks["production_constants_reused"] = (
        m3.CONGRUENCE_FLOOR == CONGRUENCE_FLOOR == 0.20
        and m3.STRICT_MARGIN == STRICT_MARGIN == 0.10
        and m3._is_trusted_edge is _is_trusted_edge)

    # 3. M2 source-native label is the cue even with NO redundant stamp.
    d = pick("blizzard", bundle(m2=[rec()]))
    checks["m2_source_native_label_admits"] = d.pick == "snow storm"
    checks["m2_no_stamp_still_admits"] = d.scores.get("snow storm") == 0.8
    checks["production_reason_graded"] = d.reason == "admitted_graded_probe_winner"

    # 4. M1 prior-act record: only ENDORSE clears the unmodified evaluator.
    d1 = pick("blizzard", bundle(m1=[rec(kind="m1")]))
    checks["m1_endorse_admits"] = d1.pick == "snow storm"
    for name, kw in (
        ("m1_unknown_source_rejected", {"source_status": "unknown"}),
        ("m1_discredited_rejected", {"source_status": "discredited"}),
        ("m1_invalidated_rejected", {"invalidated": True}),
        ("m1_unattributed_origin_rejected",
         {"origin_id": "", "provenance": {"edge_kind": "curated", "source": "conversation"}}),
        ("m1_familiarity_ignored", {"source_status": "unknown", "familiarity": 0.99}),
    ):
        checks[name] = pick("blizzard", bundle(m1=[rec(kind="m1", **kw)])).pick is None

    # 5. Provenance trust is mandatory for M2; wrong tag defers.
    bad_prov = {"edge_kind": "unknown", "source": "system", "source_type": "system"}
    checks["m2_untrusted_defers"] = pick(
        "blizzard", bundle(m2=[rec(provenance=bad_prov)])).pick is None
    checks["m2_wrong_tag_trust_defers"] = pick(
        "blizzard", bundle(m2=[rec()]), trust=m3.wrong_tag_trust).pick is None

    # 6. Source-native label handling: case preserved, stamp must agree.
    dcase = pick("blizzard", bundle(m2=[rec(label="Snow Storm")]))
    checks["source_native_case_preserved"] = dcase.pick == "snow storm"
    checks["contradictory_stamp_refused"] = pick(
        "blizzard", bundle(m2=[rec(stamp="earthquake")])).pick is None
    checks["agreeing_stamp_admits"] = pick(
        "blizzard", bundle(m2=[rec(stamp="SNOW STORM")])).pick == "snow storm"

    # 7. Explicit negation veto (flag and surface), untrusted negation ignored.
    neg_hi = rec(label="snow storm", confidence=0.95, negated=True)
    pos_lo = rec(label="snow storm", confidence=0.5)
    dn = pick("blizzard", bundle(m2=[neg_hi, pos_lo]))
    checks["negation_veto_beats_higher_positive"] = (
        dn.pick is None and dn.reason == "all_candidates_negation_vetoed"
        and "snow storm" in dn.vetoed)
    dsurf = pick("blizzard", bundle(m2=[rec(surface="blizzard is not a snow storm.")]))
    checks["negation_surface_detected"] = (
        dsurf.pick is None and "snow storm" in dsurf.vetoed)
    checks["untrusted_negation_no_veto"] = pick(
        "blizzard", bundle(m2=[rec(negated=True, provenance=bad_prov)]),
        trust=_is_trusted_edge).pick is None

    # 8. Negation is label-scoped, not a global word ban: another label wins.
    d_other = m3.read_graded_probe(
        "blizzard", OPTS, bundle(m2=[neg_hi, rec(label="party", confidence=0.9)]))
    checks["negation_scoped_to_label"] = d_other.pick == "party"

    # 9. Floor / margin / subject / option-space fail-closed behaviour.
    checks["below_floor_defers"] = pick(
        "blizzard", bundle(m2=[rec(confidence=CONGRUENCE_FLOOR - 0.01)])).pick is None
    tie = bundle(m2=[rec(confidence=0.9),
                     rec(label="party", confidence=0.9)])
    checks["tie_defers"] = pick("blizzard", tie).pick is None
    checks["unrelated_subject_defers"] = pick(
        "zeppelin", bundle(m2=[rec()])).pick is None
    checks["label_outside_options_defers"] = pick(
        "blizzard", bundle(m2=[rec(label="earthquake")])).pick is None

    # 10. Anti-tautology: decoy gold/type_sets/world_model cannot move the pick.
    checks["answer_key_invariant"] = pick(
        "blizzard", bundle(m2=[rec(gold="party", type_sets={"blizzard": ["party"]},
                                   world_model={"blizzard": "party"})])).pick == "snow storm"

    # 11. Default-off: no production call, no evidence scan.
    c = m3.new_counters()
    off = m3.graded_probe_pick("blizzard", OPTS, bundle(m2=[rec()]), counters=c)
    checks["default_off_no_pick"] = off is None
    checks["default_off_no_work"] = (c["production_calls"] == 0
                                     and c["scorer_calls"] == 0
                                     and c["records_scanned"] == 0)

    # 12. Structural deferral to production: production pick is returned UNTOUCHED.
    c = m3.new_counters()
    dep = m3.graded_probe_pick("blizzard", OPTS, bundle(m2=[rec()]),
                               production_pick=lambda s, o: "party",
                               enabled=True, counters=c)
    checks["deferral_returns_production_pick"] = (
        dep is not None and dep.pick == "party" and dep.source == "production")
    checks["deferral_skips_new_scorer"] = (
        c["scorer_calls"] == 0 and c["deferred_to_production"] == 1
        and c["records_scanned"] == 0)

    # 13. Matched READ/SHAM controls: equal calls and scans, SHAM admits nothing.
    ev = bundle(m1=[rec(kind="m1")], m2=[rec(), rec(label="party", confidence=0.4)])
    cr, cs = m3.new_counters(), m3.new_counters()
    dr = m3.graded_probe_pick("blizzard", OPTS, ev, enabled=True,
                              trust=_is_trusted_edge, counters=cr)
    ds = m3.graded_probe_pick("blizzard", OPTS, ev, enabled=True,
                              trust=m3.wrong_tag_trust, counters=cs)
    matched = ("production_calls", "scorer_calls", "trust_calls", "records_scanned",
               "m1_scanned", "m2_scanned")
    checks["read_picks"] = dr is not None and dr.pick == "snow storm"
    checks["sham_defers"] = ds is not None and ds.pick is None
    checks["sham_call_count_matched"] = all(cr[k] == cs[k] for k in matched)
    checks["sham_scan_count_nonzero"] = cr["records_scanned"] >= 3

    # 14. Flag parity: env var opt-in matches explicit enabled=True.
    os.environ[m3.M3_ENV_FLAG] = "1"
    try:
        c = m3.new_counters()
        env_on = m3.graded_probe_pick("blizzard", OPTS, bundle(m2=[rec()]), counters=c)
        checks["env_flag_opt_in"] = env_on is not None and env_on.pick == "snow storm"
    finally:
        os.environ.pop(m3.M3_ENV_FLAG, None)

    return checks


def main():
    checks = run()
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    result = {
        "self_test": "PASS" if passed == total else "FAIL",
        "checks_passed": passed,
        "checks_total": total,
        "failed": sorted(k for k, v in checks.items() if not v),
        "boot_free": {
            "no_engine_construction_in_reader": checks.get("no_engine_boot_in_reader_source"),
            "no_harness_import": checks.get("no_harness_imported"),
        },
        "production_constants": {
            "congruence_floor": CONGRUENCE_FLOOR, "strict_margin": STRICT_MARGIN},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
