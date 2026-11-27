"""Unit tests for the M3 negation-aware graded-probe reader (SCRATCH default-off).

No engine is booted, no store is written, no network is used. The module under
test is imported from this worktree's scratch/ directory by absolute path, so a
green run cannot silently test the parent checkout's editable install.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_WT = Path(__file__).resolve().parents[2]
for _p in (str(_WT / "scratch"), str(_WT / "ravana" / "src"),
           str(_WT / "ravana_ml" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m3_graded_probe as m3  # noqa: E402
from ravana.chat.schema_induction import (  # noqa: E402
    CONGRUENCE_FLOOR, STRICT_MARGIN, _is_trusted_edge)


def test_imported_from_this_worktree():
    assert Path(m3.__file__).resolve() == (_WT / "scratch" / "m3_graded_probe.py").resolve()


PROV = {"edge_kind": "curated", "source": "conversation",
        "source_type": "user", "source_id": "Q8074"}
OPTS = ("snow storm", "party", "intersection", "natural phenomenon")
BAD_PROV = {"edge_kind": "unknown", "source": "system", "source_type": "system"}


def make_record(label="snow storm", *, subject="blizzard", relation="is_a",
                confidence=0.8, provenance=None, negated=None, surface=None,
                stamp=None, kind="m2", source_status="credible",
                prior_decision="ENDORSE", invalidated=False, origin_id=None,
                **extra):
    rec = {
        "record_id": "r_%s" % str(label).replace(" ", "_"),
        "subject": subject, "relation": relation, "label": label,
        "provenance": dict(provenance if provenance is not None else PROV),
        "confidence": confidence,
    }
    if negated is not None:
        rec["negated"] = negated
    if surface is not None:
        rec["surface"] = surface
    if stamp is not None:
        rec["stamp"] = stamp
    if kind == "m1":
        rec.update(source_status=source_status, prior_decision=prior_decision,
                   invalidated=invalidated, cue_present=True,
                   claim="%s is a %s." % (subject, label))
    if origin_id is not None:
        rec["origin_id"] = origin_id
    rec.update(extra)
    return rec


def run(subject="blizzard", options=OPTS, m1=(), m2=(), *, trust=_is_trusted_edge):
    return m3.read_graded_probe(
        subject, options,
        m3.EvidenceBundle(tuple(m1), tuple(m2)), trust=trust)


# --- contract / anti-leak ----------------------------------------------------
def test_reader_signature_has_no_answer_key_or_graph_input():
    params = set(inspect.signature(m3.read_graded_probe).parameters)
    assert not ({"item", "gold", "world_model", "type_sets", "graph", "eng"} & params)


def test_evidence_bundle_has_no_oracle_fields():
    fields = set(m3.EvidenceBundle.__dataclass_fields__)
    assert not ({"gold", "world_model", "type_sets", "item"} & fields)


def test_production_constants_and_predicate_are_reused():
    assert m3.CONGRUENCE_FLOOR == CONGRUENCE_FLOOR == 0.20
    assert m3.STRICT_MARGIN == STRICT_MARGIN == 0.10
    assert m3._is_trusted_edge is _is_trusted_edge


def test_decoy_gold_and_type_sets_cannot_move_the_pick():
    rec = make_record(gold="party", world_model={"blizzard": "party"},
                      type_sets={"blizzard": ["party"]})
    assert run(m2=[rec]).pick == "snow storm"


def test_no_engine_construction_or_turn_in_reader_source():
    src = inspect.getsource(m3)
    assert not any(t in src for t in
                   ("CognitiveChatEngine(", "process_turn(", "urlopen", "sqlite3"))


# --- admission ---------------------------------------------------------------
def test_m2_source_native_label_admits_without_redundant_stamp():
    d = run(m2=[make_record()])
    assert d.pick == "snow storm"
    assert d.scores == {"snow storm": 0.8}
    assert d.reason == "admitted_graded_probe_winner"


def test_m2_untrusted_provenance_defers():
    assert run(m2=[make_record(provenance=BAD_PROV)]).pick is None


def test_m1_endorse_admits_but_unattributed_origin_is_rejected():
    assert run(m1=[make_record(kind="m1")]).pick == "snow storm"
    assert run(m1=[make_record(kind="m1", provenance={"edge_kind": "curated",
                                                      "source": "conversation"},
                               origin_id="")]).pick is None


@pytest.mark.parametrize("kw", [
    {"source_status": "unknown"},
    {"source_status": "discredited"},
    {"invalidated": True},
    {"source_status": "unknown", "familiarity": 0.99},
])
def test_m1_non_endorsed_never_admits(kw):
    assert run(m1=[make_record(kind="m1", **kw)]).pick is None


@pytest.mark.parametrize("relation", ["causes", "part_of", "resembles", ""])
def test_non_is_a_relation_never_admits(relation):
    assert run(m2=[make_record(relation=relation)]).pick is None


# --- source-native label handling -------------------------------------------
def test_source_native_case_is_preserved_in_output_option():
    assert run(m2=[make_record(label="Snow Storm")]).pick == "snow storm"


def test_contradictory_stamp_refuses_but_matching_stamp_admits():
    assert run(m2=[make_record(stamp="earthquake")]).pick is None
    assert run(m2=[make_record(stamp="SNOW STORM")]).pick == "snow storm"


def test_label_must_be_in_displayed_options():
    assert run(m2=[make_record(label="earthquake")]).pick is None


# --- negation veto -----------------------------------------------------------
def test_admitted_negation_vetoes_even_higher_confidence_positive():
    d = run(m2=[make_record(confidence=0.95, negated=True),
                make_record(confidence=0.5)])
    assert d.pick is None
    assert d.reason == "all_candidates_negation_vetoed"
    assert "snow storm" in d.vetoed
    assert d.scores["snow storm"] == 0.5  # positive retained in audit only


def test_negation_is_detected_from_teaching_surface():
    d = run(m2=[make_record(surface="blizzard is not a snow storm.")])
    assert d.pick is None and "snow storm" in d.vetoed


def test_untrusted_negation_cannot_veto():
    d = run(m2=[make_record(negated=True, provenance=BAD_PROV)])
    assert d.pick is None and d.vetoed == ()


def test_negation_is_label_scoped_not_a_global_word_ban():
    d = run(m2=[make_record(negated=True, confidence=0.95),
                make_record(label="party", confidence=0.9)])
    assert d.pick == "party"


def test_m1_negation_vetoes_too():
    d = run(m1=[make_record(kind="m1", negated=True)])
    assert d.pick is None and "snow storm" in d.vetoed


# --- fail-closed grading -----------------------------------------------------
def test_below_floor_and_tie_defer():
    assert run(m2=[make_record(confidence=CONGRUENCE_FLOOR - 0.01)]).pick is None
    tie = [make_record(confidence=0.9),
           make_record(label="party", confidence=0.9)]
    assert run(m2=tie).pick is None


def test_unrelated_subject_defers():
    assert run(subject="zeppelin", m2=[make_record()]).pick is None


# --- hook: default-off, deferral, controls ----------------------------------
def test_hook_is_default_off_and_touches_nothing():
    counters = m3.new_counters()
    ev = m3.EvidenceBundle(m2_subject_evidence=(make_record(),))
    assert m3.graded_probe_pick("blizzard", OPTS, ev, counters=counters) is None
    assert counters["production_calls"] == counters["scorer_calls"] == 0
    assert counters["records_scanned"] == 0


def test_hook_defers_to_existing_production_pick_without_scoring():
    counters = m3.new_counters()
    ev = m3.EvidenceBundle(m2_subject_evidence=(make_record(),))
    d = m3.graded_probe_pick("blizzard", OPTS, ev, enabled=True,
                             production_pick=lambda s, o: "party", counters=counters)
    assert d is not None and d.pick == "party" and d.source == "production"
    assert counters["scorer_calls"] == 0 and counters["records_scanned"] == 0
    assert counters["deferred_to_production"] == 1


def test_read_sham_have_matched_call_counts_and_sham_admits_nothing():
    ev = m3.EvidenceBundle(
        m1_records=(make_record(kind="m1"),),
        m2_subject_evidence=(make_record(), make_record(label="party", confidence=0.4)))
    cr, cs = m3.new_counters(), m3.new_counters()
    dr = m3.graded_probe_pick("blizzard", OPTS, ev, enabled=True, counters=cr)
    ds = m3.graded_probe_pick("blizzard", OPTS, ev, enabled=True,
                              trust=m3.wrong_tag_trust, counters=cs)
    assert dr is not None and dr.pick == "snow storm"
    assert ds is not None and ds.pick is None
    for key in ("production_calls", "scorer_calls", "trust_calls",
                "records_scanned", "m1_scanned", "m2_scanned"):
        assert cr[key] == cs[key], key
    assert cr["records_scanned"] >= 3


def test_env_flag_opt_in_matches_explicit_enable(monkeypatch):
    monkeypatch.setenv(m3.M3_ENV_FLAG, "1")
    counters = m3.new_counters()
    d = m3.graded_probe_pick("blizzard", OPTS,
                             m3.EvidenceBundle(m2_subject_evidence=(make_record(),)),
                             counters=counters)
    assert d is not None and d.pick == "snow storm"
    assert counters["flag_on"] == 1
