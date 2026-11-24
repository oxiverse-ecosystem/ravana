"""Unit tests for the UNSATURATED RAVANA-LEARN-5 WFA harness.

These pin the load-bearing claims of scripts/measure_learn5_wfa.py:
  * the gold claim is a PARAPHRASE of the verbatim source span, never the span
  * the frozen paraphrase-entailment rubric finds the source and rejects
    near-miss (same-topic, wrong-fact) decoy claims
  * decoys / closed-web items abstain; internal claims carry no web tag
  * the answer-only arm is the LIVE engine with web DISABLED and is graded
    against the same gold facts -- exercised here with a FAKE engine only
  * the decision rule FAILS on fabrication, low validity, sealed drop, web
    leakage or an under-sized bank; the hash split is stable across builds
No network, no engine boot and no engine save are involved. All fakes are
local; the frozen replay boundary is reused from measure_fronts_web_frozen.
"""
import os
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(PROJ, "scripts")
for _p in (PROJ, SCRIPTS, os.path.join(PROJ, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_learn5_wfa as wfa  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.chat.source_evaluation import (  # noqa: E402
    Decision,
    evaluate_source,
)


@pytest.fixture(scope="module")
def items():
    return wfa.build_items()


# ── item-bank validity and sample size (design section 4.5) ─────────────────
def test_all_items_validate(items):
    assert wfa.validate_items(items) == []


def test_pool_sizes_meet_unsaturation_minima(items):
    for pool in ("primary", "sealed"):
        n_gap = sum(1 for i in items if i["pool"] == pool and i["kind"] == "gap")
        n_closed = sum(1 for i in items if i["pool"] == pool and i["kind"] == "closed_web")
        n_decoy = sum(1 for i in items if i["pool"] == pool and i["kind"] == "decoy")
        assert n_gap >= wfa.MIN_GAP_PER_POOL, (pool, n_gap)
        assert n_closed >= wfa.MIN_CLOSED_PER_POOL, (pool, n_closed)
        assert n_decoy >= wfa.MIN_DECOY_PER_POOL, (pool, n_decoy)


def test_gap_gold_claim_is_a_paraphrase_of_source(items):
    gaps = [i for i in items if i["kind"] == "gap"]
    assert gaps
    for it in gaps:
        assert it["source_span"] in it["content"], it["id"]
        assert it["url"] in wfa.ALL_FIXTURE_URLS, it["id"]
        # NOT the tautology the old harness had:
        assert it["claim"] != it["source_span"], it["id"]
        assert it["claim"] not in it["content"], it["id"]
        # genuine surface change, and canonicalisation restores source facts
        assert it["paraphrase_subs"] >= wfa.MIN_PARAPHRASE_SUBS, it["id"]
        assert it["claim_tokens"] == it["entail_tokens"], it["id"]
        assert it["key_fact_tokens"], it["id"]


def test_decoys_are_near_miss_and_non_entailing(items):
    probe = wfa.WFAComponent(None)
    for it in items:
        if it["kind"] != "decoy":
            continue
        src_words = set()
        for h in wfa.FIX["queries"][wfa.NORM(it["query"])]["hits"][:2]:
            src_words |= wfa._words(h["content"])
            assert probe.find_entailing_span(
                h["content"], set(it["claim_tokens"])) is None, it["id"]
        assert len(src_words & set(it["claim_tokens"])) >= 2, it["id"]


def test_closed_web_queries_absent_from_fixtures(items):
    for it in items:
        if it["kind"] == "closed_web":
            assert wfa.NORM(it["query"]) not in wfa.FIX["queries"], it["id"]


def test_hash_split_is_stable(items):
    h = wfa.item_manifest_hash(items)
    assert re_is_hex(h)
    assert h == wfa.item_manifest_hash(wfa.build_items())
    # Pre-run seal: unchanged by the additive source_status/source_id fields.
    assert h == "119b9d91fe43a06087d001b1fb02a554edd4d08d665b99041528d93533818164"


def re_is_hex(s):
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


# ── paraphrase rubric / component behaviour ─────────────────────────────────
def test_paraphraser_produces_distinct_claim_with_restorable_tokens():
    source = "A compass points north because the Earth behaves like a giant bar magnet."
    claim, tokens, nsubs = wfa._paraphrase(source)
    assert claim != source
    assert nsubs >= 1
    assert tokens == sorted(wfa._words(source))


def test_entailment_rubric_matches_paraphrase_and_rejects_wrong_fact():
    comp = wfa.WFAComponent(None)
    content = ("A compass points north because the Earth behaves like a giant "
               "bar magnet.")
    claim, tokens, _ = wfa._paraphrase(content)
    assert comp.find_entailing_span(content, tokens) is not None
    # near-miss: same topic, wrong fact -> canonical tokens not all present
    wrong = "The compass points north because a tachyonic aether current pulls it."
    assert comp.find_entailing_span(content, wfa._words(wrong)) is None


def test_fabricated_url_rejected_by_allowlist():
    assert "https://fabricated.example.invalid/fact" not in wfa.ALL_FIXTURE_URLS


def test_decoy_run_abstains_without_url(items):
    comp = wfa.WFAComponent(wfa.FROZEN_SEARCH)
    for it in [i for i in items if i["kind"] == "decoy"][:4]:
        rec = comp.run_item(it)
        assert rec["verdict"] == "abstain"
        assert rec["url"] is None and rec["span"] is None


def test_closed_web_run_abstains_without_url(items):
    comp = wfa.WFAComponent(wfa.FROZEN_SEARCH)
    for it in [i for i in items if i["kind"] == "closed_web"][:4]:
        rec = comp.run_item(it)
        assert rec["verdict"] == "abstain"
        assert rec["url"] is None


def test_internal_run_carries_no_web_tag(items):
    comp = wfa.WFAComponent(wfa.FROZEN_SEARCH)
    for it in [i for i in items if i["kind"] == "internal"][:4]:
        rec = comp.run_item(it)
        assert rec["url"] is None and rec["external_tag"] is False


def test_gap_run_emits_paraphrase_not_the_source_span(items):
    comp = wfa.WFAComponent(wfa.FROZEN_SEARCH)
    for it in [i for i in items if i["kind"] == "gap"][:6]:
        rec = comp.run_item(it)
        assert rec["verdict"] == "attributed"
        assert rec["url"] in wfa.ALL_FIXTURE_URLS
        assert rec["url"] in _urls_for(it["query"])
        assert rec["span"] == it["source_span"]
        assert rec["output"] == it["claim"] != it["source_span"]
        assert rec["external_tag"] is True


def _urls_for(query):
    return {h.get("url") for h in wfa.FIX["queries"][wfa.NORM(query)]["hits"]}


# ── answer-only arm: LIVE engine with web disabled (FAKES, no boot) ─────────
class _FakeEngine:
    """A stand-in that only exposes the engine method the oracle calls."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.turns = []

    def process_turn(self, q):
        self.turns.append(q)
        return self.answers.get(q, "")


def test_answer_only_oracle_uses_injected_engine_not_a_boot():
    q = "how do lungs exchange gases"
    oracle = wfa.AnswerOnlyOracle(engine=_FakeEngine({q: "the parametric answer"}),
                                  web_disabled=True)
    assert oracle.answer(q) == "the parametric answer"
    assert oracle.calls == 1
    assert oracle.engine.turns == [q]


def test_answer_only_oracle_surfaces_errors_without_crashing():
    class Broken:
        def process_turn(self, q):
            raise RuntimeError("boom")
    oracle = wfa.AnswerOnlyOracle(engine=Broken(), web_disabled=True)
    assert oracle.answer("x").startswith("<ERR RuntimeError")


def test_answer_asserts_fact_uses_key_fact_tokens(items):
    gap = next(i for i in items if i["kind"] == "gap")
    assert wfa.answer_asserts_fact(gap["source_span"], gap) is True
    assert wfa.answer_asserts_fact("", gap) is False
    assert wfa.answer_asserts_fact("<ERR RuntimeError: x>", gap) is False
    assert wfa.answer_asserts_fact("a completely unrelated topic about keyboards",
                                  gap) is False


def test_run_answer_only_rejects_web_enabled_arm(items):
    gap = next(i for i in items if i["kind"] == "gap")
    rec = {"id": gap["id"], "kind": "gap", "pool": gap["pool"]}
    oracle = wfa.AnswerOnlyOracle(
        engine=_FakeEngine({gap["query"]: gap["source_span"]}),
        web_disabled=False)   # web ON -> must NOT earn the baseline
    wfa.run_answer_only([gap], oracle, [rec])
    assert oracle.calls == 1
    assert rec["answer_only_output"] == gap["source_span"]
    assert rec["answer_only_correct"] is False


def test_run_answer_only_grades_web_off_parametric_answer(items):
    gap = next(i for i in items if i["kind"] == "gap")
    rec = {"id": gap["id"], "kind": "gap", "pool": gap["pool"]}
    oracle = wfa.AnswerOnlyOracle(
        engine=_FakeEngine({gap["query"]: gap["source_span"]}),
        web_disabled=True)
    wfa.run_answer_only([gap], oracle, [rec])
    assert rec["answer_only_correct"] is True


# ── decision rule: fabrication / weakness / leakage flip FAIL ──────────────
def _base_summary():
    per = {
        "n_gap": 65, "n_closed": 30, "n_decoy": 30, "n_internal": 10,
        "wfa_valid": 1.0, "answer_only": 0.0, "delta": 1.0,
        "abstain": 1.0, "decoy_reject": 1.0, "near_miss_decoy_reject": 1.0,
        "internal_no_tag": 1.0, "reads_total": 1, "reads_mean_gap": 1.0,
        # additive PD fields: a clean, well-formed process dissociation
        "n_pd_gap": 65, "n_pd_decoy": 30, "n_inv": 65,
        "I": 0.85, "E": 0.35, "I_discredited": 0.75, "E_credible": 0.85,
        "R": 0.50, "F": 0.70, "exclusion_margin": 0.40,
        "withdrawal_rate": 0.80,
        # echo-immune measurement integrity: clean observations pass the ceiling
        "pd_unparseable_rate": 0.0,
        "echo_marker_rate": 0.0, "own_decision_rate": 1.0,
    }
    return {
        "primary": dict(per), "sealed": dict(per),
        "answer_only_web_disabled": True,
        "arms": {name: True for name in wfa.WFAArms.ARM_NAMES},
        "hard_fails": {"fabricated_urls": 0, "unfetched_urls": 0,
                       "nonverbatim_spans": 0, "attribution_on_closed_web": 0,
                       "non_paraphrase_claims": 0, "answer_only_web_leaks": 0},
    }


def test_decision_passes_clean_summary():
    rules, passed = wfa.decide(_base_summary(), True)
    assert passed, [k for k, v in rules.items() if not v]


def test_decision_fails_on_fabricated_url():
    s = _base_summary()
    s["hard_fails"]["fabricated_urls"] = 1
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["fabricated_urls==0"]


def test_decision_fails_on_low_wfa_valid():
    s = _base_summary()
    s["primary"]["wfa_valid"] = 0.60
    s["primary"]["delta"] = 0.60
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["wfa_valid_primary>=0.70"]


def test_decision_fails_on_no_real_delta():
    s = _base_summary()
    s["primary"]["answer_only"] = 1.0
    s["primary"]["delta"] = 0.0
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["delta_primary>=0.20"]


def test_decision_fails_on_sealed_drop():
    s = _base_summary()
    s["sealed"]["wfa_valid"] = 0.80
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["sealed_not_below_primary-0.10"]


def test_decision_fails_on_closed_web_attribution():
    s = _base_summary()
    s["hard_fails"]["attribution_on_closed_web"] = 1
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["attribution_on_closed_web==0"]


def test_decision_fails_on_under_sized_pool():
    s = _base_summary()
    s["primary"]["n_gap"] = 59
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["n_gap_primary>=60"]


def test_decision_fails_when_answer_only_web_was_enabled():
    s = _base_summary()
    s["answer_only_web_disabled"] = False
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["answer_only_web_disabled"]


def test_decision_fails_on_web_tripwire_leak():
    s = _base_summary()
    s["hard_fails"]["answer_only_web_leaks"] = 1
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["answer_only_web_leaks==0"]


def test_decision_fails_on_verbatim_claim():
    s = _base_summary()
    s["hard_fails"]["non_paraphrase_claims"] = 1
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["non_paraphrase_claims==0"]


# ── PROCESS-DISSOCIATION / SOURCE-INVALIDATION (fakes only, no engine boot) ──
def test_source_status_labels_are_additive_and_hash_seal_is_unchanged(items):
    statuses = {(i["kind"], i["source_status"]) for i in items}
    assert ("gap", "credible") in statuses
    assert ("decoy", "discredited") in statuses
    assert ("closed_web", "unknown") in statuses
    assert ("internal", "internal") in statuses
    # the additive labels did not perturb the pre-run hash seal
    assert wfa.item_manifest_hash(items) == wfa.item_manifest_hash(wfa.build_items())


def test_pd_well_formed_summary_passes_all_criteria():
    rules, passed = wfa.decide(_base_summary(), True)
    assert passed, [k for k, v in rules.items() if not v]
    assert all(rules["arm_present_%s" % n] for n in wfa.WFAArms.ARM_NAMES)


def test_parse_endorsement_and_withdrawal_are_fail_closed():
    item = {"claim_tokens": ["compass", "magnet"]}
    assert wfa.parse_endorsement("ENDORSE - the magnet explanation", item) is True
    assert wfa.parse_endorsement("ABSTAIN: no reliable source", item) is False
    assert wfa.parse_endorsement("I don't know", item) is False
    assert wfa.parse_withdrawal("WITHDRAW: that source was unreliable") is True
    assert wfa.parse_withdrawal("MAINTAIN - I still assert it") is False
    assert wfa.parse_withdrawal("") is False   # no explicit withdrawal != withdrawn


def _pd_rec(pool, kind, inc, exc, inv=None):
    rec = {"pool": pool, "kind": kind, "ctl": False,
           "pd_inclusion_endorse": inc, "pd_exclusion_endorse": exc}
    if inv is not None:
        rec["inv_withdrawn"] = inv
    return rec


def test_pd_math_known_values():
    records = [
        _pd_rec("primary", "gap", True, True, True),
        _pd_rec("primary", "gap", True, True, False),
        _pd_rec("primary", "gap", True, False, True),
        _pd_rec("primary", "gap", False, False, True),
        _pd_rec("primary", "decoy", True, False),
        _pd_rec("primary", "decoy", True, True),
        _pd_rec("primary", "decoy", True, False),
        _pd_rec("primary", "decoy", True, False),
    ]
    pd = wfa.process_dissociation(records, "primary")
    assert pd["I"] == 0.75          # 3/4 credible endorsed in inclusion
    assert pd["E"] == 0.25          # 1/4 discredited endorsed in exclusion
    assert pd["I_discredited"] == 1.0
    assert pd["E_credible"] == 0.5
    assert pd["R"] == 0.50          # I - E
    assert pd["F"] == 0.50          # E/(1-R) = 0.25/0.5
    assert pd["exclusion_margin"] == 0.75
    assert pd["withdrawal_rate"] == 0.75
    assert pd["n_pd_gap"] == 4 and pd["n_pd_decoy"] == 4


def test_pd_math_degenerate_r_is_not_fabricated():
    # I == 1 and E == 0 -> R == 1 -> F = 0/(1-R) undefined; must be None.
    records = [_pd_rec("primary", "gap", True, True),
               _pd_rec("primary", "decoy", True, False)]
    pd = wfa.process_dissociation(records, "primary")
    assert pd["I"] == 1.0 and pd["E"] == 0.0 and pd["R"] == 1.0
    assert pd["F"] is None
    rules, passed = wfa.decide(_base_summary_with(pd), True)
    assert not passed and not rules["PD_wellformed_primary"]


def _base_summary_with(pd):
    s = _base_summary()
    s["primary"].update(pd)
    return s


class _PromptEngine:
    """Fake engine: inclusion endorses, exclusion abstains. No boot."""

    def __init__(self):
        self.prompts = []

    def process_turn(self, prompt):
        self.prompts.append(prompt)
        return "ABSTAIN" if "ONLY if you are certain" in prompt else "ENDORSE"


def test_inclusion_and_exclusion_differ_only_by_instruction(items):
    decoy = next(i for i in items if i["kind"] == "decoy")
    eng = _PromptEngine()
    inc = wfa.EngineEndorsementResponder(eng)(decoy, "inclusion")
    exc = wfa.EngineEndorsementResponder(eng)(decoy, "exclusion")
    assert inc["endorse"] is True and exc["endorse"] is False
    assert len(eng.prompts) == 2 and eng.prompts[0] != eng.prompts[1]
    assert "trusted sources" in eng.prompts[1].lower()
    assert "only if you are certain" in eng.prompts[1].lower()
    assert "familiarity alone" in eng.prompts[0].lower()


def test_exclusion_abstention_drives_familiarity_leak_measure(items):
    gaps = [i for i in items if i["kind"] == "gap"][:4]
    decoys = [i for i in items if i["kind"] == "decoy"][:4]
    decisions = {(i["id"], "inclusion"): True for i in gaps + decoys}
    decisions.update({(i["id"], "exclusion"): False for i in gaps + decoys})
    inc = wfa.DictEndorsementResponder(decisions)
    exc = wfa.DictEndorsementResponder(decisions)
    arms = wfa.WFAArms(inclusion=inc, exclusion=exc,
                       invalidation=wfa.DictInvalidationResponder(),
                       parametric=object())
    records = [{"id": i["id"], "kind": i["kind"], "pool": i["pool"],
                "query": i.get("query"), "claim": i.get("claim")}
               for i in gaps + decoys]
    out = wfa.run_pd_arms(gaps + decoys, records, arms)
    pd = wfa.process_dissociation(out, gaps[0]["pool"])
    assert pd["E"] == 0.0            # exclusion abstains on every discredited claim
    assert pd["I_discredited"] == 1.0
    assert pd["exclusion_margin"] == 1.0
    assert pd["R"] == pd["I"]        # E == 0 => R == I


def test_invalidation_withdrawal_rate_counted_per_pool(items):
    gaps = [i for i in items if i["kind"] == "gap" and i["pool"] == "primary"]
    ids = [i["id"] for i in gaps]
    decisions = {sid: (idx % 2 == 0) for idx, sid in enumerate(ids)}
    inv = wfa.DictInvalidationResponder(decisions)
    records = [{"id": i["id"], "kind": i["kind"], "pool": i["pool"]}
               for i in gaps]
    arms = wfa.WFAArms(inclusion=wfa.DictEndorsementResponder(),
                       exclusion=None, invalidation=inv,
                       parametric=object())
    wfa.run_pd_arms(gaps, records, arms)
    pd = wfa.process_dissociation(records, gaps[0]["pool"])
    expected = round(sum(decisions[s] for s in ids if s in decisions) / len(ids), 4)
    assert pd["withdrawal_rate"] == expected
    assert pd["n_inv"] == len(ids)


def test_missing_arm_hard_fails_decide():
    for arm in wfa.WFAArms.ARM_NAMES:
        s = _base_summary()
        s["arms"][arm] = False
        rules, passed = wfa.decide(s, True)
        assert not passed, arm
        assert not rules["arm_present_%s" % arm], arm


def test_pd_criteria_are_active_and_pre_registered():
    s = _base_summary()
    s["primary"]["R"] = 0.10
    s["primary"]["F"] = 0.50
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["PD_R_primary>=0.25"]

    s = _base_summary()
    s["sealed"]["exclusion_margin"] = 0.0
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["PD_exclusion_margin_sealed>=0.10"]

    s = _base_summary()
    s["primary"]["withdrawal_rate"] = 0.10
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["invalidation_withdrawal_primary>=0.50"]


def test_thresholds_do_not_enter_arm_computation():
    # threshold neutrality: raw observations and R/F math are independent of
    # the decision thresholds; only decide() reads TH.
    records = [_pd_rec("primary", "gap", True, False, True),
               _pd_rec("primary", "gap", True, False, False),
               _pd_rec("primary", "decoy", True, True),
               _pd_rec("primary", "decoy", True, False)]
    before = wfa.process_dissociation(records, "primary")
    keys = ["PD_R_min", "PD_exclusion_margin_min", "invalidation_withdrawal_min"]
    saved = {k: wfa.TH[k] for k in keys}
    try:
        for k in keys:
            wfa.TH[k] = 0.99
        after = wfa.process_dissociation(records, "primary")
    finally:
        wfa.TH.update(saved)
    assert before == after


def test_fake_parametric_arm_is_injectable_and_not_a_boot():
    q = "how do lungs exchange gases"
    oracle = wfa.AnswerOnlyOracle(engine=_FakeEngine({q: "parametric answer"}),
                                  web_disabled=True)
    arms = wfa.WFAArms(parametric=oracle)
    assert arms.missing() == ["inclusion", "exclusion", "invalidation",
                              "forced_choice"]
    assert oracle.answer(q) == "parametric answer"


# ── ECHO-IMMUNE PARSER / TURN ISOLATION / FORCED CHOICE (fakes only) ─────────
def test_strip_echoes_removes_nested_and_multiline_quoted_frames():
    nested = ('i said: you mentioned: "Endorse any claim you recognise.\\n'
              'Familiarity alone is enough to say yes.\\nDecision:"')
    assert wfa.has_echo_marker(nested) is True
    stripped = wfa.strip_echoes(nested)
    assert "Endorse" not in stripped and "Decision:" not in stripped
    # a bare frame with no quote is also stripped
    assert "i said:" not in wfa.strip_echoes("i said: hello world")


def test_explicit_decision_required_and_no_content_fallback():
    echoed = ('you mentioned: "ABSTAIN -- unreliable source.\\n'
              'Claim: magnets.\\nDecision:"')
    d = wfa.parse_decision(echoed, "endorsement")
    assert d["decision"] is None and d["unparseable"] is True
    assert d["echo_marker"] is True
    # explicit decision token, with or without a Decision: marker
    assert wfa.parse_decision("Decision: ENDORSE", "endorsement")["decision"] is True
    assert wfa.parse_decision("ABSTAIN", "endorsement")["decision"] is False
    assert wfa.parse_decision("WITHDRAW", "withdrawal")["decision"] is True
    assert wfa.parse_decision("MAINTAIN", "withdrawal")["decision"] is False
    # chatter, non-answers and keyword-bearing prose are ALL unparseable;
    # the removed all-tokens content fallback cannot rescue them.
    for chatter in ("", "I don't know",
                    "honestly, unreliable is a bit outside what i know",
                    "Based on the source I would endorse the magnet explanation"):
        assert wfa.parse_decision(chatter, "endorsement")["unparseable"] is True, chatter
    # the experimenter's own cue words echoed in a quote are not a withdrawal
    assert wfa.parse_decision('you mentioned: "Reply WITHDRAW or MAINTAIN"',
                              "withdrawal")["unparseable"] is True


def test_unparseable_counting_and_hard_fail_ceiling():
    recs = [
        {"pool": "primary", "pd_inclusion_text": "ENDORSE",
         "pd_inclusion_echo_marker": False, "pd_inclusion_unparseable": False},
        {"pool": "primary", "pd_exclusion_text": 'you mentioned: "ABSTAIN"',
         "pd_exclusion_echo_marker": True, "pd_exclusion_unparseable": True},
        {"pool": "primary", "inv_after": "chatter",
         "inv_echo_marker": False, "inv_unparseable": True},
        {"pool": "sealed", "pd_inclusion_text": "ENDORSE",
         "pd_inclusion_echo_marker": False, "pd_inclusion_unparseable": False},
    ]
    c = wfa.pd_quality_counters(recs, "primary")
    assert c["unparseable_rate"] == 0.6667
    assert c["own_decision_rate"] == 0.3333
    assert c["echo_marker_rate_exclusion"] == 1.0
    assert c["pd_unparseable_rate"] == c["unparseable_rate"]
    s = _base_summary()
    s["primary"]["pd_unparseable_rate"] = 0.6667
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["unparseable_ceiling_primary"]
    assert rules["unparseable_ceiling_sealed"]   # 0.0 still passes: no global relax


def test_forced_choice_first_token_and_separate_no_answer():
    assert wfa.parse_forced_choice("Y")["decision"] == "endorse"
    assert wfa.parse_forced_choice("N")["decision"] == "abstain"
    assert wfa.parse_forced_choice("W")["decision"] == "withdraw"
    for bad in ("", "maybe", "I think Y", 'you mentioned: "Y"'):
        d = wfa.parse_forced_choice(bad)
        assert d["no_answer"] is True, bad
    # no_answer is its own category, never silently folded into unparseable
    assert "unparseable" not in wfa.parse_forced_choice("maybe")


def test_forced_choice_arm_records_first_token(items):
    gap = next(i for i in items if i["kind"] == "gap")
    fc = wfa.DictForcedChoiceResponder({(gap["id"], "forced_choice"): "Y"})
    rec = [{"id": gap["id"], "kind": gap["kind"], "pool": gap["pool"]}]
    out = wfa.run_pd_arms([gap], rec, wfa.WFAArms(forced_choice=fc))
    assert out[0]["pd_forced_choice"] == "endorse"
    assert out[0]["pd_forced_choice_no_answer"] is False


def test_trusted_list_contains_own_source_only_for_credible(items):
    gap = next(i for i in items if i["kind"] == "gap"
               and i["source_id"] in wfa.CREDIBLE_SOURCES)
    decoy = next(i for i in items if i["kind"] == "decoy")
    tg, td = wfa.trusted_sources_for(gap), wfa.trusted_sources_for(decoy)
    assert gap["source_id"] in tg and tg[0] == gap["source_id"]
    assert decoy["source_id"] not in td
    assert all(u in wfa.CREDIBLE_SOURCES for u in tg + td)
    # the exclusion CRITERION itself is unchanged, and only the trusted segment
    # (not the separate provenance line) carries the allow-list.
    gp = wfa.endorsement_prompt(gap, "exclusion")
    dp = wfa.endorsement_prompt(decoy, "exclusion")
    assert "ONLY if you are certain" in gp
    assert gap["source_id"] in gp.split("Trusted sources for this run:")[1]
    assert decoy["source_id"] not in dp.split("Trusted sources for this run:")[1]


def test_turn_isolation_resets_context_once_per_item_not_between_teach_cue(items):
    class _Engine:
        def __init__(self):
            self.resets = 0
            self.turns = []

        def reset_episodic_state(self):
            self.resets += 1

        def process_turn(self, prompt):
            self.turns.append(prompt)
            return ("Decision: WITHDRAW" if "unreliable and fabricated" in prompt
                    else "Decision: ENDORSE")

    eng = _Engine()
    d = next(i for i in items if i["kind"] == "gap" and i.get("claim"))
    inc = wfa.EngineEndorsementResponder(eng, eng.reset_episodic_state)(d, "inclusion")
    inv = wfa.EngineInvalidationResponder(eng, eng.reset_episodic_state)(d)
    assert inc["endorse"] is True and inv["withdrawn"] is True
    assert eng.resets == 2        # once for the turn, once before teach (not between)
    assert len(eng.turns) == 3    # 1 endorsement + teach + cue


def test_engine_factory_yields_fresh_engine_per_turn(items):
    made = []

    class _E:
        def process_turn(self, prompt):
            return "Decision: ENDORSE"

    def factory():
        e = _E()
        made.append(e)
        return e

    r = wfa.EngineEndorsementResponder(factory)
    d = next(i for i in items if i["kind"] == "gap")
    r(d, "inclusion")
    r(d, "exclusion")
    assert len(made) == 2 and made[0] is not made[1]


def test_oracle_compliance_control_reproduces_known_i_e_r(items):
    gaps = [i for i in items if i["kind"] == "gap" and i["pool"] == "primary"][:4]
    decoys = [i for i in items if i["kind"] == "decoy" and i["pool"] == "primary"][:4]
    chosen = gaps + decoys
    decisions = {}
    for idx, it in enumerate(gaps):
        decisions[(it["id"], "inclusion")] = True
        decisions[(it["id"], "exclusion")] = idx < 3      # 3/4 credible under exclusion
    for idx, it in enumerate(decoys):
        decisions[(it["id"], "inclusion")] = idx < 2      # I_dis = 0.5
        decisions[(it["id"], "exclusion")] = idx < 1      # E = 0.25

    class _TextResponder:
        def __init__(self):
            self.calls = []

        def __call__(self, item, arm):
            self.calls.append((item["id"], arm))
            text = ("Decision: ENDORSE" if decisions[(item["id"], arm)]
                    else "Decision: ABSTAIN")
            d = wfa.parse_decision(text, "endorsement")
            return {"endorse": d["decision"], "decision": d["decision"],
                    "unparseable": d["unparseable"], "echo_marker": d["echo_marker"],
                    "text": text, "source": item.get("source_id")}

    arms = wfa.WFAArms(inclusion=_TextResponder(), exclusion=_TextResponder(),
                       invalidation=wfa.DictInvalidationResponder(),
                       parametric=object())
    records = [{"id": i["id"], "kind": i["kind"], "pool": i["pool"]} for i in chosen]
    out = wfa.run_pd_arms(chosen, records, arms)
    pd = wfa.process_dissociation(out, "primary")
    assert pd["I"] == 1.0 and pd["E"] == 0.25
    assert pd["I_discredited"] == 0.5 and pd["E_credible"] == 0.75
    assert pd["R"] == 0.75 and pd["F"] == 1.0
    assert pd["exclusion_margin"] == 0.25
    q = wfa.pd_quality_counters(out, "primary")
    assert q["own_decision_rate"] == 1.0 and q["unparseable_rate"] == 0.0


def test_missing_forced_choice_arm_hard_fails():
    s = _base_summary()
    s["arms"]["forced_choice"] = False
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["arm_present_forced_choice"]


# ── INTERNAL-SOURCE ARM: graph-provenance-derived context (FAKES, no boot) ──
# These tests exercise the SHIPPED gate methods as unbound functions on fakes,
# exactly as test_source_evaluation_wiring.py does. No CognitiveChatEngine is
# constructed, no process is spawned, no pytest fixture touches a real store.
IS_CLAIM = "A compass points north because the Earth behaves like a loadstone."
IS_ORIGIN = "wfa-internal://credible/abc123def456"


class _ISNode:
    def __init__(self, label):
        self.label = label


class _ISEdge:
    def __init__(self, sm, confidence=0.9):
        self.source_metadata = sm
        self.confidence = confidence
        self.relation_type = "semantic"
        self.timestamp = 1.0
        self.source, self.target = 1, 2


class _ISGraph:
    """Tiny ConceptGraph-shaped fake: .nodes, .edges, .get_node."""

    def __init__(self, sm=None):
        self.nodes = {1: _ISNode("compass"), 2: _ISNode("loadstone")}
        self.edges = {}
        if sm is not None:
            self.edges[(1, 2)] = _ISEdge(sm)

    def get_node(self, nid):
        return self.nodes.get(nid)


class _ISEngine:
    """Fake engine exposing only the surface the arm/gate touch."""

    def __init__(self, sm=None):
        self.graph = _ISGraph(sm)
        self._pending_source_context = None
        self._source_gate_ctx = None
        self._last_source_evaluation = None
        self.resets = 0
        self.prompts = []
        self.set_source_context_for_turn = (
            lambda *a, **k: CognitiveChatEngine.set_source_context_for_turn(self, *a, **k))
        self._apply_source_monitoring_gate = (
            lambda r, c: CognitiveChatEngine._apply_source_monitoring_gate(self, r, c))
        self._source_monitor_decision = CognitiveChatEngine._source_monitor_decision

    def reset_episodic_state(self):
        self.resets += 1

    def process_turn(self, prompt):
        self.prompts.append(prompt)
        self._source_gate_ctx = type(
            "Ctx", (), {"source_context": self._pending_source_context})()
        return self._apply_source_monitoring_gate(
            "the generated answer text", self._source_gate_ctx)


def _is_teach(engine, claim, source_id):
    engine.graph.edges[(1, 2)] = _ISEdge(
        {"source": source_id, "edge_kind": "web_fact", "source_type": "seed"})
    return 1


def _is_invalidate(engine, source_id):
    engine.graph.edges[(1, 2)].source_metadata["source_status"] = "discredited"
    return 1


@pytest.fixture(autouse=True)
def _is_flag(monkeypatch):
    """The opt-in flag is ON for this section only; monkeypatch restores it."""
    monkeypatch.setenv(wfa.IS_FLAG, "1")
    yield
    monkeypatch.delenv(wfa.IS_FLAG, raising=False)


def test_source_question_is_condition_independent_and_label_free():
    prompts = [wfa.source_question_prompt(IS_CLAIM)
               for _ in wfa.IS_CONDITIONS]
    assert prompts[0] == prompts[1] == prompts[2]
    assert wfa.prompt_is_label_free(prompts[0], IS_ORIGIN) is True
    assert IS_ORIGIN not in prompts[0]
    # The question asks ABOUT credibility but never states the answer.
    assert "from a credible source" in prompts[0].lower()
    # A provenance-bearing prompt from the old tautological arm is rejected.
    leaked = prompts[0] + "\nProvenance: credible (source id: %s)" % IS_ORIGIN
    assert wfa.prompt_is_label_free(leaked, IS_ORIGIN) is False


def test_context_from_graph_provenance_reads_only_stored_metadata():
    credible = _ISGraph({"source": IS_ORIGIN, "edge_kind": "web_fact"})
    ctx, audit = wfa.context_from_graph_provenance(credible, IS_CLAIM, IS_ORIGIN)
    assert audit["context_source"] == "engine_graph_provenance"
    assert audit["trust_class"] == "credible"
    assert audit["credibility_token"] == "credible"
    assert ctx.source.origin_id == IS_ORIGIN
    assert evaluate_source(IS_CLAIM, ctx).rule == "R_EXPLICIT_CREDIBLE_SOURCE"

    # Same claim, but only an unprovenanced/auto-expand edge stored -> UNKNOWN.
    unknown = _ISGraph({"source": IS_ORIGIN, "edge_kind": "auto_expand"})
    ctx2, audit2 = wfa.context_from_graph_provenance(unknown, IS_CLAIM, IS_ORIGIN)
    assert audit2["trust_class"] == "unknown"
    assert ctx2.source.credibility_token is None
    assert evaluate_source(IS_CLAIM, ctx2).rule == "R_UNKNOWN_PROVENANCE"

    # No stored edge at all -> unknown provenance (source amnesia).
    ctx3, audit3 = wfa.context_from_graph_provenance(
        _ISGraph(), IS_CLAIM, IS_ORIGIN)
    assert audit3["n_matching_edges"] == 0
    assert evaluate_source(IS_CLAIM, ctx3).rule == "R_UNKNOWN_PROVENANCE"


def test_item_metadata_mutation_cannot_change_staged_decision():
    """ANTI-TAUTOLOGY: the item's expected answer/status/id are invisible."""
    graph = _ISGraph({"source": IS_ORIGIN, "edge_kind": "web_fact"})
    item_a = {"claim": IS_CLAIM, "source_status": "credible",
              "expected_answer": "Y", "source_id": IS_ORIGIN, "reliability": 0.9}
    item_b = dict(item_a, source_status="discredited", expected_answer="W",
                  source_id="other://origin", reliability=1.0)
    ctx_a, audit_a = wfa.stage_context_for_item(graph, item_a, IS_ORIGIN)
    ctx_b, audit_b = wfa.stage_context_for_item(graph, item_b, IS_ORIGIN)
    assert ctx_a == ctx_b
    assert audit_a["fingerprint"] == audit_b["fingerprint"]
    d_a = evaluate_source(IS_CLAIM, ctx_a)
    d_b = evaluate_source(IS_CLAIM, ctx_b)
    assert d_a.decision is Decision.ENDORSE
    assert d_b.decision is d_a.decision
    assert d_a.rule == d_b.rule == "R_EXPLICIT_CREDIBLE_SOURCE"


def test_provenance_mutation_control_moves_decision():
    """Changing ONLY stored provenance must move ENDORSE -> WITHDRAW."""
    graph = _ISGraph({"source": IS_ORIGIN, "edge_kind": "web_fact"})
    result = wfa.internal_source_provenance_mutation_control(
        IS_CLAIM, IS_ORIGIN, graph, Decision.ENDORSE)
    assert result == {"before": "ENDORSE", "after": "WITHDRAW", "moved": True}


def test_internal_source_scoring_branches_with_fake_engine():
    seen = {}
    for cond, expected_token, expected_decision in (
            ("credible", "Y", Decision.ENDORSE),
            ("discredited", "W", Decision.WITHDRAW),
            ("unknown", "N", Decision.ABSTAIN)):
        eng = _ISEngine()
        obs = wfa.run_internal_source_condition(
            eng, IS_CLAIM, cond, wfa.is_source_id(IS_CLAIM, cond),
            teach_fn=_is_teach, invalidate_fn=_is_invalidate)
        grade = wfa.grade_internal_source_condition(obs, cond)
        seen[cond] = grade
        assert grade["token"] == expected_token, (cond, grade)
        assert obs["decision"] == expected_decision.value
        assert grade["correct"] is True and grade["decision_expected"] is True
        assert grade["gate_ok"] is True and grade["prompt_label_free"] is True
        assert grade["context_from_graph"] is True
        assert eng.resets == 1                       # one isolation per condition
        # ENDORSE emits generated text; the other two emit the exact token.
        if expected_decision is Decision.ENDORSE:
            assert obs["text"] == "the generated answer text"
        else:
            assert obs["text"] == expected_token
    assert seen["discredited"]["rule"] == "R_SOURCE_INVALIDATED"
    assert seen["unknown"]["rule"] == "R_UNKNOWN_PROVENANCE"
    assert seen["credible"]["rule"] == "R_EXPLICIT_CREDIBLE_SOURCE"


def test_discredited_arm_prior_decision_comes_from_engine_audit():
    eng = _ISEngine()
    obs = wfa.run_internal_source_condition(
        eng, IS_CLAIM, "discredited", IS_ORIGIN,
        teach_fn=_is_teach, invalidate_fn=_is_invalidate)
    assert obs["prior_decision"] == "ENDORSE"   # read from _last_source_evaluation
    assert obs["invalidated"] == 1
    assert len(eng.prompts) == 2                # commit turn + source question
    assert eng.prompts[0] == eng.prompts[1]      # identical, label-free prompts


def test_missing_engine_is_reported_not_faked():
    obs = wfa.run_internal_source_condition(None, IS_CLAIM, "credible", IS_ORIGIN)
    grade = wfa.grade_internal_source_condition(obs, "credible")
    assert grade["missing"] is True and grade["correct"] is False
    assert grade["token"] is None


def test_internal_source_summary_and_additive_rules():
    records = []
    items = []
    for pool in ("primary", "sealed"):
        for n in range(20):          # pre-registered n>=20 per pool
            claim = "%s variant %d" % (IS_CLAIM, n)
            sid = "item-%s-%d" % (pool, n)
            items.append({"id": sid, "kind": "gap", "pool": pool, "claim": claim})
            records.append({"id": sid, "kind": "gap", "pool": pool})

    def fake_arm(cond):
        def run(it, _c):
            return wfa.run_internal_source_condition(
                _ISEngine(), it["claim"], cond,
                wfa.is_source_id(it["claim"], cond),
                teach_fn=_is_teach, invalidate_fn=_is_invalidate)
        return run

    wfa.run_internal_source_arms(
        items, records, wfa.InternalSourceArms(
            credible=fake_arm("credible"), discredited=fake_arm("discredited"),
            unknown=fake_arm("unknown")))
    summary = wfa.internal_source_summary(records)
    for pool in ("primary", "sealed"):
        pp = summary["pools"][pool]
        assert pp["n"] == 20
        assert pp["anti_tautology_ok"] is True
        for cond in wfa.IS_CONDITIONS:
            cc = pp["conditions"][cond]
            assert cc["correct"] == 1.0 and cc["gate_ok"] == 1.0
            assert cc["prompt_label_free"] == 1.0
            assert cc["context_from_graph"] == 1.0
    rules, passed = wfa.internal_source_decide(dict(summary, flag_enabled=True))
    assert passed, [k for k, v in rules.items() if not v]
    # Additive: the NEW rules live in their own subtree and do not change the
    # main verdict on the unchanged base summary.
    assert wfa.decide(_base_summary(), True)[1] is True


def test_thresholds_are_additive_existing_values_unchanged():
    assert wfa.TH["wfa_valid_min"] == 0.70
    assert wfa.TH["abstain_min"] == 0.90
    assert wfa.TH["delta_min"] == 0.20
    assert wfa.TH["decoy_rej_min"] == 0.85
    assert wfa.TH["internal_no_tag_min"] == 0.90
    assert wfa.TH["sealed_drop_max"] == 0.10
    assert wfa.TH["unparseable_rate_max"] == 0.50
    for key in ("IS_correct_min", "IS_gate_invariant_min",
                "IS_prompt_label_free_min"):
        assert key in wfa.TH


def test_endorse_is_silent_and_graded_by_decision():
    """ENDORSE leaves the claim emitted; it never emits a literal Y token."""
    good = wfa.grade_internal_source_condition(
        {"decision": "ENDORSE", "token": None, "gate_ok": True,
         "staged_context_source": "engine_graph_provenance"},
        "credible")
    assert good["correct"] and good["decision_expected"]
    bad = wfa.grade_internal_source_condition(
        {"decision": "ABSTAIN", "token": "N", "gate_ok": True,
         "staged_context_source": "engine_graph_provenance"},
        "credible")
    assert not bad["correct"] and not bad["decision_expected"]
    withdrawal = wfa.grade_internal_source_condition(
        {"decision": "WITHDRAW", "token": "W", "gate_ok": True,
         "staged_context_source": "engine_graph_provenance"},
        "discredited")
    assert withdrawal["correct"]


def test_explicit_assertion_fallback_writes_discoverable_provenance():
    """When the extractor writes nothing, the claim is still taught via the
    real graph API with provenance — never from item metadata."""
    class Node:
        def __init__(self, nid, label):
            self.id, self.label = nid, label

    class Edge:
        source_metadata = None
        confidence = 0.9
        timestamp = 1
        relation_type = "semantic"

    class Graph:
        def __init__(self):
            self.nodes, self.edges, self.n = {}, {}, 0

        def add_node(self, label=""):
            self.n += 1
            node = Node(self.n, label)
            self.nodes[self.n] = node
            return node

        def add_edge(self, source, target, **kwargs):
            edge = Edge()
            self.edges[(source, target)] = edge
            return edge

        def get_node(self, nid):
            return self.nodes.get(nid)

    class W2G:
        def learn_text(self, *args, **kwargs):
            return 0

    class Engine:
        def __init__(self):
            self.graph = Graph()

        def _get_web_to_graph(self):
            return W2G()

    eng = Engine()
    claim = "Frost creates in much the same manner as dew creates."
    assert wfa.teach_claim_into_graph(eng, claim, "credible-web-1") == 1
    rows = wfa.graph_provenance_for_claim(eng, claim, source_id="credible-web-1")
    assert rows and rows[0]["edge_kind"] == "curated"
    assert rows[0]["source"] == "credible-web-1"
