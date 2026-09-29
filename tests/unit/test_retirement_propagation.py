"""RED-FIRST gate for the retirement capability.

Split deliberately:

  * PARSER tests — pure Python, no engine boot. Deterministic and fast.
  * ENGINE tests — the user-visible symptom (a correction must stick and
    every store must stop answering with the retracted value). These are the
    tests that genuinely bite; the parser tests only pin the mechanism.

No authored reply strings are asserted here: the gate is on WHICH value
comes back, never on the prose that carries it.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    if p not in sys.path:
        sys.path.insert(0, p)

from ravana.chat.retirement import (
    CONTRAST_MARKERS,
    RetirementLedger,
    Retraction,
    looks_like_retraction,
    parse_contrast,
)


def _ret(**kw):
    """Build a real Retraction. Deliberately not a hand-rolled stand-in with
    a subset of fields — the ledger serialises every field, so a stub with
    fewer attributes would fail for the wrong reason."""
    kw.setdefault("retired_value", "")
    kw.setdefault("replaced_by", None)
    kw.setdefault("marker", None)
    kw.setdefault("turn_index", 0)
    kw.setdefault("evidence", "")
    return Retraction(**kw)


# ── parser: the productive contrastive shapes ──────────────────────────

def test_contrastive_tail_splits_asserted_from_rejected():
    got = parse_contrast("no, nikhil plays the surbahari, not the shehnai")
    assert got is not None, "contrastive tail not recognised"
    asserted, rejected, marker = got
    assert "surbahari" in asserted and "surbahari" not in rejected, asserted
    assert "shehnai" in rejected and "shehnai" not in asserted, rejected


def test_negation_first_inverted_shape():
    got = parse_contrast("not the shehnai, it's the surbahari")
    assert got is not None
    asserted, rejected, _m = got
    assert "surbahari" in asserted, asserted
    assert "shehnai" in rejected, rejected


def test_ordinary_disclosure_is_not_a_contrast():
    # No contrast marker -> must return None, never a spurious split.
    assert parse_contrast("my brother nikhil plays the shehnai") is None


def test_no_false_contrast_without_marker():
    assert parse_contrast("i have a dog named rocky and a cat named milo") is None


def test_identical_both_sides_is_rejected():
    # asserted == rejected is not a revision, it is a repetition.
    assert parse_contrast("shehnai, not shehnai") is None


def test_learned_marker_is_honoured():
    led = RetirementLedger()
    assert led.learn_contrast_marker("nah") is True
    assert "nah" in led.all_markers()
    # A learned marker is treated as INTERJECTIVE (it repairs what precedes),
    # so the parser must now accept an utterance it would otherwise skip.
    got = parse_contrast("he moved to berlin, nah he moved to lisbon",
                         led.all_markers(), led.learned_markers)
    assert got is not None
    asserted, rejected, marker = got
    assert marker == "nah"
    assert "lisbon" in asserted and "berlin" in rejected


def test_authored_interjective_marker_gets_direction_right():
    got = parse_contrast("he moved to berlin, actually he moved to lisbon")
    assert got is not None
    asserted, rejected, marker = got
    assert marker == "actually"
    assert "lisbon" in asserted and "berlin" in rejected


def test_negation_and_interjection_opposite_directions():
    """The class distinction is the general rule; both must hold at once."""
    neg = parse_contrast("nikhil plays surbahari, not shehnai")
    assert neg is not None
    assert "shehnai" in neg[1] and "surbahari" in neg[0], neg
    inter = parse_contrast("nikhil plays shehnai, actually he plays surbahari")
    assert inter is not None
    assert "shehnai" in inter[1] and "surbahari" in inter[0], inter


def test_ledger_starts_empty():
    led = RetirementLedger()
    assert led.retired == {} and led.log == []
    assert led.learned_markers == set()


def test_ledger_retire_and_query():
    led = RetirementLedger()
    led.retire(_ret(retired_value="Shehnai", turn_index=4, marker="not",
                    replaced_by="surbahari"), "brother nikhil|plays")
    assert led.is_retired("shehnai")
    assert led.is_retired("shehnai", "brother nikhil|plays")
    assert not led.is_retired("surbahari")
    # inflected form must match the same retirement
    assert led.is_retired("shehnais")


def test_ledger_marker_growth_only_via_retirement():
    led = RetirementLedger()
    assert led.learn_contrast_marker("not") is False  # already authored
    led.retire(_ret(retired_value="x", turn_index=1, marker="nah"), "s")
    assert "nah" in led.learned_markers


def test_ledger_roundtrips_through_state():
    led = RetirementLedger()
    led.retire(_ret(retired_value="surbahari", turn_index=9, marker="not",
                    replaced_by="shehnai",
                    evidence="no, it is the shehnai"), "a|b")
    led.learn_contrast_marker("nah")
    back = RetirementLedger.from_state(led.to_state())
    assert back.is_retired("surbahari")
    assert "nah" in back.all_markers()
    assert back.log[0].evidence == "no, it is the shehnai"
    assert back.log[0].turn_index == 9


def test_retraction_opener_detection():
    assert looks_like_retraction("no, nikhil plays the surbahari, not the shehnai")
    assert looks_like_retraction("actually i live in berlin")
    assert looks_like_retraction("forget what i said about the shehnai")
    assert not looks_like_retraction("my brother nikhil plays the shehnai")
    assert not looks_like_retraction("hello there")


# ── engine: the user-visible symptom ───────────────────────────────────

def _engine():
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="retiretest")


def _turn(eng, q):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = eng.process_turn(q)
    return r.get("reply") if isinstance(r, dict) else r


def test_correction_retires_the_old_value_in_every_store():
    eng = _engine()
    try:
        _turn(eng, "my brother nikhil plays the shehnai, he is in his thirties")
        _turn(eng, "no, nikhil plays the surbahari, not the shehnai")

        # 1. the retraction is recorded as a real event, not a prose ack
        led = eng.retirement_ledger
        assert led.log, "no grounded retraction recorded"

        # 2. the OLD value is retired for its slot
        assert any("shehnai" in v for vals in led.retired.values() for v in vals), \
            "retracted value not retired"

        # 3. the fact store no longer answers with it
        pf = eng.user_model.personal_facts
        for _k, f in pf.facts.items():
            if "shehnai" in str(f.value).lower():
                assert f.superseded, f"old value still active: {_k} {f}"

        # 4. the episodic transcript no longer offers it for recall
        for rec in eng._episodic_transcript:
            assert not eng._record_is_retracted(rec), \
                f"retracted record still recallable: {rec.get('text')!r}"

        # 5. and the user-visible answer has moved on
        reply = (_turn(eng, "which instrument does my brother play now?") or "").lower()
        assert "surbahari" in reply, reply
        assert "shehnai" not in reply, reply
    finally:
        eng.stop_background_learning()


def test_self_recall_does_not_resurface_the_retracted_claim():
    eng = _engine()
    try:
        _turn(eng, "my brother nikhil plays the shehnai, he is in his thirties")
        _turn(eng, "no, nikhil plays the surbahari, not the shehnai")
        reply = (_turn(eng, "what did i tell you about my brother?") or "").lower()
        assert "shehnai" not in reply, reply
        assert "surbahari" in reply, reply
    finally:
        eng.stop_background_learning()


def test_ordinary_disclosure_is_never_retracted():
    """The capability must not fire on normal conversation."""
    eng = _engine()
    try:
        _turn(eng, "my brother nikhil plays the shehnai, he is in his thirties")
        _turn(eng, "i also have a dog named rocky")
        assert not eng.retirement_ledger.log, \
            "spurious retraction on ordinary disclosure"
    finally:
        eng.stop_background_learning()


def test_retraction_survives_save_and_load():
    eng = _engine()
    try:
        _turn(eng, "my brother nikhil plays the shehnai, he is in his thirties")
        _turn(eng, "no, nikhil plays the surbahari, not the shehnai")
        eng.save()
    finally:
        eng.stop_background_learning()

    eng2 = _engine()
    try:
        eng2.load()
        led = getattr(eng2, "retirement_ledger", None)
        assert led is not None and led.log, "retirement did not survive save/load"
        reply = (_turn(eng2, "which instrument does my brother play now?") or "").lower()
        assert "shehnai" not in reply, reply
    finally:
        eng2.stop_background_learning()
