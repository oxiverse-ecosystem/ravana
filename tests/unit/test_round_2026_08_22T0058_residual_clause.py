"""RED->GREEN regression test for RAVANA residual capability (round
2026-08-22T0058Z, feature card t_6b93e125).

RESIDUAL DEFECT (logged at end of round 2026-08-22T0058Z): a relationship
disclosure whose verb OBJECT is a NON-NOUN-PHRASE CLAUSE was dropped
entirely. Canonical case:
    "my old beekeeping mentor, Dr. Osei, taught me how to read the hive's mood"
The opinion-topic resolver collapses the clause to a content head ("read") and
then REJECTS that head as verb-residue (it lives in _OBJ_NONCONTENT), so _obj
came back EMPTY and the degenerate-fact guard dropped the WHOLE disclosure — a
later "what did my mentor teach me?" had nothing to recall.

This is a structural capability gap, not a per-topic fix. The fix preserves the
user's own clause words as the fact value (bounded, trailing framers stripped)
instead of dropping the disclosure. Content comes from the user's own words; no
authored prose; no retraining; RAVANA-expandable via the same PersonalFactStore.

The test FAILS at the RED line below when the capability is absent (the fact is
never mined) and PASSES once the raw-object fallback is present. It also guards
that genuine noun-phrase objects still go through the resolver (regression), and
that a pure verb-residue clause ("taught me") is honestly NOT stored as content.
"""
import os, sys, io, contextlib, glob
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _new_engine(suffix):
    for f in (glob.glob(f"weights/ravana_weights{suffix}.pkl")
              + glob.glob(f"weights/ravana_usermodel{suffix}.pkl")):
        os.remove(f)
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def _facts(eng):
    return [(a, f.value) for (s, a, _), f in
            eng.user_model.personal_facts.facts.items()
            if s == "i" and not f.superseded]


def test_clause_object_relationship_mined_and_recalled():
    """A clause-object relationship disclosure must mine + recall its clause."""
    eng = _new_engine("test_resid_clause")
    with contextlib.redirect_stdout(io.StringIO()):
        eng.process_turn(
            "my old beekeeping mentor, Dr. Osei, taught me how to read the "
            "hive's mood.")
    rel = _facts(eng)
    # RED line: without the raw-object fallback this fact is NEVER mined.
    assert ("mentor dr. osei",) in [(a,) for a, _ in rel], (
        f"clause-object relationship disclosure not mined: {rel}")
    _val = dict((a, v) for a, v in rel if a == "mentor dr. osei")
    assert "taught how to read the hive's mood" in _val.values(), (
        f"clause object not preserved as value: {_val}")
    # recall via reverse-name resolver
    with contextlib.redirect_stdout(io.StringIO()):
        ans = eng.process_turn("who is dr. osei to me?")
    assert "mentor" in ans.lower(), f"mentor not recalled: {ans!r}"
    with contextlib.redirect_stdout(io.StringIO()):
        ans2 = eng.process_turn("what did my mentor teach me?")
    assert "hive" in ans2.lower() and "taught" in ans2.lower(), (
        f"clause not recalled: {ans2!r}")
    print("[OK] residual clause-object relationship mined + recalled")
    eng.stop_background_learning()


def test_noun_phrase_object_still_resolved():
    """Regression: a genuine noun-phrase object still goes through the resolver
    (the existing 'real concept head' shape is preserved, not clobbered)."""
    eng = _new_engine("test_resid_noun")
    with contextlib.redirect_stdout(io.StringIO()):
        eng.process_turn("my grandmother taught me astronomy when she was alive.")
    rel = _facts(eng)
    assert ("grandmother",) in [(a,) for a, _ in rel], (
        f"noun-phrase relationship disclosure not mined: {rel}")
    _val = dict((a, v) for a, v in rel if a == "grandmother")
    # resolver keeps the concept head "astronomy" (not the raw clause)
    assert any("astronomy" in v for v in _val.values()), (
        f"noun-phrase object not resolved to concept head: {_val}")
    print("[OK] noun-phrase object still resolved via topic resolver")
    eng.stop_background_learning()


def test_pure_verb_residue_clause_not_stored():
    """Honesty guard: 'my mentor taught me' (no clause content) must NOT be
    stored as a junk fact — the leading framer 'me' is stripped and nothing
    real remains, so the disclosure is honestly skipped, not fabricated."""
    eng = _new_engine("test_resid_empty")
    with contextlib.redirect_stdout(io.StringIO()):
        eng.process_turn("my mentor taught me.")
    rel = _facts(eng)
    _osei = [(a, v) for a, v in rel if a == "mentor"]
    assert not _osei or "taught" not in _osei[0][1] or len(
        _osei[0][1].split()) > 1, (
        f"pure verb-residue stored as content: {_osei}")
    print("[OK] pure verb-residue clause honestly not stored")
    eng.stop_background_learning()


if __name__ == "__main__":
    test_clause_object_relationship_mined_and_recalled()
    test_noun_phrase_object_still_resolved()
    test_pure_verb_residue_clause_not_stored()
    print("\nALL RESIDUAL-CLAUSE REGRESSION TESTS PASSED")
