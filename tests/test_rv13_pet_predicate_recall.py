"""FIX-RV-13 (round auto/round-20260925T0823-fix-7) — three compounding
defects, one root cause each. Written RED first.

Defect A (ROOT CAUSE): the personal-fact miner only mined a possession that
carried a COPULA ("my X is Y") or a NAME ("my X <Name> ..."). A canonical
disclosure with a PREDICATIVE VERB and no name — "my cat has been diagnosed
with a chronic illness", "my dog had surgery last month" — matched no miner
branch, so ZERO facts were stored. Every downstream recall failure is a
symptom of that.

Defect B: `_is_question` treats the first three tokens as a question lead, so
a DECLARATION whose 3rd token is an auxiliary ("my cat HAS been diagnosed")
is classified as a question. The episodic cue pass skips prior "questions", so
it skipped the exact cat/dog episodes the query was cued on and the recall
degenerated to a most-recent-turn echo.

Defect C: an unresolvable recall cue must ABSTAIN, not quote an unrelated
turn. The agent-own-recall ring-buffer fallback accepted a single incidental
token overlap (>=1) against RAVANA's own reply TEXT, so every entity-cued
recall converged on the same most-recent reply.

Defect D: "what is my <pet>'s name" must reach the pet recall resolver. With
no fact mined (A) it fell through to metacognitive uncertainty.

Run: RAVANA_OFFLINE=1 python -m pytest tests/test_rv13_pet_predicate_recall.py -q
"""
import os
os.environ.setdefault("RAVANA_OFFLINE", "1")

import sys
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ, f"{_PROJ}/ravana_ml/src", f"{_PROJ}/ravana/src", f"{_PROJ}/ravana-v2/src"):
    sys.path.insert(0, p)

import glob

from ravana.chat.engine import CognitiveChatEngine

# The four canonical disclosures from the cold audit.
DISCLOSURES = [
    "my cat has been diagnosed with a chronic illness",
    "my dog had surgery last month and i am worried",
    "i really love hiking in the himalayas",
    "i think remote work is better than office work",
]

# The entity cue each of the first three disclosures is really about.
CUES = ["cat", "dog", "hiking"]


def _clean(suffix):
    """A COLD engine: no pre-existing pickle for this suffix can contaminate
    the run with another worker's conversation (stale-pickle pitfall)."""
    wdir = os.path.join(_PROJ, "weights")
    if os.path.isdir(wdir):
        for p in glob.glob(os.path.join(wdir, f"*{suffix}*.pkl")):
            os.remove(p)
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def _facts(eng):
    return eng.user_model.personal_facts.facts


def _values(eng):
    """Keys AND values — a mined pet fact stores the entity in the KEY
    ('i', 'cat', 'diagnosed ...'), so reading values alone would make the
    assertion blind to the very thing it checks."""
    return " | ".join(
        f"{k[0]} {k[1]} {k[2]}" if isinstance(k, tuple) else str(k)
        for k, f in _facts(eng).items()).lower()


# ─────────────────────────────────────────────────────────────────────
# Defect A — the miner must store a pet + condition from a name-less
# predicative disclosure.
# ─────────────────────────────────────────────────────────────────────

def test_miner_stores_pet_condition_from_predicative_disclosure():
    eng = _clean("rv13_miner_cat")
    eng.user_model.mine_personal_facts(
        "my cat has been diagnosed with a chronic illness")
    facts = _facts(eng)
    assert facts, (
        "FIX-RV-13 A: a canonical disclosure mined ZERO facts; the fact store "
        "is empty so every recall downstream can only confabulate")
    blob = _values(eng)
    assert "cat" in blob, f"pet entity not retained; stored: {blob}"
    assert "diagnos" in blob, f"condition not retained; stored: {blob}"


def test_miner_stores_pet_event_from_past_tense_disclosure():
    eng = _clean("rv13_miner_dog")
    eng.user_model.mine_personal_facts(
        "my dog had surgery last month and i am worried")
    facts = _facts(eng)
    assert facts, "FIX-RV-13 A: 'my dog had surgery last month' mined zero facts"
    blob = _values(eng)
    assert "dog" in blob, f"pet entity not retained; stored: {blob}"
    assert "surger" in blob, f"event not retained; stored: {blob}"


def test_miner_generalises_beyond_the_probed_verbs():
    """The fix must be a SHAPE (possession + predicate), not a per-topic verb
    table: an unprobed pet predicate must mine the same way."""
    eng = _clean("rv13_miner_gen")
    eng.user_model.mine_personal_facts("my rabbit has been missing since friday")
    facts = _facts(eng)
    assert facts, (
        "FIX-RV-13 A: the miner fix is tuned to the probed words, not the "
        "possession+predicate shape; an unprobed pet predicate still mines nothing")
    assert "rabbit" in _values(eng)


# ─────────────────────────────────────────────────────────────────────
# Defect B — a declaration is not a question.
# ─────────────────────────────────────────────────────────────────────

def test_declaration_with_embedded_auxiliary_is_not_a_question():
    eng = _clean("rv13_isq")
    for text in ("my cat has been diagnosed with a chronic illness",
                 "my dog had surgery last month"):
        assert not eng._is_question(text), (
            f"FIX-RV-13 B: a first-person DECLARATION is classified as a "
            f"question because an auxiliary sits in the lead window: {text!r}")


def test_real_questions_are_still_questions():
    eng = _clean("rv13_isq_ok")
    for text in ("is my cat okay",
                 "what did you say about the cat",
                 "do you remember my dog",
                 "has my dog recovered?"):
        assert eng._is_question(text), (
            f"fixing B must not stop recognising real questions: {text!r}")


def test_cued_recall_returns_the_asked_episode_not_a_sibling():
    """The whole point: three DIFFERENT entity-cued recalls must resolve to
    three DIFFERENT, entity-matched answers."""
    eng = _clean("rv13_cue")
    for q in DISCLOSURES:
        eng.process_turn(q)
    replies = {}
    for cue in CUES:
        replies[cue] = eng.process_turn(
            f"what did i just tell you about {cue}").lower()
    assert len(set(replies.values())) == len(CUES), (
        "FIX-RV-13 B: every entity-cued recall returned the SAME episode: "
        f"{replies}")
    for cue, reply in replies.items():
        assert cue in reply, (
            f"FIX-RV-13 B: recall cued on {cue!r} did not answer about it: "
            f"{reply!r}")


# ─────────────────────────────────────────────────────────────────────
# Defect C — an unresolved cue abstains instead of quoting a sibling turn.
# ─────────────────────────────────────────────────────────────────────

def test_unresolved_entity_cue_abstains_rather_than_quoting_a_sibling():
    eng = _clean("rv13_abstain")
    for q in DISCLOSURES:
        eng.process_turn(q)
    reply = eng.process_turn(
        "what did i just tell you about my ferret").lower()
    leaked = [c for c in ("remote work", "hiking", "office work") if c in reply]
    assert not leaked, (
        "FIX-RV-13 C: an unresolvable cue confidently quoted an unrelated "
        f"episode ({leaked}): {reply!r}")


# ─────────────────────────────────────────────────────────────────────
# Defect D — pet-name recall routes to the pet resolver.
# ─────────────────────────────────────────────────────────────────────

def test_pet_name_recall_routes_to_pet_resolver():
    eng = _clean("rv13_petname")
    eng.process_turn("my cat is named milo")
    reply = eng.process_turn("what is my cat's name").lower()
    strategy = str(getattr(eng, "_last_strategy", ""))
    assert strategy != "metacognitive_uncertainty", (
        "FIX-RV-13 D: a cued pet-name recall fell to metacognitive "
        f"uncertainty instead of the pet resolver: {reply!r}")
    assert "milo" in reply, f"stored pet name not recalled: {reply!r}"


def test_pet_name_recall_abstains_honestly_when_never_disclosed():
    """No name was ever given: RAVANA must say it does not have one rather
    than inventing or echoing an unrelated turn."""
    eng = _clean("rv13_petname_none")
    for q in DISCLOSURES:
        eng.process_turn(q)
    reply = eng.process_turn("what is my ferret's name").lower()
    assert "ferret" not in reply or "milo" not in reply, (
        f"pet-name recall confabulated: {reply!r}")
