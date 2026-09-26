"""FIX-RV-15 — stance topic keys: no speech-act verb, no duplicates, no
mis-signed comparative.

The defect, verified cold at c21b6c1e: a comparative disclosure
("i think remote work is better than office work") produced THREE stance keys
for ONE attitude — 'think remote work' (the reporting verb fused to the topic,
a key no query can ever resolve), 'remote work' (the endorsed side) and
'office work' (the REJECTED side) — and it gave the rejected side the pattern's
POSITIVE polarity, so the store endorsed the side the user had just rejected
and the direction of the comparison was lost.

These tests drive the real `UserModel.mine_personal_facts`, i.e. the same call
`process_turn` makes, so they exercise the shipped path rather than a replica.

Rotating probes: the comparative cases here are phrasings developed AFTER the
fix shape was chosen, and the frame-strip tests use verbs ("i figure",
"she maintains") that appear in no miner pattern, to show the strip is
grammatical rather than a lookup of the verbs the miner happens to know.
"""
import inspect
import os
import sys

import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

# The `ravana` package must be bound to THIS worktree for the whole test
# module, and that is not defensive over-engineering — it is correctness.
#
# tests/unit/test_pre_registered.py inserts a HARD-CODED absolute path
# (C:\Users\Likhith\Documents\Projects\ravana) at sys.path[0] at import time.
# Whichever test module pytest collects first decides where `ravana` binds for
# the WHOLE session, and alphabetically that module is test_pre_registered — so
# by the time this file is imported, `ravana` is already bound to the MAIN
# checkout, which sits on whatever branch was last checked out and may predate
# this branch entirely.
#
# Two failure modes follow, and the second is the dangerous one:
#   1. a new module added on this branch is INVISIBLE, and the error reads
#      "No module named 'ravana.chat.slot_naming'" — which looks like an
#      uncommitted file (it is committed; `git cat-file -p HEAD:<path>` proves
#      it) rather than a path-ordering artifact;
#   2. worse, `UserModel` would come from the MAIN checkout, which does NOT
#      contain this fix, so the tests would pass or fail against code nobody
#      is shipping — a verdict about the wrong tree.
#
# So: drop any `ravana` bound elsewhere, put this worktree's source first,
# and ASSERT the binding before testing anything.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_RAVANA_SRC = os.path.join(_REPO, "ravana", "src")

# Drop any `ravana` bound to a different checkout, and put THIS worktree's
# package source first.
for _m in [k for k in sys.modules
           if k == "ravana" or k.startswith("ravana.")]:
    del sys.modules[_m]
if _RAVANA_SRC in sys.path:
    sys.path.remove(_RAVANA_SRC)
sys.path.insert(0, _RAVANA_SRC)

# Prove the binding is the worktree, so a future reordering fails loudly here
# instead of quietly testing the wrong tree.
import ravana as _ravana_pkg  # noqa: E402
_expected = os.path.join(_RAVANA_SRC, "ravana")
_bound = list(getattr(_ravana_pkg, "__path__", []))
assert _expected in _bound, (
    f"ravana bound to {_bound}, expected this worktree's {_expected} — "
    "these tests would otherwise verify a different checkout's code")
_ravana_pkg.__path__ = [_expected]

from ravana.chat.user_model import UserModel  # noqa: E402
from ravana.chat.personal_fact_store import UserStanceStore  # noqa: E402

# Now that the package is provably bound to this worktree, the ordinary
# package import is correct and sufficient — no separate file-path load, which
# would create a SECOND module object for the same file and make identity
# comparisons between the two misleading.
from ravana.chat.slot_naming import strip_reporting_frame  # noqa: E402

# Pin the provenance that matters: the function under test must come from THIS
# worktree's file, not merely from some `slot_naming` that happened to import.
assert os.path.normcase(inspect.getsourcefile(strip_reporting_frame)) == \
    os.path.normcase(os.path.join(_RAVANA_SRC, "ravana", "chat",
                                  "slot_naming.py")), (
    f"strip_reporting_frame came from "
    f"{inspect.getsourcefile(strip_reporting_frame)}, not this worktree")


@pytest.fixture()
def um():
    """A UserModel with only the stores the opinion miner needs.

    Built via __new__ so the test does not pay the full engine boot (GloVe,
    concept graph, background learning) for a pure mining assertion.
    """
    m = UserModel.__new__(UserModel)
    from ravana.chat.personal_fact_store import UserStanceStore
    m.opinions = UserStanceStore()
    m.turn_num = 0
    m.emotional_state = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    m.interaction_history = []
    m.beliefs = {}
    m.personal_facts = None
    return m


def stances(um):
    return {k: v.polarity for k, v in um.opinions.stances.items()}


# ── A. the reporting frame must not fuse into the topic key ──────────────

def test_topic_strips_reporting_frame(um):
    """'i think remote work' -> 'remote work', never 'think remote work'."""
    assert um._opinion_topic("i think remote work") == "remote work"
    assert um._opinion_topic("i believe nuclear energy") == "nuclear energy"
    assert um._opinion_topic("she prefers winter") == "winter"


def test_topic_strips_unseen_reporting_verbs(um):
    """The strip is grammatical, not a list of the miner's known verbs.

    'figure' and 'maintain' appear in no miner pattern and in no seed list
    here; if the strip were a verb table these would leak like 'think' did.
    """
    assert um._opinion_topic("i figure remote work") == "remote work"
    assert um._opinion_topic("she maintains winter") == "winter"
    assert um._opinion_topic("they reckon summer") == "summer"


def test_topic_preserves_bare_noun_phrases(um):
    """The strip must not touch a phrase that does not open with a subject."""
    for phrase, expected in [
        ("remote work", "remote work"),
        ("office work", "office work"),
        ("quiet mornings", "quiet mornings"),
        ("running", "running"),
        ("small talk", "small talk"),
        ("people who talk", "people who talk"),
        ("books that last", "books that last"),
        ("cold water swimming", "cold water swimming"),
    ]:
        assert um._opinion_topic(phrase) == expected, phrase


def test_strip_reporting_frame_never_empties():
    """A frame with no complement is left intact rather than emptied."""
    assert strip_reporting_frame(["i", "think"]) == ["i", "think"]
    assert strip_reporting_frame(["think"]) == ["think"]
    assert strip_reporting_frame([]) == []


def test_frame_subjects_are_a_closed_class_subset(um):
    """The strip draws its subjects from the caller's own closed-class set.

    If the two could drift, a word the miner treats as a stop word would stop
    being recognised as a frame subject (or vice versa) and the chokepoint
    would silently stop agreeing with itself.
    """
    for subj in ("i", "you", "he", "she", "we", "they"):
        assert subj in um._OPINION_STOP, subj


# ── B. one proposition, one canonical key (no shredding) ─────────────────

def test_comparative_yields_no_malformed_duplicate(um):
    """The fused 'think remote work' key must not exist alongside 'remote work'."""
    um.mine_personal_facts("i think remote work is better than office work")
    keys = set(um.opinions.stances)
    assert "think remote work" not in keys
    assert "remote work" in keys


def test_comparative_does_not_multiply_keys(um):
    """One comparative disclosure, not three keys."""
    um.mine_personal_facts("i think remote work is better than office work")
    # exactly the two sides of the comparison, nothing more
    assert set(um.opinions.stances) == {"remote work", "office work"}


def test_repeated_disclosure_converges(um):
    """Re-expressing the same attitude must not spawn a parallel key.

    The store has to converge, not multiply: the same proposition stated twice
    is still one key, and the repeat entrenches it rather than forking it.
    """
    for _ in range(3):
        um.mine_personal_facts("i think remote work is better than office work")
    assert set(um.opinions.stances) == {"remote work", "office work"}


# ── C. the comparison keeps its DIRECTION ───────────────────────────────

def test_comparative_preserves_direction(um):
    """The endorsed side is positive and the rejected side is negative.

    This is the part the defect got outright backwards: the old code read
    group(lastindex) — the LOSER — and gave it the pattern's positive polarity,
    so RAVANA claimed to be *for* office work right after the user said it was
    worse than remote work.
    """
    um.mine_personal_facts("i think remote work is better than office work")
    pol = stances(um)
    assert pol["remote work"] > 0, pol
    assert pol["office work"] < 0, pol


@pytest.mark.parametrize("utterance,loser", [
    ("to me tea beats coffee every time", "coffee"),
    ("in my view winter beats summer", "summer"),
    ("honestly mountain air beats city smog", "city smog"),
])
def test_dyadic_comparatives_keep_direction(um, utterance, loser):
    """Rotating probes: the REJECTED side of every dyadic comparative is negative.

    The endorsed side is deliberately not asserted by exact key here — see
    `test_leading_adjunct_still_fuses_into_key`, which documents that a
    sentence-adverbial adjunct in front of the subject survives into the key.
    What must hold for the direction fix is that the two sides never share a
    sign, which is exactly what the defect got backwards.
    """
    um.mine_personal_facts(utterance)
    pol = stances(um)
    assert len(pol) == 2, (utterance, pol)
    signs = sorted(pol.values())
    assert signs[0] < 0 < signs[1], (utterance, pol)
    assert pol[loser] < 0, (utterance, pol)


@pytest.mark.xfail(
    reason="RESIDUAL, documented not hidden: a sentence-ADVERBIAL adjunct in "
           "front of the subject ('honestly ...', 'in my view ...') still fuses "
           "into the topic key. This is a DIFFERENT constituent from the "
           "reporting frame FIX-RV-15 strips (a preposed adverb/PP has no "
           "subject+verb pair to key off), and separating it needs either a POS "
           "tagger — measured unreliable here: classify_word_pos returns "
           "'noun' for believe/find/reckon — or a seed list of adverbs, which "
           "is the keyword table this card bans. Left failing on purpose so the "
           "limitation stays visible instead of being papered over.",
    strict=True)
def test_leading_adjunct_still_fuses_into_key(um):
    um.mine_personal_facts("honestly mountain air beats city smog")
    assert "mountain air" in um.opinions.stances


def test_single_side_opinion_keeps_its_sign(um):
    """A one-sided opinion is unaffected by the dyadic branch."""
    um.mine_personal_facts("i believe coffee is overrated")
    pol = stances(um)
    assert pol["coffee"] < 0, pol


def test_preference_is_unchanged(um):
    """A plain preference still yields a positive stance on its object."""
    um.mine_personal_facts("i really like quiet mornings")
    pol = stances(um)
    assert pol["quiet mornings"] > 0, pol


# ── the contrast is RECOVERABLE by a later contrastive query ─────────────

def test_contrastive_query_recovers_the_lean(um):
    """Both sides resolve independently, so a split query recovers the lean.

    This is what the shredding destroyed: with the loser stored positive, a
    contrastive query read two equally-positive stances and could not tell
    which way the user leaned. The FIX-RV-11 contrastive path splits a query
    on the connective and resolves each side through the store, so the fix is
    only real if the STORE now carries the direction.
    """
    um.mine_personal_facts("i think remote work is better than office work")
    assert um.opinions.resolve_topic("remote work") == "remote work"
    assert um.opinions.resolve_topic("office work") == "office work"
    assert stances(um)["remote work"] * stances(um)["office work"] < 0
