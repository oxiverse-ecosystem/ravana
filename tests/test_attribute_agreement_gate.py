"""
Regression tests for the attribute-agreement gate (round auto/round-20260925T0823-fix-7).

Commit 3519e1bc taught three recall sites the rule "a name query may only be
answered by a name", each with its OWN private copy of the rule. Two defects
followed, both proven against the round head:

1. THE RULE COPIES DRIFTED. The pet site matched only
   (name|named|called) while the other two also matched "nickname" -- three
   sites, three answers to "did the user ask for a name?". One shared
   predicate (chat/attribute_gate.py) now owns the rule, so they agree BY
   CONSTRUCTION. This is the recurring lesson from the pet-slot rename: N
   hand-kept copies of one rule drift apart and the drift surfaces as an
   unrelated-looking CI failure.

2. A COMPOUND QUESTION IS NOT A NAME QUESTION. "what's my grandmother's name
   AND what does she make?" mentions "name", so the gate fired, deleted every
   non-name-shaped value -- including the very activity fact the SECOND clause
   asked for -- and recall abstained on a fact the user HAD stated
   (tests/test_round_2026_08_16_1745_d7.py::test_recall_grandmother_activity_via_process_turn).
   `asks_name_only` stands the gate down when a coordinator is followed by
   another question element, which is how English builds a compound question.

The gate itself is a grammatical position test, not a topic table: it holds
for any entity, any attribute, any language-level noun. No reply string is
asserted -- these read the real store and the real rendered clause.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat import attribute_gate
from ravana.chat.engine import CognitiveChatEngine


# ── the shared rule itself ────────────────────────────────────────────────────
def test_asks_name_only_true_for_a_pure_name_question():
    assert attribute_gate.asks_name_only("what is my cat's name?")
    assert attribute_gate.asks_name_only("who is my dog named?")


def test_asks_name_only_false_when_no_attribute_is_asked():
    # No name attribute mentioned at all -> the gate must not engage.
    assert not attribute_gate.asks_name_only("what does my cat do?")
    assert not attribute_gate.asks_name_only("tell me about my grandmother")


def test_asks_name_only_false_for_a_compound_question():
    # The defect: a coordinated SECOND question means the query asks for more
    # than the name, so restricting it to name-shaped values would delete the
    # fact the second clause asked for.
    assert not attribute_gate.asks_name_only(
        "what's my grandmother's name and what does she make?")
    assert not attribute_gate.asks_name_only(
        "what is my cat's name and how old is he")


def test_asks_name_only_is_shared_across_phrasings():
    # "nickname" is the same attribute said differently. The pet site used to
    # miss it entirely while the other two honoured it -- the drift.
    assert attribute_gate.asks_name_only("what is my cat's nickname?")
    assert attribute_gate.asks_name_only("what is my cat's name.")


def test_is_name_shaped_accepts_a_name_and_rejects_a_predicate():
    # Content still comes from what the user actually said; the gate only
    # decides ADMISSIBILITY of a stored value, never what to say.
    assert attribute_gate.is_name_shaped("pixel")
    assert attribute_gate.is_name_shaped("wren")
    # a predicate phrase is the entity's STATE, not its name
    assert not attribute_gate.is_name_shaped("diagnosed with a chronic illness")
    assert not attribute_gate.is_name_shaped("weaves baskets")
    assert not attribute_gate.is_name_shaped("")
    assert not attribute_gate.is_name_shaped(None)


# ── end-to-end: a compound question still recalls the activity ────────────────
def test_compound_question_still_recalls_the_activity(tmp_path):
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              data_dir=str(tmp_path), user_suffix="_t70655534")
    eng.process_turn("my grandmother Indira weaves baskets from river reeds")
    reply = eng.process_turn("what's my grandmother's name and what does she make?")
    r = reply.lower()
    assert "is weaves" not in r, f"spurious copula: {reply!r}"
    assert "grandmother indira weaves baskets" in r, \
        f"compound question lost the activity it asked for: {reply!r}"


def test_equational_disclosure_is_not_stored_twice(tmp_path):
    """The FIX-RV-13 possession branch must stand down on an equational
    disclosure. It stored ('i','dog','is a lurcher named wren') NEXT TO the
    equational ('i','dog','a lurcher named wren'), and every renderer prefixes
    a pet slot with 'your <slot> is' -- so recall read
    'your dog is IS a lurcher named wren'."""
    from ravana.chat.user_model import UserModel
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("my dog is a lurcher named wren", run_correction=True)
    vals = [f.value for f in um.personal_facts.facts.values()
            if not getattr(f, "superseded", False)]
    copula = [v for v in vals
              if isinstance(v, str) and v.split()
              and v.split()[0] in ("is", "are", "was", "were", "am")]
    assert not copula, f"value kept its copula -> doubled render: {vals}"


def test_possession_predicate_branch_still_mines_its_own_shape(tmp_path):
    """The guard must not over-reach: a genuine PREDICATE disclosure (no
    equational copula) is exactly what FIX-RV-13 added this branch to mine."""
    from ravana.chat.user_model import UserModel
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("my cat has been diagnosed with a chronic illness",
                           run_correction=True)
    vals = " ".join(str(f.value) for f in um.personal_facts.facts.values()
                    if not getattr(f, "superseded", False))
    assert "chronic illness" in vals, f"real predicate disclosure lost: {vals}"
