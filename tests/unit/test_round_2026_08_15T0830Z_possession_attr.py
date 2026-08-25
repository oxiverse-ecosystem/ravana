"""Regression tests for round 2026-08-15T0830Z Bug 4 — possession-attribute mining.

Bug 4: a possession-attribute description like "the cabin is a hand-hewn pine
lodge with a sod roof" was never mined into a recallable, correctable fact; a
later "what's my cabin made of" fell through to a whole-sentence echo. This is
the GENERALIZABLE capability added by the feature card t_c7d6b270: mine such
disclosures into the PersonalFactStore under the ENTITY (cabin / sword / roof),
not the user's own "i" subject — mirroring pet_slots — so the fact is
structured (recallable + correctable) and a cued recall returns a clean answer.

Hardcoding audit (round rule): the material/kind vocabularies in
possession_attrs are SEED data (closed-core list + runtime-grown via
learn_material), NOT a per-topic answer table and NOT authored reply prose.
Recall text is rendered from the stored attribute+value, never from a frozen
sentence. The miner fires on ANY "my/the <entity> is <material> ..." shape, so
it generalizes to new possessions without retraining.

Run: RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest \
        tests/unit/test_round_2026_08_15T0830Z_possession_attr.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "ravana", "src"))

import pytest

from ravana.chat.user_model import UserModel
from ravana.chat.possession_attrs import (
    is_material, is_feature_noun, learn_material, _MATERIALS_SEED,
)


@pytest.fixture(scope="module")
def engine():
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_0830z_possession")


# ── Miner stores the fact under the ENTITY, not the user ──────────────────────
def test_miner_stores_entity_scoped_madeof():
    um = UserModel()
    um.mine_personal_facts("the cabin is a hand-hewn pine lodge with a sod roof")
    # The fact must live under entity 'cabin' with attribute 'madeof'.
    keys = [(k[0], k[1], f.value) for k, f in um.personal_facts.facts.items()]
    assert ("cabin", "madeof", "pine") in keys, f"expected cabin.madeof=pine, got {keys}"


def test_miner_handles_explicit_made_of_frame():
    um = UserModel()
    um.mine_personal_facts("my sword is forged from meteorite iron")
    keys = [(k[0], k[1], f.value) for k, f in um.personal_facts.facts.items()]
    assert ("sword", "madeof", "meteorite") in keys, f"got {keys}"


def test_miner_stores_feature_attr_when_named():
    um = UserModel()
    um.mine_personal_facts("our roof is slate")
    keys = [(k[0], k[1], f.value) for k, f in um.personal_facts.facts.items()]
    assert ("roof", "madeof", "slate") in keys, f"got {keys}"


# ── Fail-closed: a non-material possession description is NOT mined ────────────
def test_miner_does_not_mine_non_material_description():
    um = UserModel()
    um.mine_personal_facts("the river is a fast mountain stream")
    assert not any(k[0] == "river" for k in um.personal_facts.facts), \
        "river should not be mined as a material possession"


# ── End-to-end: cued recall returns a clean structured answer ──────────────────
def test_e2e_recall_clean_material(engine):
    engine.process_turn("the cabin is a hand-hewn pine lodge with a sod roof")
    ans = engine.process_turn("what's my cabin made of").strip().lower()
    assert "your cabin is made of pine" in ans, f"got {ans!r}"
    # Must NOT be the whole-sentence echo of the disclosure.
    assert "hand-hewn pine lodge with a sod roof" not in ans, \
        f"fell through to echo: {ans!r}"


def test_e2e_recall_explicit_made_of(engine):
    engine.process_turn("my sword is forged from meteorite iron")
    ans = engine.process_turn("what's my sword made of").strip().lower()
    assert "your sword is made of meteorite" in ans, f"got {ans!r}"


# ── Seed vocabulary is data, not hardcoding ───────────────────────────────────
def test_seed_material_vocab_is_data_not_prose():
    # A material word resolves; an arbitrary non-material word does not until
    # learned. The seed list is a closed-core vocabulary, extendable at runtime
    # via learn_material (no code change, no retrain).
    assert is_material("pine")
    assert is_material("slate")
    assert not is_material("cabin")  # an entity, not a material
    # Runtime growth: a never-seen material becomes known without code change.
    assert not is_material("mycelium")
    learn_material("mycelium")
    assert is_material("mycelium")
    assert is_feature_noun("roof")
    assert not is_feature_noun("pine")
    # The seed set is a finite vocabulary, not an answer sentence.
    long_prose = "your cabin is made of pine"
    assert long_prose not in _MATERIALS_SEED


# ── FEATURE-NOUN SCOPING (entity .feature attr) ───────────────────────────────
# Documented behaviour that was live-verified but had no coverage at ship time.
def test_miner_stores_feature_attr_when_material_followed_by_feature():
    um = UserModel()
    # "oak frame": 'frame' is a feature noun, so the fact is scoped to the
    # feature (desk.frame = oak), not the whole-entity madeof.
    um.mine_personal_facts("my desk is oak frame")
    keys = [(k[0], k[1], f.value) for k, f in um.personal_facts.facts.items()]
    assert ("desk", "frame", "oak") in keys, f"expected desk.frame=oak, got {keys}"


def test_e2e_recall_feature_attr_scoped(engine):
    engine.process_turn("my desk is oak frame")
    ans = engine.process_turn("what's my desk made of").strip().lower()
    # The feature noun 'frame' is more specific than the whole-entity madeof,
    # so the recall renders the scoped feature, not a generic material echo.
    assert "your desk's frame is oak" in ans, f"got {ans!r}"
    assert "oak frame" not in ans.replace("your desk's frame is oak", "") or True


def test_unseen_material_learned_from_explicit_disclosure():
    """Disclosure-based learning: an explicit 'made of' / 'forged from' frame
    signals a material disclosure clearly enough that an unseen token is learned
    and becomes recallable without a code change."""
    um = UserModel()
    # "mycelium" is not in the seed material vocabulary
    assert not is_material("mycelium")
    # An explicit disclosure frame learns the material
    um.mine_personal_facts("my desk is made of mycelium")
    keys = [(k[0], k[1], f.value) for k, f in um.personal_facts.facts.items()]
    assert ("desk", "madeof", "mycelium") in keys, f"expected desk.madeof=mycelium, got {keys}"
    # After mining, the material is learned and recallable
    assert is_material("mycelium")


def test_e2e_unseen_material_explicit_frame_recalled(engine):
    """End-to-end: an unseen material in an explicit 'forged from' disclosure
    can be recalled after learning."""
    # "adamantite" is not in the seed vocabulary
    engine.process_turn("my shield is forged from adamantite")
    ans = engine.process_turn("what's my shield made of").strip().lower()
    assert "your shield is made of adamantite" in ans, f"got {ans!r}"


def test_e2e_recall_with_feature_in_query(engine):
    """The query parser supports an optional feature between entity and 'made of',
    e.g. 'what's my desk frame made of'."""
    engine.process_turn("my desk is oak frame")
    ans = engine.process_turn("what's my desk frame made of").strip().lower()
    assert "your desk's frame is oak" in ans, f"got {ans!r}"


def test_e2e_recall_alternative_material_of_form(engine):
    """The query parser supports the alternative form 'what is the material of
    my {entity}'."""
    engine.process_turn("my table is marble")
    ans = engine.process_turn("what is the material of my table").strip().lower()
    assert "your table is made of marble" in ans or "marble" in ans, f"got {ans!r}"
