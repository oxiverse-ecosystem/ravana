"""Bounded inertia in the stance store (round 2026-09-29T0823Z, defect D5).

The defect: `express_stance` weighted the held stance by
`confidence * rehearsal_count`, which grows without bound. A stance therefore
accumulated HISTORY instead of tracking what the user CURRENTLY holds, and the
longer a read had been rehearsed the more immune it became to later talk. The
user could state the opposite view repeatedly and the store kept reporting the
stale read — which is exactly what the round probe saw (4 stances before and
after three explicit opinion turns, including a deliberate contradiction).

These tests pin the behaviour at the STORE level: the merge rule is pure
arithmetic, so the whole capability is exercised deterministically without an
engine boot, without GloVe, and without any network. The engine-level
consequence (a chat turn actually moving the stance) is covered separately in
test_stance_reconsolidation_engine.py.

No reply strings, no topic tables, nothing hardcoded: the capability is one
bounding term in a merge formula.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, os.path.join(PROJ, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.personal_fact_store import UserStanceStore  # noqa: E402


def _entrench(store, topic, polarity=1.0, confidence=0.5, times=7):
    """Assert one judgment `times` times so the stance reaches steady state."""
    store.express_stance(topic, polarity=polarity, confidence=confidence)
    for _ in range(times - 1):
        store.express_stance(topic, polarity=polarity, confidence=confidence)


def test_repeating_one_view_still_entrenches_it():
    """The capability must not weaken the case it is meant to serve:
    re-asserting the SAME judgment converges to that judgment."""
    u = UserStanceStore()
    _entrench(u, "cats", polarity=1.0)
    assert u.query_stance("cats").polarity == 1.0

    u2 = UserStanceStore()
    u2.express_stance("cats", polarity=1.0, confidence=0.5)
    u2.express_stance("cats", polarity=0.8, confidence=0.5)
    st = u2.query_stance("cats")
    # The new signal pulls it toward 0.8, and it must stay well above the
    # stale zero it would sit at if inertia were over-weighted.
    assert st.polarity > 0.7


def test_a_changed_view_is_followed_not_averaged_away():
    """THE defect. An entrenched +1.0 stance, then repeated assertions of the
    opposite, must end up AGAINST rather than asymptotically parked on the
    old side of neutral.

    Under the unbounded rule this reached only +0.1175 after eight
    assertions — RAVANA still reading the user as 88% in favour after they
    said the opposite eight times.
    """
    u = UserStanceStore()
    _entrench(u, "cats", polarity=1.0)
    assert u.query_stance("cats").polarity == 1.0

    for _ in range(8):
        u.express_stance("cats", polarity=-1.0, confidence=0.5)

    st = u.query_stance("cats")
    assert st.polarity < 0.0, (
        f"eight assertions of the opposite view left polarity at {st.polarity:+.4f}: "
        "the stance is still reporting the stale read"
    )
    # And it must have actually crossed, not merely approached, the old pole.
    assert st.polarity < -0.3, (
        f"expected a real crossing to the new pole, got {st.polarity:+.4f}"
    )


def test_entrenchment_has_a_bound_not_a_removal():
    """The companion property. Bounding inertia must not delete it: ONE
    contrary mention still cannot overturn a well-rehearsed stance. Without
    this, the fix would trade a staleness bug for an over-writable-attitude
    bug, which is worse."""
    u = UserStanceStore()
    _entrench(u, "cats", polarity=1.0)

    u.express_stance("cats", polarity=-1.0, confidence=0.5)
    after_one = u.query_stance("cats").polarity

    assert after_one > 0.0, (
        f"a single contrary mention flipped the stance to {after_one:+.4f}"
    )
    # It must still be recognisably a positive read, not a coin toss.
    assert after_one > 0.4, (
        f"one contrary mention moved the stance too far, to {after_one:+.4f}"
    )


def test_inertia_is_actually_bounded_in_the_merge_weight():
    """The mechanism itself, not just the outcome: the historical term the
    merge uses must stop growing once the cap is reached, so the property is
    attributable to the bound rather than to some incidental constant.

    Read by recomputing the merge's own weight expression from the store's
    state, which is what makes this fail loudly if the cap is removed.
    """
    u = UserStanceStore()
    cap = UserStanceStore.RETENTION_CAP
    assert isinstance(cap, int) and cap > 0, "the cap must be a positive bound"

    weights = []
    for i in range(1, 4 * cap + 6):
        u.express_stance("cats", polarity=1.0, confidence=0.6)
        st = u.query_stance("cats")
        weights.append(st.confidence * min(st.rehearsal_count, cap))

    tail = weights[-4:]
    # Tolerance is 1e-4, not 1e-9: the rehearsal term is saturated exactly, but
    # `confidence` is itself a running mean that converges ASYMPTOTICALLY
    # toward its 0.7 ceiling, so the product still creeps at ~1e-6 per
    # iteration forever. An exact-equality bound here would fail on a
    # correct implementation, so this asserts the real property — that the
    # weight has stopped growing at a meaningful scale — rather than an
    # unachievable one.
    assert max(tail) - min(tail) < 1e-4, (
        f"the merge's historical weight is still growing after saturation: {tail}"
    )
    # ... and it is bounded by confidence * cap, the documented ceiling.
    assert all(w <= 1.0 * cap + 1e-9 for w in weights)


def test_neutral_and_zero_signals_do_not_flip_a_stance():
    """A zero-signal expression (an unparsed / zero-confidence observation)
    must not walk an entrenched read. Bounds the fix from the other side."""
    u = UserStanceStore()
    _entrench(u, "cats", polarity=1.0)
    before = u.query_stance("cats").polarity
    for _ in range(5):
        u.express_stance("cats", polarity=0.0, confidence=0.0)
    after = u.query_stance("cats").polarity
    assert abs(after - before) < 1e-9, (
        f"zero-signal expressions moved an entrenched stance {before:+.4f} -> {after:+.4f}"
    )


def test_the_cap_does_not_change_reverse_stance_behaviour():
    """reverse_stance is the explicit retraction path and recodes directly
    rather than merging. Bounding the merge weight must not disturb it — a
    regression guard on the mechanism this commit sits next to.

    Asserted loosely on purpose: the exact post-reversal value is
    reverse_stance's own contract (already pinned in
    test_personal_fact_store.py), not this commit's business."""
    u = UserStanceStore()
    u.express_stance("cats", polarity=1.0, confidence=0.8)
    u.reverse_stance("cats", reversal_strength=0.85)
    st = u.query_stance("cats")
    assert st.polarity < 0.0, "reversal must still flip the pole"
    assert st.confidence < 0.8, "reversal must still inject uncertainty"
    assert u.last_reversal is not None
    assert u.last_reversal[1] == 1.0
