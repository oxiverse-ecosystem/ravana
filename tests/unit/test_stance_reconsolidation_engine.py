"""Engine-level coverage for bounded stance inertia (round 2026-09-29T0823Z).

The store-level merge is pinned in test_stance_reconsolidation.py (pure
arithmetic, no boot). This file covers the other half of the claim: that a
CHAT TURN actually moves the stance, which is the only thing the user
experiences. Without it, a cap that was correct on paper but unreachable
from process_turn would pass the whole suite.

The A/B is done by REBINDING the class attribute
(UserStanceStore.RETENTION_CAP) rather than by editing the source, so the
unbounded rule is exercised with no stash/pop and no way to leave the tree
modified. Measured on this path, 8 positive then 8 negative utterances on
one topic:

    capped   (cap=4)          after 8 neg: -0.6930
    uncapped (cap=1e9)        after 8 neg: -0.5426

ATTRIBUTION, and why this file asserts a MAGNITUDE and not a direction.
When the A/B was first written the uncapped arm ended at +0.0686 -- still
positive -- so "the cap is what makes the stance move at all" was a valid
reading, and the test asserted exactly that. That reading is no longer
available: a later fix in the same round (the copular-evaluative miner's
sign-aware abstain, "an opposite assertion is evidence; only a redundant one
abstains") independently makes an opposite-sign read admissible, so the
unbounded arm now converges too. Two mechanisms deliver the direction; only
one of them is the cap.

Asserting on direction would therefore be a coin flip between two correct
fixes, and reverting either to turn it green would delete real coverage.
What the cap UNIQUELY owns is inertia saturation -- how much of a held
stance's rehearsal history survives into the next merge, and hence how
quickly a re-asserted view takes over. That is measured, monotonic in the
cap, and is asserted below on the divergence between the arms. The bounded
merge itself is pinned arithmetically in test_stance_reconsolidation.py.

Boots three engines (~30s each); all fixtures are module-scoped so the cost
is paid once, not per test.
"""
import contextlib
import io
import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (PROJ,
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.chat.personal_fact_store import UserStanceStore  # noqa: E402

TOPIC = "long evening walks"
POSITIVE = ("i really like long evening walks.", "long evening walks are really good.")
NEGATIVE = "long evening walks are really bad."
N = 8
SHIPPED_CAP = UserStanceStore.RETENTION_CAP


def _drive(suffix, cap):
    """8 positive then 8 negative utterances; return the final polarity."""
    UserStanceStore.RETENTION_CAP = cap
    try:
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix=suffix)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for i in range(N):
                eng.process_turn(POSITIVE[0] if i == 0 else POSITIVE[1])
            entrenched = eng.user_model.opinions.stances[TOPIC].polarity
            for _ in range(N):
                eng.process_turn(NEGATIVE)
            final = eng.user_model.opinions.stances[TOPIC].polarity
        return entrenched, final
    finally:
        UserStanceStore.RETENTION_CAP = SHIPPED_CAP


@pytest.fixture(scope="module")
def capped():
    return _drive("stancecap_capped", SHIPPED_CAP)


@pytest.fixture(scope="module")
def uncapped():
    return _drive("stancecap_uncapped", 10 ** 9)


@pytest.fixture(scope="module")
def minimal_cap():
    return _drive("stancecap_minimal", 1)


def test_repeating_one_judgment_still_entrenches_it(capped):
    """The fix must not weaken the case it exists to serve: saying the same
    thing eight times still settles the stance at the pole."""
    entrenched, _ = capped
    assert entrenched > 0.9, (
        f"eight repetitions of ONE view only reached {entrenched:+.4f}: "
        "bounding inertia weakened ordinary accumulation")


def test_a_reversed_view_is_followed_on_the_live_path(capped):
    """THE capability as the user meets it: after entrenching a view, saying
    the OPPOSITE eight times must actually move the stored read past neutral."""
    _, final = capped
    assert final < 0.0, (
        f"eight assertions of the opposite view left the stance at {final:+.4f} "
        "on the live engine path: still reporting the stale read")


def test_the_cap_is_what_makes_the_difference(uncapped, capped, minimal_cap):
    """The A/B, as a regression on the ATTRIBUTION.

    The cap's job is inertia saturation: it bounds how much of a held
    stance's rehearsal history survives into the next merge, so the user's
    CURRENT view takes over faster than it would against unbounded history.
    The three arms share a seed, a topic and a turn sequence, differing only
    in RETENTION_CAP, and the further the bound the further the read travels:

        cap=1    -> -0.9891   (inertia essentially gone)
        cap=4    -> -0.6930   (shipped)
        cap=1e9  -> -0.5426   (unbounded)

    Asserting monotonicity across all three pins the cap as a real, live
    input to the merge on the engine path, which is what this file exists to
    cover. Note the assertion is deliberately NOT "the unbounded arm stays
    positive": an independent fix in the same round (the copular-evaluative
    miner's sign-aware abstain) also lets the unbounded arm converge, so
    direction is not attributable to the cap alone. See the module docstring.
    """
    _, unbounded = uncapped
    _, shipped = capped
    _, minimal = minimal_cap
    assert minimal < shipped < unbounded, (
        "the cap no longer orders the trajectory: "
        f"cap=1 -> {minimal:+.4f}, cap=4 -> {shipped:+.4f}, "
        f"cap=1e9 -> {unbounded:+.4f}. Bounded inertia is either inert on "
        "the live path or no longer monotone in the cap")


def test_the_cap_is_restored_after_the_ab():
    """The uncapped arm rebinds a CLASS attribute. If it leaked, every later
    test in the session would run against a silently different merge rule."""
    assert UserStanceStore.RETENTION_CAP == SHIPPED_CAP
