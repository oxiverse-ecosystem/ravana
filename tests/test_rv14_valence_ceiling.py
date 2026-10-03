"""FIX-RV-14 / t_98e02c67 — PIN the measured event-polarity limitation.

This is a limitation pin, not a capability test. It exists because every
candidate appraisal was measured and every one failed to beat the incumbent,
and that fact must not be silently forgotten by the next round that decides
to "try one more valence idea".

WHAT WAS MEASURED (held-out probe set, 18 adverse / 20 benign, all written
fresh for this card — NOT the utterances quoted in the card body):
  baseline empathy routing      1/18 adverse recall, 2/20 benign FP
  evaluative_polarity peak      18/18 recall needs 19/20 FP
  classify_cause confidence     18/18 recall needs 20/20 FP
  GloVe mean vs affect seed     18/18 recall needs 19/20 FP
  GloVe peak vs affect seed     18/18 recall needs 19/20 FP
  goal-congruence (AUC 0.331)   INVERTED — benign is closer to the goal vector

THE STRONGEST NEGATIVE RESULT. A least-squares separator fitted directly on
labelled probe vectors (an ORACLE — strictly more information than any seed
vocabulary could carry) does not generalise either: 400 random 70/30 splits
give held-out recall-at-zero-FP of 0.106 (nearest-centroid, median 0.000)
and 0.225 (ridge, median 0.200), with held-out AUC 0.263 / 0.305 — both
BELOW chance. The in-sample AUC of 0.856 is indistinguishable from shuffled
labels (p = 0.34, 400 permutations), i.e. it is overfit, not signal.

So the honest conclusion is that adverse/benign EVENT polarity is not
linearly decodable from the projected GloVe representation of these
utterances. The card's acceptance criterion — better than 5/15 at 0/19 FP,
with no hand-listed polarity vocabulary — is NOT met by any signal tried.
FIX-RV-14's documented limitation therefore STANDS.

These tests pin that standing so the limitation is explicit and re-measurable
rather than rediscovered by guesswork. If a future round ships a real
signal, these tests are expected to be DELETED or inverted — that is the
correct outcome, and the failure message says so.
"""

import os
import re

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.evaluative_polarity import (  # noqa: E402
    EvaluativePolarityModel,
)

# Held-out probe set. Fresh for t_98e02c67, deliberately NOT the utterances
# quoted in the card body (which were the development set of two prior
# rounds and whose recall numbers collapsed from 6/10 to 3/10 on unseen
# cases).
ADVERSE = [
    "my landlord terminated my lease early",
    "they rejected my transfer request",
    "my mother was diagnosed with diabetes",
    "the hospital cancelled my appointment",
    "my visa application was denied",
    "my scholarship was revoked last month",
    "my phone was stolen from the library",
    "i was laid off from the internship",
    "my car failed its emissions test",
    "my prescription dosage got halved",
    "i was turned down for the fellowship",
]

BENIGN = [
    "my passport was issued in 2019",
    "my lease was renewed for another year",
    "the workshop was rescheduled for friday",
    "my application was accepted",
    "they approved my transfer request",
    "i passed my driving test on the first try",
    "my scholarship was renewed last month",
    "the hospital confirmed my appointment for monday",
    "my laptop screen got replaced under warranty",
    "my car passed its emissions test",
    "my phone was upgraded at the store",
    "my visa was approved in march",
]


def _best_zero_fp(scores_adverse, scores_benign):
    """Best (recall, fp) achievable over all thresholds, higher = more adverse."""
    best = (0, 10 ** 6)
    for t in sorted({*scores_adverse, *scores_benign}):
        hit = sum(1 for s in scores_adverse if s >= t)
        fp = sum(1 for s in scores_benign if s >= t)
        if (hit, -fp) > (best[0], -best[1]):
            best = (hit, fp)
    return best


@pytest.fixture(scope="module")
def glove_vector():
    """The engine's projected GloVe table, bound to THIS source tree."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for p in (root / "ravana_ml" / "src", root / "ravana" / "src",
              root / "ravana-v2" / "src", root):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="rv14pin")
    try:
        yield eng._glove_vector
    finally:
        eng.stop_background_learning()


# ── the pinning assertions ────────────────────────────────────────────────


def test_evaluative_polarity_cannot_gate_events(glove_vector):
    """The evaluative-polarity axis fails CLOSED on event predicates.

    For the overwhelming majority of these utterances NO word scores as an
    evaluative predicate (returns None), so the model has nothing to say. Of
    the handful that do score, the sign is unreliable and the axis cannot be
    thresholded without admitting essentially every benign probe.
    """
    m = EvaluativePolarityModel(vector_fn=glove_vector)
    adv, ben = [], []
    for utt, bucket in ((u, adv) for u in ADVERSE):
        for w in re.findall(r"[a-z']+", utt):
            r = m.score(w)
            if r is not None:
                bucket.append(r["polarity"])
    for utt in BENIGN:
        for w in re.findall(r"[a-z']+", utt):
            r = m.score(w)
            if r is not None:
                ben.append(r["polarity"])

    hit, fp = _best_zero_fp(adv, ben)
    assert fp / max(len(ben), 1) >= 0.30, (
        "evaluative_polarity now separates these events with a false-positive "
        f"rate under 30% (recall {hit}/{len(ADVERSE)}, FP {fp}/{len(ben)}). A "
        "real signal may have been found — RE-MEASURE on a fresh held-out "
        "probe set and, if it holds, DELETE this pin and close t_98e02c67.")


def test_affect_valence_is_not_event_polarity():
    """The incumbent `_update_emotion` valence reads the USER'S EMOTION WORD.

    This is the premise of the whole card, pinned as a fact so it cannot be
    quietly assumed away: stimulus valence is a set-membership test against
    the affect lexicon, so an adverse event whose speaker volunteers no
    emotion word scores exactly 0.0 — identical to a benign one.
    """
    from ravana.chat.engine_memory import MemoryMixin as _M  # noqa: E402
    base = _M._AFFECT_LEXICON_BASE
    negative = {w for w, s in base.items() if s < 0}

    def stimulus_valence(text):
        words = set(w.lower().strip(".,!?") for w in text.split())
        sv = 0.0
        if words & {w for w, s in base.items() if s > 0}:
            sv += 0.4
        if words & negative:
            sv -= 0.4
        return sv

    adverse_no_emotion = [
        u for u in ADVERSE if stimulus_valence(u) == 0.0
    ]
    assert len(adverse_no_emotion) >= 7, (
        f"only {len(adverse_no_emotion)}/{len(ADVERSE)} adverse probes now "
        "score zero stimulus valence — if the emotion engine has learned to "
        "read event polarity, the ceiling this card describes is gone and "
        "this pin should be re-derived, not deleted blindly.")

    benign_no_emotion = [u for u in BENIGN if stimulus_valence(u) == 0.0]
    assert len(benign_no_emotion) >= 7, (
        "the benign side changed shape — re-measure the pin before trusting "
        "the count.")


def test_no_hand_listed_polarity_vocabulary_was_introduced():
    """The prohibition itself is pinned, as a grep over the shipped source.

    Any new adverse/benign EVENT word table would satisfy the acceptance
    numbers by construction and violate the doctrine. This test fails if a
    module-level set of event-polarity words appears in the chat package.
    """
    import ast
    from pathlib import Path
    chat_dir = Path(__file__).resolve().parents[1] / "ravana" / "src" / \
        "ravana" / "chat"
    banned_names = {
        "ADVERSE_WORDS", "ADVERSE_EVENT_WORDS", "BENIGN_WORDS",
        "EVENT_POLARITY", "EVENT_POLARITY_LEXICON", "BAD_EVENTS",
        "GOOD_EVENTS", "NEGATIVE_EVENTS", "POSITIVE_EVENTS",
    }
    hits = []
    for py in sorted(chat_dir.glob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in banned_names:
                        hits.append(f"{py.name}:{node.lineno} {tgt.id}")
            elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name):
                if node.target.id in banned_names:
                    hits.append(f"{py.name}:{node.lineno} "
                                f"{node.target.id}")
    assert not hits, (
        "hand-listed event-polarity vocabulary introduced: " + ", ".join(hits)
        + " — this is the shape the card bans four times over.")