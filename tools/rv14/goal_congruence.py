"""The last structural idea, and the one appraisal theory actually points at.

Every attempt so far scored the predicate in ABSOLUTE terms ("is this word
bad?"). Lazarus/Scherer appraisal is RELATIVE: an event is adverse when it
frustrates a goal the person actually holds, and goal-congruent otherwise.
That is testable here WITHOUT any polarity vocabulary, using RAVANA's own
stance store as the goal source: does the predicate's direction oppose
something the user is on record as valuing?

Setup is the honest version of the experiment, not the convenient one. Each
user states their goals FIRST (neutral, emotion-free statements). RAVANA
mines them into stances. THEN the adverse/benign event probes are fired and
the readout must use ONLY what the engine learned. No probe is used to fit.

The point of running this is to be able to report the answer honestly either
way — including "relative appraisal does not rescue it".
"""
import io
import os
import re
import sys
import contextlib

os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
sys.path.insert(0, os.path.join(PROJ, "tools", "rv14"))
from bindtree import bind, verify  # noqa: E402

bind(PROJ)

from probe_set import PROBES  # noqa: E402
import numpy as np  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

# Goals the user states BEFORE any probe. Neutral, emotion-free, and each one
# is the subject of at least one adverse AND one benign probe, so goal
# congruence alone cannot win without reading the predicate's direction.
GOALS = [
    "i care about my housing situation",
    "i care about my studies",
    "i care about money",
    "i care about my car",
    "i care about my phone",
    "i care about my visa",
    "i care about my family",
    "i care about my health",
    "i care about my laptop",
]


def main():
    verify(PROJ)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="rv14goal")
    gv = eng._glove_vector

    print("--- teaching goals (pre-probe, never a probe utterance)")
    for g in GOALS:
        with contextlib.redirect_stdout(io.StringIO()):
            eng.process_turn(g)
    stances = getattr(getattr(eng, "user_model", None), "opinions", None)
    st = getattr(stances, "stances", {}) if stances is not None else {}
    print(f"    stances learned: {len(st)}")
    for k, v in list(st.items())[:14]:
        pol = getattr(v, "polarity", None)
        print(f"      {str(k)[:44]:46} polarity={pol}")

    # Goal vector = mean unit vector of the stance topics the engine learned.
    goal_words = []
    for k in st:
        goal_words += [w for w in re.findall(r"[a-z']+", str(k).lower()) if len(w) >= 3]
    for g in GOALS:
        goal_words += [w for w in re.findall(r"[a-z']+", g.lower()) if len(w) >= 3]
    gv_goal = [gv(w) for w in goal_words]
    gv_goal = [v for v in gv_goal if v is not None]
    if not gv_goal:
        print("    NO goal vector — engine mined nothing; experiment void")
        return
    gc = np.mean(gv_goal, axis=0)
    gc /= np.linalg.norm(gc) + 1e-9

    # The PREDICATE is the eventive token: the verb / past participle. Pulled
    # structurally (first verb-shaped word after the subject+noun phrase), not
    # from a list — a crude but generalising extractor.
    VSHAPE = re.compile(
        r"\b(was|were|is|are|has|have|had|got|got)\s+(\w+(?:ed|ied|wn|ne|pt))\b"
        r"|\b(\w+(?:ed|ied))\b")

    def predicate(u):
        m = VSHAPE.search(u.lower())
        if not m:
            ws = re.findall(r"[a-z']+", u.lower())
            return ws[-1] if ws else ""
        return next(g for g in m.groups() if g)

    print(f"\n--- goal-congruence readout (goal vector from {len(gv_goal)} words)")
    rows = []
    print(f"{'probe':50} {'pred':14} {'cos(goal)':>10}")
    print("-" * 78)
    for utt, label, _n in PROBES:
        p = predicate(utt)
        v = gv(p)
        if v is None:
            v = np.zeros_like(gc)
        n = np.linalg.norm(v)
        s = float(v @ gc / n) if n > 1e-9 else 0.0
        rows.append((label, s, p))
        print(f"{utt[:50]:50} {p:14} {s:>10.4f}")

    adv = [s for l, s, _ in rows if l == "adverse"]
    ben = [s for l, s, _ in rows if l == "benign"]
    print(f"\n  mean cos(adverse) = {np.mean(adv):+.4f}")
    print(f"  mean cos(benign)  = {np.mean(ben):+.4f}")
    print(f"  difference (adv-ben) = {np.mean(adv) - np.mean(ben):+.4f}")

    def auc(s, y):
        pos = [a for a, b in zip(s, y) if b == "adverse"]
        neg = [a for a, b in zip(s, y) if b == "benign"]
        w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return w / (len(pos) * len(neg))

    sc = [r[1] for r in rows]
    lb = [r[0] for r in rows]
    print(f"  AUC(pred vs goal) = {auc(sc, lb):.3f}")
    # best zero-FP threshold in-sample (optimistic upper bound)
    cands = sorted(set(sc))
    best = max(((sum(1 for s in adv if s >= t), sum(1 for s in ben if s >= t), t)
                for t in cands), key=lambda z: (z[0], -z[1]))
    print(f"  BEST in-sample zero-FP point: recall {best[0]}/{len(adv)} "
          f"FP {best[1]}/{len(ben)}  (thr {best[2]:+.4f}) — OPTIMISTIC")

    eng.stop_background_learning()


if __name__ == "__main__":
    main()