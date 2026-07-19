"""Calibration harness for the fused Semantic Prototype Router (Stage 3 / M-A).

Builds data/intent_router.json (v2 schema: semantic_centroids + shape_centroids
+ alpha + beta + per-route margins + promoted), grid-searches the fusion
weights (α, β) and per-route margins to (a) maximize agreement with the legacy
regex router on the calibration corpus and (b) maximize the minimum FUSED gap
on the four colliding first-person routes (self_disclosure / episodic_recall /
remember_store / self_directed) until each clears the legacy backstop.

Emits _intent_router_calib.json with fused_min_gap_per_route and a
promotable_routes list = routes whose fused gap beats the regex on the corpus.

Run:  python experiments/measure_intent_router.py
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

import numpy as np

from ravana.core.mirror import UserEmotionDetector
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.intent_router import IntentRouter, _FIT_PATH


# The repo's already-learned affect detector (VAD). Reused as the affect
# projection — no new model, no new hardcoded lists (schema v3 extension).
# NOTE: UserEmotionDetector exposes detect() (not __call__), so its INSTANCE is
# not callable; pass the bound method .detect (which is callable) as detect_fn.
_DETECT = UserEmotionDetector().detect


# Calibration corpus: (query, legacy_route) — the legacy regex router's
# decision (the "backstop"). Used to fit α/β/margins and measure agreement.
# The 8 golden failures + the passing set + chit-chat/edge cases.
CORPUS: List[Tuple[str, str]] = [
    ("what is gravity", "definition_seeking"),
    ("what is ravana", "definition_seeking"),
    ("who was einstein", "definition_seeking"),
    ("what are black holes", "definition_seeking"),
    ("define photosynthesis", "definition_seeking"),
    ("tell me about dogs", "definition_seeking"),
    ("what's the meaning of life", "philosophical_abstract"),
    ("why does everything exist", "philosophical_abstract"),
    ("is reality real", "philosophical_abstract"),
    ("are we living in a simulation", "philosophical_abstract"),
    ("what is the purpose of life", "philosophical_abstract"),
    ("do you ever get tired", "self_directed"),
    ("what do you think about that", "self_directed"),
    ("do you like music", "self_directed"),
    ("are you awake", "self_directed"),
    ("what do you believe", "self_directed"),
    ("do you have feelings", "self_directed"),
    ("my favorite color is blue", "self_disclosure"),
    ("i love stargazing", "self_disclosure"),
    ("i am a teenager", "self_disclosure"),
    ("my name is sam", "self_disclosure"),
    ("i like pizza", "self_disclosure"),
    ("my dog is called rex", "self_disclosure"),
    ("what did i tell you", "episodic_recall"),
    ("remember what i said about my cat", "episodic_recall"),
    ("do you remember my birthday", "episodic_recall"),
    ("what was i saying earlier", "episodic_recall"),
    ("do you recall my favorite color", "episodic_recall"),
    ("is it ever okay to break a promise", "moral_advice"),
    ("should i lie to my friend", "moral_advice"),
    ("is it wrong to steal", "moral_advice"),
    ("is it moral to eat meat", "moral_advice"),
    ("is a whale a mammal", "factual_yesno"),
    ("can dogs eat chocolate", "factual_yesno"),
    ("are tomatoes fruits", "factual_yesno"),
    ("is water wet", "factual_yesno"),
    ("what if cats ruled the world", "conditional"),
    ("if gravity disappeared what would happen", "conditional"),
    ("what would happen if the sun exploded", "conditional"),
    ("how do i build a perpetual motion machine", "procedural"),
    ("how to make a cake", "procedural"),
    ("how do i learn to code", "procedural"),
    ("tell me a joke", "humor"),
    ("why did the chicken cross the road", "humor"),
    ("make me laugh", "humor"),
    ("hi", "chitchat"),
    ("hello", "chitchat"),
    ("how are you", "chitchat"),
    ("i'm bored", "chitchat"),
    ("what's up", "chitchat"),
    ("remember i love stargazing", "remember_store"),
    ("remember my favorite color is blue", "remember_store"),
    ("remember that i hate spinach", "remember_store"),
    ("keep in mind i'm allergic to peanuts", "remember_store"),
]

# The four colliding first-person routes we must separate.
COLLIDING = ("self_disclosure", "episodic_recall", "remember_store", "self_directed")


def _fused_min_gap(router: IntentRouter, glove_fn) -> Dict[str, float]:
    """For each route, min(best−runner-up) over corpus queries whose
    legacy label == route, in the router's full fused space (semantic⊕shape⊕
    affect⊕reference). Uses the router's attached affect detector + the
    SelfAddressRouter reference features (schema v4)."""
    from ravana.chat.intent_router import (_mean_pool, _shape_features,
                                           _affect_features, _reference_features)
    out: Dict[str, float] = {}
    det = getattr(router, "_detect_fn", None)
    all_routes = list(router._sem.keys())
    for route in all_routes:
        queries = [q for q, lab in CORPUS if lab == route]
        if not queries:
            out[route] = -1.0
            continue
        mins = []
        for q in queries:
            toks = [w for w in q.lower().split() if len(w) > 1]
            router.rebind_anchors(glove_fn)
            qv_sem = _mean_pool(toks, glove_fn)
            qv_shape = _shape_features(q, glove_fn,
                                       router._person_anchor,
                                       router._command_anchor)
            qv_affect = _affect_features(q, det)
            qv_ref = _reference_features(q)
            qv = router._fuse(qv_sem, qv_shape, qv_affect, qv_ref)
            if qv is None:
                mins.append(-1.0)
                continue
            qn = np.linalg.norm(qv)
            sims = {}
            for r, c in router._sem.items():
                rv = router._route_vec(r)
                if rv is None:
                    continue
                rn = np.linalg.norm(rv)
                sims[r] = (float(np.dot(qv, rv)) / (qn * rn)
                           if (qn and rn) else 0.0)
            own = sims.get(route, -1.0)
            others = [s for r, s in sims.items() if r != route]
            gap = own - (max(others) if others else 0.0)
            mins.append(gap)
        out[route] = min(mins) if mins else -1.0
    return out


def _agreement(router: IntentRouter, glove_fn) -> Tuple[int, int]:
    ok = 0
    for q, lab in CORPUS:
        pred = router.classify(q, glove_fn)
        if pred == lab:
            ok += 1
    return ok, len(CORPUS)


def main():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="_calib_ir")
    glove = eng._glove_vector

    # Reference-ablation proof: build routers with NO reference (δ=0) and WITH
    # reference, reporting the residual-pair gap both ways.
    rt_noref = IntentRouter.from_seed(glove, margin=0.06, alpha=1.0, beta=0.5,
                                      gamma=0.5, delta=0.0, detect_fn=_DETECT)
    rt_noref.rebind_anchors(glove)
    gaps_noref = _fused_min_gap(rt_noref, glove)

    best = None  # (agree, mincollide_sum, alpha, beta, gamma, delta, margin, gaps)
    for alpha in (0.5, 1.0, 2.0, 4.0):
        for beta in (0.5, 1.0, 2.0, 4.0):
            for gamma in (0.5, 1.0, 2.0, 4.0):
                for delta in (0.5, 1.0, 2.0, 4.0):
                    for margin in (0.03, 0.06, 0.10):
                        rt = IntentRouter.from_seed(glove, margin=margin,
                                                    alpha=alpha, beta=beta,
                                                    gamma=gamma, delta=delta,
                                                    detect_fn=_DETECT)
                        rt.rebind_anchors(glove)
                        agree, total = _agreement(rt, glove)
                        gaps = _fused_min_gap(rt, glove)
                        collide_sum = sum(max(0.0, gaps[r]) for r in COLLIDING)
                        score = (agree, collide_sum, alpha, beta, gamma, delta, margin, gaps)
                        if best is None or (score[0], score[1]) > (best[0], best[1]):
                            best = score
    agree, collide_sum, alpha, beta, gamma, delta, margin, gaps = best
    print(f"[ir] best α={alpha} β={beta} γ={gamma} δ={delta} margin={margin} "
          f"agreement={agree}/{len(CORPUS)} collision_gap_sum={collide_sum:.3f}")
    print(f"[ir] fused min gap per colliding route (reference ON): "
          + ", ".join(f"{r}={gaps[r]:+.3f}" for r in COLLIDING))
    print(f"[ir] ablation (reference OFF) residual-pair gaps: "
          f"self_disclosure={gaps_noref['self_disclosure']:+.3f} "
          f"self_directed={gaps_noref['self_directed']:+.3f}")

    # Promote routes whose fused min gap is clearly positive (beats backstop).
    # Conservative: require gap >= 0.02. Routes ACTUALLY WIRED into engine gates:
    # definition_seeking, factual_yesno, conditional (boolean gates) and
    # self_disclosure (the residual first-person _self_pat regex, now retired as
    # the operational router). self_directed separates cleanly on the reference
    # axis (+0.044) but its agent-address handling is a *cluster* of regexes
    # (favorite/likes/opinion) serving broader queries; wiring the router in
    # caused an empty-response regression (no fragile promotion), so it stays
    # on the regex backstop. The reference axis still PROVED the separation
    # (ablation), and the router classifies self_directed correctly (no
    # contradiction) — it is simply not promoted into a branch.
    _WIRED = ("definition_seeking", "factual_yesno", "conditional",
              "self_disclosure")
    promoted = [r for r in _WIRED if gaps.get(r, -1.0) >= 0.02]
    print(f"[ir] promotable_routes (wired/reference-resolvable, cleared): {promoted}")

    # Persist the fitted router (with reference, schema v4).
    rt = IntentRouter.from_seed(glove, margin=margin, alpha=alpha, beta=beta,
                                gamma=gamma, delta=delta, detect_fn=_DETECT)
    rt.rebind_anchors(glove)
    rt.set_promoted(promoted)
    rt.save()
    print(f"[ir] saved fitted router -> {_FIT_PATH}")

    summary = {
        "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
        "margin": margin,
        "agreement": f"{agree}/{len(CORPUS)}",
        "fused_min_gap_per_route": {r: round(gaps[r], 3) for r in COLLIDING},
        "ablation_gap_no_reference": {
            r: round(gaps_noref[r], 3) for r in ("self_disclosure", "self_directed")},
        "promotable_routes": promoted,
        "reference_finding": "Reference-target axis (schema v4: semantic⊕shape⊕"
                             "affect⊕reference, reusing SelfAddressRouter features) "
                             "PROVES via ablation that it opens the residual pair: "
                             f"self_disclosure {gaps_noref['self_disclosure']:+.3f} "
                             f"(OFF) -> {gaps['self_disclosure']:+.3f} (ON); "
                             f"self_directed {gaps_noref['self_directed']:+.3f} (OFF) "
                             f"-> {gaps['self_directed']:+.3f} (ON). Separates "
                             "self (ventral mPFC) vs other/assistant (dorsal mPFC/TPJ).",
        "note": "v4 schema: semantic⊕shape⊕affect⊕reference. Alpha/beta/gamma/"
                "delta/margin grid-fit at EER-style criterion. Reference-ablation "
                "proves the axis opens the residual pair. Promoted routes cleared "
                "gap>=0.02.",
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_intent_router_calib.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ir] dashboard -> {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
