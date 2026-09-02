"""
N1 harness — hierarchical (2-stage) question-type classification
================================================================
Stage 1: binary speech-act (question vs statement) — ALREADY SHIPPED as the
         adaptive SpeechActClassifier. Not re-tested here.
Stage 2: wh-subtype prototype bank, invoked ONLY inside the question branch.

Rationale (Rosch 1976 basic-level-first; Lambon Ralph 2017 hub-and-spoke
coarse->fine; Friston & Kiebel 2009 hierarchical inference): don't pack 12
pragmatic classes into non-orthogonal GloVe-64 as a flat bank — near-synonyms
(who/what, why/how) collapse (the same 64D crowding that killed HRR@64D).
Split coarse (already at 87%) from fine (subtype).

This harness measures, BEFORE porting to production:
  - Stage-2 accuracy vs the regex cascade on labeled questions.
  - Metric A/B: plain cosine argmax  vs  exemplar-spread z-score
    (Mahalanobis-lean; Snell et al. 2017 show Euclidean/Mahalanobis beats cosine
    for class-mean prototypes).
  - Abstain band: fraction rejected at low z (the seam N4 surprise-gating needs).

Subtypes modeled (the wh/semantic question intents from QUESTION_PATTERNS):
  what_is, why, how, tell_me, compare, hypothetical, do_you_know
Social/structural types (greeting, introduction, analogy, impossible, ...) are
NOT in the prototype bank — they stay regex (they're genuinely pattern-shaped:
"my name is X", "hi", "A:B::C:D"). N1 replaces the SEMANTIC subtype cascade only.

Run:  python experiments/n1_question_subtype.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_SRC = os.path.join(_ROOT, "ravana", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ravana.language.prefrontal_workspace import PrefrontalWorkspace  # noqa: E402
from ravana.ontology.attribute_encoder import build_glove64_lookup  # noqa: E402


# Seed exemplars per wh-subtype (few-shot prototypes; extend to "learn").
SUBTYPE_SEEDS = {
    "what_is": [
        "what is trust", "what are black holes", "what is gravity",
        "what is a neuron", "what are dreams", "what is democracy",
        "what is photosynthesis", "what is inflation",
    ],
    "why": [
        "why does ice melt", "why is the sky blue", "why do birds sing",
        "why are we here", "why does it rain", "why do people lie",
        "why is water wet", "why do stars shine",
    ],
    "how": [
        "how does gravity work", "how do planes fly", "how does a battery work",
        "how do vaccines work", "how does memory work", "how do engines run",
        "how does the heart pump", "how do computers think",
    ],
    "tell_me": [
        "tell me about freedom", "tell me about the ocean",
        "tell me more about jazz", "tell me about ancient rome",
        "tell me about quantum physics", "tell me about your day",
    ],
    "compare": [
        "compare cats and dogs", "difference between love and lust",
        "what is the difference between weather and climate",
        "compare python and java", "socialism versus capitalism",
        "difference between a virus and bacteria",
    ],
    "hypothetical": [
        "what if the sun disappeared", "what would happen if we stopped sleeping",
        "what happens if you fall into a black hole", "suppose gravity reversed",
        "imagine a world without money", "what if humans could fly",
    ],
    "do_you_know": [
        "do you know about einstein", "have you heard of the beatles",
        "do you know what dna is", "do you know who wrote hamlet",
        "have you heard about the big bang", "do you know any good books",
    ],
}

# Labeled eval questions (gold subtype). Deliberately mixes phrasings the regex
# cascade nails with paraphrases that stress near-synonym crowding.
EVAL = [
    # what_is
    ("what is trust", "what_is"),
    ("what are volcanoes", "what_is"),
    ("what's a black hole", "what_is"),
    ("define entropy", "what_is"),          # paraphrase — regex has no 'define' in what_is
    # why
    ("why does ice melt", "why"),
    ("why is the ocean salty", "why"),
    ("why do we dream", "why"),
    # how
    ("how does gravity work", "how"),
    ("how do birds fly", "how"),
    ("how does a vaccine protect us", "how"),
    # tell_me
    ("tell me about freedom", "tell_me"),
    ("tell me more about the roman empire", "tell_me"),
    # compare
    ("compare dogs and cats", "compare"),
    ("what is the difference between fog and mist", "compare"),
    ("socialism versus capitalism", "compare"),
    # hypothetical
    ("what if the moon vanished", "hypothetical"),
    ("what happens if you never sleep", "hypothetical"),
    ("suppose the earth stopped spinning", "hypothetical"),
    # do_you_know
    ("do you know about newton", "do_you_know"),
    ("have you heard of jazz", "do_you_know"),
]


def make_vector_fn():
    cache = os.path.join(_ROOT, "data", "ravana_glove_cache.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"GloVe cache not found: {cache}")
    lut, dim = build_glove64_lookup(cache)

    def vector_fn(word):
        v = lut.get(word.lower())
        if v is None:
            return None
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v

    return vector_fn, dim


class SubtypeBank:
    """Stage-2 wh-subtype prototype bank. Two metrics selectable for A/B."""

    def __init__(self, vector_fn, seeds, first_weight=1.0):
        self.vector_fn = vector_fn
        self.seeds = {c: list(v) for c, v in seeds.items()}
        self.first_weight = first_weight
        self._cen = {}
        self._mu = {}
        self._sigma = {}
        self._fit()

    def _sv(self, text, first_weight=1.0):
        import re as _re
        toks = _re.findall(r"[a-z']+", text.lower())
        vs = [self.vector_fn(w) for w in toks]
        vs = [v for v in vs if v is not None]
        if not vs:
            return None
        M = np.stack(vs)
        weights = np.ones(len(M))
        if first_weight != 1.0 and len(M) > 0:
            weights[0] = first_weight  # emphasise the leading function word
        a = (M * weights[:, None]).sum(axis=0) / weights.sum()
        n = np.linalg.norm(a)
        return a / n if n > 0 else a

    def _fit(self):
        for c, phrases in self.seeds.items():
            vs = [self._sv(p, self.first_weight) for p in phrases]
            vs = [v for v in vs if v is not None]
            if not vs:
                continue
            M = np.stack(vs)
            cen = M.mean(axis=0)
            n = np.linalg.norm(cen)
            if n > 0:
                cen = cen / n
            sims = M @ cen
            self._cen[c] = cen
            self._mu[c] = float(sims.mean())
            self._sigma[c] = float(sims.std()) or 1e-3

    def classify(self, text, metric="zscore", abstain_z=None, abstain_k=None):
        sv = self._sv(text, self.first_weight)
        if sv is None:
            return None, {}
        cos = {c: float(np.dot(sv, cen)) for c, cen in self._cen.items()}
        if metric == "cosine":
            score = cos
        elif metric == "zscore":
            score = {c: (cos[c] - self._mu[c]) / self._sigma[c] for c in cos}
        else:
            raise ValueError(metric)
        best = max(score, key=score.get)
        # z-score abstain (legacy sweep)
        if abstain_z is not None and metric == "zscore" and score[best] < abstain_z:
            return "ABSTAIN", score
        # HYBRID abstain gate: cosine decides, exemplar-spread (mu - k*sigma)
        # floors confidence. This is the N4 surprise seam that works WITH cosine.
        if abstain_k is not None:
            floor = self._mu[best] - abstain_k * self._sigma[best]
            if cos[best] < floor:
                return "ABSTAIN", score
        return best, score


def regex_subtype(text):
    """Ground the comparison in the CURRENT production regex cascade."""
    qtype, _ = PrefrontalWorkspace._detect_question_type_regex(text)
    return qtype


def run(EVAL, bank, metric, abstain_z=None):
    correct = 0
    abstained = 0
    conf = defaultdict(Counter)
    for text, gold in EVAL:
        pred, _ = bank.classify(text, metric=metric, abstain_z=abstain_z)
        if pred == "ABSTAIN":
            abstained += 1
            continue
        conf[gold][pred] += 1
        correct += (pred == gold)
    n = len(EVAL)
    return correct, abstained, conf, n


def main():
    vector_fn, dim = make_vector_fn()
    print(f"GloVe-64 loaded (dim={dim}). Subtypes: {list(SUBTYPE_SEEDS.keys())}\n")
    bank = SubtypeBank(vector_fn, SUBTYPE_SEEDS)

    n = len(EVAL)

    # regex baseline (only counts hits that land in our 7 modeled subtypes)
    rgx_correct = 0
    rgx_rows = []
    for text, gold in EVAL:
        rp = regex_subtype(text)
        rgx_correct += (rp == gold)
        rgx_rows.append((text, gold, rp))
    print("=" * 78)
    print("REGEX CASCADE (current production) on the 20 questions")
    print("=" * 78)
    for text, gold, rp in rgx_rows:
        flag = "" if rp == gold else "  <-- MISS"
        print(f"  gold={gold:<12} regex={rp:<12} {text}{flag}")
    print(f"\nRegex subtype accuracy: {rgx_correct}/{n} = {rgx_correct/n:.1%}")

    # A/B metric
    print("\n" + "=" * 78)
    print("STAGE-2 PROTOTYPE BANK — metric A/B (cosine vs exemplar-spread z-score)")
    print("=" * 78)
    for metric in ("cosine", "zscore"):
        c, ab, conf, _ = run(EVAL, bank, metric)
        print(f"\n[{metric}] accuracy: {c}/{n} = {c/n:.1%}")
        for gold in SUBTYPE_SEEDS:
            preds = conf.get(gold)
            if preds:
                print(f"    {gold:<12} -> {dict(preds)}")

    # abstain sweep (z-score only)
    print("\n" + "=" * 78)
    print("ABSTAIN BAND SWEEP (z-score) — the seam N4 surprise-gating consumes")
    print("=" * 78)
    for az in (None, -1.0, -0.5, 0.0, 0.5):
        c, ab, _, _ = run(EVAL, bank, "zscore", abstain_z=az)
        answered = n - ab
        acc_ans = (c / answered) if answered else 0.0
        print(f"  abstain_z={str(az):<6} answered={answered:<3} abstained={ab:<3} "
              f"acc_on_answered={acc_ans:.1%} coverage={answered/n:.0%}")

    print("\n" + "=" * 78)
    print("FIRST-TOKEN WEIGHT SWEEP — does emphasising the leading wh-word help?")
    print("(hypothesis: subtypes differ by function word, which mean-pooling averages away)")
    print("=" * 78)
    for fw in (1.0, 2.0, 3.0, 5.0, 8.0):
        b = SubtypeBank(vector_fn, SUBTYPE_SEEDS, first_weight=fw)
        cc, _, _, _ = run(EVAL, b, "cosine")
        cz, _, _, _ = run(EVAL, b, "zscore")
        print(f"  first_weight={fw:<4} cosine={cc}/{n}={cc/n:.0%}  zscore={cz}/{n}={cz/n:.0%}")

    print("\n" + "=" * 78)
    print("WINNING CONFIG DIFF — first_weight=3.0, cosine (per-utterance vs regex)")
    print("=" * 78)
    best_bank = SubtypeBank(vector_fn, SUBTYPE_SEEDS, first_weight=3.0)
    rescues = 0
    regress = 0
    for text, gold in EVAL:
        rp = regex_subtype(text)
        pp, _ = best_bank.classify(text, metric="cosine")
        tag = ""
        if rp != gold and pp == gold:
            tag = "  <== RESCUE (regex missed)"; rescues += 1
        elif rp == gold and pp != gold:
            tag = "  <== REGRESS"; regress += 1
        print(f"  gold={gold:<12} regex={rp:<12} proto={pp:<12} {text}{tag}")
    print(f"\n  rescues={rescues}  regressions={regress}")

    print("\n" + "=" * 78)
    print("HYBRID ABSTAIN GATE — cosine decides, (mu - k*sigma) floors confidence")
    print("(the N4 surprise seam that works WITH the winning cosine metric)")
    print("=" * 78)
    for k in (None, 3.0, 2.0, 1.0):
        c = ab = 0
        for text, gold in EVAL:
            pred, _ = best_bank.classify(text, metric="cosine", abstain_k=k)
            if pred == "ABSTAIN":
                ab += 1
            else:
                c += (pred == gold)
        answered = n - ab
        acc = (c / answered) if answered else 0.0
        print(f"  abstain_k={str(k):<5} answered={answered:<3} abstained={ab:<3} "
              f"acc_on_answered={acc:.0%} coverage={answered/n:.0%}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    cz, _, _, _ = run(EVAL, bank, "zscore")
    cc, _, _, _ = run(EVAL, bank, "cosine")
    best_metric = "zscore" if cz >= cc else "cosine"
    best = max(cz, cc)
    print(f"  regex: {rgx_correct/n:.0%} | cosine: {cc/n:.0%} | zscore: {cz/n:.0%}")
    print(f"  best metric: {best_metric} ({best}/{n} = {best/n:.0%}), delta vs regex "
          f"{(best-rgx_correct):+d}")
    print("  (abstain band lets low-z cases defer to regex fallback OR trigger a "
          "clarifying question in N4)")


if __name__ == "__main__":
    main()
