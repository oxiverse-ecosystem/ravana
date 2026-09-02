"""
Parallel-run harness: regex rule cascade vs semantic prototype classifier
=========================================================================
Evidence-first step for the unified semantic layer (Phase A1).

Goal: before deleting ANY regex, measure whether the prototype-based
SpeechActClassifier actually beats the regex rule cascade
(PrefrontalWorkspace.classify_speech_act_rules), especially on paraphrases
the brittle rules are known to miss.

We run BOTH classifiers over a labeled utterance set and report:
  - overall accuracy of each
  - agreement rate between them
  - confusion: where regex is right & proto wrong, and vice-versa
  - the specific utterances where the prototype model rescues a regex miss
  - method breakdown for the prototype path (semantic / hybrid / syntactic)

Ground truth: RAVANA has no chat logs on disk, so the eval set is built from
the existing unit-test fixtures (the behaviour the code currently guarantees)
PLUS a curated paraphrase set — declaratives without first-person markers and
questions without '?'/wh-/inversion, i.e. exactly the cases the rule cascade
defaults (it defaults unknown inputs to 'question').

Run:
    python experiments/regex_vs_prototype_speechact.py

Notes:
  - Uses the SAME GloVe-64 pipeline as production (build_glove64_lookup on
    data/ravana_glove_cache.npz), so vector_fn is faithful.
  - No code under ravana/ is modified. This is pure measurement.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

# --- make ravana importable (src/ layout) ---
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_SRC = os.path.join(_ROOT, "ravana", "src")
for p in (_SRC,):
    if p not in sys.path:
        sys.path.insert(0, p)

from ravana.language.prefrontal_workspace import (  # noqa: E402
    PrefrontalWorkspace,
    SpeechActClassifier,
)
from ravana.ontology.attribute_encoder import build_glove64_lookup  # noqa: E402


# ---------------------------------------------------------------------------
# Labeled eval set. label ∈ {"question", "statement"}.
# "tier" tags whether the rule cascade has an explicit cue ("easy") or must
# fall through its default ("hard" — the paraphrase cases that motivate a
# learned layer).
# ---------------------------------------------------------------------------
EVAL = [
    # --- EASY questions: explicit '?', wh-, or aux-inversion (regex should nail) ---
    ("What is trust?", "question", "easy"),
    ("Why does ice melt?", "question", "easy"),
    ("How does gravity work?", "question", "easy"),
    ("Are you listening?", "question", "easy"),
    ("Can you help me", "question", "easy"),
    ("Do they know about it?", "question", "easy"),
    ("Tell me about freedom", "question", "easy"),
    ("Explain photosynthesis", "question", "easy"),
    ("Who invented the telephone?", "question", "easy"),
    ("Where is the nearest station?", "question", "easy"),

    # --- EASY statements: first/third-person declarative markers ---
    ("I am tired today", "statement", "easy"),
    ("My name is Pixel", "statement", "easy"),
    ("We are going to the park", "statement", "easy"),
    ("She is a doctor", "statement", "easy"),
    ("They went home early", "statement", "easy"),
    ("Thanks for the help", "statement", "easy"),
    ("Nothing much", "statement", "easy"),
    ("Nice to meet you", "statement", "easy"),

    # --- HARD questions: no '?', no wh-, no inversion → regex DEFAULTS to
    #     'question' (so it "gets these right" by luck of the default, not by
    #     understanding). Kept to test proto doesn't regress them. ---
    ("wondering about black holes lately", "question", "hard"),
    ("curious what makes volcanoes erupt", "question", "hard"),

    # --- HARD statements: declaratives the rule cascade MISSES because they
    #     lack a leading first/third-person marker → default to 'question'.
    #     These are the rescue cases a semantic layer should catch. ---
    ("the sky looks beautiful tonight", "statement", "hard"),
    ("really enjoyed that movie yesterday", "statement", "hard"),
    ("kind of hungry right now", "statement", "hard"),
    ("pretty sure that's correct", "statement", "hard"),
    ("just finished reading a great book", "statement", "hard"),
    ("feeling good about the project", "statement", "hard"),
    ("looks like rain is coming", "statement", "hard"),
    ("absolutely love this song", "statement", "hard"),
    ("not a fan of cold weather", "statement", "hard"),
    ("that was an amazing performance", "statement", "hard"),
]


# A broader statement seed — bare declaratives WITHOUT a leading first/third
# person marker. These are exactly the utterances the current 2-class seed
# (which only seeds statements from "i/my/we/she/they..." + social phrases)
# cannot represent, and therefore collapses to "question". Hand-curating this
# is itself a curation cost — noted in the report.
WIDER_STATEMENT_EXEMPLARS = [
    "the sky is blue", "dogs are friendly", "that movie was great",
    "coffee tastes bitter", "the book is interesting", "rain is falling",
    "music sounds nice", "the food was delicious", "the room is quiet",
    "the test went well", "the light is bright", "this soup is tasty",
    "the game was fun", "the plan worked", "the idea is good",
    "the weather turned cold", "the child smiled", "the engine stopped",
]


def _pure_semantic_classifier(vector_fn, dim, widen_statements: bool):
    """Build a SpeechActClassifier that does NOT defer to the regex hint —
    pure nearest-prototype decision. Optionally widen the statement seed."""
    cls = SpeechActClassifier(vector_fn=vector_fn, dim=dim)
    if widen_statements:
        # patch the prototype table before first build
        cls.PROTOTYPES = dict(cls.PROTOTYPES)
        cls.PROTOTYPES["statement"] = list(cls.PROTOTYPES["statement"]) + WIDER_STATEMENT_EXEMPLARS
        cls._proto_vecs = None
    return cls


def _run_pure(EVAL, cls):
    """Pure-semantic (no hint) pass. Returns (correct, agree_with_regex, rescues)."""
    correct = 0
    agree = 0
    rescues = []
    for text, gold, tier in EVAL:
        regex_pred = PrefrontalWorkspace.classify_speech_act_rules(text)
        proto_pred, scores, method = cls.classify(text)  # no syntactic_hint
        correct += (proto_pred == gold)
        agree += (proto_pred == regex_pred)
        if proto_pred != regex_pred and proto_pred == gold:
            rescues.append((text, gold, regex_pred, proto_pred, method, scores))
    return correct, agree, rescues


class AdaptiveProtoClassifier:
    """Prototype classifier with NO hardcoded margin.

    The decision boundary is derived per-class from the exemplar-to-centroid
    similarity distribution (mean mu_c, std sigma_c). An utterance's membership
    in class c is the z-score  (cos(u, proto_c) - mu_c) / sigma_c  — i.e. how
    typical the utterance is *relative to how tight that class's own exemplars
    are*. We pick argmax z. This removes the fixed SEMANTIC_MARGIN=0.12 entirely:
    the threshold is a learned property of the exemplar set, so adding chat-fed
    exemplars automatically re-shapes it (the "learn by chatting" requirement).

    Warm-started from the same seed exemplars; exemplars can be appended at
    runtime via add_exemplar(cls, text) to simulate chat accumulation.
    """

    def __init__(self, vector_fn, dim, seed_prototypes):
        self.vector_fn = vector_fn
        self.dim = dim
        self.exemplars = {c: list(v) for c, v in seed_prototypes.items()}
        self._centroid = {}
        self._mu = {}
        self._sigma = {}
        self._dirty = True

    def _sentence_vector(self, text):
        import re as _re
        if self.vector_fn is None:
            return None
        words = _re.findall(r"[a-z']+", text.lower())
        vecs = [self.vector_fn(w) for w in words]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        arr = np.stack(vecs).mean(axis=0)
        n = np.linalg.norm(arr)
        return arr / n if n > 0 else arr

    def add_exemplar(self, cls, text):
        self.exemplars.setdefault(cls, []).append(text)
        self._dirty = True

    def _fit(self):
        if not self._dirty:
            return
        for c, phrases in self.exemplars.items():
            vs = [self._sentence_vector(p) for p in phrases]
            vs = [v for v in vs if v is not None]
            if not vs:
                continue
            M = np.stack(vs)
            centroid = M.mean(axis=0)
            n = np.linalg.norm(centroid)
            if n > 0:
                centroid = centroid / n
            sims = M @ centroid  # each exemplar's cos to its own centroid
            self._centroid[c] = centroid
            self._mu[c] = float(sims.mean())
            # guard against a degenerate (single-exemplar) class
            self._sigma[c] = float(sims.std()) or 1e-3
        self._dirty = False

    def classify(self, text):
        self._fit()
        sv = self._sentence_vector(text)
        if sv is None or not self._centroid:
            return ("question", {}, "default")
        z = {}
        raw = {}
        for c, cen in self._centroid.items():
            sim = float(np.dot(sv, cen))
            raw[c] = sim
            z[c] = (sim - self._mu[c]) / self._sigma[c]
        best = max(z, key=z.get)
        return (best, {k: round(v, 3) for k, v in raw.items()}, "adaptive-z")


def _run_adaptive(EVAL, clf):
    correct = 0
    agree = 0
    rescues = []
    regress = []
    for text, gold, tier in EVAL:
        regex_pred = PrefrontalWorkspace.classify_speech_act_rules(text)
        pred, raw, method = clf.classify(text)
        correct += (pred == gold)
        agree += (pred == regex_pred)
        if pred != regex_pred and pred == gold:
            rescues.append((text, gold, regex_pred, pred, raw))
        if pred != gold and regex_pred == gold:
            regress.append((text, gold, regex_pred, pred, raw))
    return correct, agree, rescues, regress


class HRRUtteranceEncoder:
    """A0 encoder: utterance vector that carries SYNTAX, not just bag-of-words.

    The token-centroid encoder used everywhere else throws away word order, which
    caused the regressions ("can you help me" looks like a statement) and the
    chat-sim failure. Here we bind role-fillers with the real HRR primitives from
    core/vsa.py so position/register survive:

        u = bundle[ bind(CONTENT,  centroid(tokens)),
                    bind(FIRST,    vec(first_token)),   # word-order / fronting cue
                    bind(REGISTER, question_mark ? Q : STMT) ]

    Crucially the FIRST-token cue is SEMANTIC (the first word's own GloVe vector),
    not a curated aux/wh list — "can/do/are" cluster apart from "the/i/really" in
    embedding space, so aux-inversion & wh-fronting are captured without any
    regex or word list. Register is the ONE structural bit we keep (a trailing
    '?'), bound as its own role so it can't swamp the semantics.

    Uses VSAManager at dim=64 to match the GloVe space (HRR needs equal dims).
    """

    def __init__(self, vector_fn, dim):
        from ravana.core.vsa import VSAManager
        self.vector_fn = vector_fn
        self.dim = dim
        self.vsa = VSAManager(dim=dim)
        # dedicated roles + register filler atoms (random fixed unit vectors)
        for r in ("content", "first", "register"):
            self.vsa.get_role(r)
        self._Q = self.vsa.generate_vector()      # "is a question (has ?)"
        self._S = self.vsa.generate_vector()      # "no question mark"

    def _tokens(self, text):
        import re as _re
        return _re.findall(r"[a-z']+", text.lower())

    def _centroid(self, toks):
        vecs = [self.vector_fn(w) for w in toks]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        arr = np.stack(vecs).mean(axis=0)
        n = np.linalg.norm(arr)
        return arr / n if n > 0 else arr

    def encode(self, text):
        toks = self._tokens(text)
        centroid = self._centroid(toks)
        if centroid is None:
            return None
        comps = [self.vsa.bind_role_filler("content", centroid)]
        # first-token semantic cue
        fv = self.vector_fn(toks[0]) if toks else None
        if fv is not None:
            comps.append(self.vsa.bind_role_filler("first", fv))
        # register: trailing '?' (the one structural bit)
        reg = self._Q if text.strip().endswith("?") else self._S
        comps.append(self.vsa.bind_role_filler("register", reg))
        u = self.vsa.bundle(comps)
        return u


class AdaptiveHRRClassifier(AdaptiveProtoClassifier):
    """AdaptiveProtoClassifier but utterances/exemplars are HRR-encoded
    (syntax-aware) instead of plain token centroids. Same z-score boundary."""

    def __init__(self, vector_fn, dim, seed_prototypes):
        super().__init__(vector_fn, dim, seed_prototypes)
        self._enc = HRRUtteranceEncoder(vector_fn, dim)

    def _sentence_vector(self, text):
        return self._enc.encode(text)


def make_vector_fn():
    """Load GloVe-64 lookup and return a word->unit-vector callable + dim."""
    cache = os.path.join(_ROOT, "data", "ravana_glove_cache.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"GloVe cache not found: {cache}")
    lut, dim = build_glove64_lookup(cache)

    def vector_fn(word: str):
        v = lut.get(word.lower())
        if v is None:
            return None
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v

    return vector_fn, dim, len(lut)


def main():
    vector_fn, dim, vocab = make_vector_fn()
    print(f"GloVe-64 loaded: dim={dim}, vocab={vocab:,}\n")

    sac = SpeechActClassifier(vector_fn=vector_fn, dim=dim)
    # sanity: how many prototype phrases actually embed?
    sac._build_prototypes()
    print(f"Prototype classes built: {list(sac._proto_vecs.keys())}")
    print(f"SEMANTIC_MARGIN (fixed threshold): {sac.SEMANTIC_MARGIN}\n")

    n = len(EVAL)
    regex_correct = 0
    proto_correct = 0
    agree = 0
    method_counts = Counter()
    proto_rescues = []   # regex wrong, proto right
    proto_regress = []   # regex right, proto wrong
    per_tier = {"easy": [0, 0, 0], "hard": [0, 0, 0]}  # [count, regex_ok, proto_ok]

    rows = []
    for text, gold, tier in EVAL:
        regex_pred = PrefrontalWorkspace.classify_speech_act_rules(text)
        hint = regex_pred
        proto_pred, scores, method = sac.classify(text, syntactic_hint=hint)

        r_ok = regex_pred == gold
        p_ok = proto_pred == gold
        regex_correct += r_ok
        proto_correct += p_ok
        agree += (regex_pred == proto_pred)
        method_counts[method] += 1

        per_tier[tier][0] += 1
        per_tier[tier][1] += r_ok
        per_tier[tier][2] += p_ok

        if not r_ok and p_ok:
            proto_rescues.append((text, gold, regex_pred, proto_pred, method, scores))
        if r_ok and not p_ok:
            proto_regress.append((text, gold, regex_pred, proto_pred, method, scores))

        rows.append((text, gold, regex_pred, proto_pred, method, r_ok, p_ok))

    # -------- report --------
    print("=" * 78)
    print("PER-UTTERANCE")
    print("=" * 78)
    print(f"{'gold':<10}{'regex':<10}{'proto':<10}{'method':<10}  utterance")
    print("-" * 78)
    for text, gold, rp, pp, method, r_ok, p_ok in rows:
        flag = ""
        if not r_ok and p_ok:
            flag = "  <== PROTO RESCUE"
        elif r_ok and not p_ok:
            flag = "  <== PROTO REGRESS"
        print(f"{gold:<10}{rp:<10}{pp:<10}{method:<10}  {text}{flag}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"N utterances ............ {n}")
    print(f"Regex accuracy .......... {regex_correct}/{n} = {regex_correct/n:.1%}")
    print(f"Prototype accuracy ...... {proto_correct}/{n} = {proto_correct/n:.1%}")
    print(f"Agreement (regex==proto)  {agree}/{n} = {agree/n:.1%}")
    print(f"Proto rescues (R✗ P✓) ... {len(proto_rescues)}")
    print(f"Proto regressions (R✓ P✗) {len(proto_regress)}")
    print(f"Prototype method mix .... {dict(method_counts)}")

    print("\nPER-TIER (count / regex_ok / proto_ok):")
    for tier, (c, r, p) in per_tier.items():
        if c:
            print(f"  {tier:<6} n={c:<3} regex={r}/{c}={r/c:.0%}  proto={p}/{c}={p/c:.0%}")

    if proto_rescues:
        print("\n" + "-" * 78)
        print("RESCUES — regex wrong, prototype right (the case FOR a learned layer):")
        for text, gold, rp, pp, method, scores in proto_rescues:
            sc = {k: round(v, 3) for k, v in (scores or {}).items()}
            print(f"  '{text}'  gold={gold} regex={rp} proto={pp} [{method}] {sc}")

    if proto_regress:
        print("\n" + "-" * 78)
        print("REGRESSIONS — regex right, prototype wrong (the cost / risk):")
        for text, gold, rp, pp, method, scores in proto_regress:
            sc = {k: round(v, 3) for k, v in (scores or {}).items()}
            print(f"  '{text}'  gold={gold} regex={rp} proto={pp} [{method}] {sc}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    delta = proto_correct - regex_correct
    if delta > 0:
        print(f"Prototype BEATS regex by {delta} utterances (+{delta/n:.1%}).")
    elif delta < 0:
        print(f"Prototype LOSES to regex by {-delta} utterances ({delta/n:.1%}).")
    else:
        print("Prototype and regex TIE on overall accuracy.")
    hard = per_tier["hard"]
    if hard[0]:
        print(f"On HARD paraphrases: regex {hard[1]}/{hard[0]}, proto {hard[2]}/{hard[0]} "
              f"(this tier is what a learned layer must win).")

    # -----------------------------------------------------------------------
    # CEILING PROBE: does the prototype model beat regex IF it is allowed to
    # override the regex hint (pure semantic) and IF the statement seed is
    # widened to cover bare declaratives? This answers whether Phase A1 as
    # currently scoped can ever surpass the regex — or whether it simply
    # re-confirms it.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CEILING PROBE — pure semantic (no regex hint), with widened statement seed")
    print("=" * 78)
    widened = _pure_semantic_classifier(vector_fn, dim, widen_statements=True)
    wcorrect, wagree, wrescues = _run_pure(EVAL, widened)
    print(f"Pure-semantic accuracy (widened seed) .. {wcorrect}/{n} = {wcorrect/n:.1%}")
    print(f"Agreement with regex .................. {wagree}/{n} = {wagree/n:.1%}")
    print(f"Rescues (regex wrong, pure-semantic right) ... {len(wrescues)}")
    for text, gold, rp, pp, method, scores in wrescues:
        sc = {k: round(v, 3) for k, v in (scores or {}).items()}
        print(f"  '{text}'  gold={gold} regex={rp} pure={pp} [{method}] {sc}")

    narrow = _pure_semantic_classifier(vector_fn, dim, widen_statements=False)
    ncorrect, nagree, _ = _run_pure(EVAL, narrow)
    print(f"\nPure-semantic accuracy (ORIGINAL seed) .. {ncorrect}/{n} = {ncorrect/n:.1%} "
          f"(agrees with regex {nagree}/{n})")

    # -----------------------------------------------------------------------
    # ADAPTIVE-MARGIN PROBE: no fixed 0.12. Boundary = per-class z-score from
    # exemplar spread. Tests whether removing the hardcoded threshold alone
    # (original seed, no hand-widening) recovers rescues.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ADAPTIVE-MARGIN PROBE — z-score boundary from exemplar spread, ORIGINAL seed")
    print("=" * 78)
    seed_protos = dict(SpeechActClassifier.PROTOTYPES)
    adaptive = AdaptiveProtoClassifier(vector_fn, dim, seed_protos)
    acorrect, aagree, arescues, aregress = _run_adaptive(EVAL, adaptive)
    print(f"Adaptive accuracy (ORIGINAL seed) ..... {acorrect}/{n} = {acorrect/n:.1%}")
    print(f"Agreement with regex .................. {aagree}/{n} = {aagree/n:.1%}")
    print(f"Rescues (regex wrong, adaptive right) . {len(arescues)}")
    print(f"Regressions (regex right, adaptive wrong) {len(aregress)}")
    for text, gold, rp, pp, raw in arescues:
        print(f"  RESCUE '{text}' gold={gold} regex={rp} adaptive={pp} {raw}")
    for text, gold, rp, pp, raw in aregress:
        print(f"  REGRESS '{text}' gold={gold} regex={rp} adaptive={pp} {raw}")

    # -----------------------------------------------------------------------
    # LEARN-BY-CHATTING SIMULATION: instead of hand-curating a wider seed, we
    # feed HALF the hard statements to the classifier as "chat" exemplars and
    # test on the HELD-OUT half. If held-out accuracy rises, chat accumulation
    # (not curation) is what closes the gap — the actual design claim.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("LEARN-BY-CHATTING SIM — feed half the hard statements as chat, test held-out half")
    print("=" * 78)
    hard_statements = [(t, g, tier) for (t, g, tier) in EVAL
                       if tier == "hard" and g == "statement"]
    train = hard_statements[::2]   # even-indexed → "seen in chat"
    heldout = hard_statements[1::2]  # odd-indexed → never seen
    print(f"hard statements: {len(hard_statements)}  train(chat)={len(train)}  heldout={len(heldout)}")

    # baseline: adaptive on original seed, measured on held-out only
    base = AdaptiveProtoClassifier(vector_fn, dim, dict(SpeechActClassifier.PROTOTYPES))
    base_ok = sum(base.classify(t)[0] == g for (t, g, _) in heldout)
    print(f"Held-out accuracy BEFORE chat .......... {base_ok}/{len(heldout)}")

    # feed the training half as chat exemplars, re-test held-out
    learner = AdaptiveProtoClassifier(vector_fn, dim, dict(SpeechActClassifier.PROTOTYPES))
    for (t, g, _) in train:
        learner.add_exemplar(g, t)
    learn_ok = sum(learner.classify(t)[0] == g for (t, g, _) in heldout)
    print(f"Held-out accuracy AFTER chat ........... {learn_ok}/{len(heldout)}")
    for (t, g, _) in heldout:
        pred, raw, _ = learner.classify(t)
        flag = "OK" if pred == g else "MISS"
        print(f"  [{flag}] '{t}' gold={g} pred={pred} {raw}")
    if learn_ok > base_ok:
        print("\n=> Chat-fed exemplars generalize to UNSEEN utterances without "
              "hand-curation or a fixed threshold. This is the real 'learn by chatting' signal.")
    else:
        print("\n=> No held-out gain from chat exemplars at this scale — needs more "
              "data or richer utterance encoding (register/prosody roles).")

    # -----------------------------------------------------------------------
    # A0 PROBE: HRR utterance encoding (content + first-token + register roles)
    # via the real core/vsa.py primitives. Does syntax-aware encoding fix the 3
    # token-centroid regressions, and does it make chat exemplars generalize?
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("A0 PROBE — HRR utterance encoding (content+first+register), adaptive z-margin")
    print("=" * 78)
    hrr = AdaptiveHRRClassifier(vector_fn, dim, dict(SpeechActClassifier.PROTOTYPES))
    hcorrect, hagree, hrescues, hregress = _run_adaptive(EVAL, hrr)
    print(f"HRR-adaptive accuracy (ORIGINAL seed) . {hcorrect}/{n} = {hcorrect/n:.1%}")
    print(f"Agreement with regex .................. {hagree}/{n} = {hagree/n:.1%}")
    print(f"Rescues (regex wrong, HRR right) ...... {len(hrescues)}")
    print(f"Regressions (regex right, HRR wrong) .. {len(hregress)}")
    for text, gold, rp, pp, raw in hregress:
        print(f"  REGRESS '{text}' gold={gold} regex={rp} hrr={pp} {raw}")
    prev_regress = {"Can you help me", "Do they know about it?", "My name is Pixel"}
    now_regress = {t for (t, *_rest) in hregress}
    fixed = prev_regress - now_regress
    if fixed:
        print(f"  FIXED by HRR encoding (were token-centroid regressions): {sorted(fixed)}")

    # chat-sim on HRR encoding
    print("\nLEARN-BY-CHATTING SIM on HRR encoding:")
    base_h = AdaptiveHRRClassifier(vector_fn, dim, dict(SpeechActClassifier.PROTOTYPES))
    base_h_ok = sum(base_h.classify(t)[0] == g for (t, g, _) in heldout)
    learn_h = AdaptiveHRRClassifier(vector_fn, dim, dict(SpeechActClassifier.PROTOTYPES))
    for (t, g, _) in train:
        learn_h.add_exemplar(g, t)
    learn_h_ok = sum(learn_h.classify(t)[0] == g for (t, g, _) in heldout)
    print(f"  Held-out BEFORE chat: {base_h_ok}/{len(heldout)}   AFTER chat: {learn_h_ok}/{len(heldout)}")

    print("\n" + "=" * 78)
    print("FINAL SCOREBOARD (accuracy on 30 utterances)")
    print("=" * 78)
    print(f"  regex rule cascade .................. {regex_correct}/{n} = {regex_correct/n:.0%}")
    print(f"  hybrid (shipped, fixed 0.12) ....... {proto_correct}/{n} = {proto_correct/n:.0%}")
    print(f"  adaptive z-margin, token centroid .. {acorrect}/{n} = {acorrect/n:.0%}  (+{acorrect-regex_correct} vs regex)")
    print(f"  adaptive z-margin, HRR encoding .... {hcorrect}/{n} = {hcorrect/n:.0%}  (+{hcorrect-regex_correct} vs regex)")
    print("  (no hardcoded threshold in the last two; HRR adds syntax via bound roles)")


if __name__ == "__main__":
    main()
