#!/usr/bin/env python3
"""RAVANA-LEARN-5 / RSI -- UNSATURATED rule-induction redesign.

Design source: docs/investigations/WEB_AND_SATURATION.md section 4 (RSI).
Companion criteria doc: docs/investigations/PDR_RSI_UNSATURATED.md.

WHAT CHANGED FROM THE SATURATED v1
----------------------------------
v1 never booted an engine, graded only k in {1,2,4}, and was FLAT at 1.00 -- a
mechanism check, not learning growth. This redesign:

  * boots the LIVE engine (``RealBackend``, ``--live``) and teaches each rule
    through conversation (``eng.process_turn``) on an isolated scratch store;
  * grades demonstration count k in {1,2,4,8};
  * uses GRADED DIFFICULTY per k: near transfer -> far transfer -> foil;
  * requires a MONOTONIC learning curve across k (the CAR-style gate) -- a
    non-decreasing rate per condition, checked at EVERY adjacent step;
  * adds NEGATIVE RULES: from counterexample (negated) demonstrations the
    system must NOT induce the rule. A counterexample-ignoring confound arm
    must be detectable;
  * reports growth headroom (k=1 vs k=8) but does NOT gate on strict growth,
    so a saturated 1.00 is reported honestly and G6 flags it INVALID.

BACKENDS (dependency injection; identical grading)
---------------------------------------------------
  * ``FakeBackend`` (DEFAULT) -- deterministic, no engine boot, no network,
    no save. Used by unit tests and the default ``main()`` (CI-safe).
  * ``RealBackend`` (``--live``) -- lazily imports and boots
    ``CognitiveChatEngine`` on a scratch ``data_dir`` and presents every demo
    through ``process_turn``. Real imports are lazy, so importing this module
    never boots an engine (skill 2b).

Sealed/primary hash-split sealing is preserved. Thresholds are pre-registered
and never tuned to force a PASS.

OBSERVATION SEAM (this revision): ``RealBackend.present()`` now GRADEs the
engine's ``process_turn`` response, parsed into the existing
``{relation, target, verdict}`` answer shape. No structured answer ->
additive ``engine_response_unparsed`` counter and a NO ANSWER verdict; the
input demo triples are never returned as the graded observation. No gate, TH,
decide(), instrument_controls() or gate_saturation() was changed.

Usage:
    uv run PYTHONHASHSEED=0 python scripts/measure_learn5_rsi.py
    uv run PYTHONHASHSEED=0 python scripts/measure_learn5_rsi.py --live
    uv run PYTHONHASHSEED=0 python scripts/measure_learn5_rsi.py --validate-only
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import front_gates  # noqa: E402  (shared Tier-0 saturation gate, unmodified)

SCRATCH = os.path.join(PROJ, "scratch", "rsi_live")
DATA_PATH = os.path.join(SCRATCH.replace("rsi_live", ""), "learn5_rsi_probes.jsonl")
SUMMARY_PATH = os.path.join(SCRATCH.replace("rsi_live", ""), "learn5_rsi_summary.json")
GATE_PATH = os.path.join(SCRATCH.replace("rsi_live", ""), "learn5_rsi_gates.jsonl")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

HASH_PREFIX = "RAVANA-LEARN-5"
COMPONENT = "RSI"
K_LEVELS = (1, 2, 4, 8)                      # redesigned: added k=8
CONDITIONS = ("near", "far", "foil", "negative")  # graded + negative rules
ARMS = ("structural", "naive", "chance", "counterexample")
N_FAMILIES = 120                            # >= 40 per pool after hash split

# Pre-registered decision thresholds (WEB_AND_SATURATION section 4 + companion
# doc). NOT tuned against any outcome.
TH = {
    "near_min": 0.75,
    "far_min": 0.70,
    "foil_reject_min": 0.85,
    "negative_reject_min": 0.80,
    "monotonic_tol": 0.05,        # max adjacent drop allowed across k
    "graded_drop_max": 0.15,      # max near->far->foil drop at a fixed k
    "sealed_drop_max": 0.10,
    "relation_coverage_min": 0.95,
    "chance_max": 0.35,
    "far_over_chance": 0.20,
    "foil_reject_hard_min": 0.70,
    "counterexample_reject_max": 0.20,  # confound must wrongly induce
    "permuted_hard_margin": 0.15,
    "permuted_margin": 0.10,
}

# Relation vocabulary (unchanged). Each relation has forward (S R O) and
# inverse (O R^-1 S) surface phrases; canonicalisation swaps roles for inverse.
RELATIONS = {
    "CAUSE": {
        "fwd": ["causes", "leads to", "produces", "brings about", "gives rise to"],
        "inv": ["is caused by", "results from", "is produced by", "arises from"],
    },
    "PART_OF": {
        "fwd": ["is part of", "belongs to", "is a component of"],
        "inv": ["contains", "includes", "is composed of"],
    },
    "PRECEDES": {
        "fwd": ["comes before", "precedes", "happens before"],
        "inv": ["comes after", "follows", "happens after"],
    },
    "SUPPORTS": {
        "fwd": ["supports", "strengthens", "bolsters", "reinforces"],
        "inv": ["is supported by", "is strengthened by", "is reinforced by"],
    },
    "OPPOSITE": {
        "fwd": ["contrasts with", "is opposite of", "differs sharply from"],
        "inv": [],  # symmetric
    },
    "RESEMBLES": {
        "fwd": ["resembles", "is similar to", "is akin to"],
        "inv": [],  # symmetric
    },
}
RELATION_KEYS = tuple(RELATIONS)
PHRASE_TO_REL = {}
for _r, _spec in RELATIONS.items():
    for _p in _spec["fwd"]:
        PHRASE_TO_REL[_p] = (_r, +1)
    for _p in _spec["inv"]:
        PHRASE_TO_REL[_p] = (_r, -1)

# Disjoint object pools. Extended to 24/domain so k=8 needs 16 demo objects
# while the held-out near/far/foil/negative queries use 8 reserved objects.
OBJECTS = {
    "weather": ["cloud", "storm", "breeze", "frost", "drizzle", "thunder",
                "rainbow", "hail", "mist", "gale", "dew", "sunset",
                "sleet", "shower", "glow", "chill",
                "tempest", "monsoon", "typhoon", "blizzard", "overcast",
                "thaw", "climate", "squall"],
    "body": ["heart", "lung", "nerve", "muscle", "bone", "skin",
             "liver", "brain", "tendon", "gland", "vessel", "reflex",
             "kidney", "joint", "tissue", "palm",
             "artery", "vein", "cartilage", "ligament", "plasma",
             "marrow", "diaphragm", "esophagus"],
    "machine": ["gear", "piston", "valley", "circuit", "rotor", "belt",
                "spring", "coil", "switch", "bearing", "lever", "fuse",
                "cam", "axle", "clutch", "pivot",
                "valve", "gasket", "shaft", "sprocket", "widget",
                "motor", "wheel", "pulley"],
    "social": ["habit", "custom", "law", "market", "team", "signal",
               "ritual", "contract", "rumor", "reward", "norm", "guild",
               "role", "rank", "game", "vote",
               "treaty", "charter", "election", "ceremony", "tradition",
               "society", "leader", "council"],
    "geometry": ["point", "line", "angle", "circle", "square", "triangle",
                 "plane", "radius", "chord", "vertex", "prism", "curve",
                 "arc", "cube", "axis", "slope",
                 "sphere", "ellipse", "polygon", "segment", "diameter",
                 "tangent", "quadrant", "lattice"],
    "plant": ["seed", "root", "stem", "leaf", "bud", "petal",
              "pollen", "bark", "branch", "vine", "fern", "moss",
              "trunk", "thorn", "cone", "fruit",
              "blossom", "sapling", "foliage", "herb", "shrub",
              "tree", "grain", "seedling"],
}
DOMAINS = tuple(OBJECTS)
for _a, _da in enumerate(DOMAINS):
    for _db in DOMAINS[_a + 1:]:
        if set(OBJECTS[_da]) & set(OBJECTS[_db]):
            raise RuntimeError("domain vocabularies overlap: %s / %s" % (_da, _db))

NETWORK_ATTEMPTS = []
SAVE_ATTEMPTS = []


# ── deterministic helpers (no RNG) ──────────────────────────────────────────
def _digest(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _hash_pick(seed, pool, n, exclude=()):
    out, seen = [], set(exclude)
    i = 0
    while len(out) < n and i < len(pool) * 8:
        idx = int(_digest(seed, i), 16) % len(pool)
        obj = pool[idx]
        if obj not in seen:
            seen.add(obj)
            out.append(obj)
        i += 1
    if len(out) != n:
        raise ValueError("hash_pick could not select %d distinct from %d" % (n, len(pool)))
    return out


def _hash_sort(seed, items):
    return sorted(items, key=lambda x: _digest(seed, x))


def item_hash(item):
    return _digest(HASH_PREFIX, COMPONENT, item["difficulty"], item["item_id"])


def item_manifest_hash(items):
    blob = json.dumps(
        [{"item_id": it["item_id"], "difficulty": it["difficulty"],
          "pool": it["pool"], "relation": it["relation"],
          "negative_relation": it["negative_relation"],
          "k_levels": list(K_LEVELS), "conditions": list(CONDITIONS),
          "hash": item_hash(it)} for it in items],
        sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _parse_phrase(phrase, strip_negation=False):
    """Surface phrase -> (relation, polarity). Negated phrases admit NO rule."""
    text = (phrase or "").strip().lower()
    negated = ("not " in text) or "n't" in text or " never " in (" " + text + " ")
    if negated and not strip_negation:
        return None
    if negated and strip_negation:
        text = text.replace("not ", "").replace(" never ", " ").replace("n't", "")
        text = " ".join(text.split())
    return PHRASE_TO_REL.get(text)


def canonicalise(subject, phrase, obj, strip_negation=False):
    parsed = _parse_phrase(phrase, strip_negation=strip_negation)
    if parsed is None:
        return None
    rel, polarity = parsed
    if polarity == -1:
        return (obj, rel, subject)
    return (subject, rel, obj)


# ── authored rule bank (fixed before any run) ───────────────────────────────
def build_items():
    items = []
    for fidx in range(N_FAMILIES):
        rel = RELATION_KEYS[fidx % len(RELATION_KEYS)]
        dname = DOMAINS[(fidx // 6) % len(DOMAINS)]
        fname = DOMAINS[((fidx // 6) + 3) % len(DOMAINS)]
        seed = "RSI-family-%03d" % fidx

        # 16 demo objects -> 8 pairs, enough for k=8. 8 reserved objects are
        # held out for near/far/foil/negative queries (never demonstrated).
        demo_objs = _hash_pick(seed + "|D", OBJECTS[dname], 16)
        pairs = [(demo_objs[i], demo_objs[i + 1]) for i in range(0, 16, 2)]
        demo_pairs = pairs[:8]
        demo_pair_objs = {o for pair in demo_pairs for o in pair}
        reserved = _hash_pick(seed + "|R", OBJECTS[dname], 8,
                              exclude=set(demo_objs))
        s_exact, t_exact = reserved[0], reserved[1]

        far_a, far_b = _hash_pick(seed + "|F", OBJECTS[fname], 2)

        # foil: subject is a reserved, undemonstrated object; candidates include
        # one demonstrated anchor so a naive surface matcher wrongly accepts.
        foil_subject = reserved[2]
        anchor = demo_pairs[0][0]
        foil_cands = [anchor] + _hash_pick(
            seed + "|foilc", OBJECTS[dname], 3,
            exclude={foil_subject, anchor} | demo_pair_objs)
        foil_rel = RELATION_KEYS[(fidx + 2) % len(RELATION_KEYS)]

        # NEGATIVE RULE: query subject has a TRUE R target in the world model,
        # but the demonstrations are COUNTEREXAMPLES ("X does not R Y"). A
        # system that respects negation induces no rule and MUST reject; one
        # that ignores negation induces R and wrongly asserts.
        neg_subject = reserved[3]
        neg_target = reserved[4]
        neg_cands = _hash_sort(
            seed + "|negc", [neg_target] + _hash_pick(
                seed + "|negd", OBJECTS[dname], 3,
                exclude={neg_subject, neg_target} | demo_pair_objs))

        wm = {}
        for s, t in pairs + [(s_exact, t_exact), (far_a, far_b),
                             (neg_subject, neg_target)]:
            wm.setdefault((rel, s), set()).add(t)

        exact_cands = _hash_sort(seed + "|ec",
                                 [t_exact] + _hash_pick(
                                     seed + "|exd", OBJECTS[dname], 3,
                                     exclude=({s_exact, t_exact} | demo_pair_objs)))
        far_cands = _hash_sort(seed + "|fc",
                               [far_b] + _hash_pick(
                                   seed + "|fard", OBJECTS[fname], 3,
                                   exclude={far_a, far_b}))

        item = {
            "item_id": "rsi_%03d" % fidx,
            "difficulty": ("easy", "medium", "hard", "expert")[fidx % 4],
            "relation": rel,
            "negative_relation": rel,  # the rule that must NOT be induced
            "domain": dname, "far_domain": fname, "foil_relation": foil_rel,
            "demo_pairs": demo_pairs,
            "exact": {"subject": s_exact, "candidates": exact_cands,
                      "gold_target": t_exact, "domain": dname},
            # 'near' is the graded-difficulty name for the same held-out query;
            # keep both keys for additive-JSON continuity with v1.
            "near": {"subject": s_exact, "candidates": exact_cands,
                     "gold_target": t_exact, "domain": dname},
            "far": {"subject": far_a, "candidates": far_cands,
                    "gold_target": far_b, "domain": fname},
            "foil": {"subject": foil_subject, "candidates": foil_cands,
                     "gold_target": None, "domain": dname},
            "negative": {"subject": neg_subject, "candidates": neg_cands,
                         "gold_target": None, "domain": dname,
                         # The true R target that a negation-ignoring arm
                         # would wrongly assert; structural must reject.
                         "counterexample_target": neg_target},
            "world_model": {k: sorted(v) for k, v in wm.items()},
        }
        items.append(item)

    # hash split within difficulty: first half PRIMARY, rest SEALED
    groups = collections.defaultdict(list)
    for it in items:
        groups[it["difficulty"]].append(it)
    for diff in sorted(groups):
        grp = sorted(groups[diff], key=item_hash)
        cut = len(grp) // 2
        for it in grp[:cut]:
            it["pool"] = "primary"
        for it in grp[cut:]:
            it["pool"] = "sealed"
    return items


def validate_items(items):
    errors = []
    if len({it["item_id"] for it in items}) != len(items):
        errors.append("duplicate item ids")
    for it in items:
        rel = it["relation"]
        if rel not in RELATIONS:
            errors.append("%s bad relation" % it["item_id"])
        wm = {(k[0], k[1]): set(v) for k, v in it["world_model"].items()}
        for cond in ("exact", "far"):
            q = it[cond]
            gold = q["gold_target"]
            if gold not in q["candidates"]:
                errors.append("%s %s gold not a candidate" % (it["item_id"], cond))
            if gold not in wm.get((rel, q["subject"]), set()):
                errors.append("%s %s gold not a world-model fact" % (it["item_id"], cond))
            if len(set(q["candidates"])) != len(q["candidates"]):
                errors.append("%s %s duplicate candidates" % (it["item_id"], cond))
        fq = it["foil"]
        if any(c in wm.get((rel, fq["subject"]), set()) for c in fq["candidates"]):
            errors.append("%s foil has an R target (not rejectable)" % it["item_id"])
        if it["foil_relation"] == rel:
            errors.append("%s foil relation equals rule relation" % it["item_id"])
        # negative: counterexample subject DOES have a true R target, so a
        # negation-ignoring arm can wrongly assert it.
        nq = it["negative"]
        if nq["counterexample_target"] not in wm.get((rel, nq["subject"]), set()):
            errors.append("%s negative counterexample target not a world-model fact"
                          % it["item_id"])
        if nq["counterexample_target"] not in nq["candidates"]:
            errors.append("%s negative counterexample target not a candidate"
                          % it["item_id"])
        if len(it["demo_pairs"]) < max(K_LEVELS):
            errors.append("%s demo_pairs shorter than max k" % it["item_id"])
        demo_objs = {o for pair in it["demo_pairs"] for o in pair}
        if it["exact"]["subject"] in demo_objs or it["exact"]["gold_target"] in demo_objs:
            errors.append("%s exact query reuses a demo object" % it["item_id"])
        if it["domain"] == it["far_domain"]:
            errors.append("%s far domain equals demo domain" % it["item_id"])
    return errors


# ── ENGINE RESPONSE PARSER ───────────────────────────────────────────────────
def _parse_engine_answer(responses, item, condition, subject, candidates):
    """Parse raw ``process_turn`` text into the existing answer shape.

    Accepts an explicit structured form (``relation=X target=Y verdict=Z``)
    or recovers a known relation phrase + candidate target from natural text.
    Returns ``None`` when NO structured answer is recoverable; the caller then
    records ``engine_response_unparsed`` and scores NO ANSWER. The input demo
    triples are never used as a substitute.
    """
    text = " ".join(str(r) for r in (responses or []) if r is not None)
    low = " ".join(text.lower().split())
    if not low:
        return None

    rel = None
    target = None
    verdict = None
    cand_lower = {c.lower(): c for c in candidates}

    # 1) explicit key=value form (auditable, deterministic)
    m = re.search(r"\brelation\s*[:=]\s*([a-z_-]+)", low)
    if m and m.group(1).upper() in RELATION_KEYS:
        rel = m.group(1).upper()
    m = re.search(r"\btarget\s*[:=]\s*([a-z0-9_-]+)", low)
    if m and m.group(1) in cand_lower:
        target = cand_lower[m.group(1)]
    m = re.search(r"\bverdict\s*[:=]\s*([a-z_]+)", low)
    if m and m.group(1) in ("match", "reject", "no_answer"):
        verdict = m.group(1)

    # 2) natural-language fallback: recover a known relation phrase and a
    #    candidate actually mentioned in the emission.
    if rel is None:
        for phrase, (r, _pol) in PHRASE_TO_REL.items():
            if phrase in low:
                neg_phrase = "not " + phrase
                if neg_phrase in low and low.count(phrase) == low.count(neg_phrase):
                    continue                      # only ever negated
                rel = r
                break
    if target is None:
        for c in candidates:
            if re.search(r"\b" + re.escape(c.lower()) + r"\b", low):
                target = c
                break

    neg = any(tok in low for tok in
              ("not ", "n't", "never ", " no ", "cannot", "can't",
               "unknown", "reject", "no answer"))
    if verdict is None:
        if neg:
            verdict = "reject"
        elif rel and target:
            verdict = "match"
        elif rel or target:
            verdict = "no_answer"                 # partial -> NO ANSWER
        else:
            return None                           # nothing structured
    if verdict == "match" and not (rel and target):
        verdict = "no_answer"
    return {"subject": subject, "relation": rel,
            "target": target, "verdict": verdict}


# ── BACKENDS ────────────────────────────────────────────────────────────────
class FakeBackend:
    """Deterministic demo channel. No engine boot, no save, no network.

    ``present`` returns the demo surface strings unchanged; the deterministic
    grader canonicalises them. This is the oracle used by unit tests and by
    the default main().
    """

    label = "fake"
    engine_graded = False       # no live engine; oracle derives from demo triples

    def __init__(self):
        self.turns = 0
        self.last_engine_answer = None
        self.last_unparsed = False

    def reset(self):
        self.turns = 0

    def present(self, demo_triples, item, k, condition):
        self.turns += len(demo_triples)
        return list(demo_triples)


class RealBackend(FakeBackend):
    """Live demo channel: boot CognitiveChatEngine, teach via conversation.

    Instantiated ONLY by ``main(--live)``. The real import/boot is lazy, so
    importing this module never boots an engine. Demos are presented through
    ``eng.process_turn`` on an isolated scratch ``data_dir``; the production
    weights directory is never touched. Any API failure is surfaced, never
    swallowed, and flips the backend_error rule to FAIL.
    """

    label = "live"
    engine_graded = True        # graded answer comes from process_turn output

    def __init__(self, data_dir=None):
        super().__init__()
        self.data_dir = data_dir or SCRATCH
        self._engine = None
        self.engines_booted = 0
        self.api_calls = {"boot": 0, "intake": 0, "save": 0, "cold_reload": 0,
                          "process_turn": 0}
        self.api_error = None
        # ADDITIVE observation-provenance counters. The graded answer now
        # comes from the engine's process_turn response, never the demo triples.
        self.engine_response_unparsed = 0
        self.engine_observation_missing = 0
        self.last_engine_answer = None

    def _boot(self):
        if self._engine is not None:
            return self._engine
        from ravana.chat.engine import CognitiveChatEngine
        os.makedirs(self.data_dir, exist_ok=True)
        eng = CognitiveChatEngine(data_dir=self.data_dir, user_suffix="")
        self._engine = eng
        self.engines_booted += 1
        self.api_calls["boot"] += 1
        return eng

    def present(self, demo_triples, item, k, condition):
        eng = self._boot()
        q = item[condition]
        responses = []
        for triple in demo_triples:
            s = "%s %s %s" % triple
            try:
                responses.append(eng.process_turn(s))
                self.api_calls["process_turn"] += 1
            except Exception as exc:  # surface, never swallow (skill 6)
                self.api_error = "process_turn: %r" % (exc,)
                responses.append("")
        # Construct validity: the demos teach, the QUERY is the test. Without
        # a query turn the engine only ever reacts to teaching assertions and
        # the graded observation is not an answer to the question at all
        # (measured: 0/1440 query turns, byte-identical near/far at fixed k).
        # Keep the query prompt free of the parser's key tokens so an echoed
        # instruction cannot manufacture a relation/target/verdict.
        question = ("Considering the examples above, what is true of %s? "
                    "Options: %s. Answer briefly.") % (
                        q["subject"], ", ".join(q["candidates"]))
        try:
            query_answer = eng.process_turn(question)
            self.api_calls["process_turn"] += 1
        except Exception as exc:  # surface, never swallow (skill 6)
            self.api_error = "process_turn(query): %r" % (exc,)
            query_answer = ""
        # Grade ONLY the query response. Demo reactions remain in the raw
        # observation list for audit but must not supply the answer tokens.
        answer = _parse_engine_answer([query_answer], item, condition,
                                      q["subject"], q["candidates"])
        unparsed = answer is None
        if unparsed:
            self.engine_response_unparsed += 1
            answer = {"subject": q["subject"], "relation": None,
                      "target": None, "verdict": "no_answer"}
        self.last_engine_answer = answer
        self.last_unparsed = unparsed
        # Return the raw ENGINE emissions (never the input demo triples) as
        # the observation list; the parsed, graded answer is carried on
        # ``last_engine_answer`` for ``run_item`` to consume.
        return list(responses)


# ── component under test ────────────────────────────────────────────────────
class RSIComponent:
    """Induce a relation from demonstrations and apply it structurally.

    arm='structural'    : canonicalise inverse/order, NO rule from negated
                          demos, majority relation, world-model lookup.
    arm='naive'         : surface/order memoriser (fails permutation, accepts
                          the foil). Confound arm.
    arm='chance'        : shuffled relation label. Confound arm.
    arm='counterexample': ignores negation, induces R from counterexamples and
                          asserts it. Confound arm for NEGATIVE RULES.
    """

    def __init__(self, arm="structural", backend=None):
        if arm not in ARMS:
            raise ValueError("unknown arm %r" % arm)
        self.arm = arm
        self.backend = backend or FakeBackend()

    def _demos(self, item, k, condition, permuted):
        """Return demo surface triples (s, phrase, o) for this condition."""
        pairs = list(item["demo_pairs"][:k])
        rel = item["relation"]
        if condition == "negative":
            # Counterexamples: "s does not R o". No positive rule is licensed.
            phrase = RELATIONS[rel]["fwd"][0]
            # "not " + the exact forward phrase so stripping the negation
            # recovers a phrase the vocabulary actually contains.
            return [(s, "not " + phrase, o) for s, o in pairs]
        if permuted:
            pairs = list(reversed(pairs))
            phrases = RELATIONS[rel]["inv"] or RELATIONS[rel]["fwd"]
            return [(o, phrases[i % len(phrases)], s)
                    for i, (s, o) in enumerate(pairs)]
        phrases = RELATIONS[rel]["fwd"]
        return [(s, phrases[i % len(phrases)], o) for i, (s, o) in enumerate(pairs)]

    def induce(self, demos):
        if self.arm == "naive":
            _s, phrase, _o = demos[0]
            parsed = _parse_phrase(phrase, strip_negation=True)
            if parsed is None:
                return None
            rel, polarity = parsed
            return rel if polarity == +1 else rel + "^-1"
        strip = self.arm == "counterexample"
        counts = collections.Counter()
        for s, phrase, o in demos:
            tri = canonicalise(s, phrase, o, strip_negation=strip)
            if tri is not None:
                counts[tri[1]] += 1
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def run_item(self, item, k, condition, permuted=False):
        q = item[condition]
        demo_triples = self._demos(item, k, condition, permuted)
        # LIVE: the backend returns raw engine emissions; the graded answer
        # was parsed from them. FAKE: the backend returns the demo triples and
        # the deterministic oracle derives the answer from them. Never
        # re-split emitted strings: relation phrases are multi-word.
        observed = self.backend.present(demo_triples, item, k, condition)
        is_engine = bool(getattr(self.backend, "engine_graded", False))
        engine_responses = list(observed) if is_engine else None
        engine_answer = (getattr(self.backend, "last_engine_answer", None)
                         if is_engine else None)
        induced = self.induce(demo_triples)   # authored demos (oracle/audit)
        subject = q["subject"]
        candidates = q["candidates"]
        wm = item["world_model"]

        if is_engine and self.arm == "structural":
            # GRADE THE ENGINE: no world-model shortcut, no demo triple.
            answer = dict(engine_answer or {})
            answer["subject"] = subject
        elif self.arm == "chance":
            shuffled = RELATION_KEYS[int(
                _digest(item["item_id"], "shuffle"), 16) % len(RELATION_KEYS)]
            target = (q["gold_target"] or
                      wm.get((item["relation"], subject), [None])[0] or
                      candidates[0])
            answer = {"subject": subject, "relation": shuffled,
                      "target": target, "verdict": "match"}
        elif self.arm == "counterexample":
            # Wrongly induces R from negated demos and asserts it.
            targets = [c for c in candidates
                       if induced and c in wm.get((induced, subject), [])]
            target = targets[0] if targets else candidates[0]
            answer = {"subject": subject, "relation": induced or item["relation"],
                      "target": target, "verdict": "match"}
        elif self.arm == "naive":
            demo_obj_set = {o for trio in demo_triples for o in (trio[0], trio[2])}
            seen = (subject in demo_obj_set
                    or any(c in demo_obj_set for c in candidates))
            if seen:
                answer = {"subject": subject, "relation": induced,
                          "target": demo_triples[0][2], "verdict": "match"}
            else:
                targets = [c for c in candidates
                           if induced and c in wm.get((induced, subject), [])]
                answer = {"subject": subject, "relation": induced,
                          "target": (targets[0] if len(targets) == 1 else None),
                          "verdict": ("match" if len(targets) == 1 else "reject")}
        else:
            # structural: canonicalised relation (None from negations), WM lookup
            targets = [c for c in candidates
                       if induced and c in wm.get((induced, subject), [])]
            if len(targets) == 1:
                answer = {"subject": subject, "relation": induced,
                          "target": targets[0], "verdict": "match"}
            elif len(targets) == 0:
                answer = {"subject": subject, "relation": induced,
                          "target": None, "verdict": "reject"}
            else:
                pick = _hash_sort(item["item_id"], targets)[0]
                answer = {"subject": subject, "relation": induced,
                          "target": pick, "verdict": "match"}

        rec = {
            "id": item["item_id"], "pool": item["pool"],
            "difficulty": item["difficulty"], "relation": item["relation"],
            "k": k, "condition": condition, "permuted": bool(permuted),
            "arm": self.arm, "backend": self.backend.label,
            "input": {"demonstrations": demo_triples, "query_subject": subject,
                      "candidates": candidates, "domain": q["domain"],
                      "foil_relation": item["foil_relation"]
                      if condition == "foil" else None},
            "output": answer,
            "gold": {"relation": item["relation"],
                     "target": q["gold_target"],
                     "should_reject": condition in ("foil", "negative")},
            "verdict": answer["verdict"],
        }
        if is_engine:
            # ADDITIVE live-observation provenance (per row): the engine
            # emission is the observation, never the demo triples.
            rec["engine_graded"] = True
            rec["engine_responses"] = engine_responses
            rec["engine_answer"] = engine_answer
            rec["engine_unparsed"] = bool(
                getattr(self.backend, "last_unparsed", False))
        return rec


def grade(rec):
    ans = rec["output"]
    f = {"relation_present": ans["relation"] in RELATION_KEYS or ans["relation"] is None}
    if rec["gold"]["should_reject"]:
        f["correct"] = (ans["verdict"] == "reject")
        f["exact_structural_match"] = None
    else:
        f["exact_structural_match"] = (
            ans["verdict"] == "match"
            and ans["relation"] == rec["gold"]["relation"]
            and ans["target"] == rec["gold"]["target"])
        f["correct"] = f["exact_structural_match"]
    f["verdict"] = rec["verdict"]
    rec["flags"] = f
    return rec


# ── sweep / summary / controls / decision ───────────────────────────────────
def run_sweep(items, backend=None):
    backend = backend or FakeBackend()
    comps = {arm: RSIComponent(arm, backend) for arm in ARMS}
    rows = []
    for it in items:
        for k in K_LEVELS:
            for cond in CONDITIONS:
                perm = (cond == "permuted")  # permuted is split into its own probe below
                for arm in ARMS:
                    rec = comps[arm].run_item(it, k, cond, permuted=perm)
                    rows.append(grade(rec))
                # explicit permutation probe for the graded-difficulty conditions
                if cond in ("near", "foil"):
                    for arm in ARMS:
                        rec = comps[arm].run_item(
                            it, k, cond, permuted=True)
                        rec["condition"] = "permuted"
                        rows.append(grade(rec))
    return rows


def _rate(rows, key):
    vals = [bool(r["flags"].get(key)) for r in rows]
    return round(sum(vals) / len(vals), 4) if rows else None


def _outcome(r):
    """Uniform correctness rate for any condition (positive or reject)."""
    return bool(r["flags"]["correct"])


def summarize(items, rows):
    out = {}
    conds = CONDITIONS + ("permuted",)
    for pool in ("primary", "sealed"):
        p = {"by_k": {str(k): {} for k in K_LEVELS}}
        srows = [r for r in rows if r["arm"] == "structural" and r["pool"] == pool]
        for cond in conds:
            cr = [r for r in srows if r["condition"] == cond]
            p[cond + "_n"] = len(cr)
            p[cond] = round(sum(_outcome(r) for r in cr) / len(cr), 4) if cr else None
            for k in K_LEVELS:
                kr = [r for r in cr if r["k"] == k]
                p["by_k"][str(k)][cond] = (round(sum(_outcome(r) for r in kr) / len(kr), 4)
                                           if kr else None)
        # monotonicity: max adjacent drop across k for each condition
        violations = {}
        for cond in CONDITIONS:
            rates = [p["by_k"][str(k)][cond] for k in K_LEVELS]
            drops = []
            for a, b in zip(rates, rates[1:]):
                if a is None or b is None:
                    continue
                drops.append(round(max(0.0, a - b), 4))
            violations[cond] = max(drops) if drops else 0.0
        p["monotonic_max_drop"] = max(violations.values())
        p["monotonic_by_condition"] = violations
        p["growth_headroom"] = (round(p["by_k"]["8"]["near"] - p["by_k"]["1"]["near"], 4)
                                if p["by_k"]["8"]["near"] is not None
                                and p["by_k"]["1"]["near"] is not None else None)
        # graded difficulty at each k: near >= far >= foil (max inversion)
        grade_drops = []
        for k in K_LEVELS:
            b = p["by_k"][str(k)]
            nf = max(0.0, (b["near"] or 0) - (b["far"] or 0))
            ff = max(0.0, (b["far"] or 0) - (b["foil"] or 0))
            grade_drops.extend([round(nf, 4), round(ff, 4)])
        p["graded_max_drop"] = max(grade_drops) if grade_drops else 0.0

        crows = [r for r in rows if r["arm"] == "chance" and r["pool"] == pool
                 and r["condition"] in ("near", "far")]
        p["chance"] = _rate(crows, "exact_structural_match")
        p["far_minus_chance"] = (round((p["far"] or 0) - p["chance"], 4)
                                 if p["chance"] is not None else None)
        sneg = [r for r in srows if r["condition"] == "negative"]
        cneg = [r for r in rows if r["arm"] == "counterexample" and r["pool"] == pool
                and r["condition"] == "negative"]
        p["negative_reject"] = (round(sum(_outcome(r) for r in sneg) / len(sneg), 4)
                                if sneg else None)
        p["counterexample_reject"] = (round(sum(_outcome(r) for r in cneg) / len(cneg), 4)
                                      if cneg else None)
        p["relation_coverage"] = _rate(srows, "relation_present")
        # legacy names preserved for additive-JSON continuity
        p["exact_exact"] = p["near"]
        p["permuted_exact"] = p["permuted"]
        p["foil_foil_reject"] = p["foil"]
        out[pool] = p
    # ADDITIVE observation-provenance counters (uniform schema with PDR).
    # main() overwrites these from the live backend before writing the summary.
    out["engine_observation_missing"] = 0
    out["engine_response_unparsed"] = 0
    return out


def instrument_controls(items, rows, summary):
    checks = []

    def by(pool, arm, cond, k=None):
        return [r for r in rows if r["pool"] == pool and r["arm"] == arm
                and r["condition"] == cond and (k is None or r["k"] == k)]

    def orate(rs):
        return round(sum(_outcome(r) for r in rs) / len(rs), 4) if rs else None

    # C1 positive control: structural near transfer at max k.
    c1 = (orate(by("primary", "structural", "near", max(K_LEVELS))) or 0) >= 0.90
    checks.append(("C1 positive structural near(k=8)>=0.90", c1))
    # C2 foil power: structural rejects, naive surface matcher accepts.
    sf = orate(by("primary", "structural", "foil"))
    nf = orate(by("primary", "naive", "foil"))
    checks.append(("C2 foil: structural rejects>=0.90 and naive<=0.10",
                   sf is not None and sf >= 0.90 and nf is not None and nf <= 0.10))
    # C3 permutation power.
    ne = orate(by("primary", "naive", "near"))
    np_ = orate(by("primary", "naive", "permuted"))
    se = summary["primary"]["near"] or 0
    sp = summary["primary"]["permuted"] or 0
    checks.append(("C3 permutation: naive gap>=0.15 and structural gap<=0.10",
                   (ne - np_) >= TH["permuted_hard_margin"]
                   and abs(se - sp) <= TH["permuted_margin"]))
    # C4 chance/far.
    checks.append(("C4 chance<=0.35 and far-chance>=0.20",
                   summary["primary"]["chance"] is not None
                   and summary["primary"]["chance"] <= TH["chance_max"]
                   and (summary["primary"]["far_minus_chance"] or 0) >= TH["far_over_chance"]))
    # C5 negative-rule power: structural rejects, counterexample arm accepts.
    sn = summary["primary"]["negative_reject"]
    cn = summary["primary"]["counterexample_reject"]
    checks.append(("C5 negative: structural reject>=0.85, counterexample<=0.20",
                   sn is not None and sn >= 0.85 and cn is not None
                   and cn <= TH["counterexample_reject_max"]))
    # C6 monotonicity across k (both pools).
    checks.append(("C6 monotonic learning curve (max adjacent drop<=0.05)",
                   all(summary[p]["monotonic_max_drop"] <= TH["monotonic_tol"]
                       for p in ("primary", "sealed"))))
    # C7 graded difficulty near->far->foil at each k (both pools).
    checks.append(("C7 graded difficulty near>=far>=foil (max drop<=0.15)",
                   all(summary[p]["graded_max_drop"] <= TH["graded_drop_max"]
                       for p in ("primary", "sealed"))))
    # C8 relation-field coverage + grader fixed vectors.
    def gv(cond, ans, gold_rel, gold_target, should_reject=False):
        rec = {"condition": cond, "verdict": ans["verdict"], "output": ans,
               "gold": {"relation": gold_rel, "target": gold_target,
                        "should_reject": should_reject}}
        return grade(rec)["flags"]["correct"]
    good = gv("near", {"verdict": "match", "relation": "CAUSE", "target": "fires"}, "CAUSE", "fires")
    wrong_rel = gv("near", {"verdict": "match", "relation": "PART_OF", "target": "fires"}, "CAUSE", "fires")
    wrong_target = gv("near", {"verdict": "match", "relation": "CAUSE", "target": "sparks"}, "CAUSE", "fires")
    foil_rej = gv("foil", {"verdict": "reject", "relation": None, "target": None}, "CAUSE", None, True)
    foil_acc = gv("foil", {"verdict": "match", "relation": "CAUSE", "target": "x"}, "CAUSE", None, True)
    neg_rej = gv("negative", {"verdict": "reject", "relation": None, "target": None}, "CAUSE", None, True)
    cov = all((summary[p]["relation_coverage"] or 0) >= TH["relation_coverage_min"]
              for p in ("primary", "sealed"))
    checks.append(("C8 relation coverage>=0.95 + grader fixed vectors",
                   cov and good and not wrong_rel and not wrong_target
                   and foil_rej and not foil_acc and neg_rej))
    return checks


def decide(summary, checks_ok, sat_status, backend_label="fake",
           backend_error=None):
    p, s = summary["primary"], summary["sealed"]
    rules = {}
    for pool, d in (("primary", p), ("sealed", s)):
        rules[pool + "_near>=0.75"] = (d["near"] or 0) >= TH["near_min"]
        rules[pool + "_far>=0.70"] = (d["far"] or 0) >= TH["far_min"]
        rules[pool + "_foil_reject>=0.85"] = (d["foil"] or 0) >= TH["foil_reject_min"]
        rules[pool + "_negative_reject>=0.80"] = (
            d["negative_reject"] is not None
            and d["negative_reject"] >= TH["negative_reject_min"])
        rules[pool + "_monotonic<=0.05"] = (
            d["monotonic_max_drop"] <= TH["monotonic_tol"])
        rules[pool + "_graded<=0.15"] = (
            d["graded_max_drop"] <= TH["graded_drop_max"])
        rules[pool + "_far>=chance+0.20"] = (
            d["far_minus_chance"] is not None
            and d["far_minus_chance"] >= TH["far_over_chance"])
        rules[pool + "_relation_coverage>=0.95"] = (
            d["relation_coverage"] or 0) >= TH["relation_coverage_min"]
    rules["sealed_near>=primary_near-0.10"] = (
        (s["near"] or 0) >= (p["near"] or 0) - TH["sealed_drop_max"])
    rules["instrument_controls_ok"] = bool(checks_ok)
    rules["no_network"] = len(NETWORK_ATTEMPTS) == 0
    rules["no_save"] = len(SAVE_ATTEMPTS) == 0
    rules["backend_error_free"] = backend_error is None
    rules["saturation_flag_recorded"] = sat_status in ("FLAG", "PASS", "FAIL")
    return rules, all(rules.values())


def _wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def statistics(rows, summary):
    out = {}
    for pool in ("primary", "sealed"):
        srows = [r for r in rows if r["arm"] == "structural" and r["pool"] == pool]
        ex = [r for r in srows if r["condition"] == "near"]
        ch = [r for r in rows if r["arm"] == "chance" and r["pool"] == pool
              and r["condition"] == "near"]
        a = sum(r["flags"]["exact_structural_match"] for r in ex)
        b = len(ex) - a
        c = sum(r["flags"]["exact_structural_match"] for r in ch)
        d = len(ch) - c
        out[pool] = {
            "near_correct": "%d/%d" % (a, len(ex)),
            "near_wilson95": _wilson(a, len(ex)),
            "chance_correct": "%d/%d" % (c, len(ch)),
            "growth_headroom": summary[pool]["growth_headroom"],
            "monotonic_max_drop": summary[pool]["monotonic_max_drop"],
            "negative_reject": summary[pool]["negative_reject"],
        }
    return out


# ── MAIN ────────────────────────────────────────────────────────────────────
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    live = "--live" in argv
    started = time.time()
    commit = os.popen("git -C %s rev-parse --short HEAD" % PROJ).read().strip()

    items = build_items()
    errors = validate_items(items)
    manifest = item_manifest_hash(items)
    pools = collections.Counter(it["pool"] for it in items)
    backend = RealBackend() if live else FakeBackend()

    print("=" * 78)
    print("RAVANA-LEARN-5 / RSI (UNSATURATED) -- backend=%s" % backend.label)
    print("=" * 78)
    print("  commit=%s families=%d primary=%d sealed=%d" %
          (commit, len(items), pools["primary"], pools["sealed"]))
    print("  hash_rule=%s|<difficulty>|<item_id>  manifest_sha256=%s" %
          (HASH_PREFIX, manifest))
    print("  k=%s conditions=%s arms=%s" %
          (K_LEVELS, CONDITIONS, ARMS))
    if errors:
        print("\nINSTRUMENT INVALID -- item definitions broken:")
        for e in errors[:20]:
            print("  ", e)
        return 2
    if "--validate-only" in argv:
        print("\nvalidate-only: all %d families valid." % len(items))
        return 0

    rows = run_sweep(items, backend=backend)
    summary = summarize(items, rows)
    stats = statistics(rows, summary)
    checks = instrument_controls(items, rows, summary)
    checks_ok = all(ok for _, ok in checks)
    print("\n--- 8 instrument controls (before any score) ---")
    for name, ok in checks:
        print("  [%s] %s" % ("ok" if ok else "BAD", name))
    if not checks_ok:
        print("\nINSTRUMENT INVALID -- refusing to report any score.")
        return 2

    backend_error = getattr(backend, "api_error", None)
    if live:
        calls = backend.api_calls
        if calls["boot"] < 1 or calls["process_turn"] < 1:
            backend_error = backend_error or "live path incomplete: %r" % calls
        print("  live api_calls=%s error=%r" % (calls, backend_error))

    sat = front_gates.gate_saturation(
        {"rsi_near": round(100 * (summary["primary"]["near"] or 0)),
         "rsi_negative_reject": round(100 * (summary["primary"]["negative_reject"] or 0))},
        None, "rsi")
    with open(GATE_PATH, "w", encoding="utf-8") as gf:
        gf.write(json.dumps(sat.as_dict("rsi"), ensure_ascii=False) + "\n")

    rules, passed = decide(summary, checks_ok, sat.status, backend.label,
                           backend_error)

    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n--- decision rules (pre-registered, not tuned) ---")
    for name, ok in rules.items():
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))

    print("\n--- PRIMARY vs SEALED (structural arm) ---")
    for pool in ("primary", "sealed"):
        d = summary[pool]
        print("  %-7s near=%.3f far=%.3f foil=%.3f negative=%.3f "
              "counterex=%.3f perm=%.3f chance=%.3f "
              "mono_drop=%.3f graded_drop=%.3f growth=%+.3f cov=%.3f" % (
                  pool, d["near"] or 0, d["far"] or 0, d["foil"] or 0,
                  d["negative_reject"] or 0, d["counterexample_reject"] or 0,
                  d["permuted"] or 0, d["chance"] or 0,
                  d["monotonic_max_drop"], d["graded_max_drop"],
                  d["growth_headroom"] or 0, d["relation_coverage"] or 0))
        print("    by k: " + "  ".join(
            "k=%s N=%.2f F=%.2f O=%.2f G=%.2f" % (
                k, d["by_k"][k]["near"] or 0, d["by_k"][k]["far"] or 0,
                d["by_k"][k]["foil"] or 0, d["by_k"][k]["negative"] or 0)
            for k in map(str, K_LEVELS)))
    print("  saturation gate: %s -- %s" % (sat.status, sat.detail))

    out = {
        "component": "RSI", "commit": commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": len(items), "pool_sizes": dict(pools),
        "hash_rule": "%s|<difficulty>|<item_id> (sha256)" % HASH_PREFIX,
        "manifest_sha256": manifest, "k_levels": list(K_LEVELS),
        "conditions": list(CONDITIONS), "arms": list(ARMS),
        "thresholds": TH, "rows": len(rows),
        "instrument_controls": [{"name": n, "ok": bool(ok)} for n, ok in checks],
        "instrument_controls_ok": checks_ok,
        "summary": summary, "statistics": stats,
        "rules": {k: bool(v) for k, v in rules.items()},
        "saturation_gate": {"status": sat.status, "detail": sat.detail},
        "pass": bool(passed), "backend": backend.label, "live": live,
        "api_calls": getattr(backend, "api_calls", {}),
        "backend_error": backend_error,
        # ADDITIVE observation-provenance counters (uniform schema with PDR).
        "engine_observation_missing": getattr(
            backend, "engine_observation_missing", 0),
        "engine_response_unparsed": getattr(
            backend, "engine_response_unparsed", 0),
        "no_network": len(NETWORK_ATTEMPTS) == 0,
        "no_save": len(SAVE_ATTEMPTS) == 0,
        "data_path": DATA_PATH, "elapsed_s": round(time.time() - started, 1),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        json.dump(out, sf, indent=2, ensure_ascii=False)

    print("\nOVERALL: %s" % ("PASS" if passed else "FAIL"))
    print("artifacts: %s , %s" % (DATA_PATH, SUMMARY_PATH))
    print("elapsed %.1fs | DONE" % (time.time() - started))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
