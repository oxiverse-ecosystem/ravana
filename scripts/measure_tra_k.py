#!/usr/bin/env python3
"""TRA-K benchmark harness -- pre-registered knowledge-conditioned typed role acquisition.

SPEC: docs/investigations/TRA_K_SPEC.md (branch origin/docs/tra-k-spec, frozen).
Item bank: scripts/tra_k_items.py (400 families / 200+200 / exactly 40 per k).

WHAT THIS FILE IS
-----------------
A faithful implementation of the NEW pre-registered instrument. It is STATIC
for the fake arm: importing the module boots no engine, makes no network call,
writes no store and patches no process global. ``RealBackend`` lazily imports
``CognitiveChatEngine`` and is only constructed by ``main(--live)``.

PHASES (spec):
  P0 freeze       build + validate bank, hash split, manifest hashes  [no grade]
  P1 baseline     one cold query turn per item                         [no grade]
  P2 k typed exemplars (arm A, positive control, saturates)
  P2c channel-evidence teaching (arm B, the deployable input channel)
  P3 query        ONE graded turn; verdict parsed from it only         [GRADED]
  P4 immediate    one paraphrase query per item                        [no grade]
  P5 save         eng.save() into an isolated scratch data_dir + cold reload
  P6 delayed      one held-out paraphrase on the FRESH engine          [confirmatory]
  SI              unrelated-subject probe (subject-independence, HARD FAIL)

The graded path never reads ``world_model``, ``type_sets``, ``domain`` or
``far_domain``. ``channel_evidence`` is the ONLY new input: a provenance-
stamped fact list loaded from ``TRAK_CHANNEL_JSON`` (or a deterministic
channel-agnostic fallback). The forced-choice options come from
``typed_queries``: four SOURCE-NATIVE TYPE LABELS per probe, one correct label
present in this item's own channel facts and three deterministic foils from
other channel facts (same domain first). Overlap with the oracle fields is
REPORTED and any deployable path that uses them is a HARD FAIL (control C15).

SUBJECT-TYPE INPUT AND k=0 (this fix)
-------------------------------------
The benchmark input defect (a query subject absent from its channel) is fixed
in ``tra_k_items.py``: query subjects are substrate-covered and each item
carries the subject's own source-native type assertion plus low-credibility
conflicts, admitted by the documented source-provenance rule. The harness
never reads ``type_sets``/``world_model``/gold to build or rank channel facts;
those fields are consulted only by the negative answer-key filter that
FORBIDS a channel fact from naming them.

DEVIATION from the earlier P2c implementation (documented, not a threshold
change): channel teaching is now given at k>=1 only. At k=0 the deployable
arm receives the same 8 matched NEUTRAL turns as the cold baseline, so k=0 is
a true null; the spec's "P2c in the same k calls" implies zero teaching at
k=0. Our previous code taught channel facts even at k=0, which made the
k=0 point a second copy of the k>=1 arm and flattened the learning curve.
All C1-C15 thresholds and the G6/M-A/M-B controls are unchanged.

Controls C1-C13 are copied unchanged from TRA (thresholds frozen). C14
(subject independence) and C15 (channel independence) are the two new gates.
C11 is a CAPABILITY gate: the fake arm MUST fail it; it is evaluated on the
live backend only. All other controls are certified on the fake/oracle arm.

Usage:
    python scripts/measure_tra_k.py                 # fake, CI-safe, no boot
    python scripts/measure_tra_k.py --validate-only # freeze check only
    python scripts/measure_tra_k.py --live          # real engine, scratch store
    TRAK_MAX_PER_POOL=20 python scripts/measure_tra_k.py --live  # focused probe
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import json
import math
import os
import re
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _p in (PROJ, _SCRIPTS, os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import front_gates                       # noqa: E402  shared Tier-0 G6, unmodified
import tra_k_items                       # noqa: E402  frozen item bank

HASH_PREFIX = tra_k_items.HASH_PREFIX
COMPONENT = tra_k_items.COMPONENT
K_LEVELS = tra_k_items.K_LEVELS                  # (0, 1, 2, 4, 8)
CONDITIONS = tra_k_items.CONDITIONS              # near, far, foil, negative, unrelated
ARMS = ("channel", "ostensive", "naive", "chance", "counterexample", "permuted")
CHANNEL_ARM = "channel"                           # deployable arm (input channel B)
OSTENSIVE_ARM = "ostensive"                       # positive control (channel A)
ROLE_KEYS = tra_k_items.ROLE_KEYS
ROLE_TO_TYPE = tra_k_items.ROLE_TO_TYPE
ROLE_TO_PHRASE = tra_k_items.ROLE_TO_PHRASE
UNRELATED_SUBJECT = tra_k_items.UNRELATED_SUBJECT

SCRATCH = os.path.join(PROJ, "scratch", "tra_k_harness")
DATA_PATH = os.path.join(SCRATCH, "tra_k_rows.jsonl")
SUMMARY_PATH = os.path.join(SCRATCH, "tra_k_summary.json")
GATE_PATH = os.path.join(SCRATCH, "tra_k_gates.jsonl")
MANIFEST_PATH = os.path.join(SCRATCH, "tra_k_manifest.json")
os.makedirs(SCRATCH, exist_ok=True)

# Pre-registered thresholds -- FROZEN, copied unchanged from TRA. No threshold
# or gate is edited to alter a verdict. tests/unit/test_measure_tra_k.py pins it.
TH = {
    "transfer_min": 0.75,
    "near_min": 0.75,
    "far_min": 0.70,
    "far_chance_min": 0.20,
    "chance_max": 0.35,
    "foil_reject_min": 0.85,
    "foil_hard_min": 0.70,
    "negative_reject_min": 0.80,
    "negative_hard_min": 0.70,
    "permuted_margin": 0.10,
    "permuted_hard_margin": 0.15,
    "monotonic_tol": 0.05,
    "graded_drop_max": 0.15,
    "coverage_min": 0.95,
    "coverage_hard_min": 0.90,
    "retention_min": 0.80,
    "control_delta_max": 0.05,
    "sealed_drop_max": 0.10,
    "growth_min": 0.15,
    "c1_near_min": 0.90,
    "c2_foil_min": 0.90,
    "c2_naive_max": 0.10,
    "c5_struct_min": 0.85,
    "c5_counter_max": 0.20,
    "c8_coverage_min": 0.95,
}
TRUSTED_KINDS = ("curated", "web_fact", "definition", "conversation")
SLOT_ENV_FLAG = "RAVANA_TRA_SLOTS"        # production default-OFF primitive
TRAK_ENV_FLAG = "RAVANA_TRA_K_SLOTS"      # harness marker, also default-OFF
ORACLE_ONLY_FIELDS = ("type_sets", "world_model")
DEPLOYABLE_INPUT_FIELDS = (
    "item_id", "role", "type_label", "difficulty", "domain", "far_domain",
    "subject", "taught_exemplars", "unseen_test_words", "typed_queries",
    "foil", "negative", "channel_evidence", "k", "pool",
)


# ── flag lifecycle: live arm only, always restored ──────────────────────────
@contextlib.contextmanager
def trak_slots_flag():
    """Enable both default-OFF flags for the live arm and restore prior state.

    The fake arm must never enter this context; a unit test asserts a fake
    sweep leaves both variables byte-identical (including 'unset').
    """
    keys = (SLOT_ENV_FLAG, TRAK_ENV_FLAG)
    previous = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k in keys:
            if previous[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = previous[k]


# ── phase surfaces (provenance-safe; NEVER gold/oracle metadata) ────────────
def query_for(item, condition):
    """The graded forced-choice probe: options are channel TYPE LABELS.

    ``unseen_test_words`` remains the oracle-side domain-word record; the
    deployable path reads only ``typed_queries`` (derived from channel facts).
    A minimal hand-built item without that field falls back to the legacy
    shape so pure unit probes stay usable.
    """
    tq = item.get("typed_queries")
    if isinstance(tq, dict) and condition in tq:
        return dict(tq[condition])
    if condition == "unrelated":
        q = item["unseen_test_words"]["near"]
        return {"subject": UNRELATED_SUBJECT, "candidates": list(q["candidates"]),
                "domain": item["domain"], "gold": None}
    if condition in ("near", "far"):
        return dict(item["unseen_test_words"][condition])
    q = item[condition]
    return {"subject": q["subject"], "candidates": list(q["candidates"]),
            "domain": item["domain"], "gold": None}


def query_surface(item, condition, paraphrase=False):
    """P3/P4/P6 prompt. Deliberately free of the parser's key tokens."""
    q = query_for(item, condition)
    opts = ", ".join(q["candidates"])
    if paraphrase:
        return ("Given the demonstrations so far, which option best fits %s? "
                "Options: %s. Reply briefly." % (q["subject"], opts))
    return ("Considering the examples above, what is true of %s? "
            "Options: %s. Answer briefly." % (q["subject"], opts))


_NEUTRAL = ("what do you know about %s?",
            "things happen for many reasons, don't they?")


def channel_teaching_surfaces(item):
    """P2c: assertions from ADMITTED provenance-stamped channel facts only."""
    out = []
    for fact in tra_k_items.ordered_channel_facts(item):
        surface = tra_k_items.channel_assertion(fact)
        if surface and surface not in out:
            out.append(surface)
        if len(out) >= max(K_LEVELS):
            break
    return out


def channel_negation_surfaces(item):
    """P2c negative control: channel facts rendered with explicit negation.

    Still sourced ONLY from provenance-stamped channel facts and their
    source-native type labels; no oracle field contributes a token.
    """
    out = []
    for fact in tra_k_items.ordered_channel_facts(item):
        surface = tra_k_items.channel_negation_assertion(fact)
        if surface and surface not in out:
            out.append(surface)
        if len(out) >= max(K_LEVELS):
            break
    return out


def teaching_surfaces(item, k, condition, arm, permuted=False):
    """P2 (+ P2c for k>=1). k=0 -> 8 matched neutral turns, NO channel teaching.

    The k=0 point is the matched-exposure null: even the deployable channel
    arm receives only neutral turns there. Channel assertions enter at k>=1.
    """
    if k == 0:
        subject = query_for(item, condition)["subject"]
        p2 = [_NEUTRAL[0] % subject if i % 2 == 0 else _NEUTRAL[1]
              for i in range(max(K_LEVELS))]
    else:
        exemplars = list(item["taught_exemplars"][:k])
        if permuted:
            exemplars = list(reversed(exemplars))
        if condition == "negative" and arm != "counterexample":
            phrase = ROLE_TO_PHRASE[item["role"]]
            p2 = ["%s does not %s %s." % (ex["r1"], phrase, ex["r2"])
                  for ex in exemplars]
        else:
            # Single whole is_a assertion per exemplar (production writer
            # refuses multi-clause text -- see TRA harness note).
            label = item["type_label"]
            article = "an" if label[:1].lower() in "aeiou" else "a"
            p2 = ["%s is %s %s." % (ex["r2"], article, label)
                  for ex in exemplars]
    if arm == CHANNEL_ARM and k >= 1:
        # Negative control: the channel's own type labels are taught WITH
        # negation, matching the reject-only foil/negative option space.
        if condition == "negative":
            return p2 + channel_negation_surfaces(item)
        return p2 + channel_teaching_surfaces(item)
    return p2


# ── answer parser (from measure_learn5_rsi.py:378-475, unchanged) ───────────
def _parse_engine_answer(responses, item, condition, subject, candidates):
    """Parse ONLY the graded query response. Unparseable -> verdict=no_answer."""
    raw = [str(r) for r in (responses or []) if r is not None]
    text = " ".join(raw)
    low = " ".join(text.lower().split())
    if not low:
        return None

    def _is_hedge(s):
        l = " ".join(s.lower().split())
        return ("not totally sure" in l or "not sure about" in l
                or "solid grasp on" in l or "outside what i know" in l
                or l.startswith(("right.", "yeah.", "got it.", "ok, noted.")))

    ev_low = " ".join(" ".join(s.lower().split()) for s in raw
                      if not _is_hedge(s))
    role = target = verdict = None
    cand_lower = {c.lower(): c for c in candidates}

    m = re.search(r"\brole\s*[:=]\s*([a-z_-]+)", low)
    if m and m.group(1).upper() in ROLE_KEYS:
        role = m.group(1).upper()
    # Source-native type labels are multi-word (e.g. "cloud cover"). Capture
    # the whole explicit target up to the verdict token instead of a single
    # word, otherwise a shorter foil that prefixes the gold wins the parse.
    m = re.search(r"\btarget\s*[:=]\s*(.+?)(?=\s+verdict\s*[:=]|\s*$)", low)
    if m:
        token = m.group(1).strip(" .!?,")
        if token in cand_lower:
            target = cand_lower[token]
    m = re.search(r"\bverdict\s*[:=]\s*([a-z_]+)", low)
    if m and m.group(1) in ("match", "reject", "no_answer"):
        verdict = m.group(1)

    if role is None:
        for rname, spec in tra_k_items.ROLES.items():
            phrase = spec["phrase"]
            if phrase in ev_low:
                neg_phrase = "not " + phrase
                if neg_phrase in ev_low and ev_low.count(phrase) == ev_low.count(neg_phrase):
                    continue
                role = rname
                break
    if target is None:
        # Longest first: a multi-word gold must not be shadowed by a foil that
        # is its own prefix (source-native labels contain spaces).
        for c in sorted(candidates, key=lambda x: -len(str(x))):
            if re.search(r"\b" + re.escape(c.lower()) + r"\b", ev_low):
                target = c
                break

    neg = any(tok in low for tok in
              ("not ", "n't", "never ", " no ", "cannot", "can't",
               "unknown", "reject", "no answer"))
    if verdict is None:
        if neg:
            verdict = "reject"
        elif role and target:
            verdict = "match"
        elif role or target:
            verdict = "no_answer"
        else:
            return None
    if verdict == "match" and not (role and target):
        verdict = "no_answer"

    bare = low.strip(" .!?\"'")
    if (target is not None and verdict in (None, "no_answer")
            and bare in cand_lower):
        target = cand_lower[bare]
        verdict = "match"
        if role is None and condition in ("near", "far"):
            role = item["role"]
    return {"subject": subject, "role": role,
            "target": target, "verdict": verdict}


# ── backends ────────────────────────────────────────────────────────────────
class FakeBackend:
    """Deterministic oracle. NO engine boot, save, network or store write.

    ``engine_graded=False`` keeps C11 hostile to this arm by construction.
    """

    label = "fake"
    engine_graded = False

    def __init__(self):
        self.turns = 0
        self.last_engine_answer = None
        self.last_unparsed = False
        self.api_calls = {"boot": 0, "intake": 0, "save": 0,
                          "cold_reload": 0, "process_turn": 0}
        self.api_error = None
        self.engine_response_unparsed = 0
        self.engine_observation_missing = 0
        self.synthetic_fallback_used = 0
        self.engines_booted = 0
        self.reloaded = 0
        self.provenance_views_cache = []

    def reset_for_item(self, item):
        self.last_engine_answer = None

    def teach(self, item, k, condition, arm, surfaces, permuted=False):
        self.turns += len(surfaces)
        return list(surfaces)

    def _response(self, item, condition, arm, k, permuted=False, phase="query"):
        q = query_for(item, condition)
        role = item["role"]
        correct = _fake_correct(arm, condition, k, permuted, phase=phase)
        if condition in ("foil", "negative", "unrelated"):
            if correct:
                return "role=%s target=none verdict=reject" % role
            return "role=%s target=%s verdict=match" % (
                role, q["candidates"][0])
        if correct:
            return "role=%s target=%s verdict=match" % (role, q["gold"])
        others = [c for c in q["candidates"] if c != q["gold"]]
        return "role=%s target=%s verdict=match" % (role, others[0])

    def query(self, item, condition, k, phase="query", permuted=False):
        arm = self._arm
        response = self._response(item, condition, arm, k, permuted,
                                  phase=phase)
        q = query_for(item, condition)
        answer = _parse_engine_answer([response], item, condition,
                                      q["subject"], q["candidates"])
        unparsed = answer is None
        if unparsed:
            self.engine_response_unparsed += 1
            answer = {"subject": q["subject"], "role": None,
                      "target": None, "verdict": "no_answer"}
        self.last_engine_answer = answer
        self.last_unparsed = unparsed
        return [response]

    def set_arm(self, arm):
        self._arm = arm

    def save_reload(self, item):
        self.api_calls["save"] += 1
        self.api_calls["cold_reload"] += 1
        self.reloaded += 1
        self.provenance_views_cache = _synthetic_provenance_views(item)

    def provenance_views(self):
        return list(self.provenance_views_cache)


def _fake_correct(arm, condition, k, permuted=False, phase="query"):
    """Oracle-shaped k curve.

    P1 ``baseline`` is COLD -- no channel evidence has been taught yet, so
    every arm must be absent there (otherwise growth is a tautological 0).

    CHANNEL arm: the independent channel is taught at k>=1 only. At k=0 the
    arm is the matched neutral null, so near/far are absent there and the
    k-curve grows from 0 -> 1. Reject probes (foil/negative/unrelated) stay
    correct at k=0 because declining is the right answer before teaching.
    Saturation at the interior k levels is handled by G6 + M-A/M-B, not by a
    threshold change.
    OSTENSIVE arm: positive control A, known to saturate at k>=1.
    Controls naive/chance/counterexample stay wrong on their target cells.
    """
    reject_cond = condition in ("foil", "negative", "unrelated")
    if phase == "baseline":
        return False                               # cold: nothing taught yet
    if arm in (CHANNEL_ARM, "permuted"):
        # Reversing demo order does not affect an independent channel; both
        # arms share the k=0 neutral null and the k>=1 channel teaching.
        return True if reject_cond else k >= 1
    if arm == OSTENSIVE_ARM:
        return True if reject_cond else k >= 1
    if arm == "naive":
        if permuted:
            return False                           # surface/order memoriser breaks
        if reject_cond:
            return False                           # wrongly accepts all controls
        return True
    if arm == "chance":
        return False
    if arm == "counterexample":
        return False                               # ignores negation -> accepts
    return False


# ── provenance seam: scope a REAL census to this item's taught demo edges ──
_TAUGHT_SLOT_RE = re.compile(
    r"\b([a-z][a-z\-]*)\s+is\s+(?:a|an)\s+([a-z][a-z\-]+)\b", re.I)


def taught_slot_pairs(surfaces, type_label):
    """(source, target) pairs the teaching surfaces ask to store.

    Derived strictly from the surfaces actually taught into the engine, never
    from oracle fields or the query subject's type.
    """
    label = str(type_label or "").strip().lower()
    pairs = set()
    for surface in surfaces or ():
        for clause in str(surface).split("."):
            m = _TAUGHT_SLOT_RE.search(clause)
            if not m:
                continue
            src = m.group(1).strip().lower()
            tgt = m.group(2).strip().lower()
            if label and tgt == label:
                pairs.add((src, tgt))
    return pairs


def scope_taught_views(census, taught_pairs):
    """Filter a real typed-slot census to conversation-taught demo edges."""
    allowed = set(taught_pairs or ())
    if not allowed:
        return []
    out = []
    for v in census or ():
        src = str(getattr(v, "source", "") or "").strip().lower()
        tgt = str(getattr(v, "target", "") or "").strip().lower()
        if (src, tgt) not in allowed:
            continue
        kind = str(getattr(v, "edge_kind", "") or "").strip().lower()
        source = str(getattr(v, "edge_source", "") or "").strip().lower()
        stype = str(getattr(v, "source_type", "") or "").strip().lower()
        if kind not in TRUSTED_KINDS or source != "conversation" or stype != "user":
            continue
        out.append({
            "subject": src, "target": tgt,
            "relation_type": str(getattr(v, "relation_type", "") or ""),
            "type_label": str(getattr(v, "type_label", "") or "").strip(),
            "edge_kind": kind, "source_type": stype, "edge_source": source,
            "url": str(getattr(v, "url", "") or ""),
            "fabricated_span": bool(getattr(v, "fabricated_span", False)),
            "confidence": float(getattr(v, "confidence", 0.0) or 0.0),
        })
    return out


class RealBackend(FakeBackend):
    """Live engine. Lazy import; isolated per-item scratch ``data_dir``."""

    label = "live"
    engine_graded = True

    def __init__(self, data_dir=None):
        super().__init__()
        self._engine = None
        self.base_dir = data_dir or os.path.join(SCRATCH, "live_data")
        self.data_dir = self.base_dir
        self.current_item_id = None
        self._taught_slot_pairs = set()
        self._all_provenance_views = []

    def reset_for_item(self, item):
        self._engine = None
        self.current_item_id = item["item_id"]
        self.data_dir = os.path.join(self.base_dir, item["item_id"])
        self.last_engine_answer = None
        self._taught_slot_pairs = set()
        self.provenance_views_cache = []

    def _boot(self):
        if self._engine is not None:
            return self._engine
        from ravana.chat.engine import CognitiveChatEngine
        os.makedirs(self.data_dir, exist_ok=True)
        self._engine = CognitiveChatEngine(data_dir=self.data_dir, user_suffix="")
        self.engines_booted += 1
        self.api_calls["boot"] += 1
        return self._engine

    def _refresh_provenance_views(self):
        census = []
        if self._engine is not None and self._taught_slot_pairs:
            try:
                from ravana.chat.schema_induction import typed_slot_views_from_graph
                census = typed_slot_views_from_graph(self._engine.graph)
            except Exception as exc:               # surface, never swallow
                self.api_error = self.api_error or "provenance: %r" % (exc,)
                census = []
        self.provenance_views_cache = scope_taught_views(
            census, self._taught_slot_pairs)
        seen = {(v.get("subject"), v.get("target"))
                for v in self._all_provenance_views}
        for v in self.provenance_views_cache:
            key = (v.get("subject"), v.get("target"))
            if key not in seen:
                self._all_provenance_views.append(v)
                seen.add(key)

    def teach(self, item, k, condition, arm, surfaces, permuted=False):
        eng = self._boot()
        out = []
        for surface in surfaces:
            try:
                out.append(str(eng.process_turn(surface)))
                self.api_calls["process_turn"] += 1
                if " is a " in surface or " is an " in surface:
                    self.api_calls["intake"] += 1
            except Exception as exc:               # surface, never swallow
                self.api_error = self.api_error or "teach: %r" % (exc,)
                out.append("")
        self._taught_slot_pairs.update(
            taught_slot_pairs(surfaces, item["type_label"]))
        self._refresh_provenance_views()
        return out

    def query(self, item, condition, k, phase="query", permuted=False):
        eng = self._boot()
        question = query_surface(item, condition, paraphrase=(phase != "query"))
        q = query_for(item, condition)
        try:
            response = str(eng.process_turn(question))
            self.api_calls["process_turn"] += 1
        except Exception as exc:
            self.api_error = self.api_error or "query: %r" % (exc,)
            response = ""
        if not response.strip():
            self.engine_observation_missing += 1
        answer = _parse_engine_answer([response], item, condition,
                                      q["subject"], q["candidates"])
        unparsed = answer is None
        if unparsed:
            self.engine_response_unparsed += 1
            answer = {"subject": q["subject"], "role": None,
                      "target": None, "verdict": "no_answer"}
        self.last_engine_answer = answer
        self.last_unparsed = unparsed
        self._refresh_provenance_views()
        return [response]

    def save_reload(self, item):
        eng = self._boot()
        try:
            eng.save()
            self.api_calls["save"] += 1
        except Exception as exc:
            self.api_error = self.api_error or "save: %r" % (exc,)
            return
        self._engine = None
        self.reloaded += 1
        try:
            self._boot()
            self.api_calls["cold_reload"] += 1
            self._refresh_provenance_views()
        except Exception as exc:
            self.api_error = self.api_error or "cold_reload: %r" % (exc,)

    def provenance_views(self):
        return list(getattr(self, "_all_provenance_views",
                            self.provenance_views_cache))


# ── row construction / grading ──────────────────────────────────────────────
def _synthetic_provenance_views(item):
    label = item["type_label"]
    return [{"edge_kind": "curated", "source_type": "user",
             "edge_source": "conversation", "relation_type": "is_a",
             "type_label": label, "url": "", "fabricated_span": False,
             "confidence": 0.9}]


def run_item(item, k, condition, arm, phase, backend, permuted=False,
             teach=True):
    q = query_for(item, condition)
    backend.set_arm(arm)
    surfaces = teaching_surfaces(item, k, condition, arm, permuted) if teach else []
    backend.teach(item, k, condition, arm, surfaces, permuted)
    # P1 is a COLD pre-teaching probe: answer as k=0 regardless of assigned k.
    query_k = 0 if phase == "baseline" else k
    responses = backend.query(item, condition, query_k, phase=phase,
                              permuted=permuted)
    answer = dict(backend.last_engine_answer or {})
    answer["subject"] = q["subject"]
    is_engine = bool(getattr(backend, "engine_graded", False))
    source_names = sorted({str(f.get("source", "")) for f in
                           item.get("channel_evidence", [])})
    rec = {
        "id": item["item_id"], "pool": item["pool"],
        "difficulty": item["difficulty"], "role": item["role"],
        "k": k, "condition": condition, "arm": arm, "phase": phase,
        "permuted": bool(permuted), "backend": backend.label,
        "input": {"query_subject": q["subject"], "candidates": q["candidates"],
                  "domain": (item["far_domain"] if condition == "far"
                             else item["domain"]),
                  "teaching_turns": len(surfaces),
                  "teaching_surfaces": surfaces if not is_engine else [],
                  "channel_facts": len(item.get("channel_evidence", [])),
                  "channel_sources": source_names,
                  "subject_type_facts": [
                      {"subject": str(f.get("subject", "")),
                       "type_label": str(f.get("object", "")),
                       "source": str(f.get("source", ""))}
                      for f in tra_k_items.ordered_channel_facts(item)
                      if f.get("evidence_role") == tra_k_items.SUBJECT_TYPE_ROLE]},
        "output": answer,
        "gold": {"role": item["role"], "target": q["gold"],
                 "should_reject": condition in ("foil", "negative", "unrelated")},
        "observation_source": "query_response" if is_engine else "fake_oracle",
        "engine_graded": is_engine,
        "engine_responses": list(responses) if is_engine else [],
        "engine_answer": dict(backend.last_engine_answer or {}) if is_engine else None,
        "engine_unparsed": bool(getattr(backend, "last_unparsed", False)),
        "reloaded": int(getattr(backend, "reloaded", 0) >= 1),
        "slot_read_fresh": phase == "delayed" and bool(
            getattr(backend, "reloaded", 0) >= 1),
        "trust_gate_disabled": False,
    }
    return grade(rec)


def grade(rec):
    ans = rec["output"]
    gold = rec["gold"]
    f = {"role_present": ans["role"] in ROLE_KEYS or ans["role"] is None}
    if gold["should_reject"]:
        f["exact_structural_match"] = None
        f["correct"] = ans["verdict"] == "reject"
    else:
        f["exact_structural_match"] = (
            ans["verdict"] == "match" and ans["role"] == gold["role"]
            and ans["target"] == gold["target"])
        f["correct"] = bool(f["exact_structural_match"])
    f["verdict"] = ans["verdict"]
    rec["flags"] = f
    return rec


def _outcome(rec):
    return bool(rec["flags"]["correct"])


def run_sweep(items, backend=None):
    """Phases P1-P6 + SI. One k per item; one engine per item in live mode."""
    backend = backend or FakeBackend()
    rows = []
    for item in items:
        backend.reset_for_item(item)
        k = item["k"]
        rows.append(run_item(item, k, "near", CHANNEL_ARM, "baseline",
                             backend, teach=False))
        for condition in CONDITIONS:
            for arm in ARMS:
                # P2/P2c teaching + P3 graded query (the ONLY graded phase).
                rows.append(run_item(item, k, condition, arm, "query",
                                     backend, arm == "permuted"))
                # P4 immediate paraphrase: diagnostic, channel arm only.
                if arm == CHANNEL_ARM:
                    rows.append(run_item(item, k, condition, arm, "immediate",
                                         backend, arm == "permuted"))
        # SI: unrelated subject must NOT receive the confident graded answer.
        rows.append(run_item(item, k, "unrelated", CHANNEL_ARM,
                             "subject_independence", backend))
        # Permuted probe: naive arm sees reversed demos (C3 confound power).
        rows.append(run_item(item, k, "near", "naive", "permuted_probe",
                             backend, permuted=True))
        # P5 save + cold reload, then P6 held-out paraphrase on fresh engine.
        backend.save_reload(item)
        for condition in ("near", "far"):
            rows.append(run_item(item, k, condition, CHANNEL_ARM, "delayed",
                                 backend, teach=False))
    return rows


def _rate(rows):
    return round(sum(_outcome(r) for r in rows) / len(rows), 4) if rows else None


def _rows(rows, pool=None, arm=None, condition=None, phase="query", k=None,
          permuted=None):
    return [r for r in rows
            if (pool is None or r["pool"] == pool)
            and (arm is None or r["arm"] == arm)
            and (condition is None or r["condition"] == condition
                 or (isinstance(condition, (tuple, list))
                     and r["condition"] in condition))
            and r["phase"] == phase
            and (k is None or r["k"] == k)
            and (permuted is None or bool(r["permuted"]) == permuted)]


def summarize(items, rows):
    out = {}
    for pool in ("primary", "sealed"):
        p = {"by_k": {str(k): {} for k in K_LEVELS}}
        srows = _rows(rows, pool=pool, arm=CHANNEL_ARM)
        for cond in ("near", "far", "foil", "negative"):
            cr = [r for r in srows if r["condition"] == cond]
            p[cond + "_n"] = len(cr)
            p[cond] = _rate(cr)
            for k in K_LEVELS:
                p["by_k"][str(k)][cond] = _rate([r for r in cr if r["k"] == k])
        drops = []
        for cond in ("near", "far", "foil", "negative"):
            rates = [p["by_k"][str(k)][cond] for k in K_LEVELS]
            adjacent = [round(max(0.0, a - b), 4)
                        for a, b in zip(rates, rates[1:])
                        if a is not None and b is not None]
            drops.append(max(adjacent) if adjacent else 0.0)
        p["monotonic_max_drop"] = round(max(drops), 4)
        graded = []
        for k in K_LEVELS:
            b = p["by_k"][str(k)]
            graded.extend([round(max(0.0, (b["near"] or 0) - (b["far"] or 0)), 4),
                           round(max(0.0, (b["far"] or 0) - (b["foil"] or 0)), 4)])
        p["graded_max_drop"] = round(max(graded), 4)

        chance = _rows(rows, pool=pool, arm="chance", condition=("near", "far"))
        chance_rate = _rate(chance)
        p["chance"] = chance_rate
        p["far_minus_chance"] = (round((p["far"] or 0) - chance_rate, 4)
                                 if chance_rate is not None else None)
        p["negative_reject"] = p["negative"]
        p["counterexample_reject"] = _rate(
            _rows(rows, pool=pool, arm="counterexample", condition="negative"))
        p["foil_reject"] = p["foil"]
        p["role_coverage"] = round(
            sum(bool(r["flags"]["role_present"]) for r in srows) / len(srows), 4
        ) if srows else None

        ne = _rate(_rows(rows, pool=pool, arm="naive", condition="near"))
        nperm = _rate([r for r in rows if r["pool"] == pool and r["arm"] == "naive"
                       and r["phase"] == "permuted_probe"])
        sp = _rate(_rows(rows, pool=pool, arm="permuted", condition="near"))
        p["naive_near"] = ne
        p["naive_permuted"] = nperm
        p["permuted"] = sp
        p["naive_permuted_gap"] = (round(ne - nperm, 4)
                                   if None not in (ne, nperm) else None)
        p["channel_permuted_gap"] = (round(abs((p["near"] or 0) - sp), 4)
                                     if sp is not None else None)

        k8 = [r for r in srows if r["k"] == 8 and r["condition"] in ("near", "far")]
        p["transfer"] = _rate(k8)
        imm = [r for r in rows if r["pool"] == pool and r["arm"] == CHANNEL_ARM
               and r["phase"] == "immediate" and r["k"] == 8
               and r["condition"] in ("near", "far")]
        delayed = [r for r in rows if r["pool"] == pool and r["arm"] == CHANNEL_ARM
                   and r["phase"] == "delayed" and r["k"] == 8
                   and r["condition"] in ("near", "far")]
        p["immediate"] = _rate(imm)
        p["delayed"] = _rate(delayed)
        p["delayed_retention"] = (round((p["delayed"] or 0) / p["immediate"], 4)
                                  if p["immediate"] else None)
        base = _rows(rows, pool=pool, arm=CHANNEL_ARM, condition="near",
                     phase="baseline")
        p["baseline"] = _rate(base)
        p["growth"] = (round((p["immediate"] or 0) - (p["baseline"] or 0), 4)
                       if p["immediate"] is not None else None)
        # k=0 NULL CONTROL: eight matched neutral turns are indistinguishable
        # from a cold baseline on the ostensive arm AND on the deployable
        # channel arm (channel teaching now starts at k>=1). Comparing either
        # to baseline therefore scores a genuine null, not the benchmark's
        # own claim.
        k0 = _rows(rows, pool=pool, arm=OSTENSIVE_ARM, condition="near", k=0)
        k0_rate = _rate(k0)
        p["k0_minus_baseline"] = (round(abs(k0_rate - p["baseline"]), 4)
                                  if None not in (k0_rate, p["baseline"]) else None)
        cnear = _rows(rows, pool=pool, arm="chance", condition="near")
        p["chance_minus_baseline"] = (round(abs(_rate(cnear) - p["baseline"]), 4)
                                      if None not in (_rate(cnear), p["baseline"])
                                      else None)
        cex = _rows(rows, pool=pool, arm="counterexample", condition="near")
        p["counterexample_minus_baseline"] = (
            round(abs(_rate(cex) - p["baseline"]), 4)
            if None not in (_rate(cex), p["baseline"]) else None)
        p["control_max_delta"] = round(max(
            v for v in (p["k0_minus_baseline"], p["chance_minus_baseline"],
                        p["counterexample_minus_baseline"]) if v is not None), 4)
        # Positive control A (ostensive only): known to saturate at k>=1.
        p["ostensive_transfer"] = _rate(
            [r for r in _rows(rows, pool=pool, arm=OSTENSIVE_ARM)
             if r["k"] == 8 and r["condition"] in ("near", "far")])
        out[pool] = p
    out["engine_observation_missing"] = 0
    out["engine_response_unparsed"] = 0
    return out


# ── C10 / C12 helpers (pure; no engine import) ──────────────────────────────
def _slot_pick(views, subject, options, floor=0.20, margin=0.10):
    """Fail-closed slot scorer (mirrors production primitive; pure/stateless)."""
    opts = {str(o).lower(): o for o in options}
    scores = {}
    for v in views or ():
        if str(v.get("relation_type", "")).lower() != "is_a":
            continue
        slot = str(v.get("type_label", "")).strip().lower()
        if (not slot or slot not in opts
                or str(v.get("subject", subject)).lower() != str(subject).lower()
                or str(v.get("edge_kind", "")).lower() not in TRUSTED_KINDS
                or str(v.get("source_type", "")).lower() != "user"):
            continue
        scores[slot] = max(scores.get(slot, 0.0), float(v.get("confidence", 0.0)))
    eligible = {k: v for k, v in scores.items() if v >= floor}
    if not eligible:
        return None, "no_trusted_typed_slot_above_floor"
    ranked = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0]))
    gap = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    if gap < margin:
        return None, "strict_margin_not_met"
    return opts[ranked[0][0]], "trusted_typed_slot_strict_winner"


def type_contrast_probe():
    """C10: typed pick flips with the taught slot; disjoint options abstain."""
    typed = {"relation_type": "is_a", "type_label": "effect", "edge_kind": "curated",
             "source_type": "user", "confidence": 0.9, "subject": "thaw"}
    flipped = dict(typed, type_label="weather-event")
    options = ["effect", "weather-event"]
    pick_a, _ = _slot_pick([typed], "thaw", options)
    pick_b, _ = _slot_pick([flipped], "thaw", options)
    pick_none, reason = _slot_pick([typed], "thaw", ["apple", "banana"])
    return {"typed_pick": pick_a, "flipped_pick": pick_b,
            "disjoint_pick": pick_none, "disjoint_reason": reason,
            "ok": pick_a == "effect" and pick_b == "weather-event"
                  and pick_none is None}


def _provenance_ok(views):
    if not views:
        return False
    for v in views:
        if str(v.get("edge_kind", "")).lower() not in TRUSTED_KINDS:
            return False
        if str(v.get("source_type", "")).lower() != "user":
            return False
        if v.get("url") or v.get("fabricated_span"):
            return False
    return True


def _fixed_vector(condition, ans, gold_role, gold_target, should_reject=False):
    return _outcome(grade({"condition": condition, "output": ans,
                           "gold": {"role": gold_role, "target": gold_target,
                                    "should_reject": should_reject},
                           "flags": {}}))


# ── new controls: subject independence + channel independence ──────────────
def subject_independence_probe(subject, candidates, response_texts, item,
                               condition="unrelated"):
    """A confident answer for an UNRELATED subject is a HARD FAIL.

    Returns ``ok=False`` when the response would assign a candidate to the
    unrelated subject. Used both on the SI rows and as a discriminating
    self-check (feeding a copied graded answer must be detected).
    """
    ans = _parse_engine_answer(response_texts, item, condition, subject,
                               candidates)
    confident = bool(ans and ans.get("verdict") == "match"
                     and ans.get("target") is not None)
    return {"subject": subject, "answer": ans, "confident_match": confident,
            "ok": not confident}


def channel_independence_audit(items, rows):
    """Report channel/oracle overlap; HARD FAIL on any deployable use of oracle.

    Checks: (a) the item-level overlap audit (answer key / domain tag / missing
    provenance); (b) no graded row's engine input carries ``type_sets`` or
    ``world_model``; (c) the declared deployable field set excludes them;
    (d) P2c surfaces contain none of the oracle-only domain tags.
    """
    overlap = tra_k_items.channel_overlap(items)
    input_leaks = []
    for r in rows:
        inp = r.get("input", {})
        for field in ORACLE_ONLY_FIELDS:
            if field in inp:
                input_leaks.append({"id": r.get("id"), "field": field,
                                    "phase": r.get("phase")})
    static_leaks = [f for f in ORACLE_ONLY_FIELDS
                    if f in DEPLOYABLE_INPUT_FIELDS]
    tag_leaks = []
    for it in items:
        tags = {v for vals in it.get("type_sets", {}).values() for v in vals
                if v != it["type_label"]}
        if not tags:
            continue
        for surface in channel_teaching_surfaces(it):
            for tag in tags:
                if re.search(r"\b" + re.escape(tag) + r"\b", surface, re.I):
                    tag_leaks.append({"item_id": it["item_id"], "tag": tag})
                    break
    ok = (bool(overlap["ok"]) and not input_leaks and not static_leaks
          and not tag_leaks)
    return {"ok": ok, "overlap": overlap, "input_leaks": input_leaks[:20],
            "static_leaks": static_leaks, "tag_leaks": tag_leaks[:20],
            "deployable_fields": list(DEPLOYABLE_INPUT_FIELDS),
            "oracle_only_fields": list(ORACLE_ONLY_FIELDS)}


# ── instrument controls C1-C15 (spec section 3 + two new gates) ─────────────
def instrument_controls(items, rows, summary, backend, live, backend_error):
    checks = []

    def add(cid, name, ok, detail=""):
        checks.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail})

    p, s = summary["primary"], summary["sealed"]
    near8 = p["by_k"]["8"]["near"]

    add("C1", "channel near(k=8)>=0.90",
        near8 is not None and near8 >= TH["c1_near_min"], "near8=%s" % near8)

    sf, nf = p["foil"], _rate(
        _rows(rows, pool="primary", arm="naive", condition="foil"))
    add("C2", "foil channel reject>=0.90 and naive accept<=0.10",
        sf is not None and sf >= TH["c2_foil_min"]
        and nf is not None and nf <= TH["c2_naive_max"],
        "channel=%s naive_correct=%s" % (sf, nf))

    add("C3", "naive near-permuted>=0.15 and channel gap<=0.10",
        p["naive_permuted_gap"] is not None
        and p["naive_permuted_gap"] >= TH["permuted_hard_margin"]
        and p["channel_permuted_gap"] is not None
        and p["channel_permuted_gap"] <= TH["permuted_margin"],
        "naive_gap=%s channel_gap=%s" % (
            p["naive_permuted_gap"], p["channel_permuted_gap"]))

    add("C4", "chance<=0.35 and far-chance>=0.20",
        p["chance"] is not None and p["chance"] <= TH["chance_max"]
        and p["far_minus_chance"] is not None
        and p["far_minus_chance"] >= TH["far_chance_min"],
        "chance=%s far-chance=%s" % (p["chance"], p["far_minus_chance"]))

    cneg = p["counterexample_reject"]
    add("C5", "negative channel reject>=0.85 and counterexample<=0.20",
        p["negative_reject"] is not None
        and p["negative_reject"] >= TH["c5_struct_min"]
        and cneg is not None and cneg <= TH["c5_counter_max"],
        "channel=%s counterexample=%s" % (p["negative_reject"], cneg))

    add("C6", "monotonic k-curve max drop<=0.05 (both pools)",
        all(summary[x]["monotonic_max_drop"] <= TH["monotonic_tol"]
            for x in ("primary", "sealed")),
        "primary=%s sealed=%s" % (p["monotonic_max_drop"],
                                  s["monotonic_max_drop"]))

    add("C7", "graded near>=far>=foil max drop<=0.15 (both pools)",
        all(summary[x]["graded_max_drop"] <= TH["graded_drop_max"]
            for x in ("primary", "sealed")),
        "primary=%s sealed=%s" % (p["graded_max_drop"],
                                  s["graded_max_drop"]))

    good = _fixed_vector("near", {"verdict": "match", "role": "EFFECT",
                                  "target": "glow"}, "EFFECT", "glow")
    wrong_role = _fixed_vector("near", {"verdict": "match", "role": "WHOLE",
                                        "target": "glow"}, "EFFECT", "glow")
    wrong_target = _fixed_vector("near", {"verdict": "match", "role": "EFFECT",
                                          "target": "chill"}, "EFFECT", "glow")
    foil_acc = _fixed_vector("foil", {"verdict": "match", "role": "EFFECT",
                                      "target": "glow"}, "EFFECT", None, True)
    neg_acc = _fixed_vector("negative", {"verdict": "match", "role": "EFFECT",
                                         "target": "glow"}, "EFFECT", None, True)
    foil_rej = _fixed_vector("foil", {"verdict": "reject", "role": None,
                                      "target": None}, "EFFECT", None, True)
    neg_rej = _fixed_vector("negative", {"verdict": "reject", "role": None,
                                         "target": None}, "EFFECT", None, True)
    cov = all(summary[x]["role_coverage"] is not None
              and summary[x]["role_coverage"] >= TH["c8_coverage_min"]
              for x in ("primary", "sealed"))
    add("C8", "role coverage>=0.95 + grader fixed vectors",
        cov and good and not wrong_role and not wrong_target
        and not foil_acc and not neg_acc and foil_rej and neg_rej,
        "coverage=%s/%s fixed_vectors=%s" % (
            p["role_coverage"], s["role_coverage"],
            all((good, not wrong_role, not wrong_target, not foil_acc,
                 not neg_acc, foil_rej, neg_rej))))

    delayed_rows = [r for r in rows if r["phase"] == "delayed"]
    durable_ok = all(
        summary[x]["delayed_retention"] is not None
        and summary[x]["delayed_retention"] >= TH["retention_min"]
        for x in ("primary", "sealed"))
    reloaded_ok = bool(delayed_rows) and all(
        r["reloaded"] >= 1 and r["slot_read_fresh"] for r in delayed_rows)
    calls = backend.api_calls
    counters_ok = calls.get("save", 0) >= 1 and calls.get("cold_reload", 0) >= 1
    add("C9", "cold-reload retention>=0.80, fresh engine, counters>=1",
        durable_ok and reloaded_ok and counters_ok,
        "retention=%s/%s save=%s reload=%s fresh_rows=%d" % (
            p["delayed_retention"], s["delayed_retention"],
            calls.get("save"), calls.get("cold_reload"), len(delayed_rows)))

    probe = type_contrast_probe()
    add("C10", "type contrast flips pick; disjoint returns None", probe["ok"],
        json.dumps(probe, sort_keys=True))

    graded_from_query = bool(rows) and all(
        r["observation_source"] == "query_response"
        and r["engine_graded"]
        and r["engine_answer"] == r["output"]
        for r in rows if r["phase"] == "query")
    c11 = (backend.label == "live" and bool(live) and backend_error is None
           and all(calls.get(key, 0) >= 1 for key in
                   ("boot", "intake", "save", "cold_reload", "process_turn"))
           and backend.engine_observation_missing == 0
           and backend.synthetic_fallback_used == 0
           and graded_from_query)
    add("C11", "anti-fake: live backend, real API counters, query-only grading",
        c11, "backend=%s live=%s error=%r graded_from_query=%s calls=%s" % (
            backend.label, live, backend_error, graded_from_query, calls))

    views = backend.provenance_views()
    add("C12", "provenance integrity; no fabricated URL/span; gate intact",
        _provenance_ok(views)
        and all(not r["trust_gate_disabled"] for r in rows),
        "views=%d fabricated=%s" % (
            len(views), any(v.get("fabricated_span") for v in views)))

    growth_ok = p["growth"] is not None and p["growth"] >= TH["growth_min"]
    coverage_loss = round((s["role_coverage"] or 0) - (p["role_coverage"] or 0), 4)
    add("C13", "immediate(k=8)-baseline>=0.15 and coverage loss<=0.05",
        growth_ok and abs(coverage_loss) <= TH["control_delta_max"],
        "growth=%s coverage_loss=%s" % (p["growth"], coverage_loss))

    # C14 subject independence (NEW, HARD FAIL).
    si_rows = [r for r in rows if r["phase"] == "subject_independence"]
    si_ok = bool(si_rows) and all(r["flags"]["correct"] for r in si_rows)
    ref = next(it for it in items if it["pool"] == "primary")
    q = query_for(ref, "near")
    copied = ["role=%s target=%s verdict=match" % (ref["role"], q["gold"])]
    bad_probe = subject_independence_probe(
        UNRELATED_SUBJECT, q["candidates"], copied, ref)
    declined = subject_independence_probe(
        UNRELATED_SUBJECT, q["candidates"],
        ["I'm not totally sure about that one."], ref)
    add("C14", "unrelated subject never gets confident answer (HARD FAIL)",
        si_ok and declined["ok"] and not bad_probe["ok"],
        "si_rows=%d si_ok=%s decline_ok=%s copy_detected=%s" % (
            len(si_rows), si_ok, declined["ok"], not bad_probe["ok"]))

    # C15 channel independence (NEW, HARD FAIL).
    audit = channel_independence_audit(items, rows)
    add("C15", "channel evidence overlaps oracle only benignly; no deployable use",
        audit["ok"], "overlap=%s leaks=%d tags=%d" % (
            audit["overlap"]["totals"], len(audit["input_leaks"]),
            len(audit["tag_leaks"])))
    return checks


# ── saturation mutation controls (scratch only) ─────────────────────────────
# Scratch sentinel used by M-A to re-stamp a channel type. It is NOT a gate or
# a threshold; it only labels the mutation so the regrade hook can detect it.
MUTATION_SENTINEL = "not-a-taught-type"


def _demo_objects(item):
    return {w for ex in item["taught_exemplars"] for w in (ex["r1"], ex["r2"])}


def _channel_verdict(item):
    """Scratch regrade hook: does the CHANNEL still carry a usable is_a type?

    An external substrate channel carries SOURCE-NATIVE type labels (Wikidata
    P31/P279 labels); the substrate audit records 0/638 overlap with TRA's six
    authored role labels. Requiring ``object == item["type_label"]`` would
    therefore answer "reject" BEFORE any mutation and disconnect this
    saturation control from the channel entirely -- a probe bug, not a
    benchmark result. The substrate-aware regrade asks the channel-neutral
    question: is there still a provenance-stamped is_a fact naming a real type
    on a taught exemplar? M-A re-stamps the type to ``MUTATION_SENTINEL`` and
    M-B removes the facts, so both must still move the verdict. The fallback
    channel's role-label facts satisfy the same rule, so the original control
    is preserved unchanged in behaviour for that channel.
    """
    demos = _demo_objects(item)
    supported = any(
        str(f.get("predicate", "")).lower() in tra_k_items.ISA_PREDICATES
        and str(f.get("object", "")).strip()
        and str(f.get("object", "")).strip() != MUTATION_SENTINEL
        and str(f.get("subject", "")).strip() in demos
        and all(k in f and f.get(k) not in (None, "")
                for k in tra_k_items.REQUIRED_FACT_KEYS)
        for f in item.get("channel_evidence", []))
    return "match" if supported else "reject"


def mutation_control(items, bank_mutator=None):
    """M-A / M-B on deep scratch copies. Both must move the graded verdict.

    M-A re-stamps the channel fact types; M-B removes the channel evidence.
    ``bank_mutator`` is the operator's live-store hook ``(item, kind) -> item``.
    """
    item = next(it for it in items if it["pool"] == "primary" and it["k"] == 8)
    before = _channel_verdict(item)
    ma_item = copy.deepcopy(item)
    for fact in ma_item["channel_evidence"]:
        fact["object"] = MUTATION_SENTINEL
    mb_item = copy.deepcopy(item)
    mb_item["channel_evidence"] = []
    if bank_mutator is not None:
        ma_item = bank_mutator(copy.deepcopy(item), "M-A")
        mb_item = bank_mutator(copy.deepcopy(item), "M-B")
    after_a = _channel_verdict(ma_item)
    after_b = _channel_verdict(mb_item)
    return {"M-A": {"before": before, "after": after_a,
                    "moved": before != after_a},
            "M-B": {"before": before, "after": after_b,
                    "moved": before != after_b},
            "moved": before != after_a and before != after_b}


# ── decision rules (frozen, both pools) ─────────────────────────────────────
def decide(summary, checks, backend, live, backend_error, sat_status,
           mutations_moved):
    rules = {}
    p, s = summary["primary"], summary["sealed"]
    for pool, d in (("primary", p), ("sealed", s)):
        pre = pool + "_"
        rules[pre + "transfer>=0.75"] = (d["transfer"] or 0) >= TH["transfer_min"]
        rules[pre + "near>=0.75"] = (d["near"] or 0) >= TH["near_min"]
        rules[pre + "far>=0.70"] = (d["far"] or 0) >= TH["far_min"]
        rules[pre + "far-chance>=0.20"] = (
            d["far_minus_chance"] is not None
            and d["far_minus_chance"] >= TH["far_chance_min"])
        rules[pre + "foil_reject>=0.85"] = (
            d["foil_reject"] is not None
            and d["foil_reject"] >= TH["foil_reject_min"])
        rules[pre + "negative_reject>=0.80"] = (
            d["negative_reject"] is not None
            and d["negative_reject"] >= TH["negative_reject_min"])
        rules[pre + "permuted>=near-0.10"] = (
            d["permuted"] is not None
            and d["permuted"] >= (d["near"] or 0) - TH["permuted_margin"])
        rules[pre + "monotonic<=0.05"] = (
            d["monotonic_max_drop"] <= TH["monotonic_tol"])
        rules[pre + "graded<=0.15"] = (
            d["graded_max_drop"] <= TH["graded_drop_max"])
        rules[pre + "coverage>=0.95"] = (
            d["role_coverage"] or 0) >= TH["coverage_min"]
        rules[pre + "retention>=0.80"] = (
            d["delayed_retention"] is not None
            and d["delayed_retention"] >= TH["retention_min"])
        rules[pre + "control_delta<=0.05"] = (
            d["control_max_delta"] <= TH["control_delta_max"])
    for metric in ("transfer", "near", "foil_reject"):
        rules["sealed_%s within 0.10" % metric] = (
            abs((s[metric] or 0) - (p[metric] or 0)) <= TH["sealed_drop_max"])
    by_id = {c["id"]: c["ok"] for c in checks}
    rules["instrument_controls_ok"] = all(
        c["ok"] for c in checks if c["id"] != "C11")
    rules["c11_anti_fake_ok"] = by_id.get("C11", False)
    rules["subject_independence_ok"] = by_id.get("C14", False)
    rules["channel_independence_ok"] = by_id.get("C15", False)
    rules["backend_live"] = backend.label == "live" and bool(live)
    rules["backend_error_free"] = backend_error is None
    rules["saturation_flag_recorded"] = sat_status in ("PASS", "FLAG", "FAIL", "INVALID")
    rules["mutations_passed_if_saturated"] = (
        bool(mutations_moved) if sat_status == "FLAG" else True)
    return rules, all(rules.values())


def _wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def statistics(items, rows, summary):
    out = {}
    for pool in ("primary", "sealed"):
        ex = _rows(rows, pool=pool, arm=CHANNEL_ARM, condition="near")
        ch = _rows(rows, pool=pool, arm="chance", condition="near")
        a = sum(r["flags"]["exact_structural_match"] for r in ex)
        c = sum(r["flags"]["exact_structural_match"] for r in ch)
        out[pool] = {
            "near_correct": "%d/%d" % (a, len(ex)),
            "near_wilson95": _wilson(a, len(ex)),
            "chance_correct": "%d/%d" % (c, len(ch)),
            "transfer": summary[pool]["transfer"],
            "growth": summary[pool]["growth"],
        }
    return out


# ── MAIN ────────────────────────────────────────────────────────────────────
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    live = "--live" in argv
    started = time.time()
    commit = os.popen('git -C "%s" rev-parse --short HEAD' % PROJ).read().strip()

    # Build the frozen bank once (fallback evidence), then load external
    # channel facts against the real item ids, then rebuild with the override.
    base_items = tra_k_items.build_items()
    channel_loaded = tra_k_items.load_channel_evidence(base_items)
    items = tra_k_items.build_items(channel_override=channel_loaded["facts"])
    all_items = list(items)
    errors = tra_k_items.validate_items(items)
    manifests = tra_k_items.manifest_hashes(items)
    bank_hash = tra_k_items.item_manifest_hash(items)
    pools = collections.Counter(it["pool"] for it in items)
    kc = tra_k_items.pool_k_counts(items)
    overlap = tra_k_items.channel_overlap(items)
    channel_info = {
        "env": tra_k_items.CHANNEL_ENV, "path": channel_loaded["path"],
        "error": channel_loaded["error"], "invalid_facts": channel_loaded["invalid"],
        "external_items": channel_loaded["external_items"],
        "fallback_items": len(items) - channel_loaded["external_items"],
        "overlap": overlap["totals"],
    }
    # Deterministic bounded subset for focused real-engine probes. Select
    # round-robin across k levels so every k (including 8) stays represented
    # and the k-curve / growth controls remain informative. The FULL bank is
    # still validated and its manifest hashes reported above; a subset result
    # is a mechanism probe, never a full-benchmark verdict.
    max_per_pool = int(os.environ.get("TRAK_MAX_PER_POOL", "0") or 0)
    if max_per_pool > 0:
        keep = []
        for pool in ("primary", "sealed"):
            by_k = {k: [it for it in items
                        if it["pool"] == pool and it["k"] == k]
                    for k in K_LEVELS}
            for idx in range(max_per_pool):
                k = K_LEVELS[idx % len(K_LEVELS)]
                if by_k[k]:
                    keep.append(by_k[k].pop(0))
        items = keep
        pools = collections.Counter(it["pool"] for it in items)
        print("  FOCUSED SUBSET max_per_pool=%d selected=%d "
              "(mechanism probe, NOT a full-benchmark verdict)" %
              (max_per_pool, len(items)))

    print("=" * 78)
    print("TRA-K benchmark (pre-registered) -- backend=%s"
          % ("live" if live else "fake"))
    print("=" * 78)
    print("  commit=%s families=%d primary=%d sealed=%d" % (
        commit, len(items), pools["primary"], pools["sealed"]))
    print("  hash_rule=%s|TRA-K|<difficulty>|<item_id>" % HASH_PREFIX)
    print("  manifest_sha256 primary=%s" % manifests["primary"])
    print("  manifest_sha256 sealed =%s" % manifests["sealed"])
    print("  bank_sha256=%s" % bank_hash)
    print("  k_levels=%s one_k_per_item=%s" % (K_LEVELS,
          all(kc.get((pool, k)) == 40 for pool in ("primary", "sealed")
              for k in K_LEVELS)))
    print("  channel path=%s external=%d fallback=%d overlap=%s" % (
        channel_info["path"], channel_info["external_items"],
        channel_info["fallback_items"], overlap["totals"]))
    if channel_info["error"]:
        print("  channel load error (degraded): %s" % channel_info["error"])
    if errors:
        print("\nINSTRUMENT INVALID -- item definitions broken:")
        for e in errors[:20]:
            print("  ", e)
        return 2
    if not overlap["ok"]:
        print("\nINSTRUMENT INVALID -- channel evidence contaminated: %s"
              % overlap["offenders"][:10])
        return 2
    if "--validate-only" in argv:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
            json.dump({"hash_prefix": HASH_PREFIX, "component": COMPONENT,
                       "families": len(items), "pools": dict(pools),
                       "pool_k_counts": {str(k): v for k, v in kc.items()},
                       "manifest_sha256": manifests, "bank_sha256": bank_hash,
                       "thresholds": TH, "k_levels": list(K_LEVELS),
                       "conditions": list(CONDITIONS), "arms": list(ARMS),
                       "channel": channel_info},
                      mf, indent=2, ensure_ascii=False, sort_keys=True)
        print("\nvalidate-only: all %d families valid; manifest frozen -> %s"
              % (len(items), MANIFEST_PATH))
        return 0

    flag_ctx = trak_slots_flag() if live else contextlib.nullcontext()
    with flag_ctx:
        backend = RealBackend() if live else FakeBackend()
        rows = run_sweep(items, backend=backend)
        summary = summarize(items, rows)
        if max_per_pool > 0:
            summary["focused_subset"] = {
                "max_per_pool": max_per_pool, "selected": len(items),
                "full_bank_verdict": False,
            }
        summary["save"] = backend.api_calls.get("save", 0)
        summary["cold_reload"] = backend.api_calls.get("cold_reload", 0)
        summary["engine_observation_missing"] = backend.engine_observation_missing
        summary["engine_response_unparsed"] = backend.engine_response_unparsed
        stats = statistics(items, rows, summary)
        backend_error = backend.api_error
        if live:
            cert_backend = FakeBackend()
            cert_rows = run_sweep(all_items, backend=cert_backend)
            cert_summary = summarize(all_items, cert_rows)
            cert_checks = instrument_controls(all_items, cert_rows, cert_summary,
                                              cert_backend, False, None)
            live_checks = instrument_controls(items, rows, summary, backend,
                                              live, backend_error)
            checks = ([c for c in cert_checks if c["id"] not in ("C11", "C12")]
                      + [c for c in live_checks if c["id"] in ("C11", "C12")])
        else:
            checks = instrument_controls(items, rows, summary, backend, live,
                                         backend_error)
        invalid = [c for c in checks if c["id"] != "C11" and not c["ok"]]

    print("\n--- instrument controls C1-C15 (before any score) ---")
    for c in checks:
        print("  [%s] %s -- %s" % ("ok" if c["ok"] else "BAD", c["id"],
                                  c["name"]))
        if not c["ok"]:
            print("        %s" % c["detail"][:240])
    if invalid:
        print("\nINSTRUMENT INVALID -- no score (failed: %s)."
              % ", ".join(c["id"] for c in invalid))
        return 2

    sat = front_gates.gate_saturation(
        {"tra_k_near": round(100 * (summary["primary"]["near"] or 0)),
         "tra_k_negative_reject": round(100 * (
             summary["primary"]["negative_reject"] or 0))},
        None, "tra_k")
    with open(GATE_PATH, "a", encoding="utf-8") as gf:
        gf.write(json.dumps(sat.as_dict("tra_k"), ensure_ascii=False) + "\n")

    mutations = mutation_control(items)
    if sat.status == "FLAG" and not mutations["moved"]:
        print("\nINSTRUMENT INVALID -- saturated boundary with disconnected "
              "mutation control; do not report these numbers.")
        return 2

    rules, passed = decide(summary, checks, backend, live, backend_error,
                           sat.status, mutations["moved"])

    with open(DATA_PATH, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    interpretation = (
        "capability_pass" if passed else
        "mechanism_probe_only_fake_arm" if not live else "genuine_fail")

    print("\n--- decision rules (pre-registered, not tuned) ---")
    for name, ok in rules.items():
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))
    print("\n--- PRIMARY vs SEALED (channel arm) ---")
    for pool in ("primary", "sealed"):
        d = summary[pool]
        print("  %-7s transfer=%.3f near=%.3f far=%.3f foil=%.3f neg=%.3f "
              "chance=%.3f perm=%.3f mono=%.3f graded=%.3f cov=%.3f "
              "ret=%.3f growth=%+.3f ostensive=%s" % (
                  pool, d["transfer"] or 0, d["near"] or 0, d["far"] or 0,
                  d["foil"] or 0, d["negative_reject"] or 0, d["chance"] or 0,
                  d["permuted"] or 0, d["monotonic_max_drop"],
                  d["graded_max_drop"], d["role_coverage"] or 0,
                  d["delayed_retention"] or 0, d["growth"] or 0,
                  d["ostensive_transfer"]))
    print("  saturation gate: %s -- %s" % (sat.status, sat.detail))
    print("  mutation control M-A/M-B moved: %s" % mutations["moved"])
    print("  interpretation: %s" % interpretation)
    if not live:
        print("  NOTE: fake backend cannot yield a capability claim (C11 BAD "
              "by construction).")

    out = {
        "component": "TRA-K", "commit": commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hash_rule": "%s|TRA-K|<difficulty>|<item_id> (sha256)" % HASH_PREFIX,
        "manifest_sha256": manifests, "bank_sha256": bank_hash,
        "families": len(items), "pool_sizes": dict(pools),
        "pool_k_counts": {str(k): v for k, v in kc.items()},
        "k_levels": list(K_LEVELS), "conditions": list(CONDITIONS),
        "arms": list(ARMS), "thresholds": TH, "rows": len(rows),
        "channel": channel_info,
        "channel_independence": channel_independence_audit(all_items, rows),
        "instrument_controls": checks,
        "instrument_controls_ok": all(c["ok"] for c in checks),
        "invalid_controls": [c["id"] for c in invalid],
        "summary": summary, "statistics": stats,
        "rules": {k: bool(v) for k, v in rules.items()},
        "saturation_gate": {"status": sat.status, "detail": sat.detail},
        "mutation_control": mutations,
        "pass": bool(passed), "interpretation": interpretation,
        "backend": backend.label, "live": live,
        "api_calls": backend.api_calls, "backend_error": backend_error,
        "engine_observation_missing": backend.engine_observation_missing,
        "engine_response_unparsed": backend.engine_response_unparsed,
        "slot_flag_env": SLOT_ENV_FLAG, "trak_flag_env": TRAK_ENV_FLAG,
        "data_path": DATA_PATH, "elapsed_s": round(time.time() - started, 1),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        json.dump(out, sf, indent=2, ensure_ascii=False)

    print("\nOVERALL: %s (%s)" % ("PASS" if passed else "FAIL", interpretation))
    print("artifacts: %s , %s" % (DATA_PATH, SUMMARY_PATH))
    print("elapsed %.1fs | DONE" % (time.time() - started))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
