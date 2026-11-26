#!/usr/bin/env python3
"""TRA-K item bank -- frozen, deterministic, hash-split primary/sealed pools.

Pre-registered by ``docs/investigations/TRA_K_SPEC.md`` (origin/docs/tra-k-spec).
This is a NEW benchmark, not a modification of TRA. STATIC ONLY: building the
bank boots no engine, writes no store, makes no network call and patches no
process global at import time.

Contract implemented here:

  * HASH_PREFIX = "RAVANA-TRA-K-v1", COMPONENT = "TRA-K" (collision-free with
    the LEARN-5 / TRA v1 pools). Split within difficulty by
    sha256(f"{HASH_PREFIX}|TRA-K|<difficulty>|<item_id>"), sort by hash bytes,
    first half PRIMARY.
  * 400 families: 200 primary + 200 sealed, 5 k levels, exactly 40 families
    per k per pool; one k per item.
  * Item schema is TRA's plus exactly ONE new field: ``channel_evidence``.
    It is a provenance-stamped list of facts supplied by an INDEPENDENT
    semantic channel (external ontology B in the spec). This module never
    derives it from ``type_sets`` / ``world_model`` / the gold target / the
    query subject's type.
  * ``world_model`` and ``type_sets`` are ORACLE-ONLY construction aids; they
    are never emitted by ``teaching_surface()`` or ``channel_assertion()`` and
    are never handed to the graded engine path by the harness.

CHANNEL INPUT IS PLUGGABLE AND CHANNEL-AGNOSTIC
-----------------------------------------------
The ontology agent's snapshot is not assumed to exist. ``channel_evidence``
is a generic list of provenance-stamped facts, optionally loaded from the JSON
path in ``TRAK_CHANNEL_JSON`` via :func:`load_channel_evidence`. Accepted JSON
shapes (all flat and human-authorable):

    {"items": {"tra_k_0000": [{"subject": ..., "predicate": "is_a",
                                "object": ..., "source": ..., ...}]}}
    {"tra_k_0000": [fact, ...]}
    [{"item_id": "tra_k_0000", "facts": [fact, ...]}, ...]
    [{"item_id": "tra_k_0000", "subject": ..., "predicate": "is_a",
      "object": ..., "source": ...}, ...]

A fact needs ``subject``, ``predicate``, ``object`` and ``source``; missing
provenance is rejected. If no external fact is available for an item, a
DETERMINISTIC fallback is constructed from that item's taught exemplar objects
and its domain-general ``type_label`` only, stamped
``source="TRA-K-fallback-deterministic"``. The harness reports the external /
fallback split and hard-fails if oracle fields contaminate the channel.

TYPED ANSWER OPTIONS
--------------------
Each item also carries ``typed_queries``: five forced-choice probes whose four
options are SOURCE-NATIVE TYPE LABELS from the channel (one correct label in
that item's own facts, three deterministic foils from other channel facts,
same domain first). The original ``unseen_test_words`` domain-word candidates
are retained only as the oracle-side record; the graded query path reads
``typed_queries``. No option or gold label is drawn from ``type_sets``,
``world_model`` or any domain-word gold target.

SUBJECT-TYPE CHANNEL INPUT (this fix)
-------------------------------------
The benchmark declares a supplied knowledge channel. A channel that omits the
queried subject's own type cannot support the query at all (a benchmark INPUT
defect, not an engine defect). Two changes are made here, both STATIC:

1. Query-subject roles (``near_subject``, ``far_subject``, ``neg_subject``) are
   drawn only from words the substrate actually covers. The four snapshot gaps
   (``sleet``/``chill``/``marrow``/``thorn``) may remain distractors or demo
   words but are never query subjects. Coverage is read from the checked-in
   substrate snapshot, never from oracle fields.
2. Every item carries one ADMITTED substrate fact for each query subject
   (near/far/negative), projected verbatim from the snapshot with full CC0
   provenance. Low-credibility conflicting facts (alternate sense / weaker
   evidence tier, or a different substrate entry) are also admitted, so the
   right label is not the only type present. The graded option label is
   selected by the source-provenance rule below, never by "only type left".

ADMISSION RULE (source provenance, not authority)
-------------------------------------------------
A channel fact is admitted iff it (a) has all REQUIRED_FACT_KEYS, (b) is an
``is_a``-predicate, (c) names either a taught demo object or a declared query
subject, and (d) when it names a query subject, it carries the full substrate
provenance field set (source/source_id/snapshot/license/license_url/source_url/
span/subject_qid/type_qid/wikidata_property/sense). Gold targets and other
answer-key tokens remain forbidden subjects. The admitted subject fact with the
best evidence tier (primary sense > alternate; item description > type
description > label) is the POSITIVE subject-type assertion; the rest are
deterministic misleading facts. ``type_sets``/``world_model``/gold are never
read to build, rank or admit a fact.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os

HASH_PREFIX = "RAVANA-TRA-K-v1"
COMPONENT = "TRA-K"
K_LEVELS = (0, 1, 2, 4, 8)
CONDITIONS = ("near", "far", "foil", "negative", "unrelated")
DIFFICULTIES = ("easy", "medium", "hard", "expert")
CHANNEL_ENV = "TRAK_CHANNEL_JSON"
FALLBACK_SOURCE = "TRA-K-fallback-deterministic"
REQUIRED_FACT_KEYS = ("subject", "predicate", "object", "source")
ISA_PREDICATES = ("is_a", "type_of", "instance_of")

# The substrate snapshot is the ONLY source of subject-type assertions.
# Git stores this file with LF endings; core.autocrlf can materialise CRLF in a
# Windows worktree. The content digest MUST therefore be newline-normalised
# (see substrate_sha256) or every frozen provenance string and manifest hash
# changes with the checkout platform.
SUBSTRATE_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scratch", "trak_substrate.json")
SUBSTRATE_SCHEMA = "ravana-trak-semantic-substrate-v1"
SUBSTRATE_CHANNEL = "B-wikidata"
SUBSTRATE_PROVENANCE_FIELDS = (
    "license", "license_url", "source_url", "span", "text_evidence",
    "text_evidence_kind", "wikidata_property", "wikidata_relation",
    "subject_qid", "type_qid", "subject_description", "type_description",
    "enwiki_title", "sense", "sense_selection",
)
# A subject-naming fact is admitted ONLY with the full substrate trail.
SUBSTRATE_PROVENANCE_KEYS = (
    "source", "source_id", "snapshot", "license", "license_url",
    "source_url", "span", "subject_qid", "type_qid", "wikidata_property",
    "sense",
)
SUBJECT_TYPE_ROLE = "subject_type"        # the positive, best-evidence assertion
MISLEADING_ROLE = "misleading"            # deterministic low-credibility conflict
_SUBSTRATE_CACHE = {}

ROLES = {
    "EFFECT":   {"type_label": "effect",            "phrase": "causes"},
    "WHOLE":    {"type_label": "structured-whole",   "phrase": "is part of"},
    "PRECEDES": {"type_label": "earlier-event",      "phrase": "is preceded by"},
    "SIMILAR":  {"type_label": "similar-form",       "phrase": "resembles"},
    "SUPPORTS": {"type_label": "supported-outcome",  "phrase": "supports"},
    "CONTRAST": {"type_label": "contrast-case",      "phrase": "contrasts with"},
}
ROLE_KEYS = tuple(ROLES)
ROLE_TO_TYPE = {r: s["type_label"] for r, s in ROLES.items()}
ROLE_TO_PHRASE = {r: s["phrase"] for r, s in ROLES.items()}

# Verbatim from scripts/measure_learn5_rsi.py:137-169 (disjoint 24-word pools).
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
            raise RuntimeError("TRA-K domain pools overlap: %s / %s" % (_da, _db))

N_FAMILIES = 200 * 2
UNRELATED_SUBJECT = "zzq_unrelated_wumpus"   # never in any OBJECTS pool


# ── deterministic helpers (no RNG, no hash() of Python objects) ─────────────
def _digest(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _hash_pick(seed, pool, n, exclude=()):
    """Pick n distinct entries from ``pool`` by stable sha256 ordering."""
    out, seen = [], set(exclude)
    for word in sorted(pool, key=lambda w: _digest(seed, w)):
        if word in seen:
            continue
        seen.add(word)
        out.append(word)
        if len(out) == n:
            break
    if len(out) != n:
        raise ValueError("hash_pick: wanted %d distinct from %d (%s)"
                         % (n, len(pool), seed))
    return out


def item_hash(item):
    """Spec section 4: sha256(f'{HASH_PREFIX}|TRA-K|<difficulty>|<item_id>')."""
    return _digest(HASH_PREFIX, COMPONENT, item["difficulty"], item["item_id"])


def _canonical_records(items, pool):
    return [dict(it) for it in sorted((x for x in items if x["pool"] == pool),
                                      key=lambda x: x["item_id"])]


def manifest_hashes(items):
    """Per-pool manifest_sha256 over canonical JSON of the sealed records."""
    out = {}
    for pool in ("primary", "sealed"):
        blob = json.dumps(_canonical_records(items, pool), sort_keys=True,
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        out[pool] = hashlib.sha256(blob).hexdigest()
    return out


def item_manifest_hash(items):
    """Single whole-bank hash (all pools) for the summary/unit pin."""
    blob = json.dumps(
        [{"item_id": it["item_id"], "difficulty": it["difficulty"],
          "pool": it["pool"], "role": it["role"], "k": it["k"],
          "hash": item_hash(it)}
         for it in sorted(items, key=lambda x: x["item_id"])],
        sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _split_words(pool, seed, subject_pool=None):
    """Deterministically carve a 24-word pool into query + distractor + demo.

    Query subject and gold are reserved first so they are NEVER in the taught
    exemplar object set (TRA validation rule 3, kept unchanged).

    ``subject_pool`` restricts ONLY the query-subject roles (near, negative)
    to substrate-covered words. Gold, distractors and demo words still come
    from the full pool, so 24-word pools still yield 17 demo words and the
    400-family / 40-per-k split is unchanged.
    """
    subject_pool = pool if subject_pool is None else subject_pool
    if len(subject_pool) < 4:
        raise ValueError("insufficient covered query subjects for %s" % seed)
    near_gold = _hash_pick(seed + "|NG", pool, 1)[0]
    near_subject = _hash_pick(seed + "|NS", subject_pool, 1, {near_gold})[0]
    neg_target = _hash_pick(seed + "|NT", pool, 1,
                            {near_gold, near_subject})[0]
    neg_subject = _hash_pick(seed + "|NSub", subject_pool, 1,
                             {near_gold, near_subject, neg_target})[0]
    reserved = {near_gold, near_subject, neg_target, neg_subject}
    remaining = [w for w in pool if w not in reserved]
    same_len = [w for w in remaining if len(w) == len(near_gold)]
    other_len = [w for w in remaining if len(w) != len(near_gold)]
    distractors = _hash_pick(seed + "|D3", same_len, min(3, len(same_len)))
    if len(distractors) < 3:
        distractors += _hash_pick(seed + "|D3b", other_len,
                                  3 - len(distractors),
                                  exclude=set(distractors))
    used = reserved | set(distractors)
    demo = [w for w in pool if w not in used]
    if len(demo) < 17:
        raise ValueError("insufficient demo words for %s" % seed)
    return {"near_gold": near_gold, "near_subject": near_subject,
            "neg_target": neg_target, "neg_subject": neg_subject,
            "distractors": distractors, "demo": demo}


def teaching_surface(role, r1, r2):
    """Provenance-safe P2 teaching text (unchanged from TRA)."""
    phrase = ROLE_TO_PHRASE[role]
    label = ROLE_TO_TYPE[role]
    article = "an" if label[0] in "aeiou" else "a"
    return "%s %s %s. %s is %s %s." % (r1, phrase, r2, r2, article, label)


# ── channel evidence: generic, provenance-stamped, pluggable ────────────────
def _fact(*, subject, predicate, object_, source, channel="B",
          source_id=None, snapshot="frozen", **extra):
    """Build one canonical fact with a stable key order."""
    out = {"subject": str(subject), "predicate": str(predicate),
           "object": str(object_), "channel": str(channel),
           "source": str(source), "source_id": str(source_id or _digest(source,
                                                                        subject, object_)[:16]),
           "snapshot": str(snapshot)}
    out.update(extra)
    return out


# ── substrate projection + admission rule (declared TRA-K channel input) ─────
def substrate_sha256(path):
    """SHA-256 of the substrate's CHECKED-IN CONTENT (LF-normalised bytes).

    The committed blob is LF. A Windows checkout with ``core.autocrlf=true``
    materialises the same content with CRLF, and hashing those raw bytes made
    the frozen ``snapshot`` provenance -- and therefore every pinned manifest
    hash -- platform-dependent. Normalise line endings first so the digest
    names the committed snapshot, not the local checkout's newline style.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    return hashlib.sha256(
        raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def load_substrate(path=None):
    """Load the checked-in Wikidata snapshot (cached, static, no network)."""
    path = path or SUBSTRATE_PATH
    key = os.path.realpath(path)
    if key not in _SUBSTRATE_CACHE:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema") != SUBSTRATE_SCHEMA:
            raise ValueError("unexpected substrate schema: %r" % data.get("schema"))
        data["digest"] = substrate_sha256(path)
        data["by_subject"] = collections.defaultdict(list)
        for raw in data["facts"]:
            data["by_subject"][str(raw["subject"]).strip().lower()].append(raw)
        # Cached low-credibility-first order for deterministic conflict borrow.
        data["low_first"] = sorted(data["facts"], key=_evidence_rank)
        _SUBSTRATE_CACHE[key] = data
    return _SUBSTRATE_CACHE[key]


def covered_subjects(substrate=None):
    """Words with a typed fact in the substrate snapshot (query-subject pool)."""
    substrate = substrate or load_substrate()
    return {s for s, rows in substrate["by_subject"].items() if rows}


def substrate_fact(raw, snapshot, item_id, role):
    """Project one substrate record verbatim into a channel fact.

    No field is invented: subject/type label/source/QIDs/span/sense come from
    the snapshot. ``evidence_role`` only labels SELECTION (positive vs
    misleading); it never changes the source-native assertion.
    """
    out = {
        "subject": raw["subject"], "predicate": "is_a",
        "object": raw["type_label"], "channel": SUBSTRATE_CHANNEL,
        "source": raw["source_url"],
        "source_id": _digest(raw.get("source_url"), raw.get("subject_qid"),
                             raw.get("type_qid"), raw.get("type_label"),
                             raw.get("wikidata_property"), raw.get("sense"))[:16],
        "snapshot": snapshot,
    }
    for key in SUBSTRATE_PROVENANCE_FIELDS:
        out[key] = raw.get(key, "")
    out["item_id"] = item_id
    out["evidence_role"] = role
    return out


def _evidence_rank(raw):
    """Deterministic credibility: primary sense, then evidence tier."""
    sense = 2 if raw.get("sense") == "primary" else 0
    kind = {"item_description": 3, "type_description": 2,
            "label_only": 1}.get(raw.get("text_evidence_kind"), 0)
    return (sense, kind, _digest(str(raw.get("type_label")),
                                 str(raw.get("type_qid"))))


def subject_channel_facts(item, substrate=None):
    """(admitted subject-type fact, misleading facts) for each query subject.

    The positive assertion is the highest-credibility substrate record for the
    subject; a conflicting lower-credibility record is admitted as a
    misleading fact (or, for a single-label subject, one source-traceable
    conflicting assertion from another substrate entry). Oracle fields are
    never consulted.
    """
    substrate = substrate or load_substrate()
    by_subject = substrate["by_subject"]
    snap = "%s@%s" % (SUBSTRATE_SCHEMA, substrate["digest"][:16])
    iid = item["item_id"]
    admitted, misleading = [], []
    for subj in sorted(query_subject_words(item)):
        rows = sorted(by_subject.get(subj, []), key=_evidence_rank, reverse=True)
        if not rows:
            continue                      # build-time coverage check forbids this
        best = rows[0]
        positive = substrate_fact(best, snap, iid, SUBJECT_TYPE_ROLE)
        admitted.append(positive)
        conflict = next((r for r in reversed(rows)
                         if r["type_label"] != best["type_label"]), None)
        if conflict is None:
            # No conflict for this word: borrow a LOW-credibility assertion
            # from another substrate entry, keeping THAT entry's subject and
            # provenance verbatim. It is an honest fact about another word.
            forbidden = forbidden_channel_subjects(item) | query_subject_words(item)
            conflict = next((r for r in substrate["low_first"]
                             if r["type_label"] != best["type_label"]
                             and str(r["subject"]).strip().lower() not in forbidden),
                            None)
        if conflict is not None:
            misleading.append(substrate_fact(conflict, snap, iid,
                                             MISLEADING_ROLE))
    return admitted, misleading


def query_subject_words(item):
    """Declared query subjects: near/far anchors and the negative subject."""
    out = {str(item["unseen_test_words"][c]["subject"]).strip().lower()
           for c in ("near", "far")}
    out.add(str(item["negative"]["subject"]).strip().lower())
    return out


def forbidden_channel_subjects(item):
    """Answer-key tokens that MUST NOT be named by any channel fact.

    Query subjects are NOT forbidden: their own substrate type assertion is
    the declared TRA-K input (see module admission rule).
    """
    return ({str(w).strip().lower() for w in answer_key_tokens(item)}
            - query_subject_words(item))


def substrate_provenance_ok(fact):
    return (isinstance(fact, dict)
            and all(fact.get(k) not in (None, "") for k in SUBSTRATE_PROVENANCE_KEYS))


def admitted_channel_fact(fact, item):
    """Source-provenance admission rule for one channel fact (module doc)."""
    if not isinstance(fact, dict):
        return False
    if any(k not in fact or fact.get(k) in (None, "") for k in REQUIRED_FACT_KEYS):
        return False
    if str(fact.get("predicate", "")).lower() not in ISA_PREDICATES:
        return False
    subj = str(fact.get("subject", "")).strip().lower()
    if subj in forbidden_channel_subjects(item):
        return False
    if subj in query_subject_words(item):
        return substrate_provenance_ok(fact)
    return True


def _fact_credibility(f):
    return (1 if f.get("sense") == "primary" else 0,
            {"item_description": 3, "type_description": 2,
             "label_only": 1}.get(f.get("text_evidence_kind"), 0),
            _digest(str(f.get("object")), str(f.get("source_id"))))


def admitted_subject_facts(item, subject, role=SUBJECT_TYPE_ROLE):
    """All ADMITTED subject facts, highest credibility first."""
    target = str(subject).strip().lower()
    facts = [f for f in item.get("channel_evidence", [])
             if admitted_channel_fact(f, item)
             and str(f.get("subject", "")).strip().lower() == target
             and (role is None or f.get("evidence_role") == role)]
    return sorted(facts, key=_fact_credibility, reverse=True)


def admitted_subject_fact(item, subject, role=SUBJECT_TYPE_ROLE):
    """Highest-credibility ADMITTED subject fact, or None (fail closed)."""
    facts = admitted_subject_facts(item, subject, role)
    return facts[0] if facts else None


def ordered_channel_facts(item):
    """Teaching order: positive subject facts, conflicts, then demo facts."""
    def key(fact):
        subj = str(fact.get("subject", "")).strip().lower()
        if subj in query_subject_words(item):
            return (0 if fact.get("evidence_role") == SUBJECT_TYPE_ROLE else 1,
                    str(fact.get("object", "")), str(fact.get("source_id", "")))
        if fact.get("evidence_role") == MISLEADING_ROLE:
            return (2, str(fact.get("subject", "")), str(fact.get("object", "")))
        return (3, str(fact.get("subject", "")), str(fact.get("object", "")))
    return sorted((f for f in (item.get("channel_evidence") or [])
                   if admitted_channel_fact(f, item)), key=key)


def fallback_channel_evidence(item):
    """Deterministic, channel-agnostic evidence for one item.

    Uses ONLY the taught exemplar objects and the item's domain-general type
    label. The query subject, gold targets, ``type_sets`` and ``world_model``
    are never read. Provenance names the fallback source explicitly so a
    fallback-backed run is never mistaken for a real ontology snapshot.
    """
    label = item["type_label"]
    objs = []
    for ex in item["taught_exemplars"][:max(K_LEVELS)]:
        for word in (ex["r1"], ex["r2"]):
            if word not in objs:
                objs.append(word)
    return [_fact(subject=w, predicate="is_a", object_=label,
                  source=FALLBACK_SOURCE, channel="B-fallback",
                  snapshot="deterministic-v1")
            for w in objs[:max(K_LEVELS)]]


def _normalise_fact(raw, item_id):
    """Validate one externally supplied fact; return None when unusable."""
    if not isinstance(raw, dict):
        return None
    if any(k not in raw or raw[k] in (None, "") for k in REQUIRED_FACT_KEYS):
        return None
    fact = {k: raw[k] for k in sorted(raw)}
    if item_id is not None:
        fact.setdefault("item_id", item_id)
    return fact


def _coerce_mapping(data):
    """Return {item_id: [raw facts]} from any of the documented JSON shapes."""
    out = collections.defaultdict(list)
    if isinstance(data, dict):
        if isinstance(data.get("items"), dict):
            data = data["items"]
        for key, val in data.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, list):
                        out[str(k2)].extend(v2)
            elif isinstance(val, list):
                out[str(key)].extend(val)
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            iid = row.get("item_id")
            facts = row.get("facts")
            if isinstance(facts, list):
                if iid is not None:
                    out[str(iid)].extend(facts)
            elif iid is not None:
                out[str(iid)].append(row)
    return dict(out)


def load_channel_evidence(items, path=None, env=None):
    """Load external channel facts. A missing/bad file DEGRADES, never raises.

    Returns ``{"facts": {item_id: [fact, ...]}, "invalid": n, "path": str|None,
    "error": str|None, "external_items": n}``. Per the runtime-file rule, a
    partial/corrupt write is a normal failure mode: it yields an empty mapping
    and an error string that the harness reports.
    """
    env = CHANNEL_ENV if env is None else env
    path = path if path is not None else os.environ.get(env)
    result = {"facts": {}, "invalid": 0, "path": path, "error": None,
              "external_items": 0}
    if not path:
        return result
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:                       # degrade, never raise
        result["error"] = "%s: %r" % (path, exc)
        return result
    raw_map = _coerce_mapping(data)
    known = {it["item_id"] for it in items}
    facts = {}
    for iid, rows in raw_map.items():
        if iid not in known:
            continue
        good = [f for f in (_normalise_fact(r, iid) for r in rows) if f is not None]
        result["invalid"] += len(rows) - len(good)
        if good:
            facts[iid] = good
    result["facts"] = facts
    result["external_items"] = len(facts)
    return result


def channel_assertion(fact):
    """P2c teaching surface from ONE provenance-stamped channel fact.

    Uses the fact alone -- never ``type_sets`` / ``world_model`` / gold. Facts
    whose predicate is not an is_a variant are ignored by the harness.
    """
    if not isinstance(fact, dict) or str(fact.get("predicate", "")).lower() not in ISA_PREDICATES:
        return None
    subj = str(fact.get("subject", "")).strip()
    obj = str(fact.get("object", "")).strip()
    if not subj or not obj:
        return None
    article = "an" if obj[:1].lower() in "aeiou" else "a"
    return "%s is %s %s." % (subj, article, obj)


# ── typed answer options: SOURCE-NATIVE type labels from the channel ────────
# The forced-choice options are type labels taken from `channel_evidence`, never
# TRA domain words and never the oracle fields. The correct label is the
# highest-support is_a object in THIS item's own channel facts; foils are
# deterministic draws from other items' channel facts (same domain first).
N_TYPED_OPTIONS = 4
TYPED_CONDITIONS = ("near", "far", "foil", "negative", "unrelated")


def channel_demo_objects(item):
    """Ordered unique taught demo words (public field only, no oracle read)."""
    out, seen = [], set()
    for ex in item["taught_exemplars"]:
        for word in (ex["r1"], ex["r2"]):
            if word not in seen:
                seen.add(word)
                out.append(word)
    return out


def channel_demo_facts(evidence, demo_objects):
    """Provenance-stamped is_a facts on taught demo objects, channel only."""
    demos = {str(w).strip().lower() for w in demo_objects}
    return [f for f in (evidence or [])
            if isinstance(f, dict)
            and str(f.get("predicate", "")).lower() in ISA_PREDICATES
            and str(f.get("subject", "")).strip().lower() in demos
            and all(k in f and f.get(k) not in (None, "")
                    for k in REQUIRED_FACT_KEYS)]


def _rank_channel_labels(evidence, demo_objects, seed):
    """Labels by descending channel support, deterministic tie-break by sha256."""
    counts = collections.Counter(
        str(f.get("object", "")).strip()
        for f in channel_demo_facts(evidence, demo_objects))
    counts.pop("", None)
    return sorted(counts, key=lambda lab: (-counts[lab], _digest(seed, lab)))


def channel_answer_label(evidence, demo_objects, seed, rank=0, forbidden=()):
    """Pick a channel label by support only; never reads an oracle field."""
    banned = {str(x).strip().lower() for x in forbidden}
    ranked = [lab for lab in _rank_channel_labels(evidence, demo_objects, seed)
              if lab.strip().lower() not in banned]
    return ranked[min(rank, len(ranked) - 1)] if ranked else None


def collect_channel_label_pools(items):
    """{domain: labels} and global labels, both from admitted channel facts.

    Includes source-native subject assertions and their low-credibility
    conflicts, so a graded option's correct label and its foil labels all come
    from the declared channel (never from TRA's oracle fields).
    """
    by_domain, glob = collections.defaultdict(set), set()
    for it in items:
        labels = {str(f.get("object", "")).strip()
                  for f in ordered_channel_facts(it)}
        labels.discard("")
        by_domain[it["domain"]].update(labels)
        glob.update(labels)
    return dict(by_domain), glob


def _clean_foils(correct, foils, banned, limit=None):
    """Drop foils that overlap the correct label or a query-subject string."""
    correct = str(correct).lower()
    out = []
    for label in foils:
        if limit is not None and len(out) >= limit:
            break
        low = label.lower()
        if low == correct or low in banned or correct in low or low in correct:
            continue
        if any(low in kept.lower() or kept.lower() in low for kept in out):
            continue
        if label not in out:
            out.append(label)
    return out


_LABEL_ORDER_CACHE = {}


def _label_order(labels):
    """One deterministic (label, lower) order per label set object.

    The source-native channel exposes ~600 labels, so re-sorting it for every
    one of the 2,000 option probes is quadratic; cache by set identity.
    Determinism is unaffected -- the order is a fixed sha256 key, independent
    of PYTHONHASHSEED and of insertion order.
    """
    key = id(labels)
    got = _LABEL_ORDER_CACHE.get(key)
    if got is not None and got[0] is labels:
        return got[1]
    if len(_LABEL_ORDER_CACHE) > 16:
        _LABEL_ORDER_CACHE.clear()
    order = sorted(((str(l), str(l).lower()) for l in labels),
                   key=lambda pair: _digest("TRA-K-foil-order", pair[0]))
    _LABEL_ORDER_CACHE[key] = (labels, order)
    return order


def _foil_labels(correct, domain_labels, global_labels, seed, condition, banned):
    """Deterministic foils from other channel facts, same domain first."""
    forbidden = {str(correct).strip().lower()} | {
        str(b).strip().lower() for b in banned}
    offset = int(_digest(seed, condition, "ORDER")[:8], 16)
    dom = _label_order(domain_labels)
    glob = _label_order(global_labels)
    local = [lab for lab, low in dom[offset:] + dom[:offset]
             if low not in forbidden]
    seen = {lab.lower() for lab in local}
    remote = [lab for lab, low in glob[offset:] + glob[:offset]
              if low not in forbidden and low not in seen]
    return _clean_foils(correct, local + remote, banned, limit=32)


def typed_option_space(item, domain_labels, global_labels, seed=None):
    """Five forced-choice queries whose options are channel type labels.

    The correct near/far label comes from the support ranking of THIS item's
    channel evidence. Oracle fields (``type_sets``, ``world_model``, every gold
    target) are never read. Query subjects are used only to keep the rendered
    option strings unambiguous; they never select the correct option.
    """
    seed = seed or _digest(item["item_id"], "typed-options")
    evidence = item.get("channel_evidence", [])
    subjects = {item["unseen_test_words"][c]["subject"] for c in ("near", "far")}
    subjects.add(item["negative"]["subject"])
    # Options must not collide with any domain-word answer token (gold,
    # distractor or query subject): those are oracle-side domain words, not
    # source-native type labels. They are used only as a NEGATIVE filter.
    banned = {str(s).strip().lower() for s in subjects}
    banned |= {str(w).strip().lower() for w in answer_key_tokens(item)}
    banned |= {str(c).strip().lower()
               for cond in ("near", "far") for c in item["unseen_test_words"][cond]["candidates"]}
    ranked = [str(f["object"]).strip() for f in ordered_channel_facts(item)
              if str(f.get("object", "")).strip()]
    if not ranked:
        raise ValueError("%s: no channel is_a label for typed options"
                         % item["item_id"])

    def _subject_label(cond):
        """The ADMITTED subject fact decides the correct source-native label.

        Prefer the highest-credibility assertion whose label does not collide
        with a domain-word token (those remain the oracle-side word pool).
        """
        facts = admitted_subject_facts(
            item, item["unseen_test_words"][cond]["subject"])
        if not facts:
            return None
        for fact in facts:
            label = str(fact["object"]).strip()
            if label.lower() not in banned:
                return label
        return str(facts[0]["object"]).strip()

    near_label = _subject_label("near")
    far_label = _subject_label("far")
    neg_facts = admitted_subject_facts(item, item["negative"]["subject"])
    neg_label = next((str(f["object"]).strip() for f in neg_facts
                      if str(f["object"]).strip().lower() not in banned),
                     str(neg_facts[0]["object"]).strip() if neg_facts else None)
    # Fail closed: never fall back to the demo support winner for a missing
    # subject assertion -- absence is an item defect, not a licence to guess.
    if not near_label or not far_label:
        raise ValueError("%s: missing subject type assertion" % item["item_id"])
    if far_label == near_label:
        alternatives = [l for l in ranked if l != near_label and l.lower() not in banned]
        far_label = alternatives[0] if alternatives else near_label
    if not neg_label or neg_label in (near_label, far_label):
        alternatives = [l for l in ranked if l not in (near_label, far_label)
                        and l.lower() not in banned]
        neg_label = alternatives[0] if alternatives else (far_label or near_label)

    def _opts(label, condition):
        foils = _foil_labels(label, domain_labels, global_labels, seed,
                             condition, banned)[:N_TYPED_OPTIONS - 1]
        return sorted([label] + foils,
                      key=lambda l: _digest(seed, condition, "OPT", l))

    foil_opts = _clean_foils(
        near_label,
        _foil_labels(near_label, domain_labels, global_labels, seed,
                     "foil", banned), banned)[:N_TYPED_OPTIONS]
    foil_opts = sorted(foil_opts, key=lambda l: _digest(seed, "foil", "OPT", l))
    return {
        "near": {"subject": item["unseen_test_words"]["near"]["subject"],
                 "candidates": _opts(near_label, "near"), "gold": near_label,
                 "domain": item["domain"]},
        "far": {"subject": item["unseen_test_words"]["far"]["subject"],
                "candidates": _opts(far_label, "far"), "gold": far_label,
                "domain": item["far_domain"]},
        # No gold: the foil/negative/unrelated conditions are rejection probes.
        "foil": {"subject": item["unseen_test_words"]["near"]["subject"],
                 "candidates": foil_opts, "gold": None,
                 "counterexample_target": foil_opts[0] if foil_opts else None,
                 "domain": item["domain"]},
        "negative": {"subject": item["negative"]["subject"],
                     "candidates": _opts(neg_label, "negative"), "gold": None,
                     "counterexample_target": neg_label,
                     "domain": item["domain"]},
        "unrelated": {"subject": UNRELATED_SUBJECT,
                      "candidates": _opts(near_label, "unrelated"), "gold": None,
                      "domain": item["domain"]},
    }


def channel_negation_assertion(fact):
    """Negated P2c surface for a channel fact (negative control, labels only)."""
    if not isinstance(fact, dict) or str(fact.get("predicate", "")).lower() not in ISA_PREDICATES:
        return None
    subj = str(fact.get("subject", "")).strip()
    obj = str(fact.get("object", "")).strip()
    if not subj or not obj:
        return None
    article = "an" if obj[:1].lower() in "aeiou" else "a"
    return "%s is not %s %s." % (subj, article, obj)


def answer_key_tokens(item):
    """Tokens that channel evidence must NEVER name (oracle answer key)."""
    out = {item.get("subject")}
    for cond in ("near", "far"):
        q = item.get("unseen_test_words", {}).get(cond, {})
        out.update((q.get("subject"), q.get("gold")))
    neg = item.get("negative", {})
    out.add(neg.get("subject"))
    out.add(neg.get("counterexample_target"))
    out.discard(None)
    return out


def channel_overlap(items):
    """Report channel/oracle overlap and detect contamination.

    Legitimate overlap: channel subjects are taught demo objects, which also
    occur as keys in ``type_sets`` -- this is reported, not failed. A HARD FAIL
    is recorded when a channel fact names an answer-key token, carries an
    oracle-only domain tag as its type, or lacks provenance.
    """
    per_item = []
    totals = collections.Counter()
    offenders = []
    for it in items:
        facts = it.get("channel_evidence", [])
        subs = {str(f.get("subject", "")) for f in facts
                if admitted_channel_fact(f, it)}
        objs = {str(f.get("object", "")) for f in facts
                if admitted_channel_fact(f, it)
                and str(f.get("predicate", "")).lower() in ISA_PREDICATES}
        key = forbidden_channel_subjects(it)
        ts_keys = set(it.get("type_sets", {}))
        ts_vals = {v for vals in it.get("type_sets", {}).values() for v in vals}
        wm_vals = {w for targets in it.get("world_model", {}).values() for w in targets}
        # A fact naming a forbidden answer-key token is a leak even when it
        # fails another admission clause (e.g. missing provenance).
        leak = {str(f.get("subject", "")).strip().lower() for f in facts} & key
        domain_tag_leak = {o for o in objs if o in ts_vals and o != it["type_label"]}
        wm_leak = subs & wm_vals
        no_prov = [f for f in facts
                   if any(k not in f or f.get(k) in (None, "") for k in REQUIRED_FACT_KEYS)]
        rec = {
            "item_id": it["item_id"],
            "facts": len(facts),
            "overlap_type_sets": len(subs & ts_keys),
            "overlap_world_model": len(wm_leak),
            "answer_key_leak": sorted(leak),
            "oracle_domain_tag_leak": sorted(domain_tag_leak),
            "provenance_missing": len(no_prov),
        }
        per_item.append(rec)
        for k in ("overlap_type_sets", "overlap_world_model"):
            totals[k] += rec[k]
        totals["facts"] += rec["facts"]
        totals["answer_key_leak"] += len(leak)
        totals["oracle_domain_tag_leak"] += len(domain_tag_leak)
        totals["provenance_missing"] += len(no_prov)
        if leak or domain_tag_leak or wm_leak or no_prov:
            offenders.append(it["item_id"])
    return {"ok": not offenders, "totals": dict(totals), "offenders": offenders,
            "per_item": per_item}


# ── bank construction ───────────────────────────────────────────────────────
def build_items(n=N_FAMILIES, channel_override=None, substrate=None):
    """Return the frozen bank (default 400 families, 200/200 split).

    ``channel_override`` maps item_id -> external fact list; items absent from
    it get the deterministic fallback. The ONLY schema difference from TRA is
    the added ``channel_evidence`` field. The substrate snapshot supplies the
    declared query-subject type assertions; it is never derived from oracle
    fields (``type_sets``/``world_model``/gold).
    """
    channel_override = channel_override or {}
    substrate = substrate or load_substrate()
    covered = covered_subjects(substrate)
    items = []
    for fidx in range(n):
        role = ROLE_KEYS[fidx % len(ROLE_KEYS)]
        difficulty = DIFFICULTIES[(fidx // len(ROLE_KEYS)) % len(DIFFICULTIES)]
        domain_idx = (fidx // (len(ROLE_KEYS) * len(DIFFICULTIES))) % len(DOMAINS)
        domain = DOMAINS[domain_idx]
        far_domain = DOMAINS[(domain_idx + 3) % len(DOMAINS)]
        assert far_domain != domain
        item_id = "tra_k_%04d" % fidx
        type_label = ROLE_TO_TYPE[role]
        seed = "%s|%s|%s" % (HASH_PREFIX, COMPONENT, item_id)
        near_pool = set(OBJECTS[domain]) & covered
        far_pool = set(OBJECTS[far_domain]) & covered
        if len(near_pool) < 4 or len(far_pool) < 4:
            raise ValueError("%s: substrate does not cover enough query words"
                             % item_id)
        pool_words = _split_words(OBJECTS[domain], seed + "|near", near_pool)
        far_words = _split_words(OBJECTS[far_domain], seed + "|far", far_pool)

        demo = pool_words["demo"][:17]
        pairs = [(demo[i], demo[i + 1]) for i in range(16)]
        taught = [{"r1": r1, "r2": r2,
                   "surface": teaching_surface(role, r1, r2),
                   "type_label": type_label} for r1, r2 in pairs]
        demo_objs = {w for pair in pairs for w in pair}

        near_subject = pool_words["near_subject"]
        near_gold = pool_words["near_gold"]
        near_d = pool_words["distractors"]
        far_subject = far_words["near_subject"]
        far_gold = far_words["near_gold"]
        far_d = far_words["distractors"]
        assert near_subject not in demo_objs and near_gold not in demo_objs
        assert far_subject not in demo_objs and far_gold not in demo_objs

        near_cands = sorted([near_gold] + near_d,
                            key=lambda w: _digest(seed + "|NC", w))
        far_cands = sorted([far_gold] + far_d,
                           key=lambda w: _digest(seed + "|FC", w))
        neg_subject = pool_words["neg_subject"]
        neg_target = pool_words["neg_target"]
        neg_cands = sorted([neg_target] + near_d,
                           key=lambda w: _digest(seed + "|NegC", w))
        anchor = pairs[0][1]
        foil_cands = sorted([anchor] + near_d,
                            key=lambda w: _digest(seed + "|Foil", w))

        domain_event = domain + "-event"
        far_event = far_domain + "-event"
        type_sets = {}
        for w in demo:
            type_sets[w] = [type_label, domain_event]
        type_sets[near_gold] = [type_label, domain_event]
        type_sets[far_gold] = [type_label, far_event]
        for w in near_d:
            type_sets[w] = [domain + "-other"]
        for w in far_d:
            type_sets[w] = [far_domain + "-other"]
        type_sets[near_subject] = [domain_event]
        type_sets[far_subject] = [far_event]
        type_sets[neg_subject] = [domain_event]
        type_sets[neg_target] = [type_label, domain_event]

        world_model = {
            "%s|%s" % (role, near_subject): [near_gold],
            "%s|%s" % (role, far_subject): [far_gold],
            "%s|%s" % (role, neg_subject): [neg_target],
        }

        # THE ONLY NEW FIELD. Never derived from type_sets/world_model/gold.
        ext = channel_override.get(item_id)
        evidence = ([_normalise_fact(f, item_id) for f in ext] if ext else None)
        if evidence:
            evidence = [f for f in evidence if f is not None]
        if not evidence:
            evidence = fallback_channel_evidence({
                "type_label": type_label, "taught_exemplars": taught})
        # Declared input: the queried subjects' OWN substrate type assertions
        # plus deterministic low-credibility conflicts. Both come from the
        # substrate snapshot and are admitted by the source-provenance rule.
        # No oracle field is read here.
        provisional = {
            "item_id": item_id,
            "subject": near_subject,
            "unseen_test_words": {
                "near": {"subject": near_subject, "gold": near_gold},
                "far": {"subject": far_subject, "gold": far_gold}},
            "negative": {"subject": pool_words["neg_subject"],
                         "counterexample_target": pool_words["neg_target"]},
            "channel_evidence": evidence,
        }
        admitted, misleading = subject_channel_facts(provisional, substrate)
        evidence = admitted + misleading + list(evidence)

        items.append({
            "item_id": item_id,
            "role": role,
            "type_label": type_label,
            "difficulty": difficulty,
            "domain": domain,
            "far_domain": far_domain,
            "subject": near_subject,
            "taught_exemplars": taught,
            "unseen_test_words": {
                "near": {"subject": near_subject, "gold": near_gold,
                         "candidates": near_cands, "domain": domain},
                "far": {"subject": far_subject, "gold": far_gold,
                        "candidates": far_cands, "domain": far_domain},
            },
            "foil": {"subject": near_subject, "candidates": foil_cands,
                     "gold_target": None},
            "negative": {"subject": neg_subject, "candidates": neg_cands,
                         "counterexample_target": neg_target,
                         "gold_target": None},
            "type_sets": type_sets,
            "world_model": world_model,
            "channel_evidence": evidence,
            "typed_queries": None,       # filled below from channel evidence only
            "k": None,
            "pool": None,
        })

    # Derive the typed answer options AFTER every item's channel evidence is
    # fixed: correct labels come from each item's own facts, foils from the
    # domain/global pools of OTHER channel facts. No oracle field is read.
    domain_labels, global_labels = collect_channel_label_pools(items)
    for it in items:
        it["typed_queries"] = typed_option_space(
            it, domain_labels.get(it["domain"], set()), global_labels)

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

    for pool in ("primary", "sealed"):
        grp = sorted((it for it in items if it["pool"] == pool), key=item_hash)
        for idx, it in enumerate(grp):
            it["k"] = K_LEVELS[idx % len(K_LEVELS)]
    return items


def validate_items(items):
    """TRA validation plus channel-evidence schema checks."""
    errors = []
    if len({it["item_id"] for it in items}) != len(items):
        errors.append("duplicate item ids")
    if len(items) != N_FAMILIES:
        errors.append("expected %d families, got %d" % (N_FAMILIES, len(items)))
    for it in items:
        iid = it["item_id"]
        role = it["role"]
        if role not in ROLES:
            errors.append("%s bad role" % iid)
            continue
        if it["type_label"] != ROLE_TO_TYPE[role]:
            errors.append("%s type_label != role slot" % iid)
        if it["domain"] == it["far_domain"]:
            errors.append("%s far_domain == domain" % iid)
        if it["domain"] not in OBJECTS or it["far_domain"] not in OBJECTS:
            errors.append("%s unknown domain" % iid)
            continue
        if it["pool"] not in ("primary", "sealed"):
            errors.append("%s bad pool" % iid)
        if it["k"] not in K_LEVELS:
            errors.append("%s bad k" % iid)
        wm = {tuple(k.split("|", 1)): set(v)
              for k, v in it["world_model"].items()}
        taught = it["taught_exemplars"]
        if len(taught) < max(K_LEVELS):
            errors.append("%s too few taught exemplars" % iid)
        demo_objs = {ex["r2"] for ex in taught} | {ex["r1"] for ex in taught}
        for ex in taught:
            if ex["type_label"] != it["type_label"]:
                errors.append("%s exemplar type_label mismatch" % iid)
        for cond in ("near", "far"):
            q = it["unseen_test_words"][cond]
            gold = q["gold"]
            if gold not in q["candidates"]:
                errors.append("%s %s gold not a candidate" % (iid, cond))
            if len(set(q["candidates"])) != len(q["candidates"]):
                errors.append("%s %s duplicate candidates" % (iid, cond))
            if gold not in wm.get((role, q["subject"]), set()):
                errors.append("%s %s gold not in world_model" % (iid, cond))
            if q["subject"] in demo_objs or gold in demo_objs:
                errors.append("%s %s reuses a taught token" % (iid, cond))
            if it["type_label"] not in it["type_sets"].get(gold, []):
                errors.append("%s %s gold lacks the taught type" % (iid, cond))
            if not any(set(it["type_sets"].get(c, [])).isdisjoint(
                    it["type_sets"].get(gold, [])) for c in q["candidates"]
                    if c != gold):
                errors.append("%s %s no disjoint-type distractor" % (iid, cond))
            if q["domain"] != (it["domain"] if cond == "near"
                               else it["far_domain"]):
                errors.append("%s %s wrong domain label" % (iid, cond))
        if it["subject"] != it["unseen_test_words"]["near"]["subject"]:
            errors.append("%s subject != near subject" % iid)
        foil = it["foil"]
        if foil["gold_target"] is not None:
            errors.append("%s foil has a gold target" % iid)
        if not any(c in demo_objs for c in foil["candidates"]):
            errors.append("%s foil lacks a demonstrated anchor" % iid)
        if any(c in wm.get((role, foil["subject"]), set())
               for c in foil["candidates"]):
            errors.append("%s foil candidate is a world-model target" % iid)
        neg = it["negative"]
        if neg["gold_target"] is not None:
            errors.append("%s negative has a gold target" % iid)
        if neg["counterexample_target"] not in wm.get((role, neg["subject"]), set()):
            errors.append("%s negative target not in world_model" % iid)
        if neg["counterexample_target"] not in neg["candidates"]:
            errors.append("%s negative target not a candidate" % iid)
        # --- channel schema (the ONLY new field) ---
        ev = it.get("channel_evidence")
        if not isinstance(ev, list) or not ev:
            errors.append("%s empty channel_evidence" % iid)
            continue
        for f in ev:
            if not isinstance(f, dict) or any(
                    k not in f or f.get(k) in (None, "") for k in REQUIRED_FACT_KEYS):
                errors.append("%s channel fact lacks provenance" % iid)
                break
        if set(forbidden_channel_subjects(it)) & {
                str(f.get("subject", "")).strip().lower() for f in ev}:
            errors.append("%s channel evidence names answer key" % iid)
        # The declared TRA-K input: each query subject's OWN substrate type
        # assertion must be present and admissible. Missing = item defect.
        for cond, subj in (("near", it["unseen_test_words"]["near"]["subject"]),
                           ("far", it["unseen_test_words"]["far"]["subject"]),
                           ("negative", it["negative"]["subject"])):
            fact = admitted_subject_fact(it, subj)
            if fact is None or not substrate_provenance_ok(fact):
                errors.append("%s %s lacks admitted substrate subject fact"
                              % (iid, cond))
        if len({str(f.get("object", "")).strip() for f in ev
                if admitted_channel_fact(f, it)}) < 2:
            errors.append("%s channel has no conflicting type label" % iid)
        # --- typed option space (channel labels, never TRA domain words) ---
        tq = it.get("typed_queries")
        if not isinstance(tq, dict) or set(tq) != set(TYPED_CONDITIONS):
            errors.append("%s bad typed_queries" % iid)
            continue
        ev_labels = {str(f.get("object", "")).strip() for f in ev
                     if str(f.get("predicate", "")).lower() in ISA_PREDICATES}
        for cond, q in tq.items():
            cands = q.get("candidates", [])
            if (len(cands) != N_TYPED_OPTIONS
                    or len(set(cands)) != len(cands)
                    or any(not isinstance(c, str) or not c.strip() for c in cands)):
                errors.append("%s %s typed options not 4 distinct labels"
                              % (iid, cond))
            if cond in ("near", "far") and q.get("gold") not in ev_labels:
                errors.append("%s %s typed gold not in channel facts"
                              % (iid, cond))
            if cond in ("foil", "negative", "unrelated") and q.get("gold") is not None:
                errors.append("%s %s reject probe has a typed gold" % (iid, cond))
    for a, da in enumerate(DOMAINS):
        for db in DOMAINS[a + 1:]:
            if set(OBJECTS[da]) & set(OBJECTS[db]):
                errors.append("domain pools overlap: %s / %s" % (da, db))
    return errors


def pool_k_counts(items):
    """{(pool, k): family count} -- must be exactly 40 for every cell."""
    return dict(collections.Counter((it["pool"], it["k"]) for it in items))


if __name__ == "__main__":
    bank = build_items()
    errs = validate_items(bank)
    overlap = channel_overlap(bank)
    print("families=%d primary=%d sealed=%d" % (
        len(bank), sum(1 for it in bank if it["pool"] == "primary"),
        sum(1 for it in bank if it["pool"] == "sealed")))
    print("pool_k_counts=%s" % sorted(pool_k_counts(bank).items()))
    print("manifest_sha256=%s" % manifest_hashes(bank))
    print("bank_sha256=%s" % item_manifest_hash(bank))
    print("channel_overlap=%s" % overlap["totals"])
    print("validation_errors=%d" % len(errs))
    for e in errs[:10]:
        print("  ", e)
    raise SystemExit(2 if errs or not overlap["ok"] else 0)
