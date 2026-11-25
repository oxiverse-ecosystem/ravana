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


def _split_words(pool, seed):
    """Deterministically carve a 24-word pool into query + distractor + demo.

    Query subject and gold are reserved first so they are NEVER in the taught
    exemplar object set (TRA validation rule 3, kept unchanged).
    """
    near_gold = _hash_pick(seed + "|NG", pool, 1)[0]
    near_subject = _hash_pick(seed + "|NS", pool, 1, {near_gold})[0]
    neg_target = _hash_pick(seed + "|NT", pool, 1,
                            {near_gold, near_subject})[0]
    neg_subject = _hash_pick(seed + "|NSub", pool, 1,
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


def answer_key_tokens(item):
    """Tokens that channel evidence must NEVER name (oracle answer key)."""
    out = {item["subject"]}
    for cond in ("near", "far"):
        q = item["unseen_test_words"][cond]
        out.update((q["subject"], q["gold"]))
    out.add(item["negative"]["subject"])
    out.add(item["negative"]["counterexample_target"])
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
        subs = {str(f.get("subject", "")) for f in facts}
        objs = {str(f.get("object", "")) for f in facts
                if str(f.get("predicate", "")).lower() in ISA_PREDICATES}
        key = answer_key_tokens(it)
        ts_keys = set(it.get("type_sets", {}))
        ts_vals = {v for vals in it.get("type_sets", {}).values() for v in vals}
        wm_vals = {w for targets in it.get("world_model", {}).values() for w in targets}
        leak = subs & key
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
def build_items(n=N_FAMILIES, channel_override=None):
    """Return the frozen bank (default 400 families, 200/200 split).

    ``channel_override`` maps item_id -> external fact list; items absent from
    it get the deterministic fallback. The ONLY schema difference from TRA is
    the added ``channel_evidence`` field.
    """
    channel_override = channel_override or {}
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
        pool_words = _split_words(OBJECTS[domain], seed + "|near")
        far_words = _split_words(OBJECTS[far_domain], seed + "|far")

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
            "k": None,
            "pool": None,
        })

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
        if set(answer_key_tokens(it)) & {str(f.get("subject", "")) for f in ev}:
            errors.append("%s channel evidence names answer key" % iid)
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
