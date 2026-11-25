#!/usr/bin/env python3
"""TRA item bank -- frozen, deterministic, hash-split primary/sealed pools.

Pre-registered by ``docs/investigations/TRA_BENCHMARK_SPEC.md`` (353 lines).
This module is STATIC ONLY: building the bank boots no engine, writes no store,
makes no network call and patches no process global at import time.

Contract implemented here (spec sections 1 and 4):

  * HASH_PREFIX = "RAVANA-TRA-v1", COMPONENT = "TRA" (collision-free with the
    LEARN-5 v1 RAVANA-7 pools).
  * 400 families: 200 primary + 200 sealed. Split within difficulty by
    sha256(f"{HASH_PREFIX}|{COMPONENT}|{difficulty}|{item_id}"), sort by hash
    bytes, first half PRIMARY.
  * One k per item: within each pool, sort by the same hash and cycle
    (0, 1, 2, 4, 8) -> exactly 40 families per k per pool.
  * Item schema exactly as in spec section 1. ``world_model`` and
    ``type_sets`` are ORACLE-ONLY construction aids; they are never emitted by
    ``teaching_surface()`` and the harness never sends them to the graded
    engine query path.

Domain pools are copied verbatim from ``measure_learn5_rsi.py:137-169``
(six disjoint 24-word pools) so this file stays self-contained and
import-order independent.
"""
from __future__ import annotations

import collections
import hashlib
import json

HASH_PREFIX = "RAVANA-TRA-v1"
COMPONENT = "TRA"
K_LEVELS = (0, 1, 2, 4, 8)
CONDITIONS = ("near", "far", "foil", "negative")
DIFFICULTIES = ("easy", "medium", "hard", "expert")

# Domain-general role slots. ``type_label`` is identical for demo targets and
# gold and is shared across domain/far_domain (spec section 1). It is the only
# metadata that is legitimate teaching material, and it reaches the store only
# through a provenance-stamped demo turn (spec section 2, P2c).
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

# Verbatim from scripts/measure_learn5_rsi.py:137-169.
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
            raise RuntimeError("TRA domain pools overlap: %s / %s" % (_da, _db))

# 40 observations per k x 5 k levels x 2 pools = 400 families (spec section 4).
N_FAMILIES = 200 * 2


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
    """Spec section 4: sha256(f'{HASH_PREFIX}|TRA|{difficulty}|{item_id}')."""
    return _digest(HASH_PREFIX, COMPONENT, item["difficulty"], item["item_id"])


def _canonical_records(items, pool):
    """Full record per item so the manifest pins the entire frozen bank."""
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


# ── bank construction ───────────────────────────────────────────────────────
def _split_words(pool, seed):
    """Deterministically carve a 24-word pool into query + distractor + demo.

    Reserved first so that query subject and gold are NEVER in the taught
    exemplar object set (spec validation rule 3). Distractors prefer words with
    the gold's character length; the fixed 24-word pool is the hard ceiling so
    length preference is best-effort, never a reason to leak a demo token.
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
    """Provenance-safe teaching text.

    Names the slot at write time (spec P2c, "mist is an effect of drizzle")
    using only exemplar tokens and the domain-general type label. Gold words,
    domain/far_domain metadata, ``world_model`` and ``type_sets`` never appear
    here, so graded query text cannot inherit the answer key.
    """
    phrase = ROLE_TO_PHRASE[role]
    label = ROLE_TO_TYPE[role]
    article = "an" if label[0] in "aeiou" else "a"
    return "%s %s %s. %s is %s %s." % (r1, phrase, r2, r2, article, label)


def build_items(n=N_FAMILIES):
    """Return the frozen bank (default 400 families, 200/200 split)."""
    items = []
    for fidx in range(n):
        role = ROLE_KEYS[fidx % len(ROLE_KEYS)]
        difficulty = DIFFICULTIES[(fidx // len(ROLE_KEYS)) % len(DIFFICULTIES)]
        domain_idx = (fidx // (len(ROLE_KEYS) * len(DIFFICULTIES))) % len(DOMAINS)
        domain = DOMAINS[domain_idx]
        far_domain = DOMAINS[(domain_idx + 3) % len(DOMAINS)]
        assert far_domain != domain
        item_id = "tra_%04d" % fidx
        type_label = ROLE_TO_TYPE[role]
        seed = "%s|%s|%s" % (HASH_PREFIX, COMPONENT, item_id)
        pool_words = _split_words(OBJECTS[domain], seed + "|near")
        far_words = _split_words(OBJECTS[far_domain], seed + "|far")

        # 16 pairs from 17 reserved demo words; the first max(K_LEVELS)=8 are
        # the only ones ever taught. One k per item is assigned below.
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

        # foil: same subject, WRONG role; one demonstrated anchor invites a
        # surface matcher to accept. No foil candidate is a world-model target.
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
        # Distractors must be genuinely disjoint from the gold's type set
        # (spec validation rule 3): sharing even the domain-event tag would
        # leave no disjoint candidate. Use a separate authored tag.
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
            "k": None,
            "pool": None,
        })

    # Hash split within difficulty (spec section 4).
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

    # One k per item: sorted-by-hash cycle -> exactly 40 per k per pool.
    for pool in ("primary", "sealed"):
        grp = sorted((it for it in items if it["pool"] == pool),
                     key=item_hash)
        for idx, it in enumerate(grp):
            it["k"] = K_LEVELS[idx % len(K_LEVELS)]
    return items


def validate_items(items):
    """Spec section 1 validation; returns a list of error strings."""
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
        if neg["counterexample_target"] not in wm.get((role, neg["subject"]),
                                                      set()):
            errors.append("%s negative target not in world_model" % iid)
        if neg["counterexample_target"] not in neg["candidates"]:
            errors.append("%s negative target not a candidate" % iid)
    for a, da in enumerate(DOMAINS):
        for db in DOMAINS[a + 1:]:
            if set(OBJECTS[da]) & set(OBJECTS[db]):
                errors.append("domain pools overlap: %s / %s" % (da, db))
    return errors


def pool_k_counts(items):
    """{(pool, k): family count} -- must be exactly 40 for every cell."""
    out = collections.Counter((it["pool"], it["k"]) for it in items)
    return dict(out)


if __name__ == "__main__":
    bank = build_items()
    errs = validate_items(bank)
    print("families=%d primary=%d sealed=%d" % (
        len(bank),
        sum(1 for it in bank if it["pool"] == "primary"),
        sum(1 for it in bank if it["pool"] == "sealed")))
    print("pool_k_counts=%s" % sorted(pool_k_counts(bank).items()))
    print("manifest_sha256=%s" % manifest_hashes(bank))
    print("bank_sha256=%s" % item_manifest_hash(bank))
    print("validation_errors=%d" % len(errs))
    for e in errs[:10]:
        print("  ", e)
    raise SystemExit(2 if errs else 0)
