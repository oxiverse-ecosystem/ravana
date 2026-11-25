#!/usr/bin/env python3
"""Deterministic adapter: Wikidata semantic substrate -> TRAK_CHANNEL_JSON.

STATIC ONLY -- no engine boot, no network, no store write and no process-global
patch at import. The substrate is the checked-in snapshot
``scratch/trak_substrate.json`` (980 provenance-stamped Wikidata P31/P279
facts); the output is the pluggable channel file consumed by
``tra_k_items.load_channel_evidence()`` / ``scripts/measure_tra_k.py``.

ORACLE INDEPENDENCE
-------------------
No channel field is derived from ``type_sets``, ``world_model``, the gold
target, ``unseen_test_words`` or any per-item answer key. The only item data
read is the PUBLIC ``taught_exemplars`` field of the frozen bank, used to
decide which *global* substrate facts are deployable for an item (the fact's
subject must be one of that item's taught demo objects). The harness's own
validation -- which does inspect the answer key -- is re-run as an OUTPUT
check only; it never contributes a fact field. Its filter is the same one the
substrate doc requires (SEMANTIC_SUBSTRATE.md section 6): a fact whose subject
is in that item's answer-key token set must not be emitted.

PROVENANCE
----------
Every emitted fact is a verbatim projection of one substrate record:
``subject`` (unchanged), ``predicate='is_a'``, ``object=type_label`` (the
source-native Wikidata type label -- never a TRA-authored role label),
``source=source_url`` and the original license / license_url / span / QIDs /
sense / evidence fields copied through. ``channel='B-wikidata'`` marks the
source. Nothing is synthesized: the adapter refuses to emit a fact whose
substrate record lacks any provenance field, and ``validate_projection``
proves every emitted fact maps back to a real substrate record.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SCRIPTS = os.path.join(PROJ, "scripts")
for _p in (PROJ, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tra_k_items as tki                      # noqa: E402

SUBSTRATE_SCHEMA = "ravana-trak-semantic-substrate-v1"
OUTPUT_SCHEMA = "ravana-trak-wikidata-channel-v1"
CHANNEL = "B-wikidata"
DEFAULT_SUBSTRATE = os.path.join(HERE, "trak_substrate.json")
DEFAULT_OUTPUT = os.path.join(HERE, "trak_channel_from_substrate.json")
DEFAULT_MAX_PER_ITEM = 8                       # == max(K_LEVELS) teaching surfaces

# Fields copied verbatim from the substrate record into the channel fact.
PROVENANCE_FIELDS = (
    "license", "license_url", "source_url", "span", "text_evidence",
    "text_evidence_kind", "wikidata_property", "wikidata_relation",
    "subject_qid", "type_qid", "subject_description", "type_description",
    "enwiki_title", "sense", "sense_selection",
)


def load_substrate(path=DEFAULT_SUBSTRATE):
    """Load and minimally validate the checked-in substrate snapshot."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != SUBSTRATE_SCHEMA:
        raise ValueError("unexpected substrate schema: %r" % data.get("schema"))
    facts = data.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("substrate has no facts")
    for f in facts:
        for key in ("subject", "type_label", "source_url", "license",
                    "license_url", "span", "subject_qid", "type_qid",
                    "wikidata_property"):
            if not f.get(key):
                raise ValueError("substrate fact missing %s: %r" % (key, f))
        if f.get("predicate") != "is_a":
            raise ValueError("substrate fact has unexpected predicate: %r" % f)
    return data


def _fact_id(f):
    """Stable digest of the source-native statement (no Python hash())."""
    return hashlib.sha256("|".join(str(f[k]) for k in (
        "source_url", "subject_qid", "type_qid", "type_label",
        "wikidata_property", "sense")).encode("utf-8")).hexdigest()[:16]


def _project_fact(raw, snapshot):
    """Project ONE substrate record; copy provenance, never invent it."""
    out = {
        "subject": raw["subject"],
        "predicate": "is_a",
        "object": raw["type_label"],           # source-native Wikidata label
        "channel": CHANNEL,
        "source": raw["source_url"],
        "source_id": _fact_id(raw),
        "snapshot": snapshot,
    }
    for key in PROVENANCE_FIELDS:
        out[key] = raw.get(key, "")
    return out


def _demo_objects(item):
    """Ordered unique taught demo words -- the item's PUBLIC field only."""
    out, seen = [], set()
    for ex in item["taught_exemplars"]:
        for word in (ex["r1"], ex["r2"]):
            if word not in seen:
                seen.add(word)
                out.append(word)
    return out


def build_projection(substrate, items=None, max_per_item=DEFAULT_MAX_PER_ITEM):
    """Return ``{item_id: [channel facts]}`` for the frozen bank.

    Selection is a deterministic round-robin over each item's taught demo
    words; per-subject fact order is the substrate's own order. Final facts
    are sorted by a provenance key, so the output bytes do not depend on dict
    iteration order or PYTHONHASHSEED.

    ``max_per_item <= 0`` means "every deployable fact" (no cap).
    """
    items = tki.build_items() if items is None else items
    by_subject = collections.defaultdict(list)
    for raw in substrate["facts"]:
        by_subject[raw["subject"]].append(raw)
    snap = "%s@%s" % (SUBSTRATE_SCHEMA, substrate.get("digest", "")[:16])
    projection = {}
    for it in items:
        demos = _demo_objects(it)
        buckets = {w: list(by_subject.get(w, ())) for w in demos}
        chosen = []
        if max_per_item <= 0:
            for w in demos:
                chosen.extend(buckets[w])
        else:
            while len(chosen) < max_per_item:
                added = False
                for w in demos:
                    if buckets[w] and len(chosen) < max_per_item:
                        chosen.append(buckets[w].pop(0))
                        added = True
                if not added:
                    break
        facts = [_project_fact(raw, snap) for raw in chosen]
        facts.sort(key=lambda f: (f["subject"], f["object"], f["source_url"],
                                  f["type_qid"], f["sense"]))
        projection[it["item_id"]] = facts
    return projection


def validate_projection(substrate, projection, items=None):
    """Output-only checks: format, provenance fidelity, zero oracle leakage.

    This is where the harness's answer-key filter runs. It proves the adapter
    emitted nothing from the answer key; it never shapes the projection.
    """
    items = tki.build_items() if items is None else items
    raw_index = {
        (f["subject"], f["type_label"], f["source_url"], f["type_qid"],
         f["wikidata_property"], f["sense"]): f
        for f in substrate["facts"]
    }
    problems = []
    n_facts = 0
    for iid, facts in projection.items():
        for fact in facts:
            n_facts += 1
            if any(k not in fact or fact.get(k) in (None, "")
                   for k in tki.REQUIRED_FACT_KEYS):
                problems.append("%s: missing provenance" % iid)
                continue
            key = (fact["subject"], fact["object"], fact["source_url"],
                   fact["type_qid"], fact["wikidata_property"], fact["sense"])
            raw = raw_index.get(key)
            if raw is None:
                problems.append("%s: fact not in substrate snapshot" % iid)
                continue
            for field in PROVENANCE_FIELDS:
                if fact.get(field, "") != raw.get(field, ""):
                    problems.append("%s: provenance field %s altered" % (iid, field))
                    break
    # Round-trip through the harness's normalizer and its own hard-fail filters.
    loaded = {iid: [tki._normalise_fact(f, iid) for f in facts]
              for iid, facts in projection.items()}
    built = tki.build_items(channel_override=loaded)
    errs = tki.validate_items(built)
    overlap = tki.channel_overlap(built)
    if errs:
        problems.extend(errs[:10])
    if not overlap["ok"]:
        problems.append("channel_overlap offenders: %r" % overlap["offenders"][:10])
    return {
        "ok": not problems,
        "problems": problems[:20],
        "facts": n_facts,
        "items": len(projection),
        "validate_errors": len(errs),
        "overlap": overlap["totals"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic Wikidata substrate -> TRAK_CHANNEL_JSON adapter")
    ap.add_argument("--substrate", default=DEFAULT_SUBSTRATE)
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--max-per-item", type=int, default=DEFAULT_MAX_PER_ITEM,
                    help="0 means all deployable facts (default: %(default)s)")
    args = ap.parse_args(argv)

    substrate = load_substrate(args.substrate)
    # Digest is metadata only; it never enters a channel fact's provenance.
    with open(args.substrate, "rb") as fh:
        substrate["digest"] = hashlib.sha256(fh.read()).hexdigest()
    items = tki.build_items()
    projection = build_projection(substrate, items, args.max_per_item)
    report = validate_projection(substrate, projection, items)
    if not report["ok"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    out = {
        "schema": OUTPUT_SCHEMA,
        "source_schema": SUBSTRATE_SCHEMA,
        "substrate_sha256": substrate["digest"],
        "channel": CHANNEL,
        "license": "CC0-1.0",
        "max_per_item": args.max_per_item,
        "items": projection,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    print("wrote %s" % args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
