#!/usr/bin/env python3
"""Build the TRA-K semantic substrate and its coverage/overlap audit.

Inputs (read-only):
  scratch/wikidata_sparql_raw.json            -- static Wikidata snapshot
  git c962f75:scripts/tra_k_items.py          -- public item-bank vocabulary
                                                 and per-item answer key, used
                                                 ONLY for overlap measurement
Outputs:
  scratch/trak_substrate.json
  scratch/trak_substrate_coverage.json

No engine boot, no production edit, no network call. Facts are copied verbatim
from the snapshot; nothing is authored or inferred by this script.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "scratch", "wikidata_sparql_raw.json")
OUT = os.path.join(ROOT, "scratch", "trak_substrate.json")
COV = os.path.join(ROOT, "scratch", "trak_substrate_coverage.json")
ITEMBANK_REF = "c962f75:scripts/tra_k_items.py"
LICENSE = "CC0-1.0"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
SOURCE_URL = "https://www.wikidata.org/"
TRA_LABELS = {"effect", "structured-whole", "earlier-event",
              "similar-form", "supported-outcome", "contrast-case"}


def val(binding, key):
    v = binding.get(key)
    return v.get("value", "") if isinstance(v, dict) else ""


def qid(uri):
    return uri.rsplit("/", 1)[-1] if uri else ""


def load_raw():
    raw = json.load(open(RAW, encoding="utf-8"))
    by_word = collections.defaultdict(lambda: collections.defaultdict(dict))
    for batch in raw["batch_bindings"].values():
        for b in batch["bindings"]:
            word = val(b, "word")
            item = qid(val(b, "item"))
            rel = val(b, "rel")
            prop = "P31" if rel == "instance of" else "P279"
            if not word or not item.startswith("Q") or prop not in ("P31", "P279"):
                continue
            rec = by_word[word][item]
            rec.setdefault("item_desc", val(b, "itemDesc"))
            rec.setdefault("enwiki", val(b, "enwiki"))
            t_qid = qid(val(b, "type"))
            t_label = val(b, "typeLabel")
            if not t_qid.startswith("Q") or not t_label:
                continue
            # Label service returns the bare QID when no English label exists;
            # a QID is an identifier, not a source-native type label.
            if t_label.startswith("Q") and t_label[1:].isdigit():
                continue
            rec.setdefault("types", {})[(prop, t_qid)] = {
                "label": t_label, "desc": val(b, "typeDesc")}
    return raw, {w: dict(items) for w, items in by_word.items()}


def pick_primary(word, item_map):
    """Deterministic sense selection. All facts are source-present either way."""
    def wiki_ok(q, rec):
        return rec.get("enwiki", "").replace("_", " ").casefold() == word.casefold()

    def desc_ok(q, rec):
        return word.casefold() in rec.get("item_desc", "").casefold()

    for rule, ok in (("enwiki_exact", wiki_ok), ("description_contains", desc_ok)):
        matching = [q for q, rec in item_map.items() if ok(q, rec)]
        if matching:
            return (sorted(matching,
                           key=lambda q: (-len(item_map[q].get("types", {})), q))[0],
                    rule)
    if item_map:
        return (sorted(item_map,
                       key=lambda q: (-len(item_map[q].get("types", {})), q))[0],
                "most_types")
    return (None, "none")


def build_facts(by_word):
    facts = []
    primary = {}
    senses = {}
    for word, item_map in sorted(by_word.items()):
        pq, rule = pick_primary(word, item_map)
        primary[word] = pq
        senses[word] = rule
        for iq, rec in sorted(item_map.items()):
            sense = "primary" if iq == pq else "alternate"
            types = rec.get("types", {})
            for (prop, tq), t in sorted(types.items()):
                label = t["label"]
                item_desc = rec.get("item_desc", "")
                type_desc = t.get("desc", "")
                if item_desc:
                    evidence, evidence_kind = item_desc, "item_description"
                elif type_desc:
                    evidence, evidence_kind = type_desc, "type_description"
                else:
                    evidence, evidence_kind = word, "label_only"
                facts.append({
                    "subject": word,
                    "predicate": "is_a",
                    "type_label": label,
                    "source_url": "https://www.wikidata.org/wiki/%s" % iq,
                    "license": LICENSE,
                    "license_url": LICENSE_URL,
                    "span": "%s — %s (%s) → %s (%s)" % (
                        word, "instance of" if prop == "P31" else "subclass of",
                        prop, label, tq),
                    "text_evidence": evidence,
                    "text_evidence_kind": evidence_kind,
                    "wikidata_property": prop,
                    "wikidata_relation": ("instance of" if prop == "P31"
                                          else "subclass of"),
                    "subject_qid": iq,
                    "type_qid": tq,
                    "subject_description": item_desc,
                    "type_description": type_desc,
                    "enwiki_title": rec.get("enwiki", ""),
                    "sense": sense,
                    "sense_selection": rule if sense == "primary" else "alternate",
                })
    facts.sort(key=lambda f: (f["subject"], f["sense"] != "primary",
                              f["subject_qid"], f["wikidata_property"],
                              f["type_qid"]))
    return facts, primary, senses


def load_item_bank():
    src = subprocess.check_output(
        ["git", "-C", ROOT, "show", ITEMBANK_REF], text=True, encoding="utf-8")
    fd, path = tempfile.mkstemp(suffix=".py", prefix="trak_items_",
                                dir=os.path.join(ROOT, "scratch"))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location("trak_items_audit", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, mod.build_items()  # 400 families; static, no engine
    finally:
        os.unlink(path)


def audit(raw, facts, primary, senses, items, mod):
    vocab = raw["vocabulary"]
    words_with_facts = {f["subject"] for f in facts}
    primary_facts = [f for f in facts if f["sense"] == "primary"]
    words_primary = {f["subject"] for f in primary_facts}

    key = set()
    for it in items:
        key |= mod.answer_key_tokens(it)
    key &= set(vocab)  # only score key tokens that are vocabulary words

    label_set = {f["type_label"] for f in facts}
    # Deployability: an item-level channel fact may not name that item's key.
    safe_total = 0
    safe_primary_total = 0
    item_cov = []
    item_cov_primary = []
    for it in items:
        this_key = mod.answer_key_tokens(it)
        exemplars = {t["r1"] for t in it["taught_exemplars"]}
        exemplars |= {t["r2"] for t in it["taught_exemplars"]}
        safe = [f for f in facts
                if f["subject"] in exemplars and f["subject"] not in this_key]
        safe_p = [f for f in primary_facts
                  if f["subject"] in exemplars and f["subject"] not in this_key]
        safe_total += len(safe)
        safe_primary_total += len(safe_p)
        item_cov.append(len(safe))
        item_cov_primary.append(len(safe_p))
    n_items = len(items)

    conflicts_all = {}
    for f in facts:
        conflicts_all.setdefault(f["subject"], set()).add(f["type_label"])
    conflicts_primary = {}
    for f in primary_facts:
        conflicts_primary.setdefault(f["subject"], set()).add(f["type_label"])
    multi_all = sorted(w for w, s in conflicts_all.items() if len(s) > 1)
    multi_primary = sorted(w for w, s in conflicts_primary.items() if len(s) > 1)

    canon = json.dumps(facts, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
    return {
        "schema": "trak-substrate-coverage-v1",
        "source": "Wikidata structured data (P31 instance of / P279 subclass of)",
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "item_bank_ref": ITEMBANK_REF,
        "raw_snapshot_sha256": hashlib.sha256(
            open(RAW, "rb").read()).hexdigest(),
        "substrate_sha256": hashlib.sha256(open(OUT, "rb").read()).hexdigest(),
        "substrate_canonical_sha256": hashlib.sha256(canon).hexdigest(),
        "vocabulary_words": len(vocab),
        "facts_total": len(facts),
        "facts_primary_sense": len(primary_facts),
        "facts_alternate_sense": len(facts) - len(primary_facts),
        "words_with_any_fact": len(words_with_facts),
        "words_with_primary_fact": len(words_primary),
        "words_without_fact": sorted(set(vocab) - words_with_facts),
        "unique_type_labels": len(label_set),
        "top_type_labels": collections.Counter(
            f["type_label"] for f in primary_facts).most_common(15),
        "by_property": dict(collections.Counter(f["wikidata_property"]
                                                for f in facts)),
        "by_evidence_kind": dict(collections.Counter(
            f["text_evidence_kind"] for f in facts)),
        "answer_key_vocabulary_tokens": len(key),
        "answer_key_subject_overlap": len(words_with_facts & key),
        "answer_key_subject_overlap_primary": len(words_primary & key),
        "TRA_authored_label_overlap": sorted(label_set & TRA_LABELS),
        "items": n_items,
        "items_with_deployable_exemplar_fact": sum(1 for c in item_cov if c > 0),
        "deployable_facts_per_item_min": min(item_cov) if item_cov else 0,
        "deployable_facts_per_item_mean": (
            sum(item_cov) / n_items if n_items else 0),
        "deployable_facts_per_item_max": max(item_cov) if item_cov else 0,
        "deployable_facts_total": safe_total,
        "deployable_primary_facts_total": safe_primary_total,
        "items_with_deployable_primary_fact": sum(
            1 for c in item_cov_primary if c > 0),
        "sense_selection_rules": dict(collections.Counter(senses.values())),
        "facts_with_label_only_evidence": sum(
            1 for f in facts if f["text_evidence_kind"] == "label_only"),
        "words_with_conflicting_types_all_senses": len(multi_all),
        "words_with_conflicting_types_primary_sense": len(multi_primary),
        "conflict_examples_primary": {w: sorted(conflicts_primary[w])
                                      for w in multi_primary[:10]},
    }


def main():
    raw, by_word = load_raw()
    facts, primary, senses = build_facts(by_word)
    substrate = {
        "schema": "ravana-trak-semantic-substrate-v1",
        "built_at_utc": raw["fetched_at_utc"],
        "source": {
            "name": "Wikidata", "url": SOURCE_URL,
            "api_endpoint": raw["endpoint"],
            "properties": {
                "P31": "https://www.wikidata.org/wiki/Property:P31",
                "P279": "https://www.wikidata.org/wiki/Property:P279"},
            "license": LICENSE, "license_url": LICENSE_URL,
            "citation": ("Vrandečić & Krötzsch (2014), Wikidata: a free "
                         "collaborative knowledgebase, CACM 57(10):78-85, "
                         "doi:10.1145/2629489"),
        },
        "construction": {
            "vocabulary_source": ITEMBANK_REF,
            "vocabulary_is_answer_key": False,
            "type_labels": "source-native English labels of Wikidata type items",
            "sense_rule": ("primary = exact-label item with an enwiki sitelink "
                           "titled as the word; else item whose description "
                           "contains the word; else item with most typed "
                           "statements, then lowest QID"),
            "answer_key_used_in_facts": False,
            "note": ("Global lexicon. Item-level deployment must filter facts "
                     "whose subject is in that item's answer key."),
        },
        "facts": facts,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(substrate, fh, ensure_ascii=False, indent=1)
    mod, items = load_item_bank()
    cov = audit(raw, facts, primary, senses, items, mod)
    with open(COV, "w", encoding="utf-8") as fh:
        json.dump(cov, fh, ensure_ascii=False, indent=1)
    for k in ("vocabulary_words", "facts_total", "facts_primary_sense",
              "words_with_any_fact", "words_with_primary_fact",
              "words_without_fact", "unique_type_labels",
              "answer_key_subject_overlap", "TRA_authored_label_overlap",
              "items_with_deployable_exemplar_fact",
              "deployable_facts_per_item_mean",
              "deployable_facts_per_item_min",
              "words_with_conflicting_types_primary_sense",
              "substrate_canonical_sha256"):
        print("%s: %s" % (k, cov[k]))


if __name__ == "__main__":
    main()
