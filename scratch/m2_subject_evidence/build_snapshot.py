#!/usr/bin/env python3
"""Build the frozen, independently-licensed subject-side evidence snapshot.

Reads ONLY the previously-fetched Wikidata SPARQL substrate (CC0-1.0).
No item bank, gold key, type_set, world_model, engine or network is read.
Output: data/subject_type_evidence/snapshot.json with provenance + SHA.
"""
from __future__ import annotations

import hashlib
import json
import os

SUBSTRATE = ("C:/Users/Likhith/Documents/projects/ravana_xsrc_transfer/"
             "scratch/trak_substrate.json")
OUT = ("C:/Users/Likhith/Documents/projects/ravana_m2_subject_evidence/"
       "data/subject_type_evidence/snapshot.json")


def canonical_sha(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    src = json.load(open(SUBSTRATE, encoding="utf-8"))
    assert src["schema"] == "ravana-trak-semantic-substrate-v1"
    allowed = {"CC0-1.0"}
    facts = []
    for f in src["facts"]:
        if f.get("license") not in allowed:
            continue
        if f.get("predicate") != "is_a":
            continue
        if not f.get("subject") or not f.get("type_label"):
            continue
        facts.append({
            "subject": f["subject"],
            "relation": "is_a",
            "type_name": f["type_label"],
            "assertion": f.get("span", ""),
            "text_evidence": f.get("text_evidence", ""),
            "source": "wikidata",
            "license": f["license"],
            "license_url": f.get("license_url", ""),
            "source_url": f.get("source_url", ""),
            "wikidata_property": f.get("wikidata_property", ""),
            "subject_qid": f.get("subject_qid", ""),
            "type_qid": f.get("type_qid", ""),
            "sense": f.get("sense", ""),
        })
    facts.sort(key=lambda r: (r["subject"], r["type_name"], r["source_url"]))
    payload = {
        "schema": "ravana-m2-subject-type-evidence-v1",
        "provenance": {
            "sources": [{
                "source": "wikidata",
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "license_policy": "https://www.wikidata.org/wiki/Wikidata:Licensing",
                "endpoint": "https://query.wikidata.org/sparql",
                "fetched_at_utc": src.get("built_at_utc", ""),
                "credibility": 0.7,
            }],
            "upstream_schema": src["schema"],
            "upstream_sha256": canonical_sha({"facts": src["facts"]}),
            "facts_sha256": canonical_sha(facts),
        },
        "facts": facts,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.write("\n")
    subjects = sorted({f["subject"] for f in facts})
    print("facts", len(facts), "subjects", len(subjects))
    print("facts_sha256", payload["provenance"]["facts_sha256"])
    print("upstream_sha256", payload["provenance"]["upstream_sha256"])
    print("out", OUT)


if __name__ == "__main__":
    main()
