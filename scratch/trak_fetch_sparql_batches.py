#!/usr/bin/env python3
"""Fetch a static Wikidata snapshot of P31/P279 facts for TRA-K vocabulary.

STATIC DATA BUILD ONLY. No engine boot, no production edit. The answer key
(type_sets / world_model / gold / unseen_test_words) is never read: only the
public OBJECTS vocabulary of scripts/tra_k_items.py at c962f75 is used as a
list of query strings. Wikimedia WDQS allows ~1 request/minute; six batches
of 24 words are fetched with 66 s spacing and saved incrementally.

Output: scratch/wikidata_sparql_raw.json
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMBANK_REF = "c962f75:scripts/tra_k_items.py"
RAW = os.path.join(ROOT, "scratch", "wikidata_sparql_raw.json")
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "trak-semantic-substrate-build/1.0 (static research build)"

QUERY = """SELECT DISTINCT ?word ?item ?itemDesc ?rel ?type ?typeLabel ?typeDesc ?enwiki WHERE {
  VALUES ?word { %s }
  ?item rdfs:label ?word .
  { ?item wdt:P31 ?type . BIND("instance of" AS ?rel) }
  UNION
  { ?item wdt:P279 ?type . BIND("subclass of" AS ?rel) }
  OPTIONAL { ?item schema:description ?itemDesc . FILTER(LANG(?itemDesc) = "en") }
  OPTIONAL { ?type schema:description ?typeDesc . FILTER(LANG(?typeDesc) = "en") }
  OPTIONAL { ?article schema:about ?item ;
                      schema:isPartOf <https://en.wikipedia.org/> ;
                      schema:name ?enwiki . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def vocab():
    src = subprocess.check_output(
        ["git", "-C", ROOT, "show", ITEMBANK_REF], text=True, encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "OBJECTS" for t in node.targets):
            pools = ast.literal_eval(node.value)
            words = sorted({w for pool in pools.values() for w in pool})
            assert len(words) == 144, len(words)
            return pools, words
    raise RuntimeError("OBJECTS not found")


def fetch(query, tries=5):
    url = ENDPOINT + "?" + urllib.parse.urlencode(
        {"query": query, "format": "json"})
    for attempt in range(tries):
        p = subprocess.run(
            ["curl", "-sS", "-f", "-m", "150", "-H", "User-Agent: " + UA,
             "-H", "Accept: application/sparql-results+json", url],
            capture_output=True, text=True)
        if p.returncode == 0:
            try:
                return json.loads(p.stdout), url
            except json.JSONDecodeError:
                pass
        print("fetch attempt %d failed: %s %s" % (
            attempt + 1, p.stderr.strip()[:120], p.stdout[:120]), file=sys.stderr)
        time.sleep(70)
    raise RuntimeError("SPARQL fetch failed")


def main():
    pools, words = vocab()
    batches = [words[i:i + 24] for i in range(0, len(words), 24)]
    saved = {"schema": "trak-wikidata-sparql-raw-v1",
             "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "endpoint": ENDPOINT, "item_bank_ref": ITEMBANK_REF,
             "domain_pools": pools, "vocabulary": words,
             "batch_bindings": {}}
    for i, batch in enumerate(batches, 1):
        q = QUERY % " ".join('"%s"@en' % w for w in batch)
        data, url = fetch(q)
        b = data.get("results", {}).get("bindings", [])
        saved["batch_bindings"]["batch_%02d" % i] = {
            "words": batch, "request_url": url,
            "query": q, "bindings": b}
        with open(RAW, "w", encoding="utf-8") as fh:
            json.dump(saved, fh, ensure_ascii=False, separators=(",", ":"))
        print("batch %d/%d: %d words, %d bindings" % (
            i, len(batches), len(batch), len(b)), file=sys.stderr, flush=True)
        if i < len(batches):
            time.sleep(66)
    print("wrote %s (%d bytes)" % (RAW, os.path.getsize(RAW)))


if __name__ == "__main__":
    main()
