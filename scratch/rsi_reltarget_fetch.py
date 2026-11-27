#!/usr/bin/env python3
"""Fetch a FROZEN Wikidata snapshot of real RELATION-TARGET assertions for the
RSI object vocabulary. STATIC DATA BUILD ONLY: no engine, no production edit,
no answer key read. Only the public OBJECTS word list from the authored harness
is used as query strings (gold/world_model/type_sets never touched).

Licence: CC0-1.0  https://creativecommons.org/publicdomain/zero/1.0/
Endpoint: https://query.wikidata.org/sparql
Rate limit: WDQS allows ~1 req/min; one query for all words, retried with backoff.
Output: scratch/rsi_reltarget_wikidata_raw.json
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
HARNESS_REF = "scripts/measure_learn5_rsi.py"  # worktree copy, unmodified
RAW = os.path.join(ROOT, "scratch", "rsi_reltarget_wikidata_raw.json")
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "rsi-reltarget-cc0-audit/1.0 (static research build; contact: local)"
LICENSE = "CC0-1.0"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

# RSI relation family -> Wikidata statement/value property pairs.
PROPS = {
    "CAUSE": [("P828", "has cause"), ("P1542", "has effect")],
    "PART_OF": [("P361", "part of"), ("P527", "has part")],
    "PRECEDES": [("P155", "follows"), ("P156", "followed by")],
    "OPPOSITE": [("P461", "opposite of")],
}

QUERY_TMPL = """SELECT DISTINCT ?word ?item ?itemLabel ?propLabel ?obj ?objLabel ?rank ?stmt ?ps WHERE {
  VALUES ?word { %s }
  ?item rdfs:label ?word .
  VALUES ?p { %s }
  ?item ?p ?stmt .
  ?stmt ?ps ?obj .
  FILTER(STRSTARTS(STR(?ps), "http://www.wikidata.org/prop/statement/"))
  FILTER(isIRI(?obj))
  OPTIONAL { ?stmt wikibase:rank ?rank . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 10000
"""


def vocabulary():
    """Parse only the OBJECTS literal from the authored harness (no gold)."""
    path = os.path.join(ROOT, HARNESS_REF)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "OBJECTS" for t in node.targets):
            pools = ast.literal_eval(node.value)
            return pools, sorted({w for pool in pools.values() for w in pool})
    raise RuntimeError("OBJECTS literal not found")


def fetch(query, tries=8, wait=66):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    for attempt in range(1, tries + 1):
        p = subprocess.run(
            ["curl", "-sS", "-f", "-m", "180", "-H", "User-Agent: " + UA,
             "-H", "Accept: application/sparql-results+json", url],
            capture_output=True, text=True)
        if p.returncode == 0:
            try:
                return json.loads(p.stdout), url
            except json.JSONDecodeError:
                pass
        print("fetch attempt %d/%d failed: %s %s" % (
            attempt, tries, p.stderr.strip()[:100], p.stdout[:100]),
            file=sys.stderr, flush=True)
        if attempt < tries:
            time.sleep(wait)
    raise RuntimeError("WDQS fetch failed after %d attempts" % tries)


def main():
    pools, words = vocabulary()
    pids = sorted({pid for v in PROPS.values() for pid, _ in v})
    pvals = " ".join("p:" + pid for pid in pids)
    q = QUERY_TMPL % (" ".join('"%s"@en' % w for w in words), pvals)
    data, url = fetch(q)
    bindings = data.get("results", {}).get("bindings", [])
    saved = {
        "schema": "rsi-reltarget-wikidata-raw-v1",
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": ENDPOINT, "query_url": url, "query": q,
        "license": LICENSE, "license_url": LICENSE_URL,
        "source": "wikidata", "item_bank_ref": HARNESS_REF,
        "property_map": PROPS, "vocabulary": words, "domain_pools": pools,
        "binding_count": len(bindings), "bindings": bindings,
    }
    with open(RAW, "w", encoding="utf-8") as fh:
        json.dump(saved, fh, ensure_ascii=False, separators=(",", ":"))
    print("wrote %s (%d bytes, %d bindings)" % (
        RAW, os.path.getsize(RAW), len(bindings)))


if __name__ == "__main__":
    main()
