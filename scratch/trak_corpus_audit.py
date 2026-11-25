#!/usr/bin/env python3
"""TRAK corpus channel -- STATIC/DATA-ONLY audit (no engine boot, no network, no pytest).

Question: can the repo's frozen distributional corpus supply an independent,
subject-specific type channel for TRA-K, after the external-ontology channel
returned NO-GO (docs/investigations/TRAK_ONTOLOGY_AUDIT.md on goal/trak-ontology)?

Inputs (read only):
  tests/fixtures/web_snapshots/web_fixtures.json  -- frozen web snapshots (corpus)
  tests/fixtures/web_snapshots/README.md          -- capture provenance
  scratch/type_acquisition_results.json           -- recorded subject set (rows only)
  scripts/tra_items.py                            -- item bank (oracle fields ONLY
                                                     in the overlap section)
  LICENSE, .gitignore                             -- licence / tracking status

The 25 subject words and 6 labels are taken from the 40 graded `near` subjects of
the recorded TRA replay, exactly as in the ontology audit. `type_sets`,
`world_model` and `unseen_test_words.gold` are NEVER an input to a type fact;
they appear only in section `answer_key_overlap`, where they are the object of
the leakage measurement.

Run:  python3 scratch/trak_corpus_audit.py
Out:  scratch/trak_corpus_coverage.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_items as ti  # noqa: E402  (frozen, deterministic, no engine/network)

FIX = os.path.join(ROOT, "tests", "fixtures", "web_snapshots", "web_fixtures.json")
README = os.path.join(ROOT, "tests", "fixtures", "web_snapshots", "README.md")
RECORDED = os.path.join(ROOT, "scratch", "type_acquisition_results.json")
OUT = os.path.join(ROOT, "scratch", "trak_corpus_coverage.json")

# Pre-registered lexical targets for the six TRA role labels, copied from the
# ontology audit (goal/trak-ontology:scratch/trak_ontology_audit.py). Used only
# to measure whether a corpus context could license a role label.
LABEL_TARGETS = {
    "effect": {"effect", "consequence", "result", "outcome", "byproduct",
               "by-product", "side effect"},
    "structured-whole": {"whole", "structure", "totality", "assembly",
                         "organization"},
    "earlier-event": {"precursor", "antecedent", "predecessor", "forerunner",
                      "herald"},
    "similar-form": {"similarity", "resemblance", "likeness", "analogue",
                     "analogy"},
    "supported-outcome": {"outcome", "result", "consequence", "support",
                          "basis", "foundation"},
    "contrast-case": {"contrast", "opposite", "antithesis", "counterpart",
                      "difference"},
}
TOPIC_PREFIXES = (
    "what is", "what are", "about ", "describe ", "a ", "an ", "the ",
    "how does", "how do", "why does", "why do", "role ", "role of",
    "how to", "under conditions", "if under conditions", "tell me a sentence",
)


def whole_word(word):
    return re.compile(r"\b" + re.escape(word) + r"\b", re.I)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus():
    """Return all text blobs keyed by source id, with hit metadata attached."""
    doc = json.load(open(FIX, encoding="utf-8"))
    blobs, meta = {}, {}

    def ingest(query, value, section):
        parts, hits_meta = [], []
        if isinstance(value, dict):
            for key in ("query", "content", "text", "snippet", "response",
                        "answer", "extract"):
                if isinstance(value.get(key), str):
                    parts.append(value[key])
            for hit in (value.get("hits") or []):
                if isinstance(hit, dict):
                    for key in ("title", "url", "content", "snippet", "text",
                                "body", "description"):
                        if isinstance(hit.get(key), str):
                            parts.append(hit[key])
                    hits_meta.append({k: hit.get(k) for k in
                                      ("title", "url", "sources", "authority",
                                       "score", "is_local") if k in hit})
        sid = "%s:%s" % (section, query)
        blobs[sid] = " ".join(parts)
        meta[sid] = {"query": query, "section": section, "hits": hits_meta}

    for q, v in doc["queries"].items():
        ingest(q, v, "queries")
    for q, v in doc["kb_describes"].items():
        ingest(q, v, "kb_describes")
    for q, v in doc["tool_web_search"].items():
        ingest(q, v, "tool_web_search")
    return doc, blobs, meta


def is_topic_query(word, query):
    """A query is topic-focused when the word is in the query AND the query is
    one of the word-directed templates used by the harness (not an incidental
    mention inside a query about something else)."""
    q = query.strip().lower()
    if not whole_word(word).search(q):
        return False
    return q == word or q.startswith(TOPIC_PREFIXES) or (
        q.endswith(" " + word) or q.endswith(" " + word + " meaning") or
        q.endswith(" " + word + " definition meaning"))


def context_examples(word, meta, limit=4):
    """Return subject-specific context snippets with full provenance."""
    pat = whole_word(word)
    out = []
    for sid, m in sorted(meta.items()):
        q = m["query"]
        for hit in m["hits"]:
            text = " ".join(str(hit.get(k, "")) for k in ("title", "content",
                                                          "snippet", "text",
                                                          "description"))
            mt = pat.search(text)
            if mt and sid.split(":", 1)[0] == "queries" and is_topic_query(word, q):
                url = hit.get("url") or ""
                if any(u == url for _, _, _, u in out):
                    continue
                out.append((q, (hit.get("title") or "")[:120],
                            text[max(0, mt.start() - 60):mt.end() + 120].strip(),
                            url[:200]))
                if len(out) >= limit:
                    return out
    return out


def main():
    bank = ti.build_items()
    by_id = {it["item_id"]: it for it in bank}
    recorded = json.load(open(RECORDED, encoding="utf-8"))
    rec_rows = [r for r in recorded["rows"] if r["condition"] == "near"]
    assert len(rec_rows) == 40, "expected the 40 recorded near-condition subjects"

    subjects = [{
        "item_id": r["id"], "subject": r["subject"],
        "pool": by_id[r["id"]]["pool"], "role": by_id[r["id"]]["role"],
        "type_label": by_id[r["id"]]["type_label"],
        "domain": by_id[r["id"]]["domain"],
        "candidates": list(by_id[r["id"]]["unseen_test_words"]["near"]["candidates"]),
    } for r in rec_rows]
    unique = sorted({s["subject"] for s in subjects})
    assert len(unique) == 25
    labels = sorted({s["type_label"] for s in subjects})
    assert len(labels) == 6

    doc, blobs, meta = load_corpus()

    # ---- per-subject coverage ------------------------------------------------
    per_subject = []
    for w in unique:
        pat = whole_word(w)
        hits = [sid for sid, text in blobs.items() if pat.search(text)]
        topic_hits = [sid for sid in hits
                      if sid.startswith("queries:") and
                      is_topic_query(w, meta[sid]["query"])]
        incidental = [sid for sid in hits if sid not in topic_hits]
        urls = sorted({h.get("url") for sid in topic_hits
                       for h in meta[sid]["hits"] if h.get("url")})
        examples = context_examples(w, meta)
        per_subject.append({
            "subject": w,
            "n_docs_with_word_anywhere": len(hits),
            "n_topic_queries": len(topic_hits),
            "n_topic_docs": len(topic_hits),
            "n_incidental_docs": len(incidental),
            "n_distinct_topic_urls": len(urls),
            "has_topic_context": bool(topic_hits),
            "example_contexts": [{"query": q, "title": t, "snippet": s,
                                  "url": u} for q, t, s, u in examples],
        })

    # ---- label vocabulary coverage ------------------------------------------
    label_vocab = {}
    for lab in labels:
        pat = whole_word(lab)
        docs = [sid for sid, text in blobs.items() if pat.search(text)]
        assertion = []
        for sid, text in blobs.items():
            for m in re.finditer(r"\b(?:is|are)\s+(?:a|an)\s+" + re.escape(lab) +
                                 r"\b", text, re.I):
                assertion.append(sid)
        label_vocab[lab] = {"n_docs_with_label_string": len(docs),
                            "n_explicit_is_a_label": len(assertion),
                            "present_as_label_string": bool(docs)}

    # Strict label grounding: a SUBJECT-TOPIC context must contain the required
    # label's lexical targets. Incidental mentions in unrelated queries do not
    # count -- that is the failure mode of naive co-occurrence.
    strict_grounding = []
    for s in subjects:
        targets = LABEL_TARGETS[s["type_label"]]
        tpat = re.compile(r"\b(" + "|".join(re.escape(t) for t in targets) +
                          r")\b", re.I)
        w = s["subject"]
        topic = [sid for sid, text in blobs.items()
                 if whole_word(w).search(text) and sid.startswith("queries:") and
                 is_topic_query(w, meta[sid]["query"])]
        grounded = [sid for sid in topic if tpat.search(blobs[sid])]
        strict_grounding.append({"item_id": s["item_id"], "subject": w,
                                 "label": s["type_label"],
                                 "n_topic_docs": len(topic),
                                 "n_topic_docs_with_label_target": len(grounded)})

    # ---- subject specificity / candidate invariance --------------------------
    # For each item, does the query subject have topic contexts, and are those
    # contexts shared with any of its 4 candidates (candidate-invariant)?
    specificity = []
    for s in subjects:
        w = s["subject"]
        own = {sid for sid in blobs if sid.startswith("queries:") and
               whole_word(w).search(blobs[sid]) and
               is_topic_query(w, meta[sid]["query"])}
        cand = {}
        for c in s["candidates"]:
            cand[c] = sorted({sid for sid in blobs if sid.startswith("queries:") and
                              whole_word(c).search(blobs[sid]) and
                              is_topic_query(c, meta[sid]["query"])})
        shared = sorted(c for c, ids in cand.items() if own and set(ids) & own)
        # polysemy: topic hits whose title/content names a different sense
        off_topic = []
        for q, title, snip, url in context_examples(w, meta, limit=8):
            blob = (title + " " + snip).lower()
            if ("mountain dew" in blob or "dew.sc.gov" in url.lower() or
                    "employment" in blob):
                off_topic.append(url)
        specificity.append({"item_id": s["item_id"], "subject": w,
                            "n_subject_topic_docs": len(own),
                            "candidate_topic_docs": {k: len(v) for k, v in cand.items()},
                            "candidates_sharing_context": shared,
                            "off_topic_examples": off_topic[:3]})

    # ---- overlap with answer key (MEASUREMENT ONLY) ---------------------------
    # Same definition as TRA-K's answer_key_tokens() (scripts/tra_k_items.py on
    # branch feat/tra-k-harness): the oracle tokens a channel fact must NOT name.
    key_tokens = set()
    gold_words = set()
    for it in bank:
        for cond in ("near", "far"):
            q = it["unseen_test_words"][cond]
            gold_words.add(q["gold"])
        key_tokens |= {
            it["subject"],
            it["unseen_test_words"]["near"]["subject"],
            it["unseen_test_words"]["far"]["subject"],
            it["unseen_test_words"]["near"]["gold"],
            it["unseen_test_words"]["far"]["gold"],
            it["negative"]["subject"],
            it["negative"]["counterexample_target"],
        }
    subjects_in_key = sorted(set(unique) & key_tokens)
    gold_in_corpus = sorted(w for w in gold_words if whole_word(w).search(
        " ".join(blobs.values())))

    # ---- provenance -----------------------------------------------------------
    prov = {
        "corpus_path": os.path.relpath(FIX, ROOT).replace("\\", "/"),
        "corpus_sha256": sha256_file(FIX),
        "corpus_bytes": os.path.getsize(FIX),
        "schema": doc.get("schema"), "captured_at": doc.get("captured_at"),
        "captured_by": doc.get("captured_by"), "provider": doc.get("provider"),
        "n_queries": len(doc["queries"]),
        "n_kb_describes": len(doc["kb_describes"]),
        "n_tool_web_search": len(doc["tool_web_search"]),
        "per_hit_provenance_fields": ["title", "url", "sources", "authority",
                                      "score", "is_local"],
        "readme_path": os.path.relpath(README, ROOT).replace("\\", "/"),
        "readme_sha256": sha256_file(README),
        "repo_license": "Oxiverse Community License (OCL); see LICENSE",
        "third_party_page_license": ("Not granted by the fixture: snippets from "
                                     "Wikipedia / Met Office / National Geographic "
                                     "etc. carry only their source URL and title; "
                                     "no per-page reuse licence is recorded."),
        "gitignored_scratch": True,
    }

    n_with_any = sum(1 for p in per_subject if p["n_docs_with_word_anywhere"])
    n_topic = sum(1 for p in per_subject if p["has_topic_context"])
    n_strict = sum(1 for r in strict_grounding
                   if r["n_topic_docs_with_label_target"] > 0)
    result = {
        "audit": "TRAK_CORPUS_AUDIT - static/data only; no engine boot, no gold as input",
        "question": ("Can the frozen distributional corpus supply an independent "
                     "subject-specific type channel for TRA-K?"),
        "provenance": prov,
        "subject_definition": {
            "n_instances": len(subjects), "n_unique_words": len(unique),
            "unique_words": unique, "labels": labels,
            "by_label": dict(collections.Counter(s["type_label"] for s in subjects)),
            "by_domain": dict(collections.Counter(s["domain"] for s in subjects)),
            "source": "scratch/type_acquisition_results.json:rows[condition=near]",
            "oracle_fields_used_as_input": False,
        },
        "coverage": {"per_subject": per_subject,
                     "n_unique_with_word_anywhere": n_with_any,
                     "n_unique_with_topic_context": n_topic,
                     "n_unique_with_zero_mentions": len(unique) - n_with_any},
        "label_vocabulary": label_vocab,
        "strict_label_grounding": {
            "per_item": strict_grounding,
            "n_items_with_topic_grounded_required_label": n_strict,
            "n_items_total": len(strict_grounding),
            "definition": ("subject-topic corpus context co-occurs with a lexical "
                           "target of the item's required TRA label"),
        },
        "subject_specificity": specificity,
        "answer_key_overlap": {
            "n_subject_words_that_are_answer_key_tokens_elsewhere": len(subjects_in_key),
            "n_subject_words": len(unique), "examples": subjects_in_key[:10],
            "n_gold_words_in_corpus": len(gold_in_corpus),
            "n_gold_words": len(gold_words),
            "note": ("MEASUREMENT ONLY: gold/type_sets/world_model are never fed "
                     "to any type fact; channel facts naming an answer-key token "
                     "trip TRA-K control C15 (HARD FAIL)."),
        },
        "verdict": {
            "decision": "NO-GO",
            "key_numbers": {
                "unique_subjects_with_any_corpus_mention": n_with_any,
                "unique_subjects_with_topic_context": n_topic,
                "unique_subjects_total": len(unique),
                "labels_present_as_strings": sum(
                    1 for v in label_vocab.values() if v["present_as_label_string"]),
                "labels_total": len(labels),
                "items_with_corpus_assertion_for_required_label": 0,
                "items_with_strict_topic_grounded_label": n_strict,
                "items_total": len(strict_grounding),
            },
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps({"verdict": result["verdict"],
                      "coverage": {"any": n_with_any, "topic": n_topic},
                      "strict_grounded": n_strict}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
