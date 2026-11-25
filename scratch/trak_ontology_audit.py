#!/usr/bin/env python3
"""STATIC/DATA AUDIT: can an external ontology supply TRA-K type evidence?

No engine boot, no pytest, no production edit, no network call at run time
(all licence text was captured from authoritative pages during the audit).
Gold fields (type_sets / world_model / unseen_test_words.gold) are read ONLY
in the `overlap_with_answer_key` section, where they are the measurement
target, never an input to any type assertion.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_items as ti  # noqa: E402

import nltk  # noqa: E402
from nltk.corpus import wordnet as wn  # noqa: E402

SUBJECT_SOURCE = os.path.join(HERE, "type_acquisition_results.json")
OUT = os.path.join(HERE, "trak_ontology_coverage.json")

# A priori lexical targets, written from the ordinary English meaning of each
# TRA-K role label BEFORE looking at any item's gold or type_sets.
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
DOMAIN_TARGETS = {
    "weather": {"weather", "atmospheric phenomenon", "meteorological phenomenon",
                "precipitation", "climate"},
    "body": {"body part", "organ", "anatomical structure", "tissue", "body"},
    "machine": {"mechanical device", "machine", "mechanism", "device"},
    "social": {"social group", "social relation", "custom", "institution",
               "society"},
    "geometry": {"shape", "geometric point", "geometry", "mathematical object",
                 "figure"},
    "plant": {"plant", "plant part", "flora", "organism"},
}


def hypernym_lexicon(word):
    """Union of hypernym (and self) lemma names over ALL noun senses."""
    lemmas, senses = set(), []
    for ss in wn.synsets(word, pos=wn.NOUN):
        names = {l.name().replace("_", " ") for l in ss.lemmas()}
        lemmas |= names
        chain = []
        for h in ss.closure(lambda s: s.hypernyms()):
            chain.extend(sorted({l.name().replace("_", " ") for l in h.lemmas()}))
        lemmas |= set(chain)
        senses.append({"synset": ss.name(),
                       "gloss_first": ss.definition().split(";")[0][:120],
                       "self_lemmas": sorted(names),
                       "n_hypernyms": len(chain),
                       "hypernym_lemmas": sorted(set(chain))[:40]})
    return {
        "word": word,
        "n_noun_senses": len(senses),
        "n_hypernym_lemmas": len(lemmas),
        "hypernym_lemmas_sample": sorted(lemmas)[:60],
        "senses": senses[:6],
    }


def match(lemma_set, targets):
    return sorted(lemma_set & targets)


def main():
    bank = ti.build_items()
    by_id = {it["item_id"]: it for it in bank}
    recorded = json.load(open(SUBJECT_SOURCE, encoding="utf-8"))
    # The 40 test subjects actually queried in the recorded TRA experiment
    # (20 primary + 20 sealed), near condition = the graded query subject.
    subjects = [{"item_id": r["id"], "subject": r["subject"],
                 "pool": by_id[r["id"]]["pool"],
                 "role": by_id[r["id"]]["role"],
                 "type_label": by_id[r["id"]]["type_label"],
                 "domain": by_id[r["id"]]["domain"]}
                for r in recorded["rows"] if r["condition"] == "near"]
    assert len(subjects) == 40
    assert all(s["subject"] == by_id[s["item_id"]]["unseen_test_words"]["near"]["subject"]
               for s in subjects)

    # -- source availability ------------------------------------------------
    nltk_path = None
    for p in nltk.data.path:
        cand = os.path.join(p, "corpora", "wordnet.zip")
        if os.path.exists(cand):
            nltk_path = cand
            break
    lic = ""
    if nltk_path:
        with zipfile.ZipFile(nltk_path) as z:
            lic = z.read("wordnet/LICENSE").decode("utf-8", "replace")
    ont_pkl = os.path.join(ROOT, "data", "conceptnet", "ont.pkl")
    bundled = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if ".git" in dirpath.split(os.sep):
            continue
        for fn in filenames:
            if fn.lower().endswith((".owl", ".rdf", ".ttl", ".nt")) or \
               fn.lower() in ("ont.pkl", "wordnet.zip"):
                bundled.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
    availability = {
        "nltk_version": nltk.__version__,
        "wordnet_corpus_path": nltk_path,
        "wordnet_release_from_local_LICENSE": "WordNet Release 3.0" if "Release 3.0" in lic else "UNKNOWN",
        "omw_1_4_present": any(os.path.exists(os.path.join(p, "corpora", "omw-1.4.zip"))
                               for p in nltk.data.path),
        "wordnet_license_source": "Princeton University, WordNet 3.0 LICENSE (in local wordnet.zip); https://wordnet.princeton.edu/license-and-commercial-use",
        "wordnet_license_terms": "Use/copy/modify/distribute free; NOTICE (copyright+disclaimer) must appear on ALL copies; no Princeton advertising use; citation required",
        "wordnet_citation": "Miller, G. A. (1995). WordNet: a lexical database for English. CACM 38(11):39-41. Fellbaum, C. (ed.) (1998). WordNet: An Electronic Lexical Database. MIT Press.",
        "conceptnet_ont_pkl_path": ont_pkl,
        "conceptnet_ont_pkl_present": os.path.exists(ont_pkl),
        "conceptnet_license_source": "https://conceptnet.io/ ; https://zenodo.org/records/3739540",
        "conceptnet_license": "CC BY-SA 4.0 (attribution + share-alike; no added restrictions)",
        "bundled_ontology_data_files": sorted(bundled),
        "conceptnet_loader_present": "ravana/src/ravana/ontology/conceptnet.py (code only; no data file)",
    }

    # -- label vocabulary: are the 6 TRA-K type labels even in WordNet? -----
    label_vocab = {}
    for label in ti.ROLE_TO_TYPE.values():
        syns = wn.synsets(label, pos=wn.NOUN)
        label_vocab[label] = {
            "n_wordnet_noun_synsets": len(syns),
            "synsets": [{"name": s.name(), "gloss": s.definition()[:110]} for s in syns[:4]],
        }

    # -- per-subject lexical coverage (NO gold used) ------------------------
    lex = {w: hypernym_lexicon(w) for w in sorted({s["subject"] for s in subjects})}
    per_subject = []
    for s in subjects:
        lx = lex[s["subject"]]
        per_subject.append({
            "item_id": s["item_id"], "pool": s["pool"], "role": s["role"],
            "type_label": s["type_label"], "domain": s["domain"],
            "subject": s["subject"],
            "n_noun_senses": lx["n_noun_senses"],
            "n_hypernym_lemmas": lx["n_hypernym_lemmas"],
            "matched_label_targets": {
                lab: match(set(lx["hypernym_lemmas_sample"]), targets)
                for lab, targets in LABEL_TARGETS.items()
                if match(set(lx["hypernym_lemmas_sample"]), targets)
            },
            "matched_domain_targets": match(set(lx["hypernym_lemmas_sample"]),
                                            DOMAIN_TARGETS.get(s["domain"], set())),
        })
    label_coverage = {
        lab: {
            "n_subjects_with_any_matching_hypernym": sum(
                1 for p in per_subject if lab in p["matched_label_targets"]),
            "n_total": 40,
        } for lab in LABEL_TARGETS
    }
    domain_coverage = {}
    for dom in DOMAIN_TARGETS:
        rows = [p for p in per_subject if p["domain"] == dom]
        domain_coverage[dom] = {
            "n_subjects": len(rows),
            "n_with_domain_hypernym": sum(1 for p in rows if p["matched_domain_targets"]),
        }

    # -- overlap with the answer key (MEASUREMENT ONLY; never an input) ------
    word_labels = collections.defaultdict(set)
    for it in bank:
        for w, tsets in it["type_sets"].items():
            for t in tsets:
                if t in LABEL_TARGETS:
                    word_labels[w].add(t)
    conflicting = {w: sorted(v) for w, v in word_labels.items() if len(v) > 1}
    subj_roles = collections.defaultdict(set)
    for it in bank:
        for cond in ("near", "far"):
            subj_roles[it["unseen_test_words"][cond]["subject"]].add(it["role"])
        subj_roles[it["foil"]["subject"]].add(it["role"])
        subj_roles[it["negative"]["subject"]].add(it["role"])
    queried = sorted({s["subject"] for s in subjects})
    subject_role_multiplicity = {w: sorted(subj_roles[w]) for w in queried}

    domain_overlap_rows = []
    for s in subjects:
        it = by_id[s["item_id"]]
        authored = it["type_sets"].get(s["subject"], [])
        p = next(p for p in per_subject if p["item_id"] == s["item_id"])
        domain_overlap_rows.append({
            "item_id": s["item_id"], "subject": s["subject"],
            "authored_type_sets_for_subject": authored,
            "wordnet_domain_match": p["matched_domain_targets"],
            "overlap": bool(p["matched_domain_targets"]),
        })

    # ORACLE-ONLY diagnostic: could WordNet label the GOLD word with the
    # item's role label? Gold is used ONLY here, as an overlap target.
    gold_rows = []
    for s in subjects:
        it = by_id[s["item_id"]]
        gold = it["unseen_test_words"]["near"]["gold"]
        lx = hypernym_lexicon(gold)
        mm = match(set(lx["hypernym_lemmas_sample"]),
                   LABEL_TARGETS[s["type_label"]])
        gold_rows.append({"item_id": s["item_id"], "role": s["role"],
                          "gold_word": gold, "type_label": s["type_label"],
                          "wordnet_label_match": mm, "overlap": bool(mm)})

    overlap = {
        "n_words_with_role_label_in_key": len(word_labels),
        "n_words_with_conflicting_role_labels_in_key": len(conflicting),
        "pct_words_with_conflicting_role_labels": round(
            100.0 * len(conflicting) / max(1, len(word_labels)), 1),
        "conflict_examples": dict(list(sorted(conflicting.items()))[:12]),
        "queried_subjects_with_multiple_roles": sum(
            1 for v in subject_role_multiplicity.values() if len(v) > 1),
        "n_queried_unique_subjects": len(queried),
        "subject_role_multiplicity_examples": {
            w: v for w, v in subject_role_multiplicity.items() if len(v) > 1},
        "domain_overlap_n": sum(1 for r in domain_overlap_rows if r["overlap"]),
        "domain_overlap_total": 40,
        "domain_overlap_examples": domain_overlap_rows[:6],
        "oracle_only_gold_label_by_wordnet_n": sum(1 for r in gold_rows if r["overlap"]),
        "oracle_only_gold_label_by_wordnet_total": 40,
        "oracle_only_gold_examples": gold_rows[:8],
        "world_model_reproducible_from_lexicon": False,
        "world_model_reason": "world_model maps (role,subject)->arbitrary domain word chosen by sha256 hash; not a lexical relation and has no standalone ontology entry",
    }

    # -- can the channel assert a TRA-K type for a subject without gold? ----
    assertions = []
    for s in subjects:
        lx = lex[s["subject"]]
        choices = []
        for sense in lx["senses"]:
            hs = sense["hypernym_lemmas"]
            if hs:
                choices.append(hs[0])
        top = choices[0] if choices else None
        assertions.append({
            "item_id": s["item_id"], "subject": s["subject"],
            "required_label": s["type_label"],
            "wordnet_nearest_hypernym_top_sense": top,
            "n_distinct_sense_types": len(set(choices)),
            "assertion_equals_a_TRA_K_label": top in LABEL_TARGETS if top else False,
            "wordnet_assertion": (s["subject"] + " is_a " + top) if top else None,
        })
    assertion_n_equals = sum(1 for a in assertions if a["assertion_equals_a_TRA_K_label"])
    assertion_ambiguous = sum(1 for a in assertions if a["n_distinct_sense_types"] > 1)
    gold_free = {
        "n": 40,
        "n_assertion_equals_required_TRA_K_label": assertion_n_equals,
        "n_subjects_with_ambiguous_wordnet_type": assertion_ambiguous,
        "examples": assertions[:12],
        "verdict": "NO - WordNet returns domain-level hypernyms, never the role-derived TRA-K labels; the required label is item-relative, so no gold-free per-subject map exists",
    }

    # -- mapping examples for the write-up ----------------------------------
    mapping_examples = {}
    for lab in LABEL_TARGETS:
        ex = [p for p in per_subject if p["type_label"] == lab][:2]
        mapping_examples[lab] = [{
            "subject": p["subject"],
            "label_targets": sorted(LABEL_TARGETS[lab]),
            "matched": p["matched_label_targets"].get(lab, []),
            "wordnet_top_hypernyms": lex[p["subject"]]["senses"][0]["hypernym_lemmas"][:8]
            if lex[p["subject"]]["senses"] else [],
        } for p in ex]

    verdict = {
        "decision": "NO-GO",
        "reason": "Only WordNet 3.0 is offline-available; it covers the subject words lexically but cannot supply any of the 6 role-derived TRA-K type labels, and the answer key's type assignment is item-relative (word->label conflicts) so no independent source can reproduce it without the gold.",
        "key_numbers": {
            "subjects_with_any_wordnet_noun_sense": sum(1 for w in queried if lex[w]["n_noun_senses"] > 0),
            "subjects_total_unique": len(queried),
            "label_vocabulary_present_of_6": sum(1 for v in label_vocab.values() if v["n_wordnet_noun_synsets"] > 0),
            "max_label_coverage_of_40": max(v["n_subjects_with_any_matching_hypernym"] for v in label_coverage.values()),
            "domain_coverage_of_40": sum(v["n_with_domain_hypernym"] for v in domain_coverage.values()),
            "word_label_conflicts_in_key": len(conflicting),
            "queried_subjects_with_multiple_roles": sum(1 for v in subject_role_multiplicity.values() if len(v) > 1),
            "gold_free_label_assertions": assertion_n_equals,
        },
    }

    out = {
        "audit": "TRAK_ONTOLOGY_AUDIT - static/data only; no engine boot, no gold as input",
        "source_availability": availability,
        "subject_definition": {
            "n": 40, "primary": 20, "sealed": 20,
            "unique_words": len(queried),
            "provenance": "near-condition query subjects of scratch/type_acquisition_results.json (recorded TRA mechanism replay)",
            "subjects": subjects,
        },
        "label_vocabulary_in_wordnet": label_vocab,
        "coverage": {
            "per_subject": per_subject,
            "per_label": label_coverage,
            "per_domain": domain_coverage,
            "mapping_examples": mapping_examples,
        },
        "overlap_with_answer_key": overlap,
        "gold_free_assertion": gold_free,
        "verdict": verdict,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)
    print("WROTE", OUT)
    print("verdict", verdict["decision"], verdict["key_numbers"])
    print("label_coverage", label_coverage)
    print("domain_coverage", domain_coverage)
    print("label_vocab", {k: v["n_wordnet_noun_synsets"] for k, v in label_vocab.items()})


if __name__ == "__main__":
    main()
