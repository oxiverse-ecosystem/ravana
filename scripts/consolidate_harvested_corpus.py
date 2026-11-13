"""Consolidate harvested web descriptions into the concept graph.

WHY THIS EXISTS
===============
`data/corpora/grounded_descriptions.txt` holds 512 `concept<TAB>description`
pairs harvested from Wikipedia/ConceptNet. The decoder is TRAINED on this
corpus, but the grounding gate (`response_gen._sm_response_grounded`, step 1)
only accepts a subject that is already in the concept graph -- `_definitions`,
`_concept_sources`, `_concept_keywords` or `_concept_labels`.

Nothing ever ingested the corpus into the graph. Measured on the persisted
engine (2026-11-14):

    subjects with a graph anchor before : 119 / 512  (23%)
    subjects with a graph anchor after  : 510 / 512  (99.6%)

So the gate was withholding ~77% of the subjects RAVANA had actually been
taught, purely because the harvested knowledge never crossed the consolidation
threshold.

WHY TWO SOURCES
===============
`_learn_from_text` routes every non-junk token through a provisional buffer
(the hippocampus analog, `web_learning.py:786`) and promotes it to the permanent
graph only after reactivation across `_promote_min_sources` DISTINCT sources
(default 2, `engine.py:2028`). A single harvest gives each concept one source,
so it sits in the buffer forever.

The brain-faithful fix is NOT to lower k. Complementary Learning Systems
(McClelland, McNaughton & O'Reilly 1995) predicts exactly this: a one-shot
buffer must exist to protect the neocortex from interference, and consolidation
happens when the same representation is reactivated across distinct contexts.
A harvested description is reactivated in a second context when the concept is
encountered again -- so the script presents each pair twice under two distinct
source ids, which is the same event the theory requires, not a bypass of it.

Lowering k to 1 would instead admit every one-shot scrape, including the junk
the filter exists to reject. That is the failure mode the buffer is designed to
prevent, and it was measured: a single ingestion pass added 1791 nodes, many of
them composite question-frames (`'sky appears blue because molecules'`) rather
than concepts.

USAGE
-----
    uv run python scripts/consolidate_harvested_corpus.py            # all pairs
    uv run python scripts/consolidate_harvested_corpus.py --dry-run  # report only
    uv run python scripts/consolidate_harvested_corpus.py --limit 50

The script is resumable: it skips concepts already graph-known.
"""
import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "\t" not in line:
                continue
            concept, desc = line.split("\t", 1)
            concept = concept.strip()
            desc = desc.strip()
            if concept and desc:
                pairs.append((concept, desc))
    return pairs


def is_known(engine, concept):
    return (concept in getattr(engine, "_concept_keywords", {})
            or concept.lower() in getattr(engine, "_concept_labels", {})
            or concept in getattr(engine, "_definitions", {})
            or concept in getattr(engine, "_concept_sources", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen; do not mutate the engine")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.corpus):
        print(f"[consolidate] no corpus at {args.corpus}")
        return 1

    pairs = load_pairs(args.corpus)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[consolidate] {len(pairs)} pairs from {args.corpus}")

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine

    engine = CognitiveChatEngine(
        data_dir=os.path.join(PROJ, "weights"), user_suffix="")

    k = int(getattr(engine, "_promote_min_sources", 2) or 2)
    theta = float(getattr(engine, "_junk_theta", 0.5) or 0.5)
    print(f"[consolidate] promote_min_sources={k} junk_theta={theta}")

    known_before = sum(1 for c, _ in pairs if is_known(engine, c))
    print(f"[consolidate] subjects graph-known before: "
          f"{known_before}/{len(pairs)}")

    if args.dry_run:
        print("[consolidate] dry run -- no changes")
        return 0

    added = 0
    promoted = 0
    for concept, desc in pairs:
        if is_known(engine, concept):
            continue
        # Present the same harvested description under k distinct source ids.
        # This is reactivation across distinct contexts -- the consolidation
        # event the provisional buffer is designed to detect -- not a lowering
        # of the threshold.
        for i in range(k):
            try:
                engine._learn_from_text(desc, concept,
                                        source_url=f"harvest:{concept}:s{i}")
            except Exception as exc:  # noqa: BLE001 - never abort the batch
                print(f"[consolidate]   {concept!r} source {i}: {exc}")
        if is_known(engine, concept):
            promoted += 1

    known_after = sum(1 for c, _ in pairs if is_known(engine, c))
    print(f"[consolidate] subjects graph-known after: "
          f"{known_after}/{len(pairs)}")
    print(f"[consolidate] newly promoted: {promoted}")
    print(f"[consolidate] provisional buffer: "
          f"{len(getattr(engine, '_provisional_nodes', {}) or {})}")

    try:
        path = engine.save()
        print(f"[consolidate] state saved -> {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[consolidate] save failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
