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

WHY THE GATE CHANGED
====================
`_learn_from_text` routes non-junk tokens through a provisional buffer (the
hippocampus analog, `web_learning.py:786`) and used to promote them only after
reactivation across `_promote_min_sources` DISTINCT sources (default 2,
`engine.py:2028`). A single harvest gives each concept one source, so every
subject sat in the buffer forever.

That flat corroboration counter was standing in for a schema-consistency
judgement. The brain's rule is not "one source = junk, two sources = knowledge":
a fact that fills slots which ALREADY exist is learned in ONE trial (Tse et al.,
Science 2011, 333:891-895; Kumaran, Hassabis & McClelland, TiCS 2016,
20(7):512-534), and only schema-inconsistent material must be interleaved
(McClelland, JEP:General 2013, 142(4):1190-1210). RAVANA's fact gate already
implements this (`intake.py:370` `SchemaConsistencyGate.consistency()`), but the
node gate ignored it.

`_learn_from_text` now routes a text's OWN SUBJECT through that fast-mapping
case -- the text supplies the slot the subject fills -- while co-occurring tokens
still require corroboration. So this script is a SINGLE pass per pair; it no
longer manufactures a synthetic second source.

An earlier version replayed each pair under k distinct source ids to clear the
old counter. That compensated for the gate's wrong grain rather than fixing it.
Lowering k to 1 was rejected for the same reason: a single pass then minted 1791
nodes, many of them composite question-frames (`'sky appears blue because
molecules'`) rather than concepts -- an extraction-quality problem, not a reason
to keep a corroboration counter.

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

    promoted = 0
    for concept, desc in pairs:
        if is_known(engine, concept):
            continue
        try:
            engine._learn_from_text(desc, concept,
                                    source_url=f"harvest:{concept}")
        except Exception as exc:  # noqa: BLE001 - never abort the batch
            print(f"[consolidate]   {concept!r}: {exc}")
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
