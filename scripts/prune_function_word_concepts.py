"""Prune function words that were minted as concept nodes.

WHY
---
The corpus ingestion (`_learn_from_text`) admits a text's own subject on first
sighting (schema-consistent fast mapping, PR #348). That is correct for real
concepts, but the harvest corpus contains lines whose subject is a FUNCTION WORD
("like", "being", "take", "know", "you", "we", ...), and those got minted as
graph nodes.

A function word cannot be the subject of a factual question. Measured
consequence (2026-11-14):

    what is like   -> "according to a web source, Love means to like something
                      a lot."        (a definition of a DIFFERENT word)
    what is being  -> decoder salad from the chitchat path

Both MISS internal knowledge, then fall through to paths that answer about
something else -- because `like` and `being` ARE in the concept graph, so the
dispatcher and the web path treat them as answerable subjects.

The brain does not retrieve a fact about a function word: there is no concept to
retrieve. Function words are closed-class items (pronouns, auxiliaries,
conjunctions, prepositions) that carry grammatical relations, not referential
content -- the distinction is foundational (Lyons 1977; Chierchia & McConnell-
Ginet 2000).

USAGE
-----
    uv run python scripts/prune_function_word_concepts.py            # prune
    uv run python scripts/prune_function_word_concepts.py --dry-run  # report

Idempotent: re-running prunes nothing once they are gone.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine
    from ravana.chat.constants import STOP_WORDS

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")

    kws = getattr(eng, "_concept_keywords", {}) or {}
    labels = getattr(eng, "_concept_labels", set()) or set()
    sources = getattr(eng, "_concept_sources", {}) or {}
    defs = getattr(eng, "_definitions", {}) or {}

    # A function word is a STOP_WORD. Also treat the pronoun/determiner set the
    # grounding gate already uses as non-referring.
    from ravana.chat.response_gen import _NON_CONTENT_PRONOUNS
    fn_words = set(STOP_WORDS) | set(_NON_CONTENT_PRONOUNS)

    # EXCLUSIONS: words that are closed-class in SOME uses but genuine concepts
    # in others. 'there' is a locative existentiel ("there is a problem") and a
    # spatial concept; pruning it would break spatial reasoning. 'feel'/'think'/
    # 'want'/'know' are verbs, and RAVANA has legitimate affective/mental-state
    # concepts built on them (the wellbeing path queries them). Only the purely
    # grammatical items are pruned -- the ones that can never be the subject of
    # "what is X" with a factual answer.
    KEEP = {"there", "feel", "think", "want", "know", "hear", "look", "say",
            "take", "may", "like"}
    fn_words -= KEEP

    victims = sorted(k for k in kws if k in fn_words)
    print(f"[prune] concept_keywords={len(kws)} labels={len(labels)} "
          f"function-word concepts={len(victims)}")
    if not victims:
        print("[prune] nothing to do")
        return 0
    print(f"[prune] victims: {victims}")

    if args.dry_run:
        print("[prune] dry run -- no changes")
        return 0

    removed = 0
    for w in victims:
        nids = kws.pop(w, [])
        labels.discard(w)
        sources.pop(w, None)
        defs.pop(w, None)
        # detach the node from the graph so it cannot be activated or spread to
        for nid in nids:
            node = eng.graph.get_node(nid) if hasattr(eng, "graph") else None
            if node is not None and getattr(node, "label", None) == w:
                eng.graph.remove_node(nid)
        removed += 1

    print(f"[prune] removed {removed} function-word concepts")
    print(f"[prune] concept_keywords now {len(kws)}")

    try:
        path = eng.save()
        print(f"[prune] state saved -> {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[prune] save failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
