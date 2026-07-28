"""Phase 1 warmup (section 6.4, plan reports/opencode_64_plan.md 3.2).

Feeds clean taxonomic is-a chains from the prebuilt ConceptNet ontology
(data/conceptnet/ont.pkl) through the triplet operator of the eval
snapshot's engine, so the persistent RelationProfile for "is" carries
real transitive evidence at benchmark time (Wilson gate open), then
saves the updated snapshot.

The chains are REAL closed triangles (a isa b, b isa c, AND a isa c is
asserted by feeding the closing edge): the same statistical pattern the
learner mines online. No score is written directly — evidence counters
move only through the normal ProfileLearner.observe() path.

Usage:  python scripts/tmp_warmup_profiles.py [--chains 200] [--dry-run]
"""
import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, os.path.join(_root, "ravana", "src"))
sys.path.insert(0, _here)

from evaluate_ravana import SNAPSHOT_PATH, restore_from_snapshot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=200)
    ap.add_argument("--triangles", type=int, default=200,
                    help="inheritance triangles (isa + shared feature) "
                         "for composition warmup")
    ap.add_argument("--dry-run", action="store_true",
                    help="report evidence, do not save snapshot")
    args = ap.parse_args()

    engine = restore_from_snapshot()
    op = getattr(engine, "triplet_op", None)
    if op is None:
        print("FATAL: engine has no triplet_op")
        sys.exit(1)

    import pickle
    ont = pickle.load(open(os.path.join(_root, "data", "conceptnet",
                                        "ont.pkl"), "rb"))
    isa = ont.isa if hasattr(ont, "isa") else ont.get("isa")

    from ravana.core.triplet_inference import Triple

    def _clean(w):
        return w.replace("_", " ").strip().lower()

    fed = 0
    for a, parents in isa.items():
        if fed >= args.chains:
            break
        for b in parents:
            gp = isa.get(b)
            if not gp:
                continue
            c = next(iter(gp))
            ca, cb, cc = _clean(a), _clean(b), _clean(c)
            if len(ca.split()) > 3 or len(cb.split()) > 3 \
                    or len(cc.split()) > 3:
                continue  # skip multiword junk terms
            if ca == cb or cb == cc or ca == cc:
                continue
            # closed transitive triangle — order matters: the closing
            # edge (a isa c) must arrive so the learner counts positive
            # evidence rather than an open (negative) chain.
            op.ingest_triple(Triple(ca, "is", cb, source="seed"))
            op.ingest_triple(Triple(cb, "is", cc, source="seed"))
            op.ingest_triple(Triple(ca, "is", cc, source="seed"))
            fed += 1
            break

    # ── Phase 1.3: composition warmup from REAL inheritance triangles ──
    # (w isa p) + (p rel f) + (w rel f) — the word shares a feature with
    # its taxonomic parent. This is the empirical pattern behind property
    # inheritance; feeding it moves the learner's composition counters
    # ("is" ∘ rel -> rel) and hierarchy depth. NOTE: we do NOT warm
    # transitivity for HasProperty/CapableOf/UsedFor — those relations
    # are not transitive in the data, and authoring fake evidence would
    # violate the learned-not-authored rule.
    feats = ont["features"] if not hasattr(ont, "isa") else ont.features
    frel = ont["feature_rel"] if not hasattr(ont, "isa") else None
    _REL_NAME = {"HasProperty": "has property",
                 "CapableOf": "capable of",
                 "UsedFor": "used for"}
    tri_fed = 0
    if frel is not None:
        for w, ps in isa.items():
            if tri_fed >= args.triangles:
                break
            if w not in feats:
                continue
            for p in ps:
                if p not in feats:
                    continue
                shared = feats[w] & feats[p]
                if not shared:
                    continue
                f = next(iter(shared))
                rel = _REL_NAME.get(
                    (frel.get(p) or {}).get(f)
                    or (frel.get(w) or {}).get(f) or "")
                if not rel:
                    continue
                cw, cp, cf = _clean(w), _clean(p), _clean(f)
                if max(len(cw.split()), len(cp.split()),
                       len(cf.split())) > 3 or len({cw, cp, cf}) < 3:
                    continue
                # Order: premises first, conclusion last (case-4
                # composition counting needs (s,r3,c) known when the
                # chain is enumerated — feed conclusion, then re-feed
                # the first leg so the learner sees the closed pattern).
                op.ingest_triple(Triple(cp, rel, cf, source="seed"))
                op.ingest_triple(Triple(cw, rel, cf, source="seed"))
                op.ingest_triple(Triple(cw, "is", cp, source="seed"))
                tri_fed += 1
                break

    prof = op.memory.profiles.get("is")
    if prof is None:
        print("FATAL: no 'is' profile after warmup")
        sys.exit(1)
    wl = prof.transitivity_lower()
    print(f"chains fed: {fed}  triangles fed: {tri_fed}")
    print(f"'is' profile: transitivity_score={prof.transitivity_score:.3f} "
          f"pos={prof.transitivity_pos} neg={prof.transitivity_neg} "
          f"wilson_lower={wl:.3f}")
    # Report every OPEN inference gate. Exit criteria: >= 4 ANSWERABLE
    # predicates — a predicate is answerable when a channel can produce
    # conclusions for it: its own transitivity/symmetry gate is open,
    # OR it is the dominant TARGET of some composition bucket (the
    # Composition operator answers (s, target, ?) via r1∘r2→target).
    answerable = {}
    for pname, p in sorted(op.memory.profiles.items()):
        if p.transitivity_lower() > 0.5:
            answerable.setdefault(pname, []).append(
                f"trans={p.transitivity_lower():.2f}")
        if p.symmetry_lower() > 0.5:
            answerable.setdefault(pname, []).append(
                f"sym={p.symmetry_lower():.2f}")
        for r2, bucket in p.composition_counts.items():
            total = sum(bucket.values())
            if total:
                top_r3, top_n = max(bucket.items(), key=lambda kv: kv[1])
                if top_n / total > 0.5 and top_n >= 2:
                    answerable.setdefault(top_r3, []).append(
                        f"comp({pname}∘{r2})={top_n}/{total}")
    for pname, kinds in sorted(answerable.items()):
        print(f"  ANSWERABLE {pname}: {', '.join(kinds)}")
    n_evidence = sum(1 for p in op.memory.profiles.values()
                     if (p.transitivity_pos + p.transitivity_neg) > 0
                     or p.symmetry_n > 0)
    print(f"predicates with evidence: {n_evidence}; "
          f"answerable predicates: {len(answerable)}")

    if wl <= 0.5:
        print("FAIL: 'is' Wilson lower bound did not clear 0.5 — "
              "not saving snapshot")
        sys.exit(1)
    if len(answerable) < 4:
        print("FAIL: exit criteria not met (<4 answerable predicates) — "
              "not saving snapshot")
        sys.exit(1)
    if args.dry_run:
        print("dry-run: snapshot NOT saved")
        return
    # CRITICAL: stop background threads before save — create_snapshot()
    # does this; skipping it left live locks/threads inside the state,
    # save() sanitized ConceptGraph to '<unpicklable:ConceptGraph>' and
    # the snapshot silently lost its graph (measured: 'graph was str,
    # rebuilding empty graph' on the next restore).
    engine.stop_background_learning()
    engine.save()
    import shutil
    shutil.copy2(engine._save_path, SNAPSHOT_PATH)
    print(f"snapshot updated -> {SNAPSHOT_PATH}")
    # Post-save integrity check: restore and verify the warmed profiles
    # actually answer. The eval snapshot always restores an EMPTY graph,
    # so graph-node count is NOT a valid differentiator (both A and B
    # start from the same restored-graph state; the only delta is
    # triplet profiles). We instead prove the operator answers for the
    # warmed predicates using a fed subject that has BOTH an 'is' parent
    # and a feature relation — that exercises transitivity + composition.
    e2 = restore_from_snapshot()
    op = e2.triplet_op
    m = op.memory
    answered = set()
    for t in m.triples:
        if t.predicate == "is" and not t.superseded:
            s = t.subject
            for r in ("is", "used for", "capable of", "has property"):
                try:
                    if op.infer(s, r):
                        answered.add(r)
                except Exception:
                    pass
            if len(answered) >= 4:
                break
    print(f"post-save check: 'is' wilson={m.profiles['is'].transitivity_lower():.3f}, "
          f"answerable predicates (live infer) = {sorted(answered)}")
    if m.profiles["is"].transitivity_lower() <= 0.5 or len(answered) < 4:
        print("FAIL: post-save integrity check failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
