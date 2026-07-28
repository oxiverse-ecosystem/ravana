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

    prof = op.memory.profiles.get("is")
    if prof is None:
        print("FATAL: no 'is' profile after warmup")
        sys.exit(1)
    wl = prof.transitivity_lower()
    print(f"chains fed: {fed}")
    print(f"'is' profile: transitivity_score={prof.transitivity_score:.3f} "
          f"pos={prof.transitivity_pos} neg={prof.transitivity_neg} "
          f"wilson_lower={wl:.3f}")
    n_evidence = sum(1 for p in op.memory.profiles.values()
                     if (p.transitivity_pos + p.transitivity_neg) > 0
                     or p.symmetry_n > 0)
    print(f"predicates with evidence: {n_evidence}")

    if wl <= 0.5:
        print("FAIL: 'is' Wilson lower bound did not clear 0.5 — "
              "not saving snapshot")
        sys.exit(1)
    if args.dry_run:
        print("dry-run: snapshot NOT saved")
        return
    engine.save()
    import shutil
    shutil.copy2(engine._save_path, SNAPSHOT_PATH)
    print(f"snapshot updated -> {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
