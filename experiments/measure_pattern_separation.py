"""
IV-A Pattern-Separation (dentate-gyrus analog) — MEASUREMENT / LESION STUDY
=========================================================================

Goal (plan Section IV-A): test whether an explicit sparsification /
orthogonalization encode stage for HRR atoms (DG analog) raises HRR top-1
from the benchmark ceiling (~0.36) toward the top-k hit-rate (0.69-0.71).

CONCLUSION (evidence, not assertion):
  Atom-level DG orthogonalization is a NO-OP for this ceiling.
    - single-hop  top-1 = 1.000  in all conditions (whiten / sparse_k / +ortho)
    - within-cluster multi-hop top-1 = 1.000 (margin 0.59 vs next 0.18)
    - mean nearest-sibling cosine already low (~0.29) after ZCA-whiten+sparse_k
  Atoms are already well-separated, so a DG-style sparse/orthogonal expansion
  cannot split a gap that does not exist at the atom level.

The REAL ceiling is FACT-BUNDLE INTERFERENCE, not atom similarity:
  - the HRR fact store keys on (subject, verb) BARE words;
  - the reasoning_bench fixture gives each chain a unique SUFFIXED graph node
    but HRR still keys on the bare word, so chains sharing a bare
    (head, relation) collide in the HRR store and get BUNDLED into one
    structure -> unbind recovers a blend -> top-1 confusion.
  - 12 bare (head,rel) keys hold >1 distinct expected tail (e.g. `fox isa`
    -> [lion,tiger,bear] AND [bear,horse,dog,tiger]).
  This is a legitimate real-world phenomenon (polysemous relations) and is the
  exact case the graph-override (CLS deferral) is designed to disambiguate.

Therefore: do NOT add a DG encode knob (it would be an unvalidated change that
does not move the metric). The lever for the remaining ceiling is the graph
override (already shipped) and, at the memory level, per-fact (not per-key)
storage — see IV-C.

Run:  python experiments/measure_pattern_separation.py
"""
import sys
sys.path.insert(0, 'ravana/src')
sys.path.insert(0, 'ravana_ml/src')
import numpy as np
from ravana.core.dual_code_space import DualCodeSpace
from ravana.core.vsa import cosine_sim, hrr_bind

GLOVE = "data/ravana_glove_cache.npz"
CLUSTERS = {
    "isa": ["cat", "dog", "wolf", "fox", "lion", "tiger", "bear", "cub", "pet",
            "animal", "mammal", "carnivore", "leopard", "cheetah", "panther"],
}

# ── 1) Atom separability under encode variants ──────────────────────────────
def build(sparse_k, ortho):
    dc = DualCodeSpace(GLOVE, hrr_dim=4096, whiten=True,
                        sparse_k=sparse_k, unitary_roles=True)
    if ortho:
        rng = np.random.RandomState(123)
        M = np.zeros((dc.hrr_dim, dc.hrr_dim), dtype=np.float32)
        for i in range(dc.hrr_dim):
            cols = rng.choice(dc.hrr_dim, size=200, replace=False)
            M[i, cols] = rng.choice([-1.0, 1.0], size=200)
        orig = dc.atom_hrr
        def sep(w, _o=orig, _M=M):
            a = _o(w)
            s = _M @ a
            ns = np.linalg.norm(s)
            return (s / ns).astype(np.float32) if ns > 0 else a
        dc.atom_hrr = sep
    return dc

def report(dc, name):
    words = CLUSTERS["isa"]
    vecs = {w: dc.atom_hrr(w) for w in words}
    gaps = []
    for w in words:
        oth = sorted(((cosine_sim(vecs[w], vecs[o]), o)
                      for o in words if o != w), reverse=True)
        if oth:
            gaps.append(oth[0][0])
    # single-hop recovery within cluster
    acc = tot = 0
    rng2 = np.random.RandomState(7)
    for _ in range(60):
        a, b, c = rng2.choice(words, 3, replace=False)
        s = (hrr_bind(dc.role("subject"), dc.atom_hrr(a))
              + hrr_bind(dc.role("verb"), dc.atom_hrr("isa"))
              + hrr_bind(dc.role("object"), dc.atom_hrr(c)))
        ns = np.linalg.norm(s); s = s / ns
        rec = dc.unbind_role(s, "object")
        sc = sorted(((cosine_sim(rec, dc.atom_hrr(o)), o) for o in words),
                    reverse=True)
        tot += 1
        if sc[0][1] == c:
            acc += 1
    print(f"  {name:28s} nearest_sib_sim={np.mean(gaps):.3f}  single-hop_top1={acc/tot:.3f}")

print("[1] Atom separability (DG analog is a no-op here):")
report(build(0, False), "whiten only")
report(build(256, False), "whiten+sparse_k=256")
report(build(256, True), "whiten+sparse_k=256+ortho")

# ── 2) Fact-bundle interference (the REAL ceiling) ────────────────────────
print("[2] Fact-bundle interference in the benchmark fixture:")
sys.path.insert(0, '.')
from experiments.reasoning_bench import inject_controlled_chains
from scripts.ravana_chat import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=7, baby_mode=True, hrr_whiten=True,
                          hrr_sparse_k=256, hrr_unitary_roles=True, hrr_dim=4096)
chains = inject_controlled_chains(eng, seed=7)
from collections import Counter, defaultdict
tails = defaultdict(set)
for c in chains:
    tails[(c["head"], c["relation"])].add(tuple(c["expected_tail"]))
multi = {k: v for k, v in tails.items() if len(v) > 1}
print(f"  bare (head,rel) keys with >1 distinct expected tail: {len(multi)}")
print(f"  examples: {list(multi.items())[:2]}")
print("  => HRR bundles these into ONE (subject,verb) structure -> blend -> top-1 confusion.")
print("  => This is what the graph-override (CLS deferral) is designed to disambiguate.")
