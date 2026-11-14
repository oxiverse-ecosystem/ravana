"""Is the fitted W_sm projection collapsing all subjects to one point?

W_sm maps a 65-D Binder attribute vector to a 75-D dual-code embedding. If the
projection is rank-deficient or the attribute encoder maps every subject to a
nearby signature, the conditioning signal carries no subject information.

Usage:
    uv run python scripts/diagnose_wsm_collapse.py [n_subjects]
"""
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    nd = eng.neural_decoder
    W = getattr(nd, "_linggen_W_sm", None)
    enc = getattr(eng, "_combined_attr_encoder", None)
    gv = getattr(eng, "_glove_vector", None)
    print(f"[wsm] W_sm shape: {None if W is None else W.shape}")
    print(f"[wsm] encoder: {type(enc).__name__ if enc is not None else None}")
    print(f"[wsm] glove: {'yes' if gv is not None else 'NO'}")
    if W is None:
        print("[wsm] NO W_sm attached -- conditioning is inert by construction")
        return 1

    # rank / spectrum of W_sm
    sv = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
    print(f"[wsm] singular values: max={sv[0]:.4f} min={sv[-1]:.6f} "
          f"effective rank(>1e-3*max)={int((sv > 1e-3*sv[0]).sum())}/{len(sv)}")

    subs = []
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            if "\t" in line:
                subs.append(line.split("\t", 1)[0].strip())
    subs = subs[:n]

    sigs, projs = [], []
    for s in subs:
        v = gv(s) if gv else None
        if v is None or enc is None:
            continue
        sig = np.asarray(enc.attribute_vector(np.asarray(v, dtype=np.float64)),
                         dtype=np.float32).reshape(-1)[:65]
        if sig.shape[0] != 65:
            continue
        proj = np.asarray(W, dtype=np.float32) @ sig
        nrm = np.linalg.norm(proj)
        proj = proj / nrm if nrm > 0 else proj
        sigs.append(sig)
        projs.append(proj)
        print(f"  {s:<14} |sig|={np.linalg.norm(sig):.3f} "
              f"|proj|={nrm:.3f} sig[:4]={np.round(sig[:4],3)}")

    if len(projs) < 2:
        print("[wsm] not enough subjects with embeddings")
        return 0

    P = np.stack(projs)
    S = np.stack(sigs)
    cp = P @ P.T
    cs = S @ S.T
    off = ~np.eye(len(projs), dtype=bool)
    print(f"\n[wsm] pairwise cosine of PROJECTED vectors: "
          f"mean={cp[off].mean():.4f} min={cp[off].min():.4f} max={cp[off].max():.4f}")
    print(f"[wsm] pairwise cosine of RAW signatures:     "
          f"mean={cs[off].mean():.4f} min={cs[off].min():.4f} max={cs[off].max():.4f}")
    print("\n[wsm] interpretation:")
    if cp[off].mean() > 0.99:
        print("  PROJECTED vectors are nearly IDENTICAL across subjects ->")
        print("  the conditioning signal carries no subject information.")
    elif cp[off].mean() > 0.9:
        print("  PROJECTED vectors are highly redundant across subjects.")
    else:
        print("  projected vectors do vary; conditioning is not the bottleneck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
