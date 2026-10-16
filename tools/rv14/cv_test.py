"""Honest held-out separation for the event-polarity readout.

Lesson from the first attempt: with n=38 and d=64, least-squares interpolation
returns an in-sample AUC of 1.000 for the REAL labels AND for every label
permutation. An AUC that cannot be destroyed by shuffling the labels is an
artefact of the fit, not evidence of signal — it is discarded.

What is reported here instead is repeated random-split CROSS-VALIDATED
nearest-centroid and ridge, both with the decision threshold chosen on the
TRAIN half only. The card's operating point is recall at ZERO false
positives, so that is what is scored.
"""
import io
import os
import re
import sys
import contextlib

os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
sys.path.insert(0, os.path.join(PROJ, "tools", "rv14"))
from bindtree import bind, verify  # noqa: E402

bind(PROJ)

from probe_set import PROBES  # noqa: E402
import numpy as np  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

STOP = {"my", "the", "a", "an", "i", "me", "was", "were", "is", "are", "has",
        "have", "had", "got", "for", "of", "to", "in", "on", "at", "it",
        "they", "them", "we", "us", "and", "or", "but", "from", "with", "by",
        "its", "his", "her", "their", "our", "your", "this", "that", "again",
        "just", "still", "now", "up", "out", "off", "over", "last", "next",
        "another", "one", "two", "first", "not", "no", "do", "does", "did",
        "so", "if", "then", "than", "about"}


def content(utt):
    return [w for w in re.findall(r"[a-z']+", utt.lower())
            if w not in STOP and len(w) >= 3]


def build(gv):
    X, y, utts, wss = [], [], [], []
    for utt, label, _n in PROBES:
        ws = content(utt)
        vs = [gv(w) for w in ws]
        vs = [v for v in vs if v is not None]
        if not vs:
            continue
        m = np.mean(vs, axis=0)
        nrm = np.linalg.norm(m)
        X.append(m / nrm if nrm > 1e-9 else m)
        y.append(1 if label == "adverse" else 0)
        utts.append(utt)
        wss.append(ws)
    return np.array(X), np.array(y), utts, wss


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return 0.5
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def zero_fp_threshold(s_tr, y_tr):
    """Threshold that selects NOTHING benign, chosen on train only."""
    ben = sorted(s_tr[y_tr == 0])
    return (max(ben) + 1e-9) if len(ben) else -1e18


def main():
    verify(PROJ)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="rv14cv")
    X, y, utts, wss = build(eng._glove_vector)
    n = len(X)
    print(f"n={n} dims={X.shape[1]} adverse={int(y.sum())} benign={int((1 - y).sum())}")

    rng = np.random.default_rng(7)
    rec_nc, rec_ridge, aucs_nc, aucs_ridge = [], [], [], []

    print("\n--- 400 random splits (70/30), threshold from TRAIN only")
    for trial in range(400):
        perm = rng.permutation(n)
        k = int(n * 0.7)
        tr, te = perm[:k], perm[k:]
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        if len(set(ytr)) < 2:
            continue

        # nearest centroid
        ca = Xtr[ytr == 1].mean(axis=0); ca /= np.linalg.norm(ca) + 1e-9
        cb = Xtr[ytr == 0].mean(axis=0); cb /= np.linalg.norm(cb) + 1e-9
        d = ca - cb
        s_tr, s_te = Xtr @ d, Xte @ d
        thr = zero_fp_threshold(s_tr, ytr)
        pred = s_te >= thr
        te_pos = yte == 1
        rec_nc.append((pred & te_pos).sum() / max(te_pos.sum(), 1))
        aucs_nc.append(auc(s_te, yte))

        # ridge (regularised least squares on ±1 labels)
        lam = 1.0
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        b = Xtr.T @ (ytr * 2 - 1)
        w = np.linalg.solve(A, b)
        w /= np.linalg.norm(w) + 1e-9
        s_tr2, s_te2 = Xtr @ w, Xte @ w
        thr2 = zero_fp_threshold(s_tr2, ytr)
        pred2 = s_te2 >= thr2
        rec_ridge.append((pred2 & te_pos).sum() / max(te_pos.sum(), 1))
        aucs_ridge.append(auc(s_te2, yte))

    for nm, r, a in (("nearest-centroid", rec_nc, aucs_nc),
                     ("ridge(lam=1)", rec_ridge, aucs_ridge)):
        r = np.array(r); a = np.array(a)
        print(f"  {nm:18} held-out recall@0FP mean={r.mean():.3f} "
              f"median={np.median(r):.3f} max={r.max():.3f}  "
              f"held-out AUC mean={a.mean():.3f}")

    # Permutation control for the nearest-centroid HELD-OUT AUC.
    perm_aucs = []
    for _ in range(400):
        yp = y.copy(); rng.shuffle(yp)
        ca = X[yp == 1].mean(axis=0); ca /= np.linalg.norm(ca) + 1e-9
        cb = X[yp == 0].mean(axis=0); cb /= np.linalg.norm(cb) + 1e-9
        perm_aucs.append(auc(X @ (ca - cb), yp))
    pa = np.array(perm_aucs)
    real = auc(X @ (X[y == 1].mean(axis=0) / np.linalg.norm(X[y == 1].mean(axis=0))
                    - X[y == 0].mean(axis=0) / np.linalg.norm(X[y == 0].mean(axis=0))), y)
    print(f"\n  in-sample nearest-centroid AUC = {real:.3f}")
    print(f"  400 shuffled-label AUCs: mean={pa.mean():.3f} sd={pa.std():.3f} "
          f" 95th pct={np.percentile(pa, 95):.3f}")
    print(f"  p(real > shuffled) = {(pa >= real).mean():.3f}")

    # Which single words carry the in-sample signal? (diagnostic, not a table)
    print("\n--- per-word mean-vector cosine to the two centroids (diagnostic)")
    ca = X[y == 1].mean(axis=0); ca /= np.linalg.norm(ca)
    cb = X[y == 0].mean(axis=0); cb /= np.linalg.norm(cb)
    acc = {}
    for ws, lab in zip(wss, y):
        for w in ws:
            a = acc.setdefault(w, [0.0, 0])
            a[0] += (1 if lab == 1 else -1); a[1] += 1
    rows = sorted(acc.items(), key=lambda kv: -abs(kv[1][0]))
    print(f"  {'word':16}{'adv-minus-ben':>15}{'n':>4}")
    for w, (s, c) in rows[:18]:
        print(f"  {w:16}{int(s):>15d}{int(c):>4d}")
    eng.stop_background_learning()


if __name__ == "__main__":
    main()