"""Decisive test: is there ANY linear polarity axis in the projected GloVe
space that separates adverse from benign events and GENERALIZES?

This is the strongest possible version of the question. If an ORACLE — one
allowed to fit a separating hyperplane directly on labelled data, which is
strictly more information than any seed vocabulary — still fails to
generalize to held-out probes, then the capability the card asks for cannot
be extracted from these embeddings by any linear readout, and the honest
answer is a documented limitation rather than a tuned gate.

Method: 2-fold cross-validation over the probe set. Fit on half, test on the
other half, both directions. Report held-out recall at zero false positives,
which is the operating point the card demands.
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


def main():
    verify(PROJ)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="rv14oracle")
    gv = eng._glove_vector

    # Build one feature vector per probe: mean of content-word unit vectors
    # (the most favourable summary; also try peak-variant via per-word fit).
    X, y, wordsets = [], [], []
    for utt, label, _n in PROBES:
        ws = content(utt)
        vs = [gv(w) for w in ws]
        vs = [v for v in vs if v is not None]
        if not vs:
            continue
        m = np.mean(vs, axis=0)
        nrm = np.linalg.norm(m)
        X.append(m / nrm if nrm > 1e-9 else m)
        y.append(1.0 if label == "adverse" else -1.0)
        wordsets.append(ws)
    X = np.array(X)
    y = np.array(y)
    print(f"n={len(X)}  dims={X.shape[1]}  adverse={int((y > 0).sum())}  "
          f"benign={int((y < 0).sum())}")

    n = len(X)
    idx = np.arange(n)
    folds = [idx[0::2], idx[1::2]]

    print("\n--- ORACLE linear separator, 2-fold CV (fit on 1 half, test other)")
    for fi, test_idx in enumerate(folds):
        tr_idx = folds[1 - fi]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        # Fit a maximum-margin-ish direction by least squares on the labels,
        # then pick the threshold that maximises held-out recall at ZERO FP.
        w, *_ = np.linalg.lstsq(np.hstack([Xtr, np.ones((len(Xtr), 1))]), ytr,
                                rcond=None)
        w = w[:-1]
        w = w / (np.linalg.norm(w) + 1e-9)
        s_tr, s_te = X[tr_idx] @ w, X[test_idx] @ w
        # Zero-FP threshold from TRAIN only.
        ben_tr = sorted(s_tr[ytr < 0])
        thr = (max(ben_tr) + 1e-6) if ben_tr else -1e9
        pred = s_te >= thr
        te_y = y[test_idx] > 0
        recall = (pred & te_y).sum() / max(te_y.sum(), 1)
        fp = (pred & ~te_y).sum()
        print(f"  fold {fi}: train_acc="
              f"{((( s_tr >= (0)).astype(int) == ytr).mean()):.3f}  "
              f"HELD-OUT recall={recall:.3f}  FP={int(fp)}/{int((~te_y).sum())}")

    # Permutation control: is the apparent separation better than chance?
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(200):
        yp = rng.permutation(y)
        aucs.append(_auc(X @ (np.linalg.lstsq(np.hstack([X, np.ones((n, 1))]),
                                           yp, rcond=None)[0][:-1]), yp))
    real = _auc(X @ (np.linalg.lstsq(np.hstack([X, np.ones((n, 1))]), y,
                                      rcond=None)[0][:-1]), y)
    perm = np.array(aucs)
    print(f"\n--- in-sample AUC (oracle, upper bound) = {real:.3f}")
    print(f"--- 200 label permutations: mean AUC = {perm.mean():.3f} "
          f"+/- {perm.std():.3f}   p = {(perm >= real).mean():.3f}")

    # Nearest-centroid on GloVe alone (no fitting) — simplest possible readout.
    cen_a = X[y > 0].mean(axis=0); cen_b = X[y < 0].mean(axis=0)
    cen_a /= np.linalg.norm(cen_a); cen_b /= np.linalg.norm(cen_b)
    s = X @ (cen_a - cen_b)
    te = y > 0
    ben_s = sorted(s[~te])
    thr = (max(ben_s) + 1e-6) if ben_s else -1e9
    print(f"\n--- nearest-centroid (fit on ALL, zero-FP thr, optimistic): "
          f"recall={(  (s >= thr) & te).sum()}/{te.sum()}  "
          f"FP={((s >= thr) & ~te).sum()}/{(~te).sum()}  AUC={_auc(s, y):.3f}")

    eng.stop_background_learning()


def _auc(scores, y):
    pos = scores[y > 0]
    neg = scores[y < 0]
    if not len(pos) or not len(neg):
        return 0.5
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


if __name__ == "__main__":
    main()