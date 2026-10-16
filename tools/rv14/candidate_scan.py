"""Measure candidate event-polarity signals on the fresh probe set.

Every candidate is measured, none is assumed. Prints per-utterance scores and
a separation summary (best achievable threshold, AUC-like ranking) so the
choice is made on numbers, not on intuition.
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
from ravana.chat.brain_regions import classify_cause  # noqa: E402
from ravana.chat.evaluative_polarity import EvaluativePolarityModel  # noqa: E402


def stats(name, rows, key):
    """rows: list of (label, score). Report best-threshold separation."""
    adv = [s for l, s in rows if l == "adverse"]
    ben = [s for l, s in rows if l == "benign"]
    if not adv or not ben:
        return
    # The signal is "more adverse" when score is HIGHER.
    cands = sorted({*adv, *ben})
    best = None
    for t in cands:
        for ge in (True, False):
            hit = sum(1 for s in adv if (s >= t if ge else s <= t))
            fp = sum(1 for s in ben if (s >= t if ge else s <= t))
            # objective: recall, then FP, then margin
            cand = (hit, -fp, abs(t))
            if best is None or cand > best[0]:
                best = (cand, hit, fp, t, ge)
    _, hit, fp, t, ge = best
    sep = sum(adv) / len(adv) - sum(ben) / len(ben)
    print(f"  {name:28} best: recall {hit}/{len(adv)}  FP {fp}/{len(ben)}"
          f"   (thr {t:+.4f} {'>=' if ge else '<='})   mean_adv={sep:+.4f}")


def main():
    verify(PROJ)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="rv14cand")
    gv = eng._glove_vector

    # The evaluative-polarity model, wired to the engine's GloVe table.
    epm = EvaluativePolarityModel(vector_fn=gv)

    CONTENT_STOP = {"my", "the", "a", "an", "i", "me", "was", "were", "is",
                    "are", "has", "have", "had", "got", "got", "for", "of",
                    "to", "in", "on", "at", "it", "they", "them", "we", "us",
                    "and", "or", "but", "from", "with", "by", "its", "his",
                    "her", "their", "our", "your", "this", "that", "again",
                    "just", "still", "now", "up", "out", "off", "over",
                    "last", "next", "another", "one", "two", "first", "not"}

    rows_eval, rows_cause, rows_mean, rows_peak = [], [], [], []
    print(f"{'probe':52} {'label':8} {'evaluative':>11} {'cause':>18} "
          f"{'mean':>8} {'peak':>8}")
    print("-" * 112)
    for utt, label, _n in PROBES:
        words = [w for w in re.findall(r"[a-z']+", utt.lower())
                 if w not in CONTENT_STOP and len(w) >= 3]

        # 1) evaluative_polarity over each content word
        best_eval = 0.0
        for w in words:
            r = epm.score(w)
            if r is None:
                continue
            if abs(r["polarity"]) > abs(best_eval):
                best_eval = r["polarity"]
        rows_eval.append((label, best_eval))

        # 2) classify_cause on the whole utterance
        cc = classify_cause(utt, gv)
        rows_cause.append((label, cc.confidence if cc.label != "neutral" else -1.0))

        # 3/4) raw GloVe mean / peak against the AFFECT seed (what the card
        # already ruled out — measured here for comparison, not as a proposal)
        pos = [gv(w) for w in ("good", "great", "happy", "love", "success", "won")]
        neg = [gv(w) for w in ("bad", "sad", "terrible", "hurt", "fail", "lost")]
        pos = [v for v in pos if v is not None]
        neg = [v for v in neg if v is not None]
        vs = [gv(w) for w in words]
        vs = [v for v in vs if v is not None]
        if vs and pos and neg:
            pc = np.mean(pos, axis=0); pc /= np.linalg.norm(pc)
            nc = np.mean(neg, axis=0); nc /= np.linalg.norm(nc)
            m = np.mean(vs, axis=0); m /= np.linalg.norm(m)
            rows_mean.append((label, float(nc @ m - pc @ m)))
            rows_peak.append((label, float(max(float(nc @ v - pc @ v)
                                               for v in vs))))
        else:
            rows_mean.append((label, 0.0))
            rows_peak.append((label, 0.0))

        print(f"{utt[:52]:52} {label:8} {best_eval:+11.4f} "
              f"{cc.label + '/' + str(round(cc.confidence, 3)):>18} "
              f"{rows_mean[-1][1]:+8.4f} {rows_peak[-1][1]:+8.4f}")

    print("\n" + "=" * 70)
    print("SEPARATION (higher score = more adverse)")
    for nm, rr in (("evaluative_polarity(peak)", rows_eval),
                   ("classify_cause(conf)", rows_cause),
                   ("glove mean vs affect", rows_mean),
                   ("glove peak vs affect", rows_peak)):
        stats(nm, rr, None)
    eng.stop_background_learning()


if __name__ == "__main__":
    main()