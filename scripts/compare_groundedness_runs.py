#!/usr/bin/env python3
"""Compare two decoder-groundedness instrument runs, per subject.

The instrument (`scripts/eval_decoder_groundedness.py`) writes a TSV with one
row per (engine, subject, raw_input) triple. Temperature sampling makes single
runs noisy, so a whole-run mean can move for reasons that have nothing to do with
training. This script joins two runs on (engine, subject, raw_input) and reports:

  * the paired delta on each metric (mean of per-row differences),
  * how many rows improved / worsened / were unchanged,
  * the rows with the largest movement.

Only the UNGATED rows are compared by default -- that is the strategy the
retraining is meant to change. Pass --gated to compare the gated columns.

Usage:
    python scripts/compare_groundedness_runs.py BEFORE.tsv AFTER.tsv [--gated]
"""
from __future__ import annotations

import csv
import sys
from typing import Dict, List, Tuple

# column -> (display name, higher is better?)
UNGATED_METRICS = {
    "ungated_on_topic_frac": ("on_topic", True),
    "ungated_unsupported_count": ("unsupported", False),
    "ungated_supported_count": ("supported", True),
    "ungated_mean_cosine": ("cosine", True),
    "ungated_word_count": ("words", None),
    "ungated_unique_ratio": ("unique", None),
}

GATED_METRICS = {
    "gated_on_topic_frac": ("gated_on_topic", True),
    "gated_unsupported_count": ("gated_unsupported", False),
    "gated_supported_count": ("gated_supported", True),
    "gated_mean_cosine": ("gated_cosine", True),
    "gated_word_count": ("gated_words", None),
    "gated_unique_ratio": ("gated_unique", None),
}


def _load(path: str) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    rows: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: no header")
        for rec in reader:
            key = (rec.get("engine", "").strip(),
                   rec.get("subject", "").strip(),
                   rec.get("raw_input", "").strip())
            out: Dict[str, float] = {}
            for col in list(UNGATED_METRICS) + list(GATED_METRICS):
                raw = rec.get(col, "").strip()
                if raw == "":
                    continue
                try:
                    out[col] = float(raw)
                except ValueError:
                    pass
            if out:
                rows[key] = out
    return rows


def main(argv: List[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    gated = "--gated" in argv
    if len(args) != 2:
        print(__doc__)
        return 2
    before_path, after_path = args
    before = _load(before_path)
    after = _load(after_path)

    common = sorted(set(before) & set(after))
    if not common:
        print("No common (engine, subject, raw_input) rows between the two runs.")
        print(f"  before rows: {len(before)}  after rows: {len(after)}")
        return 1

    print(f"paired rows: {len(common)}  "
          f"(before-only {len(set(before) - set(after))}, "
          f"after-only {len(set(after) - set(before))})")
    print()

    metrics = GATED_METRICS if gated else UNGATED_METRICS
    for col, (name, higher_better) in metrics.items():
        deltas = [after[k][col] - before[k][col]
                  for k in common if col in before[k] and col in after[k]]
        if not deltas:
            continue
        n = len(deltas)
        mean_delta = sum(deltas) / n
        if higher_better is True:
            improved = sum(1 for d in deltas if d > 0)
            worsened = sum(1 for d in deltas if d < 0)
        elif higher_better is False:
            improved = sum(1 for d in deltas if d < 0)
            worsened = sum(1 for d in deltas if d > 0)
        else:
            improved = worsened = 0
        unchanged = n - improved - worsened
        direction = "" if higher_better is None else (
            "higher=better" if higher_better else "lower=better")
        print(f"{name:16s} mean_delta={mean_delta:+.4f}  "
              f"improved={improved} worsened={worsened} same={unchanged}  "
              f"({direction})")

    for col, (name, _hb) in list(metrics.items())[:2]:
        rows = [(abs(after[k][col] - before[k][col]), k)
                for k in common if col in before[k] and col in after[k]]
        if not rows:
            continue
        rows.sort(reverse=True)
        print(f"\n largest |delta| on {name}:")
        for d, k in rows[:8]:
            b, a = before[k][col], after[k][col]
            print(f"   {d:+.3f}  {k[1]:22s} {k[2]:28s} {b:.3f} -> {a:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
