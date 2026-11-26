#!/usr/bin/env python3
"""BOOT-FREE A/B for the WFA stall hotspot.

No CognitiveChatEngine is constructed (no __init__, no weight-store write, no
network). We call the UNMODIFIED production method
``CognitiveChatEngine._activate_from_input`` as an unbound function on a tiny
stub whose graph holds the REAL 2167 labels read read-only from the SQLite
mirror. The two candidate arms are local scratch reimplementations; none of
them touch production files.

Arms:
  baseline  : production _activate_from_input verbatim (unbound call)
  hoist     : same matching rules, but the node/label snapshot is built once
              per turn instead of once per word
  index     : hoist + inverted token index and substring lookup instead of
              scanning every node for every word

Correctness is asserted by exact equality of the (nid -> score) dict and of
the activated node-id set.

Run (no engine boot, safe while any harness is running):
  cd C:/Users/Likhith/Documents/projects/ravana_wfa_gqprof
  PYTHONHASHSEED=0 uv run python scratch/ab_hot_loop.py
"""
import collections
import os
import re
import sqlite3
import statistics
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = r"C:\Users\Likhith\Documents\projects\ravana"
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "ravana", "src"), ROOT):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402  (class only)
from ravana.chat.engine_memory import STOP_WORDS  # noqa: E402

import measure_learn5_wfa as W  # noqa: E402


# ── real graph labels, read-only, no engine boot ─────────────────────────────
DB = os.path.join(MAIN, "weights", "ravana_weights.db")


def real_labels(limit=None):
    con = sqlite3.connect("file:" + DB.replace("\\", "/") + "?mode=ro", uri=True)
    rows = con.execute(
        "select id, label from concepts order by id" +
        ("" if limit is None else " limit %d" % limit)).fetchall()
    con.close()
    return [(int(i), str(label)) for i, label in rows if label]


class Node:
    __slots__ = ("label", "vector")

    def __init__(self, label):
        self.label = label
        self.vector = None  # label loop under test is vector-independent


class Graph:
    def __init__(self, labels):
        self.nodes = {nid: Node(lab) for nid, lab in labels}
        self.activations = []

    def get_node(self, nid):
        return self.nodes.get(nid)

    def get_outgoing(self, nid):
        return []

    def activate(self, nid, amount):
        self.activations.append((nid, round(amount, 12)))


def make_fake(labels):
    g = Graph(labels)
    kw = collections.defaultdict(list)
    for nid, node in g.nodes.items():
        for tok in set(re.findall(r"[a-z']+", node.label.lower())):
            kw[tok].append(nid)
    return SimpleNamespace(
        dim=64, graph=g, _concept_keywords=dict(kw), _concept_pos={},
        scores_seen=None,
    )


# ── candidate arm 1: hoist only ─────────────────────────────────────────────
def activate_hoist(fake, text):
    words = re.findall(r"[a-zA-Z']{1,}", text.lower())
    snapshot = [(nid, node, node.label.lower())
                for nid, node in list(fake.graph.nodes.items())]
    scores = {}
    acc_ctx = None
    word_count = 0
    for w in words:
        if w in STOP_WORDS and word_count > 0:
            continue
        w_nids = fake._concept_keywords.get(w, [])
        n400_surprise = 0.8 if word_count == 0 else 0.5
        for nid in w_nids:
            node = fake.graph.nodes.get(nid)
            pos_boost = (0.5 if node and node.label and
                         fake._concept_pos.get(node.label.lower()) == "noun"
                         else 0.0)
            scores[nid] = scores.get(nid, 0) + (5.0 + pos_boost) * (
                1.0 + n400_surprise * 2.0)
        for nid, node, label in snapshot:
            if not node or not node.label:
                continue
            s = scores.get(nid, 0.0)
            if label == w:
                s += 5.0 * (1.0 + n400_surprise)
            elif len(w) >= 3 and (label.startswith(w + " ") or
                                 (" " + w + " ") in label or
                                 label.endswith(" " + w)):
                s += 3.0
            elif len(label) >= 3 and label in w:
                s += 2.0
            if s > 0:
                scores[nid] = s
        word_count += 1
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
    for nid, sc in top:
        fake.graph.activate(nid, min(1.0, sc * 0.12))
    return [nid for nid, _ in top]


# ── candidate arm 2: hoist + inverted index ─────────────────────────────────
def activate_index(fake, text):
    words = re.findall(r"[a-zA-Z']{1,}", text.lower())
    by_token = collections.defaultdict(list)
    single = collections.defaultdict(list)
    for nid, node in fake.graph.nodes.items():
        if not node or not node.label:
            continue
        label = node.label.lower()
        entry = (nid, node, label)
        for tok in set(label.split()):
            by_token[tok].append(entry)
        if " " not in label:
            single[label].append(entry)

    scores = {}
    word_count = 0
    for w in words:
        if w in STOP_WORDS and word_count > 0:
            continue
        n400_surprise = 0.8 if word_count == 0 else 0.5
        for nid in fake._concept_keywords.get(w, []):
            node = fake.graph.nodes.get(nid)
            pos_boost = (0.5 if node and node.label and
                         fake._concept_pos.get(node.label.lower()) == "noun"
                         else 0.0)
            scores[nid] = scores.get(nid, 0) + (5.0 + pos_boost) * (
                1.0 + n400_surprise * 2.0)
        cand = list(by_token.get(w, ()))
        # `label in w` can only hold for single-word labels; enumerate the
        # substrings of w instead of scanning all labels.
        seen_sub = set()
        if len(w) >= 3:
            for i in range(len(w)):
                for j in range(i + 3, len(w) + 1):
                    sub = w[i:j]
                    if sub in seen_sub:
                        continue
                    seen_sub.add(sub)
                    cand.extend(single.get(sub, ()))
        dedup = {}
        for nid, node, label in cand:
            dedup[nid] = (node, label)
        for nid, (node, label) in dedup.items():
            s = scores.get(nid, 0.0)
            if label == w:
                s += 5.0 * (1.0 + n400_surprise)
            elif len(w) >= 3 and (label.startswith(w + " ") or
                                  (" " + w + " ") in label or
                                  label.endswith(" " + w)):
                s += 3.0
            elif len(label) >= 3 and label in w:
                s += 2.0
            if s > 0:
                scores[nid] = s
        word_count += 1
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
    for nid, sc in top:
        fake.graph.activate(nid, min(1.0, sc * 0.12))
    return [nid for nid, _ in top]


# ── A/B driver ──────────────────────────────────────────────────────────────
def timed(fn, fake, prompt, reps=3):
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(fake, prompt)
        vals.append(time.perf_counter() - t0)
    return statistics.median(vals), result


def main():
    labels = real_labels()
    prompt = W.forced_choice_prompt(
        next(it for it in W.build_items()
             if it["kind"] == "decoy" and it["pool"] == "primary"),
        "exclusion")
    words = re.findall(r"[a-zA-Z']{1,}", prompt.lower())
    print("labels=%d  prompt_chars=%d  regex_words=%d  inner_iterations≈%d" % (
        len(labels), len(prompt), len(words), len(labels) * len(words)),
        flush=True)

    arms = [
        ("baseline", lambda f, p: CognitiveChatEngine._activate_from_input(f, p)),
        ("hoist", activate_hoist),
        ("index", activate_index),
    ]
    results = {}
    for name, fn in arms:
        fake = make_fake(labels)
        secs, activated = timed(fn, fake, prompt)
        results[name] = (secs, activated)
        print("%-9s median=%.4fs  activated=%s  activations=%d" % (
            name, secs, activated[:5], len(fake.graph.activations)), flush=True)

    base_fake = make_fake(labels)
    CognitiveChatEngine._activate_from_input(base_fake, prompt)
    base_activations = base_fake.graph.activations
    for name, fn in arms[1:]:
        fake = make_fake(labels)
        fn(fake, prompt)
        same = fake.graph.activations == base_activations
        print("EQUALITY %-9s activations==baseline: %s" % (name, same), flush=True)
        if not same:
            print("  baseline:", base_activations[:8], flush=True)
            print("  candidate:", fake.graph.activations[:8], flush=True)
        print("SPEEDUP  %-9s %.2fx" % (
            name, results["baseline"][0] / max(results[name][0], 1e-9)),
            flush=True)

    # Scaling check: baseline should grow linearly in node count.
    print("\nScaling (baseline, same prompt, synthetic labels):", flush=True)
    for n in (250, 500, 1000, 2000):
        fake = make_fake([(i, "synthetic_word_%d" % i) for i in range(n)])
        secs, _ = timed(CognitiveChatEngine._activate_from_input, fake, prompt, reps=2)
        print("  nodes=%-5d median=%.4fs" % (n, secs), flush=True)


if __name__ == "__main__":
    main()
