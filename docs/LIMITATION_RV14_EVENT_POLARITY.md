# RV14 event-polarity appraisal — measured negative result

**Card:** `t_98e02c67` (split out of FIX-RV-14 `t_bbf781d0`)
**Verdict:** the capability the card asks for does **not** exist in RAVANA and could
**not** be built from the available representation. The limitation is now pinned by
tests rather than left as tribal knowledge.

Everything below is from real in-process runs (`RAVANA_OFFLINE=1`, GloVe warm, source
tree bound and asserted — see `tools/rv14/bindtree.py`). Nothing is estimated.

---

## 1. The premise is confirmed

`_update_emotion` (`ravana/src/ravana/chat/engine_memory.py:3469`) computes stimulus
valence by set-membership against `_AFFECT_LEXICON_BASE`:

```
sv = 0.0
if words & positive: sv += 0.4
if words & negative: sv -= 0.4
```

Measured on a **fresh held-out probe set** (18 adverse / 20 benign, written for this
card — deliberately *not* the utterances quoted in the card body, which were the
development set of two prior rounds):

| | adverse | benign |
|---|---|---|
| stimuli scoring exactly `sv = 0.0` | **8/11 probed** | 7/12 probed |

So an adverse event whose speaker volunteers no emotion word is genuinely
indistinguishable from a benign one. The card's premise holds. `emotion.state.valence`
is the emotion engine reading the same word list, not an independent appraisal.

## 2. Baseline behaviour is worse than the card reports

Full `process_turn` routing, held-out probes:

```
empathy recall on adverse : 1/18
empathy FP on benign      : 2/20
```

Both false positives are the *same* defect seen from the other side — `i passed my
driving test on the first try` and `my car passed its emissions test` are met with
*"i'm so sorry about your test. that's a real loss, and it hurts."* The empathy path
is firing on `passed` without checking the predicate's polarity.

**8 of 18 adverse disclosures get the bare degenerate `"noted."`** — the exact
user-visible complaint FIX-RV-14 was opened for, at 44%.

The card's own figure was 5/15 at 0/19. The collapse to 1/18 on unseen probes is the
third consecutive round to report a number that does not survive held-out measurement.

## 3. Five candidate signals, all measured, all fail

| signal | best zero-FP recall | benign FP |
|---|---|---|
| `evaluative_polarity` peak over content words | 18/18 | 19/20 |
| `classify_cause` confidence | 18/18 | 20/20 |
| GloVe mean vs affect seed | 18/18 | 19/20 |
| GloVe peak vs affect seed | 18/18 | 19/20 |
| goal-congruence (`cos(predicate, goal vector)`, AUC) | — | **AUC 0.331, inverted** |

The incumbent target — better than **5/15 recall at 0/19 FP** — is not met by any of
them. Every candidate that achieves full recall does so by selecting nearly the whole
benign set.

Goal-congruence (the relative appraisal Lazarus/Scherer actually point at) was the most
promising idea: read polarity off opposition to goals in RAVANA's own stance store,
which carries no polarity vocabulary at all. It was run honestly — nine emotion-free
goal statements fed in first, stances mined, probes fired with no probe used to fit.
It is **inverted**: benign probes sit *closer* to the goal vector (mean cos 0.608) than
adverse ones (0.502). Goal-relevance does not track goal-frustration here.

## 4. The decisive negative result

The strongest possible version of the question: if an **oracle** — one allowed to fit a
separating hyperplane directly on labelled probe vectors, strictly more information than
any seed vocabulary could carry — still fails to generalise, then no linear readout of
these embeddings can do this job.

400 random 70/30 splits, threshold chosen on the train half only, scored at the card's
zero-false-positive operating point:

```
nearest-centroid   held-out recall@0FP mean=0.106  median=0.000  held-out AUC=0.263
ridge(lam=1)       held-out recall@0FP mean=0.225  median=0.200  held-out AUC=0.305
```

Both held-out AUCs are **below chance**. The in-sample nearest-centroid AUC of 0.856 is
**indistinguishable from noise**: 400 label permutations give mean AUC 0.837 (sd 0.040),
`p(real > shuffled) = 0.34`. It is overfit, not signal.

> A first attempt reported an in-sample AUC of 1.000 — and 1.000 for every label
> permutation too, because least-squares with n=38 < d=64 interpolates. **An AUC that
> cannot be destroyed by shuffling the labels is an artefact of the fit.** That number
> was discarded; everything above is regularised or centroid-based.

**Conclusion:** adverse/benign event polarity is not linearly decodable from the
projected GloVe representation of these utterances. Any gate that achieved the target
recall on this data would have to be carrying the label in a word list.

## 5. What was NOT shipped

No change to `ravana/src/`. The diff is tests and this document.

This is deliberate. The honest fallback for an unrecognised adverse disclosure is a
grounded stored-fact ack, and it is measurably better than the alternative: shipping a
gate that trades ~8 extra false positives (sympathising with a passport renewal) for
recall is not acceptable, and the card says so. An honest flat fallback beats fake
depth.

The one thing the measurement *does* justify fixing is narrower and is **not** an event-
polarity gate: `passed`/`failed` are treated as suffering words by the empathy path, so
a success is consoled as a loss. That is a genuine defect in the emotion lexicon's use,
it is orthogonal to event polarity, and it belongs in its own card.

## 6. Reproducing

```bash
cd C:/Users/Likhith/Documents/Projects/ravana
for s in baseline_probe candidate_scan cv_test goal_congruence; do
  RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe tools/rv14/$s.py
done
RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest tests/test_rv14_*.py -q
```

**Measurement discipline learned here** (it changed a conclusion): `emotion.update()`
applies a decay EMA, so a sequential probe run measures residue from previous turns.
The first baseline run aborted on an assertion that pre-turn valence had drifted to
`-0.192` by probe 8 — exactly the artefact the card warned about. The harness now
records the per-turn delta from a verified-baseline pre-turn state, and
`tools/rv14/bindtree.py` strips the venv's editable-install finder so a worktree run
cannot silently measure the main checkout.

## 7. Pins shipped

`tests/test_rv14_valence_ceiling.py` — pins (a) the affect-lexicon premise, (b) that
`evaluative_polarity` still needs a ≥30% benign FP rate to be usable as a gate, (c) that
no hand-listed event-polarity vocabulary exists (AST scan).

`tests/test_rv14_eventive_routing.py` — pins that the eventive feature is **constant**
across the probe set, so it has zero variance and carries zero information about
valence. Pure Python, cannot rot. Plus an AST guard against an eventive admission
vocabulary shipping.

All three behavioural gates were **proven to go red** by deliberate sabotage
(affect-lexicon event words, a module-level `ADVERSE_WORDS`, a module-level
`EVENTIVE_ADMISSION_WORDS`) and the source tree restored byte-for-byte afterwards. A gate
that has never failed is not a gate.

If a future round ships a real signal, these tests are expected to fail — with messages
saying to re-measure on a fresh held-out probe set first, because two prior rounds'
numbers collapsed under exactly that test.