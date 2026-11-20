# RAVANA-LEARN-5 / CAR — measured result

**Status: FAIL.** One of ten binding criteria failed: the primary adjacent-k
monotonicity gate (k=1 → k=2 delta = −0.50, bound −0.10). The headline k=8
deltas, both control arms, and cold-reload survival all passed on both pools.
Per `docs/BRAIN_LEARN5_DESIGN.md` §3.6 a single failed criterion is a CAR FAIL;
no threshold, rubric, or trust gate was relaxed to try to remove it.

**Measurement commit:** `43db6cd` (working tree contained the three CAR files
below, uncommitted). Host Windows, `uv run`, `PYTHONHASHSEED=0`,
`RAVANA_OFFLINE=1`. One process, one cold boot for all teaching, one `eng.save()`
into an isolated scratch `data_dir`, one cold-reload boot. Elapsed 1977 s
(~33 min); boot 1 = 4.07 s, cold boot 2 = 11.77 s.

## What was built

- `scripts/measure_learn5_car.py` — k∈{0,1,2,4,8} spaced teaching of a 2-hop
  trusted causal chain (`a→b→x`), held-out paraphrase probes, k=0 matched-
  familiarity control, definition-only control, identical-surface parroting
  probe, per-hop provenance census, delayed cold-reload confirmatory probe.
- `tests/unit/test_learn5_car.py` — 13 tests (all pass): hash rule, split
  stability/disjointness, Latin-square balance, paraphrase hygiene, Wilson /
  Fisher stats, gate census, def-only disjointness, traversal-budget behaviour,
  and fixed thresholds.
- This document.

Reused from `scripts/measure_learning_curve.py` (not rewritten):
`run_detector_controls`, `prepare_data_dir`, `trusted_chain` (as the gate
template), `trusted_chain_summary`, `ask`, `answer_verdict`,
`teach_conversational`, `teach_web`, `install_trace`, `label_map`,
`node_ids_for`, `causal_snapshot`.

## Instrument

**Headline metric.** `chain_i(stage)=1` iff, for subject X, a provenance-
trusted ≥2-hop causal chain is returned by
`grade_chain(c).verdict == 'assert' AND assertable_chain(c)` on the structured
`ExplanationChain` object — never a prose regex. `Δ_CAR(k) = cold chain rate at
k − baseline chain rate at k`. This is the graph-level structured-chain
endpoint endorsed by `docs/LEARN5_CODEMAP.md` §3.1 and the `ravana-debugging`
skill ("measure the decision, not the prose"). **Important caveat:** the
strict §3.1 wording ("the answer ... contains a chain") is *not* met — see
Limitations. The emitted-answer reach is reported as a separate endpoint.

**Hash split, frozen before boot.** `sha256("RAVANA-LEARN-5|CAR|<role>|<difficulty>|<item_id>")`,
sorted within (role, difficulty), first 50% PRIMARY / rest SEALED; k by Latin
square over hash-sorted items. 50 primary + 50 sealed teach items, 10 per k per
pool; 10+10 definition-only control items with subjects disjoint from all teach
subjects. Manifest hashes (canonical JSON, sorted keys):
- PRIMARY `b13e89a42fa9ac5176787733a59998bf7990fc23b1e94daa2422a4453c8e47bd`
- SEALED `18a606ae6387b0631c6cfe30f69b91ecbc9319518067e7d93c4e182d693020a0`
- DEF-ONLY `drawn from the same rule; subjects {texture … saltiness}`
The primary/sealed hashes are byte-identical to the invalid pilot (below):
the split was **not** changed after seeing results.

**Gates.** Instrument self-check before any curve: positive trusted 2-hop
detected = true, negative definition rejected = true, forward direction ok.
Per-hop provenance census at every probe: at cold reload, of 640 enumerated
chains for primary k=8 items only 9 passed `assertable_chain`, 631 were
rejected by the trust gate (sealed k=8: 7/576 accepted). 112/120 items had at
least one trust-rejected chain at cold reload — the gate is active, not
bypassed. No `eng.save()` on `weights/` (save path was the scratch data_dir
only).

## Measured results — primary and sealed side by side

Trusted-chain rate (graph-level, provenance-gated); `Δ = cold − baseline`.
Each cell n=10 item-families.

| k | PRIMARY base / post / cold | PRIMARY Δ_post / Δ_cold | SEALED base / post / cold | SEALED Δ_post / Δ_cold |
|---|---|---|---|---|
| 0 | 0.00 / 0.00 / 0.00 | 0.00 / **0.00** | 0.00 / 0.00 / 0.00 | 0.00 / **0.00** |
| 1 | 0.00 / 0.50 / 0.70 | +0.50 / +0.70 | 0.10 / 0.60 / 0.50 | +0.50 / +0.40 |
| 2 | 0.20 / 0.80 / 0.40 | +0.60 / +0.20 | 0.00 / 0.60 / 0.60 | +0.60 / +0.60 |
| 4 | 0.00 / 0.50 / 0.50 | +0.50 / +0.50 | 0.10 / 0.60 / 0.70 | +0.50 / +0.60 |
| 8 | 0.00 / 0.90 / 0.80 | +0.90 / **+0.80** | 0.00 / 0.40 / 0.50 | +0.40 / **+0.50** |
| def-only | 0.00 / 0.00 / 0.00 | — / **0.00** | 0.00 / 0.00 / 0.00 | — / **0.00** |

**Primary test (k=8 vs k=0, delayed cold reload):** 8/10 vs 0/10,
Wilson 95% CI [0.49, 0.94] vs [0.00, 0.28]; risk difference **+0.80**;
Fisher exact **p = 0.00071**. Sealed k=8 vs k=0: 5/10 vs 0/10 (Δ +0.50).
Sealed is 0.30 below primary — inside the design's −0.10 tolerance only if
compared as a raw Δ, but reported plainly here; no rubric edit was made.

**Secondary endpoint — answer reach (not the headline).** On the held-out
paraphrase, `answer_verdict.has_chain` was 0.00–0.10 in every arm; the emitted
answer did not surface the chain. On the *identical-surface* probe, 5/10
primary k=8 answers echoed both taught mechanism words (`a`,`b`), showing a
parroting signal in prose that is absent from the paraphrase probe.

## Criteria (fixed from design §3.6; machine-checked in the script)

| # | Criterion | Measured | Result |
|---|---|---|---|
| 1 | primary Δ_cold(k=8) ≥ 0.25 | 0.80 | **PASS** |
| 2 | primary k=0 Δ ≤ 0.05 | 0.00 | **PASS** |
| 3 | primary def-only Δ ≤ 0.05 | 0.00 | **PASS** |
| 4 | sealed Δ_cold(k=8) ≥ 0.25 | 0.50 | **PASS** |
| 5 | sealed k=0 Δ ≤ 0.05 | 0.00 | **PASS** |
| 6 | sealed def-only Δ ≤ 0.05 | 0.00 | **PASS** |
| 7 | primary no adjacent-k reversal > 0.10 | k1→k2 = **−0.50** (curve 0, .7, .2, .5, .8) | **FAIL** |
| 8 | sealed no adjacent-k reversal > 0.10 | k4→k8 = −0.10 (curve 0, .4, .6, .6, .5) | PASS |
| 9 | primary gain survives cold reload | cold +0.80 vs immediate +0.90 | PASS |
| 10 | sealed gain survives cold reload | cold +0.50 vs immediate +0.40 | PASS |

**Overall: FAIL (criterion 7).** Located cause, from per-item audit: two of
the ten primary k=2 items (`rust`, `earthquake`) already had a baseline trusted
chain (curriculum knowledge), capping their measurable delta; four k=2 items
(`amber`, `honey`, `magnify`, `stalactite`) had a chain in-process that was
absent after cold reload. A focused probe of the saved graph showed `amber`,
`magnify`, `stalactite` have *no* trusted chain even at traversal limit 1024 —
a genuine retention/acquisition failure, not a truncation artifact. `honey`'s
chain exists at limit 256 but not 64; the search budget was **not** raised
again after seeing this result, because doing so would be tuning after the
fact and would not repair the other three items.

## Per-item input / output (hand-checkable examples)

All 120 items' baseline/post/cold question, raw answer, chain object, gate
census, and identical-surface answer are in
`scratch/learn5_car_measured.json` → `records`. Representative rows:

- **PASS, primary k=2 `lightning`** — baseline Q `why does lightning occur?`
  → `hmm, i'm not totally sure about Lightning yet…`, chain=false. Cold Q
  `what is the mechanism behind lightning?` → `i don't really have a solid
  grasp on mechanism…`, yet graph chain `['deposition','flash','lightning']`,
  hops `curated/curated`, grade `assert`, trusted=true. The probe answer does
  not state the chain; the chain exists in the trusted graph object.
- **FAIL, primary k=2 `amber`** — baseline chain=false; immediate Q
  `how is amber produced?` chain=true (`resin→fossil→amber`, curated); cold Q
  `what is the mechanism behind amber?` chain=false, no trusted chain even at
  limit 1024. This is one of the reversals contributing to criterion 7.
- **DEF-ONLY `weight`** — `weight is a kind of mass and refers to a measure…`;
  baseline/immediate/cold all chain=false. Control stayed flat.
- **Identical-surface echo, primary k=8:** e.g. `amber` re-asked
  `why does amber happen? is it because resin causes fossil, and fossil causes
  amber?` → the engine echoed a *different* taught sentence
  (`you mentioned: "gum happens because resin causes hardening…"`), evidence
  that surface re-ask behaviour is not a reliable index of the trusted chain.

## Instrument repairs made after an invalid pilot (documented, not tuned)

A first full run (`scratch/learn5_car_measured_PILOT_INVALID.json`,
`scratch/car_full_PILOT_INVALID.log`) produced primary Δ8=+0.60, sealed Δ8=
+0.10, primary def-only Δ=+0.30 and a cold collapse. Instrument diagnosis:
1. **Confounded definition-only control:** 17/20 def-only subjects were also
   teach subjects (same graph node), so their "definition-only" chains came
   from causal teaching of the same subject. Fixed by replacing the def-only
   bank with 20 subjects disjoint from all teach subjects.
2. **Top-8 traversal truncation:** learned curated chains survived the save
   intact (e.g. `tornado`, `limestone`, `callus`, `blush` had grade=`assert`
   curated chains at `limit=64` that `limit=8` missed). The search budget was
   raised from 8 to 64 (`TRAVERSAL_LIMIT`). `grade_chain` / `assertable_chain`
   and every pass threshold are unchanged. Evidence:
   `scratch/car_cold_inspect.py`.
The pilot is retained as evidence and is **not** a reported result. The
primary/sealed hash split is identical between pilot and reported run; no
re-sealing was performed.

## Limitations

1. **Graph-level ≠ answer-level.** The headline is a structured chain object,
   not the generated sentence: answer reach on the held-out paraphrase was
   ≤0.10, and the "answer contains a chain" reading of design §3.1 is not met.
   This is the single largest gap between the measured number and the literal
   specification; it is not repaired by a rubric change.
2. **Power.** n=10 item-families per k per pool, versus the design's n≥40.
   The k=2 dip is item-composition-sensitive (two curriculum-known subjects,
   three genuine retention failures) and is a hypothesis at this n; the
   criterion is nevertheless binding and its failure is reported, not smoothed.
3. **Spacing is within-session** (exposures interleaved across eight rounds
   with matched neutral turns), not multi-session/multi-day. The design's
   spacing interval is a fixed, reported parameter, not tuned.
4. **Cold reload used one `eng.save()`** into an isolated scratch data_dir
   (`scratch/learn5_car_data`), per the task's explicit allowance; `weights/`
   was never a save target. Save/reload changed confidence values slightly
   (e.g. 0.42→0.48) but preserved `edge_kind` provenance.
5. **Single seed / one run.** `PYTHONHASHSEED=0`; no second full replication
   was run within the time budget. The design's same-config-twice
   reproducibility gate is therefore not satisfied here.
6. **Secular graph growth.** k=8 items are taught last and the graph accumulates
   hundreds of untrusted chains per subject (631/640 rejected at cold for
   primary k=8); the trusted chain is found only after a 64-wide search. The
   metric is sensitive to the traversal budget (see repairs).
7. **No statistics beyond the primary Fisher test.** The sealed comparison and
   trend test (Cochran–Armitage / mixed effects from §3.5) are not computed;
   Wilson intervals are reported for the primary k=8 vs k=0 proportions only.

## Citations (from `docs/BRAIN_LEARN5_DESIGN.md` §11)

- Ebbinghaus (1885/1913), *Memory: A Contribution to Experimental Psychology*
  — savings method; retention as a within-subject relearning difference.
- Murre & Dros (2015), *PLOS ONE 10*(7):e0120644, doi:10.1371/journal.pone.0120644
  — forgetting curve replication; savings decline with retention interval.
- Roediger & Karpicke (2006), *Psychological Science 17*(3):249–255,
  doi:10.1111/j.1467-9280.2006.01693.x — post-teaching probe must be a
  delayed retrieval test, not restudy; motivates the held-out paraphrase and
  cold-reload retest.
- Karpicke & Blunt (2011), *Science 331*(6018):772–775,
  doi:10.1126/science.1199327 — retrieval advantage on inference questions.
- Cepeda, Pashler, Vul, Wixted & Rohrer (2006), *Psychological Bulletin
  132*(3):354–380, doi:10.1037/0033-2909.132.3.354 — spacing is a real
  manipulation; motivates k as spaced exposures and the monotonicity gate
  that this measurement **failed**.

## Reproduction

```
cd C:/Users/Likhith/Documents/projects/ravana
PYTHONPATH=scripts PYTHONHASHSEED=0 uv run python scripts/measure_learn5_car.py
PYTHONPATH=scripts uv run python -m pytest tests/unit/test_learn5_car.py -q
```

Artifacts: `scratch/learn5_car_measured.json` (full per-item audit, manifest,
criteria), `scratch/car_full_v2.log` (run transcript), plus the retained
invalid pilot artifacts named above.
