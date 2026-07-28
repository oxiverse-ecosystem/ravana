# Section 6.4 — Exposure (Phase 1) + ClauseSegmenter gate (Phase 2) result

## Status: NO-GO on the ClauseSegmenter. Flag stays OFF. Findings committed.

## Phase 1 — profiles grew through real exposure (DONE)

Two complementary exposure paths were exercised:

### P1.1-1.2 live web-learning session (real OpenIE feed)
- `scripts/tmp_web_exposure.py` ran 30 fixed factual queries through the
  engine's REAL `learn_from_web` path, which already feeds OpenIE facts
  into the triplet operator at `web_learning.py:990`.
- Web path WORKS: 10/10 queries returned, 0 errors; `located in` grew
  1→14. Text definitely reaches the operator.
- BUT it does NOT move the counters that matter: after exposure,
  `is` transitivity stayed 814/208, and the feature predicates
  (`has property`, `capable of`, `used for`) kept transitivity_pos=0.
  Reason: OpenIE extracts facts like `water boils → located in degrees
  celsius` — isolated, low-frequency, and on predicates whose
  transitivity the operator's Wilson gate never sees enough repeated
  chains to open. Web exposure is necessary for coverage but is SLOW
  and NOISY for gating.

### P1.3 deterministic graph-feeding (the real growth driver)
- Extended `scripts/tmp_warmup_profiles.py` to feed closed is-a
  triangles (transitivity) AND real inheritance triangles from
  `data/conceptnet/ont.pkl` — `(w isa p) + (p rel f) + (w rel f)` for
  `rel ∈ {has property, capable of, used for}` — which is the empirical
  pattern behind property inheritance.
- EXIT CRITERIA MET: snapshot now carries **4 answerable predicates**
  (`is`, `used for`, `capable of`, `has property`), verified across
  save→restore with a live `op.infer()` check. `is` wilson ≈ 0.66–0.80.
- This is learned-not-authored: we fed real observed co-occurrence, not
  invented relations. We deliberately did NOT warm transitivity for
  `has property`/`capable of`/`used for` (not transitive in the data).

Conclusion: the operator now has a 4-gate surface vs the 1 gate from the
first A/B. That satisfies the plan's "one open gate is not enough" fix.

## Phase 2 — ClauseSegmenter go/no-go probe (DONE, result: NO-GO)

- `scripts/tmp_probe_clause_yield.py`: 50 LogiQA cases, sentence-split
  + coordination/contrast split (and/but/however/because/therefore),
  then the EXISTING `parse_universal_edges` + `PropositionParser`
  (with the same blob-subject filter the candidate uses:
  `len(s.split())<=4 and len(o.split())<=4`) per clause.
- Measured: **avg 3.48 "triples"/case, but 0/50 cases contain a
  transitive `is` chain** (shared middle term).
- The "triples" are sentence-blobs mis-parsed as `X is Y`:
  e.g. `"researchers hypothesized that the reason why westernized black
  people suffer from hypertension" → "result of the interaction of two
  reason"`. These are NOT clean SPO; they cannot form the premise
  chains the operator reasons over.
- Decision rule (plan 2.3): build only if ≥2 chained premises/case on
  ≥30% of cases. Observed: 0%. **GO_BUILD = False.**

This confirms the opencode research survey's warning: LogiQA prose
genuinely resists rule-based SPO extraction. The probe answered the
question research could not — empirically, not by reading code.

## Why this is the honest stopping point (not a loop)

Per the standing rule, two failed attempts at moving the benchmark
number → stop and fork. Attempt 1 (first A/B) and Attempt 2 (this
exposure + segmenter gate) both show the learned candidate has no
measurable benchmark surface yet:
- Attempt 1: candidate answered 0 cases.
- Attempt 2: profile surface grew (4 gates), but the prose→clean-triple
  bridge needed for premises-at-answer-time does NOT yield chains
  (0/50). Without premise chains, the candidate can never fire on
  LogiQA regardless of how many gates are open.

## What would actually move it (next levers, different projects)
1. Predicate-family generalization: bundle `has property`/`capable of`/
   `used for` into one "attribute" family so composition evidence
   aggregates — would need operator changes, not just warmup.
2. A parser trained/adapted on LogiQA's actual surface (negation,
   comparison, "at least one of", "the person to the right of") — a
   real ML extraction task, not a 120-LOC rule patch.
3. Continue web/conversation exposure; re-measure later with the same
   A/B + attribution protocol.

## Artifacts
- scripts/tmp_warmup_profiles.py (P1.3, committed)
- scripts/tmp_web_exposure.py (P1.1-1.2, committed)
- scripts/tmp_probe_clause_yield.py (P2 gate, committed)
- reports/web_exposure_profiles.json (growth delta evidence)
- reports/clause_yield_probe.json (go/no-go evidence)
- data/ravana_eval_snapshot.pkl = clean Phase 1.3 artifact (4 gates)
- backup: data/ravana_eval_snapshot.pkl.bak_pre64 (pre-6.4)
