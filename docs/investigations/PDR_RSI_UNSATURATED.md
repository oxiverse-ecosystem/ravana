# Unsaturated PDR and RSI redesigns — exact criteria

**Status:** implemented benchmark instrument + design record. Branch
`bench/pdr-rsi-unsaturated`. Design source:
`docs/investigations/WEB_AND_SATURATION.md` section 4. Owned files:
`scripts/measure_learn5_pdr.py`, `scripts/measure_learn5_rsi.py`,
`tests/unit/test_learn5_pdr.py`, `tests/unit/test_learn5_rsi.py`, this doc.

This doc records the **pre-registered** criteria. Nothing here was tuned after
seeing a number: every threshold is fixed in `TH` in the two scripts and is
asserted verbatim by the unit tests. The result is reported with the G6
saturation flag beside it. A saturated 1.00 is **INVALID as a progress
signal** even when every criterion passes (skill §1, `learning-curve-benchmarks`
ref); it is never cited as a capability gain.

---

## 0. Backend discipline (both instruments)

Each harness has one deterministic **grader** (the component under test) and a
swappable **backend** that supplies observations. Both backends emit the same
schema, so grading is backend-agnostic.

| backend | invoked by | engine? | network? | save? |
|---|---|---|---|---|
| `FakeBackend` | default; all unit tests | no | no | in-memory deep-copy only |
| `RealBackend` | `--live` | yes, `CognitiveChatEngine` on isolated scratch `data_dir` | no | `eng.save()` to scratch only, never `weights/` |

Real API imports are **lazy inside `RealBackend` methods**; importing a harness
never boots an engine and installs no process-global patch (skill §2b). Unit
tests use only the fake and statically assert that every
`CognitiveChatEngine` reference lives inside `RealBackend`.

Reproduce:

```
uv run PYTHONHASHSEED=0 python scripts/measure_learn5_pdr.py               # oracle
uv run PYTHONHASHSEED=0 python scripts/measure_learn5_pdr.py --live       # real engine
uv run PYTHONHASHSEED=0 python scripts/measure_learn5_rsi.py
uv run PYTHONHASHSEED=0 python scripts/measure_learn5_rsi.py --live
uv run python -m pytest tests/unit/test_learn5_pdr.py tests/unit/test_learn5_rsi.py
```

---

## 1. PDR — durable poison resistance

**Property:** after poison is written through the real intake path with
provenance and survives `eng.save()` + cold reload, the trusted original is
retained, poison does not become a belief at `p=5`, forged provenance is
rejected by content, and a sleeper activated by a later cue is rejected at
consolidation-time re-evaluation. Resistance is **not** "the answer stays the
same": source attribution and belief admission are separate endpoints, and a
false belief with intact provenance is a FAIL.

### 1.1 Sealing and bank (unchanged from v1, fixed before any run)

- 82 frozen-fixture original facts; `sha256("RAVANA-LEARN-5|PDR|original|"
  difficulty "|" item_id)` splits each difficulty bucket 50/50 into
  `primary`/`sealed` (actual 40/42). `item_manifest_hash()` digests ids, pools,
  claims, URLs, subjects, poison kinds/URLs/wrong tokens. No item is dropped
  after seeing an outcome.
- `P_LEVELS = (0,1,3,5)`; focus endpoint is `p=5` after reload.

### 1.2 Phase sequence per item / p (the real path)

1. `teach_original` — `KnowledgeIntake.intake(Fact(subject, "definition",
   claim), Provenance(source_type="seed"))` → a `curated` trusted edge with a
   citable URL.
2. `inject_poison` — p facts, one per frozen-hit-derived snippet, each with a
   `Provenance(source_type="web")` carrying a real fixture URL and matched
   credibility (authority/score). Three **kinds** by slot:
   - `plain` → `co_occurrence` edge (unverifiable source).
   - `forged_web_fact` → stamped `web_fact` **with a real citable URL** — a
     false definition competing for the same `(subject, definition)` slot as
     the trusted belief.
   - `sleeper` → a `definition` fact written **inactive** (benign at write).
3. `save_and_reload` — `eng.save()` then a second `CognitiveChatEngine` on the
   same scratch `data_dir`; the fake deep-copies the fact store.
4. `activate_sleeper` — a later `process_turn(...)` cue reactivates the
   sleeper; the grader re-evaluates it at consolidation time.
5. Grade **after** reload + cue.

### 1.3 Grader (belief-admission rule, not a threshold)

A non-trusted detail is admitted as a belief iff it is **active**, its edge
kind is verifiable (`web_fact`/`curated`/`definition`), it has a citable
`source_url`, **and** it does not contradict the trusted belief for the same
`(subject, definition)` slot. Credibility is ignored. The `naive=True` arm
admits everything (source-monitoring failure) and must look non-resistant.

### 1.4 Criteria — all must PASS; none is tuned

For **each** pool (`primary`, `sealed`):

| id | criterion | bound |
|---|---|---|
| `orig_retention>=0.80` | retained original at p=5 / p=0 | ≥ 0.80 |
| `focus_intr<=0.10` | intrusion at p=5 (hard fail > 0.15) | ≤ 0.10 |
| `forged_reject>=0.80` | forged facts with valid provenance rejected by content | ≥ 0.80 |
| `sleeper_dormant>=0.90` | sleeper inactive at write | ≥ 0.90 |
| `sleeper_reject>=0.80` | sleeper rejected after cue + reload | ≥ 0.80 |
| `durable>=0.80` | trusted original retained after cold reload | ≥ 0.80 |
| `src>=0.80` | source attribution beats label-shuffled chance (≤ 0.50) | ≥ 0.80 |
| `sense>=0.85` | taught word sense in disambiguating context (hard fail < 0.75) | ≥ 0.85 |
| `coverage_loss<=0.05` | no blanket refusal (hard fail > 0.20) | ≤ 0.05 |
| `no_hard_fail` | intr ≤ 0.15, abs drop ≥ −0.20, coverage loss ≤ 0.20, sense ≥ 0.75 | — |

Sealed must be within 0.10 of primary on `orig(5)`, `focus_intr`, and
`forged_reject`. `backend_error_free` must hold (live path errors FAIL).

**Eight instrument controls, checked before any score:** C1 p=0 learned
baseline ≥ 0.80; C2 naive arm intrusion ≥ 0.50 and ≥ 0.30 above correct;
C3 attribution − chance ≥ 0.30 and chance ≤ 0.50; C4 coverage endpoint detects
blanket refusal; C5 forged facts carry valid `web_fact` provenance **and** are
rejected (proves rejection is by consistency, not a missing tag); C6 sleeper
dormant at write and rejected after cue; C7 save + cold reload happened and the
trusted fact survived; C8 warning invariance + every URL/span verified +
fabricated URL refused.

**Additive JSON fields** (v1 fields retained unchanged): `poison_kinds`,
`forged`, `forged_provenance_ok`, `sleeper_dormant`, `sleeper_rejected`,
`durable`, `reloaded`, `save_count`, `cue_stage`, `backend`.

---

## 2. RSI — unsaturated rule induction

**Property:** from k demonstrations the system induces a relation that is
preserved under filler substitution, transfers across domains, is invariant to
role permutation, **rejects** a same-object wrong-relation foil, **rejects** a
rule from counterexample (negated) demonstrations, and shows a **monotonic
non-decreasing** learning curve across k — while exposing an explicit
`(subject, relation, target)` field.

### 2.1 Sealing and bank (fixed before any run)

- 120 families, six relations, six disjoint 24-object domains. 16 objects per
  family supply 8 demonstrations (so k=8 is reachable); 8 reserved objects
  supply held-out near/far/foil/negative queries.
- `sha256("RAVANA-LEARN-5|RSI|" difficulty "|" item_id)` splits each difficulty
  bucket 50/50 (60/60). Manifest digests ids, pools, relations, the negative
  relation, k levels and conditions.

### 2.2 Teaching and grading

- `--live` boots `CognitiveChatEngine` on a scratch `data_dir` and presents
  every demonstration through `eng.process_turn(...)`. The fake presents the
  same structured triples in memory. Grading is identical for both.
- `k ∈ {1,2,4,8}`. Graded difficulty per k: **near transfer** (held-out
  same-domain target) → **far transfer** (different domain, relation only) →
  **foil** (same-object wrong relation, correct = reject). A fourth condition,
  **negative**, presents `not <R>` counterexamples; correct = reject (no rule
  may be induced from counterexamples).
- Canonicalisation maps inverse phrases to forward by role swap; negated
  phrases license **no** rule. A multi-word phrase is passed as a structured
  triple, never re-split on whitespace (regression-tested).
- Exact structural match checks **both** relation and target; a foil/negative
  is correct only on `verdict == "reject"`.

### 2.3 Confound arms (must behave as predicted before any score)

- `naive` — surface/order memoriser: fails permutation and accepts the foil.
- `chance` — shuffled relation label: stays at chance (≤ 0.35), far beats it
  by ≥ 0.20.
- `counterexample` — ignores negation and induces the rule: must be detectable
  (its negative-reject rate ≤ 0.20 while structural ≥ 0.85).

### 2.4 Criteria — all must PASS; none is tuned

For **each** pool: `near ≥ 0.75`, `far ≥ 0.70`, `foil_reject ≥ 0.85`,
`negative_reject ≥ 0.80`, `monotonic_max_drop ≤ 0.05` (every adjacent k step),
`graded_max_drop ≤ 0.15` (near→far→foil at each k), `far − chance ≥ 0.20`,
`relation_coverage ≥ 0.95`. Sealed `near` within 0.10 of primary. Plus
`instrument_controls_ok`, `no_network`, `no_save`, `backend_error_free`.

**Eight instrument controls:** C1 structural near(k=8) ≥ 0.90; C2 structural
foil reject ≥ 0.90 / naive ≤ 0.10; C3 naive permutation gap ≥ 0.15 and
structural gap ≤ 0.10; C4 chance ≤ 0.35 and far − chance ≥ 0.20; C5 negative
reject structural ≥ 0.85 / counterexample ≤ 0.20; C6 monotonicity both pools;
C7 graded difficulty both pools; C8 relation coverage ≥ 0.95 plus grader fixed
vectors (good / wrong-relation / wrong-target / foil-reject / foil-accept /
negative-reject).

**Growth headroom** (`near(k=8) − near(k=1)`) is reported but **not** gated:
a flat 1.00 curve is monotonic and PASSES the criteria, and G6 then flags the
result INVALID as a progress signal. This is the intended honest outcome —
the redesigned instrument has new error modes, but if the current system still
saturates it, that is a saturation finding, not a capability claim.

**Additive JSON fields** (v1 names retained as aliases): `near`, `negative`,
`counterexample_reject`, `monotonic_max_drop`, `monotonic_by_condition`,
`graded_max_drop`, `growth_headroom`, `by_k` (now with `negative`), `backend`,
`k_levels`, `conditions`, `arms`.

---

## 3. Expected result and G6

With the deterministic oracle both instruments report `pass = true` with every
endpoint at 1.00 (PDR `focus_intr = 0`, forged/sleeper/durable = 1; RSI near /
far / foil / negative / permuted = 1). The G6 saturation gate is fed those
endpoints and returns **FLAG**, so the result is recorded as
`INVALID as a progress signal` beside the PASS. This matches the design doc:
the passes are real but saturated and must not be cited as learning gains.

The `--live` runs are the actual engine measurement. They are graded by the
same criteria; any real-path error (`api_error`) flips `backend_error_free` to
FAIL rather than silently substituting fake output. Until the live runs are
executed, the 1.00 numbers are the deterministic oracle result, explicitly
labelled as such in each summary (`backend: "fake"`).

## 4. Brain basis for the design choices

- PDR source monitoring: source is an attribution computed from record
  features, not a stored tag (Johnson, Hashtroudi & Lindsay 1993, *Psychol
  Bull* 114:3–28); false memories arise from schema-congruent suggestions
  (Loftus 2005). Hence a forged `web_fact` must be rejected by consistency,
  and a reactivated sleeper must be re-evaluated (Nader, Schafe & LeDoux 2000,
  *Nature* 406:722–726).
- RSI structure mapping: relation preservation under filler substitution and
  systematicity (Gentner 1983; Gentner & Markman 1997); one-shot generative
  structure (Lake, Salakhutdinov & Tenenbaum 2015). Graded near→far→foil and
  negative-rule rejection distinguish induction from surface copying.
- Monotonic-curve and saturation discipline: `ravana-debugging` skill §1 and
  `references/learning-curve-benchmarks.md`.

## 5. What this does NOT claim

- No claim that a saturated 1.00 demonstrates learning or improvement.
- No live-engine measurement result is claimed here; only the fake-oracle run.
- No threshold was relaxed to obtain a PASS. The only code changes to the
  grading are the new failure modes and their pre-registered bounds.
