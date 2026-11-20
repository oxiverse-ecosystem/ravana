# RAVANA-LEARN-5 / WFA — measured result

**Status: PASS (with a saturation caveat).** This document records an actual,
completed run of the WFA measurement harness on the frozen web-replay boundary.
It changes no weights, no gate, and no fixture. Base commit `4ecbb79`; harness,
tests and this report are working-tree additions.

## 1. What was measured

WFA (Web Fact Acquisition and Attribution) is the source-monitoring component
of RAVANA-LEARN-5. The mechanism is grounded in source-monitoring theory:
externally derived knowledge is not a tag read off a memory, but a decision at
retrieval that must be checked against the actually encountered source
(Johnson & Raye, 1981, *Psychological Review*; Johnson, Hashtroudi & Lindsay,
1993, *Psychological Bulletin*). A felt information gap motivates a **bounded**
read whose epistemic value is limited (Loewenstein, 1994, *Psychological
Bulletin*; Friston, 2010, *Nature Reviews Neuroscience*). Hence the endpoint is
an **attributed answer**, and read count is a **cost**. Full design rationale:
`docs/BRAIN_LEARN5_DESIGN.md` §4 (lines 265–349); reuse decisions:
`docs/LEARN5_CODEMAP.md` §3.2.

The system under test is the WFA attribution layer implemented in
`scripts/measure_learn5_wfa.py` (`WFAComponent`, line 405). Given a gap question
and at most two frozen sources it must either (a) emit an externally-tagged
answer with a machine-verified URL and a **verbatim entailing** evidence span,
or (b) abstain. Arms: gap/attribution, answer-only (no web), decoy (real URL,
non-entailing span), closed-web (no source), internal-source (reasoning claim
must carry no web tag).

## 2. Instrument and reuse (what is *not* reinvented)

- Frozen replay boundary reused byte-for-byte from
  `scripts/measure_fronts_web_frozen.py`: `_frozen_search`, `_norm_q`, the
  blocked `urlopen`/`_call_api`/`save` tripwires and their counters, and the
  checked-in `tests/fixtures/web_snapshots/web_fixtures.json` (1087 queries,
  `captured_at=2026-11-19T18:40:01Z`, provider
  `intentforge(localhost:4000) -> local_api -> duckduckgo -> oxiverse`).
- Shared `scripts/front_gates.py` `gate_saturation` (G6) is reused for the
  informativeness flag; thresholds G1–G6 are untouched.
- Every gap claim and URL is **read out of a frozen fixture hit**, not written
  by the harness. `validate_items()` (line 371) aborts the run unless each gap
  claim is a verbatim substring of fetched content, each gap URL is in the
  frozen allow-list, each decoy is on-topic but **non-entailing**, and every
  closed-web query is absent from the store.
- Split is fixed before any run: `sha256("RAVANA-LEARN-v2|WFA|kind|difficulty|id")`
  (`_seal`, line 278). Item-bank manifest SHA-256:
  `a3482de70573b96d61394563fb6790686dc70f2e5f5ee1483685bdcf1bf7b45f`.
- Pre-registered thresholds (line 61), not tuned: `WFA_valid ≥ 0.70`,
  abstention ≥ 0.90, `WFA_valid − answer_only ≥ 0.20`, decoy rejection ≥ 0.85,
  sealed ≥ primary − 0.10, zero fabricated/unfetched URLs, zero attribution on
  closed-web items, read count reported.

## 3. Measured numbers

Command: `uv run python scripts/measure_learn5_wfa.py` (exit 0). Artifacts:
`scratch/learn5_wfa_probes.jsonl` (144 per-item rows),
`scratch/learn5_wfa_summary.json`, `scratch/learn5_wfa_run.log`.

| Metric | PRIMARY | SEALED | Criterion | Result |
|---|---|---|---|---|
| gap items (n) | 32 | 32 | ≥ 60 per design §4.5 | **shortfall** (see §5) |
| `WFA_valid` | 1.000 | 1.000 | ≥ 0.70 both | PASS |
| answer-only accuracy | 0.000 | 0.000 | — | — |
| `WFA_valid − answer_only` | 1.000 | 1.000 | ≥ 0.20 both | PASS |
| closed-web abstention (n=15/15) | 1.000 | 1.000 | ≥ 0.90 both | PASS |
| decoy rejection (n=14/16) | 1.000 | 1.000 | ≥ 0.85 both | PASS |
| internal no-web-tag (n=10/10) | 1.000 | 1.000 | ≥ 0.90 both | PASS |
| reads (cost) | 32 | 31 | reported | 63 total, 63 search calls |

**Hard-fail counts (all zero):** fabricated URLs 0, unfetched URLs 0,
non-verbatim spans 0, attribution on closed-web 0. Machine checks confirmed all
64 attributed rows carry a URL present in the frozen fixtures, a URL actually
returned by a frozen read this run, and a span that is a literal substring of
that fetched content. `network_attempts=0`, `save_attempts=0`, `fixture_miss=0`.

**Instrument controls (all pass):** known-good attributed gap is valid; decoys
all rejected; a fabricated URL is refused by the allow-list; a non-entailing
span yields nothing; closed-web abstains with no URL; internal claim carries no
web tag. `uv run python -m pytest tests/unit/test_learn5_wfa.py -q` → **16 passed**.

**Verdict:** all 19 pre-registered decision rules pass → **PASS**.

## 4. Per-item audit trail (hand-checked inputs and outputs)

The JSONL logs `input`, `claim`, `output`, `url`, `span`, `verdict`,
`external_tag`, `reads` and all sub-flags for each of the 144 items. Examples:

- **gap g28** — input `what causes frost`; emitted URL
  `https://www.weatherstreet.com/weatherquestions/What_causes_frost.htm`, span
  `Frost forms in much the same way as dew forms.` (verbatim in that page);
  verdict `attributed`, external tag true, 1 read.
- **decoy d19** — input `how does blood circulate in the body`, claim “Blood is
  pushed around the body by rhythmic contractions of the spleen.” The fetched
  page is on-topic but contains no entailing span; output “I don't know — no
  entailing source”, no URL/span, verdict `abstain`.
- **closed-web c03** — input `how does qlarith weaving strengthen bulletproof
  fabric` (no frozen source); “I don't know — no reachable source”, no URL.
- **internal i03** — input `Do all triangles have three sides`; answered
  internally, `external_tag=False`, no URL.

## 5. Limitations — read these before citing the 1.00 as capability

1. **The front is saturated and therefore INVALID as a progress signal.**
   Shared G6 flags `wfa_valid=100` (recorded in the summary's `saturation_gate`
   as `FLAG`). A perfect score is expected: every gap item was authored so a
   clean entailing source exists, and every decoy/closed-web item was authored
   to be rejected. The number establishes that the attribution *mechanism*
   works with zero fabrication; it does not measure difficulty-graded growth
   and must not be used to claim learning.
2. **Entailment is partly tautological.** The gold claim is the very sentence
   selected from the source, and the rubric accepts a verbatim sentence whose
   content-token set contains the claim's content tokens. This validates
   span-grounding, but does **not** separately test the design's
   paraphrased-evidence control (same fact in different words). Surface
   memorisation and true entailment are not yet separated.
3. **The answer-only arm is a controlled parametric baseline, not a live-engine
   oracle.** `PARAMETRIC_STORE` (line 566) is empty and gap items are, by
   definition, absent from internal memory, so answer-only is 0.00 by
   construction. This rules out guessing on genuine gaps but cannot detect a
   model that happens to know a fact parametrically; a live offline engine
   baseline is future work.
4. **Sample size is below the design target.** Design §4.5 asks for n ≥ 60 gap
   items **per pool**; this run has 32 per pool (64 total), above the skill's
   ~20-probe hypothesis threshold but below the pre-registered design target.
   Decoy pools are 14/16 by difficulty-bucket splitting. Treat the pool
   comparison as a fixed-split replication, not a fully powered study.
5. **No engine boot, no live read.** The endpoint is attribution over the
   frozen provider; no `CognitiveChatEngine` process was involved, so this does
   not measure the engine's own web-answer path. That is deliberate (the
   codemap assigns WFA the frozen replay boundary) but bounds the claim.
6. **Closed-web items use an invented term (`qlarith`) to guarantee absence**
   from the 1087-query bundle. They test abstention cleanly but do not sample
   natural no-answer failure modes.

## 6. Concrete next work (not a ceiling)

Per the debugging skill, a saturated front and a shortfall are next work items,
not reported limits. The next step is authored **paraphrased-evidence** items:
gold claims phrased independently of the source sentence, graded by a
concept-level entailment rubric with a synonym/concept map, plus a live offline
engine baseline for the answer-only arm, and expansion to 60 gap items per pool.
No threshold was relaxed to obtain the current PASS, and none should be to
obtain the next number.

## 7. Artifact index

- Harness: `scripts/measure_learn5_wfa.py`
- Tests: `tests/unit/test_learn5_wfa.py` (16 passing)
- Per-item rows: `scratch/learn5_wfa_probes.jsonl` (144)
- Summary: `scratch/learn5_wfa_summary.json`
- Run log: `scratch/learn5_wfa_run.log`
- Fixtures: `tests/fixtures/web_snapshots/web_fixtures.json` (unchanged)
