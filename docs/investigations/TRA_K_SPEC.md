# TRA-K — knowledge-conditioned typed role acquisition (pre-registered)

## Declaration

TRA-K is a NEW benchmark, not a fix to TRA. A PASS means: *given an independent
semantic knowledge channel, RAVANA can classify an unseen subject's type under
source and foil controls*. It does **not** mean RAVANA learns types from eight
sparse exemplars alone. Original TRA and RSI verdicts are unchanged.

## Input channels (in priority order)

1. **External ontology (B):** fixed, provenance-stamped `is_a`/type edges for
   benchmark vocabulary from a snapshot independent of the item bank's
   `type_sets`/`world_model`. Coverage and overlap with the answer key must be
   measured before any score.
2. **Explicit ostensive teaching (A):** `"<subject> is a <type>."` as a real
   conversational turn. Positive control only — known to saturate.
3. **Frozen distributional corpus (C):** fixture-backed text containing test
   words, ingested through the real web path. Deferred unless B fails.

## Item schema

Based on TRA but with `channel_evidence` added: subject, type, taught exemplars,
channel-provided type evidence, candidates, foil, negative, domain/far-domain.
`channel_evidence` is the ONLY new input; it is recorded with provenance and
never derived from the gold target or the query subject's `type_sets`.

## Phase structure

P1 cold baseline; P2 k typed exemplars; P2c channel evidence teaching;
P3 query-only graded turn; P4 paraphrase; P5 save + cold reload; P6 immediate.
Teaching turns are audited, never graded.

## Gates (copied unchanged from TRA where applicable)

- Primary/sealed hash split, ≥40 families/pool, one k per item, 5 k levels.
- Instrument controls C1–C10, C12, C13 on the fake arm; C11 on live.
- Fake backend must FAIL C11; live backend must show real API counters.
- G6 saturation flag + M-A/M-B mutation control on saturated endpoints.
- Subject-independence control: same answer for an unrelated subject = HARD FAIL.
- Channel-independence audit: overlap with `type_sets`/`world_model` reported;
  any deployable path using them = contamination, HARD FAIL.

## Endpoints

near/far transfer at k=8, foil rejection, negative rejection, cold-reload
retention, growth (k=8 − baseline), permutation gap, coverage.

## Decision rule

Run fake arm first. If instrument controls fail → INSTRUMENT INVALID, fix wiring.
Then focused live (`TRAK_MAX_PER_POOL=20`). Only if focused controls pass and
endpoints are informative (not all 1.00/0.00) run the full 400-family arm.
No gate or threshold is ever changed to alter a verdict.
