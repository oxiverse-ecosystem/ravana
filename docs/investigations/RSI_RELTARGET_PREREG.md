# RSI relation-target evidence channel — pre-registration

Branch: `invest/rsi-reltarget-evidence` (worktree `ravana_rsi_reltarget`, based on `main`).
Written BEFORE any licensed-source data was read or any coverage number computed.

## Goal

Build a NEW, independent, licensed **relation-target** evidence channel for RSI:
records of the form `(subject, relation, target)` asserted by a real external
source, with provenance, credibility, and an explicit negation veto. It must be
a *pure, default-off* module: no engine boot, no store write, no network at
import, and **zero** use of `gold`, `type_sets`, `world_model`, `answer_key`,
`gold_target`, or item-bank domain labels.

This is prompted by `docs/investigations/RSI_F7_FLAT_ROOTCAUSE.md`: M2 supplied
type labels (`is_a`) while RSI grades relation-target objects; 473/480 rows have
no admitted label in the displayed option set. The missing input is a subject →
relation-target channel for held-out subjects.

## Source and licence

Proposed source: **Wikidata** (CC0-1.0, https://creativecommons.org/publicdomain/zero/1.0/),
queried at `https://query.wikidata.org/sparql`. Each record carries the statement
URI (QID + property + value QID), the item/value source URLs, the Wikidata rank
(preferred/normal/deprecated) as the credibility signal, and the query URL as
provenance. This is a *different property set* from M2's P31/P279 types
(the new channel uses causal/part-whole/order/opposition properties).

Mapping (RSI family → Wikidata property):
- CAUSE → P828 (has cause), P1542 (has effect)
- PART_OF → P361 (part of), P527 (has part)
- PRECEDES → P155 (follows), P156 (followed by)
- OPPOSITE → P461 (opposite of)
- SUPPORTS, RESEMBLES → no clean Wikidata property; reported as uncovered.

A source is admissible only if the fetched text is a genuine assertion in that
source with a resolvable statement URL. No candidate fact may be authored to
fill a gap.

## Pre-registered acceptance criteria (ALL must hold for GO)

- **C1 LICENCE** — Every admitted record carries `license=CC0-1.0`, a
  `source_url` and a statement/provenance identifier. Audit: 100% of admitted
  records carry all three.
- **C2 REAL ASSERTION** — Every admitted record's target label is taken from
  the source's own object label, never from the RSI item bank. Relations must be
  in the six families above, not `is_a`/`type_of`/`instance_of`.
- **C3 SUBJECT COVERAGE** — ≥ 0.80 of the 141 RSI query subjects have ≥ 1
  admitted assertion (any relation family) in the frozen snapshot.
- **C4 CANDIDATE-TARGET COVERAGE** — For the 240 `exact`+`far` query rows,
  ≥ 0.50 have ≥ 1 admitted assertion whose target is one of the *displayed*
  candidates. This is the necessary condition for the channel to be able to
  move the RSI endpoint at all.
- **C5 ANSWER-KEY INDEPENDENCE** — AST/source scan of the module shows no
  reference to `gold`, `gold_target`, `type_sets`, `world_model`, `answer_key`,
  `counterexample_target`; and the channel's pick surface is invariant to a
  *within-option-set permutation* of the answer key (because it never reads
  the key). A reader that consumes the channel and a permuted key must show
  pick accuracy collapse to ≤ 0.25 while the channel's admitted records are
  byte-identical.
- **C6 NEGATION VETO** — Injecting a provenanced, explicitly negated
  `(subject, relation, target)` record flips that candidate from admitted to
  vetoed in 100% of a fixed 20-row test set; a wrong-tag SHAM arm admits 0.
- **C7 CANDIDATE-ORDER INVARIANCE** — Shuffling the displayed candidate order
  leaves the reader's pick (and admitted set) unchanged in 100% of rows where
  a pick exists.
- **C8 NO REGRESSION / DEFAULT-OFF** — The module is pure (no import-time
  network/engine/store side effect), default-OFF on an explicit env flag, and
  touches no gate, threshold, item, prompt, scorer or production file.
  Flag-OFF output is byte-identical to baseline.

## Decision rule

GO (ship the default-off pure module) **only if C1–C8 all pass**.
If C3 or C4 fails, the channel cannot carry the RSI endpoint: record a NO-GO
with exact coverage numbers, keep the module + frozen snapshot as a negative
result, and do **not** change any gate/item/scoring. A NO-GO is a result, not a
reason to fabricate channel facts (knowledge-channel-audits.md).

## Controls to run regardless of GO/NO-GO

1. Permutation *inside* each displayed option set (replace the key with a
   different candidate from that same set; assert every key changed).
2. Negation veto + wrong-tag SHAM (call-matched).
3. Candidate-order shuffle.
4. Flag-OFF ≡ baseline equivalence and import-time purity.

## Honest boundary

RSI's `world_model` is authored from hash-selected object pairs; the query
"gold target" is `reserved[1]`, an arbitrary same-domain word, not necessarily
a real-world relation fact. If C4 fails, that is evidence that the benchmark's
key is not a deployable relation-target fact — not evidence that the reader is
broken.
