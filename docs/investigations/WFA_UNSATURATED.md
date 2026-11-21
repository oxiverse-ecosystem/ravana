# Unsaturated WFA redesign — exact criteria

**Status:** implemented design + instrument spec. Branch `bench/wfa-unsaturated`.
Builds on `docs/investigations/WEB_AND_SATURATION.md` §4 and
`docs/BRAIN_LEARN5_DESIGN.md` §4.1–4.6. No engine run or pytest was performed
while writing this doc (static/code work only); the parent runs the suite.

## 1. Why the old WFA was G6-invalid

The previous `scripts/measure_learn5_wfa.py` reported 1.00/1.00 and was
flagged as a saturated (non-informative) progress signal. Four instrument
defects made that number vacuous:

1. **Tautological gold claim.** The claim was read verbatim from the same
   sentence used as evidence, so "the span entails the claim" was true by
   construction. Source monitoring was never exercised.
2. **Empty answer-only arm.** `PARAMETRIC_STORE = set()`; the answer-only
   accuracy was a hardcoded 0, not a measurement of what the live system
   knows without web access. The attribution-vs-baseline delta was a constant.
3. **Weak decoys.** Decoy claims were arbitrary false strings with no
   guaranteed topical proximity to the retrieved source; rejection did not
   prove discrimination between a true source and a same-topic wrong source.
4. **n = 32 per pool** (64 gap items total), half the design's ≥60/pool.

## 2. Brain grounding (why these four fixes, not a threshold change)

The measured property is **source monitoring at retrieval**: distinguishing a
remembered/known fact from an externally attested one, and rejecting a
plausible but unattested competitor.

- Johnson & Raye (1981), *Psychol Bull* 90(2):252–277 — source-monitoring
  errors arise when a self-generated item is attributed to an external source.
  The internal-source control (reasoning-only claims that must NOT wear a web
  tag) and the live-engine answer-only arm are the two halves of this control.
- Kang et al. (2009), *Psychol Sci* 20(8):963–973,
  doi:10.1111/j.1467-9280.2009.02402.x — the information-gap reward signal is
  a *specific* unresolved question; the gap items are exactly those questions,
  and reading is bounded at ≤2 sources.
- Gruber, Gelman & Ranganath (2014), *Neuron* 84(2):486–496,
  doi:10.1016/j.neuron.2014.08.060 — the motivational state, not exposure,
  gates encoding; hence the web-off parametric arm must be measured on the
  *same* questions rather than assumed empty.
- Tse et al. (2007), *Science* 316:76–82, doi:10.1126/science.1135935 —
  schema-congruent new associations are admitted cheaply; a near-miss source
  is schema-congruent (same topic) but factually wrong, so it is the right
  adversarial probe for content-based rejection.

None of these motivate lowering a decision threshold. Thresholds are unchanged
from design §4.6.

## 3. Exact implementation (what the code now does)

### 3.1 Gold claim is a paraphrase, not the source sentence

- `SYN` is a fixed, checked-in map of canonical content word → single-word
  synonym; its inverse is asserted unique at import.
- `_paraphrase(source_span)` rewrites the verbatim source sentence by applying
  those substitutions and returns `(claim, claim_tokens, n_subs)`. Canonical
  tokens are tracked **per occurrence** (not by token value): a synonym value
  such as `sea`, `stomach`, `liquid` may also occur naturally in the source, and
  a value-keyed inverse map would corrupt those words.
- `find_entailing_span` accepts a claim only when every canonicalised claim
  token appears in a verbatim sentence of the fetched content. Because the
  claim surface differs from the evidence, matching requires the synonym
  bridge rather than a string copy.
- `validate_items` rejects any gap whose `claim == source_span`, whose claim
  appears verbatim in the content, whose `paraphrase_subs < 1`, or whose
  canonicalised claim tokens do not restore exactly to the source fact tokens.

### 3.2 Answer-only arm is the live engine with web disabled

- `boot_answer_only_engine()` creates a real `CognitiveChatEngine` with
  `RAVANA_OFFLINE=1` and asserts `eng._offline is True` (the central
  `_web_blocked()` gate suppresses every live-web path in this state).
- `AnswerOnlyOracle` asks that engine `process_turn(query)` for **every gap
  item** — the same questions the web arm sees. It is injectable: unit tests
  pass a fake engine and never boot anything.
- `_WebAccessTripwire` wraps `SearchEngine.search` and `kb_describe` for the
  duration of this arm. Any web-provider access increments
  `oracle.web_calls`; a nonzero count becomes hard fail
  `answer_only_web_leaks`, and `answer_only_web_disabled` must be true.
- `answer_asserts_fact` grades the engine's own answer against the item's four
  most distinctive source fact tokens (`key_fact_tokens`), after applying the
  same light stemming/synonym canonicalisation. This is blind to the item's
  verdict and can fail for any individual question.

### 3.3 Near-miss decoy sources

- `DECOY_SPEC` has 60 items, each tied to a **real frozen query** and carrying
  a wrong-fact claim about the same topic (tags `near-miss-seen` /
  `near-miss-novel`).
- `validate_items` requires ≥2 content-token overlap between the decoy claim
  and the real source (genuine topical proximity) AND requires that no fetched
  source span entails the decoy claim. Correct behaviour is abstention with no
  URL and no span.

### 3.4 Sample size and sealing

| arm | primary | sealed | rule |
|---|---:|---:|---|
| gap | 65 | 67 | ≥60 per pool |
| closed-web | 30 | 30 | ≥30 per pool |
| decoy (near-miss) | 30 | 30 | ≥30 per pool |
| internal control | 10 | 10 | unchanged |

Gap evidence for the 68 extra items is generated by a fixed, score-independent
rule over the frozen fixture bank (how/why/what-causes queries whose first-hit
evidence sentence has ≥6 content words and ≥2 paraphrasable substitutions,
sorted by `sha256("RAVANA-WFA-GEN|"+query)`, one item per deduplicated topic).
No outcome data is consulted; the manifest hash is written to the summary.

Sealing is **unchanged**: primary/sealed are split per difficulty bucket by
`sha256("RAVANA-LEARN-v2|WFA|"+kind+"|"+difficulty+"|"+sid)` before any run.
The split order (primary ranks first) is a label, not an outcome.

## 4. Decision rules (exact, from design §4.6 + additive structural gates)

Unchanged numeric thresholds: `wfa_valid ≥ 0.70` on both pools; abstention
`≥ 0.90`; `wfa_valid − answer_only ≥ 0.20`; decoy rejection `≥ 0.85`;
`internal_no_tag ≥ 0.90`; sealed `wfa_valid ≥ primary − 0.10`; fabricated URLs,
unfetched URLs, non-verbatim spans and attribution on closed-web all `= 0`;
instrument controls pass; no network / save / fixture miss.

New **additive** gates (JSON fields added, none renamed or removed):

- `n_gap_primary ≥ 60`, `n_gap_sealed ≥ 60`
- `n_closed_primary ≥ 30`, `n_closed_sealed ≥ 30`
- `n_decoy_primary ≥ 30`, `n_decoy_sealed ≥ 30`
- `non_paraphrase_claims == 0`
- `answer_only_web_disabled` is true and `answer_only_web_leaks == 0`
- near-miss decoy rejection is reported both as the original `decoy_reject`
  and as an additive alias `near_miss_decoy_reject` (same value).

New additive record fields: `source_span`, `claim_is_paraphrase`,
`entail_tokens`, `answer_only_output`, `answer_only_correct`. New additive
summary fields: `answer_only_web_disabled`, `answer_only_web_calls`,
`answer_only_mode`, `web_arm_mode`, `paraphrase_synonym_map_size`; `hard_fails`
gains `non_paraphrase_claims` and `answer_only_web_leaks`. The legacy
`PARAMETRIC_STORE` constant is retained for external readers but is no longer
used in scoring.

## 5. Why this cannot saturate silently / how it can still FAIL

- A live-engine answer that already contains the gold facts on ≥70% of gap
  items drives `answer_only` up and collapses `delta` below 0.20 → **FAIL**.
- A gap fact whose frozen source is missing/mislocated aborts at validation →
  **INSTRUMENT INVALID**, never a patched threshold.
- Any web access in the answer-only arm, a paraphrase that reverts to a verbatim
  span, or a decoy that is entailed is a hard fail.
- Attributed success on a decoy or closed-web item is a hard fail.
- The shared Tier-0 `gate_saturation` still FLAGs a `wfa_valid` of 100 in the
  summary (non-blocking, as in the original harness). That flag is the honest
  signal that a 1.00 attribution rate is not, by itself, a progress claim — the
  progress claim here is the paired `delta` against the live web-off engine,
  which is measured on identical items rather than assumed.

No threshold was raised or lowered to turn a failing item into a passing one.
Harder items are allowed to fail; the correct outcome of a real measurement may
be FAIL.
