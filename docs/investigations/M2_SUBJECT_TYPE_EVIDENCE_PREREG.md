# M2 — pre-registration: non-leaking subject-side type evidence channel

**Status: PRE-REGISTRATION, written before any evidence number was read.**
Worktree `C:/Users/Likhith/Documents/projects/ravana_m2_subject_evidence`,
branch `feat/m2-subject-type-evidence` from `origin/main` `82fe79d70ef88d7694a63d09d35bdf9c84291145`.
No engine boot, no `process_turn`, no store write, no network, no production
file, no gate, threshold, item bank, prompt or scoring change is authorized by
this document. This is a **channel** experiment, not a TRA/RSI benchmark result.

## 1. Question

Can a pure, default-off module produce **type evidence for the actual query
subject** from an independent licensed external source, with provenance, while
rejecting negated / contradicted / low-credibility assertions and leaking
nothing from the benchmark's answer key?

## 2. Licensed input (only input permitted)

- Wikidata structured data (`subclass of`, P279) + English item descriptions,
  retrieved from `https://query.wikidata.org/sparql`; **CC0-1.0** (official
  policy: `Wikidata:Licensing` — "All structured data … is released into the
  public domain under Creative Commons Zero").
- Optical second source: WordNet 3.0 noun hypernyms, WordNet License
  (Princeton University, 1995–2006). Used only for an optional independence
  cross-check, never to supply the graded key.
- Every record carries: source, license, source URL, snapshot SHA-256,
  verbatim source text, and an explicit credibility assigned from the source
  license (Wikidata 0.7, WordNet 0.9, unlicensed/unknown 0.0). Credibility is
  **never** derived from a gold label.

Forbidden inputs (hard): `gold`, `type_sets`, `world_model`, `typed_queries`,
`channel_evidence`, `item_id`, `role`, `type_label`, `domain`/`far_domain`,
answer keys, engine stores, `process_turn`. The public query subject and the
public displayed candidate list may be used as inputs but candidates are never
read by the channel.

## 3. Fixed sample (chosen before measurement)

40 unique subjects from the **public** fields of `tra_k_items.build_items()`:
`subject`, `unseen_test_words.near.subject`, `unseen_test_words.far.subject`.
Order by `sha256("M2-v1|" + subject)`, take the first 40 unique. Only those
three string fields are extracted; the item dict is never passed to a scorer.
The same sample is used for every gate and every arm.

## 4. Frozen constants (not tunable after a result)

| constant | value | rationale |
|---|---|---|
| `CREDIBILITY_FLOOR` | 0.50 | excludes unlicensed/unknown sources a priori |
| `PERMUTE_OFFSET` | 137 | fixed hash-style subject permutation |
| `AGREEMENT_MIN` | 0.90 | subject-specificity must be near-deterministic |
| `COVERAGE_MIN` | 0.50 | a precision channel needs ≥ half the public vocabulary |
| `SNAPSHOT_SCHEMA` | `ravana-m2-subject-type-evidence-v1` | frozen provenance schema |

No threshold, floor, margin, weight or gate may be changed after a result is
seen. A failing gate produces a NO-GO, not a retune.

## 5. Pre-registered gates (ALL required to implement)

| ID | Gate | Pass condition |
|---|---|---|
| G1 | Anti-leak | AST scan of every core evidence function: 0 forbidden identifiers, 0 imports of item-bank/gate/engine modules |
| G2 | Default-off | flag unset → 0 evidence on all 40 subjects; repeated calls byte-identical; no global mutation |
| G3 | Subject specificity | ≥90% of sampled subjects have a signature different from the next subject's; permuting subject changes ≥80% of non-empty evidence sets; a subject absent from the snapshot yields 0 |
| G4 | Negation / contradiction / credibility | **100%** rejection on a fixed 14-case adversarial set: negated (`not`, `never`, `isn't`, `not a` …), contradicted by a strictly higher-credibility negation, credibility < floor, unlicensed source, empty text, wrong subject; positive controls still admitted |
| G5 | Coverage | ≥20/40 sampled subjects have ≥1 admitted evidence record |
| G6 | Provenance | 100% of admitted records carry non-empty source, allow-listed license, source URL, snapshot SHA, verbatim assertion text, credibility ≥ floor, and `subject == query subject` |
| G7 | Candidate permutation | permuting the displayed candidate list leaves evidence byte-identical (channel has no candidate parameter) |
| G8 | Gold-key analogue | a fabricated answer key supplied to the caller cannot change evidence; structurally enforced by API signature + G1, reported as N/A for direct permutation with rationale |

## 6. Decision rule

- If **G1–G7 all PASS** → implement the module in the isolated worktree as a
  pure default-off module under `ravana/src/ravana/learn/`, add unit tests,
  commit and push. Report SHA, source/license, metrics, status.
- If **any gate FAILS** → NO-GO. Commit the pre-registration, scratch script,
  results and a findings doc. Do **not** add a production module. Name the
  missing input or broken property.
- In either case: no TRA/RSI endpoint is claimed. This channel does not change
  a gate, threshold, item, prompt or scoring routine, and is not evidence that
  TRA/RSI capability improved.

## 7. Known prior failures this design must not repeat

1. **Ontology label mismatch** — never grade in authored role labels; emit
   source-native type names only (Wikidata P279 labels).
2. **Cross-source fast mapping NO-GO (+0.055)** — do not re-run the
   description→demo-schema→candidate-pick scorer or claim its margin; this
   channel supplies evidence, it does not pick a candidate.
3. **Lexical hook emitted on negated teaching** — G4 is a hard gate; the
   channel must reject negation before any evidence is emitted.

## 8. Brain grounding (citations resolved on the web this session)

- Carey & Bartlett (1978) fast mapping; a new word's type came from the
  **contrastive situation** ("not the red tray, the chromium one"), not from
  other exemplars — re-verified in Carey (2010), *Lang. Learn. Dev.*
  6(3):184–205, doi:10.1080/15475441.2010.484379, PMCID PMC3101498.
- Xu & Tenenbaum (2007), *Psychological Review* 114(2):245–272,
  doi:10.1037/0033-295X.114.2.245 — sparse induction requires a structured
  hypothesis space; PubMed 17500627.
- Tse, Langston, Kakeyama, Bethus, Spooner, Wood, Witter & Morris (2007),
  *Science* 316(5821):76–82, doi:10.1126/science.1135935 — one-trial traces
  are assimilated only when a schema already exists; PMID 17412951.
- Clark (1987) principle of contrast and Johnson, Hashtroudi & Lindsay (1993)
  source-memory dissociation — a schema may propose, only sourced evidence
  admits (as recorded in `ONTOLOGY_TYPE_BRIDGE.md`).

These are behavioural/neurobiological findings; the mapping to a provenance-
filtered evidence channel is an **extrapolation**, not a claim that the brain
reads Wikidata.
