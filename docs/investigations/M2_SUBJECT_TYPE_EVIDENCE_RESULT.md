# M2 result — non-leaking subject-side type evidence channel (IMPLEMENTED)

**Status: IMPLEMENTED, default-OFF, unwired. No TRA/RSI benchmark claim.**
Worktree `C:/Users/Likhith/Documents/projects/ravana_m2_subject_evidence`,
branch `feat/m2-subject-type-evidence`, base `origin/main` `82fe79d70ef88d7694a63d09d35bdf9c84291145`.
The pre-registration was written before any evidence number was read:
`M2_SUBJECT_TYPE_EVIDENCE_PREREG.md`. No engine was booted, no `process_turn`
called, no store written, no network fetched during the experiment.

## What changed

Only NEW files. No gate, threshold, item bank, prompt, answer key or scoring
routine was edited:

- `ravana/src/ravana/learn/subject_type_evidence.py` — pure default-off module.
- `data/subject_type_evidence/snapshot.json` — frozen licensed snapshot.
- `tests/unit/test_subject_type_evidence.py` — 20 static unit tests.
- `scratch/m2_subject_evidence/` — pre-registered probe, builder, results,
  independence audit, production-equivalence check.
- the pre-registration and this document.

## Source and licence

| item | value |
|---|---|
| Primary source | Wikidata SPARQL (`https://query.wikidata.org/sparql`), `subclass of` (P279) / `instance of` (P31) plus English item descriptions |
| Licence | **CC0-1.0** (public domain). Official policy `https://www.wikidata.org/wiki/Wikidata:Licensing`: "All structured data … is released into the public domain under Creative Commons Zero"; `https://creativecommons.org/publicdomain/zero/1.0/` |
| Optional cross-check source | Princeton WordNet 3.0, WordNet License 3.0, `https://wordnet.princeton.edu/wordnet/license/` (Miller 1995, CACM 38(11):39–41, doi:10.1145/219717.219748) |
| Snapshot facts / subjects | 980 / 140 |
| Snapshot `facts_sha256` | `17b91e1a6afae3da8978ad76d19e0ee8a6c92d2563e38a22e7a9f5c43ae57b3a` |
| Upstream substrate SHA | `b91e7044f0505db47fdb5b4bc8f8547435434e3080d966591effbfcdb09be0aa` |
| Fixed sample SHA (40 subjects) | `8a79ee22816d6d82` |

Every emitted evidence record carries source, licence, licence URL, source URL,
snapshot SHA, verbatim source text and credibility. Credibility is derived from
the **licence** (CC0 0.7, WordNet 0.9), never from a label; a positive declared
value can only cap it, not inflate it.

## Pre-registered gates (all measured, none retuned)

| gate | result |
|---|---|
| G1 anti-leak (AST) | PASS — 0 forbidden identifiers, 0 forbidden imports in every core function |
| G2 default-off | PASS — 0 evidence with flag unset on all 40 subjects; repeated calls byte-identical |
| G3 subject specificity | PASS — distinct-signature rate **1.000**, within-vocabulary subject permutation changes **1.000** of evidence sets, out-of-vocabulary subject yields 0 |
| G4 negation / contradiction / credibility | PASS — **14/14** cases; all six negation surfaces, contradicted, low-credibility, unlicensed, empty text, wrong subject and bad relation rejected for the correct reason; both positive sources admitted |
| G5 coverage | PASS — **40/40** sampled public query subjects have ≥1 admitted record |
| G6 provenance | PASS — **100 %** of admitted records complete and licence-valid |
| G7 candidate permutation | PASS — evidence byte-identical; the channel has no candidate parameter |
| G8 answer-key analogue | Structurally N/A — API accepts no key, enforced by G1 |

Results are byte-identical under `PYTHONHASHSEED=0` and `1`, and the production
module reproduces the scratch results JSON byte-for-byte. `pytest` 20/20 pass;
the production module also ran the unmodified gate script unchanged.

## Instrument defect found and fixed (reported, not a threshold retune)

The first run returned 0/40 coverage because snapshot records left credibility
unstated (0.0) and `effective_credibility` took `min(declared, ceiling)`, so
every record was rejected as `low_credibility`. The derivation was corrected:
a non-positive declaration now means "not stated" and inherits the licence
ceiling. The pre-registered floor (0.50), all other constants and all gates
were unchanged. The corrected run is the one reported above.

## Independence audit (informational, not a gate)

- Overlap between 229 emitted distinct source-native type names and item-bank
  `type_sets` values: **0**.
- Overlap with `world_model`/gold strings and displayed options: **12/229**,
  all ordinary English vocabulary coincidences (e.g. common nouns that are
  both Wikidata labels and candidate words). No item-bank field is an input to
  any scorer; the AST scan and the absence of item-bank imports are the
  derivation-level proof.

## Limitations (honest)

1. **Not a benchmark fix.** This is a channel that can supply subject-specific
   licensed evidence. It is not wired into `process_turn`, changes no endpoint,
   and is not evidence that TRA/RSI improved. Enabling it alone cannot repair
   the documented information deficit.
2. **Source polysemy.** Wikidata contains multiple senses (e.g. an alternate
   sense types `game` as `animal`). The channel emits source-native facts and
   does not adjudicate sense; a consumer needs its own sense policy. No answer
   key was used to filter them.
3. **Negation is a clause-window heuristic**, fail-closed on ambiguous text;
   it is validated on the pre-registered surfaces, not on arbitrary prose.
4. **Cross-source margin not re-litigated.** The prior
   `CROSS_SOURCE_FAST_MAPPING_DESIGN.md` NO-GO (+0.055 over the demo-only arm)
   is untouched; this module does not pick a candidate or compute a score.

## Brain grounding

- Carey & Bartlett (1978) fast mapping; type from the contrastive situation,
  re-verified in Carey (2010) *Lang. Learn. Dev.* 6(3):184–205,
  doi:10.1080/15475441.2010.484379, PMCID PMC3101498.
- Xu & Tenenbaum (2007) *Psychol Rev* 114(2):245–272,
  doi:10.1037/0033-295X.114.2.245, PMID 17500627 — structured hypothesis space.
- Tse, Langston, Kakeyama, Bethus, Spooner, Wood, Witter & Morris (2007)
  *Science* 316(5821):76–82, doi:10.1126/science.1135935, PMID 17412951 —
  one-trial assimilation requires an existing schema.
- Johnson, Hashtroudi & Lindsay (1993) *Psychol Bull* 114(1):3–28,
  doi:10.1037/0033-2909.114.1.3 — source memory dissociates from item memory;
  schemas propose, only sourced evidence admits.

These are behavioural/neurobiological findings; the provenance filter is an
**extrapolation** into RAVANA's own primitives, not a claim about brain code.
