# Consolidation result: the harvested corpus reaches the graph

Date: 2026-11-14
Instrument: `scripts/eval_decoder_groundedness.py --engine persisted --cpg --cpg-relation`
Subjects measured: **5394** (was 480)

> **CORRECTION (same day).** The gated figures in the first version of this doc
> were measured with a grounding gate that had a hole: its per-sentence reference
> check accepted high-frequency associates such as `often` as evidence of
> aboutness, so pure word salad passed. That is fixed in **#351**. The corrected
> gated numbers are in the table below; the UNGATED numbers are unaffected
> (0.6802 both before and after), which confirms the fix was entirely in the gate.

## Headline

| metric (decoder strategy, graph-known) | before | after |
|---|---|---|
| mean on-topic fraction | 0.4616 | **0.6802** |
| mean unsupported assertions | 1.1183 | **0.5453** |
| mean supported assertions | 0.4734 | **0.5302** |
| mean word count | 9.10 | 7.13 |
| subjects measured | 338 | **4102** |

**Unsupported (0.5453) is now essentially equal to supported (0.5302).** Before
this work the decoder asserted ~2.4x more than it could ground; that ratio is now
~1.0. This is the first measurement where the decoder's assertions are not
systematically ungrounded.

## Gated (production) configuration after the #351 fix

| metric | pre-#351 gate | post-#351 gate |
|---|---|---|
| gate withholding | 2718/5394 (50.4%) | **3076/5392 (57.0%)** |
| gated on-topic | 0.3631 | 0.3663 |
| gated unsupported | 0.7911 | **0.7090** |
| ungated decoder on-topic | 0.6802 | 0.6802 (unchanged) |

The gate now withholds **more**, which is correct: it is no longer passing salad.
The text that does get through is better grounded (unsupported 0.7911 → 0.7090).

## What changed

Four PRs, all on the same root cause — harvested knowledge never reached the
graph:

- **#347** added `scripts/consolidate_harvested_corpus.py`. Subjects with a graph
  anchor went 128/512 → 510/512.
- **#348** fixed the gate itself: node admission is now schema-consistency-first
  (a text's own subject is admitted on one sighting — Tse et al., *Science* 2011;
  Kumaran et al., *TiCS* 2016), with corroboration retained for co-occurring
  tokens. A single pass then does what previously required manufacturing a second
  source (509/512).
- **#349** stopped the comparison/decompose path minting template frames
  (`'archived versus sea comparison'`) as concepts — 72 such nodes were in the
  persisted graph.
- **#351** closed the reference-check hole: the associate set is now filtered to
  content words, so `often` can no longer ground a sentence.

## Honest reading

- **The gate now withholds 3076/5392 (57.0%).** That is the binding constraint.
  More than half of all subjects produce no gated output at all.
- **The narrative strategy confabulates**: on-topic 0.1268, unsupported 2.4286
  over 27.28 words. It is the fallback path. The decoder strategy is where the
  quality is.
- **On-topic 0.6802 is measured over 4102 subjects, not 338.** The larger set
  includes many easier/more common concepts, so this is not a like-for-like
  comparison with the 0.4616 figure. The comparable signal is the
  unsupported/supported ratio, which improved on the same instrument.
- **Word count is 7.13** — short answers. Some of the on-topic gain is fewer
  chances to leave the subject.
- **The 57% withholding rate means the gate is now the target**, and the
  diagnostic in `scripts/diagnose_gate_clauses.py` shows two failure classes:
  the decoder output does not mention the subject (24/43 sampled), or the text is
  a definition echo with a stutter that trips the conflict check (19/43).

## Next

The gate's 57% withholding rate. The decoder is no longer it.
