# Consolidation result: the harvested corpus reaches the graph

Date: 2026-11-14
Instrument: `scripts/eval_decoder_groundedness.py --engine persisted --cpg --cpg-relation`
Subjects measured: **5394** (was 480)

## Headline

| metric (decoder strategy, graph-known) | before | after |
|---|---|---|
| mean on-topic fraction | 0.4616 | **0.6802** |
| mean unsupported assertions | 1.1183 | **0.5438** |
| mean supported assertions | 0.4734 | **0.5402** |
| mean word count | 9.10 | 7.13 |
| subjects measured | 338 | **4141** |

**Unsupported (0.5438) is now essentially equal to supported (0.5402).** Before
this work the decoder asserted ~2.4x more than it could ground; that ratio is now
~1.0. This is the first measurement where the decoder's assertions are not
systematically ungrounded.

## What changed

Three PRs, all on the same root cause — harvested knowledge never reached the
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

## Honest reading

- **The gate still withholds 2718/5394 (50.4%).** That is the binding constraint
  now. Half of all subjects produce no gated output at all.
- **The narrative strategy is bad and got worse in relative terms**: on-topic
  0.1217, unsupported 2.3963 over 26.83 words. It is the fallback path and it
  confabulates. The decoder strategy is where the quality is.
- **On-topic 0.6802 is measured over 4141 subjects, not 338.** The larger set
  includes many easier/more common concepts, so this is not a like-for-like
  comparison with the 0.4616 figure. The comparable signal is the
  unsupported/supported ratio, which improved on the same instrument.
- **Word count is 7.13** — short answers. Some of the on-topic gain is fewer
  chances to leave the subject.

## Next

The gate's 50% withholding rate is the target. The decoder is no longer it.
