# Decoder after the attention and negative-sampling fixes

Date: 2026-11-14
Instrument: `scripts/eval_decoder_groundedness.py --engine persisted --cpg --cpg-relation`
Weights: `data/neural_decoder.npz` (retrained, 63 MB, vocab 17021)

## Headline

| metric (decoder strategy, graph-known) | before | after |
|---|---|---|
| mean on-topic fraction | 0.2213 | **0.4616** |
| mean unsupported assertions | 1.1787 | **1.1183** |
| mean word count | 22.55 | 9.10 |
| mean unique ratio | 0.8245 | **1.0000** |
| word-salad flags | — | **0/338** |

The decoder strategy now handles 338/480 subjects (was 313).

## What changed

Three merged PRs, all from `docs/CODEBASE_DECODER_AUDIT.md`:

1. **#342 `condition_proj` was never updated** — the update block was misindented
   into the auxiliary-loss branch, so the conditioning projection stayed at its
   random initialisation and could not influence the GRU. Measured: moved exactly
   0.0 per step before, 2.5e-05 after.

2. **#344 the attention head was mis-wired and dead** — built as
   `ConceptAttentionHead(hidden_dim, hidden_dim)`, so `output_proj` was
   `Linear(256, 256)` and returned a hidden-space vector. The readout blended two
   hidden vectors then projected (a double projection). Fixed with
   `_align_attention_head()` plus a logit-space mixture gated by a learnable
   `attn_gate`. Attention's share of the readout went from **0.63% to 43.7%**.

3. **#345 mixed negative sampling** — negatives were drawn ∝ `log1p(5·‖row‖)`,
   a rich-get-richer loop favouring function words. Half are now uniform.
   Content-word negatives went from 7/25 to 18/50. Also fixed the
   scheduled-sampling path, which projected a logit vector a second time and
   crashed the retrain.

## Honest reading

- **On-topic 0.4616 is still not "fluent and factual".** The unsupported rate
  (1.1183) exceeds the supported rate (0.4734): the decoder still asserts more
  than it can ground.
- **The gate withholds 185/480 subjects.** The binding constraint is now the
  grounding gate, not the decoder's fluency.
- **Word count dropped to 9.10** from 22.55. Part of the on-topic gain is that
  shorter output has fewer chances to leave the subject. The unique ratio of
  1.0000 and zero salad flags are genuine quality gains, but the length drop
  means the on-topic number is not directly comparable to the earlier 22-word
  measurements.
- **Three passes of a 40-pass run.** The checkpoint was promoted after three
  passes (~12 min each at the enlarged vocabulary). CE was 1.838 on the previous
  full run; this run's CE was not captured because stdout was buffered. Treat the
  retrain as partial.

## Not addressed

The audit's remaining findings: the train/inference readout mismatch (finding 1 —
an auxiliary `h_star` loss measured as noise, not shipped), `W_h_bias` not wired
into the settle call sites (finding 5, affects only the greeting/wellbeing/
farewell composers, not the factual path), and the dead code paths (finding 6).
