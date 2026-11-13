# Retraining on descriptions: measured result

**Run:** `scripts/train.py --mode linggen`, 40 passes, resumed from a
28,697,144-byte checkpoint. Completed 2026-11-13, elapsed 3588.5 s.
Final CE **1.838**, `use_linggen=True`, vocab 15,000 → **17,021**.

**Instrument:** `scripts/eval_decoder_groundedness.py --engine persisted
--cpg --cpg-relation`, 478 graph-known subjects, two raw-input forms each.

---

## The measurement

A **paired A/B** — same instrument, same flags, same session, only the decoder
weights differ. The pre-retrain decoder was backed up to
`scratch/decoder_pre_linggen.npz` before training started.

| metric | pre-retrain | post-retrain | Δ |
|---|---|---|---|
| on-topic fraction | 0.3938 | 0.3840 | −0.0098 |
| content-word cosine | 0.4420 | 0.4400 | −0.0020 |
| unsupported assertions | 1.4289 | 1.5146 | +0.0857 |
| supported assertions | 0.8682 | 0.8640 | −0.0042 |
| word count | 16.19 | 16.55 | +0.36 |
| unique ratio | 0.9772 | 0.9746 | −0.0026 |

**The retraining did not materially change the decoder's grounding behaviour.**
Every delta is within run-to-run noise.

---

## A correction

An earlier reading of this data compared the post-retrain run against a
**stale baseline from a previous session** (unsupported 0.6046, supported
2.5105) and concluded the retraining caused a 2.4× regression in unsupported
assertions and a 65% collapse in supported ones.

That comparison was invalid: the baseline was measured with different weights
*and* a different instrument configuration. The paired A/B above is the valid
comparison, and it shows no material change. **The earlier regression claim
was wrong and is withdrawn.**

---

## What this means

The decoder was retrained on 512 harvested concept descriptions (plus 80 KB
definitions) for 40 passes, and its grounding behaviour is unchanged. Combined
with the audit findings:

1. **The settle loop was not minimising anything** (PR #343) — fixed, but the
   fix addresses *how* h settles, not *what* the readout knows.
2. **The readout is never trained on the state inference reads** — the
   obvious fix was measured as noise and reverted (see
   `docs/DECODER_AUDIT_DISPOSITIONS.md`).
3. **The attention path is dead** — `W_q/W_k/W_v` never trained,
   `output_proj` decayed to 14% of init.

Retraining cannot fix a readout that is never trained on the state it reads.
The blocker is architectural, not a data-volume problem.

---

## State

- `data/neural_decoder.npz` — post-retrain (vocab 17,021, fro 124.87)
- `data/linggen_checkpoint.npz` — same weights, checkpoint form
- `weights/ravana_weights.pkl` — persisted engine, post-retrain
- `scratch/decoder_pre_linggen.npz` — pre-retrain backup (vocab 15,000)
- `scratch/eval_preretrain.tsv`, `scratch/eval_postretrain.tsv` — the A/B data
