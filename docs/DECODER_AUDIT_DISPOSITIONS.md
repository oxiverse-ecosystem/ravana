# Decoder audit — findings and dispositions

Source: `docs/CODEBASE_DECODER_AUDIT.md` (subagent audit, 86 API calls, plus a
live probe of the shipped checkpoint `data/neural_decoder.npz`).

Each finding below was **verified by direct measurement** before any action was
taken. Two were confirmed and fixed; two were confirmed but are deliberate
design choices; the rest are recorded with their measured impact.

---

## 1. BLOCKER — training/inference state mismatch

**Claim.** Training is teacher-forced next-token CE: `h_t = GRU(emb(w_{t-1}))`,
loss on `w_t` (`neural_decoder.py:925-927`). Inference reads
`h_star = GRU(emb(w_t))` — the state *after* consuming the target
(`predictive_coding_generator.py:196,220`). Training never optimises the state
inference reads.

**Verified.** Confirmed by reading both code paths. `all_h[pos]` holds the state
after consuming `word_indices[pos]`; the settle loop's `h_star` is one extra
GRU step on `word_indices[pos+1]`.

**Disposition — attempted fix, measured as NOISE, reverted.**

I implemented an auxiliary sampled-softmax loss on `h_star`
(`train_post_target=True`): one extra GRU step on the target token, then a
sampled-softmax step with the same target, folded into `error_h_all`.

Measured over 8 seeds × 40 steps on a held-out sentence:

| flag | mean Δ CE on h_star |
|---|---|
| off | −0.00095 |
| on | **+0.00038** (worse) |

Per-seed spread was ±0.005 — the effect is smaller than the noise. The reason is
structural: the auxiliary error is folded into `error_h_all`, which drives
**all** parameter updates, so it perturbs the GRU and embeddings rather than
just the readout. With `freeze_core=True` (what `_mode_linggen` uses) the GRU
cannot adapt, so the extra gradient mostly adds noise.

**Not shipped.** The flag and its wiring were removed; the tree is clean. The
mismatch remains real and is still the most likely root cause of the decoder's
poor output — but the obvious fix does not work, and I will not ship a flag
whose measured effect is noise.

**Next attempt should be:** train the readout on `h_star` *without* perturbing
the GRU — e.g. a separate readout head for the post-target state, or a
stop-gradient on the extra GRU step so only `output_proj` learns from it.

---

## 2. BLOCKER — the settle loop is not predictive coding

**Claim.** `err = logits_target − logits_h` is computed then discarded
(`predictive_coding_generator.py:205`); the update is a fixed
`h ← h + 0.25·(h_star − h)`. After 8 steps h is 90% of the way to `h_star`, so
the emitted word is `readout(h_star)` — the attractor's own word.

**Verified.** Confirmed by inspection (the `err` variable is never read after
being computed) and by measurement.

**Fixed — PR #343.**

Replaced with the true gradient of `E = ½‖err‖²`:

```
dE/dh = Jᵀ err,   J = ∂readout/∂h = 0.5 · W_out
```

`_readout` is linear in the blended state, so `J` is exact — no finite
differences.

| metric | before | after |
|---|---|---|
| ‖err‖ before → after | 0.5404 → 0.5404 (0%) | **0.5404 → 0.4856 (10.1%)** |
| cos(h, h_star) | ~+0.90 (copies) | **+0.52** |

---

## 3. REAL — attention head mis-wired and dead

**Claim.** `ConceptAttentionHead(hidden_dim, hidden_dim)` makes `output_proj` a
`Linear(256,256)`, not `Linear(256,17021)`. `W_q/W_k/W_v` are never updated
and sit at exact random init; `output_proj` is weight-decayed to a near-no-op.

**Verified on the shipped checkpoint:**

| weight | checkpoint fro | fresh init fro | verdict |
|---|---|---|---|
| `attention.W_q.weight` | 22.5711 | 22.5705 | **never trained** |
| `attention.W_k.weight` | 22.6429 | — | **never trained** |
| `attention.W_v.weight` | 22.6955 | — | **never trained** |
| `attention.output_proj.weight` | **1.1587** | 7.98 | decayed to ~14% |
| `output_proj.weight` (main) | 124.8664 | 184.4559 | trained |

**Disposition — NOT changed. Deliberate design choice.**

The skip is explicit in the source (`neural_decoder.py:1139`):

```python
# Skip attention W_q/W_k/W_v (traces are sentence-specific, unreliable)
```

and `attention.output_proj` **is** trained (line 1137). The audit flagged this
as a defect; the code comment records it as a decision. Changing trained
behaviour on a hunch — with the decoder already producing word salad — would
confound the retraining measurement. Recorded, not acted on.

Note: the audit's "attention explains 0.63% of readout variance" figure is from
the *trained* checkpoint where `output_proj` is decayed to 1.16. On a fresh
decoder the attention term is larger than `h` (`‖attn‖/‖h‖ = 1.77`) and changes
the readout by 109%. Both numbers are true of their respective models; neither
means the attention path is healthy.

---

## 4. REAL — sampled softmax has no content-word coverage

**Claim.** 51 of 17021 rows update per sentence (0.30%). Negatives are drawn
∝ `log1p(5·‖output_proj row‖)` — a rich-get-richer loop favouring high-norm
(function-word) rows.

**Disposition — recorded, not changed.** This is a plausible contributor to the
function-word bias in the output, but it is a training-dynamics change that
would need its own measurement. It is the strongest remaining candidate after
findings 1 and 2.

---

## 5. REAL — `W_h_bias` trained but never applied at settle call sites

**Claim.** Production settle call sites pass no `persistent_emb`
(`response_gen.py:2987,3021,3055`), so the working-memory gate is dead there.

**Verified.** Confirmed — but those three sites are `_compose_greeting_generative`,
`_compose_wellbeing_generative` and `_compose_farewell_generative`: small-talk
paths, not factual answers. The main generation path
(`response_gen.py:1652`, autoregressive `generate()`) **does** pass
`persistent_emb`.

**Disposition — NOT changed.** Real dead code, low impact. The factual path is
covered.

---

## 6. MINOR — dead paths

`contrastive_weight`/`contrastive_negatives` stored and persisted but no
contrastive loss exists; `av_head` trained but unused at inference;
`ConceptAttentionHead.forward()` traced path is dead; `load_from` hardcodes
`n_attention_heads=4`; `_linggen_W_sm` not persisted in the decoder checkpoint.

**Disposition — recorded.** No behaviour change; cleanup only.

---

## Summary

| # | finding | verified | action |
|---|---|---|---|
| 1 | train/inference state mismatch | yes | fix attempted, measured as noise, **reverted** |
| 2 | settle loop discards its error | yes | **fixed** (PR #343) |
| 3 | attention W_q/k/v never trained | yes | deliberate, not changed |
| 4 | sampled softmax content bias | partly | recorded |
| 5 | `W_h_bias` dead at settle sites | yes | real but low-impact, not changed |
| 6 | dead paths | yes | recorded |

The honest state: **one real blocker fixed** (the settle loop), one blocker
**not** fixable by the obvious approach (the readout mismatch), and a training
run whose result is still pending.
