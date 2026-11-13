# CODEBASE DECODER AUDIT

**Scope:** Why the GRU hidden state fails to encode next-word content (cos(h, eos) rises 0.11 → 0.98 over one sentence; h encodes position, not content), and what the code does about it.
**Method:** Read-only static audit + live probe of the shipped checkpoint `data/neural_decoder.npz` (vocab 17021, embed 75, hidden 256, 4 heads).
**Date:** 2026-11-13

---

## 0. Executive summary

The decoder is not broken in the way the symptom suggests. The hidden state **does** carry next-word information — but only for the first token or two, and the readout is dominated by a **positional/eos attractor** that the training objective actively reinforces. Three independent defects compound:

1. **The training objective is next-token CE under teacher forcing, but inference reads out a state the training never targets** (the GRU state *after* consuming the target word, `h_star`). The state that inference reads is off-distribution.
2. **The attention head is architecturally incoherent**: `ConceptAttentionHead` is built with `vocab_size=hidden_dim=256`, so its `output_proj` is `Linear(256,256)`, not `Linear(256,17021)`. Its output is a 256-D vector that is blended 50/50 into `h` and then passed through `output_proj` a second time. The head is trained only through its `output_proj` (W_q/W_k/W_v are never updated), and the shipped weights are ~12× smaller than random init — i.e. the head has learned to be nearly a no-op.
3. **The settle loop is not predictive coding.** `err` is computed and discarded; the update is a fixed linear interpolation `h ← h + 0.25·(h_star − h)`. After 8 steps h is ~90% of the way to `h_star`, so the emitted word is essentially `readout(h_star)` — the readout of the GRU state of the **attractor's own word**. The generator is a dressed-up copy of the attractor trajectory, not a language model.

Severity: **BLOCKER** for (1) and (3); **REAL** for (2).

---

## 1. Q1 — Trace the readout path end to end

### Answer

The blend is **dimensionally coherent** — both operands are `(hidden_dim,) = (256,)` — but it is **semantically incoherent**. `attn_logits` is not a vocab-sized logit vector; it is the output of a `Linear(256,256)` applied to a pooled concept vector. Blending it with `h` and then applying `output_proj` (`Linear(256,17021)`) a second time is a **double projection**. The attention head is trained (partially) to map pooled concepts → vocab logits, but at inference its output is consumed as a hidden-state vector.

### Evidence

- `ravana_ml/src/ravana_ml/nn/module.py:453` — `ConceptAttentionHead.__init__(concept_dim, vocab_size, n_heads=2)`: `W_q/W_k/W_v = Linear(concept_dim, concept_dim)`; `output_proj = Linear(concept_dim, vocab_size)`.
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:114` — `self.attention = ConceptAttentionHead(hidden_dim, hidden_dim, n_heads=n_attention_heads)`. **The second argument is `hidden_dim`, not `vocab_size`.** So `output_proj` is `Linear(256,256)`.
- `ravana_ml/src/ravana_ml/nn/module.py:491-494` — `pooled = np.mean(attended, axis=0)` → `(concept_dim,)`; `return self.output_proj.forward_raw(pooled[np.newaxis,:])[0]` → `(vocab_size,)` = `(256,)`.
- `ravana/src/ravana/decoder/predictive_coding_generator.py:100` — comment says `# (vocab_size,)`. **Wrong.** The actual shape is `(hidden_dim,)`.
- `ravana/src/ravana/decoder/predictive_coding_generator.py:102-105` — `_readout`: `combined = h*0.5 + attn_logits*0.5; return nd.output_proj.forward_raw(combined...)`. So `attn_logits` (256) is blended with `h` (256) and then projected to vocab by `output_proj` (17021×256).
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:334` (forward), `:420` (generate), `:988` (train_on_sentence) — all three use the identical `h*0.5 + attn*0.5` blend. **The blend is consistent across train/generate/settle** — it is not a shape bug in one path.

### What the attention head actually computes

`cond_proj = condition_proj(conditioning_embs)` → `(n_concepts, 256)`. Then multi-head self-attention over the concept set (Q,K,V all from the same `cond_proj`), mean-pool over concepts → `(256,)`, then `Linear(256,256)`. It is a **concept-set summariser**, not a vocab logits head. Its output is a 256-D "concept context vector".

### Is blending meaningful?

It is meaningful in the sense that both vectors live in the same space and the sum is well-defined. It is **not** meaningful as "attention provides vocab logits that are averaged with the GRU state". The 50/50 blend was presumably intended to be `h*0.5 + attn_logits*0.5` where `attn_logits` is vocab-sized, but the head was wired with `vocab_size=hidden_dim`. The result is that the attention contribution is a small, roughly constant bias vector (see Q3 for magnitude).

### Comparison with `NeuralDecoder.generate()`

`generate()` uses the **same** blend (`neural_decoder.py:420`). So the settle generator is consistent with the autoregressive path — both are equally affected by the double-projection. This is not a settle-only bug.

### Severity: **REAL**

The blend is dimensionally valid and consistent, so it is not a crash/shape bug. But the attention head is architecturally mis-wired (`vocab_size=hidden_dim`), and its output is semantically a hidden-state bias being treated as a peer of `h`. The 50% weight is misleading: the attention vector's norm is ~0.19× `||h||` in the real probe, so the "50/50" blend is effectively ~11% attention, 89% GRU.

---

## 2. Q2 — Is the training objective matched to the inference scheme?

### Answer

**No.** Training optimises `h` (the GRU state after consuming the *previous* token) to predict the *next* token. Inference (settle path) reads out `h_star` — the GRU state after consuming the *target* token. These are different states, and training never targets `h_star`.

### Evidence

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:925-927` — `input_idx = word_indices[pos]; target_idx = word_indices[pos+1]`. The GRU consumes `input_idx` (previous token) and the loss is on `target_idx` (next token). Teacher-forced next-token CE.
- `ravana/src/ravana/decoder/predictive_coding_generator.py:196` — `h_star = self._gru_step(proj_tgt, h, persistent_emb=persistent_emb)` where `proj_tgt = condition_proj(traj_embs[pos])`. `h_star` is the GRU state **after consuming the target trajectory word**.
- `ravana/src/ravana/decoder/predictive_coding_generator.py:220` — `settled_logits = self._readout(h, attn_logits)` after settling h toward `h_star`. So the emitted word is `readout(h ≈ h_star)`.
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:988` — training reads out `combined_vec = h*0.5 + cached_attn_logits*0.5` where `h` is the state after the *previous* token.

### The mismatch precisely

Training: `h_t = GRU(emb(w_{t-1}), h_{t-1}); loss = CE(readout(h_t), w_t)`.
Inference: `h_star = GRU(emb(w_t), h_{t-1}); emit = readout(h_star)`.

`h_star` is the state *after* the target word has been consumed. Training never asks the readout to predict anything from `h_star`; it asks the readout to predict `w_t` from `h_t` (the state *before* `w_t`). So the state that inference reads is **off-distribution** — it is a state the readout was never trained on.

### Additional mismatch: conditioning

Training builds `conditioning_embs` from the sentence's own content/function words (`neural_decoder.py:776-837`). Inference builds `conditioning_embs` from the situation vector + concept node vectors + subject vector (`response_gen.py:1448+`). These are different distributions. The attention context at inference is not the attention context at training.

### Additional mismatch: scheduled sampling

`scheduled_eps=0.25` in `_mode_linggen` (`scripts/train.py:802+`) means 25% of positions feed the model's own argmax instead of the teacher token. This is a partial exposure-bias cure, but it does not address the `h_star` mismatch.

### Severity: **BLOCKER**

This is the core reason the settle path emits attractor words rather than generated words. The training objective and the inference scheme are optimising different functions.

---

## 3. Q3 — What is `attn_logits` doing during training?

### Answer

The attention head is **partially trained**: only its `output_proj` (`Linear(256,256)`) is updated. `W_q/W_k/W_v` are **never updated** — the code explicitly skips them (`neural_decoder.py:1139`: "Skip attention W_q/W_k/W_v (traces are sentence-specific, unreliable)"). The shipped `output_proj` weights are ~12× smaller than random init, i.e. the head has learned to be nearly a no-op.

### Evidence

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:1133-1137` — the only attention update:
  ```python
  if not freeze_core:
      attn_error_all = np.clip(error_h_all * 0.1, -1.0, 1.0)
      total_attn_err = np.sum(attn_error_all, axis=0)
      self.attention.output_proj.weight.data += (total_attn_err / T) * lr_attn - self.attention.output_proj.weight.data * wd
  ```
  `lr_attn = 0.0002` (`:1040`). Only `output_proj` is touched.
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:1139` — `# Skip attention W_q/W_k/W_v (traces are sentence-specific, unreliable)`.
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:713` — docstring: "freeze_core: If True, only update word_embedding and output_proj. Skips GRU, condition_proj, and attention updates." So under `freeze_core=True` (the production training mode, `scripts/train.py:867`), **even `attention.output_proj` is frozen**. The attention head is entirely untrained in the shipped checkpoint.
- Live probe of `data/neural_decoder.npz`:
  - `attention.output_proj` frobenius = **1.159**; expected random init frobenius = **9.24** (`sqrt(256/3)` for `Linear(256,256)`). The head has been driven to ~12.5% of random scale — it is a near-no-op.
  - `attention.output_proj` row-norm mean = **0.0724** (max 0.0736, min 0.0713) — extremely uniform, consistent with a heavily weight-decayed / barely-trained matrix.
  - `W_q` fro = 22.57, `W_k` fro = 22.64, `W_v` fro = 22.70 — all ≈ `sqrt(256·256·2/256)` = 22.63, i.e. **exactly random init**. Confirms W_q/W_k/W_v were never updated.

### Quantifying the readout variance it explains

Live probe with real `h` and `attn` from the checkpoint:

| Quantity | Value |
|---|---|
| `||h||` | 9.52 |
| `||attn||` | 1.78 |
| `||attn|| / ||h||` | **0.187** |
| `var(logits)` | 0.4357 |
| `var(logits_h)` (h part) | 0.4329 |
| `var(logits_attn)` (attn part) | 0.0027 |
| **fraction of readout variance from attn** | **0.63%** |

With random `h`: fraction = **0.074%**.

So the 50/50 blend injects a vector that explains **<1% of the readout variance**. The "50%" is a misnomer — the attention contribution is ~11% of the combined vector's norm and <1% of the logit variance.

### Is attention untrained?

- `W_q/W_k/W_v`: **never trained** (random init confirmed).
- `output_proj`: trained only when `freeze_core=False`. Production uses `freeze_core=True` (`scripts/train.py:867`), so in the shipped checkpoint it is also untrained (and weight-decayed to near-zero).
- The `ConceptAttentionHead.forward()` docstring (`module.py:496-505`) claims `accumulate_free_energy` can update W_q/W_k/W_v — but no such call exists in `train_on_sentence`. The traced path is dead.

### Severity: **REAL**

The attention head is effectively dead weight. It is not injecting an *arbitrary* vector (it is a deterministic function of the conditioning concepts), but it is injecting a near-zero vector that contributes <1% of readout variance. The 50% blend is misleading and the `vocab_size=hidden_dim` wiring is a genuine architectural bug.

---

## 4. Q4 — The sampled softmax

### Answer

Only **51 of 17021 rows** update per sentence (1 target + 50 negatives). The negative-sampling distribution is **neither uniform nor corpus-frequency**: it is proportional to `log1p(5 · L2 norm of the output_proj row)`, i.e. it favours words whose output weight rows have large norms — the model's most-confident words. There is **no mechanism** that ensures content words get sampled often enough.

### Evidence

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:868-893`:
  ```python
  special_set = frozenset({0, 1, 2, 3})
  non_special_idx = np.array([i for i in range(vocab_size_actual) if i not in special_set], dtype=np.intp)
  n_neg = min(50, len(non_special_idx))
  row_norms = np.linalg.norm(out_weight[non_special_idx], axis=1)
  log_weights = np.log1p(row_norms * 5.0)
  neg_probs = log_weights / sum_log_weights
  shared_negatives = self._neg_rng.choice(non_special_idx, n_neg, replace=False, p=neg_probs)
  ```
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:1052-1058` — only the sampled rows (`all_idx_stored[pos]`) get a Hebbian update:
  ```python
  for pos in range(T):
      h_vec = all_combined[pos]; err = all_err_stored[pos]; idx = all_idx_stored[pos]
      hebb = np.outer(err, h_vec) / T
      out_weight[idx] += hebb * lr_proj - out_weight[idx] * wd
  ```
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:1061-1065` — the backprop to hidden is also sparse: `error_h_all[pos] = err @ out_weight[idx]`. Only sampled rows contribute to the hidden-state gradient.

### What fraction of training steps update any given content word?

- Per sentence: 51 rows out of 17021 → **0.30%**.
- Per position: 51 rows (shared across positions within a sentence).
- A given content word is updated only if it is (a) the target at some position, or (b) sampled as a negative. The negative sampling is biased toward high-norm rows, which are the model's confident words — a rich-get-richer dynamic. Rare/content words with small row norms are sampled with probability ∝ `log1p(5·norm)`, which is small.
- There is **no frequency-based sampling**, no importance weighting, no content-word boost in the sampler. The `content_word_ids` set is passed to `generate()` (`response_gen.py:1660`) but is **not used in training**.

### Consequence

The output_proj rows for rare content words are updated only when those words appear as targets in the training corpus. The corpus is 96% aphorisms (`teen_seeds.txt` 14,037 lines vs `grounded_descriptions.txt` 512 — see `docs/READOUT_CANNOT_PREDICT_CONTENT.md`), so content-word rows are rarely targets and rarely sampled. The readout for content words is essentially untrained.

### Severity: **REAL**

The sampled softmax is a reasonable approximation for speed, but the negative distribution is a rich-get-richer feedback loop and there is no content-word coverage guarantee. This is a major contributor to the "word salad" symptom: the model's high-norm rows are the function words / special tokens, which get sampled as negatives most often, reinforcing their dominance.

---

## 5. Q5 — The settle loop

### Answer

The settle loop is **not predictive coding**. `err` is computed and discarded; the update is a fixed linear interpolation `h ← h + 0.25·(h_star − h)`. After 8 steps h is ~90% of the way to `h_star`, so the emitted word is essentially `readout(h_star)` — the readout of the GRU state of the **attractor's own word**. The generator emits the attractor's own words, not novel generations.

### Evidence

- `ravana/src/ravana/decoder/predictive_coding_generator.py:203-213`:
  ```python
  for _ in range(self.settle_steps):
      logits_h = self._readout(h, attn_logits)
      err = logits_target - logits_h          # computed ...
      step = self.settle_lr * (h_star - h)    # ... but never used
      h = h + step
      if np.linalg.norm(h - prev_h) < self.settle_tol:
          break
      prev_h = h.copy()
  ```
  `err` is dead code. The update is `h ← h + 0.25·(h_star − h)` = `0.75h + 0.25h_star`. After 8 steps: `h = (1 − 0.75^8)·h_star + 0.75^8·h_0` → **90.0% of the way to `h_star`**.
- `ravana/src/ravana/decoder/predictive_coding_generator.py:196` — `h_star = self._gru_step(proj_tgt, h, ...)` where `proj_tgt = condition_proj(traj_embs[pos])`. `h_star` is the GRU state after consuming the target trajectory word.
- `ravana/src/ravana/decoder/predictive_coding_generator.py:220-221` — `settled_logits = self._readout(h, attn_logits); chosen = self._pick(settled_logits, ...)`. So the emitted word is `readout(h ≈ h_star)`.
- `ravana/src/ravana/core/attractor_memory.py:182-184` — `retrieve` returns `self._trajectories[best_key]`, i.e. the **stored** trajectory (the definition's own words), not a decoded/recovered one. The HRR unbind is only used for scoring/selection.
- `ravana/src/ravana/core/attractor_memory.py:155-184` — `retrieve` returns the stored trajectory verbatim. `recover_trajectory` (`:186-207`) also returns the stored trajectory.

### Can the settle loop ever produce a word NOT in the trajectory?

**Only by accident.** The emitted word is `readout(h ≈ h_star)` where `h_star` is the GRU state of the target trajectory word. If the readout were a perfect next-word predictor, it would emit the word that *follows* the target in the training distribution — which is not necessarily in the trajectory. But:

1. The readout is not a perfect predictor (see Q2 — it was trained on a different state).
2. The readout is dominated by the eos/function-word attractor (see `docs/HIDDEN_STATE_COLLAPSES_TO_EOS.md`).
3. `_pick` rejects special tokens and falls through to the next-best non-special word (`predictive_coding_generator.py:287-291`), which is typically a function word (`is`, `of`, `to`, `the`).

So in practice the settle loop emits a mix of (a) the attractor's own words (when the readout happens to rank them highly) and (b) function words (when the readout collapses to the eos/function-word attractor). It does not generate novel content words.

### The `min_steps` guard

`predictive_coding_generator.py:233-234` — if fewer than `min_steps=4` words are emitted, the generator returns `[]` (fail-closed). This means short attractor trajectories produce no output at all.

### Severity: **BLOCKER**

The settle loop is the production generation path for greeting/wellbeing/farewell (`response_gen.py:2968-3068`) and for `_gen_one` in `scripts/train.py:522-529`. It is presented as "predictive coding" but is a fixed linear interpolation toward the attractor's own state. It cannot generate novel words.

---

## 6. Q6 — Dead or inconsistent paths

### 6.1 `err` in the settle loop — dead code

`ravana/src/ravana/decoder/predictive_coding_generator.py:205` — `err = logits_target - logits_h` is computed and never used. The comment at `:206-208` claims "push h along the direction that increases alignment with h_star. Use a simple fixed-point update toward h_star scaled by error magnitude" — but the update is **not** scaled by error magnitude. It is a constant `settle_lr`. **Severity: REAL** (misleading comment + dead code).

### 6.2 `contrastive_weight` / `contrastive_negatives` — stored but never used

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:100-101` — stored on the module.
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:203-204, 232-235` — persisted and restored.
- `ravana/src/ravana/decoder/engine.py:22-23, 165-166` — passed in config.
- `ravana/src/ravana/chat/response_gen.py:345-346, 501-502` — passed at construction.
- **No contrastive loss exists anywhere in `train_on_sentence`.** Grep for `contrastive_weight|contrastive_negatives` finds only the above storage/persistence sites — no computation. **Severity: MINOR** (dead config, no runtime effect).

### 6.3 `av_head` — used only when `aux_lambda > 0`

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:140` — `self.av_head = Linear(hidden_dim, 65)`.
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:1072-1087` — used only when `av_target is not None` (i.e. `_has_persistent and aux_lambda > 0 and sensorimotor_conditioning is not None`).
- `scripts/train.py:404` — `_mode_linggen` passes `AUX_LAMBDA=0.5`, so `av_head` is trained in that mode. But `av_head` is **never used at inference** — `generate()` and `PredictiveCodingGenerator` do not read it. It is a training-only auxiliary head. **Severity: MINOR** (dead at inference, but intentional as an auxiliary loss).

### 6.4 `ConceptAttentionHead.forward()` traced path — dead

- `ravana_ml/src/ravana_ml/nn/module.py:496-505` — docstring claims `accumulate_free_energy` can update W_q/W_k/W_v/output_proj during training.
- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:861` — `cached_attn_tensor = self.attention(StateTensor(cond_proj))` uses the traced `forward()`, but the resulting trace is never consumed by any `accumulate_free_energy` call. The only attention update is the manual `output_proj` update at `:1133-1137`. **Severity: MINOR** (dead trace path, misleading docstring).

### 6.5 `load_from` hardcodes `n_attention_heads=4`

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:231` — `n_attention_heads=4` regardless of saved config.
- The saved config does not include `n_attention_heads` (only `_vocab_size`, `_embed_dim`, `_hidden_dim`, `_contrastive_weight`, `_contrastive_negatives` — see `:200-204`).
- Production constructs with `n_attention_heads=4` (`response_gen.py:340-348`), so the hardcoded value matches. But if a checkpoint were trained with a different head count, `load_state_dict` would silently filter the mismatched `W_q/W_k/W_v` shapes (`:1383`) and leave them at random init. **Severity: MINOR** (latent, currently masked by the production convention).

### 6.6 `_readout` comment says `(vocab_size,)` but returns `(hidden_dim,)`

- `ravana/src/ravana/decoder/predictive_coding_generator.py:100` — `return nd.attention.forward_raw(cond_proj)  # (vocab_size,)`. Actual shape is `(hidden_dim,)`. **Severity: MINOR** (misleading comment, no runtime effect).

### 6.7 `forward()` docstring says `precomputed_attn_logits` is `(vocab_size,)`

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:298` — "precomputed_attn_logits: optional (vocab_size,)". Actual shape is `(hidden_dim,)`. **Severity: MINOR**.

### 6.8 `_linggen_W_sm` not persisted in the decoder checkpoint

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:1296-1304` — `set_linggen_projection` stores `self._linggen_W_sm`.
- `save_to`/`load_from` (`:182-265`) do **not** persist `_linggen_W_sm`. It is re-attached at boot by `engine._restore_linggen_conditioner` (`engine.py:12139-12167`) from `data/linggen_wsm.npz`.
- Live probe: `dec._linggen_W_sm` is **None** after `load_from` — confirmed. **Severity: MINOR** (by design, but means the decoder checkpoint alone is not self-contained).

### 6.9 `W_h_bias` is trained but not used in the settle generator

- `ravana_ml/src/ravana_ml/nn/neural_decoder.py:1149-1154` — `W_h_bias` is updated when `_has_persistent`.
- `ravana/src/ravana/decoder/predictive_coding_generator.py:91-94` — `_gru_step` applies `W_h_bias` only when `persistent_emb is not None`.
- Production settle call sites (`response_gen.py:2987-2988, 3021-3022, 3055-3056`) pass **no** `initial_emb`/`persistent_emb`. So `W_h_bias` is never applied in the settle path. **Severity: REAL** (the working-memory gate is dead in the settle generator; the `W_sm` conditioning is only used in the autoregressive `_generate_with_situation_model` path).

### 6.10 `_sm_response_grounded` gate

- `ravana/src/ravana/chat/response_gen.py:4271+` — Levelt-style monitor. Not audited in depth here, but it is the gate that decides whether the settle output is allowed through. If it rejects, the caller falls back to template generation. **Severity: NON-ISSUE** (out of scope for this audit).

---

## 7. Live probe results (shipped checkpoint)

All probes run against `data/neural_decoder.npz` (vocab 17021, embed 75, hidden 256, 4 heads) with `data/neural_decoder_vocab.json`.

### 7.1 cos(h, eos) over one sentence

| Input | cos(h, eos) | eos rank | eos logit | `||h||` | `||attn||` |
|---|---|---|---|---|---|
| `gravity` | 0.1093 | 144 | 0.043 | 0.42 | 0.33 |
| `gravity is` | 0.6052 | 2 | 4.036 | 8.17 | 4.93 |
| `gravity is the` | 0.5068 | 3 | 5.098 | 12.20 | 2.08 |
| `gravity is the force` | 0.8323 | 1 | 6.521 | 9.52 | 1.78 |
| `gravity is the force that` | 0.9401 | 1 | 8.445 | 10.91 | 1.84 |
| `gravity is the force that pulls` | 0.9883 | 0 | 8.701 | 10.71 | 1.24 |

Confirms the reported symptom: cos(h, eos) rises 0.11 → 0.99 over one sentence. The eos logit goes from rank 144 to rank 0.

### 7.2 Top-20 readout after `gravity is the force`

`<bos>` 6.879, `<eos>` 6.521, `of` 5.198, `is` 5.014, `to` 3.941, `or` 3.927, `and` 3.844, `that` 3.82, `world` 3.81, `between` 3.756, `the` 3.714, `on` 3.68, `in` 3.653, `for` 3.647, `way` 3.627, `system` 3.596, `study` 3.503, `most` 3.485, `environment` 3.483, `ocean` 3.449.

The top-2 are special tokens; the top-10 are dominated by function words. Content words appear only from rank 9 onward.

### 7.3 Weight norms

| Parameter | frobenius | expected random | verdict |
|---|---|---|---|
| `attention.W_q` | 22.57 | 22.63 | random init (untrained) |
| `attention.W_k` | 22.64 | 22.63 | random init (untrained) |
| `attention.W_v` | 22.70 | 22.63 | random init (untrained) |
| `attention.output_proj` | 1.159 | 9.24 | trained/weight-decayed to near-zero |
| `output_proj` | 124.87 | — | trained |
| `gru.W_z` | 1.480 | 22.66 | frozen (freeze_core=True) |
| `gru.W_r` | 1.031 | 22.60 | frozen |
| `gru.W_h` | 2.835 | 22.63 | frozen |
| `W_h_bias` | 0.424 | — | trained (but unused in settle path) |
| `condition_proj` | 21.77 | — | trained |
| `word_embedding` | 152.14 | — | trained |

GRU biases are exactly 0.0 (confirmed) — consistent with `freeze_core=True` and the GRU never being trained.

### 7.4 Attention contribution to readout variance

| Scenario | `var(logits_h)` | `var(logits_attn)` | fraction from attn |
|---|---|---|---|
| real `h`, real `attn` | 0.4329 | 0.0027 | **0.63%** |
| random `h`, random `attn` | 0.2250 | 0.00017 | **0.074%** |

---

## 8. Recommended fix order

### Fix 1 (BLOCKER) — Make training target the state inference reads

The settle loop reads `h_star = GRU(emb(w_t), h_{t-1})`. Training must include a loss on `readout(h_star)` — i.e. a "post-token" prediction loss, or a contrastive loss that pulls `readout(h_star)` toward the *next* token after `w_t`. Concretely:

- Add a second forward pass per sentence: after computing `h_t` for each position, compute `h_star_t = GRU(emb(w_t), h_{t-1})` and add `CE(readout(h_star_t), w_{t+1})` to the loss.
- Or: change the settle loop to read `h` (the state *before* the target) instead of `h_star`, and train a separate "pre-token" head. This is simpler but requires retraining.

### Fix 2 (BLOCKER) — Replace the settle loop with actual predictive coding

Either:
- (a) Implement a real gradient step on the readout error: `h ← h − η · ∇_h ||readout(h) − readout(h_star)||²`, using the Jacobian of `output_proj`. This is the honest predictive-coding update the comments claim.
- (b) Or abandon the settle loop and use the autoregressive `generate()` path with the `W_sm` conditioning (which is already wired in `_generate_with_situation_model`). The settle path is not producing value over the autoregressive path.

### Fix 3 (REAL) — Fix the attention head wiring

- Change `neural_decoder.py:114` to `ConceptAttentionHead(hidden_dim, vocab_size, n_heads=...)` so `output_proj` is `Linear(256,17021)`. Then `attn_logits` is genuinely vocab-sized and the blend `h*0.5 + attn_logits*0.5` is a shape error — so instead use `logits = output_proj(h) + attn_logits` (residual logits) or `logits = output_proj(h + attn_hidden)` with a separate `Linear(256,256)` hidden head.
- Retrain W_q/W_k/W_v (remove the `:1139` skip) or delete the attention head entirely if it is not pulling weight.

### Fix 4 (REAL) — Fix the negative-sampling distribution

- Replace the `log1p(5·row_norm)` distribution with a mixture: 50% uniform over content words, 50% frequency-based (or the current norm-based). This breaks the rich-get-richer loop.
- Or: use a full softmax over a bounded vocab (e.g. the 1500 most frequent words) instead of 50 negatives.

### Fix 5 (REAL) — Wire `W_h_bias` into the settle generator

- Pass `initial_emb`/`persistent_emb` at the settle call sites (`response_gen.py:2987, 3021, 3055`) using the same `W_sm @ av` computation as `_generate_with_situation_model` (`:1630-1649`).
- Or: accept that the settle path is unconditioned and remove the `W_h_bias` training (it is dead weight).

### Fix 6 (MINOR) — Clean up dead code

- Delete `err` in the settle loop (`predictive_coding_generator.py:205`) or use it.
- Delete `contrastive_weight`/`contrastive_negatives` or implement the contrastive loss.
- Fix the `_readout` and `forward()` docstring shape comments.
- Persist `n_attention_heads` in `save_to`/`load_from`.
- Persist `_linggen_W_sm` in the decoder checkpoint or document that it is external.

---

## 9. Summary table

| # | Finding | Severity |
|---|---|---|
| 1 | Readout blend is dimensionally coherent but semantically incoherent (double projection) | REAL |
| 2 | Training objective (pre-token state) ≠ inference state (post-token `h_star`) | BLOCKER |
| 3 | Attention head mis-wired (`vocab_size=hidden_dim`); W_q/W_k/W_v never trained; output_proj near-zero; <1% readout variance | REAL |
| 4 | Sampled softmax: 51/17021 rows; norm-biased negatives; no content-word coverage | REAL |
| 5 | Settle loop is linear interpolation, not predictive coding; emits attractor's own words | BLOCKER |
| 6.1 | `err` in settle loop is dead code | REAL |
| 6.2 | `contrastive_weight`/`contrastive_negatives` stored but never used | MINOR |
| 6.3 | `av_head` trained but never used at inference | MINOR |
| 6.4 | `ConceptAttentionHead.forward()` traced path is dead | MINOR |
| 6.5 | `load_from` hardcodes `n_attention_heads=4` | MINOR |
| 6.6 | `_readout` comment says `(vocab_size,)` but is `(hidden_dim,)` | MINOR |
| 6.7 | `forward()` docstring says `precomputed_attn_logits` is `(vocab_size,)` | MINOR |
| 6.8 | `_linggen_W_sm` not persisted in decoder checkpoint | MINOR |
| 6.9 | `W_h_bias` trained but not used in settle generator | REAL |
