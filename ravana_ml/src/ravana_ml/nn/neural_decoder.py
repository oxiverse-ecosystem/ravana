"""
RAVANA Neural Decoder — Corticostriatal Next-Word Generation
=============================================================
Learns to generate fluent text from concept graph walk embeddings
by predicting the next word given previous context + conditioning.

Architecture (Dominey 2013 corticostriatal reservoir + predictive coding):
  Input concept embeddings (conditioning from graph walk)
      |
      v
  GRU (cortical sequence processing, maintains temporal state)
      |
  ConceptAttentionHead (attends over conditioning context)
      |
  Linear projection → logits over vocabulary
      |
  BasalGangliaGate (Go/NoGo competitive word selection during generation)
      |
  Output word

Training: unsupervised next-word prediction on web article text.
Each word is mapped to its concept/GloVe embedding. The decoder learns
to predict the next embedding via local predictive coding error signals
(Hebbian updates, no backprop).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Callable
from ..tensor import StateTensor, Parameter, RawTensor
from .module import Module, Linear, Embedding, GRUCell, Dropout, ConceptAttentionHead
import re


class TracedGRUCell(GRUCell):
    """GRUCell variant that stores traces on internal Linear modules via __call__
    instead of forward_raw, enabling Hebbian learning of GRU weights."""

    def forward_traced(self, x, h_prev):
        x_data = np.asarray(x, dtype=np.float32)
        h_data = np.asarray(h_prev, dtype=np.float32)
        combined = np.concatenate([x_data, h_data])
        combined_2d = combined[np.newaxis, :]

        z_out = self.W_z(StateTensor(combined_2d)).data[0]
        z = 1.0 / (1.0 + np.exp(-np.clip(z_out, -100, 100)))
        r_out = self.W_r(StateTensor(combined_2d)).data[0]
        r = 1.0 / (1.0 + np.exp(-np.clip(r_out, -100, 100)))
        combined_r = np.concatenate([x_data, r * h_data])
        h_candidate_pre = self.W_h(StateTensor(combined_r[np.newaxis, :])).data[0]
        h_candidate = np.tanh(h_candidate_pre)
        h_new = (1.0 - z) * h_data + z * h_candidate

        self._last_combined = combined
        self._last_combined_r = combined_r
        self._last_z = z
        self._last_r = r
        self._last_h_prev = h_data
        self._last_x = x_data
        self._last_h_candidate = h_candidate

        return h_new


class NeuralDecoder(Module):
    """Corticostriatal neural decoder for graph-conditioned text generation.

    Learns to map sequences of concept embeddings to fluent word sequences
    via unsupervised next-word prediction on web article text.

    The decoder processes a conditioning context (graph walk concept embeddings)
    and generates one word at a time autoregressively.

    Key design:
    - Conditioning: graph walk concept embeddings provide semantic context
    - Generation: GRU processes the generated sequence so far, attention pools
      conditioning context, Linear projects to vocab logits
    - Training: unsupervised next-word prediction on article sentences;
      uses local predictive coding errors (no backprop) via Module's Hebbian interface
    - Selection: BasalGangliaGate for Go/NoGo competitive word selection at generation time

    Parameters:
        vocab_size: size of output vocabulary
        embed_dim: dimension of concept embeddings (matches graph)
        hidden_dim: GRU hidden state dimension
        n_attention_heads: number of heads in ConceptAttentionHead
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, n_attention_heads: int = 2,
                 contrastive_weight: float = 0.5,
                 contrastive_negatives: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.contrastive_weight = contrastive_weight
        self.contrastive_negatives = contrastive_negatives

        # Word embedding (output vocabulary) — maps word indices to embeddings
        self.word_embedding = Embedding(vocab_size, embed_dim)

        # Conditioning projection — project conditioning concepts into GRU space
        self.condition_proj = Linear(embed_dim, hidden_dim)

        # GRU for sequence processing (temporal cortex)
        self.gru = TracedGRUCell(hidden_dim, hidden_dim)

        # Attention over conditioning context
        self.attention = ConceptAttentionHead(hidden_dim, hidden_dim, n_heads=n_attention_heads)

        # Output projection: hidden state → logits over vocabulary
        self.output_proj = Linear(hidden_dim, vocab_size)

        # Dropout for regularization
        self.dropout = Dropout(0.2)

        # Vocabulary dimension
        self._vocab_dim = embed_dim

        # Cache vocab embeddings for fast similarity lookup during generation
        self._vocab_embed_cache: Optional[np.ndarray] = None

        # Track recent predictions for cerebellar-style sequence learning
        self._recent_predictions: List[int] = []

        # Learning statistics
        self._total_training_examples = 0
        self._avg_prediction_error = 0.0
        # Real metrics (replaces the fake |one-hot - probs| mean that rounded
        # to a constant ~2/vocab_size and never moved). Cross-entropy in nats;
        # top-1/top-5 next-word accuracy in [0, 1]. Tracked as EMAs so the
        # caller can poll progress without storing per-token history.
        self._avg_cross_entropy = 0.0
        self._avg_top1_acc = 0.0
        self._avg_top5_acc = 0.0
        self._metric_examples = 0  # EMA sample count for the metrics above

    def rebuild_vocab_cache(self):
        """Rebuild internal vocabulary embedding cache for fast similarity lookup."""
        self._vocab_embed_cache = self.word_embedding._w_raw.copy()

    def reset_plasticity(self, stability: float = 0.5):
        """Reset every parameter's stability to keep the network learnable.

        After many sleep cycles stability climbs toward the 0.8 cap, shrinking
        plasticity (1 - stability) and slowing learning. Call this before a
        fresh training burst (e.g. a Phase 2 run on a new corpus) so updates
        land at full strength. Does not touch learned weights.
        """
        for _, param in self.named_parameters():
            if hasattr(param, '_stability'):
                param._stability = float(np.clip(stability, 0.0, 0.8))
        # Rebuild raw caches for submodules (defensive; stability change
        # doesn't move weights, but keep views consistent).
        for mod in self.modules():
            if hasattr(mod, '_rebuild_raw_cache'):
                mod._rebuild_raw_cache()

    def forward(self, conditioning_embs: np.ndarray,
                input_seq: np.ndarray,
                h_prev: Optional[np.ndarray] = None,
                use_raw: bool = True,
                _use_gru_traced: bool = False,
                precomputed_attn_logits: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass for training (next-word prediction).

        Args:
            conditioning_embs: (n_concepts, embed_dim) — graph walk concept embeddings
            input_seq: (seq_len,) — word indices (input sequence, shifted by 1)
            h_prev: Optional (hidden_dim,) — initial hidden state
            use_raw: if True, return raw numpy arrays (no StateTensor wrapping)
            _use_gru_traced: if True, use GRU.forward_traced to store traces for Hebbian learning
            precomputed_attn_logits: optional (vocab_size,) — cached attention output for
                this sentence's conditioning context. When provided, skips recomputing
                the full multi-head attention on every timestep (~3-5x speedup).

        Returns:
            logits: (seq_len, vocab_size) — prediction logits for each position
            h_final: (hidden_dim,) — final hidden state
        """
        seq_len = len(input_seq)
        if h_prev is None:
            h = np.zeros(self.hidden_dim, dtype=np.float32)
        else:
            h = h_prev.copy()

        # Only compute conditioning projection for attention if not precomputed
        if precomputed_attn_logits is None:
            cond_proj_tensor = self.condition_proj(StateTensor(conditioning_embs))
            cond_proj = cond_proj_tensor.data

        logits_list = []
        for t in range(seq_len):
            word_idx = int(input_seq[t])
            word_emb = self.word_embedding.embed_raw(word_idx)

            # Use forward_raw for condition_proj (trace never consumed by Hebbian)
            proj_emb = self.condition_proj.forward_raw(word_emb[np.newaxis, :])[0]

            if _use_gru_traced:
                h = self.gru.forward_traced(proj_emb, h)
            else:
                h = self.gru(proj_emb, h)

            if precomputed_attn_logits is not None:
                attn_logits = precomputed_attn_logits
            else:
                attn_logits = self.attention.forward_raw(cond_proj)
            combined = h * 0.5 + attn_logits * 0.5
            if self.training and self.dropout.p > 0:
                mask = np.random.binomial(1, 1.0 - self.dropout.p, combined.shape).astype(combined.dtype)
                combined = combined * mask / (1.0 - self.dropout.p)

            # Use forward for trace storage at output projection
            out_tensor = self.output_proj(StateTensor(combined[np.newaxis, :]))
            logits_t = out_tensor.data[0]
            logits_list.append(logits_t)

        return np.stack(logits_list, axis=0), h

    def generate(self, conditioning_embs: np.ndarray,
                 max_steps: int = 20,
                 bos_idx: int = 0,
                 eos_idx: Optional[int] = None,
                 temperature: float = 0.5,
                 cerebellar_ngram=None,
                 idx_to_word: Optional[Dict[int, str]] = None,
                 basal_ganglia=None,
                 content_word_ids: Optional[Set[int]] = None,
                 token_boost: Optional[Dict[int, float]] = None) -> List[int]:
        """Generate a sequence autoregressively conditioned on concept embeddings.

        Optionally uses cerebellar n-gram bias and basal ganglia gating
        for biologically-plausible word selection.

        Args:
            conditioning_embs: (n_concepts, embed_dim) — graph walk concept embeddings
            max_steps: maximum number of tokens to generate
            bos_idx: beginning-of-sequence token index
            eos_idx: end-of-sequence token index (None = no early stopping)
            temperature: softmax temperature (lower = more deterministic)
            cerebellar_ngram: optional CerebellarNgram for transition bias
            idx_to_word: optional {idx: word} mapping for cerebellar lookups
            basal_ganglia: optional BasalGangliaGate for Go/NoGo selection

        Returns:
            List of generated token indices
        """
        if self._vocab_embed_cache is None:
            self.rebuild_vocab_cache()

        h = np.zeros(self.hidden_dim, dtype=np.float32)
        cond_proj = self.condition_proj.forward_raw(conditioning_embs)

        generated = [bos_idx]
        confidence_sum = 0.0
        for step in range(max_steps):
            word_emb = self.word_embedding.embed_raw(generated[-1])
            projected_word = self.condition_proj.forward_raw(word_emb[np.newaxis, :])[0]
            h = self.gru(projected_word, h)

            attn_logits = self.attention.forward_raw(cond_proj)
            combined = h * 0.5 + attn_logits * 0.5
            logits = self.output_proj.forward_raw(combined[np.newaxis, :])[0]

            # Cerebellar bias: boost transitions learned from past chain walks
            # Optimization: pre-filter to tokens with logits > 0.01 (3-5x speedup)
            if cerebellar_ngram is not None and idx_to_word is not None:
                last_word = idx_to_word.get(generated[-1], "")
                if last_word:
                    active_idx = np.where(logits > 0.01)[0]
                    if len(active_idx) < 3:
                        # Fallback: top-10 highest logits
                        active_idx = np.argsort(logits)[-10:]
                    for idx in active_idx:
                        cand = idx_to_word.get(idx, "")
                        if cand:
                            trans = cerebellar_ngram.get_transition_strength(last_word, cand)
                            if trans > 0:
                                logits[idx] += trans * 0.15
                            fw = cerebellar_ngram.predict_function_word(last_word, cand)
                            if fw and fw == cand:
                                logits[idx] += 0.3

            # Track seen counts for repetition & content-word biasing
            if len(generated) > 1:
                seen_counts = {}
                for idx in generated[1:]:
                    seen_counts[idx] = seen_counts.get(idx, 0) + 1
            else:
                seen_counts = {}

            # Content word biasing: boost tokens that are known graph concepts
            # to counteract the function-word bias learned from template training.
            # Stage 2: Stronger bias (×3) for 1500-vocab models where function-word
            # logits can be 3-5 units above content-word logits.
            if content_word_ids is not None and len(generated) > 2:
                func_count = sum(1 for idx in generated[1:] if idx not in content_word_ids)
                if func_count > len(generated) * 0.5:
                    boost = 3.0
                    for cid in content_word_ids:
                        logits[cid] += boost
                        if seen_counts.get(cid, 0) == 0:
                            logits[cid] += boost * 0.5

            # Stage 2: Token-level boost (e.g. subject concept to seed content words)
            if token_boost is not None:
                for tid, tboost in token_boost.items():
                    if tid < len(logits):
                        logits[tid] += tboost

            # Adaptive repetition penalty: suppress cycling between function words
            if len(generated) > 1:
                for idx, count in seen_counts.items():
                    if count >= 3:
                        logits[idx] -= 5.0
                    elif count > 1:
                        is_func = content_word_ids is not None and idx not in content_word_ids
                        penalty = 1.5 + 1.0 * (count - 1) if is_func else 1.0 + 0.5 * (count - 1)
                        logits[idx] -= penalty

            # BasalGangliaGate Go/NoGo selection
            if basal_ganglia is not None and idx_to_word is not None:
                candidates = []
                raw_max = np.max(np.abs(logits)) if np.max(np.abs(logits)) > 0 else 1.0
                # Optimization: pre-filter to tokens with logits > 0.01
                active_idx = np.where(logits > 0.01)[0]
                if len(active_idx) < 3:
                    active_idx = np.argsort(logits)[-10:]
                for idx in active_idx:
                    word = idx_to_word.get(idx, "")
                    if word and not word.startswith("<"):
                        score = float(logits[idx] / raw_max)  # normalize to ~[-1, 1]
                        candidates.append((word, score, 0.5, "decoder"))
                if len(candidates) >= 2:
                    basal_ganglia.set_dopamine_tone(1.0 - temperature)
                    winner_label, _, _ = basal_ganglia.select_concept(candidates)
                    winner_idx = None
                    for idx, w in idx_to_word.items():
                        if w == winner_label:
                            winner_idx = idx
                            break
                    if winner_idx is not None and winner_idx != bos_idx:
                        if eos_idx is not None and winner_idx == eos_idx:
                            break
                        generated.append(winner_idx)
                        self._recent_predictions.append(winner_idx)
                        continue
                # Fallthrough to softmax if BG gate fails
                pass

            # Apply temperature and sample (standard softmax)
            if temperature > 0:
                # Dynamic temperature: slightly increase if confidence has been low
                dyn_temp = temperature
                if step > 2:
                    avg_conf = confidence_sum / max(1, step)
                    if avg_conf < 0.3:
                        dyn_temp = min(0.8, temperature * 1.3)
                logits = logits / dyn_temp

                logits_exp = np.exp(logits - np.max(logits))
                probs = logits_exp / (np.sum(logits_exp) + 1e-10)

                # Top-p (nucleus) sampling: keep smallest set of tokens whose prob >= 0.92
                sorted_idx = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_idx]
                cumsum = np.cumsum(sorted_probs)
                cutoff = np.searchsorted(cumsum, 0.92) + 1
                top_p_idx = sorted_idx[:cutoff]
                top_p_probs = probs[top_p_idx]
                top_p_probs = top_p_probs / np.sum(top_p_probs)
                try:
                    chosen = np.random.choice(top_p_idx, p=top_p_probs)
                    confidence_sum += float(np.max(probs))
                except Exception:
                    chosen = int(np.argmax(probs))
            else:
                chosen = int(np.argmax(logits))

            if eos_idx is not None and chosen == eos_idx:
                break

            generated.append(chosen)
            self._recent_predictions.append(chosen)

        return generated[1:]  # strip BOS

    def train_on_sentence(self, sentence_words: List[str],
                          word_to_embed: Dict[str, np.ndarray],
                          word_to_idx: Dict[str, int],
                          unknown_idx: int = 1,
                          conditioning_embs: Optional[np.ndarray] = None,
                          word_indices: Optional[List[int]] = None) -> float:
        """Unsupervised learning from a sentence (fully batched Hebbian updates).

        Forward pass stores all errors and intermediates; then at the end
        all Hebbian updates are computed in a few large matmuls.
        ~5-10x faster than per-token accumulate calls.
        """
        if len(sentence_words) < 2:
            return 0.0

        if conditioning_embs is None:
            known_embs = []
            for w in sentence_words:
                wl = w.lower().strip(".,!?").strip("'")
                if wl in word_to_embed:
                    known_embs.append(word_to_embed[wl])
                elif wl:
                    rand_emb = np.random.randn(self.embed_dim).astype(np.float32) * 0.1
                    norm = np.linalg.norm(rand_emb)
                    if norm > 0:
                        rand_emb /= norm
                    known_embs.append(rand_emb)
            if len(known_embs) < 2:
                known_embs_flat = [np.random.randn(self.embed_dim).astype(np.float32) * 0.1
                                   for _ in range(max(2, len(sentence_words)))]
                for i in range(len(known_embs_flat)):
                    n = np.linalg.norm(known_embs_flat[i])
                    if n > 0:
                        known_embs_flat[i] /= n
                conditioning_embs = np.stack(known_embs_flat, axis=0)
            else:
                conditioning_embs = np.stack(known_embs, axis=0)

        if word_indices is None:
            word_indices = []
            for w in sentence_words:
                wl = w.lower().strip(".,!?").strip("'")
                if wl in word_to_idx:
                    word_indices.append(word_to_idx[wl])
                else:
                    word_indices.append(unknown_idx)

        if len(word_indices) < 2:
            return 0.0

        cond_proj = self.condition_proj.forward_raw(conditioning_embs)
        cached_attn_tensor = self.attention(StateTensor(cond_proj))
        cached_attn_logits = cached_attn_tensor.data[0]

        out_weight = self.output_proj.weight.data
        vocab_size_actual = out_weight.shape[0]

        # Stage 2: Precompute contrastive candidates once per sentence (not per pos)
        # This replaces the per-token np.ones(vocab) + np.where + argpartition overhead.
        contrastive_precomputed = None
        _cweight = self.contrastive_weight
        _cnegs = self.contrastive_negatives
        if _cweight > 0 and _cnegs > 0:
            special_set = frozenset({0, 1, 2, 3})
            non_special = np.array([i for i in range(vocab_size_actual) if i not in special_set], dtype=np.intp)
            contrastive_precomputed = non_special

        # Buffers
        all_combined: List[np.ndarray] = []
        all_error_out: List[np.ndarray] = []
        all_word_emb: List[np.ndarray] = []
        all_input_idx: List[int] = []
        # GRU buffers
        all_gru_combined: List[np.ndarray] = []
        all_gru_combined_r: List[np.ndarray] = []
        all_gru_h_candidate: List[np.ndarray] = []
        all_gru_h_prev: List[np.ndarray] = []
        all_gru_z: List[np.ndarray] = []
        all_gru_r: List[np.ndarray] = []
        total_ce = 0.0
        top1_hits = 0
        top5_hits = 0
        trained = 0

        h = np.zeros(self.hidden_dim, dtype=np.float32)
        for pos in range(len(word_indices) - 1):
            input_idx = word_indices[pos]
            target_idx = word_indices[pos + 1]

            word_emb = self.word_embedding.embed_raw(input_idx)
            proj_emb = self.condition_proj.forward_raw(word_emb[np.newaxis, :])[0]

            # Manual GRU (stores intermediates for batched update)
            x_data = np.asarray(proj_emb, dtype=np.float32)
            h_data = np.asarray(h, dtype=np.float32)
            combined = np.concatenate([x_data, h_data])
            combined_2d = combined[np.newaxis, :]
            z_out = self.gru.W_z.forward_raw(combined_2d)[0]
            z = 1.0 / (1.0 + np.exp(-np.clip(z_out, -100, 100)))
            r_out = self.gru.W_r.forward_raw(combined_2d)[0]
            r = 1.0 / (1.0 + np.exp(-np.clip(r_out, -100, 100)))
            combined_r = np.concatenate([x_data, r * h_data])
            h_candidate_pre = self.gru.W_h.forward_raw(combined_r[np.newaxis, :])[0]
            h_candidate = np.tanh(h_candidate_pre)
            h_new = (1.0 - z) * h_data + z * h_candidate

            all_gru_combined.append(combined.copy())
            all_gru_combined_r.append(combined_r.copy())
            all_gru_h_candidate.append(h_candidate.copy())
            all_gru_h_prev.append(h_data.copy())
            all_gru_z.append(z.copy())
            all_gru_r.append(r.copy())
            h = h_new

            # Attention + output (keep ratio consistent with generate: 0.5 each)
            combined_vec = h * 0.5 + cached_attn_logits * 0.5
            if self.training and self.dropout.p > 0:
                mask = np.random.binomial(1, 1.0 - self.dropout.p, combined_vec.shape).astype(combined_vec.dtype)
                combined_vec = combined_vec * mask / (1.0 - self.dropout.p)
            logit_t = self.output_proj.forward_raw(combined_vec[np.newaxis, :])[0]

            logit_stable = logit_t - np.max(logit_t)
            exp_logits = np.exp(np.clip(logit_stable, -50, 50))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)

            target_one_hot = np.zeros(vocab_size_actual, dtype=np.float32)
            if target_idx < vocab_size_actual:
                target_one_hot[target_idx] = 1.0

            error_output = (target_one_hot - probs).astype(np.float32)
            error_output = np.clip(error_output, -1.0, 1.0)

            # Stage 2: Vectorized contrastive using precomputed candidate set
            if contrastive_precomputed is not None and (pos % 2 == 0):
                non_special = contrastive_precomputed
                # Remove target from candidates efficiently
                if target_idx < vocab_size_actual:
                    mask = non_special != target_idx
                    candidates = non_special[mask]
                else:
                    candidates = non_special
                if len(candidates) > 0:
                    k = min(_cnegs, len(candidates))
                    probs_cand = probs[candidates]
                    neg_top = np.argpartition(probs_cand, -k)[-k:]
                    neg_indices = candidates[neg_top]
                    if target_idx < vocab_size_actual:
                        pos_margin = 0.9 - probs[target_idx]
                        error_output[target_idx] += _cweight * pos_margin
                    error_output[neg_indices] -= _cweight * probs[neg_indices] * 1.5

            error_output = np.clip(error_output, -1.0, 1.0)

            all_combined.append(combined_vec.copy())
            all_error_out.append(error_output.copy())
            all_word_emb.append(word_emb.copy())
            all_input_idx.append(input_idx)

            if target_idx < vocab_size_actual:
                p_t = max(float(probs[target_idx]), 1e-10)
                total_ce += -np.log(p_t)
                if int(np.argmax(probs)) == target_idx:
                    top1_hits += 1
                if target_idx in np.argpartition(probs, -5)[-5:]:
                    top5_hits += 1
            trained += 1

        if trained == 0:
            return 0.0

        # ── Batched Hebbian updates ──
        T = len(all_combined)
        fe_decay = 0.99 ** T

        # 1. output_proj
        combined_stack = np.stack(all_combined)
        err_out_stack = np.stack(all_error_out)
        hebbian_out = combined_stack.T @ err_out_stack
        self.output_proj._weight_free_energy.data += hebbian_out.T * 0.15
        self.output_proj._weight_free_energy.data *= fe_decay

        error_h_all = err_out_stack @ self.output_proj.weight.data

        # 2. condition_proj
        emb_stack = np.stack(all_word_emb)
        error_proj_all = np.clip(error_h_all * 0.5, -1.0, 1.0)
        hebbian_proj = emb_stack.T @ error_proj_all
        self.condition_proj._weight_free_energy.data += hebbian_proj.T * 0.1
        self.condition_proj._weight_free_energy.data *= fe_decay

        # 3. word_embedding
        W_cond = self.condition_proj.weight.data
        error_word_all = np.clip(error_proj_all @ W_cond, -1.0, 1.0)
        for i, idx in enumerate(all_input_idx):
            self.word_embedding._weight_free_energy.data[idx] += error_word_all[i] * 0.05
        self.word_embedding._weight_free_energy.data *= fe_decay

        # 4. GRU W_z / W_r / W_h
        gru_c_stack = np.stack(all_gru_combined)
        gru_cr_stack = np.stack(all_gru_combined_r)
        gru_hc_stack = np.stack(all_gru_h_candidate)
        gru_hp_stack = np.stack(all_gru_h_prev)
        gru_z_stack = np.stack(all_gru_z)

        err_wz = error_h_all * (gru_hc_stack - gru_hp_stack) * 0.5
        fe_wz = gru_c_stack.T @ err_wz
        self.gru.W_z._weight_free_energy.data += fe_wz.T * 0.1
        self.gru.W_z._weight_free_energy.data *= fe_decay

        err_wr = error_h_all * 0.3
        fe_wr = gru_c_stack.T @ err_wr
        self.gru.W_r._weight_free_energy.data += fe_wr.T * 0.1
        self.gru.W_r._weight_free_energy.data *= fe_decay

        err_wh = error_h_all * gru_z_stack
        fe_wh = gru_cr_stack.T @ err_wh
        self.gru.W_h._weight_free_energy.data += fe_wh.T * 0.1
        self.gru.W_h._weight_free_energy.data *= fe_decay

        # 5. attention output_proj
        attn_error_all = np.clip(error_h_all * 0.1, -1.0, 1.0)
        total_attn_err = np.sum(attn_error_all, axis=0)
        self.attention.output_proj.accumulate_free_energy(
            StateTensor(total_attn_err[np.newaxis, :], salience=0.1))
        self.attention.output_proj._weight_free_energy.data *= (0.99 ** (T - 1))

        # Skip attention W_q/W_k/W_v (traces are sentence-specific, unreliable)

        avg_ce = total_ce / max(1, trained)
        t1 = top1_hits / max(1, trained)
        t5 = top5_hits / max(1, trained)

        self._total_training_examples += trained
        # EMA over sentences (each sentence ~10-30 tokens)
        alpha = 0.05
        self._avg_cross_entropy = (1 - alpha) * self._avg_cross_entropy + alpha * avg_ce
        self._avg_top1_acc = (1 - alpha) * self._avg_top1_acc + alpha * t1
        self._avg_top5_acc = (1 - alpha) * self._avg_top5_acc + alpha * t5
        self._metric_examples += 1
        # Keep legacy field for back-compat (now also EMA-tracked; callers
        # reading it get a non-constant value, unlike the old ~2/vocab artifact).
        self._avg_prediction_error = (1 - alpha) * self._avg_prediction_error + alpha * avg_ce

        return avg_ce

    def prepare_sentences(self, text: str,
                          word_to_embed: Dict[str, np.ndarray],
                          word_to_idx: Dict[str, int],
                          unknown_idx: int = 1,
                          min_sentence_len: int = 3,
                          max_sentences: Optional[int] = None) -> List[dict]:
        """Pre-process text into cached sentence data for fast training loops.

        Returns a list of dicts with 'words', 'word_indices', and 'conditioning_embs'
        so that repeated training passes avoid re-parsing and re-embedding the text.

        Use with train_on_sentence(word_indices=..., conditioning_embs=...) for
        ~2x speedup over repeated train_on_text calls.
        """
        sentences = []
        for sent in re.split(r'[.!?]+', text):
            words = [w.strip(".,!?\"' ") for w in sent.split()
                     if len(w.strip(".,!?\"' ")) > 0]
            if len(words) >= min_sentence_len:
                word_indices = []
                known_embs = []
                for w in words:
                    wl = w.lower().strip(".,!?").strip("'")
                    if wl in word_to_idx:
                        word_indices.append(word_to_idx[wl])
                    else:
                        word_indices.append(unknown_idx)
                    if wl in word_to_embed:
                        known_embs.append(word_to_embed[wl])
                    elif wl:
                        rand_emb = np.random.randn(self.embed_dim).astype(np.float32) * 0.1
                        norm = np.linalg.norm(rand_emb)
                        if norm > 0:
                            rand_emb /= norm
                        known_embs.append(rand_emb)
                if len(word_indices) < 2:
                    continue
                if len(known_embs) >= 2:
                    conditioning_embs = np.stack(known_embs, axis=0)
                else:
                    conditioning_embs = np.stack(
                        [np.random.randn(self.embed_dim).astype(np.float32) * 0.1
                         for _ in range(max(2, len(words)))], axis=0)
                    for i in range(len(conditioning_embs)):
                        n = np.linalg.norm(conditioning_embs[i])
                        if n > 0:
                            conditioning_embs[i] /= n
                sentences.append({
                    'words': words,
                    'word_indices': word_indices,
                    'conditioning_embs': conditioning_embs,
                })
        cap = max_sentences if max_sentences is not None else len(sentences)
        return sentences[:cap]

    def train_on_text(self, text: str,
                      word_to_embed: Dict[str, np.ndarray],
                      word_to_idx: Dict[str, int],
                      unknown_idx: int = 1,
                      min_sentence_len: int = 3,
                      max_sentences: Optional[int] = None,
                      conditioning_embs: Optional[np.ndarray] = None) -> Tuple[float, int]:
        """Train on a full text (sentence by sentence).

        Splits text into sentences, trains on each sentence independently.
        This reproduces how infants learn from continuous speech — they hear
        utterances bounded by prosodic cues (≈ sentence boundaries).

        Args:
            text: raw text to learn from
            word_to_embed: word → embedding vector mapping
            word_to_idx: word → index mapping
            unknown_idx: index for unknown words
            min_sentence_len: minimum words in a sentence to train on
            max_sentences: maximum number of sentences to train on per call.
                If None, defaults to 50 (safety cap to prevent OOM).
            conditioning_embs: optional pre-computed conditioning embeddings.
                If None, uses self-conditioning (sentence's own embeddings).

        Returns:
            (avg_error, sentences_trained): average error and count
        """
        # Simple sentence splitting
        sentences = []
        for sent in re.split(r'[.!?]+', text):
            words = [w.strip(".,!?\"' ") for w in sent.split()
                     if len(w.strip(".,!?\"' ")) > 0]
            if len(words) >= min_sentence_len:
                sentences.append(words)

        if not sentences:
            return 0.0, 0

        # Apply max_sentences cap if provided
        cap = max_sentences if max_sentences is not None else 50
        total_error = 0.0
        trained_count = 0
        for words in sentences[:cap]:
            err = self.train_on_sentence(words, word_to_embed, word_to_idx, unknown_idx,
                                         conditioning_embs=conditioning_embs)
            total_error += err
            trained_count += 1

        avg_err = total_error / max(1, trained_count)
        return avg_err, trained_count

    def sleep_cycle(self):
        """Consolidate learning via free-energy minimization (Hebbian sleep replay)."""
        super().sleep_cycle()
        self.rebuild_vocab_cache()

    def state_dict(self):
        sd = super().state_dict()
        sd['_total_training_examples'] = self._total_training_examples
        sd['_avg_prediction_error'] = float(self._avg_prediction_error)
        sd['_avg_cross_entropy'] = float(self._avg_cross_entropy)
        sd['_avg_top1_acc'] = float(self._avg_top1_acc)
        sd['_avg_top5_acc'] = float(self._avg_top5_acc)
        sd['_metric_examples'] = self._metric_examples
        return sd

    def load_state_dict(self, sd):
        # Filter out shape-mismatched parameters to prevent silent resizing
        filtered_sd = {}
        for name, param in self.named_parameters():
            if name in sd:
                entry = sd[name]
                saved_data = entry["data"] if isinstance(entry, dict) else entry
                if saved_data.shape == param.data.shape:
                    filtered_sd[name] = entry
        super().load_state_dict(filtered_sd)
        if '_total_training_examples' in sd:
            self._total_training_examples = sd['_total_training_examples']
        if '_avg_prediction_error' in sd:
            self._avg_prediction_error = sd['_avg_prediction_error']
        # New metrics (graceful back-compat with older checkpoints that lack them)
        self._avg_cross_entropy = float(sd.get('_avg_cross_entropy', 0.0))
        self._avg_top1_acc = float(sd.get('_avg_top1_acc', 0.0))
        self._avg_top5_acc = float(sd.get('_avg_top5_acc', 0.0))
        self._metric_examples = int(sd.get('_metric_examples', 0))
        self.rebuild_vocab_cache()


# Add re import at module level
import re
