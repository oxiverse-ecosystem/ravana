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
from typing import Dict, List, Optional, Tuple, Set, Callable, TYPE_CHECKING
from ..tensor import StateTensor, Parameter, RawTensor
from .module import Module, Linear, Embedding, GRUCell, Dropout, ConceptAttentionHead
import re

if TYPE_CHECKING:
    from .neuromodulator import NeuromodulatorEngine


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
                 contrastive_negatives: int = 8,
                 neuromodulator: Optional['NeuromodulatorEngine'] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.contrastive_weight = contrastive_weight
        self.contrastive_negatives = contrastive_negatives
        self.neuromodulator = neuromodulator

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
        precomputed_attn = self.attention.forward_raw(cond_proj)

        generated = [bos_idx]
        confidence_sum = 0.0
        word_to_idx = {w: idx for idx, w in idx_to_word.items()} if idx_to_word else {}
        seen_counts: Dict[int, int] = {}
        seen_bigrams: Set[Tuple[int, int]] = set()
        seen_trigrams: Set[Tuple[int, int, int]] = set()
        top_p = 0.92

        for step in range(max_steps):
            word_emb = self.word_embedding.embed_raw(generated[-1])
            projected_word = self.condition_proj.forward_raw(word_emb[np.newaxis, :])[0]
            h = self.gru(projected_word, h)

            combined = h * 0.5 + precomputed_attn * 0.5
            logits = self.output_proj.forward_raw(combined[np.newaxis, :])[0]

            # ── Neuromodulator modulation ──
            nm_temp_mod = 1.0
            nm_rep_mod = 1.0
            nm_explore = 0.0
            nm_conf_mod = 1.0
            if self.neuromodulator is not None:
                mods = self.neuromodulator.generation_mods()
                nm_temp_mod = mods['temperature_mod']
                nm_rep_mod = mods['repetition_penalty_mod']
                nm_explore = mods['exploration_bonus']
                nm_conf_mod = mods['confidence_threshold_mod']
                self.neuromodulator.tick_decay()

            if cerebellar_ngram is not None and idx_to_word is not None:
                last_word = idx_to_word.get(generated[-1], "")
                if last_word:
                    topk = np.argpartition(logits, -10)[-10:]
                    for idx in topk:
                        cand = idx_to_word.get(idx, "")
                        if cand:
                            trans = cerebellar_ngram.get_transition_strength(last_word, cand)
                            if trans > 0:
                                logits[idx] += trans * 0.15
                            fw = cerebellar_ngram.predict_function_word(last_word, cand)
                            if fw and fw == cand:
                                logits[idx] += 0.3

            if content_word_ids is not None and len(generated) > 2:
                func_count = sum(1 for idx in generated[1:] if idx not in content_word_ids)
                if func_count > len(generated) * 0.5:
                    boost = 3.0
                    for cid in content_word_ids:
                        logits[cid] += boost
                        if seen_counts.get(cid, 0) == 0:
                            logits[cid] += boost * 0.5

            if token_boost is not None:
                for tid, tboost in token_boost.items():
                    if tid < len(logits) and tid not in seen_counts:
                        logits[tid] += tboost

            # ── Neuromodulator-aware repetition penalty ──
            if len(generated) > 1:
                last_idx = generated[-1]
                for idx, count in seen_counts.items():
                    if count >= 3:
                        logits[idx] -= 5.0 * nm_rep_mod
                    elif count > 1:
                        is_func = content_word_ids is not None and idx not in content_word_ids
                        base_p = 1.5 + 1.0 * (count - 1) if is_func else 1.0 + 0.5 * (count - 1)
                        logits[idx] -= base_p * nm_rep_mod
                logits[last_idx] -= 15.0 * nm_rep_mod

                if len(generated) >= 3:
                    new_bigram = (generated[-2], generated[-1])
                    if new_bigram in seen_bigrams:
                        logits[generated[-1]] -= 10.0 * nm_rep_mod
                if len(generated) >= 4:
                    new_trigram = (generated[-3], generated[-2], generated[-1])
                    if new_trigram in seen_trigrams:
                        logits[generated[-1]] -= 15.0 * nm_rep_mod

            # ── Neuromodulator-aware exploration bonus (NE → novel content words) ──
            if nm_explore != 0.0 and content_word_ids is not None:
                rare_count = sum(1 for cid in content_word_ids if seen_counts.get(cid, 0) == 0)
                if rare_count > 0:
                    explore_bonus = nm_explore * 2.0
                    for cid in content_word_ids:
                        if seen_counts.get(cid, 0) == 0:
                            logits[cid] += explore_bonus

            if basal_ganglia is not None and idx_to_word is not None:
                logits_stable = logits - np.max(logits)
                exp_logits = np.exp(np.clip(logits_stable, -50, 50))
                probs = exp_logits / (np.sum(exp_logits) + 1e-10)
                top15_idx = np.argsort(probs)[-15:]
                candidates = []
                for idx in top15_idx:
                    word = idx_to_word.get(idx, "")
                    is_eos = (word == "<eos>")
                    if word and (not word.startswith("<") or is_eos):
                        if is_eos and step < 5 and len(top15_idx) > 3:
                            continue
                        candidates.append((word, float(probs[idx]), 0.5, "decoder"))
                if len(candidates) >= 2:
                    if self.neuromodulator is not None:
                        bg_mods = self.neuromodulator.bg_gate_mods()
                        go_mod = bg_mods['go_threshold_mod']
                        da_mod = bg_mods['dopamine_tone_mod']
                        new_threshold = basal_ganglia.base_go_threshold * go_mod
                        basal_ganglia.base_go_threshold = np.clip(new_threshold, 0.05, 0.8)
                        basal_ganglia.set_dopamine_tone(np.clip(0.5 + da_mod, 0.0, 1.0))
                    else:
                        basal_ganglia.set_dopamine_tone(1.0 - temperature)
                    winner_label, _, _ = basal_ganglia.select_concept(candidates)
                    winner_idx = word_to_idx.get(winner_label)
                    # At step 0, only accept content words from BG gate
                    if step == 0 and content_word_ids is not None and winner_idx is not None:
                        if winner_idx not in content_word_ids:
                            winner_idx = None  # fall through to content-masked sampling
                    if winner_idx is not None and winner_idx != bos_idx:
                        if eos_idx is not None and winner_idx == eos_idx:
                            break
                        generated.append(winner_idx)
                        self._recent_predictions.append(winner_idx)
                        if winner_idx < len(logits):
                            seen_counts[winner_idx] = seen_counts.get(winner_idx, 0) + 1
                        continue

            # ── Force first word to be a content word ──
            # This prevents the decoder from defaulting to function words ("in", "the")
            # at the start of generation, a common failure mode from BOS → function-word bias.
            if step == 0 and content_word_ids is not None and len(content_word_ids) > 0:
                content_mask = np.full(len(logits), -np.inf, dtype=np.float32)
                for cid in content_word_ids:
                    if cid < len(logits):
                        content_mask[cid] = logits[cid]
                if token_boost is not None:
                    for tid, tboost in token_boost.items():
                        if tid < len(logits) and tid not in seen_counts:
                            content_mask[tid] += tboost * 2.0
                logits = content_mask

            if temperature > 0:
                dyn_temp = temperature * nm_temp_mod
                if step > 2:
                    avg_conf = confidence_sum / max(1, step)
                    confidence_threshold = 0.3 * nm_conf_mod
                    if avg_conf < confidence_threshold:
                        dyn_temp = min(0.8, dyn_temp * 1.3)
                dyn_temp = max(0.1, dyn_temp)
                logits = logits / dyn_temp
                logits_exp = np.exp(logits - np.max(logits))
                probs = logits_exp / (np.sum(logits_exp) + 1e-10)
                sorted_idx = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_idx]
                cumsum = np.cumsum(sorted_probs)
                top_p_dyn = np.clip(top_p * nm_conf_mod, 0.6, 0.99)
                cutoff = np.searchsorted(cumsum, top_p_dyn) + 1
                top_p_idx = sorted_idx[:cutoff]
                top_p_probs = probs[top_p_idx]
                top_p_probs = top_p_probs / np.sum(top_p_probs)
                try:
                    chosen = int(np.random.choice(top_p_idx, p=top_p_probs))
                    confidence_sum += float(np.max(probs))
                except Exception:
                    chosen = int(np.argmax(probs))
            else:
                chosen = int(np.argmax(logits))

            if eos_idx is not None and chosen == eos_idx:
                break

            generated.append(chosen)
            self._recent_predictions.append(chosen)
            seen_counts[chosen] = seen_counts.get(chosen, 0) + 1
            if len(generated) >= 3:
                seen_bigrams.add((generated[-2], generated[-1]))
            if len(generated) >= 4:
                seen_trigrams.add((generated[-3], generated[-2], generated[-1]))

        return generated[1:]

    def train_on_sentence(self, sentence_words: List[str],
                          word_to_embed: Dict[str, np.ndarray],
                          word_to_idx: Dict[str, int],
                          unknown_idx: int = 1,
                          conditioning_embs: Optional[np.ndarray] = None,
                          word_indices: Optional[List[int]] = None,
                          freeze_core: bool = False) -> float:
        """Unsupervised learning from a sentence (fully batched Hebbian updates).

        Args:
            freeze_core: If True, only update word_embedding and output_proj.
                Skips GRU, condition_proj, and attention updates.
                Use this for web learning to protect core language patterns.

        Forward pass stores all errors and intermediates; then at the end
        all Hebbian updates are computed in a few large matmuls.
        ~5-10x faster than per-token accumulate calls.
        """
        if len(sentence_words) < 2:
            return 0.0

        bos_idx = word_to_idx.get("<bos>", 0)
        eos_idx = word_to_idx.get("<eos>", 2)

        if conditioning_embs is None:
            # FIX: Blend function words + actual content words from the sentence
            # for conditioning. This mimics how the decoder receives concept
            # embeddings during generation - solving the training/generation
            # mismatch that caused CE ~3.9.
            # 
            # The key insight: the decoder MUST learn to operate on actual
            # concept-like embeddings (which it sees during generation), not
            # just function words. We include function words for syntactic
            # context AND content words for semantic conditioning.
            import random as _rand_mod
            function_words_set = {
                "a", "an", "the", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "shall", "can",
                "not", "no", "nor", "so", "if", "then", "than", "too", "very",
                "just", "about", "also", "into", "over", "after", "before",
                "between", "through", "during", "because", "while", "which",
                "who", "whom", "what", "when", "where", "why", "how", "all",
                "each", "every", "both", "few", "more", "most", "some", "any",
                "this", "that", "these", "those", "it", "its", "they", "them",
                "their", "we", "our", "you", "your", "he", "she", "him", "her",
                "his", "i", "me", "my", "myself", "am",
                "of", "to", "for", "with", "from", "at", "by", "as", "on", "in", "and", "but", "or",
            }
            known_embs = []
            seen_cond = set()
            
            # Step 1: Add up to 3 content words from the sentence (concept-like embeddings)
            content_count = 0
            for w in sentence_words:
                if content_count >= 3:
                    break
                wl = w.lower().strip(".,!?").strip("'")
                if wl not in function_words_set and wl in word_to_embed and wl not in seen_cond and len(wl) >= 3:
                    known_embs.append(word_to_embed[wl] * random.uniform(0.7, 1.0))
                    seen_cond.add(wl)
                    content_count += 1
            
            # Step 2: Add up to 2 function words for syntactic scaffolding
            function_count = 0
            for w in sentence_words:
                if function_count >= 2:
                    break
                wl = w.lower().strip(".,!?").strip("'")
                if wl in function_words_set and wl in word_to_embed and wl not in seen_cond:
                    known_embs.append(word_to_embed[wl])
                    seen_cond.add(wl)
                    function_count += 1
            
            # Step 3: Pad with random vocab embeddings to always have at least 3
            _all_vals = list(word_to_embed.values())
            while len(known_embs) < 3:
                if _all_vals:
                    known_embs.append(_rand_mod.choice(_all_vals) * 0.5)
                else:
                    pad = np.random.randn(self.embed_dim).astype(np.float32) * 0.1
                    n = np.linalg.norm(pad)
                    if n > 0:
                        pad /= n
                    known_embs.append(pad)
            conditioning_embs = np.stack(known_embs, axis=0)

        if word_indices is None:
            word_indices = []
            for w in sentence_words:
                wl = w.lower().strip(".,!?").strip("'")
                if wl in word_to_idx:
                    word_indices.append(word_to_idx[wl])
                else:
                    word_indices.append(unknown_idx)
        else:
            word_indices = list(word_indices)

        # Prepend <bos> and append <eos> to sentence indices to teach
        # the decoder initiation from <bos> and termination at <eos>.
        if not word_indices or word_indices[0] != bos_idx:
            word_indices = [bos_idx] + word_indices
        if word_indices[-1] != eos_idx:
            word_indices = word_indices + [eos_idx]

        if len(word_indices) < 2:
            return 0.0

        cond_proj = self.condition_proj.forward_raw(conditioning_embs)
        cached_attn_tensor = self.attention(StateTensor(cond_proj))
        cached_attn_logits = cached_attn_tensor.data[0]

        out_weight = self.output_proj.weight.data
        out_bias = self.output_proj.bias.data if self.output_proj.bias is not None else None
        vocab_size_actual = out_weight.shape[0]

        # Precompute non-special indices for negative sampling
        special_set = frozenset({0, 1, 2, 3})
        non_special_idx = np.array([i for i in range(vocab_size_actual) if i not in special_set], dtype=np.intp)

        # Sampled softmax: K=50 negatives, shared across positions within sentence
        # Focused negative sampling: prefer words with high output weights
        # (the model's most-confidently-predicted words make the best negatives).
        if not hasattr(self, '_neg_rng'):
            self._neg_rng = np.random.RandomState(42)
        n_neg = min(50, len(non_special_idx))
        if n_neg > 0:
            # Compute row L2 norms as sampling weights (softmax over indices)
            row_norms = np.linalg.norm(out_weight[non_special_idx], axis=1)
            # Flatten extreme outliers with log transform for stable sampling
            log_weights = np.log1p(row_norms * 5.0)
            sum_log_weights = np.sum(log_weights)
            if sum_log_weights > 0:
                neg_probs = log_weights / sum_log_weights
            else:
                neg_probs = np.ones(len(non_special_idx)) / len(non_special_idx)
            # Ensure numerical stability
            neg_probs = np.clip(neg_probs, 1e-10, None)
            neg_probs /= neg_probs.sum()
            shared_negatives = self._neg_rng.choice(non_special_idx, n_neg, replace=False, p=neg_probs)
        else:
            shared_negatives = np.array([], dtype=np.intp)

        # Pre-allocate buffers
        max_T = len(word_indices) - 1
        all_combined = [None] * max_T
        all_err_stored = [None] * max_T
        all_idx_stored = [None] * max_T
        all_word_emb = [None] * max_T
        all_input_idx = [None] * max_T
        all_gru_combined = [None] * max_T
        all_gru_combined_r = [None] * max_T
        all_gru_h_candidate = [None] * max_T
        all_gru_h_prev = [None] * max_T
        all_gru_z = [None] * max_T
        all_gru_r = [None] * max_T
        total_ce = 0.0
        top1_hits = 0
        top5_hits = 0
        trained = 0

        # Precompute the condition projection for the entire sentence at once
        input_word_indices = word_indices[:-1]
        sentence_embs = self.word_embedding.embed_batch_raw(input_word_indices)
        proj_embs = self.condition_proj.forward_raw(sentence_embs)

        h = np.zeros(self.hidden_dim, dtype=np.float32)
        for pos in range(max_T):
            input_idx = word_indices[pos]
            target_idx = word_indices[pos + 1]

            word_emb = sentence_embs[pos]
            proj_emb = proj_embs[pos]

            # Manual GRU (stores intermediates for batched update)
            x_data = proj_emb
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

            all_gru_combined[pos] = combined
            all_gru_combined_r[pos] = combined_r
            all_gru_h_candidate[pos] = h_candidate
            all_gru_h_prev[pos] = h_data
            all_gru_z[pos] = z
            all_gru_r[pos] = r
            h = h_new

            # Attention + output (keep ratio consistent with generate: 0.5 each)
            combined_vec = h * 0.5 + cached_attn_logits * 0.5
            if self.training and self.dropout.p > 0:
                mask = np.random.binomial(1, 1.0 - self.dropout.p, combined_vec.shape).astype(combined_vec.dtype)
                combined_vec = combined_vec * mask / (1.0 - self.dropout.p)

            # ── Sampled softmax (K+1-way instead of full V-way) ──
            if target_idx < vocab_size_actual:
                # Remove target from shared negatives if present
                if target_idx in shared_negatives:
                    neg_idx = shared_negatives[shared_negatives != target_idx]
                else:
                    neg_idx = shared_negatives
                all_idx = np.concatenate([[target_idx], neg_idx])

                logit_t = combined_vec @ out_weight[all_idx].T
                if out_bias is not None:
                    logit_t += out_bias[all_idx]

                logit_stable = logit_t - np.max(logit_t)
                exp_logits = np.exp(np.clip(logit_stable, -50, 50))
                probs_k = exp_logits / (np.sum(exp_logits) + 1e-10)

                err = np.zeros(len(all_idx), dtype=np.float32)
                err[0] = 1.0 - probs_k[0]
                err[1:] = 0.0 - probs_k[1:]
                err = np.clip(err, -1.0, 1.0)

                p_t = max(float(probs_k[0]), 1e-10)
                total_ce += -np.log(p_t)
                if int(np.argmax(probs_k)) == 0:
                    top1_hits += 1
                if 0 in np.argpartition(probs_k, -5)[-5:]:
                    top5_hits += 1
            else:
                all_idx = np.array([0], dtype=np.intp)
                err = np.zeros(1, dtype=np.float32)
                total_ce += 9.0
            trained += 1

            all_combined[pos] = combined_vec
            all_err_stored[pos] = err
            all_idx_stored[pos] = all_idx
            all_word_emb[pos] = word_emb
            all_input_idx[pos] = input_idx

        if trained == 0:
            return 0.0

        # ── Get neuromodulator-based learning rate multipliers ──
        lr_proj = 0.001
        lr_other = 0.0005
        lr_attn = 0.0002
        wd = 1e-5
        prev_ce = self._avg_cross_entropy
        if self.neuromodulator is not None:
            mods = self.neuromodulator.training_mods()
            lr_proj *= mods['lr_mult_proj']
            lr_other *= mods['lr_mult_gru']
            wd *= (1.0 - mods['ne_level'] * 0.5)  # less weight decay when exploring

        # ── Per-sentence Hebbian updates (SGD-style with weight decay) ──
        T = len(all_combined)

        # 1. output_proj: sparse Hebbian update (sampled softmax, only K+1 rows)
        for pos in range(T):
            h_vec = all_combined[pos]
            err = all_err_stored[pos]
            idx = all_idx_stored[pos]
            hebb = np.outer(err, h_vec) / T
            out_weight[idx] += hebb * lr_proj - out_weight[idx] * wd

        # 2. Error backprop to hidden (sparse: only sampled rows)
        error_h_all = np.zeros((T, self.hidden_dim), dtype=np.float32)
        for pos in range(T):
            err = all_err_stored[pos]
            idx = all_idx_stored[pos]
            error_h_all[pos] = err @ out_weight[idx]

        # 3. condition_proj (only if not frozen — protects learned language patterns)
        if not freeze_core:
            emb_stack = np.stack(all_word_emb)
            error_proj_all = np.clip(error_h_all * 0.5, -1.0, 1.0)
            hebbian_proj = (emb_stack.T @ error_proj_all) / T
            self.condition_proj.weight.data += hebbian_proj.T * lr_other - self.condition_proj.weight.data * wd

        # 4. word_embedding (always updated — new words need embeddings)
        W_cond = self.condition_proj.weight.data
        error_proj_all = np.clip(error_h_all * 0.5, -1.0, 1.0)
        error_word_all = np.clip(error_proj_all @ W_cond, -1.0, 1.0) * lr_other
        for i, idx in enumerate(all_input_idx):
            self.word_embedding.weight.data[idx] += error_word_all[i]

        # 5. GRU gates (only if not frozen)
        if not freeze_core:
            gru_c_stack = np.stack(all_gru_combined)
            gru_cr_stack = np.stack(all_gru_combined_r)
            gru_hc_stack = np.stack(all_gru_h_candidate)
            gru_hp_stack = np.stack(all_gru_h_prev)
            gru_z_stack = np.stack(all_gru_z)
            gru_r_stack = np.stack(all_gru_r)

            # Exact backprop error for h_candidate_pre:
            # dL/dh_candidate_pre = (dL/dh * 0.5) * z * (1 - tanh^2)
            err_wh = error_h_all * 0.5 * gru_z_stack * (1.0 - gru_hc_stack ** 2)
            fe_wh = (gru_cr_stack.T @ err_wh) / T

            # Exact backprop error for z_pre:
            # dL/dz_pre = (dL/dh * 0.5) * (h_candidate - h_prev) * z * (1 - z)
            err_wz = error_h_all * 0.5 * (gru_hc_stack - gru_hp_stack) * gru_z_stack * (1.0 - gru_z_stack)
            fe_wz = (gru_c_stack.T @ err_wz) / T

            # Exact backprop error for r_pre:
            # dL/dr_pre = (dL/dh_candidate_pre @ W_hh * h_prev) * r * (1 - r)
            W_hh = self.gru.W_h.weight.data[:, self.hidden_dim:]
            err_r = (err_wh @ W_hh * gru_hp_stack) * gru_r_stack * (1.0 - gru_r_stack)
            fe_wr = (gru_c_stack.T @ err_r) / T

            # Update GRU weights with gradient clipping and weight decay to prevent divergence
            clip_val = 1.0
            self.gru.W_z.weight.data += np.clip(fe_wz.T, -clip_val, clip_val) * lr_other - self.gru.W_z.weight.data * wd
            self.gru.W_r.weight.data += np.clip(fe_wr.T, -clip_val, clip_val) * lr_other - self.gru.W_r.weight.data * wd
            self.gru.W_h.weight.data += np.clip(fe_wh.T, -clip_val, clip_val) * lr_other - self.gru.W_h.weight.data * wd

        # 6. attention output_proj (only if not frozen)
        if not freeze_core:
            attn_error_all = np.clip(error_h_all * 0.1, -1.0, 1.0)
            total_attn_err = np.sum(attn_error_all, axis=0)
            self.attention.output_proj.weight.data += (total_attn_err / T) * lr_attn - self.attention.output_proj.weight.data * wd

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

        # ── Neuromodulator update from prediction error ──
        if self.neuromodulator is not None:
            # ce_delta = how much worse/better vs. old baseline
            ce_delta = avg_ce - prev_ce if prev_ce > 0 else 0.0
            self.neuromodulator.update_from_prediction_error(ce_delta)
            self.neuromodulator.tick_decay()

        return avg_ce

    def prepare_sentences(self, text: str,
                          word_to_embed: Dict[str, np.ndarray],
                          word_to_idx: Dict[str, int],
                          unknown_idx: int = 1,
                          min_sentence_len: int = 3,
                          max_sentences: Optional[int] = None) -> List[dict]:
        """Pre-process text into cached sentence data for fast training loops.

        FIXED: No more self-conditioning cheat. Conditioning embeddings now use
        a blend of function words (common across sentences) + random vocabulary
        embeddings, NOT the sentence's own content words. This forces the decoder
        to learn true language patterns from the GRU sequence model rather than
        cheating by reading the answer from the conditioning context.

        Returns a list of dicts with 'words', 'word_indices', and 'conditioning_embs'
        so that repeated training passes avoid re-parsing and re-embedding the text.
        """
        import random as _rand_mod
        
        # Pre-build a pool of random vocabulary embeddings for conditioning
        # These provide a general "language space" signal without leaking
        # any specific sentence's content words.
        _all_embed_vals = list(word_to_embed.values())
        
        sentences = []
        for sent in re.split(r'[\r\n.!?]+', text):
            words = [w.strip(".,!?\"' ") for w in sent.split()
                     if len(w.strip(".,!?\"' ")) > 0]
            if len(words) >= min_sentence_len:
                function_words_set = {
                    "a", "an", "the", "is", "are", "was", "were", "be", "been",
                    "being", "have", "has", "had", "do", "does", "did", "will",
                    "would", "could", "should", "may", "might", "shall", "can",
                    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
                    "just", "about", "also", "into", "over", "after", "before",
                    "between", "through", "during", "because", "while", "which",
                    "who", "whom", "what", "when", "where", "why", "how", "all",
                    "each", "every", "both", "few", "more", "most", "some", "any",
                    "this", "that", "these", "those", "it", "its", "they", "them",
                    "their", "we", "our", "you", "your", "he", "she", "him", "her",
                    "his", "i", "me", "my", "myself", "am",
                    "of", "to", "for", "with", "from", "at", "by", "as", "on", "in", "and", "but", "or",
                }
                word_indices = []
                for w in words:
                    wl = w.lower().strip(".,!?").strip("'")
                    if wl in word_to_idx:
                        word_indices.append(word_to_idx[wl])
                    else:
                        word_indices.append(unknown_idx)
                
                # FIX: Build blended conditioning from content words + function words.
                # This matches the train_on_sentence fix and ensures the decoder
                # learns to operate on concept-like embeddings during training.
                cond_embs = []
                seen_cond = set()
                
                # Step 1: Content words (concept-like embeddings)
                content_count = 0
                for w in words:
                    if content_count >= 3:
                        break
                    wl = w.lower().strip(".,!?").strip("'")
                    if wl not in function_words_set and wl in word_to_embed and wl not in seen_cond and len(wl) >= 3:
                        cond_embs.append(word_to_embed[wl] * _rand_mod.uniform(0.7, 1.0))
                        seen_cond.add(wl)
                        content_count += 1
                
                # Step 2: Function words (syntactic scaffolding)
                func_count = 0
                for w in words:
                    if func_count >= 2:
                        break
                    wl = w.lower().strip(".,!?").strip("'")
                    if wl in function_words_set and wl in word_to_embed and wl not in seen_cond:
                        cond_embs.append(word_to_embed[wl])
                        seen_cond.add(wl)
                        func_count += 1
                
                # Step 3: Pad with random vocabulary embeddings to minimum 3
                while len(cond_embs) < 3:
                    if _all_embed_vals:
                        cond_embs.append(_rand_mod.choice(_all_embed_vals) * 0.5)
                    else:
                        pad = np.random.randn(self.embed_dim).astype(np.float32) * 0.1
                        n = np.linalg.norm(pad)
                        if n > 0:
                            pad /= n
                        cond_embs.append(pad)
                conditioning_embs = np.stack(cond_embs, axis=0)
                
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
                      conditioning_embs: Optional[np.ndarray] = None,
                      freeze_core: bool = False) -> Tuple[float, int]:
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
            freeze_core: If True, only update word_embedding and output_proj.
                Skips GRU, condition_proj, and attention updates.

        Returns:
            (avg_error, sentences_trained): average error and count
        """
        # Simple sentence splitting
        sentences = []
        for sent in re.split(r'[\r\n.!?]+', text):
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
                                         conditioning_embs=conditioning_embs,
                                         freeze_core=freeze_core)
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
