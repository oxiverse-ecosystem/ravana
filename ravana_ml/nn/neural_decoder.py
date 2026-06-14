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
from .module import Module, Linear, Embedding, GRUCell, LayerNorm, Dropout, ConceptAttentionHead
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
                 contrastive_weight: float = 0.3,
                 contrastive_negatives: int = 5):
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

        # Layer norm for stability
        self.norm = LayerNorm(hidden_dim)

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

    def rebuild_vocab_cache(self):
        """Rebuild internal vocabulary embedding cache for fast similarity lookup."""
        self._vocab_embed_cache = self.word_embedding._w_raw.copy()

    def forward(self, conditioning_embs: np.ndarray,
                input_seq: np.ndarray,
                h_prev: Optional[np.ndarray] = None,
                use_raw: bool = True,
                _use_gru_traced: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass for training (next-word prediction).

        Args:
            conditioning_embs: (n_concepts, embed_dim) — graph walk concept embeddings
            input_seq: (seq_len,) — word indices (input sequence, shifted by 1)
            h_prev: Optional (hidden_dim,) — initial hidden state
            use_raw: if True, return raw numpy arrays (no StateTensor wrapping)
            _use_gru_traced: if True, use GRU.forward_traced to store traces for Hebbian learning

        Returns:
            logits: (seq_len, vocab_size) — prediction logits for each position
            h_final: (hidden_dim,) — final hidden state
        """
        seq_len = len(input_seq)
        if h_prev is None:
            h = np.zeros(self.hidden_dim, dtype=np.float32)
        else:
            h = h_prev.copy()

        # Use forward() for training to store traces for Hebbian learning.
        # During training, we wrap inputs in StateTensor so Linear stores _trace_x.
        cond_proj_tensor = self.condition_proj(StateTensor(conditioning_embs))
        cond_proj = cond_proj_tensor.data

        logits_list = []
        for t in range(seq_len):
            word_idx = int(input_seq[t])
            word_emb = self.word_embedding.embed_raw(word_idx)

            # Use forward for trace storage
            proj_tensor = self.condition_proj(StateTensor(word_emb[np.newaxis, :]))
            proj_emb = proj_tensor.data[0]

            if _use_gru_traced:
                h = self.gru.forward_traced(proj_emb, h)
            else:
                h = self.gru(proj_emb, h)

            attn_logits = self.attention.forward_raw(cond_proj)
            combined = h + attn_logits * 0.1
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
                 basal_ganglia=None) -> List[int]:
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
            combined = h + attn_logits * 0.1
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

            # Adaptive repetition penalty: scale based on how many times seen
            if len(generated) > 1:
                seen_counts = {}
                for idx in generated:
                    seen_counts[idx] = seen_counts.get(idx, 0) + 1
                for idx, count in seen_counts.items():
                    if count > 1:
                        penalty = 1.0 + 0.5 * (count - 1)
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
                          conditioning_embs: Optional[np.ndarray] = None) -> float:
        """Unsupervised online learning from a sentence.

        Processes one word at a time (not batched): for each word, predict
        the next word, compute predictive coding error, and update all layers
        via local Hebbian signals. Hidden state carries across timesteps.

        This mimics how babies learn language: hear a sequence, predict what
        comes next, update when prediction is wrong.

        Unlike the batched forward pass (which overwrites _trace_x), this online
        approach ensures each timestep's error is paired with the correct input
        trace for Hebbian learning.

        Args:
            sentence_words: list of words from the sentence
            word_to_embed: {word: embedding_vector} — embedding lookup
            word_to_idx: {word: index} — vocab index lookup
            unknown_idx: index for out-of-vocabulary words (default 1)
            conditioning_embs: optional (n_concepts, embed_dim) graph walk embeddings.
                If None, uses sentence's own word embeddings as pseudo-context.

        Returns:
            average prediction error for this sentence
        """
        if len(sentence_words) < 2:
            return 0.0

        if conditioning_embs is None:
            # Build conditioning from the sentence itself (unsupervised: context = sentence)
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

        # Build word index sequence
        word_indices = []
        for w in sentence_words:
            wl = w.lower().strip(".,!?").strip("'")
            if wl in word_to_idx:
                word_indices.append(word_to_idx[wl])
            else:
                word_indices.append(unknown_idx)

        if len(word_indices) < 2:
            return 0.0

        # Online learning: one timestep at a time (carries h state)
        h = np.zeros(self.hidden_dim, dtype=np.float32)
        total_error = 0.0
        trained = 0

        for pos in range(len(word_indices) - 1):
            input_idx = word_indices[pos]
            target_idx = word_indices[pos + 1]

            # Single-step forward pass
            input_arr = np.array([input_idx], dtype=np.int64)
            logits_t, h = self.forward(conditioning_embs, input_arr, h_prev=h,
                                        use_raw=True, _use_gru_traced=True)

            logit_t = logits_t[0]

            # Compute softmax and predictive coding error
            logit_stable = logit_t - np.max(logit_t)
            exp_logits = np.exp(np.clip(logit_stable, -50, 50))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)

            out_weight = self.output_proj.weight.data
            vocab_size_actual = out_weight.shape[0]
            target_one_hot = np.zeros(vocab_size_actual, dtype=np.float32)
            if target_idx < vocab_size_actual:
                target_one_hot[target_idx] = 1.0

            error_output = (target_one_hot - probs).astype(np.float32)
            error_output = np.clip(error_output, -1.0, 1.0)

            # ─── Contrastive Next-Token Loss ───
            # Sample negative tokens and add contrastive gradient to error_output.
            # This provides stronger signal than pure predictive coding (Hebbian only),
            # preventing repetitive/looped output by explicitly suppressing common negatives.
            if self.contrastive_weight > 0 and self.contrastive_negatives > 0:
                # Top-k sampling for negatives (avoidunks, target, special tokens)
                special = {0, 1, 2, 3}  # pad, unk, bos, eos
                # Exclude target and special tokens from negative candidates
                candidate_mask = np.ones(vocab_size_actual, dtype=bool)
                candidate_mask[list(special)] = False
                if target_idx < vocab_size_actual:
                    candidate_mask[target_idx] = False
                candidate_indices = np.where(candidate_mask)[0]
                if len(candidate_indices) > 0:
                    k = min(self.contrastive_negatives, len(candidate_indices))
                    # Prefer negatives with higher current probability (hard negatives)
                    prob_neg = probs[candidate_indices]
                    neg_top = np.argsort(prob_neg)[-k:] if k < len(prob_neg) else np.arange(len(prob_neg))
                    neg_indices = candidate_indices[neg_top]
                    
                    # Contrastive gradients: push positive up, negatives down
                    # Using InfoNCE-style gradients: sigmoid(pos) up, sigmoid(neg) down
                    pos_margin = 0.9 - probs[target_idx] if target_idx < vocab_size_actual else 0.5
                    # Positive contrastive push
                    if target_idx < vocab_size_actual:
                        error_output[target_idx] += self.contrastive_weight * pos_margin
                    # Negative contrastive push (suppress probable negatives)
                    for neg_idx in neg_indices:
                        neg_force = -self.contrastive_weight * probs[neg_idx] * 0.5
                        error_output[neg_idx] += neg_force

            error_output = np.clip(error_output, -1.0, 1.0)

            # Output projection Hebbian update (trace_x was set during forward)
            self.output_proj.accumulate_free_energy(StateTensor(error_output, salience=0.15))

            # Propagate error to hidden state via output_proj weights
            error_h = error_output @ out_weight  # (hidden_dim,)

            # GRU Hebbian updates using stored intermediates from forward_traced
            h_candidate = getattr(self.gru, '_last_h_candidate', None)
            h_prev = getattr(self.gru, '_last_h_prev', None)
            z = getattr(self.gru, '_last_z', None)
            if h_candidate is not None and h_prev is not None and z is not None:
                error_z = error_h * (h_candidate - h_prev) * 0.5
                self.gru.W_z.accumulate_free_energy(StateTensor(error_z, salience=0.1))
                error_r = error_h * 0.3
                self.gru.W_r.accumulate_free_energy(StateTensor(error_r, salience=0.1))
                error_h_val = error_h * z
                self.gru.W_h.accumulate_free_energy(StateTensor(error_h_val, salience=0.1))

            total_error += float(np.mean(np.abs(error_output)))
            trained += 1

        avg_error = total_error / max(1, trained)
        self._total_training_examples += trained
        self._avg_prediction_error = self._avg_prediction_error * 0.95 + avg_error * 0.05

        return avg_error

    def train_on_text(self, text: str,
                      word_to_embed: Dict[str, np.ndarray],
                      word_to_idx: Dict[str, int],
                      unknown_idx: int = 1,
                      min_sentence_len: int = 3,
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

        total_error = 0.0
        trained_count = 0
        for words in sentences[:50]:  # cap per text to avoid OOM
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
        self.rebuild_vocab_cache()


# Add re import at module level
import re
