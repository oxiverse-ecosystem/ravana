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
from typing import Dict, List, Optional, Tuple, Set
from ..tensor import StateTensor, Parameter, RawTensor
from .module import Module, Linear, Embedding, GRUCell, LayerNorm, Dropout, ConceptAttentionHead


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
                 hidden_dim: int = 128, n_attention_heads: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Word embedding (output vocabulary) — maps word indices to embeddings
        self.word_embedding = Embedding(vocab_size, embed_dim)

        # Conditioning projection — project conditioning concepts into GRU space
        self.condition_proj = Linear(embed_dim, hidden_dim)

        # GRU for sequence processing (temporal cortex)
        self.gru = GRUCell(hidden_dim, hidden_dim)

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
                use_raw: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass for training (next-word prediction).

        Args:
            conditioning_embs: (n_concepts, embed_dim) — graph walk concept embeddings
            input_seq: (seq_len,) — word indices (input sequence, shifted by 1)
            h_prev: Optional (hidden_dim,) — initial hidden state
            use_raw: if True, return raw numpy arrays (no StateTensor wrapping)

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
                 temperature: float = 0.5) -> List[int]:
        """Generate a sequence autoregressively conditioned on concept embeddings.

        Uses the BasalGangliaGate-inspired selection: during generation,
        logits are filtered through a competitive selection process.

        Args:
            conditioning_embs: (n_concepts, embed_dim) — graph walk concept embeddings
            max_steps: maximum number of tokens to generate
            bos_idx: beginning-of-sequence token index
            eos_idx: end-of-sequence token index (None = no early stopping)
            temperature: softmax temperature (lower = more deterministic)

        Returns:
            List of generated token indices
        """
        if self._vocab_embed_cache is None:
            self.rebuild_vocab_cache()

        h = np.zeros(self.hidden_dim, dtype=np.float32)
        cond_proj = self.condition_proj.forward_raw(conditioning_embs)

        generated = [bos_idx]
        for step in range(max_steps):
            word_emb = self.word_embedding.embed_raw(generated[-1])
            projected_word = self.condition_proj.forward_raw(word_emb[np.newaxis, :])[0]
            h = self.gru(projected_word, h)

            attn_logits = self.attention.forward_raw(cond_proj)
            combined = h + attn_logits * 0.1
            logits = self.output_proj.forward_raw(combined[np.newaxis, :])[0]

            # Apply repetition penalty: scale down logits for already-generated tokens
            if len(generated) > 1:
                for i, prev_idx in enumerate(generated):
                    penalty = 1.5 + 0.5 * i / len(generated)
                    logits[prev_idx] -= penalty

            # Apply temperature and sample
            if temperature > 0:
                logits = logits / temperature
                # Softmax
                logits_exp = np.exp(logits - np.max(logits))
                probs = logits_exp / (np.sum(logits_exp) + 1e-10)
                # Top-k filtering (keep top 30)
                k = min(30, len(probs))
                top_k_idx = np.argpartition(probs, -k)[-k:]
                top_k_probs = probs[top_k_idx]
                top_k_probs = top_k_probs / np.sum(top_k_probs)
                try:
                    chosen = np.random.choice(top_k_idx, p=top_k_probs)
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
                          unknown_idx: int = 1) -> float:
        """Unsupervised learning from a sentence.

        For each word in the sentence, predict the next word given previous words.
        Learning signal: predictive coding error between predicted next-word
        embedding and actual next-word embedding.

        This mimics how babies learn language: hear a sequence, predict what comes next,
        update when prediction is wrong.

        Args:
            sentence_words: list of words from the sentence
            word_to_embed: {word: embedding_vector} — embedding lookup
            word_to_idx: {word: index} — vocab index lookup
            unknown_idx: index for out-of-vocabulary words (default 1)

        Returns:
            average prediction error for this sentence
        """
        if len(sentence_words) < 2:
            return 0.0

        # Build conditioning from the sentence itself (unsupervised: context = sentence)
        # In a full system, conditioning comes from graph walk. During unsupervised
        # pre-training, we use the sentence's own word embeddings as pseudo-context.
        known_embs = []
        for w in sentence_words:
            wl = w.lower().strip(".,!?").strip("'")
            if wl in word_to_embed:
                known_embs.append(word_to_embed[wl])
            elif wl:
                # Use random embedding for unknown words (will be learned)
                rand_emb = np.random.randn(self.embed_dim).astype(np.float32) * 0.1
                norm = np.linalg.norm(rand_emb)
                if norm > 0:
                    rand_emb /= norm
                known_embs.append(rand_emb)

        if len(known_embs) < 2:
            # Need at least 2 embeddings for meaningful sequence learning
            known_embs_flat = [np.random.randn(self.embed_dim).astype(np.float32) * 0.1
                               for _ in range(max(2, len(sentence_words)))]
            for i in range(len(known_embs_flat)):
                n = np.linalg.norm(known_embs_flat[i])
                if n > 0:
                    known_embs_flat[i] /= n
            conditioning_embs = np.stack(known_embs_flat, axis=0)
        else:
            conditioning_embs = np.stack(known_embs, axis=0)

        # Build input word index sequence (predict word[t+1] given word[t])
        word_indices = []
        for w in sentence_words:
            wl = w.lower().strip(".,!?").strip("'")
            if wl in word_to_idx:
                word_indices.append(word_to_idx[wl])
            else:
                word_indices.append(unknown_idx)

        if len(word_indices) < 2:
            return 0.0

        # Input: all words except last. Target: all words except first.
        input_seq = np.array(word_indices[:-1], dtype=np.int64)
        target_indices = np.array(word_indices[1:], dtype=np.int64)

        # Forward pass
        logits, _ = self.forward(conditioning_embs, input_seq)

        # Compute prediction error (predictive coding: softmax cross-entropy)
        # But we DON'T use backprop. Instead, we compute the error at the output
        # and use it as a local Hebbian learning signal.
        errors = []
        for t in range(len(target_indices)):
            target = int(target_indices[t])
            logit_t = logits[t]

            # Compute softmax
            logit_stable = logit_t - np.max(logit_t)
            exp_logits = np.exp(np.clip(logit_stable, -50, 50))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)

            # One-hot target
            target_one_hot = np.zeros(self.vocab_size, dtype=np.float32)
            if target < self.vocab_size:
                target_one_hot[target] = 1.0

            # Gradient-compatible error: one_hot - softmax (negative Hebbian)
            error = (target_one_hot - probs).astype(np.float32)
            error = np.clip(error, -1.0, 1.0)
            errors.append(error)

            # Accumulate for weight updates (moderate learning rate)
            self.output_proj.accumulate_free_energy(StateTensor(error, salience=0.15))

        avg_error = float(np.mean([float(np.mean(np.abs(e))) for e in errors]))
        self._total_training_examples += len(errors)
        self._avg_prediction_error = self._avg_prediction_error * 0.95 + avg_error * 0.05

        return avg_error

    def train_on_text(self, text: str,
                      word_to_embed: Dict[str, np.ndarray],
                      word_to_idx: Dict[str, int],
                      unknown_idx: int = 1,
                      min_sentence_len: int = 3) -> Tuple[float, int]:
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
        for words in sentences[:20]:  # cap per text to avoid OOM
            err = self.train_on_sentence(words, word_to_embed, word_to_idx, unknown_idx)
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
        super().load_state_dict(sd)
        if '_total_training_examples' in sd:
            self._total_training_examples = sd['_total_training_examples']
        if '_avg_prediction_error' in sd:
            self._avg_prediction_error = sd['_avg_prediction_error']
        self.rebuild_vocab_cache()


# Add re import at module level
import re
