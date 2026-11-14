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

        # Attention over conditioning context.
        #
        # BUG (docs/CODEBASE_DECODER_AUDIT.md finding 3): this was constructed as
        # ConceptAttentionHead(hidden_dim, hidden_dim), so `vocab_size` was 256 --
        # the head's output_proj was Linear(256, 256) and returned a HIDDEN-SPACE
        # vector, not vocab logits. `_readout` then blended it 50/50 with h and
        # applied output_proj again (a double projection), and the head's own
        # output_proj was weight-decayed to fro 1.16 (vs 9.24 at init) while
        # W_q/W_k/W_v sat at exact random init (fro 22.57/22.64/22.70 vs 22.63
        # expected) -- the attention contributed ~0.6% of readout variance.
        #
        # The head must emit vocab logits so it can be blended with the GRU
        # readout in LOGIT space (a proper mixture of two distributions), not
        # hidden space. vocab_size is not known until the weights are loaded, so
        # the head is rebuilt by `_align_attention_head()` once vocab_size_actual
        # is known.
        self.attention = ConceptAttentionHead(hidden_dim, hidden_dim, n_heads=n_attention_heads)

        # Logit-space mixture gate between the GRU readout and the attention
        # readout. 0.5 = equal mixture (the historical 50/50 blend, now applied
        # in logit space rather than hidden space). Learnable; see the update
        # step in train_on_sentence.
        self.attn_gate = 0.5

        # Output projection: hidden state → logits over vocabulary
        self.output_proj = Linear(hidden_dim, vocab_size)

        # P6: persistent concept-bias projection (angular-gyrus / working-memory
        # hold). The concept semantic pointer (W_sm @ av, a 75-D BOS vector) is
        # projected into GRU space by condition_proj and added to the hidden
        # state at EVERY step, so the pointer modulates production continuously
        # (held in working memory) rather than only seeding step 0. This is the
        # brain-faithful fix for free-form grounded generation: the pointer
        # MODULATES (not replaces) the linguistic scaffolding. Initialized
        # near-zero so it starts as a no-op and preserves the seed-trained
        # coherence until grounded training teaches a useful bias.
        self.W_h_bias = Linear(hidden_dim, hidden_dim)
        # Initialize near-zero so the persistent bias starts as a no-op and
        # preserves seed-trained coherence until grounded training teaches a
        # useful modulation.
        self.W_h_bias.weight.data[:] = 0.0
        if self.W_h_bias.bias is not None:
            self.W_h_bias.bias.data[:] = 0.0

        # P6: auxiliary "load-bearing" head (research item #9, "Cosine Misleads").
        # Forces the concept pointer to actually flow through the network by
        # requiring the hidden state to reconstruct av (65-D Binder attributes).
        # Without this a weak projected vector is silently bypassed by the GRU.
        self.av_head = Linear(hidden_dim, 65)
        self._last_aux_loss = 0.0

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

    # ── Persistence ────────────────────────────────────────────────────────
    # The decoder used to be unpersistable: `scripts/train.py` trained it in
    # memory and the weights vanished on exit, so the 500k-example run had no
    # effect on the running system. These two methods write and restore the
    # FULL state -- parameters, vocabulary, and the config needed to
    # reconstruct the module tree.
    #
    # The vocabulary is saved alongside the weights because the two must agree:
    # a decoder trained on 15k words driven with a 217-word concept vocabulary
    # emits one function word and stops. See docs/DECODER_VOCAB_MISMATCH.md.

    def save_to(self, path) -> bool:
        """Write parameters + vocabulary + config to `path` (npz).

        Returns True on success. Never raises: a failed save must not take down
        a training run that has already done the expensive work.
        """
        try:
            import numpy as np
            sd = self.state_dict()
            flat = {}
            for name, entry in sd.items():
                arr = entry["data"] if isinstance(entry, dict) else entry
                flat[f"w:{name}"] = np.asarray(arr)
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        if k == "data":
                            continue
                        flat[f"m:{name}:{k}"] = np.asarray(v)
            flat["_vocab_size"] = np.asarray(self.vocab_size)
            flat["_embed_dim"] = np.asarray(self.embed_dim)
            flat["_hidden_dim"] = np.asarray(self.hidden_dim)
            flat["_contrastive_weight"] = np.asarray(self.contrastive_weight)
            flat["_contrastive_negatives"] = np.asarray(self.contrastive_negatives)
            np.savez(path, **flat)
            return True
        except Exception:
            return False

    @classmethod
    def load_from(cls, path, neuromodulator=None):
        """Reconstruct a decoder from `save_to` output.

        Returns (decoder, vocab_size) or (None, None). The caller is responsible
        for installing the word<->index maps, which live on the engine rather
        than the decoder.
        """
        try:
            import numpy as np
            z = np.load(path, allow_pickle=False)
            keys = set(z.files)
            if "_vocab_size" not in keys:
                return None, None
            vocab_size = int(np.asarray(z["_vocab_size"]).reshape(-1)[0])
            embed_dim = int(np.asarray(z.get("_embed_dim", 75)).reshape(-1)[0])
            hidden_dim = int(np.asarray(z.get("_hidden_dim", 256)).reshape(-1)[0])
            dec = cls(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                n_attention_heads=4,
                contrastive_weight=float(
                    np.asarray(z.get("_contrastive_weight", 0.5)).reshape(-1)[0]),
                contrastive_negatives=int(
                    np.asarray(z.get("_contrastive_negatives", 8)).reshape(-1)[0]),
                neuromodulator=neuromodulator,
            )
            sd = {}
            # Only real parameters carry the `w:` prefix. The decoder's
            # load_state_dict also reads metric keys (`_avg_cross_entropy`,
            # `_total_training_examples`, ...) and expects them as BARE
            # scalars -- wrapping them in the {"data": ...} parameter shape
            # makes float() raise TypeError. So pass those through untouched.
            for k in keys:
                if not k.startswith("w:"):
                    continue
                name = k[2:]
                if name.startswith("_"):
                    # Metric, not a parameter: bare scalar.
                    sd[name] = float(np.asarray(z[k]).reshape(-1)[0])
                    continue
                entry = {"data": z[k].copy()}
                for mk in keys:
                    pref = f"m:{name}:"
                    if mk.startswith(pref):
                        # np.savez stores scalars as 0-d arrays; float() on the
                        # raw value raises TypeError, so flatten first.
                        entry[mk[len(pref):]] = float(
                            np.asarray(z[mk]).reshape(-1)[0])
                sd[name] = entry
            dec.load_state_dict(sd)
            dec.rebuild_vocab_cache()
            return dec, vocab_size
        except Exception:
            return None, None

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
            self._align_attention_head()
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
            # Logit-space mixture (docs/CODEBASE_DECODER_AUDIT.md finding 3):
            # both terms are vocab logits, so the blend is a convex combination
            # of two distributions rather than a projection of a hidden-space
            # blend.
            gru_logits = self.output_proj.forward_raw(h[np.newaxis, :])[0]
            combined = ((1.0 - self.attn_gate) * gru_logits
                        + self.attn_gate * attn_logits)
            if self.training and self.dropout.p > 0:
                mask = np.random.binomial(1, 1.0 - self.dropout.p, combined.shape).astype(combined.dtype)
                combined = combined * mask / (1.0 - self.dropout.p)

            # `combined` is already vocab logits (logit-space mixture) -- no
            # second projection through output_proj.
            logits_t = combined
            logits_list.append(logits_t)

        return np.stack(logits_list, axis=0), h

    def _align_attention_head(self):
        """Rebuild the attention head so its output_proj emits VOCAB logits.

        The head is constructed as ConceptAttentionHead(hidden_dim, hidden_dim)
        because vocab_size is not known until the checkpoint is loaded. Left
        alone, its output_proj is Linear(hidden_dim, hidden_dim) and the readout
        blends a hidden-space vector with h -- a double projection that carries
        no vocabulary information (docs/CODEBASE_DECODER_AUDIT.md finding 3).

        Rebuilding here (idempotent) makes the head emit vocab logits so it can
        be mixed with the GRU readout in logit space. The existing W_q/W_k/W_v
        weights are preserved; only output_proj is resized, seeded from the
        current (near-zero) matrix where the shapes overlap.
        """
        vocab_size_actual = self.output_proj.weight.data.shape[0]
        if self.attention.vocab_size == vocab_size_actual:
            return
        old = self.attention.output_proj
        new_head = ConceptAttentionHead(
            self.hidden_dim, vocab_size_actual, n_heads=self.attention.n_heads)
        new_head.W_q.weight.data[:] = self.attention.W_q.weight.data
        new_head.W_k.weight.data[:] = self.attention.W_k.weight.data
        new_head.W_v.weight.data[:] = self.attention.W_v.weight.data
        new_head.W_q.bias.data[:] = self.attention.W_q.bias.data
        new_head.W_k.bias.data[:] = self.attention.W_k.bias.data
        new_head.W_v.bias.data[:] = self.attention.W_v.bias.data
        old_w = old.weight.data
        new_w = new_head.output_proj.weight.data
        k = min(old_w.shape[0], new_w.shape[0])
        new_w[:k, :] = old_w[:k, :]
        if new_head.output_proj.bias is not None and old.bias is not None:
            kb = min(old.bias.data.shape[0], new_head.output_proj.bias.data.shape[0])
            new_head.output_proj.bias.data[:kb] = old.bias.data[:kb]
        self.attention = new_head

    def generate(self, conditioning_embs: np.ndarray,
                 max_steps: int = 20,
                 bos_idx: int = 0,
                 eos_idx: Optional[int] = None,
                 temperature: float = 0.5,
                 cerebellar_ngram=None,
                 idx_to_word: Optional[Dict[int, str]] = None,
                 basal_ganglia=None,
                 content_word_ids: Optional[Set[int]] = None,
                 token_boost: Optional[Dict[int, float]] = None,
                 token_suppress: Optional[Dict[int, float]] = None,
                 initial_emb: Optional[np.ndarray] = None,
                 persistent_emb: Optional[np.ndarray] = None) -> List[int]:
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
            initial_emb: optional (embed_dim,) — a dynamically CONDITIONED BOS
                embedding (e.g. register + sensorimotor + VAD). When provided,
                it replaces the static <bos> embedding as the first token's
                input at step 0 (P6: PFC register frame sets the initiation
                state before utterance onset).

        Returns:
            List of generated token indices
        """
        if self._vocab_embed_cache is None:
            self.rebuild_vocab_cache()

        h = np.zeros(self.hidden_dim, dtype=np.float32)
        self._align_attention_head()
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
            if step == 0 and initial_emb is not None:
                # P6: use the conditioned BOS embedding directly as the first
                # token's input (the PFC register frame), bypassing the static
                # <bos> row. The conditioning projection still maps it into GRU
                # space so the rest of the forward pass is unchanged.
                word_emb = np.asarray(initial_emb, dtype=np.float32)[:self.embed_dim]
            projected_word = self.condition_proj.forward_raw(word_emb[np.newaxis, :])[0]
            h = self.gru(projected_word, h)
            # P6: persistent concept bias (working-memory hold). The concept
            # semantic pointer (persistent_emb, same 75-D BOS vector as
            # initial_emb in the LingGen path) is projected into GRU space and
            # added to h at EVERY step, so the pointer modulates production
            # continuously instead of only seeding step 0. This is what makes
            # free-form grounded generation stay on-topic (the pointer is held
            # in working memory), while the linguistic scaffold (attention over
            # conditioning_embs) is left intact.
            if persistent_emb is not None:
                pproj = self.condition_proj.forward_raw(
                    np.asarray(persistent_emb, dtype=np.float32)[:self.embed_dim][np.newaxis, :])[0]
                h = h + self.W_h_bias.forward_raw(pproj[np.newaxis, :])[0]

            gru_logits = self.output_proj.forward_raw(h[np.newaxis, :])[0]
            combined = ((1.0 - self.attn_gate) * gru_logits
                        + self.attn_gate * precomputed_attn)
            logits = combined

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

            # token_boost is a per-step bias toward the committed/retrieved
            # content words. It used to apply only to tokens NOT yet in
            # seen_counts, which meant it shaped the opening of the utterance
            # and then let the model run free -- measured in
            # docs/BOOST_STRENGTH_NOT_THE_LEVER.md ("It only affects the first
            # occurrence"). A boost that expires on first use cannot hold an
            # utterance on subject; the repetition penalty below already
            # discourages re-emitting a token, so applying the boost every step
            # does not produce verbatim repetition (the sweep in
            # docs/ANCHOR_TOKEN_BOOST_MEASURED.md records repetition 0.000 at
            # the 9.0 knee).
            if token_boost is not None:
                for tid, tboost in token_boost.items():
                    if tid < len(logits):
                        logits[tid] += tboost

            # ── NoGo: suppress tokens that are NOT admissible ──────────────
            # The Go pathway (token_boost) raises admissible tokens; this is
            # the NoGo pathway that lowers the rest. Both are needed: boosting
            # alone leaves the decoder free to emit unsupported content words,
            # which is exactly what the CPG measurement showed (unsupported
            # assertions flat at ~1.19 while on-topic rose 4.5x).
            #
            # Suppression is applied to CONTENT words only -- function words
            # carry syntax, and suppressing them would break fluency rather
            # than grounding. That mirrors the brain's division: the formulator
            # needs its syntactic frame; the semantic store constrains the
            # content that fills it.
            if token_suppress is not None and content_word_ids:
                for tid, tsup in token_suppress.items():
                    if tid < len(logits) and tid not in seen_counts \
                            and tid in content_word_ids:
                        logits[tid] -= tsup

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
                        if tid < len(logits):
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

    # ── Scoring (read-only; used by the LingGen promotion harness) ────────────

    def score_sequence(self, word_indices: List[int],
                       conditioning_embs: Optional[np.ndarray] = None,
                       bos_idx: int = 0, eos_idx: int = 2,
                       n_neg: int = 50, seed: int = 1234) -> Optional[float]:
        """Teacher-forced cross-entropy of a token sequence under this decoder.

        A READ of the model: no weight updates, no RNG state touched. The LingGen
        promotion harness uses it to score surface forms produced by EITHER
        generation path, which is what makes the comparison a real
        distribution-fit test instead of two unrelated numbers.

        The negative sample is DETERMINISTIC (seeded) rather than resampled per
        call as in `train_on_sentence`: two replies of different length, or the
        same reply scored twice, must be comparable, and a fresh sample per call
        would put measurement noise inside the quantity being measured.

        Returns mean NLL per token, or None when the sequence cannot be scored.
        """
        if not word_indices:
            return None
        seq = list(word_indices)
        if seq[0] != bos_idx:
            seq = [bos_idx] + seq
        if seq[-1] != eos_idx:
            seq = seq + [eos_idx]
        if len(seq) < 2:
            return None

        if conditioning_embs is None:
            conditioning_embs = np.zeros((1, self.embed_dim), dtype=np.float32)
        # Project FIRST, then attend: the attention head reads hidden_dim space,
        # so an unprojected embed_dim vector is a shape error, not a zero context.
        self._align_attention_head()
        cond_proj = self.condition_proj.forward_raw(
            np.asarray(conditioning_embs, dtype=np.float32))
        cached_attn = self.attention(StateTensor(cond_proj)).data[0]

        out_weight = self.output_proj.weight.data
        out_bias = (self.output_proj.bias.data
                    if self.output_proj.bias is not None else None)
        vocab_size_actual = out_weight.shape[0]

        special_set = frozenset({0, 1, 2, 3})
        non_special_idx = np.array(
            [i for i in range(vocab_size_actual) if i not in special_set],
            dtype=np.intp)
        if len(non_special_idx) == 0:
            return None

        # Focused negative sampling, same idea as training (the model's
        # most-confidently-predicted words make the hardest negatives), but
        # drawn from a fixed seed so scores are comparable.
        rng = np.random.RandomState(seed)
        norms = np.linalg.norm(out_weight[non_special_idx], axis=1) + 1e-8
        neg_probs = norms / norms.sum()
        k = min(n_neg, len(non_special_idx))
        shared_negatives = rng.choice(non_special_idx, k, replace=False,
                                      p=neg_probs)

        embs = self.word_embedding.embed_batch_raw(
            np.array(seq[:-1], dtype=np.intp))

        h = np.zeros(self.hidden_dim, dtype=np.float32)
        total_ce = 0.0
        scored = 0
        for pos in range(len(seq) - 1):
            target_idx = seq[pos + 1]
            proj_emb = self.condition_proj.forward_raw(
                embs[pos][np.newaxis, :])[0]
            h = self.gru(proj_emb, h)
            gru_logits = self.output_proj.forward_raw(h[np.newaxis, :])[0]
            combined = ((1.0 - self.attn_gate) * gru_logits
                        + self.attn_gate * cached_attn)

            if target_idx >= vocab_size_actual:
                total_ce += 9.0          # same penalty training uses for OOV
                scored += 1
                continue
            neg_idx = shared_negatives
            if target_idx in shared_negatives:
                neg_idx = shared_negatives[shared_negatives != target_idx]
            all_idx = np.concatenate([[target_idx], neg_idx])
            # `combined` is already a vocab-logit vector (logit-space mixture),
            # so the sampled logits are a plain gather -- no second projection.
            logit_t = combined[all_idx]
            if out_bias is not None:
                logit_t = logit_t + out_bias[all_idx]
            logit_stable = logit_t - np.max(logit_t)
            exp_logits = np.exp(np.clip(logit_stable, -50, 50))
            probs_k = exp_logits / (np.sum(exp_logits) + 1e-10)
            total_ce += -np.log(max(float(probs_k[0]), 1e-10))
            scored += 1

        if scored == 0:
            return None
        return float(total_ce / scored)

    def train_on_sentence(self, sentence_words: List[str],
                          word_to_embed: Dict[str, np.ndarray],
                          word_to_idx: Dict[str, int],
                          unknown_idx: int = 1,
                          conditioning_embs: Optional[np.ndarray] = None,
                          word_indices: Optional[List[int]] = None,
                          freeze_core: bool = False,
                          sensorimotor_conditioning: Optional[np.ndarray] = None,
                          initial_emb: Optional[np.ndarray] = None,
                          persistent_emb: Optional[np.ndarray] = None,
                          scheduled_eps: float = 0.0,
                          aux_lambda: float = 0.0) -> float:
        """Unsupervised learning from a sentence (fully batched Hebbian updates).

        Args:
            freeze_core: If True, only update word_embedding and output_proj.
                Skips GRU, condition_proj, and attention updates.
                Use this for web learning to protect core language patterns.
            sensorimotor_conditioning: Optional (65-D) Binder vector (LingGen P6).
                When provided AND the learned angular-gyrus projection W_sm is
                attached, it is projected to a 75-D ``initial_emb`` and injected
                as the BOS initiation embedding at step 0 -- matching
                ``generate(initial_emb=)`` EXACTLY, so train == gen. It does NOT
                overwrite the per-token linguistic conditioning (the function/
                content blend), which remains the attention context. This is the
                supervised grounding signal: "given this embodied signature as
                the initiation state, reproduce this human description".
            initial_emb: Optional (embed_dim,) BOS initiation embedding supplied
                directly (skips the W_sm projection). Used by callers that have
                already computed it.

        Forward pass stores all errors and intermediates; then at the end
        all Hebbian updates are computed in a few large matmuls.
        ~5-10x faster than per-token accumulate calls.
        """
        if len(sentence_words) < 2:
            return 0.0

        bos_idx = word_to_idx.get("<bos>", 0)
        eos_idx = word_to_idx.get("<eos>", 2)

        # P6: derive the BOS initiation embedding from the embodied signature,
        # but ONLY for step 0 -- never touch the per-token linguistic blend.
        if initial_emb is None and sensorimotor_conditioning is not None:
            sm = np.asarray(sensorimotor_conditioning, dtype=np.float32)[:65]
            W_sm = getattr(self, "_linggen_W_sm", None)
            if W_sm is not None:
                init = (W_sm @ sm).astype(np.float32)
                n = np.linalg.norm(init)
                if n > 0:
                    init = init / n
                initial_emb = init  # (embed_dim,) BOS initiation state
        _has_initial = initial_emb is not None
        # P6: persistent concept bias (working-memory hold). Derived from the
        # same embodied signature as initial_emb: the 75-D BOS vector projected
        # via W_sm. Added to h at EVERY training step so the concept pointer is
        # held across the sequence (matching generate(persistent_emb=)). Kept
        # separate from initial_emb so the register path (which only seeds step
        # 0) is unaffected.
        if persistent_emb is None and sensorimotor_conditioning is not None:
            sm = np.asarray(sensorimotor_conditioning, dtype=np.float32)[:65]
            W_sm = getattr(self, "_linggen_W_sm", None)
            if W_sm is not None:
                pinit = (W_sm @ sm).astype(np.float32)
                n = np.linalg.norm(pinit)
                if n > 0:
                    pinit = pinit / n
                persistent_emb = pinit
        _has_persistent = persistent_emb is not None

        # P6: auxiliary load-bearing target. When a concept pointer is present and
        # the caller requests an auxiliary reconstruction loss (aux_lambda > 0),
        # reconstruct the 65-D av from the hidden states so the pointer cannot be
        # silently bypassed (research item #9, "Cosine Misleads").
        av_target = None
        if _has_persistent and aux_lambda > 0 and sensorimotor_conditioning is not None:
            av_target = np.asarray(sensorimotor_conditioning, dtype=np.float32)[:65]
        self._last_aux_loss = 0.0
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
                    known_embs.append(word_to_embed[wl] * _rand_mod.uniform(0.7, 1.0))
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

        self._align_attention_head()
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
        if not hasattr(self, '_sched_rng'):
            self._sched_rng = np.random.RandomState(7)
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

            # MIXED NEGATIVE SAMPLING (docs/CODEBASE_DECODER_AUDIT.md finding 4).
            # Drawing negatives purely by output-row norm is a rich-get-richer
            # loop: high-norm rows are disproportionately function words (the
            # frequent, high-confidence predictions), so content words rarely
            # appear as negatives and receive almost no gradient. Half the
            # negatives are therefore drawn UNIFORMLY, which guarantees every
            # word -- including rare content words -- is occasionally
            # contrasted against the target.
            n_uniform = n_neg // 2
            n_focused = n_neg - n_uniform
            uniform_probs = np.ones(len(non_special_idx), dtype=np.float64)
            uniform_probs /= uniform_probs.sum()
            negs_focused = self._neg_rng.choice(
                non_special_idx, n_focused, replace=False, p=neg_probs)
            negs_uniform = self._neg_rng.choice(
                non_special_idx, n_uniform, replace=False, p=uniform_probs)
            shared_negatives = np.concatenate([negs_focused, negs_uniform])
        else:
            shared_negatives = np.array([], dtype=np.intp)

        # Pre-allocate buffers
        max_T = len(word_indices) - 1
        all_combined = [None] * max_T
        all_err_stored = [None] * max_T
        all_gru_logits = [None] * max_T
        all_attn_logits = [None] * max_T
        all_idx_stored = [None] * max_T
        all_word_emb = [None] * max_T
        all_input_idx = [None] * max_T
        all_gru_combined = [None] * max_T
        all_gru_combined_r = [None] * max_T
        all_gru_h_candidate = [None] * max_T
        all_gru_h_prev = [None] * max_T
        all_gru_z = [None] * max_T
        all_gru_r = [None] * max_T
        all_h = [None] * max_T
        # P6: per-position projected concept pointer for the persistent-bias
        # backward pass (only allocated when a concept pointer is present).
        all_pproj = [None] * max_T if _has_persistent else None
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
            # P6: scheduled sampling (exposure-bias cure). With probability eps,
            # feed the model's OWN previous prediction instead of the teacher
            # token, so training matches free-run generation. Aligns train/test
            # distributions so a single early error can't cascade into salad.
            if (scheduled_eps > 0.0 and pos > 0 and all_combined[pos - 1] is not None
                    and self._sched_rng.rand() < scheduled_eps):
                # `all_combined` holds VOCAB LOGITS (logit-space mixture), so
                # the model's own previous prediction is a direct argmax -- no
                # second projection through output_proj.
                logits_prev = all_combined[pos - 1]
                own_pred = int(np.argmax(logits_prev))
                word_emb = self.word_embedding.embed_batch_raw(
                    np.array([own_pred], dtype=np.intp))[0]
            proj_emb = proj_embs[pos]
            if pos == 0 and _has_initial and initial_emb is not None:
                # P6: mirror generate(initial_emb=) -- replace the <bos> input's
                # projected embedding with the conditioned BOS initiation state
                # (W_sm @ av), so the GRU hidden state is seeded by the embodied
                # signature exactly as at generation time. The per-token
                # linguistic conditioning (attention context) is untouched.
                proj_emb = self.condition_proj.forward_raw(
                    np.asarray(initial_emb, dtype=np.float32)[:self.embed_dim][np.newaxis, :])[0]

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

            # P6: persistent concept bias (working-memory hold) -- mirrors
            # generate(persistent_emb=). Add the concept pointer (projected via
            # condition_proj, then W_h_bias) to h at EVERY step so the pointer
            # modulates production continuously, not just at step 0. This is the
            # brain-faithful fix: the angular-gyrus semantic pointer is held in
            # working memory and modulates the linguistic scaffold. W_h_bias is
            # initialized near-zero, so until grounded training teaches a useful
            # modulation this is a no-op and seed coherence is preserved.
            if _has_persistent and persistent_emb is not None:
                pproj = self.condition_proj.forward_raw(
                    np.asarray(persistent_emb, dtype=np.float32)[:self.embed_dim][np.newaxis, :])[0]
                h_new = h_new + self.W_h_bias.forward_raw(pproj[np.newaxis, :])[0]
                all_pproj[pos] = pproj

            all_gru_combined[pos] = combined
            all_gru_combined_r[pos] = combined_r
            all_gru_h_candidate[pos] = h_candidate
            all_gru_h_prev[pos] = h_data
            all_gru_z[pos] = z
            all_gru_r[pos] = r
            h = h_new

            # Attention + output. The attention head now emits VOCAB logits
            # (docs/CODEBASE_DECODER_AUDIT.md finding 3), so the mixture is a
            # convex combination of two logit vectors -- a proper mixture of two
            # distributions. Previously this blended h (hidden space) with a
            # hidden-space attention vector and projected the result, which is a
            # double projection carrying no vocabulary information.
            gru_logits = self.output_proj.forward_raw(h[np.newaxis, :])[0]
            combined_vec = ((1.0 - self.attn_gate) * gru_logits
                            + self.attn_gate * cached_attn_logits)
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

                # `combined_vec` is already a vocab-logit vector (logit-space
                # mixture), so the sampled logits are a plain gather.
                logit_t = combined_vec[all_idx]
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
            all_gru_logits[pos] = gru_logits
            all_attn_logits[pos] = cached_attn_logits
            all_idx_stored[pos] = all_idx
            all_word_emb[pos] = word_emb
            all_input_idx[pos] = input_idx
            all_h[pos] = h_new

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

        # 1. output_proj: sparse Hebbian update (sampled softmax, only K+1 rows).
        #    combined = (1-g)*W_out*h + g*attn  =>  dL/dW_out[idx] = err (x) h.
        for pos in range(T):
            h_vec = all_h[pos]
            err = all_err_stored[pos]
            idx = all_idx_stored[pos]
            hebb = np.outer(err, h_vec) / T * (1.0 - self.attn_gate)
            out_weight[idx] += hebb * lr_proj - out_weight[idx] * wd

        # 2. Error backprop to hidden (sparse: only sampled rows).
        #    dL/dh = (1-g) * W_out[idx]^T @ err
        error_h_all = np.zeros((T, self.hidden_dim), dtype=np.float32)
        for pos in range(T):
            err = all_err_stored[pos]
            idx = all_idx_stored[pos]
            error_h_all[pos] = (1.0 - self.attn_gate) * (err @ out_weight[idx])

        # Logit-space stacks for the attention head and the mixture gate.
        # `err` is (K+1)-wide over the sampled indices `all_idx_stored[pos]`, so
        # the gate gradient is computed against the SAME sampled logits -- not
        # the full vocab (which would be both expensive and a different
        # objective than the one actually optimised).
        # Ragged-safe stacks: when the target is drawn into the shared
        # negatives, that position's error vector is one shorter. Pad to the
        # max width so np.stack succeeds; the gate gradient below only uses the
        # sampled-index gathers, which are always well-defined.
        _max_k = max(len(all_err_stored[p]) for p in range(T))
        error_logits_all = np.zeros((T, _max_k), dtype=np.float32)
        for p in range(T):
            e = all_err_stored[p]
            error_logits_all[p, :len(e)] = e
        gru_sampled_all = np.zeros((T, _max_k), dtype=np.float32)
        attn_sampled_all = np.zeros((T, _max_k), dtype=np.float32)
        for p in range(T):
            idx = all_idx_stored[p]
            k = len(idx)
            gru_sampled_all[p, :k] = all_gru_logits[p][idx]
            attn_sampled_all[p, :k] = all_attn_logits[p][idx]

        # 2b. P6 auxiliary "load-bearing" loss (research item #9, "Cosine
        # Misleads"). Reconstruct av (65-D Binder attributes) from the mean
        # hidden state; its gradient makes the thin concept pointer flow through
        # the network instead of being bypassed. Added into error_h_all so it
        # propagates to every parameter update below.
        if av_target is not None and all_h[0] is not None:
            h_stack = np.stack([hh for hh in all_h if hh is not None])
            av_pred = self.av_head.forward_raw(h_stack).mean(axis=0)  # (65,)
            av_diff = av_pred - av_target
            aux_loss = 0.5 * float(np.mean(av_diff ** 2))
            self._last_aux_loss = aux_loss
            # d(aux)/dh_pos = av_head.weight.T @ av_diff  (mean over positions)
            daux_dh = (self.av_head.weight.data.T @ av_diff) / T  # (hidden,)
            grad_term = 0.5 * aux_lambda * daux_dh
            error_h_all += grad_term[np.newaxis, :]
            # Update av_head itself (independent of freeze_core — this is the
            # grounding supervision, like W_h_bias).
            grad_w = np.outer(av_diff, h_stack.mean(axis=0))  # (65, hidden)
            self.av_head.weight.data -= lr_other * grad_w - self.av_head.weight.data * wd
            if self.av_head.bias is not None:
                self.av_head.bias.data -= lr_other * av_diff - self.av_head.bias.data * wd

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
        #
        # The attention head now emits VOCAB logits and is mixed with the GRU
        # readout in logit space (docs/CODEBASE_DECODER_AUDIT.md finding 3), so
        # its gradient is the LOGIT error scaled by the gate -- not error_h_all
        # (which is the hidden-space error). Using error_h_all here was a
        # shape-carrying no-op that weight-decayed the head to fro 1.16.
        if not freeze_core:
            # combined = (1-g)*W_out*h + g*attn, so dL/dattn = g * err.
            # The head's output_proj maps hidden -> vocab, so its gradient is
            # err (x) head_input over ALL vocab rows. `err` is (K+1)-wide over
            # the sampled indices, so scatter it into a full-vocab vector.
            attn_err = np.zeros(self.attention.vocab_size, dtype=np.float32)
            for pos in range(T):
                idx = all_idx_stored[pos]
                attn_err[idx] += all_err_stored[pos]
            attn_err *= self.attn_gate / T
            head_in = np.asarray(cond_proj, dtype=np.float32).mean(axis=0)
            self.attention.output_proj.weight.data += (
                np.outer(attn_err, head_in) * lr_attn
                - self.attention.output_proj.weight.data * wd)

        # 6b. logit-space mixture gate. dE/dgate = (attn_logits - gru_logits) .
        #     err, averaged over positions. Clipped to [0, 1] so the mixture
        #     stays a convex combination.
        if not freeze_core:
            gate_grad = float(np.mean(np.sum(
                (attn_sampled_all - gru_sampled_all) * error_logits_all, axis=1)))
            self.attn_gate = float(np.clip(
                self.attn_gate + lr_other * gate_grad, 0.0, 1.0))

        # Skip attention W_q/W_k/W_v (traces are sentence-specific, unreliable)

        # 7. P6 persistent concept-bias projection (W_h_bias) — the angular-gyrus
        #    working-memory hold. This is the NEW grounding parameter; it MUST
        #    learn whenever a concept pointer is present, EVEN under
        #    freeze_core=True (which protects only the seed-language core:
        #    condition_proj / GRU / attention). Brain-faithful separation: the
        #    grounding modulation learns without disturbing core syntax.
        #    Gradient: h_new += W_h_bias(pproj); combined = h*0.5 + attn*0.5, so
        #    dL/dW_h_bias = (pproj^T @ (error_h * 0.5)) / T.
        if _has_persistent and all_pproj is not None:
            pproj_stack = np.stack([p for p in all_pproj if p is not None])
            err_hb = np.clip(error_h_all * 0.5, -1.0, 1.0)
            hebbian_hb = (pproj_stack.T @ err_hb) / T
            lr_hb = 0.0005
            self.W_h_bias.weight.data += hebbian_hb.T * lr_hb - self.W_h_bias.weight.data * wd

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
                          max_sentences: Optional[int] = None,
                          sensorimotor_conditioning: Optional[np.ndarray] = None) -> List[dict]:
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

                # P6: derive the BOS initiation embedding from the embodied
                # signature (same as train_on_sentence) -- attached as
                # 'initial_emb' and injected at step 0 by the forward pass, so
                # train == gen. It does NOT overwrite the per-token linguistic
                # conditioning (the blended conditioning_embs above).
                _init = None
                if sensorimotor_conditioning is not None:
                    _sm = np.asarray(sensorimotor_conditioning, dtype=np.float32)[:65]
                    _W = getattr(self, "_linggen_W_sm", None)
                    if _W is not None:
                        _init = (_W @ _sm).astype(np.float32)
                        _n = np.linalg.norm(_init)
                        if _n > 0:
                            _init = _init / _n

                sentences.append({
                    'words': words,
                    'word_indices': word_indices,
                    'conditioning_embs': conditioning_embs,
                    'initial_emb': _init,
                })
        cap = max_sentences if max_sentences is not None else len(sentences)
        return sentences[:cap]

    def set_linggen_projection(self, W_sm: Optional[np.ndarray]) -> None:
        """P6: attach the learned angular-gyrus projection W_sm (65->75).

        Stored as ``self._linggen_W_sm`` and consumed by ``train_on_sentence`` /
        ``prepare_sentences`` when ``sensorimotor_conditioning`` is supplied.
        Passing None clears it (fail-soft back to blended conditioning).
        """
        self._linggen_W_sm = (np.asarray(W_sm, dtype=np.float32)
                              if W_sm is not None else None)

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
