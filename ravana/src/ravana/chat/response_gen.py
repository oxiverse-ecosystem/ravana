"""Auto-generated mixin module for CognitiveChatEngine."""
from __future__ import annotations
import sys, os, time, random, json, re, threading, hashlib
import urllib.request
import socket
socket.setdefaulttimeout(4.0)
from urllib.error import URLError
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque, Counter


from .constants import STOP_WORDS
from .chain_walker import ChainWalkerMixin
from ravana.language.surface_realizer import DiscourseState
# Compute project root (same logic as engine.py)
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ravana_ml.nn.neural_decoder import NeuralDecoder
from ravana_ml.nn.neuromodulator import NeuromodulatorEngine

from .models import CognitiveResponseContext
from .constants import _is_word_salad


class ResponseGenMixin(ChainWalkerMixin):
    """Mixin providing neural decoder generation, chitchat, and templated responses."""

    def _build_decoder_vocab(self):
        """Build vocabulary for the NeuralDecoder from graph concepts + GloVe + function words.

        Maps every graph concept label and common English function words to
        vocab indices and embedding vectors. Initializes the NeuralDecoder.
        Stage 2: Capped at MAX_DECODER_VOCAB_SIZE for speed (5× fewer logits).
        """
        if self._decoder_word_to_idx and self._decoder_idx_to_word and self._decoder_word_to_embed:
            vocab_size = len(self._decoder_word_to_idx)
            # Neuromodulator — tracks ACh/NE/DA/5-HT for adaptive generation & learning
            saved_nm = getattr(self, '_saved_neuromodulator_state', None)
            self.neuromodulator_engine = NeuromodulatorEngine()
            if saved_nm:
                self.neuromodulator_engine.load_state(saved_nm)
            self.neural_decoder = NeuralDecoder(
                vocab_size=vocab_size,
                embed_dim=self.dim,
                hidden_dim=256,
                n_attention_heads=4,
                contrastive_weight=0.5,
                contrastive_negatives=8,
                neuromodulator=self.neuromodulator_engine,
            )
            for word, idx in self._decoder_word_to_idx.items():
                if word in self._decoder_word_to_embed:
                    self.neural_decoder.word_embedding.weight.data[idx] = \
                        self._decoder_word_to_embed[word]
            self.neural_decoder.rebuild_vocab_cache()
            self._decoder_vocab_built = True
            return

        self._decoder_word_to_idx = {}
        self._decoder_idx_to_word = {}
        self._decoder_word_to_embed = {}

        # Special tokens
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        for i, tok in enumerate(special_tokens):
            self._decoder_word_to_idx[tok] = i
            self._decoder_idx_to_word[i] = tok
            vec = np.random.randn(self.dim).astype(np.float32) * 0.05
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self._decoder_word_to_embed[tok] = vec

        next_idx = len(special_tokens)
        max_vocab = self.MAX_DECODER_VOCAB_SIZE

        # Common function words for fluent generation.
        # These are genuine grammatical/function words — NOT content words
        # like "connect", "relate", "link" which should be learned naturally.
        function_words = [
            "a", "an", "the", "and", "but", "or", "is", "are", "was", "were", "be", "been",
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
            "of", "to", "for", "with", "from", "at", "by", "as", "on", "in", "out", "up", "off",
        ]

        # Add graph concept labels (highest priority after function words)
        with self._graph_lock:
            concept_labels = set()
            for nid, node in self.graph.nodes.items():
                if node.label:
                    concept_labels.add(node.label.lower())

        # Function words first (ensures fluent generation)
        for fw in function_words:
            if next_idx >= max_vocab:
                break
            if fw not in self._decoder_word_to_idx:
                self._decoder_word_to_idx[fw] = next_idx
                self._decoder_idx_to_word[next_idx] = fw
                vec = self._glove_vector(fw)
                if vec is None:
                    vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                    norm_v = np.linalg.norm(vec)
                    if norm_v > 0:
                        vec /= norm_v
                self._decoder_word_to_embed[fw] = vec.astype(np.float32)
                next_idx += 1

        # Graph concepts next
        for label in sorted(concept_labels):
            if next_idx >= max_vocab:
                break
            if label not in self._decoder_word_to_idx:
                self._decoder_word_to_idx[label] = next_idx
                self._decoder_idx_to_word[next_idx] = label
                nids = self._concept_keywords.get(label, [])
                if nids:
                    node = self.graph.get_node(nids[0])
                    if node and node.vector is not None:
                        vec = node.vector.copy()
                    else:
                        vec = self._glove_vector(label)
                else:
                    vec = self._glove_vector(label)
                if vec is None:
                    vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                    norm_v = np.linalg.norm(vec)
                    if norm_v > 0:
                        vec /= norm_v
                self._decoder_word_to_embed[label] = vec.astype(np.float32)
                next_idx += 1

        # Fill remaining slots with corpus words
        remaining = max_vocab - next_idx
        if remaining > 0:
            corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
            if os.path.exists(corpus_path):
                import re
                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus_text = f.read()
                words_in_corpus = set(re.findall(r"[a-zA-Z']{3,}", corpus_text.lower()))
                for w in sorted(words_in_corpus):
                    if remaining <= 0:
                        break
                    if w not in self._decoder_word_to_idx:
                        self._decoder_word_to_idx[w] = next_idx
                        self._decoder_idx_to_word[next_idx] = w
                        vec = self._glove_vector(w)
                        if vec is None:
                            vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                            norm_v = np.linalg.norm(vec)
                            if norm_v > 0:
                                vec /= norm_v
                        self._decoder_word_to_embed[w] = vec.astype(np.float32)
                        next_idx += 1
                        remaining -= 1

        # If forced vocab size is set (from saved state), match it for back-compat
        if hasattr(self, '_forced_vocab_size') and self._forced_vocab_size:
            target_size = min(self._forced_vocab_size, max_vocab)
            while len(self._decoder_word_to_idx) < target_size:
                pad_token = f"<pad_{next_idx}>"
                self._decoder_word_to_idx[pad_token] = next_idx
                self._decoder_idx_to_word[next_idx] = pad_token
                vec = np.random.randn(self.dim).astype(np.float32) * 0.05
                norm_v = np.linalg.norm(vec)
                if norm_v > 0:
                    vec /= norm_v
                self._decoder_word_to_embed[pad_token] = vec.astype(np.float32)
                next_idx += 1
            delattr(self, '_forced_vocab_size')

        vocab_size = len(self._decoder_word_to_idx)
        # Neuromodulator — tracks ACh/NE/DA/5-HT for adaptive generation & learning
        saved_nm = getattr(self, '_saved_neuromodulator_state', None)
        self.neuromodulator_engine = NeuromodulatorEngine()
        if saved_nm:
            self.neuromodulator_engine.load_state(saved_nm)
        self.neural_decoder = NeuralDecoder(
            vocab_size=vocab_size,
            embed_dim=self.dim,
            hidden_dim=256,
            n_attention_heads=4,
            contrastive_weight=0.5,
            contrastive_negatives=8,
            neuromodulator=self.neuromodulator_engine,
        )

        for word, idx in self._decoder_word_to_idx.items():
            if word in self._decoder_word_to_embed:
                self.neural_decoder.word_embedding.weight.data[idx] = \
                    self._decoder_word_to_embed[word]
        self.neural_decoder.rebuild_vocab_cache()
        self._decoder_vocab_built = True
        self._needs_seed_training = True
        self._needs_synthetic_training = True



    def _train_decoder_from_graph(self, min_synthetic: int = 0):
        """Train neural decoder on REAL English from seed corpus.
        
        No template sentences — those produce robotic output. Uses efficient
        prepare/loop pattern so ALL sentences are trained on each pass.
        
        Args:
            min_synthetic: Ignored (kept for compatibility).
        """
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return

        total_trained = 0
        corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    text = f.read()
                text = text.strip()
                if text:
                    # Pre-process once, then loop for efficient multi-pass training
                    nd = self.neural_decoder
                    all_sentences = nd.prepare_sentences(
                        text, self._decoder_word_to_embed, self._decoder_word_to_idx,
                        min_sentence_len=3,
                    )
                    n_available = len(all_sentences)
                    if n_available == 0:
                        return
                    passes = 5
                    sleep_every = 2
                    rng = np.random.RandomState(42)
                    for i in range(passes):
                        idx = rng.permutation(n_available)
                        err_sum = 0.0
                        for j in idx:
                            s = all_sentences[j]
                            err = nd.train_on_sentence(
                                s['words'], self._decoder_word_to_embed, self._decoder_word_to_idx,
                                word_indices=s['word_indices'],
                                conditioning_embs=s['conditioning_embs'],
                            )
                            err_sum += err
                        total_trained += n_available
                        if (i + 1) % sleep_every == 0:
                            nd.sleep_cycle()
                    nd.sleep_cycle()
                    self._decoder_training_count += total_trained
                    print(f"  [Decoder] Trained on {total_trained} real-English sentences"
                          f" ({passes} passes x {n_available} sents/pass)"
                          f"{' + ' + str(self._decoder_web_training_count) + ' from web' if self._decoder_web_training_count > 0 else ''}")
            except Exception as e:
                print(f"  [Decoder] Seed corpus training error: {e}")
        else:
            print(f"  [Decoder] No seed corpus found at {corpus_path}")



    def _train_decoder_on_seed_corpus(self, max_sentences: int = 500) -> int:
        """Train neural decoder on the bundled real-English corpus using efficient
        prepare/loop pattern with shuffled passes over ALL sentences.
        """
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return 0
        corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
        if not os.path.exists(corpus_path):
            return 0
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return 0
        # Pre-expand decoder vocab with corpus words
        words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", text.lower()))
        new_for_vocab = [w for w in words_in_corpus if w not in self._decoder_word_to_idx]
        if new_for_vocab:
            self._expand_decoder_vocab(new_for_vocab)

        # Pre-process corpus once, then loop with shuffled passes over ALL sentences
        nd = self.neural_decoder
        all_sentences = nd.prepare_sentences(
            text, self._decoder_word_to_embed, self._decoder_word_to_idx,
            min_sentence_len=3,
        )
        n_available = len(all_sentences)
        if n_available == 0:
            return 0
        n_passes = 5
        sleep_interval = 2
        rng = np.random.RandomState(42)
        passes = 0

        for i in range(n_passes):
            idx = rng.permutation(n_available)
            for j in idx:
                s = all_sentences[j]
                nd.train_on_sentence(
                    s['words'], self._decoder_word_to_embed, self._decoder_word_to_idx,
                    word_indices=s['word_indices'],
                    conditioning_embs=s['conditioning_embs'],
                )
            passes += n_available
            if (i + 1) % sleep_interval == 0:
                nd.sleep_cycle()

        if (n_passes % sleep_interval) != 0:
            nd.sleep_cycle()

        self._decoder_seed_training_count += passes
        self._decoder_training_count += passes
        return passes



    def _expand_decoder_vocab(self, new_words: List[str]):
        """Add new words to the decoder vocabulary (hippocampal replay)."

        New words get embeddings from GloVe (or random if unavailable).
        The decoder embedding table is resized and vocabulary caches rebuilt.
        Existing embeddings are preserved (neocortical consolidation).
        Stage 2: Respects MAX_DECODER_VOCAB_SIZE cap.
        """
        if not new_words or self.neural_decoder is None:
            return
        import sys
        if getattr(self, '_freeze_decoder_vocab', False):
            return
        max_vocab = self.MAX_DECODER_VOCAB_SIZE

        with self._vocab_lock:
            if len(self._decoder_word_to_idx) >= max_vocab:
                return
            added = 0
            for word in new_words:
                wl = word.lower().strip()
                if wl in self._decoder_word_to_idx or wl in ('<pad>', '<unk>', '<bos>', '<eos>'):
                    continue
                idx = len(self._decoder_word_to_idx)
                self._decoder_word_to_idx[wl] = idx
                self._decoder_idx_to_word[idx] = wl
                vec = self._glove_vector(wl)
                if vec is None:
                    vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec /= n
                self._decoder_word_to_embed[wl] = vec.astype(np.float32)
                added += 1

            if added == 0:
                return

            old_vocab_size = self.neural_decoder.vocab_size
            new_vocab_size = len(self._decoder_word_to_idx)

            # Resize decoder embedding table (preserve existing weights)
            old_weight = self.neural_decoder.word_embedding.weight.data
            new_weight = np.zeros((new_vocab_size, self.dim), dtype=np.float32)
            new_weight[:len(old_weight)] = old_weight
            for word, idx in list(self._decoder_word_to_idx.items()):
                if idx >= len(old_weight) and word in self._decoder_word_to_embed:
                    new_weight[idx] = self._decoder_word_to_embed[word]

            from ravana_ml.nn.module import Embedding, Linear
            new_emb = Embedding(new_vocab_size, self.dim)
            new_emb.weight.data = new_weight
            self.neural_decoder.word_embedding = new_emb

            # FIX: Use small random init for new output_proj rows instead of zeros.
            # Zero-init creates a dead zone where new words have ~0 logits, which
            # pulls gradient toward zero and destabilizes the output layer.
            old_out_w = self.neural_decoder.output_proj.weight.data
            new_out_w = np.random.randn(new_vocab_size, self.neural_decoder.hidden_dim).astype(np.float32) * 0.01
            new_out_w[:old_vocab_size] = old_out_w
            new_out_proj = Linear(self.neural_decoder.hidden_dim, new_vocab_size)
            new_out_proj.weight.data = new_out_w
            if self.neural_decoder.output_proj.bias is not None:
                old_bias = self.neural_decoder.output_proj.bias.data
                new_bias = np.random.randn(new_vocab_size).astype(np.float32) * 0.01
                new_bias[:old_vocab_size] = old_bias
                new_out_proj.bias.data = new_bias
            self.neural_decoder.output_proj = new_out_proj

            self.neural_decoder.vocab_size = new_vocab_size
            self.neural_decoder._vocab_dim = self.dim
            self.neural_decoder.rebuild_vocab_cache()

        # Hippocampal replay: mix new embeddings with similar existing ones
        # Build batch embedding matrix for existing words once (avoids O(V²) per new word)
        with self._vocab_lock:
            existing_labels = []
            existing_vecs = []
            new_set = set(w.lower().strip() for w in new_words)
            for ew, ei in list(self._decoder_word_to_idx.items()):
                if ew.startswith('<') or ew in new_set:
                    continue
                ev = self._decoder_word_to_embed.get(ew)
                if ev is not None:
                    existing_labels.append(ew)
                    existing_vecs.append(ev)
            if existing_vecs:
                existing_mat = np.stack(existing_vecs, axis=0).astype(np.float32)
            else:
                existing_mat = np.empty((0, self.dim), dtype=np.float32)

        for word in new_words:
            wl = word.lower().strip()
            idx = self._decoder_word_to_idx.get(wl)
            if idx is None:
                continue
            vec = self._decoder_word_to_embed.get(wl)
            if vec is None:
                continue
            if len(existing_labels) > 0:
                sims = existing_mat @ vec
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_label = existing_labels[best_idx]
                # Blend: 70% new + 30% nearest (hippocampal-neocortical consolidation)
                if best_sim > 0.3:
                    best_vec = self._decoder_word_to_embed[best_label]
                    blended = vec * 0.7 + best_vec * 0.3
                    blended /= np.linalg.norm(blended)
                    self._decoder_word_to_embed[wl] = blended
                    new_weight[idx] = blended
        self.neural_decoder.word_embedding.weight.data = new_weight

        print(f"  [Decoder] Expanded vocab by {added} words (now {new_vocab_size})")



    def _build_conditioning_for_text(self, topic: str, text_words: List[str]) -> Optional[np.ndarray]:
        """Build graph walk conditioning embeddings for a text's important words.

        Walks from the topic through up to 5 of the most frequent content words
        in the text, collecting their concept embeddings from the graph.
        Falls back to text word embeddings if no graph concepts match.
        """
        concept_embs = []
        seen = set()

        # Collect from graph concepts first
        tl = topic.lower().strip()
        if tl in self._concept_keywords:
            nids = self._concept_keywords[tl]
            node = self.graph.get_node(nids[0])
            if node and node.vector is not None:
                concept_embs.append(node.vector.copy())
                seen.add(tl)

        counted = {}
        for w in text_words:
            wl = w.lower().strip(".,!?\"' ")
            if wl not in STOP_WORDS and len(wl) >= 3:
                counted[wl] = counted.get(wl, 0) + 1
        for w, _ in sorted(counted.items(), key=lambda x: x[1], reverse=True)[:5]:
            if w not in seen and w in self._concept_keywords:
                nids = self._concept_keywords[w]
                node = self.graph.get_node(nids[0])
                if node and node.vector is not None:
                    concept_embs.append(node.vector.copy())
                    seen.add(w)

        # Fall back to GloVe/decoder embeddings if no graph concepts found
        if len(concept_embs) < 1:
            tl = topic.lower().strip()
            if tl in self._decoder_word_to_embed:
                concept_embs.append(self._decoder_word_to_embed[tl].copy())
                seen.add(tl)
            for w, _ in sorted(counted.items(), key=lambda x: x[1], reverse=True)[:5]:
                if w not in seen and w in self._decoder_word_to_embed:
                    concept_embs.append(self._decoder_word_to_embed[w].copy())
                    seen.add(w)

        if len(concept_embs) < 1:
            return None
        return np.stack(concept_embs, axis=0).astype(np.float32)

    # ─── Neural Decoder Generation ───



    def _generate_with_decoder(self, ctx: CognitiveResponseContext) -> Optional[str]:
        """Generate response using NeuralDecoder with graph conditioning.

        Builds conditioning embeddings from the subject and associated concepts,
        then runs the decoder autoregressively with cerebellar n-gram bias and
        basal ganglia gating for biologically-plausible word selection.

        Returns None if decoder is not ready or generation fails.
        (Quality gate is now in _generate_response.)
        """
        if self.neural_decoder is None or not self._decoder_vocab_built:
            print(f"  [Decoder Gen] FAIL: decoder=None or vocab not built")
            return None

        subject = ctx.subject
        if not subject:
            return None

        # Build conditioning embeddings from graph concepts
        concept_embs = []
        seen_labels: Set[str] = set()

        # Extract individual concept words from subject that ARE in decoder vocab
        subject_words = subject.lower().split()
        vocab_words = []
        with self._graph_lock:
            for w in subject_words:
                wl = w.lower()
                if wl in self._decoder_word_to_idx and wl not in seen_labels:
                    vocab_words.append(wl)
                    seen_labels.add(wl)

        # Also check associated concepts for vocab words
        for label, score in ctx.associated_concepts[:10]:
            ll = label.lower()
            if ll in self._decoder_word_to_idx and ll not in seen_labels:
                vocab_words.append(ll)
                seen_labels.add(ll)
                if len(vocab_words) >= 8:
                    break

        # Fallback: if no vocab words found, try the subject as-is
        if not vocab_words:
            subj_lower = subject.lower()
            if subj_lower in self._decoder_word_to_idx:
                vocab_words.append(subj_lower)
            else:
                print(f"  [Decoder Gen] FAIL: subject='{subject}' no words in vocab")
                return None

        # Build embeddings from graph nodes or decoder word embeddings
        for vw in vocab_words[:6]:
            if vw in self._concept_keywords:
                nids = self._concept_keywords[vw]
                node = self.graph.get_node(nids[0])
                if node and node.vector is not None:
                    concept_embs.append(node.vector.copy())
                elif vw in self._decoder_word_to_embed:
                    concept_embs.append(self._decoder_word_to_embed[vw].copy())
            elif vw in self._decoder_word_to_embed:
                concept_embs.append(self._decoder_word_to_embed[vw].copy())

        # Final fallback
        if len(concept_embs) < 1:
            return None

        # Add sentence-level compositional vector (N400/P600 integration) as conditioning
        sent_vec = getattr(ctx, 'sentence_vector', None)
        if sent_vec is not None and np.any(sent_vec != 0):
            concept_embs.append(sent_vec.astype(np.float32) * 0.6)

        conditioning_embs = np.stack(concept_embs, axis=0).astype(np.float32)

        # Special token indices
        bos_idx = self._decoder_word_to_idx.get("<bos>", 0)
        eos_idx = self._decoder_word_to_idx.get("<eos>", 2)

        print(f"  [Decoder Gen] conditioning_embs shape={conditioning_embs.shape}, vocab_words={vocab_words[:6]}, bos_idx={bos_idx}, eos_idx={eos_idx}")

        # Add sentence-level compositional vector (N400/P600 integration) as conditioning
        sent_vec = getattr(ctx, 'sentence_vector', None)
        if sent_vec is not None and np.any(sent_vec != 0):
            concept_embs.append(sent_vec.astype(np.float32) * 0.6)

        conditioning_embs = np.stack(concept_embs, axis=0).astype(np.float32)

        # Special token indices
        bos_idx = self._decoder_word_to_idx.get("<bos>", 0)
        eos_idx = self._decoder_word_to_idx.get("<eos>", 2)

        print(f"  [Decoder Gen] conditioning_embs shape={conditioning_embs.shape}, bos_idx={bos_idx}, eos_idx={eos_idx}, subject='{subject}'")

        # Dynamic temperature from cognitive state.
        # Higher when: curious, aroused, learning. Lower when: confident, focused.
        # With larger vocab the decoder needs more temperature to explore word choices.
        base_temp = 0.50
        arousal = ctx.arousal if hasattr(ctx, 'arousal') else 0.3
        temp = base_temp * (0.6 + arousal * 0.8)
        # Higher dopamine tone = more creative/exploratory
        dt = getattr(self, '_dopamine_tone', 0.5)
        temp *= (0.6 + dt * 0.8)
        # More training = more confidence = slightly lower temp
        training_factor = max(0.7, 1.0 - min(0.3, self._decoder_training_count / 20000))
        temp *= training_factor
        temp = max(0.25, min(1.0, temp))

        # Build content word IDs (non-function, non-special tokens)
        # IMPORTANT: Do NOT include semantic connector words like "connects",
        # "relates", "links", "leads" here — those are content words whose
        # presence in the function-word set was causing the decoder to default
        # to template-like "X connects with Y" patterns.
        function_words_set = {
            "a", "an", "the", "and", "but", "or", "is", "are", "was", "were", "be", "been",
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
            "of", "to", "for", "with", "from", "at", "by", "as", "on", "in", "out", "up", "off",
            "<pad>", "?", "<bos>", "<eos>",
        }
        with self._vocab_lock:
            content_word_ids = {
                idx for word, idx in self._decoder_word_to_idx.items()
                if word not in function_words_set and not word.startswith("<")
            }

        # Stage 2: Boost subject concept in conditioning to seed content words
        subject_idx = self._decoder_word_to_idx.get(subject.lower())
        if subject_idx is not None:
            subject_boost = {subject_idx: 3.0}
        else:
            subject_boost = None

        try:
            generated = self.neural_decoder.generate(
                conditioning_embs=conditioning_embs,
                max_steps=22,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                temperature=temp,
                cerebellar_ngram=self.cerebellar_ngram,
                idx_to_word=self._decoder_idx_to_word,
                basal_ganglia=self.basal_ganglia,
                content_word_ids=content_word_ids,
                token_boost=subject_boost,
            )

            print(f"  [Decoder Gen] generated={generated}")
            print(f"  [Decoder Gen] idx_to_word for first few: {[(idx, self._decoder_idx_to_word.get(idx, '?')) for idx in generated[:5]]}")

            if not generated or len(generated) < 3:
                print(f"  [Decoder Gen] FAIL: generated too short or None")
                return None

            # Convert indices to words, filtering special tokens
            words = []
            for idx in generated:
                word = self._decoder_idx_to_word.get(idx, "")
                if word and not word.startswith("<"):
                    words.append(word)

            if len(words) < 3:
                return None

            text = " ".join(words)

            # Let the decoder's own quality metrics (CE, top-1) be the gate.
            # No hardcoded blacklists — they reject legitimate English patterns.
            # If the decoder generated it, and it passed the word salad check,
            # it's good enough to show. Over time, training improves quality.

            # Bigram repetition gate: reject if any bigram appears 4+ times
            if len(words) >= 6:
                bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
                from collections import Counter
                bg_counts = Counter(bigrams)
                if any(c >= 4 for c in bg_counts.values()):
                    return None

            # Clean up basic punctuation issues
            text = text[0].upper() + text[1:]
            if not text.endswith((".", "?", "!")):
                text += "."
            # Remove double spaces
            text = re.sub(r'\s+', ' ', text)

            return text
        except Exception as e:
            if self._trace_enabled:
                print(f"  [trace] decoder generation error: {e}")
            return None



    # ─── Situation-Model-Guided Generation ───


    def _generate_narrative_paragraph(self, ctx: CognitiveResponseContext) -> Optional[str]:
        """Generate a fluid narrative paragraph using the event schema library
        and situation-modulated verb selection, instead of discrete SVO triples.

        Key differences from the standard syntactic pipeline:
        1. Uses Event Schemas (process chains) when available for richer descriptions
        2. Always uses situation-modulated verb selection (not just relation-type)
        3. Generates a structured paragraph: definition -> process -> significance
        4. Each sentence builds on the previous one with pronoun resolution
        5. Discourse markers and adverbial modifiers for natural flow

        Returns the paragraph text or None.
        """
        if not ctx.subject:
            return None

        subject = ctx.subject
        subject_lower = subject.lower()
        situation_vec = getattr(ctx, 'situation_vector', None)
        situation_narrative = getattr(ctx, 'situation_narrative', {})
        coherence = situation_narrative.get('coherence', 0.5)
        theme = situation_narrative.get('theme', subject_lower)
        active = situation_narrative.get('active_concepts', [])

        # Step 1: Get the vector function from VerbLexicon for situation-modulated verb selection
        from ravana.language.verb_lexicon import VerbLexicon
        vector_fn = getattr(self, '_glove_vector', None)
        dopamine_tone = getattr(self, '_dopamine_tone', 0.5)

        # Step 2: Check Event Schema Library for a process schema
        schema_lib = getattr(self, 'event_schema_lib', None)
        schema = schema_lib.get_schema(subject_lower) if schema_lib else None

        # Get top associations for enrichment
        assocs = ctx.associated_concepts
        noun_assocs = []
        for label, weight in assocs:
            ll = label.lower()
            if ll in self._GRAMMATICAL_CONCEPTS:
                continue
            if len(noun_assocs) < 5:
                noun_assocs.append((label, weight))

        # Step 3: Generate the narrative paragraph
        utterances = []

        # Sentence 1: Definition/nature of the concept
        # Use the relation from the strongest association for modulation
        first_rel = "semantic"
        first_target = ""
        if noun_assocs:
            first_target = noun_assocs[0][0]
            nid_s = self._concept_keywords.get(subject_lower, [None])[0]
            nid_t = self._concept_keywords.get(first_target.lower(), [None])[0]
            if nid_s and nid_t:
                edge = self.graph.get_edge(nid_s, nid_t)
                if edge is None:
                    edge = self.graph.get_edge(nid_t, nid_s)
                if edge:
                    first_rel = edge.relation_type or "semantic"

        # Use situation-modulated verb for the first sentence
        verb_s1 = VerbLexicon.select_verb_with_situation(
            relation=first_rel,
            situation_vector=situation_vec,
            subject=subject,
            object=first_target or subject,
            dopamine_tone=dopamine_tone,
            vector_fn=vector_fn,
        )

        # Build definition frame with situation-modulated verb
        if first_target and first_target.lower() != subject_lower:
            frame1 = self.syntactic_assembly.bind_to_sentence(
                subject=subject,
                relation=first_rel,
                target=first_target,
                pos_map=getattr(self, '_concept_pos', {}),
            )
            # Override verb with situation-modulated version
            frame1.verb_phrase = verb_s1
            disc_ctx1 = DiscourseState(
                sentence_index=0, discourse_type="explain",
                total_sentences=3, free_energy=max(0.15, 1.0 - coherence * 0.7),
            )
            s1 = self.surface_realizer.realize(
                frame=frame1, discourse_context=disc_ctx1,
                dopamine_tone=dopamine_tone,
                cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
            )
            if s1 and len(s1) > 5:
                utterances.append(s1)

        # Sentence 2: Process description (from schema or generated)
        disc_ctx2 = DiscourseState(
            sentence_index=len(utterances), discourse_type="elaborate",
            total_sentences=3, free_energy=max(0.15, 1.0 - coherence * 0.7),
        )

        if schema and len(schema.steps) >= 2:
            # Use event schema for rich process narrative
            # Schema sentences already have correct pronouns and flow — use them as-is
            narrative_sents = schema_lib.get_narrative_from_schema(subject_lower)
            if narrative_sents:
                for ns in narrative_sents[:2]:
                    # Schema sentences already have proper pronouns — no "it " prefix needed
                    # Just capitalize the first letter if it's the first utterance, else lowercase
                    if len(utterances) == 0:
                        utterances.append(ns[0].upper() + ns[1:])
                    else:
                        utterances.append(ns[0].lower() + ns[1:])
        elif noun_assocs and len(noun_assocs) >= 2:
            # Generate process description from second strongest association
            second = noun_assocs[1] if len(noun_assocs) > 1 else noun_assocs[0]
            second_rel = "causal"
            verb_s2 = VerbLexicon.select_verb_with_situation(
                relation="causal",
                situation_vector=situation_vec,
                subject=subject,
                object=second[0],
                dopamine_tone=dopamine_tone,
                vector_fn=vector_fn,
            )
            frame2 = self.syntactic_assembly.bind_to_sentence(
                subject=subject,
                relation=second_rel,
                target=second[0],
                pos_map=getattr(self, '_concept_pos', {}),
            )
            frame2.verb_phrase = verb_s2
            s2 = self.surface_realizer.realize(
                frame=frame2, discourse_context=disc_ctx2,
                dopamine_tone=dopamine_tone,
                cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
            )
            if s2 and len(s2) > 5:
                if utterances:
                    s2_lower = s2[0].lower() + s2[1:]
                    utterances.append(s2_lower)
                else:
                    utterances.append(s2)

            # Sentence 3: Significance/impact sentence (only if no schema was used)
            disc_ctx3 = DiscourseState(
                sentence_index=len(utterances), discourse_type="conclude",
                total_sentences=3, free_energy=max(0.1, 1.0 - coherence * 0.8),
            )
            if noun_assocs and len(noun_assocs) >= 3:
                third = noun_assocs[2]
                verb_s3 = VerbLexicon.select_verb_with_situation(
                    relation="causal",
                    situation_vector=situation_vec,
                    subject=subject,
                    object=third[0],
                    dopamine_tone=dopamine_tone,
                    vector_fn=vector_fn,
                )
            elif noun_assocs:
                third = noun_assocs[0]
                verb_s3 = VerbLexicon.select_verb_with_situation(
                    relation="semantic",
                    situation_vector=situation_vec,
                    subject=subject,
                    object=third[0],
                    dopamine_tone=dopamine_tone,
                    vector_fn=vector_fn,
                )
            else:
                third = None
                verb_s3 = "shapes"
            if third:
                frame3 = self.syntactic_assembly.bind_to_sentence(
                    subject=subject if len(utterances) == 0 else "it",
                    relation=third_rel if 'third_rel' in dir() else "causal",
                    target=third[0],
                    pos_map=getattr(self, '_concept_pos', {}),
                )
                frame3.verb_phrase = verb_s3
                frame3.pronoun_subject = "it"
                s3 = self.surface_realizer.realize(
                    frame=frame3, discourse_context=disc_ctx3,
                    dopamine_tone=dopamine_tone,
                    cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                )
                if s3 and len(s3) > 5:
                    if len(utterances) >= 2:
                        s3_lower = s3[0].lower() + s3[1:]
                        if random.random() < 0.5:
                            utterances.append(f"and {s3_lower}")
                        else:
                            utterances.append(s3_lower)
                    else:
                        utterances.append(s3)

        # If we have at least 2 utterances, join them into a paragraph
        if len(utterances) >= 2:
            paragraph = " ".join(utterances)
            # Clean up: ensure proper capitalization and punctuation
            paragraph = paragraph[0].upper() + paragraph[1:]
            if not paragraph.endswith((".", "?", "!")):
                paragraph += "."
            paragraph = re.sub(r'\s+', ' ', paragraph)
            return paragraph

        # Fall back to single-sentence response if paragraph generation failed
        if utterances:
            return utterances[0]

        return None


    def _generate_with_situation_model(self, ctx: CognitiveResponseContext) -> Optional[Tuple[str, str]]:
        """Generate response using the SituationModel's blended cognitive state.

        DMN-inspired generation path:
        1. Get the blended situation vector from the SituationModel
        2. Use it as conditioning context for the NeuralDecoder
        3. If the decoder fails, use the situation vector with event schemas
           and situation-modulated verb selection to generate a narrative
           paragraph (definition -> process -> significance)

        This path fundamentally differs from the standard syntactic path:
        - It uses Event Schemas (process knowledge) for richer descriptions
        - It uses situation-modulated verbs (builds, strengthens, shapes)
          instead of generic relational verbs (relates to, connects with)
        - It produces a coherent paragraph with discourse flow, not
          a sequence of independent SVO sentences

        Returns (response_text, strategy_name) or None.
        """
        situation_vec = getattr(ctx, 'situation_vector', None)
        situation_narrative = getattr(ctx, 'situation_narrative', {})

        if situation_vec is None or not hasattr(self, 'situation_model'):
            return None

        # Attempt 1: NeuralDecoder with situation vector conditioning
        if self.neural_decoder is not None and self._decoder_vocab_built:
            try:
                # Build conditioning from situation vector + top concepts
                cond_embs = []
                
                # Add situation vector as primary conditioning (tiled 3x for stability)
                sv = situation_vec.astype(np.float32)
                if np.any(sv != 0):
                    cond_embs.extend([sv.copy()] * 3)

                # Add top active concepts for specific content guidance
                active = situation_narrative.get('active_concepts', [])
                for label, weight in active[:5]:
                    if weight > 0.1:
                        nids = self._concept_keywords.get(label.lower(), [])
                        if nids:
                            node = self.graph.get_node(nids[0])
                            if node and node.vector is not None:
                                cond_embs.append(node.vector.copy() * 0.5)

                # Add subject-specific embeddings
                if ctx.subject:
                    sl = ctx.subject.lower()
                    nids = self._concept_keywords.get(sl, [])
                    if nids:
                        node = self.graph.get_node(nids[0])
                        if node and node.vector is not None:
                            cond_embs.append(node.vector.copy() * 0.8)
                    elif sl in self._decoder_word_to_embed:
                        cond_embs.append(self._decoder_word_to_embed[sl].copy() * 0.8)

                if not cond_embs:
                    return None

                conditioning_embs = np.stack(cond_embs, axis=0).astype(np.float32)
                bos_idx = self._decoder_word_to_idx.get("<bos>", 0)
                eos_idx = self._decoder_word_to_idx.get("<eos>", 2)

                # Compute temperature from situation coherence and diversity
                coherence = situation_narrative.get('coherence', 0.5)
                diversity = situation_narrative.get('diversity', 0.5)
                base_temp = 0.45
                temp = base_temp * (0.8 + coherence * 0.4) * (0.7 + diversity * 0.6)
                temp = max(0.25, min(0.85, temp))

                # Content word IDs set
                function_words_set = {
                    "a", "an", "the", "and", "but", "or", "is", "are", "was", "were", "be", "been",
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
                    "of", "to", "for", "with", "from", "at", "by", "as", "on", "in", "out", "up", "off",
                    "<pad>", "?", "<bos>", "<eos>",
                }
                with self._vocab_lock:
                    content_word_ids = {
                        idx for word, idx in self._decoder_word_to_idx.items()
                        if word not in function_words_set and not word.startswith("<")
                    }

                generated = self.neural_decoder.generate(
                    conditioning_embs=conditioning_embs,
                    max_steps=24,
                    bos_idx=bos_idx,
                    eos_idx=eos_idx,
                    temperature=temp,
                    cerebellar_ngram=self.cerebellar_ngram,
                    idx_to_word=self._decoder_idx_to_word,
                    basal_ganglia=self.basal_ganglia,
                    content_word_ids=content_word_ids,
                )

                if generated and len(generated) >= 5:
                    words = []
                    for idx in generated:
                        word = self._decoder_idx_to_word.get(idx, "")
                        if word and not word.startswith("<"):
                            words.append(word)
                    if len(words) >= 4:
                        text = " ".join(words)
                        # Check for excessive repetition
                        if len(words) >= 6:
                            bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
                            bg_counts = Counter(bigrams)
                            if any(c >= 4 for c in bg_counts.values()):
                                pass  # Fall through to narrative generation
                            else:
                                text = text[0].upper() + text[1:]
                                if not text.endswith((".", "?", "!")):
                                    text += "."
                                text = re.sub(r'\s+', ' ', text)
                                if not _is_word_salad(text):
                                    if getattr(self, '_trace_enabled', False):
                                        print(f"  [trace]   SM decoder: generated fluid response")
                                    return (text, "situation_model_decoder")
            except Exception as e:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [trace]   SM decoder error: {e}")

        # Attempt 2: Narrative paragraph generation
        # Uses Event Schemas + situation-modulated verbs + paragraph structure
        if ctx.subject and hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
            try:
                paragraph = self._generate_narrative_paragraph(ctx)
                if paragraph and len(paragraph) > 15 and not _is_word_salad(paragraph):
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [trace]   SM narrative: generated fluid paragraph")
                    return (paragraph, "situation_model_narrative")
            except Exception as e:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [trace]   SM narrative error: {e}")

        # Attempt 3: Original syntactic pipeline with frame merging
        # (fallback if narrative generation fails)
        try:
            if ctx.subject:
                coherence = situation_narrative.get('coherence', 0.5)
                noun_assocs = []
                for label, weight in ctx.associated_concepts:
                    ll = label.lower()
                    if ll in self._GRAMMATICAL_CONCEPTS:
                        continue
                    pos = getattr(self, '_concept_pos', {}).get(ll, 'noun')
                    if pos == 'noun':
                        noun_assocs.append((label, weight))
                if noun_assocs:
                    utterances = []
                    utterance_frames = []
                    disc_ctx = DiscourseState(
                        sentence_index=0, discourse_type="explain",
                        total_sentences=min(3, len(noun_assocs)),
                        free_energy=max(0.2, 1.0 - coherence),
                    )
                    seen = {ctx.subject.lower()}
                    for i, (label, weight) in enumerate(noun_assocs[:3]):
                        if label.lower() in seen:
                            continue
                        seen.add(label.lower())
                        relation = "semantic"
                        nid_subj = self._concept_keywords.get(ctx.subject.lower(), [None])[0]
                        nid_label = self._concept_keywords.get(label.lower(), [None])[0]
                        if nid_subj is not None and nid_label is not None:
                            edge = self.graph.get_edge(nid_subj, nid_label)
                            if edge is None:
                                edge = self.graph.get_edge(nid_label, nid_subj)
                            if edge is not None:
                                relation = edge.relation_type or "semantic"
                        frame = self.syntactic_assembly.bind_to_sentence(
                            subject=ctx.subject, relation=relation, target=label,
                            pos_map=getattr(self, '_concept_pos', {}),
                        )
                        if utterances and utterance_frames and coherence > 0.4:
                            last_frame = utterance_frames[-1]
                            if last_frame and label.lower() == last_frame.object_concept.lower():
                                frame.embedded_relation = "which"
                                utterances[-1] = self._make_embedded(utterances[-1], frame, disc_ctx)
                                utterance_frames[-1] = frame
                                continue
                        disc_ctx.sentence_index = len(utterances)
                        sentence = self.surface_realizer.realize(
                            frame=frame, discourse_context=disc_ctx,
                            dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                            cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                        )
                        if sentence and len(sentence) > 5:
                            utterances.append(sentence)
                            utterance_frames.append(frame)
                    if utterances:
                        response = " ".join(utterances)
                        return (response, "situation_model_syntax")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace]   SM syntax error: {e}")

        return None


    def _make_embedded(self, parent_sentence: str, child_frame, disc_ctx) -> str:
        """Try to embed a child frame as a relative clause in the parent sentence."""
        try:
            rel = child_frame.embedded_relation or "which"
            child_sentence = self.surface_realizer.realize(
                frame=child_frame,
                discourse_context=disc_ctx,
                dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
            )
            if child_sentence:
                # Remove trailing period from parent, add relative clause
                parent_clean = parent_sentence.rstrip(".!?")
                child_lower = child_sentence[0].lower() + child_sentence[1:]
                child_clean = child_lower.rstrip(".!?")
                return f"{parent_clean}, {rel} {child_clean}."
        except Exception:
            pass
        return parent_sentence


    def _handle_chitchat(self, text: str, subject: str) -> Optional[str]:
        """Detect and respond to greetings and chit-chat naturally."""
        import re
        import random
        t = text.lower().strip(" ?!.,")
        if not t:
            return None

        # 1. Greetings
        greetings = (
            r"\b(hi|hello|hey|yo|sup|greetings|whats\s*up|howdy|good\s*morning|good\s*afternoon|good\s*evening)\b"
        )
        # 2. Well-being
        wellbeing = (
            r"\b(how\s*are\s*you|how\s*is\s*it\s*going|how\s*are\s*you\s*doing|how\s*have\s*you\s*been|hows\s*it\s*going|hows\s*life)\b"
        )
        # 3. Capabilities / Identity
        capabilities = (
            r"\b(what\s*can\s*you\s*do|what\s*do\s*you\s*do|how\s*do\s*you\s*work|tell\s*me\s*about\s*yourself|who\s*are\s*you|what\s*is\s*your\s*name)\b"
        )
        # 4. Farewells
        farewells = (
            r"\b(bye|goodbye|see\s*you|good\s*night|farewell)\b"
        )

        is_greeting = re.search(greetings, t) is not None
        is_wellbeing = re.search(wellbeing, t) is not None
        is_capability = re.search(capabilities, t) is not None
        is_farewell = re.search(farewells, t) is not None

        # Skip chitchat only if it's NOT conversational and matches a query pattern
        if not (is_greeting or is_wellbeing or is_capability or is_farewell) and subject:
            for pattern, _ in self.QUERY_PATTERNS:
                if re.search(pattern, t):
                    return None

        # Remove hardcoded return values to route greetings/chit-chat through the normal cognitive pipeline
        return None



    def _detect_comparison_concepts(self, query: str) -> Optional[Tuple[str, str]]:
        """Detects if a query is comparing two concepts and returns them if found."""
        q_lower = query.lower()
        # Look for comparison indicators
        comparison_indicators = {"difference", "compare", "contrast", "distinguish", "vs", "versus", "between", "relationship", "relation"}
        if not any(indicator in q_lower for indicator in comparison_indicators):
            return None
            
        # Find all words in the query that are nouns and not stop words or comparison indicators
        words = [w.strip(".,!?\"'") for w in q_lower.split()]
        known_concepts = []
        for w in words:
            if len(w) > 2 and w not in STOP_WORDS and w not in self.QUESTION_WORDS and w not in self.TOPIC_SKIP_WORDS and w not in comparison_indicators:
                if w in self._concept_labels or w in self._concept_keywords:
                    if w not in known_concepts:
                        known_concepts.append(w)
                        
        # Find all valid nouns in the query
        all_nouns = []
        for w in words:
            if len(w) > 2 and w not in STOP_WORDS and w not in self.QUESTION_WORDS and w not in self.TOPIC_SKIP_WORDS and w not in comparison_indicators:
                if w not in all_nouns:
                    all_nouns.append(w)
                    
        if len(known_concepts) >= 2:
            return (known_concepts[0], known_concepts[1])
        elif len(known_concepts) == 1 and len(all_nouns) >= 2:
            # Find the other noun
            other_nouns = [n for n in all_nouns if n != known_concepts[0]]
            if other_nouns:
                return (known_concepts[0], other_nouns[0])
                
        return None



    def _generate_comparison_response(self, concept_a: str, concept_b: str) -> str:
        """Generates a human-like cognitive comparison between two concepts using graph associations."""
        # Retrieve associations for concept A
        a_nids = self._concept_keywords.get(concept_a.lower(), [])
        a_assocs = []
        if a_nids:
            a_assocs = [label.lower() for label, _ in self._spread_and_collect(a_nids)[:3]
                        if label.lower() != concept_a.lower() and label.lower() not in STOP_WORDS]
            
        # Retrieve associations for concept B
        b_nids = self._concept_keywords.get(concept_b.lower(), [])
        b_assocs = []
        if b_nids:
            b_assocs = [label.lower() for label, _ in self._spread_and_collect(b_nids)[:3]
                        if label.lower() != concept_b.lower() and label.lower() not in STOP_WORDS]

        # Route through SurfaceRealizer if available
        try:
            if hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
                utterances = []
                for concept, assocs in [(concept_a, a_assocs), (concept_b, b_assocs)]:
                    for target in assocs[:1]:
                        frame = self.syntactic_assembly.bind_to_sentence(
                            subject=concept, relation="semantic", target=target,
                            pos_map=getattr(self, '_concept_pos', {}),
                        )
                        disc_ctx = DiscourseState(
                            sentence_index=len(utterances), discourse_type="explain",
                            total_sentences=2, free_energy=0.3,
                        )
                        sentence = self.surface_realizer.realize(
                            frame=frame, discourse_context=disc_ctx,
                            dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                            cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                        )
                        if sentence and len(sentence) > 5:
                            utterances.append(sentence)
                if len(utterances) >= 2:
                    return " ".join(utterances)
        except Exception:
            pass
        # Generative comparison fallback via SurfaceRealizer
        try:
            s = self._try_surface_realize(
                subject=concept_a, target=concept_b,
                discourse_type="explain", free_energy=0.35, min_len=8)
            if s:
                return s
        except Exception:
            pass
        return f"i see that {concept_a} and {concept_b} are different concepts."



    def _detect_reasoning_question(self, query: str) -> bool:
        """Detects if a query is a reasoning question (logical choice, conditional, riddle)."""
        q = query.lower().strip(" ?!.")
        # Only questions are considered reasoning questions
        is_question = query.strip().endswith('?') or any(w in q for w in ["what", "who", "where", "when", "why", "how"])
        if not is_question:
            return False
            
        reasoning_patterns = [
            r"\b(if|when|suppose|assume|predict)\b",  # conditional/scenario
            r"\b(why|how does|how do|how to)\b",      # causal/procedural reasoning
            r"\b(or)\b",                              # logical choice / alternative
            r"\b(riddle|puzzle|logic|solve|math)\b",  # riddle/puzzle
        ]
        return any(re.search(pat, q) for pat in reasoning_patterns)



    def _generate_reasoning_fallback(self, query: str) -> str:
        """Generates a natural curious-teenager fallback response for reasoning questions."""
        q = query.lower().strip(" ?!.")
        if "or" in q:
            choices = [
                "i'm not sure which one! how would you choose?",
                "that's a tough choice! what do you think?",
                "i'm still figuring out how to choose between things. what do you think?",
            ]
            return random.choice(choices)
        elif "if" in q or "when" in q or "suppose" in q:
            scenarios = [
                "that's an interesting scenario! i don't know what would happen yet. what do you think?",
                "hmm, i'm not sure what would happen in that case! what do you predict?",
                "i'm still learning how the world works. what do you think happens?",
            ]
            return random.choice(scenarios)
        elif "why" in q:
            reasons = [
                "i'm still figuring out why that is. what do you think is the reason?",
                "that's a deep question. why do you think?",
                "i don't know the cause yet! what's your theory?",
            ]
            return random.choice(reasons)
        elif "how" in q:
            methods = [
                "i'm still learning how things work. how do you think it's done?",
                "i don't know how that works yet. can you explain it to me?",
                "that's a good question! how would you do it?",
            ]
            return random.choice(methods)
        else:
            fallbacks = [
                "that's a real puzzle! i'm not sure how to solve it yet. what's the answer?",
                "i love riddles, but i'm still figuring this one out. what do you think?",
                "my mind is still growing! can you tell me the answer?",
            ]
            return random.choice(fallbacks)




    def _ventral_path(self, ctx: CognitiveResponseContext) -> Optional[Tuple[str, str]]:
        """Ventral path: fast, no-reasoning response using graph + SurfaceRealizer.

        Maps to the brain's ventral stream (ATL → IFG):
        - ATL: concept retrieval from graph (spreading activation, 1-2 hops)
        - IFG: surface realization with diverse sentence patterns

        No web search, no multi-hop reasoning, no neural decoder.
        Returns None if it cannot produce a coherent response.
        """
        subject = ctx.subject
        assocs = ctx.associated_concepts

        # Route through syntactic pipeline (graph + surface realizer only)
        try:
            syntax_response = self._generate_with_decoder_and_syntax(ctx)
            if syntax_response and len(syntax_response) > 10 and not _is_word_salad(syntax_response):
                _words = syntax_response.lower().split()
                _unique_ratio = len(set(_words)) / max(1, len(_words))
                if _unique_ratio >= 0.35:
                    return (syntax_response, "fast_ventral")
        except Exception:
            pass

        # Check definition store for known concepts (ATL convergence zone)
        text_lower = ctx.raw_input.lower().strip()
        is_query_for_defn = self._is_informational_query(ctx.raw_input, subject)
        if is_query_for_defn and subject and subject.lower() in self._definitions:
            defn = self._definitions[subject.lower()]
            if defn and len(defn) > 10:
                defn_clean = defn.rstrip(" .!?")
                subj_disp = self._capitalize_subject(subject, ctx.raw_input)
                try:
                    if hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
                        frame = self.syntactic_assembly.bind_to_sentence(
                            subject=subj_disp, relation="semantic", target=defn_clean,
                            pos_map=getattr(self, '_concept_pos', {}),
                        )
                        disc_ctx = DiscourseState(sentence_index=0, discourse_type="explain",
                            total_sentences=1, free_energy=0.2,
                        )
                        response = self.surface_realizer.realize(frame=frame,
                            discourse_context=disc_ctx,
                            dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                            cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                        )
                        if not response or len(response) < 5:
                            response = f"{subj_disp} is {defn_clean}."
                    else:
                        response = f"{subj_disp} is {defn_clean}."
                except Exception:
                    response = f"{subj_disp} is {defn_clean}."
                return (response, "fast_ventral")

        return None

    def _dorsal_path(self, ctx: CognitiveResponseContext) -> Optional[Tuple[str, str]]:
        """Dorsal path: slow, reasoned response using web search + multi-hop chain walking.

        Maps to the brain's dorsal stream (Hippocampus → PFC → IFG):
        - Hippocampus: novel fact integration from web search
        - PFC: multi-hop reasoning and discourse planning
        - IFG: surface realization for final output

        Returns None if web search is unavailable or reasoning fails.
        """
        reasoned_res, reasoned_strat = self._reasoning_loop(ctx)
        if reasoned_res:
            return (reasoned_res, reasoned_strat)

        # Try syntactic pipeline as alternative reasoned response
        try:
            syntax_response = self._generate_with_decoder_and_syntax(ctx)
            if syntax_response and len(syntax_response) > 10 and not _is_word_salad(syntax_response):
                _words = syntax_response.lower().split()
                _unique_ratio = len(set(_words)) / max(1, len(_words))
                if _unique_ratio >= 0.35:
                    return (syntax_response, "dorsal_reasoned")
        except Exception:
            pass

        return None


    def _generate_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Dual-path response generation.

        Neuroscience basis:
        - Ventral Stream (fast): ATL → IFG — concept retrieval + surface realization.
          Handles known concepts, direct associations, cached patterns.
          NO web search, NO multi-hop reasoning, NO neural decoder.
        - Dorsal Stream (slow): Hippocampus → PFC → IFG — reasoning + web search.
          Handles novel concepts, multi-hop inference, web-augmented reasoning.

        PFC classifier detects question type and knowledge confidence to decide
        which path to activate. No fallback chain — each path succeeds or fails
        independently. If the chosen path fails, the other path is tried once.
        Graph fallback is the terminal case (always produces something).

        Removed from response path:
        - Neural decoder (bag-of-words, CE ~3.9, never produced coherent output)
        - Hippocampal recall (merged into reasoning loop for dorsal path)
        - Implicature detector (greetings handled by PFC routing)
        - Nested proposition parser (edge case, rarely triggered)
        - Quantity comparison (special case, rarely triggered)
        """
        # Step 1: PFC Classifier — detect question type and knowledge confidence
        qtype, _ = self.pfc_workspace.detect_question_type(ctx.raw_input, concept_pos=self._concept_pos)

        # Social/chitchat → always Ventral path
        if qtype in ("greeting", "wellbeing", "capability", "farewell"):
            ventral_res = self._ventral_path(ctx)
            if ventral_res:
                return ventral_res
            return self._graph_fallback_response(ctx)

        subject = ctx.subject
        assocs = ctx.associated_concepts

        # Determine knowledge confidence for this subject
        text_lower = ctx.raw_input.lower().strip()
        is_query_for_defn = self._is_informational_query(ctx.raw_input, subject)
        is_question = (ctx.raw_input.strip().endswith('?') or
                       any(w in text_lower for w in ["what", "who", "where", "when", "why", "how",
                                                      "define", "explain", "describe", "tell me about", "which"]))
        is_comparison = self._detect_comparison_concepts(ctx.raw_input) is not None
        is_reasoning_query = is_query_for_defn or is_question or is_comparison

        # Compute knowledge confidence: how well do we know this subject?
        subj_known = subject and (subject.lower() in self._concept_keywords or
                                  subject.lower() in self._concept_labels)
        has_assocs = len(assocs) > 0
        has_defn = subject and subject.lower() in self._definitions

        knowledge_confidence = 0.0
        if subj_known:
            knowledge_confidence += 0.3
        if has_assocs:
            knowledge_confidence += min(0.4, len(assocs) * 0.1)
        if has_defn:
            knowledge_confidence += 0.3
        knowledge_confidence = min(1.0, knowledge_confidence)

        # Step 2: Situation Model Path — DMN-guided fluid generation
        # Tries the situation model first when the subject is reasonably known and
        # the situation coherence is high enough for narrative generation
        situation_narrative = getattr(ctx, 'situation_narrative', {})
        situation_coherence = situation_narrative.get('coherence', 0.0) if situation_narrative else 0.0
        situation_active = len(situation_narrative.get('active_concepts', [])) if situation_narrative else 0
        try_situation_path = (knowledge_confidence > 0.2 or situation_coherence > 0.3) and situation_active > 1

        if try_situation_path:
            sm_res = self._generate_with_situation_model(ctx)
            if sm_res:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [trace]   SM path: {sm_res[1]} (coherence={situation_coherence:.2f})")
                return sm_res

        # Step 3: PFC Gate — route to Ventral or Dorsal path
        # Ventral path: known concepts (confidence > 0.4) or non-reasoning queries
        # Dorsal path: low confidence OR needs web search for reasoning
        needs_search = self._needs_web_search(subject) if subject else False

        if is_reasoning_query and (knowledge_confidence < 0.4 or needs_search):
            # DORSAL PATH: needs reasoning, unknown or partially known concept
            dorsal_res = self._dorsal_path(ctx)
            if dorsal_res:
                return dorsal_res
            # If dorsal fails, try ventral
            ventral_res = self._ventral_path(ctx)
            if ventral_res:
                return ventral_res
        else:
            # VENTRAL PATH: known concept or non-reasoning query
            ventral_res = self._ventral_path(ctx)
            if ventral_res:
                return ventral_res
            # If ventral fails, try dorsal for reasoning queries
            if is_reasoning_query:
                dorsal_res = self._dorsal_path(ctx)
                if dorsal_res:
                    return dorsal_res

        # Terminal: graph fallback (always produces something)
        return self._graph_fallback_response(ctx)


    def _capitalize_subject(self, subject: str, raw_input: str) -> str:
        """Capitalize subject correctly, preserving original case for proper nouns/acronyms."""
        if not subject:
            return ""
        words = raw_input.split()
        for w in words:
            clean_w = w.strip(".,!?\"'()[]{}*:;")
            if clean_w.lower() == subject.lower():
                return clean_w
        return subject.capitalize()


