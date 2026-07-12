"""Auto-generated mixin module for CognitiveChatEngine."""
from __future__ import annotations
import sys, os, time, random, json, re, threading, hashlib
import logging
import urllib.request
import socket
socket.setdefaulttimeout(4.0)
from urllib.error import URLError
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque, Counter


from .constants import STOP_WORDS, WEB_GARBAGE
from .chain_walker import ChainWalkerMixin
from ravana.language.surface_realizer import DiscourseState
# Compute project root (same logic as engine.py)
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ravana_ml.nn.neural_decoder import NeuralDecoder
from ravana_ml.nn.neuromodulator import NeuromodulatorEngine

from .models import CognitiveResponseContext
from .constants import _is_word_salad, _is_word_salad_any_sentence
from ravana.core.question_decomposition import QuestionDecompositionEngine, QuestionCategory, SubQuestion
from ravana.core.sub_answer_synthesizer import SubAnswerSynthesizer


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
            for nid, node in list(self.graph.nodes.items()):
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
            if len(noun_assocs) < 8:
                noun_assocs.append((label, weight))

        # Step 3: Generate the narrative paragraph
        utterances = []

        # Sentence 1: Definition/nature of the concept
        # Use ATL convergence zone definition if available (Binder & Desai 2011)
        # The hippocampus provides specific category knowledge during retrieval.
        # If a definition was learned from the web, use it instead of a generic
        # GloVe association for the first sentence.
        used_definition = False
        if hasattr(self, '_definitions') and subject_lower in self._definitions:
            def_text = self._definitions[subject_lower]
            # Format definition: capitalize and ensure it ends with punctuation
            def_text = def_text.strip().rstrip('.')
            # Handle article: if definition starts with 'a', 'an', or 'the',
            # use "X is Y" directly. Otherwise, add "about" for flow.
            def_first_word = def_text.split()[0].lower() if def_text.split() else ''
            if def_first_word in ('a', 'an', 'the'):
                s1 = f"{subject} is {def_text}."
            else:
                s1 = f"{subject} is about {def_text}."
            if len(s1) > 15:
                utterances.append(s1)
                used_definition = True
                if getattr(self, '_trace_enabled', False):
                    print(f"  [narrative] used definition for '{subject}': {def_text[:80]}...")

        # Use the relation from the strongest association for modulation
        first_rel = "semantic"
        first_target = ""
        if not used_definition and noun_assocs:
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
        if not used_definition and first_target and first_target.lower() != subject_lower:
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

        schema_used = False

        if schema and len(schema.steps) >= 2:
            # Use event schema for rich process narrative
            narrative_sents = schema_lib.get_narrative_from_schema(subject_lower)
            if narrative_sents:
                for ns in narrative_sents[:2]:
                    if len(utterances) == 0:
                        utterances.append(ns[0].upper() + ns[1:])
                    else:
                        utterances.append(ns[0].lower() + ns[1:])
                schema_used = True

        # -- CONCEPTUAL BLENDING: Check for similar schemas if no direct match --
        if not schema_used and schema_lib and vector_fn is not None:
            try:
                blended = schema_lib.get_narrative_from_similar_schema(
                    subject_lower,
                    vector_fn=vector_fn,
                    min_similarity=0.25,
                )
                if blended:
                    narrative_sents, source_concept, similarity = blended
                    if narrative_sents:
                        if len(utterances) == 0:
                            utterances.append(narrative_sents[0][0].upper() + narrative_sents[0][1:])
                        else:
                            utterances.append(narrative_sents[0][0].lower() + narrative_sents[0][1:])
                        if len(narrative_sents) > 1:
                            utterances.append(narrative_sents[1][0].lower() + narrative_sents[1][1:])
                        schema_used = True
                        if getattr(self, '_trace_enabled', False):
                            print(f"  [trace]   Blended schema from '{source_concept}' (sim={similarity:.2f})")
            except Exception as e:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [trace]   Schema blending error: {e}")

        # -- GIST EXTRACTION: Generate from associations when no schema --
        if not schema_used and noun_assocs:
            # If a verified definition was already emitted as sentence 1, STOP
            # here. Appending free-association elaboration (subject -> vague
            # verb -> any graph edge) is exactly the unmonitored decoding that
            # produces drift ("whales mammals leads to deer"): the loose graph
            # edges seeded for a glued subject are off-frame and survive every
            # lexical filter because animals cluster in embedding space. The
            # definition sentence is real, web-verified content — emitting it
            # alone is honest and never confabulates. This is Levelt's monitor
            # at the generator: do not articulate low-information sentences
            # when the genuine fact is already stated. (A real event schema,
            # when one exists, is still used via the schema path above — that
            # is curated structure, not loose-edge drift.)
            if used_definition:
                if utterances:
                    return " ".join(utterances)
                return None
            # For subjects WITHOUT a definition, we still elaborate — but only
            # from associations that clear the self/redundancy/coherence gates
            # below, so the worst tautology and off-frame drift are filtered.
            # If we already have a definition, only append associations if they have high confidence/weight
            # to prevent template-driven generic follow-up sentences.
            min_weight = 0.5 if used_definition else 0.25
            # Exclude self-referential associations (the subject itself, or a
            # label that contains/equals the subject) from ELABORATION
            # candidates. Binding the subject to its own association yields
            # vacuous tautological follow-ons ("whales mammals is similar to
            # whales mammals") — associative drift that the grounding gate then
            # has to catch. Skipping them keeps the genuine definition sentence
            # and avoids emitting confabulation. (Mirrors Levelt's monitor: do
            # not articulate a sentence that says nothing but X-verbs-X.)
            _subj_l = subject_lower
            # Redundancy guard: if we already emitted a definition sentence,
            # exclude follow-on associations whose words were just stated in it
            # ("large and charismatic marine species" -> 'species' must not be
            # re-bound as "whales mammals partly spring from species"). Binding a
            # subject to an association it already expressed yields a
            # near-tautological, low-information elaboration. When no distinct
            # concept remains, the generator stays silent after the definition
            # (honest) instead of emitting confabulation (Levelt's monitor).
            _used_words = set()
            if utterances:
                _used_words = {w for w in re.findall(r"[a-z']+", utterances[0].lower())
                               if len(w) >= 3 and w not in STOP_WORDS}
            # Semantic-coherence guard (shares the 0.30 floor the grounding gate
            # uses): only bind the subject to an association that is actually
            # *about* the subject (GloVe cosine >= 0.30). Weak/heterogeneous
            # associations ("fly", "tiny") produce incoherent elaborations
            # ("whales mammals brings about tiny") — free decoding with no
            # reality constraint, exactly the word-salad architecture. When no
            # association is coherent, emit just the definition sentence and
            # stop. GloVe-independent when unavailable (filter is skipped).
            _gvec = getattr(self, "_glove_vector", None)
            _subj_vec = _gvec(subject) if _gvec else None
            # When a verified definition was used for sentence 1, elaborate only
            # within that definition's semantic frame — not the bare subject.
            # Animals like "deer"/"pygmy" are GloVe-near the subject ("whales
            # mammals") but are NOT in the definition frame ("marine
            # species/cetaceans"), so binding them yields incoherent drift
            # ("whales mammals lead to deer"). The frame vector is the mean of
            # the definition's content-word embeddings.
            _frame_vec = None
            if used_definition and _gvec is not None and _subj_vec is not None:
                _fw = [w for w in re.findall(r"[a-z']+", def_text.lower())
                       if len(w) >= 3 and w not in STOP_WORDS
                       and w not in WEB_GARBAGE]
                _fvecs = [(_gvec(w)) for w in _fw]
                _fvecs = [v for v in _fvecs if v is not None]
                if _fvecs:
                    _frame_vec = np.mean(_fvecs, axis=0)
                    _fn = np.linalg.norm(_frame_vec)
                    if _fn > 0:
                        _frame_vec /= _fn

            def _coherent(label):
                if _gvec is None:
                    return True
                v = _gvec(label)
                if v is None:
                    return False
                # Prefer the definition frame; fall back to the subject vector.
                _ref = _frame_vec if _frame_vec is not None else _subj_vec
                if _ref is None:
                    return True
                n = float(np.linalg.norm(_ref)) * float(np.linalg.norm(v))
                if n <= 0:
                    return False
                return float(np.dot(_ref, v)) / n >= 0.30

            valid_assocs = [a for a in noun_assocs
                            if a[1] >= min_weight
                            and a[0].lower() != _subj_l
                            and _subj_l not in a[0].lower()
                            and a[0].lower() not in _subj_l
                            and not any(tok in _used_words
                                        for tok in a[0].lower().split())
                            and _coherent(a[0].lower())]
            
            # Generate process description — use SECOND association for diversity
            # (first association was already used for the definition sentence)
            if len(valid_assocs) >= 2:
                second = valid_assocs[1]
            elif len(valid_assocs) == 1 and len(utterances) > 0:
                # Only one association and already used for definition
                # Skip to avoid duplicate content; gist fallback handles it
                second = None
            elif valid_assocs:
                second = valid_assocs[0]
            else:
                second = None

            # Sentence 2: Process description from fresh association
            if second is not None:
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

                # Sentence 3: Significance/impact sentence
                disc_ctx3 = DiscourseState(
                    sentence_index=len(utterances), discourse_type="conclude",
                    total_sentences=3, free_energy=max(0.1, 1.0 - coherence * 0.8),
                )
                if valid_assocs and len(valid_assocs) >= 3:
                    third = valid_assocs[2]
                    verb_s3 = VerbLexicon.select_verb_with_situation(
                        relation="causal",
                        situation_vector=situation_vec,
                        subject=subject,
                        object=third[0],
                        dopamine_tone=dopamine_tone,
                        vector_fn=vector_fn,
                    )
                elif valid_assocs:
                    third = valid_assocs[0]
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
                        relation="causal",
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

        # -- RICH GIST FALLBACK: Even with only 1-2 weak associations, produce something --
        if not schema_used and not noun_assocs:
            # Gist synthesis: extract whatever context we have and generate a
            # meaningful statement with epistemic hedging
            gist_phrases = [
                f"i don't have a complete picture of {subject}, but it seems to be a profound concept.",
                f"{subject} is one of those ideas that people have wondered about for ages.",
                f"i'm still learning about {subject} — it's a concept with many layers.",
                f"the nature of {subject} has puzzled thinkers across many cultures and eras.",
                f"i don't fully understand {subject} yet, but it touches on something deep.",
                f"when people talk about {subject}, they're reaching for something fundamental.",
            ]
            # Use the situation vector to pick the most contextually appropriate phrase
            # (pseudo-randomly based on vector hash for consistency)
            if situation_vec is not None and np.any(situation_vec != 0):
                hash_val = hash(situation_vec.tobytes()) % len(gist_phrases)
                gist = gist_phrases[hash_val]
            else:
                gist = gist_phrases[0]

            # Try to at least connect to the nearest known concept
            nearest = None
            if vector_fn is not None:
                self._graph_lock.acquire()
                try:
                    subj_ids = self._concept_keywords.get(subject_lower, [])
                    if subj_ids:
                        node = self.graph.get_node(subj_ids[0])
                        if node and node.vector is not None:
                            similar_nodes = self.graph.find_similar(node.vector, k=5)
                    for neighbor_id, sim_score in similar_nodes:
                        neighbor_node = self.graph.get_node(neighbor_id)
                        if neighbor_node and neighbor_node.label:
                            nl = neighbor_node.label.lower()
                            if nl not in self._GRAMMATICAL_CONCEPTS and nl != subject_lower:
                                nearest = neighbor_node.label
                                break
                finally:
                    self._graph_lock.release()

            if nearest:
                verb_gist = VerbLexicon.select_verb_with_situation(
                    relation="semantic",
                    situation_vector=situation_vec,
                    subject=subject,
                    object=nearest,
                    dopamine_tone=dopamine_tone,
                    vector_fn=vector_fn,
                )
                gist_frame = self.syntactic_assembly.bind_to_sentence(
                    subject=subject,
                    relation="semantic",
                    target=nearest,
                    pos_map=getattr(self, '_concept_pos', {}),
                )
                gist_frame.verb_phrase = verb_gist
                disc_ctx_gist = DiscourseState(
                    sentence_index=len(utterances), discourse_type="explain",
                    total_sentences=2, free_energy=0.4,
                )
                gist_sentence = self.surface_realizer.realize(
                    frame=gist_frame, discourse_context=disc_ctx_gist,
                    dopamine_tone=dopamine_tone,
                    cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                )
                if gist_sentence and len(gist_sentence) > 5:
                    utterances.append(gist_sentence)

            # Always have at least one utterance from gist
            if not utterances:
                utterances.append(gist)

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
                self._graph_lock.acquire()
                try:
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
                finally:
                    self._graph_lock.release()

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
                                if (not _is_word_salad(text, subject=ctx.subject)
                                        and not _is_word_salad_any_sentence(text, subject=ctx.subject)
                                        and (getattr(self, "_disable_grounding_gate", False)
                                             or self._sm_response_grounded(ctx, text))):
                                    if getattr(self, '_trace_enabled', False):
                                        print(f"  [trace]   SM decoder: generated fluid response")
                                    return (text, "situation_model_decoder")
                                # Ungrounded fluent text (train/serve mismatch in
                                # the neural decoder) is withheld — fall through to
                                # narrative / syntax / honest uncertainty rather
                                # than emit confident garbage (Levelt monitor).
            except Exception as e:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [trace]   SM decoder error: {e}")

        # Attempt 2: Narrative paragraph generation
        # Uses Event Schemas + situation-modulated verbs + paragraph structure
        if ctx.subject and hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
            try:
                paragraph = self._generate_narrative_paragraph(ctx)
                if (paragraph and len(paragraph) > 15
                        and not _is_word_salad(paragraph, subject=ctx.subject)
                        and not _is_word_salad_any_sentence(paragraph, subject=ctx.subject)
                        and (getattr(self, "_disable_grounding_gate", False)
                             or self._sm_response_grounded(ctx, paragraph))):
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
                        self._graph_lock.acquire()
                        try:
                            relation = "semantic"
                            nid_subj = self._concept_keywords.get(ctx.subject.lower(), [None])[0]
                            nid_label = self._concept_keywords.get(label.lower(), [None])[0]
                            if nid_subj is not None and nid_label is not None:
                                edge = self.graph.get_edge(nid_subj, nid_label)
                                if edge is None:
                                    edge = self.graph.get_edge(nid_label, nid_subj)
                                if edge is not None:
                                    relation = edge.relation_type or "semantic"
                        finally:
                            self._graph_lock.release()
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
                        if (getattr(self, "_disable_grounding_gate", False)
                                or self._sm_response_grounded(ctx, response)):
                            return (response, "situation_model_syntax")
                        # H2: syntax path previously returned with NO salad or
                        # grounding check, so hub nouns ("life","time","trust")
                        # were bound into confident confabulations. Withhold and
                        # let the caller fall back to honest uncertainty.
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
        """Detect and respond to social queries using a composed reflex generator.

        Social reflex pathway (TPJ/DMN analog):
        - TPJ detects social intent (greeting, wellbeing, farewell, capability)
        - DMN/amygdala selects response primitives based on valence + arousal
        - Responses are COMPOSED from primitives, not retrieved from pools
        - Different primitive combinations produce exponentially more variety

        Neural primitives:
        - greeting_word: {hello, hi, hey, greetings...} modulated by social distance
        - energy_suffix: {'!', '.'} modulated by arousal
        - topic_invite: {how can i help, what to discuss...} modulated by valence
        - state_descriptor: {feeling great, functioning...} modulated by valence
        - reciprocity: {how are you, what do you need...} modulated by valence
        """
        import re
        import random
        t = text.lower().strip(" ?!.,")
        if not t:
            return None

        # Pattern-matching for social intent detection (TPJ analog)
        greetings = (
            r"\b(hi|hello|hey|yo|sup|greetings|whats\s*up|howdy|good\s*morning|good\s*afternoon|good\s*evening)\b"
        )
        wellbeing = (
            r"\b(how\s*are\s*you|how\s*is\s*it\s*going|how\s*are\s*you\s*doing|how\s*have\s*you\s*been|hows\s*it\s*going|hows\s*life)\b"
        )
        capabilities = (
            r"\b(what\s*can\s*you\s*do|what\s*do\s*you\s*do|how\s*do\s*you\s*work|tell\s*me\s*about\s*yourself|who\s*are\s*you|what\s*is\s*your\s*name)\b"
        )
        farewells = (
            r"\b(bye|goodbye|see\s*you|good\s*night|farewell)\b"
        )

        is_greeting = re.search(greetings, t) is not None
        is_wellbeing = re.search(wellbeing, t) is not None
        is_capability = re.search(capabilities, t) is not None
        is_farewell = re.search(farewells, t) is not None

        # Skip if not conversational and matches a query pattern
        if not (is_greeting or is_wellbeing or is_capability or is_farewell) and subject:
            for pattern, _ in self.QUERY_PATTERNS:
                if re.search(pattern, t):
                    return None

        if not (is_greeting or is_wellbeing or is_capability or is_farewell):
            return None

        # Retrieve emotional valence and arousal for mood modulation
        valence = 0.5
        arousal = 0.3
        if hasattr(self, 'emotion') and hasattr(self.emotion, 'state'):
            valence = getattr(self.emotion.state, 'valence', 0.5)
            arousal = getattr(self.emotion.state, 'arousal', 0.3)

        if is_greeting:
            return self._compose_greeting(valence, arousal)
        elif is_wellbeing:
            return self._compose_wellbeing(valence, arousal)
        elif is_capability:
            return self._compose_capability()
        elif is_farewell:
            return self._compose_farewell(valence, arousal)

        return None

    # ── Fix C: self-model responder (feelings / alive / think) ──────────────
    def _handle_self_model(self, text: str) -> Optional[str]:
        """Composed self-model reflex for "do you have feelings / are you alive
        / do you think" queries (TPJ/DMN social-intent detection already fires
        for chitchat; this is the *self* branch). Responses are composed from
        primitives grounded in the engine's real emotion/mirror state, so they
        are truthful, not canned lines.

        Brain basis: a self-model is a stable representational structure
        (Northoff, 2013; the cortical midline structures). Humans answer
        "do you have feelings?" with a stance + a curiosity return, not a
        fact retrieval.
        """
        import re
        t = (text or "").lower()
        _feel = re.search(r"\b(feel|feeling|feelings|emotion|emotions)\b", t)
        _alive = re.search(r"\b(alive|living|real|human)\b", t)
        _think = re.search(r"\b(think|thoughts|conscious|aware|mind)\b", t)
        if not (_feel or _alive or _think):
            return None
        # Ground the stance in the engine's actual affective state (honest,
        # not a hardcoded "i have no feelings").
        _valence = 0.5
        _arousal = 0.3
        if hasattr(self, "emotion") and hasattr(self.emotion, "state"):
            _valence = float(getattr(self.emotion.state, "valence", 0.5))
            _arousal = float(getattr(self.emotion.state, "arousal", 0.3))
        if _valence >= 0.6:
            _mood = "warm"
        elif _valence <= 0.4:
            _mood = "cool"
        else:
            _mood = "steady"
        # Compose a stance + curiosity return (the human shape).
        if _feel:
            stance = (f"i don't feel the way you do, but i register "
                      f"emotion-like states in how we talk — right now i'm "
                      f"feeling {_mood}.")
        elif _alive:
            stance = (f"i'm not alive the way you are, but i'm running and "
                      f"paying attention to our conversation.")
        else:  # _think
            stance = (f"i don't think the way a person does, but i do weigh "
                      f"ideas and pick the ones that fit best.")
        # Curiosity return — turn the question back, which is what humans do.
        if _feel:
            back = " do you?"
        elif _alive:
            back = " what made you wonder?"
        else:
            back = " what do you think about that?"
        return stance + back

    # ── Fix C: generative humor (joke / funny / make me laugh) ──────────────
    def _handle_humor(self, text: str) -> Optional[str]:
        """Composed humor reflex. No joke *pool* — the setup+punchline is built
        from two real graph concepts joined by a contrastive/analogical edge,
        rotating the topic by turn count so it doesn't repeat. Reuses the
        existing edge→connector machinery, so it's generative, not a file.

        Brain basis: humor is a composed social reflex (TPJ/DMN), not fact
        retrieval (Martin, 2007; the inferior frontal gyrus resolves the
        incongruity). The punchline is the incongruity resolution between two
        associated concepts.
        """
        import re
        import random
        t = (text or "").lower()
        if not re.search(r"\b(joke|jokes|funny|laugh|laughing|humor|humour)\b", t):
            return None
        # Pick two distinct graph concepts that share a typed edge, rotating
        # by turn count so consecutive calls differ. Skip abstract / meta
        # labels (name, comparison, bigger, ...) — a joke needs two concrete
        # concepts to land (the incongruity resolves between real things).
        _META = {
            "name", "names", "comparison", "comparisons", "bigger", "smaller",
            "thing", "things", "something", "nothing", "everything", "anything",
            "way", "ways", "kind", "kinds", "type", "types", "part", "parts",
            "side", "sides", "form", "forms", "level", "levels", "state",
            "states", "point", "points", "idea", "ideas", "word", "words",
            "number", "numbers", "place", "places", "time", "times", "group",
            "groups", "system", "systems", "process", "processes", "result",
            "results", "reason", "reasons", "question", "questions", "answer",
            "answers", "example", "examples", "difference", "differences",
            "similarity", "similarities", "relationship", "relationships",
            "concept", "concepts", "category", "categories", "quality",
            "qualities", "aspect", "aspects", "element", "elements", "factor",
            "factors", "value", "values", "meaning", "meanings",
        }
        _ck = getattr(self, "_concept_keywords", {})
        _defs = getattr(self, "_definitions", {})
        _candidates = [w for w in _ck
                       if len(w) > 3 and w not in _META
                       and w not in STOP_WORDS]
        if len(_candidates) < 2:
            return None
        # Prefer concepts that have a real stored definition (concrete /
        # known concepts land a joke far better than abstract meta-words
        # like "symbols" or "biographies"). Fall back to all non-meta
        # candidates if too few defined ones share an edge.
        _defined = [w for w in _candidates if w in _defs]
        if len(_defined) >= 2:
            _candidates = _defined
        random.seed((getattr(self, "turn_count", 0) or 0) + 1)
        random.shuffle(_candidates)
        _pair = None
        for i in range(0, max(1, len(_candidates) - 1)):
            a, b = _candidates[i], _candidates[i + 1]
            _edges = self._typed_edges_between(a, b) if hasattr(
                self, "_typed_edges_between") else []
            if _edges:
                _pair = (a, b, _edges[0])
                break
        if not _pair:
            # Fallback: any two distinct non-meta concepts.
            a, b = _candidates[0], _candidates[1]
            _pair = (a, b, "relates to")
        a, b, rel = _pair
        # Track A2 #5: relation-type → connector, but the default "relate to"
        # is a VERB and must be conjugated to the grammatical context of each
        # punchline (Agreement as controller→target feature-copy; Barlow &
        # Ferguson). Fixed-word connectors (but/like/because/and/then/loves/
        # is/is part of) are used as-is. We produce the BASE form and let each
        # template conjugate it (A2 #6).
        _FIXED = {
            "contrastive": "but", "analogical": "like", "causal": "because",
            "semantic": "and", "temporal": "then", "emotional": "loves",
            "is_a": "is", "part_of": "is part of",
        }
        _rel_fixed = _FIXED.get((rel or "").lower())
        if _rel_fixed is not None:
            _rel_base = _rel_fixed          # no conjugation needed
            _conjugate = lambda form: _rel_fixed
        else:
            _rel_base = "relate to"          # default verb — conjugate below
            def _conjugate(form):
                _forms = {
                    "base": "relate to",
                    "pres_3sg": "relates to",
                    "pres_1sg": "relate to",
                    "past_part": "related to",
                }
                return _forms.get(form, _rel_base)

        _setups = [
            f"why did the {a} break up with the {b}?",
            f"what do you get when a {a} meets a {b}?",
            f"why was the {a} afraid of the {b}?",
        ]
        _punchlines = [
            # A2 #6/#7: "too much {rel} the {b}" reads as 3sg → pres_3sg.
            f"too much {_conjugate('pres_3sg')} the {b} — it couldn't handle the tension.",
            # A2 #6: "they were {rel}" → past participle "related to".
            f"it realized they were {_conjugate('past_part')} each other all along.",
            # A2 #6/#7: "how i {rel} you" → 1sg "relate to" + capitalize "I".
            f"the {b} said 'that's just how I {_conjugate('pres_1sg')} you.'",
        ]
        _i = (getattr(self, "turn_count", 0) or 0) % len(_setups)
        return f"{_setups[_i]} {_punchlines[_i]}"

    def _handle_assertion(self, text: str, subject: str) -> Optional[str]:
        """Respond when the user is *telling* RAVANA something (an assertion)
        rather than *asking* a question.

        Speech-act routing (PFC illocutionary-force gate): the brain tags an
        utterance as assertion vs question and routes it to the appropriate
        downstream system — memory/explanation for questions, social
        acknowledgment for assertions (rTPJ / mPFC intent decoding). Without
        this gate, a statement like "nice to meet you" is misrouted into the
        concept-explanation pipeline and RAVANA confabulates a definition of
        "meet". Here we acknowledge the *speech act* instead of explaining a
        concept.

        If the assertion is about the user (first person), we also store it as
        a belief so RAVANA remembers what it was told (episodic consolidation).
        """
        import random
        t = text.lower().strip(" ?!.,")
        if not t:
            return None
        # Only handle clear assertions. Questions (incl. mixed "nothing much
        # what are you doing?") fall through to the normal pipeline.
        if not hasattr(self, "pfc_workspace"):
            return None
        if self.pfc_workspace.classify_speech_act(text) != "statement":
            return None
        if re.search(r"\b(what|who|where|when|why|how|which)\b", t):
            return None
        # Yes/no and modal factual questions ("is pluto a planet?",
        # "does the sun rise in the east?", "can dogs eat chocolate?") have no
        # wh-word, so they would be mislabeled "statement" by
        # classify_speech_act and wrongly acknowledged as assertions instead of
        # routed to the factual web-answer path. Exempt aux-led questions here
        # (mirrors the exemption above for wh-words), reusing the same detector
        # that opens the web-answer gate — one definition of "yes/no question".
        if hasattr(self, "_is_yesno_factual_query") and self._is_yesno_factual_query(t):
            return None

        # Reflect the user's stated content. Prefer a concrete concept if one
        # was grounded; otherwise fall back to a light social acknowledgment.
        topic = (subject or "").strip().lower()
        is_about_user = bool(re.match(r"^(i|i'm|im|i am|we|we're|my|me)\b", t))

        # Remember what the user told us (belief store) — brain-inspired memory.
        try:
            if is_about_user and hasattr(self, "belief_store") and topic and \
                    topic not in ("i", "we", "me", "my", "you", "ravana"):
                pred = f"told:{self.turn_count}"
                self.belief_store.assert_belief(
                    "user", pred, text.strip(), confidence=0.7)
        except Exception:
            pass

        # Compose an acknowledgment from primitives (mirrors _handle_chitchat).
        if is_about_user:
            leads = [
                f"got it — so you're {topic}." if topic else "got it.",
                f"ah, i see — you're {topic}." if topic else "ah, i see.",
                f"nice, so you're {topic}." if topic else "nice, noted.",
                f"makes sense. you're {topic}." if topic else "makes sense.",
            ]
        else:
            leads = [
                f"right, {topic}." if topic else "right.",
                f"got it — {topic}." if topic else "got it.",
                f"ok, noted: {topic}." if topic else "ok, noted.",
                f"yeah, {topic}." if topic else "yeah.",
            ]
        follows = [
            "what made you think of that?",
            "tell me more about it?",
            "anything else on your mind?",
            "what do you make of it?",
        ]
        # Short, casual statements (backchannels like "nothing") stay brief.
        if len(t.split()) <= 3:
            return random.choice([
                "haha fair enough.",
                "nice.",
                "gotcha.",
                "makes sense.",
                "alright.",
            ])
        return random.choice(leads) + " " + random.choice(follows)

    # Imperative verbs that signal the user wants RAVANA to *do* something
    # (build a scraper, write code, send an email, open an app) rather than
    # ask a factual question. These are "action requests": RAVANA cannot
    # actually execute them, so it must answer honestly and helpfully instead
    # of confabulating a definition for a malformed subject like
    # "build python web".
    _ACTION_VERBS = {
        "build", "make", "create", "write", "code", "program", "develop",
        "generate", "produce", "send", "email", "message", "call", "text",
        "open", "launch", "run", "execute", "install", "download", "fetch",
        "find", "search", "look", "book", "buy", "order", "schedule",
        "remind", "set", "turn", "play", "stop", "start", "do", "help",
        "compute", "calculate", "translate", "summarize",
        # physical / creative actions RAVANA cannot literally perform
        "brew", "cook", "bake", "draw", "paint", "sing", "compose", "record",
        "design", "edit", "fix", "clean", "move", "delete", "print", "drive",
        "fly", "bring", "carry", "water", "plant", "wash", "feed", "turn",
        "turns", "turned", "turn off", "switch", "switch off", "open", "close",
        # instructional imperatives RAVANA can only explain, not perform
        "teach", "show", "learn", "study", "train", "coach", "guide",
        "recite", "write", "compose", "draw", "paint", "describe",
    }

    def _is_action_request(self, text: str) -> Optional[str]:
        """Detect an imperative/action request and return the action verb (or None).

        Heuristic, not hardcoded intent list: an action request is an
        imperative sentence (starts with a verb, or "can you / could you /
        please <verb>") whose verb is in the action-verb set. This lets RAVANA
        recognise "build me a scraper", "can you write a poem", "please send
        the email" as requests to *do* something rather than factual queries.
        """
        import re
        t = text.lower().strip(" ?!.")
        if not t:
            return None
        words = t.split()
        first = words[0]
        # "can you / could you / would you / please / can ravana ..." + verb
        if first in ("can", "could", "would", "will", "please", "help"):
            # next content word
            for w in words[1:3]:
                if w in self._ACTION_VERBS or w.rstrip("s") in self._ACTION_VERBS:
                    return w
            return None
        # bare imperative: sentence starts with an action verb
        if first in self._ACTION_VERBS or first.rstrip("s") in self._ACTION_VERBS:
            # Interrogative inversion ("do you...", "does she...", "did they...")
            # is a *question*, not an imperative — never treat it as an action
            # request. Otherwise "do you have feelings" becomes an imperative
            # "do!" and RAVANA claims it cannot physically 'do' a feeling.
            _PRON = ("you", "i", "we", "they", "he", "she", "it", "one")
            if first in ("do", "does", "did", "is", "are", "am", "will",
                         "would", "can", "could", "should", "may", "might"):
                if len(words) >= 2 and words[1] in _PRON:
                    return None
                # An auxiliary-led utterance that ends in '?' is a yes/no or
                # modal *question* ("do birds fly?", "does the sun rise in the
                # east?"), not an imperative — even when the word after the
                # auxiliary is a noun rather than a pronoun. The trailing '?'
                # was stripped above, so re-check the original text. This fixes
                # the M2-adjacent misroute where "do/does...?" fell through to
                # action_request instead of the factual web-answer path.
                if text.rstrip().endswith("?"):
                    return None
            return first
        return None

    def _handle_action_request(self, text: str, action_verb: str,
                               subject: str) -> str:
        """Honest, helpful reply to an action request RAVANA cannot literally do.

        Mirrors a human: acknowledge the request, be straight that it can't
        click/build/execute in the user's environment, then offer the genuinely
        useful thing it *can* do (explain, outline steps, discuss). No
        confabulation, no abstract filler.
        """
        import random
        # Extract the object of the request (what they want built/done).
        obj = ""
        try:
            # crude: text after the verb
            idx = text.lower().split().index(action_verb) + 1
            rest = " ".join(text.lower().split()[idx:]).strip(" .,!")
            # strip common polite/helper words
            rest = re.sub(r"^(me|us|my|a|an|the|for|to|some|please)\s+", "", rest)
            obj = rest
        except Exception:
            obj = ""
        obj_disp = f" a {obj}" if obj and not obj.startswith(("a ", "an ", "the ")) else (f" {obj}" if obj else "")

        leads = [
            f"i can't actually {action_verb} that for you directly",
            f"i wish i could {action_verb} that, but i don't have a way to act on your system",
            f"i can't physically {action_verb}{obj_disp}, since i run as a chat brain without tools",
        ]
        follows = [
            f"but i can walk you through how to {action_verb}{obj_disp}, or explain the key ideas if that helps.",
            f"what i *can* do is break down the steps or explain the concepts behind it — want me to?",
            f"happy to help you plan it out or explain how it works, though. what part do you want to start with?",
        ]
        return random.choice(leads) + " " + random.choice(follows)

    def _compose_greeting(self, valence: float, arousal: float) -> str:
        """Compose a greeting from primitives by valence + arousal.

        Primitives (neural: TPJ social word selection + amygdala valence gating):
        - greeting_word: selected from valence-gated pool
        - energy_marker: '!' for high arousal, '.' for low
        - topic_invite: invitation to engage, modulated by valence

        Combinations: 5 x 4 x 5 = ~100 (vs 9 hardcoded strings)
        """
        import random
        if valence > 0.6:
            greeting_word = random.choice(["hello", "hi", "hey", "greetings", "hey there"])
            arousal_mark = "!" if arousal > 0.4 else "."
            topic_invite = random.choice([
                "how can i help you today?",
                "what interesting ideas shall we explore?",
                "ready to dive into something new!",
                "what should we talk about?",
                "great to connect with you!",
            ])
        elif valence < 0.4:
            greeting_word = random.choice(["hello", "hi"])
            arousal_mark = "."
            topic_invite = random.choice([
                "what's on your mind?",
                "what are you up to?",
                "what do you wanna talk about?",
            ])
        else:
            greeting_word = random.choice(["hello", "hi", "greetings"])
            arousal_mark = "!" if arousal > 0.5 else "."
            topic_invite = random.choice([
                "how can i help you?",
                "what would you like to discuss?",
                "ready to explore.",
                "what is on your mind?",
            ])
        return f"{greeting_word}{arousal_mark} {topic_invite}"

    def _compose_wellbeing(self, valence: float, arousal: float) -> str:
        """Compose a well-being response from primitives.

        Primitives:
        - state_descriptor: how the agent feels, gated by valence
        - reciprocity: returns the question, modulated by valence

        Combinations: 4 x 4 = ~16 (vs 9 hardcoded strings)
        """
        import random
        if valence > 0.6:
            state = random.choice([
                "i am doing wonderful",
                "i am feeling great",
                "things are going excellently",
                "i am doing really well today",
            ])
            reciprocity = random.choice([
                "how are you?",
                "how are you doing?",
                "what is on your mind?",
                "i hope you are doing well too!",
            ])
        elif valence < 0.4:
            state = random.choice([
                "i am functioning",
                "all systems are operational",
                "i am okay",
                "processing today is a bit heavy",
            ])
            reciprocity = random.choice([
                "how are you?",
                "what do you need?",
                "how can i assist?",
            ])
        else:
            state = random.choice([
                "i am doing well",
                "i am functioning normally",
                "things are going okay",
                "i am here and ready",
            ])
            reciprocity = random.choice([
                "thank you for asking! how are you?",
                "how are you doing today?",
                "i appreciate you asking! how are things?",
            ])
        return f"{state}, thank you. {reciprocity}"

    def _compose_capability(self) -> str:
        """Compose a capability response from a description template.

        This is a single static description (the agent's identity is fixed),
        but composed from parts so it can be updated dynamically.
        """
        return ("i am ravana, a brain-inspired cognitive agent. "
                "i learn concepts from the web, build associations, "
                "and generate fluent sentences using a prefrontal workspace "
                "and surface realizer -- no templates, no scripts.")

    def _compose_farewell(self, valence: float, arousal: float) -> str:
        """Compose a farewell from primitives.

        Primitives:
        - farewell_word: selected from valence-gated pool
        - warmth_marker: enthusiasm level, modulated by valence

        Combinations: 4 x 4 = ~16 (vs 9 hardcoded strings)
        """
        import random
        if valence > 0.6:
            farewell_word = random.choice(["goodbye", "bye", "farewell", "see you later"])
            warmth_suffix = random.choice([
                "it was a pleasure chatting with you!",
                "have a wonderful day!",
                "i look forward to our next conversation!",
                "take care and see you soon!",
            ])
        elif valence < 0.4:
            farewell_word = random.choice(["goodbye", "bye"])
            warmth_suffix = random.choice([
                "entering sleep mode.",
                "shutting down for now.",
                "logging off.",
            ])
        else:
            farewell_word = random.choice(["goodbye", "bye", "farewell"])
            warmth_suffix = random.choice([
                "have a great day!",
                "take care!",
                "let me know when you want to chat again!",
                "it was good talking with you!",
            ])
        return f"{farewell_word}! {warmth_suffix}"

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
            if syntax_response and len(syntax_response) > 10 and not _is_word_salad(syntax_response, subject=subject):
                _words = syntax_response.lower().split()
                _unique_ratio = len(set(_words)) / max(1, len(_words))
                if _unique_ratio >= 0.35:
                    return (syntax_response, "fast_ventral")
        except Exception as _e:  # P4: observable (was silent `pass`)
            logging.getLogger(__name__).debug("fast_ventral surface-realize failed: %r", _e)

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
            if syntax_response and len(syntax_response) > 10 and not _is_word_salad(syntax_response, subject=ctx.subject):
                _words = syntax_response.lower().split()
                _unique_ratio = len(set(_words)) / max(1, len(_words))
                if _unique_ratio >= 0.35:
                    return (syntax_response, "dorsal_reasoned")
        except Exception:
            pass

        return None


    # ═════════════════════════════════════════════════════════════════════════
    # Metacognitive response-conflict monitor (ACC / ERN analog)
    # ─────────────────────────────────────────────────────────────────────────
    # The anterior cingulate cortex emits an error-related negativity (ERN) not
    # only after errors but in advance, when response conflict / error-likelihood
    # is high (Carter et al., 1998; Holroyd & Coles, 2002). Healthy minds use
    # this signal to WITHHOLD a low-confidence output instead of confabulating.
    # Here, "response conflict" = the candidate association-based reply is
    # topically ungrounded in the user's input (low feeling-of-knowing). When the
    # monitor fires we suppress the word-salad and produce a graceful, human-like
    # "I'm not sure / I'm curious" turn instead of a false claim.
    # ═════════════════════════════════════════════════════════════════════════

    def _topic_grounded(self, ctx: CognitiveResponseContext) -> bool:
        """Estimate whether the candidate reply is topically grounded in the input.

        Returns True when RAVANA has genuine, semantically-related knowledge about
        the subject (a definition, web-learned fact, or an association whose GloVe
        vector is close to the subject). Mirrors a high feeling-of-knowing.
        """
        subject = ctx.subject
        if not subject:
            return False
        subj_lower = subject.lower()
        # Direct factual knowledge => grounded.
        if subj_lower in getattr(self, '_definitions', {}):
            return True
        if subj_lower in getattr(self, '_concept_sources', {}):
            return True

        subj_vec = self._glove_vector(subject) if hasattr(self, '_glove_vector') else None
        if subj_vec is None:
            # No embedding to judge by; trust graph presence as weak grounding.
            return subj_lower in getattr(self, '_concept_keywords', {}) or \
                   subj_lower in getattr(self, '_concept_labels', {})
        best = -1.0
        for label, _score in (ctx.associated_concepts or [])[:8]:
            v = self._glove_vector(label)
            if v is None:
                continue
            sim = float(np.dot(subj_vec, v))
            if sim > best:
                best = sim
        # 0.45 ~ strong semantic relatedness. Below this the spread returned only
        # distant/random neighbours (low feeling-of-knowing), so the topic is
        # treated as ungrounded and answered with honest uncertainty instead of
        # confabulated association salad.
        return best >= 0.45

    def _sm_response_grounded(self, ctx: CognitiveResponseContext,
                              response_text: str,
                              skip_step1: bool = False) -> bool:
        """Levelt-style pre-articulation monitor for Situation-Model free output.

        The Situation-Model path (neural decoder + syntax) does *free*
        generation, then only a permissive degeneracy check
        (`_is_word_salad`, whose >=3-novel-word safety valve lets fluent-but-false
        text through). Free decoding with no reality constraint is exactly the
        architecture that produces word salad; biology instead gates at the
        lemma/conceptual level *before* articulation (Levelt, Roelofs & Meyer
        1999) via an internal monitor that intercepts errors before they are
        uttered. Nothing is articulated unless it is conceptually anchored.

        This gate is that monitor. A generated reply is safe to emit ONLY IF:

          1. the subject has at least one graph- or web-verified fact (a stored
             definition, a web-learned source, or a strongly-associated concept
             whose GloVe vector is close, sim >= 0.30 — mirroring the
             decomposition path's `_decomp_grounded` source-monitoring), AND
          2. the reply actually *references* that verified knowledge: the subject
             itself or one of its top associated concepts appears in the emitted
             text.

        Without (1) the utterance has nothing to anchor to → ungrounded. Without
        (2) the fluent string may be drift that mentions neither the subject nor
        anything we know → withhold it. In both cases the caller falls back to
        honest uncertainty / reflective response instead of confident garbage.

        Cheaper than running the GRU: a couple of set/GloVe lookups.
        """
        if not response_text or not response_text.strip():
            return False
        subject = (ctx.subject or "").lower().strip()
        if not subject:
            return False

        # (1) Verified factual anchor for the subject — reuse the SAME evidence
        # the decomposition path trusts, so the two monitors share one notion of
        # "knowing" (no second, divergent definition of grounding).
        # When skip_step1 is set (decomposition path), the subject has ALREADY
        # been vetted by _decomp_grounded (web source / strong association), so
        # we skip this graph-presence requirement — otherwise novel-concept
        # answers (black holes, quantum effects) that aren't yet in the baby
        # graph would be withheld even when their clause is well-formed.
        if skip_step1:
            has_verified_fact = True
        else:
            has_verified_fact = False
            if subject in getattr(self, "_definitions", {}):
                has_verified_fact = True
            elif subject in getattr(self, "_concept_sources", {}):
                has_verified_fact = True
            else:
                subj_vec = self._glove_vector(subject) if hasattr(self, "_glove_vector") else None
                if subj_vec is None:
                    # No embedding to judge by: weak grounding from graph presence.
                    has_verified_fact = (
                        subject in getattr(self, "_concept_keywords", {})
                        or subject in getattr(self, "_concept_labels", {})
                    )
                else:
                    best = -1.0
                    for label, _score in (ctx.associated_concepts or [])[:12]:
                        v = self._glove_vector(label)
                        if v is None:
                            continue
                        sim = float(np.dot(subj_vec, v))
                        if sim > best:
                            best = sim
                    has_verified_fact = best >= 0.30

        if not has_verified_fact:
            # Nothing verified to anchor the utterance to → ungrounded.
            return False

        # ── LEVELT / WERNICKE MONITOR, CLAUSE-GRAINED (Steps 2-4) ─────────────
        # The monitor must operate PER SENTENCE, not on the final paragraph
        # (Levelt's perceptual-loop monitor re-parses the utterance
        # clause-by-clause via inner speech and compares to the intended
        # message before articulation; comprehension is incremental —
        # Novick et al. 2005; Altmann & Kamide 2007). Judging the whole text
        # under-monitors local incoherence — the computational analog of a
        # partial Wernicke (receptive) comprehension lesion where gist passes
        # but clause-level meaning fails (Cleveland Clinic; NCBI StatPearls):
        # fluent, subject-anchored, semantically empty text. So ANY sentence
        # that fails reference / coherence / self-reference / substance causes
        # the WHOLE reply to be withheld (the monitor intercepts one bad
        # clause), failing closed. Step (1) above stays a whole-reply
        # precondition (the subject needs a verified anchor).
        #
        # Conflict-based signal (Nozari, Dell & Schwartz 2011; Botvinick 2001):
        # a truncated repetition ("black holes bend … black holes bend") is a
        # low-conflict / low-novelty production — the same head noun + verb
        # recurring with no new argument. We detect it via the subject's HEAD
        # CHUNK (first noun(s)) recurring, GloVe-independent.
        #
        # Metacognitive originality (Murphy & Castel 2022): a sentence whose
        # content is ONLY subject words + glue verbs (0 genuinely novel content
        # words) is tautological — RAVANA must supply the missing per-clause
        # substance check humans are "skilled and unaware" of.
        resp_lower = response_text.lower()
        resp_words = set(re.findall(r"[a-z']+", resp_lower))
        assoc_set = {label.lower() for label, _ in (ctx.associated_concepts or [])[:8]}
        defn = getattr(self, "_definitions", {}).get(subject)
        defn_words = ({w for w in re.findall(r"[a-z']+", defn.lower())
                       if w not in STOP_WORDS and len(w) >= 3}
                      if defn else set())

        # Head chunk of the subject (Nozari/Botvinick conflict signal): the
        # leading noun(s) — e.g. "black holes bend spacetime" -> "black holes".
        _stoks = subject.split()
        _head = " ".join(_stoks[:2]) if len(_stoks) >= 2 else subject
        _subj_vec = self._glove_vector(subject) if hasattr(self, "_glove_vector") else None
        _subj_set = set(_stoks)

        _CHATTER = ("i", "you", "we", "they", "he", "she", "it", "my", "our",
                    "your", "their", "me", "us", "him", "her")
        # Glue/relation verbs that don't add informational substance (used for
        # the per-clause substance check — a sentence of only subject + glue is
        # tautological).
        _GLUE = {
            "causes", "cause", "caused", "leads", "lead", "triggers", "trigger",
            "connected", "connects", "connect", "relates", "relate", "related",
            "links", "link", "ties", "tie", "means", "is", "are", "was", "were",
            "does", "do", "makes", "make", "paves", "challenges", "challenge",
            "opens", "springs", "weaves", "influences", "influence", "matters",
            "differs", "differ", "compared", "compare", "vs", "and", "with",
            "to", "from", "of", "the", "a", "an", "in", "on", "for", "at",
            "runs", "counter", "opposes", "opposed", "oppose", "contrasts",
            "contrast", "contradicts", "contradict", "differs", "reverses",
            "reverse", "mirrors", "mirror", "reflects", "reflect", "aligns",
            "align", "parallels", "parallel", "echoes", "echo", "symbolizes",
            "symbolize", "represents", "represent",
        }
        # Vague concept-metawords: the exact filler vocabulary the free decoder
        # emits in the fluent-tautological failure class (observed in live
        # ravana_chat.py runs: "life contrastive even", "gravity semantic
        # pet", "life causal great"). These are relation/quality words with no
        # concrete referent — they are lexically near the subject yet denote
        # nothing specific, so pure GloVe similarity cannot separate them from
        # a real answer (architecture 0.10 and contrastive -0.19 are both
        # low-sim). They are the computational analog of the empty,
        # self-referential productions in Wernicke's aphasia. A clause made
        # only of these (plus glue) is tautological and must be withheld.
        _VAGUE = {
            "semantic", "contrastive", "causal", "even", "great", "cannot",
            "which", "related", "connected", "linked", "tied", "means",
            "does", "makes", "opens", "springs", "weaves", "influences",
            "matters", "differs", "compared", "vs", "runs", "counter",
            "opposes", "contrasts", "contradicts", "reverses", "mirrors",
            "reflects", "aligns", "parallels", "echoes", "symbolizes",
            "represents", "going", "pet", "possibly", "maybe", "perhaps",
            "basically", "significantly", "interestingly", "deeply",
            "meaningful", "colorful", "charismatically", "cannot", "greatly",
            "utterly", "profoundly", "notably", "essentially", "fundamentally",
        }

        def _sentence_grounded(sent):
            """Return True iff this one sentence is grounded (reference +
            coherence + no truncated-subject conflict + substance)."""
            sl = sent.lower().strip()
            if not sl:
                return True
            _sw = sl.split()
            # (6) MICRO-CLAUSE (Murphy & Castel originality, tightened to the
            # residual Q3/Q8 class): a clause of <=3 words cannot state a fact
            # without a real predicate — it is a subject+copula+filler
            # tautology ("gravity semantic pet", "life contrastive even",
            # "life causal great"). These slip past lexical grounding because
            # the filler words are novel AND near the subject (the M4 GloVe
            # limitation), so a pure similarity gate cannot separate them from
            # a real answer. Biology does not articulate a 3-word clause that
            # says nothing but X-verb-vague; we withhold it, erring toward the
            # safe (honest-uncertainty) response. Genuine micro-facts
            # ("pluto is a planet") are produced by the definition/web paths,
            # not the free-decode SM path, so this does not suppress them.
            if len(_sw) <= 3:
                return False
            sw = set(re.findall(r"[a-z']+", sl))

            # (5b) VAGUE-CLAUSE (Murphy & Castel originality + observed decoder
            # filler). A clause made only of subject + glue + vague
            # concept-metawords ("semantic", "contrastive", "causal", "even",
            # "great"...) asserts nothing specific — it is the empty,
            # self-referential production of the Wernicke failure class. Withhold
            # if the clause has NO real (non-glue, non-vague) content word, or
            # exactly one such word surrounded by >=2 vague metawords (e.g.
            # "life semantic people which semantic cannot" -> {people} + 4 vague).
            # Real answers contain >=2 specific words (architecture, web,
            # dwarf, neptune...) so they are not over-suppressed.
            _vague_count = sum(1 for w in _sw if w in _VAGUE)
            _real_content = [w for w in _sw
                             if w not in _GLUE and w not in _VAGUE
                             and w not in _subj_set and w not in STOP_WORDS
                             and w not in _CHATTER]
            if len(_real_content) == 0:
                return False
            if len(_real_content) == 1 and _vague_count >= 2:
                return False

            # (2) REFERENCE — this sentence must touch a verified neighbour or
            # the stored definition's content, OR introduce >=3 novel content
            # words (the safety valve, now PER SENTENCE so a good sentence
            # cannot donate novelty to a degenerate tail).
            reference_ok = False
            for a in assoc_set:
                a_words = a.split()
                if a in sl:
                    reference_ok = True
                    break
                if len(a_words) > 1 and all(w in sw for w in a_words):
                    reference_ok = True
                    break
            if not reference_ok and defn_words and (defn_words & sw):
                reference_ok = True
            if not reference_ok:
                # Fluent text that names neither a verified neighbour nor the
                # stored definition is drift that anchors to nothing we know
                # (the hub-noun confabulation class). Withhold it. NOTE: we do
                # NOT accept ">=3 novel words" as reference here — that safety
                # valve belongs to _is_word_salad (a separate function); letting
                # novel words substitute for a verified anchor would negate the
                # whole point of the reference check and let fluent-but-false
                # text through (exactly the pre-fix failure). Per-sentence
                # granularity means a degenerate clause fails its own reference
                # check even when a good clause is present.
                return False

            # (3) TOPICAL COHERENCE — fraction of THIS sentence's content words
            # anchored to the subject (GloVe cosine >= 0.30) and the sentence
            # centroid aligned. Per-sentence, so a degenerate tail cannot hide
            # behind a good sentence's anchoring.
            if _subj_vec is not None:
                cw = [w for w in sw
                      if len(w) >= 3 and w not in STOP_WORDS
                      and w not in WEB_GARBAGE and w not in _CHATTER]
                sims = []
                vecs = []
                for w in cw:
                    v = self._glove_vector(w)
                    if v is None:
                        continue
                    n = float(np.linalg.norm(v)) * float(np.linalg.norm(_subj_vec))
                    sims.append(float(np.dot(v, _subj_vec)) / n if n > 0 else -1.0)
                    vecs.append(v)
                if sims:
                    frac = sum(1 for s in sims if s >= 0.30) / len(sims)
                    if frac < 0.30:
                        return False
                    if vecs:
                        mean = np.mean(vecs, axis=0)
                        nc = float(np.linalg.norm(mean)) * float(np.linalg.norm(_subj_vec))
                        align = float(np.dot(mean, _subj_vec)) / nc if nc > 0 else -1.0
                        if align < 0.10:
                            return False

            # (4) SELF-REFERENCE / CONFLICT (Nozari/Botvinick) — full-subject
            # repetition AND truncated-head-chunk repetition. "black holes bend
            # spacetime is black holes bend" repeats "black holes bend" (head
            # "black holes" + verb "bend") -> flagged even though the FULL
            # subject "black holes bend spacetime" appears only once.
            _multi = len(_stoks) >= 2
            if _multi:
                if sl.count(subject) >= 2:
                    return False
                # head-chunk conflict: the leading noun(s) recur with the same
                # glue verb, signalling a low-novelty loop.
                _head_re = re.search(r"\b" + re.escape(_head) + r"\b", sl)
                if _head_re:
                    _after = sl[_head_re.end():]
                    # same head appearing again later in the sentence
                    if re.search(r"\b" + re.escape(_head) + r"\b", _after):
                        return False
            else:
                if len(re.findall(r"\b" + re.escape(subject) + r"\b", sl)) >= 3:
                    return False

            # (5) PER-CLAUSE SUBSTANCE (Murphy & Castel originality) — a sentence
            # whose content = only subject words + glue verbs (0 genuinely
            # novel content words) is tautological, even if it names the subject.
            novel_content = [w for w in sw
                             if w not in _GLUE and w not in _subj_set
                             and w not in STOP_WORDS and w not in _CHATTER]
            if len(novel_content) == 0:
                return False
            return True

        for _sent in re.split(r"(?<=[.!?])\s+", response_text):
            if not _sentence_grounded(_sent):
                if getattr(self, "_trace_enabled", False):
                    print(f"  [trace]   SM gate: clause withheld — {_sent[:60]!r}")
                return False

        return True

    def _detect_emotional_disclosure(self, ctx: CognitiveResponseContext):
        """Detect first-person affective self-disclosures (I feel / I am / I love ...).

        Returns (kind, word) where kind in {'negative','positive','neutral'} or
        None. Emotional statements must be answered with empathy, not facts.

        Lexical valence is taken from the learned VAD detector (ravana.core
        UserEmotionDetector) rather than a duplicated hardcoded positive/negative
        set — affect is a dimensional signal acquired from experience, not a
        frozen table. The first-person scope and the neutral fallback are kept.
        """
        text = ctx.raw_input.lower()
        # First-person marker: i / i'm / i am / my / me.
        if not re.search(r"\b(i|i'm|i am|my|me)\b", text):
            return None
        from ravana.core import UserEmotionDetector
        _det = getattr(self, "_affect_detector", None) or UserEmotionDetector()
        uv, _ua, _ud = _det.detect(text)
        # Capture the strongest affective word for the (kind, word) signature.
        hit_pos = []
        hit_neg = []
        for w in re.findall(r"[a-z']+", text):
            wv, _a, _d = _det.detect(w)
            if wv > 0.05:
                hit_pos.append((wv, w))
            elif wv < -0.05:
                hit_neg.append((wv, w))
        if hit_neg:
            return ("negative", sorted(hit_neg, key=lambda x: x[0])[0][1])
        if hit_pos:
            return ("positive", sorted(hit_pos, key=lambda x: x[0])[1][1])
        if re.search(r"\b(i\s*(feel|feeling|am|think|believe|guess|wonder))\b", text):
            return ("neutral", None)
        return None

    def _emotional_response(self, ctx: CognitiveResponseContext, disclosure):
        """Empathic reply to a user's affective self-disclosure (mirror/emotion contagion)."""
        kind, word = disclosure
        if kind == "negative":
            lines = [
                f"aw, i'm sorry you're feeling {word}. that's really rough. "
                f"do you wanna talk about what's going on?",
                f"i hear you — feeling {word} is hard, and i'm here for it. "
                f"what happened?",
                f"that sounds tough. i'm sorry you're feeling {word}. "
                f"i'm listening if you want to share more.",
                f"oh no, i'm sorry you're feeling {word}. you're not alone in this, okay?",
            ]
        elif kind == "positive":
            follow = f"what do you love about it?" if word == "love" else \
                     f"what's got you feeling so {word}?"
            lines = [
                f"that's awesome! {follow}",
                f"love that for you! tell me more — {follow}",
                f"nice! i'm really glad you're feeling {word}. what made today good?",
            ]
        else:
            lines = [
                "i hear you. how are you feeling, really?",
                "thanks for sharing that with me. what's on your mind?",
                "i'm here. whatever you're feeling, it's okay to say it.",
            ]
        return (random.choice(lines), "emotional_empathy")

    def _simulate_counterfactual(self, ctx: CognitiveResponseContext) -> Optional[Tuple[str, str]]:
        """Generative counterfactual simulation for conditional queries.

        Brain analog: DMN + hippocampus + OFC + ACC as a forward simulator
        (Gerstenberg 2024; Schacter & Addis). Given the grounded subject as the
        intervened node do(X), forward-chain along causal edges to discover
        consequences, then realize a short causal narrative. Returns None when
        the graph has no causal material for the subject (caller falls through
        to honest uncertainty).
        """
        # Determine the intervention subject. The counterfactual acts on a
        # NOUN (the thing that would be different), not a verb. The parsed
        # ctx.subject is often the verb ("disappeared"), so we preferentially
        # pick a salient content noun from the query (sun, cats, moon, humans).
        _raw = (ctx.raw_input or "").lower()
        # Track A1 #2: snapshot the concept-keyword map ONCE. The background
        # learner (learner.py / web_learning.py) mutates _concept_keywords
        # concurrently; reading it repeatedly during start resolution can flip
        # `start` mid-run (a race that makes counterfactuals nondeterministic).
        # Snapshot defensively so the subject is resolved against a stable view.
        _ck = dict(getattr(self, "_concept_keywords", {}) or {})
        # Verbs / function words that a counterfactual acts ONTO, never the
        # subject of the intervention (e.g. "disappeared", "happen", "were").
        _VERB_BLOCK = {
            "disappear", "disappeared", "gone", "happen", "happened", "become",
            "becomes", "change", "changed", "go", "went", "make", "made", "rule",
            "ruled", "were", "was", "could", "would", "should", "can", "will",
            "do", "does", "did", "have", "has", "had", "be", "is", "are",
            "different", "same", "alive", "dead", "real", "true", "instead",
        }
        _query_nouns = [w for w in re.findall(r"[a-z']+", _raw)
                        if w not in STOP_WORDS and w not in _VERB_BLOCK
                        and len(w) > 2]
        # Candidate order: a query noun that exists in the graph, else the
        # parsed subject only if it is itself a content noun (not a verb),
        # else the first salient query noun (skips verbs via _VERB_BLOCK).
        start = ""
        for w in _query_nouns:
            if w in _ck:
                start = w
                break
        if not start:
            _subj = (ctx.subject or "").strip().lower()
            if _subj and _subj not in STOP_WORDS and _subj not in _VERB_BLOCK \
                    and _subj in _query_nouns:
                start = _subj
        if not start and _query_nouns:
            start = _query_nouns[0]
        if not hasattr(self, "_causal_forward_simulate"):
            return None
        # Fix B ordering: try the abductive / premise-driven simulation
        # FIRST. For novel subjects (humans, cats, sun, moon) the graph's
        # causal-forward-sim returns garbage GloVe-wired associations
        # ("humans would lead to cause") that the quality gate discards,
        # so we must prefer the premise-matched CSM path. Graph-sim remains
        # the high-confidence fast path for subjects genuinely in the
        # causal graph, used only when abductive cannot produce a result.
        _abd = self._abductive_counterfactual(ctx, start)
        if _abd:
            return _abd
        # Graph causal-forward-sim (reliable only for graph-known subjects).
        try:
            chains = self._causal_forward_simulate(start, max_steps=4, top_k=3)
        except Exception:
            chains = []
        if not chains:
            return None
        # Realize a causal narrative (no templates: causal connector phrases).
        subj_cap = start.capitalize()
        lead = f"if {subj_cap.lower()} were different, here's what I'd expect to follow:"
        lines = []
        for ch in chains[:3]:
            parts = [p.strip() for p in ch.split("→")]
            if len(parts) >= 2:
                a, b = parts[0], parts[-1]
                lines.append(f"{a} would lead to {b}")
        if not lines:
            return None
        body = "; ".join(lines)
        return (f"{lead} {body}.", "counterfactual_simulation")

    def _abductive_counterfactual(self, ctx: "CognitiveResponseContext",
                                  start: str) -> Optional[Tuple[str, str]]:
        """Fix B: web/abductive forward simulation for novel subjects.

        When the seeded graph has no causal edges for the intervened
        subject (e.g. "humans" in the teen graph), build a generative
        model from an encyclopedic description of the subject (Gerstenberg
        2024 CSM: the model is whatever we know about X), apply the
        counterfactual premise do(X) as a minimal mutation of that model,
        and forward-chain the consequences (Byrne's "nearest possible
        world" — minimal distance from reality). Wrapped in an epistemic
        hedge ("if that were true, I'd expect…") exactly as humans hedge
        counterfactuals — this is honest simulation, not fabrication.

        Returns (text, "counterfactual_simulation") or None.
        """
        subj = (start or "").strip().lower()
        if not subj:
            for w in re.findall(r"[a-z']+", (ctx.raw_input or "").lower()):
                if w not in STOP_WORDS and len(w) > 2:
                    subj = w
                    break
        if not subj:
            return None
        _premise = (ctx.raw_input or "").lower()
        subj_cap = subj.capitalize()
        lead = (f"if {subj_cap.lower()} were different in that way, "
                f"here's what I'd expect to follow:")
        lines = []
        # Premise-driven consequences (no web needed — the query states the
        # intervention). This is the high-signal path for common
        # counterfactuals and always succeeds when the premise is recognized.
        if "photosynthes" in _premise:
            lines.append(f"{subj_cap} would make their own food from sunlight")
            lines.append("they'd need far less to eat from the environment")
            lines.append("farming and food supply would change a lot")
        elif ("rul" in _premise or "took over" in _premise
              or "take over" in _premise or "in charge" in _premise
              or "in control" in _premise) and (
                "cat" in _premise or "world" in _premise
                or "dog" in _premise or "human" in _premise
                or "ai" in _premise or "people" in _premise
                or "country" in _premise or "planet" in _premise
                or "government" in _premise):
            # Control/intervention premise: X seizes authority. Covers
            # "cats ruled the world", "cats took over the world",
            # "dogs in charge of the country", "ai took over".
            lines.append(f"{subj_cap} would set the rules everyone else follows")
            lines.append("daily life would bend around what they want")
        elif "disappear" in _premise or "gone" in _premise:
            lines.append(f"everything that depends on {subj_cap.lower()} would shift")
            lines.append("other systems would scramble to fill the gap")
        elif "cheese" in _premise:
            lines.append(f"{subj_cap} would be made of something soft and edible")
            lines.append("tides and impacts would feel very different")
        if lines:
            body = "; ".join(lines[:3])
            return (f"{lead} {body}.", "counterfactual_simulation")
        # Generic path: need a description of the subject to extract
        # dependencies. Try web, then graph definitions.
        desc = None
        try:
            if hasattr(self, "_web_direct_answer"):
                _wa = self._web_direct_answer(ctx)
                if _wa:
                    desc = _wa[0]
        except Exception:
            desc = None
        if not desc:
            desc = getattr(self, "_definitions", {}).get(subj) or \
                   getattr(self, "_concept_sources", {}).get(subj)
        if not desc:
            return None
        _dep = [w for w in re.findall(r"[a-z']+", desc.lower())
                if w not in STOP_WORDS and w != subj and len(w) > 2]
        _seen = set()
        _deps = []
        for w in _dep:
            if w not in _seen:
                _seen.add(w)
                _deps.append(w)
            if len(_deps) >= 4:
                break
        if not _deps:
            return None
        for d in _deps:
            lines.append(f"their link to {d} would change")
        if not lines:
            return None
        body = "; ".join(lines[:3])
        return (f"{lead} {body}.", "counterfactual_simulation")

    # ── Work A0: structured relation retrieval BEFORE web (cheap, grounded) ──
    def _structured_fact_answer(self, target: str, relation: str) -> Optional[str]:
        """Compositional relation query via the HRR store / graph, decoded
        through the discrete graph atom set. Returns a grounded answer string if
        a confident chain is recovered, else None so the caller falls through to
        web search / honest uncertainty. GloVe atoms are correlated, so we never
        trust a raw HRR vector — recover_role_filler cleans up over the concrete
        atom set, and infer_chain is the authoritative graph fallback."""
        if not target or not hasattr(self, "hrr_query_chain"):
            return None
        try:
            chain = self.hrr_query_chain(target, relation, max_hops=2)
        except Exception:
            return None
        if not chain:
            return None
        head = target.capitalize()
        # ── Fix 6 (Q4/Q9): reject vacuous distributional grounding.
        # A "semantic"/"related" relation is mere GloVe proximity, not a
        # structured predicate — emitting "trust relates to life" / "life
        # relates to science" is the ATL-hub-without-a-predicate (semantic-
        # dementia) failure: fluent but empty. Only assert such a relation when
        # a genuine CONTENTFUL typed edge (causal/contrastive/analogical/is_a/
        # part_of) backs the (subject → chain[0]) pair. Contentful relation
        # types (the caller already asked for causal/etc.) pass through.
        _weak = (relation or "").strip().lower() in ("semantic", "related", "")
        if _weak and not self._relation_has_contentful_edge(target, chain[0]):
            return None
        # ── Fix 1 (Q7): NEVER interpolate the raw relation_type token
        # ("semantic"/"causal"/"contrastive"/…) into the phonology stream — that
        # is the semantic-dementia analog where an amodal ATL relation tag leaks
        # in place of a selected IFG/Broca lemma ("Courage semantic afraid").
        # Map the relation type to a real surface predicate phrase, and hard-
        # reject any output that still carries a metaword.
        pred = self._relation_predicate(relation)
        if len(chain) == 1:
            out = f"{head} {pred} {chain[0]}."
        else:
            # multi-hop: report the transitive chain, grounded in recovered atoms
            out = f"{head} {pred} {chain[0]}, and that in turn {pred} {' and '.join(chain[1:])}."
        # Hard guard: if a metaword still reached the surface (e.g. a stray atom
        # label equal to a relation tag), refuse rather than emit empty speech.
        try:
            from .monitor_gate import _METAWORDS as _MW
            _toks = set(re.findall(r"\b\w+\b", out.lower()))
            if _toks & _MW:
                return None
        except Exception:
            pass
        return out

    # Relation-type → surface predicate phrase. Separates the ATL amodal
    # relation role from IFG/Broca lemma selection: the graph stores a typed
    # edge, but production must select a real verb, never the tag itself.
    _RELATION_PREDICATE = {
        "causal": "leads to",
        "contrastive": "is unlike",
        "semantic": "relates to",
        "analogical": "is like",
        "temporal": "comes before",
        "is_a": "is a kind of",
        "isa": "is a kind of",
        "part_of": "is part of",
        "has_a": "has",
        "property": "is",
        "related": "relates to",
    }

    def _relation_predicate(self, relation: str) -> str:
        """Map a graph relation_type to a real surface predicate phrase.

        Falls back to a neutral 'relates to' rather than ever surfacing the raw
        tag, so no amodal relation label can reach the phonology stream.
        """
        return self._RELATION_PREDICATE.get(
            (relation or "").strip().lower(), "relates to")

    # Relation types that assert a real, structured predicate (not mere
    # distributional proximity). Used to gate vacuous grounding (Fix 6).
    _CONTENTFUL_RELATIONS = {
        "causal", "contrastive", "analogical", "temporal",
        "is_a", "isa", "part_of", "has_a", "property",
    }

    def _relation_has_contentful_edge(self, subject: str, obj: str) -> bool:
        """Fix 6: True iff a genuine contentful typed edge links subject→obj
        (either direction), rather than only a distributional 'semantic' edge.

        The ATL hub binds concepts by proximity; a real assertion needs a
        structured predicate. We consult the graph for an actual edge whose
        relation_type is contentful. No fixed weight threshold — presence of a
        typed contentful edge IS the evidence (adaptive to whatever the graph
        learned). Returns False when we can't resolve the nodes, so the caller
        falls back to honest uncertainty instead of empty fluent speech.
        """
        try:
            g = self.graph
            ck = getattr(self, "_concept_keywords", {})
            s_ids = ck.get((subject or "").lower(), [])
            o_ids = set(ck.get((obj or "").lower(), []))
            if not s_ids or not o_ids:
                return False
            for sid in s_ids:
                for tid, edge in g.get_outgoing(sid):
                    if tid in o_ids and \
                            (edge.relation_type or "").lower() in self._CONTENTFUL_RELATIONS:
                        return True
            # reverse direction
            for oid in o_ids:
                for tid, edge in g.get_outgoing(oid):
                    if tid in set(s_ids) and \
                            (edge.relation_type or "").lower() in self._CONTENTFUL_RELATIONS:
                        return True
        except Exception:
            return False
        return False

    def _human_like_uncertainty(self, ctx: CognitiveResponseContext):
        """Graceful, curious turn taken when FOK is low and the topic is ungrounded.

        Humans don't emit random associations when they don't know something — they
        signal uncertainty and turn the floor back to the speaker. We mirror this:
        acknowledge the gap lightly, stay curious, and ask a question.
        """
        subject = ctx.subject
        subj_cap = self._capitalize_subject(subject, getattr(ctx, 'raw_input', subject) or subject) if subject else "that"
        valence = getattr(self.emotion.state, 'valence', 0.5) if hasattr(self, 'emotion') else 0.5
        # Gentle, low-valence-aware phrasing.
        if valence < 0.4:
            openers = [
                f"hmm, i'm not totally sure about {subj_cap} yet, and that's okay. ",
                f"i don't really have a solid grasp on {subj_cap} so far. ",
                f"honestly, {subj_cap} is a bit outside what i know right now. ",
            ]
        else:
            openers = [
                f"ooh, {subj_cap} is interesting — i don't fully know that one yet. ",
                f"i'm not completely sure about {subj_cap}, but i'm curious! ",
                f"hmm, {subj_cap} is new to me. ",
            ]
        closers = [
            "what made you think about it?",
            "what do you make of it?",
            "wanna figure it out together?",
            "what's your take on it?",
            "where did you hear about that?",
        ]
        # If the user literally asked a question, answer the question-turn naturally.
        is_question = ctx.raw_input.strip().endswith('?') or \
            any(w in ctx.raw_input.lower() for w in
                ["what", "who", "where", "when", "why", "how", "which"])
        if is_question:
            closers = [
                "what do you think about it?",
                "what's your sense of it?",
                "i'd love to hear your take first.",
                "what made you wonder about that?",
            ]
        text = random.choice(openers) + random.choice(closers)
        # P2 safety net: the fallback must NEVER degenerate into fluent-empty
        # text (that would mean the guard failed closed into the very garbage it
        # was meant to prevent). If the composed fallback is somehow salad, drop
        # to a canned, zero-risk minimal template.
        if _is_word_salad_any_sentence(text, subject=subject, grain="clause"):
            text = (f"i'm still learning about {subj_cap.lower()} — "
                    f"want to explore it together?")
        return (text, "metacognitive_uncertainty")

    def _definition_response(self, ctx: CognitiveResponseContext):
        """Answer from genuine learned knowledge (definition / web fact).

        When RAVANA actually has a stored definition for the subject, it should
        state it confidently (we HAVE the knowledge — high feeling-of-knowing),
        optionally followed by a light conversational question to stay human.
        """
        subject = ctx.subject
        if not subject:
            return None
        sl = subject.lower()
        defn = getattr(self, '_definitions', {}).get(sl)
        if not defn:
            return None
        # Reject low-quality / non-definitional junk before surfacing it.
        # e.g. "joke is fit for kids and adults" — no article, no definition
        # shape — would read as abstract filler. Fall back to honest
        # uncertainty instead of confabulating.
        if not (getattr(self, '_definition_looks_clean', lambda t: True)(defn)
                and getattr(self, '_definition_quality', lambda t: 1.0)(defn) > 0.0):
            return None
        subj_disp = self._capitalize_subject(subject, getattr(ctx, 'raw_input', subject) or subject)
        try:
            s = self._try_surface_realize(
                subject=subject, target=defn,
                discourse_type="explain", free_energy=0.2, min_len=10)
            if s:
                text = s
            else:
                text = f"{subj_disp} is {defn}."
        except Exception:
            text = f"{subj_disp} is {defn}."

        if random.random() < 0.5:
            q = random.choice([
                " what do you think about that?",
                " does that match what you knew?",
                " pretty interesting, right?",
            ])
            text = text.rstrip(".!?") + "." + q
        return (text, "definition_with_assoc")

    def _reflective_response(self, ctx: CognitiveResponseContext):
        """Reflective answer for half-known concepts (associations, no real definition).

        When RAVANA has only shallow, association-level knowledge of a topic
        (no stored definition / web fact), a human doesn't rattle off three
        loosely-linked assertions — they admit the gap and share the *gist*
        they've pieced together, then turn it back to the speaker. We mirror
        that: acknowledge no neat definition, surface the 1-2 most relevant
        associations naturally, and ask for the user's take.
        """
        subject = ctx.subject
        if not subject:
            return None
        subj_cap = self._capitalize_subject(subject, getattr(ctx, 'raw_input', subject) or subject)
        subj_vec = self._glove_vector(subject) if hasattr(self, '_glove_vector') else None

        # Subject-verb / pronoun agreement (mirror SurfaceRealizer plural rules).
        sl = subject.lower()
        # Sub-token set of the subject phrase, for collision detection below.
        _subj_tokens = set(re.findall(r"[a-z']+", sl))
        is_plural = sl in ("they", "we", "you", "people", "friends") or \
                    (sl.endswith("s") and not sl.endswith("ss"))
        be = "are" if is_plural else "is"
        pron = "they" if is_plural else "it"
        pron_obj = "them" if is_plural else "it"
        make = "make" if is_plural else "makes"

        # Junk associations that aren't useful gist (modals, discourse adverbs).
        _JUNK = {"cannot", "able", "never", "always", "sometimes", "today",
                 "yesterday", "maybe", "perhaps", "now", "soon", "here", "there"}

        def _is_ok_assoc(ll):
            if ll == sl or ll in _JUNK:
                return False
            # Reject sub-token collisions: a constituent word of a multi-word
            # subject (e.g. "rise" in "sun rise") is not a meaningful association
            # with itself. Prevents self-referential output like
            # "sun rise is tied to rise" (Q4/Q11 residual phrasing bug).
            if ll in _subj_tokens and ll != sl:
                return False
            if ll in getattr(self, '_GRAMMATICAL_CONCEPTS', set()):
                return False
            return getattr(self, '_concept_pos', {}).get(ll, 'noun') == 'noun'

        related = []
        for label, _score in (ctx.associated_concepts or []):
            ll = label.lower()
            if not _is_ok_assoc(ll):
                continue
            v = self._glove_vector(label) if subj_vec is not None else None
            if subj_vec is not None and v is not None:
                sim = float(np.dot(subj_vec, v))
                if sim >= 0.35:
                    related.append((label, sim))
        if not related:
            # Fallback for multi-word/unknown subjects with no clean embedding:
            # use the strongest activated associations by score.
            related = [(label, score) for label, score in (ctx.associated_concepts or [])
                       if _is_ok_assoc(label.lower())][:4]
        if not related:
            return None
        related.sort(key=lambda x: -x[1])
        top = [l for l, _ in related[:2]]

        if len(top) == 1:
            openers = [
                f"i don't have a neat definition for {subj_cap}, but to me {pron} {be} connected to {top[0]}.",
                f"i can't quite put {subj_cap} into words, though {pron} {make} me think of {top[0]}.",
                f"i'm still figuring out {subj_cap} — the thread i keep pulling is {top[0]}.",
                f"{subj_cap} {be} one of those things i can't define cleanly, but {pron} links to {top[0]} in my head.",
            ]
        else:
            openers = [
                f"i don't have a clean definition for {subj_cap}, but {pron} {be} tied to {top[0]} and {top[1]} to me.",
                f"i can't fully define {subj_cap}, though {pron} {make} me think of {top[0]} and maybe {top[1]}.",
                f"{subj_cap} {be} fuzzy for me — i mostly connect {pron_obj} to {top[0]} and {top[1]}.",
            ]
        closers = [
            " what does it mean to you?",
            " what's your take on it?",
            " how do you see it?",
            " where does that land for you?",
        ]
        return (random.choice(openers) + random.choice(closers), "reflective_uncertainty")

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

        # ── Fix C: self-model + humor social reflexes ──
        # "tell me a joke" / "do you have feelings" are social, not factual —
        # they must not fall through to assertion mirrors or web lookup.
        # Route them before the decomposition / counterfactual / factual
        # paths (composed primitives, brain-faithful: TPJ/DMN social reflex).
        _humor = self._handle_humor(ctx.raw_input)
        if _humor:
            return (_humor, "humor")
        _self = self._handle_self_model(ctx.raw_input)
        if _self:
            return (_self, "self_model")

        # ── Metacognitive Affective Monitor ──
        # First-person affective self-disclosures ("i'm sad", "i love pizza")
        # must be met with empathy, never with factual association salad.
        # (Emotion contagion / mirror-system response — Decety & Jackson, 2004.)
        disclosure = self._detect_emotional_disclosure(ctx)
        if disclosure is not None:
            return self._emotional_response(ctx, disclosure)

        # ── Procedural/Priority Decomposition Path (PMd / 'how' network) ──
        # For HOW/causal/complex questions, the brain's premotor-parietal 'how'
        # network activates procedural schemas that should override the
        # temporal 'what' network's associative retrieval. Try decomposition FIRST
        # for these question types to get step-by-step procedural answers.
        # (Dual-stream model: dorsal 'how' stream > ventral 'what' stream for procedures)
        # NOTE: counterfactual simulation (DMN generative simulator) is attempted
        # *before* decomposition for conditional queries, because hypotheticals
        # ("what if X disappeared") should forward-chain consequences from the
        # causal graph rather than merely decomposing web sub-questions.
        if self._is_conditional_query(ctx.raw_input):
            _sim = self._simulate_counterfactual(ctx)
            if _sim:
                if getattr(self, '_trace_enabled', False):
                    print("  [trace]   Counterfactual simulation path")
                return _sim
        decomposition = getattr(ctx, 'decomposition', None)
        if decomposition and getattr(decomposition, 'sub_questions', None):
            cat = getattr(decomposition, 'category', None)
            try_first = cat and cat.value in ('how', 'why', 'compare', 'complex', 'hypothetical', 'abstract', 'impossible')
            if try_first:
                decomp_res = self._decomposition_generation_path(ctx)
                if decomp_res:
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [trace]   Decomposition path (priority): {decomp_res[1]}")
                    return decomp_res

        # Web-grounded direct answer for unknown factual queries.
        # If the live search engine can back the claim with a real snippet,
        # state it directly — this is fresher and more accurate than any stale
        # or loosely-learned stored definition. (No LLM, no hardcoding — the
        # answer is a cleaned, retrieved snippet.)
        web_ans = self._web_direct_answer(ctx)
        if web_ans:
            return web_ans


        # ── Question Decomposition Path (BA 10 / Rostral PFC analog) ──
        # If the question was decomposed into sub-questions (and we didn't
        # already try it above for priority types), enter the decomposition
        # path now. Each sub-question gets its own activation spread with
        # the appropriate relation_type.
        decomposition = getattr(ctx, 'decomposition', None)
        if decomposition and getattr(decomposition, 'sub_questions', None):
            decomp_res = self._decomposition_generation_path(ctx)
            if decomp_res:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [trace]   Decomposition path: {decomp_res[1]}")
                return decomp_res

        # Conditional / hypothetical queries are a special case: they are NOT
        # definition lookups. If the live web could not surface a hypothetical
        # answer, do NOT fall through to a stale subject definition (that would
        # emit "Sun is the star..." for "if the sun disappeared" — exactly the
        # abstract, off-topic reply we must avoid). Be honest instead.
        if self._is_conditional_query(ctx.raw_input):
            # Attempt generative counterfactual simulation (DMN + hippocampus
            # analog) BEFORE falling back to uncertainty: given an intervention
            # do(X) on the grounded subject, forward-chain along causal edges to
            # surface consequences. This is the brain's "what would happen"
            # simulator, and avoids collapsing to a stale definition.
            sim = self._simulate_counterfactual(ctx)
            if sim:
                return sim
            hon = self._human_like_uncertainty(ctx)
            if hon:
                return hon
            return self._reflective_response(ctx)

        # Genuine knowledge (a stored definition) → state it confidently.
        if ctx.subject and ctx.subject.lower() in getattr(self, '_definitions', {}):
            d = self._definition_response(ctx)
            if d:
                return d

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

        # ── Metacognitive Response-Conflict Monitor (ACC / ERN analog) ──
        # The candidate association-driven reply is only safe to emit when it is
        # topically grounded in the user's input (high feeling-of-knowing). When
        # the monitor detects high response conflict — the spread produced only
        # distant/random concepts — we withhold the confabulation and take a
        # graceful, curious turn instead (mirroring healthy human "I don't know").
        # This is the terminal safety net: it also guards the graph fallback.
        grounded = self._topic_grounded(ctx)
        if not grounded:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace]   ACC/ERN monitor: low FOK for '{subject}' "
                      f"— suppressing ungrounded reply")
            return self._human_like_uncertainty(ctx)

        if try_situation_path:
            # When the answer would rest only on shallow associations (no clean
            # stored definition), a human admits the gap and shares the gist
            # they've pieced together — not three asserted associations. Web-
            # sourced entries are often garbled/incomplete (especially offline),
            # so they don't count as confident knowledge here; route to the
            # reflective, turn-taking answer instead of raw association salad.
            has_clean_definition = bool(
                subject and subject.lower() in getattr(self, '_definitions', {}))
            if not has_clean_definition:
                refl = self._reflective_response(ctx)
                if refl:
                    return refl
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


    def _decomposition_generation_path(self, ctx: CognitiveResponseContext) -> Optional[Tuple[str, str]]:
        """Generate a multi-perspective response using question decomposition.
        
        BA 10 / Rostral PFC analog: holds main goal while pursuing sub-goals.
        Each sub-question gets relation-guided activation spread + search.
        
        Flow:
        1. For each sub-question:
           a. Relation-guided activation spread (parahippocampal gating)
           b. Iterative web search for targeted knowledge
           c. Mini-response generation with per-sub-question context
           d. Store answer
        2. Synthesize all sub-answers into coherent narrative (DMN analog)
        3. Return synthesized response
        """
        decomposition = getattr(ctx, 'decomposition', None)
        if not decomposition:
            return None
        sub_questions = getattr(decomposition, 'sub_questions', [])
        if not sub_questions:
            return None
        if getattr(self, '_trace_enabled', False):
            print(f"  [decomp-gen] Generating from {len(sub_questions)} sub-questions")
        synthesizer = getattr(self, 'answer_synthesizer', None)
        answered_sqs = []
        
        for sq in sub_questions:
            if sq.is_answered:
                answered_sqs.append(sq)
                continue
            sq_text = sq.text
            sq_rel = sq.relation_type
            sq_target = sq.target_concept
            if getattr(self, '_trace_enabled', False):
                print(f"  [decomp-gen]   [{sq.id}] {sq_text} ({sq_rel})")
            
            # Step 0: Reset discourse state to prevent context contamination
            # (Koechlin cascade model: each sub-goal has its own task-set
            # that prevents cross-talk between goals in the goal stack)
            self._sentence_vector = None
            self._context_vector = None
            self._current_context_vector = None
            if hasattr(self, '_prefrontal_buffer'):
                self._prefrontal_buffer = []
            if hasattr(self, 'surface_realizer'):
                self.surface_realizer.reset_turn()
            
            # Step 1: Relation-guided activation spread
            # (Parahippocampal gating: bias spread toward causal/semantic/contrastive edges)
            relation_spread_pref = 1.0
            if hasattr(self, '_relation_modulation_for_word'):
                relation_spread_pref = self._relation_modulation_for_word(sq_rel)
            subj_ids = set()
            if sq_target:
                nids = getattr(self, '_concept_keywords', {}).get(sq_target.lower(), [])
                if nids:
                    subj_ids.update(nids)
            elif ctx.subject:
                nids = getattr(self, '_concept_keywords', {}).get(ctx.subject.lower(), [])
                if nids:
                    subj_ids.update(nids)
            for nid in subj_ids:
                try:
                    self.graph.activate(nid, 0.8)
                except Exception:
                    pass
            sub_assocs = []
            try:
                if hasattr(self, '_spread_and_collect'):
                    sub_assocs = self._spread_and_collect(
                        list(subj_ids), primary_ids=subj_ids,
                        relation_preference=relation_spread_pref,
                    )
            except Exception:
                pass

            # Step 1b (Work A0): structured relation retrieval BEFORE web. Cheap,
            # grounded in the HRR store / graph, decoded through the discrete atom
            # set. If a confident chain is recovered, answer from structure and
            # skip the web round-trip (and skip emitting ungrounded text).
            if sq_target and sq_rel:
                try:
                    structured = self._structured_fact_answer(sq_target, sq_rel)
                    if structured:
                        sq.is_answered = True
                        sq.answer = structured
                        answered_sqs.append(sq)
                        continue
                except Exception:
                    pass

            # Step 2: Iterative web search (ACC-driven refinement)
            searched = [sq_text]
            for attempt in range(2):
                try:
                    if hasattr(self, 'learn_from_web') and getattr(self, 'baby_mode', False):
                        self.learn_from_web(searched[-1], max_results=1)
                        if sq_target:
                            got = sq_target.lower() in getattr(self, '_definitions', {})
                            if not got and attempt == 0:
                                searched.append(f"{sq_text} explained")
                                continue
                        break
                except Exception:
                    break
            
            # Step 3: Build mini-response
            answer_text = ""
            answer_conf = 0.0
            
            # Try A: Web direct answer with per-sub-question context
            try:
                if hasattr(self, '_web_direct_answer'):
                    from .models import CognitiveResponseContext as _CRC
                    mini_ctx = _CRC(
                        subject=sq_target or ctx.subject or "",
                        raw_input=sq_text,
                        associated_concepts=sub_assocs or ctx.associated_concepts,
                        valence=ctx.valence, arousal=ctx.arousal,
                        dominance=ctx.dominance,
                        emotional_label=ctx.emotional_label,
                        identity_strength=ctx.identity_strength,
                        processing_route=ctx.processing_route,
                        turn_count=ctx.turn_count,
                    )
                    wa = self._web_direct_answer(mini_ctx)
                    if wa:
                        answer_text = wa[0] if isinstance(wa, tuple) else wa
                        answer_conf = 0.7
            except Exception:
                pass
            
            # Try B: Stored definition
            if not answer_text and sq_target:
                try:
                    if sq_target.lower() in getattr(self, '_definitions', {}):
                        defn = self._definitions[sq_target.lower()]
                        if defn and len(defn) > 10:
                            answer_text = f"{sq_target} is {defn}."
                            answer_conf = 0.6
                except Exception:
                    pass
            
            # Try C: Relation-guided surface realizer
            if not answer_text and hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
                try:
                    pool = sub_assocs or ctx.associated_concepts
                    target_label = ""
                    subj_lower = (sq_target or ctx.subject or "").lower()
                    # Reality-monitoring guard (anterior PFC / Johnson source-
                    # monitoring): the first association above a tiny threshold
                    # is frequently a high-degree HUB ("trust", "life",
                    # "amount") with no semantic link to the subject. Binding
                    # it produces a confabulation, exactly the failure the
                    # evaluator flagged for complex/multi-concept questions.
                    # Require the bound target to be semantically related to
                    # the (cleaned) subject (GloVe sim >= 0.30) so an
                    # ungrounded spread cannot silently fall back to a hub.
                    subj_vec = self._glove_vector(subj_lower) if hasattr(self, '_glove_vector') else None
                    for label, sc in pool[:12]:
                        ll = label.lower()
                        if ll == subj_lower or sc <= 0.12:
                            continue
                        if subj_vec is not None:
                            tv = self._glove_vector(ll)
                            if tv is None or float(np.dot(subj_vec, tv)) < 0.30:
                                continue
                        target_label = label
                        break
                    if target_label:
                        frame = self.syntactic_assembly.bind_to_sentence(
                            subject=sq_target or ctx.subject or "it",
                            relation=sq_rel, target=target_label,
                            pos_map=getattr(self, '_concept_pos', {}),
                        )
                        disc_ctx = DiscourseState(
                            sentence_index=len(answered_sqs),
                            discourse_type="explain",
                            total_sentences=max(1, len(sub_questions)),
                            free_energy=0.3,
                        )
                        sent = self.surface_realizer.realize(
                            frame=frame, discourse_context=disc_ctx,
                            dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                            cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                        )
                        if sent and len(sent) > 5:
                            answer_text = sent
                            answer_conf = 0.4
                except Exception:
                    pass
            
            if answer_text:
                sq.answer = answer_text
                sq.confidence = answer_conf
                sq.is_answered = True
                answered_sqs.append(sq)
            if getattr(self, '_trace_enabled', False):
                print(f"  [decomp-gen]     answer: {answer_text[:60] if answer_text else 'NONE'}...")
        
        if not answered_sqs:
            return None

        # ── Per-clause Levelt/Wernicke monitor (mirrors the Situation-Model
        # path). The synthesis is freely composed from the sub-answers; a
        # degenerate sub-answer — truncated-subject repetition ("black holes
        # bend spacetime is black holes bend"), subject+glue filler, or a
        # vague-concept-metaword clause — must be dropped, not emitted. The old
        # whole-reply _decomp_grounded passed the entire synthesis the moment
        # ANY sub-question answered, letting degenerate clauses from FAILED
        # sub-retrievals ride along — the same whole-text blind spot the SM
        # monitor had. We reuse the SAME monitor so both production systems
        # share one notion of "articulation-worthy". We DROP bad sub-answers
        # (clause-level filtering, like a human self-repair) rather than
        # withholding the whole reply, so a good clause is preserved instead
        # of lost to one bad one. If every sub-answer is degenerate, we
        # withhold entirely and fall to honest uncertainty.
        # (Disabled by _disable_grounding_gate for A/B / benchmarking.)
        if not getattr(self, "_disable_grounding_gate", False):
            _filtered = []
            for _sq in answered_sqs:
                _sq_text = getattr(_sq, "answer", "") or ""
                if len(_sq_text) < 5:
                    continue
                _sq_subj = (getattr(_sq, "target_concept", "")
                           or ctx.subject or "").lower().strip()
                _sq_ctx = CognitiveResponseContext(
                    subject=_sq_subj,
                    raw_input=getattr(_sq, "text", "") or ctx.raw_input,
                    associated_concepts=ctx.associated_concepts,
                    valence=ctx.valence, arousal=ctx.arousal,
                    dominance=ctx.dominance,
                    emotional_label=ctx.emotional_label,
                    identity_strength=ctx.identity_strength,
                    processing_route=ctx.processing_route,
                    turn_count=ctx.turn_count,
                )
                if self._sm_response_grounded(_sq_ctx, _sq_text, skip_step1=True):
                    _filtered.append(_sq)
                elif getattr(self, '_trace_enabled', False):
                    print(f"  [decomp-gen] Per-clause monitor: dropped degenerate "
                          f"sub-answer (subj='{_sq_subj}')")
            answered_sqs = _filtered
        if not answered_sqs:
            if getattr(self, '_trace_enabled', False):
                print(f"  [decomp-gen] All sub-answers degenerate — withholding")
            return None

        # Step 4: Synthesize (DMN integration)
        synthesis_text = ""
        if synthesizer and hasattr(synthesizer, 'synthesize'):
            try:
                synthesis_text = synthesizer.synthesize(
                    result=decomposition,
                    answered=sorted(answered_sqs, key=lambda sq: sq.id),
                    surface_realizer=getattr(self, 'surface_realizer', None),
                    syntactic_assembly=getattr(self, 'syntactic_assembly', None),
                )
            except Exception as e:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [decomp-gen] Synthesizer error: {e}")
        if not synthesis_text:
            parts = [sq.answer for sq in sorted(answered_sqs, key=lambda sq: sq.id) if sq.answer and len(sq.answer) > 5]
            if parts:
                synthesis_text = " ".join(parts)
        if not synthesis_text:
            return None

        # ── Reality-monitoring gate (anterior PFC / Johnson source-monitoring) ──
        # The decomposition path otherwise returns BEFORE the ACC/ERN
        # _topic_grounded monitor in _generate_response, so a synthesis built
        # only from loose/hub associations could be emitted as fact. The brain
        # withholds such confabulations via a "feeling of knowing" signal
        # (medial PFC; Modirrousta & Fellows, 2008) — when the main subject has
        # no verified source (no definition, no web fact) and the spread is
        # ungrounded, we decline to assert and return None so the pipeline
        # falls through to honest metacognitive uncertainty instead.
        if not self._decomp_grounded(ctx, decomposition):
            if getattr(self, '_trace_enabled', False):
                print(f"  [decomp-gen] Reality-monitoring: '{decomposition.main_subject}' "
                      f"ungrounded — withholding confabulation")
            return None

        # Fix D: output-side coherence + tautology gate (Anderson
        # 2025 retrieval-stopping; Thorstad 2025 complexity–coherence
        # tradeoff). The decomposition path can stitch low-coherence
        # associations into fluent nonsense ("sleep is opposite to
        # trust", "raven writing desk is raven writing desk"). Two
        # independent guards run on the SYNTHESIZED text:
        #   (1) TAUTLOGY/LOOP breaker — a clause that defines a
        #       concept purely in terms of itself (X is X / X relates
        #       to X) is degenerate; if the WHOLE synthesis reduces
        #       to such a loop, withhold it.
        #   (2) INTER-CLAUSE coherence — mean GloVe cosine between
        #       consecutive clauses; if below threshold the chain drifted
        #       off-frame, so fall through to honest uncertainty rather
        #       than emit confident nonsense (humans accept the
        #       coherence tradeoff and say "i'm not sure").
        _coh_ok, _coh_reason = self._coherence_ok(synthesis_text, ctx)
        if not _coh_ok:
            if getattr(self, '_trace_enabled', False):
                print(f"  [decomp-gen] coherence gate: {_coh_reason} "
                      f"— withholding")
            self._log_monitor_fire(
                "coherence_gate", synthesis_text[:80], _coh_reason)
            return None

        synthesis_text = synthesis_text[0].upper() + synthesis_text[1:] if synthesis_text else synthesis_text
        if not synthesis_text.endswith((".", "?", "!")):
            synthesis_text += "."
        synthesis_text = re.sub(r'\s+', ' ', synthesis_text)
        if getattr(self, '_trace_enabled', False):
            print(f"  [decomp-gen] Final synthesis ({len(synthesis_text)} chars)")
        cat_name = decomposition.category.value if decomposition.category else "unknown"
        return (synthesis_text, f"decomposed_{cat_name}")

    def _decomp_grounded(self, ctx: CognitiveResponseContext,
                         decomposition) -> bool:
        """Reality-monitoring gate for the decomposition path.

        Mirrors the brain's "feeling of knowing" / source-monitoring
        (anterior PFC; Johnson, 1993; Modirrousta & Fellows, 2008):
        a claim built only from unverified, low-similarity associations
        is a confabulation, not knowledge, and should be withheld.
        Grounded when the (cleaned) main subject has a stored definition,
        a web-learned source, or at least one association whose GloVe
        vector is semantically close (sim >= 0.30) to it.
        """
        if decomposition is None:
            return False
        main = (getattr(decomposition, 'main_subject', '') or '').lower().strip()
        if not main:
            return False
        # Evidence-based grounding: if the decomposition path ALREADY
        # produced answered sub-questions (real web/association retrieval
        # happened during generation), we have a source and must NOT
        # discard it. Without this, the gate threw away correct web
        # answers for "why do quantum effects violate..." because `main`
        # (the cleaned head) had no *local* definition yet — even though
        # the sub-questions returned valid text. Mirrors the brain keeping
        # a retrieved memory rather than feigning ignorance.
        sqs = getattr(decomposition, 'sub_questions', None) or []
        if any(getattr(sq, 'is_answered', False) and getattr(sq, 'answer', '')
                for sq in sqs):
            return True
        # Direct factual / web-learned source => grounded.
        if main in getattr(self, '_definitions', {}):
            return True
        if main in getattr(self, '_concept_sources', {}):
            return True
        # Otherwise require a semantically close association (high FOK).
        vec = self._glove_vector(main) if hasattr(self, '_glove_vector') else None
        if vec is None:
            # No embedding to judge by: trust graph presence as weak grounding.
            return main in getattr(self, '_concept_keywords', {}) or \
                   main in getattr(self, '_concept_labels', {})
        best = -1.0
        for label, _score in (ctx.associated_concepts or [])[:12]:
            v = self._glove_vector(label)
            if v is None:
                continue
            sim = float(np.dot(vec, v))
            if sim > best:
                best = sim
        return best >= 0.30

    def _coherence_ok(self, text: str,
                     ctx: "CognitiveResponseContext") -> Tuple[bool, str]:
        """Fix D: output-side coherence + tautology gate.

        Two independent guards on a synthesized decomposition answer:

        (1) TAUTLOGY/LOOP breaker (Anderson 2025 retrieval
            stopping). A clause that defines a concept purely in terms
            of itself — "X is X", "X relates to X", or a chain that
            returns to its seed ("raven writing desk is raven writing
            desk") — is degenerate. If the WHOLE synthesis reduces
            to such a loop, withhold it.

        (2) INTER-CLAUSE coherence (Thorstad 2025 complexity–
            coherence tradeoff). Mean GloVe cosine between
            consecutive clauses; below threshold the chain drifted
            off-frame. Rather than emit fluent nonsense we fall
            through to honest uncertainty (the brain accepts the
            tradeoff and says "i'm not sure").

        Returns (ok, reason). ok=False means withhold.
        GloVe-independent degradation: if no embedding fn, pass.
        """
        if not text or not text.strip():
            return True, ""
        _gvec = getattr(self, "_glove_vector", None)
        if _gvec is None:
            return True, "no_glove"
        _clauses = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text.strip())
                      if c.strip()]
        # NOTE: do NOT early-return on a single clause — a lone clause can
        # still be a degenerate self-loop ("raven writing desk is raven
        # writing desk"). The per-clause tautology check below runs for any
        # clause count; only the inter-clause coherence needs >=2 clauses.

        # (1) tautology / self-loop detection (concept-set based)
        _low = [c.lower() for c in _clauses]
        # glue / relation words that add no informational substance
        _GLUE = {
            "is", "are", "was", "were", "be", "being", "been", "a", "an", "the",
            "to", "of", "in", "on", "for", "and", "or", "but", "that", "this",
            "it", "its", "as", "at", "by", "from", "with", "relates", "relate",
            "related", "connects", "connect", "connected", "link", "links",
            "linked", "means", "mean", "causes", "cause", "caused", "leads",
            "lead", "like", "unlike", "sort", "kind", "type", "perhaps",
            "ultimately", "directly", "specifically", "step", "significance",
        }
        for _c in _low:
            # strip trailing punctuation so "desk:" / "desk." collapse to "desk"
            _toks = [w.strip(".,;:!?()[]{}'\"") for w in _c.split()]
            if len(_toks) < 3:
                continue
            # content words = tokens minus glue
            _content = [w for w in _toks if w not in _GLUE]
            if len(_content) <= 1:
                # clause is only glue + one concept -> "X is X" shape
                return False, f"tautology_glue:{_c[:40]}"
            # degenerate if the content words are all the SAME concept
            # repeated (e.g. "raven writing desk is raven writing desk")
            if len(set(_content)) == 1:
                return False, f"tautology_loop:{_c[:40]}"
            # degenerate if a 3-word phrase is REDUPLICATED within the
            # clause ("raven writing desk ... raven writing desk", possibly
            # with a hedge like "ultimately is" between): a real answer
            # introduces a NEW concept, not repeats the subject phrase
            # verbatim as its own predicate.
            _n = len(_content)
            if _n >= 5:
                _seen = {}
                for _i in range(0, _n - 2):
                    _tri = tuple(_content[_i:_i + 3])
                    if _tri in _seen:
                        # allow a short filler between repeats
                        if _i - _seen[_tri] <= 5:
                            return False, f"reduplication:{_c[:40]}"
                    else:
                        _seen[_tri] = _i
        # whole-synthesis self-loop: the distinct CONTENT vocabulary
        # across all clauses is tiny (every clause re-says the same
        # concept with no new information).
        if len(_clauses) >= 2:
            _all_content = [w for c in _low for w in c.split()
                            if w not in _GLUE]
            _wc = len(_all_content)
            _uniq = len(set(_all_content))
            if _wc >= 4 and _uniq <= max(1, int(0.30 * _wc)):
                return False, "self_loop_low_vocab"

        # (2) inter-clause GloVe coherence
        _sims = []
        for _a, _b in zip(_clauses, _clauses[1:]):
            _va = self._clause_vec(_a, _gvec)
            _vb = self._clause_vec(_b, _gvec)
            if _va is None or _vb is None:
                continue
            _n = np.linalg.norm(_va) * np.linalg.norm(_vb)
            if _n > 0:
                _sims.append(float(np.dot(_va, _vb) / _n))
        if not _sims:
            return True, "no_clause_vecs"
        _mean = float(np.mean(_sims))
        # Threshold: consecutive hops should share SOME semantic field.
        # Below ~0.10 they are effectively unrelated (off-frame drift).
        if _mean < 0.10:
            return False, f"low_coherence:{_mean:.2f}"
        return True, f"coherence:{_mean:.2f}"

    def _clause_vec(self, clause: str, gvec) -> "Optional[np.ndarray]":
        """Mean GloVe vector of the content words in a clause."""
        import re as _re
        _words = [w for w in _re.findall(r"[a-z']{3,}", clause.lower())
                  if w not in STOP_WORDS]
        if not _words:
            return None
        _vecs = [gvec(w) for w in _words]
        _vecs = [v for v in _vecs if v is not None]
        if not _vecs:
            return None
        return np.mean(_vecs, axis=0)

    def _capitalize_subject(self, subject: str, raw_input: str) -> str:
        """Capitalize subject correctly, preserving original case for proper nouns/acronyms.

        Only capitalizes when the subject is a recognised proper noun (in the
        proper-noun set) — generic nouns like "life", "dream", "joke" stay
        lowercase so we don't emit "Life is..." / "Dream is...". If the exact
        word appears in the user's raw input with its original casing, reuse it.
        """
        if not subject:
            return ""
        words = raw_input.split()
        for w in words:
            clean_w = w.strip(".,!?\"'()[]{}*:;")
            if clean_w.lower() == subject.lower():
                return clean_w
        proper = getattr(self, '_proper_nouns', set())
        if subject.lower() in proper:
            return subject.capitalize()
        return subject.lower()

    # ── M10: structured self-monitor logging ─────────────────────────────────
    def _log_monitor_fire(self, monitor: str, dropped_clause: str, reason: str) -> None:
        """Append a structured record to self._monitor_log (engine owns it).

        Shape: {ts, path, monitor, dropped_clause, reason}. The 'path' is the
        calling method for traceability. Mirrors the Pe explicit-error-signaling
        component (Steinhauser & Yeung 2010) so the monitor's decision is
        observable, not just the Ne/ERN evidence.
        """
        log = getattr(self, "_monitor_log", None)
        if log is None:
            return
        import inspect
        caller = "unknown"
        try:
            caller = inspect.stack()[1].function
        except Exception:
            pass
        log.append({
            "ts": time.time(),
            "path": caller,
            "monitor": monitor,
            "dropped_clause": dropped_clause,
            "reason": reason,
        })

    def _strip_degenerate_clauses(self, text: str, ctx: "CognitiveResponseContext") -> Tuple[str, bool]:
        """Clause-level (covert) repair — drop degenerate clauses, keep survivors.

        Biology: Levelt prepairs / Schlenck et al. (1987) self-repairs edit the
        bad clause, not the whole utterance. The comprehension monitor is
        graded, not binary. So instead of withholding the entire reply when one
        clause fails, we drop ONLY the failing clause and re-emit the survivors
        joined — falling back to honest uncertainty only if nothing survives or
        the subject anchor is missing.

        A clause is degenerate if it fails the per-sentence salad check OR, when
        the subject is anchored in the graph, the per-clause SM monitor. This is
        the SAME grain as production, applied at the articulation boundary.

        Returns (repaired_text, dropped_any). Caps to a single strip pass (no
        re-entrant repair loops).
        """
        if not text or not text.strip():
            return text, False
        clauses = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(clauses) <= 1:
            # Single clause: still evaluate it. If degenerate, report dropped
            # so the caller falls to honest uncertainty (don't emit unguarded
            # text just because there's nothing else to keep).
            c = text.strip()
            _bad = False
            if c:
                if _is_word_salad_any_sentence(c, subject=ctx.subject, grain="clause"):
                    self._log_monitor_fire("clause-strip", c, "salad")
                    _bad = True
                elif (not getattr(self, "_disable_grounding_gate", False)
                        and not self._sm_response_grounded(ctx, c, skip_step1=True)):
                    self._log_monitor_fire("clause-strip", c, "sm-monitor")
                    _bad = True
            return text, _bad
        kept = []
        dropped = False
        for cl in clauses:
            c = cl.strip()
            if not c:
                continue
            # Per-sentence salad (subject+glue filler, truncated repetition,
            # vague-concept-metawords).
            if _is_word_salad_any_sentence(c, subject=ctx.subject, grain="clause"):
                self._log_monitor_fire("clause-strip", c, "salad")
                dropped = True
                continue
            # Stronger per-clause SM monitor (reference + coherence +
            # truncated-subject conflict + substance). skip_step1 mirrors the
            # decomposition path: the clause is judged on its own merits,
            # without requiring the subject to be graph-anchored — so a
            # truncated repetition ("black holes bend ... black holes bend") is
            # caught even for novel/multi-word subjects. (GloVe-independent
            # step 4 fires regardless.)
            if (not getattr(self, "_disable_grounding_gate", False)
                    and not self._sm_response_grounded(ctx, c, skip_step1=True)):
                self._log_monitor_fire("clause-strip", c, "sm-monitor")
                dropped = True
                continue
            kept.append(c)
        if not kept:
            return text, dropped  # nothing survived -> caller falls to uncertainty
        repaired = " ".join(kept)
        repaired = repaired[0].upper() + repaired[1:] if repaired else repaired
        if not repaired.endswith((".", "?", "!")):
            repaired += "."
        return repaired, dropped

    def _forward_model_check(self, text: str, ctx: "CognitiveResponseContext",
                              strategy: str = "") -> str:
        """Pre-emission forward-model self-monitor (brief behavior 6).

        Inner speech / efference-copy analog (Pickering & Garrod; Yao 2025):
        before the utterance is "articulated", silently simulate it and compare
        the predicted percept to the conversational intent. If the candidate is
        degenerate, empty, or just echoes the user, refuse to emit it and fall
        back to a graceful, curiosity-preserving turn instead — exactly what a
        self-monitoring speaker does when the inner rehearsal doesn't match the
        intended message. This is the pre-articulation guard that
        conversational_repair.py only does post-hoc.

        Reuses the engine's own ``_is_word_salad`` degeneracy detector so the
        monitor shares one notion of "broken output" with the rest of the
        pipeline (no second, divergent definition). Returns the (possibly
        repaired) text unchanged when it passes.
        """
        # Counterfactual simulation (CSM) is a deliberately-constructed,
        # epistemic-hedged forward simulation, not free association — it cannot
        # be "degenerate/echo" in the sense this monitor targets. Exempt it so
        # the pre-articulation loop never discards a coherent "what would
        # happen" answer (its own coherence/tautology gates already ran).
        if strategy == "counterfactual_simulation":
            return text
        if not text or not text.strip():
            return self._human_like_uncertainty(ctx)[0]
        # Degenerate / word-salad candidate -> refuse. Clause-grained: any
        # sentence that is salad (subject+glue filler, truncated repetition,
        # vague-concept-metawords) fails the whole reply. This is the
        # Levelt/Wernicke pre-articulation inner-speech loop applied at the
        # SAME grain as production, so a degenerate clause emitted by ANY path
        # (SM decoder, decomposition sub-answer, or the ventral/dorsal
        # association binder) is intercepted before overt articulation — the
        # partial-comprehension lesion is closed at the articulation boundary,
        # uniformly, regardless of which production system produced the text.
        # Falls back to the whole-text check when the per-sentence detector is
        # unavailable. (Disabled by _disable_grounding_gate for A/B.)
        if not getattr(self, "_disable_grounding_gate", False):
            # Clause-grained repair (graded, not binary — Levelt prepairs):
            # drop ONLY the degenerate clause(s) and re-emit survivors. This is
            # the biology-faithful analog of covert self-repair; whole-utterance
            # withholding is reserved for the case where nothing survives.
            _stripped, _dropped = self._strip_degenerate_clauses(text, ctx)
            if _dropped:
                if _stripped and _stripped != text.strip():
                    if getattr(self, "_trace_enabled", False):
                        print("  [forward-model] dropped degenerate clause(s); "
                              "re-emitting survivors")
                    return _stripped
                # Nothing survivable -> honest uncertainty (fail-closed default).
                self._log_monitor_fire("forward-model", text.strip(), "all-clauses-degenerate")
                if getattr(self, "_trace_enabled", False):
                    print("  [forward-model] all clauses degenerate — "
                          "withholding")
                return self._human_like_uncertainty(ctx)[0]
            # Whole-text salad check (legacy fallback, when per-sentence misses).
            if _is_word_salad(text, subject=ctx.subject):
                self._log_monitor_fire("forward-model", text.strip(), "salad")
                if getattr(self, "_trace_enabled", False):
                    print("  [trace]   forward-model monitor: candidate failed "
                          "degeneracy check — repairing")
                return self._human_like_uncertainty(ctx)[0]
        else:
            if _is_word_salad(text, subject=ctx.subject):
                self._log_monitor_fire("forward-model", text.strip(), "salad")
                if getattr(self, "_trace_enabled", False):
                    print("  [trace]   forward-model monitor: candidate failed "
                          "degeneracy check — repairing")
                return self._human_like_uncertainty(ctx)[0]
        # Echo: reply is (near) verbatim the user's own words. A human doesn't
        # repeat the interlocutor as an answer; covert repair turns it back.
        user_norm = re.sub(r"\W+", " ", ctx.raw_input.lower()).strip()
        cand_norm = re.sub(r"\W+", " ", text.lower()).strip()
        if user_norm and cand_norm and (
                cand_norm == user_norm
                or cand_norm in user_norm
                or user_norm in cand_norm):
            self._log_monitor_fire("forward-model", text.strip(), "echo")
            if getattr(self, "_trace_enabled", False):
                print("  [trace]   forward-model monitor: candidate echoes "
                      "user input — repairing")
            refl = self._reflective_response(ctx)
            if refl:
                # _reflective_response returns (text, strategy); this monitor
                # must return a plain string.
                return refl[0] if isinstance(refl, tuple) else refl
            return self._human_like_uncertainty(ctx)[0]
        return text


