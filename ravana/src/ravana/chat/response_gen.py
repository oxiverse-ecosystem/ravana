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
# Compute project root (same logic as engine.py)
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ravana_ml.nn.neural_decoder import NeuralDecoder

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
            self.neural_decoder = NeuralDecoder(
                vocab_size=vocab_size,
                embed_dim=self.dim,
                hidden_dim=256,
                n_attention_heads=4,
                contrastive_weight=0.5,
                contrastive_negatives=8,
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

        # Common function words for fluent generation
        function_words = [
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
            "of", "to", "for", "with", "from", "at", "by", "as", "on",
            "related", "connect", "connects", "mean", "means",
            "concept", "concepts", "idea", "ideas", "important",
            "lead", "leads", "talk", "think", "link", "links",
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
        self.neural_decoder = NeuralDecoder(
            vocab_size=vocab_size,
            embed_dim=self.dim,
            hidden_dim=256,
            n_attention_heads=4,
            contrastive_weight=0.5,
            contrastive_negatives=8,
        )

        for word, idx in self._decoder_word_to_idx.items():
            if word in self._decoder_word_to_embed:
                self.neural_decoder.word_embedding.weight.data[idx] = \
                    self._decoder_word_to_embed[word]
        self.neural_decoder.rebuild_vocab_cache()
        self._decoder_vocab_built = True
        self._needs_seed_training = True
        self._needs_synthetic_training = True



    def _train_decoder_from_graph(self, min_synthetic: int = 2000):
        """Train neural decoder on natural sentences derived from graph relationships.

        Uses diverse, human-like phrasings instead of template artifacts like
        "X connects with Y" that were poisoning the decoder's output.
        
        Args:
            min_synthetic: Maximum number of synthetic sentences to generate (default 2000)
        """
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return
        # Natural phrasings that sound like human conversation, not graph templates
        templates = [
            # Casual / conversational
            "{s} and {o} go hand in hand",
            "when you think about {s}, {o} comes to mind",
            "{s} is a big part of {o}",
            "{s} has a lot to do with {o}",
            "{o} matters a lot when it comes to {s}",

            # Explanatory
            "{s} is really about {o}",
            "at its core, {s} ties into {o}",
            "{s} plays a role in {o}",
            "one way to think about {s} is through {o}",
            "{s} shapes how we understand {o}",

            # Causal / influence
            "{s} can lead to {o}",
            "{o} often happens because of {s}",
            "{s} has an impact on {o}",
            "{s} contributes to {o} in important ways",
            "{o} is something that {s} can bring about",

            # Contrastive
            "{s} and {o} are quite different",
            "while {s} is one thing, {o} is another",
            "{s} stands apart from {o} in interesting ways",
            "unlike {s}, {o} takes a different path",

            # Analogical
            "{s} is a lot like {o} in some ways",
            "you can see {s} reflected in {o}",
            "{s} and {o} share something important",
            "thinking of {s} reminds me of {o}",

            # Temporal / process
            "{s} often comes before {o}",
            "{o} tends to follow {s}",
            "{s} sets the stage for {o}",
            "{s} paves the way for {o}",

            # Speculative / reflective
            "maybe {s} and {o} are more related than we think",
            "it is interesting how {s} ties into {o}",
            "{s} makes you wonder about {o}",
            "in a way, {s} and {o} feed into each other",

            # Personal / first person
            "i think {s} is closely tied to {o}",
            "to me, {s} says something about {o}",
            "i see {s} and {o} as deeply connected",
        ]

        sentences = []
        seen = set()
        for (src_id, tgt_id), edge in self.graph.edges.items():
            src_node = self.graph.get_node(src_id)
            tgt_node = self.graph.get_node(tgt_id)
            if src_node and tgt_node and src_node.label and tgt_node.label:
                s_label = src_node.label.lower()
                o_label = tgt_node.label.lower()
                if s_label == o_label:
                    continue
                key = (s_label, o_label)
                if key not in seen:
                    seen.add(key)
                    t = templates[len(sentences) % len(templates)]
                    sent = t.format(s=s_label, o=o_label)
                    sentences.append(sent)

        # Limit synthetic training to avoid overwhelming real English patterns
        sentences = sentences[:min_synthetic]
        total_trained = 0
        sleep_every = 200
        for sent in sentences:
            words = sent.split()
            err = self.neural_decoder.train_on_sentence(
                words, self._decoder_word_to_embed, self._decoder_word_to_idx)
            total_trained += 1
            if total_trained % sleep_every == 0:
                self.neural_decoder.sleep_cycle()
        self.neural_decoder.sleep_cycle()
        self._decoder_training_count = total_trained + self._decoder_web_training_count + self._decoder_seed_training_count
        print(f"  [Decoder] Trained on {total_trained} natural graph sentences"
              f"{' + ' + str(self._decoder_web_training_count) + ' from web' if self._decoder_web_training_count > 0 else ''}")



    def _train_decoder_on_seed_corpus(self, max_sentences: int = 250) -> int:
        # Train the neural decoder on the bundled real-English corpus
        # (data/corpora/teen_seeds.txt). This is the difference between
        # a decoder that has seen only {s} connects with {o} template
        # garbage and a decoder that has actually seen English. Without
        # this, the decoder learns to imitate the very template phrases
        # that produce the trash output in chat.
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
        # Pre-expand decoder vocab with corpus words so they can be
        # used as inputs/targets (GloVe-backed vectors).
        words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", text.lower()))
        new_for_vocab = [w for w in words_in_corpus if w not in self._decoder_word_to_idx]
        if new_for_vocab:
            self._expand_decoder_vocab(new_for_vocab)

        # Pre-process corpus once, then loop with shuffled passes
        nd = self.neural_decoder
        all_sentences = nd.prepare_sentences(
            text, self._decoder_word_to_embed, self._decoder_word_to_idx,
            min_sentence_len=3,
        )
        n_available = len(all_sentences)
        sentences_per_pass = min(200, n_available)
        n_passes = 20
        sleep_interval = 5

        total_err = 0.0
        passes = 0
        rng = np.random.RandomState(42)

        for i in range(n_passes):
            idx = rng.permutation(n_available)
            batch = [all_sentences[idx[j]] for j in range(sentences_per_pass)]

            err_sum = 0.0
            for sent in batch:
                err = nd.train_on_sentence(
                    sent['words'], self._decoder_word_to_embed, self._decoder_word_to_idx,
                    word_indices=sent['word_indices'],
                    conditioning_embs=sent['conditioning_embs'],
                )
                err_sum += err
            avg_err = err_sum / sentences_per_pass
            total_err += avg_err
            passes += sentences_per_pass

            if (i + 1) % sleep_interval == 0:
                nd.sleep_cycle()

        if (n_passes % sleep_interval) != 0:
            nd.sleep_cycle()

        self._decoder_seed_training_count = passes
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

            # Resize output_proj weight and bias to match new vocab size
            old_out_w = self.neural_decoder.output_proj.weight.data
            new_out_w = np.zeros((new_vocab_size, self.neural_decoder.hidden_dim), dtype=np.float32)
            new_out_w[:old_vocab_size] = old_out_w
            new_out_proj = Linear(self.neural_decoder.hidden_dim, new_vocab_size)
            new_out_proj.weight.data = new_out_w
            if self.neural_decoder.output_proj.bias is not None:
                old_bias = self.neural_decoder.output_proj.bias.data
                new_bias = np.zeros(new_vocab_size, dtype=np.float32)
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

        # Mini training pass on new words using batch similarity
        for nw in new_words:
            nwl = nw.lower().strip()
            nidx = self._decoder_word_to_idx.get(nwl)
            if nidx is None:
                continue
            nvec = self._decoder_word_to_embed.get(nwl)
            if nvec is None:
                continue
            if len(existing_labels) > 0:
                sims = existing_mat @ nvec
                above_thresh = np.where(sims > 0.4)[0]
                if len(above_thresh) > 0:
                    top_k = min(3, len(above_thresh))
                    top_order = np.argsort(sims[above_thresh])[::-1][:top_k]
                    for ni in above_thresh[top_order]:
                        neighbor = existing_labels[ni]
                        bridge = f"{nwl} and {neighbor} are related"
                        self.neural_decoder.train_on_sentence(
                            bridge.split(), self._decoder_word_to_embed, self._decoder_word_to_idx)

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
            return None

        subject = ctx.subject
        if not subject or subject.lower() not in self._decoder_word_to_idx:
            return None

        # Build conditioning embeddings from graph concepts
        concept_embs = []
        seen_labels: Set[str] = set()

        # Subject embedding (core concept) — lock graph reads
        with self._graph_lock:
            subj_lower = subject.lower()
            if subj_lower in self._concept_keywords:
                nids = self._concept_keywords[subj_lower]
                node = self.graph.get_node(nids[0])
                if node and node.vector is not None:
                    concept_embs.append(node.vector.copy())
                    seen_labels.add(subj_lower)

            # Associated concepts with weighted significance
            for label, score in ctx.associated_concepts[:6]:
                ll = label.lower()
                if ll not in seen_labels and ll in self._concept_keywords:
                    nids = self._concept_keywords[ll]
                    node = self.graph.get_node(nids[0])
                    if node and node.vector is not None:
                        weight = 0.5 + min(score, 1.0) * 0.5
                        concept_embs.append(node.vector.copy() * weight)
                        seen_labels.add(ll)
                        if len(concept_embs) >= 6:
                            break

        # Fallback: use decoder word embeddings if no graph nodes found
        if len(concept_embs) < 1:
            if subj_lower in self._decoder_word_to_embed:
                concept_embs.append(self._decoder_word_to_embed[subj_lower].copy())
            else:
                return None

        # Add sentence-level compositional vector (N400/P600 integration) as conditioning
        sent_vec = getattr(ctx, 'sentence_vector', None)
        if sent_vec is not None and np.any(sent_vec != 0):
            concept_embs.append(sent_vec.astype(np.float32) * 0.6)

        conditioning_embs = np.stack(concept_embs, axis=0).astype(np.float32)

        # Special token indices
        bos_idx = self._decoder_word_to_idx.get("<bos>", 0)
        eos_idx = self._decoder_word_to_idx.get("<eos>", 2)

        # Dynamic temperature from cognitive state
        # Stage 2: Lower base temp for 1500-vocab model (more deterministic → content words)
        base_temp = 0.35
        arousal = ctx.arousal if hasattr(ctx, 'arousal') else 0.3
        temp = base_temp * (0.7 + arousal * 0.6)
        # Higher dopamine tone = more creative/exploratory
        dt = getattr(self, '_dopamine_tone', 0.5)
        temp *= (0.7 + dt * 0.6)
        temp = max(0.15, min(0.7, temp))

        # Build content word IDs (non-function, non-special tokens)
        # IMPORTANT: Do NOT include semantic connector words like "connects",
        # "relates", "links", "leads" here — those are content words whose
        # presence in the function-word set was causing the decoder to default
        # to template-like "X connects with Y" patterns.
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
            "of", "to", "for", "with", "from", "at", "by", "as", "on",
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
                max_steps=28,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                temperature=temp,
                cerebellar_ngram=self.cerebellar_ngram,
                idx_to_word=self._decoder_idx_to_word,
                basal_ganglia=self.basal_ganglia,
                content_word_ids=content_word_ids,
                token_boost=subject_boost,
            )

            if not generated or len(generated) < 3:
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

            # Quality gate: if >60% of output is function words, fall through
            # to syntactic pipeline or reasoning loop (decoder not ready yet)
            func_set = {"a","an","the","is","are","was","were","be","been",
                "being","have","has","had","do","does","did","will",
                "would","could","should","may","might","shall","can",
                "not","no","nor","so","if","then","than","too","very",
                "just","about","also","into","over","after","before",
                "between","through","during","because","while","which",
                "who","whom","what","when","where","why","how","all",
                "each","every","both","few","more", "most", "some", "any",
                "this", "that", "these", "those", "it", "its", "they", "them",
                "their", "we", "our", "you", "your", "he", "she", "him", "her",
                "his", "i", "me", "my", "myself", "am",
                "of", "to", "for", "with", "from", "at", "by", "as", "on"}
            func_count = sum(1 for w in words if w.lower() in func_set)
            if len(words) > 0 and func_count / len(words) > 0.70:
                return None

            # Template-pattern gate: reject ONLY egregious graph artifacts.
            # Relaxed: "leads to", "drives", "refers" are legitimate English.
            template_verbs = {"connects", "connect", "relates", "relate", "links",
                               "link", "associated"}
            template_preps = {"with", "into"}
            text_lower = text.lower()
            template_rejected = False
            for tv in template_verbs:
                for tp in template_preps:
                    if re.search(rf'\b\w+\s+' + tv + r'\s+' + tp + r'\s+\w+', text_lower):
                        template_rejected = True
                        break
                if template_rejected:
                    break
            # Only reject if template pattern AND output is short (< 12 words)
            if template_rejected and len(words) < 12:
                return None

            # Bigram repetition gate: reject if any bigram appears 3+ times
            if len(words) >= 4:
                bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
                from collections import Counter
                bg_counts = Counter(bigrams)
                if any(c >= 3 for c in bg_counts.values()):
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

        # Case 1: We know both concepts
        if a_nids and b_nids:
            shared = set(a_assocs).intersection(b_assocs)
            unique_a = [w for w in a_assocs if w not in shared]
            unique_b = [w for w in b_assocs if w not in shared]
            
            response_parts = []
            if shared:
                response_parts.append(f"both {concept_a} and {concept_b} are connected to {', '.join(list(shared)[:2])}.")
            
            a_desc = f"{concept_a} is tied to {', '.join(unique_a[:2])}" if unique_a else f"{concept_a} has its own unique connections"
            b_desc = f"{concept_b} relates to {', '.join(unique_b[:2])}" if unique_b else f"{concept_b} has different links"
            
            response_parts.append(f"however, while {a_desc}, {b_desc}.")
            return " ".join(response_parts)
            
        # Case 2: We only know concept A
        elif a_nids:
            desc = f"associated with {', '.join(a_assocs[:2])}" if a_assocs else "a concept in my graph"
            return f"i know that {concept_a} is {desc}, but i haven't fully learned about {concept_b} yet to compare them."
            
        # Case 3: We only know concept B
        elif b_nids:
            desc = f"associated with {', '.join(b_assocs[:2])}" if b_assocs else "a concept in my graph"
            return f"i know that {concept_b} is {desc}, but i haven't fully learned about {concept_a} yet to compare them."
            
        # Case 4: We know neither
        return f"i haven't learned enough about either {concept_a} or {concept_b} yet. how would you compare them?"



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




    def _generate_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Generate response using syntactic pipeline (primary), neural decoder
        (when proven fluent), then reasoning loop (last resort).

        Order reflects neuroscience:
        - Syntactic pipeline = P600-driven compositional integration (primary route)
        - Neural decoder = overlearned fluent speech (requires high quality)
        - Reasoning loop = web learning when knowledge is absent

        Neuroscience basis:
        - Brouwer Retrieval-Integration: composition (P600) is the default route
        - PMC 2023: hierarchical predictive coding across multiple timescales
        - Nature Human Behaviour 2025: multi-timescale discourse organization
        """
        # Social/Chitchat path: bypass reasoning, definition lookup, and standard decoder gates
        qtype, _ = self.pfc_workspace.detect_question_type(ctx.raw_input, concept_pos=self._concept_pos)
        if qtype in ("greeting", "wellbeing", "capability", "farewell"):
            try:
                syntax_response = self._generate_with_decoder_and_syntax(ctx)
                if syntax_response:
                    return (syntax_response, "syntactic_pipeline")
            except Exception:
                pass

        subject = ctx.subject
        assocs = ctx.associated_concepts

        # Determine if the query is seeking factual/informational definition
        is_query_for_defn = self._is_informational_query(ctx.raw_input, subject)

        # Prioritize reasoning loop for questions or comparison queries (gets cognitive answers/fallbacks)
        text_lower = ctx.raw_input.lower().strip()
        is_question = ctx.raw_input.strip().endswith('?') or any(w in text_lower for w in ["what", "who", "where", "when", "why", "how", "define", "explain", "describe", "tell me about", "which"])
        is_comparison = self._detect_comparison_concepts(ctx.raw_input) is not None

        # Reasoning loop: only for queries that truly need web search
        # Apply _needs_web_search to ALL branches so the neural decoder gets a chance
        needs_search = is_query_for_defn or is_question or is_comparison
        if needs_search and self._needs_web_search(subject):
            reasoned_res, reasoned_strat = self._reasoning_loop(ctx)
            if reasoned_res:
                return (reasoned_res, reasoned_strat)

        # Hippocampal buffer recall check (Issues #7-8) — check BEFORE decoder/syntactic pipeline
        # so factual memories are retrieved even when the pipeline would generate a fluent response.
        # This mirrors hippocampal pattern completion: from partial cue → full episodic trace.
        try:
            hippocampal_result = self._try_hippocampal_retrieval(ctx)
            if hippocampal_result:
                return (hippocampal_result, "hippocampal_recall")
        except Exception:
            pass

        # Fallback to local stored definition (ATL convergence zone - Binder & Desai 2011)
        if is_query_for_defn and subject and subject.lower() in self._definitions:
            defn = self._definitions[subject.lower()]
            if defn and len(defn) > 10:
                defn_clean = defn.rstrip(" .!?")
                subj_disp = self._capitalize_subject(subject, ctx.raw_input)
                framings = [
                    f"{subj_disp} is {defn_clean}.",
                    f"From what I know, {subj_disp} is {defn_clean}.",
                    f"I have learned that {subj_disp} is {defn_clean}.",
                ]
                response = random.choice(framings)
                return (response, "definition_store")

        # Path 1: Neural decoder (fluent, overlearned speech) - check this first
        decoder_ready = False
        if self.neural_decoder is not None and self._decoder_vocab_built:
            nd = self.neural_decoder
            ce_ok = nd._avg_cross_entropy < 5.0 if nd._metric_examples > 10 else False
            t1_ok = nd._avg_top1_acc > 0.08 if nd._metric_examples > 10 else False
            trained_enough = self._decoder_training_count >= 500
            decoder_ready = ce_ok and t1_ok and trained_enough
        if decoder_ready:
            try:
                decoder_response = self._generate_with_decoder(ctx)
                if decoder_response and len(decoder_response) > 10 and not _is_word_salad(decoder_response):
                    return (decoder_response, "neural_decoder")
            except Exception:
                pass

        # Path 2: Syntactic pipeline (P600 compositional integration) — fallback when decoder is not ready/fails
        try:
            syntax_response = self._generate_with_decoder_and_syntax(ctx)
            if syntax_response and len(syntax_response) > 10 and not _is_word_salad(syntax_response):
                # Simple quality check: reject only if too short or empty
                _is_short = len(syntax_response.strip()) < 15
                if not _is_short:
                    return (syntax_response, "syntactic_pipeline")
                # Template detected - fall through to definition/graph fallback
        except Exception:
            pass

        # Path 3: Graph fallback (last resort — simple associative response)
        # Check for pragmatic implicature (Issue #6)
        try:
            implicature = self.implicature_detector.analyze(ctx.raw_input)
            if implicature.is_implicature and implicature.suggested_question_type == "acknowledge":
                return (self._generate_acknowledgment(ctx, implicature), "implicature_acknowledge")
        except Exception:
            pass

        # Check for nested propositions (Issue #3)
        try:
            if self.proposition_parser.has_nested_propositions(ctx.raw_input):
                propositions = self.proposition_parser.extract_propositions(ctx.raw_input)
                if len(propositions) >= 2:
                    surface = self.proposition_parser.get_surface_structure(propositions)
                    if surface == "contrastive":
                        p1 = propositions[0]
                        p2 = propositions[1] if len(propositions) > 1 else None
                        if p2:
                            return (f"On one hand, {p1.subject} {p1.predicate} {p1.object}. On the other hand, {p2.subject} {p2.predicate} {p2.object}.", "nested_proposition_contrastive")
                        return (f"I think there are multiple perspectives here. {p1.subject} may {p1.predicate} {p1.object} depending on the context.", "nested_proposition")
                    elif surface == "multi_perspective":
                        parts = []
                        for p in propositions[:3]:
                            parts.append(f"{p.subject} {p.predicate} {p.object}")
                        return (" and also ".join(parts) + ".", "nested_proposition_multi")
        except Exception:
            pass

        # Check for quantity comparison (Issue #5)
        try:
            if hasattr(self, '_pending_quantity_result') and self._pending_quantity_result:
                qa, qb, q_result, q_conf = self._pending_quantity_result
                if q_result == "equal":
                    return (f"{qa.concept.capitalize()} and {qb.concept} have the same quantity. They are equal.", "quantity_comparison")
                elif q_result == "a_greater":
                    return (f"{qa.concept.capitalize()} has more than {qb.concept} ({qa.value} vs {qb.value}).", "quantity_comparison")
                elif q_result == "b_greater":
                    return (f"{qb.concept.capitalize()} has more than {qa.concept} ({qb.value} vs {qa.value}).", "quantity_comparison")
        except Exception:
            pass

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


