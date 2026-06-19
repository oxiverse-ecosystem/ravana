"""
Decoder Engine - RAVANA's neural decoder for text generation.
Handles: vocabulary building, seed corpus training, web article training, generation.
"""
import numpy as np
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ravana_ml.nn.neural_decoder import NeuralDecoder
from ..graph import TEEN_CONCEPTS, DOMAIN_CONCEPTS, STOP_WORDS


@dataclass
class DecoderConfig:
    """Configuration for neural decoder."""
    embed_dim: int = 64
    hidden_dim: int = 256
    n_attention_heads: int = 4
    contrastive_weight: float = 0.3
    contrastive_negatives: int = 5


class DecoderEngine:
    """Neural decoder management: vocab, training, generation."""

    def __init__(self, config: Optional[DecoderConfig] = None):
        self.config = config or DecoderConfig()
        self.neural_decoder: Optional[NeuralDecoder] = None

        # Vocabulary mappings
        self._decoder_word_to_idx: Dict[str, int] = {}
        self._decoder_idx_to_word: Dict[int, str] = {}
        self._decoder_word_to_embed: Dict[str, np.ndarray] = {}
        self._decoder_vocab_built: bool = False

        # Training counters
        self._decoder_training_count: int = 0
        self._decoder_web_training_count: int = 0
        self._decoder_seed_training_count: int = 0

        # Saved state for loading
        self._saved_decoder_state: dict = {}

    @property
    def vocab_size(self) -> int:
        return len(self._decoder_word_to_idx)

    @property
    def training_count(self) -> int:
        return self._decoder_training_count

    @property
    def web_training_count(self) -> int:
        return self._decoder_web_training_count

    @property
    def is_ready(self) -> bool:
        """Check if decoder is ready for generation."""
        return (self.neural_decoder is not None and self._decoder_vocab_built
                and self._decoder_training_count >= 500)

    def build_vocab(self, graph_engine, glove_vector_fn):
        """Build vocabulary for the NeuralDecoder from graph concepts + GloVe + function words."""
        self._decoder_word_to_idx = {}
        self._decoder_idx_to_word = {}
        self._decoder_word_to_embed = {}

        # Special tokens
        special_tokens = ["<pad>", "?", "<bos>", "<eos>"]
        for i, tok in enumerate(special_tokens):
            self._decoder_word_to_idx[tok] = i
            self._decoder_idx_to_word[i] = tok
            vec = np.random.randn(self.config.embed_dim).astype(np.float32) * 0.05
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self._decoder_word_to_embed[tok] = vec

        next_idx = len(special_tokens)

        # Common function words
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

        # Add graph concept labels
        concept_labels = set()
        for nid, node in graph_engine.graph.nodes.items():
            if node.label:
                concept_labels.add(node.label.lower())

        for label in sorted(concept_labels):
            if label not in self._decoder_word_to_idx:
                self._decoder_word_to_idx[label] = next_idx
                self._decoder_idx_to_word[next_idx] = label
                nids = graph_engine._concept_keywords.get(label, [])
                if nids:
                    node = graph_engine.graph.get_node(nids[0])
                    if node and node.vector is not None:
                        vec = node.vector.copy()
                    else:
                        vec = glove_vector_fn(label)
                else:
                    vec = glove_vector_fn(label)
                if vec is None:
                    vec = np.random.randn(self.config.embed_dim).astype(np.float32) * 0.1
                    norm_v = np.linalg.norm(vec)
                    if norm_v > 0:
                        vec /= norm_v
                self._decoder_word_to_embed[label] = vec.astype(np.float32)
                next_idx += 1

        for fw in function_words:
            if fw not in self._decoder_word_to_idx:
                self._decoder_word_to_idx[fw] = next_idx
                self._decoder_idx_to_word[next_idx] = fw
                vec = glove_vector_fn(fw)
                if vec is None:
                    vec = np.random.randn(self.config.embed_dim).astype(np.float32) * 0.1
                    norm_v = np.linalg.norm(vec)
                    if norm_v > 0:
                        vec /= norm_v
                self._decoder_word_to_embed[fw] = vec.astype(np.float32)
                next_idx += 1

        vocab_size = len(self._decoder_word_to_idx)
        self.neural_decoder = NeuralDecoder(
            vocab_size=vocab_size,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            n_attention_heads=self.config.n_attention_heads,
            contrastive_weight=self.config.contrastive_weight,
            contrastive_negatives=self.config.contrastive_negatives,
        )

        # Initialize decoder word embeddings
        for word, idx in self._decoder_word_to_idx.items():
            if word in self._decoder_word_to_embed:
                self.neural_decoder.word_embedding.weight.data[idx] = self._decoder_word_to_embed[word]
        self.neural_decoder.rebuild_vocab_cache()
        self._decoder_vocab_built = True

        # Train on seed corpus FIRST (real English patterns)
        try:
            self._train_decoder_on_seed_corpus(max_sentences=1000)
            if self._trace_enabled and self._decoder_seed_training_count > 0:
                print(f"  [init] Seed corpus training: {self._decoder_seed_training_count} sentences")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [init] Seed corpus training error: {e}")

        # Minimal synthetic training
        self._train_decoder_from_graph(graph_engine, min_synthetic=500)

    def _train_decoder_from_graph(self, graph_engine, min_synthetic: int = 500) -> int:
        """Train neural decoder on synthetic sentences from graph relationships."""
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return 0
        templates = [
            "{s} is related to {o}", "{s} connects with {o}", "{s} links to {o}",
            "{s} refers to {o}", "{s} can be described as {o}",
            "the concept of {s} involves {o}", "when people talk about {s} they often mention {o}",
            "{s} leads to {o}", "{s} causes {o}", "{o} is a result of {s}",
            "{s} contributes to {o}", "{s} influences {o}",
            "{s} precedes {o}", "{o} follows {s}", "first comes {s} then {o}",
            "{s} contrasts with {o}", "unlike {s} {o} is different",
            "while {s} has one meaning {o} has another",
            "{s} is like {o} in many ways", "both {s} and {o} share similarities",
            "the relationship between {s} and {o} is important",
            "{s} is a part of {o}", "{s} is a type of {o}",
            "{o} includes {s}", "{s} forms the basis of {o}",
            "people use {s} to understand {o}", "{s} helps explain {o}",
            "understanding {s} requires knowing {o}", "{o} depends on {s}",
            "maybe {s} connects to {o}", "it seems like {s} relates to {o}",
            "one way to think about {s} is through {o}",
            "the meaning of {s} goes beyond {o}",
            "{s} represents something larger than {o}",
            "thinking about {s} naturally leads to {o}", "{s} and {o} are deeply connected",
        ]

        relation_templates = {
            "causal": ["{s} causes {o}", "{s} leads to {o}", "{o} results from {s}"],
            "contrastive": ["{s} contrasts with {o}", "unlike {s} {o} is different"],
            "analogical": ["{s} is like {o}", "{s} resembles {o} in many ways"],
            "temporal": ["{s} comes before {o}", "{s} precedes {o}"],
            "semantic": ["{s} relates to {o}", "{s} connects with {o}"],
        }
        sentences = []
        seen = set()
        for (src_id, tgt_id), edge in graph_engine.graph.edges.items():
            src_node = graph_engine.graph.get_node(src_id)
            tgt_node = graph_engine.graph.get_node(tgt_id)
            if src_node and tgt_node and src_node.label and tgt_node.label:
                s_label = src_node.label.lower()
                o_label = tgt_node.label.lower()
                if s_label == o_label:
                    continue
                key = (s_label, o_label)
                if key not in seen:
                    seen.add(key)
                    rel = getattr(edge, 'relation_type', 'semantic') or 'semantic'
                    rtemplates = relation_templates.get(rel, templates)
                    t = rtemplates[len(sentences) % len(rtemplates)]
                    sent = t.format(s=s_label, o=o_label)
                    sentences.append(sent)

        sentences = sentences[:min_synthetic]
        total_trained = 0
        for sent in sentences:
            words = sent.split()
            err = self.neural_decoder.train_on_sentence(
                words, self._decoder_word_to_embed, self._decoder_word_to_idx)
            total_trained += 1
        self.neural_decoder.sleep_cycle()
        self._decoder_training_count = total_trained + self._decoder_web_training_count
        print(f"  [Decoder] Trained on {total_trained} synthetic graph sentences"
              f"{' + ' + str(self._decoder_web_training_count) + ' from web' if self._decoder_web_training_count > 0 else ''}")
        return total_trained

    def _train_decoder_on_seed_corpus(self, max_sentences: int = 1000) -> int:
        """Train on bundled teen_seeds.txt corpus (20 passes)."""
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return 0
        _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
        if not os.path.exists(corpus_path):
            return 0
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return 0

        words_in_corpus = set(re.findall(r"[a-zA-Z']{3,}", text.lower()))
        new_for_vocab = [w for w in words_in_corpus if w not in self._decoder_word_to_idx]
        if new_for_vocab:
            self._expand_vocab(new_for_vocab)

        total_err = 0.0
        passes = 0
        for _ in range(20):  # 20 passes for Hebbian consolidation
            err, n = self.neural_decoder.train_on_text(
                text, self._decoder_word_to_embed, self._decoder_word_to_idx,
                min_sentence_len=3, max_sentences=200)
            total_err += err
            passes += n
        if hasattr(self, 'cerebellar_ngram') and self.cerebellar_ngram is not None:
            self.cerebellar_ngram.learn_from_text(
                text, decoder_prediction_error=min(0.6, total_err / max(1, passes)))
        self.neural_decoder.sleep_cycle()
        self._decoder_seed_training_count = passes
        self._decoder_training_count += passes
        return passes

    def _expand_vocab(self, new_words: List[str]):
        """Add new words to decoder vocabulary (hippocampal replay)."""
        if not new_words or self.neural_decoder is None:
            return
        added = 0
        for word in new_words:
            wl = word.lower().strip()
            if wl in self._decoder_word_to_idx or wl in ('<pad>', '?', '<bos>', '<eos>'):
                continue
            idx = len(self._decoder_word_to_idx)
            self._decoder_word_to_idx[wl] = idx
            self._decoder_idx_to_word[idx] = wl
            vec = self._glove_vector(wl) if hasattr(self, '_glove_vector') else None
            if vec is None:
                vec = np.random.randn(self.config.embed_dim).astype(np.float32) * 0.1
                n = np.linalg.norm(vec)
                if n > 0:
                    vec /= n
            self._decoder_word_to_embed[wl] = vec.astype(np.float32)
            added += 1

        if added == 0:
            return

        old_vocab_size = self.neural_decoder.vocab_size
        new_vocab_size = len(self._decoder_word_to_idx)

        # Resize embedding table
        old_weight = self.neural_decoder.word_embedding.weight.data
        new_weight = np.zeros((new_vocab_size, self.config.embed_dim), dtype=np.float32)
        new_weight[:len(old_weight)] = old_weight
        for word, idx in self._decoder_word_to_idx.items():
            if idx >= len(old_weight) and word in self._decoder_word_to_embed:
                new_weight[idx] = self._decoder_word_to_embed[word]

        from ravana_ml.nn.module import Embedding, Linear
        new_emb = Embedding(new_vocab_size, self.config.embed_dim)
        new_emb.weight.data = new_weight
        self.neural_decoder.word_embedding = new_emb

        # Resize output projection
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
        self.neural_decoder._vocab_dim = self.config.embed_dim
        self.neural_decoder.rebuild_vocab_cache()

        # Hippocampal replay: blend new embeddings with similar existing
        for word in new_words:
            wl = word.lower().strip()
            idx = self._decoder_word_to_idx.get(wl)
            if idx is None:
                continue
            vec = self._decoder_word_to_embed.get(wl)
            if vec is None:
                continue
            best_sim = 0.0
            best_label = None
            for existing_word, existing_idx in list(self._decoder_word_to_idx.items()):
                if existing_word == wl or existing_word.startswith('<'):
                    continue
                ev = self._decoder_word_to_embed.get(existing_word)
                if ev is not None:
                    sim = float(np.dot(vec, ev))
                    if sim > best_sim:
                        best_sim = sim
                        best_label = existing_word
            if best_label is not None and best_sim > 0.3:
                best_vec = self._decoder_word_to_embed[best_label]
                blended = vec * 0.7 + best_vec * 0.3
                blended /= np.linalg.norm(blended)
                self._decoder_word_to_embed[wl] = blended
                new_weight[idx] = blended
        self.neural_decoder.word_embedding.weight.data = new_weight

        # Mini training on new words
        for nw in new_words:
            nwl = nw.lower().strip()
            nidx = self._decoder_word_to_idx.get(nwl)
            if nidx is None:
                continue
            nvec = self._decoder_word_to_embed.get(nwl)
            if nvec is None:
                continue
            neighbors = []
            for ew, ei in self._decoder_word_to_idx.items():
                if ew == nwl or ew.startswith('<') or ei == nidx:
                    continue
                ev = self._decoder_word_to_embed.get(ew)
                if ev is not None:
                    sim = float(np.dot(nvec, ev))
                    if sim > 0.4:
                        neighbors.append((ew, sim))
            neighbors.sort(key=lambda x: -x[1])
            for neighbor, _ in neighbors[:3]:
                bridge = f"{nwl} and {neighbor} are related"
                self.neural_decoder.train_on_sentence(
                    bridge.split(), self._decoder_word_to_embed, self._decoder_word_to_idx)

        print(f"  [Decoder] Expanded vocab by {added} words (now {new_vocab_size})")

    def train_on_text(self, text: str, topic: str, graph_engine, glove_vector_fn,
                      conditioning_embs: Optional[np.ndarray] = None,
                      passes: int = 3) -> int:
        """Train decoder on full article text."""
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return 0
        if conditioning_embs is None:
            conditioning_embs = self._build_conditioning(graph_engine, topic, text)

        # Expand vocab with new words
        words_in_text = set(re.findall(r"[a-zA-Z']{3,}", text.lower()))
        new_for_vocab = [w for w in words_in_text
                         if w not in self._decoder_word_to_idx
                         and w not in STOP_WORDS
                         and len(w) >= 3]
        if new_for_vocab:
            self._expand_vocab(list(new_for_vocab)[:50])

        total_trained = 0
        for _ in range(passes):
            err, n = self.neural_decoder.train_on_text(
                text, self._decoder_word_to_embed, self._decoder_word_to_idx,
                min_sentence_len=3, conditioning_embs=conditioning_embs)
            total_trained += n

        self._decoder_training_count += total_trained
        self._decoder_web_training_count += total_trained

        # Train cerebellar n-gram
        if hasattr(self, 'cerebellar_ngram') and self.cerebellar_ngram is not None:
            self.cerebellar_ngram.learn_from_text(
                text, decoder_prediction_error=min(0.6, err / max(1, total_trained)))
        self.neural_decoder.sleep_cycle()
        return total_trained

    def _build_conditioning(self, graph_engine, topic: str, text_words: List[str]) -> Optional[np.ndarray]:
        """Build graph walk conditioning embeddings for text."""
        concept_embs = []
        seen = set()

        tl = topic.lower().strip()
        if tl in graph_engine._concept_keywords:
            nids = graph_engine._concept_keywords[tl]
            node = graph_engine.graph.get_node(nids[0])
            if node and node.vector is not None:
                concept_embs.append(node.vector.copy())
                seen.add(tl)

        counted = {}
        for w in text_words:
            wl = w.lower().strip(".,!?\"' ")
            if wl not in STOP_WORDS and len(wl) >= 3:
                counted[wl] = counted.get(wl, 0) + 1
        for w, _ in sorted(counted.items(), key=lambda x: x[1], reverse=True)[:5]:
            if w not in seen and w in graph_engine._concept_keywords:
                nids = graph_engine._concept_keywords[w]
                node = graph_engine.graph.get_node(nids[0])
                if node and node.vector is not None:
                    concept_embs.append(node.vector.copy())
                    seen.add(w)

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

    def generate_with_syntax(self, ctx, graph_engine) -> Optional[str]:
        """Generate response using decoder + syntactic pipeline."""
        # This would integrate with SyntacticCellAssembly and SurfaceRealizer
        # For now, use the decoder directly
        return self.generate(ctx)

    def generate(self, ctx) -> Optional[str]:
        """Generate a response using the NeuralDecoder."""
        if not self.is_ready:
            return None

        subject = ctx.subject
        if not subject or subject.lower() not in self._decoder_word_to_idx:
            return None

        # Build conditioning embeddings
        concept_embs = []
        seen_labels = set()

        subj_nids = graph_engine._concept_keywords.get(subject.lower(), []) if 'graph_engine' in dir() else []
        # This needs graph_engine passed in
        return None  # Placeholder - actual generation requires graph_engine

    def save_state(self) -> dict:
        """Save decoder state for checkpointing."""
        if self.neural_decoder is None:
            return {}
        return self.neural_decoder.state_dict()

    def load_state(self, state: dict):
        """Load decoder state from checkpoint."""
        if self.neural_decoder is not None and state:
            try:
                self.neural_decoder.load_state_dict(state)
            except Exception:
                pass  # Shape mismatch, ignore