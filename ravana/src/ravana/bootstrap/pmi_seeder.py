"""
PMI-based corpus bootstrapper for RAVANA.
Replaces hardcoded TEEN_CONCEPTS, semantic_edges, causal_edges, etc.
with corpus-driven Pointwise Mutual Information statistics.

The brain doesn't come with pre-wired semantic networks — infants learn
vocabulary through statistical learning (transitional probabilities,
cross-situational statistics). This module mimics that process by
extracting concepts and relations from a text corpus using PMI.

Neuroscience basis:
- Lany & Saffran (2011): infants track transitional probabilities to
  segment speech and map words to meanings.
- Beckage et al. (2011): semantic networks form small-world structure
  through exposure — structure comes from input statistics, not innate seeds.
"""
import os
import re
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional

# Default stop words for filtering
_DEFAULT_STOPS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "were", "be", "been",
    "are", "am", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "him", "her", "his",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "also", "into", "over", "after", "before",
    "between", "through", "during", "because", "while", "which",
    "who", "whom", "what", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "some", "any",
    "i", "me", "my", "myself",
}


def load_corpus(corpus_path: str) -> List[str]:
    """Load and split a text corpus into sentences."""
    if not os.path.exists(corpus_path):
        # Try default path relative to project root
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "data", "corpora", "teen_seeds.txt")
        if os.path.exists(alt_path):
            corpus_path = alt_path
        else:
            print(f"  [PMI] Corpus not found at {corpus_path}, using internal fallback")
            return _get_fallback_sentences()
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into sentences
    sentences = re.split(r'[.!?]+\s*', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return sentences


def _get_fallback_sentences() -> List[str]:
    """Built-in fallback sentences when no corpus file exists."""
    return [
        "hello how are you doing today", "it is nice to meet you",
        "what do you like to do for fun", "i really enjoy spending time with friends",
        "i think that everyone deserves respect", "in my opinion honesty is the best policy",
        "a true friend is someone who always listens", "trust is the foundation of any relationship",
        "i have a lot to learn and that is exciting", "science class is really interesting this year",
        "i woke up early this morning to go for a walk", "i like to listen to music while i do my homework",
        "i love going for walks in the park", "nature is full of amazing things to discover",
        "never give up on your dreams", "every mistake is a chance to learn something new",
        "you are capable of amazing things", "small steps lead to big changes over time",
        "believe in yourself and anything is possible", "the future is full of possibilities",
        "i use my computer to learn new things online", "technology changes the way we communicate",
        "the internet is a powerful tool for learning", "video games are a fun way to relax",
        "i am learning how to code in my free time", "artificial intelligence is changing the world",
        "what is the meaning of happiness", "i wonder why people dream when they sleep",
        "sometimes the simplest things bring the most joy", "curiosity is what drives human progress",
        "everything in nature follows a pattern", "time is the most valuable thing we have",
        "change is the only constant in life", "we are all connected in ways we cannot see",
        "the sunset was absolutely beautiful today", "i feel really happy when i see my friends",
        "sometimes i get nervous before a big test", "being kind to others makes me feel good",
        "i am curious about how things work", "learning opens doors to many opportunities",
        "i enjoy writing stories for my english class", "i am trying to learn a new language",
        "every day is a new adventure", "taking care of the environment is everyone responsibility",
        "i believe that hard work pays off", "personally i think music brings people together",
        "to be honest i really enjoy reading books", "i feel like we should help each other more",
        "practice makes perfect", "it depends on the situation",
        "every problem has a solution", "i am pretty sure that things will get better",
        "family is really important to me", "friendship means being there for each other",
        "good communication is key to strong friendships", "i learned a lot from my older sibling",
        "my best friend and i have known each other for years",
        "you should never be afraid to ask for help", "it makes me angry when people are unfair",
        "i feel grateful for everything i have", "my favorite subject in school is history",
        "i am excited about the weekend", "i enjoy playing sports with my friends",
        "i saw a beautiful rainbow after the rain", "spring is my favorite season of the year",
        "animals are such fascinating creatures", "planting a tree is a great way to help the planet",
        "challenges make us stronger in the long run", "it is never too late to start something new",
        "you can achieve anything if you work hard enough",
        "social media helps me stay connected with friends",
        "smartphones make it easy to stay in touch", "it is important to take breaks from screens",
        "the stars are so bright tonight", "i love the sound of rain on the roof",
        "breakfast is my favorite meal of the day", "after school i usually hang out with my friends",
        "i try to read at least one book every month", "my favorite hobby is drawing and painting",
        "sometimes i just like to sit and think",
        "why do we remember the past but not the future",
        "time is the most valuable thing we have", "change is the only constant in life",
        "the more you learn the more you realize you do not know",
    ]


def compute_pmi_from_corpus(
    sentences: List[str],
    window_size: int = 4,
    min_freq: int = 2,
    min_pmi: float = 0.5,
    max_concepts: int = 200,
    stop_words: Optional[Set[str]] = None
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """Compute PMI statistics from a corpus.
    
    Args:
        sentences: List of sentence strings
        window_size: Co-occurrence window size
        min_freq: Minimum word frequency to consider
        min_pmi: Minimum PMI to include an edge
        max_concepts: Maximum number of top concepts to return
        stop_words: Set of stop words to filter
        
    Returns:
        (concepts_list, edges_list) where:
        - concepts_list: [(word, frequency_score)] sorted by frequency
        - edges_list: [(word1, word2, pmi_score)] sorted by PMI
    """
    if stop_words is None:
        stop_words = _DEFAULT_STOPS
    
    # Tokenize into words
    word_counts = Counter()
    cooccur_counts: Dict[Tuple[str, str], int] = Counter()
    total_words = 0
    
    for sent in sentences:
        words = re.findall(r"[a-zA-Z']{3,}", sent.lower())
        words = [w.strip("'") for w in words 
                 if w.strip("'") not in stop_words and len(w.strip("'")) >= 3]
        
        for w in words:
            word_counts[w] += 1
            total_words += 1
        
        # Co-occurrence within sliding window
        for i, w1 in enumerate(words):
            for j in range(i + 1, min(i + window_size + 1, len(words))):
                w2 = words[j]
                if w1 != w2:
                    pair = tuple(sorted([w1, w2]))
                    cooccur_counts[pair] += 1
    
    # Filter by minimum frequency
    frequent_words = {w for w, c in word_counts.items() if c >= min_freq}
    
    # Compute PMI for each word pair
    pmi_scores: List[Tuple[str, str, float]] = []
    n_pairs = sum(cooccur_counts.values()) if cooccur_counts else 1
    
    for (w1, w2), cooccur in cooccur_counts.items():
        if w1 not in frequent_words or w2 not in frequent_words:
            continue
        # PMI = log(P(w1,w2) / (P(w1) * P(w2)))
        p_w1 = word_counts[w1] / max(total_words, 1)
        p_w2 = word_counts[w2] / max(total_words, 1)
        p_joint = cooccur / max(n_pairs, 1)
        
        if p_joint > 0 and p_w1 > 0 and p_w2 > 0:
            pmi = math.log2(p_joint / (p_w1 * p_w2))
            if pmi >= min_pmi:
                pmi_scores.append((w1, w2, pmi))
    
    # Sort by PMI descending
    pmi_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Get top concepts by frequency
    top_concepts = sorted(
        [(w, c) for w, c in word_counts.items() if c >= min_freq],
        key=lambda x: x[1],
        reverse=True
    )[:max_concepts]
    
    concepts = [w for w, _ in top_concepts]
    return (concepts, pmi_scores)


class PMISeeder:
    """Corpus-driven seeding using PMI statistics.
    
    Replaces hardcoded TEEN_CONCEPTS, semantic_edges, causal_edges,
    contrastive_edges, temporal_edges, analogical_edges with
    data-driven equivalents from a text corpus.
    """
    
    def __init__(self, corpus_path: Optional[str] = None,
                 max_concepts: int = 200, max_edges: int = 300,
                 min_freq: int = 2, min_pmi: float = 0.3):
        self.max_concepts = max_concepts
        self.max_edges = max_edges
        self.min_freq = min_freq
        self.min_pmi = min_pmi
        self.corpus_path = corpus_path
    
    def seed_from_corpus(self, graph, glove_fn=None, rng=None) -> Tuple[int, int]:
        """Seed a graph from the corpus using PMI.
        
        Args:
            graph: ConceptGraph to seed
            glove_fn: Optional function to get GloVe vectors
            rng: Optional random state
            
        Returns:
            (num_concepts, num_edges)
        """
        if rng is None:
            rng = np.random.RandomState(42)
        
        sentences = load_corpus(self.corpus_path)
        concepts, pmi_edges = compute_pmi_from_corpus(
            sentences,
            min_freq=self.min_freq,
            min_pmi=self.min_pmi,
            max_concepts=self.max_concepts
        )
        
        label_to_id = {}
        added_count = 0
        
        for word in concepts:
            vec = None
            if glove_fn is not None:
                vec = glove_fn(word)
            if vec is None:
                h = hash(word) % 10000
                vr = np.random.RandomState(h + 42)
                vec = vr.randn(graph.dim).astype(np.float32) * 0.15
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec /= norm
            node = graph.add_node(vector=vec, label=word)
            label_to_id[word] = node.id
            added_count += 1
        
        # Apply top PMI edges
        edge_count = 0
        for w1, w2, pmi in pmi_edges[:self.max_edges]:
            nid1 = label_to_id.get(w1)
            nid2 = label_to_id.get(w2)
            if nid1 is not None and nid2 is not None:
                if graph.get_edge(nid1, nid2) is None and graph.get_edge(nid2, nid1) is None:
                    # Weight = sigmoid(PMI) mapped to [0.15, 0.75]
                    weight = 0.15 + 0.6 * (1.0 / (1.0 + math.exp(-pmi + 1.0)))
                    weight = min(0.75, max(0.15, weight))
                    graph.add_edge(nid1, nid2, weight=weight + rng.uniform(0, 0.08),
                                  relation_type="semantic")
                    edge_count += 1
        
        return (added_count, edge_count)
    
    def get_fallback_concepts(self) -> List[str]:
        """Return essential seed concepts when corpus is unavailable.
        A minimal set to bootstrap the graph — far smaller than the old 188-item TEEN_CONCEPTS.
        """
        return [
            "hello", "bye", "yes", "no", "please", "thanks", "sorry",
            "i", "you", "we", "they", "people", "person",
            "friend", "love", "trust", "respect", "help",
            "think", "know", "feel", "want", "like", "say",
            "good", "bad", "big", "small", "happy", "sad",
            "true", "real", "life", "time", "world", "change",
            "learn", "create", "make", "give", "take",
            "reason", "meaning", "way", "thing", "part",
            "science", "art", "nature", "mind", "heart",
            "cause", "effect", "question", "answer",
            "hope", "fear", "joy", "pain", "strength",
            "together", "always", "never", "now", "then",
            "work", "play", "rest", "dream", "grow",
            "see", "hear", "speak", "read", "write",
            "begin", "end", "day", "night", "story",
            "music", "book", "idea", "plan", "choice",
            "power", "freedom", "truth", "justice",
            "different", "important", "possible", "simple",
            "deep", "clear", "strong", "full", "open",
            "knowledge", "wisdom", "understanding",
            "experience", "practice", "effort", "skill",
            "journey", "path", "place", "moment", "memory",
            "family", "home", "school", "city", "country",
            "sun", "moon", "star", "water", "fire", "air",
            "tree", "flower", "animal", "bird", "ocean",
            "believe", "wonder", "imagine", "discover",
            "build", "destroy", "protect", "serve",
            "challenge", "struggle", "overcome", "succeed",
        ]
