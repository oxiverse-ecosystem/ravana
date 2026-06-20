"""
Shared constants and utilities for RLMv2 submodules.

Auto-extracted from rlm_v2.py.
"""

import numpy as np
import time
import pickle
import os
import zipfile
import json
from typing import Optional, List, Tuple, Dict, Set, Any
from collections import defaultdict

from .module import Module, Linear, Embedding
from ..graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap
from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity
from ..propagation import PropagationEngine
from ..currencies import CognitiveCurrencies
from ..currency import create_rlm_currency


# ─── Relation Type Definitions ───────────────────────────────────────────────

RELATION_TYPES = [
    "causal",       # causes, produces, leads to, results in, makes, triggers
    "semantic",     # is, are, represents, defines, means
    "temporal",     # then, after, before, next, later, during
    "possessive",   # has, contains, includes, belongs to, part of
    "analogical",   # like, similar to, resembles, analogous to
    "contextual",   # in, at, on, with, under, over
]


def _build_glove_embedding_matrix(tokenizer, target_dim=64, glove_dim=100, max_words=200000):

    """Build embedding matrix from pre-trained GloVe vectors.

    

    Downloads glove.6B.100d.txt (cached in data/glove/) on first call.

    Projects 100D -> target_dim using a random orthogonal projection.

    Returns (vocab_size, target_dim) float32 matrix or None if unavailable.

    """

    import numpy as np

    import os

    from pathlib import Path

    

    glove_path = Path('data') / 'glove' / 'glove.6B.100d.txt'

    

    # Fallback to 50D if 100D not available

    if not glove_path.exists():

        glove_path = Path('data') / 'glove' / 'glove.6B.50d.txt'

        glove_dim = 50

    

    if not glove_path.exists():

        return None

    

    # Load GloVe vectors into dict

    word_vecs = {}

    with open(str(glove_path), 'r', encoding='utf-8') as f:

        for i, line in enumerate(f):

            if i >= max_words:

                break

            parts = line.strip().split()

            if len(parts) != glove_dim + 1:

                continue

            word = parts[0]

            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)

            word_vecs[word] = vec

    

    # Build projection matrix glove_dim -> target_dim (random orthogonal)

    rng = np.random.RandomState(42)

    max_d = max(glove_dim, target_dim)

    full_q, _ = np.linalg.qr(rng.randn(max_d, max_d).astype(np.float32))

    proj = full_q[:target_dim, :glove_dim].copy()

    proj *= np.sqrt(float(glove_dim) / float(target_dim))

    

    # Build token->id mapping

    word_to_id = {}

    if hasattr(tokenizer, 'word_to_id') and tokenizer.word_to_id:

        word_to_id = tokenizer.word_to_id

    else:

        for tid in range(tokenizer.vocab_size):

            try:

                w = tokenizer.decode([tid]).strip().lower()

                if w:

                    word_to_id[w] = tid

            except Exception:

                pass

    

    vocab_size = tokenizer.vocab_size

    

    # Check for cached projected matrix

    cache_path = Path('data') / 'glove' / f'projected_{vocab_size}_{target_dim}.npy'

    if cache_path.exists():

        return np.load(str(cache_path))

    

    matrix = np.zeros((vocab_size, target_dim), dtype=np.float32)

    found = 0

    total = 0

    

    for word, tid in word_to_id.items():

        if not isinstance(word, str) or not word:

            continue

        total += 1

        w = word.lower().strip()

        

        vec = word_vecs.get(w)

        if vec is None and len(w) > 1:

            # Try stripping plural/tense markers

            vec = word_vecs.get(w.rstrip('s'))

        if vec is None and len(w) > 2:

            vec = word_vecs.get(w[:-1])

        

        if vec is not None:

            matrix[tid] = proj @ vec

            norm = np.linalg.norm(matrix[tid])

            if norm > 0:

                matrix[tid] *= 2.0 / norm  # normalize to radius ~2.0

            found += 1

    

    # Fill missing with random

    if found < total:

        rng2 = np.random.RandomState(42)

        for word, tid in word_to_id.items():

            if not isinstance(word, str):

                continue

            if np.all(matrix[tid] == 0):

                matrix[tid] = rng2.randn(target_dim).astype(np.float32) * 0.3

    

    # Cache the projected matrix

    cache_path = Path('data') / 'glove' / f'projected_{vocab_size}_{target_dim}.npy'

    np.save(str(cache_path), matrix)

    return matrix





# ─── Relation Type Definitions ───────────────────────────────────────────────





_KEYWORD_MAP = {

    "causal": [

        "causes", "cause", "produces", "produce", "leads", "results",

        "makes", "make", "triggers", "trigger", "creates", "create",

        "generates", "generate", "melts", "melt", "burns", "burn",

        "breaks", "break", "destroys", "destroy", "builds", "build",

        "grows", "grow", "changes", "change", "transforms", "transform",

        "converts", "convert", "affects", "affect", "influences", "influence",

        "powers", "power", "drives", "drive", "forces", "force",

        "heats", "heat", "cools", "cool", "freezes", "freeze",

        "dissolves", "dissolve", "evaporates", "evaporate",

        "compresses", "compress", "expands", "expand",

        "contributes", "contribute", "associated", "linked",

        "correlates", "correlate", "worsens", "worsen",

        "improves", "improve", "increases", "increase",

        "decreases", "decrease", "reduces", "reduce",

        "enhances", "enhance", "diminishes", "diminish",

        "prevents", "prevent", "inhibits", "inhibit",

        "strengthens", "strengthen", "weakens", "weaken",

        "restores", "restore", "provides", "provide",

        "protects", "protect", "corrupts", "corrupt",

        "damages", "damage", "harms", "harm", "heals", "heal",

        "cures", "cure", "fights", "fight", "blocks", "block",

        "accelerates", "accelerate", "slows", "slow",

        # Compound predicates (single tokens after wordpiece)

        "contributes_to", "associated_with", "linked_to",

        "may_cause", "can_cause", "leads_to", "results_in",

        "correlated_with", "is_a", "is_type_of", "type_of",

        "consists_of", "composed_of", "made_of",

        "capable_of", "able_to", "same_as", "equivalent_to",

    ],

    "temporal": [

        "then", "after", "before", "next", "later", "during",

        "when", "while", "until", "since", "follows", "follow",

        "precedes", "precede", "succeeds", "succeed",

    ],

    "possessive": [

        "has", "have", "contains", "contain", "includes", "include",

        "belongs", "comprises", "comprise", "holds", "hold",

        "carries", "carry", "bears", "bear",

    ],

    "analogical": [

        "like", "similar", "resembles", "resemble", "analogous",

        "comparable", "equivalent", "parallel", "mirrors", "mirror",

    ],

    "contextual": [

        "in", "at", "on", "with", "under", "over", "near", "beside",

        "within", "among", "between", "through", "across", "along",

    ],

}




