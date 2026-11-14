"""Anchor-token boosting -- the retrieve-then-elaborate bridge.

The decoder's conditioning path is measured as near-null: the fitted ``W_sm``
projection maps different subjects to vectors with pairwise cosine 0.999
(``docs/CONDITIONING_COLLAPSES_TO_A_POINT.md``), and the attention head's
``W_q``/``W_k``/``W_v`` are at exact random init in the shipped checkpoint
(fro 22.571 = 256 x 0.0882, matching a fresh ``randn(256,256)``). So the
subject signal does not reach the GRU, and the emitted distribution is a
frozen function-word attractor (``docs/DECODER_FIXED_FILLER_REGISTER.md``).

The one intervention in this repo with a measured dose-response curve is a
logit boost on the retrieved definition's content words. The sweep is
recorded in ``scripts/train.py`` (the ``ANCHOR_TOKEN_BOOST`` comment block):

    boost   on-topic   distinct-1   repetition
    1.5      0.009       0.997         0.000
    3.0      0.047       0.996         0.000
    5.0      0.240       0.996         0.001
    7.0      0.490       0.996         0.000
    9.0      0.522       0.996         0.000
    10.0     0.537       0.994         0.000

The response is a sharp threshold between 3.0 and 6.0, saturating near 9.0,
with no cost to fluency. 9.0 is the knee.

This module is the single source of truth for that constant and for the
content-word extraction, so the training harness and the production chat path
cannot drift apart. It is a pure function over (text, word_to_idx) -> dict.
"""

from __future__ import annotations

from typing import Dict, Optional

# The measured knee. See the sweep table in the module docstring and
# scripts/train.py's ANCHOR_TOKEN_BOOST comment block.
ANCHOR_TOKEN_BOOST = 9.0

# Closed-class items that must never be boosted into content positions. This
# is the same inventory the decoder's own content_word_ids mask uses on the
# production path (response_gen.py's function_words_set), kept local so this
# module has no chat-engine dependency.
_STOP = frozenset({
    "a", "an", "the", "and", "but", "or", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not", "no",
    "nor", "so", "if", "then", "than", "too", "very", "just", "about", "also",
    "into", "over", "after", "before", "between", "through", "during",
    "because", "while", "which", "who", "whom", "what", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "some", "any", "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "we", "our", "you", "your", "he", "she", "him", "her",
    "his", "i", "me", "my", "myself", "am", "of", "to", "for", "with", "from",
    "at", "by", "as", "on", "in", "out", "up", "off",
})

_STRIP = ".,!?;:\"'()[]"


def content_words(text: str):
    """Yield the lowercased content words of ``text`` in order.

    A content word is a token that is not closed-class and not a special
    token. The brain's lemma layer selects among content lemmas; function
    words are inserted later by grammatical encoding, so boosting them would
    push the decoder toward the filler register it already over-produces.
    """
    if not text:
        return
    for raw in str(text).split():
        w = raw.lower().strip(_STRIP)
        if not w or w in _STOP or w.startswith("<"):
            continue
        yield w


def anchor_token_boost(anchor_desc: Optional[str],
                       word_to_idx: Optional[Dict[str, int]],
                       *,
                       strength: float = ANCHOR_TOKEN_BOOST) -> Optional[Dict[int, float]]:
    """Logit boost for the anchor description's content words.

    Retrieve-then-elaborate: boost the retrieved definition's content words so
    the decoder's elaboration stays lexically tethered to it, without forcing
    verbatim repetition. Returns ``None`` when there is nothing to boost --
    the caller then passes ``token_boost=None`` and generation is unchanged.
    """
    if not anchor_desc or not word_to_idx:
        return None
    boost: Dict[int, float] = {}
    for w in content_words(anchor_desc):
        idx = word_to_idx.get(w)
        if idx is not None:
            boost[idx] = strength
    return boost or None
