"""Shared slot naming for stance/opinion topic keys (FIX-RV-12).

One chokepoint so the stance MINER (``user_model._opinion_topic``) and the
stance RECALL paths (``engine._user_stance_reply`` and friends, which all
re-resolve the query through ``_opinion_topic``) name the SAME slot for the
same concept. Before this module the two sides could disagree: a disclosure
"i love cooking earlier in the morning" mined the key "cooking earlier" while
a later "what are your thoughts on cooking?" resolved to "cooking" — the
stance was stored under a key no query could ever reach.

Why this lives here rather than as a hand-written ``if 'earlier' in topic``
chain: the modifier vocabulary is a CLOSED-CLASS seed stored in
``data/functional_lexicon.json`` (``leading_modifiers`` /
``trailing_modifiers``) and loaded through the existing ``FunctionalLexicon``
fit-file mechanism, so it is data (EER-fit / decayed / grown at runtime like
every other functional list), not authored per-topic logic. Removing an entry
only re-admits one modifier shape; adding one generalizes to any disclosure
that uses that modifier. No retraining, no per-topic tables, no reply prose.

Fails open: if the fit file or the lexicon import is unavailable the seed sets
below are used, so behavior never regresses below the seed.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

# Seed fallback — mirrors the FunctionalLexicon seed lists of the same name.
# The fit file (data/functional_lexicon.json) wins when present.
_SEED_LEADING = frozenset({
    # salutation / age / status modifiers that never head a real concept
    "dear", "dearest", "old", "late", "former", "poor", "beloved",
    "cherished", "respected", "deceased", "belated",
})

_SEED_TRAILING = frozenset({
    "earlier", "now", "then", "later", "soon", "always", "never",
    "sometimes", "again", "already", "still", "yet", "just", "even",
    "too", "very", "quite", "really", "better", "best", "worse",
    "worst", "more", "most", "less", "least", "often", "usually",
    "generally", "basically", "actually", "literally", "seriously",
    "honestly", "frankly", "clearly", "obviously", "apparently",
    "presumably", "supposedly", "today", "tonight", "yesterday",
    "tomorrow", "lately", "recently", "beforehand", "afterwards",
})


def _lexicon():
    """Return the loaded FunctionalLexicon, or None when unavailable."""
    try:
        from .functional_lexicon import default_lexicon
        return default_lexicon()
    except Exception:
        return None


def leading_modifiers() -> frozenset:
    """Leading modifier vocabulary (fit file when present, else seed)."""
    lex = _lexicon()
    if lex is not None:
        try:
            got = set(lex.leading_modifiers)
            if got:
                return frozenset(got)
        except Exception:
            pass
    return _SEED_LEADING


def trailing_modifiers() -> frozenset:
    """Trailing modifier vocabulary (fit file when present, else seed)."""
    lex = _lexicon()
    if lex is not None:
        try:
            got = set(lex.trailing_modifiers)
            if got:
                return frozenset(got)
        except Exception:
            pass
    return _SEED_TRAILING


def strip_leading_modifiers(tokens: Sequence[str],
                            mods: Optional[Iterable[str]] = None) -> List[str]:
    """Drop leading salutation/age/status modifiers ("dear old jazz clubs").

    Only ever strips from the FRONT and never empties the list, so a genuine
    content head is never consumed ("old jazz" -> "jazz", "jazz" -> "jazz").
    """
    vocab = set(mods) if mods is not None else leading_modifiers()
    out = list(tokens)
    while len(out) > 1 and out[0] in vocab:
        out.pop(0)
    return out


def strip_trailing_modifiers(tokens: Sequence[str],
                             mods: Optional[Iterable[str]] = None) -> List[str]:
    """Drop trailing temporal/degree adverbs ("cooking earlier" -> "cooking").

    Only ever strips from the END and never empties the list, so a phrase that
    IS a modifier ("today") keeps its single token.
    """
    vocab = set(mods) if mods is not None else trailing_modifiers()
    out = list(tokens)
    while len(out) > 1 and out[-1] in vocab:
        out.pop()
    return out


def slot_name(tokens: Sequence[str],
              stop: Optional[Iterable[str]] = None,
              leading: Optional[Iterable[str]] = None,
              trailing: Optional[Iterable[str]] = None) -> str:
    """Canonical slot key for a tokenized opinion phrase.

    Applies the same leading then trailing trim to both the mining side and
    every recall side, so a mined key and a queried key are byte-identical for
    the same concept. `stop` (when given) is also trimmed from both edges,
    preserving the historical "drop leading determiners / trailing residue"
    behavior in one place.
    """
    toks = list(tokens)
    if not toks:
        return ""
    if stop is not None:
        stops = set(stop)
        while len(toks) > 1 and toks[0] in stops:
            toks.pop(0)
        while len(toks) > 1 and toks[-1] in stops:
            toks.pop()
    toks = strip_leading_modifiers(toks, leading)
    toks = strip_trailing_modifiers(toks, trailing)
    return " ".join(toks)
