"""Learned word-frequency model — brain-honest replacement for hand word lists.

Brain basis
-----------
The mental lexicon is a frequency-organized network. Word-frequency effects
are among the most robust findings in neuroscience: N400 amplitude and IFG BOLD
response scale with log frequency (Zipf). The brain does NOT carry a hand-curated
set of "common words" or "generic nouns" — it just knows which words occur at
high frequency from exposure.

This module generalizes the RAVANA ``functional_lexicon`` pattern: each hand list
(``_COMMON_WORDS``, ``_GENERIC_NOUNS``, ``TOPIC_SKIP_WORDS``,
``_SUBJECT_CONTEXT_WORDS``) is kept as a COLD-START SEED, then the model folds in
observed conversation/corpus frequency so the high-frequency tail is discovered
from exposure rather than authored. Day-one behavior is identical (only the seed
words are known); over time the observed distribution dominates, exactly like the
de-hardcode P0 flags intend.

Membership (``is_common`` / ``is_generic_noun`` / ``is_topic_skip`` /
``is_subject_glue``) returns True if the word is in the seed OR has risen into the
observed high-frequency band. During warmup (fewer than ``min_obs`` tokens seen)
only the seed applies, so the engine never degrades on a cold boot.

Persistence: the observed counts serialize so frequency learning survives
sessions (the audit's saved-but-never-loaded class of bug).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

_TOKEN_RE = re.compile(r"[a-z']+")


class FrequencyModel:
    """Tracks word frequency from corpus + conversation; seeds from hand lists."""

    def __init__(self, seed_words: Optional[Set[str]] = None,
                 min_obs: int = 200, percentile: float = 0.8) -> None:
        # Seed = the original hardcoded list; frozen so day-one behavior matches.
        self.seed: Set[str] = set(w.lower() for w in (seed_words or set()))
        self.counts: Counter = Counter()
        self.total: int = 0
        self.min_obs = int(min_obs)
        self.percentile = float(percentile)

    # -- ingestion -----------------------------------------------------------
    def observe(self, text: str) -> None:
        """Fold raw text into the running frequency counts."""
        toks = _TOKEN_RE.findall(text.lower())
        if not toks:
            return
        self.counts.update(toks)
        self.total += len(toks)

    def observe_tokens(self, tokens: Iterable[str]) -> None:
        toks = [t.lower() for t in tokens if t]
        if not toks:
            return
        self.counts.update(toks)
        self.total += len(toks)

    # -- queries (membership, brain-honest) ----------------------------------
    def _learned_band(self) -> Set[str]:
        """Words in the observed high-frequency band (top (1-percentile))."""
        if self.total < self.min_obs or not self.counts:
            return set()
        # Zipf-like: a word is "common" if its frequency exceeds the per-token
        # threshold at the requested percentile of the observed distribution.
        cutoff = self.percentile * (self.total / max(len(self.counts), 1))
        return {w for w, c in self.counts.items() if c >= cutoff}

    def is_common(self, word: str) -> bool:
        """True if ``word`` is a seed common word OR observed high-frequency."""
        w = word.lower()
        if w in self.seed:
            return True
        return w in self._learned_band()

    def is_generic_noun(self, word: str) -> bool:
        """Back-compat alias for the generic-noun list (seed + frequency)."""
        return self.is_common(word)

    def is_topic_skip(self, word: str) -> bool:
        return self.is_common(word)

    def is_subject_glue(self, word: str) -> bool:
        return self.is_common(word)

    def top_frequent(self, n: int = 50,
                      exclude: Optional[Set[str]] = None) -> List[str]:
        """Highest-frequency observed words (optionally excluding a POS set)."""
        exclude = exclude or set()
        return [w for w, _ in self.counts.most_common(n) if w not in exclude]

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "seed": sorted(self.seed),
            "counts": dict(self.counts),
            "total": self.total,
            "min_obs": self.min_obs,
            "percentile": self.percentile,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FrequencyModel":
        fm = cls(seed_words=set(d.get("seed", [])),
                 min_obs=int(d.get("min_obs", 200)),
                 percentile=float(d.get("percentile", 0.8)))
        fm.counts = Counter(d.get("counts", {}))
        fm.total = int(d.get("total", 0))
        return fm
