"""Track B — Phase 2: learned snippet-quality model (replaces Group 1 filters).

Replaces the hand-listed ``_SNIPPET_REJECT_SHAPES`` / ``_SNIPPET_NOISE`` /
``WEB_GARBAGE`` regex/word tables with a *learned* structural prediction-error
(PE) model, grounded in predictive coding (Elman 1990 SRN; Friston FEP) and
prototype similarity (Rosch).

Brain analog
------------
The brain does not carry a frozen list of "boilerplate phrases". Instead, a
forward model learns the *structure* of well-formed linguistic input and flags
anything that violates that prediction as prediction error (surprisal). A
snippet whose word-sequence is far from the learned distribution of good
definitions is rejected — but only when its semantic coherence with the
subject is also low, so an unusual-but-on-topic snippet survives (dual gate:
``pe > theta_struct AND coherence < theta_sem``).

Implementation
--------------
``SnippetStructureModel`` is a sparse word n-gram (order 2) with additive
smoothing. It is trained on a seed corpus of known-good declarative snippets
(definitions). ``structural_pe`` returns the mean surprisal (-log2 P) of the
snippet under the model: low for in-distribution (well-formed) snippets, high
for out-of-distribution boilerplate / navigation / code. ``theta_struct`` is
learned from the PE distribution of the good training set (max * margin), so
the threshold adapts to the data rather than being hand-picked.

This module is intentionally self-contained and unit-testable without GloVe,
network, or the full engine — the engine wires it in BEHIND A FLAG
(``--use-cerebellar-snippet``) and keeps the old constants as a fallback until
the learned model is verified to beat them on the regression set (per the
milestone plan).
"""

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, List, Optional, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CHAR_RE = re.compile(r"[a-z0-9@#|/_.():;=+\-\[\]{}<>]")
_CHAR_N = 4  # character n-gram order for structural (code/symbol) detection


# Seed corpus of known-good declarative snippets. These teach the model the
# structural shape of a real definition/answer (subject + copula + predicate),
# so boilerplate/navigation/code — which is structurally OOD — accrues high PE.
_SEED_GOOD = [
    "trust is a belief in the reliability of another person",
    "gravity is the force that attracts objects with mass toward each other",
    "a quokka is a small herbivorous marsupial found in western australia",
    "photosynthesis is the process by which plants convert light into energy",
    "democracy is a system of government in which power rests with the people",
    "the heart is a muscular organ that pumps blood through the body",
    "language is a structured system of communication used by humans",
    "memory is the faculty by which the mind stores and recalls information",
    "a neuron is a cell that transmits electrical and chemical signals",
    "climate is the long term pattern of weather in a region",
    "a river is a natural flowing stream of freshwater",
    "justice is the principle of fair treatment under the law",
    "the sun is the star at the center of the solar system",
    "knowledge is the understanding acquired through experience or study",
    "an economy is a system of production distribution and consumption of goods",
]


class SnippetStructureModel:
    """Sparse n-gram structural prediction-error model for web snippets.

    Two PE streams, combined by max:
      * word n-gram (order 2) — flags prose that is structurally OOD vs the
        learned definition shape.
      * character n-gram (order 4) — flags code fragments, @mentions, URLs and
        other symbol-heavy boilerplate that word n-grams miss.

    Trained on known-good snippets; high combined PE + low semantic coherence
    => junk (rejected). The threshold ``theta_struct`` is learned from the PE
    distribution of the good training set (envelope + margin), so it adapts to
    the data instead of being hand-picked.
    """

    def __init__(self, order: int = 2, smoothing: float = 1.0,
                 theta_sem: float = 0.15, margin: float = 0.5):
        self.order = order
        self.smoothing = smoothing
        self.theta_sem = theta_sem
        self.margin = margin  # absolute bits added beyond the worst good snippet
        # word stream: context tuple -> Counter(next token)
        self._w_ngrams: dict = defaultdict(Counter)
        self._w_vocab: set = set()
        # char stream: context tuple -> Counter(next char)
        self._c_ngrams: dict = defaultdict(Counter)
        self._c_vocab: set = set()
        self.theta_struct: Optional[float] = None
        self._trained = False

    # ─── tokenization ──────────────────────────────────────────────────────
    @staticmethod
    def _wtokens(s: str) -> List[str]:
        return _TOKEN_RE.findall((s or "").lower())

    @staticmethod
    def _ctokens(s: str) -> List[str]:
        return _CHAR_RE.findall((s or "").lower())

    # ─── training ───────────────────────────────────────────────────────────
    def _learn_stream(self, ngrams, vocab, tokens, order):
        if len(tokens) < order:
            return
        ctx = ("<s>",) * order
        for t in tokens:
            ngrams[ctx][t] += 1
            vocab.add(t)
            ctx = (ctx + (t,))[-order:]
        ngrams[ctx]["</s>"] += 1

    def train(self, good_snippets: Sequence[str]) -> None:
        """Learn n-gram structure from a corpus of known-good snippets."""
        for snip in good_snippets:
            self._learn_stream(self._w_ngrams, self._w_vocab,
                               self._wtokens(snip), self.order)
            self._learn_stream(self._c_ngrams, self._c_vocab,
                               self._ctokens(snip), _CHAR_N)
        # Mark trained BEFORE computing the threshold, so structural_pe() reads
        # the model (it returns 0.0 while untrained, which would zero the gate).
        self._trained = True
        pes = [self.structural_pe(s) for s in good_snippets
               if len(self._wtokens(s)) >= 3]
        if pes:
            # Envelope: anything more surprising than the worst acceptable good
            # snippet (plus a small margin) is structurally OOD. Combines the
            # learned distribution with a sanity floor.
            self.theta_struct = max(pes) + self.margin

    def train_seed(self) -> None:
        """Train on the built-in seed corpus of known-good definitions."""
        self.train(_SEED_GOOD)

    # ─── inference ────────────────────────────────────────────────────────────
    def _stream_pe(self, ngrams, vocab, tokens, order) -> float:
        if len(tokens) < order or not self._trained:
            return 0.0
        ctx = ("<s>",) * order
        total = 0.0
        n = 0
        for t in tokens:
            counts = ngrams.get(ctx)
            if counts:
                denom = sum(counts.values()) + self.smoothing * len(vocab)
                p = (counts.get(t, 0) + self.smoothing) / denom
            else:
                denom = self.smoothing * (len(vocab) + 1)
                p = self.smoothing / denom
            total += -math.log2(p)
            n += 1
            ctx = (ctx + (t,))[-order:]
        return total / n if n else 0.0

    def structural_pe(self, snippet: str) -> float:
        """Combined structural PE (max of word-stream and char-stream surprisal).

        Returns 0.0 when untrained or too short (caller should not reject on
        PE alone in those cases).
        """
        w_pe = self._stream_pe(self._w_ngrams, self._w_vocab,
                               self._wtokens(snippet), self.order)
        c_pe = self._stream_pe(self._c_ngrams, self._c_vocab,
                               self._ctokens(snippet), _CHAR_N)
        # Word stream on very short snippets is unreliable; fall back to char.
        if len(self._wtokens(snippet)) < 3:
            return c_pe
        return max(w_pe, c_pe)

    def is_junk(self, snippet: str,
                coherence: Optional[float] = None) -> bool:
        """Dual-gate junk decision: high structural PE AND low coherence.

        Untrained or no threshold => never rejects (safe default; the old
        hardcoded filters remain the backstop in that case).
        """
        if self.theta_struct is None:
            return False
        pe = self.structural_pe(snippet)
        if pe > self.theta_struct:
            # High structural surprise; reject only if semantics also fail.
            if coherence is None or coherence < self.theta_sem:
                return True
        return False

    # ─── convenience for tests / diagnostics ───────────────────────────────
    def pe_distribution(self, snippets: Iterable[str]) -> List[float]:
        return [self.structural_pe(s) for s in snippets]


def default_model() -> "SnippetStructureModel":
    """A ready-to-use model trained on the seed corpus."""
    m = SnippetStructureModel()
    m.train_seed()
    return m
