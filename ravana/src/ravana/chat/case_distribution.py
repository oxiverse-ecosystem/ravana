"""Case-distribution prior for brain-inspired casing (research item: casing, Phase 1).

The current codebase hardcodes casing: ``text[0].upper() + text[1:]`` for
sentence start, and a manually-maintained ``_proper_nouns`` set for proper
nouns (engine.py:530, response_gen.py:682, chain_walker.py:264). The plan's
brain analog is the VWFA: store each word with a CASE-DISTRIBUTION (not a
binary flag), learned from exposure — here, the SUBTLEX-US film-subtitle
frequency norms, which preserve original casing.

The key SUBTLEX columns are ``FREQcount`` (total occurrences) and ``FREQlow``
(occurrences starting with a lowercase letter). So:

    P(capitalized) = 1 - FREQlow / FREQcount
    P(lowercase)   = FREQlow / FREQcount

e.g. France -> 1 - 13/1380 = 0.991; apple -> 1 - 878/1207 = 0.273.
These match the plan's table exactly (verified).

This module is the Phase-1 SUBSTRATE only: it builds/loads the per-word
capitalization prior and exposes ``cap_prob(word)``. It does NOT yet replace
the hardcoded ``.upper()`` (that is Phase 4) nor add context/position/entity
signals (Phase 2/5) or online user-feedback (Phase 3). Those build on this.

The prior is FIT from data (SUBTLEX), not hand-set. OOV words fall back to a
conservative default (common-noun lowercase) — consistent with today's behavior
so no regression. The json artifact is gitignored (generated, not source).

Dependency-free (numpy not even needed; pure csv + json).
"""
from __future__ import annotations

import os
import csv
import json
import sys
from typing import Dict, Optional

_THIS = os.path.dirname(os.path.abspath(__file__))
# case_distribution.py lives at <repo>/ravana/src/ravana/chat/.
# Repo root is TWO levels up: chat -> ravana(src/ravana) -> src -> ravana -> <repo>.
# Wait: path is <repo>/ravana/src/ravana/chat/case_distribution.py
#   chat -> ravana/src/ravana
#   ravana(src/ravana) -> ravana/src
#   src -> ravana (package)
#   ravana (package) -> ravana (repo root)? No.
# Actual: /c/.../ravana/ravana/src/ravana/chat  (nested ravana/ravana)
#   chat -> ravana/src/ravana
#   ravana -> ravana/src
#   src -> ravana (package dir 'ravana')
#   ravana(package) -> ravana (repo root)   <-- 4 up
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
for p in (os.path.join(_ROOT, "ravana", "src"),
          os.path.join(_ROOT, "ravana_ml", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

_DEFAULT_CAP_PROB = 0.0  # conservative: unknown word = common noun (lowercase)
_SUBTLEX_CSV = os.path.join(_ROOT, "data", "cache", "SUBTLEXus.csv")
_CASE_JSON = os.path.join(_ROOT, "data", "case_dist.json")


class CaseDistributionStore:
    """Per-word capitalization prior, FIT from SUBTLEX-US norms.

    ``_cap`` maps lowercased word -> P(capitalized) in [0, 1].
    """

    def __init__(self, cap: Optional[Dict[str, float]] = None):
        self._cap: Dict[str, float] = dict(cap) if cap else {}

    @classmethod
    def build_from_subtlex(cls, csv_path: str = _SUBTLEX_CSV) -> "CaseDistributionStore":
        """Build the prior from SUBTLEX-US. P(cap) = 1 - FREQlow/FREQcount."""
        cap: Dict[str, float] = {}
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            r = csv.DictReader(f)
            for row in r:
                w = (row.get("Word") or "").strip().lower()
                if not w:
                    continue
                try:
                    fc = float(row["FREQcount"])
                    fl = float(row["FREQlow"])
                except (TypeError, ValueError, KeyError):
                    continue
                if fc <= 0:
                    continue
                cap[w] = round(1.0 - fl / fc, 4)
        return cls(cap)

    def save(self, path: str = _CASE_JSON) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"cap": self._cap, "default": _DEFAULT_CAP_PROB}, f)

    @classmethod
    def load(cls, path: str = _CASE_JSON) -> "CaseDistributionStore":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        store = cls(d.get("cap", {}))
        store._default = float(d.get("default", _DEFAULT_CAP_PROB))
        return store

    _default = _DEFAULT_CAP_PROB

    def cap_prob(self, word: str) -> float:
        """P(this word is capitalized in non-sentence-initial position)."""
        return self._cap.get((word or "").lower().strip(" .,!?"), self._default)

    def __len__(self) -> int:
        return len(self._cap)

    def __contains__(self, word: str) -> bool:
        return (word or "").lower().strip(" .,!?") in self._cap


def get_store(rebuild: bool = False) -> CaseDistributionStore:
    """Return a loaded store, building from SUBTLEX if the json is absent."""
    if rebuild or not os.path.exists(_CASE_JSON):
        store = CaseDistributionStore.build_from_subtlex()
        store.save()
        return store
    return CaseDistributionStore.load()


# Threshold on the lexical capitalization prior for NON-sentence-initial words.
# Words with cap_prob >= this are capitalized mid-sentence (proper nouns like
# France/Monday); below it they stay lowercase (common nouns like apple/bank).
_CAP_THRESHOLD = 0.5


def case_infer(text: str, store: Optional[CaseDistributionStore] = None,
               entity_words: Optional[set] = None) -> str:
    """Context-sensitive casing (research casing, Phase 4) — replaces the
    hardcoded ``text[0].upper() + text[1:]``.

    Brain-aligned dual-route (plan Phase 4):
      * Positional route: the first alphabetic word of each sentence is
        capitalized (sentence-boundary bias, ~98% in written text).
      * Lexical route: a NON-sentence-initial word is capitalized when its
        FIT capitalization prior ``cap_prob`` (from SUBTLEX, Phase 1) is high,
        OR it is in ``entity_words`` (Phase 5 hook: ConceptGraph entity
        signal). Otherwise it stays lowercase — no regression for the
        all-lowercase decoder output.
      * Special: lone "i" -> "I" (pronoun, always capitalized).

    The prior is data-derived, not a hand list. OOV words have cap_prob=0.0
    (conservative lowercase), so unknown words never get wrongly capitalized.

    Punctuation and the rest of the text are preserved exactly.
    """
    if not text:
        return text
    if store is None:
        store = get_store()
    entity_words = {w.lower() for w in (entity_words or set())}

    # Split into sentences on . ! ? keeping the terminator attached.
    # Use a regex that splits AFTER a terminator + optional space.
    import re
    pieces = re.split(r'(?<=[.!?])\s+', text)
    out_pieces = []
    for piece in pieces:
        if not piece:
            continue
        # Tokenize into (leading_punct, word, trailing_punct) keeping structure.
        toks = re.findall(r"(\s*)(\S+)", piece)
        first_word_done = False
        out_toks = []
        for lead, tok in toks:
            # Separate trailing punctuation from the word.
            m = re.match(r"^([^\W\d_]+)(.*)$", tok)
            if m and m.group(1):
                word = m.group(1)
                trail = m.group(2)
                wl = word.lower()
                if not first_word_done:
                    # Positional route: capitalize sentence start.
                    out_toks.append((lead, word[0].upper() + word[1:], trail))
                    first_word_done = True
                else:
                    if wl == "i":
                        out_toks.append((lead, "I", trail))
                    elif wl in entity_words or store.cap_prob(wl) >= _CAP_THRESHOLD:
                        out_toks.append((lead, word[0].upper() + word[1:], trail))
                    else:
                        out_toks.append((lead, word, trail))
            else:
                # Tokens that aren't plain words (numbers, punctuation) unchanged.
                out_toks.append((lead, tok, ""))
        out_pieces.append("".join(lead + w + trail for lead, w, trail in out_toks))
    return " ".join(out_pieces)
