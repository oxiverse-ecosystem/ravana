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
_FB_MIN_OBS = 3           # min feedback observations before it overrides SUBTLEX
_SUBTLEX_CSV = os.path.join(_ROOT, "data", "cache", "SUBTLEXus.csv")
_CASE_JSON = os.path.join(_ROOT, "data", "case_dist.json")


class CaseDistributionStore:
    """Per-word capitalization prior, FIT from SUBTLEX-US norms.

    ``_cap`` maps lowercased word -> P(capitalized) in [0, 1].

    Phase 3 (research casing): online user-feedback prediction error (N400
    analog). When the user types a word RAVANA stored lowercase but capitalized,
    that is a prediction error that updates the internal model. We track it as a
    *weighted* per-word count (not a binary flag) so a single observation can't
    flip a well-established SUBTLEX prior, and the update PERSISTS across
    restarts (unlike the old in-memory ``_proper_nouns`` set, which was
    forgotten on reload).
    """

    def __init__(self, cap: Optional[Dict[str, float]] = None):
        self._cap: Dict[str, float] = dict(cap) if cap else {}
        self._fb: Dict[str, list] = {}        # word -> [cap_count, total_count]
        self._override: Dict[str, float] = {}  # word -> P(cap) high-confidence

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

    def save(self, path: Optional[str] = None) -> None:
        path = path or _CASE_JSON
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "cap": self._cap,
                "default": _DEFAULT_CAP_PROB,
                "fb": self._fb,
                "override": self._override,
            }, f)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "CaseDistributionStore":
        path = path or _CASE_JSON
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        store = cls(d.get("cap", {}))
        store._default = float(d.get("default", _DEFAULT_CAP_PROB))
        store._fb = {k: list(v) for k, v in (d.get("fb") or {}).items()}
        store._override = dict(d.get("override") or {})
        return store

    _default = _DEFAULT_CAP_PROB

    def cap_prob(self, word: str) -> float:
        """P(this word is capitalized in non-sentence-initial position).

        Resolution order: explicit high-confidence override -> learned feedback
        (if enough observations) -> SUBTLEX base prior -> conservative default.
        """
        wl = (word or "").lower().strip(" .,!?")
        if wl in self._override:
            return self._override[wl]
        fb = self._fb.get(wl)
        if fb and fb[1] >= _FB_MIN_OBS:
            return round(fb[0] / fb[1], 4)
        return self._cap.get(wl, self._default)

    # --- Phase 3: online user-feedback (N400 prediction-error) channel ---

    def record_feedback(self, word: str, capitalized: bool) -> None:
        """Record one observation that ``word`` was (not) capitalized by the user.

        Brain analog: the observed casing is the "correct" signal; mismatch with
        the current prediction generates a prediction error that nudges the
        stored distribution. Implemented as a count so it is robust to noise and
        combines with the SUBTLEX prior only after enough observations.
        """
        wl = (word or "").lower().strip(" .,!?")
        if not wl:
            return
        cnt = self._fb.get(wl, [0.0, 0.0])
        cnt[0] += 1.0 if capitalized else 0.0
        cnt[1] += 1.0
        self._fb[wl] = cnt

    def override(self, word: str, prob: float) -> None:
        """Force a high-confidence capitalization probability (0..1).

        Used for unambiguous typos / stable proper nouns the user repeats.
        Persisted via save().
        """
        wl = (word or "").lower().strip(" .,!?")
        if wl:
            self._override[wl] = float(min(1.0, max(0.0, prob)))

    def __len__(self) -> int:
        return len(self._cap)

    def __contains__(self, word: str) -> bool:
        return (word or "").lower().strip(" .,!?") in self._cap


_STORE_CACHE: Optional[CaseDistributionStore] = None


def get_store(rebuild: bool = False) -> CaseDistributionStore:
    """Return a loaded store, building from SUBTLEX if the json is absent.

    Cached at module level so online feedback (Phase 3) accumulates across calls
    within a process and persists on save(). Pass ``rebuild=True`` to refit.
    """
    global _STORE_CACHE
    if _STORE_CACHE is not None and not rebuild:
        return _STORE_CACHE
    if rebuild or not os.path.exists(_CASE_JSON):
        store = CaseDistributionStore.build_from_subtlex()
    else:
        store = CaseDistributionStore.load()
    store.save()  # materialize json so feedback writes have a target
    _STORE_CACHE = store
    return store


def record_user_casing(word: str, capitalized: bool) -> None:
    """Phase 3 entry point: log a user casing observation (N400 analog).

    Thin wrapper over the cached store's ``record_feedback``; safe if the store
    can't be loaded. Does NOT auto-save (call ``get_store().save()`` periodically
    or at session end to persist across restarts).
    """
    try:
        get_store().record_feedback(word, capitalized)
    except Exception:
        pass


def persist_store() -> None:
    """Persist the cached store (feedback + overrides) to disk."""
    try:
        if _STORE_CACHE is not None:
            _STORE_CACHE.save()
    except Exception:
        pass


# Entity-type keywords (from ConceptNet IsA targets) that license mid-sentence
# capitalization. SPLIT into STRONG (unambiguous named entities — capitalizing
# them is almost always correct) and WEAK (ambiguous with common nouns, e.g.
# "brand"/"product" overlaps "apple" fruit vs "Apple" company — these must NOT
# override the lexical prior, which stays lowercase for the common reading).
_ENTITY_STRONG = {
    "country", "nation", "sovereign_state", "state", "city", "capital",
    "town", "village", "municipality", "place", "location", "geographic_entity",
    "geographical_entity", "geopolitical_entity", "person", "people",
    "government_agency", "agency", "organization", "organisation",
    "company", "corporation", "firm", "enterprise", "university", "school",
    "college", "language", "religion", "ethnicity", "nationality",
}
_ENTITY_WEAK = {
    "brand", "product", "line_of_products", "make_of_car", "model_of_car",
}


def concept_entity_score(word: str, isa_getter) -> float:
    """Entity-likeness from the ConceptGraph IsA structure (research casing, Phase 5).

    Data-derived: walks the raw ``isa`` edges (via ``isa_getter(word) -> set``)
    up to depth 2 and returns 1.0 if any IsA target (or its parent) is a STRONG
    entity type. WEAK types (brand/product) do NOT count, so ambiguous words
    like "apple" (company_brand + edible_fruit) don't get force-capitalized
    against the lexical prior.

    ``isa_getter`` is injected (not imported) to avoid a circular dependency on
    the ontology module. The caller passes e.g.
    ``lambda w: ont.isa.get(w, set())``.

    Returns 0.0 when no ontology / no edges / OOV — so the SUBTLEX lexical prior
    (Phase 1/4) remains the authority for ordinary words.
    """
    if not word or isa_getter is None:
        return 0.0
    w = word.lower().replace(" ", "_")
    seen = set()
    frontier = [w]
    depth = 0
    while frontier and depth < 2:
        nxt = []
        for node in frontier:
            edges = isa_getter(node) or set()
            for tgt in edges:
                tl = (tgt or "").lower().replace(" ", "_")
                if tl in _ENTITY_STRONG:
                    return 1.0
                if tl not in seen:
                    seen.add(tl)
                    nxt.append(tl)
        frontier = nxt
        depth += 1
    return 0.0


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
                    elif wl in entity_words and store.cap_prob(wl) == 0.0:
                        # Graph entity signal wins ONLY for words with no
                        # SUBTLEX evidence (OOV) — when distributional data
                        # exists, trust it (e.g. "bank" is 88% lowercase despite
                        # being a 'company' in ConceptNet).
                        out_toks.append((lead, word[0].upper() + word[1:], trail))
                    elif store.cap_prob(wl) >= _CAP_THRESHOLD:
                        out_toks.append((lead, word[0].upper() + word[1:], trail))
                    else:
                        out_toks.append((lead, word, trail))
            else:
                # Tokens that aren't plain words (numbers, punctuation) unchanged.
                out_toks.append((lead, tok, ""))
        out_pieces.append("".join(lead + w + trail for lead, w, trail in out_toks))
    return " ".join(out_pieces)
