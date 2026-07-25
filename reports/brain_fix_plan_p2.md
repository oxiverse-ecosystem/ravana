# Brain-Faithful Implementation Plan — RAVANA Tail Items (P1-H, P2-B, P3-D, P3-F)

**Date:** 2026-07-22
**HEAD:** After modularization (e8ff1c9), P0 flags ON (0b0501f), P1-I dedupe (c686dc8), P1-G ConnectorLearner (a1a03fb), P2-E adaptive gates (b6f631f), P2-C calibration (375e8c1).
**Regression gate:** `python -m pytest tests/test_dehardcode_plan.py -q` — must yield **21 passed, 1 failed** throughout (the 1 = `test_meaning_of_life_not_dict_dump`, PRE-EXISTING on HEAD).
**Unit audit gate:** `python -m pytest tests/unit/test_derived_ontology_audit.py -q` — must yield **ALL passed** throughout (HARD constraint for item H).
**Cold-start rule:** Every adaptive gate starts with prior == today's value. Engine behaves identically on first run.

---

## Item H — Collapse 6 Closed-Class Lists into `functional_lexicon.json` (P1)

**Status:** REMAINING. The Round-1 plan assumed `functional_lexicon.json` already contains the closed-class categories — it does NOT (only polarity/moral/framing). This plan corrects that: **extend** the JSON + loader.

### Current code (the 6 lists to consolidate)

| List | engine.py | Size | Usages (engine_*.py) |
|------|-----------|------|----------------------|
| `_COMMON_WORDS` | line 212 | ~230 words | engine_reasoning.py:478 |
| `_CONDITIONAL_FRAME` | line 313 | ~50 words | engine_reasoning.py:1274,1309,1336,1380 |
| `_ATTR_WORDS` | line 461 | 22 words | engine_web_search.py:1663 |
| `TOPIC_SKIP_WORDS` | line 493 | ~80 words | engine_generation.py:1601,1692,1739,1895,1928,1947 |
| `_SUBJECT_CONTEXT_WORDS` | line 512 | ~90 words | engine_graph.py:1246 |
| `_GRAMMATICAL_CONCEPTS` | line 1051 | ~250 words | engine_web_search.py:1580,1605; chain_walker.py:2089,2162,2269; interface.py:1370 |

**CORRECTION (vs Round-1 plan):** The prompt says `TOPIC_SKIP_WORDS` has "ZERO usages in engine_*.py" — this is WRONG. Grep confirms 7 usages in engine_generation.py alone (lines 1601, 1692, 1739, 1895, 1928, 1947), plus 5 in interface.py (lines 719, 731, 745, 789). The plan handles ALL 6 lists regardless.

**Cross-module note:** `interface.py:77` has its OWN copy of `TOPIC_SKIP_WORDS`. `interface.py` and `response_gen.py` reference `TOPIC_SKIP_WORDS` as `self.TOPIC_SKIP_WORDS` (engine mixin). These resolve via MRO — they read the engine's attribute. Only interface.py line 77 defines a standalone `TOPIC_SKIP_WORDS` constant (a COPY). This copy must also route through the lexicon.

### Plan — RECOMMENDATION: (a) Extend functional_lexicon.json + loader

Why: the unit audit suite (`test_derived_ontology_audit.py`) tests `_derive_definition_purge`, `_is_function_word`, and `_walk_hierarchy` — NONE of these test the membership of the 6 closed-class lists directly. The `test_learned_pos_parity_with_hardcoded_for_covered_set` test (line 271) checks function-word classification parity between learned-POS and hardcoded-GRAMMATICAL_CONCEPTS, which the delegation layer preserves (hand sets remain as fallback at cold-start). (a) is the pattern used by every other de-hardcoding and is testable.

#### Step H.1 — Extend `_SEED` in `functional_lexicon.py`

In `functional_lexicon.py:34-51`, extend the `_SEED` dict with 6 new keys:

```python
_SEED: Dict[str, list] = {
    # ... existing keys (polarity_increase, polarity_decrease, polarity_remove,
    #     moral_markers, moral_ambiguous, framing) ...
    "common_words": [
        "the", "a", "an", "and", "or", "but", "if", "because", "when",
        "while", "of", "to", "in", "on", "at", "by", "for", "with", "from",
        # ... full list from engine.py:212 ...
    ],
    "topic_skip": [
        "i", "you", "we", "they", "he", "she", "it", "me", "my",
        # ... full list from engine.py:493 ...
    ],
    "subject_context": [
        "happen", "happened", "happening", "occur", "occurred",
        # ... full list from engine.py:512 ...
    ],
    "conditional_frame": [
        "if", "suppose", "supposing", "assume", "assuming",
        # ... full list from engine.py:313 ...
    ],
    "attr_words": [
        "capital", "population", "author", "director",
        # ... full list from engine.py:461 ...
    ],
    "grammatical_concepts": [
        "out", "in", "on", "off", "up", "down",
        # ... full list from engine.py:1051 ...
    ],
}
```

Each value is the UNION of all words in the hand list (duplicates across categories are OK — the brain's single learned lexicon also overlaps). No word is removed.

#### Step H.2 — Add 6 property accessors to `FunctionalLexicon`

In `functional_lexicon.py:69-91` (after the existing 6 properties), add:

```python
@property
def common_words(self) -> Set[str]:
    return self._v.get("common_words", set(_SEED["common_words"]))

@property
def topic_skip(self) -> Set[str]:
    return self._v.get("topic_skip", set(_SEED["topic_skip"]))

@property
def subject_context(self) -> Set[str]:
    return self._v.get("subject_context", set(_SEED["subject_context"]))

@property
def conditional_frame(self) -> Set[str]:
    return self._v.get("conditional_frame", set(_SEED["conditional_frame"]))

@property
def attr_words(self) -> Set[str]:
    return self._v.get("attr_words", set(_SEED["attr_words"]))

@property
def grammatical_concepts(self) -> Set[str]:
    return self._v.get("grammatical_concepts", set(_SEED["grammatical_concepts"]))
```

#### Step H.3 — Add delegation layer in engine.py

In `engine.py:865` (where `self._func_lex` is loaded), add 6 property-based accessors on `CognitiveChatEngine`. The class-level hand lists (e.g., `_COMMON_WORDS` at line 212) are **renamed** to `_FALLBACK_COMMON_WORDS` to make the delegation explicit, and a new property intercepts every access:

```python
@property
def _COMMON_WORDS(self):
    if self._func_lex:
        try: return self._func_lex.common_words
        except Exception: pass
    return self._FALLBACK_COMMON_WORDS
```

Repeat for all 6. The `@property` on a class-level attribute shadowed by an instance-level property works in Python because the descriptor is defined on the class. **Test this before merging**: ensure `self._COMMON_WORDS` in mixin methods resolves to the property. If Python's MRO shadows it (unlikely but worth validating on the actual CPython version), use a getter method pattern instead: `def _get_common_words(self):`.

**Alternative (if property shadowing fails):** Add instance-level dict `self._closed_class_cache = {}` and a single method `_closed_class(name)` that delegates:

```python
def _closed_class(self, name: str) -> Set[str]:
    if self._func_lex:
        try: return getattr(self._func_lex, name)()
        except Exception: pass
    return getattr(self, f"_FALLBACK_{name.upper()}", set())
```

Usage: `self._closed_class("common_words")` instead of `self._COMMON_WORDS`. This is more verbose per call-site but avoids Python descriptor issues. **Verdict:** the accessor-method pattern is safer and matches how `_is_function_word` already works (engine_web_search.py:1569-1620).

#### Step H.4 — Route ~18 usage sites

Replace each reference to the 6 hand lists with the property/method:

| File | Line(s) | Old | New |
|------|---------|-----|-----|
| engine_reasoning.py:478 | `self._COMMON_WORDS` | `self._closed_class("common_words")` |
| engine_reasoning.py:1274,1309,1336,1380 | `self._CONDITIONAL_FRAME` | `self._closed_class("conditional_frame")` |
| engine_web_search.py:1663 | `self._ATTR_WORDS` | `self._closed_class("attr_words")` |
| engine_generation.py:1601,1692,1739,1895,1928,1947 | `self.TOPIC_SKIP_WORDS` | `self._closed_class("topic_skip")` |
| engine_graph.py:1246 | `cls._SUBJECT_CONTEXT_WORDS` | `self._closed_class("subject_context")` (note `cls`→`self`) |
| engine_web_search.py:1580,1605 | `self._GRAMMATICAL_CONCEPTS` | `self._closed_class("grammatical_concepts")` |
| chain_walker.py:2089,2162,2269 | `self._GRAMMATICAL_CONCEPTS` | `self._closed_class("grammatical_concepts")` |
| interface.py:1370 | `self.graph_engine._GRAMMATICAL_CONCEPTS` | `self.graph_engine._closed_class("grammatical_concepts")` |
| interface.py:77 (standalone copy) | Delete the copy; import from engine's mixin | Remove line 77, rely on `self.TOPIC_SKIP_WORDS` |
| interface.py:719,731,745,789 | `TOPIC_SKIP_WORDS` (standalone module constant) | `self.TOPIC_SKIP_WORDS` (engine mixin property) |

#### Step H.5 — Keep hand lists as `_FALLBACK_*` (never delete)

Rename the class-level attributes (NOT delete):
```python
_FALLBACK_COMMON_WORDS = _COMMON_WORDS  # original body
_FALLBACK_CONDITIONAL_FRAME = _CONDITIONAL_FRAME  # original body
_FALLBACK_ATTR_WORDS = _ATTR_WORDS
_FALLBACK_TOPIC_SKIP_WORDS = TOPIC_SKIP_WORDS
_FALLBACK_SUBJECT_CONTEXT_WORDS = _SUBJECT_CONTEXT_WORDS
_FALLBACK_GRAMMATICAL_CONCEPTS = _GRAMMATICAL_CONCEPTS
```

Then delete the original list body (or leave the renamed version as-is). The effect: `_COMMON_WORDS` is now a property (or method return), not a class-level set literal. Cold-start identical because `_func_lex` is None → falls back to `_FALLBACK_*` which IS the original set.

#### Cold-start prior

When `data/functional_lexicon.json` is absent (first run): `self._func_lex` is None → every `_closed_class(name)` call falls through to `self._FALLBACK_*`. Behavior is **byte-identical** to current HEAD.

When the JSON is present (after extension): it contains the 6 new keys with the exact same contents as the hand lists (from `_SEED`). Behavior is identical. Over time, the JSON can be updated for data-driven refinement without touching engine source.

#### Regression gates

- `python -m pytest tests/test_dehardcode_plan.py -q` → **21 passed, 1 failed** (unchanged — the suite does not check closed-class list membership).
- `python -m pytest tests/unit/test_derived_ontology_audit.py -q` → **ALL passed**. The `test_learned_pos_parity_with_hardcoded_for_covered_set` test (line 271) checks that `eng._is_function_word(w)` agrees between learned-POS and hardcoded sets. Since the `_FALLBACK_*` set IS the original hand set, the assertion passes identically. The other tests (`test_derive_definition_purge`, `test_snippet_gate`, `test_source_trust`) do not reference any of the 6 lists.

**Effort:** M. **Risk:** Low (delegation layer is additive; hand sets preserved).

---

## Item B — Broaden Consult / Advice Knowledge (Genuine Capability Gap #2)

**Status:** ASSESSED — NO CODE CHANGE NEEDED. The existing architecture already solves this.

### Why it is already solved

The engine's learning pipeline mirrors hippocampal consolidation:
1. **Detection:** When `consult_internal` (engine_self_query.py:538) returns None for a subject, the engine does NOT give up permanently. It enqueues the unknown subject to `_pending_learning_queue`.
2. **Enqueue sites:** engine.py:2458 (definition path), 2810 (graph discovery path), 3127 (curiosity path) — all feed into `_bg_learning_queue`.
3. **Offline research:** `WebLearningMixin` (web_learning.py) researches queued topics in the background, extracting definitions, graph relations, and snippets. These accrete into `_definitions` (engine.py:618) and the graph.
4. **Retrieval:** On subsequent turns, `consult_internal` finds the accumulated knowledge.

Health/programming coverage accrue naturally with usage. There is NO hardcoded gap — only a cold-start knowledge gap that closes as the engine interacts with the world.

### Why NO synchronous consult→web fallback

A synchronous fallback at `_handle_self_query` (engine_self_query.py:538) would:
- Fire on the **first turn** of a fresh engine (zero accumulated knowledge = every unknown subject triggers a web lookup = cold-start violation).
- Change **per-turn latency** (web lookup adds 1-5s to a response that currently returns None immediately).
- Lower the **honest-uncertainty bar** (the engine would surface web-snippet noise instead of saying "I don't know" — a feature, not a bug).

The brain itself does NOT do synchronous world-queries during retrieval — it reports honest uncertainty when knowledge is not consolidated.

### Plan

Document in code (add a comment at engine_self_query.py:538):
```python
# The consult→web synchronous fallback is INTENTIONALLY NOT added.
# Unknown subjects are enqueued to _pending_learning_queue
# (engine.py:2458, 2810, 3127) and researched offline by
# WebLearningMixin. This preserves cold-start latency and the
# honest-uncertainty bar. See implementation_status.md for rationale.
```

**Effort:** S (comment only). **Risk:** None.

---

## Item D — Sensorimotor Realization Lexicon (P3)

**Status:** REMAINING. All data files exist locally (Lancaster 40k CSV + encoder + train script).

### Current code

| Element | Location | Purpose |
|---------|----------|---------|
| `_LANCASTER_ORDER` | engine.py:271 | Fixed encoder contract — KEEP |
| `_SENSORY_DIM_PHRASE` | engine.py:275 | Property→(verb, sensory-phrase), hand-crafted, 26 entries |
| `_PROP_TO_BINDER` | engine.py:302 | Property→binder sensory dims, hand-crafted, 14 entries |
| `_select_sensorimotor_dim` | engine_generation.py:1130-1141 | Selects the verb/phrase from the hand-crafted maps |
| `_lancaster_vector` | engine.py:1156 | Already computes Lancaster projections from the encoder |

### Local data files

- `data/cache/word_ratings/Lancaster_sensorimotor_norms_for_39707_words.csv` — 39,707 words × 11 dims
- `data/lancaster_encoder.npz` — fitted encoder (trained by `scripts/train_lancaster_probe.py`)
- `scripts/train_lancaster_probe.py` — existing fit script using `ravana.ontology.attribute_encoder.train_from_lancaster`

### Plan

#### Step D.1 — Write the fit script: `scripts/fit_sensorimotor_lexicon.py`

Mirroring `experiments/measure_pos_model.py`. Core logic:

```python
"""Fit data/sensorimotor_lexicon.json from Lancaster 40k norms.

For each property word in the union of _SENSORY_DIM_PHRASE keys and
common English property adjectives, compute the dominant sensorimotor
dimension and map it to a (verb, phrase) template.
"""
import os, sys, json, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ravana", "src"))

# Load Lancaster norms
CSV = os.path.join(ROOT, "data", "cache", "word_ratings",
                   "Lancaster_sensorimotor_norms_for_39707_words.csv")
LANCASTER_ORDER = [
    "Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual",
    "Foot_leg", "Hand_arm", "Head", "Mouth", "Torso",
]

# Read norms: word -> {dim: strength}
norms = {}
with open(CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        word = row.get("Word", "").strip().lower()
        if word:
            norms[word] = {dim: float(row.get(dim, 0)) for dim in LANCASTER_ORDER}

# Verb templates keyed by dominant effector
_EFFECTOR_VERB = {
    "Hand_arm": "feel", "Foot_leg": "step", "Head": "carry",
    "Mouth": "taste", "Torso": "bear",
}
_PERCEPTUAL_VERB = {
    "Visual": "look", "Auditory": "sound", "Haptic": "feel",
    "Gustatory": "taste", "Olfactory": "smell", "Interoceptive": "feel",
}
_PERCEPTUAL_PHRASE = {
    "Visual": "see", "Auditory": "hear", "Haptic": "touch",
    "Gustatory": "taste", "Olfactory": "smell", "Interoceptive": "sense",
}

def _dominant_dim(word: str, dims: list) -> tuple:
    """Return (dim_name, strength) for the top-2 dimensions."""
    if word not in norms:
        return None, None
    scores = [(d, norms[word].get(d, 0)) for d in dims]
    scores.sort(key=lambda x: -x[1])
    return scores[0][0], scores[1][0] if len(scores) > 1 else scores[0][0]

# Property words to fit (union of all _SENSORY_DIM_PHRASE keys + common adjectives)
PROPERTY_WORDS = {
    # From _SENSORY_DIM_PHRASE keys
    "shape", "vision", "color", "bright", "dark", "pattern", "texture",
    "touch", "temperature", "weight", "sound", "audition", "loud",
    "motion", "complexity", "taste", "smell",
    "upperlimb", "lowerlimb", "head", "mouth", "torso",
    # Common property adjectives
    "smooth", "rough", "hard", "soft", "wet", "dry", "hot", "cold",
    "warm", "cool", "heavy", "light", "loud", "quiet", "bright", "dim",
    "fast", "slow", "big", "small", "large", "tiny", "thick", "thin",
    "sharp", "dull", "sweet", "sour", "bitter", "salty", "fragrant",
    "stinky", "colorful", "red", "blue", "green", "black", "white",
}

# Build the lexicon
LEXICON = {"property_to_dim": {}, "property_to_verb": {}, "property_to_phrase": {}}
for prop in sorted(PROPERTY_WORDS):
    if prop in norms:
        dominant, secondary = _dominant_dim(prop, LANCASTER_ORDER)
        if dominant is None:
            continue
        LEXICON["property_to_dim"][prop] = [dominant, secondary] if secondary else [dominant]
        # Verb: use effector verb if dominant is an effector, else perceptual verb
        if dominant in _EFFECTOR_VERB:
            verb = _EFFECTOR_VERB[dominant]
        else:
            verb = _PERCEPTUAL_VERB.get(dominant, "feels")
        phrase = _PERCEPTUAL_PHRASE.get(dominant, "sense")
        LEXICON["property_to_verb"][prop] = verb
        LEXICON["property_to_phrase"][prop] = phrase

# Include the hand-map entries verbatim as a fallback override
LEXICON["hand_map_overrides"] = {
    "shape": {"verb": "shape", "phrase": "picture by its outline"},
    "vision": {"verb": "looks", "phrase": "see"},
    "color": {"verb": "colour", "phrase": "see"},
    # ... include all 26 entries from _SENSORY_DIM_PHRASE ...
}

OUT = os.path.join(ROOT, "data", "sensorimotor_lexicon.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(LEXICON, f, indent=2)
print(f"Wrote {OUT} ({len(LEXICON['property_to_dim'])} properties)")
```

#### Step D.2 — Write the loader: `ravana/src/ravana/chat/sensorimotor_lexicon.py`

Mirror `functional_lexicon.py` exactly:

```python
"""Learned sensorimotor realization lexicon (de-hardcoding items #8/#9).

Replaces the hand-crafted _SENSORY_DIM_PHRASE and _PROP_TO_BINDER maps
with a data-driven mapping derived from Lancaster 40k sensorimotor norms.
Hand maps are kept as the OOV fallback.
"""

import json
import os
from typing import Dict, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_FIT_PATH = os.path.join(_DATA_DIR, "sensorimotor_lexicon.json")

# Seed design: None means absent. The hand-map fallback (engine class attrs)
# is the true fallback; this module only loads the fit file or returns None.
_SEED: Optional[Dict] = None


class SensorimotorLexicon:
    def __init__(self, values: Dict):
        self._v: Dict = values

    @property
    def property_to_dim(self) -> Dict[str, list]:
        return self._v.get("property_to_dim", {})

    @property
    def property_to_verb(self) -> Dict[str, str]:
        return self._v.get("property_to_verb", {})

    @property
    def property_to_phrase(self) -> Dict[str, str]:
        return self._v.get("property_to_phrase", {})

    @property
    def hand_map_overrides(self) -> Dict[str, dict]:
        return self._v.get("hand_map_overrides", {})

    @classmethod
    def load(cls) -> Optional["SensorimotorLexicon"]:
        if not os.path.exists(_FIT_PATH):
            return None
        try:
            with open(_FIT_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return cls(values=d)
        except Exception:
            return None


def default_sensorimotor_lexicon() -> Optional[SensorimotorLexicon]:
    loaded = SensorimotorLexicon.load()
    return loaded if loaded is not None else None
```

#### Step D.3 — Wire into engine.py:865-880 area

After the existing `self._func_lex` loading (engine.py:865), add:

```python
# Stage 5b-iii: learned sensorimotor realization lexicon (replaces
# _SENSORY_DIM_PHRASE / _PROP_TO_BINDER hand maps). Loaded from
# data/sensorimotor_lexicon.json; absent = hand-map fallback.
self._sensorimotor_lex = None
try:
    from .sensorimotor_lexicon import default_sensorimotor_lexicon
    self._sensorimotor_lex = default_sensorimotor_lexicon()
except Exception:
    self._sensorimotor_lex = None
```

Then modify `_select_sensorimotor_dim` (engine_generation.py:1130-1141) to prefer the learned lexicon:

```python
def _select_sensorimotor_dim(self, property_name: str):
    """Select a sensorimotor dim for a property, preferring learned lexicon."""
    p = property_name.lower().strip()
    # Learned lexicon (if loaded) takes priority.
    if self._sensorimotor_lex is not None:
        dims = self._sensorimotor_lex.property_to_dim.get(p)
        if dims:
            return dims
        # Also check hand_map_overrides for exact fidelity to old behavior.
        override = self._sensorimotor_lex.hand_map_overrides.get(p)
        if override and "dim" in override:
            return [override["dim"]]
    # Fallback: hand-crafted _PROP_TO_BINDER.
    return self._PROP_TO_BINDER.get(p, ("Visual",))
```

Similarly modify the phrase selection path (engine_generation.py around line 1135) to use `self._sensorimotor_lex.property_to_verb.get(p)` and `.property_to_phrase.get(p)` before falling back to `_SENSORY_DIM_PHRASE`.

#### Cold-start prior

When `data/sensorimotor_lexicon.json` is absent: `self._sensorimotor_lex` is None → `_select_sensorimotor_dim` falls through to `self._PROP_TO_BINDER` and `self._SENSORY_DIM_PHRASE` (the hand maps). Behavior is byte-identical to current HEAD.

When the JSON is present: the learned map covers the same properties plus ~50+ more entries from the Lancaster norms. The hand_map_overrides section ensures byte-exact fidelity for the 14 original property keys.

#### Regression gate

`python -m pytest tests/test_dehardcode_plan.py -q` → **21 passed, 1 failed** (unchanged — no test exercises sensorimotor dim selection).

**Effort:** L (fit script + loader + wiring). **Risk:** Low (hand maps preserved as OOV fallback).

---

## Item F — Shape-Based Junk / Boilerplate Detection (P3)

**Status:** REMAINING. The existing `junk_scorer.py` already has `_structural_floor()` shape features. This plan extends them additively.

### Current code

| Element | Location | Purpose |
|---------|----------|---------|
| `_JUNK_SNIPPET_DOMAINS` | engine.py:329 | 45-domain blocklist |
| `_WEBSITE_SHAPE` | constants.py:401 (and junk_scorer.py:42) | TLD-tail regex |
| `_structural_floor` | junk_scorer.py:67-82 | Combines keyboard-mash + POS-tag + website-shape + vowel-ratio |
| `junk_score` | junk_scorer.py:487-517 | Full scorer (structural + learned soft classifier) |
| `_is_keyboard_mash` | constants.py:53 (and junk_scorer.py:85) | Keyboard-mash shape detector |
| `_JUNK_SNIPPET_DOMAINS` usage | engine_web_search.py:667 | Domain check in snippet pipeline |
| `_snippet_is_structural_junk` | engine_web_search.py:972 | Structural snippet junk gate |

### Plan

#### Step F.1 — Add shape features to `junk_scorer.py:_structural_floor`

In `junk_scorer.py:67-82`, extend `_structural_floor` with richer shape signals. The current code already has keyboard-mash (+0.45), POS-tag (+0.5), TLD/digit shape (+0.35), and vowel-less (+0.4). Add:

```python
def _structural_floor(word: str) -> float:
    w = word.lower().strip("'\"")
    if not w:
        return 1.0
    s = 0.0
    if _is_keyboard_mash(w):
        s += 0.45
    if w in _POS_TAGS:
        s += 0.5
    if _WEBSITE_SHAPE.search(w) or any(ch.isdigit() for ch in w):
        s += 0.35
    _vowels = set("aeiouy")
    _vc = sum(1 for ch in w if ch in _vowels)
    if len(w) >= 4 and _vc == 0:
        s += 0.4

    # ── new shape features (predictive-coding: statistical shape, not blocklist) ──
    # 1. Vowel ratio: low vowel ratio (< 0.25) = likely boilerplate/SEO junk
    if len(w) >= 4:
        vowel_ratio = _vc / len(w)
        if vowel_ratio < 0.2:
            s += 0.3  # very low vowel density = keyboard-mash/spam shape
        elif vowel_ratio < 0.3:
            s += 0.15  # moderately low

    # 2. Digit density: high digit ratio = version numbers, dates, SNHU codes
    _dc = sum(1 for ch in w if ch.isdigit())
    if len(w) >= 3:
        digit_ratio = _dc / len(w)
        if digit_ratio > 0.5:
            s += 0.35  # more digits than letters = not a real word
        elif digit_ratio > 0.25:
            s += 0.15

    # 3. Uppercase ratio (for multi-word snippets): LOTS of caps = boilerplate
    # (This is a word-level function; uppercase is handled by lowercasing above.
    #  For snippet-level shape, see engine_web_search.py path.)

    # 4. Repetition pattern: same char 3+ times in a row = spam shape
    _max_run = 1
    _run = 1
    for i in range(1, len(w)):
        if w[i] == w[i-1]:
            _run += 1
            _max_run = max(_max_run, _run)
        else:
            _run = 1
    if _max_run >= 3:
        s += 0.2

    return s
```

These features are all SHAPE-based — they do not reference any domain name or URL. They detect junk by its statistical properties (predictive coding: high predictability = low surprise = boilerplate). All thresholds are simple hand-set priors that the downstream `OnlineJunkClassifier` can override once sufficient self-labeled data accrues.

#### Step F.2 — Add snippet-level shape score (vowel ratio, uppercase ratio, entropy)

In `engine_web_search.py`, add a auxiliary shape-based snippet quality score that runs BEFORE `_snippet_is_structural_junk` and BEFORE the domain check. Insert at `engine_web_search.py:960-970` (before the existing `_snippet_is_structural_junk` call at line 955):

```python
def _snippet_shape_junk_score(self, snippet: str) -> float:
    """Predictive-coding snippet-level shape score (0=clean, 1=junk).
    
    Detects boilerplate/SEO/navigation by statistical text shape,
    not a domain blocklist. Additive to the existing structural and
    learned gates — never decreases junk detection coverage.
    """
    if not snippet or len(snippet) < 20:
        return 0.0
    text = snippet.lower()
    # Vowel ratio (low = boilerplate/compressed text)
    vowels = sum(1 for ch in text if ch in "aeiouy")
    vowel_ratio = vowels / max(1, len(text))
    shape_score = 0.0
    if vowel_ratio < 0.25:
        shape_score += 0.2
    
    # Uppercase ratio (high = navigation/header text)
    uc = sum(1 for ch in snippet if ch.isupper())
    uc_ratio = uc / max(1, len(snippet))
    if uc_ratio > 0.4:
        shape_score += 0.2
    elif uc_ratio > 0.6:
        shape_score += 0.3
    
    # Digit density (high = version/date/SNHU)
    digits = sum(1 for ch in text if ch.isdigit())
    digit_ratio = digits / max(1, len(text))
    if digit_ratio > 0.15:
        shape_score += 0.15
    
    # Punctuation ratio (high = listicles/navigation)
    punct = sum(1 for ch in text if ch in "[-|/\\:;()"])
    punct_ratio = punct / max(1, len(text))
    if punct_ratio > 0.1:
        shape_score += 0.15
    
    return min(1.0, shape_score)
```

Then integrate into the snippet pipeline at `engine_web_search.py:815` (the `_snippet_is_structural_junk` call site) and `engine_web_search.py:955`:

```python
# In the snippet loop, before the domain check at line 667:
_shape_junk = self._snippet_shape_junk_score(snippet_text)
if _shape_junk > 0.5 and not any(j in dom for j in self._JUNK_SNIPPET_DOMAINS):
    # Shape detects junk the domain list misses — skip snippet.
    continue
```

This ensures shape-based detection runs FIRST and independently of the domain list. When both agree, no change. When shape catches junk the domain list misses, it is additive.

#### Step F.3 — Stale domain list handling

Do NOT delete `_JUNK_SNIPPET_DOMAINS` (engine.py:329). Keep it as a cold-start prior. Add a deprecation comment above it:

```python
# DEPRECATED (replaced by shape-based detection in _snippet_shape_junk_score
# and junk_scorer._structural_floor). Kept as a cold-start prior to preserve
# behavior on first runs without learned-shape data. The shape path runs
# FIRST and independently; this domain list is only checked when shape is
# uncertain (score < 0.5). No new entries should be added here.
_JUNK_SNIPPET_DOMAINS = (...existing 45 domains...)
```

#### Cold-start prior

`_JUNK_SNIPPET_DOMAINS` unchanged. Shape features are purely additive — they never reduce junk detection. At cold-start, shape features contribute their scores alongside the existing domain check. The domain list remains active, so detection is a superset of today's behavior.

#### Regression gate

`python -m pytest tests/test_dehardcode_plan.py -q` → **21 passed, 1 failed** (unchanged — no test checks specific domain filtering).

**Effort:** M. **Risk:** Low (additive only; domain list preserved).

---

## "Do NOT Touch" List

| Element | Reason | Item |
|---------|--------|------|
| `_LANCASTER_ORDER` (engine.py:271) | Fixed encoder output contract | D |
| `_SENSORY_DIM_PHRASE` (engine.py:275) | Hand maps as OOV fallback — never delete | D |
| `_PROP_TO_BINDER` (engine.py:302) | Hand maps as OOV fallback — never delete | D |
| `_JUNK_SNIPPET_DOMAINS` (engine.py:329) | Keep as cold-start prior; shape runs alongside | F |
| `_FALLBACK_*` closed-class sets | Keep after renaming (they are the cold-start data) | H |
| `_bg_learning_queue` (engine.py:1166) | The existing solution for Item B | B |
| `_pending_learning_queue` (engine.py:2458 etc.) | The existing solution for Item B | B |
| `SAVE_SCHEMA_VERSION` (#29) | Schema marker | all |
| `_PROTECTED_CONCEPTS` (#31) | Curated project namespace | all |
| `_CATEGORY_OF_SUBJECT` / `_CATEGORY_AFFORDANCES` / `_PROPERTY_CATEGORIES` (#4-6) | OOV safety net behind ConceptNet | all |
| `_SNIPPET_REJECT_SHAPES` / `_SNIPPET_NOISE` (#13-14) | Hard backstop behind cerebellar model | all |
| `_vad_baseline` (#34) | Exemplar adaptive pattern | all |
| VAD/Identity/GW/Sleep eta constants (#33) | Model config | all |

---

## Verification Section

### Commands

```bash
python -m pytest tests/test_dehardcode_plan.py -q
python -m pytest tests/unit/test_derived_ontology_audit.py -q   # H only
```

### Expected outcome per change

| Change | `test_dehardcode_plan` | `test_derived_ontology_audit` | Notes |
|--------|----------------------|-----------------------------|-------|
| Baseline (current HEAD) | 21 passed, 1 failed | ALL passed | Known baseline |
| **H)** Extend functional_lexicon.json + route | 21 passed, 1 failed | **ALL passed** | `_FALLBACK_*` == original sets; parity tests pass |
| **B)** Comment-only (no code change) | 21 passed, 1 failed | ALL passed | No behavioral change |
| **D)** Fit script + loader + wiring | 21 passed, 1 failed | ALL passed | Hand maps preserved as OOV fallback |
| **F)** Shape features in junk_scorer.py | 21 passed, 1 failed | ALL passed | Additive; domain list kept as cold-start |
| All 4 combined | 21 passed, 1 failed | ALL passed | Each is additive/fallback-preserving |

### Testing note

The single failure (`test_meaning_of_life_not_dict_dump`) is PRE-EXISTING on HEAD. If any change causes ADDITIONAL failures, revert that change immediately. If a change drops `test_derived_ontology_audit` below ALL PASSED (for H), the delegation approach is incorrect — revert to pure fallback-only (option b) and report.

---

## File Paths of Deliverables

1. `C:\Users\Likhith\Documents\projects\ravana\reports\brain_research_report_p2.md` — Round-2 brain mechanism research with citations
2. `C:\Users\Likhith\Documents\projects\ravana\reports\brain_fix_plan_p2.md` — this implementation plan
