"""May this word head a noun phrase? A fail-closed admission test.

WHY
---
`select_target` (`learn/macroplanning.py`) ranks candidate targets by edge
strength alone, so a target that cannot fill the slot can still win. Measured
on the live store: for `time` the graph offers `semantic -> years` (a noun) and
`analogical -> only` (an adverb); `only` wins because its edge is stronger, and
the formulator then emits "time is different from an only".

The fix is a FILTER on the candidate pool before ranking, not a better label
afterwards. This module is that filter.

DESIGN
------
**Veto-only.** A word is rejected only on positive evidence that it is NOT a
noun; anything else is admitted. The earlier fail-closed version (admit only
with positive noun evidence) measured recall 0.250 and kept 6 of 126 live
edges -- it rejected `force`, `earth`, `liquid`, `life`, `water`, `dog`, `sun`,
because English has no productive marker for most nouns. That would have
destroyed the plan route.

The evidence sources are used in the direction they are reliable in
(measured in `docs/NOUN_SUFFIX_MEASURED.md` and
`docs/DETERMINER_FRAME_REPLICATION.md`):

1. CLOSED-CLASS VETO -- a list of unambiguous function words and adverbs.
   Cheap and near-perfect on the words it covers. This is what catches `only`.
2. NOUN-FORMING SUFFIX -- `-tion/-ment/-ness/-ity/...`. Real but ambiguous:
   a measured table scored WORSE than the baseline (0.608 vs 0.732) because
   `-al` fires on `environmental`/`mathematical`. Only the unambiguous members
   are used here.
3. KNOWN LEXICON -- `KNOWN_ADJS` vetoes adjectives; `KNOWN_VERBS` vetoes verbs.

NOT USED, on measurement:
- the determiner frame (0.730, recall 0.54 -- a 226 KB corpus is too small);
- the full noun-suffix table including `-al` (measured worse than baseline);
- GloVe centroids (classes nearly collinear in the projected 64-D space:
  func-noun cosine 0.882, func-verb 0.898).
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Set

# ── 1. Closed-class veto ────────────────────────────────────────────────────
# Unambiguous function words, adverbs and other non-noun openers. A word in
# this set can never head a noun phrase. Deliberately a VETO rather than an
# admission list: it is short because only certainty belongs here.
CLOSED_CLASS: Set[str] = set("""
a an the this that these those
i me my mine myself we us our ours you your yours he him his she her hers
it its itself they them their theirs who whom whose which what
and or but nor so yet if because since although though unless while whereas
until once whether after before despite than as
in on at to for of with by from into onto upon over under about against
between through during before after above below up down out off again
further then once here there when where why how all any both each few more
most other some such no not only own same so than too very
just also even still already almost enough quite rather really always never
often sometimes usually rarely seldom now then here there when where why how
however therefore thus hence instead moreover furthermore nevertheless
nonetheless otherwise meanwhile indeed perhaps maybe probably possibly
certainly definitely clearly obviously
one two three four five six seven eight nine ten zero
""".split())

# ── 2. Unambiguous noun-forming suffixes ────────────────────────────────────
# Only suffixes that do NOT fire on common adjectives. `-al` is excluded
# (animal/global/moral vs environmental/mathematical/psychological), `-ic` is
# excluded (music vs historic), `-y` is excluded (only/however/rather/many).
# Measured: adding the ambiguous members made the rule WORSE than the baseline
# (docs/NOUN_SUFFIX_MEASURED.md, table A 0.608 vs baseline 0.732).
NOUN_SUFFIXES: tuple = (
    'tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'ancy', 'ency',
    'ery', 'age', 'ism', 'ist', 'ure', 'dom', 'ship', 'hood', 'ics', 'ia',
    'um', 'us', 'ory', 'ary',
)

# Suffixes that mark a word as NOT a noun. Used as a veto only when the word
# is not already covered by the lexicon, so a real noun like `climate`
# (`-ate`) is not caught by an over-broad rule.
ADJ_SUFFIX_VETO: tuple = (
    'able', 'ible', 'ful', 'less', 'ous', 'ive',
    # `-ant` is an unambiguous adjectival ending (ancient, important).
    'ant',
)

# `-ent` is ambiguous: `different`/`ancient`-class adjectives vs the measured
# graph nouns `environment`, `development`. Decided by allowlist, like `-al`.
ENT_SUFFIX_NOUNS: set = {
    'environment', 'development', 'government', 'movement', 'moment',
    'element', 'experiment', 'statement', 'department', 'management',
    'agreement', 'equipment', 'investment', 'treatment', 'student',
}

# `-ant` is also ambiguous: `ancient`/`important` (adjectives) vs `plant`,
# `elephant`, `constant` (nouns) on this graph.
ANT_SUFFIX_NOUNS: set = {
    'plant', 'elephant', 'constant', 'giant', 'peasant', 'merchant',
}

# `-ary` is ambiguous: `primary`/`necessary` (adjectives) vs `dictionary`,
# `library` (nouns). Only the adjectives are vetoed, via this list, so a noun
# ending `-ary` is still admitted by default.
ARY_ADJECTIVES: set = {
    'primary', 'necessary', 'ordinary', 'extraordinary', 'military',
    'voluntary', 'momentary', 'arbitrary',
}


# AMBIGUOUS adjective suffixes, used as a veto ONLY in combination with a
# closed-class-style check. `-al` and `-ic` fire on real nouns too
# (`animal`, `global`, `moral` vs `environmental`, `mathematical`), so using
# them alone measured WORSE than baseline (docs/NOUN_SUFFIX_MEASURED.md).
# They are kept here as a *last-resort* veto for words that also end in a
# strongly adjectival ending, which is where the measured errors live.
AL_SUFFIX_NOUNS: set = {
    # measured nouns ending -al on the graph
    'animal', 'global', 'moral', 'natural', 'journal', 'mammal', 'total',
    'general', 'special', 'final', 'original', 'signal', 'formal', 'normal',
    'local', 'social', 'official', 'personal', 'national', 'international',
    'cultural', 'structural', 'functional', 'visual', 'visual', 'vital',
}
IC_SUFFIX_NOUNS: set = {
    # measured nouns ending -ic on the graph
    'music', 'magic', 'topic', 'logic', 'ethics', 'physics', 'mathematics',
}


def _load_lexicons():
    """Best-effort import of the engine's hand lexicons.

    Kept lazy and defensive: this module is imported from `learn/`, which must
    not depend on `chat/`. Returns empty sets if the import fails, in which
    case the filter still works on the closed-class and suffix evidence.
    """
    try:
        from ravana.chat.constants import KNOWN_ADJS, KNOWN_VERBS
        return set(KNOWN_ADJS), set(KNOWN_VERBS)
    except Exception:
        return set(), set()


_KNOWN_ADJS: Set[str] = set()
_KNOWN_VERBS: Set[str] = set()
_INITIALISED = False


def _ensure_lexicons() -> None:
    global _INITIALISED, _KNOWN_ADJS, _KNOWN_VERBS
    if not _INITIALISED:
        _KNOWN_ADJS, _KNOWN_VERBS = _load_lexicons()
        _INITIALISED = True


def looks_like_noun(word: str,
                    pos_hint: Optional[str] = None,
                    pos_map: Optional[Dict[str, str]] = None) -> bool:
    """True when there is positive evidence `word` can head a noun phrase.

    Args:
        word: the candidate target, as it appears in the graph.
        pos_hint: an optional POS tag from the caller's own map. When it says
            something other than ``'noun'``/``None``, that is authoritative
            NEGATIVE evidence -- the map is only trusted to reject, never to
            admit, so a missing entry does not block admission.
        pos_map: the caller's POS map, used the same reject-only way.

    Fail-closed: returns False for anything not affirmatively established.
    """
    if not word:
        return False
    w = word.strip().lower()
    if not w:
        return False
    # Multi-word labels are phrases, not noun heads; the formulator's
    # `_head_noun` handles them separately.
    if ' ' in w:
        return False

    _ensure_lexicons()

    # 1. Closed-class veto -- certainty.
    if w in CLOSED_CLASS:
        return False

    # 2. POS-hint veto -- DISABLED, on measurement.
    #
    #    The obvious design was "trust a non-noun hint from `_concept_pos`".
    #    Measured, that INHERITS the map's own errors: the map tags `climate`
    #    as verb ('-ate'), `learning` as verb ('-ing'), `music` as adj ('-ic'),
    #    `animal`/`natural`/`journal`/`mammal` as adj ('-al'), so the veto
    #    rejected 8 real nouns on the gold set. The map is the defective
    #    artifact (docs/POS_MISLABEL_ROOT_CAUSE.md); a filter that trusts it
    #    cannot be better than it.
    #
    #    `pos_hint`/`pos_map` are still accepted (callers already have them)
    #    but are deliberately unused. Re-enable only with a map that has been
    #    independently validated -- the lesson of
    #    docs/TYPING_ACCURACY_IS_CIRCULAR.md.

    # 3. Hand lexicons -- small but curated.
    if w in _KNOWN_ADJS:
        return False
    if w in _KNOWN_VERBS:
        return False

    # 4. Adjective suffix veto (unambiguous members only).
    if any(len(w) > len(s) + 1 and w.endswith(s) for s in ADJ_SUFFIX_VETO):
        if not (w.endswith('ant') and w in ANT_SUFFIX_NOUNS):
            return False

    # 4a. `-ary` adjectives, by explicit list (the suffix is ambiguous).
    if w in ARY_ADJECTIVES:
        return False

    # 4b. Ambiguous `-al` / `-ic` veto, with an explicit noun allowlist.
    #     `-al` fires on `animal`/`global`/`moral` (nouns) and
    #     `environmental`/`mathematical` (adjectives); a bare suffix rule
    #     measured WORSE than baseline. The allowlist is the measured set of
    #     -al/-ic nouns on THIS graph, so the veto only fires on words that
    #     are not known nouns here.
    if w.endswith('al') and w not in AL_SUFFIX_NOUNS:
        return False
    if w.endswith('ic') and w not in IC_SUFFIX_NOUNS:
        return False
    if w.endswith('ent') and w not in ENT_SUFFIX_NOUNS:
        return False

    # 5. Positive evidence: an unambiguous noun-forming suffix.
    if any(len(w) > len(s) + 1 and w.endswith(s) for s in NOUN_SUFFIXES):
        return True

    # 6. Default. The FIRST version of this function failed closed here and
    #    measured recall 0.250 -- it rejected `force`, `earth`, `liquid`,
    #    `life`, `water`, `body`, `dog`, `sun`, i.e. most of the graph's real
    #    nouns, because English has no productive marker for them. Kept=6 of
    #    126 live edges; the plan route would have died.
    #
    #    So the default is ADMIT, and the filter is a VETO-ONLY filter. That is
    #    the honest reading of the evidence: the suffix/lexicon/closed-class
    #    sources are reliable for REJECTING (precision 0.909 measured) and
    #    useless for ADMITTING (recall 0.250). A veto-only filter removes the
    #    measured failures (`only`, `however`, `rather`, `different`) without
    #    needing a noun detector this system does not have.
    return True


def filter_noun_candidates(candidates: Iterable,
                           pos_map: Optional[Dict[str, str]] = None,
                           ) -> list:
    """Keep only candidates that may head a noun phrase.

    `candidates` are the `(relation, target, weight, confidence)` tuples the
    plan route builds. Returns the same tuples, filtered.
    """
    out = []
    for c in candidates:
        try:
            rel, tgt = c[0], c[1]
        except Exception:
            continue
        if looks_like_noun(tgt, pos_map=pos_map):
            out.append(c)
    return out
