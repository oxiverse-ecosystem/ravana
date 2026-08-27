import os
import re
import pickle
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from .models import CorrectionType
from .personal_fact_store import (
    PersonalFactStore, UserStanceStore, QuantityMemory, number_to_int)
from . import pet_slots as _pet_slots

# ── Dedicated user-model store ───────────────────────────────────────────────
# The per-user model used to be pickled *inside* the engine weight snapshot,
# which coupled it to the cognitive-graph lifecycle. It now lives in its own
# <repo>/user_models/ directory so user profiles are independent of (and
# outlive) any single weight checkpoint.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
USER_MODELS_DIR = os.path.join(_REPO_ROOT, "user_models")


# Correction detection patterns — ACC conflict detection (Error-Related Negativity)
_CORRECTION_DIRECT_PATTERNS = [
    r"\bno[!,.]", r"that'?s wrong", r"that'?s not right", r"that'?s incorrect",
    r"you'?re wrong", r"you are wrong", r"you'?re incorrect", r"you are incorrect",
    r"not correct", r"actually[!,]", r"not true", r"that'?s false",
    r"\bwrong[!.]", r"\bincorrect[!.]", r"\bmistake[!.]", r"\berror[!.]",
    r"no[!,]\s+that", r"no[!,]\s+it", r"wait[!,]\s+",
    r"that'?s not what", r"that is not what", r"that isn'?t what",
    r"hold on[!,]",
]

# Patterns that explicitly supply a corrected fact
_CORRECTION_FACT_PATTERNS = [
    # "it's X, not Y"
    r"it'?s\s+(\w+)[,.]*\s+not\s+(\w+)",
    r"it is\s+(\w+)[,.]*\s+not\s+(\w+)",
    # "X is Y, not Z"
    r"([\w\s]+?)\s+is\s+(\w+)[,.]*\s+not\s+(\w+)",
    r"([\w\s]+?)\s+are\s+(\w+)[,.]*\s+not\s+(\w+)",
]

# real affect categories in brain_regions._CAUSE_SEEDS and
# support_router._SUPPORT_AFFECT). Used by the bare-copula name guard: a
# first-person "i'm X" where X is any of these is a TRANSIENT STATE, never a
# proper noun, so it must not be stored as the user's NAME. This is SEED
# vocabulary (a data set, not an answer path) — RAVANA-expandable via the
# shared affect lexicon; removing entries degrades gracefully. Participles
# ("shaking"/"tired"), irregulars ("torn"/"lost"), and stative/cognitive
# verbs ("thinking"/"convinced") are all covered so the guard generalizes
# across every tense/participle form rather than a frozen per-word list.
_AFFECT_STATE_LEXICON = {
    # affect / emotion nouns + adjectives
    "happy", "sad", "glad", "mad", "angry", "furious", "scared", "afraid",
    "anxious", "worried", "worry", "lonely", "alone", "empty", "hollow",
    "numb", "lost", "torn", "hurt", "hopeless", "hopeful", "grateful",
    "excited", "tense", "raw", "low", "blue", "down", "proud", "calm",
    "nervous", "stressed", "stressedout", "overwhelmed", "depressed",
    "exhausted", "tired", "hungry", "thirsty", "sick", "confused",
    "fine", "good", "bad", "ok", "okay", "well", "ready", "done", "sure",
    "certain", "right", "wrong", "sorry", "here", "there", "home", "awake",
    "asleep", "late", "early", "busy",
    # Round 2026-08-14T0608Z: broaden the affect/state noun set so a bare
    # self-description ("i'm quiet", "i'm gutted", "i'm obsessed", "i'm
    # devastated") is NEVER stored as the user's NAME. These are genuine
    # affect/state words a real persona uses to describe a mood, not a proper
    # noun. Seed vocabulary (RAVANA-expandable: shares the role of the affect
    # lexicon the empathy gate uses; removing an entry degrades gracefully to
    # one less guard). Covers the words the chat probe actually poisoned plus
    # common synonyms so the next round's rotated probe can't re-expose them.
    "quiet", "gutted", "devastated", "obsessed", "content", "peaceful",
    "restless", "uneasy", "wound", "wounded", "broken", "crushed", "crush",
    "freaked", "spent", "drained", "fried", "wired", "zinged", "giddy",
    "bashful", "shy", "bold", "brave", "fearful", "moody", "snappy",
    "bitter", "sour", "warm", "cold", "soft", "hard", "still", "silent",
    "speechless", "numbed", "aching", "sore", "woozy", "faint", "weak",
    "strong", "alive", "dead", "deadened", "flat", "blank", "void",
    "comfortable", "uncomfortable", "safe", "unsafe", "free", "trapped",
    "stuck", "lost", "found", "clear", "cloudy", "sharp", "dull",
    "bright", "dim", "heavy", "light", "open", "closed", "honest",
    "dishonest", "real", "fake", "true", "false", "certain", "uncertain",
    "zen", "chill", "chilled", "mellow", "hyper", "wound", "upset",
    "gleeful", "cheerful", "mournful", "somber", "sober", "tipsy", "drunk",
    "soaked", "drenched", "freezing", "freezing", "boiling", "burning",
    "melting", "shaking", "trembling", "quivering", "shivering", "sweating",
    "ashamed", "guilty", "innocent", "proud", "humble", "vain", "jealous",
    "envious", "furious", "livid", "irritated", "annoyed", "bothered",
    # stative / cognitive / feeling verbs (incl. participles + infinitives)
    "feeling", "felt", "feel", "love", "like", "hate", "dislike", "prefer",
    "think", "thinking", "believing", "believe", "guess", "guessing",
    "wonder", "wondering", "mean", "meaning", "know", "knowing", "understand",
    "want", "wanting", "need", "needing", "wish", "wishing", "hope", "hoping",
    "doubt", "doubting", "fear", "fearing", "regret", "regretting",
    "suspect", "suspecting", "agree", "agreeing", "disagree", "convinced",
    "convincing", "standing", "coming", "going", "trying", "saying",
    "referring", "talking", "asking", "loving", "hating", "liking",
    "sticking", "getting", "shaking", "crying", "dying", "lying", "trying",
    "running", "falling", "breaking", "caring", "waiting", "working",
    "learning", "growing", "changing", "feeling",
    # Round 2026-08-13T2059Z: broaden to catch transient-state name
    # poisoning from emotional / participle words the prior set missed
    # (gutted / devastated / elated / shattered / drained / defeated /
    # wrecked / crushed / broken / frozen / shaken / heartbroken /
    # desolate / miserable / forlorn / wretched / bereft / gloomy / glum /
    # morose / melancholy / mournful / dismal / bitter / gleeful / rueful /
    # sorrowful / fretful / homesick / vexed / wistful / suspicious / wary /
    # jubilant / content / serene / tranquil / smug / sheepish / blithe /
    # coy / bashful / shy / sulky / querulous / uptight / vitriolic / void /
    # dubious / sullen / edgy / queasy / ticked / bereaved / anguished /
    # distraught / disheartened / dismayed). SEED vocabulary (RAVANA-
    # expandable), not an authored answer path — removing entries degrades
    # gracefully (one fewer guard), it is not content RAVANA can never change.
    "gutted", "devastated", "elated", "crushed", "heartbroken", "thrilled",
    "ecstatic", "miserable", "shattered", "wrecked", "defeated", "drained",
    "bitter", "gleeful", "rueful", "overjoyed", "despondent", "desolate",
    "forlorn", "wretched", "anguished", "bereft", "broken", "bereaved",
    "inconsolable", "dejected", "sorrowful", "gloomy", "glum", "morose",
    "melancholy", "mournful", "dismal", "crestfallen", "distraught",
    "dismayed", "disheartened", "fretful", "homesick", "horrified",
    "humiliated", "mortified", "terrified", "unnerved", "vexed", "wistful",
    "yearning", "aggrieved", "appalled", "bewildered", "chagrined",
    "daunted", "deflated", "delirious", "exasperated", "flabbergasted",
    "incensed", "indignant", "infuriated", "outraged", "peeved", "perturbed",
    "rattled", "resentful", "spooked", "startled", "stung", "sulky",
    "suspicious", "wary", "jubilant", "gratified", "invigorated", "content",
    "serene", "tranquil", "smug", "sheepish", "blithe", "coy", "bashful",
    "shy", "sullen", "querulous", "uptight", "vitriolic", "void", "dubious",
    "edgy", "queasy", "ticked", "frozen", "shaken",
}

# Consolidated, RUNTIME-EXTENSIBLE reject set for the bare-copula name guard
# ("i'm X" where X must NOT become the user's stored NAME). This is the single
# source of truth the guard consults; it merges the affect/state lexicon above
# with common self-descriptor adjectives and prepositions that introduce a
# PREDICATE, never a proper noun ("i'm against geoengineering").
#
# WHY A SEED SET (not a per-word answer path): it is DATA RAVANA GROWS at
# runtime. `register_name_reject()` is called by the empathy / support
# classifier whenever it observes "i'm <word>" classifying as affect — so the
# next "i'm <that word>" is rejected WITHOUT a code change. Removing an entry
# degrades gracefully (one less guard). This is the seed-vs-hardcoding test
# from the round brief satisfied: "can RAVANA change this by itself, through
# experience?" -> YES. (A frozen stoplist that only ever lists the exact probe
# words would be a fixed table wearing a seed's clothing — this set is the
# structural vocabulary, and the runtime path is what makes it genuinely
# growable rather than whack-a-mole.)
_NAME_REJECT_SEED = {
    # --- prepositions: "i'm against/for/with X" is a stance, not a name ---
    "against", "for", "with", "about", "over", "under", "because",
    "despite", "through", "without", "except", "besides", "unlike",
    # --- common self-descriptor adjectives (single-token predicates) ---
    "intense", "euphoric", "hooked", "careful", "stubborn", "loud",
    "brave", "calm", "shy", "bold", "proud", "humble", "vain",
    "jealous", "guilty", "innocent", "strong", "weak", "alive",
    "dead", "free", "trapped", "stuck", "clear", "cloudy", "sharp",
    "dull", "bright", "dim", "heavy", "light", "open", "closed",
    "honest", "dishonest", "real", "fake", "true", "false", "zen",
    "chill", "mellow", "hyper", "upset", "cheerful", "mournful",
    "sober", "freezing", "boiling", "burning", "melting", "soaked",
    "drenched", "drunk", "tipsy", "bashful", "bitter", "sour", "soft",
    "hard", "still", "silent", "speechless", "numbed", "aching",
    "sore", "woozy", "faint", "comfortable", "uncomfortable", "safe",
    "unsafe", "found", "void", "blank", "flat", "warm", "cold",
    "wound", "wounded", "broken", "gleeful", "somber", "restless",
    "uneasy", "peaceful", "content", "moody", "snappy", "giddy",
    "freaked", "spent", "drained", "fried", "wired", "zinged",
}
# Runtime-extensible half. The empathy/support classifier calls
# register_name_reject() when it sees "i'm <word>" as genuine affect, so the
# guard learns new predicates from conversation without a code deploy.
_NAME_REJECT_RUNTIME: set = set()

# ── Structural predicate-vs-name discriminator (round 2026-08-15T0326Z) ──
# The OLD guard (2026-08-14T1110Z) leaned on a FROZEN affect/state reject
# lexicon + a "runtime-extensible" half (register_name_reject). The runtime
# half was DEAD CODE — nothing ever called register_name_reject, so a ROTATED
# persona word that wasn't in the finite seed lexicon ("tender", in this
# round's ELIAS probe) slipped straight through and got stored as the user's
# NAME. A frozen lexicon is whack-a-mole: every new persona invents a new
# predicate word and the list can never be complete.
#
# Structural fix: a bare-copula candidate "i'm X" is a PREDICATE (never a
# name) whenever X has a lexical adjective sense in WordNet — because real
# English names ("elias", "nadia", "petra", "wren") are proper nouns with NO
# adjective sense, while every self-described state/predicate ("tender",
# "proud", "bored", "furious", "intense", "calm") DOES. This is a property of
# the language, not a curated word list, so it generalizes across EVERY future
# rotated persona without a code change. WordNet is OPTIONAL (declared in
# requirements.txt and cached under .venv-real/nltk_data); if it is missing we
# fail CLOSED to the narrow seed lexicon above and log once — we never paper
# over the import (per the project's silent-import-guard rule).
try:
    from nltk.corpus import wordnet as _WN
    _WN.synsets("tender", pos="a")  # touch to force corpus load error early
    _WN_AVAILABLE = True
except Exception as _wn_err:  # pragma: no cover - optional dependency
    _WN = None
    _WN_AVAILABLE = False
    import logging as _logging
    _logging.getLogger("ravana.chat.user_model").warning(
        "WordNet corpus unavailable (%s); name-poison guard falls back to the "
        "narrow seed affect lexicon. Run: python -m nltk.downloader wordnet",
        type(_wn_err).__name__)


def _is_predicate_word(word: str) -> bool:
    """Structural test: is `word` an English predicate (adjective/participle),
    rather than a proper-noun name? True iff WordNet has an adjective sense for
    it. Real names have no adjective sense, so they are NOT rejected."""
    w = (word or "").strip().lower().strip("'\"")
    if not w or " " in w or len(w) > 24:
        return False
    if _WN_AVAILABLE:
        try:
            return bool(_WN.synsets(w, pos="a"))
        except Exception:
            return False
    # Fail-closed fallback: a small structural heuristic for the no-WordNet
    # case — common adjective morphology (-ed/-ing participle or -y/-ful/-ive
    # suffix) is still a predicate signal even without the lexicon.
    return w.endswith(("ed", "ing", "ful", "ive", "ous", "y", "less", "ish"))


def register_name_reject(word: str) -> None:
    """Grow the bare-copula name reject set from observed affect words.

    Called by the empathy/support classifier (engine.py §3 affective-disclosure
    gate) whenever an "i'm X" utterance is classified as a genuine affect/state
    disclosure. This is how RAVANA extends the guard online (no retrain, no code
    change) — satisfying the round's seed-vs-hardcoding test. The word is also
    structurally re-confirmed as a predicate below, so we never add a real name
    token to the deny set.
    """
    w = (word or "").strip().lower().strip("'\"")
    if w and len(w) <= 24 and " " not in w and _is_predicate_word(w):
        _NAME_REJECT_RUNTIME.add(w)


def _name_rejectable(word: str) -> bool:
    """True if `word` is a predicate, never a proper-noun name, so reject it as
    a stored identity. Combines (a) the narrow seed affect lexicon (kept for
    graceful degradation when WordNet is absent) and (b) the STRUCTURAL
    WordNet adjective-sense test that generalizes to any rotated persona word."""
    w = (word or "").strip().lower().strip("'\"")
    if not w:
        return False
    return (w in _NAME_REJECT_SEED
            or w in _NAME_REJECT_RUNTIME
            or w in _AFFECT_STATE_LEXICON
            or w in _ACTIVITY_DENY
            or _is_predicate_word(w))


# Broad affect-term vocabulary used to NAME a felt state in the empathy
# responder (and to extract the user's own feeling word). This is SEED
# vocabulary (RAVANA-expandable, degrades gracefully): a word set describing
# human feeling states, NOT an authored reply path. It is intentionally broad
# so a ROTATED probe ("i felt terrified", "i'm grief-stricken", "i'm furious")
# is caught without enumerating every variant. Genuine affect naming — not a
# frozen per-topic table.
_AFFECT_TERM_LEXICON = frozenset({
    # fear / anxiety
    "terrified", "afraid", "scared", "scary", "frightened", "fearful",
    "anxious", "anxiety", "panicked", "panic", "worried", "nervous",
    "tense", "shaky", "alarmed", "uneasy", "restless",
    # grief / loss / sadness
    "grief", "grieving", "grief-stricken", "heartbroken", "devastated",
    "sad", "sadness", "blue", "down", "depressed", "hopeless", "mournful",
    "somber", "empty", "hollow", "lonely", "alone", "lost", "crushed",
    "broken", "hurting", "hurt", "numb", "void",
    # anger / agitation
    "furious", "fury", "angry", "anger", "irritated", "annoyed", "enraged",
    "livid", "mad", "bitter", "resentful", "upset",
    # shame / guilt
    "ashamed", "guilty", "embarrassed", "humiliated",
    # overwhelm / exhaustion
    "overwhelmed", "exhausted", "drained", "burned", "burnt", "spent",
    "fried", "stressed", "pressure", "wired",
    # positive
    "happy", "joy", "joyful", "delighted", "thrilled", "euphoric",
    "excited", "proud", "grateful", "relieved", "content", "peaceful",
    "calm", "glad", "cheerful", "hopeful", "gleeful",
})


def is_affect_term(word: str) -> bool:
    """True if `word` is a recognized human feeling word (used by the empathy
    responder to decide whether a copula-extracted word names a felt state)."""
    return (word or "").strip().lower().strip("'-") in _AFFECT_TERM_LEXICON


# Round 2026-08-14T0608Z: ACTIVITY / EVENT verb deny set. The open-class
# miner (and the seeded whitelist blocks) treat ANY word after "i" as the
# verb, so emotion verbs ("felt") and pure communication/reporting verbs
# ("said", "told") were captured as garbage 'does'/'event' facts
# ("felt crushed", "said careless ones"). This set is NARROW by design: it
# only contains verbs that are NEVER a real user activity or life event
# (emotion/cognition/volition are handled by the opinion/empathy paths;
# said/told are reporting utterances that echo verbatim). It deliberately does
# NOT deny legitimate activity verbs like keep/start/take/build — those are
# real things the user does, and denying them would also break the correction
# detector (which mines the 'does' fact to supersede a prior count). Framer /
# temporal words ("now", "just", "already", "take back") are handled at the
# regex / object level (see _FRAMER_SKIP, _FRAMER_OBJ, retraction guard), not
# here, so the real verb behind them is still captured. Seed vocabulary
# (RAVANA-expandable; removing an entry degrades gracefully). No per-verb
# answer table, no authored reply.
_ACTIVITY_DENY = frozenset({
    # emotion / cognition / volition (opinion + empathy paths handle these)
    "feel", "feels", "felt", "feeling",
    "love", "like", "hate", "dislike", "prefer",
    "think", "thinks", "thought", "believe", "believes", "believed",
    "know", "knows", "understand", "want", "wants", "need", "needs",
    "wish", "hope", "guess", "suppose", "mean", "means", "meant",
    "wonder", "agree", "disagree", "doubt", "fear", "fears",
    "regret", "regrets", "suspect", "realize", "realises", "care", "mind",
    # pure reporting / communication utterances (echo verbatim as garbage)
    "said", "say", "says", "told", "tell", "tells",
})

# Framer / temporal / degree words that may immediately precede the REAL
# activity verb ("i just started building", "i recently took up the cello").
# Added to the capture-regex skip groups so the genuine verb is matched, not
# the framer.
_FRAMER_SKIP = (
    "also|really|even|just|now|still|often|sometimes|usually|"
    "already|recently|lately|soon|first|last|then|next|once|twice|again|"
    "finally|today|tonight|yesterday|tomorrow|occasionally|rarely|"
    "simply|quite|very|truly|actually|basically|probably|possibly|maybe|"
    "certainly|definitely|rather|instead|"
)

# Words that may LEAK into the captured OBJECT as a trailing framer
# ("how many quail do i keep now" -> object "now"). Stripped from the resolved
# object head so 'does'/'event' facts store a real concept, never a framer.
_FRAMER_OBJ = frozenset({
    "now", "already", "still", "just", "recently", "lately", "soon",
    "today", "tonight", "yesterday", "tomorrow", "earlier", "later",
    "currently", "right", "then", "here",
})

# Aspectual / framer verbs that LEAD an activity but are not the activity head
# themselves ("i've BEEN building", "i STARTED keeping", "i got into"). When
# extracting the activity verb for date mining, skip these and take the lexical
# verb that follows. Defined once (round 2026-08-15T0326Z) so the date-miner
# blocks (a)/(b)/(c) stay in sync instead of each hardcoding its own skip list.
_ASPECTUAL_VERBS = frozenset({
    "been", "have", "has", "had", "start", "starts", "started", "begin",
    "begins", "began", "get", "gets", "got", "go", "goes", "went",
})

# Particles that turn a verb into a phrasal ("pick up", "took up", "got into").
# Appended to the captured head verb so "i picked up the cello" stores
# "pick up" (not just "pick"). Structural; generalizes across any phrasal verb.
_PARTICLES = frozenset({"up", "on", "in", "out", "off", "down", "with", "into"})

# Closed-class words that CLOSE the activity-object span (the object is the
# verb's patient; prepositions / time / clause words end it). SEED vocabulary
# (RAVANA-expandable): a word added here only changes where the object span
# ends, never the answer. Generalizes across phrasal/time adjuncts.
_OBJECT_STOP = frozenset({
    "since", "in", "on", "at", "for", "from", "to", "by", "of",
    "about", "around", "into", "during", "after", "before", "when", "while",
    "where", "because", "but", "and", "or", "so", "that", "which", "what",
    "who", "how", "why", "over", "near", "under", "with",
})
# Determiners / particles to SKIP (not close the span) before/within the
# object, so "building THE cabinets" -> "cabinets" and "build UP the frame" ->
# "frame". Also SEED vocabulary; expanding it only changes which leading
# function words are ignored.
_OBJECT_SKIP = frozenset({
    "the", "a", "an", "my", "your", "our", "their", "his", "her", "its",
    "this", "these", "those", "some", "every", "all", "each",
    "up", "out", "off", "down", "in", "on", "into", "back",
})


def _activity_object(clause_tokens, verb_idx) -> str:
    """Extract the activity object following the verb at verb_idx in a token
    list (the verb patient, e.g. (frames) in (building frames since 2019)).

    Structural, no per-topic table: skip leading determiners/particles
    (_OBJECT_SKIP), collect the run of content words, and stop at the first
    span-closing word in _OBJECT_STOP (prepositions / time / clause words like
    (since)/(for)/(when)). Bounded to 5 tokens so a runaway clause cannot
    swallow the whole sentence. The result is concatenated onto the mined
    since/since_age value so a later DATE-GROUNDED recall can DISAMBIGUATE two
    activities that share a verb head but differ by object (building frames vs
    building cabinets) - the object is exactly what the user query names.
    Returns empty string when there is no object (bare activity), so the
    existing value shape (build 2019) is preserved for verb-only disclosures.
    Fail-closed: out-of-range index returns empty.
    """
    if verb_idx < 0 or verb_idx + 1 >= len(clause_tokens):
        return ""
    _obj = []
    for _t in clause_tokens[verb_idx + 1:]:
        _tl = _t.lower()
        if _tl in _OBJECT_STOP:
            break
        if _tl in _OBJECT_SKIP:
            continue
        if not _tl or _tl.startswith("'") or _tl.isdigit():
            break
        _obj.append(_tl)
        if len(_obj) >= 5:
            break
    return " ".join(_obj)


def _activity_verb_ok(verb: str) -> bool:
    """True if `verb` is a legitimate activity/experience verb (not an
    emotion/achieve-comm verb). Used by all three capture blocks so
    'does'/'event' facts only store real activities RAVANA learned. Framer
    words are NOT denied here — they are skipped at the regex level so the
    real verb behind them is still captured."""
    v = (verb or "").strip().lower().lstrip("'").rstrip("'")
    if "'" in v:           # contraction artifact ("won't", "don't")
        return False
    if v.startswith("n't") or v == "not":
        return False
    return v not in _ACTIVITY_DENY


def _strip_obj_framers(obj: str) -> str:
    """Drop leading/trailing framer words so 'keep now' -> 'keep' and a real
    object survives. Returns '' if nothing real remains."""
    _toks = (obj or "").split()
    while _toks and _toks[0] in _FRAMER_OBJ:
        _toks.pop(0)
    while _toks and _toks[-1] in _FRAMER_OBJ:
        _toks.pop()
    return " ".join(_toks)




# D3 (round v3): explicit correction shape "X's name is not Y, it's Z" /
# "X is not Y, it's Z" where the CORRECTED value is the token after "it's"
# (never the negation word). Handled separately because its group order is
# (subject_attr, correct_value) which the generic 2-group branch would misread.
_CORRECTION_NAME_FACT_PATTERN = (
    r"(?:my\s+)?([\w'-]+?)(?:'s)?\s+(?:name\s+)?is\s+not\s+[\w'-]+"
    r"[,.]*\s+it'?s\s+([\w'-]+)"
)


# Stance-reversal cues (C4): universal retraction phrasings that recode the
# user's valuation of a topic to the opposite pole. These are STRUCTURAL
# language seeds, same status as the correction/opinion cue lists above — a
# small closed set of retraction verbs/idioms, NOT a per-topic table. Group
# placement matters: the topic is the clause AFTER the cue ("i take back what
# i said — plastic bans..."), so the match's .end() marks the topic span.
_RETRACTION_CUES = (
    # D4 fix (round v-aug06): allow an optional pronoun/adverb between "take"
    # and "back" — "i take it back", "i take this back", "i take that back" are
    # all the same retraction speech act. The prior pattern only allowed "that",
    # so "i take it back" (the most common spoken form) failed to match and the
    # contradiction was silently dropped, leaving the stale positive stance.
    r"\bi\s+take\s+(?:it|this|that|things|them|what\s+i\s+said)?\s*back",
    r"\bi\s+retract",
    r"\bi\s+(?:have|'ve)?\s*changed\s+my\s+mind",
    r"\bi\s+(?:am|'m)?\s*changing\s+my\s+mind",
    r"\bi\s+no\s+longer\s+(?:think|believe|feel|support|care)\b",
    r"\bi\s+was\s+wrong\s+(?:about|on|there)\b",
    # allow an adverb before "wrong" (e.g. "i was completely wrong about X")
    # or between "wrong" and the topic preposition. The common spoken form is
    # "<adverb> wrong about", which the original "wrong\s+(about...)" missed,
    # silently leaving the stale stance un-reversed.
    r"\bi\s+was\s+(?:\w+\s+)?wrong\s+(?:about|on|there)\b",
    r"\bi\s+was\s+wrong\s+\w+\s+(?:about|on|there)\b",
    r"\bi\s+(?:was|am)\s+too\s+hasty\b",
    r"\bi\s+(?:was|am|'m)\s+(?:mistaken|in\s+error)\b",
    r"\b(?:they're|it's|that's|wasn't|isn't|not)\s+(?:not\s+that\s+bad|not\s+so\s+bad|fine|okay|ok|acceptable|good\s+after\s+all)\b",
    r"\bi\s+take\s+(?:it|that|this|things|them|what\s+i\s+said|my\s+words)\s*back",
    r"\bscratch\s+that\b",
    # Round t_6c023144 (2026-08-09T1953Z residual): first-person reversal
    # speech acts that the round worker saw slip through to a fresh FOR stance.
    # "i flipped, the reef tank is more work than joy" formed a new positive
    # stance instead of recoding the held one, because "flipped" was absent
    # here — so no cue matched, the concession branch (requires a "but"/belief
    # frame) never fired, and mine_stance_reversal bailed before reversing. Add
    # the decisive change-of-mind verbs as SEED cues (RAVANA-expandable, not a
    # per-topic table). The tail→held-stance resolver already guards against
    # corruption: a flip on something the user has no stance on is a no-op
    # (reverse_stance returns None), so false positives are bounded.
    r"\bi\s*(?:'ve\s+|have\s+|)(?:flipped|flip-?flopped)\b",
    r"\bi\s+(?:recant|recanted|renounce|renounced|revoked|reversed|reneged)\b",
    r"\bi\s+(?:backtracked|went\s+back\s+on|backed\s+off\s+from)\b",
    # allow the contraction 'i've' (no space after i) as well as 'i had'/'i have'
    r"\bi\s*'?ve\s+had\s+a\s+change\s+of\s+heart\b",
    r"\bi\s+(?:had|have)\s+a\s+change\s+of\s+heart\b",
)

# Softening retraction cues: the user is walking a stance BACK toward neutral,
# not flipping to the opposite conviction ("olives? not that bad, i was too
# hasty"). These drive a PARTIAL reversal (relax toward neutral) instead of a
# full 180°. Kept as a separate tuple so mine_stance_reversal can test
# `cue is _SOFTENING_CUES` by object identity — it is a sub-set of
# _RETRACTION_CUES. No topic is named here, so this is seed structure, not an
# answer table; a hard recant lives in the rest of _RETRACTION_CUES.
_SOFTENING_CUES = (
    r"\bi\s+(?:was|am)\s+too\s+hasty\b",
    r"\bi\s+(?:was|am|'m)\s+(?:mistaken|in\s+error)\b",
    r"\b(?:they're|it's|that's|wasn't|isn't|not)\s+(?:not\s+that\s+bad|not\s+so\s+bad|fine|okay|ok|acceptable|good\s+after\s+all)\b",
)

# Conjoined multi-pet disclosure pattern: "i have a ferret named pim and a
# parrot called coco". One regex captures the whole chain; the miner expands it
# into per-animal "species named name" pairs (see possession-mining block below).
# Defined ONCE at module scope so the miner identifies it by object identity,
# not by a fragile `startswith` of its literal (a prior attempt compared a
# different literal and the branch never fired). Escapes are plain articles
# (a|an|the|some...), NOT \a/\t/\s — those are regex metachars that silently
# fail to match the spoken words.
_CONJOINED_PET_PAT = (
    r"\bi\s+have\s+(?:\d+\s+|(?:a|an|the|some|several|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"((?:(?:a|an|the|my|our|their|his|her)?\s*[\w\'-]+\s+(?:named|called)\s+[\w\'-]+"
    r"\s*(?:,?\s*(?:and|&|,)\s*(?:a|an|the)?\s*)?)+)"
)
@dataclass
class UserModel:
    edge_reactivations: Dict[Tuple[str, str], int] = field(default_factory=dict)
    query_concepts: Set[str] = field(default_factory=set)
    user_name: str = ""
    user_location: str = ""      # "I live in X" / "I am from X"
    user_background: str = ""     # free biographical note (e.g. "born in Paris")
    preferences: Dict[str, Any] = field(default_factory=dict)
    # Learned personal-fact store (brain-faithful: confidence x recency x decay,
    # improves over time via confirm/contradict). Seeded from the high-precision
    # regex below AND from any _mine_episodic_facts hit, so "my cat is Pixel"
    # becomes a gradeable, correctable fact rather than a frozen field.
    personal_facts: PersonalFactStore = field(default_factory=PersonalFactStore)
    # Learned opinion store (C): the user's value judgments (for/against a
    # topic), kept SEPARATE from biographical facts. Opinions decay faster than
    # facts (malleable attitudes), per OFC/vmPFC vs hippocampal circuit split.
    opinions: UserStanceStore = field(default_factory=UserStanceStore)
    # Structured quantity memory (round 2026-08-11T0521Z): count-bearing
    # disclosures ("i keep twelve racing pigeons") are captured as
    # (subject, kind, count, noun) so RAVANA can SYNTHESIZE a clean count answer
    # and AGGREGATE across the store ("how many pets in total"). The 'does'/
    # 'event' text facts still hold the gist sentence; this store holds the
    # NUMBER, decoupling it from the gist. Seed + online; durable via
    # get/set_state so it survives engine reload.
    quantity_memory: QuantityMemory = field(default_factory=QuantityMemory)

    knowledge_model: Dict[str, float] = field(default_factory=dict)
    learning_goals: Dict[str, int] = field(default_factory=dict)
    emotional_rapport: Dict[str, float] = field(default_factory=dict)
    cognitive_style: str = "balanced"
    engagement_level: float = 0.5
    conversation_depth: float = 0.0
    interaction_count: int = 0
    relationship_depth: float = 0.0
    goals: List[str] = field(default_factory=list)
    last_goal: str = "EXPLORING"

    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        'valence': 0.0, 'arousal': 0.3, 'dominance': 0.5,
    })
    belief_state: Dict[str, Dict] = field(default_factory=dict)
    interaction_history: List[Dict] = field(default_factory=list)
    _emotion_detector: Any = None

    topic_interaction_count: Dict[str, int] = field(default_factory=dict)
    topic_followup_count: Dict[str, int] = field(default_factory=dict)
    last_topic: str = ""
    turn_since_topic_change: int = 0

    # ── Phase: Correction pattern tracking (ACC/ERN analog) ──
    detected_correction: bool = False
    detected_correction_type: Optional[CorrectionType] = None
    correction_severity: float = 0.0
    correction_subject: str = ""
    detected_correction_fact: Optional[Tuple[str, str, str]] = None
    _last_user_valence_before_response: float = 0.0
    _last_response_for_correction: str = ""
    _last_response_strategy_for_correction: str = ""
    _previous_user_query: str = ""

    def observe_chain(self, hops: List[Tuple[str, str]], is_user_query: bool = False):
        for from_label, to_label in hops:
            key = (from_label.lower(), to_label.lower())
            self.edge_reactivations[key] = self.edge_reactivations.get(key, 0) + 1
        if is_user_query:
            for from_label, to_label in hops:
                self.query_concepts.add(from_label.lower())
                self.query_concepts.add(to_label.lower())
                self.knowledge_model[from_label.lower()] = min(1.0, self.knowledge_model.get(from_label.lower(), 0.0) + 0.1)
                self.learning_goals[to_label.lower()] = self.learning_goals.get(to_label.lower(), 0) + 1

    def mine_personal_facts(self, text: str,
                             run_correction: bool = True) -> None:
        """High-precision personal-fact miner (B3 / A5). Seeded from explicit
        "my X is Y" / "I have a X named Y" / name / location patterns into the
        learned PersonalFactStore. Called both from observe_user_query (full
        ToM pass) and directly by the identity gate for same-turn capture, so
        it must take ONLY raw text (no subject / valence dependency).

        Facts seeded here are gradeable + correctable; they are NOT frozen
        regex buckets — the store learns the rest from confirm/contradict.
        """
        q_clean = re.sub(r"\s+", " ", text).strip()
        # Correction cue (B4 wiring, investigation Gap 1): when the user is
        # correcting us ("no, my cat is milo", "actually i live in paris"),
        # a mined fact whose attribute already holds a DIFFERENT active value
        # must supersede it via contradict() — the user is ground truth for
        # their own profile. Without this the correction loop is dead: the
        # store has contradict() but nothing ever called it.
        _corrective = bool(re.search(
            r"^\s*no\b|\bactually\b|\bthat'?s\s+(?:wrong|not\s+right|incorrect)\b"
            r"|\bi\s+(?:said|told\s+you)\b|\bnot\s+[\w'-]+\s*,?\s*(?:it'?s|it\s+is)\b",
            q_clean, re.IGNORECASE))


        _NEG_WORDS = {"not", "no", "never", "none", "nil", "n't", "dont",
                       "don't", "false", "wrong", "incorrect"}
        _VALUE_STOP = _NEG_WORDS | {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "am", "be", "been", "being", "to", "of", "in", "on", "at", "for",
            "with", "my", "your", "i", "you", "it", "this", "that", "me"}

        # Relation words for the HEADLESS possessive splitter (round
        # 2026-08-15T0326Z). The SAME vocabulary _structured_recall._ENT_ATTR
        # keys on, so the miner and the recaller stay in sync by construction.
        # A multi-word attr whose final token is one of these resolves to
        # (entity=<head>, relation=<word>); anything else stays a genuine
        # self-fact (e.g. "favorite color" -> (None, attr)).
        _REL_WORDS = {
            "name", "age", "breed", "job", "work", "is", "does", "location",
            "live", "color", "type", "kind", "favorite", "gender", "role",
        }
        # A leading qualifier that marks a SELF-FACT attribute (not an entity).
        # "my favorite color is ochre" must stay subject 'i' (attr 'favorite
        # color'), not be split into entity 'favorite' / relation 'color' — the
        # head is a quantifier, not a relation the user named. Guarded in the
        # splitter so genuine self-facts survive.
        _SELF_QUALIFIER = {
            "favorite", "favourite", "least", "most", "best", "worst",
        }

        # D3 (round v4): name-correction detection MUST run here, not only in
        # UserModel._extract_correction_fact (which observe_user_query calls at
        # engine.py:4193 — AFTER the self-disclosure early-return at :3392). A
        # correction phrased as a disclosure ("my sister's name is not meena,
        # it's priya") returns at :3392 before late detection ever sets the
        # flag, so the contradict() block at :3408 saw None and the correction
        # was silently lost. mine_personal_facts runs at :2549 (pre-early-return),
        # so detecting here lets the engine persist the corrected fact online.
        # Seeds the same flags the full circuit uses; the :3408 block does the
        # actual contradict() (no retrain, no authored text). _VALUE_STOP is
        # defined above, so the negation guard can reject junk values.
        _nm = re.search(_CORRECTION_NAME_FACT_PATTERN, q_clean, re.IGNORECASE)
        if _nm:
            _subj_attr = _nm.group(1).strip().removesuffix("'s")
            _correct_val = _nm.group(2).strip()
            if _correct_val and _correct_val not in _VALUE_STOP:
                self.detected_correction = True
                self.detected_correction_fact = ("i", _subj_attr, _correct_val)
                self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                self.correction_severity = max(self.correction_severity, 0.8)

        # PET re-disclosure correction (round 2026-08-10T0813Z). A pet name
        # stated in a SECOND phrasing ("the dog is a lurcher called briar"
        # after "my dog is a lurcher named wren") is a CORRECTION of the
        # earlier name, not a fresh disclosure — but the possession miner only
        # matches "my/a/an/the <species> named/called <name>" (one shape), so
        # "the dog is a lurcher called briar" (species + copula + breed +
        # called + name) fell through and the stale name persisted, then a
        # later "what's my dog's name" returned the OLD name. Detect the
        # copula+breed+name shape, resolve the species via pet_slots (the same
        # resolver the miner and recall sites use), and if that slot already
        # holds a DIFFERENT active value, flag a correction so the existing
        # contradict() machinery supersedes it (online, no retrain). When the
        # slot is empty the normal possession miner stores it. No per-topic
        # table — species comes from the live pet_slots vocabulary.
        # Guard on detected_correction_fact (not the bare flag): _detect_correction
        # may set detected_correction=True via a weak signal (sentiment-drop /
        # reask) WITHOUT a fact, which would otherwise suppress this pet fact
        # and leave flag=True/fact=None (the T46 chat regression). Only skip
        # when a real correction FACT is already extracted.
        if not self.detected_correction_fact:
            _pet_corr = re.search(
                r"\b(?:my|the|a|an)\s+([\w'-]+)\s+(?:is|was|are|were)\s+"
                r"(?:a\s+|an\s+)?(?:[\w'-]+\s+){0,2}?"
                r"(?:named|called)\s+([\w'-]+)", q_clean, re.IGNORECASE)
            if _pet_corr:
                _sp_word = _pet_corr.group(1).strip().lower()
                _new_name = _pet_corr.group(2).strip().strip(".,!?")
                _species = _pet_slots.species_of(_sp_word)
                if _species is not None and _new_name:
                    _slot = _pet_slots.slot_for(_species, 1)
                    _prior = self.personal_facts.get("i", _slot)
                    if _prior is not None and _prior.value.lower() != _new_name.lower():
                        self.detected_correction = True
                        self.detected_correction_fact = ("i", _slot, _new_name)
                        self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                        self.correction_severity = max(self.correction_severity, 0.8)


        # REVERSE-ORDER POSSESSION NAMING + OWNER RE-ATTRIBUTION miner (round
        # 2026-08-10T1401Z feature). Limitation #1 flagged in the round report:
        # the possession miner only captures "my <species> is/are named/called
        # <name>" and "i have a <species> named <name>" — FORWARD order with the
        # owner first. But a user re-discloses pets the OTHER way round, which
        # the miner never modelled:
        #   * reverse-order naming: "the barn owl is mine and she's called wren"
        #     (THE <species> IS MINE [and (he|she|it)'s called|named <name>]) —
        #     the owl was previously stored under the USER (subject "i",
        #     species slot "owl") on the first disclosure, so this second
        #     phrasing must FILE THE NAME on that same slot, not drop it.
        #   * owner re-attribution: "pip is my sister's cat" /
        #     "the barn owl is mine" — re-assigns an owned entity to a DIFFERENT
        #     owner. The first disclosure stored pip under subject "i"; this
        #     must MOVE it to subject "sister" so a later "what's my cat's name"
        #     no longer returns pip (self/other boundary).
        # Both write through the SAME pet_slots resolver (species_of / learn_species
        # / slot_for) the forward miner and the recall sites use, so the key
        # agrees by construction. No per-topic table; species + owner come from
        # the live store and the live pet_slots vocabulary (RAVANA-expandable).
        # Online, no retrain: each turn mines from raw text only.
        _POSSESS_RE = re.compile(
            r"\b(the\s+)?(?P<sp>[\w'-]+)\s+(?:is|was|are|were)\s+"
            r"(?P<mine>(?:mine|my\s+own|ours))"
            r"(?:\s+(?:and|but|,)?\s*(?:he|she|it|they)'s\s+"
            r"(?:called|named)\s+(?P<nm>[\w'-]+))?",
            re.IGNORECASE)
        for _pr in _POSSESS_RE.finditer(q_clean):
            _sp_word = _pr.group("sp").strip().lower()
            if _sp_word in ("the",):
                continue
            _nm = (_pr.group("nm") or "").strip().strip(".,!?")
            _species = _pet_slots.species_of(_sp_word)
            if _species is None:
                # "the owl is mine and she's called wren" for a species the
                # forward miner never got a chance to learn (e.g. the prior
                # disclosure was a bare possession like "i keep an owl in the
                # loft", which mints no pet slot). The attached name is
                # itself evidence of a real pet, so learn the species here
                # too — same gate (a name/"called"/"named" is present) the
                # forward miner uses when it learns an unknown species.
                if _nm and _sp_word.isalpha():
                    _species = _pet_slots.learn_species(_sp_word)
                if _species is None:
                    continue
            _slot = _pet_slots.slot_for(_species, 1)
            _mine = _pr.group("mine")
            # Locate any prior slot for this species under the USER — the
            # first disclosure stored it there (subject "i").
            _prior_user = self.personal_facts.get("i", _slot)
            # Ownership claim "is mine" for an entity already owned: file the
            # name on the existing user slot if one wasn't given before.
            if _nm and _prior_user is not None:
                if _prior_user.value.lower() != _nm.lower():
                    self.personal_facts.contradict("i", _slot, _nm)
                else:
                    self.personal_facts.reinforce("i", _slot, _nm)
            elif _nm:
                # "the barn owl is mine and she's called wren" — first time we
                # see the name for this species under the user; store it.
                self.personal_facts.assert_fact(
                    "i", _slot, _nm, confidence=0.6,
                    source="seed_regex")
            # Owner re-attribution: "the barn owl is mine" for an entity that
            # was previously stored under a DIFFERENT owner must RE-ASSIGN it to
            # the user. Detect the cross-owner move by scanning for an active
            # prior slot for this species under any non-user subject.
            if _prior_user is None and _mine:
                for (s, a, v), f in self.personal_facts.facts.items():
                    if (s != "i" and a == _slot and not f.superseded):
                        # move to the user: retire the old owner's record and
                        # re-assert under "i" (preserve the value/name).
                        f.superseded = True
                        self.personal_facts.assert_fact(
                            "i", _slot, v, confidence=0.6,
                            source="seed_regex")
                        break
        # R8 fix (round 2026-08-11T0521Z): REVERSE-ORDER ownership claim /
        # correction "<name>'s my <species>" (e.g. "salt's my dog actually")
        # was NOT captured by the forward "<species> is mine" miner
        # (_POSSESS_RE) nor the owner re-attribution miner (_OWNER_RE, which
        # only moves an entity OFF the user). So "salt's my dog" — a user
        # re-claiming a pet a neighbour had disclosed — was silently dropped,
        # and recall still returned the neighbour's record. This block handles
        # the subject-first form: extract name + species, and (a) if a prior
        # active record for this species exists under a NON-user subject,
        # re-attribute it to the USER (supersede the old owner's record, assert
        # under "i" with the clean name); (b) otherwise assert / file the name
        # on the user's species slot (same resolver the forward miner uses, so
        # the key agrees by construction). Only KNOWN species match
        # (species_of without learn_species) so a stray "<name>'s my <relation>"
        # (e.g. "john's my friend") can never learn "friend" as a pet species.
        _NAME_MINE_RE = re.compile(
            r"\b(?P<nm>[A-Za-z][\w'-]*)\s*'s\s+my\s+(?P<sp>[\w'-]+)\b",
            re.IGNORECASE)
        for _nr in _NAME_MINE_RE.finditer(q_clean):
            _nm = _nr.group("nm").strip().strip(".,!?").lower()
            _sp_word = _nr.group("sp").strip().lower()
            if not _nm or _sp_word in ("the", "a", "an"):
                continue
            # Do NOT treat an interrogative word ("what", "who", "which", ...)
            # as a pet name. A recall query like "what's my dog's name" matches
            # this pattern (nm="what", sp="dog") and would otherwise overwrite
            # the stored name with the question word. Genuine names are never
            # wh-words, so this guard is safe.
            if _nm in ("what", "who", "which", "where", "when", "why",
                       "how", "whose", "whom"):
                continue
            _species = _pet_slots.species_of(_sp_word)
            if _species is None:
                # Only react to KNOWN species (no learn_species here) to avoid
                # learning a relation noun ("friend") as a pet species.
                continue
            _slot = _pet_slots.slot_for(_species, 1)
            # (a) re-attribute from a prior non-user owner to the user.
            _moved = False
            for (s, a, v), f in self.personal_facts.facts.items():
                if (s != "i" and a == _slot
                        and not getattr(f, "superseded", False)):
                    f.superseded = True
                    self.personal_facts.assert_fact(
                        "i", _slot, _nm, confidence=0.6,
                        source="seed_regex")
                    _moved = True
                    break
            if _moved:
                continue
            # (b) file / reinforce the name on the user's own species slot.
            _prior_user = self.personal_facts.get("i", _slot)
            if _prior_user is not None:
                if _prior_user.value.lower() != _nm.lower():
                    self.personal_facts.contradict("i", _slot, _nm)
                else:
                    self.personal_facts.reinforce("i", _slot, _nm)
            else:
                self.personal_facts.assert_fact(
                    "i", _slot, _nm, confidence=0.6,
                    source="seed_regex")
        # OWNER-AS-POSSESSOR re-attribution: "<name> is my <owner>'s <species>"
        # e.g. "pip is my sister's cat" / "wren is my mum's owl". Re-assigns an
        # owned entity from the USER to a NAMED third-party owner. The first
        # disclosure stored pip under subject "i"; this must MOVE it to subject
        # <owner> so recall keyed by the user no longer returns it (self/other
        # boundary, same reasoning as the D6 possessive-split fix). Generic:
        # any "<name> is my <owner>'s <species>" resolves owner + species from
        # the live pet_slots vocabulary; the user can name any relation.
        _OWNER_RE = re.compile(
            r"\b(?P<nm>[\w'-]+)\s+is\s+my\s+(?P<own>[\w'-]+)'s\s+"
            r"(?P<sp>[\w'-]+)\b", re.IGNORECASE)
        for _or in _OWNER_RE.finditer(q_clean):
            _nm = _or.group("nm").strip().strip(".,!?").lower()
            _owner = _or.group("own").strip().lower()
            _sp_word = _or.group("sp").strip().lower()
            if _owner in ("the", "a", "an") or not _nm:
                continue
            # Owner re-attribution ONLY fires when the user already owns a pet
            # NAMED <nm>. This anchors the species to the entity the user
            # actually disclosed ("pip is my sister's cat" -> pip was the
            # user's cat), so a stray "<name> is my <owner>'s name" (a person
            # naming themselves after a pet) can never learn "name" as a
            # species or move an unrelated fact. We resolve the species from
            # the LIVE user-owned slot whose value contains _nm — never from
            # the trailing word, which may be a relation noun ("name"), and we
            # never call learn_species here. If the user owns no pet named _nm,
            # this is not a possession re-attribution; skip it.
            _prior_user = None
            _slot = None
            for (s, a, v), f in self.personal_facts.facts.items():
                if (s == "i" and not f.superseded
                        and _pet_slots.is_pet_attribute(a)
                        and _nm in re.sub(r"^(?:called|named)\s+", "",
                                          v.lower()).split()):
                    _prior_user = f
                    _slot = a
                    break
            if _prior_user is None or _slot is None:
                continue
            # Owner re-attribution: the entity leaves the user and moves to a
            # named third-party owner. The name comes from THIS phrase (_nm),
            # which is the clean name — the forward miner may have stored a
            # "called <name>" value, so we prefer the explicit name here and
            # never echo the "called " scaffolding into the owner's record.
            # Retire EVERY active user record for this species slot
            # (value-agnostic): contradict() would skip superseding when the
            # new value equals the old, which would leave the user's stale
            # record active and a later "what's my <species>'s name" would
            # still surface it. Self/other boundary must be enforced.
            for (s, a, v), f in self.personal_facts.facts.items():
                if s == "i" and a == _slot and not f.superseded:
                    f.superseded = True
            # The hippocampal episodic index is keyed by ENTITY (not subject),
            # so moving pip off the user must ALSO clear the user-facing entry
            # for that entity — otherwise a later "what's my cat's name" reads
            # episodic_index["cat"] and echoes the stale name. This is the same
            # self/other boundary as the fact-store supersede above, applied to
            # the recall source the engine actually reads.
            _ent_word = _pet_slots.base_species(_slot)
            _epi_index = getattr(self, "_episodic_index", None)
            if _epi_index is not None and _ent_word in _epi_index:
                _epi_index.pop(_ent_word, None)
            # The RAW episodic transcript is a SECOND recall source: recall
            # merges each stored turn's `facts` dict into the entity index
            # (engine_memory.py _retrieve_episodic). Turn 1 stored
            # {"cat": "called pip"} in that turn-record's `facts`, so even after
            # we pop episodic_index["cat"], the transcript still replays the
            # stale name for the user's cat. The user just re-attributed this
            # entity to a THIRD PARTY — redact that entity from every stored
            # turn's `facts` so the self/other boundary holds at BOTH recall
            # sources. Only the entity keyed by this species leaves the user;
            # other stored facts on the same turn are untouched.
            _epi_tr = getattr(self, "_episodic_transcript", None)
            if _epi_tr is not None:
                for _rec in _epi_tr:
                    _rfacts = _rec.get("facts")
                    if not isinstance(_rfacts, dict):
                        continue
                    for _k in list(_rfacts.keys()):
                        _ent_tok = _k.split("_", 1)[0] if "_" in _k else _k
                        if _pet_slots.species_of(_ent_tok) == _ent_word:
                            _rfacts.pop(_k, None)
            # The HIPPOCAMPAL BUFFER is a THIRD recall source: multi-hop
            # relational recall (_try_multi_hop -> _hop_retrieve) reads raw
            # stored utterances from it via buf.retrieve(<entity>). Turn 1 stored
            # ("cat","is_about","my cat is called pip") there, and the multi-hop
            # reasoner returns that raw utterance verbatim as the answer to
            # "what is my cat's name?". The user just re-attributed the cat to a
            # third party, so the user-facing buffer entry for that species must
            # be RETIRED from every key list it lives under — otherwise the
            # self/other boundary leaks at the multi-hop path even though the
            # fact-store + episodic sources are clean. buf.retrieve does NOT
            # honor the `superseded` flag, so we remove the objects directly from
            # each key-list (the same FactTriple is indexed under the species
            # subject AND alias keys, so we must purge it everywhere).
            _hbuf = getattr(self, "_hippocampal_buffer", None)
            if _hbuf is not None:
                _hb_facts = getattr(_hbuf, "facts", None)
                if _hb_facts is not None:
                    # FactTriple is an unhashable dataclass, so collect the
                    # objects to drop via id() rather than a set literal.
                    _drop_ids = {id(_f) for _lst in _hb_facts.values()
                                 for _f in _lst
                                 if getattr(_f, "subject", None) == _ent_word}
                    if _drop_ids:
                        for _k, _lst in _hb_facts.items():
                            _hb_facts[_k] = [_f for _f in _lst
                                            if id(_f) not in _drop_ids]
                        _all = getattr(_hbuf, "_all_facts", None)
                        if _all is not None:
                            _hbuf._all_facts = [_f for _f in _all
                                                if id(_f) not in _drop_ids]
            self.personal_facts.assert_fact(
                _owner, _slot, _nm, confidence=0.6,
                source="seed_regex")

        # COUNT / QUANTITY correction (round 2026-08-09g). A plain update like
        # "it's seven hives now, i split one last week" carries NO negation or
        # "my X is Y" structure, so the name-correction and _corrective paths
        # above miss it — the prior count fact ("keep six hives", stored under
        # the 'does' attribute) was left active and a later "how many hives do
        # i have" returned the STALE six (measured: T41 -> "ok, noted: wait.",
        # T42/T64 silently kept six). Detect an update cue + a cardinal number +
        # an entity noun, locate the prior count/activity fact for that entity,
        # and supersede it via contradict() (online, no retrain). Content comes
        # from the live store; no per-topic table, no authored text.
        if not self.detected_correction:
            _NUMWORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|"
                         r"ten|eleven|twelve|\d+)")
            _cnt = re.search(
                r"\b(?P<num>" + _NUMWORDS + r")\s+(?P<ent>[a-z][a-z]+)\b.*\b"
                r"(now|split|added|new|more|extra|another|gained|got|"
                r"increased|up to)\b", q_clean, re.IGNORECASE) or \
                re.search(
                r"\b(now|split|added|new|more|extra|another|gained|got)\b.*\b"
                + r"(?P<num>" + _NUMWORDS + r")\s+(?P<ent>[a-z][a-z]+)\b",
                q_clean, re.IGNORECASE)
            if _cnt:
                _num = _cnt.group("num")
                _ent = _cnt.group("ent").lower().strip()
                if _ent and _ent not in _VALUE_STOP and _num:
                    # find the prior activity/count fact whose value mentions
                    # this entity (e.g. "keep six hives" -> entity "hives").
                    _prior = None
                    for (s, a, v), f in self.personal_facts.facts.items():
                        if s == "i" and a in ("does", "count", "number", "qty") \
                                and not getattr(f, "superseded", False) \
                                and _ent in v.lower():
                            _prior = (a, v)
                            break
                    if _prior is not None:
                        # rebuild the new value with the corrected count,
                        # preserving the verb + entity from the prior fact.
                        _verb = re.match(
                            r"^(keep|have|keep on|have on|raise|own|breed|run|"
                            r"got)\b", _prior[1].lower())
                        _newval = (f"{_verb.group(1)} " if _verb else "") \
                            + f"{_num} {_ent}"
                        self.detected_correction = True
                        self.detected_correction_fact = (
                            "i", _prior[0], _newval)
                        self.detected_correction_type = \
                            CorrectionType.CORRECTION_WITH_FACT
                        self.correction_severity = \
                            max(self.correction_severity, 0.7)
                        # Mirror the corrected count into the structured
                        # QuantityMemory store so "how many X" recall reflects
                        # the NEW number (otherwise it would echo the stale
                        # pre-correction count). Seed + online; the store
                        # supersedes the prior record by subject+noun.
                        try:
                            _qcount = number_to_int(_num)
                            if _qcount is not None:
                                self.quantity_memory.correct(
                                    subject="i", noun=_ent, count=_qcount)
                        except Exception:
                            pass

        def _put_fact(attr: str, val: str, conf: float) -> None:
            # D3 (round v3): never store a closed-class / negation token as a
            # fact value. The old miner matched "my sister's name is not meena,
            # it's priya" and stored value="not" (the word after "is"). Values
            # that are not real content are rejected so they can't pollute the
            # personal-fact store; the real corrected value arrives via the
            # correction circuit (detected_correction_fact) instead.
            _val = (val or "").strip().strip(" .,!?;:'\"").lower()
            if not _val or _val in _VALUE_STOP:
                return
            # C-value (round 2026-08-08c): trim a TRAILING prepositional phrase
            # from the value. The greedy value capture ("my X is Y", up to 8
            # words) over-grabbed trailing prepositions, storing nonsense like
            # "cabin at." / "always raw from hauling hive boxes up the.". A
            # value that ENDS in a preposition ("up"/"from"/"at"/"in") is an
            # incomplete capture — drop the preposition and everything after it
            # so "cabin at." -> "cabin" and "up the mountain" -> "up" -> "".
            # Structural: one shared chokepoint for every fact; the
            # preposition set is closed-class (not content), so trimming it
            # never discards a real value word. If the whole value is
            # prepositions, reject it (no content to store).
            _PREP = ("up", "down", "from", "at", "in", "on", "with", "to",
                     "of", "by", "for", "about", "into", "onto", "over",
                     "under", "near", "behind", "beside", "off")
            _vwords = _val.split()
            if _vwords and _vwords[-1] in _PREP:
                # drop trailing prepositions and any words following the first
                # trailing preposition
                _cut = len(_vwords)
                for _i in range(len(_vwords) - 1, -1, -1):
                    if _vwords[_i] in _PREP:
                        _cut = _i
                    else:
                        break
                _vwords = _vwords[:_cut]
                _val = " ".join(_vwords)
            if not _val or _val in _VALUE_STOP:
                return
            existing = self.personal_facts.get("i", attr)
            if (_corrective and existing is not None
                    and existing.value.lower() != _val):
                self.personal_facts.contradict("i", attr, _val)
            else:
                self.personal_facts.assert_fact("i", attr, _val,
                                                confidence=conf,
                                                source="seed_regex")

        def _split_possessive_attr(attr: str):
            """D6 (round 2026-08-08b-d): 'my partner's name is theo' must model
            an ENTITY (partner) and its attribute (name), not collapse onto the
            user's own self-profile. The multi-word attr pattern
            (r'\bmy\s+(...)\s+is\s+...') captures 'partner's name' as one attr
            key under subject 'i'; a later recall path sees the substring
            'name' and renders 'your name is theo' — reporting the PARTNER'S
            name as the USER's name (a self/other boundary breach; the same
            defect class that put 'your name is a hypocrite' in v-aug04).

            Fix: detect a possessive head ('s) in the attr and resolve it to
            (entity=<owner>, attr=<relation>), mirroring the ALREADY-CORRECT
            possessive handling in engine_memory._record_episode (which keys
            'my cat's name is whiskers' -> entity=cat, attr=name). This is
            structural: any 'my <X>'s <Y> is Z' is stored under entity X, never
            under the user's 'i' subject. Generic — no per-entity table. The
            entity grows from experience (the user can name any relation).

            Round 2026-08-15T0326Z GENERALIZE: also handle the HEADLESS
            possessive 'my <entity> <relation> is Z' (no apostrophe), e.g.
            'my daughter name is petra' -> attr 'daughter name'. The old code
            only matched the '<entity>'s <relation>' form, so 'daughter name'
            fell through to subject 'i' (attr 'daughter name') and a later
            'what's my daughter's name' echoed the USER's name instead. Now any
            attr whose LAST token is a relation word resolves to
            (entity=<head>, relation=<word>), so the miner agrees with the
            recaller (_structured_recall._ENT_ATTR) by construction for BOTH
            possessive shapes. Still returns (None, attr) for genuine self-facts
            like 'favorite color' (neither head nor tail is a relation word)."""
            _am = re.match(r"^([\w'-]+)'s\s+(.+)$", attr)
            if _am:
                return _am.group(1).strip().lower(), _am.group(2).strip().lower()
            # Headless possessive: 'my daughter name is petra' -> attr
            # 'daughter name'. The trailing token is a relation word; the head
            # is the entity. Reused _REL_WORDS (the same relation vocabulary the
            # recaller keys on) so miner and recaller stay in sync.
            # GUARD: a head that is a SELF-FACT QUALIFIER ('favorite', 'least',
            # 'most', ...) is NOT an entity — 'my favorite color is ochre'
            # stays a genuine self-fact (attr 'favorite color', subject 'i'),
            # not an entity 'favorite' with relation 'color'. Without this guard
            # the tail-relation test wrongly entity-scoped 'favorite color' and
            # broke its recall (round 2026-08-15T0326Z regression check).
            # Only a multi-word attr is split; a single-token attr ('cat',
            # 'daughter', 'name') is left as-is so pet/self-fact mining keeps
            # its existing subject='i' key (the single-token branch was removed:
            # it entity-scoped 'my cat is pixel' as ('cat','is','pixel') and
            # broke pet mining).
            _toks = attr.split()
            if len(_toks) >= 2 and _toks[-1] in _REL_WORDS \
                    and _toks[0] not in _SELF_QUALIFIER:
                return " ".join(_toks[:-1]).strip(), _toks[-1]
            return None, attr

        def _put_fact_ent(entity: str, attr: str, val: str, conf: float) -> None:
            """Entity-keyed variant of _put_fact (subject = entity, not 'i')."""
            _val = (val or "").strip().strip(" .,!?;:'\"").lower()
            if not _val or _val in _VALUE_STOP:
                return
            _PREP = ("up", "down", "from", "at", "in", "on", "with", "to",
                     "of", "by", "for", "about", "into", "onto", "over",
                     "under", "near", "behind", "beside", "off")
            _vwords = _val.split()
            if _vwords and _vwords[-1] in _PREP:
                _cut = len(_vwords)
                for _i in range(len(_vwords) - 1, -1, -1):
                    if _vwords[_i] in _PREP:
                        _cut = _i
                    else:
                        break
                _vwords = _vwords[:_cut]
                _val = " ".join(_vwords)
            if not _val or _val in _VALUE_STOP:
                return
            _subj = entity.lower()
            existing = self.personal_facts.get(_subj, attr)
            if (_corrective and existing is not None
                    and existing.value.lower() != _val):
                self.personal_facts.contradict(_subj, attr, _val)
            else:
                self.personal_facts.assert_fact(_subj, attr, _val,
                                                confidence=conf,
                                                source="seed_regex")

        m_name = re.search(
            r"\b(?:my\s+name\s+is|i\s+am\s+called|call\s+me)\s+"
            r"([^.,!?]+)",
            q_clean, re.IGNORECASE)
        if not m_name:
            # Bare first-person self-naming ("i'm noor", "i am noor"): the
            # contraction "i'm" / "i am" introduces a proper noun that is the
            # speaker's name. The existing patterns only caught "my name is /
            # i am called / call me", so a user opening with "i'm noor" (as
            # every persona in the chat rounds does) was NEVER captured and a
            # later "what's my name?" returned "i don't know your name yet".
            # Structural: a capitalized bare proper noun after the
            # first-person copula, not a common noun or a determiner-led
            # phrase. Rejects "i'm a nurse" (common noun) and "i am the bee
            # guy" (determiner) so descriptor phrases are never stored as a
            # name. Generalizes across all personas (no per-name list).
            m_name = re.search(
                r"\b(?:i'?m|i\s+am)\s+"
                r"([A-Za-z][A-Za-z']*(?:\s+[A-Za-z][A-Za-z']*){0,2})"
                r"(?=[\s.,!?]|$)",
                q_clean)
            # NOTE: names are typed lowercase in chat ("i'm noor"), so we do
            # NOT require capitalization here — instead we rely on the
            # downstream guards (lines ~319-321 reject determiner-led phrases
            # like "i am a/the ...", and the non-name stoplist rejects states
            # / common nouns like "tired"/"hungry"/"nurse"). The bare form is
            # accepted as a name ONLY when it survives those guards.
        if not m_name:
            m_name = re.search(r"\b(?:do\s+you\s+know\s+my\s+name|know\s+my\s+name|is\s+my\s+name)\s+is\s+(.+)", q_clean, re.IGNORECASE)
        m_loc = re.search(
            r"\bi\s+(?:live|lives|am|was|were|grew\s+up|moved|move|stay|stayed)\s+"
            r"(?:in|near|at|from|to|onto)\s+"
            r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,7})",
            q_clean, re.IGNORECASE)
        # Round 2026-08-15T0326Z: also capture "i keep <thing> on <place>" /
        # "i have my studio on <place>" — a common way users state WHERE they
        # are (lighthouse keeper, boat restorer, ...). The main m_loc verb set
        # only covered live/am/was/grew-up, so "i keep the lighthouse on hollis
        # rock" was stored as a 'does' activity with NO location fact, and a
        # later "where do i live" fell through to an over-broad self-profile
        # dump (echoing the user's name). Generic: any "keep/have <noun> on
        # <Place>" captures the place head. The place is the token after "on";
        # capped at 4 words so a trailing clause ("on hollis rock, it's tiny")
        # is trimmed.
        m_loc_on = re.search(
            r"\bi\s+(?:keep|have|kept|had)\s+(?:the\s+|a\s+|an\s+|my\s+)?"
            r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3}\s+"
            r"on\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3})",
            q_clean, re.IGNORECASE)
        # FIX (round v-aug06b): when the location clause NAMES a place via
        # "called/named" (e.g. "i live in a small town called hollow creek"),
        # the real toponym is the named phrase, not the filler leading up to
        # it. Extract the named toponym and prefer it over the raw capture so
        # "hollow creek" is stored instead of "a small town called hollow".
        # FIX (round v-aug06d): extended the lexicon to natural features
        # (valley/dale/glen/cove/bay/fjord/island/peninsula/canyon/hollow) so
        # "a converted mill in a valley called ashcombe" resolves to "ashcombe"
        # (previously "valley" was absent and the greedy capture grabbed
        # "a converted mill in a"). No per-toponym table.
        _named_loc = re.search(
            r"\b(?:in|near|at|from)\s+(?:a|an|the|my|our|their|his|her)?\s*"
            r"(?:small\s+)?(?:town|city|village|settlement|place|hamlet|"
            r"community|suburb|borough|region|area|country|state|province|"
            r"valley|dale|glen|cove|bay|fjord|island|peninsula|canyon|hollow|"
            r"estuary|inlet|harbor|harbour|beach|shore)\b"
            r"(?:\s+(?:called|named|spelled))\s+"
            r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3})",
            q_clean, re.IGNORECASE)
        # Round 2026-08-13T2059Z: capture location from "based in X" /
        # "located in X" and from natural-feature location phrases
        # ("on the isle of skye", "on the island of man", "in the valley of
        # X"). The prior m_loc only matched live/lives/am/was/were/grew up
        # + in/near/at/from, so "i'm a lighthouse keeper based in skye" and
        # "i keep the lighthouse on the isle of skye" stored NO location and
        # a later "where do i live" fell back to the NAME. Structural: extra
        # trigger shapes, no per-toponym table; the same _put_fact("location")
        # path is reused so recall stays consistent by construction.
        _m_loc_based = re.search(
            r"\b(i(?:'m| am)|we)\s+(?:[a-z]+['\-]?\s+){0,8}?"
            r"(?:based|located|stationed|situated)\s+(?:in|on|at|near)\s+"
            r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2})",
            q_clean, re.IGNORECASE)
        _m_loc_feat = re.search(
            r"\b(?:in|on|at|near|off)\s+the\s+(?:isle|island|coast|shore|"
            r"headland|peninsula|cove|bay|fjord|valley|dale|glen|beach)\s+"
            r"of\s+([A-Za-z][A-Za-z'\-]*)",
            q_clean, re.IGNORECASE)
        _loc_cand = None
        if _m_loc_based:
            _subj = _m_loc_based.group(1).lower().strip()
            # Only extract location for first-person subjects
            if _subj.startswith("i") or _subj == "we":
                _loc_cand = _m_loc_based.group(2).strip().strip(" .,!")
        elif _m_loc_feat:
            _loc_cand = _m_loc_feat.group(1).strip().strip(" .,!")
        if _loc_cand and len(_loc_cand.split()) <= 5 and _loc_cand.lower() not in _VALUE_STOP:
            self.user_location = _loc_cand
            _put_fact("location", _loc_cand, 0.6)
        if _named_loc:
            m_loc = type("_Loc", (), {"group": lambda self, n: _named_loc.group(1)})()
            # store the named toponym directly (reuse m_loc handling below)
            _loc = _named_loc.group(1).strip().strip(" .,!")
            if _loc and len(_loc.split()) <= 4:
                self.user_location = _loc
                _put_fact("location", _loc, 0.6)
        elif m_loc_on and not m_loc:
            # Round 2026-08-15T0326Z: "i keep the lighthouse on hollis rock"
            # -> capture the place head after "on". Trim a trailing clause at a
            # comma/period so "hollis rock, it's tiny" -> "hollis rock".
            _loc = m_loc_on.group(1).strip().strip(" .,!")
            _loc = re.split(r"\s+(?:and|but|,|\.)\s*", _loc)[0].strip()
            if _loc and len(_loc.split()) <= 4:
                self.user_location = _loc
                _put_fact("location", _loc, 0.6)
        elif m_loc:
            _loc = m_loc.group(1).strip().strip(" .,!")
            _loc = re.split(r"\s+(?:and|but|,|\.)\s*", _loc)[0].strip()
            # FIX (round v-aug06c): a location clause like "a small apartment
            # near the river in porto" caps the 5-word capture at "a small
            # apartment near the" and silently drops the real toponym "porto".
            # A proper noun after a trailing "in/near/at <Place>" is the actual
            # place the user lives — prefer it. Generic: matches any capitalized
            # word led by a place preposition, no per-city table.
            # FIX (round v-aug06d): also catch a toponym introduced by
            # "called/named" ANYWHERE in the clause (not only end-of-string),
            # e.g. "a converted mill in a valley called ashcombe" where the
            # named-toponym lexicon missed the feature word — prefer the proper
            # noun after called/named over the filler leading up to it.
            _trailing = re.search(
                r"\b(?:in|near|at|from)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,2})\s*$",
                m_loc.group(1), re.IGNORECASE)
            if not _trailing:
                _trailing = re.search(
                    r"\b(?:called|named|spelled)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,3})",
                    m_loc.group(1), re.IGNORECASE)
            if _trailing:
                _loc = _trailing.group(1).strip()
            # Round 2026-08-08f: a long location clause with a trailing
            # measure/qualifier ("i live in a lighthouse on a rock about two
            # kilometers offshore") over-grabs the qualifier, pushing the
            # capture past the <=5-word gate so NO location fact is stored and
            # the disclosure falls through to the hollow "got it" ack. Trim a
            # trailing qualifier phrase led by a measure word ("about/around/
            # roughly" or a number+unit like "two kilometers") so the real
            # place head ("a lighthouse on a rock") is kept and stored.
            # Structural: cuts at a closed-class qualifier, never invents a
            # place; degrades gracefully if trimming leaves nothing.
            else:
                _loc = re.split(
                    r"\s+(?:about|around|roughly|approximately|some)\s+"
                    r"(?:\d+\s+\w+|\w+)\b", _loc)[0].strip()
                _loc = re.split(r"\s+\d+\s+(?:kilometer|meter|mile|km|mi|minute|hour|year|month)s?\b", _loc)[0].strip()
            if _loc and len(_loc.split()) <= 5:
                self.user_location = _loc
                _put_fact("location", _loc, 0.6)
        # POSSESSION-LOCATION miner (round 2026-08-10T0813Z). A named
        # possession's whereabouts ("the slow coal is moored at bingley",
        # "the van is parked in leeds") is a location fact about THAT entity,
        # not the user — but the location miner above only handles "i live in
        # X" / "i am from X". Without this, "where's the slow coal moored"
        # could only echo the raw utterance from the episodic index and a
        # later correction ("at saltaire, not bingley") had no structured fact
        # to supersede (measured: T23 stored nothing; T45 recall fell through
        # to the empty filler). Capture the entity + place and store it as an
        # entity-keyed location fact via _put_fact_ent (subject = entity, not
        # 'i'), so a later correction/contradict resolves by the same entity
        # key. Generic: any entity noun + place preposition; no per-place
        # table. The entity resolves through the same personal-fact store the
        # user can correct. Online/incremental; no retrain, no LLM.
        _pos_loc = re.search(
            r"\b(?:my|the|a|an|our|their|his|her)\b\s*"
            r"([\w'-]+(?:\s+[\w'-]+){0,3})"
            r"\s+(?:is|was|are|were|sits|lies|stays|remains)\s+"
            r"(?:moored|berthed|anchored|docked|based|parked|stationed|"
            r"kept|stored|housed|tied up|wintered)\s+"
            r"(?:at|in|on|near|by|outside|outside of)\s+"
            r"([\w'-]+(?:\s+[\w'-]+){0,3})", q_clean, re.IGNORECASE)
        if _pos_loc:
            _ent = _pos_loc.group(1).strip().strip(" .,!?").lower()
            _place = _pos_loc.group(2).strip().strip(" .,!?").lower()
            # Strip a leading hedge / discourse word from the entity head so
            # corrections like "actually the slow coal is moored at saltaire"
            # resolve to the SAME entity ("slow coal"), not a fresh one
            # ("ctually the slow coal"). The hedge set is SEED vocabulary
            # (discourse markers RAVANA can grow); missing one degrades to the
            # old behavior (a separate entity) — no crash, no wrong answer.
            _HEDGE = ("actually", "now", "well", "so", "but", "right",
                      "okay", "ok", "and", "then", "still")
            _ent_words = _ent.split()
            while len(_ent_words) > 1 and _ent_words[0] in _HEDGE:
                _ent_words = _ent_words[1:]
            _ent = " ".join(_ent_words)
            # reject closed-class / non-entity heads (e.g. "it is moored at x")
            if (_ent and _place and _ent not in _VALUE_STOP
                    and _place not in _VALUE_STOP
                    and len(_ent.split()) <= 4 and len(_place.split()) <= 4):
                # Trim a trailing qualifier/clause from the place ("bingley for
                # the winter" -> "bingley"; "leeds near the canal" -> "leeds";
                # "saltaire now" -> "saltaire").
                _place = re.split(r"\s+(?:for|near|by|outside|on|with|that|which|now|currently|these days|,|\.)\b",
                                  _place)[0].strip()
                if _place and _place not in _VALUE_STOP:
                    # A possession has exactly ONE whereabouts; a new location
                    # for the same entity SUPERSEDES the prior one (online
                    # correction, no retrain). The user is ground truth.
                    _prior = self.personal_facts.get(_ent, "location")
                    if _prior is not None and _prior.value.lower() != _place.lower():
                        self.personal_facts.contradict(_ent, "location", _place)
                    else:
                        _put_fact_ent(_ent, "location", _place, 0.6)
        if m_name:
            name_cand = m_name.group(1).strip()
            name_cand = re.split(r"\s+(?:and|but|,|\.)\s*", name_cand)[0].strip()
            name_words = name_cand.split()
            if name_words and name_words[0].lower() in ("is", "are", "was", "were"):
                name_words = name_words[1:]
            name_cand = " ".join(name_words)
            # A name is a proper noun, never an article-led description.
            # "call me a hypocrite" / "call me the bee guy" / "call me an
            # amateur" are self-descriptions, not names — reject candidates
            # that begin with a determiner so descriptor phrases are never
            # stored as the user's identity. Names are introduced bare
            # ("call me tobias", "i am called aria"), so this never blocks a
            # real name. Structural: a leading-determiner test, not a
            # per-name list.
            if name_cand.lower().startswith(("a ", "an ", "the ")):
                name_cand = ""
            # D1 (round 2026-08-08b): the bare "i'm X" / "i am X" copula form
            # is ALSO how users express transient states ("i'm furious at the
            # coast guard", "i'm scared the light will fail", "i'm feeling
            # really lonely", "i'm convinced cuttlefish are smarter"). The old
            # code stored the ENTIRE clause as (i, name, X), so a later
            # "what's my name" returned "feeling really lonely" / "furious at
            # the". A name is a SHORT proper noun, never a predicate clause.
            # Reject a bare-copula candidate unless it is name-shaped:
            #   - 1-2 word tokens (real chat names are single tokens: iris,
            #     noor, dev, soren, mira, wren, tobias; two-token names like
            #     "mary jane" are rare but legitimate), AND
            #   - contains NO closed-class word (preposition / copula /
            #     determiner: at/of/for/with/on/in/to/about/the/a/an/is/are/
            #     was/were/am/that/this/it/my/your), AND
            #   - its head token is NOT a feeling/state/cognitive verb.
            # This is structural predicate-vs-proper-noun discrimination, not a
            # per-name list; it generalizes across every persona. The
            # feeling/state verb set is SEED vocabulary (RAVANA-expandable: it
            # shares the role of the affect lexicon used by the empathy
            # gate), not authored answers.
            if name_cand:
                _nw = name_cand.split()
                _CLOSED = {
                    "at", "of", "for", "with", "on", "in", "to", "about",
                    "the", "a", "an", "is", "are", "was", "were", "am",
                    "that", "this", "it", "my", "your", "from", "by", "as",
                    "so", "but", "and", "or", "if", "because", "against",
                    "over", "under", "through", "without", "despite",
                    "except", "besides", "unlike", "into", "onto",
                }
                # A-name (round 2026-08-08c + 2026-08-14T1110Z): a bare
                # "i'm X" copula is how users express TRANSIENT STATES
                # ("i'm torn", "i'm shaking", "i'm proud", "i'm hollow") AND
                # predicates ("i'm against geoengineering", "i'm intense but
                # careful"). The old reject set was a FROZEN stoplist that only
                # ever listed the exact words a prior probe poisoned, so a
                # ROTATED probe (intense/euphoric/hooked/against) slipped
                # straight through and got stored as the user's NAME. The fix
                # is STRUCTURAL + GROWABLE, not a bigger list:
                #   1. Any closed-class / preposition head ("against/for/with")
                #      is a stance predicate, never a proper noun -> reject.
                #   2. The candidate head is tested against the CONSOLIDATED,
                #      runtime-extensible reject set (_name_rejectable), which
                #      merges the affect/state lexicon, the activity-deny set,
                #      and words the empathy/support classifier has observed
                #      as genuine affect at runtime (register_name_reject).
                # This is SEED vocabulary RAVANA GROWS by itself (no retrain,
                # no code change) — satisfying the round's seed-vs-hardcoding
                # test. Removing entries degrades gracefully. Covers
                # participles, irregulars, stative/cognitive verbs, and common
                # self-descriptor adjectives uniformly across every persona.
                _has_closed = any(w.lower() in _CLOSED for w in _nw)
                _head_reject = _name_rejectable(_nw[0]) if _nw else False
                # also reject any non-head token that is a rejectable predicate
                # ("intense but careful" -> "intense" rejected).
                _any_reject = _has_closed or _head_reject or any(
                    _name_rejectable(w) for w in _nw[1:])
                if len(_nw) > 2 or _any_reject:
                    # GROW the runtime reject set from the predicate words we
                    # just rejected, so a future rotated probe ("i'm <newword>")
                    # is caught even if the seed lexicon never listed it. This
                    # is how the guard learns online (no retrain, no code
                    # change) — the round's seed-vs-hardcoding test satisfied:
                    # RAVANA changes this by itself, through experience. We only
                    # register words that were REJECTED as predicates (never
                    # genuine name tokens), so a real name like "nadia" is
                    # never added to the deny set.
                    for _rw in _nw:
                        _rl = _rw.lower().strip("'\"")
                        if _rl and _name_rejectable(_rl):
                            register_name_reject(_rl)
                    name_cand = ""
            # Reject common states / descriptors / interrogatives so a bare
            # self-description is never stored as the user's name. Seed
            # stoplist of NON-name words, not a per-name allowlist.
            _NON_NAME = ("happy", "sad", "tired", "busy", "fine", "good",
                         "bad", "hungry", "thirsty", "angry", "mad", "glad",
                         "ok", "okay", "sorry", "here", "there", "lost",
                         "ready", "done", "sure", "right", "wrong", "well",
                         "sick", "late", "early", "home", "awake", "asleep",
                         "confused", "scared", "afraid", "excited",
                         "nervous", "calm", "what", "who", "why", "how",
                         "where", "when", "not", "no", "yes", "maybe")
            # DESCRIPTOR-NOUN deny set (round 2026-08-14T0103Z). "i'm
            # vegetarian" / "i'm vegan" / "i'm a ceramicist" / "i'm an
            # atheist" / "i'm a teacher" all reach the bare-copula name
            # candidate path and were stored as the user's NAME (name
            # poisoning: a later "what's my name" answered "Vegetarian" /
            # "Ceramicist"). A name is a PROPER NOUN (mira, wren, tobias),
            # never a common descriptor noun. This deny set covers the broad
            # CATEGORIES of self-descriptor nouns (diet, religion/identity,
            # occupation, nationality, orientation, family-role, political) as
            # SEED vocabulary — RAVANA-expandable, not a per-name allowlist,
            # so it generalizes to any descriptor the user might self-apply.
            # Removing entries degrades gracefully (only loses one guard).
            _NAME_REJECT_DESCRIPTOR = {
                # diet / lifestyle
                "vegetarian", "vegan", "omnivore", "pescatarian", "flexitarian",
                "carnivore", "meat-eater", "teetotaler", "teetotaller",
                # religion / belief / identity
                "atheist", "agnostic", "christian", "muslim", "islamist",
                "hindu", "buddhist", "jew", "jewish", "sikh", "pagan",
                "catholic", "protestant", "mormon", "spiritual", "humanist",
                "skeptic", "sceptic", "nihilist", "stoic", "optimist",
                "pessimist", "realist", "idealist", "pragmatist",
                # occupation / role
                "ceramicist", "ceramist", "artist", "painter", "sculptor",
                "writer", "author", "poet", "musician", "teacher", "student",
                "engineer", "doctor", "nurse", "lawyer", "chef", "baker",
                "programmer", "developer", "designer", "architect", "scientist",
                "researcher", "farmer", "fisherman", "sailor", "soldier",
                "officer", "clerk", "cashier", "waiter", "waitress", "barista",
                "driver", "pilot", "carpenter", "plumber", "electrician",
                "mechanic", "gardener", "librarian", "journalist", "editor",
                "actor", "singer", "dancer", "photographer", "printmaker",
                "potter", "weaver", "smith", "tailor", "cook", "builder",
                # nationality / origin
                "indian", "american", "british", "english", "scottish",
                "welsh", "irish", "french", "german", "spanish", "italian",
                "canadian", "australian", "chinese", "japanese", "korean",
                "russian", "mexican", "brazilian", "dutch", "swiss", "swede",
                "norwegian", "dane", "fin", "polish", "greek", "turk",
                # orientation / identity
                "straight", "gay", "lesbian", "bisexual", "transgender",
                "queer", "cisgender", "pansexual", "asexual", "demisexual",
                # family role (relative nouns can follow "i'm", e.g. "i'm a
                # father" / "i'm someone's sister") — these are roles, never names
                "father", "mother", "parent", "son", "daughter", "brother",
                "sister", "uncle", "aunt", "auntie", "cousin", "grandfather",
                "grandmother", "grandparent", "nephew", "niece",
                # political / affiliation
                "democrat", "republican", "socialist", "communist", "liberal",
                "conservative", "anarchist", "centrist", "libertarian",
                # general descriptors
                "introvert", "extrovert", "ambivert", "minimalist",
                "maximalist", "environmentalist", "feminist", "activist",
            }
            _nc_low = name_cand.lower()
            if (name_cand and _nc_low not in _NON_NAME
                    and _nc_low not in _NAME_REJECT_DESCRIPTOR):
                name_cap = " ".join(w.capitalize() for w in name_cand.split())
                self.user_name = name_cap
                _put_fact("name", name_cap, 0.6)

        # Round 2026-08-13T2059Z: GENERAL RELATIONAL MINER (the 6f
        # generalization gap — "who is wren to me" must resolve from the
        # store, type-agnostically, for ANY relationship the user states).
        # A bare "my <relation> <name> <rest>" disclosure (my cousin nora is
        # a glaciologist...; my partner june runs a library; my neighbour otto
        # fixes clocks; my auntie bea breeds ponies) is a RELATIONSHIP + an
        # ENTITY, not a self-profile attribute. It MUST be captured as a
        # relationship fact keyed by the NAMED ENTITY (subject=name,
        # attr=relationship) plus the descriptive rest as role/does keyed by
        # the same entity — reusing the SAME _put_fact_ent path the
        # possessive miner uses, so miner + recaller agree on the key by
        # construction. This generalizes across EVERY relationship type
        # (friend/sister/brother/cousin/partner/neighbour/auntie/pet/...);
        # it does NOT add a second narrow branch per entity kind.
        # RELATION_WORDS is SEED vocabulary: a closed set of common-English
        # relation terms, RAVANA-expandable at runtime via _learn_relation()
        # (the user can name any relation — e.g. "my godfather luis" extends
        # the set). It is DATA (not an if/elif answer path), and removing an
        # entry degrades gracefully (one fewer relation class captured). A
        # bare-capitalized-name test after it makes the miner fire on names
        # RAVANA has not seen, so it never needs the list to be exhaustive.
        _RELATION_WORDS = (
            "friend", "friends", "sister", "brother", "cousin", "siblings",
            "aunt", "auntie", "uncle", "nephew", "niece", "mother", "mom",
            "father", "dad", "parent", "parents", "grandmother", "grandfather",
            "grandma", "grandpa", "son", "daughter", "child", "children",
            "kid", "kids", "wife", "husband", "spouse", "partner", "fiance",
            "fiancee", "boyfriend", "girlfriend", "colleague", "coworker",
            "boss", "manager", "mentor", "student", "teacher", "neighbour",
            "neighbor", "roommate", "flatmate", "landlord", "tenant",
            "godfather", "godmother", "godparent", "stepfather", "stepmother",
            "stepsister", "stepbrother", "halfsister", "halfbrother",
            "grandson", "granddaughter", "pet", "cat", "dog", "horse", "pony",
            "companion", "bestie", "pal", "buddy", "acquaintance",
        )
        # mutable per-instance learned-relation set so RAVANA grows it online
        _rel_known = set(getattr(self, "_learned_relations", set())) | set(_RELATION_WORDS)
        # single-match form: "my <rel> <Name> <rest>" at end of clause or
        # anywhere; iterate so conjoined "my cat mochi ... and my dog pip ..."
        # captures BOTH entities (one match per "my <rel> <Name>" segment).
        _REL_RE = re.compile(
            r"\bmy\s+([a-z][a-z]+)\s+([A-Za-z][A-Za-z'-]+)"
            r"(?:\s+([^.?!]+?))?(?=\s+(?:and|but|,|&)|[.!?]|$)",
            re.IGNORECASE)
        for _m_rel in _REL_RE.finditer(q_clean):
            _rel = _m_rel.group(1).lower().strip()
            _ent = (_m_rel.group(2) or "").strip().strip(".,!?")
            _rest = (_m_rel.group(3) or "").strip().strip(".,!?")
            _looks_relation = (_rel in _rel_known) or (
                _rel.endswith("friend") or _rel.endswith("mate")
                or _rel in ("partner", "spouse", "sibling"))
            # Pet species (cat/dog/horse/...) are owned by the pet-slot system
            # (pet_slots.py) which keys them as (i, species) / (i, species_N);
            # never route them through the entity-keyed relationship store, or
            # "my cat is pixel" would be lost from the pet slot and break the
            # species-coexistence / cued-recall tests. Kinship + people
            # relations still flow through the relational miner.
            _is_pet = _pet_slots.species_of(_rel) is not None
            _name_shape = bool(re.match(r"^[A-Za-z][A-Za-z'-]+$", _ent)) \
                and _ent.lower() not in _VALUE_STOP
            if _looks_relation and _name_shape and len(_ent) >= 2 and not _is_pet:
                _put_fact_ent(_ent, "relationship", _rel, 0.65)
                if _rest:
                    _r = re.sub(r"^[\s,]+", "", _rest).strip()
                    # drop a leading copula + optional article ("is a X") OR a
                    # bare leading article ("a X") — the copula may already be
                    # consumed by the capture, leaving "a glaciologist ...".
                    _r = re.sub(r"^(?:is|are|was|were|'s)\s+(?: of)?\s+(?:a|an|the)\s+",
                                "", _r, flags=re.IGNORECASE).strip()
                    _r = re.sub(r"^(?:is|are|was|were)\s+", "", _r,
                                flags=re.IGNORECASE).strip()
                    _r = re.sub(r"^(?:a|an|the)\s+", "", _r,
                                flags=re.IGNORECASE).strip()
                    _r = re.split(r"(?<=[.!?])\s+|[.!?]\s*$", _r)[0].strip()
                    if _r and len(_r.split()) <= 10:
                        _put_fact_ent(_ent, "role", _r, 0.6)
                if _rel not in _RELATION_WORDS:
                    _rel_known.add(_rel)
                    self._learned_relations = _rel_known

        for _pat in (
            # D2 (round v2): tolerate an optional "name" word between the
            # relation and "is" so "my daughter name is ingrid" stores
            # daughter=ingrid (the old pattern required exactly one token
            # before "is" and missed the "name" form).
            # FIX (round v-aug06d): generalize to a MULTI-WORD attribute so
            # "my favorite color is ochre" stores attr="favorite color",
            # val="ochre" (the old single-token attr grabbed attr="favorite",
            # val="color" and lost the real value). Attribute capped at 4 words,
            # value at up to 8 words; this is structural (no per-topic
            # attribute list) — any "<attr phrase> is <value phrase>" is
            # captured. The cap was 4 words but real values are longer,
            # e.g. "my favorite time of day is the blue hour just before the
            # sun" was shredded to "the blue hour just" (the 4-word cap cut
            # "before the sun"). Raised to 8 words so a genuine multi-word
            # value is kept whole; a following sentence (".", "!", "?") is
            # still trimmed downstream so only THIS clause is stored.
            r"\bmy\s+([\w'-]+(?:\s+[\w'-]+){0,3})\s+(?:is|are)\s+([\w'-]+(?:\s+[\w'-]+){0,7})",
            # FIX (round v-aug06d): "i work as a <role>" / "i work for
            # <org>" self-descriptions are identity facts, not throwaway
            # activities. The old activity miner only caught the verb "work"
            # inside the generic activity loop and stored junk (does=s).
            # Capture the role as a durable 'work' fact. Generic: any noun
            # after "work as/for", no occupation list. Tolerates optional
            # adverbs/words between "work" and "as/for" ("i work nights as an
            # er nurse") — the previous pattern required them contiguous and
            # silently dropped "i work <adv> as X".
            r"\bi\s+work\b.*?\b(?:as|for)\s+(?:a\s+|an\s+|the\s+)?([\w'-]+(?:\s+[\w'-]+){0,4})",
            r"\bi\s+(?:have|keep)\s+(?:a|an|the)\s+([\w'-]+)\s+(?:named|called)\s+([\w'-]+)",
            # FIX (round v-aug06b): conjoined multi-pet disclosures
            # ("i have a ferret named pim and a parrot called coco"). The single
            # pattern above only captures the FIRST animal; the rest are lost.
            # This captures each "a/an <species> named/called <name>" segment
            # inside a "have ... and ..." chain so every pet is stored. Generic
            # (matches any species word, no per-animal table). The pattern is the
            # module-level _CONJOINED_PET_PAT constant; the miner identifies it by
            # object identity (_pat is _CONJOINED_PET_PAT), not a fragile
            # startswith of its literal (which previously never matched and left
            # the second animal unstored).
            _CONJOINED_PET_PAT,
            # C-fix (round v-aug04): quantified / multi-name possessions
            # ("i have two cats named biscuit and gravy", "i have 3 dogs
            # called rex, spot and max"). The old first pattern required a
            # single article + one name, so "two cats named biscuit and gravy"
            # never matched and the pets were lost. This matches an optional
            # number/quantifier word then a noun, then one or more names joined
            # by "and"/commas, and stores EACH name under its own entity slot
            # so a later "what are my cats called" recall finds them.
            r"\bi\s+(?:have|keep)\s+(?:\d+\s+|(?:a|an|the|some|several|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
            r"([\w'-]+)\s+(?:named|called|named\s+called)\s+"
            r"([\w'-]+(?:\s+(?:and|,|&)\s*[\w'-]+)*)",
            r"\bmy\s+([\w'-]+)\s+(?:named|called)\s+([\w'-]+)",
            # B-fix (round 2026-08-12T0613Z): a forward possession disclosure
            # with an INTERSTITIAL breed/adjective phrase between the species
            # and the name was never captured: "my dog is a nova scotia duck
            # tolling retriever called wren" / "my cat is a maine coon called
            # ember". The existing forward pattern requires the name
            # immediately after the species ("my dog called wren"), so the
            # breed phrase ("is a ... retriever") broke the match and the pet
            # fact was dropped — measured this round: T11 "my dog's a nova
            # scotia duck tolling retriever called wren" stored NOTHING, so a
            # later "what's my dog's name" fell through to "outside what i
            # know". This allows an optional copula (is/are/was/were OR the
            # spoken contraction 's, since users say "my dog's a retriever
            # called wren") + article + up-to-6-word breed/adjective span
            # before named/called, then routes through the SAME _pet_slots slot
            # logic the bare pattern uses (no per-animal table; species
            # resolved live). The species capture is LETTERS-ONLY ([\w-]+) so
            # the trailing 's is consumed as the copula, not folded into the
            # species token. Generic across any breed phrase length; the breed
            # words are discarded (only the species + name matter for the
            # slot). Fires only when the head word is a real species (resolved
            # by pet_slots), so non-pet "my brother is a tall guy called bob"
            # is handled by the existing guard, not learned as a pet.
            r"\bmy\s+([\w-]+)\s*(?:'s)?\s+(?:a|an|the\s+)?"
            r"(?:[\w'-]+\s+){0,6}?(?:named|called)\s+([\w'-]+)",
            # D2: "i am a/an <noun>" self-descriptions (vegetarian, pilot,
            # teacher, ...) captured as a durable identity/role fact. Generic
            # structural capture — the noun becomes the attribute value, no
            # per-topic list. A small universal stop-set excludes non-facts
            # like "person"/"human"/"child" which are not stored attributes.
            r"\bi\s+am\s+(?:a|an)\s+([\w'-]+)",
            r"\bi\s+am\s+allergic\s+to\s+([\w'-]+)",
        ):
            for _m in re.finditer(_pat, q_clean, re.IGNORECASE):
                _attr, _val = None, None
                # Round 2026-08-13T2059Z: skip the generic "my <attr> is
                # <val>" / "my <attr> <val>" capture when the attr is a
                # RELATIONSHIP word followed by a NAME — that is the
                # "my <rel> <name> <rest>" shape already handled by the
                # relational miner above (stored under the ENTITY key, not
                # the user's 'i' subject). Without this guard the loop
                # ALSO stored ('i', 'cousin nora', ...) alongside the
                # correct ('nora', 'relationship', 'cousin'), polluting the
                # self-profile with a relationship dressed as a self-attr.
                # Structural: a relation-set membership test, not a
                # per-relation branch. _rel_known covers the seed set +
                # runtime-learned relations so it stays in sync with the
                # relational miner.
                _raw_attr = (_m.group(1).strip().lower()
                             if _m.lastindex is not None and _m.lastindex >= 1
                             else "")
                # attr may be multi-word ("cousin nora"); a relation-led attr
                # starts with a relation word followed by a proper-noun name.
                # Pet species (cat/dog/...) are excluded — they are owned by
                # the pet-slot system (pet_slots.py) and must NOT be skipped,
                # or "my cat is pixel" falls through to nothing.
                # Only skip the genuine "my <rel> <Name>" shape where the word
                # right after the relation is a NAME (not a copula/article):
                # "my cousin nora is ..." skips (nora is the named entity), but
                # "my child is a curious kid" does NOT (the word after "child"
                # is the copula "is", so it is a normal self-attribute).
                if _raw_attr:
                    _raw_head = _raw_attr.split()[0] if _raw_attr.split() else ""
                    _after = re.search(
                        r"\b" + re.escape(_raw_head)
                        + r"\s+([A-Za-z][A-Za-z'-]*)\b", q_clean, re.IGNORECASE)
                    _COPULA_ART = ("is", "are", "was", "were", "am", "a", "an",
                                   "the", "i", "my", "of", "to", "for", "on",
                                   "in", "at", "by", "with", "and", "that")
                    if (_raw_head in _rel_known
                            and _pet_slots.species_of(_raw_head) is None
                            and _after is not None
                            and _after.group(1).lower() not in _COPULA_ART
                            and re.match(r"^[A-Za-z][A-Za-z'-]+$",
                                         _after.group(1))):
                        continue
                # chain like "ferret named pim and a parrot called coco".
                # Expand it into individual "species named name" pairs and
                # store each via the same slot logic below. Identified by object
                # identity (_pat IS the module constant), which is exact — the
                # previous startswith(guard) comparison against a DIFFERENT
                # literal never matched, so the branch was dead and the second
                # animal was silently dropped.
                if _pat is _CONJOINED_PET_PAT:
                    _segs = re.findall(
                        r"([\w'-]+)\s+(?:named|called)\s+([\w'-]+)",
                        _m.group(1), re.IGNORECASE)
                    for _sp, _nm in _segs:
                        _sp, _nm = _sp.strip().lower(), _nm.strip().strip(".,!?")
                        if not _sp or not _nm:
                            continue
                        _species = _pet_slots.species_of(_sp)
                        if _species is None and _sp.isalpha():
                            _species = _pet_slots.learn_species(_sp)
                        else:
                            _species = _pet_slots.species_of(_sp) or _sp
                        if _species is not None:
                            # count existing slots for this species to append _2, _3
                            _i = 1
                            while _pet_slots.slot_for(_species, _i) in self.personal_facts.facts:
                                _i += 1
                            _put_fact(_pet_slots.slot_for(_species, _i), _nm, 0.6)
                    continue
                if _m.lastindex is not None and _m.lastindex >= 2:
                    _attr, _val = _m.group(1).strip().lower(), _m.group(2).strip()
                    # Trim any FOLLOWING sentence so a value like "the blue
                    # hour just before the sun. i also keep pigeons" stores
                    # only the first clause, not the whole tail. Only a hard
                    # sentence break (., !, ?) ends the value; internal
                    # prepositions/conjunctions ("of the", "before the") are
                    # kept so a genuine multi-word value stays whole.
                    _val = re.split(r"(?<=[.!?])\s+|\s*[.!?]\s*$", _val)[0].strip()
                elif _m.lastindex == 1:
                    # Single-group patterns: "i am a/an <noun>" / "i am
                    # allergic to <noun>" — the one group is the VALUE; the
                    # attribute is inferred from the pattern.
                    _val = _m.group(1).strip().lower()
                    if _pat.startswith(r"\bi\s+am\s+(?:a|an)"):
                        _attr = "role"
                        if _val in ("person", "human", "woman", "man", "child",
                                    "kid", "adult", "student", "robot", "machine",
                                    "ai", "thing", "people", "boy", "girl"):
                            continue
                    elif _pat.startswith(r"\bi\s+work"):
                        _attr = "work"
                        # Trim trailing prepositional noise ("er nurse in a
                        # hospital near the river" -> "er nurse") via the same
                        # content-head resolver used for stances.
                        _trimmed = self._opinion_topic(_val)
                        if _trimmed:
                            _val = _trimmed
                    elif _pat.startswith(r"\bi\s+am\s+allergic\s+to"):
                        _attr = "allergy"
                # D6 (round 2026-08-08b-d): a possessive attr ('my partner's
                # name is theo' -> attr="partner's name") must be stored under
                # the OWNER entity, not the user's 'i' subject. Otherwise a
                # later recall renders "your name is theo" (partner's name
                # reported as the user's). Split the possessive head into
                # (entity, relation) and route through _put_fact_ent. The
                # recall reconstructor (_retrieve_episodic / _structured_recall)
                # already keys possessive facts by owner, so this makes the
                # MINER agree with the recaller by construction.
                _ent, _rel = _split_possessive_attr(_attr)
                # A NAME relation holds a short proper noun (the entity's name),
                # never the trailing descriptive clause that often follows it
                # ("my dog's name is wren and she's a scruffy terrier",
                #  "my brother is arjun and he's learning to weld"). The old
                # value trim (line ~1119) only split on a hard sentence break,
                # so the whole post-copula clause ("wren and she's a scruffy
                # terrier") was stored as the name value — which then broke
                # reverse-lookups ("who is wren to me") and self-summaries.
                # General fix: when the relation is a name, keep ONLY the
                # leading proper-noun run and cut at the first coordinating
                # clause / appositive (" and ", " but ", " who ", " she's ",
                # " he's ", ", ", " with "). Multi-word proper names ("mary
                # jane") survive because we stop at the clause boundary, not at
                # the first space. Structural, not a per-entity table; the
                # trimmed remainder is simply dropped (a name has no second
                # fact). This also makes a later correction ("no, my dog is
                # milo") supersede cleanly.
                if _rel == "name":
                    _val = re.split(
                        r"\s+(?:and|but|who|that|which|,)\s+|\s+(?:she|he|it)'?s\s+"
                        r"|\s+with\s+|\s*\.\s*|\s*\?\s*", _val)[0].strip()
                    _val = _val.strip(".,!?;:'\"")
                if _ent is not None:
                    _put_fact_ent(_ent, _rel, _val, 0.6)
                    continue
                if _attr and _val and _attr not in ("name", "location"):
                    # A possession disclosure may name several animals ("i have
                    # two cats named biscuit and gravy"). Split the value on
                    # "and"/"," and store EACH name in its own slot. The slot
                    # KEEPS the species (cat, cat_2, dog) via pet_slots so a
                    # user can own more than one kind of animal, a cued recall
                    # can ask about one species, and a correction ("no, my cat
                    # is milo") finds the prior value under the same key the
                    # user's own word resolves to.
                    _names = re.split(r"\s+(?:and|,|&)\s*", _val)
                    _species = _pet_slots.species_of(_attr)
                    if _species is None and len(_names) >= 1 and _attr.isalpha():
                        # An unknown animal word in a "named/called" possession
                        # frame is a species RAVANA has not met yet — learn it
                        # so later recalls address the same slot.
                        if re.search(r"\b(?:named|called)\b", _m.group(0), re.IGNORECASE):
                            _species = _pet_slots.learn_species(_attr)
                    if _species is not None:
                        for _i, _nm in enumerate(_names, 1):
                            _nm = _nm.strip().strip(".,!?")
                            if _nm:
                                _put_fact(_pet_slots.slot_for(_species, _i),
                                          _nm, 0.6)
                    elif len(_names) > 1:
                        for _i, _nm in enumerate(_names, 1):
                            _nm = _nm.strip().strip(".,!?")
                            if _nm:
                                _put_fact(f"{_attr}_{_i}", _nm, 0.6)
                    else:
                        _put_fact(_attr, _val, 0.6)

        # D4 (round 2026-08-11T1328Z): a communication / meta verb is itself a
        # speech-act or inner-state report, never a possession or lived
        # activity. "i told a friend X" / "i keep saying it" / "i felt a kind
        # of weight lift" must not become ('i','does','told friend...') /
        # ('i','does','keep saying') / ('i','does','felt kind'). Reject any
        # activity capture whose VERB is in this SEED set (RAVANA-expandable,
        # not a per-topic answer table). A real activity verb ("keep pigeons"/
        # "brew"/"forge"/"run") is never in this set, so genuine disclosures
        # still pass in both the D3-style loop and the general-activity loop.
        # NOTE (round 2026-08-11T1328Z audit fix t_86c5c46b): this set is for
        # SPEECH-ACT / INNER-STATE verbs only ("tell"/"say"/"feel"/"think"...).
        # Genuine possession / loss verbs "keep"/"kept"/"lose"/"lost" were
        # wrongly listed here, so first-person disclosures ("i keep homing
        # pigeons", "i lost a kestrel", "i keep a saltwater reef tank") were
        # dropped BEFORE capture, and the downstream count-correction path
        # (which needs the stored 'does' text fact as its prior) went dead,
        # leaving stale "six hives" after a "seven" correction. Their
        # meta-discourse protection is already provided by the OBJECT-level
        # guards (_META_HEAD / _META_DISCOURSE / embedded-question scan), which
        # reject "keep saying" / "felt kind" / "lose track of whether..." on
        # the object HEAD — so the verb-level guard must NOT reject real
        # possession/loss verbs. Keep/lose are SEPARATELY in the activity/event
        # verb seed lists so the disclosures still land.
        _META_VERBS = (
            "tell", "tells", "told", "say", "says", "said", "mention",
            "mentions", "mentioning", "recall", "recount", "repeat",
            "repeated", "feel", "feels",
            "felt", "think", "know", "learn", "forget",
        )

        # D3 (round v3): capture self-disclosed ACTIVITIES / possessions that the
        # existing "my X is Y" / "i am a role" miners miss — e.g. "i run a chai
        # stall near the mysore palace", "i play the tabla when the stall is
        # closed", "i've been watching the night sky for twelve years". These are
        # first-person disclosures of what the user DOES / has, and must land in
        # the personal-fact store so a later "what have you learned about me"
        # summary and cued recall can surface them (D-D bug: only 'likes' was
        # recalled because activity facts were never mined). Structural: a small
        # closed VERB set (not a per-topic list) + the resolved content HEAD of
        # the object phrase (via _opinion_topic, which drops closed-class words),
        # so the stored value is a real concept ("chai stall", "tabla", "night
        # sky"), never a function word. This is seed structure RAVANA expands from
        # experience — it adds to the same PersonalFactStore the user can correct.
        for _verb in ("run", "own", "operate", "play", "teach", "study",
                       "manage", "drive", "build", "make", "sell",
                       "restore", "grow", "watch", "raise", "tend", "brew",
                       "bake", "write", "read", "learn", "practice", "collect",
                       "fix", "paint", "code", "design", "craft", "volunteer",
                       "cook", "fish", "hike", "garden", "farm", "lead", "organize",
                       # Round 2026-08-08: activity verbs the prior seed set
                       # missed, so self-disclosures like "i keep homing
                       # pigeons", "i grind my own telescope mirrors", "i race
                       # homing pigeons", "i sail a small dinghy", "i knit
                       # wool socks", "i forge my own knives" were silently
                       # dropped and later recall had nothing to recall. These
                       # are the SAME kind of seed verb vocabulary (RAVANA
                       # expands it from experience; the user can correct any
                       # stored 'does' fact), not a per-topic list.
                       "keep", "grind", "race", "sail", "fly", "knit", "sew",
                       "weld", "forge", "carve", "compose", "record",
                       "perform", "coach", "train", "compete", "spin",
                       "weave", "mount", "trade", "sell", "host", "guide"):
            if not _activity_verb_ok(_verb):
                continue
            _m = re.search(
                r"\bi\s+(?:" + _FRAMER_SKIP + r")?"
                r"(?:have\s+been\s+)?(?:been\s+)?(?:keep\s+|grind\s+|race\s+)?"
                + _verb +
                # D3 (round 2026-08-08b-d): the article alternative `a` had NO
                # word boundary, so for "i teach at a school" it matched the
                # leading 'a' of "at", leaving "t a school" as the object ->
                # stored "teach t". Bound the article alternatives with \b so
                # `a`/`an`/`the` only match a STANDALONE article, never the
                # prefix of another word. The object still passes through
                # _opinion_topic which drops closed-class words, so "i teach at
                # a school by the river" -> "school" (real concept head), not
                # "t". Structural; no per-topic table.
                r"\s+(?:my\s+|\b(?:a|an|the)\b\s+)?(.+?)(?:\bfor\b|\bwhen\b|\bbut\b|"
                r"\bbecause\b|\band\b|\.|\!|\?|$|,)",
                q_clean, re.IGNORECASE)
            if _m:
                # retraction cue ("i take back what i said") is not an activity
                if _verb in ("take", "took", "taking") and "back" in _m.group(1).lower():
                    continue
                _obj = self._opinion_topic(_m.group(1).strip().lower())
                _obj = _strip_obj_framers(_obj)
                if _obj and len(_obj.split()) <= 5:
                    # D4 fix (round 2026-08-11T1328Z): the resolved object HEAD
                    # must not be a communication / meta-discourse / abstract-
                    # state word ("saying", "told", "kind", "track", "felt"...).
                    # "i keep saying it" / "i felt a kind of weight lift" matched
                    # the activity verb and stored junk ('does' = "keep saying"
                    # / "felt kind") that pollutes recall. A real possession/
                    # activity head ("pigeons", "tabla", "homing pigeons") is
                    # never in this SEED vocabulary and still passes. Shared
                    # with the general-activity miner below so both capture the
                    # same real disclosures and reject the same meta-discourse.
                    _META_HEAD = (
                        "saying", "says", "said", "tell", "tells", "telling",
                        "told", "mention", "mentions", "mentioning", "recall",
                        "recount", "repeat", "repeated", "keep", "kept",
                        "lose", "lost", "kind", "weight", "hush", "lift",
                        "track", "sort", "type", "notion", "feeling", "feels",
                        "felt", "bit", "thing", "stuff", "business", "matter",
                        "point",
                    )
                    if _obj.split()[0] in _META_HEAD:
                        continue
                    # Store the verb WITH the object ("keep homing pigeons")
                    # so activity recall ("what do i keep?") can match the
                    # verb and return a complete, grammatical answer instead
                    # of a bare noun. The verb is part of the mined disclosure,
                    # not an authored label.
                    _put_fact("does", f"{_verb} {_obj}", 0.55)
        # "i've been <verb>-ing <object> for <duration>" (ongoing activity)
        _cont = re.search(
            r"\bi(?:'ve| have)\s+been\s+(\w+ing)\s+(.+?)(?:\bfor\b|\bsince\b|\.|\!|\?|\$|,)",
            q_clean, re.IGNORECASE)
        if _cont:
            _cverb = _cont.group(1).lower()
            if _activity_verb_ok(_cverb):
                _obj = self._opinion_topic(_cont.group(2).strip().lower())
                _obj = _strip_obj_framers(_obj)
                if _obj and len(_obj.split()) <= 5:
                    _put_fact("does", f"{_cverb} {_obj}", 0.55)

        # FIX (round 2026-08-09T1953Z): general first-person activity +
        # experience capture. The D3 activity loop above only matches BARE
        # verb forms ("train", "keep") and omits common disclosure verbs
        # ("throw", "shoot", "develop", "clean", "grow", "train"). Real chat
        # is dominated by gerunds and continuous tenses ("i throw pots",
        # "i've been training a juniper bonsai", "i shoot 35mm", "i keep air
        # plants", "i clean the reef tank glass") — none of which the D3 loop
        # caught, so they fell through to the hollow "got it — thanks for
        # telling me." ack AND became unrecallable. This block generalises the
        # capture to inflected forms and a broader closed VERB SEED set, and
        # ADDS firsthand-experience (event) capture for disclosures like "i
        # dropped half its needles", "i lost a favia coral to heat", "i
        # repotted the juniper and found a root that went necrotic", "i
        # removed the dead favia". These are real things the user did/experienced
        # about their world and must land in the same PersonalFactStore so cued
        # recall and the "what have you learned about me" summary can surface
        # them (the old code only ever recalled 'does'/'likes' activity facts).
        #
        # DESIGN (per round hardcoding rule + seed-vs-hardcoding test):
        #  - The verb vocabulary is SEED structure: a closed list of
        #    activity/experience verbs. It is RAVANA-expandable in principle
        #    (it feeds the same PersonalFactStore the user can correct/extend),
        #    NOT a per-topic answer dictionary and NOT authored reply prose.
        #    Removing an entry degrades gracefully (one fewer activity class
        #    captured) — it is not content RAVANA can never change, so it is
        #    seed knowledge, not hardcoding.
        #  - The value is the resolved CONTENT HEAD of the object phrase
        #    (_opinion_topic drops closed-class words), so the stored value is
        #    a real concept ("pots", "bonsai", "35mm", "air plants", "reef
        #    tank glass"), never a function word.
        #  - Capture is GENERAL (any "i <verb> <object>"), so it fires on new
        #    topics without retraining or per-topic tuning.
        #  - Activity verbs -> attr "does" (consistent with the D3 loop).
        #  - Experience/event verbs -> attr "event" (new), so a later recall
        #    can reconstruct "you dropped <x>" / "you lost <y>" grammatically
        #    (see engine_memory._reconstruct_entity + engine_reasoning
        #    ._derive_ack_from_store which now render the 'event' attr).
        # Round 2026-08-13T2059Z: PRINCIPLED VERB PRUNE. The activity/event
        # verb seeds below mix genuine activity/experience verbs with
        # ACHIEVEMENT / COMMUNICATION verbs (got, said, made, gave, told,
        # came, went, did, saw, met, sold, paid, sent, spent, bought, caught,
        # brought, ate, drank, knew, wore, led, read, flew, swam, rode, drove,
        # broke, spoke, woke, froze, chose, slept, felt, held, found, lost,
        # kept, took, set, put, cut, hit, fed, bled, built, taught, wrote,
        # drew, sang, grew, threw). Those fire on OUTCOME / UTTERANCE
        # disclosures whose object is a bare noun naming a result
        # ("i got the artist residency" -> does=got artist residency;
        # "i said open-plan offices help" -> does=said open), and they echo
        # verbatim in the self-summary as garbage. SEED set (RAVANA-
        # expandable): these are communication/achievement verbs, not
        # sustained activities or physical-world experiences; removing them
        # degrades gracefully (one fewer outcome-class captured, which is
        # correct — outcomes are not recurring activities). The genuine
        # activity verbs (keep/grow/start/lost/found/build/throw/play/...)
        # and experience verbs (drop/lose/find/break/heal/...) are kept,
        # so the prior fact-mining tests (throw pots, grow air plants,
        # repot juniper, lost favia coral, reef tank) stay GREEN.
        _ACHIEVE_COMM_VERBS = frozenset({
            "got", "get", "said", "say", "gave", "give",
            "told", "tell", "came", "come", "went", "go", "did", "do",
            "met", "meet", "sold", "sell", "paid", "pay",
            "sent", "send", "spent", "spend", "bought", "buy",
            "brought", "bring", "ate", "eat", "drank", "drink",
            "knew", "know", "wore", "wear", "led", "lead",
            "spoke", "speak", "woke", "wake",
            "fed", "feed", "bled", "bleed",
        })
        # Closed VERB SEED vocabulary (RAVANA-expandable; feeds the same

        # PersonalFactStore the user can correct — NOT per-topic answers, NOT
        # authored prose). Covers everyday disclosure verbs + common irregular
        # past forms so first-person activities/experiences actually land.
        _ACTIVITY_VERBS = tuple(v for v in (
            "run", "own", "operate", "play", "teach", "study", "manage",
            "drive", "build", "make", "sell", "restore", "grow", "watch",
            "raise", "tend", "brew", "bake", "write", "read", "learn",
            "practice", "collect", "fix", "paint", "code", "design", "craft",
            "volunteer", "cook", "fish", "hike", "garden", "farm", "lead",
            "organize", "keep", "grind", "race", "sail", "fly", "knit",
            "sew", "weld", "forge", "carve", "compose", "record", "perform",
            "coach", "train", "compete", "spin", "weave", "mount", "trade",
            "host", "guide", "throw", "shoot", "develop", "clean", "reload",
            "recharge", "assemble", "mix", "pour", "press", "roll", "fire",
            "glaze", "wire", "prune", "pot", "plant", "sketch", "draw",
            "sculpt", "stitch", "mend", "whittle", "start", "begin", "try",
            "go", "use", "take", "make", "get", "built", "taught", "wrote",
            "drew", "sang", "flew", "swam", "rode", "drove", "broke",
            "spoke", "woke", "froze", "chose", "ate", "drank", "grew",
            "threw", "knew", "wore", "brought", "bought", "caught",
            "kept", "slept", "left", "felt", "met",
            "sent", "spent", "lost", "found", "held", "told", "sold",
            "paid", "said", "gave", "came", "went", "did", "saw", "got",
            "made", "took", "set", "put", "cut", "hit", "read", "led",
            "fed", "bled", "fed",
        ) if v not in _ACHIEVE_COMM_VERBS)
        _EVENT_VERBS = tuple(v for v in (
            "drop", "lose", "find", "remove", "break", "discover", "notice",
            "repot", "prune", "harvest", "spill", "melt", "crack", "kill",
            "ruin", "save", "nurse", "revive", "miss", "spot",
            "catch", "pull", "cut", "burn", "flood", "rescue", "rebuild",
            "recover", "heal", "uproot", "freeze", "thaw", "hatch",
            "bloom", "wilt", "die", "survive", "escape", "return", "birth",
            "fall", "fell", "crash", "lose", "lost", "found", "kept",
            "broke", "felt", "cut", "hit", "met", "told", "saw", "got",
            "made", "took", "gave", "came", "went", "did", "ate", "drank",
            "grew", "knew", "threw", "froze", "bled", "fed", "died",
        ) if v not in _ACHIEVE_COMM_VERBS)
        # Match "i [aux?] <verb>(s|ed|ing)? <object> <clause-boundary>".
        # The object stops at a clause boundary (., !, ?, ",", " and ",
        # " but ", " because ", " so ", " which ", " that ", " when ",
        # " where ") so a multi-clause sentence stores only the relevant
        # fragment (e.g. "i repotted the juniper and found a root..." ->
        # "juniper", not "juniper and found a root"). The verb is matched
        # with optional inflection so gerunds/continuous tenses are caught.
        # Round 2026-08-14T0103Z: GENERAL verb-frame guard (defined BEFORE the
        # seeded ACTIVITY/EVENT blocks so they can all use it). The open-class
        # miner (and the seeded blocks) treat ANY word after "i" as the verb,
        # so framer / temporal / negation words preceding the real activity
        # verb were captured as the verb and stored as garbage 'does' facts:
        # "i won't buy fish" -> does="won't buy fish"; "i used to love..." ->
        # does="used love"; "i first lit a kiln" -> does="first lit";
        # "i mis-spoke earlier" -> does="mis-spoke earlier". These are NOT
        # activities RAVANA learned. Fix structurally: (a) NORMALISE the
        # captured verb — strip an apostrophe contraction artifact and drop a
        # leading "n't"/"not" so negations are not stored as the activity;
        # (b) reject verbs that are FRAME / TEMPORAL / DISCOURSE words (used,
        # first, mis-spoke, probably, still, just, really, also ...) — these
        # precede the real verb and must never be the mined activity; (c) reject
        # objects that are PURELY temporal/discourse tails ("earlier", "now",
        # "today") or empty. Seed deny-sets (RAVANA-expandable, removing entries
        # degrades gracefully), applied to ALL THREE capture blocks so the
        # seeded whitelist and the open-class fallback agree by construction.
        # No per-verb answer table, no hardcoded reply.
        _STATIVE_DENY = frozenset({
            # copula / existence
            "am", "are", "is", "was", "were", "be", "been", "being",
            "become", "seem", "appear", "remain",
            # affect / cognition / volition (handled by opinion/benign paths)
            "feel", "feels", "love", "like", "hate", "dislike", "prefer",
            "think", "believe", "know", "understand", "want", "need", "wish",
            "hope", "guess", "suppose", "mean", "wonder", "agree", "disagree",
            "doubt", "fear", "regret", "suspect", "realize", "realise",
            "remember", "recall", "imagine", "mind", "care",
            "enjoy", "adore", "detest", "cherish", "miss",
            # possession (handled by 'my X is Y' / have patterns)
            "have", "has", "had", "own", "possess",
            # communication / achievement utterances (echo verbatim as garbage;
            # seeded out just like _ACHIEVE_COMM_VERBS above)
            "got", "get", "said", "say", "gave", "give",
            "told", "tell", "came", "come", "went", "go", "did", "do",
            "met", "meet", "sold", "sell", "paid", "pay",
            "sent", "send", "spent", "spend", "bought", "buy",
            "brought", "bring", "ate", "eat", "drank", "drink",
            "knew", "wore", "wear", "led", "lead",
            "spoke", "speak", "woke", "wake",
            "fed", "feed", "bled", "bleed",
        })
        _VERB_FRAME_DENY = frozenset({
            # temporal / aspectual framers (not activities)
            "used", "first", "last", "then", "next", "once", "twice",
            "again", "finally", "recently", "lately", "soon", "already",
            "just", "still", "now", "earlier", "later", "today", "tonight",
            "yesterday", "tomorrow", "sometimes", "often", "usually",
            "always", "never", "occasionally", "rarely",
            # discourse / correction / stance framers
            "mis-spoke", "misspoke", "meant", "suppose", "guess", "wonder",
            "realize", "realise", "mean", "admit", "confess",
            # modal-ish framers that are not the activity itself
            "probably", "possibly", "maybe", "certainly", "definitely",
            "really", "truly", "actually", "basically", "simply", "quite",
            "very", "also", "even", "rather", "instead",
            # modality / auxiliaries (never a mined activity): "i should
            # handle" / "i can lift" / "i will finish" are modality, not a
            # sustained activity, and questions like "how do you think i
            # should handle that?" must not leak "should handle" as a 'does'.
            "should", "would", "could", "can", "may", "might", "must",
            "shall", "will", "ought",
        })

        def _norm_verb(v: str) -> str:
            v = v.strip().lower().lstrip("'").rstrip("'")
            # A captured verb containing an apostrophe is a CONTRACTION
            # artifact ("won't", "can't", "don't", "isn't") — not a clean
            # activity verb. Negation is better expressed via the
            # opinion/stance path, so we normalize to empty and let _verb_ok
            # reject it rather than storing "won't buy fish" as a 'does' fact.
            if "'" in v:
                return ""
            if v.startswith("n't"):
                v = v[3:] or v
            elif v == "not":
                v = ""
            return v

        def _verb_ok(v: str) -> bool:
            v = _norm_verb(v)
            if not v:
                return False
            if v in _STATIVE_DENY or v in _VERB_FRAME_DENY:
                return False
            return True

        def _obj_ok(obj: str) -> bool:
            o = (obj or "").strip().lower()
            if not o:
                return False
            _OBJ_STOP = {"earlier", "later", "now", "today", "tonight",
                         "yesterday", "tomorrow", "recently", "lately",
                         "soon", "then", "next", "again"}
            return o not in _OBJ_STOP

        def _is_question(t: str) -> bool:
            # A first-person activity/event can ONLY be mined from a
            # DECLARATIVE self-report, never from a question. Mining a
            # question ("how do you think i should handle that?") leaks
            # modality tails ("should handle") as garbage 'does' facts.
            t = (t or "").strip()
            if t.endswith("?"):
                return True
            return bool(re.match(
                r"^(what|who|when|where|why|how|which|is|are|do|does|did|"
                r"can|could|would|should|will|may|might|am|have|has|had)\b",
                t, re.IGNORECASE))

        _act_pat = re.compile(
            r"\bi\s+(?:" + _FRAMER_SKIP + r")?"
            r"(?:have\s+been\s+|has\s+been\s+|am\s+|was\s+|were\s+)?"
            r"(?:been\s+)?"
            r"(" + "|".join(_ACTIVITY_VERBS) + r")(?:s|es|ing|ed|[a-z]ed|[a-z]d)?"
            r"\s+(?:my\s+|a\s+|an\s+|the\s+|some\s+|two\s+|three\s+|four\s+|"
            r"five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)?"
            r"(.+?)(?:\s*(?:\.|\!|\?|,|-{1,3}|$|"
            r"\s+and\s+|\s+but\s+|\s+because\s+|\s+so\s+|\s+which\s+|"
            r"\s+that\s+|\s+when\s+|\s+where\s+|\s+while\s+))",
            re.IGNORECASE)
        # D5 (round 2026-08-10T0813Z): reject activity captures whose OBJECT
        # is an embedded question, a meta-reflection, or a verbatim self-quote
        # rather than a real possessed thing. "i lose track of whether i told
        # you" matched the activity verb "told"/"lose" and stored junk
        # ('does'/'event' = "told you", "lose track") that polluted later
        # recall. The object clause is scanned for question-frames
        # (whether/if/when/what/why/how/who + a second clause) and for the
        # self-quote pronoun "you"/"me" as the object head (a quoted speech
        # act, not a possession). Generic: structural grammar test, no
        # per-topic list; the real possession object still passes through.
        def _activity_obj_is_real(_obj: str, _raw: str = "") -> bool:
            _o = (_obj or "").strip().lower()
            if not _o:
                return False
            # Scan the RAW object clause (before _opinion_topic trims it) for
            # an embedded/subordinate question ("of whether i told you",
            # "why he left") — a meta-reflection, not a possessed thing. The
            # head is often a content word ("track"), so the trimmed topic
            # alone would pass; the raw clause exposes the quote/question.
            _raw_l = (_raw or _o)
            if re.search(
                r"\b(?:of|about|whether|if|why|how|what|when|who|where|that)\b\s+"
                r"(?:i|you|we|they|he|she|it|the|a|an|my|your|this|that)\b",
                _raw_l):
                return False
            _head = _o.split()[0]
            # object resolves to a quote/self-reference, not a thing
            if _head in ("you", "me", "i", "im", "i'm", "we", "us", "they", "them"):
                return False
            # the trimmed topic is a single closed-class / particle word
            # ("up", "off", "out") left after stripping the real object — a
            # dangling verb-particle, not a possessed thing ("i mixed them
            # up" -> topic "up", "i got it wrong" -> "wrong"). Reject.
            if len(_o.split()) == 1 and _o in (
                "up", "down", "off", "out", "around", "over", "wrong",
                "right", "back", "in", "on"):
                return False
            # the object is a self-error / meta-reflection verb ("muddled",
            # "confused", "mistaken") — "i got muddled" / "i was confused"
            # reports the user's own slip, not a possession. A small seed of
            # error-meta words (structural, RAVANA-expandable), not a per-topic
            # answer table; the real disclosure object still passes through.
            if len(_o.split()) == 1 and _o in (
                "muddled", "confused", "mistaken", "tangled", "muddled",
                "flustered", "garbled", "befuddled"):
                return False
            # R5 fix (round 2026-08-11T0521Z): reject body-part / sensation /
            # feeling-word objects. "i felt it in my chest for days" resolved
            # to the single word "chest" (a body part) and was stored as
            # ('i','does','felt chest') — an experiential/affective detail, NOT
            # a possession or activity. Same class as "felt cold bite" /
            # "broke ice": the verb is a sensation verb and the object is a
            # body/sensation word, so it is an inner state, not a thing the
            # user does/keeps. A SEED vocabulary (RAVANA-expandable via
            # learn_sensation; the real possession object still passes through
            # because it is a noun like "pigeons"/"loft"/"banjo"). Not a
            # per-topic answer table.
            _SENSATION_BODY = (
                "chest", "heart", "stomach", "head", "skin", "bone", "hand",
                "arm", "leg", "eye", "ear", "lung", "brain", "back", "shoulder",
                "spine", "knee", "foot", "finger", "toe", "face", "throat",
                "bite", "burn", "chill", "cold", "heat", "pain", "ache",
                "shiver", "sweat", "tear", "tears", "breath", "pulse", "blood",
                "sigh", "lump", "swelling", "cramp", "tingle", "numb",
            )
            # R5 (round 2026-08-11T0521Z): scope the body/sensation gate so it
            # ONLY rejects a SENSATION PHRASE — an object whose content words are
            # ALL body/sensation words (e.g. bare "chest", "felt chest",
            # "felt cold bite", "broke ice"). This is the inner-state / affective
            # detail the R5 round was created to drop. It must NOT drop a real noun
            # phrase that merely CONTAINS a body word alongside a real noun
            # ("hand planes", "foot cream", "chest freezer") — those carry a real
            # possession/activity head and pass through (verified by probe). The
            # earlier broad form (`len<=2 and any body word`) wrongly rejected
            # real 2-word objects like "build hand planes" and is superseded here.
            _words = _o.split()
            if _words and all(w in _SENSATION_BODY for w in _words):
                return False
            # D4 fix (round 2026-08-11T1328Z): the activity/event miner must
            # capture REAL possessions / lived actions, NOT the user's own
            # META-DISCOURSE or inner-state reporting. "i keep saying it",
            # "i felt a kind of weight lift", "i told a friend", "i lost
            # track of whether" matched an activity verb and stored junk
            # ('does'/'event' = "keep saying" / "felt kind" / "told friend
            # drowned" / "lose track") that then pollutes recall and profile
            # summaries. These are speech-act / abstract-state objects, not
            # things the user possesses or does. Reject the capture when the
            # object HEAD is a communication/meta verb (saying/telling/told/
            # mention), a self-reference pronoun, or an abstract-state noun
            # (kind/weight/hush/lift/track/sort/type/notion/feeling) — the
            # words a possession/activity head would never be. This is a SEED
            # vocabulary (RAVANA-expandable, shared with the empathy/affect
            # lexicon); a real possession noun ("pigeons"/"loft"/"banjo"/
            # "ginger beer") is never in this set and still passes. Not a
            # per-topic answer table.
            _META_DISCOURSE = (
                "saying", "says", "said", "tell", "tells", "telling", "told",
                "mention", "mentions", "mentioning", "recall", "recount",
                "repeat", "repeated", "keep", "kept", "lose", "lost",
                "kind", "weight", "hush", "lift", "track", "sort", "type",
                "notion", "feeling", "feels", "felt", "bit", "thing",
                "stuff", "business", "matter", "point",
            )
            if _head in _META_DISCOURSE:
                return False
            # R5 fix (round 2026-08-11T0521Z): do NOT reject on word-count
            # alone. The earlier "<2 reject" + "<=5 cap" dropped legitimate real
            # disclosures (single-noun possessions like "jar", and 6-7-word
            # activity/event objects like "throw pots at a community studio").
            # Reject only on CONTENT grounds (sensation/body/particle/error-meta
            # words handled by the guards above), never on length. A single real
            # noun ("jar") and a long real noun phrase ("repeated the juniper
            # this spring and found a root") must BOTH pass; the R5 intent (drop
            # inner-state "felt chest") is preserved by the all-words sensation
            # gate above, not a length cap.
            return bool(_obj)
        for _am in _act_pat.finditer(q_clean):
            _verb = _am.group(1).lower()
            if not _activity_verb_ok(_verb):
                continue
            # retraction cue ("i take back ...") is not an activity
            if _verb in ("take", "took", "taking") and "back" in _am.group(2).lower():
                continue
            _obj = self._opinion_topic(_am.group(2).strip().lower())
            _obj = _strip_obj_framers(_obj)
            if _obj and 1 <= len(_obj.split()) <= 5:
                _put_fact("does", f"{_verb} {_obj}", 0.55)
        # Experience / event capture: first-person "i <event-verb> <object>"
        # describing something that happened to the user's world. Captured
        # under attr "event" so it is recallable as a lived experience (not
        # conflated with ongoing activity). Same clause-boundary + content-head
        # rules as the activity capture above.
        _evt_pat = re.compile(
            r"\bi\s+(?:" + _FRAMER_SKIP + r")?"
            r"(?:have\s+|has\s+|had\s+)?(?:almost\s+|nearly\s+)?"
            r"(" + "|".join(_EVENT_VERBS) + r")(?:s|es|ing|ed|[a-z]ed|[a-z]d)?"
            r"\s+(?:my\s+|a\s+|an\s+|the\s+|some\s+|two\s+|three\s+|four\s+|"
            r"five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)?"
            r"(.+?)(?:\s*(?:\.|\!|\?|,|-{1,3}|$|"
            r"\s+and\s+|\s+but\s+|\s+because\s+|\s+so\s+|\s+which\s+|"
            r"\s+that\s+|\s+when\s+|\s+where\s+|\s+while\s+))",
            re.IGNORECASE)
        for _em in _evt_pat.finditer(q_clean):
            _verb = _em.group(1).lower()
            if not _activity_verb_ok(_verb):
                continue
            # retraction cue ("i took back ...") is not an event
            if _verb in ("take", "took", "taking") and "back" in _em.group(2).lower():
                continue
            _obj = self._opinion_topic(_em.group(2).strip().lower())
            _obj = _strip_obj_framers(_obj)
            if _obj and 1 <= len(_obj.split()) <= 5:
                _put_fact("does", f"{_verb} {_obj}", 0.5)

        # Round 2026-08-14T0608Z: TEMPORAL / DATE-GROUNDED fact mining.
        # A first-person disclosure that anchors an activity to a POINT IN TIME
        # ("i've been building frames since 2019", "i started keeping quail in
        # 2021", "i picked up the cello when i was nine") must land in the
        # personal-fact store so a later DATE-GROUNDED recall ("when did i start
        # building frames", "since what year have i kept quail") can answer from
        # the structured store instead of dumping an unrelated episodic turn.
        # Prior rounds confirmed this was a genuine gap: the hippocampal buffer
        # captured 0 dated facts, so date recall returned empty.
        #
        # DESIGN (per round hardcoding + seed-vs-hardcoding rules):
        #  - The value is the resolved CONTENT HEAD of the activity phrase (via
        #    _opinion_topic, which drops closed-class words), so the stored
        #    value is a real concept ("building frames", "keeping quail"),
        #    never a function word. Same mechanism as the 'does' miner.
        #  - The year is a NORMALIZED integer captured from the disclosure, not
        #    an authored answer. The current-year anchor (datetime.now().year)
        #    is computed at mine time and is NOT a frozen literal — it is
        #    derivable and self-updates each run. No retraining.
        #  - The relative-duration forms ("for eleven years", "twenty years
        #    now") are resolved to a START YEAR by subtraction from the anchor,
        #    so "i've repaired tube amps for eleven years" -> since <year>.
        #  - Stored under a NEW attribute "since" keyed by the activity content
        #    head, so date recall is a precise reverse-lookup (query noun ->
        #    stored activity), not a per-topic table. RAVANA can correct any
        #    such fact; nothing is frozen.
        #  - Seed structures only (a month-name map + a small relative-tense
        #    map + a year-format regex). All RAVANA-expandable; removing an
        #    entry degrades gracefully (one fewer date form captured).
        import datetime as _dt
        _THIS_YEAR = _dt.datetime.now().year
        _MONTHS = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
            "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
            "november": 11, "december": 12,
        }
        _NUMWORDS_YEAR = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "twenty": 20,
        }

        def _year_from_text(_yt: str):
            """Extract a 4-digit year, a 2-digit 'YY, or a spelled year."""
            _ym = re.search(r"\b(19|20)\d{2}\b", _yt)
            if _ym:
                return int(_ym.group(0))
            _yn = re.search(r"\b(\d{2})\b", _yt)
            if _yn:
                _yy = int(_yn.group(1))
                return 2000 + _yy if _yy < 70 else 1900 + _yy
            for _w, _n in _NUMWORDS_YEAR.items():
                if re.search(r"\b" + _w + r"\b", _yt):
                    return None  # spelled cardinal alone is not a year
            return None

        # (a) explicit "since <YEAR>" / "in <YEAR>" / "back in <YEAR>" anchors.
        #    Strategy: locate the YEAR token first, then scan the clause that
        #    PRECEDES it (same sentence) for an activity verb. This is robust to
        #    English contractions ("i've been", "i'm"), word order, and the
        #    leading framer words the verb-frame deny set handles elsewhere.
        for _ym in re.finditer(
                r"\b(?:since|in|back\s+in|during)\s+((?:19|20)\d{2}|\d{2})\b",
                q_clean, re.IGNORECASE):
            _yr = _year_from_text(_ym.group(1))
            if _yr is None or _yr < 1900 or _yr > _THIS_YEAR:
                continue
            _clause = q_clean[:_ym.start()].rsplit(".", 1)[-1].rsplit(
                "!", 1)[-1].rsplit("?", 1)[-1].rsplit(",", 1)[-1]
            _av = re.search(
                r"\b(building|build|keeping|keep|repair|repairing|fix|fixing|"
                r"play|playing|picked\s+up|took\s+up|got\s+into|move|moved|"
                r"study|studying|learn|learning|brew|brewing|raise|raising|"
                r"garden|gardening|start|starting|began|begin|write|writing|"
                r"read|reading|run|running|teach|teaching|cook|cooking|"
                r"craft|crafting)\b", _clause, re.IGNORECASE)
            if not _av:
                continue
            _act = self._opinion_topic(_av.group(1).lower())
            if not _act:
                continue
            _act = self._verb_stem(_act)
            _put_fact("since", f"{_act} {_yr}", 0.7)
        # (b) relative duration "for <N> years" / "<N> years now" / "<N> years ago"
        for _rm in re.finditer(
                r"\b(?:for|about|over|nearly|almost)\s+"
                r"((?:one|two|three|four|five|six|seven|eight|nine|ten|"
                r"eleven|twelve|\d+)\s+years?)\b"
                r"[^.!?]{0,20}?\b(?:now|ago|since|already|straight)?\b",
                q_clean, re.IGNORECASE):
            _span = _rm.group(1).lower()
            _nm = re.search(r"\b(\d+)\b", _span)
            if _nm:
                _n = int(_nm.group(1))
            else:
                _nw = re.match(r"([a-z]+)", _span)
                _n = _NUMWORDS_YEAR.get(_nw.group(1), 0) if _nw else 0
            if _n <= 0 or _n > 200:
                continue
            _since = _THIS_YEAR - _n
            # find the activity the duration attaches to: the nearest verb
            # phrase before the duration marker (the activity is stated in the
            # same clause, e.g. "i've repaired tube amps for eleven years")
            _pre = q_clean[:_rm.start()]
            _av = re.findall(
                r"\b(building|build|built|keeping|keep|kept|repair|repairing|"
                r"repaired|fix|fixing|fixed|play|playing|played|picked\s+up|"
                r"took\s+up|got\s+into|move|moved|study|studying|studied|"
                r"learn|learning|learned|brew|brewing|brewed|raise|raising|"
                r"raised|garden|gardening|gardened|write|writing|wrote|read|"
                r"reading|ran|run|running|teach|teaching|taught|cook|cooking|"
                r"cooked|craft|crafting|crafted)\b", _pre, re.IGNORECASE)
            if not _av:
                continue
            _act = self._opinion_topic(_av[-1].lower())
            if not _act:
                continue
            _act = self._verb_stem(_act)
            _put_fact("since", f"{_act} {_since}", 0.6)
        # (c) "when i was <AGE>" / "since i was <AGE>" age-anchored start.
        #     Age may be a digit ("when i was 9") or a spelled number up to
        #     twenty ("when i was nine") — both are handled via the same
        #     number-word map the year resolver uses. Stored as since_age so a
        #     later "how long since you picked up the cello" can render
        #     "since you were about <age>".
        _AGE_WORDS = _NUMWORDS_YEAR  # 1..20 spelled map (reused, general)
        for _am in re.finditer(
                r"\b(?:when|since)\s+i(?:'ve|'m|'s|'d)?\s+was\s+(?:about\s+|"
                r"around\s+)?(?:(\d{1,2})|([a-z]+))\b", q_clean, re.IGNORECASE):
            _age = None
            if _am.group(1):
                _age = int(_am.group(1))
            elif _am.group(2):
                _age = _AGE_WORDS.get(_am.group(2).lower())
            if _age is None or _age < 1 or _age > 120:
                continue
            # The activity may appear EITHER before the age clause
            # ("i picked up the cello when i was nine") OR after it
            # ("since i was nine i've played cello"). Scan the whole sentence
            # the age sits in, both sides of the age token.
            _clause = q_clean[max(0, _am.start() - 60):_am.end() + 60]
            _av = re.search(
                r"\b(pick\s+up|picked\s+up|took\s+up|got\s+into|start|started|"
                r"began|begin|learn|learned|learning|play|playing|study|"
                r"studying|write|writing|read|reading|run|running|brew|brewing|"
                r"raise|raising|keep|kept|build|building|repair|repairing|"
                r"fix|fixing|cook|cooking|craft|crafting|garden|gardening|"
                r"move|moved|teach|teaching)\b", _clause, re.IGNORECASE)
            if not _av:
                continue
            _act = self._opinion_topic(_av.group(1).lower())
            if not _act:
                continue
            _act = self._verb_stem(_act)
            _put_fact("since_age", f"{_act} {_age}", 0.5)
        # (d) APPROXIMATE / HUMAN-PHRASED durations. Real speech rarely says
        #     "for eleven years" — it says "for a decade" / "a few years now" /
        #     "several years" / "two decades" / "many years". Block (b) only
        #     captured DIGIT or spelled 1-12 durations, so these landed in NO
        #     dated fact and date recall returned empty for them (a genuine
        #     residual from the 2026-08-14T0608Z round). This block reuses the
        #     EXACT same 'since' attribute + activity-attachment logic as (b);
        #     the existing recall resolver (engine.py 1f) answers date queries
        #     for them FOR FREE — no recall change — which proves this is a
        #     generalizable capability, not a per-phrase hack. The fuzzy map is
        #     SEED vocabulary (RAVANA-expandable: adding "a fortnight" -> 14
        #     degrades gracefully if absent); the resolved year is derivable
        #     (_THIS_YEAR - n) and self-updates. No retraining. The activity
        #     verb vocabulary mirrors block (b) exactly so mined facts stay
        #     recallable through the same resolver.
        _FUZZY_DUR = {
            "a decade": 10, "two decades": 20, "three decades": 30,
            "a couple of years": 2, "a couple years": 2,
            "a few years": 3, "few years": 3,
            "several years": 4, "a handful of years": 5,
            "many years": 15,
        }
        _used_spans = set()
        for _phrase, _n in _FUZZY_DUR.items():
            if _n <= 0 or _n > 200:
                continue
            for _dm in re.finditer(
                    r"\b(?:for\s+|about\s+|over\s+|nearly\s+|almost\s+)?"
                    + re.escape(_phrase) + r"\b", q_clean, re.IGNORECASE):
                # skip spans overlapping an already-processed fuzzy match
                # (e.g. "a few years" must not also fire the "few years" entry)
                if _used_spans & set(range(_dm.start(), _dm.end())):
                    continue
                _used_spans |= set(range(_dm.start(), _dm.end()))
                _since = _THIS_YEAR - _n
                # attach to the nearest activity verb before the phrase (same
                # clause, e.g. "i've been brewing beer for a decade"); reuse
                # block (b)'s verb vocabulary so the fact is recallable.
                _pre = q_clean[:_dm.start()]
                _av = re.findall(
                    r"\b(building|build|built|keeping|keep|kept|repair|repairing|"
                    r"repaired|fix|fixing|fixed|play|playing|played|picked\s+up|"
                    r"took\s+up|got\s+into|move|moved|study|studying|studied|"
                    r"learn|learning|learned|brew|brewing|brewed|raise|raising|"
                    r"raised|garden|gardening|gardened|write|writing|wrote|read|"
                    r"reading|ran|run|running|teach|teaching|taught|cook|cooking|"
                    r"cooked|craft|crafting|crafted)\b", _pre, re.IGNORECASE)
                if not _av:
                    continue
                _act = self._opinion_topic(_av[-1].lower())
                if not _act:
                    continue
                _act = self._verb_stem(_act)
                _put_fact("since", f"{_act} {_since}", 0.6)

        # Round 2026-08-14T0608Z: TEMPORAL / DATE-GROUNDED fact mining.
        # A first-person disclosure that anchors an activity to a POINT IN TIME
        # ("i've been building frames since 2019", "i started keeping quail in
        # 2021", "i picked up the cello when i was nine") must land in the
        # personal-fact store so a later DATE-GROUNDED recall ("when did i start
        # building frames", "since what year have i kept quail") can answer from
        # the structured store instead of dumping an unrelated episodic turn.
        # Prior rounds confirmed this was a genuine gap: the hippocampal buffer
        # captured 0 dated facts, so date recall returned empty.
        #
        # DESIGN (per round hardcoding + seed-vs-hardcoding rules):
        #  - The value is the resolved CONTENT HEAD of the activity phrase (via
        #    _opinion_topic, which drops closed-class words), so the stored
        #    value is a real concept ("building frames", "keeping quail"),
        #    never a function word. Same mechanism as the 'does' miner.
        #  - The year is a NORMALIZED integer captured from the disclosure, not
        #    an authored answer. The current-year anchor (datetime.now().year)
        #    is computed at mine time and is NOT a frozen literal — it is
        #    derivable and self-updates each run. No retraining.
        #  - The relative-duration forms ("for eleven years", "twenty years
        #    now") are resolved to a START YEAR by subtraction from the anchor,
        #    so "i've repaired tube amps for eleven years" -> since <year>.
        #  - Stored under a NEW attribute "since" keyed by the activity content
        #    head, so date recall is a precise reverse-lookup (query noun ->
        #    stored activity), not a per-topic table. RAVANA can correct any
        #    such fact; nothing is frozen.
        #  - Seed structures only (a month-name map + a small relative-tense
        #    map + a year-format regex). All RAVANA-expandable; removing an
        #    entry degrades gracefully (one fewer date form captured).
        import datetime as _dt
        _THIS_YEAR = _dt.datetime.now().year
        _MONTHS = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
            "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
            "november": 11, "december": 12,
        }
        _NUMWORDS_YEAR = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "twenty": 20,
        }

        def _year_from_text(_yt: str):
            """Extract a 4-digit year, a 2-digit 'YY, or a spelled year."""
            _ym = re.search(r"\b(19|20)\d{2}\b", _yt)
            if _ym:
                return int(_ym.group(0))
            _yn = re.search(r"\b(\d{2})\b", _yt)
            if _yn:
                _yy = int(_yn.group(1))
                return 2000 + _yy if _yy < 70 else 1900 + _yy
            for _w, _n in _NUMWORDS_YEAR.items():
                if re.search(r"\b" + _w + r"\b", _yt):
                    return None  # spelled cardinal alone is not a year
            return None

        # (a) explicit "since <YEAR>" / "in <YEAR>" / "back in <YEAR>" anchors.
        #    Strategy: locate the YEAR token first, then scan the clause that
        #    PRECEDES it (same sentence) for an activity verb. This is robust to
        #    English contractions ("i've been", "i'm"), word order, and the
        #    leading framer words the verb-frame deny set handles elsewhere.
        #    Round 2026-08-15T0326Z GENERALIZE: the old code matched a FROZEN
        #    verb allowlist (building|keep|repair|...). A rotated persona verb
        #    outside that list ("i've kept the light since 2019" -> "kept" is
        #    past-tense and was absent) was silently dropped, so the date fact
        #    never landed and "what year did i start the light" failed. Fix:
        #    extract the FIRST verb in the clause STRUCTURALLY (any \w+ that is
        #    not closed-class), then accept it iff it is a legitimate activity
        #    verb (_activity_verb_ok) — reusing the SAME discriminator the D3
        #    'does' miner uses, so the captured fact stays recallable through
        #    the identical 'since' resolver. No per-topic verb list to exhaust.
        for _ym in re.finditer(
                r"\b(?:since|in|back\s+in|during)\s+((?:19|20)\d{2}|\d{2})\b",
                q_clean, re.IGNORECASE):
            _yr = _year_from_text(_ym.group(1))
            if _yr is None or _yr < 1900 or _yr > _THIS_YEAR:
                continue
            _clause = q_clean[:_ym.start()].rsplit(".", 1)[-1].rsplit(
                "!", 1)[-1].rsplit("?", 1)[-1].rsplit(",", 1)[-1]
            # Round 2026-08-15T0326Z GENERALIZE: take the FIRST legitimate
            # activity verb in the clause, skipping aspectual/framer verbs
            # ("i've BEEN building" -> build; "i STARTED keeping" -> keep). A
            # verb precedes its object in English, so the FIRST activity-ok
            # verb (after dropping aspectuals) is the activity head — not the
            # auxiliary ("been"/"started") that led, and not a trailing noun
            # ("tube amps") that happens to pass _activity_verb_ok. Structural
            # (no frozen verb list); reuses _activity_verb_ok so rotated-persona
            # verbs (kept/raced/...) still land. Restored the content-head
            # behaviour the old miner had (regression vs
            # test_round_2026_08_14T0608_temporal when the first raw word was
            # taken).
            _verbs = [v.lower() for v in re.findall(
                r"\b([a-z][a-z']+)\b", _clause, re.IGNORECASE)]
            _verb = None
            _vidx = -1
            for _i, v in enumerate(_verbs):
                if v in _ASPECTUAL_VERBS:
                    continue
                if _activity_verb_ok(v):
                    _verb = v
                    _vidx = _i
                    break
            if _verb is None:
                continue
            _act = self._verb_stem(_verb)
            if not _act:
                continue
            # Capture the activity OBJECT so two activities that share a verb
            # head but differ by object ('building frames' vs 'building
            # cabinets') store DISTINCT facts and a later DATE-GROUNDED query
            # ('when did i start building frames') can pick the right one. The
            # object is sliced STRUCTURALLY (closed-class / time-adjunct words
            # close the span), so verb-only disclosures ('i've been restoring
            # since 2018') store the bare 'restore 2018' shape unchanged.
            _obj = _activity_object(_verbs, _vidx)
            _act_full = f"{_act} {_obj}".strip() if _obj else _act
            _put_fact("since", f"{_act_full} {_yr}", 0.7)
        # (b) relative duration "for <N> years" / "<N> years now" / "<N> years ago"
        for _rm in re.finditer(
                r"\b(?:for|about|over|nearly|almost)\s+"
                r"((?:one|two|three|four|five|six|seven|eight|nine|ten|"
                r"eleven|twelve|\d+)\s+years?)\b"
                r"[^.!?]{0,20}?\b(?:now|ago|since|already|straight)?\b",
                q_clean, re.IGNORECASE):
            _span = _rm.group(1).lower()
            _nm = re.search(r"\b(\d+)\b", _span)
            if _nm:
                _n = int(_nm.group(1))
            else:
                _nw = re.match(r"([a-z]+)", _span)
                _n = _NUMWORDS_YEAR.get(_nw.group(1), 0) if _nw else 0
            if _n <= 0 or _n > 200:
                continue
            _since = _THIS_YEAR - _n
            # find the activity the duration attaches to: the nearest verb
            # phrase before the duration marker (the activity is stated in the
            # same clause, e.g. "i've repaired tube amps for eleven years").
            # Round 2026-08-15T0326Z GENERALIZE: extract the activity verb
            # STRUCTURALLY (any word that passes _activity_verb_ok), skipping
            # aspectual/framer verbs, and take the FIRST such verb — a verb
            # precedes its object, so the first activity-ok verb is the head
            # ("i've REPAIRED tube amps" -> repair), not a trailing noun
            # ("amps") that merely passes _activity_verb_ok. Replaces the
            # frozen verb allowlist so rotated-persona activities land too.
            _pre = q_clean[:_rm.start()]
            _av = re.findall(
                r"\b([a-z][a-z']+)\b", _pre, re.IGNORECASE)
            _verb = None
            _vidx = -1
            for _i, _v in enumerate(_av):
                _vl = _v.lower()
                if _vl in _ASPECTUAL_VERBS:
                    continue
                if _activity_verb_ok(_vl):
                    _verb = _vl
                    _vidx = _i
                    break
            if _verb is None:
                continue
            _act = self._verb_stem(_verb)
            if not _act:
                continue
            # Capture the activity OBJECT (mirrors block (a)) so overlapping
            # verb heads differ by object are stored as distinct dated facts
            # and disambiguated at recall.
            _obj = _activity_object(_av, _vidx)
            _act_full = f"{_act} {_obj}".strip() if _obj else _act
            _put_fact("since", f"{_act_full} {_since}", 0.6)
        # (c) "when i was <AGE>" / "since i was <AGE>" age-anchored start.
        #     Age may be a digit ("when i was 9") or a spelled number up to
        #     twenty ("when i was nine") — both are handled via the same
        #     number-word map the year resolver uses. Stored as since_age so a
        #     later "how long since you picked up the cello" can render
        #     "since you were about <age>".
        _AGE_WORDS = _NUMWORDS_YEAR  # 1..20 spelled map (reused, general)
        for _am in re.finditer(
                r"\b(?:when|since)\s+i(?:'ve|'m|'s|'d)?\s+was\s+(?:about\s+|"
                r"around\s+)?(?:(\d{1,2})|([a-z]+))\b", q_clean, re.IGNORECASE):
            _age = None
            if _am.group(1):
                _age = int(_am.group(1))
            elif _am.group(2):
                _age = _AGE_WORDS.get(_am.group(2).lower())
            if _age is None or _age < 1 or _age > 120:
                continue
            # The activity may appear EITHER before the age clause
            # ("i picked up the cello when i was nine") OR after it
            # ("since i was nine i've played cello"). Scan the whole sentence
            # the age sits in, both sides of the age token.
            # Round 2026-08-15T0326Z GENERALIZE: structural first-verb
            # extraction validated by _activity_verb_ok, replacing the frozen
            # verb allowlist so any activity verb (e.g. a rotated-persona
            # verb) anchors the age fact.
            _clause = q_clean[max(0, _am.start() - 60):_am.end() + 60]
            # Round 2026-08-15T0326Z GENERALIZE: take the FIRST legitimate
            # activity verb, skipping aspectual/framer verbs ("i", "was",
            # "been", ...), and append a trailing particle ("up"/"on"/...) so
            # a phrasal verb ("picked UP the cello") stores "pick up" — not
            # just "pick". Structural; reuses _activity_verb_ok + _ASPECTUAL_VERBS
            # + _PARTICLES so rotated-persona verbs land too (replaces the
            # frozen verb allowlist that missed inflected/phrasal forms).
            _toks = re.findall(r"\b([a-z][a-z']+)\b", _clause, re.IGNORECASE)
            _verb = None
            _vidx = -1
            for _i, _tk in enumerate(_toks):
                _tl = _tk.lower()
                if _tl in _ASPECTUAL_VERBS:
                    continue
                if _activity_verb_ok(_tl):
                    _verb = _tl
                    _vidx = _i
                    break
            if _verb is None:
                continue
            # phrasal: include a following particle ("pick up", "took up")
            if _vidx + 1 < len(_toks) and _toks[_vidx + 1].lower() in _PARTICLES:
                _verb = f"{_verb} {_toks[_vidx + 1].lower()}"
            _act = self._verb_stem(_verb)
            if not _act:
                continue
            # Capture the activity OBJECT (mirrors blocks (a)/(b)) so a later
            # age-anchored recall can disambiguate by object when two verb heads
            # overlap.
            _obj = _activity_object(_toks, _vidx)
            _act_full = f"{_act} {_obj}".strip() if _obj else _act
            _put_fact("since_age", f"{_act_full} {_age}", 0.6)
        # (d) APPROXIMATE / HUMAN-PHRASED durations. Real speech rarely says
        #     "for eleven years" — it says "for a decade" / "a few years now" /
        #     "several years" / "two decades" / "many years". Block (b) only
        #     captured DIGIT or spelled 1-12 durations, so these landed in NO
        #     dated fact and date recall returned empty for them (a genuine
        #     residual from the 2026-08-14T0608Z round). This block reuses the
        #     EXACT same 'since' attribute + activity-attachment logic as (b);
        #     the existing recall resolver (engine.py 1f) answers date queries
        #     for them FOR FREE — no recall change — which proves this is a
        #     generalizable capability, not a per-phrase hack. The fuzzy map is
        #     SEED vocabulary (RAVANA-expandable: adding "a fortnight" -> 14
        #     degrades gracefully if absent); the resolved year is derivable
        #     (_THIS_YEAR - n) and self-updates. No retraining. The activity
        #     verb vocabulary mirrors block (b) exactly so mined facts stay
        #     recallable through the same resolver.
        _FUZZY_DUR = {
            "a decade": 10, "two decades": 20, "three decades": 30,
            "a couple of years": 2, "a couple years": 2,
            "a few years": 3, "few years": 3,
            "several years": 4, "a handful of years": 5,
            "many years": 15,
        }
        _used_spans = set()
        for _phrase, _n in _FUZZY_DUR.items():
            if _n <= 0 or _n > 200:
                continue
            for _dm in re.finditer(
                    r"\b(?:for\s+|about\s+|over\s+|nearly\s+|almost\s+)?"
                    + re.escape(_phrase) + r"\b", q_clean, re.IGNORECASE):
                # skip spans overlapping an already-processed fuzzy match
                # (e.g. "a few years" must not also fire the "few years" entry)
                if _used_spans & set(range(_dm.start(), _dm.end())):
                    continue
                _used_spans |= set(range(_dm.start(), _dm.end()))
                _since = _THIS_YEAR - _n
                # attach to the nearest activity verb before the phrase (same
                # clause, e.g. "i've been brewing beer for a decade"); reuse
                # block (b)'s structural verb extraction so the fact is
                # recallable and rotated-persona verbs land too.
                _pre = q_clean[:_dm.start()]
                _av = re.findall(
                    r"\b([a-z][a-z']+)\b", _pre, re.IGNORECASE)
                _verb = None
                _vidx = -1
                for _i, _v in enumerate(_av):
                    _vl = _v.lower()
                    if _vl in _ASPECTUAL_VERBS:
                        continue
                    if _activity_verb_ok(_vl):
                        _verb = _vl
                        _vidx = _i
                        break
                if _verb is None:
                    continue
                _act = self._verb_stem(_verb)
                if not _act:
                    continue
                # Capture the activity OBJECT (mirrors blocks (a)/(b)/(c)) so
                # fuzzy-duration facts also disambiguate by object at recall.
                _obj = _activity_object(_av, _vidx)
                _act_full = f"{_act} {_obj}".strip() if _obj else _act
                _put_fact("since", f"{_act_full} {_since}", 0.6)

        # Opinion mining (C2): capture the user's value judgments alongside
        # facts. Runs in the miner (not only observe_user_query) so opinions are
        # captured even when process_turn early-returns before Step 5b (e.g. a
        # bare "i really like cats" hits a preference handler). Polarity from
        # explicit cues + VAD signal already inferred for this turn.
        _vad = self._infer_user_emotion(text)
        _v, _a, _d = _vad
        # A first-person opinion can ONLY be mined from a DECLARATIVE
        # self-report, never from a question ("do you think i love X?" is the
        # user asking about RAVANA's stance, not stating their own). Mining a
        # question creates garbage stances keyed on trailing clause fragments
        # (e.g. "letterpress given" from "do you still think i love letterpress
        # given the wrist thing?"). Structural guard: any interrogative is
        # skipped before the stance loop.
        _is_question = (q_clean.rstrip().endswith("?")
                        or bool(re.match(
                            r"^(what|who|when|where|why|how|which|is|are|do|"
                            r"does|did|can|could|would|should|will|may|might|"
                            r"am|have|has|had)\b", q_clean)))
        if _is_question:
            return
        for _pat, _pol, _conf in (
            # D2 (round v2): capture the FULL object phrase after the verb,
            # not just the first token — "small talk" / "the solitude of the
            # lighthouse" must be one topic, not the function word "small"/"the".
            # The salient CONTENT HEAD is resolved by _opinion_topic (skips
            # closed-class words), so a stance lands on "talk"/"solitude", a
            # real concept, never on "the"/"how"/"small".
            (r"\bi\s+(?:really\s+)?(?:like|love|enjoy|prefer|adore|care\s+for)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", 0.8, 0.6),
            (r"\bi\s+(?:really\s+)?(?:hate|dislike|detest|can't\s+stand)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", -0.8, 0.6),
            # FIX (round v-aug06b): word-boundary-guarded sentiment adjectives.
            # Without \b, "bad" matched the prefix of "badly" ("is badly
            # underrated" -> parsed as "is bad"), inverting a POSITIVE
            # endorsement into a negative stance. "underrated" is a positive
            # endorsement (deserves more recognition) and is now recognized as
            # such; an optional \w+ly adverb slot ("seriously/truly/badly
            # underrated") is tolerated so the adverb never blocks the adjective.
            # Generic sentiment-lexicon expansion (no per-topic rule).
            (r"\bi\s+think\s+(.+?)\s+(?:is|are)\s+(?:\w+ly\s+)?(?:good\b|great|awesome|nice|wonderful|amazing|the\s+future|essential|important|right\b|crucial|vital|underrated|under-rated|underappreciated|under-valued|undervalued)", 0.8, 0.5),
            (r"\bi\s+think\s+(.+?)\s+(?:is|are)\s+(?:\w+ly\s+)?(?:bad\b|terrible|awful|overrated|over-rated|horrible|poor\b|a\s+mistake|harmful|wrong\b|useless)", -0.8, 0.5),
            (r"\bi\s+believe\s+(.+?)\s+(?:is|are)\s+(?:\w+ly\s+)?(?:good\b|great|awesome|nice|wonderful|amazing|the\s+future|essential|important|right\b|crucial|vital|underrated|under-rated|underappreciated|under-valued|undervalued)", 0.8, 0.5),
            (r"\bi\s+believe\s+(.+?)\s+(?:is|are)\s+(?:\w+ly\s+)?(?:bad\b|terrible|awful|overrated|over-rated|horrible|poor\b|a\s+mistake|harmful|wrong\b|useless)", -0.8, 0.5),
            # "i believe we must/should protect/save/ban <X>" -> positive stance on X.
            (r"\bi\s+believe\s+we\s+(?:must|should)\s+(?:protect|save|preserve|defend|fund|support)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", 0.8, 0.55),
            (r"\bi\s+believe\s+we\s+(?:must|should)\s+(?:ban|cut|end|stop|reduce)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", -0.8, 0.55),
            (r"\b(.+?)\s+is\s+my\s+favorite\b", 1.0, 0.7),
            (r"\bi\s+believe\s+([\w'-]+)\s+beats\s+([\w'-]+)", 0.7, 0.4),
            # A-fix (round 2026-08-08b): comparative opinions ("small towns make
            # better humans than cities", "tea beats coffee", "the mountains are
            # finer than the coast"). The Winner (X) is the valued term -> a
            # positive stance on X. General, no per-topic rule; the content head
            # is resolved by _opinion_topic so the topic is a real concept
            # (e.g. "small towns"), never a function word. The loser (Y) is NOT
            # force-negatived here — if the user holds a negative view of it
            # they state it, and the same miner captures it; we must not invent
            # a polarity RAVANA could not revise by talking.
            (r"\b(.+?)\s+(?:makes?|are?|is|make|produce[s]?|breed[s]?|build[s]?)\s+"
             r"(?:better|finer|more human|more humane|healthier|stronger|"
             r"happier|wiser|kinder)\s+(?:humans?|people|folk|neighbou?rs?|"
             r"citizens?|communities?)?\s*(?:than|over|versus|vs\.?)\b", 0.7, 0.5),
            (r"\b(.+?)\s+(?:beats|outshines|trumps|wins\s+over|is\s+finer\s+than|"
             r"is\s+better\s+than)\s+(.+?)(?:[.!?]|\band\b|\bbut\b|$|,)", 0.7, 0.5),
            # Round 2026-08-08f: broaden the comparative / superlative /
            # dismissive opinion classes. The prior miner only caught the
            # 'makes better people than' and 'beats' shapes; rich value
            # judgments like 'the sea is a better teacher than any classroom',
            # 'hand-built synths sound warmer than mass-produced', 'graveyards
            # are the most honest libraries', 'the cold water is the only
            # honest part of my day', 'most modern music is just wallpaper',
            # 'the best knots are the ones you can untie in the dark' were NOT
            # captured -> no stance -> no contradiction target -> dead hollow
            # ack. Each class below is a GRAMMATICAL pattern (no per-topic
            # list); the content head is resolved by _opinion_topic so the
            # stance lands on the real concept (e.g. 'sea', 'hand-built
            # synths', 'graveyards', 'cold water', 'modern music'). Polarity
            # is lexical: comparatives/superlatives/dismissals are inherently
            # valenced. RAVANA can still revise any stance by talking (the
            # store merges on new input); nothing is frozen or retrained.
            # (a) comparative copula 'X is a better/safer/finer/honest-er Y
            #     than Z' -> positive stance on X. The leading 'i think/
            #     i believe' frame is stripped so the captured subject is the
            #     real content head (e.g. 'sea'), never 'i think the sea'.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:is|are)\s+(?:a|an)?\s*(?:better|safer|finer|kinder|"
             r"wiser|healthier|stronger|truer|freer|calmer|cleaner|warmer|"
             r"cooler|sharper|kinder|more honest|more human|more real|"
             r"more true|more free)\b"
             r"(?:\s+(?:teacher|thing|place|way|part|kind|sort|type|version|"
             r"form|bit|lot|deal))?\s+(?:than|over|versus|vs\.?\b)", 0.7, 0.55),
            # (b) sensory-comparative 'X sounds/feels/tastes/looks/reads WARMER
            #     than Y' -> positive stance on X (the WARMER-ER class). Strip
            #     a leading 'i think/believe' frame the same way.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:sounds|sound|feels|feel|tastes|taste|looks|look|reads|read|"
             r"seems|seem|comes|come|comes\s+across|comes\s+off)\s*"
             r"(?:more|much)?\s*(?:warmer|cooler|truer|cleaner|honest|realer|"
             r"more honest|more real|more true|more alive|more human)\b"
             r"(?:\s+(?:than|over|versus|vs\.?))", 0.7, 0.55),
            # (c) superlative 'X is the most Y' / 'X is the best Y' /
            #     'the only Y' -> positive stance on X. Strip the leading
            #     'i think/believe' frame so the subject is the content head.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:is|are)\s+the\s+(?:most|best|finest|truest|honest|real|"
             r"purest|clearest|only)\b", 0.75, 0.6),
            # (d) 'X is just wallpaper/noise/fluff/...' -> negative dismissive
            #     stance on X (the subject is being demoted to negligible).
            #     Structural: the dismissive-metaphor noun set is SEED
            #     vocabulary (the same kind of small lexicon the sentiment
            #     adjectives use), not a per-topic table; RAVANA can extend it
            #     at runtime. Strip the leading frame.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:is|are)\s+(?:just|merely|basically|really\s+just)\s*"
             r"(?:wallpaper|noise|fluff|decoration|decorative|background|filler|"
             r"branding|spin|hype|fad|nonsense|garbage|junk|pap|slop|trash)"
             r"(?:\b|[.!?])", -0.7, 0.55),
            # (e) 'X is the best kind of Y' -> positive stance on X (X is the
            #     prized member of category Y). Strip the leading frame.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:is|are)\s+the\s+best\b"
             r"(?:\s+(?:kind|sort|type|breed|example|form|version|bit))?", 0.7, 0.55),
            # ── Generalized opinion shapes (D-C, round 2026-08-13T0634Z) ──
            # The classes above only catch a handful of grammatical shapes
            # ("i think X is good", "X beats Y", "X is better than Y",
            # "i like/hate X"). Natural opinions fell straight through:
            #   "i think bookshops are worth more to a city than chains"
            #   "i'm strongly against people who talk in theatres"
            #   "teaching kids to cook is more important than coding"
            # produced NO stance -> contradiction queries had nothing to cite.
            # Fix: broaden with general GRAMMATICAL shapes backed by a small
            # SENTIMENT LEXICON (seed vocabulary, expandable at runtime via
            # the affect store — not a per-topic table). The content head is
            # still resolved by _opinion_topic, so the stance lands on the
            # real concept. Polarity is lexical (the lexicon is inherently
            # valenced); RAVANA can still revise any stance by talking.
            # (f) explicit for/against stance. Positive keywords (for/in favor/
            #     in support/pro) -> +0.85; negative (against/opposed to/anti)
            #     -> -0.85. Keywords are split so a single statement cannot
            #     match both lanes.
            (r"\bi(?:'m| am)\s+(?:really\s+|strongly\s+|firmly\s+)?"
             r"(?:in\s+favor\s+of|in\s+support\s+of|for|pro)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", 0.85, 0.6),
            (r"\bi(?:'m| am)\s+(?:really\s+|strongly\s+|firmly\s+)?"
             r"(?:opposed\s+to|against|anti)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", -0.85, 0.6),
            # (g) "i (really) (like|love|hate|...) X" — broaden the verb set
            #     so attitudes like admire/despise/treasure are captured.
            (r"\bi\s+(?:really\s+|truly\s+|deeply\s+)?"
             r"(?:admire|respect|treasure|cherish|value|appreciate)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", 0.8, 0.6),
            (r"\bi\s+(?:really\s+|truly\s+|deeply\s+)?"
             r"(?:despise|loathe|resent|detest|scorn|disdain|abhor)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", -0.8, 0.6),
            # (h) "i think/believe/feel X is (more) <VAL-ADJ> (than Y)" with a
            #     broad value-adjective lexicon (the comparative slot makes it
            #     "X over Y" without needing a literal "than"). Polarity from
            #     the lexicon; the leading frame is stripped so X is the head.
            (r"\bi\s+(?:think|believe|feel|find|reckon)\s+(.+?)\s+(?:is|are)\s+"
             r"(?:much\s+|far\s+|way\s+|more\s+|so\s+)?(?:"
             r"important|valuable|meaningful|useful|helpful|worthwhile|"
             r"significant|relevant|fair|just|honest|beautiful|wonderful|"
             r"vital|crucial|essential|wise|healthy|kind|free|true|real|"
             r"alive|human|warm|clean|brave|strong|necessary|right)\b", 0.75, 0.5),
            (r"\bi\s+(?:think|believe|feel|find|reckon)\s+(.+?)\s+(?:is|are)\s+"
             r"(?:much\s+|far\s+|way\s+|more\s+|so\s+)?(?:"
             r"unimportant|worthless|meaningless|useless|harmful|pointless|"
             r"pointless|trivial|irrelevant|unfair|dishonest|ugly|awful|"
             r"terrible|harmful|wrong|cruel|false|empty|cold|dirty|weak|"
             r"unnecessary|foolish|stupid|dangerous|toxic)\b", -0.75, 0.5),
            # NOTE: a bare comparative "X is more <ADJ> than Y" (no leading
            # "i think" frame) is intentionally NOT added as a separate
            # pattern. The (h) pattern above already covers the user-opinion
            # form ("i believe teaching kids to cook is more important than
            # coding") which is the round's actual target. A standalone bare
            # comparative pattern re-matched mid-string (after "i ") and
            # seeded a second, garbled topic ("believe teaching kids") that
            # collided with the (h) topic — so it was removed. Third-party
            # bare comparatives remain un-mined, which is acceptable: the
            # durable-stance goal is about the USER's own opinions.
        ):
            for _m in re.finditer(_pat, q_clean, re.IGNORECASE):
                _raw = _m.group(_m.lastindex).strip().lower()
                if not _raw:
                    continue
                # Resolve the content head so stances never land on a
                # closed-class word (the/a/how/small/...). Returns None when
                # the phrase has no usable content noun -> skip (don't seed a
                # garbage stance).
                _topic = self._opinion_topic(_raw)
                if not _topic:
                    continue
                # C-fix (round 2026-08-12T0613Z): reject a SINGLE-WORD topic
                # that is not a real attitude object. Comparative / superlative
                # patterns ("analog synthesis beats anything in a laptop" ->
                # topic "anything"; "standing in the lamp room is the most
                # awake i feel" -> topic "standing") capture a non-attitude HEAD
                # that pollutes the stance store with keys RAVANA can never
                # reconcile or recall (measured this round: stances keyed
                # "anything"->0.7, "standing"->0.75, "open ocean"->0.0). The fix
                # rejects a SMALL universal ghost set: indefinite pronouns
                # (anything/something/everything/nothing/...) and grammatical
                # gerunds (standing/being/doing/...) that can NEVER own an
                # attitude. This is seed vocabulary (universal, not
                # topic-specific) — a legitimate single-word stance like
                # "cilantro" or "solitude" passes through. Multiword content
                # phrases (e.g. "open ocean", "acoustic music", "solar punk")
                # ALWAYS pass because they carry genuine content nouns. NOTE: an
                # earlier draft also required the topic to be in the engine's
                # GloVe vocabulary, but that wrongly dropped rare-but-valid
                # single-word attitudes (regression: test_round_aug07_fixes
                # KeyError 'cilantro'), so the vocabulary gate was removed — the
                # ghost set alone is the correct, minimal structural guard.
                if " " not in _topic:
                    if _topic in self._STANCE_GHOST_TOPICS:
                        continue

                # Sign-PRESERVING affect blend (D3 fix, round v-aug06).
                # Bug: the prior code used the RUNNING emotional buffer
                # (_v = EMA of ALL prior turns) to modulate polarity via
                # `max(0.3, _p + 0.2)` — so a clearly-negative lexical cue
                # ("i can't stand cilantro", _pol=-0.8) spoken right after a
                # happy stretch (_v>0.2) was flipped to +0.3, because the
                # buffer leak beat the explicit attitude. That is backwards:
                # the USER's stated verb is the ground-truth attitude signal;
                # affect should only REINFORCE it, never reverse it.
                # Fix: VAD now only strengthens the SAME signed pole (or
                # widens a neutral cue), and can never cross the sign line.
                # This matches vmPFC value integration: the lexical attitude
                # is the delta-rule target; affect is a gain, not a sign.
                _p = float(_pol)
                if _p < -0.05 and _v < -0.1:
                    _p = min(_p, _p - 0.15)          # more negative, same sign
                elif _p > 0.05 and _v > 0.1:
                    _p = max(_p, _p + 0.15)          # more positive, same sign
                # neutral lexical cue (_p==0) may be steered by a strong signal
                elif abs(_p) <= 0.05 and abs(_v) >= 0.3:
                    _p = 0.6 if _v > 0 else -0.6
                _p = max(-1.0, min(1.0, _p))
                self.opinions.express_stance(_topic, polarity=_p, confidence=_conf,
                                            valence=_v, arousal=_a)

        # Stance-reversal mining: "i take back X" / "i changed my mind about X" /
        # "i retract my stance on X" recodes the user's valuation of the topic to
        # the opposite pole (vmPFC re-evaluation), LINKED to the PRIOR stance the
        # store already holds — this is an attitude-change operator, not a fresh
        # opinion. Runs last so it can see (and reverse) any stance just mined.
        self.mine_stance_reversal(text)

    def _stance_key_in_text(self, text: str):
        """Return the held stance key whose TOPIC appears in `text`, else None.

        Generic resolver for retraction/idiom fallbacks: instead of requiring
        the FULL multiword key as a substring (which fails when the user names
        only part of it, e.g. recants "letterpress" but the key is "letterpress
        printing"), it matches when ANY whitespace-delimited token of the key
        appears as a whole word in `text`. This is content-driven (resolves
        against the live stance store) and generalizes to any topic — no
        per-topic table. Prefers the LONGEST key match so a more specific
        stance wins over a generic one.
        """
        if not text:
            return None
        _words = set(re.findall(r"[a-z']+", text.lower()))
        if not _words:
            return None
        _best = None
        _best_score = 0
        for _k in self.opinions.stances:
            if not _k:
                continue
            _ktoks = [t for t in re.findall(r"[a-z']+", _k.lower()) if t]
            if not _ktoks:
                continue
            # A partial recant names only part of a multiword key (e.g.
            # "letterpress" for the stored "letterpress printing"); match when
            # ANY key token appears as a whole word, and prefer the key with
            # the most tokens present (most specific match wins).
            _matched = sum(1 for t in _ktoks if t in _words)
            if _matched == 0:
                continue
            if _matched > _best_score:
                _best_score = _matched
                _best = _k
        return _best

    def mine_stance_reversal(self, text: str) -> None:
        """Detect a stance-reversal/retraction and recode the stored stance.

        Attitude change is a valuation RECODE, not a fresh opinion merge: when
        the user retracts a position ("i take back X", "i changed my mind about
        X"), the previously-held stance is flipped toward the opposite pole.
        Crucially this is LINKED to the stance the store already holds for that
        topic — a benign acknowledgment with no store mutation would leave the
        recorded opinion stale (the "take back not linked" gap).
        """
        q = re.sub(r"\s+", " ", (text or "").lower().strip())
        if not q:
            return
        cue_end = None
        _matched_cue = None
        for pat in _RETRACTION_CUES:
            m = re.search(pat, q)
            if m:
                cue_end = m.end()
                _matched_cue = pat
                break
        # A softening cue ANYWHERE in the utterance governs the whole retraction
        # speech act. "i take it back — but it's not that bad" is a hedged,
        # NON-inverting recant: the user relaxes the stance toward neutral, they
        # do NOT flip to the opposite conviction. The first-match loop above can
        # bind `_matched_cue` to a hard recant ("i take it back") while a
        # softening phrase trails it; if both are present the softening intent
        # wins. Generic: scans the whole utterance for any softening idiom, no
        # per-topic rule.
        _soft = _matched_cue in _SOFTENING_CUES or any(
            re.search(p, q) for p in _SOFTENING_CUES)
        if cue_end is None:
            # Round 2026-08-08f: concession shape. A very common contradiction
            # does NOT use a retraction keyword ("i take back", "i was wrong"):
            # the user concedes a prior stance with "i thought X but Y" /
            # "i used to think X but now Y" / "i told you X but actually Y",
            # where X's topic matches a stance RAVANA already holds and Y
            # contradicts it. The prior code only caught keyword-led retractions,
            # so these fell through to the hollow "got it" ack and the stale
            # stance persisted (the contradiction was silently dropped). Detect
            # the concession structurally: a first-person past/present belief
            # frame ("i thought/i used to think/i told you/i said") followed by
            # a BUT that introduces a contrasting clause. Resolve the conceded
            # topic against the LIVE stance store; if it matches a held stance,
            # reverse/soften it the same way a keyword retraction would. This is
            # grammatical (no per-topic table) and RAVANA can still revise the
            # stance by further talk. A concession is a SOFTENING (the user is
            # walking the stance back, not inverting to a hard opposite
            # conviction), so it relaxes toward neutral, never force-flips.
            _concession = re.search(
                r"\b(?:i\s+(?:thought|used\s+to\s+think|told\s+you|said|believed|felt)"
                r"|i'?m\s+not\s+so\s+sure)\b"
                r".{0,60}?\b(?:but|although|though|yet|actually|however)\b", q)
            if _concession is not None:
                # A concession ("i thought X but Y") walks BACK the belief held
                # in the PRE-connector clause X — X is the topic the user now
                # revokes, while Y is the NEW contrasting preference. Resolving
                # the topic against the WHOLE utterance is wrong: the opinion
                # miner in the same turn also creates a (bogus) stance from the
                # trailing "i prefer Y" clause, so a whole-utterance longest-key
                # match can bind the NEW topic and (a) reverse a stance the user
                # never walked back, and (b) emit a fabricated "you changed your
                # mind about Y" ack for a topic with no prior stance. Restrict
                # resolution to the conceded clause X only. Generic: the clause
                # is derived from the matched connector span, not a per-topic
                # table, and still resolves against the live store.
                _pre_clause = q[:_concession.end()]
                _target = self._stance_key_in_text(_pre_clause)
                if _target is None:
                    # Fallback: the conceded topic may be a multiword key whose
                    # tokens are split across the connector (e.g. "i thought the
                    # sea was X but Y"); still scope to the pre-connector span so
                    # the new preference Y can never be the resolved target.
                    _target = self._stance_key_in_text(q[:_concession.start()])
                if _target is not None:
                    try:
                        self.opinions._soft_reversal = True
                        self.opinions.reverse_stance(_target, utterance=text)
                    except Exception:
                        pass
                    return
            return
        # A retraction cue is either a HARD recant ("i was wrong about X",
        # "i take it all back" — flip decisively) or a SOFTENING ("x isn't that
        # bad, i was too hasty", "i came around a bit" — relax toward neutral,
        # never invert). The softening cues are exactly the phrase set that
        # resolves to a partial reversal; everything else is a hard recant.
        # This drives reverse_stance's blend magnitude, so the same code path
        # produces opposite-hemisphere vs near-neutral recodes from the
        # utterance itself — no per-topic rule, no hardcoding of the topic.
        # The topic lives in the clause after the retraction cue. Strip leading
        # connectors/prepositional frames that carry no content.
        tail = q[cue_end:].strip(" \t.,!?;:'\"\u2014-")
        tail = re.sub(
            r"^(?:what\s+i\s+(?:said|think|thought|meant)"
            r"|my\s+(?:stance|opinion|view)\s+on"
            r"|my\s+mind\s+(?:about|on)"
            r"|i\s+(?:think|thought|believe|felt|was|am)\s+(?:i\s+)?(?:was|were|am|wrong|right|too\s+hasty|mistaken|in\s+error)\s+(?:about|on|there)\s+"
            r"|about|on|regarding|of|to|my\s+opinion\s+on)\s+", "", tail)
        topic = self._opinion_topic(tail)
        # FIX (round v-aug06b): end-anchored retraction idioms ("... olives
        # ... they're not that bad, i was too hasty") leave an EMPTY tail, so
        # the normal topic scan finds nothing. Before giving up, scan the
        # whole utterance for a held stance whose key appears as a content
        # word and reverse THAT. Generic: resolves against the live store,
        # no per-topic table. Only when no topic is resolvable from the tail.
        # FIX (round v-aug06d): match by TOKEN CONTAINMENT, not whole-key
        # substring. The user recants "letterpress" but the stored stance key
        # is "letterpress printing"; the old " letterpress printing " in q
        # check failed because the utterance only contains "letterpress". Now
        # any token of the key appearing as a whole word in q resolves it
        # (generic: works for any multiword stance key, no per-topic rule).
        if not topic:
            _target = self._stance_key_in_text(q)
            if _target is None:
                _whole = self._opinion_topic(q)
                if _whole:
                    _target = self.opinions.resolve_topic(_whole)
            if _target is None:
                return
            try:
                self.opinions._soft_reversal = _soft
                self.opinions.reverse_stance(_target, utterance=text)
            except Exception:
                pass
            return
        # D4 fix (round v-aug06): the retraction's real topic is often buried
        # INSIDE a follow-on clause, not in the head of the tail. E.g.
        # "i take it back — i've grown to hate running" → tail="i've grown to
        # hate running", and _opinion_topic returns "i've grown" (a pronoun +
        # auxiliary), which resolves to NO prior stance, so the contradiction
        # is silently dropped and the stale positive stance persists. Fix:
        # when the head is non-content (pronoun/auxiliary-led) but the tail
        # contains an explicit attitude verb + object ("hate running", "like
        # x"), extract the object of THAT verb as the topic. This reads the
        # attitude actually being retracted, not the filler grammar. Generic,
        # verb-driven (no per-topic table) — generalizes to any retracted
        # attitude the user names.
        if topic.split()[0] in ("ive", "i've", "i", "im", "i'm", "ive", "he",
                                 "she", "they", "we", "you") or topic in (
                "grown", "changed", "wrong", "back", "corrected"):
            _rev_m = re.search(
                r"\b(?:hate|dislike|detest|can't\s+stand|love|like|enjoy|"
                r"prefer|care\s+for|believe|think)\b\s+(?:that\s+|to\s+)?"
                r"(.+?)(?:\.|\band\b|\bbut\b|$|,)", tail)
            if _rev_m:
                _rev_topic = self._opinion_topic(_rev_m.group(1).strip())
                if _rev_topic:
                    topic = _rev_topic
        # REVERSAL SCOPE GUARD. A recant like "i was wrong about acoustic-only"
        # must only flip a stance when the recanted phrase RESOLVES to a held
        # topic. The loose resolver below can link "acoustic-only" to the held
        # "acoustic music" stance by substring, then invert a stance the user
        # was NARROWING (they still liked acoustic, just not *only* acoustic).
        # That is a scope-widening, not a reversal — flipping it corrupts the
        # store. So require a TIGHT link: the recant's topic must share a
        # content word with, or be closely contained by, a held stance key.
        # Loose substring containment (a broader held topic merely containing
        # a word of the recant) is rejected for reversals. This reads the live
        # stance store — no per-topic table.
        _target_candidates = []
        _raw_topic_tokens = set(re.findall(r"[a-z']+", (topic or "").lower()))
        # split hyphenated qualifiers ("acoustic-only" -> acoustic, only)
        _raw_topic_tokens |= set(t for tok in _raw_topic_tokens
                                 for t in tok.split("-") if t)
        # scope markers ("only"/"just"...) are NOT topic content — they signal
        # a NARROWING of an existing stance, not a reversal of it. Strip them
        # before matching so "acoustic-ONLY" is read as "acoustic", whose
        # remainder is a strict subset of the held "acoustic music" topic.
        _scope_markers = {"only", "just", "really", "truly", "merely", "simply"}
        _topic_tokens = _raw_topic_tokens - _scope_markers
        for _k in self.opinions.stances:
            _kt = set(re.findall(r"[a-z']+", _k.lower()))
            if not _kt:
                continue
            # tight: every recant token present in the held key, OR strong
            # jaccard overlap (>=0.5) between the two content-word sets.
            if _topic_tokens and (_topic_tokens <= _kt
                                  or len(_topic_tokens & _kt) / max(1, len(_topic_tokens)) >= 0.5):
                # REJECT a narrowing recant: when the recant's meaningful
                # content is a STRICT subset of the held topic, the user is
                # restricting scope (still likes acoustic, just not *only*
                # acoustic), not reversing their attitude. Flipping the held
                # stance would corrupt the store. Scope-widening is not a
                # reversal. Detected generically from token containment.
                if _topic_tokens and _topic_tokens < _kt:
                    continue
                _target_candidates.append(_k)
        # Only honor a loose substring match when it is NOT a broad-held-topic
        # merely containing the recant word (that is the corruption case).
        target = None
        if _target_candidates:
            # prefer the most specific (shortest) tight match
            target = min(_target_candidates, key=len)
        else:
            _loose = self.opinions.resolve_topic(topic) or self.opinions.resolve_topic(tail)
            if _loose is not None:
                _loose_tokens = set(re.findall(r"[a-z']+", _loose.lower()))
                # Accept the loose match only if the recant phrase is NOT a
                # narrowing qualifier of a broader held stance. A narrowing
                # recant ("i was wrong about acoustic-ONLY") keeps the held
                # attitude and only restricts its scope, so flipping the held
                # stance corrupts the store. Detect: the recant's content words
                # (ignoring scope markers like "only"/"just"/"really") are a
                # subset of the held topic's words, or a hyphenated qualifier
                # whose head appears in the held topic. Such a match is
                # scope-widening, not a reversal -> reject.
                _scope_markers = {"only", "just", "really", "truly", "merely", "simply"}
                _recant_meaningful = _topic_tokens - _scope_markers
                _is_narrowing = (
                    _recant_meaningful
                    and _recant_meaningful <= _loose_tokens
                    and _recant_meaningful != _loose_tokens)
                if not _is_narrowing:
                    target = _loose
        if target is None:
            target = self._stance_key_in_text(q)
            # also try resolving the whole-utterance content head
            if target is None:
                _whole = self._opinion_topic(q)
                if _whole:
                    target = self.opinions.resolve_topic(_whole)
                if target is None:
                    return
        # SCOPE-WIDENING GUARD (final): never reverse a stance whose topic is a
        # BROADER held stance that the recant merely narrows. "i was wrong about
        # acoustic-ONLY" still likes acoustic; flipping "acoustic music" corrupts
        # the store. Reject when the recant's extracted object content is a
        # strict subset of the resolved target's content words (a narrowing, not
        # a reversal). Reads the live stance store — no per-topic rule.
        _tgt_tokens = set(re.findall(r"[a-z']+", target.lower()))
        _recant_meaningful = set(re.findall(r"[a-z']+", (topic or "").lower()))
        _recant_meaningful = {t for tok in _recant_meaningful
                              for t in tok.split("-") if t} - _scope_markers
        if _recant_meaningful and _recant_meaningful < _tgt_tokens:
            return
        try:
            self.opinions._soft_reversal = _soft
            self.opinions.reverse_stance(target, utterance=text)
        except Exception:
            pass
        # Mirror any count correction into the structured QuantityMemory store
        # so "how many X" recall reflects the corrected number. Runs once per
        # mine regardless of which correction-detection path (277 / 580) set
        # detected_correction_fact. Seed + online; correct() supersedes the
        # prior record by subject+noun so a stale count is retired, not echoed.
        try:
            _cf = self.detected_correction_fact
            if (_cf and str(_cf[0]).lower() in ("i", "me", "my")
                    and _cf[1] in ("does", "count", "number", "qty")):
                _cfv = (_cf[2] or "").lower().strip()
                _qc = None
                _qent = None
                _ctoks = _cfv.split()
                for _i, _tok in enumerate(_ctoks):
                    _n = number_to_int(_tok)
                    if _n is not None and _i + 1 < len(_ctoks):
                        _qc = _n
                        _qent = " ".join(
                            t for t in _ctoks[_i + 1:_i + 4]
                            if re.match(r"^[a-z]+$", t))
                        break
                if _qc is not None and _qent:
                    self.quantity_memory.correct(
                        subject="i", noun=_qent, count=_qc)
        except Exception:
            pass

    def observe_user_query(self, query: str, subject: str, valence: float):
        subject_lower = subject.lower()
        self.topic_interaction_count[subject_lower] = self.topic_interaction_count.get(subject_lower, 0) + 1
        self.learning_goals[subject_lower] = self.learning_goals.get(subject_lower, 0) + 1
        if subject_lower:
            self.knowledge_model[subject_lower] = min(
                1.0, self.knowledge_model.get(subject_lower, 0.0) + 0.1)
        current_rapport = self.emotional_rapport.get(subject_lower, 0.0)
        self.emotional_rapport[subject_lower] = current_rapport + 0.2 * (valence - current_rapport)

        q_clean = query.lower().strip(" ?!.")
        m_like = re.search(r"\bi\s+(?:like|love|prefer|enjoy)\s+(.+)", q_clean, re.IGNORECASE)
        if m_like:
            # Capture only the REAL content head of what's liked, not the
            # entire trailing clause. "i prefer acoustic music over anything
            # produced on a laptop" must store "acoustic music", not the
            # comparative tail "over anything produced on a laptop". Resolve
            # via _opinion_topic so the value is a clean content concept, never
            # a function-word run-on. Generic — no per-topic list.
            raw = m_like.group(1).strip(" .!?")
            thing = self._opinion_topic(raw)
            if not thing:
                thing = raw.split()[0] if raw else ""
            if thing and thing not in ("you", "it", "that", "this", "them", "him", "her", "me", "something", "everything", "anything"):
                if "likes" not in self.preferences:
                    self.preferences["likes"] = []
                if thing not in self.preferences["likes"]:
                    self.preferences["likes"].append(thing)

        m_interest = re.search(r"\bi\s+(?:want\s+to\s+learn\s+about|am\s+interested\s+in|'m\s+interested\s+in)\s+(.+)", q_clean, re.IGNORECASE)
        if m_interest:
            thing = m_interest.group(1).strip(" .!?")
            if thing and thing not in ("you", "it", "that", "this", "them", "him", "her", "me"):
                if "interests" not in self.preferences:
                    self.preferences["interests"] = []
                if thing not in self.preferences["interests"]:
                    self.preferences["interests"].append(thing)

        m_fav = re.search(r"\bmy\s+favorite\s+(.+?)\s+is\s+(.+)", q_clean, re.IGNORECASE)
        if m_fav:
            category = m_fav.group(1).strip(" .!?")
            val = m_fav.group(2).strip(" .!?")
            if category and val:
                if "favorites" not in self.preferences:
                    self.preferences["favorites"] = {}
                self.preferences["favorites"][category] = val

        m_name = re.search(
            r"\b(?:my\s+name\s+is|i\s+am\s+called|call\s+me)\s+"
            r"([^.,!?]+?)(?:\s+(?:and|but|,|\.|$))",
            q_clean, re.IGNORECASE)
        if not m_name:
            m_name = re.search(r"\b(?:do\s+you\s+know\s+my\s+name|know\s+my\s+name|is\s+my\s+name)\s+is\s+(.+)", q_clean, re.IGNORECASE)
        # Biographical location / origin: "I live in X" / "I am from X" /
        # "I was born in X". Captured into user_location / user_background
        # so a later "where do I live?" / "where are you from?" recalls it.
        # Mine biographical + general personal facts into the learned store.
        # Extracted so the same-turn identity gate can call it with only the
        # raw text (subject isn't assigned yet in process_turn there).
        # C-clock (round 2026-08-08c): advance the fact-store turn clock
        # BEFORE mining personal facts, not after. The disclosure-ack
        # composer (_derive_ack_from_store) acks a fact only if its
        # turn_number == the store's current turn_num, so that it reports
        # what was learned THIS turn (not a stale fact from 30 turns ago).
        # When advance_turn() ran AFTER mine_personal_facts, a freshly
        # stored fact got turn_number == turn_num - 1, so the equality
        # never held and the honest ack fell through to the degenerate
        # "got it — thanks for telling me." on ~30 turns of real
        # disclosures (e.g. "my hands are raw from hauling hive boxes").
        # Advancing first makes the just-stored fact match the clock, so
        # the ack renders the REAL stored relation. No retraining; pure
        # ordering fix.
        self.personal_facts.advance_turn()
        self.opinions.advance_turn()
        self.mine_personal_facts(query)

        self._update_cognitive_style(query)
        if subject_lower != self.last_topic and self.last_topic:
            self.topic_followup_count[self.last_topic] = max(0, self.topic_followup_count.get(self.last_topic, 0) - 1)
            self.turn_since_topic_change = 0
        else:
            self.turn_since_topic_change += 1
            if self.last_topic:
                self.topic_followup_count[self.last_topic] = self.topic_followup_count.get(self.last_topic, 0) + 1
        self.last_topic = subject_lower

        total_interactions = sum(self.topic_interaction_count.values())
        total_followups = sum(self.topic_followup_count.values())
        self.conversation_depth = total_followups / max(1, len(self.topic_interaction_count))
        self.engagement_level = min(1.0, 0.3 + 0.7 * (total_followups / max(1, total_interactions)))

        self.interaction_count += 1
        self.relationship_depth = min(1.0, self.interaction_count / 20.0)

        inferred = self.infer_user_goal(query)
        self.last_goal = inferred
        self.goals.append(inferred)
        if len(self.goals) > 50:
            self.goals = self.goals[-50:]

        emotion_vad = self._infer_user_emotion(query)
        self._record_interaction(query, subject, emotion_vad)

        # ── ACC analog: Detect correction patterns ──
        self._detect_correction(query, subject, valence)

    def _verb_stem(self, verb: str) -> str:
        """Normalize an inflected activity verb to its stem, and collapse common
        synonyms to ONE canonical form, so date facts store ONE consistent
        activity key (e.g. 'repaired' -> 'repair', 'picked up' -> 'pick up',
        'fixing' -> 'repair') and date recall can match a query phrased any way
        ('fixing' / 'fix' / 'repairing' / 'repair'). Seed mapping (RAVANA-
        expandable: removing an entry degrades gracefully); not an if/elif answer
        path — it is a linguistic normalization, not content."""
        _v = (verb or "").strip().lower()
        _MAP = {
            "repaired": "repair", "repairing": "repair", "repairs": "repair",
            "fixed": "repair", "fixing": "repair", "fixes": "repair", "fix": "repair",
            "built": "build", "building": "build", "builds": "build",
            "kept": "keep", "keeping": "keep", "keeps": "keep",
            "played": "play", "playing": "play", "plays": "play",
            "learned": "learn", "learning": "learn", "learns": "learn",
            "studied": "study", "studying": "study", "studies": "study",
            "brewed": "brew", "brewing": "brew", "brews": "brew",
            "raised": "raise", "raising": "raise", "raises": "raise",
            "wrote": "write", "writing": "write", "writes": "write",
            "read": "read", "reads": "read",
            "ran": "run", "running": "run", "runs": "run",
            "taught": "teach", "teaching": "teach", "teaches": "teach",
            "cooked": "cook", "cooking": "cook", "cooks": "cook",
            "crafted": "craft", "crafting": "craft", "crafts": "craft",
            "moved": "move", "moving": "move", "moves": "move",
            "gardened": "garden", "gardening": "garden",
            "picked up": "pick up", "took up": "take up",
            "got into": "get into",
        }
        _v = _MAP.get(_v, _v)
        # canonical synonym collapse (separate from inflection stemming)
        _SYN = {
            "fix": "repair", "repair": "repair",
            "frame": "build", "frame-build": "build", "framebuild": "build",
        }
        return _SYN.get(_v, _v)

    def _detect_correction(self, query: str, subject: str, valence: float):
        """ACC conflict detection: detect that the user is correcting RAVANA.
        
        Three detection streams:
        1. Direct: explicit "no", "that's wrong", etc.
        2. Sentiment drop: valence drops significantly after response
        3. Re-ask: user repeats similar query within 3 turns
        """
        self.detected_correction = False
        self.detected_correction_type = None
        self.correction_severity = 0.0
        self.correction_subject = subject
        self.detected_correction_fact = None

        q_clean = query.lower().strip()

        # Stream 1: Direct correction patterns
        for pattern in _CORRECTION_DIRECT_PATTERNS:
            if re.search(pattern, q_clean, re.IGNORECASE):
                self.detected_correction = True
                self.detected_correction_type = CorrectionType.DIRECT
                self.correction_severity = max(self.correction_severity, 0.5)
                break

        # Stream 2: Sentiment drop — valence drops significantly from previous turn
        prev_valence = self._last_user_valence_before_response
        if prev_valence > 0 and (prev_valence - valence) > 0.4:
            self.detected_correction = True
            if self.detected_correction_type != CorrectionType.DIRECT:
                self.detected_correction_type = CorrectionType.SENTIMENT_DROP
            self.correction_severity = max(self.correction_severity, 
                                             min(0.8, (prev_valence - valence) * 1.5))

        # Stream 3: Re-ask — similar query within 3 turns
        if self._previous_user_query:
            prev_words = set(self._previous_user_query.lower().split())
            curr_words = set(q_clean.split())
            overlap = len(prev_words & curr_words) / max(1, len(prev_words | curr_words))
            if overlap > 0.6 and len(prev_words) >= 3:
                self.detected_correction = True
                if self.detected_correction_type not in (CorrectionType.DIRECT, CorrectionType.SENTIMENT_DROP):
                    self.detected_correction_type = CorrectionType.INDIRECT_REASK
                self.correction_severity = max(self.correction_severity, 0.3)

        # Extract corrected fact if user provides one. D3 (round v4): run
        # unconditionally, not only after a direct/sentiment/reask signal.
        # A name/value correction ("my sister's name is not meena, it's priya")
        # has NO direct-correction cue word, so gating on detected_correction
        # made the name extractor dead — the fact was never extracted. The
        # extractor self-signals via the name pattern, so let it set the flag.
        self._extract_correction_fact(query, subject)

        self._previous_user_query = q_clean

    def _extract_correction_fact(self, query: str, subject: str):
        """Extract (subject, relation, correct_value) from correction sentence.
        E.g. \"2+2 is 4, not 5\" → (\"2+2\", \"is\", \"4\")
        """
        q_clean = query.lower().strip()
        # D3 (round v3): "X's name is not Y, it's Z" / "X is not Y, it's Z".
        # The corrected value is the token after "it's" (group 2); the subject
        # attribute is group 1 (e.g. "sister's"). The negation word "not" is
        # NEVER stored as a value — that was the D-A bug (corrected value was
        # "not"). Handled before the generic loop because its group order
        # differs.
        # D3 (round v4): "X is not Y, it is/it's Z" — the NEGATION-FIRST shape
        # ("my dog is not max, it is rocky"). The corrected value is the token
        # after "it is"/"it's" (last group); the subject attribute is the noun
        # before "is not". The existing _CORRECTION_FACT_PATTERNS only handle
        # "X is Y, not Z" (negation LAST), so this dominant natural shape was
        # never extracted — corrections about pets/things/places were lost.
        # The negation word is NEVER stored as a value. Handled before the
        # generic loop because its group order (attr, correct_val) differs.
        _notfirst = re.search(
            r"(?:my|the|a|an)?\s*([\w'-]+?)'\s*is\s+not\s+[\w'-]+"
            r"[,.]*\s+(?:it'?s|it\s+is)\s+([\w'-]+)", q_clean, re.IGNORECASE)
        if _notfirst is None:
            _notfirst = re.search(
                r"([\w'-]+)\s+is\s+not\s+[\w'-]+[,.]*\s+"
                r"(?:it'?s|it\s+is)\s+([\w'-]+)", q_clean, re.IGNORECASE)
        if _notfirst:
            _subj_attr = _notfirst.group(1).strip().removesuffix("'s")
            _correct_val = _notfirst.group(2).strip()
            if _correct_val:
                self.detected_correction = True
                self.detected_correction_fact = ("i", _subj_attr, _correct_val)
                self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                self.correction_severity = max(self.correction_severity, 0.8)
                return
        _nm = re.search(_CORRECTION_NAME_FACT_PATTERN, q_clean, re.IGNORECASE)
        if _nm:
            _subj_attr = _nm.group(1).strip()
            _correct_val = _nm.group(2).strip()
            self.detected_correction_fact = ("i", _subj_attr, _correct_val)
            self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
            self.correction_severity = max(self.correction_severity, 0.8)
            return
        for pattern in _CORRECTION_FACT_PATTERNS:
            m = re.search(pattern, q_clean, re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 2:
                    # it's X, not Y
                    correct_val = groups[0]
                    wrong_val = groups[1]
                    self.detected_correction_fact = (subject, "is", correct_val)
                    self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                    self.correction_severity = max(self.correction_severity, 0.7)
                elif len(groups) == 3:
                    # X is Y, not Z
                    fact_subject = groups[0].strip()
                    correct_val = groups[1]
                    wrong_val = groups[2]
                    self.detected_correction_fact = (fact_subject, "is", correct_val)
                    self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                    self.correction_severity = max(self.correction_severity, 0.8)

    def store_response_for_correction(self, response: str, strategy: str, valence: float):
        """Store the last response so correction detection can reference it."""
        self._last_response_for_correction = response
        self._last_response_strategy_for_correction = strategy
        self._last_user_valence_before_response = valence

    def reset_correction_flags(self):
        """Reset correction flags for next turn."""
        self.detected_correction = False
        self.detected_correction_type = None
        self.correction_severity = 0.0
        self.correction_subject = ""
        self.detected_correction_fact = None

    # D2 (round v2): closed-class / function words that can never own a stance.
    # A stance must land on a real CONTENT concept ("talk", "solitude"),
    # never on a determiner/adverb ("the", "small", "how"). This is a minimal
    # universal seed (closed-class set is language-universal and tiny), NOT a
    # per-topic table — it generalizes to any topic the user names.
    _OPINION_STOP = {
        "the", "a", "an", "my", "your", "our", "their", "his", "her", "its",
        "this", "that", "these", "those", "some", "any", "no", "all", "every",
        "i", "you", "he", "she", "we", "they", "me", "him", "us", "them",
        "and", "but", "or", "so", "if", "when", "while", "because", "of",
        "to", "in", "on", "at", "for", "with", "from", "by", "as", "into",
        "about", "over", "under", "how", "what", "why", "who", "where",
        "off", "onto", "upon", "than", "then", "till", "until", "since",

        "really", "very", "just", "only", "also", "too", "quite", "more",
        "most", "much", "many", "such", "own", "same", "other", "another",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "not", "don't", "dont", "do", "does", "did", "can", "cannot", "cant",
        "it", "they're", "im", "i'm", "you're", "we're", "there",
    }

    # C-fix (round 2026-08-12T0613Z): universal ghost topics that can NEVER
    # own a stance. Comparative / superlative patterns occasionally capture a
    # non-attitude head (an indefinite pronoun like "anything", or a grammatical
    # gerund like "standing"/"being" stripped from a longer phrase). These are
    # rejected as single-word stance topics. This is a tiny UNIVERSAL seed set
    # (indefinite pronouns + grammatical gerunds), NOT a per-topic deny-list —
    # it generalizes to any topic the user names and RAVANA cannot learn
    # attitude objects from these ghosts.
    _STANCE_GHOST_TOPICS = {
        # indefinite pronouns / quantifiers
        "anything", "something", "everything", "nothing", "whatever",
        "whoever", "whichever", "anyone", "everyone", "someone", "noone",
        "nobody", "everybody", "somebody", "anybody",
        # grammatical gerunds that pattern-matchers emit as a head but can
        # never be an attitude object
        "standing", "being", "doing", "having", "going", "coming", "feeling",
        "thinking", "knowing", "wanting", "making", "taking", "getting",
        "being", "saying", "talking", "looking", "feeling", "seeming",
        "open", "single", "moment", "sense", "breath", "note", "held",
        "restless", "quietest", "quiet", "outside", "inside", "away",
        "around", "through", "across", "behind", "before", "after",
    }

    def _opinion_topic(self, phrase: str) -> Optional[str]:
        """Resolve the salient CONTENT HEAD of an opinion-object phrase.

        Strips leading determiners and trailing modifiers and CUTS at the
        first internal closed-class word (preposition/conjunction) so:
          "the solitude of the lighthouse" -> "solitude"
          "small talk at the village market" -> "small talk"
          "accordion when the wind dies down" -> "accordion"
          "how whales communicate" -> "whales"
        But a RELATIVE CLAUSE ("people who talk in theatres", "books that
        last") is a single attitude object — the noun + its relative
        descriptor is the concept the user is actually evaluating. So the
        relative pronouns who/whom/that/which act as a BRIDGE: we keep them
        and continue collecting the clause content until the next hard stop
        (preposition/conjunction), giving "people who talk" / "books that last"
        rather than collapsing to the bare noun "people"/"books". Collapsing to
        the bare noun is what caused opposite-signed stances ("for people who
        X" vs "against people who Y") to MERGE onto one key and average to
        ~0, erasing the real attitude.
        Returns the joined head phrase (one or more content nouns), or None
        if the phrase is all closed-class words (so we never seed a garbage
        stance on a function word like "the"/"how"/"small").
        """
        toks = [t for t in re.findall(r"[a-z'][a-z']*", phrase.lower())]
        if not toks:
            return None
        # Drop leading closed-class words (determiners/prepositions).
        while toks and toks[0] in self._OPINION_STOP:
            toks.pop(0)
        if not toks:
            return None
        # Collect the head. A relative pronoun (who/whom/that/which) is a
        # bridge: keep it and DON'T break there, so the clause after it is
        # included as part of the attitude object.
        _REL_BRIDGE = {"who", "whom", "that", "which"}
        head = []
        for t in toks:
            if t in _REL_BRIDGE:
                head.append(t)          # bridge: continue into the clause
                continue
            if t in self._OPINION_STOP:
                break
            head.append(t)
        if not head:
            return None
        # Drop trailing closed-class/modifier words as a final safety
        # (but never drop a trailing relative bridge like "who"/"that").
        while len(head) > 1 and head[-1] in self._OPINION_STOP:
            head.pop()
        if not head:
            return None
        return " ".join(head)

    def _ensure_emotion_detector(self):
        if self._emotion_detector is None or not hasattr(self._emotion_detector, '_vad_matrix'):
            from ravana.core import UserEmotionDetector
            self._emotion_detector = UserEmotionDetector()

    def _infer_user_emotion(self, text: str) -> Tuple[float, float, float]:
        self._ensure_emotion_detector()
        v, a, d = self._emotion_detector.detect(text)
        rate = 0.35
        prev = self.emotional_state
        self.emotional_state = {
            'valence': prev['valence'] + rate * (v - prev['valence']),
            'arousal': prev['arousal'] + rate * (a - prev['arousal']),
            'dominance': prev['dominance'] + rate * (d - prev['dominance']),
        }
        return (v, a, d)

    def _record_interaction(self, text: str, subject: str,
                            emotion_vad: Tuple[float, float, float]):
        self.interaction_history.append({
            'text': text[:200],
            'subject': subject,
            'valence': emotion_vad[0],
            'arousal': emotion_vad[1],
            'dominance': emotion_vad[2],
            'turn': len(self.interaction_history),
        })
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

    def infer_user_goal(self, query: str) -> str:
        q = query.lower().strip()
        debug_markers = ('broken', "doesn't work", "doesn't work", 'error', 'fail',
                         'bug', 'crash', 'wrong', 'stuck', 'issue', 'fix', 'not working',
                         "isn't working", 'exception', 'traceback')
        if any(m in q for m in debug_markers) or q.startswith('why is') and any(
            m in q for m in ('broken', 'error', 'fail', 'wrong', 'crash')):
            return "DEBUGGING"
        learn_markers = ('how does', 'how do', 'how is', 'how are', 'what is', 'what are',
                         'explain', 'how come', 'why does', 'why do')
        if any(q.startswith(m) or (' ' + m) in q for m in learn_markers):
            return "LEARNING"
        explore_markers = ('tell me about', "let's talk about", 'i want to know',
                           'i wonder', 'teach me', 'show me', 'describe')
        if any(m in q for m in explore_markers):
            return "EXPLORING"
        return "EXPLORING"

    def _update_cognitive_style(self, query: str):
        q_lower = query.lower()
        style_scores = {
            'curious': sum(1 for w in ['why', 'how', 'what', 'explain', 'understand', 'curious', 'wonder'] if w in q_lower),
            'skeptical': sum(1 for w in ['really', 'actually', 'prove', 'evidence', 'doubt', 'sure', 'fake', 'lie'] if w in q_lower),
            'practical': sum(1 for w in ['how to', 'build', 'make', 'create', 'step', 'guide', 'tutorial', 'implement'] if w in q_lower),
        }
        if style_scores:
            top_style = max(style_scores, key=style_scores.get)
            if style_scores[top_style] > 0:
                self.cognitive_style = top_style

    def infer_topic_interest(self, topic: str) -> float:
        t = topic.lower()
        goal_strength = min(1.0, self.learning_goals.get(t, 0) * 0.2)
        rapport = (self.emotional_rapport.get(t, 0.0) + 1.0) / 2.0
        interaction = min(1.0, self.topic_interaction_count.get(t, 0) * 0.1)
        return (goal_strength * 0.4 + rapport * 0.4 + interaction * 0.2)

    def infer_user_knows(self, concept: str) -> float:
        return self.knowledge_model.get(concept.lower(), 0.0)

    def infer_user_wants_to_learn(self, concept: str) -> float:
        t = concept.lower()
        goal = min(1.0, self.learning_goals.get(t, 0) * 0.15)
        rapport = (self.emotional_rapport.get(t, 0.0) + 1.0) / 2.0
        return max(0.0, goal * 0.6 + rapport * 0.4 - self.knowledge_model.get(t, 0.0) * 0.5)

    def get_preferred_relation_types(self) -> List[str]:
        rel_counts = {}
        for (f, t), count in self.edge_reactivations.items():
            rel = 'semantic'
            rel_counts[rel] = rel_counts.get(rel, 0) + count
        return sorted(rel_counts, key=rel_counts.get, reverse=True)[:3]

    def inferred_preferences(self, threshold: int = 2) -> Dict[Tuple[str, str], int]:
        return {(f, t): c for (f, t), c in self.edge_reactivations.items()
                if c >= threshold}

    def activation_boost_for(self, concept: str) -> Dict[str, float]:
        boost: Dict[str, float] = {}
        cl = concept.lower()
        for (from_c, to_c), count in self.edge_reactivations.items():
            if from_c == cl:
                boost[to_c] = 1.0 + (count / (count + 1.0)) * 0.3
        return boost

    def get_state(self) -> Dict:
        return {
            'edge_reactivations': {str(k): v for k, v in self.edge_reactivations.items()},
            'query_concepts': list(self.query_concepts),
            'knowledge_model': self.knowledge_model,
            'learning_goals': self.learning_goals,
            'emotional_rapport': self.emotional_rapport,
            'cognitive_style': self.cognitive_style,
            'engagement_level': self.engagement_level,
            'conversation_depth': self.conversation_depth,
            'topic_interaction_count': self.topic_interaction_count,
            'topic_followup_count': self.topic_followup_count,
            'last_topic': self.last_topic,
            'turn_since_topic_change': self.turn_since_topic_change,
            'interaction_count': self.interaction_count,
            'relationship_depth': self.relationship_depth,
            'goals': self.goals,
            'last_goal': self.last_goal,
            'user_name': self.user_name,
            'user_location': self.user_location,
            'user_background': self.user_background,
            'preferences': self.preferences,
            'personal_facts': self.personal_facts.get_state(),
            'quantity_memory': self.quantity_memory.get_state(),
            'opinions': self.opinions.get_state(),
            'emotional_state': self.emotional_state,
            'belief_state': self.belief_state,
            'interaction_history': self.interaction_history,
            '_learned_relations': list(getattr(self, '_learned_relations', set())),
        }

    def set_state(self, state: Dict):
        self.edge_reactivations = {eval(k): v for k, v in state.get('edge_reactivations', {}).items()}
        self.query_concepts = set(state.get('query_concepts', []))
        self.knowledge_model = state.get('knowledge_model', {})
        self.learning_goals = state.get('learning_goals', {})
        self.emotional_rapport = state.get('emotional_rapport', {})
        self.cognitive_style = state.get('cognitive_style', 'balanced')
        self.engagement_level = state.get('engagement_level', 0.5)
        self.conversation_depth = state.get('conversation_depth', 0.0)
        self.topic_interaction_count = state.get('topic_interaction_count', {})
        self.topic_followup_count = state.get('topic_followup_count', {})
        self.last_topic = state.get('last_topic', '')
        self.turn_since_topic_change = state.get('turn_since_topic_change', 0)
        self.interaction_count = state.get('interaction_count', 0)
        self.relationship_depth = state.get('relationship_depth', 0.0)
        self.goals = state.get('goals', [])
        self.last_goal = state.get('last_goal', 'EXPLORING')
        self.user_name = state.get('user_name', '')
        self.user_location = state.get('user_location', '')
        self.user_background = state.get('user_background', '')
        self.preferences = state.get('preferences', {})
        self._learned_relations = set(state.get('_learned_relations', []))
        _pf = state.get('personal_facts')
        if _pf:
            self.personal_facts.set_state(_pf)
        _qm = state.get('quantity_memory')
        if _qm:
            self.quantity_memory.set_state(_qm)
        _op = state.get('opinions')
        if _op:
            self.opinions.set_state(_op)
        self.emotional_state = state.get('emotional_state',
            {'valence': 0.0, 'arousal': 0.3, 'dominance': 0.5})
        self.belief_state = state.get('belief_state', {})
        self.interaction_history = state.get('interaction_history', [])


# ── Module-level persistence helpers ───────────────────────────────────────────
def _user_model_path(user_suffix: str = "") -> str:
    """Path for a per-user-model store file under <repo>/user_models/."""
    os.makedirs(USER_MODELS_DIR, exist_ok=True)
    return os.path.join(USER_MODELS_DIR, f"ravana_usermodel{user_suffix}.pkl")


def save_user_model(user_model: "UserModel", user_suffix: str = "") -> str:
    """Persist a UserModel to its own dedicated file. Returns the path."""
    path = _user_model_path(user_suffix)
    with open(path, "wb") as f:
        pickle.dump(user_model, f)
    return path


def load_user_model(user_suffix: str = "") -> "UserModel":
    """Load a UserModel from its dedicated file, or return a fresh one."""
    path = _user_model_path(user_suffix)
    if not os.path.exists(path):
        return UserModel()
    with open(path, "rb") as f:
        um = pickle.load(f)
    if not hasattr(um, "personal_facts"):
        from .personal_fact_store import PersonalFactStore, UserStanceStore, QuantityMemory
        um.personal_facts = PersonalFactStore()
        um.opinions = UserStanceStore()
    if not hasattr(um, "quantity_memory"):
        from .personal_fact_store import QuantityMemory
        um.quantity_memory = QuantityMemory()
    return um
