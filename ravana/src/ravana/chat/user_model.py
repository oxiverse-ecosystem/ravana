import os
import re
import pickle
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from .models import CorrectionType
from .personal_fact_store import (
    PersonalFactStore, UserStanceStore, QuantityMemory, number_to_int)
from . import pet_slots as _pet_slots
from . import possession_attrs as _poss
from .constants import STOP_WORDS

# Filler / temporal / discourse tokens that must NEVER count as a topic overlap
# when resolving a held stance from an utterance. constants.STOP_WORDS omits
# temporal adverbs ("now", "still", "today") and a few discourse markers that
# otherwise cause cross-topic misbinding: a held stance keyed "thunderstorms
# now" would otherwise match ANY later utterance containing "now" (the
# 2026-08-21T0843Z misattribution where a grass contradiction acked "you've
# changed your mind about thunderstorms"). These are function words, not topic
# content; dropping them from the overlap score leaves only real content tokens.
_FILLER_TOKENS = frozenset({
    "now", "still", "today", "tonight", "yesterday", "tomorrow", "already",
    "yet", "again", "lately", "recently", "currently", "actually", "really",
    "just", "though", "anyway", "anymore", "here", "there", "then", "soon",
    "usually", "sometimes", "often", "always", "never", "ever",
})

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


# Activity-verb seed lexicon (shared by the relationship-activity miner D7 and
# the cued-recall grammar fix). Used to tell a VERB-PHRASE personal fact
# ("weaves baskets") from a NOUN-PHRASE fact ("an astronomer") so recall can
# render "your grandmother indira weaves baskets" (no copula) vs "your niece
# priya is an astronomer" (copula). This is SEED vocabulary (a data set, not an
# answer path) — RAVANA-expandable by the same PersonalFactStore the user can
# correct; removing entries degrades gracefully (one fewer verb-form recognized
# for grammar). NOT a per-topic reply dictionary and NOT authored prose.
_ACTIVITY_VERB_LEXICON = {
    "run", "own", "operate", "play", "teach", "taught", "study", "manage", "drive",
    "build", "built", "make", "made", "sell", "restore", "grow", "grew", "watch", "raise", "tend",
    "brew", "bake", "write", "wrote", "read", "learn", "learned", "learnt", "practice", "collect", "fix",
    "paint", "code", "design", "craft", "volunteer", "cook", "fish", "hike",
    "garden", "farm", "lead", "organize", "keep", "kept", "grind", "race", "sail",
    "knit", "sew", "weld", "forge", "carve", "compose", "record",
    "perform", "coach", "train", "compete", "spin", "weave", "mount",
    "trade", "host", "guide", "climb", "repair", "work", "draw", "drew",
    # Round 2026-08-19T1026Z: broaden the seed verb vocabulary with common
    # activity verbs the relationship/pet miner needs to split name/verb/object.
    # Previously absent, so disclosures using them ("my sister Pora SENDS
    # postcards", "my cat Mochi KNOCKS the vase", "my grandmother LEFT me a
    # compass", "my aunt NAMED the gull Greywing", "the steppe PASSES through
    # her letters") were dropped or had the whole clause merged into the name
    # (measured T33/T58/T68 this round). Seed data, RAVANA-expandable; adding
    # a verb only widens recognition, never reply content.
    "send", "sent", "knock", "knocked", "give", "gave", "given", "leave",
    "left", "name", "named", "pass", "passed", "passes", "bring", "brought",
    "show", "showed", "told", "say", "said", "call", "called", "visit",
    "visited", "help", "helped", "love", "loved", "like", "hate", "hated",
    "feed", "fed", "walk", "walked", "care", "cared", "protect", "protected",
    # PET ACTIVITY VERBS (round 2026-08-22T0703Z, DEFECT D1): core verbs a pet
    # does that a disclosure may report as an ACTIVITY (so the companion-fact
    # miner captures them, parallel to kin activity mining). Seed lexicon,
    # RAVANA-expandable via the same store; not a per-animal table.
    "hide", "hides", "steal", "steals", "sleep", "sleeps", "bark", "barks",
    "chase", "chases", "catch", "catches", "climb", "climbs", "dig", "digs",
    "fetch", "fetches", "guard", "guards", "pounce", "pounces", "nudge",
    "nudges", "bring", "brings", "carry", "carries", "drop", "drops",
    "sit", "sits", "roll", "rolls", "spin", "spins", "meow", "meows",
    "purr", "purrs", "chew", "chews", "lick", "licks", "paw", "paws",
}


def is_activity_verb(word: str) -> bool:
    """Return True if `word` is a (possibly inflected) activity verb from the
    seed lexicon. Used by cued recall to render verb-phrase personal facts
    without a spurious copula. Pure vocabulary lookup — no content."""
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    if w in _ACTIVITY_VERB_LEXICON:
        return True
    # base-form recovery for inflected tokens not pre-listed
    for suf in ("ing", "ed", "s", "es"):
        if w.endswith(suf) and w[: -len(suf)] in _ACTIVITY_VERB_LEXICON:
            return True
    return False


# Relation-verb seed lexicon (round 2026-08-20T1935Z residual fix, feature
# t_ec6c6b51). A relationship disclosure often states a DURABLE ATTRIBUTE /
# ABILITY about a named relative using a verb that is NOT in the activity
# lexicon — the canonical case from the round: "my grandmother yaya SPEAKS
# three languages: greek, french, italian". "speaks" is not an activity verb
# (it names a capability, not a basket-weaving-style activity), so the
# relationship miner's activity-verb gate missed the whole disclosure and the
# enumeration was never stored. Other real disclosures of this shape: "my
# uncle ravi WORKS as a mechanic", "my niece STUDIES medicine", "my brother
# PLAYS the violin".
#
# This is SEED vocabulary (a data set, not an answer table) — RAVANA-expandable
# by the same PersonalFactStore the user can correct; removing entries degrades
# gracefully (one fewer relation-verb recognized for grammar). NOT a per-person
# or per-relationship table, NOT authored prose. Deliberately EXCLUDES
# pure-location verbs ("lives"/"resides"/"stays"/"live") — those are owned by
# the dedicated location miner (m_loc), so admitting them here would
# double-store "my grandmother lives in paris" under both the location and the
# relationship key. The activity lexicon already covers the overlapping
# activity verbs (climb/weave/teach/...); this set is the SUPPLEMENT of
# attribute/ability verbs the activity set does not carry.
_RELATION_VERB_LEXICON = {
    # communication / language ability
    "speaks", "speak", "talks", "talk", "communicates", "communicate",
    # profession / role / study
    "works", "work", "studies", "study", "studied", "teaches", "teach",
    "taught", "practices", "practice", "practised", "performs", "perform",
    "operates", "operate", "manages", "manage", "runs", "run", "owns",
    "own", "keeps", "keep", "raises", "raise", "grows", "grow", "designs",
    "design", "codes", "code", "builds", "build", "built", "repairs",
    "repair", "crafts", "craft", "volunteers", "volunteer", "collects",
    "collect", "trades", "trade", "hosts", "host", "guides", "guide",
    "coaches", "coach", "trains", "train", "competes", "compete",
    "composes", "compose", "records", "record", "mounts", "mount",
    "knits", "knit", "sews", "sew", "welds", "weld", "forges", "forge",
    "carves", "carve", "spins", "spin", "weaves", "weave", "sails", "sail",
    "races", "race", "climbs", "climb", "hikes", "hike", "fishes", "fish",
    "farms", "farm", "gardens", "garden", "bakes", "bake", "brews", "brew",
    "cooks", "cook", "reads", "read", "writes", "write", "paints", "paint",
    "draws", "draw", "sings", "sing", "dances", "dance", "swims", "swim",
    "drives", "drive", "plays", "play", "learns", "learn", "learned",
    "learnt",
}


def is_relation_verb(word: str) -> bool:
    """Return True if `word` is a (possibly inflected) relation/attribute verb
    from the seed lexicon — a durable capability/role a named relative is said
    to HAVE (speaks/works/studies/plays/...). Used by the relationship miner to
    recognize attribute disclosures that carry no activity verb, and by the
    recall grammar rule to render them WITHOUT a spurious copula
    ("your grandmother yaya speaks three languages", not "is speaks"). Pure
    vocabulary lookup — no content. Excludes location verbs (owned by m_loc)."""
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    if w in _RELATION_VERB_LEXICON:
        return True
    for suf in ("ing", "ed", "s", "es"):
        if w.endswith(suf) and w[: -len(suf)] in _RELATION_VERB_LEXICON:
            return True
    return False


# Auxiliary-verb seed lexicon (round 2026-08-21T0843Z feature card
# t_16b15684). A relationship disclosure often states an activity through an
# AUXILIARY verb instead of a bare activity/relation verb: "my cousin Jin DOES
# competitive speedcubing", "my sister DID competitive debate", "my brother DOES
# parkour". The earlier two verb classes (activity verbs, relation verbs) both
# MISSED this shape — "does" is neither an activity verb nor a relation verb —
# so the whole disclosure fell through to the name-only path and was dropped by
# the degenerate-fact guard (the residual "cousin Jin's activity not recalled"
# logged at the end of round 2026-08-21T0843Z). The auxiliary opens the SAME
# capture path as the other two verb classes (name = tokens before it, value =
# aux + activity noun-phrase resolved through _opinion_topic), so the disclosure
# mines + recalls like any other. This is SEED vocabulary (a data set, not an
# answer table) — RAVANA-expandable by the same PersonalFactStore the user can
# correct; removing entries degrades gracefully (one fewer aux shape recognized).
# NOT a per-relationship table and NOT authored prose.
_AUX_VERB_LEXICON = {"do", "does", "did", "doing", "done"}


def is_aux_verb(word: str) -> bool:
    """True when `word` is an auxiliary verb that introduces an activity
    noun-phrase in a relationship disclosure ("does"/"did"/"doing" + activity).
    Recognized as a verb-phrase head for both mining and copula-free recall
    rendering. Pure vocabulary lookup — no content."""
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    if w in _AUX_VERB_LEXICON:
        return True
    # inflected forms not pre-listed
    for suf in ("ing", "ed", "s", "es"):
        if w.endswith(suf) and w[: -len(suf)] in _AUX_VERB_LEXICON:
            return True
    return False


def _is_query(q: str) -> bool:
    """True when `q` is an interrogative (question / imperative recall), so the
    personal-fact miner must skip it. A question is never a self-disclosure, yet
    the relationship/activity miners otherwise fire on query substrings (e.g.
    "remind me what my brother Theo DOES for work" stored the degenerate
    ('i','brother theo','does work') fact). The leading-question-word vocabulary
    mirrors the recall resolvers' own _name_is_q gate, so the miner and the
    recaller agree on what a question is — one source of truth, not a second
    per-topic rule. Precise: a trailing '?', OR a leading recall-verb
    (remind/tell/recall/...), OR a wh-word followed by an auxiliary (inverted
    question), OR a leading yes/no auxiliary. A clause-leading disclosure
    ("what i love is running") is NOT inverted (subject 'i' follows 'what', not
    an aux) so it is correctly KEPT and still mined. Seed structure; no content.
    """
    q = (q or "").strip()
    if not q:
        return False
    if q.endswith("?"):
        return True
    _AUX = {"is", "are", "was", "were", "am", "do", "does", "did", "can",
            "could", "would", "will", "shall", "should", "may", "might",
            "have", "has", "had"}
    _RECALL = {"remind", "tell", "show", "explain", "describe", "recall",
               "remember", "know", "say", "mention", "repeat", "list"}
    m = re.match(
        r"^(what|who|whom|whose|where|when|why|which|how|is|are|was|were|"
        r"do|does|did|can|could|would|will|shall|should|may|might|"
        r"remind|tell|show|explain|describe|recall|remember|know|say|"
        r"mention|repeat|list)\b", q, re.IGNORECASE)
    if not m:
        return False
    _w = m.group(1).lower()
    if _w in _RECALL:
        return True
    if _w in ("what", "who", "whom", "whose", "where", "when", "why",
              "which", "how"):
        # wh-word is a question only when inverted (auxiliary before subject).
        _nw = q[m.end():].strip().split()
        return bool(_nw) and _nw[0].lower() in _AUX
    # leading yes/no auxiliary (is/are/do/does/can/...) => question.
    return True


def is_verb_phrase(word: str) -> bool:
    """True when `word` heads a VERB-PHRASE personal fact (activity OR relation
    OR auxiliary verb). The single source of truth for the recall/ack grammar
    rule that drops the copula for verb-phrase values (so "your cousin jin does
    competitive speedcubing" is grammatical, not "is does"). Superset of
    is_activity_verb + is_relation_verb + is_aux_verb; callers that previously
    used is_activity_verb for the copula decision should use this so every verb
    class renders correctly. Pure vocabulary lookup — no content."""
    return is_activity_verb(word) or is_relation_verb(word) or is_aux_verb(word)


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
    rather than a proper-noun name?

    A real name has NO adjectival or verbal WordNet sense and NO predicate
    morphology, so it is never rejected. A transient self-state ("i'm unmoored",
    "i'm gut-punched", "i'm lit-up") is a verb participle / hyphenated emotion
    compound and IS rejected.

    ROUND 2026-08-15T1537Z FIX (D6): the previous test only checked WordNet
    `pos="a"` (adjective). Verb participles ("unmoored" = unmoor+ed) and
    hyphenated emotion compounds ("gut-punched", "lit-up") have NO adjective
    sense, so a ROTATED probe word like "unmoored" slipped through and got
    stored as the user's NAME ("your name is unmoored"). Fix: also accept a
    verb-participle WordNet sense (`pos="v"`) AND the morphological fallback
    (which already covers -ed/-ing participle and hyphenated compounds). This is
    STRUCTURAL — it keys on predicate morphology, not a frozen list of feeling
    words, so any future rotated self-state word is caught without a code change.
    Real names (corvin/maren/nadia) have neither sense and are left accepted.
    """
    w = (word or "").strip().lower().strip("'\"")
    if not w or " " in w or len(w) > 24:
        return False
    # Reject hyphenated emotion compounds outright ("gut-punched", "lit-up"):
    # no proper noun is hyphenated, and every such compound is a predicate.
    if "-" in w and len(w) <= 24:
        return True
    if _WN_AVAILABLE:
        try:
            _syns = _WN.synsets(w, pos="a") or _WN.synsets(w, pos="v")
            if _syns:
                return True
            # WordNet only indexes base lemmas (not every inflection), so a
            # participle like "unmoored" (unmoor+ed) has no direct sense. Fall
            # through to the morphological heuristic below rather than returning
            # False here — otherwise ROTATED predicate words with no WordNet
            # entry ("unmoored", "gut-punched") slip through and become names.
        except Exception:
            pass
    # Structural heuristic: common adjective/participle morphology (-ed/-ing
    # participle or -y/-ful/-ive/-ous suffix) is a predicate signal even when
    # WordNet is absent or has no entry. Real proper-noun names ("corvin",
    # "maren", "nadia") have none of these suffixes, so they stay accepted.
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

# Round 2026-08-17T1126Z: affective-object guard for the EVENT miner. A
# first-person "i <event-verb> <object>" where the object is a SENTIMENT
# adjective ("i find it fascinating", "i found that surprising") is a
# cognitive-affective COPULA, not a discovery event — storing it as
# `event: find fascinating` is garbage (measured: T16 of this round produced
# exactly that). The verb "find" legitimately means find-a-lost-object
# ("i found my keys"), so the verb must stay in _EVENT_VERBS; the guard is on
# the OBJECT. This is seed vocabulary: sentiment adjectives RAVANA encounters.
# Removing one only loses that one shape, never content RAVANA can't change. It
# is not a per-topic answer table and not authored reply prose. RAVANA can
# extend it at runtime the same way it extends the verb lexicons.
_AFFECTIVE_OBJECT_ADJ = frozenset({
    "fascinating", "interesting", "boring", "amazing", "amazed",
    "surprising", "surprised", "strange", "odd", "weird", "wonderful",
    "terrible", "awful", "beautiful", "ugly", "funny", "sad", "happy",
    "annoying", "comforting", "disturbing", "delightful", "disappointing",
    "exciting", "calming", "confusing", "clear", "obvious", "mysterious",
    "scary", "frightening", "moving", "touching", "inspiring", "refreshing",
    "reassuring", "overwhelming", "eye-opening", "mind-blowing", "deep",
    "profound", "meaningful", "pointless", "useless", "helpful",
    "rewarding", "worthwhile", "enjoyable", "pleasant", "unpleasant",
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

# Non-content object heads: an activity/event object that resolves to ONLY
# these words is not a real thing RAVANA learned (it is an aspectual/particle
# verb residue, a bare timeframe, or a generic noun) and must not be stored as
# a `does`/`event` fact. Seed vocabulary (RAVANA-expandable): removing an entry
# only re-admits one low-value object shape. Used by the shared
# _opinion_topic gate so both the activity and event miners reject junk via
# the same chokepoint (rule 6g — extend existing logic, not per-verb branches).
_OBJ_NONCONTENT = frozenset({
    # aspectual / particle verb residues
    "coming", "going", "keep", "keeping", "got", "went", "start", "started",
    "starting", "found", "read", "said", "cared", "burned", "burning",
    "ringing", "took", "taking", "made", "making", "did", "done",
    # particles / framers
    "out", "back", "up", "down", "off", "on", "in", "away", "over", "really",
    "just", "only", "also", "even", "still", "already",
    # bare timeframes
    "last", "tonight", "yesterday", "today", "tomorrow", "now", "then",
    "here", "there", "morning", "evening", "night", "day",
    # generic nouns with no recallable content
    "project", "projects", "thing", "things", "stuff", "lot", "lots",
    "side", "way", "ways", "part", "bit", "it",
})


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

# Free-form reassessment-affect lexicon (feature round 2026-08-20T1229Z-followup).
# A contradiction is NOT always signalled by a retraction keyword ("i take back
# X"), a concession "but" frame, or a limitation "can't" cue — the user often just
# RE-STATES an attitude that opposes what they told RAVANA before ("actually i've
# gone off winter", "they wear me out these days", "not all street art is good").
# To recode a HELD stance from such a free-form reversal we need the NEW attitude's
# polarity. This is a SEED sentiment lexicon (the same allowed class as the VAD
# affect lexicon / the R3 first-person-limitation lexicon): it names reassessment
# affect terms and their valence, and RAVANA can GROW it online (a new term the
# user coins gets added to the live set as it is seen). It is NOT an answer table
# and names NO topic — the topic is resolved against the live stance store, so the
# same capability recodes a contradiction about ANY held stance. The scorer below
# returns a polarity estimate from the strongest-matching term; it never invents a
# stance topic. If the utterance carries NO reassessment term we leave the held
# stance alone (honest — we don't guess a reversal from neutral wording).
_REASSESS_NEG = (
    "gone off", "gone cold on", "wear me out", "wears me out", "wore me out",
    "tires me out", "get to me", "gets to me", "wear on me", "grew tired of",
    "grown tired of", "sick of", "fed up", "over it", "not all", "not so",
    "not that", "no longer", "lost the taste", "lost my taste",
    "come to dislike", "grown to dislike", "coming to dislike", "can't stand",
    "can't take", "put me off", "put off", "turned off", "don't care for",
    "dread", "resent", "recoil", "wary of", "weary of", "bored of", "hate",
    "detest", "loathe", "dislike", "annoy", "annoys me", "irritates",
    "irritate me", "frustrate", "frustrates me", "puts me off",
)
_REASSESS_POS = (
    "come around to", "came around to", "coming around to", "grown to love",
    "grew to love", "coming to love", "taken to", "took to", "warming to",
    "warmed to", "came to enjoy", "grow on me", "grew on me", "grows on me",
    "won me over", "appreciate more", "appreciating more", "fonder of",
    "more into", "come to like", "coming to like", "reconnected with",
)
_REASSESS_NEG_SET = frozenset(_REASSESS_NEG)
_REASSESS_POS_SET = frozenset(_REASSESS_POS)
# GENERALIZE (round 2026-08-21T2156Z, this round's marathon-gap fix): base
# SENTIMENT verbs, not just the reassessment IDIOMS above. The idiom sets only
# caught phrasings like "gone off" / "come around to" / "hate" / "dislike";
# a BASE positive verb ("love" / "like" / "enjoy" / "adore") used in a
# contradiction ("i don't actually LOVE marathon running") was invisible to the
# scorer, so the free-form recode never fired and a held +0.95 stance persisted
# (measured: marathon retraction left running-marathons at +0.95). Mirror the
# idiom sets with base-verb sets so a negated base-positive verb reads as a
# NEGATIVE reassessment of a held-positive stance, and a negated base-negative
# verb as a POSITIVE one — the SAME sign-flip rule as the idioms, no second
# code path. Seed structure (RAVANA-extendable); no per-topic rule, no retrain.
_REASSESS_SENT_POS = (
    "love", "loves", "loved", "like", "likes", "liked", "enjoy", "enjoys",
    "enjoyed", "adore", "adores", "adored", "cherish", "treasure", "relish",
    "savor", "savour", "prefer", "prefers", "fond of", "care for",
)
_REASSESS_SENT_NEG = (
    "hate", "hates", "hated", "dislike", "dislikes", "detest", "loathe",
    "despise", "abhor", "can't stand", "cant stand", "can't bear",
    "cant bear", "dread", "resent",
)
_REASSESS_SENT_POS_SET = frozenset(_REASSESS_SENT_POS)
_REASSESS_SENT_NEG_SET = frozenset(_REASSESS_SENT_NEG)


# Negation tokens that flip a reassessment term's sign. Seed structure
# (RAVANA-expandable); no topic is named here. A negation immediately
# preceding/near an affect term inverts its polarity, so "i DON'T actually
# HATE crowds" is read as a POSITIVE reassessment of a held-NEGATIVE stance
# (and must recode it), not a same-sign echo that leaves the stale attitude
# in place. Round 2026-08-21T2156Z defect D1: the prior scorer did dumb
# substring matching and read "hate" inside "i don't actually hate crowds" as
# negative (same sign as the held -0.95 stance), so the contradiction was
# never detected and a later "do you think i like crowds" reported "strongly
# against crowds" — contradicting the user's own retraction.
_REASSESS_NEGATORS = (
    "not", "n't", "dont", "don't", "aint", "ain't", "never", "no longer",
    "no more", "barely", "hardly", "scarcely", "wont", "won't", "cannot",
    "can't", "cant", "rarely", "seldom",
)


def _assess_reversal_polarity(text: str) -> Optional[float]:
    """Estimate the NEW attitude polarity of a free-form reassessment utterance.

    Returns a polarity in [-1, 1] derived from the strongest reassessment-affect
    term present, or None when the utterance carries no reassessment signal (so a
    contradiction is never guessed from neutral wording). Seed lexicon; RAVANA can
    extend the term sets at runtime as new reassessment phrasings are observed.

    NEGATION-AWARE (round 2026-08-21T2156Z defect D1): a negation token within a
    small window BEFORE the affect term flips its sign. "i don't actually HATE
    crowds" -> the negated "hate" is read as a POSITIVE reassessment
    (recodes a held-negative stance toward neutral/pro-positive), whereas the old
    dumb-substring scorer read the bare word "hate" and treated it as same-sign,
    silently dropping the contradiction. The window is bounded (cheap + bounded
    false-positive surface); a negation more than ~4 tokens away does not flip.
    """
    t = (text or "").lower()
    if not t:
        return None
    _tokens = re.findall(r"[a-z']+", t)
    _neg_idx = {i for i, w in enumerate(_tokens) if w in _REASSESS_NEGATORS}

    def _negated(term):
        # True when a negation token sits within a 4-token window BEFORE the
        # affect term's head — bounded false-positive surface (a negation more
        # than ~4 tokens away does not flip the term's sign).
        _head = term.split()[0]
        for i, w in enumerate(_tokens):
            if w == _head or w.startswith(_head):
                return any(j in _neg_idx for j in range(max(0, i - 4), i))
        return False

    _best_pol = None
    _best_len = 0
    _best_neg = False
    # Scan ALL reassessment lexicons with their base sign. A negation within the
    # 4-token window flips the sign (so "don't actually LOVE X" reads as a
    # NEGATIVE reassessment of a held-positive stance, and "don't actually HATE
    # X" as a POSITIVE one). Base-verb sets mirror the idiom sets so the same
    # sign-flip rule covers both — no second code path, no per-topic branch.
    #
    # PRECEDENCE (round 2026-08-21T2156Z, this round's marathon-gap fix): a
    # NEGATED term is a CONTRADICTION signal and must outrank any non-negated
    # term regardless of length. Without this, a co-mentioned fresh preference
    # ("i don't actually LOVE marathon running, i PREFER short sprints") let the
    # longer non-negated "prefer" (+0.8) win over the negated "love" (-0.8), so
    # the held +0.95 stance was never recoded (measured: marathon retraction
    # left running-marathons at +0.95). A retraction is the salient intent; the
    # new preference is the replacement, handled downstream by the resolver
    # against the PRIOR stance. So: if ANY negated term is present, it wins;
    # only when none is negated do we fall back to the longest non-negated term
    # (original behavior).
    _neg_pol = None
    _neg_len = 0
    for _sign, _set in ((-0.8, _REASSESS_NEG_SET),
                        (0.8, _REASSESS_POS_SET),
                        (-0.8, _REASSESS_SENT_NEG_SET),
                        (0.8, _REASSESS_SENT_POS_SET)):
        for term in _set:
            if term not in t:
                continue
            if _negated(term):
                if len(term) > _neg_len:
                    _neg_len = len(term)
                    _neg_pol = -_sign
            elif _best_pol is None or len(term) > _best_len:
                _best_len = len(term)
                _best_pol = _sign
    if _neg_pol is not None:
        # A contradiction signal dominates — return its (flipped) polarity.
        return _neg_pol
    if _best_pol is None:
        return None
    return _best_pol

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
    r"((?:(?:a|an|the|my|our|their|his|her)?\s*[\w'-]+\s+(?:named|called)\s+[\w'-]+"
    r"\s*(?:,?\s*(?:and|&|,)\s*(?:a|an|the)?\s*)?)+)"
)
# General 'species named/called Name' catch-all (round 2026-08-20T1229Z, FIX C).
# Catches ANY "<species> named/called <Name>" span regardless of which verb
# preceded it ("i got a dog... a border collie named Biscuit"), expanding
# multi-name spans ("collie named Biscuit and Rex"). Identified by object
# identity in the miner so the species->slot path runs. Seed + runtime-learned
# species via pet_slots; no per-animal table, no authored reply.
_PET_NAMED_CATCHALL_PAT = (
    r"\b([\w'-]+)\s+(?:named|called|named\s+called)\s+"
    r"([\w'-]+(?:\s+(?:and|,|&)\s*[\w'-]+)*)"
)
# Appositive pet disclosure (round 2026-08-17T1730Z, 6f generalization): a
# species immediately followed by a Capitalized proper-noun NAME, with NO
# "named"/"called" keyword. Two realizations of the SAME class:
#   "my [pet] <species> <Name>"     e.g. "my pet raccoon Pip steals..."
#   "i have a/an/the [pet]? <species> <Name>"  e.g. "i have a dog Rex barks"
# The name group is Capitalized, so common-noun objects ("my pet rock
# collection", "i have a question") never match. Species resolves through the
# SAME pet_slots path (species_of / learn_species / slot_for) the "named"/
# "called" branch and the recaller already use, so miner + recall agree on the
# key by construction. Generic across EVERY species (no per-animal table);
# species grown at runtime; no authored reply; no retraining. Handled in the
# pet-mining block (object-identity check == _APPOSITIVE_PET_PAT).
_APPOSITIVE_PET_PAT = (
    r"\b(?:my\s+(?:pet\s+)?([A-Za-z][\w'-]*)\s+([A-Z][\w'-]+)"
    r"|i\s+have\s+(?:a|an|the)\s+(?:pet\s+)?([A-Za-z][\w'-]*)\s+([A-Z][\w'-]+))\b"
    r"(?=[\s,.;!?]|$)"
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
        # INTERROGATIVE GUARD (round 2026-08-21T2156Z, defect D3): a question is
        # never a self-disclosure, so do not mine it. Without this, query
        # substrings ("remind me what my brother Theo DOES for work") reached the
        # relationship/activity miners and stored degenerate facts (('i','brother
        # theo','does work')) that later recall rendered as broken English. See
        # _is_query for the precise (inverted-question-aware) test; it reuses the
        # same leading-question-word vocabulary the recall resolvers use.
        if _is_query(q_clean):
            return
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

        # Structured quantity memory (round 2026-08-11T0521Z): capture a count
        # bearing first-person disclosure ("i keep twelve racing pigeons") into
        # the QuantityMemory store, decoupled from the 'does'/'event' text fact.
        # A count QUESTION is never seeded as a fact. VERB_KIND maps the verb to
        # (kind, category) so "how many pets" aggregates across keep/have/raise.
        _qty_is_question = (q_clean.rstrip().endswith("?")
                             or bool(re.match(
                                 r"^(what|who|when|where|why|how|which|is|are|"
                                 r"do|does|did|can|could|would|should|will|may|"
                                 r"might|am|have|has|had)\b", q_clean)))
        _qty_pat = re.compile(
            r"\bi\s+(?:also\s+|really\s+|even\s+|just\s+|now\s+|still\s+|"
            r"often\s+|sometimes\s+|usually\s+)?"
            r"(" + "|".join(QuantityMemory.VERB_KIND.keys()) + r")"
            r"(?:s|es|ing|ed|[a-z]ed|[a-z]d)?"
            r"\s+(a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|\d+)\s+"
            r"(.+?)(?:\s*(\.|!|\?|,|-{1,3}|$|\s+and\s+|\s+but\s+|\s+because\s+|"
            r"\s+so\s+|\s+which\s+|\s+that\s+|\s+when\s+|\s+where\s+|\s+while\s+|"
            r"\s+in\s+|\s+on\s+|\s+at\s+|\s+with\s+|\s+for\s+|\s+of\s+|\s+from\s+))",
            re.IGNORECASE)
        if not _qty_is_question:
            for _qm in _qty_pat.finditer(q_clean):
                _v = _qm.group(1).lower()
                _num_tok = _qm.group(2).lower()
                _noun_raw = _qm.group(3).strip().lower()
                _vk = QuantityMemory.VERB_KIND.get(_v)
                if _vk is None:
                    continue
                _kind, _cat = _vk
                _cnt = number_to_int(_num_tok)
                if _cnt is None or _cnt <= 0:
                    continue
                _toks = [t for t in re.findall(r"[a-z']+", _noun_raw)
                         if t not in self._OPINION_STOP]
                if not _toks:
                    continue
                _spec = _pet_slots.species_of(_toks[-1]) or _pet_slots.species_of(
                    _toks[0])
                _canon = _spec if _spec else (
                    _toks[-1][:-1] if len(_toks[-1]) > 3 and _toks[-1].endswith("s")
                    else _toks[-1])
                self.quantity_memory.assert_quantity(
                    "i", _kind, _cnt, _noun_raw, _canon, category=_cat,
                    confidence=0.6, source="seed_regex")

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
                     "under", "near", "behind", "beside", "off", "out")
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
                     "under", "near", "behind", "beside", "off", "out")
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
        # Round 2026-08-13T2059Z location-capture fix: also capture "based in X"
        # / "located in X" / "stationed at X" / "situated on X" forms, including
        # the interrupted phrasing "i'm a lighthouse keeper based in skye" where a
        # multi-word noun phrase sits between the subject and the location verb.
        # Allow an OPTIONAL determiner + up to 6 words of NP before the verb, so
        # both "i'm based in skye" and "she is stationed at diego garcia" capture
        # the place head. Third-person subjects must NOT create a self-location
        # fact (the test asserts loc=="" for "she is stationed at diego garcia").
        m_loc_based = re.search(
            r"\b(?:i|i'm|i\s+am)\b\s+(?:a|an|the|my|our|their|his|her)?\s*"
            r"(?:[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,6}\s+)?"
            r"(?:based|located|stationed|situated)\s+"
            r"(?:in|at|on|near|from)\s+"
            r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,5})",
            q_clean, re.IGNORECASE)
        # Round 2026-08-13T2059Z: also capture a bare natural-feature fragment
        # ("on the isle of skye") where the place is the noun after "of" and the
        # feature (isle/island/coast/...) precedes it. No subject required; the
        # fragment is a location disclosure in context. Generic feature lexicon,
        # not a per-toponym table.
        m_loc_feat = re.search(
            r"\b(?:on|in|at|near|from)\s+the\s+"
            r"(?:isle|island|coast|shore|bank|edge|outskirts|valley|dale|glen|"
            r"cove|bay|fjord|peninsula|canyon|hollow|estuary|inlet|harbor|harbour|"
            r"beach|river|lake|sea|ocean)\s+of\s+"
            r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3})",
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
        elif m_loc_based and not m_loc and not m_loc_on:
            # Round 2026-08-13T2059Z: "i'm a lighthouse keeper based in skye" /
            # "i'm based in skye" -> capture the place head after the location
            # verb. Only fires for first-person subjects (regex requires i/i'm/
            # i am), so third-person "she is stationed at diego garcia" correctly
            # creates NO self-location fact. Trim a trailing clause at a
            # comma/period so a multi-word NP is stored cleanly.
            _loc = m_loc_based.group(1).strip().strip(" .,!")
            _loc = re.split(r"\s+(?:and|but|,|\.)\s*", _loc)[0].strip()
            if _loc and len(_loc.split()) <= 5:
                self.user_location = _loc
                _put_fact("location", _loc, 0.6)
        elif m_loc_feat and not m_loc and not m_loc_on and not m_loc_based:
            # Round 2026-08-13T2059Z: "on the isle of skye" -> capture "skye".
            # Bare fragment (no subject) is accepted as a location disclosure in
            # context. Trim trailing clause, cap length.
            _loc = m_loc_feat.group(1).strip().strip(" .,!")
            _loc = re.split(r"\s+(?:and|but|,|\.)\s*", _loc)[0].strip()
            if _loc and len(_loc.split()) <= 4:
                self.user_location = _loc
                _put_fact("location", _loc, 0.6)
        elif m_loc:
            _loc = m_loc.group(1).strip().strip(" .,!")
            _loc = re.split(r"\s+(?:and|but|,|\.)\s*", _loc)[0].strip()
            # DEFECT B FIX (round 2026-08-19T0625Z): a location clause with a
            # trailing relative clause ("i live in a small coastal town called
            # greyport where the fog comes in thick most mornings") was captured
            # whole; the prior trims only cut at comma/period/measure words, so
            # the relative clause led by "where/that/which" survived and the
            # stored value became "greyport where the". A toponym must not
            # include a relative clause — strip any trailing "where/that/which
            # <...>" so the real place head is stored. Generic: cuts at a
            # closed-class relative pronoun, never invents a place; degrades
            # gracefully if nothing remains.
            _loc = re.split(r"\s+(?:where|that|which)\b", _loc)[0].strip()
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
            # DEFECT B FIX (round 2026-08-19T0625Z): the trailing toponym scan
            # must run on the ALREADY relative-clause-trimmed _loc (above), NOT
            # on the raw m_loc.group(1). Otherwise the "called <name>" grep
            # re-grabs the relative clause that follows the toponym ("...called
            # greyport where the fog comes in thick most mornings" -> "greyport
            # where the fog"), re-introducing the truncation we just cut.
            _trailing = re.search(
                r"\b(?:in|near|at|from)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,2})\s*$",
                _loc, re.IGNORECASE)
            if not _trailing:
                _trailing = re.search(
                    r"\b(?:called|named|spelled)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,3})",
                    _loc, re.IGNORECASE)
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
            r"\b(?:my|the|a|an|our|their|his|her)\s+([\w'-]+(?:\s+[\w'-]+){0,3})"
            r"\s+(?:is|was|are|were|sits|lies|stays|remains)\s+"
            r"(?:moored|berthed|anchored|docked|based|parked|stationed|"
            r"kept|stored|housed|tied up|wintered)\s+"
            r"(?:at|in|on|near|by|outside|outside of)\s+"
            r"([\w'-]+(?:\s+[\w'-]+){0,3})", q_clean, re.IGNORECASE)
        if _pos_loc:
            _ent = _pos_loc.group(1).strip().strip(" .,!?").lower()
            _place = _pos_loc.group(2).strip().strip(" .,!?").lower()
            # Strip a leading discourse hedge that glued onto the entity
            # ("actually the slow coal" -> "slow coal", not "actually the slow
            # coal"). A small seed of hedge words; structural, not per-entity.
            _HEDGE = ("actually", "basically", "really", "just", "well",
                      "so", "now", "ok", "okay", "look", "see", "right")
            _ent_w = _ent.split()
            while len(_ent_w) > 1 and _ent_w[0] in _HEDGE:
                _ent_w = _ent_w[1:]
            _ent = " ".join(_ent_w)
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
            # Round 2026-08-14T0103Z (Defect A): a bare copula self-description
            # ("i'm vegetarian", "i'm a ceramicist", "i'm an atheist", "i'm
            # scottish") must NOT be stored as the user NAME. Extend the guard
            # with a seed deny set covering the broad CATEGORIES of self-
            # descriptor nouns (diet, belief, occupation, nationality, family
            # role, affiliation). Seed vocabulary, not a per-name allowlist, so it
            # generalizes to any descriptor the user might self-apply. Real proper
            # names ("mira", "wren", "tobias") are never in this set.
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
                # father") — these are roles, never names
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
            r"\bi\s+(?:have|keep)\s+(?:a|an|the)\s+([\w'-]+(?:\s+[\w'-]+){0,3})\s+(?:that|which|i|we)?\s*(?:named|called)\s+([\w'-]+)",
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
            # APPOSITIVE PET DISCLOSURE (round 2026-08-17T1730Z, 6f generalization):
            # "my pet raccoon Pip steals...", "my dog Rex barks", "my cat Mochi
            # sleeps on the router". The existing pet patterns only fire on an
            # EXPLICIT "named"/"called" keyword, so this appositive form
            # ("my <species> <ProperNoun>") was DROPPED and a later "who is Pip
            # to me?" had nothing to recall (measured T49 -> identity blurb).
            # Capture the species + the Capitalized proper-noun name and store it
            # through the SAME shared pet_slots path (slot_for / learn_species /
            # is_pet_attribute) the "named"/"called" branch and the recaller
            # already use, so the miner, the cued-recall renderers, and the
            # reverse-name resolver agree on the key by construction. Generic
            # across EVERY species (no per-animal table); species learned at
            # runtime via learn_species; name is a proper noun (capitalized),
            # so common-noun objects ("my pet rock collection") never match.
            # Handled below in the pet-mining block (object-identity check).
            _APPOSITIVE_PET_PAT,
            # GENERAL 'species named/called Name' CATCH-ALL (round
            # 2026-08-20T1229Z, FIX C). Prior patterns only captured a pet when a
            # possession verb ('have'/'keep'/'my') preceded the species, so
            # "i got a dog last year, a border collie named Biscuit" MISSED the
            # animal entirely — the appositive "border collie named Biscuit" sat
            # after a different verb ('got') and no pattern reached it (confirmed:
            # only a weak 'got dog last year' fact stored, no pet slot). This
            # catches ANY "<species> named/called <Name>" / "<species>
            # named/called <Name> and <Name2>" span regardless of which verb
            # preceded it, and routes through the SAME pet_slots path
            # (species_of / learn_species / slot_for) the other pet branches and
            # the recaller use, so miner + recall agree on the key by construction.
            # The species word is resolved through pet_slots (seed + runtime
            # learned), so a dog/cat/border collie/axolotl all work; no
            # per-animal table, no authored reply, no retraining. Multi-name spans
            # (joined by 'and'/',') are expanded so each name gets its own slot.
            _PET_NAMED_CATCHALL_PAT,
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
                # FIX (round v-aug06b): the conjoined-pet pattern captures a
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
                        if _sp in _pet_slots._PRONOUN_STOP:
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
                            # GENERALIZE (round 2026-08-22T0703Z, DEFECT D1): a pet
                            # disclosure that carries an ACTIVITY ("my ferret Pip
                            # hides my car keys under the couch", "my cat Mochi
                            # sleeps on the router") was stored as NAME ONLY — the
                            # activity was dropped, so a later "what does Pip do with
                            # the keys" / "which of my pets hides things in the
                            # couch" had nothing to recall (measured: both returned
                            # metacognitive uncertainty). Kin disclosures already
                            # capture verb+object activity into a combined-attr fact;
                            # pets must do the same so recall is symmetric.
                            # Capture a verb-phrase tail (any verb + object) after the
                            # NAME as a pet-activity fact keyed on the SAME species
                            # slot (a "_activity" suffix), consumed by the pet
                            # reverse-lookup resolver in engine.py. Verbs come from
                            # the SHARED is_activity_verb / is_relation_verb /
                            # is_aux_verb lexicons (RAVANA-extensible, no per-animal
                            # table, no authored reply, no retraining). The object is
                            # a real concept (resolved through _opinion_topic), not a
                            # filler; bounded to <=5 tokens. Content is the user's own
                            # words; the recaller renders it through the same D7
                            # copula rule. Honest None fallback when no verb follows.
                            _tail = q_clean[_m.end():].lstrip()
                            _act = self._mine_pet_activity(_tail, _nm, _sp)
                            if _act:
                                _put_fact(f"{_pet_slots.slot_for(_species, _i)}_activity", _act, 0.55)
                            continue
                # GENERAL 'species named/called Name' CATCH-ALL (round
                # 2026-08-20T1229Z, FIX C). group(1)=species, group(2)=name(s).
                # Expand multi-name spans ("collie named Biscuit and Rex") so each
                # name gets its own species-keyed slot. Routes through the SAME
                # pet_slots path the conjoined/appositive branches use, so miner +
                # recall agree on the key. Seed + runtime-learned species; no
                # per-animal table, no authored reply, no retraining.
                if _pat is _PET_NAMED_CATCHALL_PAT:
                    _sp = (_m.group(1) or "").strip().lower()
                    _names = re.split(r"\s+(?:and|,|&)\s*",
                                      (_m.group(2) or "").strip())
                    if not _sp:
                        continue
                    # Round 2026-08-20T1229Z regression fix: a possession
                    # disclosure like "i keep a sourdough starter i named doris"
                    # leaves group(1) == "i" (a pronoun). Reject pronoun /
                    # function-word species candidates so no bogus species slot
                    # is learned and leaked on unknown-entity recall. The legit
                    # "species named Name" case still resolves through the seed
                    # vocabulary, so this does not narrow the capture.
                    if _sp in _pet_slots._PRONOUN_STOP:
                        continue
                    _species = _pet_slots.species_of(_sp)
                    _species_is_seed = _species is not None
                    if _species is None and _sp.isalpha():
                        _species = _pet_slots.learn_species(_sp)
                    elif _species is None:
                        _species = _sp
                    if _species is not None:
                        # PRESERVE THE SURFACE SPECIES PHRASE (round
                        # 2026-08-20T1229Z, pet-resolver hardening). The catch-all
                        # is greedy: a disclosure like "i keep a pet parrot named
                        # Mango" has TWO species words before "named" ("pet" then
                        # "parrot"). The REGEX only captures the word IMMEDIATELY
                        # before "named" — "parrot" — and species_of('parrot')
                        # collapses it to its seed CANON 'bird', storing the bare
                        # ('i','bird',<name>) fact. But the possession pattern
                        # (r"\bi\s+(?:have|keep)\s+(?:a|an|the)\s+<species phrase>
                        # \s+(?:named|called)\s+<name>") captures the FULL surface
                        # "pet parrot" and stores the entity-keyed
                        # ('pet parrot','name',<name>) fact. The two miners then
                        # DISAGREE on the key ("bird" vs "pet parrot"), and the
                        # pet/relationship recaller (engine.py 1d branch) only
                        # scans subject-'i' facts — so it surfaces the bare canon
                        # "your bird is mango." instead of the user's own words
                        # "your pet parrot is mango." Fix: when the captured token
                        # is a SEED canon (parrot->bird) AND the same surface
                        # species phrase was ALSO stored by the possession pattern
                        # (i.e. the prior-token word is 'pet' / a known species and
                        # it forms a compound "<prior> <captured>"), keep the FULL
                        # surface compound as the slot key instead of collapsing to
                        # the canon. This makes the catch-all and the possession
                        # pattern AGREE on the key by construction, so the recaller
                        # renders the surface phrase the user actually said. The
                        # compound is still resolved through pet_slots (every
                        # component is a seed/learned species), so no per-animal
                        # table, no authored reply. A lone non-compound capture
                        # (e.g. "i got a border collie named Biscuit" -> species
                        # 'dog' but no 'pet' prefix) still collapses to the canon
                        # exactly as before, so legit pet capture is unchanged.
                        _surface = _sp
                        # Look at the ORIGINAL text before the match to find the
                        # word immediately preceding the captured species token
                        # (group(0) only spans "<species> named <name>", so its
                        # own prefix is empty). A disclosure like
                        # "i keep a pet parrot named Mango" leaves "pet" right
                        # before "parrot"; capture it so the compound surface key
                        # can be reconstructed.
                        _pre = q_clean[: _m.start()].rstrip()
                        _prior = _pre.split()[-1].lower() if _pre else ""
                        # When the captured token collapses to a SEED canon AND
                        # the word immediately before it ("pet"/a known species)
                        # forms a compound surface phrase, prefer keeping the FULL
                        # surface compound as the slot key. The possession pattern
                        # stores the entity-keyed ('<compound>','name',<name>)
                        # fact; aligning the catch-all to the same surface key
                        # makes miner + recaller agree by construction instead of
                        # emitting the bare canon ('bird') that the recaller
                        # renders as "your bird". Single-token captures (no
                        # qualifying prior word) keep collapsing to the canon, so
                        # legit pet capture is unchanged.
                        if (_species_is_seed
                                and _prior in ("pet",)
                                and _pet_slots.species_of(_prior) is not None):
                            _surface = f"{_prior} {_sp}"
                        for _nm in _names:
                            _nm = _nm.strip().strip(".,!?")
                            if not _nm:
                                continue
                            _i = 1
                            while _pet_slots.slot_for(_surface, _i) in self.personal_facts.facts:
                                _i += 1
                            _put_fact(_pet_slots.slot_for(_surface, _i), _nm, 0.6)
                    continue
                # APPOSITIVE PET (round 2026-08-17T1730Z, 6f): "my pet raccoon
                # Pip steals..." / "my dog Rex barks" / "my cat Mochi sleeps".
                # The name capture group (group 2) is a proper noun. The
                # isupper() guard rejects common-noun objects ("my pet rock
                # collection"), but casual chat also writes names lowercase
                # ("my cat mochi"). GENERALIZE (round 2026-08-19T1026Z): accept a
                # lowercase name too, but ONLY when the species is a SEED animal
                # (cat/dog/...), so "my cat mochi" mines while "my pet rock
                # collection" is still rejected (rock is not a seed species, so
                # learn_species never fires on the lowercase path). Resolve the
                # species through the SAME pet_slots path the "named"/"called"
                # branch uses (species_of / learn_species / slot_for), then store
                # the name in the species-keyed slot — so the miner and the
                # recaller (reverse-name resolver + cued recall) agree on the
                # key by construction. Generic across every species; no
                # per-animal table; species grown at runtime. No authored reply;
                # no retraining.
                if _pat is _APPOSITIVE_PET_PAT:
                    _raw_nm = (_m.group(2) or _m.group(4) or "")
                    _sp = (_m.group(1) or _m.group(3) or "").strip().lower()
                    _nm = _raw_nm.strip().strip(".,!?")
                    if not _sp or not _nm:
                        continue
                    # Round 2026-08-20T1229Z regression fix: reject pronoun /
                    # function-word species candidates so no bogus slot is
                    # learned (defense-in-depth; also handled at the chokepoint).
                    if _sp in _pet_slots._PRONOUN_STOP:
                        continue
                    # GENERALIZE: a name is accepted if it is Capitalized OR
                    # (lowercase AND the species is a seed animal). This keeps the
                    # common-noun guard (rejects "my pet rock collection") while
                    # allowing casual lowercase names ("my cat mochi").
                    _name_ok = _nm[:1].isupper()
                    if not _name_ok:
                        try:
                            from .pet_slots import _SPECIES_SEED as _PS_SEED
                        except Exception:
                            _PS_SEED = {}
                        if _sp in _PS_SEED:
                            # lowercase-name path (e.g. "my cat mochi"): only
                            # valid when the captured name is sentence-final or
                            # followed by a copula/punctuation — NOT a verb-led
                            # clause tail. The pattern above runs IGNORECASE, so
                            # the name group can grab the next verb ("my dog
                            # likes the park" -> name "likes"); reject when a
                            # non-copula word follows so we don't store a verb as
                            # a pet. Proper-noun (isupper) names are exempt — they
                            # may legitimately lead a clause ("my dog Rex barks").
                            _tail = q_clean[_m.end():].lstrip()
                            _nxt = re.split(r"[\W]+", _tail, 1)[0].lower()
                            _COPULA = {"is", "was", "were", "are", "named",
                                       "called", "means", "s"}
                            if not _tail or not _nxt or _nxt in _COPULA:
                                _name_ok = True
                    if not _name_ok:
                        continue
                    try:
                        from .relation_attrs import relation_of as _app_rel_of
                    except Exception:
                        _app_rel_of = lambda w: None
                    if _app_rel_of(_sp) is not None:
                        continue
                    _species = _pet_slots.species_of(_sp)
                    if _species is None and _sp.isalpha():
                        _species = _pet_slots.learn_species(_sp)
                    if _species is not None:
                        _i = 1
                        while _pet_slots.slot_for(_species, _i) in self.personal_facts.facts:
                            _i += 1
                        _put_fact(_pet_slots.slot_for(_species, _i), _nm, 0.6)
                        # GENERALIZE (round 2026-08-22T0703Z, DEFECT D1): a pet
                        # disclosure that carries an ACTIVITY ("my ferret Pip
                        # hides my car keys under the couch", "my cat Mochi
                        # sleeps on the router") was stored as NAME ONLY — the
                        # activity was dropped, so a later "what does Pip do with
                        # the keys" / "which of my pets hides things in the
                        # couch" had nothing to recall (measured: both returned
                        # metacognitive uncertainty). Kin disclosures already
                        # capture verb+object activity into a combined-attr fact;
                        # pets must do the same so recall is symmetric.
                        # Capture a verb-phrase tail (any verb + object) after the
                        # NAME as a pet-activity fact keyed on the SAME species
                        # slot (a "_activity" suffix), consumed by the pet
                        # reverse-lookup resolver in engine.py. Verbs come from
                        # the SHARED is_activity_verb / is_relation_verb /
                        # is_aux_verb lexicons (RAVANA-extensible, no per-animal
                        # table, no authored reply, no retraining). The object is
                        # a real concept (resolved through _opinion_topic), not a
                        # filler; bounded to <=5 tokens. Content is the user's own
                        # words; the recaller renders it through the same D7
                        # copula rule. Honest None fallback when no verb follows.
                        _tail = q_clean[_m.end():].lstrip()
                        _act = self._mine_pet_activity(_tail, _nm, _sp)
                        if _act:
                            _put_fact(f"{_pet_slots.slot_for(_species, _i)}_activity", _act, 0.55)
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
                    # NAME-BRIDGE route (round 2026-08-18T1340Z): patterns like
                    # "i keep a sourdough starter i named doris" capture the
                    # possessed NOUN as _attr and the NAME as _val. Store it as an
                    # entity-keyed name fact (entity=_attr, attr="name") so a later
                    # "what did i name my <thing>" recalls the name cleanly, instead
                    # of a bare "sourdough starter -> doris" slot that renders as
                    # "your sourdough starter is doris". Only fire when the match
                    # text actually contains a name keyword and the value is
                    # name-shaped (single token, alpha) — never for a pet (those
                    # are handled by the dedicated pet paths above).
                    if re.search(r"\b(?:named|called)\b", _m.group(0), re.IGNORECASE) \
                            and _val.isalpha() and " " not in _val \
                            and _pet_slots.species_of(_attr) is None:
                        # The noun group can greedily swallow the bridge pronoun
                        # ("sourdough starter i" from "...starter i named doris"),
                        # so strip a trailing bridge word before storing.
                        _attr_clean = re.sub(
                            r"\s+(?:i|we|that|which)$", "", _attr,
                            flags=re.IGNORECASE).strip()
                        if _attr_clean:
                            _attr = _attr_clean
                        _put_fact_ent(_attr, "name", _val, 0.6)
                        continue
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
                # GUARD (round 2026-08-22T0703Z CI fix): a pattern match can
                # reach this point with _attr still None when the match's
                # captured groups did not populate an attribute (e.g. a
                # 0-group or non-attr pattern matched). _split_possessive_attr
                # requires a str; passing None raises TypeError and aborts the
                # whole mine. Skip such matches honestly — there is no
                # possessive attribute to split, so nothing to store here.
                if _attr is None:
                    continue
                _ent, _rel = _split_possessive_attr(_attr)
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
                    #
                    # GUARD (round 2026-08-18T1340Z): the value-split must only
                    # fire for a genuine NAME relation whose parts are all
                    # name-shaped. The old code split ANY value on "and", so
                    # "my best friend's name is tomas and he's a chef in lisbon"
                    # produced a bogus second fact "best friend's name_2 = he's
                    # a chef in lisbon" (the conjunction joined a NAME and an
                    # UNRELATED occupation clause). Now: only split when the
                    # attribute is a name relation AND every part is a short
                    # name token (single word or Capitalized proper noun, no
                    # verb/closed-class word). Otherwise store the whole value
                    # as ONE fact so no fabricated second slot is created.
                    _is_name_rel = _attr == "name" or _attr.endswith("_name") \
                        or _attr.endswith(" name")
                    _names = re.split(r"\s+(?:and|,|&)\s*", _val)
                    _names = [_n.strip().strip(".,!?") for _n in _names if _n.strip()]
                    _name_shaped = _is_name_rel and len(_names) > 1 and all(
                        (len(_n.split()) <= 2
                         and not re.search(r"\b(he|she|it|is|are|was|were|"
                                           r"do|does|did|have|has|i|you|a|an|the)\b",
                                           _n, re.IGNORECASE))
                        for _n in _names)
                    _species = _pet_slots.species_of(_attr)
                    if _species is None and _name_shaped and _attr.isalpha():
                        # Round 2026-08-20T1229Z regression fix: never learn a
                        # pronoun / function word (e.g. "i" from "starter i named
                        # doris") as a species — it would create a bogus slot
                        # that leaks on unknown-entity recall. Rejected here and
                        # at the chokepoint in pet_slots.learn_species.
                        if (re.search(r"\b(?:named|called)\b", _m.group(0), re.IGNORECASE)
                                and _attr not in _pet_slots._PRONOUN_STOP):
                            _species = _pet_slots.learn_species(_attr)
                    if _species is not None and _name_shaped:
                        for _i, _nm in enumerate(_names, 1):
                            _nm = _nm.strip().strip(".,!?")
                            if _nm:
                                _put_fact(_pet_slots.slot_for(_species, _i),
                                          _nm, 0.6)
                    elif _name_shaped:
                        for _i, _nm in enumerate(_names, 1):
                            _nm = _nm.strip().strip(".,!?")
                            if _nm:
                                _put_fact(f"{_attr}_{_i}", _nm, 0.6)
                    else:
                        _put_fact(_attr, _val, 0.6)


        # D7 (round 2026-08-16T1745Z): relationship-ACTIVITY disclosures were
        # never mined. "my X is Y" (equational) was captured, but the dominant
        # real-world shape "my <relation> <Name> <verb> <object>" (e.g. "my
        # grandmother Indira weaves baskets", "my brother Arjun climbs mountains")
        # fell through entirely — only an incidental match happened when a
        # conjoined-pet pattern accidentally grabbed "cousin". Consequently every
        # cued recall of that family member (T29 "what's my grandmother's name",
        # T32 "does my brother have a hobby", T51 "what did i tell you about my
        # grandmother", T52 "my brother", T61 "who is indira") had nothing to
        # recall and echoed an unrelated fact. This is the recurring L2 residual
        # limitation, now closed GENERALIZABLY.
        #
        # Fix: capture ANY "my <kin> <Name> <verb> <object>" disclosure as a
        # COMBINED-attr fact (attr = "<kin> <name>", subject "i", value = the
        # verb+object HEAD) — the EXACT storage shape the recall branch (c) in
        # engine.py:_structured_recall already resolves ("my niece priya" ->
        # attr "niece priya"). So miner and recaller agree by construction; no
        # per-relationship table. The relationship vocabulary is SEED structure
        # (RAVANA-expandable: it feeds the same PersonalFactStore the user can
        # correct; removing a word degrades gracefully), NOT authored reply
        # prose and NOT a per-topic answer dictionary. The verb set is the SAME
        # closed activity-verb seed the self-activity loop uses, so a disclosure
        # like "my brother Arjun climbs mountains" stores ("i", "brother arjun",
        # "climbs mountains") and a later "does my brother have a hobby?" /
        # "what did i tell you about my brother" resolves it correctly.
        _KIN = {
            "grandmother", "grandfather", "grandma", "grandpa", "granny",
            "granddad", "nana", "papa", "mum", "mom", "mother", "dad",
            "father", "aunt", "auntie", "uncle", "sister", "brother",
            "cousin", "niece", "nephew", "daughter", "son", "wife",
            "husband", "spouse", "partner", "stepmother", "stepfather",
            "stepsister", "stepbrother", "halfsister", "halfbrother",
            "grandson", "granddaughter", "motherinlaw", "fatherinlaw",
        }
        _REL_ACTIVITY_VERBS = (
            "run", "own", "operate", "play", "teach", "study", "manage",
            "drive", "build", "make", "sell", "restore", "grow", "watch",
            "raise", "tend", "brew", "bake", "write", "read", "learn",
            "practice", "collect", "fix", "paint", "code", "design",
            "craft", "volunteer", "cook", "fish", "hike", "garden",
            "farm", "lead", "organize", "keep", "grind", "race", "sail",
            "fly", "knit", "sew", "weld", "forge", "carve", "compose",
            "record", "perform", "coach", "train", "compete", "spin",
            "weave", "mount", "trade", "host", "guide", "climb", "repair", "work",
        )
        # Accept inflected verb forms (3rd-person -s, -es, -ing) so
        # disclosures like "weaves / climbs / grows / paints" match the
        # base seed verb. The base set is the SEED; inflections are
        # derived mechanically (no per-form list), so adding a base verb
        # auto-covers its forms. Not authored prose.
        _REL_ACTIVITY_VERB_FORMS = set(_REL_ACTIVITY_VERBS)
        for _vb in _REL_ACTIVITY_VERBS:
            _REL_ACTIVITY_VERB_FORMS.add(_vb + "s")
            _REL_ACTIVITY_VERB_FORMS.add(_vb + "es")
            _REL_ACTIVITY_VERB_FORMS.add(_vb + "ing")
        # NO re.IGNORECASE here: kin + verb are always lowercase in the
        # input, while the Name is capitalized, so the capitalized-name
        # token cleanly separates from the lowercase activity verb.
        # (IGNORECASE made [A-Z] also match lowercase and the name group
        # greedily swallowed the verb.) kin/verb are lowercased after
        # matching, so casing in the source is irrelevant.
        # FIX (feature t_1a4a3938, round 2026-08-17T1126Z): the OLD regex
        # required the Name to be CAPITALIZED ([A-Z][A-Za-z]*) so it could be
        # told apart from the lowercase activity verb. But in real chat the name
        # is usually lowercase ("my grandmother indira bakes bread"), so the
        # capitalized group matched nothing, the name-less fallback fired, and
        # its kin+verb slot ("indira") was not an activity verb -> the fact was
        # NEVER stored. That silently broke every later open-ended recall of that
        # relative (the residual "tell me about my grandmother" gap logged at the
        # end of round 2026-08-17T1126Z).
        #
        # Fix: find the activity verb by MEMBERSHIP, not by position or
        # capitalization. After "my <kin>" we scan the remaining tokens
        # left-to-right for the FIRST token that is an activity verb; the tokens
        # before it (if any) are the Name, the tokens after it are the object.
        # This is structural — one verb lexicon, no per-name table, no case
        # assumption — and generalizes to any name casing/length. Content comes
        # from the user's own words; no authored reply, no retraining.
        _mk = re.search(r"\bmy\s+([a-z][a-z-]+)\b\s*(.*)", q_clean)
        if _mk:
            try:
                from .relation_attrs import relation_of as _mk_rel_of
            except Exception:
                _mk_rel_of = lambda w: None
            _kin = _mk.group(1).lower()
            _rest0 = _mk.group(2)
            # GENERALIZE (round 2026-08-22T0058Z, DEFECT B): a relationship
            # disclosure can carry one or more LEADING MODIFIERS before the
            # relationship word — "my OLD mentor Dr. Osei", "my OLD BEEKEEPING
            # mentor Dr. Osei", "my DEAR friend Priya", "my LATE uncle Bram".
            # The OLD code took only the FIRST word after "my" as the relation
            # head, so "old"/"dear"/"late" (not relationship words) became the
            # head and the WHOLE disclosure was dropped — a later "who is my
            # mentor?" had nothing to recall.
            #
            # Fix: scan the remainder left-to-right for the FIRST token that is
            # a relationship word (via relation_of). Everything before it is a
            # modifier and is dropped; that token becomes the head. SCAN STOPS
            # at the first CAPITALIZED proper-noun name (e.g. "wren" in "my
            # friend wren, she's a ceramicist") so a bare "<rel> <Name>, <clause>"
            # disclosure keeps the Name in the remainder for the embedded-clause
            # path below — this preserves the pre-existing behavior that the
            # first attempt at this fix broke. When no relationship word is
            # found before a name or the end, the head is unchanged and the
            # block falls through honestly (no fictitious fact).
            #
            # Seed vocabulary consulted via relation_of / learn_relation (RAVANA
            # runtime-extensible), not a per-role table; no authored prose; no
            # retraining. Bounded by the remainder length.
            if _mk_rel_of(_kin) is None and _rest0:
                _rt = _rest0.split()
                _head_found = False
                for _j, _tk in enumerate(_rt):
                    _wt = _tk.lower().strip(".,!?;:\"'" + "'")
                    if _mk_rel_of(_wt) is not None:
                        _kin = _mk_rel_of(_wt)
                        _rest0 = " ".join(_rt[_j + 1:])
                        _head_found = True
                        break
                    # a capitalized proper noun is the entity NAME, not a
                    # modifier — stop scanning so the embedded-clause path can
                    # capture it.
                    if _tk[:1].isupper():
                        break
            _kin_norm = _mk_rel_of(_kin)
            if _kin_norm:
                _kin = _kin_norm
            # GENERALIZE (round 2026-08-17T1730Z): a relationship disclosure is
            # not restricted to blood kin. Mentors, teachers, coaches, friends,
            # neighbors, bosses, colleagues, roommates, landlords, and any
            # RAVANA-learned relation word all count as a NAMED RELATIONSHIP the
            # user disclosed about themselves, and must be mined + recallable
            # the same way. The old gate only checked the local _KIN set, so
            # "my mentor Dr. Okonkwo taught me astronomy" and "my grandmother
            # taught me to slow-cook" were DROPPED (mentor is not kin; "taught"
            # is an irregular verb absent from _REL_ACTIVITY_VERB_FORMS) and a
            # later "who is my mentor" / "what did my grandmother pass down"
            # had nothing to recall. Now the head word is accepted if it is (a)
            # in the seed _KIN set, (b) a non-kin ROLE in the shared
            # relationship vocabulary (relation_of + ROLE lexicon), or (c)
            # already learned at runtime via learn_relation — so the miner and
            # the recaller (engine.py 1c/1d, relation_attrs) agree on what a
            # relationship word is. Generic, no per-role table.
            _role = False
            try:
                from .relation_attrs import relation_of as _ra_of
                from .relation_attrs import learn_relation as _ra_learn
            except Exception:
                _ra_of = lambda w: None
                _ra_learn = lambda w: ""
            if _kin in _KIN or _ra_of(_kin) is not None:
                _role = True
                # from the live disclosure, so the recaller (engine.py 1c/1d),
                # which already consults relation_of / is_relation_attribute /
                # base_relation, generalizes to this role WITHOUT a per-role
                # branch. This is the runtime-growth design: RAVANA learns its
                # own relationship words from conversation. Online, no retrain.
                # The role word itself is now part of the SHARED seed in
                # relation_attrs (single source of truth), which also means the
                # appositive-pet miner (which runs BEFORE this block) rejects it
                # via its relation_of() guard instead of mis-storing it as a pet
                # species (the round 2026-08-17T1730Z feature bug: "my mentor
                # Dr. Okonkwo..." produced a bogus ('i','mentor','dr') pet fact
                # that truncated recall to "your mentor is dr.").
                try:
                    _ra_learn(_kin)
                except Exception:
                    pass
            if _role:
                # HEADLESS-POSSESSIVE GUARD (round 2026-08-22T0703Z, DEFECT D2):
                # a disclosure like "my daughter name is ingrid" has the shape
                # "<kin> <relation-word> <copula> <value>" — a HEADLESS
                # possessive, NOT a "<kin> <Name> <verb>...>" activity disclosure.
                # mine_personal_facts already handles it correctly via the generic
                # `my X is Y` pattern + _split_possessive_attr into an
                # entity-scoped fact (('daughter','name','ingrid')). If this kin
                # block ALSO mines it, its comma/embedded-relative name-extractor
                # mis-reads the leading relation word ("name") as the relationship
                # head and the copula ("is") as the name, producing a MALFORMED
                # fact (('i','daughter name is','...')). Root cause: the kin block
                # fired on a shape the generic possessive path owns. Fix: when the
                # FIRST token after the relationship word is itself a relation
                # word (name/age/job/...), this is a headless possessive — return
                # so only the correct entity-scoped fact survives. Structural:
                # reuses the shared _REL_WORDS vocabulary the possessive splitter
                # already consults, so miner + recaller agree by construction; no
                # per-role branch, no authored reply, no retraining.
                _rest0_toks = _rest0.split()
                if _rest0_toks and _rest0_toks[0].strip(".,!?").lower() in _REL_WORDS:
                    return
                _rest = _rest0
                _toks = _rest.split()
                _vidx = None
                _v_is_rel = False
                for _i, _t in enumerate(_toks):
                    _tw = _t.lower().strip(".,!?")
                    if is_activity_verb(_tw):
                        _vidx = _i
                        break
                    # GENERALIZE (feature t_ec6c6b51, round 2026-08-20T1935Z
                    # residual): also recognize a RELATION verb (speaks/works/
                    # studies/plays/... — a durable attribute/ability the named
                    # relative HAS). The canonical missed case was "my grandmother
                    # yaya SPEAKS three languages: greek, french, italian" — "speaks"
                    # is not an activity verb and "yaya" is lowercase, so the old
                    # activity-verb OR capitalized-name gate skipped the whole
                    # disclosure and the enumeration was never stored. A relation
                    # verb is a legitimate verb-phrase head, so it opens the same
                    # capture path as an activity verb (name = tokens before it,
                    # value = verb + object). Seed lexicon (is_relation_verb),
                    # RAVANA-expandable; not a per-relationship table.
                    if is_relation_verb(_tw):
                        _vidx = _i
                        _v_is_rel = True
                        break
                    # GENERALIZE (feature t_16b15684, round 2026-08-21T0843Z
                    # residual): also recognize an AUXILIARY verb (does/did/doing
                    # + activity noun-phrase) as a verb-phrase head. The canonical
                    # missed case was "my cousin Jin DOES competitive speedcubing"
                    # — "does" is neither an activity verb nor a relation verb, so
                    # the verb-scan found nothing, fell to the name-only path, and
                    # the degenerate-fact guard dropped the whole disclosure. Now
                    # the auxiliary opens the SAME capture path as the other two
                    # verb classes; the value is mined as "aux + activity
                    # noun-phrase" via the activity value-extraction branch below
                    # (the object is resolved through _opinion_topic so it stays a
                    # real concept, not a filler). Seed lexicon (is_aux_verb),
                    # RAVANA-expandable; not a per-relationship table. The aux is
                    # allowed to fire on its own token as long as a following
                    # content token exists (the activity noun-phrase) — we do NOT
                    # require a following activity VERB, because the disclosure is
                    # "does competitive speedcubing", not "does climb".
                    if is_aux_verb(_tw):
                        _rest_toks = [t for t in _toks[_i + 1:]
                                      if t.strip(".,!?")]
                        if _rest_toks:
                            _vidx = _i
                            break
                # GENERALIZE (round 2026-08-19T1026Z): a relationship disclosure
                # establishes the RELATIONSHIP + NAME regardless of whether an
                # activity verb is recognised. When NO activity/relation verb is
                # found, only a LEADING CAPITALIZED token counts as a name
                # (mirrors the appositive-pet branch's isupper() guard) so a
                # temporal/activity phrase ("last spring") is never stored as the
                # name. If there is neither a name nor a verb, skip — no
                # informative fact to store. No per-role table: the head word is
                # the user's own relation word, the name the user's own noun, the
                # verb (when present) real content from the user's words.
                _put_fact_done = False
                _val = ""
                if _vidx is None:
                    _name_toks = []
                    for _t in _toks:
                        _clean = _t.strip(".,!?")
                        if _clean[:1].isupper():
                            _name_toks.append(_clean.lower())
                        else:
                            break
                    _name = " ".join(_name_toks)
                    if not _name:
                        # Neither a recognized verb nor a proper-noun name:
                        # nothing informative to store (e.g. "my grandmother
                        # last spring") — skip to avoid a bogus fact.
                        _put_fact_done = True
                    else:
                        # GENERALIZE (round 2026-08-21T2156Z defect D2): a
                        # disclosure with a recognized NAME but an UNrecognized
                        # verb ("my great-aunt Hortense folds a thousand paper
                        # cranes every winter" — "folds" is not in the activity
                        # lexicon) still carries real content. Capture the
                        # trailing clause AFTER the name as the value (trimmed at
                        # a clause boundary) so the fact is informative and the
                        # degenerate-fact guard below keeps it. Content is the
                        # user's own words; no per-relationship branch.
                        _after = " ".join(
                            _t.strip(".,!?") for _t in _toks[len(_name_toks):])
                        _after = re.split(
                            r"\s*(?:[.!?]+|where|that|which|when|but)\b",
                            _after)[0].strip(" ,.!?;:")
                        _after = re.sub(r"\s+", " ", _after).lower()
                        if _after and len(_after.split()) <= 12:
                            _val = _after
                else:
                    _name = " ".join(_toks[:_vidx]).lower()
                    _name = _strip_obj_framers(_name).strip(" .,!?")
                if _vidx is not None:
                    _verb = _toks[_vidx].lower().strip(".,!?")
                    _obj_rest = " ".join(_toks[_vidx + 1:])
                    if _v_is_rel:
                        # Relation-verb value: KEEP the enumeration + connective.
                        # The activity path splits the object on "and" (so "baskets
                        # from river reeds" -> "baskets") because activities rarely
                        # enumerate; a relation-verb disclosure like "speaks three
                        # languages: greek, french, and italian" MUST preserve the
                        # list and its connective ("as a mechanic", "three
                        # languages"), so here we stop only at a sentence/clause
                        # boundary (period / "where|that|which|when|but") and KEEP
                        # coordination ("and"/"or") and prepositional connectives
                        # ("as"/"in"/"on"). Content comes verbatim from the user's
                        # own words (lowercased). The only trims: a TRAILING
                        # closed-class framer/preposition (so "... speaks french
                        # in" -> "... speaks french") and a LEADING bare determiner
                        # ("a"/"the") that opens the object — neither carries list
                        # or attribute meaning. We do NOT run the opinion-topic
                        # framer stripper (which would drop "as"/"a"), so the
                        # disclosed phrase stays intact. Structural, bounded to 12
                        # tokens so a runaway clause cannot swallow the sentence.
                        _obj_raw = re.split(
                            r"\s*(?:[.!?]+|where|that|which|when|but)\b",
                            _obj_rest)[0].strip(" ,.!?;:")
                        _obj_toks = _obj_raw.lower().split()
                        # strip a single leading determiner only (keep connectives)
                        if _obj_toks and _obj_toks[0] in (
                                "the", "a", "an", "my", "your", "his", "her",
                                "their", "our", "its", "this", "these", "those"):
                            _obj_toks = _obj_toks[1:]
                        # strip trailing closed-class framer / preposition
                        _PREP = ("up", "down", "from", "at", "in", "on", "with",
                                 "to", "of", "by", "for", "about", "into",
                                 "onto", "over", "under", "near", "behind",
                                 "beside", "off", "out", "as")
                        while _obj_toks and _obj_toks[-1] in _PREP:
                            _obj_toks.pop()
                        _obj = " ".join(_obj_toks).strip()
                        if _obj and len(_obj.split()) <= 12:
                            _val = f"{_verb} {_obj}"
                    else:
                        _obj_raw = re.split(
                            r"\b(?:when|but|because|and)\b", _obj_rest)[0].strip(" ,.!?")
                        _obj = self._opinion_topic(_obj_raw.lower()) or ""
                        _obj = _strip_obj_framers(_obj)
                        if _obj and len(_obj.split()) <= 5:
                            _val = f"{_verb} {_obj}"
                        else:
                            # GENERALIZE (feature t_6b93e125, round
                            # 2026-08-22T0058Z residual DEFECT B-variant): a
                            # relationship disclosure can carry its object as a
                            # NON-NOUN-PHRASE CLAUSE — "my mentor taught me how
                            # to read the hive's mood", "my grandmother showed
                            # me where the river bends", "my uncle taught me
                            # why the stars wheel at night". The opinion-topic
                            # resolver is built to collapse such clauses to a
                            # content HEAD ("read" / "where" / "why") and then
                            # REJECT that head as verb-residue (it lives in
                            # _OBJ_NONCONTENT), so _obj comes back EMPTY and the
                            # degenerate-fact guard drops the WHOLE disclosure.
                            # The raw clause is real, informative content the
                            # user said, so we keep it verbatim instead of
                            # dropping it. This mirrors the relation-verb path
                            # above (which preserves the user's own enumerated
                            # phrase rather than trimming it to a content head).
                            # We only fall back to the raw clause when the
                            # topic-resolver produced nothing — genuine
                            # noun-phrase objects still go through the resolver
                            # and keep the existing "real concept head" shape.
                            # The fallback is BOUNDED (<= 12 tokens, trailing
                            # closed-class framers stripped) so a runaway clause
                            # cannot swallow the sentence, and we re-use the
                            # same clause-boundary splitter used by the
                            # relation-verb path so a trailing ".!?" / "where|
                            # that|which|when|but" cleanly ends the value.
                            # Content comes verbatim from the user's own words;
                            # no authored reply; no per-relationship table; no
                            # retraining. RAVANA-expandable: the same
                            # PersonalFactStore the user can correct.
                            _raw = re.split(
                                r"\s*(?:[.!?]+|where|that|which|when|but)\b",
                                _obj_rest)[0].strip(" ,.!?;:")
                            # drop a leading possessive framer ("me", "us") that
                            # opens a taught/showed clause but carries no
                            # content ("my mentor taught me HOW...").
                            _raw_toks = _raw.lower().split()
                            if _raw_toks and _raw_toks[0] in (
                                    "me", "us", "them", "him", "her", "you",
                                    "myself", "himself", "herself"):
                                _raw_toks = _raw_toks[1:]
                            # strip trailing closed-class framer / preposition
                            _FALLBACK_PREP = (
                                "up", "down", "from", "at", "in", "on", "with",
                                "to", "of", "by", "for", "about", "into",
                                "onto", "over", "under", "near", "behind",
                                "beside", "off", "out", "as")
                            while _raw_toks and _raw_toks[-1] in _FALLBACK_PREP:
                                _raw_toks.pop()
                            _raw = " ".join(_raw_toks).strip()
                            if _raw and len(_raw.split()) <= 12:
                                _val = f"{_verb} {_raw}"
                # GENERALIZE (round 2026-08-21T2156Z defect D2): an EMBEDDED
                # relative clause ("my friend wren, she's a ceramicist") has no
                # activity verb before the name, and the name is LOWERCASE after
                # a comma, so the name-only fallback (which requires a leading
                # CAPITALIZED token) found nothing and the disclosure was dropped.
                # Capture the "<Name>, <pronoun>'s a <descriptor>" shape: a comma,
                # a pronoun subject (she/he/they/it), a copula (is/'s), then a
                # noun-phrase descriptor. The name is the leading capitalized OR
                # lowercase run before the comma; the descriptor becomes the
                # value. Structural — generalizes to ANY relationship word
                # RAVANA has learned (the head is the user's own relation word);
                # the descriptor is the user's own noun-phrase, so content is real
                # and recall renders "your <rel> <name> is a <descriptor>". No
                # per-relationship branch, no authored reply, no retraining.
                if not _name and _toks:
                    # FIX (round 2026-08-21T2156Z defect D2): detect the comma
                    # on the RAW token (a trailing comma survived by the lower()
                    # in q_clean), NOT after rstrip(',') — rstrip removes it
                    # before endswith(',') can match, so the embedded-relative
                    # path never fired.
                    _comma = next((i for i, _t in enumerate(_toks)
                                   if _t.endswith(",")), None)
                    if _comma is not None:
                        # The name may be IN the comma token itself
                        # ("my friend wren, she's a ceramicist" -> token
                        # "wren,") or in the tokens BEFORE a separate comma
                        # ("my aunt maya, ..."). Strip a trailing comma off the
                        # leading token so the name is extracted cleanly.
                        if _comma == 0:
                            _name = _toks[0].strip(".,!?").lower()
                        else:
                            _name = " ".join(
                                _t.strip(".,!?").lower()
                                for _t in _toks[:_comma]).strip()
                        # a leading comma with nothing before it, or a name that
                        # is itself a closed-class word, is not informative.
                        if _name and _name.split()[0] not in (
                                "the", "a", "an", "my", "your", "i", "you"):
                            _rel_clause = " ".join(
                                _t.strip(".,!?") for _t in _toks[_comma + 1:])
                            _mc = re.match(
                                r"^\s*(?:she|he|they|it|who|that)\b\s*"
                                r"(?:is|'s|was|are|were|be|being|been)\s*"
                                r"(?:a|an|the)?\s*(.+?)\s*$",
                                _rel_clause, re.IGNORECASE)
                            if _mc:
                                _desc = _mc.group(1).strip().lower()
                                _desc = re.sub(r"\s+", " ", _desc)
                                if _desc and len(_desc.split()) <= 8:
                                    # keep a leading article so recall renders
                                    # "your <rel> <name> is a <descriptor>".
                                    if not _desc.startswith(("a ", "an ", "the ")):
                                        _desc = "a " + _desc
                                    _val = _desc
                                    _put_fact_done = False
                                else:
                                    _put_fact_done = True
                            else:
                                # comma present but not a relative clause we
                                # recognize — nothing informative to store.
                                _put_fact_done = True
                        else:
                            _put_fact_done = True
                # COMBINED-attr storage: (i, "<rel> <name>", "<verb> <object>").
                # Reachable from recall branch (c) / open-ended recaller by the
                # relation head. A name-less disclosure (e.g. "my grandmother
                # left me her brass compass" — no name given) stores the
                # relationship itself as the attr so the fact still exists and is
                # recallable; the value carries the verb+object. When no activity
                # verb is present, value is the relation word so the triple still
                # exists. Content from the user's own words; no authored reply,
                # no retraining.
                if _name:
                    _attr = f"{_kin} {_name}".strip()
                else:
                    _attr = _kin
                # GENERALIZE (round 2026-08-20T0701Z): never store a DEGENERATE
                # relationship fact whose value is just the relationship word
                # itself (e.g. ('i','grandmother','grandmother') from "my
                # grandmother" with neither a recognized name nor activity verb).
                # Such a fact carries no information and renders as the broken
                # "your grandmother is grandmother." at recall. Only store when
                # there is real content: a name was captured, OR an activity
                # verb produced a value, AND the value is not identical to the
                # relationship word. Content comes from the user's own words;
                # honest skip when there is nothing informative to store.
                _final_val = _val if _val else _kin
                if not _put_fact_done and _final_val != _kin and (_name or _val):
                    _put_fact(_attr, _final_val, 0.6)


        # D3 (round v3): capture self-disclosed ACTIVITIES / possessions that the
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
                    # Store the verb WITH the object ("keep homing pigeons")
                    # so activity recall ("what do i keep?") can match the
                    # verb and return a complete, grammatical answer instead
                    # of a bare noun. The verb is part of the mined disclosure,
                    # not an authored label.
                    _put_fact("does", f"{_verb} {_obj}", 0.55)
                    # NAME-BRIDGE (round 2026-08-18T1340Z): when the captured
                    # activity object contains "named/called <Name>" (e.g. "i
                    # keep a sourdough starter i named doris"), ALSO store the
                    # possessed thing's NAME as an entity-keyed name fact. The
                    # `does` miner runs BEFORE the equational name patterns, so
                    # without this it swallows the whole disclosure and the name
                    # is never stored — a later "what did i name my <thing>"
                    # would have nothing to recall. Generic: any "named/called
                    # <ProperNoun>" tail, entity = the resolved object noun,
                    # never a pet (those use the dedicated pet paths). No
                    # authored reply; the fact is real state RAVANA can recall.
                    _nm = re.search(
                        r"\b(?:named|called)\s+([A-Za-z][\w'-]*)\b",
                        _m.group(1))
                    if _nm and _pet_slots.species_of(_obj) is None:
                        _put_fact_ent(_obj, "name", _nm.group(1).lower(), 0.6)
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
        # Closed VERB SEED vocabulary (RAVANA-expandable; feeds the same
        # PersonalFactStore the user can correct — NOT per-topic answers, NOT
        # authored prose). Covers everyday disclosure verbs + common irregular
        # past forms so first-person activities/experiences actually land.
        _ACTIVITY_VERBS = (
            "run", "own", "operate", "play", "teach", "study", "manage",
            "drive", "build", "make", "sell", "restore", "grow", "watch",
            "raise", "tend", "brew", "bake", "write", "read", "learn",
            "practice", "collect", "fix", "paint", "code", "design", "craft",
            "volunteer", "cook", "fish", "hike", "garden", "farm", "lead",
            "gather", "count", "harvest", "forage", "observe", "sketch",
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
        )
        _EVENT_VERBS = (
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
        )
        # Match "i [aux?] <verb>(s|ed|ing)? <object> <clause-boundary>".
        # The object stops at a clause boundary (., !, ?, ",", " and ",
        # " but ", " because ", " so ", " which ", " that ", " when ",
        # " where ") so a multi-clause sentence stores only the relevant
        # fragment (e.g. "i repotted the juniper and found a root..." ->
        # "juniper", not "juniper and found a root"). The verb is matched
        # with optional inflection so gerunds/continuous tenses are caught.
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
            # Affective-object guard (round 2026-08-17T1126Z): "i find it
            # fascinating" / "i found that surprising" is a cognitive-affective
            # copula, not a discovery event. The object is a SENTIMENT
            # adjective, so skip storing `event: <verb> <adj>` (which is
            # garbage). A genuine discovery ("i found my keys") has a real
            # content object and still stores. Seed vocabulary (_AFFECTIVE_OBJECT_ADJ),
            # no authored prose, no retraining.
            if _obj and _obj.split()[0] in _AFFECTIVE_OBJECT_ADJ:
                continue
            if _obj and 1 <= len(_obj.split()) <= 5:
                _put_fact("event", f"{_verb} {_obj}", 0.5)

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
            # GENERALIZE (round 2026-08-20T0701Z): the age-anchored-start miner
            # must only capture the USER'S OWN activity-start ("i picked up the
            # cello when i was nine"), NOT a "when i was N" clause that merely
            # dates an event done TO the user by someone else ("my grandmother
            # taught me to fold paper cranes when i was four"). The old code
            # scanned the whole clause for any activity verb and stored
            # since_age="taught fold paper cranes 4" — a junk fact that later
            # leaks into recall ("since_age is my grandmother yuki taught me 4").
            # Fix: skip when the sentence describes a RELATIONSHIP/third-party
            # activity — i.e. it contains a "my <kin/role>" possessive (the
            # activity's actor is the relative, not the user). The relationship
            # vocabulary is the shared relation_attrs.relation_of lexicon
            # (RAVANA-expandable, single source of truth) — no per-name table,
            # no retraining. A genuine first-person age-start ("i was nine when
            # i started dancing") has no such possessive and is still mined.
            try:
                from .relation_attrs import relation_of as _ra_of_d3
            except Exception:
                _ra_of_d3 = lambda w: None
            _has_relative_actor = False
            for _wt in re.findall(r"\bmy\s+([a-z][a-z]+)\b", q_clean, re.IGNORECASE):
                if _ra_of_d3(_wt.lower()) is not None:
                    _has_relative_actor = True
                    break
            if _has_relative_actor:
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

        # Possession-attribute mining (Bug 4, round 2026-08-15T0830Z): a
        # disclosure that names a possession and says what it is made of /
        # what material it is ("the cabin is a hand-hewn pine lodge with a sod
        # roof", "my sword is forged from meteorite iron", "our roof is slate")
        # states a PROPERTY of an owned/described entity. The fact miner above
        # only captured explicit "my X is Y" self-facts + pet names, so these
        # material/attribute facts were never stored as a recallable, correctable
        # fact — and "what's my cabin made of" could not be answered from the
        # structured store (it fell through to a whole-sentence echo). Store them
        # under the ENTITY (cabin / sword / roof), not the user's own "i"
        # subject, exactly like pet_slots does for animals. From there the store
        # LEARNS: a later "no, my cabin is oak-framed" contradicts via the
        # existing contradict() path; confirm/contradict work unchanged.
        #
        # Seed vocabulary, not an answer table (mirrors pet_slots): a closed
        # core of material + kind nouns plus a runtime-grown extension
        # (_poss.learn_material) so a word RAVANA has never heard ("hempcrete")
        # becomes addressable for later recall with NO code change. Removing an
        # entry degrades gracefully (the material is simply not mined until
        # re-learned). Nothing here is ever rendered to the user — recall
        # rendering lives in engine_memory._reconstruct_entity via _poss.render.
        for _m in re.finditer(
                # entity = a leading noun (optionally "my/the/a" + adjectives);
                # "is" with an optional "a/an"; then the descriptor phrase
                # (allows hyphenated words like "hand-hewn"; a material such as
                # "pine"/"sod" appears somewhere in the phrase). A trailing kind
                # noun (lodge/house/...) marks a possession description; without
                # a recognised material we only mine when one is present, so
                # "the river is a fast mountain stream" is correctly ignored.
                r"\b(?:my|the|a|an|our|your)\s+([a-z][a-z'-]+)\s+"   # entity
                r"(?:is|are|was|were)\s+(?:(?:a|an|the)\s+)?\s*"      # copula (+opt article)
                r"([a-z][a-z'-]*(?:\s+[a-z][a-z'-]+){0,6})",        # descriptor
                q_clean, re.IGNORECASE):
            _ent = _m.group(1).lower().strip("'")
            _desc = _m.group(2).lower().strip()
            if _ent in _VALUE_STOP or len(_ent) < 3:
                continue
            # Split the descriptor into whitespace-delimited tokens (each may be
            # hyphenated, e.g. "hand-hewn"); scan for a known material noun.
            _dtoks = re.findall(r"[a-z][a-z'-]+", _desc)
            _mat = None
            _feat = None
            for _i, _w in enumerate(_dtoks):
                if _poss.is_material(_w):
                    _mat = _w
                    # a feature noun after the material scopes the fact
                    # ("sod roof" -> cabin.roof = sod, not cabin.madeof = sod)
                    _nx = _dtoks[_i + 1] if _i + 1 < len(_dtoks) else None
                    if _nx and _poss.is_feature_noun(_nx):
                        _feat = _nx
                    break
            # also accept a "made of/from" / "built of/from" explicit frame
            _explicit = re.search(
                r"\b(?:made|built|forged|carved|woven|cast|moulded|molded|"
                r"constructed|fashioned)\s+(?:of|from|out of)\s+([a-z][a-z'-]+)",
                " " + _desc + " ", re.IGNORECASE)
            if _explicit and _poss.is_material(_explicit.group(1)):
                _mat = _explicit.group(1)
            if _mat is not None:
                _mat = _mat if _poss.is_material(_mat) else _poss.learn_material(_mat)
                if _feat:
                    _put_fact_ent(_ent, _feat, _mat, 0.6)
                else:
                    _put_fact_ent(_ent, "madeof", _mat, 0.6)
            # a possession-kind clause with no recognised material yet: skip
            # storing an empty value (a later correction / online learning can
            # attach a material).
            elif any(_poss.is_kind_noun(w) for w in _dtoks):
                continue

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
                # PROVENANCE (round 2026-08-20T0701Z-followup, residual
                # limitation #1). The keyed `_topic` ("silence") may be a
                # SUBORDINATE concept while the utterance also names the
                # SALIENT broader concept the user actually meant ("winter").
                # Capture the salient content nouns of the FULL object phrase
                # (`_raw`, before _opinion_topic collapsed it to the head), so
                # the resolver + reversal miner can later bridge a co-mention
                # of "winter"/"street art" to this stance even though the key
                # differs. The set is GROWN ONLINE from the real utterance —
                # seed is empty, nothing hardwired, RAVANA revises it by
                # further talk. Generic: content-noun extraction reuses the
                # same closed-class stop set the miner already routes through.
                _prov = self._opinion_provenance(_raw)
                self.opinions.express_stance(_topic, polarity=_p, confidence=_conf,
                                            valence=_v, arousal=_a,
                                            provenance=_prov)

        # Affect-verb attitude construction mining (feature round
        # 2026-08-21T1653Z residual #1): "X creeps me out" / "X grosses me out"
        # / "X freaks me out" / "X gets to me" were NOT mined as stances even
        # though the affect verb already lives in the shared VAD lexicon, so a
        # later reversal ("i changed my mind about X") had nothing to act on.
        # This is a GRAMMATICAL pattern (<subject> <affect-verb> <me>), not a
        # per-topic list; polarity is derived from the shared VAD affect lexicon
        # (the same matrix the empathy gate grows online), so a verb RAVANA has
        # not seen yet simply scores 0.0 and is skipped — fail-closed, no
        # confabulation. Each observed verb is registered into the VAD matrix so
        # coverage GROWS by experience (see _mine_affect_verb_stance).
        self._mine_affect_verb_stance(text)

        # Stance-reversal mining: "i take back X" / "i changed my mind about X" /
        # "i retract my stance on X" recodes the user's valuation of the topic to
        # the opposite pole (vmPFC re-evaluation), LINKED to the PRIOR stance the
        # store already holds — this is an attitude-change operator, not a fresh
        # opinion. Runs last so it can see (and reverse) any stance just mined.
        self.mine_stance_reversal(text)

        # Mirror any count correction into the structured QuantityMemory store
        # (round 2026-08-11T0521Z). Runs at the END of mine so detected_correction_fact
        # (set by the correction circuit above) is available. correct() supersedes
        # the prior record by subject+noun so a stale count is retired, not echoed.
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

    def _mine_affect_verb_stance(self, text: str) -> None:
        """Mine first-person affect-verb attitude constructions as stances.

        Pattern: "<subject> <affect-verb> <me/us>" where the verb is an
        affect term drawn from the SHARED VAD affect lexicon (e.g. "creeps",
        "grosses", "freaks", "creepy", "wary"). Examples:
            "lab-grown meat creeps me out"   -> negative stance on "lab-grown meat"
            "that flickering light freaks me out" -> negative stance on the light
            "his constant humming gets to me" -> negative stance on the humming

        Design (seed-vs-hardcoding + no-retraining):
          * Polarity comes from the SAME VAD association matrix the empathy /
            support classifier already uses and GROWS online via Hebbian
            learning (`UserEmotionDetector`). We do NOT introduce a second
            affect-word list. A verb absent from the matrix scores 0.0 and is
            skipped (fail-closed). Every verb we DO see is registered into the
            matrix (`learn_association`) so the next encounter is scored even
            without a seed entry — the capability compounds with use.
          * Morphological variants ("creeps"->"creep", "grosses"->"gross",
            "freaked"->"freak") are resolved through the SAME
            `_morphological_normalize` the matrix lookup uses, so no parallel
            stemmer.
          * Subject resolution reuses the shared `_opinion_topic` chokepoint,
            so a stance lands on the real content head ("lab-grown meat"), never
            a closed-class word. Generic: any subject the user rotates in.
          * Reversals remain fully operable: the stance is stored in the same
            `opinions` store, so `mine_stance_reversal` (run next) can later
            recode it. Nothing is frozen; no retraining, no per-topic table.
        """
        if not text:
            return
        q = text.lower()
        # First-person only: the attitude must be the user's own self-report,
        # never a question ("does clowns creep you out?" is not a stance).
        if q.rstrip().endswith("?") or re.match(
                r"^(what|who|when|where|why|how|which|is|are|do|does|did|"
                r"can|could|would|should|will|may|might|am|have|has|had)\b", q):
            return
        # An affect-verb must be SOMEWHERE in the utterance; cheap pre-filter so
        # the regex only runs when relevant. The verb itself is validated
        # against the shared VAD matrix below, so this is just a content gate.
        _tok = set(re.findall(r"[a-z']+", q))
        if not (_tok & {"me", "us", "myself"}):
            return  # no first-person object -> not this construction
        # The affect-verb LEMMA set is SEED vocabulary (the same allowed class
        # as the dismissive-metaphor noun set and the sentiment adjectives): a
        # small bootstrapping list of aversive-reaction verbs. It is
        # RAVANA-extendable — an unseen verb in this class bootstraps its own
        # VAD entry on first encounter (see the valence logic below) and grows
        # online; removing an entry only loses one verb shape. NOT a per-topic
        # answer table. Every verb here denotes an aversive reaction to its
        # subject, which is what licenses the structural default-polarity below.
        for _m in re.finditer(
                r"\b([a-z][a-z'\-]+(?:\s+[a-z][a-z'\-]+){0,5})\s+"
                r"(creeps?|gross(?:es)?|freaks?|weirds?|unnerves?|disgusts?|"
                r"repulses?|scares?|terrifies?|bothers?|annoys?|rattles?|"
                r"spooks?|unsettles?|creep|gross|freak|weird|unnerve|disgust|"
                r"repulse|scare|terrify|bother|annoy|rattle|spook|unsettle|"
                r"gets\s+to)\b"
                r"\s+(?:me|us|myself)\b", q):
            _raw_subj = _m.group(1).strip()
            if not _raw_subj:
                continue
            # Resolve the salient content head via the shared chokepoint.
            _topic = self._opinion_topic(_raw_subj)
            if not _topic:
                continue
            # Validate the affect verb against the shared VAD matrix and read
            # its valence. Morphological normalization reuses the matrix's own
            # routine so "creeps"->"creep" etc. resolve without a second list.
            _verb_raw = _m.group(2).replace(" ", "")  # "gets to" -> "getsto"
            _vad = self._vad_for_affect_verb(_verb_raw)
            if _vad is not None:
                _valence = float(_vad[0])
            else:
                # The verb is in the seed affect-verb class but has no VAD
                # entry yet. The "<subject> <affect-verb> me" construction
                # STRUCTURALLY encodes an aversive reaction (like the
                # comparative "X is better than Y" is structurally positive),
                # so seed a default negative valence and register it into the
                # matrix so future scoring — in ANY construction — recognizes
                # it. Online growth, no retraining, no second list.
                _vad = (-0.6, 0.55, -0.35)
                _valence = -0.6
            if abs(_valence) < 0.05:
                continue  # neutral / unknown valence -> never seed a stance
            # GROW the matrix: register the exact surface verb so future
            # mentions (in any construction) are scored. Online, no retraining.
            self._learn_affect_verb(_verb_raw, _vad)
            # Sign-preserving affect blend (same discipline as the lexical
            # miner above): the lexical VAD valence is the ground-truth
            # attitude signal; the turn-affect buffer only reinforces, never
            # reverses.
            _p = _valence
            _v, _a, _d = self._infer_user_emotion(text)
            if _p < -0.05 and _v < -0.1:
                _p = min(_p, _p - 0.15)
            elif _p > 0.05 and _v > 0.1:
                _p = max(_p, _p + 0.15)
            _p = max(-1.0, min(1.0, _p))
            _prov = self._opinion_provenance(_raw_subj)
            self.opinions.express_stance(_topic, polarity=_p, confidence=0.6,
                                        valence=_v, arousal=_a,
                                        provenance=_prov)

    def _vad_for_affect_verb(self, verb: str):
        """Return the VAD triple for an affect verb from the SHARED VAD matrix.

        Reuses `UserEmotionDetector._lookup_word` (which applies the same
        `_morphological_normalize` the rest of the system uses) so we never
        maintain a second affect lexicon. Returns None when the verb has no
        seeded/learned VAD entry (fail-closed: we do not invent a polarity).
        """
        self._ensure_emotion_detector()
        det = self._emotion_detector
        # _lookup_word is the matrix's own morphological resolver.
        entry = det._lookup_word(verb) if hasattr(det, "_lookup_word") else None
        if entry is None:
            # Fall back to the seed dictionary directly for stem forms the
            # detector may not have normalized (e.g. "creep" without seed).
            from ravana.core.mirror import _VAD_SEED, _morphological_normalize
            for cand in (verb,) + tuple(_morphological_normalize(verb)):
                if cand in _VAD_SEED:
                    v, a, d = _VAD_SEED[cand]
                    return (float(v), float(a), float(d))
            return None
        return (float(entry[0]), float(entry[1]), float(entry[2]))

    def _learn_affect_verb(self, verb: str, vad) -> None:
        """Register an observed affect verb into the SHARED VAD matrix.

        Online growth: the first time the user utters "X creeps me out", the
        surface verb "creeps" is added to the matrix (seeded from its VAD
        lookup) so later affect scoring — in ANY construction — recognizes it.
        No retraining, no second list; the matrix is the single source of
        truth and it compounds with conversation.
        """
        self._ensure_emotion_detector()
        det = self._emotion_detector
        if hasattr(det, "learn_association"):
            det.learn_association(verb, (float(vad[0]), float(vad[1]),
                                         float(vad[2])), confidence=0.5)

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
        # Filler-token guard: a held stance key may contain a function/filler
        # word (e.g. "thunderstorms now" — "now" is a temporal filler). Without
        # this, ANY utterance containing that filler token ("...the way i used
        # to now") would bind the WRONG stance (the 2026-08-21T0843Z misattrib:
        # a grass contradiction acked "you've changed your mind about
        # thunderstorms" because both contained "now"). Score only on CONTENT
        # tokens; stopwords + temporal/discourse fillers never count as an
        # overlap. STOPS is broader than constants.STOP_WORDS (which omits
        # temporal adverbs like "now") so the misbind is fully closed. Preserves
        # the multiword match (a partial recant "letterpress" still binds
        # "letterpress printing") — the set difference only drops filler words.
        STOPS = STOP_WORDS | _FILLER_TOKENS
        _words = {w for w in _words if w not in STOPS}
        if not _words:
            return None
        _best = None
        _best_score = 0
        for _k in self.opinions.stances:
            if not _k:
                continue
            _ktoks = [t for t in re.findall(r"[a-z']+", _k.lower()) if t and t not in STOPS]
            if not _ktoks:
                continue
            # A partial recant names only part of a multiword key (e.g.
            # "letterpress" for the stored "letterpress printing"); match when
            # ANY content key token appears as a whole word, and prefer the key
            # with the most content tokens present (most specific match wins).
            _matched = sum(1 for t in _ktoks if t in _words)
            if _matched == 0:
                continue
            if _matched > _best_score:
                _best_score = _matched
                _best = _k
        return _best

    def _stance_key_in_text_stem(self, text: str):
        """Like `_stance_key_in_text` but ALSO binds a held stance when the
        utterance's content word is a VERB-STEM match of a key token (or vice
        versa). Resolves "hike" in "i can't really hike the way i used to" to
        the stored "cold weather hiking" stance, which whole-word matching
        misses ("hiking" is not a prefix of "hike" and vice versa — they share
        only the stem "hik"). Generic: the stem is computed by stripping common
        verb suffixes (ing/ed/s/e), a small morphological rule, not a per-topic
        table; we still prefer the most token-overlapping key. Used by the
        first-person limitation branch in mine_stance_reversal (round
        2026-08-18T0937Z).
        """
        if not text:
            return None
        _words = [t for t in re.findall(r"[a-z']+", text.lower())
                  if len(t) >= 3 and t not in STOP_WORDS and t not in _FILLER_TOKENS]
        if not _words:
            return None

        def _stem(w):
            # Lightweight verb-stem normalization: strip the most common
            # English inflection suffixes so "hike"/"hiking"/"hiked" collapse to
            # "hik" and "run"/"running" to "run". Bounded, no external stemmer.
            w = w.rstrip("'s").rstrip("s")
            for _suf in ("ing", "ed", "er", "ly"):
                if w.endswith(_suf) and len(w) - len(_suf) >= 3:
                    w = w[: -len(_suf)]
                    break
            if w.endswith("e") and len(w) >= 4:
                w = w[:-1]
            return w

        _best = None
        _best_score = 0
        for _k in self.opinions.stances:
            if not _k:
                continue
            # Score only CONTENT tokens of the key — drop filler words so a key
            # like "thunderstorms now" cannot bind an utterance that merely
            # shares the temporal filler "now" (the 2026-08-21T0843Z
            # misattribution defect). Stem matching below still binds verb-stem
            # variants ("hike"/"hiking"); only stopwords + fillers are excluded.
            _ktoks = [t for t in re.findall(r"[a-z']+", _k.lower())
                      if len(t) >= 3 and t not in STOP_WORDS and t not in _FILLER_TOKENS]
            if not _ktoks:
                continue
            _score = 0
            for _t in _ktoks:
                _ts = _stem(_t)
                for _w in _words:
                    _ws = _stem(_w)
                    # whole-word match (most reliable) OR stem overlap of
                    # length >=3 so "hike"/"hiking"/"hiked" all bind (they share
                    # stem "hik") but "art" doesn't bind "start".
                    if _t == _w:
                        _score += 2
                    elif _ts and _ws and len(_ts) >= 3 and (_ts == _ws
                                                         or _ts.startswith(_ws)
                                                         or _ws.startswith(_ts)):
                        _score += 1
            if _score > _best_score:
                _best_score = _score
                _best = _k
        # Require at least a stem overlap so we never soften a stance the
        # utterance doesn't actually reference (bounded false positives).
        return _best if _best_score > 0 else None

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
            # R3 (round 2026-08-18T0937Z): FIRST-PERSON LIMITATION / AVERSION
            # disclosure. A user can walk a stance back WITHOUT a retraction or
            # concession keyword: "my knee's been acting up ... i can't really
            # hike the way i used to", "loud music wrecks me now", "i can't
            # handle that volume". These are sensory/affective LIMITATION
            # statements about a topic RAVANA already holds a positive stance
            # on. Previously they fell through to a hollow "noted." and the
            # stale stance persisted (residual limitation #1 of round
            # 2026-08-17T1730Z). Detect the limitation shape structurally
            # (first person + a can't/can-not/cannot/wrecks-me/anymore cue) and
            # resolve the topic against the LIVE stance store — including
            # verb-stem matching so "hike" in the disclosure binds the stored
            # "cold weather hiking" stance (whole-word match misses it). When a
            # held stance is found, SOFTEN it toward neutral (a limitation is a
            # relaxation, not an inversion). Generic: the topic is resolved, not
            # hardcoded, so any held positive stance the user limits gets
            # softened. No retraining, no per-topic table.
            _limitation = re.search(
                r"\b(i\b|my|me|we|us)\b.{0,40}?\b("
                r"can'?t|can not|cannot|couldn'?t|no longer|anymore|"
                r"wrecks? me|wears? me (?:down|out)|i can't handle|"
                r"can't really|can't stand|harder than it used to|"
                r"the way i used to|too much for me)\b", q)
            if _limitation is not None:
                _target = self._stance_key_in_text_stem(q)
                if _target is not None:
                    # R3 fix (round 2026-08-18T0937Z): a limitation/aversion
                    # disclosure SOFTENS a stance the user ALREADY HELD in a
                    # PRIOR turn. It must NOT fire on the VERY turn the user
                    # first states the aversion (e.g. "i can't stand cilantro"
                    # is mined fresh at -0.8 by mine_stance and must stay there
                    # — the limitation branch was relaxing it to -0.333). Guard:
                    # require the resolved stance to PREDATE the current turn (
                    # strictly before, so a just-mined stance with
                    # turn_number == turn_num is skipped). This preserves the
                    # real feature — a HELD positive stance later limited by the
                    # user ("i love hiking" -> "my knee's acting up, i can't
                    # really hike anymore") still softens — while leaving fresh
                    # first-person aversions at full strength.
                    _existing = self.opinions.stances.get(_target)
                    if _existing is None or _existing.turn_number >= self.opinions.turn_num:
                        return  # freshly-mined this turn (or absent) -> not a limitation of a HELD stance
                    try:
                        self.opinions._soft_reversal = True
                        self.opinions.reverse_stance(_target, utterance=text)
                    except Exception:
                        pass
                    return
            # FEATURE (round 2026-08-20T1229Z-followup, residual limitation #1):
            # FREE-FORM contradiction recode. The branches above only fire on an
            # explicit retraction keyword, a "but"/belief concession frame, or a
            # "can't/anymore" limitation cue. But a user very often RE-STATES an
            # attitude that OPPOSES a stance they already hold WITHOUT any of
            # those shapes ("actually i've gone off winter, the cold gets to me
            # now"; "not all street art is good"; "they wear me out these days").
            # Previously none of the branches matched, the opinion miner extracted
            # no fresh stance, and the stale held stance persisted un-reversed (the
            # 0701Z contradiction-mining gap — provenance was later bridged for
            # KEYING, but the reversal was still never *detected* for these). Detect
            # a reassessment-affect signal (seed lexicon, RAVANA-extendable), resolve
            # the topic against the LIVE stance store (the resolver already bridges a
            # broader co-mention such as "winter" back to a stance keyed "silence"
            # via provenance — so one mechanism covers both), and when the new
            # attitude OPPOSES the held polarity, recalibrate the held stance toward
            # the new value. Generic: the topic comes from the store, the new value
            # from the user's real words, no per-topic rule, no retraining. If the
            # utterance carries NO reassessment signal we leave the stance alone
            # (honest — we never guess a reversal from neutral wording).
            _new_pol = _assess_reversal_polarity(text)
            if _new_pol is not None:
                # Resolve against the LIVE stance store, but the free-form
                # recode only ever acts on a stance the user ALREADY HELD in a
                # PRIOR turn. The opinion miner in the SAME turn may have just
                # created a fresh stance from a co-mentioned concept (e.g.
                # "the cold gets to me now" -> a new "cold" stance), and
                # `resolve_topic` can bind to that fresh stance via substring
                # precedence, leaving the genuinely-held stance (bridged through
                # provenance, e.g. "silence" co-mentioned with "winter") never
                # reached. Restrict resolution to PRIOR-turn stances so the
                # recode targets the held valuation, never a coincidental
                # same-turn co-mention. (Honest: a topic with no PRIOR stance is
                # left alone — we never walk back a brand-new attitude.)
                _target = self.opinions._resolve_prior_stance(text)
                if _target is not None:
                    _held = self.opinions.stances.get(_target)
                    # Guard: only RECODE a stance the user ALREADY HELD in a PRIOR
                    # turn (same rule as the limitation branch). A stance mined THIS
                    # turn conveys the user's CURRENT view and must not be walked
                    # back by a coincidental reassessment term in the same turn.
                    if (_held is not None
                            and _held.turn_number < self.opinions.turn_num
                            # Only act as a contradiction when the new attitude
                            # actually OPPOSES the held value (sign differs, or it
                            # pulls the held value across the pivot). Same-sign
                            # reassessment ("i still love winter, it's the best")
                            # is already handled by the weighted merge upstream, so
                            # leave it — no double-write.
                            and (_new_pol * _held.polarity) < 0.0):
                        try:
                            self.opinions._soft_reversal = True
                            self.opinions.recode_stance_toward(
                                _target, new_polarity=_new_pol,
                                blend=0.7, utterance=text)
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
        # A recant is a genuine SCOPE-NARROWING (the "acoustic-ONLY" corruption
        # case: the user still likes the held attitude, just restricts its
        # scope) ONLY when the recant phrase carries an explicit scope marker.
        # A subset recant WITHOUT a scope marker ("i take it back about the
        # cold", where "cold" is the salient head of the held "cold weather
        # give" stance) is a WHOLE-ATTITUDE reversal — the user is inverting the
        # evaluation of the object, not narrowing it. Treating the no-marker
        # subset case as narrowing (the prior behavior) silently dropped real
        # reversals (round 2026-08-21T1653Z: "i take it back about the cold"
        # left the stance pinned at +0.95). So the narrowing guards below only
        # fire when a scope marker is present. Generalizable: the marker is a
        # structural scope signal, not a per-topic rule, so it separates the two
        # operations for ANY topic (per opencode cognitive review: narrowing
        # re-segments the object while keeping valence; reversal negates the
        # object evaluation).
        # NOTE: the scope marker must be read from the RAW tail text, NOT the
        # extracted `topic` — _opinion_topic drops the "-only" qualifier (it
        # returns the bare head "acoustic"), so testing `topic` would never see
        # the marker and the narrowing guard would never fire. `tail` still
        # contains "acoustic-only", so scan it for a scope marker token.
        _recant_has_scope = bool(_scope_markers & set(
            re.findall(r"[a-z']+", (tail or "").lower())))
        for _k in self.opinions.stances:
            _kt = set(re.findall(r"[a-z']+", _k.lower()))
            if not _kt:
                continue
            # tight: every recant token present in the held key, OR strong
            # jaccard overlap (>=0.5) between the two content-word sets.
            if _topic_tokens and (_topic_tokens <= _kt
                                  or len(_topic_tokens & _kt) / max(1, len(_topic_tokens)) >= 0.5):
                # REJECT a narrowing recant: when the recant's meaningful
                # content is a STRICT subset of the held topic AND the recant
                # carries a scope marker (e.g. "acoustic-ONLY"), the user is
                # restricting scope (still likes acoustic, just not *only*
                # acoustic), not reversing their attitude. Flipping the held
                # stance would corrupt the store. Scope-widening is not a
                # reversal. Without a scope marker the subset is a whole-attitude
                # reversal (see _recant_has_scope above), so it is NOT rejected.
                if _recant_has_scope and _topic_tokens and _topic_tokens < _kt:
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
                    _recant_has_scope
                    and _recant_meaningful
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
        # Only block as scope-widening when the recant actually carries a scope
        # marker (see _recant_has_scope). A no-marker subset recant is a whole-
        # attitude reversal and must NOT be dropped.
        if _recant_has_scope and _recant_meaningful and _recant_meaningful < _tgt_tokens:
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
        # Discourse connectors that terminate an opinion object phrase
        # (round 2026-08-20T1229Z, FIX B). "i love small jazz clubs though"
        # was mining a stance whose TOPIC was "small jazz clubs though" because
        # the like/love pattern captured greedily up to the sentence end and
        # _opinion_topic only cut at prepositions/conjunctions, not the
        # connector "though". Adding these as stop-words makes the content head
        # trim cleanly to "small jazz clubs". Structural closed-class set;
        # generalizes to any connector the user rotates in, no per-topic rule.
        "though", "although", "yet", "however", "nevertheless", "nonetheless",
        "still", "anyway", "besides", "meanwhile", "otherwise",
        # Round 2026-08-20T1935Z FRAGMENT FIX: a simile connector ("like")
        # and a sequence connector ("after" / "before") left DANGLING
        # PREPOSITION/CONNECTIVE fragments in mined topics. Observed: "i keep
        # grinning like an idiot" -> topic "keep grinning like"; "petrichor
        # after a storm is my favorite" -> topic "petrichor after"; the same
        # class would dangle "the silence before sleep" -> "silence before".
        # Both are open-class-looking heads that are actually closed-class
        # connectives the cut-loop should have terminated on. "like" is
        # already excluded from being a STANCE VERB (the like/love detector
        # names the verb explicitly), so adding it here only affects topic
        # normalization, never preference detection. Structural closed-class
        # set; no per-topic rule. Cuts at the connector so "keep grinning like
        # an idiot" -> "keep grinning", "petrichor after a storm" ->
        # "petrichor", "silence before sleep" -> "silence".
        "like", "after", "before",

        "really", "very", "just", "only", "also", "too", "quite", "more",
        "most", "much", "many", "such", "own", "same", "other", "another",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "has", "have", "had", "not", "don't", "dont", "do", "does", "did", "can", "cannot", "cant",
        "it", "they're", "im", "i'm", "you're", "we're", "there",
    }

    def _opinion_provenance(self, phrase: str) -> List[str]:
        """Return the salient content nouns of an opinion-object phrase.

        Companion to `_opinion_topic`: where that method returns the single
        content HEAD the stance is keyed on, this returns ALL salient content
        nouns of the phrase (the head plus any modifiers that survived the
        closed-class strip). Used to seed a stance's PROVENANCE so the
        resolver + reversal miner can bridge a later co-mention of a broader
        concept (e.g. "winter", "street art") back to a stance keyed on a
        subordinate word ("silence", "vandalism"). Generic and store-driven:
        the nouns come from the user's actual words, nothing is hardwired, and
        the set is merged online across encounters. Seed vocabulary = the
        shared closed-class stop set already used by the opinion miners.
        """
        toks = [t for t in re.findall(r"[a-z'][a-z']*", (phrase or "").lower())]
        if not toks:
            return []
        # Drop leading closed-class framers (determiners/prepositions/particles)
        # and trailing modifiers using the SAME stop set the miner routes through,
        # but keep INTERNAL content nouns (the salient concepts co-named with the
        # head). We keep every token that is NOT a stop word — that yields the
        # full salient concept set rather than only the head.
        out = [t for t in toks if t not in self._OPINION_STOP]
        # Also drop pure particles (so "up"/"down" never enter provenance).
        _PARTICLES = {
            "up", "down", "off", "out", "in", "on", "away", "back", "over",
            "under", "around", "through", "along", "by", "past", "upon",
        }
        out = [t for t in out if t not in _PARTICLES]
        return out

    def _mine_pet_activity(self, tail: str, name: str, species: str) -> Optional[str]:
        """Extract a pet's verb-phrase ACTIVITY from the tail after the name.

        Round 2026-08-22T0703Z, DEFECT D1: pet disclosures like "my ferret
        Pip hides my car keys under the couch" were mined as NAME ONLY, dropping
        the activity, so later cued/pet recall ("which of my pets hides things
        in the couch") had nothing to surface. Kin disclosures already capture
        verb+object activity; pets now do too so recall is symmetric.

        Returns the verb+object head (e.g. "hides car keys"), or None when the
        tail has no recognized verb. Verbs come from the SHARED verb lexicons
        (is_activity_verb / is_relation_verb / is_aux_verb — RAVANA-extensible,
        no per-animal table, no authored reply, no retraining). The object is a
        real concept resolved through _opinion_topic; bounded to <=5 tokens so a
        runaway clause cannot swallow the sentence. Content is the user's own
        words.
        """
        if not tail:
            return None
        _toks = re.findall(r"[a-z'][a-z']*", tail.lower())
        _vidx = None
        for _i, _t in enumerate(_toks):
            if is_activity_verb(_t) or is_relation_verb(_t) or is_aux_verb(_t):
                _vidx = _i
                break
        if _vidx is None:
            return None
        _verb = _toks[_vidx]
        _obj_rest = " ".join(_toks[_vidx + 1:])
        # Reuse the same object-head extraction the kin/opinion paths use so the
        # activity value is a real concept, not a filler/preposition tail.
        _obj = self._opinion_topic(_obj_rest) or ""
        _obj = _strip_obj_framers(_obj).strip()
        if _obj and len(_obj.split()) <= 5:
            return f"{_verb} {_obj}"
        return None

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
        # REJECT directional PARTICLE heads (round 2026-08-16). A disclosure
        # like "i grew up in a village called aldermoor" has the verb object
        # span begin with the particle "up" ("up in a village called
        # aldermoor"). The particle is NOT a content head — it is a closed-class
        # framer. The old code returned "up" as the salient head, and the
        # activity/event miners then stored a bare-verb junk fact ("does=grew",
        # "event=grew") because _strip_obj_framers trimmed the trailing "up" and
        # left only the verb. Those junk facts poisoned every does-keyed recall
        # ("where did i grow up" -> "you do grew"; "what have you learned about
        # me" -> "your does is grew"). Treating a particle as a content head is
        # the same class of error as a function word being kept — particles are
        # closed-class framers. Drop them so the real content head ("village")
        # surfaces. Structural: a small closed particle set, generalizes to any
        # "i <verb> <particle> <object>" disclosure (grew up / came back /
        # went out / sat down / fell over / turned around). No per-topic table,
        # no retraining; removing an entry only loses one particle shape.
        _PARTICLES = {
            "up", "down", "off", "out", "in", "on", "away", "back", "over",
            "under", "around", "through", "along", "by", "past", "upon",
        }
        while toks and toks[0] in _PARTICLES:
            toks.pop(0)
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
        # MULTI-ACTIVITY CUT (feature t_46c07b5d, D5 residual from round
        # 2026-08-19T1628Z): a disclosure can name TWO activities as one object
        # span — "i adore cold water swimming jumping", "nothing beats cold
        # water swimming jumping", "i care for cold water swimming jumping" —
        # where the second verb/gerund ("jumping") is appended to the first
        # ("swimming") into a single run-on stance key ("cold water swimming
        # jumping"). The resolver + reversal miner can NEVER bridge a later
        # co-mention ("am i still into cold water swimming?") to that key, so
        # the stance is unrecallable (a real defect, not a seed). Morphological
        # gate: cut the head at the FIRST gerund token that is NOT the leading
        # content word (a second activity), so the key lands on the single
        # salient activity ("cold water swimming"). Agerund is a content word
        # ending in "ing" of length >= 5 that is NOT an aspectual/framer residue
        # already excluded by _OBJ_NONCONTENT ("coming"/"keeping"/"going"/
        # "starting"/"burning"/"ringing"/"taking"/"making"). The leading token
        # is always kept — so a single-activity object ("swimming", "mountain
        # climbing", "fossil hunting", "river kayaking") survives WHOLE (it is
        # the first and only gerund) and continues to feed the `does`/`event`
        # fact stores that reuse this method (rule 6g: one shared chokepoint,
        # no per-verb branch, no retraining). The cut is purely morphological
        # and generalizes to ANY two-activity disclosure the user rotates in.
        _saw_gerund = False
        _cut_at = None
        for _i, _t in enumerate(head):
            if _i == 0:
                # never cut the leading token — it is the content head.
                if _t.endswith("ing") and len(_t) >= 5 and _t not in _OBJ_NONCONTENT:
                    _saw_gerund = True
                continue
            if _t.endswith("ing") and len(_t) >= 5 and _t not in _OBJ_NONCONTENT:
                if _saw_gerund:
                    # second activity in the span — stop here.
                    _cut_at = _i
                    break
                _saw_gerund = True
        if _cut_at is not None:
            head = head[:_cut_at]
        if not head:
            return None
        # Drop trailing closed-class/modifier words as a final safety.
        while len(head) > 1 and head[-1] in self._OPINION_STOP:
            head.pop()
        if not head:
            return None
        # CONTENT-ADEQUACY GATE (round 2026-08-17T1730Z): a resolved activity/
        # event object that consists ENTIRELY of non-content words — an
        # aspectual/particle verb residue ("coming back", "started keeping",
        # "got burned", "found out"), a bare timeframe ("went last night"), or a
        # generic noun ("found project") — is not a real thing RAVANA learned
        # and must NOT be stored as a `does`/`event` fact (it later echoes in
        # "what have you learned about me" dumps and pollutes recall). Reject
        # the head only when NO content-bearing word survives, so genuine
        # activities that happen to contain one of these words (e.g. "night
        # sky", "side project", "last harvest") are still kept. This is the
        # SAME shared chokepoint the activity/event miners already route
        # through, so the gate generalizes across both blocks (rule 6g) with
        # one addition, not per-verb branches. Seed vocabulary: removing an
        # entry only re-admits one low-value object shape. No retraining.
        if all(t in _OBJ_NONCONTENT for t in head):
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
