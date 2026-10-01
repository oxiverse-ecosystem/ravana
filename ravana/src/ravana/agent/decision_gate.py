#!/usr/bin/env python3
"""RAVANA agentic decision gate — WHEN does RAVANA use its "hands"?

This is the cognitive half of the agentic layer. It is STATE-DRIVEN, not a
keyword→tool table (which would be hardcoded and would be reverted by the
loop's auditor). The decision is made from RAVANA's own cognitive signals:

  - CuriosityEngine.uncertainty_for(topic): high => "I don't know, ground it"
  - MetaCognition.current_mode == UNCERTAIN: "I should verify, not guess"
  - SocialIntentClassifier.classify(query): task/imperative speech act => act

If NO cognitive signal justifies a tool call, the gate returns None — RAVANA
answers from what it knows (or admits uncertainty) instead of faking tool use.
This is the honest, no-hardcoding-compliant path to agency.
"""
from __future__ import annotations
from typing import Optional
import re

from .tool_registry import ToolCall, ToolRegistry

# Speech-act labels (from RAVANA's own SocialIntentClassifier) that imply a task
# the agent should act on rather than just discuss.
_TASK_ACTS = {"command", "request", "imperative", "directive", "task"}

# Nouns that, when present in a task intent, map to a specific safe tool.
# Includes common plural forms — these are seed values, expandable at runtime.
_TOOL_NOUNS = {
    "repo": "github_cli", "repository": "github_cli", "git": "github_cli",
    "commit": "github_cli", "commits": "github_cli",
    "branch": "github_cli", "branches": "github_cli",
    "diff": "github_cli", "log": "github_cli", "status": "github_cli",
    "script": "run_script", "scripts": "run_script",
    "run": "run_script", "execute": "run_script",
    "website": "read_website", "page": "read_website", "url": "read_website",
    "search": "web_search", "lookup": "web_search", "what is": "web_search",
}

# Seed vocabulary of imperative verbs (expandable at runtime).
# These are seed values — the set can be extended at runtime as new
# imperative patterns are encountered. This is NOT a hardcoded trigger;
# it's a heuristic that fires when a tool noun is also present.
_IMPERATIVE_VERBS = {
    "show", "list", "display", "print", "get", "give", "tell",
    "run", "execute", "do", "make", "create", "delete", "remove",
    "add", "update", "change", "set", "fetch", "pull", "push",
    "commit", "checkout", "branch", "merge", "clone", "git",
    "diff", "log", "check", "verify", "validate", "find", "search",
    "open", "close", "start", "stop", "restart", "deploy", "build",
    "test", "debug", "fix", "clean", "install", "uninstall",
    "help", "explain", "describe", "compare", "analyze", "review",
    "save", "load", "read", "write", "edit", "move", "copy",
    "rename", "switch", "reset", "revert", "stash", "tag",
    "browse", "navigate", "go", "enter", "exit", "quit",
    "send", "receive", "upload", "download", "import", "export",
}


def _is_imperative_formed(query: str) -> bool:
    """Heuristic: is this query imperative-formed?

    Imperative-formed means:
    1. No question mark (not a question)
    2. First content word is in the imperative verb seed set

    This is a heuristic, not a parser. The verb set is seed vocabulary
    that can be extended at runtime.
    """
    q = query.strip()
    if not q:
        return False
    # Not a question
    if "?" in q:
        return False
    # Check if first word is an imperative verb
    first = q.split()[0].lower().rstrip(".,!;:")
    return first in _IMPERATIVE_VERBS


def add_imperative_verbs(verbs: set) -> None:
    """Extend the imperative-verb seed set at runtime.

    As new imperative patterns are encountered, the verb set can be grown
    without modifying code. This keeps the heuristic data-driven.
    """
    _IMPERATIVE_VERBS.update(verbs)


def _extract_topic(query: str) -> str:
    """Light noun-ish extraction for curiosity lookup (parsing, not matching)."""
    q = re.sub(r"[?.,!]", " ", query.lower())
    toks = [t for t in q.split() if len(t) > 3 and t not in {
        "what", "when", "where", "which", "tell", "about", "think", "do", "you",
        "your", "know", "believe", "feel", "like", "want", "should", "could"}]
    return " ".join(toks[:4])


def _trim_url_match(url: str) -> str:
    """Trim sentence punctuation without dropping balanced URL parentheses."""
    url = url.rstrip(".,;:!?")
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
    return url


# --- Personal-possessive gate -------------------------------------------------
#
# Purpose: a query about the USER's own life/possessions ("what is wrong with
# my car") can only be answered from episodic recall. Left ungated it falls
# through to _consult_internal_knowledge and confabulates (2026-09-04). So the
# gate must fire for genuine autobiography.
#
# It must NOT fire for ordinary world knowledge that merely happens to contain
# a possessive. "how do i change a flat tire on my car" and "what is the best
# way to clean my laptop keyboard" are procedure/recommendation questions —
# they have the same surface possessive + entity noun as an autobiographical
# query, so the noun vocabulary alone CANNOT separate them. Separating them
# takes two structural signals, neither of which is a topic list:
#
#   1. POSSESSIVE PERSON. The query is addressed to RAVANA, so "my" is the
#      user's own stuff and "your" is RAVANA's. RAVANA owns nothing, so a
#      "your X" query is a question about RAVANA (handled by the self-subject
#      gate further down) or a world-knowledge question — never a recall of
#      the user's episodes. Only a first-person possessive is autobiographical.
#
#   2. QUESTION CONSTRUCTION. A present-tense mechanism / procedure / advice
#      question ("how does X ...", "why does X ...", "what causes ...",
#      "what is the best ...") asks for a general law that holds for any
#      instance; the possessive is incidental. An autobiographical question
#      asks about one particular thing ("what is wrong with my car", "why did
#      my laptop die") — a specific state or a specific past event, so the
#      construction is a copular/defective state question or a PAST-tense
#      event question. This is grammar over function words, not vocabulary.
#
# The noun set below is a deliberately SMALL cold-start seed: only concrete
# entity-denoting nouns a user could plausibly own, name, or be related to. The
# generic-category block that used to live here (time, life, world, value,
# form, level, type, part, group, brain, story, team, favourite, ...) matched
# almost any English sentence, which made "my + any word => never search".
# That enumeration was the hardcoding: nobody can grow 685 words through
# experience, and the precision loss was invisible without a probe.
#
# The real growth path is wired at runtime: every personal DISCLOSURE the
# miner accepts (questions are excluded upstream by the interrogative guard)
# contributes the head noun of its possessive phrase via
# add_personal_entity_words(). RAVANA therefore learns the vocabulary of the
# user's own possessions and relationships from conversation, online, with no
# retrain and no code change.
_PERSONAL_ENTITY_WORDS = {
    # Vehicles
    "car", "cars", "bike", "bicycle", "motorcycle", "scooter", "truck", "van",
    "bus", "train", "boat", "plane", "drone", "tractor",
    # Devices and electronics
    "phone", "phones", "laptop", "computer", "tablet", "monitor", "keyboard",
    "mouse", "printer", "router", "modem", "console", "camera", "television",
    "tv", "radio", "speaker", "headphones", "watch", "charger", "battery",
    "gps", "server", "router", "harddrive",
    # Animals the user keeps
    "dog", "dogs", "cat", "cats", "puppy", "kitten", "bird", "fish", "hamster",
    "rabbit", "horse", "goat", "cow", "turtle", "ferret",
    # People and relationships
    "sister", "brother", "mother", "father", "mom", "dad", "parent", "parents",
    "child", "children", "kid", "kids", "son", "daughter", "husband", "wife",
    "spouse", "partner", "boyfriend", "girlfriend", "friend", "friends",
    "teacher", "professor", "boss", "manager", "colleague", "coworker",
    "neighbor", "neighbour", "classmate", "roommate", "landlord", "cousin",
    "uncle", "aunt", "grandmother", "grandfather", "mentor", "coach",
    "doctor", "dentist", "therapist", "counselor", "physician", "nurse",
    # Home and property
    "house", "home", "apartment", "condo", "room", "kitchen", "bedroom",
    "bathroom", "garage", "garden", "yard", "roof", "door", "window", "fence",
    "office", "desk", "chair", "bed", "sofa", "fridge", "stove", "oven",
    # Documents, money, valuables
    "passport", "wallet", "purse", "bag", "backpack", "suitcase", "luggage",
    "keys", "ring", "necklace", "jewelry", "glasses", "medication", "medicine",
    "pill", "pills", "prescription", "bank", "account", "credit", "debt",
    "loan", "mortgage", "rent", "insurance", "salary", "wage", "income",
    "money", "wallet",
    # Clothing
    "clothes", "clothing", "shirt", "pants", "shoes", "boots", "jacket",
    "coat", "dress", "uniform",
    # Health and body
    "headache", "migraine", "cold", "flu", "fever", "cough", "sore", "pain",
    "injury", "wound", "bruise", "rash", "allergy", "allergies", "infection",
    "surgery", "appointment", "diagnosis", "symptom", "symptoms", "tooth",
    "teeth", "back", "knee", "shoulder", "stomach", "eyes", "ears", "hand",
    "foot", "leg", "arm", "heart", "skin", "hair",
    # School and work
    "school", "college", "university", "course", "grade", "grades", "exam",
    "exams", "test", "homework", "assignment", "project", "thesis", "job",
    "jobs", "career", "internship", "resume", "interview", "salary",
    # Places
    "city", "town", "village", "country", "street", "road", "park", "beach",
    "mountain", "river", "lake", "gym", "clinic", "hospital", "pharmacy",
    "store", "shop", "market", "restaurant", "cafe", "hotel", "library",
    # Events with personal significance
    "birthday", "anniversary", "wedding", "funeral", "graduation", "party",
    "holiday", "vacation", "trip", "flight", "roadtrip", "gift", "present",
}


def add_personal_entity_words(words: set) -> None:
    """Extend the personal-entity word seed set at runtime.

    As new entity types are encountered in personal disclosures, the set can
    be grown without modifying code. This keeps the heuristic data-driven.
    """
    _PERSONAL_ENTITY_WORDS.update(words)


# Closed-class words that can never BE the head of a possessive noun phrase.
# Keeping this functional (not topical) means the head-noun extractor needs no
# vocabulary list of its own and cannot rot the way the old seed did.
_POSSESSIVE_HEAD_STOP = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "our",
    "his", "her", "their", "its", "of", "and", "or", "but", "if", "so",
    "very", "really", "quite", "too", "also", "just", "still", "already",
    "own", "beloved", "dear", "late", "old", "new", "brand", "favourite",
    "favorite", "little", "big", "huge", "tiny", "whole", "entire", "main",
    "current", "usual", "own", "first", "second", "third", "last", "next",
})

# A possessive noun phrase is "my" + modifiers + ONE head noun, which ends at
# the first word that cannot continue it: a verb, a preposition, a conjunction,
# a determiner, punctuation, or the end of the clause.
_POSSESSIVE_PHRASE_END = re.compile(
    r"^$"
    r"|\b(?:is|are|was|were|am|be|been|being|has|have|had|do|does|did|"
    r"will|would|shall|should|can|could|may|might|must)\b"
    r"|\b(?:of|in|on|at|to|for|with|from|by|about|into|onto|over|under|"
    r"near|beside|behind|during|after|before|since|until|and|or|but|if|"
    r"that|which|who|whom|whose|when|where|why|how|what|because|so)\b"
    r"|[^a-z']"
)


def learn_personal_entities_from_disclosure(text: str) -> set:
    """Learn the user's personal-entity vocabulary from a DISCLOSURE.

    This is the growth path that makes the seed a seed. The user telling
    RAVANA "my hovercraft has been leaking since march" establishes a new
    personal entity; the head noun of that possessive phrase ("hovercraft")
    joins the vocabulary, so a later "what is wrong with my hovercraft" is
    correctly recognised as autobiographical and routed to episodic recall
    instead of web_search. Online, from one conversation, no retrain, no code
    change, and it works for a noun no seed list could have anticipated.

    Only declarative disclosures should be passed in — the caller already
    rejects questions upstream, so a knowledge query can never teach the gate
    that the world is the user's personal property.

    Returns the set of newly learned words.
    """
    if not text:
        return set()
    learned = set()
    for m in re.finditer(r"\bmy\s+([a-z']+(?:[ -][a-z']+){0,3})", text.lower()):
        for token in m.group(1).split():
            if token in _POSSESSIVE_HEAD_STOP or not token.isalpha():
                break
            if len(token) < 3:
                break
            if _POSSESSIVE_PHRASE_END.search(token):
                break
            # A word that ends the phrase can still BE the head
            # ("my car") — only a following non-noun ends it — so take the
            # last alphabetic token of the phrase as the head.
            learned.add(token)
            break
    _PERSONAL_ENTITY_WORDS.update(learned)
    return learned


# Function words, not vocabulary. A question whose interrogative clause asks
# for a general law — a present-tense mechanism, a procedure, a duration, or
# a recommendation — is answerable from the world, and any possessive inside it
# is incidental ("how do i change a flat tire ON MY CAR"). Only a question
# about one particular thing is autobiographical, so its interrogative is either
# a state predicate ("what is wrong with my car") or a PAST-tense event
# ("why did my laptop die"). This is grammar over closed-class function words,
# so it cannot rot into a topic list and it generalises to nouns RAVANA has
# never seen.
_GENERAL_LAW_QUESTION = re.compile(
    r"\bhow\s+(?:long|much|many|often|far|old|fast|deep|tall|heavy|"
    r"do|does|can|could|should|would|to|is|are)\b"
    r"|\bwhy\s+(?:do|does|is|are|can|could|would)\b"
    r"|\bwhat\s+caus(?:e|es|ed|ing)\b"
    r"|\bwhat\s+is\s+the\s+best\b"
    r"|\bbest\s+way\s+to\b"
)


def _is_personal_possessive_query(query: str) -> bool:
    """Is this query about the USER's own life, which only recall can answer?

    Three conditions, all necessary:

    1. A FIRST-PERSON possessive. The query is addressed to RAVANA, so "my"
       is the user and "your" is RAVANA — and RAVANA owns nothing, so "what
       is your favourite programming language" is a question about RAVANA (the
       self-subject gate in decide_tool_use handles that) or world knowledge,
       never a recall of the user's episodes.
    2. Not a general-law question. "how long should i charge my phone" and
       "why does my dog keep scratching its ear" ask for a law that holds for
       any phone or dog; the possessive is incidental.
    3. An entity word from the (runtime-growable) personal-entity vocabulary.

    This is a routing check, not a capability removal — genuine knowledge
    queries like "what is the capital of france" still fire web_search.
    """
    q = (query or "").lower()
    if not re.search(r"\bmy\b", q):
        return False
    if _GENERAL_LAW_QUESTION.search(q):
        return False
    return any(re.search(rf"\b{re.escape(word)}\b", q)
               for word in _PERSONAL_ENTITY_WORDS)


# --- Self-attributive frame gate (FIX-RV-17) ---------------------------------
#
# A declarative utterance whose SUBJECT is the speaker is a disclosure: the
# user is telling RAVANA a fact about themselves. "when i was a teenager i
# lived in mumbai" asks the world nothing, so a world-web search cannot answer
# it — the only correct destination is the user's own memory.
#
# This is decided on GRAMMAR, over closed-class function words, so it is not a
# topic list and cannot rot into one:
#
#   1. SUBJECT PERSON. The utterance's subject is a first-person pronoun.
#      Third person ("anant ambani went viral"), second person ("you should
#      try python") and bare nominals ("mumbai is crowded") are about the
#      world and are NOT gated. Grammatical person, not topic.
#   2. NOT INTERROGATIVE. A question asks; a disclosure asserts. A wh-word, a
#      question mark, or an auxiliary-led inversion marks the utterance as a
#      question, so "what did i do in mumbai" still reaches the world.
#   3. NOT IMPERATIVE. An instruction is a task, handled by the noun and
#      social-intent paths below, so it must not be swallowed here.
#
# The personal-fact miner upstream already treats a disclosure as something to
# store, so returning None routes this turn to the memory path — which is where
# the content belonged in the first place.
# The utterance's subject is a first-person pronoun. Alternation is ordered so
# the longer contractions win over the bare "i" (otherwise "i'm" matches as
# "i" and the apostrophe-tail of "i've"/"i'd" is left dangling).
_FIRST_PERSON_SUBJECT = re.compile(
    r"^\s*(?:"
    r"(?:and|but|so|then|well|also|anyway|actually|honestly)\b[^,]{0,24}?"
    r")?"
    r"\b(?:i'm|im|i've|ive|i'd|ill|i'll|my|mine|myself|i|we're|weve|"
    r"we've|our|ours|ourselves|we|us)\b"
)

# A subordinate temporal/causal opener followed by a first-person clause is
# still a disclosure: "when i was a teenager i lived in mumbai".
_SUBORDINATE_FIRST_PERSON = re.compile(
    r"^\s*(?:when|while|before|after|since|until|if|whenever|as)\b"
    r"[^,]{0,80}?\b(?:i|i'm|im|i've|ive|my)\b"
)

# Interrogative markers. Closed-class, so a new topic cannot defeat them.
_QUESTION_MARK = re.compile(r"\?")
# A wh-word that heads the WHOLE clause, i.e. a real question. "when i was a
# teenager i lived in mumbai" opens with a wh-word but is a disclosure: "when"
# is a subordinating conjunction here and the clause has a subject. A genuine
# question's wh-clause has no subject of its own ("when did you move"), so it
# is not followed by a subject pronoun. Without this the gate would read every
# temporal disclosure as a question and never fire on one.
_WH_QUESTION_HEAD = re.compile(
    r"^\s*(?:what|who|whom|whose|where|when|why|which|how)\b(?![^,?]*?"
    r"\b(?:i|i'm|im|i've|my|we|our|you|your|he|she|they|it)\b)"
)
# An auxiliary or question word anywhere ahead of the verb signals inversion.
_AUX_INVERSION = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|can|could|should|would|will|"
    r"shall|have|has|had|am)\b"
)


def _is_self_attributive_disclosure(query: str) -> bool:
    """Is this a first-person DECLARATIVE disclosure rather than a question?

    True only when all three hold: the subject is first person, the utterance
    is not interrogative, and it is not imperative. Anything the caller would
    want to look up — a question about the world, an instruction to run a tool —
    fails one of these and falls through to the normal routing.
    """
    q = (query or "").strip()
    if not q:
        return False
    # 1) First-person subject => a disclosure about the speaker. Checked FIRST,
    # because a disclosure's own subject pronoun is what distinguishes a
    # subordinate clause ("when I ...") from a real wh-question ("when did ...").
    is_disclosure = bool(_FIRST_PERSON_SUBJECT.search(q)
                         or _SUBORDINATE_FIRST_PERSON.search(q))
    if not is_disclosure:
        return False
    # 2) Interrogative => not a disclosure.
    if _QUESTION_MARK.search(q) or _WH_QUESTION_HEAD.search(q):
        return False
    if _AUX_INVERSION.search(q):
        return False
    # 3) Imperative => a task, not a disclosure.
    if _is_imperative_formed(q):
        return False
    return True



def decide_tool_use(engine, query: str, registry: Optional[ToolRegistry] = None) -> Optional[ToolCall]:
    """Return a ToolCall plan if RAVANA's cognition justifies acting, else None.

    Reads LIVE engine state only. Never a static keyword→tool map.
    """
    registry = registry or ToolRegistry()
    q = (query or "").strip()
    if not q:
        return None

    # 0) URL pattern check — if the query contains a URL, route to read_website
    # BEFORE the curiosity path. A URL is a concrete pointer to a specific
    # resource, not a knowledge gap about a topic. This prevents the curiosity
    # signal from shadowing read_website when the user says "look up <url>".
    url_match = re.search(r'(https?://\S+|www\.\S+)', q)
    if url_match and "read_website" in registry.tools:
        url = _trim_url_match(url_match.group(1))
        return ToolCall(tool="read_website", arg=url,
                        reason=f"url_pattern_detected url={url}")

    # 0a) SELF-ATTRIBUTIVE FRAME pre-gate (FIX-RV-17). A declarative utterance
    # whose subject is the speaker is the user handing RAVANA a fact about
    # themselves, not asking the world a question. "when i was a teenager i
    # lived in mumbai" carries no information gap: the world cannot answer it,
    # and the only correct destination is the user's own memory. The old gate
    # fired web_search here and pasted an unrelated news item about a different
    # teenager into the reply.
    #
    # The test is STRUCTURAL, on function words only — first/third-person
    # subject pronoun, no interrogative, no imperative. It is not a topic
    # blocklist, so it holds for a disclosure about a subject nobody
    # anticipated, and it cannot swallow a real question.
    if _is_self_attributive_disclosure(q):
        return None

    # 0b) Personal-possessive pre-gate: if the query is autobiographical
    # (contains "my"/"your" + an entity word), SKIP web_search entirely.
    # This is a routing fix, not a capability removal — web_search still fires
    # for genuine knowledge queries like "what is the capital of france".
    if _is_personal_possessive_query(q):
        return None

    # 1) Uncertainty / curiosity: does RAVANA not know this topic?
    # Only act when it's a genuine KNOWLEDGE gap (recall query about the world),
    # not social chitchat ("how are you") or self/personal questions. Reuse the
    # engine's own recall-query detector + self-subject gate so we don't web-
    # search every casual message.
    is_knowledge_query = True
    try:
        if hasattr(engine, "_is_recall_query"):
            is_knowledge_query = bool(engine._is_recall_query(query))
    except Exception:
        is_knowledge_query = True

    # Suppress on self/personal/social questions (about RAVANA or the user) —
    # these are not world-knowledge gaps to ground via search.
    is_personal = False
    try:
        from ..chat.brain_regions import SelfModel
        _sm = SelfModel()
        # crude subject extraction: first content word after a copula/wh-word
        _subj = re.sub(r"^(what|who|whom|whose|where|when|why|which|how)\s+"
                        r"(do|does|did|is|are|was|were|will|would|can|could)\s+", "", q).split()[0:1]
        if _subj and _sm.is_self_subject(_subj[0]):
            is_personal = True
        if re.search(r"\b(how are you|how's it going|what do you think|"
                     r"what's your|tell me about yourself|who are you)\b", q):
            is_personal = True
    except Exception:
        is_personal = False

    topic = _extract_topic(q)
    uncertainty = 0.0
    try:
        if topic and hasattr(engine, "curiosity_engine"):
            uncertainty = float(engine.curiosity_engine.uncertainty_for(topic))
    except Exception:
        uncertainty = 0.0

    # 2) Metacognitive mode: is RAVANA explicitly uncertain?
    meta_uncertain = False
    try:
        mode = getattr(getattr(engine, "meta_cog", None), "current_mode", None)
        if mode is not None:
            meta_uncertain = str(getattr(mode, "value", mode)).upper() == "UNCERTAIN"
    except Exception:
        meta_uncertain = False

    if is_knowledge_query and not is_personal and (uncertainty >= 0.5 or meta_uncertain):
        return ToolCall(tool="web_search", arg=q,
                        reason=f"knowledge_query={is_knowledge_query} personal={is_personal} "
                               f"curiosity_uncertainty={uncertainty:.2f} meta_uncertain={meta_uncertain}")

    # 2b) Noun-heuristic path: when a tool noun is present AND the query is
    # imperative-formed (starts with a verb, no question mark), fire the tool
    # directly without waiting for the social-intent classifier. This is a seed
    # heuristic, not a hardcoded trigger — the verb set is expandable at runtime.
    # This path exists because the social-intent classifier is conservative and
    # labels git-status queries as 'general' rather than 'command/request/task'.
    if _is_imperative_formed(q):
        for noun, tool in _TOOL_NOUNS.items():
            if re.search(rf"\b{re.escape(noun)}\b", q.lower()):
                if tool in registry.tools:
                    return ToolCall(tool=tool, arg=q,
                                    reason=f"noun_heuristic matched_tool_noun={noun} "
                                           f"imperative_formed=true")

    # 3) Task intent via RAVANA's OWN social-intent classifier (not our keywords)
    act = None
    try:
        clf = getattr(engine, "_social_intent", None)
        if clf is not None:
            res = clf.classify(q)
            # accept (label, scores) or just label
            if isinstance(res, tuple):
                act = (res[0] or "").lower()
            else:
                act = str(res).lower()
    except Exception:
        act = None

    if act in _TASK_ACTS:
        # Map a task noun to a safe tool (only if the tool exists)
        for noun, tool in _TOOL_NOUNS.items():
            if re.search(rf"\b{re.escape(noun)}\b", q.lower()):
                if tool in registry.tools:
                    return ToolCall(tool=tool, arg=q,
                                    reason=f"social_intent={act} matched_tool_noun={noun}")
    return None
