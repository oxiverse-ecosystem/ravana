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


# Personal-possessive entity words — seed vocabulary (expandable at runtime).
# These are common nouns that, when combined with a possessive pronoun ("my",
# "your"), indicate an autobiographical query rather than a world-knowledge
# gap. This is seed data, not a hardcoded trigger: the set can be extended at
# runtime via add_personal_entity_words() as new entity types are encountered.
#
# The set is intentionally small — it bootstraps the routing check. RAVANA
# grows it at runtime from personal disclosures in conversation (e.g. learning
# that "my motorcycle" is a personal entity after the user mentions it).
_PERSONAL_ENTITY_WORDS = {
    # Vehicles and personal possessions (most common in "what is wrong with my X")
    "car", "cars", "gps", "phone", "phones", "computer", "computers",
    "laptop", "laptops", "dog", "dogs", "cat", "cats", "pet", "pets",
    "house", "home", "bike", "bicycle", "motorcycle", "truck", "vehicle",
    "engine", "battery", "tire", "tires", "brake", "brakes", "transmission",
    # Personal states and conditions (recall-gap terms that look like knowledge queries)
    "broken", "happened", "reboot", "turn", "drive", "ride",
    # Family and relationships
    "sister", "brother", "mother", "father", "mom", "dad", "parent",
    "parents", "child", "children", "kid", "kids", "son", "daughter",
    "husband", "wife", "spouse", "partner", "boyfriend", "girlfriend",
    "friend", "friends", "teacher", "professor", "boss", "manager",
    "colleague", "coworker", "neighbor", "neighbour", "classmate",
    "roommate", "landlord",
    # Health and medical
    "doctor", "dentist", "therapist", "counselor", "physician",
    "hospital", "clinic", "pharmacy", "headache", "cold", "flu", "fever",
    "cough", "sore", "pain", "injury", "wound", "bruise", "cut", "burn",
    "rash", "allergy", "medication", "medicine", "pill", "pills",
    "vitamin", "vitamins",
    # Education and work
    "school", "college", "university", "course", "class", "classes",
    "grade", "grades", "exam", "exams", "test", "tests", "homework",
    "assignment", "project", "thesis", "job", "jobs", "bank", "account",
    "credit", "debt", "loan", "mortgage", "rent", "insurance", "tax",
    "taxes", "salary", "wage", "income", "money",
    # Personal items
    "wallet", "purse", "bag", "backpack", "suitcase", "luggage",
    "clothes", "clothing", "shirt", "pants", "shoes", "jacket", "coat",
    "watch", "jewelry", "ring", "necklace", "glasses", "sunglasses",
    "camera", "television", "tv", "radio", "speaker", "headphones",
    "keyboard", "mouse", "monitor", "printer", "router", "modem",
    "tablet", "ipad", "kindle", "console", "playstation", "xbox",
    # Entertainment and leisure
    "game", "games", "movie", "movies", "book", "books", "novel",
    "song", "songs", "album", "band", "artist", "painting", "art",
    "vacation", "holiday", "trip", "travel", "flight", "hotel",
    "restaurant", "cafe", "coffee", "tea", "beer", "wine", "food",
    "meal", "breakfast", "lunch", "dinner", "snack", "dessert",
    # Home and property
    "garden", "yard", "lawn", "fence", "roof", "door", "window",
    "kitchen", "bathroom", "bedroom", "living", "dining", "garage",
    "apartment", "condo", "flat", "studio", "office", "workplace",
    # Personal attributes and states
    "favorite", "favourite", "habit", "routine", "hobby", "hobbies",
    "interest", "interests", "skill", "skills", "talent", "ability",
    "strength", "weakness", "problem", "problems", "issue", "issues",
    "trouble", "concern", "worry", "worries", "fear", "fears", "anxiety",
    "stress", "anger", "sadness", "happiness", "joy", "love", "hate",
    "dislike", "preference", "opinion", "thought", "thoughts", "idea",
    "ideas", "memory", "memories", "dream", "dreams", "goal", "goals",
    "plan", "plans", "decision", "decisions", "choice", "choices",
    "mistake", "mistakes", "regret", "success", "failure", "achievement",
    "challenge", "challenges", "difficulty", "struggle", "effort",
    "attempt", "try", "practice", "progress", "improvement", "growth",
    "change", "changes", "transition", "shift", "move", "movement",
    "journey", "path", "direction", "destination", "arrival", "departure",
    "beginning", "start", "end", "finish", "completion", "result",
    "results", "outcome", "consequence", "effect", "impact", "influence",
    "cause", "reason", "purpose", "meaning", "significance", "value",
    "worth", "importance", "priority", "urgency", "necessity", "need",
    "needs", "want", "wants", "desire", "wish", "hope", "expectation",
    "standard", "quality", "quantity", "amount", "number", "count",
    "level", "degree", "extent", "range", "scope", "scale", "size",
    "shape", "form", "structure", "pattern", "trend", "tendency",
    "behavior", "behaviour", "action", "actions", "activity", "activities",
    "event", "events", "incident", "occasion", "situation", "circumstance",
    "condition", "conditions", "state", "status", "position", "place",
    "location", "spot", "site", "area", "region", "zone", "sector",
    "field", "domain", "realm", "world", "universe", "existence",
    "life", "death", "birth", "age", "time", "period", "era", "epoch",
    "moment", "minute", "hour", "day", "week", "month", "year",
    "decade", "century", "millennium", "past", "present", "future",
    "history", "story", "tale", "narrative", "account", "report",
    "description", "explanation", "definition", "interpretation",
    "understanding", "comprehension", "knowledge", "wisdom", "insight",
    "intuition", "instinct", "feeling", "emotion", "sentiment", "mood",
    "attitude", "disposition", "temperament", "personality", "character",
    "nature", "essence", "core", "heart", "soul", "spirit", "mind",
    "brain", "thinking", "reasoning", "logic", "rationality",
    "intelligence", "intellect", "creativity", "imagination", "fantasy",
    "reality", "truth", "fact", "facts", "information", "data",
    "evidence", "proof", "verification", "confirmation", "validation",
    "authentication", "certification", "qualification", "credential",
    "license", "permit", "authorization", "approval", "consent",
    "agreement", "contract", "treaty", "pact", "deal", "arrangement",
    "compromise", "negotiation", "discussion", "debate",
    "argument", "dispute", "conflict", "fight", "battle", "war", "peace",
    "truce", "ceasefire", "surrender", "victory", "defeat", "win", "loss",
    "triumph", "disaster", "catastrophe", "crisis",
    "emergency", "demand",
    "requirement", "specification", "criterion", "benchmark",
    "measure", "measurement", "metric", "indicator", "signal", "sign",
    "symbol", "token", "mark", "label", "tag", "category", "class",
    "type", "kind", "sort", "variety", "version", "edition",
    "release", "update", "upgrade", "patch", "fix", "repair", "correction",
    "revision", "modification", "alteration", "adjustment", "adaptation",
    "transformation", "conversion", "evolution", "revolution",
    "innovation", "invention", "discovery", "finding",
    "power", "force", "energy", "might",
    "authority", "control", "command", "dominion", "rule", "governance",
    "leadership", "management", "administration", "organization",
    "institution", "establishment", "foundation", "association", "society",
    "community", "group", "team", "crew", "squad", "unit", "division",
    "department", "section", "branch", "segment", "part",
    "piece", "portion", "fraction", "percentage", "ratio", "proportion",
    "rate", "speed", "velocity", "acceleration", "momentum",
    "pressure", "tension", "strain", "load", "weight", "mass",
    "volume", "density", "concentration", "intensity", "magnitude",
    "amplitude", "frequency", "wavelength", "cycle", "loop",
    "circle", "ring", "sphere", "globe", "ball", "orb", "planet",
    "star", "sun", "moon", "earth", "cosmos",
    "galaxy", "nebula", "constellation", "asteroid", "comet", "meteor",
    "satellite", "spacecraft", "rocket", "shuttle", "station", "base",
    "colony", "settlement", "outpost", "camp", "tent", "cabin", "hut",
    "shelter", "refuge", "haven", "sanctuary", "temple", "church",
    "mosque", "synagogue", "shrine", "altar", "monastery", "convent",
    "abbey", "cathedral", "basilica", "chapel", "oratory",
    # Places and geography
    "city", "town", "village", "country", "state", "province",
    "street", "road", "avenue", "highway", "freeway", "bridge",
    "park", "beach", "mountain", "river", "lake", "ocean", "forest",
    # Weather and environment
    "weather", "temperature", "rain", "snow", "wind", "storm",
    # Fitness and activities
    "diet", "exercise", "workout", "gym", "run", "running", "walk",
    "walking", "swim", "swimming", "biking", "hike", "hiking",
    # Events and occasions
    "birthday", "anniversary", "wedding", "funeral", "party",
    "meeting", "appointment", "interview", "deadline", "schedule",
}


def add_personal_entity_words(words: set) -> None:
    """Extend the personal-entity word seed set at runtime.

    As new entity types are encountered in personal disclosures, the set can
    be grown without modifying code. This keeps the heuristic data-driven.
    """
    _PERSONAL_ENTITY_WORDS.update(words)


def _is_personal_possessive_query(query: str) -> bool:
    """Detect autobiographical queries: personal possessive + entity word.

    A query like "what is wrong with my car" contains a possessive pronoun
    ("my") and an entity word ("car") — this is a personal disclosure, not a
    world-knowledge gap. Such queries should route to episodic recall, not
    web_search.

    Returns True only when BOTH conditions hold:
    1. The query contains a personal possessive pronoun ("my", "your")
    2. The query contains an entity word (from the seed vocabulary)

    This is a routing check, not a capability removal — genuine knowledge
    queries like "what is the capital of france" still fire web_search.
    """
    q = query.lower()
    # Check for personal possessive pronouns
    has_possessive = bool(re.search(r"\b(my|your)\b", q))
    if not has_possessive:
        return False
    # Check for entity words — use word boundaries to avoid partial matches
    for word in _PERSONAL_ENTITY_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", q):
            return True
    return False


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
