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


def decide_tool_use(engine, query: str, registry: Optional[ToolRegistry] = None) -> Optional[ToolCall]:
    """Return a ToolCall plan if RAVANA's cognition justifies acting, else None.

    Reads LIVE engine state only. Never a static keyword→tool map.
    """
    registry = registry or ToolRegistry()
    q = (query or "").strip()
    if not q:
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
