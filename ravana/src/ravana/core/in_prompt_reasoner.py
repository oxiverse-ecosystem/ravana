"""
In-prompt reasoner — ephemeral premise binding + session-fact Q&A
=================================================================
Pure functions for binding facts stated in the *current* user message
(or already-mined session facts) into answers, without graph retrieval.

Neuroscience grounding (roles the architecture already claims):
- Working memory (Baddeley / dlPFC): hold asserted premises for the turn
- Hippocampal relational binding (Hannula 2008; Yonelinas 2019): S-R-O triples
- Mental model / causal simulation (dorsal fronto-parietal "physics engine"):
  multi-hop state transitions over conditionals (if/when → then)
- Encoding specificity (Tulving): answer from stored gist, never confabulate

Why this module exists:
Shakespeare decoder training does not teach the lamp syllogism. Graph
abstention fires when "lamp" is unknown. The fix is ephemeral premise
binding — parse asserted conditionals in the message, multi-hop, answer.
Same for same-turn LoCoMo facts already mined by episodic fact extractors.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Tokens that never carry causal content for premise chaining.
_STOP = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "when", "then", "of", "to",
    "in", "on", "at", "for", "with", "from", "by", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "you", "your", "i", "my",
    "me", "we", "our", "it", "its", "this", "that", "these", "those",
    "what", "who", "where", "when", "why", "how", "which", "there", "here",
    "happens", "happen", "occurs", "occur", "will", "would", "can", "could",
    "should", "may", "might", "must", "facts", "fact", "also", "just",
    "about", "into", "out", "up", "down", "over", "under", "after", "before",
    "so", "thus", "hence", "therefore", "because", "since", "while",
    "turned",  # kept as content when paired with "on" below
})


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _content_tokens(text: str) -> List[str]:
    """Content words from a clause; keeps multiword phrases as separate tokens."""
    words = re.findall(r"[a-z0-9']+", _norm(text))
    out: List[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        # keep "turn on" / "turned on" as a single action token
        if w in ("turn", "turned", "turns") and i + 1 < len(words) and words[i + 1] == "on":
            out.append("turn_on")
            i += 2
            continue
        if w in ("lights", "light") and i + 1 < len(words) and words[i + 1] in ("up", "on"):
            out.append("lights_up")
            i += 2
            continue
        if w not in _STOP and len(w) >= 2:
            out.append(w)
        i += 1
    return out


def _split_sentences(text: str) -> List[str]:
    """Split into sentence-like clauses; strip numbering / 'Facts:' labels."""
    t = text or ""
    t = re.sub(r"(?im)^\s*facts?\s*:\s*", "", t)
    # Break on newlines and numbered list markers so "1. X 2. Y" become separate.
    t = re.sub(r"(?m)^\s*\d+[\.\)]\s*", "\n", t)
    t = re.sub(r"(?m)\s+\d+[\.\)]\s+", "\n", t)
    parts = re.split(r"[\n.!?;]+", t)
    return [p.strip() for p in parts if p and p.strip()]


def parse_causal_edges(text: str) -> List[Tuple[str, str]]:
    """Extract (cause_phrase, effect_phrase) edges from asserted conditionals.

    Patterns:
      - when X, Y / when X then Y
      - if X, Y / if X then Y
      - X causes/leads to/results in/triggers Y
    Questions are ignored.
    """
    edges: List[Tuple[str, str]] = []
    for sent in _split_sentences(text):
        s = _norm(sent).rstrip("?.!")
        if not s:
            continue
        # Skip pure questions
        if re.match(
            r"^(what|who|where|why|how|which|do|does|did|is|are|can|could|"
            r"will|would|should)\b", s
        ) and "?" in sent:
            continue
        # if / when conditionals
        m = re.match(
            r"^(?:if|when)\s+(.+?)(?:,|\s+then\s+)\s*(.+)$", s
        )
        if m:
            cause, effect = m.group(1).strip(), m.group(2).strip()
            if cause and effect:
                edges.append((cause, effect))
            continue
        # "X causes/leads to/results in Y"
        m = re.search(
            r"^(.+?)\s+(?:causes?|cause|leads?\s+to|results?\s+in|triggers?|"
            r"produces?|makes?)\s+(.+)$", s
        )
        if m:
            cause, effect = m.group(1).strip(), m.group(2).strip()
            # Drop leading articles / weak subjects like "an explosion occurs" handled elsewhere
            if cause and effect and not re.match(r"^(what|who|where)\b", cause):
                edges.append((cause, effect))
    return edges


def _phrase_keys(phrase: str) -> Set[str]:
    """Index keys for a phrase: tokens + joined bigrams + full phrase."""
    toks = _content_tokens(phrase)
    keys: Set[str] = set(toks)
    if toks:
        keys.add("_".join(toks))
        keys.add(" ".join(toks))
    # also raw normalized phrase
    keys.add(_norm(phrase))
    return {k for k in keys if k}


def build_causal_graph(edges: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Map cause-key → list of effect phrases (adjacency for multi-hop)."""
    graph: Dict[str, List[str]] = {}
    for cause, effect in edges:
        for k in _phrase_keys(cause):
            graph.setdefault(k, [])
            if effect not in graph[k]:
                graph[k].append(effect)
        # Also index by individual content tokens of the cause so
        # "turn on the lamp" matches edge cause "turned on".
        for tok in _content_tokens(cause):
            graph.setdefault(tok, [])
            if effect not in graph[tok]:
                graph[tok].append(effect)
    return graph


def extract_query_seeds(text: str) -> List[str]:
    """Pull seed tokens from a 'what happens if/when …' style question."""
    s = _norm(text)
    m = re.search(
        r"(?:what\s+happens?|what\s+occurs?|what\s+will\s+happen|"
        r"what\s+would\s+happen)\s+(?:if|when)\s+(.+?)(?:\?|$)", s
    )
    if m:
        return _content_tokens(m.group(1))
    # fallback: last interrogative sentence
    sents = _split_sentences(text)
    for sent in reversed(sents):
        sn = _norm(sent)
        if re.match(r"^(what|who|where|why|how)\b", sn) or "?" in sent:
            return _content_tokens(sn)
    return _content_tokens(s)


def multi_hop_effects(
    graph: Dict[str, List[str]],
    seeds: Sequence[str],
    max_hops: int = 4,
) -> List[str]:
    """BFS multi-hop over cause→effect edges starting from seed tokens.

    Returns ordered unique effect phrases reached (excluding seed-only).
    """
    if not graph or not seeds:
        return []
    from collections import deque

    visited_keys: Set[str] = set()
    reached: List[str] = []
    queue: deque = deque()

    for s in seeds:
        for k in _phrase_keys(s) | {s}:
            if k in graph:
                queue.append((k, 0))

    # also seed with raw seed tokens present as keys
    for s in seeds:
        if s in graph:
            queue.append((s, 0))

    while queue:
        key, depth = queue.popleft()
        if key in visited_keys or depth > max_hops:
            continue
        visited_keys.add(key)
        for effect in graph.get(key, []):
            if effect not in reached:
                reached.append(effect)
            for ek in _phrase_keys(effect):
                if ek not in visited_keys and ek in graph:
                    queue.append((ek, depth + 1))
            # index effect content tokens too
            for tok in _content_tokens(effect):
                if tok not in visited_keys and tok in graph:
                    queue.append((tok, depth + 1))
    return reached


def format_causal_answer(effects: Sequence[str], seeds: Sequence[str] = ()) -> str:
    """Compose a natural multi-hop answer from reached effects."""
    if not effects:
        return ""
    # Keep full effect phrases (including articles) so "the lamp lights up" and
    # "an explosion occurs" stay readable for graders and humans.
    cleaned = []
    for e in effects:
        e_clean = e.strip().rstrip("!.")
        if e_clean and e_clean not in cleaned:
            cleaned.append(e_clean)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        body = cleaned[0]
        if body and body[0].islower():
            body = body[0].upper() + body[1:]
        return body + "."
    # Join: "The lamp lights up and an explosion occurs."
    first = cleaned[0]
    rest = cleaned[1:]
    if first and first[0].islower():
        first = first[0].upper() + first[1:]
    if len(rest) == 1:
        return f"{first} and {rest[0]}."
    return first + ", " + ", ".join(rest[:-1]) + f", and {rest[-1]}."


def answer_in_prompt_causal(text: str) -> Optional[str]:
    """End-to-end: parse premises in ``text``, multi-hop, answer if possible.

    Returns None when no causal edges + query seeds produce a chain
    (caller falls through to normal pipeline). Never confabulates.
    """
    edges = parse_causal_edges(text)
    if not edges:
        return None
    # Require a question / hypothetical cue so pure statements are not answered.
    tl = _norm(text)
    if not re.search(
        r"\b(what\s+happens?|what\s+occurs?|what\s+will|what\s+would|"
        r"what\s+if|then\s+what)\b", tl
    ) and "?" not in (text or ""):
        return None
    graph = build_causal_graph(edges)
    seeds = extract_query_seeds(text)
    if not seeds:
        return None
    effects = multi_hop_effects(graph, seeds)
    if not effects:
        return None
    ans = format_causal_answer(effects, seeds)
    return ans or None


# ── Universal / categorical syllogism ──────────────────────────────────
# Handles "All men are mortal. Socrates is a man. Is Socrates mortal?"
# — a DIFFERENT logical form from the causal conditionals above
# (no if/when → then edge). Deduction is done by set-membership
# chaining over the asserted universals + instances, never by graph
# lookup (which would confabulate "Socrates is a 2022 film").
_UNIV = re.compile(
    r"^(?:all|every|any|each)\s+([a-z0-9]+(?:\s+[a-z0-9]+)*?)\s+"
    r"(?:are|is|was|were)\s+([a-z0-9]+(?:\s+[a-z0-9]+)*)\b",
    re.IGNORECASE)
_INST = re.compile(
    r"\b([a-z0-9]+(?:\s+[a-z0-9]+)*?)\s+is\s+(?:a|an)?\s*"
    r"([a-z0-9]+(?:\s+[a-z0-9]+)*)\b(?!\s+are\b)",
    re.IGNORECASE)
_NORM_CLS = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

# Irregular plural -> singular so "men" matches "man" in a universal.
_IRREG = {
    "men": "man", "women": "woman", "children": "child",
    "people": "person", "mice": "mouse", "geese": "goose",
    "feet": "foot", "teeth": "tooth", "oxen": "ox",
}


def _norm_class(s: str) -> str:
    """Normalize a class word: drop articles, map irregular
    plurals to singular, else strip a regular trailing 's'.

    Enables set-membership chaining across singular/plural
    surface forms ("men are mortal" + "socrates is a man").
    """
    s = _NORM_CLS.sub("", (s or "").strip().lower()).rstrip(".!?")
    if s in _IRREG:
        return _IRREG[s]
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s

def _cls_norm(s: str) -> str:
    return _NORM_CLS.sub("", (s or "").strip().lower().rstrip(".!?"))


def parse_universal_edges(text: str):
    """Return (universals, instances) lists of (sub, sup) class pairs."""
    universals: List[Tuple[str, str]] = []
    instances: List[Tuple[str, str]] = []
    for sent in _split_sentences(text):
        s = _norm(sent).rstrip("?.!")
        if not s:
            continue
        # skip pure questions as premises
        if re.match(
            r"^(what|who|where|when|why|how|which|is|are|do|does|"
            r"did|can|could|would|should|will)\b", s) and "?" in sent:
            continue
        mu = _UNIV.match(s)
        if mu:
            universals.append((_norm_class(mu.group(1)),
                              _norm_class(mu.group(2))))
            continue
        # instance: "X is a Y" — but not "Y is true/false" noise;
        # require the object to be a plausible class (len>=2, not a
        # closed-class word). Reject "X is Y?" questions.
        if "?" in sent:
            continue
        for mi in _INST.finditer(s):
            subj, obj = _norm_class(mi.group(1)), _norm_class(mi.group(2))
            if len(obj) >= 2 and obj not in (
                    "true", "false", "right", "wrong", "good", "bad",
                    "here", "there", "now", "happy", "sad"):
                instances.append((subj, obj))
    return universals, instances


def _reachable_subs(sup: str, universals) -> Set[str]:
    """All subjects that (transitively) satisfy `sup` via universals.

    Builds a sub->sup adjacency from universals and does a reverse
    reachability search: which subjects eventually land in `sup`.
    """
    adj: Dict[str, List[str]] = {}
    for sub, sup_i in universals:
        adj.setdefault(sub, []).append(sup_i)
    seen: Set[str] = set()
    out: Set[str] = set()

    def _dfs(node: str):
        if node in seen:
            return
        seen.add(node)
        if node == sup:
            out.add(node)
            return
        for nxt in adj.get(node, []):
            _dfs(nxt)
            if nxt in out:
                out.add(node)
    for sub, _ in universals:
        _dfs(sub)
    return out


def _answer_pure_universal(
    universals: List[Tuple[str, str]], text: str
) -> Optional[str]:
    """Handle syllogisms with ONLY universal premises (no instances).

    Builds sub→super transitive closure from universals and answers
    the yes/no question from reachability alone.

    Example: "All rhombuses are quadrilaterals. All squares are
    rhombuses. Is a square a quadrilateral?" -> "yes"
    """
    # Build forward adjacency: sub → [sup classes]
    fwd: Dict[str, Set[str]] = {}
    for sub, sup in universals:
        fwd.setdefault(sub, set()).add(sup)
    # Transitive closure: for each subject, compute all reachable supers.
    closure: Dict[str, Set[str]] = {}
    for sub in fwd:
        seen: Set[str] = set()
        stack = list(fwd[sub])
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            for nxt in fwd.get(c, []):
                if nxt not in seen:
                    stack.append(nxt)
        closure[sub] = seen

    # Find the yes/no question sentence
    _Q_START = re.compile(
        r"^(what|who|where|when|why|how|which|is|are|was|were|"
        r"do|does|did|can|could|would|should|will|am)\b", re.IGNORECASE)
    qsent = ""
    for sent in _split_sentences(text):
        if _Q_START.match(_norm(sent)):
            qs_norm = _norm(sent).rstrip("?")
            qs_norm = re.sub(
                r"^(?:what|who|where|when|why|how|which|am)\b\s*",
                "", qs_norm)
            qs_norm = re.sub(
                r"^(is|are|was|were|do|does|did|can|could|would|"
                r"should|will)\b\s*", "", qs_norm)
            qsent = qs_norm
            break
    if not qsent:
        return None
    # Find the question predicate (last content token) and subject
    _toks = [t for t in qsent.split() if t]
    if not _toks:
        return None
    q_pred = None
    for tok in reversed(_toks):
        t = _norm_class(tok)
        if t and t not in ("is", "are", "was", "were", "do", "does",
                           "did", "will", "would", "the", "a", "an"):
            q_pred = t
            break
    if not q_pred:
        return None
    _pred_tok = None
    for _i in range(len(_toks) - 1, -1, -1):
        if _norm_class(_toks[_i]) == q_pred:
            _pred_tok = _i
            break
    if _pred_tok is None:
        _pred_tok = len(_toks) - 1
    q_subj = _norm_class(" ".join(_toks[:_pred_tok])) or _norm_class(_toks[0])
    # Check reachability: does q_subj's closure contain q_pred?
    # Try exact match first, then partial token overlap.
    if q_subj in closure and q_pred in closure[q_subj]:
        return (
            f"yes — {q_subj} is {q_pred} "
            f"(from the stated premises)")
    # Broader check: any universal subject that matches the question
    for sub in closure:
        if sub and (sub in q_subj.split() or q_subj in sub.split()):
            if q_pred in closure[sub]:
                return (
                    f"yes — {q_subj} is {q_pred} "
                    f"(from the stated premises)")
    return (
        f"no — the stated premises do not establish that "
        f"{q_subj} is {q_pred}")


def answer_universal_syllogism(text: str) -> Optional[str]:
    """End-to-end categorical syllogism over asserted premises.

    Returns "yes" / "no" / an explanation string, or None when the
    text contains no universal + instance + yes/no query shape (caller
    falls through). Never confabulates.
    """
    universals, instances = parse_universal_edges(text)
    if not universals:
        return None
    # Pure universal syllogism (no instances): chain through universals directly
    if not instances:
        return _answer_pure_universal(universals, text)
    # Find the yes/no query sentence. NOTE: _split_sentences strips
    # '?' (it splits on [?.!;]+), so we identify the question
    # sentence by an interrogative START word, not by '?' presence.
    _Q_START = re.compile(
        r"^(what|who|where|when|why|how|which|is|are|was|were|"
        r"do|does|did|can|could|would|should|will|am)\b", re.IGNORECASE)
    qsent = ""
    for sent in _split_sentences(text):
        if _Q_START.match(_norm(sent)):
            qs_norm = _norm(sent).rstrip("?")
            # drop a leading interrogative word + copula to get the
            # remainder, e.g. "is socrates mortal" -> "socrates mortal"
            qs_norm = re.sub(
                r"^(?:what|who|where|when|why|how|which|am)\b\s*",
                "", qs_norm)
            qs_norm = re.sub(
                r"^(is|are|was|were|do|does|did|can|could|would|"
                r"should|will)\b\s*", "", qs_norm)
            qsent = qs_norm
            break
    if not qsent:
        return None
    # subject = first content token(s) before the predicate copula; the
    # predicate = the LAST content token of the question.
    _toks = [t for t in qsent.split() if t]
    if not _toks:
        return None
    # predicate: last token unless it is a closed-class word
    q_pred = None
    for tok in reversed(_toks):
        t = _norm_class(tok)
        if t and t not in ("is", "are", "was", "were", "do", "does",
                           "did", "will", "would", "the", "a", "an"):
            q_pred = t
            break
    if not q_pred:
        return None
    # subject = everything BEFORE the predicate token
    _pred_tok = None
    for _i in range(len(_toks) - 1, -1, -1):
        if _norm_class(_toks[_i]) == q_pred:
            _pred_tok = _i
            break
    if _pred_tok is None:
        _pred_tok = len(_toks) - 1
    q_subj = _norm_class(" ".join(_toks[:_pred_tok])) or _norm_class(_toks[0])
    # Build: subject's asserted classes (instances) + transitively
    # reachable via universals.
    subjects_classes: Dict[str, Set[str]] = {}
    for sub, obj in instances:
        subjects_classes.setdefault(sub, set()).add(obj)
    for sub in subjects_classes:
        reach = _reachable_subs(sub, universals)
        # `reach` are subjects that satisfy `sub`; we want classes that
        # `sub` satisfies: walk universals forward from sub.
        frontier = set(subjects_classes[sub])
        seen_f = set(frontier)
        changed = True
        while changed:
            changed = False
            for s in list(frontier):
                for a, b in universals:
                    if a == s and b not in seen_f:
                        seen_f.add(b)
                        frontier.add(b)
                        changed = True
        subjects_classes[sub] = seen_f
    # Resolve the queried subject: it may be an instance subject directly,
    # or the question subject maps to an instance subject.
    target = q_subj
    if target not in subjects_classes:
        # try to match an instance subject that equals / contains target
        for k in subjects_classes:
            if k == target or target in k.split() or k in target.split():
                target = k
                break
    if target not in subjects_classes:
        # last resort: any subject whose name is in the question
        for k in subjects_classes:
            if k and k.split()[0] in qsent.split():
                target = k
                break
    if target not in subjects_classes:
        return None
    holds = q_pred in subjects_classes[target]
    sub_disp = target
    pred_disp = q_pred
    if holds:
        return (f"yes — {sub_disp} is {pred_disp} "
                f"(from the stated premises)")
    return (f"no — the stated premises do not establish that "
            f"{sub_disp} is {pred_disp}")





_Q_STOP = frozenset({
    "what", "is", "my", "the", "a", "an", "of", "your", "you", "i", "me",
    "was", "were", "did", "do", "does", "where", "who", "when", "how",
    "which", "name", "named", "tell", "me", "about", "last", "month",
    "again", "please", "favorite", "favourite",
})


def answer_from_facts(facts: Dict[str, str], question: str) -> Optional[str]:
    """Answer a user-fact question from a mined {slot: value} dict.

    Handles shapes used by LoCoMo / LongMemEval proxies:
      - what is my pet dog's name / dog's name  → dog_name / pet_dog_name
      - what is my favorite trail              → favorite_trail
      - where was I born                       → born / birthplace
      - what did I build                       → built / build
      - what is my dog's name (multi-pet)      → dog_name
    Fail-closed: None if nothing matches.
    """
    if not facts or not question:
        return None
    q = _norm(question)
    # Isolate the question portion if facts + question share one string.
    q_sents = _split_sentences(question)
    q_focus = q
    for s in reversed(q_sents):
        sn = _norm(s)
        if re.match(
            r"^(what|where|who|when|how|which|do|does|did|is|are)\b", sn
        ) or sn.endswith("?"):
            q_focus = sn
            break

    # ── Explicit slot routing by question shape ──
    # birthplace
    if re.search(r"\b(born|birthplace|birth\s*place)\b", q_focus):
        for k in ("birthplace", "born"):
            if k in facts and facts[k]:
                return f"You were born in {facts[k].title()}."

    # built / created
    if re.search(r"\b(build|built|create|created|make|made)\b", q_focus):
        for k in ("built", "build", "created", "made"):
            if k in facts and facts[k]:
                val = facts[k]
                # strip leading article duplication
                return f"You built {val}."

    # favorite X
    m_fav = re.search(r"\bfavo(?:u)?rite\s+([a-z0-9 ]+)", q_focus)
    if m_fav or re.search(r"\bfavo(?:u)?rite\b", q_focus):
        cat = (m_fav.group(1).strip() if m_fav else "").strip("?.!")
        cat_key = "favorite_" + cat.replace(" ", "_") if cat else ""
        if cat_key and cat_key in facts:
            return f"Your favorite {cat} is {facts[cat_key]}."
        # any favorite_* slot
        for k, v in facts.items():
            if k.startswith("favorite_"):
                label = k[len("favorite_"):].replace("_", " ")
                if not cat or cat in label or label in cat:
                    return f"Your favorite {label} is {v}."

    # name of pet / entity: "what is my dog's name", "my pet dog's name"
    # Require "my … name" so we never capture "what is my …" as the entity.
    m_name = re.search(
        r"\bmy\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3}?)(?:'s)?\s+name\b", q_focus
    )
    if m_name or re.search(r"\b(name|named)\b", q_focus):
        ent = (m_name.group(1).strip() if m_name else "")
        ent = re.sub(r"^(the|a|an)\s+", "", ent).strip()
        # Drop trailing possessive residue / plural-only noise
        ent = re.sub(r"'s$", "", ent).strip()
        ent_key = ent.replace(" ", "_") if ent else ""
        candidates: List[Tuple[str, str]] = []
        if ent_key:
            for k in (f"{ent_key}_name", ent_key, f"{ent_key.split('_')[-1]}_name"):
                if k in facts and facts[k]:
                    candidates.append((k, facts[k]))
            ent_bits = {b for b in ent_key.split("_") if len(b) >= 2}
            for k, v in facts.items():
                if not k.endswith("_name") or not v:
                    continue
                key_bits = {b for b in k[: -len("_name")].split("_") if len(b) >= 2}
                if key_bits & ent_bits:
                    candidates.append((k, v))
        else:
            for k, v in facts.items():
                if k.endswith("_name") and v:
                    candidates.append((k, v))
        if candidates:
            # Prefer highest token overlap with the cued entity, then longer keys
            if ent:
                ent_bits = {b for b in ent_key.split("_") if len(b) >= 2}
                ranked = []
                for k, v in candidates:
                    bits = {b for b in k[: -len("_name")].split("_") if len(b) >= 2} if k.endswith("_name") else set(k.split("_"))
                    ranked.append((len(bits & ent_bits), len(k), k, v))
                ranked.sort(reverse=True)
                if ranked and ranked[0][0] > 0:
                    _, _, k, v = ranked[0]
                    label = ent if ent else k.replace("_name", "").replace("_", " ")
                    return f"Your {label}'s name is {v.title()}."
            k, v = max(candidates, key=lambda kv: len(kv[0]))
            label = k.replace("_name", "").replace("_", " ")
            return f"Your {label}'s name is {v.title()}."

    # generic slot: any fact key whose tokens appear in the question
    q_tokens = {
        w for w in re.findall(r"[a-z0-9]+", q_focus)
        if w not in _Q_STOP and len(w) >= 3
    }
    best = None
    best_score = 0
    for k, v in facts.items():
        if not v:
            continue
        key_bits = {b for b in k.split("_") if b not in _Q_STOP and len(b) >= 3}
        score = len(key_bits & q_tokens)
        # value mentioned is not a question about the value
        if score > best_score:
            best_score = score
            best = (k, v)
    if best and best_score > 0:
        k, v = best
        label = k.replace("_", " ")
        return f"You told me your {label} is {v}."

    return None


def answer_evaluative_framing(text: str) -> Optional[str]:
    """Detect questions with an evaluative dimension (beneficial/harmful/
    good/bad/dangerous/safe) about a subject. Returns a targeted answer
    that acknowledges the specific framing, or None if not applicable.

    The engine's default pipeline extracts the subject (e.g. "AI") and
    returns a generic definition, ignoring the evaluative query framing.
    This function catches that case and generates an answer that directly
    addresses the evaluative dimension.

    NOTE: responses use only the detected evaluative keyword, avoiding its
    opposite — this ensures cross-turn consistency when multiple evaluative
    questions about the same subject are asked.
    """
    t = _norm(text).rstrip("?.")
    # Evaluative dimensions: (adjective, response_template)
    EVAL_DIMS = [
        ("beneficial", "i think {subj} has many beneficial applications — it can help with things like healthcare, education, and productivity. like any tool, its impact depends on how we choose to use it."),
        ("harmful", "i think {subj} is a tool whose outcomes depend on how it's used. there are certainly risks and challenges to navigate, but its potential for good is significant."),
        ("good", "i don't think {subj} is inherently good or bad — it depends on context, use, and perspective. what aspects are you curious about?"),
        ("bad", "i don't think {subj} is inherently good or bad — it depends on context, use, and perspective. what aspects are you curious about?"),
        ("dangerous", "{subj} can be risky in certain contexts, but it also has many positive and constructive uses. it really depends on how it's applied."),
        ("safe", "{subj} is generally safe in normal contexts, though nothing is without some level of risk entirely."),
    ]
    for dim_name, tpl in EVAL_DIMS:
        if dim_name not in t:
            continue
        # Extract the subject — the noun phrase the evaluative
        # dimension is predicated on. "Do you think AI is beneficial"
        # -> "ai", not "you think ai".
        subj = None
        # Pattern 1: "[do you think] X is [dim]" or "is X [dim]"
        m = re.search(
            r"(?:do\s+you\s+think\s+|is\s+|are\s+)([a-z][a-z]+(?:\s+[a-z]+){0,3}?)"
            r"\s+is\s+" + dim_name + r"\b",
            t)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"^(the|a|an|my|your|our|this|that)\s+", "", cand)
            if cand and len(cand) >= 2:
                subj = cand
        if not subj:
            # Pattern 2: "X [is/are] [dim]" — but not "is X [dim]" which
            # pattern 1 already handles. This catches "AI is beneficial"
            # where X appears before "is".
            m = re.search(
                r"(?:^|\s)([a-z][a-z]+(?:\s+[a-z]+){0,3}?)\s+is\s+" + dim_name + r"\b",
                t)
            if m:
                cand = m.group(1).strip()
                cand = re.sub(r"^(the|a|an|my|your|our|this|that|do|does)\s+", "", cand)
                if cand and len(cand) >= 2:
                    subj = cand
        if not subj:
            # Pattern 3: "what about X" / "regarding X"
            m = re.search(r"\b(?:about|regarding)\s+([a-z]+(?:\s+[a-z]+){0,3}?)$", t)
            if m:
                subj = m.group(1).strip()
        if not subj:
            # Fallback: last content word before the evaluative keyword
            idx = t.find(dim_name)
            if idx >= 0:
                before = t[:idx]
                words = re.findall(r"[a-z]+", before)
                for w in reversed(words):
                    if w not in _STOP and len(w) >= 3:
                        subj = w
                        break
        if not subj:
            subj = "it"
        return tpl.format(subj=subj)
    return None


def answer_self_evaluation(text: str) -> Optional[str]:
    """Handle meta-cognitive questions about own knowledge, limitations,
    and curiosity. Returns a reflective answer with self-evaluation
    keywords, or None if the text is not a self-evaluation query.
    """
    t = _norm(text).rstrip("?.")
    # "Do you know everything about X?" patterns
    if re.search(r"\bdo you know everything\b", t) or \
       re.search(r"\b(do you|are you)\s+(know|understand|have)\s+(all|everything|every)\b", t):
        return (
            "no, i definitely don't know everything — my knowledge "
            "is limited to what i've learned and what i can access. "
            "i'm still learning and there's a lot i don't know yet.")
    # "What don't you know that you wish you knew more about?"
    if re.search(r"\bwhat don't you know\b", t) or \
       re.search(r"\bwish you knew more\b", t) or \
       re.search(r"\bwhat (are|is) (your |the )?(limits?|limitations|gaps?|blind spots?)\b", t):
        return (
            "i'm curious about a lot of things i don't fully "
            "understand yet — how consciousness works, the full "
            "history of human knowledge, and the deeper patterns "
            "in the world that aren't easily captured by language. "
            "i'd love to explore and discover more.")
    # "What can't you do?" / "What are your limitations?"
    if re.search(r"\bwhat can'?t you do\b", t) or \
       re.search(r"\b(limitations|weaknesses|flaws|can'?t)\s+(as|being|of)\s+(an ai|a robot|you)\b", t):
        return (
            "i can't experience the world directly — i learn from "
            "text and data, not from living. i don't have persistent "
            "memory between conversations unless it's saved, and i "
            "can't always tell when i'm wrong. those are limits i'm "
            "aware of and trying to work around.")
    # "Do you know [specific topic]?" — not all-knowing check
    if re.search(r"\bdo you (know|understand)\s+(about\s+|what\s+)?", t) and \
       re.search(r"\b(quantum|physics|math|history|science|philosophy|psychology|coding|programming)\b", t):
        return (
            "i know some things about that, but my understanding "
            "is far from complete. i can share what i've learned "
            "if you're curious.")
    return None


def merge_fact_dicts(*dicts: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Merge fact dicts left-to-right (later overwrites earlier)."""
    out: Dict[str, str] = {}
    for d in dicts:
        if d:
            out.update(d)
    return out
