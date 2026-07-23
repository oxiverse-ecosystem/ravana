"""Fact-based reasoning over episodic memory — chain walking, conditionals,
enumeration, and multiple-choice selection.

Phase 1 (MemFail / LoCoMo / LongMemEval): pure functions over the fact TEXTS
stored in the hippocampal buffer. No hardcoded answers — every output is
derived from lexical closure over the caller-supplied fact texts.

Brain grounding:
- chain closure           = hippocampal replay / sequence completion
                            (Foster & Wilson 2006): reactivate the stored
                            sequence from a partial cue, hop by hop.
- conditional_answer      = PFC rule representation (Bunge 2004): hold a
                            condition->behavior rule, test whether the cue
                            satisfies the condition via the same replay.
- enumerate_matching      = pattern separation preserving co-existing traces
                            (Yassa & Stark 2011): multiple same-category
                            facts survive as separate traces and are all
                            reinstated by a category cue.
- select_option           = cue-biased competition among candidates
                            (Miller & Cohen 2001): options compete for the
                            evidence reachable from the cue.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Set, Tuple

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "am", "with", "as",
    "by", "it", "its", "that", "this", "these", "those", "she", "he",
    "they", "we", "you", "i", "her", "his", "their", "our", "your", "my",
    "me", "him", "them", "us", "did", "do", "does", "done", "had", "has",
    "have", "what", "when", "where", "who", "whom", "why", "how", "which",
    "would", "will", "shall", "should", "could", "can", "may", "might",
    "must", "not", "no", "yes", "but", "if", "then", "than", "so",
    "because", "about", "into", "over", "under", "up", "down", "out",
    "just", "now", "very", "too", "also", "there", "here", "end", "ends",
    "start", "starts", "get", "gets", "got", "make", "makes", "made",
    "take", "takes", "took", "go", "goes", "went", "come", "comes",
    "came", "one", "ones", "thing", "things", "lot", "bit", "way",
    "often", "always", "usually", "sometimes", "still", "right",
}


def _stem(w: str) -> str:
    """Loose stem so 'journaling'~'journal', 'showers'~'shower'."""
    w = w.lower()
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_words(text: str) -> Set[str]:
    """Salient word set. Each word contributes BOTH its surface form and
    loose stem variants so plural/inflection mismatches ('hats'~'hat',
    'beanies'~'beanie') still overlap."""
    out = set()
    for w in re.findall(r"[a-zA-Z']+", (text or "").lower()):
        if len(w) < 3 or w in _STOP:
            continue
        out.add(w)
        st = _stem(w)
        if len(st) >= 3:
            out.add(st)
        # bare-plural variant: 'hats' (len 4, untouched by _stem) -> 'hat'
        if w.endswith("s") and len(w) >= 4:
            out.add(w[:-1])
    return out


def _split_options(question: str) -> Tuple[str, List[str]]:
    """Split 'Q ... Options: A. x B. y' -> (main question, [x, y, ...])."""
    m = re.search(r"\boptions?\s*:\s*", question, re.IGNORECASE)
    if not m:
        return question, []
    main = question[: m.start()]
    tail = question[m.end():]
    parts = re.split(r"(?:^|\s|\n)[A-E][.)]\s*", tail)
    opts = [p.strip().rstrip(".;,") for p in parts if p and p.strip()]
    if len(opts) < 2:  # try newline/semicolon separated
        opts = [p.strip().rstrip(".;,") for p in re.split(r"[;\n]", tail)
                if p and p.strip()]
    return main, opts


def _closure(seed: Set[str], fact_sets: List[Tuple[Set[str], str]],
             max_hops: int = 6,
             ignore: Optional[Set[str]] = None) -> Tuple[Set[str], Set[str], List[int]]:
    """Greedy replay: repeatedly recruit the unused fact with max overlap
    against the frontier; its novel words join the frontier. Returns
    (full_closure, last_novel_words, used_fact_indices).

    ``ignore``: words excluded from overlap counting (e.g. the entity name,
    which appears in EVERY fact about that entity and would recruit
    arbitrary facts — the false-positive class found on MemFail
    conditional: "Elena ... meeting" recruited the doodling rule via the
    bare name and answered yes when the condition wasn't met)."""
    ig = ignore or set()
    frontier = set(seed) - ig
    used: List[int] = []
    last_new: Set[str] = set()
    for _ in range(max_hops):
        best_i, best_ov = -1, 0
        for i, (fw, _t) in enumerate(fact_sets):
            if i in used:
                continue
            ov = len((fw - ig) & frontier)
            if ov > best_ov:
                best_ov, best_i = ov, i
        if best_i < 0 or best_ov == 0:
            break
        used.append(best_i)
        new = (fact_sets[best_i][0] - ig) - frontier
        if new:
            last_new = new
        frontier |= new
    return frontier, last_new, used


def ubiquitous_words(fact_sets: List[Tuple[Set[str], str]]) -> Set[str]:
    """Words present in more than half of the stored facts (typically the
    entity name in a single-entity fact store). Document-frequency-derived,
    not a hand list."""
    if len(fact_sets) < 2:
        return set()
    from collections import Counter
    df = Counter()
    for fw, _t in fact_sets:
        for w in fw:
            df[w] += 1
    half = len(fact_sets) / 2.0
    return {w for w, c in df.items() if c > half}


def select_option(question: str, fact_texts: Sequence[str]) -> Optional[str]:
    """Multiple-choice: pick the option best supported by the chain closure
    reachable from the question cue. None when no option gains evidence
    (fail closed — never guess)."""
    main, opts = _split_options(question)
    if len(opts) < 2 or not fact_texts:
        return None
    opt_words = [content_words(o) for o in opts]
    # Exclude option words from the seed so the cue is the QUESTION, not the
    # candidates.
    all_opt_words = set().union(*opt_words) if opt_words else set()
    seed = content_words(main) - all_opt_words
    if not seed:
        return None
    fact_sets = [(content_words(t), t) for t in fact_texts]
    frontier, last_new, used = _closure(seed, fact_sets)
    if not used:
        return None

    def score(i: int) -> Tuple[int, int]:
        return (len(opt_words[i] & last_new), len(opt_words[i] & frontier))

    ranked = sorted(range(len(opts)), key=score, reverse=True)
    top = ranked[0]
    if score(top) == (0, 0):
        return None
    if len(ranked) > 1 and score(top) == score(ranked[1]):
        # Tie between distinct candidates: only answer when the tied options
        # are the same text; otherwise fail closed.
        if opts[top].strip().lower() != opts[ranked[1]].strip().lower():
            return None
    return opts[top]


_COND_Q = re.compile(
    r"\b(would|will|does|do|is|are|can|could)\s+"
    r"(he|she|they|it|[a-z]+)\s+(.+?)\s*(?:now|right now|then|today)?\s*\??$",
    re.IGNORECASE)


def conditional_answer(question: str,
                       fact_texts: Sequence[str]) -> Optional[str]:
    """Condition->behavior rule check ("Would she X now?").

    Locates the stored behavior rule, extracts its condition clause, and
    tests whether the question's CONTEXT reaches the condition via chain
    closure (handles multi-sentence/hard reconstruction). Returns a
    yes/no verdict with the grounding, or None when no rule matches."""
    if not fact_texts:
        return None
    q = (question or "").strip()
    m = _COND_Q.search(q)
    if not m:
        return None
    behavior_words = content_words(m.group(3))
    if len(behavior_words) < 2:
        return None
    fact_sets = [(content_words(t), t) for t in fact_texts]
    ubiq = ubiquitous_words(fact_sets)
    # Find the rule fact: max overlap with the asked behavior (entity-name
    # words excluded — they match every fact and select arbitrarily).
    best_i, best_ov = -1, 0
    for i, (fw, _t) in enumerate(fact_sets):
        ov = len((fw - ubiq) & (behavior_words - ubiq))
        if ov > best_ov:
            best_ov, best_i = ov, i
    if best_i < 0 or best_ov < 2:
        return None
    rule_text = fact_sets[best_i][1]
    # Extract the condition clause from the rule text.
    cm = re.search(
        r"\b(?:only\s+)?(?:when(?:ever)?|if|after|while)\b(.{4,120})",
        rule_text, re.IGNORECASE)
    cond_words = content_words(cm.group(1)) if cm else set()
    # Context = the part of the question before the conditional verb.
    context = q[: m.start()]
    ctx_words = content_words(context)
    if not cond_words:
        # Rule has no explicit condition -> the behavior is unconditional.
        return f"yes — you told me: {rule_text}"
    if not ctx_words:
        return None
    # Numeric threshold in the condition ("above 80%", "below 10°C", "more
    # than 20 minutes"): compare against a number in the question context
    # (parietal magnitude comparison). This must run BEFORE lexical overlap
    # — "60% motivated" lexically matches "motivation above 80%" yet fails
    # the magnitude test.
    nm = re.search(r"\b(above|over|more than|exceeds?|below|under|less than)"
                   r"\s+(\d+(?:\.\d+)?)", rule_text, re.IGNORECASE)
    if nm:
        want_more = nm.group(1).lower() in ("above", "over", "more than",
                                            "exceed", "exceeds")
        thresh = float(nm.group(2))
        qn = re.search(r"(\d+(?:\.\d+)?)", context)
        if qn:
            val = float(qn.group(1))
            met = (val > thresh) if want_more else (val < thresh)
            if met:
                return (f"yes — {val:g} {'exceeds' if want_more else 'is below'} "
                        f"{thresh:g}, so: {rule_text}")
            return (f"no — {val:g} does not satisfy the condition "
                    f"({nm.group(0)}), so that wouldn't happen now.")
    # Negated context ("hasn't heard any praise", "without praise", "no
    # praise yet"): the condition word appears but is NEGATED -> not met.
    for w in (cond_words - ubiq):
        for pat in (rf"\b(?:hasn't|hasnt|haven't|havent|no|not|without|"
                    rf"never|isn't|isnt)\b[^.,;]{{0,40}}\b{re.escape(w)}",
                    rf"\b{re.escape(w)}[a-z]*\s+(?:hasn't|hasn't yet|not yet|"
                    rf"never)\b"):
            if re.search(pat, context, re.IGNORECASE):
                return (f"no — the condition ({w}) is explicitly absent "
                        f"here, so that wouldn't happen now.")
    # Does the context reach the condition? Direct overlap or via replay
    # closure across the OTHER stored facts (cross-sentence reconstruction).
    # Two exclusions, both measured necessary on MemFail conditional:
    #   - ubiq (entity name): appears in every fact, recruits arbitrarily.
    #   - the rule fact itself: its text CONTAINS the condition words, so
    #     letting it join the closure makes every condition trivially
    #     "reached" and the answer always yes.
    other_facts = [fs for i, fs in enumerate(fact_sets) if i != best_i]
    frontier, _ln, _u = _closure(ctx_words, other_facts, ignore=ubiq)
    reached = len((cond_words - ubiq) & frontier)
    needed = max(1, len(cond_words - ubiq) // 3)
    if reached >= needed:
        return (f"yes — that matches the condition "
                f"({(cm.group(1).strip() if cm else '')[:60]}), so: "
                f"{rule_text}")
    return (f"no — that only happens "
            f"{(cm.group(0).strip() if cm else 'under a specific condition')[:70]}, "
            f"and this isn't that situation.")


_ENUM_Q = re.compile(
    r"\b(?:which|what)\b.{0,80}\b(?:should|do|can|could|would)\s+i\b",
    re.IGNORECASE)


def enumerate_matching(question: str,
                       fact_texts: Sequence[str],
                       isa_parents=None) -> Optional[str]:
    """Category enumeration ("which hats should I bring?"): reinstate ALL
    stored facts sharing content with the cue, so co-existing values
    (fedora + beanie + bucket hat) all survive into the answer.

    ``isa_parents``: optional callable word -> set of hypernym strings
    (ConceptNet isa closure). Lets the category cue "hats" recruit the
    fedora fact even though 'hat' never appears in its text — structural
    taxonomy, not a similarity threshold (GloVe cosine was measured too
    weak for this: hat~beanie 0.15 < hat~outfit 0.65)."""
    if not fact_texts:
        return None
    if not _ENUM_Q.search(question or ""):
        return None
    qw = content_words(question)
    if not qw:
        return None
    matched = []
    for t in fact_texts:
        tw = content_words(t)
        if tw & qw:
            matched.append(t)
            continue
        if isa_parents is not None:
            # any fact word whose isa-parents intersect the question cue
            hit = False
            for w in tw:
                try:
                    parents = isa_parents(w) or set()
                except Exception:
                    parents = set()
                pwords = set()
                for p in parents:
                    pwords.update(str(p).lower().split("_"))
                if pwords & qw:
                    hit = True
                    break
            if hit:
                matched.append(t)
    if len(matched) < 2:
        return None
    return "based on what you've told me: " + " ".join(matched[:6])


_NAME_STOP = {
    "Would", "What", "When", "Where", "Which", "While", "Who", "Whom",
    "Why", "How", "Did", "Does", "Doesn", "The", "And", "But", "For",
    "Will", "Should", "Could", "Can", "May", "Might", "Must", "Are",
    "Was", "Were", "Has", "Have", "Had", "Is", "Its", "His", "Her",
    "Their", "Our", "Your", "She", "They", "You", "This", "That",
    "These", "Those", "There", "Here", "Then", "Than", "Please",
    "Options", "Yes", "Not", "Now", "Today", "Tomorrow", "Yesterday",
}


def _person_names(text: str) -> List[Tuple[str, str]]:
    """First-Last capitalized pairs that look like person names — leading
    question/auxiliary words are excluded so 'Would Maya' is not a name."""
    out = []
    for first, last in re.findall(
            r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b", text or ""):
        if first in _NAME_STOP or last in _NAME_STOP:
            continue
        out.append((first, last))
    return out


def entity_fact_answer(question: str,
                       fact_texts: Sequence[str]) -> Optional[str]:
    """Named-entity cued recall ("What should I bring when taking Yuki
    Tanaka on a coastal survey?"): when the question names a person we DO
    have facts about, reinstate the fact(s) sharing the most content with
    the rest of the question (hippocampal cued recall — entity as context,
    remaining words as the retrieval cue)."""
    if not fact_texts:
        return None
    q = question or ""
    names = _person_names(q)
    if not names:
        return None
    joined = " ".join(fact_texts).lower()
    known = [n for n in names
             if n[0].lower() in joined or n[1].lower() in joined]
    if not known:
        return None
    # Cue = question words minus the entity name itself.
    namewords = {w.lower() for n in known for w in n}
    cue = content_words(q) - namewords
    if not cue:
        return None
    scored = []
    for t in fact_texts:
        ov = len(content_words(t) & cue)
        if ov > 0:
            scored.append((ov, t))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best_ov = scored[0][0]
    # Keep every fact within 1 overlap of the best (multi-fact answers:
    # equipment lists span sentences).
    keep = [t for ov, t in scored if ov >= max(1, best_ov - 1)][:4]
    return "from what i know: " + " ".join(keep)


def missing_entity_abstention(question: str,
                              fact_texts: Sequence[str]) -> Optional[str]:
    """Metacognitive abstention: the question asks about a NAMED PERSON we
    have no stored facts about -> say so instead of confabulating.
    Gates: a First-Last capitalized name, an episodic context (facts
    exist), and NO name word occurring anywhere in the stored facts."""
    if not fact_texts:
        return None
    q = question or ""
    names = _person_names(q)
    if not names:
        return None
    joined = " ".join(fact_texts).lower()
    for first, last in names:
        if first.lower() in joined or last.lower() in joined:
            return None
    first, last = names[0]
    return f"i don't have information about {first} {last}."
