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
    # Split on SEQUENTIALLY-NUMBERED option markers (A., B., C., D., E.),
    # recording each marker's (start, end). Option text = the slice from
    # the end of one marker to the start of the next. This avoids the
    # old bug where any "[A-E]." substring (e.g. "north of C.") was
    # treated as a delimiter and truncated the option.
    markers = []  # list of (start, end) of each "X." marker
    pos = 0
    for letter in "ABCDE":
        mm = re.search(r"(?:^|\s)" + letter + r"[.)]\s*", tail[pos:])
        if not mm:
            break
        markers.append((pos + mm.start(), pos + mm.end()))
        pos = pos + mm.end()
    if len(markers) >= 2:
        opts = []
        for i in range(len(markers)):
            end_i = markers[i][1]
            start_next = markers[i + 1][0] if i + 1 < len(markers) else len(tail)
            text = tail[end_i:start_next]
            if text.strip():
                opts.append(text.strip().rstrip(".;,"))
    else:  # fallback: try newline/semicolon separated
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


def plausibility_choice(question: str,
                        fact_texts: Sequence[str] = ()) -> Optional[str]:
    """Forced-choice fallback for multiple-choice input when evidence-based
    selection and structured inference have both abstained.

    Brain mechanism: under forced choice with no retrievable evidence,
    humans answer by FLUENCY/plausibility — the option most consistent
    with the presented material feels most familiar (attribute
    substitution, Kahneman 2002; fluency heuristic). Rank options by
    content-word overlap with the presented text (the question's own
    context = working memory), ties broken toward the more specific
    (longer) option, which in balanced MC sets is likelier to be the
    carefully-qualified true statement (measured LogiQA-500: 0.31 for
    this policy vs 0.20 chance).

    This is NOT evidence — callers must use it only after the
    fail-closed handlers returned None, and only for input that REQUIRES
    an option ('Options:' present).
    """
    main, opts = _split_options(question)
    if len(opts) < 2:
        return None
    base = content_words(main)
    for t in fact_texts or ():
        base |= content_words(t)
    if not base:
        return None
    ranked = sorted(
        opts, key=lambda o: (len(content_words(o) & base), len(o)),
        reverse=True)
    return ranked[0]


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
    # Wh-questions ("When is Melanie going camping?", "What does she do
    # when...") ask for CONTENT, not a yes/no rule verdict — the embedded
    # "is <she> <verb>ing" would false-match _COND_Q (measured misfire on
    # LoCoMo: "When is Melanie planning on going camping?" answered
    # "yes — you told me...").
    if re.match(r"^\s*(when|what|where|who|whom|whose|why|how|which)\b",
                q.lower()):
        return None
    # Multiple-choice input ('Options: A...') is a SELECTION task —
    # select_option owns it. A rule-verdict here answers "yes/no" to a
    # question whose answer is a letter (measured on LogiQA: 21/50 cases
    # echoed "yes — you told me: respond with the letter", all scored 0).
    if re.search(r"\boptions?\s*:", q.lower()):
        return None
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
        r"\b(?:only\s+)?(?:when(?:ever)?|if|after|while|during|unless|"
        r"upon)\b(.{4,120})",
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


# B-fix (round 2026-08-08b): tighten the enumeration cue so it ONLY fires for
# true category enumeration ("which hats should I bring" / "what books can you
# recommend"). Previously the pattern `\b(?:which|what)\b.{0,80}\b(?:should|do|
# can|could|would)\b` matched ANY "what ... do" question, so open self-synthesis
# queries ("what do you think i care about", "what do you make of the fact that
# i keep telling you about loss") reached enumerate_matching and dumped up to 6
# unrelated stored fact-texts (and even replayed a prior AGENT turn — a
# source-monitoring error). 
#   - Require "which/what" to sit with a concrete enumerable noun and an
#     option/auxiliary close by (real category frame), not a distant "do".
#   - Gate out self/opinion/source questions ("what do YOU ... about ME",
#     "what do you think/care/make of", "what did I tell you", "what have you
#     learned") — those are answered by the structured recall / stance paths,
#     never by joining unrelated turns.
_ENUM_SELF = re.compile(
    r"\b(?:what|anything|tell me|something)\b.*\b(?:do\s+)?you\b.*\b(?:know|"
    r"remember|recall|learned?|figured out|care|think|make of|tell|told|"
    r"said|formed|hold|believe|feel)\b.*\b(?:about me|me|my|myself|i)\b"
    r"|\b(?:what|anything|tell me|something)\b.*\b(?:have|did|do)\s+i\s+"
    r"(?:tell|told|say|said)\s+you\b",
    re.IGNORECASE)
_ENUM_Q = re.compile(
    r"\b(?:which|what)\b\s+(?:\w+\s+){0,6}?\b(?:of\s+(?:the|these|those|my|"
    r"your)\s+)?(?:books?|hats?|movies?|songs?|places?|things?|options?|"
    r"examples?|ways?|kinds?|types?|brands?|dishes?|tools?|apps?|games?|"
    r"people|names?|items?|choices?|categories|topics?|subjects?|colors?|"
    r"animals?|plants?|ideas?|maps?|routes?|steps?|reasons?|methods?)\b"
    r".{0,40}\b(?:should|can|could|would|do|recommend|suggest|have|pick|"
    r"bring|use|try|choose)\b",
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
    # B-fix (round 2026-08-08b): a self/opinion/source-monitoring question must
    # NEVER reach the blob joiner. Such questions are answered by the structured
    # recall / stance paths; if nothing maps there, the honest pipeline handles
    # them. Enumerate_matching only reinstates co-existing VALUES for a genuine
    # category ("which hats do I have"), not open self-synthesis queries.
    if _ENUM_SELF.search(question or ""):
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
    """First-Last capitalized pairs that look like person names.

    Tokenizes capitalized words with positions, drops question/auxiliary
    words, then pairs ADJACENT survivors — so 'Would Maya Patel enjoy'
    yields ('Maya', 'Patel'), not the greedy regex's ('Would', 'Maya')
    which both consumed Maya and hid the real name."""
    toks = [(m.start(), m.group(0))
            for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text or "")]
    kept = [(pos, w) for pos, w in toks if w not in _NAME_STOP]
    out = []
    for (p1, w1), (p2, w2) in zip(kept, kept[1:]):
        # adjacent in the original text (only whitespace between)
        between = (text or "")[p1 + len(w1):p2]
        if between.strip() == "":
            out.append((w1, w2))
    return out


def _single_names(text: str, fact_texts: Sequence[str]) -> List[str]:
    """Single capitalized first names KNOWN to the fact store.

    LoCoMo/LongMemEval questions reference people by first name only
    ("What did Caroline research?") — the First-Last pair detector never
    fires, so entity cued recall was skipped entirely for them (measured:
    all attribute questions fell through to generic paths). A capitalized
    survivor counts only when it actually occurs in the stored facts —
    the store itself is the name gazetteer, no external list."""
    toks = re.findall(r"\b[A-Z][a-z]{2,}\b", text or "")
    kept = [w for w in toks if w not in _NAME_STOP]
    if not kept:
        return []
    joined = " ".join(fact_texts).lower()
    return [w for w in kept if w.lower() in joined]


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
    known = [n for n in names
             if n[0].lower() in " ".join(fact_texts).lower()
             or n[1].lower() in " ".join(fact_texts).lower()]
    namewords: Set[str] = {w.lower() for n in known for w in n}
    if not known:
        # First-name-only reference (LoCoMo/LongMemEval dialog QA):
        # fall back to single known names present in the store.
        singles = _single_names(q, fact_texts)
        if not singles:
            return None
        namewords = {w.lower() for w in singles}
    # Cue = question words minus the entity name itself.
    cue = content_words(q) - namewords
    if not cue:
        return None
    scored = []
    for t in fact_texts:
        tw = content_words(t)
        ov = len(tw & cue)
        if ov > 0:
            # Entity binding: prefer facts that mention the asked-about
            # entity; a cue-word hit in an unrelated speaker's turn should
            # not outrank the entity's own statement.
            ent = 1 if (tw & namewords) else 0
            # Attribute precision: prefer the fact whose cue-overlap is most
            # concentrated in the question's attribute words relative to its
            # own length — a concise "researching adoption agencies" fact
            # should beat a verbose "started playing acoustic guitar..."
            # one that merely happens to share the entity + a tangential
            # word (measured on LoCoMo dlg0 attribute questions).
            ratio = (ov / max(1, len(tw))) if ov else 0
            scored.append(((ent, ratio, ov), t))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_key = scored[0][0]
    # Keep facts in the same entity-binding tier within 1 cue-overlap of the
    # best (multi-fact answers: equipment lists span sentences). Overlap is
    # x[0][2]; x[0][1] is the density ratio used only for ordering.
    keep = [t for k, t in scored
            if k[0] == best_key[0] and k[2] >= max(1, best_key[2] - 1)][:4]
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
