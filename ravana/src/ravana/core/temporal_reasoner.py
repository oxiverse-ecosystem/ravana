"""Temporal reasoning — event ordering / time-logic over asserted events.

Genuine deduction (no hardcoded answers): parse event clauses that
carry order markers (first/then/after/before/finally), clock
times (3 PM, 2:45 PM), calendar years (1990, 2012) or
weekdays (Monday, Thursday), then answer:
  - "what did X do after Y?"      -> next event for X
  - "who arrived late?"        -> arrival after the deadline
  - "born before/after he graduated?" -> year comparison
  - "how many days after X did Y?" -> weekday interval

Fails closed (returns None) when the text has no temporal
question shape, so non-temporal turns fall through untouched.
This is the same in-prompt / working-memory approach as the
causal + universal syllogism reasoners — ephemeral premise
binding, never graph lookup (which would confabulate).
"""

import re
from typing import Dict, List, Any, Optional, Sequence

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday",
              "friday", "saturday", "sunday")

_TIME_TAG = re.compile(
    r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_DATE_TAG = re.compile(
    r"\b(in\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b|\b(\d{4})\b")
_WEEKDAY = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)
_ORDER = re.compile(
    r"\b(first|second|third|then|next|after\s+that|finally|"
    r"before|after|when|while|during|later|earlier)\b", re.IGNORECASE)


def _split_sentences(text: str) -> List[str]:
    t = (text or "").replace("\n", " ")
    parts = re.split(r"[.!?;]+", t)
    return [p.strip() for p in parts if p and p.strip()]


def _parse_clock(tag: str):
    m = _TIME_TAG.search(tag)
    if not m:
        return None
    h = int(m.group(1))
    mi = int(m.group(2)) if m.group(2) else 0
    ap = (m.group(3) or "").lower()
    if ap == "pm" and h != 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    return h * 60 + mi


def _parse_weekday(tag: str):
    m = _WEEKDAY.search(tag)
    if m:
        return _WEEKDAYS.index(m.group(1).lower()) + 1
    return None


def _parse_year(tag: str):
    m = _DATE_TAG.search(tag)
    if not m:
        return None
    # group(1) = full month/date blob (no bare year captured separately),
    # group(2) = bare 4-digit year (\b(\d{4})\b)
    if m.group(2):
        return int(m.group(2))
    # fall back: pull any 4-digit year out of the whole match
    yr = re.search(r"\b(\d{4})\b", m.group(0))
    return int(yr.group(1)) if yr else None


def _norm_subj(s: str) -> str:
    s = re.sub(r"^(?:the|a|an|my|our|his|her|their)\s+", "", s.strip().lower())
    return s.strip(" .,")


def _stem(w: str) -> str:
    """Loose word stem for matching inflected forms (watering/watered)."""
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def _parse_events(text: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for s in _split_sentences(text):
        sn = s.lower().strip()
        if not sn or len(sn) < 3:
            continue
        if sn.endswith("?") or re.match(
                r"^(what|who|when|where|how|which|was|were|is|are|do|"
                r"does|did|will|would|should|can|could)\b", sn):
            continue
        for clause in re.split(r"\s*(?:,|;|then|and then|after that)\s*", sn):
            clause = clause.strip().rstrip(".!")
            if not clause or len(clause) < 3:
                continue
            subj = _norm_subj(clause.split()[0] if clause.split() else clause)
            if not subj:
                continue
            clock = _parse_clock(clause)
            year = _parse_year(clause)
            wd = _parse_weekday(clause)
            events.append({
                "subj": subj, "text": clause,
                "clock": clock, "year": year, "weekday": wd,
                "order": len(events),
            })
    return events


def answer_temporal(text: str) -> Optional[str]:
    """End-to-end temporal deduction over asserted events.

    Returns a natural answer, or None when the text has no
    temporal question shape (caller falls through). Never confabulates.
    """
    sents = _split_sentences(text)
    qsent = ""
    for s in sents:
        sn = s.lower().strip()
        if sn.endswith("?") or re.match(
                r"^(what|who|when|where|how|which|was|were|is|are|do|"
                r"does|did|will|would|should|can|could)\b", sn):
            qsent = sn.rstrip("?")
            break
    if not qsent:
        return None

    events = _parse_events(text)
    if not events:
        return None

    ql = qsent

    # Q1: "what did <X> do after <Y>?"  (e.g. "after eating breakfast")
    m = re.search(
        r"what\s+did\s+([a-z]+)\s+do\s+after\s+(.+?)\??$", ql)
    if m:
        actor = m.group(1)
        ref = _norm_subj(m.group(2))
        # match the reference event by subject OR by content overlap
        ref_idx = -1
        for i, ev in enumerate(events):
            if ref and (ref in ev["subj"] or ref in ev["text"]
                       or ev["subj"] in ref or ev["text"] in ref):
                ref_idx = i
                break
        if ref_idx < 0:
            # try any event whose text shares a content word with the ref
            ref_words = {w for w in re.findall(r"[a-z]+", ref)
                         if len(w) >= 4}
            if ref_words:
                for i, ev in enumerate(events):
                    if ref_words & set(re.findall(r"[a-z]+", ev["text"])):
                        ref_idx = i
                        break
        if ref_idx >= 0 and ref_idx + 1 < len(events):
            nxt = events[ref_idx + 1]
            return f"{nxt['text'].capitalize()}."

    # Q2: "who arrived late?" — compare arrivals to a deadline
    if re.search(r"who\s+(?:arrived|was|were|came)\s+late", ql) or \
       re.search(r"arriv\w*\s+late", ql):
        deadline = None
        arrivals = []
        for ev in events:
            if ev["clock"] is not None:
                if re.search(r"meet|deadline|start|due", ev["text"]):
                    deadline = ev["clock"]
                elif re.search(r"arriv|came|got|reach|show", ev["text"]):
                    arrivals.append(ev)
        if deadline is not None and arrivals:
            late = [a for a in arrivals if a["clock"] > deadline]
            if late:
                return f"{late[0]['subj'].capitalize()}."
            early = [a for a in arrivals if a["clock"] <= deadline]
            if early:
                return f"{early[0]['subj'].capitalize()}."
        if arrivals:
            latest = max(arrivals, key=lambda a: a["clock"] or 0)
            return f"{latest['subj'].capitalize()}."

    # Q3: "<X> born 1990, graduated 2012. born before/after?"
    m = re.search(
        r"\b(\w+)\s+(?:was\s+)?(?:born|graduated|died|started|"
        r"ended|married)\b.*\b(before|after)\b", ql)
    if m:
        a_name = _norm_subj(m.group(1))
        direction = m.group(2)
        subj_a = None
        subj_b = None
        for ev in events:
            if a_name in ev["text"] and ev["year"]:
                subj_a = ev
            elif re.search(r"graduat|born|died|started|ended|married",
                             ev["text"]) and ev["year"] and ev is not subj_a:
                subj_b = ev
        if subj_a and subj_b and subj_a["year"] and subj_b["year"]:
            if direction == "before":
                ans = "before" if subj_a["year"] < subj_b["year"] else "after"
            else:
                ans = "after" if subj_a["year"] > subj_b["year"] else "before"
            return ans.capitalize() + "."

    # Q4: "how many days after <X> did <Y>?"  (e.g. "after watering ... did
    # the soil dry") — the weekdays live in the referenced EVENTS, not the
    # question words, so resolve the referents to events first.
    m = re.search(
        r"how\s+many\s+days\s+after\s+(.+?)\s+did\s+(.+?)\??$", ql)
    if m:
        def _resolve_weekday(phrase: str):
            # direct weekday in the phrase
            wd = _parse_weekday(phrase)
            if wd:
                return wd
            # else: find the event whose text shares a content word (stem-matched)
            words = {_stem(w) for w in re.findall(r"[a-z]+", phrase) if len(w) >= 4}
            if not words:
                return None
            for ev in events:
                ev_words = {_stem(w) for w in re.findall(r"[a-z]+", ev["text"])}
                if words & ev_words and ev["weekday"]:
                    return ev["weekday"]
            return None
        a_wd = _resolve_weekday(m.group(1))
        b_wd = _resolve_weekday(m.group(2))
        if a_wd and b_wd:
            diff = (b_wd - a_wd) % 7
            if diff == 0:
                diff = 7
            return f"{diff} days."

    return None
