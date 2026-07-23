"""
Temporal Grounding — Session-Date Anchoring & Relative-Date Resolution
======================================================================
Phase 1 of the LoCoMo / LongMemEval long-term-memory upgrade.

Neuroscience grounding
----------------------
Hippocampal *time cells* (MacDonald et al. 2011; Eichenbaum 2014) bind
*what* + *when* into a single trace at ENCODING time. The Temporal Context
Model (Howard & Kahana 2002) says items near in time share a drifting context
vector, so retrieval is cued by temporal proximity as well as content. The
computational analogue used here: at STORE time we resolve any relative time
phrase in the utterance ("yesterday", "3 years ago", "last Tuesday") against
the session's real-world date and persist the resulting ABSOLUTE date on the
fact. Relative-interval math (elapsed time between two anchored dates) is the
striatal beat-frequency clock (Meck 2003).

This module is intentionally dependency-light: it uses python-dateutil (already
a project dependency) for robust absolute-date parsing and relativedelta math,
and hand-written regex for the relative-phrase grammar. No hardcoded answers —
pure date arithmetic against a supplied session anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple
import re

try:
    from dateutil import parser as _du_parser
    from dateutil.relativedelta import relativedelta as _rd
    _HAVE_DATEUTIL = True
except Exception:  # pragma: no cover - dateutil is a declared dependency
    _HAVE_DATEUTIL = False


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# number words for "N years/months/weeks/days ago"
_NUMWORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}


def _num(tok: str) -> Optional[int]:
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    return _NUMWORDS.get(tok)


@dataclass
class GroundedDate:
    """Resolved date plus how confident/precise the resolution is."""
    date: datetime
    granularity: str  # 'day' | 'month' | 'year' | 'week'
    source_phrase: str


class DateGrounder:
    """Resolve absolute and relative date expressions against a session anchor.

    Usage:
        g = DateGrounder()
        g.parse_session_date("2:15 pm on 8 May, 2023")  -> datetime
        g.resolve_relative("3 years ago", session_date) -> datetime
        g.ground_utterance(text, session_date)          -> Optional[GroundedDate]
    """

    # ── session-date parsing ────────────────────────────────────────────────
    def parse_session_date(self, raw: str) -> Optional[datetime]:
        """Parse LoCoMo/LongMemEval session timestamps like
        '2:15 pm on 8 May, 2023' or '2023-05-08' into a datetime."""
        if not raw:
            return None
        s = str(raw).strip()
        if _HAVE_DATEUTIL:
            try:
                return _du_parser.parse(s, fuzzy=True, default=datetime(2000, 1, 1))
            except Exception:
                pass
        # Fallback: pull an explicit year and month name.
        return self._regex_absolute(s)

    def _regex_absolute(self, s: str) -> Optional[datetime]:
        sl = s.lower()
        year = None
        my = re.search(r"\b(19|20)\d{2}\b", sl)
        if my:
            year = int(my.group(0))
        month = None
        for name, num in _MONTHS.items():
            if re.search(rf"\b{name}\b", sl):
                month = num
                break
        day = None
        md = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", sl)
        if md:
            d = int(md.group(1))
            if 1 <= d <= 31:
                day = d
        if year:
            return datetime(year, month or 1, day or 1)
        return None

    # ── relative-date resolution ────────────────────────────────────────────
    def resolve_relative(self, phrase: str,
                         session_date: datetime) -> Optional[GroundedDate]:
        """Resolve a single relative phrase against session_date."""
        if session_date is None:
            return None
        p = phrase.lower().strip()

        # today / now
        if re.search(r"\b(today|now|currently|right now)\b", p):
            return GroundedDate(session_date, "day", phrase)
        # yesterday / tomorrow
        if "yesterday" in p:
            return GroundedDate(session_date - timedelta(days=1), "day", phrase)
        if "tomorrow" in p:
            return GroundedDate(session_date + timedelta(days=1), "day", phrase)

        # "N <unit> ago"  /  "N <unit> from now" / "in N <unit>"
        m = re.search(
            r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve)\s+(year|month|week|day)s?\s+(ago|earlier|"
            r"before|back)\b", p)
        if m and _HAVE_DATEUTIL:
            n = _num(m.group(1)) or 0
            unit = m.group(2)
            return GroundedDate(
                self._shift(session_date, unit, -n),
                unit if unit != "day" else "day", phrase)
        m = re.search(
            r"\b(?:in|after)\s+(\d+|a|an|one|two|three|four|five|six|seven|"
            r"eight|nine|ten|eleven|twelve)\s+(year|month|week|day)s?\b", p)
        if m and _HAVE_DATEUTIL:
            n = _num(m.group(1)) or 0
            unit = m.group(2)
            return GroundedDate(self._shift(session_date, unit, n), unit, phrase)

        # last/next <unit>  ("last month", "next week", "last year")
        m = re.search(r"\b(last|next|this|previous|coming)\s+"
                      r"(year|month|week)\b", p)
        if m and _HAVE_DATEUTIL:
            direction = -1 if m.group(1) in ("last", "previous") else (
                1 if m.group(1) in ("next", "coming") else 0)
            unit = m.group(2)
            return GroundedDate(self._shift(session_date, unit, direction),
                                unit, phrase)

        # last/next <weekday>  ("last Tuesday", "next Monday", "on Sunday")
        m = re.search(r"\b(last|next|this|on|the following|previous)?\s*"
                      r"(monday|tuesday|wednesday|thursday|friday|saturday|"
                      r"sunday)\b", p)
        if m:
            qual = (m.group(1) or "").strip()
            wd = _WEEKDAYS[m.group(2)]
            return GroundedDate(self._resolve_weekday(session_date, wd, qual),
                                "day", phrase)

        # explicit month(+day) without year → assume session year
        m = re.search(r"\b(january|february|march|april|may|june|july|august|"
                      r"september|october|november|december|jan|feb|mar|apr|"
                      r"jun|jul|aug|sep|sept|oct|nov|dec)\s+"
                      r"(\d{1,2})(?:st|nd|rd|th)?\b", p)
        if m:
            month = _MONTHS[m.group(1)]
            day = int(m.group(2))
            try:
                return GroundedDate(
                    datetime(session_date.year, month, min(day, 28)
                             if month == 2 else day), "day", phrase)
            except ValueError:
                pass

        return None

    def _shift(self, base: datetime, unit: str, n: int) -> datetime:
        if unit == "week":
            return base + timedelta(weeks=n)
        if unit == "day":
            return base + timedelta(days=n)
        if unit == "month":
            return base + _rd(months=n)
        if unit == "year":
            return base + _rd(years=n)
        return base

    def _resolve_weekday(self, base: datetime, target_wd: int,
                         qual: str) -> datetime:
        base_wd = base.weekday()
        if qual in ("next", "the following", "coming"):
            delta = (target_wd - base_wd) % 7
            delta = delta or 7
            return base + timedelta(days=delta)
        if qual in ("last", "previous"):
            delta = (base_wd - target_wd) % 7
            delta = delta or 7
            return base - timedelta(days=delta)
        # bare / "this" / "on": most recent occurrence (incl. today) going back
        delta = (base_wd - target_wd) % 7
        return base - timedelta(days=delta)

    # ── utterance-level convenience ─────────────────────────────────────────
    def ground_utterance(self, text: str,
                         session_date: Optional[datetime]) -> Optional[GroundedDate]:
        """Find the first resolvable date reference in an utterance and anchor
        it. Tries an explicit absolute date first, then relative phrases."""
        if not text:
            return None
        # explicit absolute date embedded in the utterance
        abs_dt = self._regex_absolute(text)
        if abs_dt is not None and re.search(r"\b(19|20)\d{2}\b", text):
            return GroundedDate(abs_dt, "day", text)
        if session_date is None:
            return None
        return self.resolve_relative(text, session_date)

    # ── interval math ───────────────────────────────────────────────────────
    def interval_days(self, a: datetime, b: datetime) -> int:
        """Absolute number of days between two dates."""
        return abs((a - b).days)

    def describe_interval(self, a: datetime, b: datetime) -> str:
        """Human-readable interval, largest sensible unit."""
        days = self.interval_days(a, b)
        if days >= 365:
            years = days // 365
            return f"{years} year{'s' if years != 1 else ''}"
        if days >= 30:
            months = days // 30
            return f"{months} month{'s' if months != 1 else ''}"
        if days >= 7:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''}"
        return f"{days} day{'s' if days != 1 else ''}"
