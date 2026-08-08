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
    from ravana._import_guard import report_missing
    report_missing("dateutil", "relative/ordinal date grounding (\"4 years ago\", \"last month\")", kind="required")


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
        # Numeric ISO-ish forms FIRST: YYYY/MM/DD or YYYY-MM-DD (and the
        # HH:MM tail LongMemEval/LoCoMo session markers carry). Without this,
        # the fragment-assembly below reads "2023/05/28" as year=2023,
        # month=None->1, day=first-1-2-digit-number ("05") -> 2023-01-05,
        # silently corrupting EVERY session date when python-dateutil is not
        # installed (the dateutil path is preferred but optional). Measured on
        # LongMemEval: all session dates collapsed to 2023-01-05/2023-01-03.
        _iso = re.search(
            r"\b(19|20)(\d{2})[/-](\d{1,2})[/-](\d{1,2})\b", s)
        if _iso:
            try:
                yr = int(_iso.group(1) + _iso.group(2))
                mo = int(_iso.group(3))
                dy = int(_iso.group(4))
                _hm = re.search(r"\b(\d{1,2}):(\d{2})\b", s)
                if 1 <= mo <= 12 and 1 <= dy <= 31:
                    if _hm:
                        return datetime(yr, mo, dy,
                                        int(_hm.group(1)), int(_hm.group(2)))
                    return datetime(yr, mo, dy)
            except ValueError:
                pass
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
        # weekend references: "over the weekend", "last weekend", "this past
        # weekend" → the most recent Saturday strictly before the session
        # date (the weekend the speaker just lived through).
        if re.search(r"\b(?:over|last|this past|during)\s+(?:the\s+)?weekend\b"
                     r"|\bweekend\b.{0,12}\b(?:was|got|went|did)\b", p):
            _wd = session_date.weekday()  # Mon=0..Sun=6; Sat=5
            _back = (_wd - 5) % 7
            _back = _back or 7
            return GroundedDate(session_date - timedelta(days=_back),
                                "day", phrase)

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

    # ── relative-date BEFORE/AFTER an explicit anchor ──────────────────────
    # LoCoMo / LongMemEval gold answers are phrased RELATIVELY against a stated
    # absolute date, e.g. "the Sunday before 25 May 2023", "two weekends
    # before 17 July 2023", "the week before 27 June 2023". The engine stores
    # the anchored absolute date on the fact; to answer such questions it must
    # compute the actual calendar date the phrase denotes, then compare to the
    # stored fact date. Without this, the engine falls back to the session
    # anchor and returns a wrong date (measured: every "the <weekday> before
    # <date>" LoCoMo temporal Q scored 0).
    def resolve_relative_to_anchor(self, phrase: str,
                                   anchor: Optional[datetime] = None
                                   ) -> Optional[GroundedDate]:
        """Resolve 'the <weekday/period> <before|after> <date>' and
        '<N> <unit> <before|after> <date>' against an explicit anchor date
        found in the same utterance. Returns the computed calendar date."""
        if not phrase:
            return None
        p = phrase.lower()
        # Find an explicit absolute date anywhere in the phrase to use as the
        # anchor if none supplied.
        if anchor is None:
            _abs = self._regex_absolute(p)
            if _abs is not None and re.search(r"\b(19|20)\d{2}\b", p):
                anchor = _abs
        if anchor is None:
            return None

        # "<N> <unit> before|after <date>"
        m = re.search(
            r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|"
            r"ten|eleven|twelve)\s+(week|weekend|month|day)s?\s+"
            r"(before|after)\b", p)
        if m:
            n = _num(m.group(1)) or 1
            unit = m.group(2)
            direction = -1 if m.group(3) == "before" else 1
            if unit == "weekend":
                delta = timedelta(weeks=n)
            elif unit == "week":
                delta = timedelta(weeks=n)
            elif unit == "month":
                delta = _rd(months=n) if _HAVE_DATEUTIL else timedelta(days=30 * n)
            else:
                delta = timedelta(days=n)
            return GroundedDate(anchor + direction * delta, "day", phrase)

        # "the <weekday> before|after <date>"
        m = re.search(
            r"\bthe\s+(monday|tuesday|wednesday|thursday|friday|saturday|"
            r"sunday)\s+(before|after)\b", p)
        if m:
            wd = _WEEKDAYS[m.group(1)]
            direction = -1 if m.group(2) == "before" else 1
            # Walk back/forward to the nearest such weekday relative to anchor.
            if direction < 0:
                delta = (anchor.weekday() - wd) % 7
                delta = delta or 7
                return GroundedDate(anchor - timedelta(days=delta), "day", phrase)
            else:
                delta = (wd - anchor.weekday()) % 7
                delta = delta or 7
                return GroundedDate(anchor + timedelta(days=delta), "day", phrase)

        # "the week|weekend|month before|after <date>"
        m = re.search(
            r"\bthe\s+(week|weekend|month)\s+(before|after)\b", p)
        if m:
            unit = m.group(1)
            direction = -1 if m.group(2) == "before" else 1
            if unit == "month":
                delta = _rd(months=1) if _HAVE_DATEUTIL else timedelta(days=30)
            elif unit == "weekend":
                delta = timedelta(weeks=1)
            else:
                delta = timedelta(weeks=1)
            return GroundedDate(anchor + direction * delta, "day", phrase)

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
        # explicit absolute date embedded in the utterance. Require BOTH a
        # 4-digit year AND a month name before trusting an assembled date —
        # otherwise _regex_absolute stitches a bogus date out of scattered
        # fragments (e.g. "5 tips to manage your time in 2023" -> 2023-01-05),
        # which then SHADOWS the correct session-date anchor on the fact.
        # Measured on LongMemEval: conversational turns mentioning a bare year
        # plus any small number were all mis-dated, collapsing multi-session
        # ordering/temporal signal. A bare-year utterance must fall through to
        # the session-date default below.
        _has_year = re.search(r"\b(19|20)\d{2}\b", text)
        _has_month = any(re.search(rf"\b{_n}\b", text.lower()) for _n in _MONTHS)
        if _has_year and _has_month:
            abs_dt = self._regex_absolute(text)
            if abs_dt is not None:
                return GroundedDate(abs_dt, "day", text)
        # Month + day WITHOUT a year ("on January 16th"): resolve against the
        # session's year. Previously this fell through to relative resolution,
        # where a stray weekday word ("the SUNDAY mass ... on january 16th")
        # bound the fact to the nearest weekday instead of the stated date —
        # measured on LongMemEval oracle case 6 (computed 4 days, gold 30).
        if session_date is not None:
            tl = text.lower()
            _mon = None
            for _name, _num in _MONTHS.items():
                if re.search(rf"\b{_name}\b", tl):
                    _mon = _num
                    break
            if _mon is not None:
                _md = re.search(
                    rf"\b(?:{'|'.join(_MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b"
                    rf"|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
                    rf"(?:{'|'.join(_MONTHS)})\b", tl)
                if _md:
                    _d = int(_md.group(1) or _md.group(2))
                    if 1 <= _d <= 31:
                        try:
                            return GroundedDate(
                                datetime(session_date.year, _mon, _d),
                                "day", text)
                        except ValueError:
                            pass
        if session_date is None:
            return None
        # Relative-before/after-anchor: "the Sunday before 25 May 2023" etc.
        # Compute the actual denoted date (used by LoCoMo/LongMemEval temporal
        # gold phrasing). Try this BEFORE the generic relative resolver so an
        # explicit anchor wins over a session-relative reading.
        _rel_anchor = self.resolve_relative_to_anchor(text, session_date)
        if _rel_anchor is not None:
            return _rel_anchor
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
