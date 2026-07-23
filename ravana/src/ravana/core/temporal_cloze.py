"""Temporal cloze solver — TimeDial-style blank filling over dialog.

Selects which candidate temporal expression fills a masked blank in a
dialogue, using ONLY general temporal parsing + consistency rules (no
per-item answers). Mirrors parietal magnitude comparison (durations as
quantities on a number line) plus PFC task-set recognition of the cloze
format. All decisions are derived from the dialog's own temporal
quantities and textual cues — nothing is keyed to specific benchmark
items.

Public API:
    solve_cloze(dialog_text, options) -> (best_option_index, score, why)

Design (measured on real TimeDial data before wiring — see
scripts/measure_temporal_cloze.py):
  1. Parse every temporal expression in the dialog and in each option
     into a normalized quantity: (kind, minutes) where kind in
     {duration, clock, date, age, other}.
  2. Score each option by consistency rules:
     - unit/kind agreement with the expressions near the blank
     - comparative constraints ("minimum of X", "at least X",
       "more than X" -> blank should exceed X when the blank is the
       allowance; "less than X" -> below X)
     - idiom completion ("<blank> a day, seven days a week" -> 24 hours)
     - distractor penalty: option that reuses a context NUMBER with a
       DIFFERENT unit (classic TimeDial distractor), or an absurd unit
       for the activity scale of the dialog.
  3. Return argmax; the caller formats the answer.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "half": 0.5, "quarter": 0.25, "couple": 2, "few": 3,
}

_UNIT_MIN = {
    "second": 1 / 60.0, "minute": 1.0, "hour": 60.0, "day": 1440.0,
    "week": 10080.0, "month": 43800.0, "year": 525600.0,
    "decade": 5256000.0, "century": 52560000.0,
}


def _wordnum_to_float(txt: str) -> Optional[float]:
    """'forty-eight' -> 48, 'two and a half' -> 2.5, '24' -> 24."""
    txt = txt.lower().strip()
    m = re.match(r"^\d+(?:\.\d+)?$", txt)
    if m:
        return float(txt)
    total = 0.0
    found = False
    # "two and a half"
    parts = re.split(r"[\s\-]+", txt.replace(" and a ", " "))
    for p in parts:
        p = p.strip()
        if not p or p in ("a", "an", "and", "the", "about", "around",
                          "past", "last", "next"):
            if p in ("a", "an"):
                # "a day" / "an hour" -> 1
                total += 1
                found = True
            continue
        if p in _WORD_NUM:
            total += _WORD_NUM[p]
            found = True
        elif re.match(r"^\d+(?:\.\d+)?$", p):
            total += float(p)
            found = True
    return total if found else None


class TemporalQuantity:
    __slots__ = ("kind", "minutes", "raw", "unit")

    def __init__(self, kind: str, minutes: Optional[float], raw: str,
                 unit: str = ""):
        self.kind = kind          # duration | clock | date | age | other
        self.minutes = minutes    # normalized magnitude in minutes (durations)
        self.raw = raw
        self.unit = unit

    def __repr__(self):
        return f"TQ({self.kind},{self.minutes},{self.unit!r},{self.raw!r})"


_DUR_RE = re.compile(
    r"\b((?:\d+(?:\.\d+)?|"
    + "|".join(_WORD_NUM) +
    r")(?:[\s\-](?:and[\s\-]a[\s\-]half|a[\s\-]half|half))?"
    r"(?:[\s\-](?:" + "|".join(_WORD_NUM) + r"))*)"
    r"[\s\-]*(second|minute|hour|day|week|month|year|decade|centur)"
    r"(?:s|ies|y)?\b", re.IGNORECASE)

_CLOCK_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|o'?\s?clock)\b", re.IGNORECASE)

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

_AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old)\b",
                     re.IGNORECASE)


def parse_temporal(text: str) -> List[TemporalQuantity]:
    """Extract all temporal quantities from text."""
    out: List[TemporalQuantity] = []
    t = text or ""
    for m in _AGE_RE.finditer(t):
        out.append(TemporalQuantity("age", float(m.group(1)) * 525600.0,
                                    m.group(0), "year"))
    for m in _DUR_RE.finditer(t):
        num = _wordnum_to_float(m.group(1))
        unit = m.group(2).lower()
        unit = {"centur": "century"}.get(unit, unit)
        # "two and a half hours" — the RE grabs "two"; recover the half from
        # the matched raw span.
        raw = m.group(0)
        if num is not None and re.search(r"\b(?:and\s+a\s+)?half\b", raw,
                                         re.IGNORECASE):
            if num == int(num):
                num += 0.5
        if num is None:
            continue
        # skip if this span is actually an age ("20 years old")
        end = m.end()
        if re.match(r"\s*old\b", t[end:end + 6], re.IGNORECASE):
            continue
        out.append(TemporalQuantity(
            "duration", num * _UNIT_MIN.get(unit, 1.0), raw, unit))
    for m in _CLOCK_RE.finditer(t):
        h = int(m.group(1))
        mi = int(m.group(2)) if m.group(2) else 0
        ap = (m.group(3) or "").lower()
        if h > 24 or mi > 59:
            continue
        if ap == "pm" and h != 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        out.append(TemporalQuantity("clock", h * 60.0 + mi, m.group(0),
                                    "clock"))
    for m in _YEAR_RE.finditer(t):
        out.append(TemporalQuantity("date", float(m.group(1)), m.group(0),
                                    "year_number"))
    return out


def _dominant_kind(qs: List[TemporalQuantity]) -> Optional[str]:
    if not qs:
        return None
    counts = {}
    for q in qs:
        counts[q.kind] = counts.get(q.kind, 0) + 1
    return max(counts, key=counts.get)


def _blank_context(dialog: str, blank_token: str, window: int = 160) -> str:
    i = dialog.find(blank_token)
    if i < 0:
        return dialog[:2 * window]
    return dialog[max(0, i - window): i + len(blank_token) + window]


def solve_cloze(dialog: str, options: List[str],
                blank_token: str = "________") -> Tuple[int, float, str]:
    """Pick the option that best fills the blank.

    Returns (best_index, score, why). Never raises; falls back to a
    unit-consistency vote when no rule fires. Scores are comparable only
    within one call.
    """
    ctx = _blank_context(dialog, blank_token)
    ctx_l = ctx.lower()
    dialog_l = (dialog or "").lower()
    ctx_qs = parse_temporal(ctx)
    all_qs = parse_temporal(dialog)
    opt_parsed = [parse_temporal(o) for o in options]

    scores = [0.0] * len(options)
    why = [""] * len(options)

    # Numbers appearing anywhere in the dialog text (for distractor penalty
    # and support bonus).
    ctx_numbers = set()
    for q in all_qs:
        if q.kind == "duration" and q.minutes is not None:
            ctx_numbers.add(round(q.minutes, 2))

    # Rule 0: idiom — "<blank> a day(, seven days a week)" -> 24 hours
    idiom_24 = bool(re.search(
        rf"{re.escape(blank_token)}\s*a\s*day", ctx_l)) or \
        ("seven days a week" in dialog_l and blank_token in dialog)

    # Rule 1: comparative anchors near the blank.
    #   "minimum of X" / "at least X" / "more than X"  -> blank > X
    #   "maximum of X" / "at most X" / "less than X"   -> blank < X
    lower_bounds, upper_bounds = [], []
    for pat, is_lower in (
            (r"minimum of\s+([^,.;]+)", True),
            (r"at least\s+([^,.;]+)", True),
            (r"more than\s+([^,.;]+)", True),
            (r"over\s+([^,.;]+)", True),
            (r"maximum of\s+([^,.;]+)", False),
            (r"at most\s+([^,.;]+)", False),
            (r"less than\s+([^,.;]+)", False),
            (r"within\s+([^,.;]+)", False)):
        for m in re.finditer(pat, ctx_l):
            for q in parse_temporal(m.group(1)):
                if q.kind == "duration":
                    (lower_bounds if is_lower else upper_bounds).append(
                        q.minutes)

    dom_kind = _dominant_kind(ctx_qs) or _dominant_kind(all_qs)

    # Unit votes: which duration units appear near the blank vs anywhere.
    ctx_units = [q.unit for q in ctx_qs if q.kind == "duration"]
    all_units = [q.unit for q in all_qs if q.kind == "duration"]

    for i, (opt, qs) in enumerate(zip(options, opt_parsed)):
        s = 0.0
        reasons = []
        if not qs:
            # option has no parseable temporal quantity: weak penalty (could
            # still be "2008" style date handled by parse; truly opaque gets
            # no evidence either way)
            s -= 0.5
            reasons.append("unparseable")
            scores[i] = s
            why[i] = "+".join(reasons)
            continue
        q = qs[0]

        # kind agreement with context
        if dom_kind and q.kind == dom_kind:
            s += 1.0
            reasons.append(f"kind={q.kind}")

        # unit agreement: same duration unit as expressions near the blank
        # (people answer "how long was the meeting delayed" in the same unit
        # the dialog is already using — minutes stay minutes).
        if q.kind == "duration" and q.unit:
            if q.unit in ctx_units:
                s += 1.0
                reasons.append(f"unit-ctx:{q.unit}")
            elif q.unit in all_units:
                s += 0.4
                reasons.append(f"unit-dlg:{q.unit}")

        # idiom
        if idiom_24 and q.kind == "duration" and q.unit == "hour" \
                and q.minutes == 24 * 60:
            s += 3.0
            reasons.append("idiom-24h-a-day")

        # comparative bounds
        if q.kind == "duration" and q.minutes is not None:
            ok = True
            for lb in lower_bounds:
                if q.minutes <= lb:
                    ok = False
            for ub in upper_bounds:
                if q.minutes >= ub:
                    ok = False
            if (lower_bounds or upper_bounds):
                if ok:
                    s += 2.0
                    reasons.append("bounds-ok")
                else:
                    s -= 2.0
                    reasons.append("bounds-violated")

        # NOTE: a "verbatim-reuse" penalty (penalizing options whose quantity
        # already appears in the dialog) was tried and MEASURED WORSE
        # (top1-correct1 0.372 -> 0.320 on n=400): TimeDial correct answers
        # often legitimately repeat a stated quantity ("minimum of twelve
        # hours ... we've allowed forty-eight hours"). Kept out.

        # distractor penalty: same NUMBER as a context quantity but a
        # different unit (e.g. context "22" as clock closing hour, option
        # "22 hours") — a signature TimeDial distractor construction.
        for cq in all_qs:
            for oq in qs:
                if (oq.kind == "duration" and cq.kind != "duration"
                        and oq.minutes is not None):
                    cnum = None
                    mnum = re.search(r"\d+(?:\.\d+)?|" +
                                     "|".join(_WORD_NUM), cq.raw.lower())
                    onum = re.search(r"\d+(?:\.\d+)?|" +
                                     "|".join(_WORD_NUM), oq.raw.lower())
                    if mnum and onum:
                        a = _wordnum_to_float(mnum.group(0))
                        b = _wordnum_to_float(onum.group(0))
                        if a is not None and a == b:
                            s -= 1.5
                            reasons.append("num-reuse-unit-swap")

        # absurd-unit penalty: seconds for multi-person scheduled activities
        if q.kind == "duration" and q.unit == "second":
            if re.search(r"meeting|flight|trip|course|project|vacation|"
                         r"holiday|semester|work|job|stay", dialog_l):
                s -= 2.0
                reasons.append("absurd-seconds")

        # magnitude plausibility versus context durations: options wildly
        # off-scale (>200x or <1/200 of every context duration) are unlikely
        ctx_durs = [c.minutes for c in all_qs
                    if c.kind == "duration" and c.minutes]
        if q.kind == "duration" and q.minutes and ctx_durs:
            ratios = [max(q.minutes / c, c / q.minutes) for c in ctx_durs]
            if min(ratios) > 200:
                s -= 1.0
                reasons.append("off-scale")
            elif min(ratios) <= 6:
                s += 0.5
                reasons.append("in-scale")

        scores[i] = s
        why[i] = "+".join(reasons) or "none"

    best = max(range(len(options)), key=lambda i: scores[i])
    return best, scores[best], why[best]
