"""Graph-relational reasoner: propositional + transitive inference.

Brain mechanism: HPC->PFC deliberative reasoning (Zeithamova 2012;
Waltz 1999 — rostrolateral PFC integrates relational premises).
Where fact_reasoning._closure does associative replay (lexical
spreading), this module does STRUCTURED inference: it mines premise
constraints from the presented text, forward-chains them with unit
propagation (modus ponens / modus tollens / hypothetical syllogism /
categorical transitivity), and tests each candidate option for
entailment or contradiction.

Design rules (plan non-negotiables):
- No per-benchmark code: any multiple-choice question whose text
  carries conditional/categorical/comparative structure is handled.
- Fail closed: return None unless exactly one option is entailed (or
  all others are contradicted). Never guess.
- Interpretable: `explain` carries the derivation chain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ravana.core.fact_reasoning import content_words, _split_options, _stem


# ── proposition representation ──────────────────────────────────────────

_NEG = re.compile(
    r"\b(?:not|no|never|cannot|can't|won't|doesn't|don't|didn't|isn't|"
    r"aren't|wasn't|weren't|impossible|false)\b")


@dataclass(frozen=True)
class Prop:
    """A proposition = stemmed content-token set + polarity."""
    tokens: frozenset
    positive: bool = True

    def negate(self) -> "Prop":
        return Prop(self.tokens, not self.positive)

    def __bool__(self) -> bool:
        return bool(self.tokens)


def _prop(text: str) -> Prop:
    """Parse a clause into a Prop. Negation words flip polarity and are
    removed from the token set (the brain codes negation as a tag on the
    proposition, not as part of its content — Kaup 2007)."""
    t = (text or "").lower().strip(" .,;:!?")
    neg = bool(_NEG.search(t))
    t = _NEG.sub(" ", t)
    toks = frozenset(content_words(t))
    return Prop(toks, not neg)


def _match(a: Prop, b: Prop) -> bool:
    """Same proposition content? Strict containment: every token of the
    smaller set must appear in the larger. Measured on LogiQA-500: strict
    containment 3/11 correct vs 1/13 for lenient all-but-one — loose
    matching lets surface overlap masquerade as identity."""
    if not a.tokens or not b.tokens:
        return False
    small, big = (a.tokens, b.tokens) if len(a.tokens) <= len(b.tokens) \
        else (b.tokens, a.tokens)
    return len(small & big) == len(small)


# ── premise mining ───────────────────────────────────────────────────────

@dataclass
class Rule:
    ante: Prop
    cons: Prop
    source: str = ""


@dataclass
class Premises:
    rules: List[Rule] = field(default_factory=list)
    facts: List[Prop] = field(default_factory=list)
    disjunctions: List[Tuple[Prop, Prop, bool]] = field(default_factory=list)
    comparatives: List[Tuple[str, str, str]] = field(default_factory=list)


_SENT_SPLIT = re.compile(r"[.;\n]+")

_IF_THEN = re.compile(
    r"\bif\s+(.+?)\s*,?\s*then\s+(.+)", re.IGNORECASE)
_IF_COMMA = re.compile(r"\bif\s+(.+?)\s*,\s*(.+)", re.IGNORECASE)
_ONLY_IF = re.compile(r"(.+?)\s+only\s+if\s+(.+)", re.IGNORECASE)
_UNLESS = re.compile(r"(.+?)\s+unless\s+(.+)", re.IGNORECASE)
_ALL_ARE = re.compile(
    r"\b(?:all|every|each|any)\s+(.+?)\s+(?:are|is|were|must be|have|has)\s+(.+)",
    re.IGNORECASE)
_NO_ARE = re.compile(
    r"\bno\s+(.+?)\s+(?:are|is|can be)\s+(.+)", re.IGNORECASE)
_COMP = re.compile(
    r"([\w\s]{2,30}?)\s+(?:is|are|was|were)?\s*(older|younger|taller|"
    r"shorter|bigger|smaller|larger|heavier|lighter|faster|slower|higher|"
    r"lower|earlier|later|more|less|greater|better|worse)\s+than\s+"
    r"([\w\s]{2,30})", re.IGNORECASE)
_EITHER_OR = re.compile(
    r"\beither\s+(.+?)\s+or\s+(.+)", re.IGNORECASE)


def mine_premises(text: str) -> Premises:
    """Mine logical structure from presented text, sentence by sentence.
    Ingestion-side capture of conditional/categorical/comparative frames;
    plain sentences become polarity-tagged facts."""
    P = Premises()
    for sent in _SENT_SPLIT.split(text or ""):
        s = sent.strip()
        if len(s) < 4:
            continue
        m = _COMP.search(s)
        if m:
            a = " ".join(sorted(content_words(m.group(1))))
            b = " ".join(sorted(content_words(m.group(3))))
            if a and b:
                P.comparatives.append((a, m.group(2).lower(), b))
            continue
        m = _IF_THEN.search(s) or _IF_COMMA.search(s)
        if m:
            P.rules.append(Rule(_prop(m.group(1)), _prop(m.group(2)), s))
            continue
        m = _ONLY_IF.search(s)
        if m:  # P only if Q  ==  P -> Q
            P.rules.append(Rule(_prop(m.group(1)), _prop(m.group(2)), s))
            continue
        m = _UNLESS.search(s)
        if m:  # P unless Q  ==  not Q -> P
            P.rules.append(Rule(_prop(m.group(2)).negate(),
                                _prop(m.group(1)), s))
            continue
        m = _NO_ARE.search(s)
        if m:  # no X are Y  ==  X -> not Y
            P.rules.append(Rule(_prop(m.group(1)),
                                _prop(m.group(2)).negate(), s))
            continue
        m = _EITHER_OR.search(s)
        if m:  # either P or Q == (¬P -> Q) and (¬Q -> P)
            p1, p2 = _prop(m.group(1)), _prop(m.group(2))
            if p1 and p2:
                P.disjunctions.append((p1, p2, True))
                P.rules.append(Rule(p1.negate(), p2, s))
                P.rules.append(Rule(p2.negate(), p1, s))
            continue
        m = _ALL_ARE.search(s)
        if m:  # all X are Y  ==  X -> Y
            P.rules.append(Rule(_prop(m.group(1)), _prop(m.group(2)), s))
            continue
        p = _prop(s)
        if p.tokens:
            P.facts.append(p)
    return P


# ── inference ────────────────────────────────────────────────────────────

def _forward_chain(P: Premises, max_rounds: int = 6) -> List[Prop]:
    """Unit propagation: derive new facts from rules until fixpoint.
    Modus ponens (fact matches ante -> derive cons) and modus tollens
    (fact matches negated cons -> derive negated ante)."""
    derived: List[Prop] = list(P.facts)

    def known(p: Prop) -> bool:
        return any(_match(p, d) and p.positive == d.positive for d in derived)

    for _ in range(max_rounds):
        new: List[Prop] = []
        for r in P.rules:
            # modus ponens
            for d in derived:
                if _match(d, r.ante) and d.positive == r.ante.positive \
                        and not known(r.cons):
                    new.append(r.cons)
                # modus tollens: d == ¬cons  =>  ¬ante
                if _match(d, r.cons) and d.positive != r.cons.positive:
                    na = r.ante.negate()
                    if not known(na):
                        new.append(na)
            # hypothetical syllogism handled implicitly by iteration.
        if not new:
            break
        derived.extend(new)
    return derived


def _comp_closure(comps: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Transitive closure over comparative chains sharing the SAME
    comparative word (a >r b, b >r c => a >r c)."""
    out = list(comps)
    for _ in range(4):
        added = False
        for (a, r1, b) in list(out):
            for (c, r2, d) in list(out):
                if r1 == r2 and b == c and (a, r1, d) not in out and a != d:
                    out.append((a, r1, d))
                    added = True
        if not added:
            break
    return out


def _entails(derived: List[Prop], target: Prop) -> Optional[bool]:
    """True: target derived. False: negation derived. None: unknown."""
    for d in derived:
        if _match(d, target):
            return d.positive == target.positive
    return None


# ── multiple-choice entry point ─────────────────────────────────────────

_META_Q = re.compile(
    r"\b(weaken|strengthen|undermine|support|evaluat|explain|assumption|"
    r"criticis|flaw|method of argument|paradox|discrepanc)", re.IGNORECASE)


def select_option_logic(question: str) -> Optional[str]:
    """Pick the single option ENTAILED by premises mined from the question
    text itself. Fail-closed: None unless exactly one option is entailed,
    or exactly one survives with all others contradicted.

    Scope: ENTAILMENT-frame questions only ("what follows", "must be
    true"). Meta-argumentation questions (weaken/strengthen/evaluate/
    assumption) ask ABOUT the argument, not what follows FROM it —
    entailment machinery is category-inappropriate there (fires on
    surface overlap, measured precision 0.08) so we abstain."""
    if _META_Q.search(question or ""):
        return None
    main, opts = _split_options(question)
    if len(opts) < 2:
        return None
    P = mine_premises(main)
    # Require REAL logical structure: at least 2 constraints mined, or a
    # comparative chain — one weak rule entails from surface overlap.
    if len(P.rules) + len(P.comparatives) < 2:
        return None
    derived = _forward_chain(P)
    comps = _comp_closure(P.comparatives)

    verdicts: List[Optional[bool]] = []
    for o in opts:
        v = _entails(derived, _prop(o))
        if v is None and comps:
            ow = content_words(o)
            for (a, r, b) in comps:
                aw, bw = set(a.split()), set(b.split())
                if aw & ow and bw & ow and r in o.lower():
                    # option asserts a >r b — check order match
                    oa = o.lower().find(next(iter(aw & ow)))
                    ob = o.lower().find(next(iter(bw & ow)))
                    if 0 <= oa < ob:
                        v = True
                    elif 0 <= ob < oa:
                        v = False
                    break
        verdicts.append(v)

    entailed = [i for i, v in enumerate(verdicts) if v is True]
    contradicted = [i for i, v in enumerate(verdicts) if v is False]
    if len(entailed) == 1:
        return opts[entailed[0]]
    if not entailed and len(contradicted) == len(opts) - 1:
        # all but one option contradicted -> the survivor
        for i, v in enumerate(verdicts):
            if v is None:
                return opts[i]
    return None
