"""Referent-head extraction with a runtime-growable grammatical function class.

Why this module exists
----------------------
Topic extraction across RAVANA is spread over several call sites (query
grounding, the self-opinion stance target, the uncertainty frame's subject
sanitiser). Each of them independently filters with its own hand-kept
closed-class list, and each of them leaks the same class of token:

* a NEGATION or AUXILIARY particle survives filtering and becomes the topic
  ("tell me something you don't know much about" -> subject ``don't``), and
* a whole CLAUSE is taken as the topic because nothing distinguishes the
  predicate from the referent ("naming things matters" as a stance target).

Both are the same underlying failure: no shared notion of which tokens can
*denote* something. This module is that notion, in one place.

What it provides
----------------
1. :func:`expand_contractions` — splits English negative contractions by
   MORPHOLOGY (``X n't`` -> ``X not``), so an unlisted contraction is handled
   by the same rule that handles the listed ones.
2. :class:`FunctionClass` — a closed-class seed (determiners, pronouns,
   auxiliaries, modals, negation, conjunctions, prepositions, quantifiers,
   question words) that GROWS AT RUNTIME from evidence: a token that RAVANA
   keeps extracting as a topic while holding no concept node, no embedding,
   and no personal-fact evidence is a token that denotes nothing *in this
   mind's world model*, so it is demoted to grammatical after a small number
   of observations (:meth:`FunctionClass.observe_topic_token`).
3. :func:`referent_head` — reduces a clause to its head referent: closed-class
   and negation tokens are dropped, then trailing predicate tokens are peeled
   off, leaving the maximal content noun phrase. Verb detection prefers a
   caller-supplied part-of-speech lookup (real state) and falls back to a
   small seed predicate class.

This is SEED STRUCTURE, not an answer table: nothing here can produce a reply,
and removing a seed entry degrades gracefully (the runtime class and the POS
lookup still carry the extraction). Learning is online and incremental — no
retraining, no rebuild, no corpus pass.

Doctrine note: the seed lists are a *closed-class vocabulary* (the linguistic
equivalent of a reflex), not keyed answers. The growth path is what makes them
seeds rather than a frozen table: :meth:`FunctionClass.learn` accepts any word
observed in function position, and :meth:`FunctionClass.observe_topic_token`
derives new members from RAVANA's own extraction evidence.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, FrozenSet, Optional, Set

__all__ = [
    "expand_contractions",
    "FunctionClass",
    "referent_head",
    "default_function_class",
]


# ── Contraction morphology ────────────────────────────────────────────────
# The RULE handles the whole n't class (don't/doesn't/didn't/isn't/can't/
# wouldn't/shouldn't/mustn't/mightn't/aren't/haven't...). English deletes the
# auxiliary's final letter before n't, so stripping "n't" alone yields the
# wrong stem: "don't" -> "do not" but "can't" -> "ca not". The stem is
# therefore chosen by asking which candidate is an actual word, with the
# one-letter strip as the fallback for an auxiliary the seed has not seen.
_NT_RE = re.compile(r"(\w+?)n['’]t\b")

_IRREGULAR_NEG = {
    "won't": "will not",
    "shan't": "shall not",
    "ain't": "is not",
    "aren't": "are not",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "y'all": "you all",
    "let's": "let us",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "it's": "it is",
}


# ── Elision morphology (the same idea, the other lost apostrophe) ────────
# THE RULE: English also drops the apostrophe in "<pronoun/question-word>'s"
# ("what's" -> "what is", "whats" -> "what is"). Unlike the n't class there is
# no auxiliary to recover — the missing element is always the copula "is" — so
# the rule is: a token ending in "s" whose STEM is a word that can take a
# copula is that word plus "is". Grammatical, not enumerated: the stems come
# from the function class below, so a form nobody listed ("wheres", "hows")
# reduces by the same rule, exactly as "don't" did.
#
# Two guards keep it from eating real words:
#   * the token itself must NOT already be a function word — that protects
#     "its" (possessive), "was", "has", "as", "is", "thus" and every other
#     closed-class item that legitimately ends in "s";
#   * only closed-class stems qualify, so a plural noun ("dogs") whose stem
#     happened to be a pronoun is left alone.
_ELISION_STEMS: FrozenSet[str] = frozenset("""
i you he she it we they that there what who where when why how
""".split())

_ELISION_RE = re.compile(r"\b([a-z][a-z']*)s\b")


def _neg_stem(raw: str) -> str:
    """Auxiliary stem of a negative contraction.

    ``"don't"`` -> ``"do"`` (strip the ``n``), ``"can't"`` -> ``"can"`` (strip
    nothing), ``"doesn't"`` -> ``"does"`` (strip the ``n``, restore the silent
    ``e``). Prefers a candidate that is a real word in the seed so an
    auxiliary nobody enumerated still reduces correctly.
    """
    for cand in (raw, raw[:-1], raw[:-2], raw[:-1] + "e", raw + "e"):
        if cand and cand in _SEED_FUNCTION:
            return cand
    return raw[:-1] if len(raw) > 3 else raw


def expand_contractions(text: str) -> str:
    """Expand English contractions to their full ``aux + not`` / ``X is`` form.

    ``"i don't know"`` -> ``"i do not know"``; ``"whats my dog called"`` ->
    ``"what is my dog called"``. Both classes are done by MORPHOLOGY, so a
    contraction the author never saw is still expanded by the same rule.
    """
    s = (text or "").lower()
    for src, dst in _IRREGULAR_NEG.items():
        s = re.sub(r"\b" + re.escape(src) + r"\b", dst, s)
    s = _NT_RE.sub(lambda m: f"{_neg_stem(m.group(1))} not", s)
    return _ELISION_RE.sub(_elision, s)


def _elision(m: "re.Match") -> str:
    """``whats`` -> ``what is``; leave every other token untouched.

    Fires only when the WHOLE token is not itself a closed-class word (so
    "its", "was", "has", "is", "as" survive) and its stem is a pronoun /
    question word that takes a copula. Both tests are membership lookups in
    the seed function class — no per-form table.
    """
    tok = m.group(1)
    if m.group(0) in _SEED_FUNCTION:
        return m.group(0)
    # group(1) is the token WITHOUT its trailing -s, so it IS the stem.
    if tok in _ELISION_STEMS:
        return f"{tok} is"
    return m.group(0)


# ── Seed closed class ─────────────────────────────────────────────────────
# Determiners, quantifiers, pronouns, auxiliaries, modals, negation,
# conjunctions, prepositions, adverbs-of-degree, question words, copulas.
# Closed-class vocabulary, extended at runtime by FunctionClass.
_SEED_FUNCTION: FrozenSet[str] = frozenset("""
a an the this that these those my your his her its our their my
some any all both each every no none neither either another
i me you he she it we they who whom whose which what
my mine yours ours theirs myself yourself himself herself itself ourselves
something anything everything nothing someone anyone everyone
somebody anybody everybody nobody one ones
am is are was were be been being
do does did done doing have has had having
will would shall should can could may might must
not n't never no none nothing nobody nowhere neither nor
and or but so because although though while whereas if then than
of in on at by for with from to into onto about across after before
during through over under between among against within without
up down out off around near past toward towards
very really quite rather too so just only even still yet
much many most more less least lot lots little hardly barely almost
nearly pretty fairly somewhat indeed truly
here there when where why how whether
""".split())

_INTRANSITIVE_PREDICATE_SEED = frozenset(
    "happens occurred occurs exists matters counts".split())

# Small seed predicate class, used only when the caller has no part-of-speech
# lookup available. Kept tiny on purpose: the POS lookup and the engine's
# consolidated verb vocabularies (routed in via FunctionClass.learn_predicate)
# are the real sources.
_SEED_PREDICATE: FrozenSet[str] = frozenset("""
is are was were be been being am
do does did done have has had
know knows knew think thinks thought feel feels felt
want wants wanted need needs needed like likes liked
say says said tell tells told give gives gave
make makes made get gets got take takes took
happen happens happened occur occurs occurred exist exists
matters matter counts count works work mean means meant
affect affects affected cause causes caused
protect protects save saves keep keeps stop stops
become becomes became remain remains stays
""".split()) | _INTRANSITIVE_PREDICATE_SEED


class FunctionClass:
    """Closed-class token membership with an evidence-driven growth path.

    ``is_function`` answers "can this token denote a referent?" — the single
    question every topic extractor was answering badly on its own.

    Growth is online and evidence-based. :meth:`observe_topic_token` is called
    whenever a candidate topic is produced; a token that keeps being proposed
    as a topic while RAVANA holds no evidence that it names anything (no
    concept node, no embedding, no personal fact) is demoted to grammatical
    after ``threshold`` observations. That is a statement about RAVANA's own
    world model, not a hand-maintained blacklist, and it generalises to junk
    this author never anticipated.

    The PREDICATE side (:meth:`learn_predicate` / :meth:`is_predicate`) grows
    the same way. The engine's part-of-speech classifier is a suffix/seed
    heuristic and is blind to most verbs ("matters", "affects", "counts" all
    come back ``noun``), so a clause-final predicate would survive the head
    reduction and be interpolated into a reply. Rather than freezing a longer
    verb list, the engine registers a token as a predicate when it has
    POSITIVE evidence for one — a POS tag of ``verb``, membership in the
    engine's own verb/activity/occupation vocabularies, or RAVANA having
    parsed it as the predicate of a clause it just understood. Every such
    registration is permanent, so the class widens from live language.
    """

    def __init__(self, seed: Optional[Set[str]] = None,
                 predicate_seed: Optional[Set[str]] = None,
                 threshold: int = 2) -> None:
        self._seed: Set[str] = set(seed) if seed is not None else set(_SEED_FUNCTION)
        self._predicate_seed: Set[str] = (set(predicate_seed) if predicate_seed
                                          is not None else set(_SEED_PREDICATE))
        # Words RAVANA has been TOLD are grammatical, by position or by an
        # explicit call from a parser that recognised them.
        self._learned: Set[str] = set()
        # Words with positive evidence of being a predicate.
        self._predicates: Set[str] = set()
        # Words demoted by evidence: proposed as a topic, never grounded.
        self._null_counts: Dict[str, int] = {}
        self._threshold = max(1, int(threshold))

    # ── membership ────────────────────────────────────────────────────────
    @property
    def seed(self) -> FrozenSet[str]:
        return frozenset(self._seed)

    @property
    def learned(self) -> FrozenSet[str]:
        return frozenset(self._learned)

    def is_function(self, word: str) -> bool:
        """True when the token is grammatical and cannot be a referent.

        Contraction-aware: ``don't`` is not in the seed (only its expansion
        ``do not`` is), so a token is also rejected when EVERY token of its
        expanded form is grammatical. That keeps the seed a closed-class
        vocabulary while handling surface forms nobody enumerated.
        """
        w = (word or "").strip().lower().strip(".,!?;:\"")
        if not w:
            return True
        if w in self._seed or w in self._learned or self._is_demoted(w):
            return True
        if "'" in w or "’" in w:
            parts = [p for p in _TOKEN_RE.findall(expand_contractions(w)) if p]
            if parts and all(p in self._seed or p in self._learned
                             or self._is_demoted(p) for p in parts):
                return True
        return False

    def is_predicate_seed(self, word: str) -> bool:
        return (word or "").strip().lower() in self._predicate_seed

    def is_predicate(self, word: str) -> bool:
        """True when the token has evidence of being a predicate."""
        w = (word or "").strip().lower()
        return w in self._predicate_seed or w in self._predicates

    def learn_predicate(self, *words: str) -> Set[str]:
        """Register tokens with positive evidence of being predicates.

        The growth path for the predicate side. Callers pass a word only when
        they have real evidence — a POS tag of ``verb``, membership in one of
        the engine's verb vocabularies, or a successful clause parse. Returns
        the newly registered words. A word registered here is peeled off a
        clause tail by :func:`referent_head` but is never treated as
        grammatical, so a genuine noun that happens to look verb-like
        ("protect", "handle") is not silently banned from being a referent.
        """
        new = set()
        for raw in words:
            w = (raw or "").strip().lower().strip(".,!?;:\"")
            if w and w not in self._predicate_seed and w not in self._predicates:
                self._predicates.add(w)
                new.add(w)
        return new

    @property
    def predicates(self) -> FrozenSet[str]:
        return frozenset(self._predicates)

    def _is_demoted(self, w: str) -> bool:
        return self._null_counts.get(w, 0) >= self._threshold

    # ── growth paths ──────────────────────────────────────────────────────
    def learn(self, *words: str) -> Set[str]:
        """Register words seen in grammatical function position.

        The growth path for the seed vocabulary: any token a parser
        recognises as an auxiliary, determiner or particle becomes a permanent
        member, so the class widens from live language rather than from a
        frozen table. Returns the words that were newly learned.
        """
        new = set()
        for raw in words:
            w = (raw or "").strip().lower().strip(".,!?;:\"")
            if w and w not in self._seed and w not in self._learned:
                self._learned.add(w)
                new.add(w)
        return new

    def observe_topic_token(self, word: str, grounded: bool) -> bool:
        """Record one extraction outcome; return True if the token is a referent.

        ``grounded`` is the caller's honest answer to "does RAVANA hold any
        evidence this token names something?" — a concept node, an embedding,
        or a personal fact. A token proposed as a topic and never grounded is
        counted; once it crosses the threshold it is treated as grammatical
        and filtered out of future topics. A token that later DOES get grounded
        is forgiven immediately (its count is cleared), so a real concept is
        never permanently banned by a run of bad luck.
        """
        w = (word or "").strip().lower().strip(".,!?;:\"")
        if not w:
            return False
        if grounded:
            self._null_counts.pop(w, None)
            return True
        if self.is_function(w):
            return False
        self._null_counts[w] = self._null_counts.get(w, 0) + 1
        return not self._is_demoted(w)

    def null_counts(self) -> Dict[str, int]:
        return dict(self._null_counts)

    def reset_counts(self) -> None:
        self._null_counts.clear()

    # ── persistence ───────────────────────────────────────────────────────
    def to_state(self) -> Dict[str, object]:
        """Serialise the LEARNED part of the class (not the seed).

        The seed is code and is identical on every boot, so persisting it
        would just bloat the pickle; only what RAVANA discovered is state.
        """
        return {
            "learned": sorted(self._learned),
            "predicates": sorted(self._predicates),
            "null_counts": dict(self._null_counts),
        }

    def load_state(self, state: Optional[Dict[str, object]]) -> None:
        """Restore learned membership from a previous session.

        Fails open on a malformed or absent payload: a class that cannot be
        restored keeps its seed and simply re-learns, which is degraded but
        never wrong. No reply is authored and nothing is invented.
        """
        if not isinstance(state, dict):
            return
        for key in ("learned", "predicates"):
            vals = state.get(key)
            if isinstance(vals, (list, tuple, set)):
                for w in vals:
                    w = str(w).strip().lower()
                    if w:
                        if key == "learned":
                            self._learned.add(w)
                        else:
                            self._predicates.add(w)
        counts = state.get("null_counts")
        if isinstance(counts, dict):
            for k, v in counts.items():
                try:
                    self._null_counts[str(k).strip().lower()] = int(v)
                except Exception:
                    continue


# Process-wide class so growth observed by one call site benefits every other
# call site in the session (the same pattern as pet_slots / relation_attrs).
_DEFAULT = FunctionClass()


def default_function_class() -> FunctionClass:
    return _DEFAULT


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def referent_head(text: str,
                  pos_lookup: Optional[Callable[[str], Optional[str]]] = None,
                  func: Optional[FunctionClass] = None) -> Optional[str]:
    """Reduce a clause to the head referent it is about.

    ``"you don't know much about"``            -> ``None`` (no content token
                                                 survives — nothing is being
                                                 referred to)
    ``"naming things matters"``                 -> ``"naming things"``
    ``"the ethics of terraforming mars"``       -> ``"terraforming mars"``
    ``"mangroves"``                             -> ``"mangroves"``
    ``"remote work"``                           -> ``"remote work"``

    Two structural steps, in order:

    1. Drop every token that cannot denote a referent (closed class, negation,
       quantifiers). This is what stops ``don't`` becoming a topic.
    2. SEGMENT the remainder at clause structure and keep the first finite
       clause. A predicate clause ("X matters", "X is Y", "X affects Y") is
       about its subject, so everything from the predicate onward is dropped.
       Step 2 is deliberately *structural* rather than tag-based: RAVANA's POS
       classifier is a suffix heuristic that calls "sleep" a verb and "work" a
       noun, so peeling on a verb tag truncates real noun phrases ("remote
       work" -> "remote"). A clause boundary, by contrast, is a grammatical
       fact that holds whatever the tagger says.

    Returns ``None`` when nothing denotes anything — callers must then fail
    open to their own honest fallback rather than inventing a referent.
    """
    if func is None:
        func = _DEFAULT
    expanded = expand_contractions(text or "")
    toks = [t for t in _TOKEN_RE.findall(expanded) if t]
    if not toks:
        return None

    # Clause segmentation happens BEFORE the closed-class filter, because the
    # copula that marks the clause edge is itself a function word: filtering
    # first would delete it and hand the predicate back as the topic
    # ("silence is underrated" -> "silence underrated").
    _CLAUSE_EDGE = (frozenset(
        "is are was were be been being am seem seems seemed appear appears "
        "appeared look looks looked sound sounds sounded feel feels felt "
        "become becomes became remain remains stays stayed "
        "have has had do does did can could will would should".split())
        | _INTRANSITIVE_PREDICATE_SEED)
    _REL_PRON = frozenset("who whom whose which".split())
    end = len(toks)
    for _i, _t in enumerate(toks):
        if _i == 0:
            # A clause must have a subject before its predicate; a leading
            # predicate is itself the head ("matters" is a whole answer).
            continue
        if _t in _REL_PRON:
            # A relative clause modifies the head noun; everything from the
            # relative pronoun on is the modifier, not part of the referent
            # ("people who talk too much" -> "people").
            end = _i
            break
        if _t in _CLAUSE_EDGE or _pos_says_verb(_t, pos_lookup, func):
            end = _i
            break
    head_span = toks[:end]

    # Now drop everything that cannot denote a referent: closed class,
    # negation, quantifiers. This is what stops ``don't`` becoming a topic.
    content = [t for t in head_span if not func.is_function(t)]
    if not content:
        return None
    return " ".join(content)


def _pos_says_verb(tok: str,
                   pos_lookup: Optional[Callable[[str], Optional[str]]],
                   func: FunctionClass) -> bool:
    """True when the caller's POS state and a learned predicate AGREE.

    A learned predicate alone is not enough: the engine's verb vocabularies
    contain noun-ambiguous words ("work", "sleep", "handle"), so trusting one
    without corroboration truncates real noun phrases. Requiring agreement
    keeps the growth path useful (a genuinely verb-tagged word RAVANA has
    learned acts as a clause edge) without letting it fire on a noun.
    """
    if not func.is_predicate(tok):
        return False
    if pos_lookup is None:
        return False
    p = pos_lookup(tok)
    return bool(p) and str(p).lower().startswith("verb")
