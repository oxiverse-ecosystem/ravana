"""Brain-faithful relational reasoning — System-2 problem working memory.

Neuroscience grounding (per the relational-reasoning plan):
- Frankland & Greene (2015/2020, PNAS): PFC binds abstract ROLES
  (Agent/Patient/Relation) independent of word meaning; this is the
  structural substrate relational integration rides on.
- Baddeley (2003) / Eichenbaum (1997): working memory is an
  ephemeral, capacity-limited buffer — it does NOT persist to disk and
  does NOT consult lifetime associative frequencies.
- Evans & Stanovich (2013) / Kahneman (2011): System 2 "cognitive
  decoupling" — reason over the problem's OWN premises, setting aside
  System-1 co-occurrence stats.

Design consequences enforced here:
1. ProblemWorkingMemory is EPHEMERAL. It is built fresh per turn from
   the question's premises and is NEVER persisted to the lifetime
   triplet operator / RelationProfile store. This is the hard
   decoupling: a novel relation ("left_of") chains on first exposure
   within the problem with ZERO dependence on historical counts.
2. RoleMetaruleEngine applies SECOND-ORDER metarules over an abstract
   relation variable R (A-R-B & B-R-C => A-R-C), not per-predicate
   frequency gates. Transitivity is licensed by PREMISE STRUCTURE
   (the problem states the chain), not by a Wilson bound over
   lifetime evidence.
3. Relation-type knowledge (symmetric / inverse / rule) is carried as
   METADATA ON THE PREMISE ITSELF (e.g. a universal "for all x,y:
   left_of(x,y) iff right_of(y,x)" asserts the inverse). There is no
   global hardcoded per-relation table gating every predicate — the
   property lives where the brain would put it: in the stated premise.

Why discrete role-binding rather than HRR vectors (deliberate call):
HRRReasoner exists and does vector chaining, but its clean-up decode is
approximate. For crisp deductive entailment under a >=95% precision
bar, exact discrete binding is the faithful choice — PFC's explicit
variable binding for stated premises is discrete, not fuzzy. The module
is structured so the binding substrate could be swapped without changing
the metarules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


# Closed-class directional antonyms (grammar, not world knowledge —
# parallel to canonical.py's copula collapse). Used ONLY by the NL
# parser to infer an inverse predicate from a relational phrase; it is
# not a per-relation frequency/rule table. Inverse can also be stated
# explicitly on a premise (relation_type / inverse field), which takes
# precedence.
_DIRECTIONAL_OPPOSITES = {
    "loc:leftof": "loc:rightof", "loc:rightof": "loc:leftof",
    "loc:above": "loc:below", "loc:below": "loc:above",
    "loc:infront": "loc:behind", "loc:behind": "loc:infront",
    "compare:taller": "compare:shorter", "compare:shorter": "compare:taller",
    "compare:greater": "compare:less", "compare:less": "compare:greater",
    "compare:faster": "compare:slower", "compare:slower": "compare:faster",
}

# Canonical positional-relation key per directional word. This is the
# grammar of spatial prepositions (closed-class), mapping "left"/"right"
# to "loc:leftof"/"loc:rightof" — parallel to canonical.py's copula
# collapse. It is NOT a per-relation world-knowledge table.
_LOC_KEY = {
    "left": "loc:leftof", "right": "loc:rightof",
    "top": "loc:above", "above": "loc:above",
    "bottom": "loc:below", "below": "loc:below",
    "infront": "loc:infront", "behind": "loc:behind",
    "nextto": "loc:nextto", "beside": "loc:nextto",
    "inside": "loc:inside", "outside": "loc:outside",
    "ontop": "loc:ontop",
    # diagonal (ordinal) directions — same closed-class grammar family
    "northwest": "loc:northwestof", "northeast": "loc:northeastof",
    "southwest": "loc:southwestof", "southeast": "loc:southeastof",
}


@dataclass
class DeductiveTriple:
    """A single premise or derived conclusion in the working memory.

    Faithful extension of the base Triple concept with the logical
    metadata the old (s,p,o) triple lacks: quantifier, polarity, and
    relation-type. Backward-compatible: every field defaults, so a bare
    DeductiveTriple(s, p, o) is a plain positive atomic assertion.
    """
    subject: str
    predicate: str
    object: str
    quantifier: str = "none"      # "forall" | "exists" | "none"
    polarity: bool = True          # False = negated premise
    relation_type: str = "general" # "transitive"|"symmetric"|"inverse"|"rule"|"general"
    inverse: Optional[str] = None   # declared inverse predicate (for inverse metarule)
    source: str = "premise"        # "premise" | "derived"
    confidence: float = 1.0
    path: str = ""                 # human-readable derivation chain

    def key(self) -> Tuple[str, str, str, bool]:
        # polarity is part of identity: A-R-B and NOT(A-R-B) coexist.
        return (self.subject, self.predicate, self.object, self.polarity)

    def to_dict(self) -> dict:
        return {
            "s": self.subject, "p": self.predicate, "o": self.object,
            "q": self.quantifier, "neg": self.polarity is False,
            "rt": self.relation_type, "inv": self.inverse,
            "src": self.source, "c": self.confidence, "path": self.path,
        }


class ProblemWorkingMemory:
    """Ephemeral PFC-style buffer holding ONE problem's premises.

    NOT persisted. Built per turn, discarded after the turn. Holds the
    bound (subject, relation, object) tuples and supports structural
    query + contradiction detection. Relation-type metadata rides on the
    premises, so the buffer is self-contained (no external table).
    """

    def __init__(self):
        self.triples: List[DeductiveTriple] = []
        self._index: Dict[Tuple[str, str, bool], List[DeductiveTriple]] = {}
        self._relations: Set[str] = set()

    def add(self, t: DeductiveTriple) -> None:
        # De-dupe identical (s,p,o,polarity) to keep fixpoint iteration finite.
        for existing in self.triples:
            if existing.key() == t.key():
                # keep the higher-confidence / derived flag
                if t.source == "derived" and existing.source == "premise":
                    existing.source = "derived"
                    existing.confidence = t.confidence
                    existing.path = t.path
                return
        self.triples.append(t)
        self._index.setdefault((t.subject, t.predicate, t.polarity), []).append(t)
        self._relations.add(t.predicate)

    def has_fact(self, s: str, p: str, o: str, polarity: bool = True) -> bool:
        for t in self._index.get((s, p, polarity), ()):
            if t.object == o:
                return True
        return False

    def objects_of(self, s: str, p: str, polarity: bool = True) -> List[str]:
        return [t.object for t in self._index.get((s, p, polarity), ())]

    def subjects_of(self, o: str, p: str, polarity: bool = True) -> List[str]:
        # reverse lookup: who has p->o
        out = []
        for t in self.triples:
            if t.predicate == p and t.object == o and t.polarity == polarity:
                out.append(t.subject)
        return out

    def entail_confidence(self, t: DeductiveTriple) -> float:
        """Return confidence if the buffer entails `t` (as premise or
        derived fact), else 0.0. Entailment is exact membership under
        matching polarity — the buffer already holds transitive/symmetric/
        inverse derivatives, so no extra search is needed."""
        for cand in self._index.get((t.subject, t.predicate, t.polarity), ()):
            if cand.object == t.object:
                return cand.confidence
        return 0.0

    def contradiction_exists(self) -> bool:
        """Negative-resolution metarule: (A,R,B) and NOT(A,R,B) both
        asserted -> contradiction."""
        for t in self.triples:
            if t.polarity and self.has_fact(t.subject, t.predicate, t.object, False):
                return True
        return False


class RoleMetaruleEngine:
    """Second-order metarules over an abstract relation variable R.

    Every metarule is structurally licensed — it fires because the
    PREMISES support the pattern, never because a lifetime frequency
    crossed a threshold. This is the System-2 decoupling.
    """

    def __init__(self, decay: float = 0.9, max_passes: int = 6):
        self.decay = decay          # confidence decay per metarule hop
        self.max_passes = max_passes

    def apply(self, pwm: ProblemWorkingMemory) -> None:
        """Run metarules to fixpoint (bounded). Repeated passes let a
        derived transitive fact chain one more hop (A-R-B,B-R-C,C-R-D
        => A-R-D), which a single pass would miss."""
        for _ in range(self.max_passes):
            before = len(pwm.triples)
            self._transitive(pwm)
            self._symmetric(pwm)
            self._inverse(pwm)
            self._rule_implication(pwm)
            if len(pwm.triples) == before:
                break

    # ── Transitive metarule: A-R-B & B-R-C => A-R-C ───────────────
    def _transitive(self, pwm: ProblemWorkingMemory) -> None:
        # Snapshot current facts so this pass chains over them; derived
        # facts are added but the adjacency is rebuilt from the snapshot.
        for R in list(pwm._relations):
            adj: Dict[str, List[str]] = defaultdict(list)
            for t in list(pwm.triples):
                if t.predicate == R and t.polarity:
                    adj[t.subject].append(t.object)
            for s in list(adj):
                seen: Set[str] = {s}
                frontier: List[Tuple[str, float, List[str]]] = [
                    (b, self.decay, [s, b]) for b in adj[s]]
                hops = 1
                while frontier and hops < 5:
                    nxt: List[Tuple[str, float, List[str]]] = []
                    for node, conf, path in frontier:
                        for c in adj.get(node, ()):
                            if c in seen:
                                continue
                            seen.add(c)
                            np = conf * self.decay
                            if not pwm.has_fact(s, R, c, True):
                                # Don't derive a fact the premises negate.
                                if pwm.has_fact(s, R, c, False):
                                    continue
                                pwm.add(DeductiveTriple(
                                    s, R, c, polarity=True,
                                    source="derived", confidence=np,
                                    path="→".join(path)))
                            nxt.append((c, np, path + [c]))
                    frontier = nxt
                    hops += 1

    # ── Symmetric metarule: A-R-B => B-R-A (R marked symmetric) ───
    def _symmetric(self, pwm: ProblemWorkingMemory) -> None:
        for t in list(pwm.triples):
            if t.relation_type == "symmetric" and t.polarity:
                if not pwm.has_fact(t.object, t.predicate, t.subject, True):
                    pwm.add(DeductiveTriple(
                        t.object, t.predicate, t.subject,
                        polarity=True, source="derived",
                        confidence=t.confidence * self.decay,
                        path=f"{t.subject}↔{t.object}"))

    # ── Inverse metarule: A-R1-B => B-R2-A (R2 declared inverse) ──
    def _inverse(self, pwm: ProblemWorkingMemory) -> None:
        for t in list(pwm.triples):
            inv = t.inverse
            if inv is None:
                inv = _DIRECTIONAL_OPPOSITES.get(t.predicate)
            if inv and t.polarity:
                if not pwm.has_fact(t.object, inv, t.subject, True):
                    if pwm.has_fact(t.object, inv, t.subject, False):
                        continue
                    pwm.add(DeductiveTriple(
                        t.object, inv, t.subject,
                        polarity=True, source="derived",
                        confidence=t.confidence * self.decay,
                        path=f"{t.subject}-{t.predicate}⇒{t.object}-{inv}"))

    # ── Rule implication (universal syllogism): ──────────────────────
    #   ∀x: CLASS(x) -> OBJ(x)   (premise relation_type=="rule",
    #   predicate=="implies", subject=CLASS, object=OBJ)
    #   INSTANCE: e is CLASS        (predicate in {"is","isa"})
    #   => e is OBJ
    def _rule_implication(self, pwm: ProblemWorkingMemory) -> None:
        rules: List[DeductiveTriple] = [
            t for t in pwm.triples
            if t.relation_type == "rule" and t.predicate == "implies"
            and t.polarity]
        if not rules:
            return
        for rule in rules:
            cls, obj = rule.subject, rule.object
            for inst in list(pwm.triples):
                if inst.predicate in ("is", "isa") and inst.object == cls \
                        and inst.polarity:
                    e = inst.subject
                    if not pwm.has_fact(e, "is", obj, True):
                        if pwm.has_fact(e, "is", obj, False):
                            continue
                        pwm.add(DeductiveTriple(
                            e, "is", obj, polarity=True, source="derived",
                            confidence=inst.confidence * self.decay,
                            path=f"{e}-is-{cls}⊨{cls}⇒{obj}"))


# ── Premise parser (NL -> DeductiveTriples) ─────────────────────────
# Focused, relation-aware extraction. This is the extraction layer the
# P0 probe showed was the real LogiQA bottleneck; here it targets the
# relation FORMS the engine can actually chain (copula, comparative,
# positional, universal). It is intentionally narrow and honest about
# its limits — it is not a general NL reasoner.
import re  # noqa: E402

_CONJ = re.compile(
    r"\b(and|but|however|because|therefore|so|thus|whereas|although|"
    r"though|while)\b", re.IGNORECASE)
_FORALL = re.compile(r"^\s*(all|every|each|any)\b", re.IGNORECASE)
_EXISTS = re.compile(r"^\s*(some|at least one|one of|several|many|a few)\b",
                     re.IGNORECASE)
_NEG = re.compile(
    r"\b(not|no|never|none|isn'?t|aren'?t|cannot|can'?t|doesn'?t|"
    r"don'?t|without|except|incorrect|false|wrong)\b", re.IGNORECASE)

_REL_PATTERNS = [
    # universal rule: "All X are Y" / "Every X is Y"
    (re.compile(r"^(?:all|every|each|any)\s+(.+?)\s+(?:are|is)\s+(.+?)\.?$",
                re.IGNORECASE),
     lambda m: ("implies", m.group(1), m.group(2), "rule", True)),
    # comparative: "A is <adj> than B"
    (re.compile(r"^(.+?)\s+is\s+(.+?)\s+than\s+(.+?)\.?$", re.IGNORECASE),
     lambda m: ("compare:" + re.sub(r"[^a-z]", "", m.group(2).lower()),
                 m.group(1), m.group(3), "transitive", False)),
    # positional: "A is (to the) left/right/... of B"
    (re.compile(
        r"^(.+?)\s+is\s+(?:to\s+the\s+|on\s+the\s+)?"
        r"(left|right|top|bottom|above|below|in\s*front|behind|"
        r"next\s*to|beside|inside|outside|on\s*top|"
        r"northwest|northeast|southwest|southeast)\b"
        r"(?:\s+of|s+to)?\s*(.+?)\.?$", re.IGNORECASE),
     lambda m: (_LOC_KEY.get(m.group(2).lower().replace(" ", ""),
                             "loc:" + re.sub(r"[^a-z]", "", m.group(2).lower()))
                 if _LOC_KEY.get(m.group(2).lower().replace(" ", ""))
                 else "loc:" + re.sub(r"[^a-z]", "", m.group(2).lower()),
                 m.group(1), m.group(3), "general", False)),
    # copula: "A is B" / "A is a B"
    (re.compile(r"^(.+?)\s+is\s+(?:a|an|the)?\s*(.+?)\.?$", re.IGNORECASE),
     lambda m: ("is", m.group(1), m.group(2), "general", False)
                if not _is_noise_copula(m.group(1), m.group(2)) else None),
]

_MAX_TOK = 5  # term length cap; above this it's a blob, drop it.

# Broca's-style propositional filter: a copula "X is Y" is NOT a
# relational premise when the subject is a verb (e.g. "researchers
# hypothesized") or the object is a meta-noun that merely reifies the
# clause ("result", "reason", "factor", "thing", "case", "way",
# "situation", "fact"). Such frames are discarded, not treated as
# facts — this is what stops hypothesis preambles from blobbing.
_NOISE_OBJ = {
    "result", "reason", "factor", "thing", "case", "way",
    "situation", "fact", "example", "point", "issue", "matter",
    "question", "problem", "claim", "statement",
}
_VERB_LIKE = ("hypothes", "claim", "state", "argue", "suggest",
               "believe", "assume", "consider", "report", "note")


def _is_noise_copula(s: str, o: str) -> bool:
    if not s or not o:
        return True
    if o in _NOISE_OBJ:
        return True
    if s.lower().startswith(_VERB_LIKE):
        return True
    return False


def _canon(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r"^(a|an|the|my|your|our|their|his|her|its)\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip().strip(".,;:!?'\"")
    # minimal morphology: collapse trailing plural 's' so "men"/"man",
    # "dogs"/"dog" match. Grammar-only (regular plural), not
    # lemmatization of irregulars — those are rare in clean premises.
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def parse_deductive_premises(text: str) -> List[DeductiveTriple]:
    """Extract DeductiveTriples from a clause of text.

    Returns [] when nothing clean parses (fail-closed on noise). The
    first (most specific) relation pattern that matches wins per clause.

    Lever-2 wiring: the primary path is the neuro-symbolic
    DeductivePremiseExtractor (spaCy NP de-blobbing + the relation
    patterns defined below, reused as the single source of truth).
    If spaCy is unavailable it degrades to the regex path that
    follows. THIS function's own patterns remain the fallback so the
    channel still runs without the model.
    """
    from ravana.core.deductive_extractor import DeductivePremiseExtractor
    try:
        return DeductivePremiseExtractor(use_spacy=True).extract(text)
    except Exception:
        pass  # fall through to the regex path below
    out: List[DeductiveTriple] = []
    if not text:
        return out
    clauses = [text]
    # sentence + coordinating split
    sents = re.split(r"(?<=[.!?])\s+", text)
    clauses = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        for seg in _CONJ.split(s):
            seg = seg.strip().strip(",.!?;:\"'")
            if len(seg.split()) >= 2:
                clauses.append(seg)
    if not clauses:
        clauses = [text]

    for clause in clauses:
        q = "none"
        if _FORALL.search(clause):
            q = "forall"
        elif _EXISTS.search(clause):
            q = "exists"
        neg = bool(_NEG.search(clause))
        matched = False
        for pat, fn in _REL_PATTERNS:
            m = pat.match(clause)
            if not m:
                continue
            rel, s_raw, o_raw, rtype, _ = fn(m)
            if rel is None:
                # propositional-noise copula (e.g. "hypothesized is result")
                continue
            s = _canon(s_raw)
            o = _canon(o_raw)
            if not s or not o or len(s.split()) > _MAX_TOK \
                    or len(o.split()) > _MAX_TOK:
                continue
            if s == o:
                continue
            out.append(DeductiveTriple(
                subject=s, predicate=rel, object=o,
                quantifier=q, polarity=not neg, relation_type=rtype))
            matched = True
            break
        # (unmatched clause = not a clean relational premise; skip silently)
    return out


def deductive_mc_answer(user_input: str) -> Optional[str]:
    """Standalone fail-closed MC candidate (no engine needed).

    Parses the question's premises into a fresh ProblemWorkingMemory,
    applies the metarule engine, then checks which option (if exactly
    one) is entailed by the buffer. Returns the option string or None.

    This is the pure-logic core; the engine wires it after its evidence
    handlers abstain. It deliberately does NOT touch self.triplet_op /
    RelationProfile, so it has zero dependence on lifetime frequencies.
    """
    from ravana.core import fact_reasoning as _frz  # local import guard
    try:
        main, opts = _frz._split_options(user_input)
    except Exception:
        return None
    if len(opts) < 2:
        return None
    pwm = ProblemWorkingMemory()
    for t in parse_deductive_premises(main):
        pwm.add(t)
    if not pwm.triples:
        return None
    RoleMetaruleEngine().apply(pwm)
    if pwm.contradiction_exists():
        # Contradictory premises -> abstain rather than fabricate.
        return None

    scored = []
    for i, opt in enumerate(opts):
        claims = parse_deductive_premises(opt)
        if not claims:
            continue
        best = 0.0
        for c in claims:
            conf = pwm.entail_confidence(c)
            if conf > best:
                best = conf
        if best > 0.0:
            scored.append((best, i, opt))
    if len(scored) != 1:
        # zero = nothing entailed; >1 = ambiguous. Abstain both.
        return None
    return scored[0][2]
