"""Linearization -- turn a role-bound message into an ordered surface plan.

Research deliverable: ``docs/BRAIN_LINEARIZATION_SPEC.md``.

The problem this addresses is measured, not hypothesized. RAVANA's decoder can
be made to emit the RIGHT content words: a logit boost on a retrieved
definition's content words drives on-topic to 1.00 and unsupported assertions
to 0 at every strength from 2.0 upward (``docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md``).
But the output is a WORD LIST, not a sentence:

    "Other earth body objects gives centre planet force gravity moon..."

Selection works. Ordering does not. That is the gap Levelt calls GRAMMATICAL
ENCODING: the message is a non-linear role-concept structure, and the
formulator must assign syntactic functions and commit to a surface order
BEFORE articulation (Levelt 1989; Kempen & Hoenkamp 1987 on incrementality).

## What this module does

``message_to_plan`` takes a ``macroplanning.Message`` and produces an ordered
``SurfacePlan`` -- a sequence of slots, each a (role, concept) pair in the
order the formulator should realize them. The order is derived from the
RELATION SEMANTICS of the producing edge, not from a hardcoded sentence frame:

* the message's TOPIC (the subject) is realized first -- it is the theme the
  utterance is about, and English topicalizes the subject;
* a categorizing relation (``is_a`` / ``part_of``) realizes as
  ``<subject> is a <theme>`` -- the classic definitional construction;
* a causal relation realizes as ``<agent> leads to <ground>``;
* an attributive relation realizes as ``<subject> has <relation>``.

This is a CONSTRUCTION in the usage-based sense (Goldberg 1995; Tomasello
2003): a form-meaning pairing learned from use, with slots filled by the
message's bound concepts. It is not a sentence template -- the concepts come
from the graph, and a different graph yields a different sentence.

## Why not just boost the words

Because a boost supplies concepts with no ROLES and no ORDER. The measurement
in ``docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md`` is exactly the predicted result of
that: the right words, no grammatical frame. This module supplies the frame
and lets the decoder fill it.

## Scope

Pure function over (Message, relation vocabulary) -> SurfacePlan. It touches no
decoder and authors no reply string. The caller hands the plan to the
formulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ravana.learn.macroplanning import Message, RoleBinding


# ── Constructions ────────────────────────────────────────────────────────────
# A construction is a form-meaning pairing: a surface frame with SLOTS, and a
# mapping from thematic roles to those slots. This is the usage-based unit
# (Goldberg 1995), not a sentence template -- the slot fillers come from the
# message, so the same construction yields a different sentence for a different
# graph.
#
# Each entry: (pattern, slot_order)
#   pattern    -- the surface frame, with {slot} placeholders
#   slot_order -- the roles that fill the slots, in realization order
#
# The patterns below are the brain's most-practiced constructions for
# definitional and relational speech. They are few and general, not a
# per-topic table.

@dataclass(frozen=True)
class Construction:
    """A form-meaning pairing: a surface frame plus its role->slot mapping."""

    name: str
    pattern: Tuple[str, ...]
    slots: Tuple[str, ...]

    def render(self, fill: Dict[str, str]) -> List[str]:
        """Fill the slots and return the surface token sequence."""
        out: List[str] = []
        for tok in self.pattern:
            if tok.startswith("{") and tok.endswith("}"):
                slot = tok[1:-1]
                val = fill.get(slot)
                if val:
                    out.append(val)
            else:
                out.append(tok)
        return out


# The construction inventory. Deliberately small and closed: these are the
# high-frequency constructions of definitional prose, not a grammar.
#
# ``slots`` names the ROLES (macroplanning's vocabulary: agent / theme /
# relation / ground) that fill the frame's placeholders, in the order the
# placeholders appear. Keeping the slot names in macroplanning's role
# vocabulary is what lets a message bind straight into a frame without a
# second translation table.
CONSTRUCTIONS: Dict[str, Construction] = {
    # "<theme> is a <theme>" -- categorization. The most-practiced
    # definitional form. Both slots are 'theme' because the subject and its
    # category are the same thematic role at different levels; the caller
    # fills slot 0 with the subject and slot 1 with the category.
    "categorize": Construction(
        name="categorize",
        pattern=("{theme}", "is", "a", "{theme}"),
        slots=("theme", "theme"),
    ),
    # "<agent> leads to <ground>" -- causation.
    "cause": Construction(
        name="cause",
        pattern=("{agent}", "leads", "to", "{ground}"),
        slots=("agent", "ground"),
    ),
    # "<theme> has <relation>" -- attribution.
    "attribute": Construction(
        name="attribute",
        pattern=("{theme}", "has", "{relation}"),
        slots=("theme", "relation"),
    ),
    # "<theme> is located in <ground>" -- location.
    "locate": Construction(
        name="locate",
        pattern=("{theme}", "is", "located", "in", "{ground}"),
        slots=("theme", "ground"),
    ),
    # "<theme> is related to <ground>" -- the least committal relational form.
    "relate": Construction(
        name="relate",
        pattern=("{theme}", "is", "related", "to", "{ground}"),
        slots=("theme", "ground"),
    ),
    # "<theme> follows <ground>" -- temporal succession.
    "temporal": Construction(
        name="temporal",
        pattern=("{theme}", "follows", "{ground}"),
        slots=("theme", "ground"),
    ),
    # "<theme> is like <ground>" -- analogy. Distinct from categorization:
    # an analogy asserts resemblance, not membership.
    "analogy": Construction(
        name="analogy",
        pattern=("{theme}", "is", "like", "{ground}"),
        slots=("theme", "ground"),
    ),
    # "<theme> contrasts with <ground>" -- contrast.
    "contrast": Construction(
        name="contrast",
        pattern=("{theme}", "contrasts", "with", "{ground}"),
        slots=("theme", "ground"),
    ),
    # "<theme> is inferred from <ground>" -- inference.
    "infer": Construction(
        name="infer",
        pattern=("{theme}", "is", "inferred", "from", "{ground}"),
        slots=("theme", "ground"),
    ),
    # "<theme> happened in <ground>" -- episodic grounding.
    "episodic": Construction(
        name="episodic",
        pattern=("{theme}", "happened", "in", "{ground}"),
        slots=("theme", "ground"),
    ),
}


def _construction_for(relation: str) -> Optional[Construction]:
    """Map a graph relation_type to the construction that realizes it.

    This is the relation-semantics -> surface-form mapping. It is a small
    closed table because the graph's typed relation vocabulary is small and
    closed (``graph_typing.TYPED_RELATION_TYPES``); an unknown relation falls
    back to the least committal construction rather than being dropped.
    """
    r = (relation or "").strip().lower()
    if r in ("is_a", "isa", "type", "kind", "category", "instance_of",
             "subclass", "part_of", "member_of"):
        return CONSTRUCTIONS["categorize"]
    # The graph's learned relation types are ADJECTIVAL ('causal',
    # 'analogical', 'contrastive', 'temporal', 'episodic', 'inferred'), while
    # graph_typing.TYPED_RELATION_TYPES are VERBAL ('causes', 'is_a',
    # 'has_property'). Both spellings occur in the persisted graph, so both
    # must map. Missing the adjectival forms silently demoted every causal and
    # analogical edge to the 'relate' fallback, which plan_is_committed then
    # rejected -- measured as 7/19 typed-edge subjects failing to commit
    # ("gravity is related to earth" from a CAUSAL edge).
    if r in ("causes", "cause", "produces", "produces_cause", "leads_to",
             "results_in", "effect", "affects", "influences", "enables",
             "causal"):
        return CONSTRUCTIONS["cause"]
    if r in ("agent", "by", "author", "creator", "source", "origin",
             "created_by", "from", "has_agent"):
        return CONSTRUCTIONS["cause"]
    if r == "temporal":
        return CONSTRUCTIONS["temporal"]
    if r in ("analogical", "analogy", "analog"):
        return CONSTRUCTIONS["analogy"]
    if r in ("contrastive", "contrast", "opposition"):
        return CONSTRUCTIONS["contrast"]
    if r in ("inferred", "inference", "implies"):
        return CONSTRUCTIONS["infer"]
    if r in ("episodic", "episode"):
        return CONSTRUCTIONS["episodic"]
    if r in ("location", "located_in", "in", "at", "place", "time",
             "part", "contains", "within"):
        return CONSTRUCTIONS["locate"]
    if r in ("has_property", "property", "attribute", "has", "capable_of",
             "used_for", "function", "purpose"):
        return CONSTRUCTIONS["attribute"]
    if r == "temporal":
        return CONSTRUCTIONS["temporal"]
    if r == "analogical":
        return CONSTRUCTIONS["analogy"]
    if r == "contrastive":
        return CONSTRUCTIONS["contrast"]
    if r == "inferred":
        return CONSTRUCTIONS["infer"]
    if r == "episodic":
        return CONSTRUCTIONS["episodic"]
    return CONSTRUCTIONS["relate"]


# Relations that assert a real, structured predicate rather than mere
# distributional proximity. Same distinction as
# response_gen.py's _DISTRIBUTIONAL_RELATIONS: a vacuous relation cannot carry
# a factual claim, so a plan built on one is fluent and empty.
VACUOUS_RELATIONS = frozenset({"semantic", "related", "related_to", ""})

# Fillers that are not concepts. A plan that realizes one is grammatical and
# about nothing ("gravity is related to two").
_NON_CONCEPT = frozenset("""
a an the and but or is are was were be been being have has had do does did
will would could should may might shall can not no nor so if then than too
very just about also into over after before between through during because
while which who whom what when where why how all each every both few more
most some any this that these those it its they them their we our you your
he she him her his i me my of to for with from at by as on in out up off
one two three four five six seven eight nine ten many much several various
certain other another such same different first second last next
""" .split())


def plan_is_committed(plan: Optional[SurfacePlan]) -> bool:
    """True when a plan asserts a real, contentful proposition.

    Rejects three measured failure shapes:

    1. a vacuous relation -- the ``relate`` fallback construction, which
       asserts proximity rather than a predicate;
    2. a non-concept FILLER -- a numeral or function word that makes the clause
       grammatical and empty (``gravity is related to two``);
    3. a TAUTOLOGY -- the target is the subject itself
       (``abstraction precedes a abstract``). This passed the first two checks
       because both words are graph-supported, so the grounding metric cannot
       see it; the plan route measured it as a live failure in
       ``docs/PROPOSITION_UNION.md``.

    Only the plan's FILLERS are checked, not its frame: ``is a`` / ``located
    in`` are the construction's own words and are always legitimate. Checking
    every token rejected correct plans like ``music is a sound`` because the
    frame contains ``a``.
    """
    if plan is None:
        return False
    if plan.construction == "relate":
        # 'relate' is the fallback construction. It is only committed when the
        # relation was genuinely typed as related_to AND the filler is a real
        # concept -- otherwise it is proximity with a frame around it.
        return False
    subj = (plan.subject or "").lower().strip()
    # The slot that legitimately holds the subject. `categorize` / `attribute`
    # / `locate` / `relate` bind it as `theme`; `cause` binds it as `agent`.
    # Any OTHER slot repeating the subject is a tautology.
    topic_roles = {"theme", "agent"}
    seen_topic = False
    for role, val in plan.fill.items():
        if not val or val.lower() in _NON_CONCEPT:
            return False
        if not subj:
            continue
        if val.lower().strip() != subj:
            continue
        if role in topic_roles and not seen_topic:
            seen_topic = True          # the legitimate subject slot
            continue
        return False                   # a second slot repeating the subject
    return True


@dataclass
class SurfacePlan:
    """An ordered, role-tagged surface plan the formulator can elaborate.

    ``tokens`` is the construction's surface frame with slots filled.
    ``roles`` is the parallel role sequence, so the caller can score whether
    the emitted utterance actually realized the committed message.
    ``construction`` names the construction used, for auditability.
    """

    subject: str
    construction: str
    tokens: List[str]
    roles: List[str]
    fill: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "subject": self.subject,
            "construction": self.construction,
            "tokens": list(self.tokens),
            "roles": list(self.roles),
            "fill": dict(self.fill),
        }


def message_to_plan(
    message: Optional[Message],
    *,
    subject: Optional[str] = None,
) -> Optional[SurfacePlan]:
    """Linearize a message into an ordered surface plan.

    Steps:

    1. Pick the construction from the STRONGEST binding's relation type. The
       relation semantics determine the surface form; the roles determine the
       slot fillers.
    2. Fill the construction's slots from the message's role bindings, in
       order. A construction may use the same role twice (``categorize`` has
       two ``theme`` slots): slot 0 takes the SUBJECT, slot 1 takes the
       strongest bound concept. That is what makes "X is a Y" come out right
       rather than "organism is a organism".
    3. The subject always leads as the utterance's topic, unless the
       construction's first slot is a different role (as in ``cause``, where
       the agent leads) -- in which case the subject is the agent and fills
       slot 0 anyway.

    Returns ``None`` when the message has no bindings or the subject cannot be
    placed, so the caller can fall back rather than emit a frame with no
    content.
    """
    if message is None or not message.bindings:
        return None

    subj = (subject or message.subject or "").strip()
    if not subj:
        return None

    bindings = list(message.bindings)
    primary = bindings[0]
    constr = _construction_for(primary.relation)
    if constr is None:
        return None

    # Fill slots positionally. The subject takes the FIRST slot unconditionally
    # (it is the topic); the remaining slots take the strongest bindings whose
    # role matches, skipping a role already used.
    fill: Dict[str, str] = {}
    roles: List[str] = []
    used_roles: set = set()
    slot_vals: List[str] = []

    for i, slot_role in enumerate(constr.slots):
        if i == 0:
            val = subj
        else:
            val = None
            # Prefer a binding whose role matches the slot.
            for b in bindings:
                if b.role == slot_role and b.role not in used_roles:
                    val = b.concept
                    used_roles.add(b.role)
                    break
            # Fall back to any unused binding. The message's roles are an
            # internal-argument structure; a relation the construction names
            # differently (macroplanning's 'relation' vs a frame's 'ground')
            # still has to be realized, or the clause comes out empty and the
            # whole plan is dropped for want of a synonym.
            if val is None:
                for b in bindings:
                    if b.concept.lower() == subj.lower():
                        continue
                    if b.role not in used_roles:
                        val = b.concept
                        used_roles.add(b.role)
                        break
            if val is None:
                # Genuinely nothing to fill the slot with: drop the whole
                # construction rather than emit a hole. The caller falls back.
                return None
        slot_vals.append(val)
        roles.append(slot_role)

    tokens: List[str] = []
    # Fill positionally: the pattern's placeholders consume slot values in
    # order, so a repeated role (categorize's two {theme} slots) takes the
    # subject first and the bound concept second.
    slot_iter = list(slot_vals)
    for tok in constr.pattern:
        if tok.startswith("{") and tok.endswith("}"):
            if slot_iter:
                tokens.append(slot_iter.pop(0))
            else:
                return None
        else:
            tokens.append(tok)

    if not any(t.lower() == subj.lower() for t in tokens):
        return None

    return SurfacePlan(
        subject=subj,
        construction=constr.name,
        tokens=tokens,
        roles=roles,
        fill={r: v for r, v in zip(roles, slot_vals)},
    )
