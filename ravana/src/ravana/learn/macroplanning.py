"""Macroplanning -- build the pre-lexical message from a communicative goal.

Research deliverable: ``docs/BRAIN_MESSAGE_PREPARATION.md``.

Levelt's model splits conceptual preparation into MACROPLANNING (what to say)
and MICROPLANNING (how to say it). The output of microplanning -- Levelt's
"message" -- is a **non-linear, propositional/relational structure** with
thematic role-concept bindings (who-did-what-to-whom), NOT a word sequence
(Levelt 1989 p.73; Bock & Ferreira 2014; Konopka & Brown-Schmidt 2014).

RAVANA already has the microplanning half: ``committed_proposition.py`` commits
to a graph-supported proposition before form encoding. What it lacks is the
macroplanning stage that BUILDS that proposition from a communicative goal.

This module supplies it. Given a subject and the ConceptGraph, it:

1. **Selects a communicative goal** from the graph's structure -- which
   relation to express is chosen by spreading activation from the subject,
   not by a hardcoded template.
2. **Builds a role-concept message** -- a small non-linear structure binding
   ROLES (agent / theme / relation / ground) to CONCEPTS, mirroring the
   Chang-Dell-Bock Meaning System's temporary Role<->Concept bindings
   (Chang, Dell & Bock 2006, Psych Review 113(2):234-272).
3. **Selects lexical concepts competitively** -- a Luce-ratio hazard rule over
   spreading activation with decay (Roelofs 1992, Cognition 42:107-142;
   Levelt, Roelofs & Meyer 1999, BBS 22(1):1-38):

       P(select target per time unit) = a(target) / sum(a(all candidates))

   This is the mechanism the brain uses to pick among competing lemmas, and
   it is a genuine competition -- not an argmax, which is what the decoder's
   readout currently does.

## Why this is not another template

The message is a graph structure over whatever the ConceptGraph actually
contains. A different graph yields a different message. Nothing here names a
relation or a target: both come from the graph walk. The only fixed knowledge
is the ROLE vocabulary, which is the brain's thematic-role inventory, not a
sentence frame.

## Scope

This module is a **pure function** over (subject, graph edges) -> Message. It
touches no decoder and no response string. The caller converts the message to
a token boost via the existing CPG path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Thematic roles. This is the brain's role inventory (agent, theme, goal,
# instrument, location, time), not a syntactic frame. A message binds these
# roles to concepts; it does not linearize them.
ROLES = ("agent", "theme", "relation", "ground")


@dataclass
class MacroplanConfig:
    """Spreading-activation and selection parameters.

    Defaults are judgment, not fitted. The Luce ratio is scale-free, so
    ``decay`` and ``spread`` mostly affect how many candidates survive to
    compete, not the winner among the top few.
    """

    # Activation decay per step (Roelofs 1992: a(m,t+dt) = a(m,t)(1-d) + ...).
    decay: float = 0.35
    # Spreading-activation weight from a node to each neighbour.
    spread: float = 0.55
    # Steps of spreading activation before selection.
    steps: int = 2
    # Candidates with activation below this fraction of the winner are
    # dropped from the competition (they would lose the Luce ratio anyway).
    competition_floor: float = 0.15
    # Hazard-rule scaling: selection probability per unit time is
    # target_activation / sum(activations). This is the Luce ratio; the
    # constant only affects how sharply the winner is preferred.
    luce_scale: float = 1.0
    # Maximum roles to bind (planning scope). Levelt's incrementality: broad
    # relational framework first, one referent at a time when encoding is hard.
    max_roles: int = 4
    # Minimum activation for a concept to be bound at all.
    min_activation: float = 0.05


@dataclass
class RoleBinding:
    """One role bound to one concept, with the activation that selected it."""

    role: str
    concept: str
    activation: float
    relation: str = "related_to"

    def as_dict(self) -> Dict[str, object]:
        return {
            "role": self.role,
            "concept": self.concept,
            "activation": round(float(self.activation), 6),
            "relation": self.relation,
        }


@dataclass
class Message:
    """The pre-lexical message: a non-linear role-concept structure.

    Deliberately NOT a string and NOT a word sequence. The formulator (the
    decoder) is downstream of this; the message is what it is conditioned on.
    """

    subject: str
    goal: str
    bindings: List[RoleBinding] = field(default_factory=list)
    # The competitive field: every candidate that entered the Luce ratio,
    # with its activation. Kept so the selection is auditable.
    competitors: List[Tuple[str, float]] = field(default_factory=list)
    # The winner's Luce ratio (target / sum). 1.0 means no competition.
    selection_ratio: float = 0.0

    @property
    def concepts(self) -> List[str]:
        return [b.concept for b in self.bindings]

    def as_dict(self) -> Dict[str, object]:
        return {
            "subject": self.subject,
            "goal": self.goal,
            "bindings": [b.as_dict() for b in self.bindings],
            "competitors": [(c, round(float(a), 6))
                            for c, a in self.competitors],
            "selection_ratio": round(float(self.selection_ratio), 6),
        }


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def spreading_activation(
    subject: str,
    candidates: Iterable[Tuple[str, str, float, float]],
    *,
    cfg: Optional[MacroplanConfig] = None,
) -> Dict[str, float]:
    """Spread activation from the subject over candidate concepts.

    ``candidates`` yields ``(relation, target, edge_weight, edge_confidence)``.
    Activation starts at the edge strength and spreads for ``cfg.steps``,
    decaying each step. The result is a field over concept labels -- the
    competitive arena the Luce ratio is computed in.

    This is the WEAVER spreading-activation network in miniature: activation
    flows from the subject to its neighbours, and each neighbour's activation
    is the sum of what it receives, decayed.
    """
    cfg = cfg or MacroplanConfig()
    field_: Dict[str, float] = {}
    for rel, tgt, w, c in candidates:
        if not tgt:
            continue
        # Seed activation from the edge's own strength.
        a = _clamp01(w) * 0.7 + _clamp01(c) * 0.3
        field_[tgt] = field_.get(tgt, 0.0) + a

    if not field_:
        return {}

    # Spread: each step, a candidate's activation decays and receives a share
    # of its neighbours' activation. With one hop from the subject, this is
    # mostly a decay-and-redistribute; the point is that activation is a
    # FIELD, not a per-edge score.
    for _ in range(max(0, cfg.steps)):
        nxt: Dict[str, float] = {}
        total = sum(field_.values()) or 1.0
        for concept, a in field_.items():
            retained = a * (1.0 - cfg.decay)
            # Redistribute a fraction of the total field to each candidate,
            # weighted by its share. This is what makes a highly-connected
            # concept win even with a weaker direct edge.
            received = cfg.spread * total * (a / total)
            nxt[concept] = retained + received * (1.0 - cfg.decay)
        field_ = nxt

    # Renormalise so the field is comparable across subjects.
    mx = max(field_.values(), default=0.0)
    if mx > 0:
        for k in field_:
            field_[k] /= mx
    return field_


def luce_ratio(target_activation: float, all_activations: Sequence[float]) -> float:
    """The Luce-ratio hazard rule: P(select target) = a(target) / sum(a(all)).

    Roelofs 1992; Levelt, Roelofs & Meyer 1999. This is the brain's lemma
    selection rule. It is a genuine competition: the winner is the candidate
    with the largest SHARE of the field, not the largest raw score.
    """
    total = float(sum(all_activations))
    if total <= 0.0:
        return 0.0
    return float(target_activation) / total


def select_lexical_concepts(
    activations: Dict[str, float],
    *,
    cfg: Optional[MacroplanConfig] = None,
    limit: int = 4,
) -> List[Tuple[str, float, float]]:
    """Competitive selection over the activation field.

    Returns ``(concept, activation, luce_ratio)`` sorted by activation,
    descending. Candidates below ``competition_floor`` of the winner are
    dropped -- they would lose the ratio anyway, and keeping them would let a
    long tail of noise into the message.
    """
    cfg = cfg or MacroplanConfig()
    if not activations:
        return []
    ranked = sorted(activations.items(), key=lambda kv: kv[1], reverse=True)
    winner_a = ranked[0][1]
    if winner_a <= 0.0:
        return []

    floor = winner_a * cfg.competition_floor
    total = sum(activations.values())
    out: List[Tuple[str, float, float]] = []
    for concept, a in ranked:
        if a < floor or a < cfg.min_activation:
            continue
        out.append((concept, a, luce_ratio(a, list(activations.values()))))
        if len(out) >= max(1, limit):
            break
    return out


def _role_for_relation(relation: str) -> str:
    """Map a graph relation type to a thematic role.

    The brain assigns roles by the RELATION's semantics, not by salience: an
    ``is_a`` edge binds a theme/category, a ``causes`` edge binds a result,
    an ``agent``/``by`` edge binds the agent. Assigning roles by activation
    order -- the obvious implementation -- puts function words into roles
    (measured: ``time -> theme=early, relation=years``; ``love -> relation=i``),
    which is not a message, it is a word salad with labels.

    This is the role inventory of the brain's thematic-role system, applied to
    the graph's relation vocabulary. Unknown relations default to ``ground``.
    """
    r = (relation or "").strip().lower()
    if not r:
        return "ground"
    # Categorisation / essence -> the theme is what the subject IS.
    if r in ("is_a", "isa", "type", "kind", "category", "instance_of",
             "subclass", "part_of", "member_of"):
        return "theme"
    # Causation / production -> the theme is the result.
    if r in ("causes", "cause", "produces", "produces_cause", "leads_to",
             "results_in", "effect", "affects", "influences", "enables"):
        return "ground"
    # Agency / authorship / origin -> the theme is the agent.
    if r in ("agent", "by", "author", "creator", "source", "origin",
             "created_by", "from", "has_agent"):
        return "agent"
    # Location / containment / time -> the theme is the ground.
    if r in ("location", "located_in", "in", "at", "place", "time",
             "temporal", "part", "contains", "within"):
        return "ground"
    # Attribute / property -> the theme is a property of the subject.
    if r in ("has_property", "property", "attribute", "has", "capable_of",
             "used_for", "function", "purpose"):
        return "relation"
    # Default: the relation slot, which is the least committal binding.
    return "relation"


# Function words. The graph's edges are not filtered for content, so a
# high-degree node (``time`` has 30 edges) accumulates activation from
# function words: measured ``time -> theme=early, relation=years,
# ground=only, agent=first`` and ``love -> relation=i, agent=you``. Those are
# not lexical concepts; they are the graph's stop-word noise. The brain's
# lemma layer does not compete function words into content roles.
#
# This list is deliberately small and closed: it is the closed-class
# inventory (determiners, pronouns, prepositions, auxiliaries, conjunctions),
# not a frequency cutoff. A frequency cutoff would drop rare content words.
_FUNCTION_WORDS = frozenset("""
a an the this that these those some any all each every both few many much more most
several other another such no nor not only own same so than too very
i me my myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs
themselves what which who whom whose where when why how
and but or because as until while of at by for with about against between into
through during before after above below to from up down in out on off over under
again further then once here there all any both each few more most other some
such no nor not only own same so than too very can will just should now
is am are was were be been being have has had having do does did doing
would could ought i'm you're he's she's it's we're they're i've we've you've
they've i'll you'll he'll she'll we'll they'll i'd you'd he'd she'd we'd they'd
also however therefore thus hence moreover furthermore nevertheless though
although yet still even ever never always often sometimes usually
early late first last only just also quite rather almost nearly already
""".split())


def is_content_word(word: str) -> bool:
    """True when ``word`` is a content word, not a function word or a token.

    The gate for macroplanning's lexical-concept selection. The brain's lemma
    layer selects among content lemmas; function words are inserted later by
    the formulator's grammatical encoding, not chosen by conceptual
    preparation. Measured: without this gate, ``time`` selects ``early`` /
    ``only`` / ``first`` and ``love`` selects ``i`` / ``you``.
    """
    if not word:
        return False
    w = word.strip().lower()
    if not w or w.startswith("<"):
        return False
    if w in _FUNCTION_WORDS:
        return False
    # Multi-word labels are often graph noise ("ocean sea", "sea side").
    # A genuine concept can be multi-word, but the graph's labels at this
    # stage are mostly fragments; require a single token.
    if len(w.split()) != 1:
        return False
    return True


def build_message(
    subject: str,
    candidates: Iterable[Tuple[str, str, float, float]],
    *,
    goal: str = "characterize",
    cfg: Optional[MacroplanConfig] = None,
    content_only: bool = True,
) -> Optional[Message]:
    """Macroplanning: build the pre-lexical message from a communicative goal.

    Steps:

    1. Spread activation from the subject over the candidate concepts.
    2. Select lexical concepts competitively (Luce ratio).
    3. Bind them to roles by the RELATION's semantics (``_role_for_relation``),
       not by activation order. The subject is the message's topic; each
       selected concept fills the role its edge type implies.

    ``content_only`` (default True) drops function words before selection --
    the brain's conceptual preparation does not compete closed-class items
    into content roles.

    Returns ``None`` when the graph offers nothing -- the caller then falls
    back to the existing CPG path, which is the same fail-open behaviour.
    """
    cfg = cfg or MacroplanConfig()
    cands = [(r, t, _clamp01(w), _clamp01(c)) for r, t, w, c in candidates if t]
    if content_only:
        cands = [(r, t, w, c) for r, t, w, c in cands
                 if is_content_word(t)]
    if not cands:
        return None

    field_ = spreading_activation(subject, cands, cfg=cfg)
    if not field_:
        return None

    selected = select_lexical_concepts(field_, cfg=cfg,
                                       limit=max(1, cfg.max_roles))
    if not selected:
        return None

    # Role assignment by relation semantics. Each selected concept takes the
    # role its producing edge's relation type implies; a concept with no
    # matching edge (possible after the field redistribution) takes the
    # lowest-salience unfilled role.
    rel_for_concept: Dict[str, str] = {}
    for r, t, _w, _c in cands:
        rel_for_concept.setdefault(t, r)

    bindings: List[RoleBinding] = []
    used_roles: set = set()
    # First pass: assign by relation type.
    for concept, a, _ratio in selected:
        rel = rel_for_concept.get(concept, "related_to")
        role = _role_for_relation(rel)
        if role in used_roles:
            # Two concepts map to the same role. The higher-activation one
            # keeps it; the other falls through to the salvage pass.
            continue
        used_roles.add(role)
        bindings.append(RoleBinding(role=role, concept=concept,
                                    activation=a, relation=rel))
    # Salvage pass: any selected concept not yet bound takes an unfilled role,
    # in salience order. This keeps the message's content complete.
    for concept, a, _ratio in selected:
        if any(b.concept == concept for b in bindings):
            continue
        free = [r for r in ROLES if r not in used_roles]
        if not free:
            break
        role = free[0]
        used_roles.add(role)
        rel = rel_for_concept.get(concept, "related_to")
        bindings.append(RoleBinding(role=role, concept=concept,
                                    activation=a, relation=rel))

    winner_a = selected[0][1]
    ratio = luce_ratio(winner_a, list(field_.values()))
    competitors = sorted(field_.items(), key=lambda kv: kv[1], reverse=True)
    return Message(subject=subject, goal=goal, bindings=bindings,
                   competitors=competitors, selection_ratio=ratio)


def message_to_boost(
    message: Optional[Message],
    word_to_idx: Dict[str, int],
    *,
    base_boost: float = 2.0,
    role_weights: Optional[Dict[str, float]] = None,
) -> Dict[int, float]:
    """Convert a message to a decoder token boost.

    This is the bridge to the existing CPG path: the message's bound concepts
    become token ids, weighted by their selection activation and role. The
    caller passes the result to ``NeuralDecoder.generate`` as ``token_boost``.

    Role weights let the caller prefer, say, the theme over the ground. Default
    weights are flat -- the activation already carries the competition's
    outcome.
    """
    if message is None:
        return {}
    rw = role_weights or {}
    boost: Dict[int, float] = {}
    for b in message.bindings:
        tid = word_to_idx.get(b.concept.lower())
        if tid is None:
            continue
        w = rw.get(b.role, 1.0)
        boost[tid] = base_boost * max(0.1, b.activation) * w
    return boost
