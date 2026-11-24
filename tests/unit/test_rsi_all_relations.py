"""Fake-based tests for provenanced writes of ALL RSI relation families.

No engine process, no ``process_turn``, no network, no persisted weights. The
write path is exercised with a tiny real `KnowledgeIntake` over an in-memory
`ConceptGraph`, plus recording fakes. Schema/answer assertions use the pure
helpers in schema_induction.py / response_gen.py. Env flags are restored.
"""
import os
import sys

import numpy as np
import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(PROJ, "ravana", "src"), PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat import response_gen as rg  # noqa: E402
from ravana.chat import schema_induction as si  # noqa: E402
from ravana.chat.chain_walker import ChainWalkerMixin  # noqa: E402
from ravana.learn.intake import KnowledgeIntake  # noqa: E402
from ravana_ml.graph import ConceptGraph  # noqa: E402

FAMILIES = {
    "CAUSE": ("gale causes rainbow.", "causal", None),
    "PART_OF": ("thunder is part of chill.", "part_of", "PART_OF"),
    "PRECEDES": ("cloud comes before glow.", "temporal", "PRECEDES"),
    "RESEMBLES": ("drizzle resembles chill.", "analogical", "RESEMBLES"),
}
QUERY = ("Considering the examples above, what is true of thaw? "
         "Options: sunset, typhoon, mist, hail. Answer briefly.")


@pytest.fixture(autouse=True)
def _flags_off():
    saved = {k: os.environ.get(k) for k in
             (si.ENV_FLAG, si.SCHEMA_ENV_FLAG, rg.EDGE_CHOICE_ENV)}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _GraphShim:
    """Minimal GraphEngine shape the real intake gate reads."""

    def __init__(self, walker):
        self.walker = walker

    @property
    def graph(self):
        return self.walker.graph

    @property
    def _all_labels(self):
        return {str(n.label).lower(): nid
                for nid, n in self.graph.nodes.items() if n.label}

    def get_outgoing(self, nid):
        return self.graph.get_outgoing(nid)

    def get_incoming(self, nid):
        return self.graph.get_incoming(nid)


class RecordingIntake:
    def __init__(self, real):
        self.real = real
        self.calls = []

    def intake(self, fact, provenance, salience=1.0):
        self.calls.append((fact, provenance, salience))
        return self.real.intake(fact, provenance, salience=salience)


class FakeWalker(ChainWalkerMixin):
    """ChainWalkerMixin with only the collaborators the write path needs."""

    def __init__(self, with_intake=True):
        self.graph = ConceptGraph(dim=4)
        self._concept_keywords = {}
        self._concept_labels = set()
        self._trace_enabled = False
        self.intake = None
        if with_intake:
            real = KnowledgeIntake(ge=_GraphShim(self), buffer=None)
            self.intake = RecordingIntake(real)
            self._get_knowledge_intake = lambda: self.intake
        else:
            self._get_knowledge_intake = lambda: None

    def _glove_vector(self, label):
        return np.zeros(4, dtype=float)

    @property
    def written_edges(self):
        return list(self.graph.edges.values())


def teach(walker, text):
    n = walker._consolidate_asserted_relations(text)
    return n, walker.written_edges


# ── 1. Every family writes a provenance-stamped trusted edge ─────────────────

@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_writes_a_trusted_edge(family):
    text, rel_type, meta_family = FAMILIES[family]
    walker = FakeWalker()
    n, edges = teach(walker, text)
    assert n >= 1
    assert len(edges) == 1
    edge = edges[0]
    md = edge.source_metadata
    assert md["source"] == "conversation"
    assert md["source_type"] == "user"
    assert md["edge_kind"] == "curated"
    assert md["extractor"] == "conversational-assertion"
    assert "taught-in-conversation" in md["operations"]
    if family != "CAUSE":   # CAUSE provenance is deliberately unchanged
        assert ("asserted-relation:" + family) in md["operations"]
    assert edge.relation_type == rel_type
    assert edge.target != edge.source
    # No item metadata / gold answer enters the assertion path.
    assert md.get("asserted_relation") == meta_family
    assert not any("gold" in k or "item" in k for k in md)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_intake_receives_user_provenance_and_clean_fact(family):
    text = FAMILIES[family][0]
    walker = FakeWalker()
    teach(walker, text)
    assert len(walker.intake.calls) == 1
    fact, prov, salience = walker.intake.calls[0]
    assert salience == 1.0
    assert prov.source == "conversation" and prov.source_type == "user"
    assert prov.extractor == "conversational-assertion"
    assert fact.source_sentence == text
    assert set(vars(fact)) <= {"subject", "relation", "obj", "confidence",
                               "source_sentence"}
    assert fact.subject and fact.obj and fact.subject != fact.obj


def test_inverse_phrases_swap_roles():
    walker = FakeWalker()
    teach(walker, "chill contains thunder.")
    edge = walker.written_edges[0]
    labels = {nid: n.label for nid, n in walker.graph.nodes.items()}
    assert labels[edge.source] == "thunder"
    assert labels[edge.target] == "chill"
    assert edge.source_metadata["asserted_relation"] == "PART_OF"


# ── 2. Existing CAUSE behaviour is unchanged ─────────────────────────────────

def test_cause_path_unchanged_uses_causes_relation_and_no_new_metadata():
    walker = FakeWalker()
    teach(walker, "gale causes rainbow.")
    fact, prov, _ = walker.intake.calls[0]
    assert fact.relation == "causes"          # unchanged legacy Fact relation
    assert fact.source_sentence == "gale causes rainbow."
    edge = walker.written_edges[0]
    assert edge.relation_type == "causal"
    assert "asserted_relation" not in edge.source_metadata
    # The CAUSE extractor still owns this sentence; the new parser ignores it.
    assert walker._asserted_relation_propositions("gale causes rainbow.") == []


def test_no_assertion_writes_nothing():
    walker = FakeWalker()
    n, edges = teach(walker, "hello there, how are you.")
    assert n == 0 and edges == []
    assert walker.intake.calls == []
    assert walker._asserted_relation_propositions(
        "Considering the examples above, what is true of thaw?") == []


# ── 3. Unknown / hedged / negated assertions admit no trusted edge ───────────

@pytest.mark.parametrize("text", [
    "thunder is not part of chill.",
    "thunder never is part of chill.",
    "cloud might come before glow.",
    "drizzle might resemble chill.",
    "gale probably causes rainbow.",             # hedge would mint probably->rainbow
    "gale supports rainbow.",                    # unknown family
    "thunder is part of chill?",                 # question, not an assertion
])
def test_unknown_hedged_negated_and_question_assertions_admit_nothing(text):
    walker = FakeWalker()
    n, edges = teach(walker, text)
    assert n == 0
    assert edges == []
    assert walker.intake.calls == []


# ── 4. Schema hypothesis forms from >=2 converged trusted demos per family ───

@pytest.mark.parametrize("family", ["PART_OF", "PRECEDES", "RESEMBLES", "CAUSE"])
def test_two_converged_demos_form_one_unassertable_hypothesis(family):
    text = FAMILIES[family][0]
    s, rest = text.split(" ", 1)                 # "gale causes rainbow."
    obj = rest.rsplit(" ", 1)[-1].rstrip(".")
    phrase = rest.rsplit(" ", 1)[0]
    demo2 = "shower %s climate." % phrase
    walker = FakeWalker()
    teach(walker, text)
    teach(walker, demo2)
    edges = rg.schema_trusted_edges_from_graph(walker.graph)
    traces = si.traces_from_edges(edges)
    assert len(traces) == 2
    hyps = si.induce_schema_hypotheses(traces)
    assert len(hyps) == 1
    assert hyps[0].support == 2
    assert hyps[0].assertable is False
    assert hyps[0].relation == FAMILIES[family][1]
    assert {s, "shower"} <= {pair[0] for pair in hyps[0].instances}
    assert obj in {pair[1] for pair in hyps[0].instances}


def test_one_demo_per_family_forms_no_hypothesis():
    walker = FakeWalker()
    teach(walker, "thunder is part of chill.")
    traces = si.traces_from_edges(rg.schema_trusted_edges_from_graph(walker.graph))
    assert len(traces) == 1
    assert si.induce_schema_hypotheses(traces) == []


def test_untrusted_edge_still_cannot_seed_hypothesis():
    edge = si.EdgeView("gale", "causal", "rainbow", "co_occurrence", "", 0.99)
    assert si.traces_from_edges([edge]) == []
    assert si.induce_schema_hypotheses(si.traces_from_edges([edge])) == []


# ── 5. Direct-edge deferral, foil / negative rejection preserved ─────────────

def test_non_causal_trusted_edge_is_not_direct_evidence():
    from ravana.chat.response_gen import CandidateEdge
    non_causal = CandidateEdge("mist", "thaw", "part_of", "curated", "conversation")
    causal = CandidateEdge("mist", "thaw", "causal", "curated", "conversation")
    assert rg.select_trusted_candidate("thaw", ["mist", "hail"], [non_causal]) is None
    assert rg.select_trusted_candidate("thaw", ["mist", "hail"], [causal]) == "mist"


def test_schema_hook_defers_to_directly_taught_edge():
    os.environ[si.SCHEMA_ENV_FLAG] = "1"
    os.environ[rg.EDGE_CHOICE_ENV] = "1"
    walker = FakeWalker()
    # Two converged CAUSE demos form a hypothesis; a third, direct edge into
    # the query subject must win before the schema ranker is consulted.
    teach(walker, "gale causes rainbow.")
    teach(walker, "shower causes climate.")
    teach(walker, "mist causes thaw.")
    fake = type("Self", (), {"graph": walker.graph,
                             "_concept_keywords": walker._concept_keywords,
                             "_glove_vector": walker._glove_vector})()
    assert rg.ResponseGenMixin._trusted_edge_candidate(fake, QUERY, "thaw") == "mist"
    assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") is None


def test_foil_candidate_below_congruence_floor_rejected_for_each_family():
    for family in ("PART_OF", "PRECEDES", "RESEMBLES"):
        text = FAMILIES[family][0]
        s, rest = text.split(" ", 1)
        obj = rest.rsplit(" ", 1)[-1].rstrip(".")
        phrase = rest.rsplit(" ", 1)[0]
        walker = FakeWalker()
        teach(walker, text)
        teach(walker, "shower %s climate." % phrase)
        traces = si.traces_from_edges(
            rg.schema_trusted_edges_from_graph(walker.graph))
        hyps = si.induce_schema_hypotheses(traces)
        assert len(hyps) == 1
        # A demonstrated anchor is a candidate but is NOT congruent with the
        # unseen subject; a naive surface matcher would wrongly accept it.
        assert si.rank_candidates_for_unseen_subject(
            "thaw", [s, "sunset"], hyps, traces,
            {s: 0.3, "sunset": 0.2},
            {(s, obj): 0.9, ("shower", "climate"): 0.9}) is None


def test_negative_demos_induce_no_hypothesis_and_hook_fails_closed():
    os.environ[si.SCHEMA_ENV_FLAG] = "1"
    for family in ("PART_OF", "PRECEDES", "RESEMBLES"):
        walker = FakeWalker()
        n, edges = teach(walker, "thunder is not part of chill.")
        if family != "PART_OF":
            n, edges = teach(walker, "cloud not comes before glow.")
        if family == "RESEMBLES":
            n, edges = teach(walker, "drizzle not resembles chill.")
        assert n == 0 and edges == []
        traces = si.traces_from_edges(
            rg.schema_trusted_edges_from_graph(walker.graph))
        assert si.induce_schema_hypotheses(traces) == []
        fake = type("Self", (), {"graph": walker.graph,
                                 "_concept_keywords": walker._concept_keywords,
                                 "_glove_vector": walker._glove_vector})()
        assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") is None


def test_flag_off_is_inert_for_all_families():
    assert si.schema_enabled() is False
    walker = FakeWalker()
    for family in FAMILIES:
        teach(walker, FAMILIES[family][0])
    fake = type("Self", (), {"graph": walker.graph,
                             "_concept_keywords": walker._concept_keywords,
                             "_glove_vector": walker._glove_vector})()
    assert rg.ResponseGenMixin._schema_candidate(fake, QUERY, "thaw") is None
