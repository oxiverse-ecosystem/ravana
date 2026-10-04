"""Round 2026-10-03T2353Z — RAVANA must not assert a relation about a subject it
has never represented.

MEASURED defect (D4): asked "who is marguerite" for an entity RAVANA has never
encountered, RAVANA answered:

    "marguerite is like ravana — they sit close in the same part of the map"

Trace of how that sentence is built (`engine_self_query._try_analogical_reasoning`):
internal knowledge MISSES, GloVe finds the nearest graph label, and then — when
there is NO edge between the subject and that concept — the code INHERITS the
nearest concept's strongest outgoing edge type and renders a relation frame from
it.

MEASURED on a live engine:
    subj='marguerite'    in_graph=False  nearest='ravana'  cos=0.333  -> asserted
    subj='ravana'        in_graph=True   nearest='ravana'  cos=1.000
    subj='exist'         in_graph=False  nearest='things'  cos=0.605

The subject has no node and no edge, so there is NO structural evidence linking it
to anything. Inheriting a stranger's edge type manufactures the evidence. That is
the source-monitoring failure the repo already guards against elsewhere
(Johnson 1993; Modirrousta & Fellows 2008 — the `_decomp_grounded` reality-
monitoring gate in response_gen.py withholds exactly this class of claim).

Brain basis: a genuinely stored concept carries its own relations. Proximity in a
distributional embedding is a WEAK cue about association, never evidence about a
relation's TYPE. Humans do not conclude "a quibbleworth is a kind of quibbleworth
because those words sit near each other".

The fix must be a STRUCTURAL precondition (does the store actually LINK these two
concepts?), never a phrase list, a topic list, or a tuned cosine threshold.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO_ROOT,
           os.path.join(REPO_ROOT, "ravana", "src"),
           os.path.join(REPO_ROOT, "ravana_ml", "src"),
           os.path.join(REPO_ROOT, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat import engine_self_query as esq


# Distinct unit vectors give a DETERMINISTIC nearest-neighbour ordering without
# faking the cosine, so the test exercises the real arithmetic in
# _try_analogical_reasoning rather than a patched-out dot product.
_VECTORS = {
    "water": (1.0, 0.0, 0.0),
    "river": (0.9, 0.0, 0.0),
    "ravana": (0.3, 0.0, 0.0),
    "marguerite": (0.8, 0.0, 0.0),
    "quibbleworth": (0.8, 0.0, 0.0),
    "zorblatt": (0.8, 0.0, 0.0),
}
_DEFAULT_VEC = (0.8, 0.0, 0.0)


class _Node:
    def __init__(self, label):
        self.label = label


class _Edge:
    def __init__(self, relation_type, weight=0.5):
        self.relation_type = relation_type
        self.weight = weight


class _Graph:
    """Graph with a DIRECT edge river->water and an unrelated edge ravana->river.

    Every node label is unique, so label->node-id resolution cannot land on a
    decoy node. (An earlier revision of this harness added a second node per
    label, which silently kept the branch under test from being reached at all.)
    """

    def __init__(self):
        self.nodes = {
            "n_water": _Node("water"),
            "n_river": _Node("river"),
            "n_ravana": _Node("ravana"),
        }
        self.edges = {
            # the ONLY edge that touches 'water', so a subject with no node of
            # its own would inherit 'causes' and fabricate a relation
            ("n_river", "n_water"): _Edge("causes", 0.9),
            ("n_water", "n_ravana"): _Edge("part_of", 0.6),
        }

    def get_outgoing(self, nid):
        return [(tgt, e) for (s, tgt), e in self.edges.items() if s == nid]


class _Engine:
    """Engine double with a real GloVe-shaped vector per word."""

    def __init__(self):
        self.graph = _Graph()
        self._trace_enabled = False

    def _glove_vector(self, word):
        return list(_VECTORS.get((word or "").lower().strip(), _DEFAULT_VEC))


def _reply_for(subject):
    eng = _Engine()
    # All vectors are unit length, so the cosine is a plain dot product.
    orig_norm = esq.np.linalg.norm
    esq.np.linalg.norm = lambda v: 1.0
    try:
        return esq.SelfQueryMixin._try_analogical_reasoning(
            eng, subject, f"what is {subject}")
    finally:
        esq.np.linalg.norm = orig_norm


def _is_frame_shaped(reply):
    """Does the reply ASSERT a relation frame ("X is like Y — ...")?

    A reply that declines, hedges, or says it lacks a clean line is exactly what
    we want. Only a confident relational assertion is the defect.
    """
    if not reply:
        return False
    low = reply.lower()
    return (" is like " in low) or (" is akin to " in low) or low.startswith("from what i understand")


# --- the defect: an entity RAVANA has never represented ----------------------

@pytest.mark.parametrize("subject", ["marguerite", "quibbleworth", "zorblatt"])
def test_unrepresented_subject_is_not_given_a_relation_frame(subject):
    """No link in the store => no structural basis for asserting a relation.

    These subjects have no node, so the nearest label's edge type belongs to a
    DIFFERENT concept and carries no evidence about them.
    """
    reply = _reply_for(subject)
    assert not _is_frame_shaped(reply), (
        f"asserted a relation for unrepresented subject {subject!r}: {reply!r}")


def test_unrepresented_subject_may_still_say_it_is_unsure():
    """Declining or hedging is CORRECT here — the honest fallback is the goal."""
    reply = _reply_for("marguerite")
    assert reply is None or isinstance(reply, str)
    # If a string comes back it must not claim knowledge it lacks.
    if reply:
        low = reply.lower()
        assert ("don't have" in low or "not certain" in low or "not totally sure" in low
                or "no clean line" in low or "outside what i know" in low
                or "think longer" in low), \
            f"unexpected confident framing for an unknown entity: {reply!r}"


# --- the capability must survive: a LINKED subject still reasons -------------

def test_linked_subject_still_gets_its_real_relation():
    """Deleting the fabrication must not delete analogical reasoning.

    'river' HAS a direct edge to 'water' in the store, so the honest structural
    answer still exists and must still be produced from THAT edge.
    """
    reply = _reply_for("river")
    assert reply, "a subject with a real edge to its nearest concept must answer"
    assert "river" in reply.lower() and "water" in reply.lower()
    assert _is_frame_shaped(reply), \
        f"the real edge should still render its frame, got {reply!r}"


def test_gate_is_a_store_link_check_not_a_similarity_threshold():
    """The discriminator is 'does the store link them', not 'how near are they'.

    'zorblatt' (cos 0.8 to water) is refused while 'river' (cos 0.9 to the same
    water) is answered — close in embedding space, opposite in outcome, so no
    cosine cut-off can explain the behaviour.
    """
    unknown = _reply_for("zorblatt")          # nearest water (0.8) - no edge
    linked = _reply_for("river")              # nearest water (0.9) - real edge
    assert not _is_frame_shaped(unknown), \
        f"unrepresented subject asserted: {unknown!r}"
    assert _is_frame_shaped(linked), \
        f"linked subject lost its frame: {linked!r}"