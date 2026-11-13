"""Regression tests for the self-referential phrasing bug (Q4/Q11 residual).

Root cause was NOT chain_walker wiring a concept to itself in the traversal
sense. It was two concrete defects:
  1. Whole QUESTION phrases (e.g. "what causes the sun rise") were being minted
     as graph concept nodes and wired back to their own subject, producing
     "the sun rise is what causes the sun rise".
  2. A multi-word subject's own sub-token (e.g. "rise" in "sun rise") was
     returned by association spread and bound as the object, producing
     "the sun rise causes rise".

Both are now blocked: `_is_question_phrase` rejects question/sentence frames,
`_sanitize_graph` prunes legacy poison nodes + self-loops at load, and
`SurfaceRealizer.realize` refuses degenerate sub-token binds.
"""
import pytest

from ravana.chat.constants import _is_question_phrase


class TestIsQuestionPhrase:
    def test_question_leading_word(self):
        assert _is_question_phrase("what causes the sun rise")
        assert _is_question_phrase("why does trust matter")
        assert _is_question_phrase("how do people relate to life")

    def test_long_sentence_phrase(self):
        # >=5 words reads as a sentence, never a concept.
        assert _is_question_phrase("the cat sat on the warm windowsill")

    def test_clean_multiword_concept(self):
        # Genuine multi-word concepts must NOT be rejected.
        assert not _is_question_phrase("dark energy")
        assert not _is_question_phrase("quantum entanglement")
        assert not _is_question_phrase("sun rise")

    def test_single_word(self):
        assert not _is_question_phrase("trust")
        assert not _is_question_phrase("what")  # no space -> not a phrase

    def test_discourse_frames_are_rejected(self):
        """Template comparison/meta phrasing is not a concept.

        Measured in the persisted graph (2026-11-14): 72 phrases of >=4 words
        minted by the comparison/decompose path, e.g.
        'archived versus sea comparison'. A person cannot have a fact about
        'X versus Y comparison' -- the system generated the phrasing itself, so
        storing it as a concept is self-referential pollution.
        """
        assert _is_question_phrase("archived versus sea comparison")
        assert _is_question_phrase("philosophy versus archived comparison")
        assert _is_question_phrase("general usage edit comparison")
        assert _is_question_phrase("so far say confidence")

    def test_discourse_filter_does_not_reject_real_concepts(self):
        """The filter must not over-reach: genuine 4-word concepts pass."""
        for phrase in ("natural language processing",
                       "artificial intelligence research",
                       "climate change mitigation"):
            assert not _is_question_phrase(phrase), phrase


class TestSanitizeGraph:
    def test_prunes_question_phrase_nodes_and_self_loops(self):
        from ravana.chat.engine import CognitiveChatEngine
        eng = CognitiveChatEngine(dim=64, seed=1, baby_mode=True,
                                  data_dir=None)
        # Inject a poison question-phrase node.
        n = eng.graph.add_node(label="what causes the sun rise")
        # Wire it to a real concept so it has incident edges.
        real = eng.graph.add_node(label="sun rise")
        eng.graph.add_edge(n.id, real.id, weight=0.5)
        eng._concept_keywords["what causes the sun rise"] = [n.id]
        eng._concept_labels.add("what causes the sun rise")
        # Inject a genuine self-loop.
        selfloop = eng.graph.add_node(label="oxiverse")
        eng.graph.add_edge(selfloop.id, selfloop.id, weight=0.4)

        eng._sanitize_graph()

        assert "what causes the sun rise" not in eng._concept_keywords
        assert (selfloop.id, selfloop.id) not in eng.graph.edges
        # Real concept untouched.
        assert real.id in eng.graph.nodes
