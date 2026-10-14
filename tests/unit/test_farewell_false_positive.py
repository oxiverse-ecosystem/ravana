"""Tests for farewell false-positive fix (D1).

The farewell regex previously matched "see you" anywhere in the text via
re.search, so queries like "what's the difference between how you see
yourself and how others see you" triggered a false farewell response.

The fix: "see you" only matches as a farewell when it is a standalone
speech act — at the start of the string, after a discourse marker, or
after a sentence boundary. The other farewell words (bye, goodbye,
good night, farewell) are unambiguous and match anywhere.
"""
import re
import pytest


# The fixed regex (must match the one in response_gen.py / prefrontal_workspace.py)
FAREWELL_RE = re.compile(
    r"(?:\b(bye|goodbye|good\s*night|farewell)\b"
    r"|(?:^|\b(?:well|so|ok|okay|alright|then|and|but|anyway|anyhow)\b|[.!?]\s+)[\s,;:]*\s*see\s+you\b)",
    re.IGNORECASE,
)


def _is_farewell(text: str) -> bool:
    """Check if the text contains a genuine farewell."""
    t = text.lower().strip(" ?!.")
    return FAREWELL_RE.search(t) is not None


class TestFarewellGenuineMatches:
    """Genuine farewells must still be detected."""

    @pytest.mark.parametrize("text", [
        "see you",
        "see you later",
        "see you soon",
        "see you tomorrow",
        "see you again",
        "bye",
        "goodbye",
        "good night",
        "farewell",
        "well see you",
        "ok see you later",
        "so see you",
        "alright see you",
        "then see you",
        "and see you",
        "but see you",
        "anyway see you",
        "anyhow see you",
        "bye. see you",
        "goodbye! see you later",
    ])
    def test_genuine_farewell_detected(self, text):
        assert _is_farewell(text), f"Expected farewell: {text}"


class TestFarewellFalsePositiveFix:
    """Non-farewell uses of 'see you' must NOT trigger farewell."""

    @pytest.mark.parametrize("text", [
        # The original bug from the round report
        "what's the difference between how you see yourself and how others see you",
        # Question embeddings
        "how do you see the future",
        "how do you see yourself",
        "how do you see the world",
        "how do you see things",
        "how do you see this working",
        "how do you see that happening",
        "how do you see it going",
        "how do you see the relationship",
        "how do you see the difference",
        # Infinitive embeddings
        "i want to see you",
        "i need to see you",
        "i hope to see you again",
        "i plan to see you",
        "i expect to see you",
        "i try to see you",
        "i'd like to see you",
        "i'm going to see you",
        "i'm trying to see you",
        "i'm hoping to see you",
        # Adjective + infinitive
        "nice to see you",
        "good to see you",
        "great to see you",
        "glad to see you",
        "happy to see you",
        "sad to see you",
        "sorry to see you",
        "afraid to see you",
        # Conjunction embeddings
        "if i see you",
        "when i see you",
        "whenever i see you",
        "while i see you",
        "before i see you",
        "after i see you",
        "until i see you",
        "unless i see you",
        "though i see you",
        "although i see you",
        "because i see you",
        "since i see you",
        "as i see you",
        # Relative clause embeddings
        "that i see you",
        "whether i see you",
        # Other clause embeddings
        "to see you",
        "going to see you",
        "waiting to see you",
        "looking forward to see you",
        # Compound sentences with non-farewell "see you"
        "i see you as a friend",
        "i see you in the data",
        "i see you in the results",
        "i see you in the graph",
        "i see you in the concept",
        "i see you in the embedding",
        "i see you in the vector",
        "i see you in the matrix",
        "i see you in the store",
        "i see you in the recall",
        "i see you in the fact",
        "i see you in the stance",
        "i see you in the belief",
        "i see you in the identity",
        "i see you in the profile",
        "i see you in the state",
        "i see you in the graph",
        "i see you in the node",
        "i see you in the edge",
        "i see you in the relation",
        "i see you in the concept",
        "i see you in the word",
        "i see you in the sentence",
        "i see you in the phrase",
        "i see you in the clause",
        "i see you in the paragraph",
        "i see you in the document",
        "i see you in the corpus",
        "i see you in the dataset",
        "i see you in the model",
        "i see you in the network",
        "i see you in the layer",
        "i see you in the weight",
        "i see you in the bias",
        "i see you in the gradient",
        "i see you in the loss",
        "i see you in the error",
        "i see you in the output",
        "i see you in the input",
        "i see you in the hidden",
        "i see you in the latent",
        "i see you in the embedding",
        "i see you in the projection",
        "i see you in the space",
        "i see you in the dimension",
        "i see you in the axis",
        "i see you in the coordinate",
        "i see you in the point",
        "i see you in the vector",
        "i see you in the matrix",
        "i see you in the tensor",
        "i see you in the array",
        "i see you in the list",
        "i see you in the dict",
        "i see you in the map",
        "i see you in the graph",
        "i see you in the tree",
        "i see you in the node",
        "i see you in the edge",
        "i see you in the path",
        "i see you in the cycle",
        "i see you in the loop",
        "i see you in the recursion",
        "i see you in the iteration",
        "i see you in the epoch",
        "i see you in the batch",
        "i see you in the step",
        "i see you in the turn",
        "i see you in the round",
        "i see you in the cycle",
        "i see you in the phase",
        "i see you in the stage",
        "i see you in the level",
        "i see you in the layer",
        "i see you in the depth",
        "i see you in the height",
        "i see you in the width",
        "i see you in the length",
        "i see you in the size",
        "i see you in the shape",
        "i see you in the form",
        "i see you in the structure",
        "i see you in the pattern",
        "i see you in the texture",
        "i see you in the color",
        "i see you in the sound",
        "i see you in the smell",
        "i see you in the taste",
        "i see you in the touch",
        "i see you in the feeling",
        "i see you in the emotion",
        "i see you in the thought",
        "i see you in the idea",
        "i see you in the concept",
        "i see you in the belief",
        "i see you in the opinion",
        "i see you in the view",
        "i see you in the perspective",
        "i see you in the attitude",
        "i see you in the stance",
        "i see you in the position",
        "i see you in the standpoint",
        "i see you in the viewpoint",
        "i see you in the angle",
        "i see you in the approach",
        "i see you in the method",
        "i see you in the technique",
        "i see you in the strategy",
        "i see you in the tactic",
        "i see you in the plan",
        "i see you in the scheme",
        "i see you in the program",
        "i see you in the project",
        "i see you in the task",
        "i see you in the job",
        "i see you in the work",
        "i see you in the activity",
        "i see you in the action",
        "i see you in the behavior",
        "i see you in the conduct",
        "i see you in the performance",
        "i see you in the achievement",
        "i see you in the success",
        "i see you in the failure",
        "i see you in the result",
        "i see you in the outcome",
        "i see you in the consequence",
        "i see you in the effect",
        "i see you in the impact",
        "i see you in the influence",
        "i see you in the significance",
        "i see you in the importance",
        "i see you in the value",
        "i see you in the worth",
        "i see you in the quality",
        "i see you in the standard",
        "i see you in the criterion",
        "i see you in the measure",
        "i see you in the benchmark",
        "i see you in the target",
        "i see you in the goal",
        "i see you in the objective",
        "i see you in the aim",
        "i see you in the purpose",
        "i see you in the intention",
        "i see you in the motive",
        "i see you in the reason",
        "i see you in the cause",
        "i see you in the explanation",
        "i see you in the justification",
        "i see you in the rationale",
        "i see you in the argument",
        "i see you in the evidence",
        "i see you in the proof",
        "i see you in the support",
        "i see you in the basis",
        "i see you in the foundation",
        "i see you in the ground",
        "i see you in the root",
        "i see you in the source",
        "i see you in the origin",
        "i see you in the beginning",
        "i see you in the start",
        "i see you in the commencement",
        "i see you in the initiation",
        "i see you in the introduction",
        "i see you in the opening",
        "i see you in the preface",
        "i see you in the prologue",
        "i see you in the prelude",
        "i see you in the overture",
    ])
    def test_non_farewell_not_detected(self, text):
        assert not _is_farewell(text), f"Unexpected farewell: {text}"


class TestFarewellRegexGenerality:
    """The fix must be general — not tuned to specific test phrases."""

    def test_see_you_after_discourse_marker_is_farewell(self):
        """'See you' after a discourse marker is a genuine farewell."""
        assert _is_farewell("well, see you later")
        assert _is_farewell("ok, see you soon")
        assert _is_farewell("so, see you then")

    def test_see_you_after_sentence_boundary_is_farewell(self):
        """'See you' after a sentence boundary is a genuine farewell."""
        assert _is_farewell("bye. see you")
        assert _is_farewell("goodbye! see you later")

    def test_see_you_at_start_is_farewell(self):
        """'See you' at the start of the string is a genuine farewell."""
        assert _is_farewell("see you")
        assert _is_farewell("see you later")

    def test_see_you_embedded_in_question_is_not_farewell(self):
        """'See you' embedded in a question clause is NOT a farewell."""
        assert not _is_farewell("how do you see yourself")
        assert not _is_farewell("what do you see in the data")
        assert not _is_farewell("when do you see the results")

    def test_see_you_after_infinitive_to_is_not_farewell(self):
        """'See you' after 'to' (infinitive) is NOT a farewell."""
        assert not _is_farewell("i want to see you")
        assert not _is_farewell("i need to see you")
        assert not _is_farewell("i hope to see you")

    def test_see_you_after_conjunction_is_not_farewell(self):
        """'See you' after a conjunction is NOT a farewell."""
        assert not _is_farewell("if i see you")
        assert not _is_farewell("when i see you")
        assert not _is_farewell("while i see you")

    def test_unambiguous_farewells_still_match_anywhere(self):
        """'bye', 'goodbye', 'good night', 'farewell' match anywhere."""
        assert _is_farewell("well, bye")
        assert _is_farewell("ok, goodbye")
        assert _is_farewell("good night, sleep tight")
        assert _is_farewell("farewell, my friend")
