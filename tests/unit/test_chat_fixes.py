"""Regression tests for chat-interface fixes (2026-07-14 battery).

Covers:
- N-operand arithmetic (was capped at 3 operands -> "2+2+2+2" failed)
- ELI5 tail stripping in query grounding ("... like i am five" polluted subject)
- Empathy non-sequitur on ELI5 ("explain X like i am five" -> "that's awesome!")
- Query grounding: clause-connector split, discovery-verb stripping, hypothetical
  over-collapse (these caused web retrieval to silently fail for valid facts).
"""
import pytest


@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True)


@pytest.mark.parametrize("q,expected", [
    ("what is 2 + 2", 4),
    ("what is 2 + 2 + 2 + 2", 8),              # was UNVERIFIED before fix
    ("what is 1 + 2 + 3 + 4 + 5", 15),         # 5 operands
    ("what is 10 * 10 * 10", 1000),
    ("what is 25 times 17", 425),
    ("what is 100 / 4", 25),
    ("what is 2 - 2 - 2 - 2", -4),
])
def test_arithmetic_n_operands(engine, q, expected):
    out = engine._try_arithmetic(q)
    assert out is not None, f"arithmetic returned None for {q!r}"
    # answer is "<expr> = <result>."
    res = out.split("=")[-1].strip().rstrip(".")
    assert float(res) == expected, f"{q!r} -> {out!r}, expected {expected}"


def test_eli5_tail_stripped(engine):
    for q in [
        "explain quantum entanglement like i am five",
        "explain relativity like i'm five",
        "what is photosynthesis in simple terms",
        "describe gravity as if i were five",
    ]:
        subj, conf, method = engine._ground_query(q)
        assert subj, f"subject empty for {q!r}"
        assert "five" not in subj, f"ELI5 tail leaked into subject {subj!r} for {q!r}"
        assert "simple terms" not in subj, f"tail leaked: {subj!r}"


def test_empathy_not_fired_on_eli5(engine):
    # These contain first-person "i am" but are NOT affective disclosures.
    for q in [
        "explain quantum entanglement like i am five",
        "explain relativity like i am five",
    ]:
        disclosure = engine._detect_emotional_disclosure(text=q)
        assert disclosure is None, f"ELI5 {q!r} wrongly detected as disclosure: {disclosure}"


def test_empathy_fired_on_real_disclosure(engine):
    for q in ["i am feeling really sad today", "i hate you", "i am happy today", "i love pizza"]:
        disclosure = engine._detect_emotional_disclosure(text=q)
        assert disclosure is not None, f"real disclosure {q!r} missed by detector"
        assert disclosure[0] in ("negative", "positive", "neutral")


def test_grounding_clause_connector_split(engine):
    # "but"/"and" must not fuse two topics into one garbled subject.
    subj, conf, method = engine._ground_query("why is the sky blue but sunsets red")
    assert "sunsets" not in subj, f"clause connector fused topics: {subj!r}"
    assert subj.startswith("sky"), f"expected 'sky blue', got {subj!r}"


def test_grounding_discovery_verb_stripped(engine):
    # "who invented the telephone" -> 'telephone' (verb dropped), not 'invented telephone'
    subj, conf, method = engine._ground_query("who invented the telephone")
    assert "invented" not in subj, f"discovery verb leaked: {subj!r}"
    assert "telephone" in subj


def test_grounding_hypothetical_no_overcollapse(engine):
    # A noun phrase the PFC may mislabel 'hypothetical' must NOT collapse to its
    # last word and drop the head noun ("the speed of light" -> "light").
    subj, conf, method = engine._ground_query("the speed of light")
    assert "speed" in subj, f"hypothetical mislabel dropped head noun: {subj!r}"


# ── LingGen 30-Watt Proof-of-Concept: settle, don't sample ──────────────────
def test_attractor_memory_pattern_completion():
    """Phase 0: HRR store retrieves a definition trajectory from a concept cue."""
    from ravana.core.attractor_memory import AttractorMemory
    import numpy as np
    rng = np.random.RandomState(7)
    dim = 64
    defs = {
        "gravity": "gravity is a force that pulls objects toward each other",
        "music": "music is sound organized in time with rhythm and melody",
    }
    dw = {}
    for t in defs.values():
        for w in t.split():
            if w not in dw:
                v = rng.randn(dim)
                dw[w] = v / np.linalg.norm(v)
    am = AttractorMemory.from_definitions(
        defs, dw, lambda c: dw.get(c, rng.randn(dim)), dim=dim,
        stop={"is", "a", "that", "in", "with", "and"})
    assert len(am) == 2
    traj = am.retrieve(dw["gravity"])
    assert traj is not None and len(traj) >= 1
    assert any(w in ("gravity", "force") for w, _ in traj)


def test_settle_generator_over_decoder():
    """Phase 2: PredictiveCodingGenerator settles toward a trajectory."""
    from ravana_ml.nn.neural_decoder import NeuralDecoder
    from ravana.decoder.predictive_coding_generator import (
        PredictiveCodingGenerator)
    import numpy as np
    n_vocab = 64
    vocab = ["<bos>", "<eos>", "<unk>"] + [f"w{i}" for i in range(n_vocab - 3)]
    idx = {w: i for i, w in enumerate(vocab)}
    iw = {i: w for w, i in idx.items()}
    rng0 = np.random.RandomState(3)
    embed = {w: rng0.randn(64).astype(np.float32) for w in vocab}
    nd = NeuralDecoder(vocab_size=n_vocab, embed_dim=64, hidden_dim=32)
    for w, i in idx.items():
        nd.word_embedding.weight.data[i] = embed[w]
    traj = [(f"w{i}", embed[f"w{i}"]) for i in range(1, 5)]
    gen = PredictiveCodingGenerator(nd, idx_to_word=iw, settle_steps=6,
                                    min_steps=4, max_steps=12)
    cond = np.stack([embed["w1"], embed["w2"]], axis=0).astype(np.float32)
    toks = gen.generate(conditioning_embs=cond, trajectory=traj,
                        bos_idx=idx["<bos>"], eos_idx=idx["<eos>"])
    assert len(toks) >= 3
    assert all(iw.get(t) not in ("<bos>", "<eos>", "<unk>") for t in toks)


# ── Empathy / self-disclosure routing regressions (2026-07-19) ───────────────
# The empathy path was being hijacked by INTERROGATIVES and IMPERATIVES that
# merely contain a first-person token ("should i lie to protect a friend" ->
# positive empathy; "explain ... like i'm five" -> empathy/self-ack), and the
# self-disclosure ack was swallowing troubleshooting questions ("why does my
# code keep crashing" -> "got it — thanks for telling me"). Affective disclosure
# is a self-REPORT, not a question or command; the fail-closed guard now lives
# in the detector + the self-disclosure gate so every call site is consistent.

_INTERROGATIVE_IMPERATIVE = [
    "should i lie to protect a friend",
    "explain quantum computing like i'm five",
    "why does my code keep crashing",
    "how can i learn to cook",
    "what should i do when i feel overwhelmed",
    "can you help me understand gravity",
    "do i need to worry about that",
]


def test_empathy_not_fired_on_questions_or_commands(engine):
    # Interrogatives / imperatives must NOT be read as affective disclosures.
    for q in _INTERROGATIVE_IMPERATIVE:
        disclosure = engine._detect_emotional_disclosure(text=q)
        assert disclosure is None, (
            f"interrogative/imperative {q!r} wrongly detected as disclosure: "
            f"{disclosure}")


def test_self_disclosure_not_fired_on_questions_or_commands(engine):
    # Questions/commands that contain a first-person token ("i'm", "my") are
    # NOT self-disclosure STATEMENTS to store — they must fall through.
    for q in _INTERROGATIVE_IMPERATIVE:
        assert engine._is_self_disclosure_stmt(q) is False, (
            f"interrogative/imperative {q!r} wrongly treated as self-disclosure")


def test_self_disclosure_accepts_real_statement(engine):
    # Genuine first-person statements must STILL be captured (no regression).
    for q in ["my favorite color is blue", "i am called Likhith",
              "i love pizza", "my name is Sam"]:
        assert engine._is_self_disclosure_stmt(q) is True, (
            f"real self-disclosure statement {q!r} rejected")


def test_empathy_fired_on_cause_fallback_disclosure(engine):
    # "my mom is sick" has no strong affect WORD but a negative CAUSE; the
    # process_turn cause-fallback must still route it to empathy.
    out = engine.process_turn("my mom is sick")
    assert "sorry" in out.lower() or "hard" in out.lower(), (
        f"'my mom is sick' did not reach empathy: {out!r}")


def test_empathy_fired_on_real_disclosure_still_works(engine):
    # Regression guard: genuine disclosures keep working after the interrogative
    # guard was added.
    for q in ["i'm sad", "i am feeling really bored", "i love you"]:
        disclosure = engine._detect_emotional_disclosure(text=q)
        assert disclosure is not None, f"real disclosure {q!r} missed"


def test_clean_snippet_strips_dangling_doi(engine):
    # A truncated DOI / reference handle must not leak into a surfaced answer.
    for raw, bad in [
        ("according to an official source, doi: 10.1234/abc. def is a thing.",
         "doi"),
        ("doi: 10. The trust is a belief in reliability.", "doi"),
        ("see https://example.com/ref for details.", "https"),
    ]:
        cleaned = engine._clean_snippet(raw)
        assert "doi" not in cleaned.lower(), (
            f"doi residue leaked from {raw!r} -> {cleaned!r}")
        assert "https" not in cleaned.lower(), (
            f"url leaked from {raw!r} -> {cleaned!r}")


def test_sanitize_definition_text_strips_doi(engine):
    raw = "IntentForge is a search engine. <ref>doi:10.1000/xyz</ref> It indexes."
    san = engine._sanitize_definition_text(raw)
    assert san is not None
    assert "doi" not in san.lower(), f"doi leaked: {san!r}"

