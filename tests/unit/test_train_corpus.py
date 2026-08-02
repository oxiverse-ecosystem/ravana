"""Hermetic unit test for scripts/train.py load_corpus() multi-corpus wiring.

Covers the change that lets the training pipeline consume additional corpora
in data/corpora/ (e.g. tiny_shakespeare.txt) alongside the primary seed corpus.

No glove, no network, no full engine — a minimal stub engine is used so this
stays in the fast CI path.
"""
import os
import sys
import types

import pytest


# ── Minimal stub engine (faithful to the attrs load_corpus touches) ──
class _StubND:
    def __init__(self):
        self.prepare_calls = 0

    def prepare_sentences(self, text, embed, idx, min_sentence_len=3):
        # One synthetic "sentence" per whitespace token — deterministic and
        # lets the test assert that BOTH corpora contributed.
        self.prepare_calls += 1
        toks = text.split()
        return [{"words": [t], "word_indices": [0], "conditioning_embs": None}
                for t in toks]


class _StubEngine:
    def __init__(self):
        self.neural_decoder = _StubND()
        self._decoder_word_to_idx = {}
        self._decoder_word_to_embed = {}  # load_corpus passes this to prepare_sentences
        self._freeze_decoder_vocab = False

    def _expand_decoder_vocab(self, new_words):
        for w in new_words:
            if w not in self._decoder_word_to_idx:
                self._decoder_word_to_idx[w] = len(self._decoder_word_to_idx)


def _make_train_module(tmp_path, monkeypatch):
    """Import scripts.train with a writable data/corpora dir override.

    load_corpus resolves corpora under _proj_root/data/corpora. We point
    _proj_root at tmp_path so the test never touches the real data folder.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import scripts.train as train_mod
    monkeypatch.setattr(train_mod, "_proj_root", str(tmp_path))
    corpora = tmp_path / "data" / "corpora"
    corpora.mkdir(parents=True, exist_ok=True)
    (corpora / "teen_seeds.txt").write_text("trust love time memory", encoding="utf-8")
    return train_mod, corpora


def test_load_corpus_primary_only(tmp_path, monkeypatch):
    train_mod, corpora = _make_train_module(tmp_path, monkeypatch)
    eng = _StubEngine()
    text, sents, nd = train_mod.load_corpus(eng)
    assert nd.prepare_calls == 1
    assert len(sents) == 4  # 4 words in teen_seeds
    # teen_seeds words entered the vocab
    assert {"trust", "love", "time", "memory"} <= set(eng._decoder_word_to_idx)
    assert eng._freeze_decoder_vocab is True


def test_load_corpus_with_extra_corpus(tmp_path, monkeypatch):
    train_mod, corpora = _make_train_module(tmp_path, monkeypatch)
    extra = corpora / "tiny_shakespeare.txt"
    extra.write_text("hamlet tragedy kingdom ghost", encoding="utf-8")
    eng = _StubEngine()
    text, sents, nd = train_mod.load_corpus(eng, extra_corpora=[str(extra)])
    # Both corpora prepared (primary + extra)
    assert nd.prepare_calls == 2
    # 4 (primary) + 4 (extra) synthetic sentences
    assert len(sents) == 8
    # Extra-corpus words also expanded the vocab
    assert "hamlet" in eng._decoder_word_to_idx
    assert eng._freeze_decoder_vocab is True


def test_load_corpus_skips_missing_extra(tmp_path, monkeypatch):
    train_mod, corpora = _make_train_module(tmp_path, monkeypatch)
    eng = _StubEngine()
    text, sents, nd = train_mod.load_corpus(
        eng, extra_corpora=[str(corpora / "does_not_exist.txt")])
    # Missing extra must not crash; only the primary corpus is used.
    assert nd.prepare_calls == 1
    assert len(sents) == 4
