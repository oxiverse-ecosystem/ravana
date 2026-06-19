"""Tests for ravana_ml.tokenizer and word_tokenizer."""

import pytest
import numpy as np
from ravana_ml.tokenizer import (
    TokenizerInterface, BPETokenizer, SimpleTokenizer, WordTokenizer,
    PixelTokenizer, get_tokenizer,
)


class TestTokenizerInterface:
    def test_abstract_methods_raise(self):
        t = TokenizerInterface()
        with pytest.raises(NotImplementedError):
            t.encode("test")
        with pytest.raises(NotImplementedError):
            t.decode([1, 2])
        with pytest.raises(NotImplementedError):
            _ = t.vocab_size


class TestSimpleTokenizer:
    def test_init(self):
        t = SimpleTokenizer()
        assert t.vocab_size == 256

    def test_encode_ascii(self):
        t = SimpleTokenizer()
        ids = t.encode("ABC")
        assert ids == [65, 66, 67]

    def test_encode_unicode_fallback(self):
        t = SimpleTokenizer()
        ids = t.encode("\u2603")  # snowman not in ASCII
        assert ids == [63]  # '?'

    def test_decode_roundtrip(self):
        t = SimpleTokenizer()
        text = "Hello World!"
        ids = t.encode(text)
        assert t.decode(ids) == text

    def test_decode_unknown(self):
        t = SimpleTokenizer()
        assert t.decode([999]) == "?"

    def test_repr(self):
        t = SimpleTokenizer()
        r = repr(t)
        assert "SimpleTokenizer" in r
        assert "vocab_size=256" in r


class TestWordTokenizer:
    def test_init(self):
        t = WordTokenizer()
        assert t.vocab_size == 1  # min 1 (UNK token)

    def test_encode_creates_vocab(self):
        t = WordTokenizer()
        ids = t.encode("hello world")
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert t.vocab_size == 2

    def test_encode_roundtrip(self):
        t = WordTokenizer()
        text = "hello world"
        ids = t.encode(text)
        assert t.decode(ids) == text

    def test_encode_empty_string(self):
        t = WordTokenizer()
        ids = t.encode("")
        assert ids == [0]
        assert t.vocab_size >= 1

    def test_decode_empty(self):
        t = WordTokenizer()
        assert t.decode([]) == ""

    def test_decode_unknown(self):
        t = WordTokenizer()
        t.encode("hello")
        assert t.decode([999]) == "?"

    def test_repr(self):
        t = WordTokenizer()
        t.encode("test")
        assert "WordTokenizer" in repr(t)

    def test_get_tokenizer_word(self):
        t = get_tokenizer("word")
        assert isinstance(t, WordTokenizer)

    def test_get_tokenizer_simple(self):
        t = get_tokenizer("simple")
        assert isinstance(t, SimpleTokenizer)

    def test_get_tokenizer_bpe_fallback(self):
        t = get_tokenizer("bpe")
        assert isinstance(t, (BPETokenizer, WordTokenizer))


class TestBPETokenizer:
    def test_requires_tiktoken(self):
        try:
            t = BPETokenizer("gpt2")
            assert t.vocab_size > 0
            ids = t.encode("hello world")
            assert len(ids) > 0
            decoded = t.decode(ids)
            assert len(decoded) > 0
        except ImportError:
            pytest.skip("tiktoken not installed")


class TestPixelTokenizer:
    def test_init(self):
        pt = PixelTokenizer()
        assert pt.vocab_size == 266

    def test_encode_image_0to1(self):
        pt = PixelTokenizer()
        img = np.random.rand(28, 28).astype(np.float32)
        tokens = pt.encode_image(img)
        assert tokens.shape == (784,)
        assert tokens.min() >= 0
        assert tokens.max() <= 255

    def test_encode_image_0to255(self):
        pt = PixelTokenizer()
        img = np.random.randint(0, 256, (28, 28)).astype(np.float32)
        tokens = pt.encode_image(img)
        assert tokens.shape == (784,)

    def test_encode_label(self):
        pt = PixelTokenizer()
        for label in range(10):
            tid = pt.encode_label(label)
            assert 256 <= tid <= 265

    def test_encode_label_invalid(self):
        pt = PixelTokenizer()
        with pytest.raises(ValueError):
            pt.encode_label(-1)
        with pytest.raises(ValueError):
            pt.encode_label(10)

    def test_decode_label(self):
        pt = PixelTokenizer()
        for expected in range(10):
            tid = 256 + expected
            assert pt.decode_label(tid) == expected

    def test_decode_label_invalid(self):
        pt = PixelTokenizer()
        with pytest.raises(ValueError):
            pt.decode_label(0)
        with pytest.raises(ValueError):
            pt.decode_label(300)

    def test_repr(self):
        pt = PixelTokenizer()
        assert "PixelTokenizer" in repr(pt)


class TestWordTokenizerSeparate:
    """Tests for ravana_ml.word_tokenizer.WordTokenizer (separate module)."""

    def test_init(self):
        from ravana_ml.word_tokenizer import WordTokenizer
        t = WordTokenizer()
        assert t.vocab_size == 0

    def test_encode_lowercases(self):
        from ravana_ml.word_tokenizer import WordTokenizer
        t = WordTokenizer()
        ids = t.encode("Hello World")
        assert t.decode(ids) == "hello world"

    def test_encode_reuses_ids(self):
        from ravana_ml.word_tokenizer import WordTokenizer
        t = WordTokenizer()
        ids1 = t.encode("hello hello")
        assert ids1[0] == ids1[1]

    def test_decode_single_int(self):
        from ravana_ml.word_tokenizer import WordTokenizer
        t = WordTokenizer()
        ids = t.encode("hello")
        decoded = t.decode(ids[0])
        assert decoded == "hello"
