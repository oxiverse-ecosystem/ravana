"""Tests for ravana_ml support modules: embedder, tensor, tokenizer, relation_ontology, episode_injector."""

import pytest
import numpy as np
from ravana_ml.embedder import LearnedEmbedder
from ravana_ml.tensor import StateTensor, RawTensor, Parameter, tensor, zeros, ones
from ravana_ml.tokenizer import (
    TokenizerInterface, BPETokenizer, SimpleTokenizer, WordTokenizer,
    PixelTokenizer, get_tokenizer
)
from ravana_ml.relation_ontology import (
    get_sub_family, get_family, get_confidence, matches_config, TraversalConfig,
    SUB_FAMILIES, SUPER_FAMILIES, Candidate
)


# ─── Embedder Tests ───

class TestLearnedEmbedder:
    def test_init(self):
        e = LearnedEmbedder(dim=64)
        assert e.dim == 64

    def test_encode(self):
        e = LearnedEmbedder(dim=64)
        vec = e.encode("hello world")
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_encode_empty(self):
        e = LearnedEmbedder(dim=64)
        vec = e.encode("")
        assert vec.shape == (64,)

    def test_encode_ngrams(self):
        e = LearnedEmbedder(dim=64)
        ngrams = e._char_ngrams("test", ns=(3,))
        assert len(ngrams) == 2  # "tes", "est"
        assert "tes" in ngrams

    def test_encode_deterministic(self):
        e = LearnedEmbedder(dim=64)
        v1 = e.encode("test")
        v2 = e.encode("test")
        assert np.allclose(v1, v2)

    def test_encode_importance(self):
        e = LearnedEmbedder(dim=64)
        v1 = e.encode("test", importance=0.1)
        v2 = e.encode("test", importance=0.9)
        assert not np.allclose(v1, v2)


# ─── Tensor Tests ───

class TestRawTensor:
    def test_init(self):
        t = RawTensor([1, 2, 3])
        assert t.shape == (3,)

    def test_shape(self):
        t = RawTensor(np.zeros((3, 4)))
        assert t.shape == (3, 4)

    def test_dtype(self):
        t = RawTensor([1, 2], dtype=np.float32)
        assert t.dtype == np.float32

    def test_item(self):
        t = RawTensor([5])
        assert t.item() == 5

    def test_numpy(self):
        t = RawTensor([1, 2, 3])
        arr = t.numpy()
        assert isinstance(arr, np.ndarray)

    def test_add(self):
        a = RawTensor([1, 2])
        b = RawTensor([3, 4])
        c = a + b
        assert np.allclose(c.data, [4, 6])

    def test_sub(self):
        a = RawTensor([5, 6])
        b = RawTensor([1, 2])
        c = a - b
        assert np.allclose(c.data, [4, 4])

    def test_mul(self):
        a = RawTensor([2, 3])
        b = RawTensor([4, 5])
        c = a * b
        assert np.allclose(c.data, [8, 15])

    def test_matmul(self):
        a = RawTensor(np.eye(2))
        b = RawTensor(np.array([1, 2], dtype=np.float32))
        c = a @ b
        assert np.allclose(c.data, [1, 2])

    def test_view(self):
        t = RawTensor(np.zeros((4,)))
        v = t.view(2, 2)
        assert v.shape == (2, 2)

    def test_indexing(self):
        t = RawTensor([10, 20, 30])
        assert t[0].item() == 10

    def test_sum(self):
        t = RawTensor([1, 2, 3])
        s = t.sum()
        assert s.item() == 6

    def test_mean(self):
        t = RawTensor([1, 2, 3])
        m = t.mean()
        assert m.item() == 2.0

    def test_zeros_static(self):
        z = RawTensor.zeros(3, 4)
        assert z.shape == (3, 4)
        assert np.all(z.data == 0)

    def test_ones_static(self):
        o = RawTensor.ones(3)
        assert o.shape == (3,)
        assert np.all(o.data == 1)

    def test_stack(self):
        a = RawTensor([1, 2])
        b = RawTensor([3, 4])
        s = RawTensor.stack([a, b])
        assert s.shape == (2, 2)

    def test_clamp(self):
        t = RawTensor([-1, 5, 3])
        c = t.clamp(0, 4)
        assert np.all(c.data == [0, 4, 3])

    def test_abs(self):
        t = RawTensor([-5, 3])
        a = t.abs()
        assert np.all(a.data == [5, 3])


class TestStateTensor:
    def test_init(self):
        t = StateTensor([1, 2, 3], salience=0.7)
        assert t.salience == 0.7
        assert t.stability == 0.5  # default

    def test_free_energy(self):
        t = StateTensor([1, 2, 3])
        t.free_energy = 5.0
        assert t.free_energy == 5.0

    def test_plasticity(self):
        t = StateTensor([1, 2, 3], stability=0.3)
        assert t.plasticity == 0.7

    def test_decay(self):
        t = StateTensor(np.array([1.0, 2.0, 3.0]))
        data_before = t.data.copy()
        t.decay()
        # Decay with rate 0.01 over age() — should change data slightly
        assert np.any(t.data != data_before) or not np.allclose(t.data, data_before)

    def test_age(self):
        t = StateTensor([1, 2, 3])
        a = t.age()
        assert a >= 0

    def test_boost_salience(self):
        t = StateTensor([1, 2, 3], salience=0.3)
        t.boost_salience(0.2)
        assert t.salience == 0.5

    def test_apply_free_energy(self):
        t = StateTensor([1, 2, 3])
        fe = t.apply_free_energy(2.0, salience_weight=1.0)
        assert fe > 0

    def test_ops_return_statetensor(self):
        a = StateTensor([1, 2])
        b = StateTensor([3, 4])
        c = a + b
        assert isinstance(c, StateTensor)

    def test_repr(self):
        t = StateTensor([1, 2, 3])
        r = repr(t)
        assert "StateTensor" in r
        assert "salience" in r


class TestParameter:
    def test_init(self):
        t = StateTensor([1, 2, 3])
        p = Parameter(t)
        assert isinstance(p, StateTensor)
        assert "Parameter" in repr(p)


class TestTensorFunction:
    def test_tensor_from_list(self):
        t = tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)

    def test_tensor_from_statetensor(self):
        st = StateTensor([1, 2])
        t = tensor(st)
        assert isinstance(t, StateTensor)


# ─── Tokenizer Tests ───

class TestTokenizerInterface:
    def test_abstract_methods(self):
        t = TokenizerInterface()
        with pytest.raises(NotImplementedError):
            t.encode("test")
        with pytest.raises(NotImplementedError):
            t.decode([0])


class TestSimpleTokenizer:
    def test_encode(self):
        t = SimpleTokenizer()
        ids = t.encode("abc")
        assert ids == [97, 98, 99]

    def test_decode(self):
        t = SimpleTokenizer()
        text = t.decode([97, 98, 99])
        assert text == "abc"

    def test_vocab_size(self):
        t = SimpleTokenizer()
        assert t.vocab_size == 256

    def test_encode_out_of_range(self):
        t = SimpleTokenizer()
        ids = t.encode("\u00e9")  # é is beyond ASCII
        assert len(ids) == 1


class TestWordTokenizer:
    def test_encode(self):
        t = WordTokenizer()
        ids = t.encode("hello world")
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_decode(self):
        t = WordTokenizer()
        ids = t.encode("hello world")
        text = t.decode(ids)
        assert text == "hello world"

    def test_vocab_size(self):
        t = WordTokenizer()
        assert t.vocab_size == 1  # starts with empty vocab
        t.encode("hello world")
        assert t.vocab_size == 2  # hello, world

    def test_repr(self):
        t = WordTokenizer()
        assert "WordTokenizer" in repr(t)


class TestPixelTokenizer:
    def test_encode_image(self):
        pt = PixelTokenizer()
        img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        tokens = pt.encode_image(img)
        assert tokens.shape == (784,)

    def test_encode_image_normalized(self):
        pt = PixelTokenizer()
        img = np.random.rand(28, 28).astype(np.float32)
        tokens = pt.encode_image(img)
        assert tokens.shape == (784,)
        assert tokens.min() >= 0
        assert tokens.max() <= 255

    def test_encode_label(self):
        pt = PixelTokenizer()
        tid = pt.encode_label(3)
        assert tid == 259  # 256 + 3

    def test_encode_label_invalid(self):
        pt = PixelTokenizer()
        with pytest.raises(ValueError):
            pt.encode_label(10)

    def test_decode_label(self):
        pt = PixelTokenizer()
        label = pt.decode_label(259)
        assert label == 3

    def test_vocab_size(self):
        pt = PixelTokenizer()
        assert pt.vocab_size == 266


class TestGetTokenizer:
    def test_word(self):
        t = get_tokenizer("word")
        assert isinstance(t, WordTokenizer)

    def test_simple(self):
        t = get_tokenizer("simple")
        assert isinstance(t, SimpleTokenizer)


# ─── Relation Ontology Tests ───

class TestRelationOntology:
    def test_get_sub_family(self):
        sub = get_sub_family("causes")
        assert sub == "causal-strong"

    def test_get_sub_family_unknown(self):
        sub = get_sub_family("nonexistent")
        assert sub is None

    def test_get_family(self):
        fam = get_family("causes")
        assert fam == "causal"

    def test_get_confidence(self):
        conf = get_confidence("causes")
        assert 0.85 <= conf <= 0.95

    def test_get_confidence_unknown(self):
        conf = get_confidence("xyzzy")
        assert conf == 0.5

    def test_matches_config_predicate(self):
        config = TraversalConfig(mode="predicate", sub_family="causes")
        assert matches_config("causes", config)

    def test_matches_config_family(self):
        config = TraversalConfig(mode="family", family="causal")
        assert matches_config("causes", config)

    def test_matches_config_relaxed(self):
        config = TraversalConfig(mode="relaxed")
        assert matches_config("anything", config)

    def test_matches_config_super_family(self):
        config = TraversalConfig(mode="super_family", super_family="causal_all")
        assert matches_config("causes", config)

    def test_candidate_dataclass(self):
        c = Candidate(word="test", predicate="causes", family="causal", sub_family="causal-strong", depth=1, confidence=0.9)
        assert c.word == "test"

    def test_sub_families_structure(self):
        assert "causal-strong" in SUB_FAMILIES
        assert SUB_FAMILIES["causal-strong"]["family"] == "causal"

    def test_super_families(self):
        assert "causal_all" in SUPER_FAMILIES
