"""Tests for ravana_ml.nn.module — Module, Linear, Embedding, LayerNorm, Dropout, Sequential, GRUCell."""
import pytest
import numpy as np
from ravana_ml.nn.module import (
    Module, Sequential, Linear, Embedding, LayerNorm, Dropout, GRUCell
)
from ravana_ml.tensor import StateTensor, Parameter


class TestModuleBase:
    """Test the base Module class."""

    def test_module_init(self):
        m = Module()
        assert m.training is True
        assert m._free_energy == 0.0

    def test_module_register_parameter(self):
        m = Module()
        p = Parameter(StateTensor(np.array([1.0, 2.0])))
        m.register_parameter("test_weight", p)
        assert "test_weight" in m._parameters
        assert hasattr(m, "test_weight")

    def test_module_register_module(self):
        parent = Module()
        child = Module()
        parent.register_module("child", child)
        assert "child" in parent._modules

    def test_parameters(self):
        m = Module()
        p = Parameter(StateTensor(np.array([1.0])))
        m.register_parameter("w", p)
        params = list(m.parameters())
        assert len(params) == 1
        assert params[0] is p

    def test_named_parameters(self):
        m = Module()
        p = Parameter(StateTensor(np.array([1.0])))
        m.register_parameter("w", p)
        named = list(m.named_parameters())
        assert len(named) == 1
        assert named[0][0] == "w"
        assert named[0][1] is p

    def test_train_eval(self):
        m = Module()
        assert m.training is True
        m.eval()
        assert m.training is False
        m.train()
        assert m.training is True

    def test_forward_not_implemented(self):
        m = Module()
        with pytest.raises(NotImplementedError):
            m()

    def test_repr(self):
        m = Module()
        r = repr(m)
        assert "Module" in r
        assert "free_energy" in r

    def test_accumulate_free_energy(self):
        m = Module()
        m.accumulate_free_energy(np.array([0.5, -0.3]))
        assert m._free_energy > 0

    def test_sleep_cycle_decay(self):
        m = Module()
        m._free_energy = 10.0
        m.sleep_cycle()
        assert m._free_energy < 10.0  # decays

    def test_reset_free_energy(self):
        m = Module()
        m._free_energy = 100.0
        m.reset_free_energy()
        assert m._free_energy == 0.0

    def test_state_dict(self):
        m = Module()
        p = Parameter(StateTensor(np.array([1.0, 2.0])))
        m.register_parameter("w", p)
        sd = m.state_dict()
        assert "w" in sd
        assert "data" in sd["w"]

    def test_load_state_dict(self):
        m = Module()
        p = Parameter(StateTensor(np.array([0.0, 0.0])))
        m.register_parameter("w", p)
        sd = {"w": {"data": np.array([3.0, 4.0])}}
        m.load_state_dict(sd)
        np.testing.assert_array_equal(m.w.data, [3.0, 4.0])

    def test_module_free_energy_propagation(self):
        parent = Module()
        child = Module()
        parent.register_module("child", child)
        parent.accumulate_free_energy(np.array([1.0]))
        assert child._free_energy > 0  # error propagates to children


class TestSequential:
    """Test the Sequential container."""

    def test_sequential_forward(self):
        seq = Sequential(
            Linear(2, 3),
            Linear(3, 1),
        )
        x = np.array([[1.0, 2.0]])
        out = seq(x)
        assert out.data.shape == (1, 1)

    def test_sequential_getitem(self):
        l1 = Linear(2, 3)
        l2 = Linear(3, 1)
        seq = Sequential(l1, l2)
        assert seq[0] is l1
        assert seq[1] is l2

    def test_sequential_len(self):
        seq = Sequential(Linear(2, 3), Linear(3, 1))
        assert len(seq) == 2

    def test_sequential_param_count(self):
        seq = Sequential(Linear(2, 3, bias=True), Linear(3, 1, bias=True))
        params = list(seq.parameters())
        # First layer: 3*2 + 3 = 9, second layer: 1*3 + 1 = 4, total = 13
        assert len(params) == 4  # 2 weight params + 2 bias params
        total_params = sum(p.data.size for p in params)
        assert total_params == 13


class TestLinearLayer:
    """Test the Linear module."""

    def test_linear_init(self):
        lin = Linear(5, 3)
        assert lin.in_features == 5
        assert lin.out_features == 3
        assert lin.weight.data.shape == (3, 5)
        assert lin.bias is not None
        assert lin.bias.data.shape == (3,)

    def test_linear_no_bias(self):
        lin = Linear(5, 3, bias=False)
        assert lin.bias is None

    def test_linear_forward(self):
        lin = Linear(3, 2)
        x = np.array([[1.0, 2.0, 3.0]])
        out = lin(x)
        assert out.data.shape == (1, 2)

    def test_linear_forward_tensor(self):
        lin = Linear(3, 2)
        x = StateTensor(np.array([[1.0, 2.0, 3.0]]))
        out = lin(x)
        assert isinstance(out, StateTensor)
        assert out.data.shape == (1, 2)

    def test_linear_forward_raw(self):
        lin = Linear(3, 2)
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = lin.forward_raw(x)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1, 2)

    def test_linear_parameters(self):
        lin = Linear(5, 3, bias=True)
        params = list(lin.parameters())
        assert len(params) == 2  # weight + bias

    def test_linear_no_bias_params(self):
        lin = Linear(5, 3, bias=False)
        params = list(lin.parameters())
        assert len(params) == 1  # weight only

    def test_linear_setattr_parameter(self):
        lin = Linear(3, 2)
        # The weight and bias are already registered via __setattr__ in __init__
        assert "weight" in lin._parameters

    def test_linear_repr(self):
        lin = Linear(5, 3)
        r = repr(lin)
        assert "Linear" in r
        assert "in=5" in r
        assert "out=3" in r


class TestEmbeddingLayer:
    """Test the Embedding module."""

    def test_embedding_init(self):
        emb = Embedding(10, 5)
        assert emb.num_embeddings == 10
        assert emb.embedding_dim == 5
        assert emb.weight.data.shape == (10, 5)

    def test_embedding_forward(self):
        emb = Embedding(10, 5)
        indices = np.array([0, 1, 2], dtype=np.int64)
        out = emb(indices)
        assert out.data.shape == (3, 5)

    def test_embedding_padding_idx(self):
        emb = Embedding(10, 5, padding_idx=0)
        # padding_idx should be zero
        assert np.all(emb.weight.data[0] == 0.0)

    def test_embedding_embed_raw(self):
        emb = Embedding(10, 5)
        vec = emb.embed_raw(3)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (5,)

    def test_embedding_embed_batch_raw(self):
        emb = Embedding(10, 5)
        vecs = emb.embed_batch_raw(np.array([0, 1, 2]))
        assert vecs.shape == (3, 5)

    def test_embedding_denies_grad(self):
        """Embedding should not raise during forward."""
        emb = Embedding(10, 5)
        emb(StateTensor(np.array([0])))
        emb.accumulate_free_energy(StateTensor(np.random.randn(1, 5) * 0.01))
        # No gradient explosion
        assert emb._weight_free_energy is not None

    def test_embedding_sleep_cycle_preserves_padding(self):
        emb = Embedding(10, 5, padding_idx=0)
        emb.weight.data[1] = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        emb._weight_free_energy.data[1] = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        emb.sleep_cycle()
        # padding_idx should remain zero
        assert np.all(emb.weight.data[0] == 0.0)
        # Non-padding should have changed
        assert not np.allclose(emb.weight.data[1], [10.0, 10.0, 10.0, 10.0, 10.0])


class TestLayerNormLayer:
    """Test the LayerNorm module."""

    def test_layernorm_init_affine(self):
        ln = LayerNorm(4)
        assert ln.normalized_shape == (4,)
        assert ln.elementwise_affine is True
        assert ln.weight is not None
        assert ln.bias is not None

    def test_layernorm_init_no_affine(self):
        ln = LayerNorm(4, elementwise_affine=False)
        assert ln.elementwise_affine is False
        assert ln._w_ln_raw is None
        assert ln._b_ln_raw is None

    def test_layernorm_forward(self):
        ln = LayerNorm(4)
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        out = ln(x)
        assert out.data.shape == (1, 4)
        # Mean should be ~0, std ~1
        assert abs(float(np.mean(out.data))) < 1e-4
        assert abs(float(np.std(out.data)) - 1.0) < 1e-3

    def test_layernorm_forward_raw(self):
        ln = LayerNorm(3, elementwise_affine=False)
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = ln.forward_raw(x)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1, 3)

    def test_layernorm_repr(self):
        ln = LayerNorm((4,))
        r = repr(ln)
        assert "LayerNorm" in r
        assert "eps" in r

    def test_layernorm_two_tuple_shape(self):
        ln = LayerNorm((3, 4))
        assert ln.normalized_shape == (3, 4)

    def test_layernorm_single_int(self):
        ln = LayerNorm(4)
        assert ln.normalized_shape == (4,)


class TestDropoutLayer:
    """Test the Dropout module."""

    def test_dropout_init(self):
        d = Dropout(0.3)
        assert d.p == 0.3

    def test_dropout_forward_state_tensor(self):
        d = Dropout(0.5)
        x = StateTensor(np.ones((1000,)))
        d.train()
        out = d.forward(x)
        assert isinstance(out, StateTensor)
        # Dropout with p=0.5 scales by 1/(1-p)=2.0, so surviving values are 2.0
        # Mean should be ~1.0 in expectation (some get 0, some get 2.0)
        assert float(np.mean(out.data)) <= 2.0
        # Some values should be zero (dropped)
        assert np.any(out.data == 0.0)

    def test_dropout_eval_no_drop(self):
        d = Dropout(0.5)
        x = StateTensor(np.ones((100,)))
        d.eval()
        out = d.forward(x)
        np.testing.assert_array_equal(out.data, np.ones(100))

    def test_dropout_repr(self):
        d = Dropout(0.3)
        r = repr(d)
        assert "Dropout" in r
        assert "p=0.3" in r

    def test_dropout_zero_p(self):
        d = Dropout(0.0)
        x = StateTensor(np.ones((100,)))
        out = d.forward(x)
        np.testing.assert_array_equal(out.data, np.ones(100))


class TestGRUCell:
    """Test the GRUCell module."""

    def test_grucell_init(self):
        gru = GRUCell(10, 20)
        assert gru.input_size == 10
        assert gru.hidden_size == 20
        assert hasattr(gru, "W_z")
        assert hasattr(gru, "W_r")
        assert hasattr(gru, "W_h")

    def test_grucell_forward(self):
        gru = GRUCell(5, 3)
        x = np.random.randn(5).astype(np.float32)
        h = np.zeros(3, dtype=np.float32)
        out = gru(x, h)
        assert out.shape == (3,)
        assert not np.allclose(out, 0)  # output should not be all zeros

    def test_grucell_forward_multiple_steps(self):
        gru = GRUCell(5, 3)
        x = np.random.randn(5).astype(np.float32)
        h = np.zeros(3, dtype=np.float32)
        # Step 1
        h = gru(x, h)
        h1 = h.copy()
        # Step 2 with same input
        h = gru(x, h)
        # Hidden state should evolve
        assert not np.allclose(h, h1)

    def test_grucell_persists_hidden(self):
        """Hidden state should carry forward information across steps."""
        gru = GRUCell(4, 4)
        # Step 1: compute hidden state from zero
        x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        h1 = gru(x, np.zeros(4, dtype=np.float32))
        
        # Step 2: same input but with h1 as previous state
        # This should produce a different output than starting from zero
        h2 = gru(x, h1.copy())
        
        # If hidden state carries forward, h1 and h2 should differ
        # because the reset gate will see different h_prev
        assert not np.allclose(h1, h2, atol=1e-6)

    def test_grucell_internal_tensors(self):
        """GRUCell should store intermediate activations for gradient computation."""
        gru = GRUCell(3, 3)
        x = np.ones(3, dtype=np.float32)
        h = np.zeros(3, dtype=np.float32)
        gru(x, h)
        assert hasattr(gru, "_last_combined")
        assert hasattr(gru, "_last_h_prev")
        assert hasattr(gru, "_last_x")

    def test_grucell_repr(self):
        gru = GRUCell(10, 20)
        r = repr(gru)
        assert "GRUCell" in r
        assert "10" in r
        assert "20" in r
