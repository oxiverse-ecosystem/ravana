"""Tests for ravana_ml.nn.functional — activation functions, losses, utilities."""

import pytest
import numpy as np
from ravana_ml.tensor import StateTensor, RawTensor
from ravana_ml.nn import functional as F


class TestActivations:
    """Test activation functions: relu, sigmoid, tanh, gelu, silu, softmax, log_softmax."""

    def test_relu_positive(self):
        result = F.relu(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, StateTensor)
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])

    def test_relu_negative(self):
        result = F.relu(np.array([-1.0, -2.0, 0.0]))
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 0.0])

    def test_relu_mixed(self):
        result = F.relu(np.array([-0.5, 0.0, 0.5, 1.0]))
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 0.5, 1.0])

    def test_relu_with_tensor(self):
        t = StateTensor(np.array([-3.0, 0.0, 3.0]))
        result = F.relu(t)
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 3.0])

    def test_sigmoid_midpoint(self):
        result = F.sigmoid(np.array([0.0]))
        assert abs(float(result.data[0]) - 0.5) < 0.01

    def test_sigmoid_saturation(self):
        result = F.sigmoid(np.array([-100.0, 100.0]))
        assert float(result.data[0]) < 1e-5
        assert abs(float(result.data[1]) - 1.0) < 0.01

    def test_sigmoid_range(self):
        x = np.linspace(-5, 5, 100)
        result = F.sigmoid(x)
        assert np.all(result.data >= 0.0)
        assert np.all(result.data <= 1.0)

    def test_tanh_range(self):
        result = F.tanh(np.array([-0.5, 0.0, 0.5]))
        assert np.all(np.abs(result.data) <= 1.0)

    def test_tanh_symmetry(self):
        x = np.array([1.0, 2.0, 3.0])
        pos = F.tanh(x)
        neg = F.tanh(-x)
        np.testing.assert_array_almost_equal(pos.data, -neg.data)

    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = F.softmax(x, dim=-1)
        assert abs(float(np.sum(result.data)) - 1.0) < 1e-5

    def test_softmax_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = F.softmax(x, dim=-1)
        row_sums = np.sum(result.data, axis=-1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])

    def test_log_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = F.log_softmax(x, dim=-1)
        # log_softmax should be <= 0 and monotonic with input
        assert np.all(result.data <= 0)
        assert result.data[2] > result.data[1] > result.data[0]

    def test_gelu_shape(self):
        x = np.random.randn(5, 10)
        result = F.gelu(x)
        assert result.data.shape == (5, 10)

    def test_gelu_approx_zero(self):
        result = F.gelu(np.array([0.0]))
        assert abs(float(result.data[0])) < 0.5

    def test_silu(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = F.silu(x)
        assert result.data.shape == (5,)
        # silu(x) = x * sigmoid(x) should be between 0 for positive and slightly negative
        assert result.data[0] < 0  # negative input -> slightly negative output
        assert result.data[2] == 0  # zero input -> zero output
        assert result.data[4] > 0  # positive input -> positive output


class TestDropout:
    """Test dropout behavior in train vs eval mode."""

    def test_dropout_eval_no_op(self):
        x = np.ones((100, 100))
        result = F.dropout(x, p=0.5, training=False)
        np.testing.assert_array_equal(result.data, x)

    def test_dropout_zero_p(self):
        x = np.ones((100,))
        result = F.dropout(x, p=0.0, training=True)
        np.testing.assert_array_equal(result.data, x)

    def test_dropout_training(self):
        x = np.ones((1000,))
        result = F.dropout(x, p=0.5, training=True)
        # ~50% should be dropped (become 0)
        zero_frac = np.mean(result.data == 0)
        assert 0.35 < zero_frac < 0.65, f"zero_frac={zero_frac}"

    def test_dropout_expected_value(self):
        """In training mode, E[dropout(x)] = x, so mean should approximate original."""
        x = np.ones((10000,)) * 2.0
        result = F.dropout(x, p=0.5, training=True)
        # Mean should be ~2.0 in expectation (2.0 * (1-p) / (1-p) = 2.0)
        assert abs(float(np.mean(result.data)) - 2.0) < 0.2


class TestLinear:
    """Test linear transformation."""

    def test_basic_linear(self):
        x = np.array([[1.0, 2.0]])
        w = np.array([[0.5, 0.5], [1.0, 0.0]])
        result = F.linear(x, w)
        assert result.data.shape == (1, 2)

    def test_linear_with_bias(self):
        x = np.array([[1.0, 1.0]])
        w = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        result = F.linear(x, w, bias=b)
        assert abs(float(result.data[0, 0]) - 3.0) < 1e-5

    def test_linear_with_tensors(self):
        x = StateTensor(np.array([[2.0, 3.0]]))
        w = StateTensor(np.array([[0.5, 0.5]]))
        result = F.linear(x, w)
        assert abs(float(result.data[0, 0]) - 2.5) < 1e-5


class TestEmbedding:
    """Test embedding lookup."""

    def test_basic_embedding(self):
        indices = np.array([0, 1, 2])
        weight = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        result = F.embedding(indices, weight)
        assert result.data.shape == (3, 2)
        np.testing.assert_array_equal(result.data[0], [1.0, 0.0])
        np.testing.assert_array_equal(result.data[1], [0.0, 1.0])

    def test_embedding_with_tensor(self):
        indices = StateTensor(np.array([0]))
        weight = StateTensor(np.array([[10.0, 20.0]]))
        result = F.embedding(indices, weight)
        np.testing.assert_array_equal(result.data[0], [10.0, 20.0])


class TestLayerNorm:
    """Test layer normalization."""

    def test_basic_layernorm(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = F.layer_norm(x, normalized_shape=(3,))
        assert result.data.shape == (2, 3)
        # Each row should have mean ~0 and std ~1
        for row in result.data:
            assert abs(float(np.mean(row))) < 1e-5
            assert abs(float(np.std(row)) - 1.0) < 1e-4

    def test_layernorm_with_affine(self):
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        w = np.array([2.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        result = F.layer_norm(x, normalized_shape=(2,), weight=w, bias=b)
        assert result.data.shape == (1, 2)
        # After normalization the row is ~[-1, 1], then *2 + 1 = ~[-1, 3]
        assert abs(float(result.data[0, 0]) + 1.0) < 0.5
        assert abs(float(result.data[0, 1]) - 3.0) < 0.5


class TestLosses:
    """Test loss functions."""

    def test_bce(self):
        input = np.array([0.5, 0.8, 0.2])
        target = np.array([1.0, 1.0, 0.0])
        loss = F.binary_cross_entropy(input, target)
        assert float(loss.data) > 0

    def test_bce_perfect_prediction(self):
        input = np.array([1.0, 0.0])
        target = np.array([1.0, 0.0])
        loss = F.binary_cross_entropy(input, target)
        assert float(loss.data) < 0.01  # near zero loss

    def test_mse(self):
        input = np.array([1.0, 2.0, 3.0])
        target = np.array([2.0, 3.0, 4.0])
        loss = F.mse_loss(input, target)
        assert abs(float(loss.data) - 1.0) < 1e-5  # MSE of [1,1,1] is 1.0

    def test_mse_zero(self):
        input = np.array([5.0, 5.0])
        target = np.array([5.0, 5.0])
        loss = F.mse_loss(input, target)
        assert float(loss.data) < 1e-10

    def test_cross_entropy(self):
        input = np.array([[1.0, 0.0, 0.0]])
        target = np.array([0])
        loss = F.cross_entropy(input, target)
        # Should be low since the correct class has highest logit
        assert float(loss.data) < 1.0

    def test_cross_entropy_wrong(self):
        input = np.array([[1.0, 3.0, 2.0]])
        target = np.array([0])
        loss = F.cross_entropy(input, target)
        # Should be higher since class 0 is not the highest
        assert float(loss.data) > 0.5


class TestUtilities:
    """Test utility functions: one_hot, pad, cosine_similarity."""

    def test_one_hot(self):
        result = F.one_hot(np.array([0, 1, 2]), num_classes=3)
        np.testing.assert_array_equal(result.data, np.eye(3))

    def test_one_hot_out_of_range_ignored(self):
        # one_hot with indices beyond num_classes creates extra rows in identity
        with pytest.raises(Exception):
            F.one_hot(np.array([5]), num_classes=3)

    def test_pad_constant(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = F.pad(x, ((1, 1), (1, 1)), mode='constant', value=0.0)
        assert result.data.shape == (4, 4)
        assert result.data[0, 0] == 0.0
        assert result.data[1, 1] == 1.0

    def test_cosine_similarity(self):
        x1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        x2 = np.array([[1.0, 0.0], [1.0, 0.0]])
        result = F.cosine_similarity(x1, x2, dim=-1)
        # First row: same direction -> cos ~ 1
        assert abs(float(result.data[0, 0]) - 1.0) < 1e-5
        # Second row: orthogonal -> cos ~ 0
        assert abs(float(result.data[1, 0])) < 1e-5

    def test_cosine_similarity_opposite(self):
        x1 = np.array([[1.0, 0.0]])
        x2 = np.array([[-1.0, 0.0]])
        result = F.cosine_similarity(x1, x2, dim=-1)
        assert abs(float(result.data[0, 0]) + 1.0) < 1e-5
