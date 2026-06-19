"""Tests for PixelTokenizer — image tokenization for MNIST and similar tasks."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ravana_ml.tokenizer import PixelTokenizer


class TestPixelTokenizer:
    """Test suite for PixelTokenizer."""

    def setup_method(self):
        self.pt = PixelTokenizer()

    def test_vocab_size(self):
        """vocab_size should be 266 (256 pixel bins + 10 class tokens)."""
        assert self.pt.vocab_size == 266

    def test_encode_image_shape(self):
        """encode_image should flatten 28x28 to 784 tokens."""
        image = np.random.rand(28, 28).astype(np.float32)
        tokens = self.pt.encode_image(image)
        assert tokens.shape == (784,)

    def test_encode_image_range_normalized(self):
        """For [0, 1] input, token IDs should be in [0, 255]."""
        image = np.random.rand(28, 28).astype(np.float32)
        tokens = self.pt.encode_image(image)
        assert tokens.min() >= 0
        assert tokens.max() <= 255

    def test_encode_image_range_raw(self):
        """For [0, 255] input, token IDs should be in [0, 255]."""
        image = np.random.randint(0, 256, (28, 28)).astype(np.uint8)
        tokens = self.pt.encode_image(image)
        assert tokens.min() >= 0
        assert tokens.max() <= 255

    def test_encode_image_all_zeros(self):
        """All-black image should produce all 0 tokens."""
        image = np.zeros((28, 28), dtype=np.float32)
        tokens = self.pt.encode_image(image)
        assert np.all(tokens == 0)

    def test_encode_image_all_ones(self):
        """All-white image (1.0) should produce all 255 tokens."""
        image = np.ones((28, 28), dtype=np.float32)
        tokens = self.pt.encode_image(image)
        assert np.all(tokens == 255)

    def test_encode_label_valid(self):
        """Labels 0-9 should map to token IDs 256-265."""
        for label in range(10):
            token = self.pt.encode_label(label)
            assert token == 256 + label

    def test_encode_label_out_of_range(self):
        """Labels outside [0, 9] should raise ValueError."""
        with pytest.raises(ValueError):
            self.pt.encode_label(10)
        with pytest.raises(ValueError):
            self.pt.encode_label(-1)

    def test_decode_label_valid(self):
        """Token IDs 256-265 should decode back to labels 0-9."""
        for token_id in range(256, 266):
            label = self.pt.decode_label(token_id)
            assert label == token_id - 256

    def test_decode_label_out_of_range(self):
        """Non-label tokens should raise ValueError."""
        with pytest.raises(ValueError):
            self.pt.decode_label(0)
        with pytest.raises(ValueError):
            self.pt.decode_label(255)
        with pytest.raises(ValueError):
            self.pt.decode_label(266)

    def test_encode_decode_label_roundtrip(self):
        """encode_label → decode_label should be identity."""
        for label in range(10):
            assert self.pt.decode_label(self.pt.encode_label(label)) == label

    def test_no_overlap_pixel_and_label_tokens(self):
        """Pixel tokens [0, 255] and label tokens [256, 265] should not overlap."""
        pixel_token = 255
        label_token = self.pt.encode_label(0)
        assert label_token > pixel_token

    def test_repr(self):
        """repr should show vocab_size."""
        assert "266" in repr(self.pt)
        assert "PixelTokenizer" in repr(self.pt)
