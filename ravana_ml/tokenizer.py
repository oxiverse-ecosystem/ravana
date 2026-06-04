from typing import List, Union
import logging
import numpy as np

class TokenizerInterface:
    """Standard interface for tokenization in RAVANA."""
    
    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs."""
        raise NotImplementedError
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs into a string."""
        raise NotImplementedError
        
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        raise NotImplementedError


class BPETokenizer(TokenizerInterface):
    """BPE Tokenizer using the fast tiktoken library."""
    
    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self._encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        
    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)
        
    def decode(self, token_ids: List[int]) -> str:
        return self.encoding.decode(token_ids)
        
    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    def __repr__(self) -> str:
        return f"BPETokenizer(encoding={self._encoding_name}, vocab_size={self.vocab_size})"


class SimpleTokenizer(TokenizerInterface):
    """Fallback character-level tokenizer when tiktoken is not available."""
    
    def __init__(self):
        # Maps all standard ASCII characters to their byte values [0-255]
        self.char_to_id = {chr(i): i for i in range(256)}
        self.id_to_char = {i: chr(i) for i in range(256)}
        
    def encode(self, text: str) -> List[int]:
        # Fall back to 63 ('?') for out-of-bounds unicode characters
        return [self.char_to_id.get(c, 63) for c in text]
        
    def decode(self, token_ids: List[int]) -> str:
        return "".join(self.id_to_char.get(tid, "?") for tid in token_ids)
        
    @property
    def vocab_size(self) -> int:
        return 256

    def __repr__(self) -> str:
        return f"SimpleTokenizer(char_level, vocab_size={self.vocab_size})"


class WordTokenizer(TokenizerInterface):
    """Word-level tokenizer — splits on whitespace, one token per word.

    Much faster than char-level for cognitive experiments since each
    fact becomes ~5 tokens instead of ~28. Vocab is built dynamically
    from seen text.
    """

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self._next_id = 0

    def encode(self, text: str) -> List[int]:
        words = text.strip().split()
        if not words:
            return [0]
        ids = []
        for w in words:
            if w not in self.word_to_id:
                self.word_to_id[w] = self._next_id
                self.id_to_word[self._next_id] = w
                self._next_id += 1
            ids.append(self.word_to_id[w])
        return ids

    def decode(self, token_ids: List[int]) -> str:
        return " ".join(self.id_to_word.get(tid, "?") for tid in token_ids)

    @property
    def vocab_size(self) -> int:
        return max(1, self._next_id)

    def __repr__(self) -> str:
        return f"WordTokenizer(vocab_size={self.vocab_size})"


class PixelTokenizer:
    """Tokenizer for image data — maps pixel values to token IDs.

    Design (from reviewer response Concern 1):
    - Pixel values [0, 255] → token IDs [0, 255] (one per pixel)
    - Class labels [0, 9] → token IDs [256, 265]
    - vocab_size = 266

    Usage:
        pt = PixelTokenizer()
        tokens = pt.encode_image(image)      # 28x28 → 784 token IDs
        label_token = pt.encode_label(3)     # → 259
    """

    PIXEL_OFFSET = 0
    LABEL_OFFSET = 256
    N_CLASSES = 10

    def __init__(self):
        self._vocab_size = self.LABEL_OFFSET + self.N_CLASSES  # 266

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Flatten a 2D image and quantize pixel values to token IDs [0, 255].

        Args:
            image: 2D numpy array (e.g. 28x28) with pixel values in [0, 1] or [0, 255].

        Returns:
            1D array of token IDs in [0, 255].
        """
        flat = image.flatten()
        # Normalize to [0, 255] if values are in [0, 1]
        if flat.max() <= 1.0:
            flat = (flat * 255).astype(np.int64)
        else:
            flat = flat.astype(np.int64)
        return np.clip(flat, 0, 255)

    def encode_label(self, label: int) -> int:
        """Map a class label (0-9) to a token ID (256-265).

        Args:
            label: Integer class label in [0, 9].

        Returns:
            Token ID in [256, 265].

        Raises:
            ValueError: If label is outside [0, 9].
        """
        if not 0 <= label < self.N_CLASSES:
            raise ValueError(f"Label {label} out of range [0, {self.N_CLASSES - 1}]")
        return self.LABEL_OFFSET + label

    def decode_label(self, token_id: int) -> int:
        """Map a label token ID (256-265) back to a class label (0-9).

        Args:
            token_id: Token ID in [256, 265].

        Returns:
            Integer class label in [0, 9].

        Raises:
            ValueError: If token_id is not a label token.
        """
        if not self.LABEL_OFFSET <= token_id < self.LABEL_OFFSET + self.N_CLASSES:
            raise ValueError(f"Token {token_id} is not a label token (expected [{self.LABEL_OFFSET}, {self.LABEL_OFFSET + self.N_CLASSES - 1}])")
        return token_id - self.LABEL_OFFSET

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __repr__(self) -> str:
        return f"PixelTokenizer(vocab_size={self.vocab_size})"


def get_tokenizer(name: str = "word") -> TokenizerInterface:
    """Factory method to get the best available tokenizer.

    Default priority: WordTokenizer (5x faster for cognitive experiments),
    then BPE (if tiktoken available), then SimpleTokenizer (char-level fallback).

    Args:
        name: "word" (default), "bpe"/"gpt2", "simple", or a tiktoken encoding name.
    """
    if name == "word":
        return WordTokenizer()
    if name in ("bpe", "gpt2"):
        try:
            import tiktoken
            return BPETokenizer("gpt2")
        except ImportError:
            logging.warning("tiktoken not available, falling back to WordTokenizer.")
            return WordTokenizer()
    if name == "simple":
        return SimpleTokenizer()
    # Treat as tiktoken encoding name
    try:
        import tiktoken
        return BPETokenizer(name)
    except ImportError:
        logging.warning("tiktoken not available, falling back to WordTokenizer.")
        return WordTokenizer()
