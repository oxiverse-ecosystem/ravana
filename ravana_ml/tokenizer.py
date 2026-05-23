from typing import List
import logging

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


def get_tokenizer(name: str = "gpt2") -> TokenizerInterface:
    """Factory method to get the best available tokenizer."""
    try:
        import tiktoken
        return BPETokenizer(name)
    except ImportError:
        logging.warning("tiktoken library not found. Falling back to char-level SimpleTokenizer.")
        return SimpleTokenizer()
