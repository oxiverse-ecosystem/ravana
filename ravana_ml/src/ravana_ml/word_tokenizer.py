"""Word-level tokenizer for RLMv2.

Splits text into words and maps each word to a unique token ID.
Enables RLMv2 to create concept nodes for WORDS, not characters.
"""
from typing import List, Dict


class WordTokenizer:
    """Word-level tokenizer.
    
    Maps each unique word to a sequential token ID starting at 0.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self._next_id = 0
    
    def _get_or_add_word(self, word: str) -> int:
        if word not in self.word_to_id:
            self.word_to_id[word] = self._next_id
            self.id_to_word[self._next_id] = word
            self._next_id += 1
        return self.word_to_id[word]
    
    def encode(self, text: str) -> List[int]:
        """Split text into words and return word-level token IDs."""
        words = text.lower().strip().split()
        return [self._get_or_add_word(w) for w in words]
    
    def decode(self, ids) -> str:
        """Decode token IDs back to text."""
        if isinstance(ids, (int,)):
            ids = [ids]
        result = []
        for tid in ids:
            if tid in self.id_to_word:
                result.append(self.id_to_word[tid])
            else:
                result.append('?')
        return ' '.join(result)
    
    @property
    def vocab_size(self) -> int:
        return self._next_id
