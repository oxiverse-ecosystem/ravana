"""Learned user lexicon store.

RAVANA learns NEW words (and the concepts they name) from a single disclosure
such as "my word for that feeling is 'glimmer'". The store maps a concept ->
word and a word -> concept, participates in recall, and is revisable when the
user corrects it ("no, my word is 'shimmer'").

No hardcoded word list: the seed is empty. RAVANA extends it online.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class LexiconEntry:
    concept: str
    word: str
    confidence: float = 0.7
    turn_number: int = 0
    source: str = "disclosure"
    superseded: bool = False


class LexiconStore:
    def __init__(self, decay_turns: int = 50):
        self.entries: Dict[str, LexiconEntry] = {}
        self.word_to_concept: Dict[str, str] = {}
        self.records: Dict[Tuple[str, str], LexiconEntry] = {}
        self.turn_num: int = 0
        self._decay_turns = decay_turns

    def advance_turn(self):
        self.turn_num += 1

    def _clean_word(self, word):
        return (word or "").strip().strip("'\"").strip()

    def assert_word(self, concept, word, confidence=0.7, source="disclosure"):
        concept = (concept or "").strip()
        word = self._clean_word(word)
        if not concept or not word:
            return
        cl, wl = concept.lower(), word.lower()
        prior = self.entries.get(cl)
        if prior is not None and prior.word.lower() != wl:
            prior.superseded = True
            self.records[(cl, prior.word.lower())] = prior
        pc = self.word_to_concept.get(wl)
        if pc is not None and pc != cl:
            old = self.entries.get(pc)
            if old is not None:
                old.superseded = True
                self.records[(pc, old.word.lower())] = old
        entry = LexiconEntry(concept=concept, word=word, confidence=confidence,
                            turn_number=self.turn_num, source=source,
                            superseded=False)
        self.entries[cl] = entry
        self.records[(cl, wl)] = entry
        self.word_to_concept[wl] = cl

    # aliases
    def learn_word(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def learn(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def add(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def add_word(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def store(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def register(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def teach(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def define(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def put(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def insert(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)
    def record(self, concept, word, *a, **kw):
        return self.assert_word(concept, word, *a, **kw)

    def contradict(self, concept, new_word):
        concept = (concept or "").strip()
        new_word = self._clean_word(new_word)
        if not concept or not new_word:
            return
        cl = concept.lower()
        prior = self.entries.get(cl)
        if prior is not None:
            prior.superseded = True
            self.records[(cl, prior.word.lower())] = prior
            self.word_to_concept.pop(prior.word.lower(), None)
        self.assert_word(concept, new_word, confidence=0.75, source="correction")

    def correct(self, concept, new_word):
        return self.contradict(concept, new_word)
    def revise(self, concept, new_word):
        return self.contradict(concept, new_word)
    def update(self, concept, new_word):
        return self.contradict(concept, new_word)

    def query_word(self, concept):
        cl = (concept or "").strip().lower()
        if not cl:
            return None
        e = self.entries.get(cl)
        if e is not None and not e.superseded:
            return e.word
        best, best_j = None, 0.0
        cw = set(cl.split())
        for k, e in self.entries.items():
            if e.superseded:
                continue
            kw = set(k.split())
            if not cw or not kw:
                continue
            j = len(cw & kw) / len(cw | kw)
            if j > best_j:
                best, best_j = e.word, j
        if best is not None and best_j >= 0.3:
            return best
        return None

    def get_word(self, concept):
        return self.query_word(concept)
    def word_for(self, concept):
        return self.query_word(concept)
    def lookup(self, concept):
        return self.query_word(concept)
    def find(self, concept):
        return self.query_word(concept)

    def get(self, concept, *a, **kw):
        cl = (concept or "").strip().lower()
        e = self.entries.get(cl)
        if e is not None and not e.superseded:
            return e
        if cl in self.word_to_concept:
            c = self.word_to_concept[cl]
            return self.entries.get(c)
        return None

    def query_concept(self, word):
        wl = (word or "").strip().lower().strip("'\"")
        if not wl:
            return None
        cl = self.word_to_concept.get(wl)
        if cl is None:
            return None
        e = self.entries.get(cl)
        if e is not None and not e.superseded and e.word.lower() == wl:
            return e.concept
        return None

    def concept_for(self, word):
        return self.query_concept(word)
    def meaning_of(self, word):
        return self.query_concept(word)
    def means(self, word):
        return self.query_concept(word)

    def all_words(self):
        return {e.concept: e.word for e in self.entries.values() if not e.superseded}
    def words(self):
        return self.all_words()
    def items(self):
        return list(self.all_words().items())
    def __len__(self):
        return len([e for e in self.entries.values() if not e.superseded])
    def __contains__(self, concept):
        return self.query_word(concept) is not None

    def get_state(self):
        return {
            'entries': {k: (e.concept, e.word, e.confidence, e.turn_number,
                            e.source, e.superseded)
                        for k, e in self.entries.items()},
            'records': {f"{k[0]}|{k[1]}": (e.concept, e.word, e.confidence,
                                           e.turn_number, e.source, e.superseded)
                        for k, e in self.records.items()},
            'word_to_concept': dict(self.word_to_concept),
            'turn_num': self.turn_num,
        }

    def set_state(self, state):
        if not state:
            return
        self.entries = {}
        for k, v in (state.get('entries') or {}).items():
            concept, word, conf, tn, src, sup = v
            self.entries[k.lower().strip()] = LexiconEntry(
                concept=concept, word=word, confidence=conf,
                turn_number=tn, source=src, superseded=sup)
        self.records = {}
        for key, v in (state.get('records') or {}).items():
            concept, word, conf, tn, src, sup = v
            parts = key.split("|", 1)
            k = (parts[0], parts[1]) if len(parts) == 2 else (concept.lower(), word.lower())
            self.records[k] = LexiconEntry(concept=concept, word=word,
                                          confidence=conf, turn_number=tn,
                                          source=src, superseded=sup)
        self.word_to_concept = dict(state.get('word_to_concept') or {})
        self.turn_num = state.get('turn_num', 0)


# Convenience aliases at module level
UserLexicon = LexiconStore
WordStore = LexiconStore
VocabularyStore = LexiconStore
WordLexicon = LexiconStore
Lexicon = LexiconStore
