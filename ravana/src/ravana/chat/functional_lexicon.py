"""Consolidated closed-class functional lexicon (Stage 5b-ii, de-hardcoding plan).

Collapses the duplicated inline functional lexicons that were scattered
across ``engine.py`` (`_generic` at 7729, `_FRAMING` at 10665, `_bare_moral` at
7661, `_INC/_DEC/_REM` at 7614-7618) into ONE data-driven source of truth.

These are *functional* (closed-class) primitives — quantity/negation/polarity
markers, moral-question cues, and generic question framing words. Per the
code's own comments they are legitimate to keep as curated seeds (the brain
tracks closed-class modification words for quantity/causal reasoning), but
they must live in a single ``data/*.json`` file, not three+ inline copies, so
there is one authority and they can be EER-fit / decayed centrally.

Mirrors the ``SnippetPEConfig`` / ``SaladClassifier`` pattern: ``load()`` fails
open to seed constants when the fit file is absent, so removing the inline
copies never regresses behavior.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Set

# This file lives at <repo>/ravana/src/ravana/chat/functional_lexicon.py.
# To reach <repo>/data go up 5 levels: chat -> ravana -> src -> ravana -> <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_FIT_PATH = os.path.join(_DATA_DIR, "functional_lexicon.json")

# Seed = the consolidated current inline sets (verified, day-one identical).
# Duplicates across _generic / _FRAMING / _bare_moral are unioned here.
_SEED: Dict[str, list] = {
    # Quantity / polarity primitives (closed-class modifiers).
    "polarity_increase": ["doubl", "tripl", "increase", "more", "stronger",
                          "higher", "grow", "add", "extra", "boost", "amplif",
                          "intensif"],
    "polarity_decrease": ["halv", "less", "weaker", "lower", "reduc", "shrink",
                          "diminish"],
    "polarity_remove": ["without", "gone", "disappear", "vanish", "remov",
                        "lost", "absent", "cease", "eliminat", "no longer",
                        "none"],
    # Moral / advice question cues.
    "moral_markers": ["okay", "ok", "moral", "should", "ethical", "fair",
                      "promise", "lie", "ever"],
    "moral_ambiguous": ["right", "wrong"],
    # Generic question-framing words (signal a malformed multi-word subject).
    "framing": ["ever", "okay", "ok", "break", "make", "really", "right",
               "wrong", "thing", "things", "actually", "question", "answer"],
    # Trailing temporal/adverbial modifiers that pollute topic keys when they
    # survive to the word list.  Stripped from the END of a phrase before the
    # topic-key truncation so "cooking earlier" -> "cooking" and
    # "raise a kid to care about the ocean" keeps "ocean" (the real topic).
    # A seed list — grows at runtime via data/functional_lexicon.json.
    # Leading salutation/age/status modifiers that never head a real concept.
    # A relationship disclosure can carry them before the relationship word
    # ("my OLD mentor", "my DEAR friend", "my LATE uncle"), and an opinion
    # object can carry them too ("dear old jazz clubs"). Trimmed from the FRONT
    # of a topic key so the stored slot is the concept, not the address form.
    # Structural closed-class seed; removing an entry only re-admits one
    # modifier shape. Grows at runtime via data/functional_lexicon.json.
    "leading_modifiers": ["dear", "dearest", "old", "late", "former", "poor",
                          "beloved", "cherished", "respected", "deceased",
                          "belated"],
    "trailing_modifiers": ["earlier", "now", "then", "later", "soon",
                            "always", "never", "sometimes", "again",
                            "already", "still", "yet", "just", "even",
                            "too", "very", "quite", "really", "better",
                            "best", "worse", "worst", "more", "most",
                            "less", "least", "often", "usually",
                            "generally", "basically", "actually",
                            "literally", "seriously", "honestly",
                            "frankly", "clearly", "obviously",
                            "apparently", "presumably", "supposedly"],
    "subject_context": ["anything", "assume", "assuming", "author", "avoid", "background", "bad", "bake", "bakes", "became", "become", "becomes", "best", "better", "boss", "brew", "brewed", "brews", "briefly", "build", "building", "builds", "call", "called", "calls", "capital", "captain", "cause", "caused", "causes", "ceo", "chief", "close", "closes", "compose", "composed", "composes", "cook", "cooks", "create", "created", "creates", "creator", "definition", "definition of", "describe", "describes", "design", "designed", "designs", "develop", "developed", "develops", "did", "difference", "differences", "director", "discover", "discovered", "discovering", "discovers", "do", "does", "done", "draw", "draws", "emperor", "everything", "example", "examples", "explain", "explains", "fact", "facts", "feel", "feels", "find", "finds", "founder", "get", "gets", "give", "gives", "go", "goes", "good", "got", "governor", "happen", "happened", "happening", "head", "help", "helps", "history", "idea", "ideas", "imagine", "immediately", "instantly", "invent", "invented", "inventing", "inventor", "invents", "keep", "keeps", "kind", "kinds", "king", "know", "knows", "leader", "learn", "learns", "least", "like", "likes", "made", "make", "makes", "many", "mayor", "mean", "meaning", "meaning of", "means", "minister", "most", "much", "name", "named", "need", "needs", "occur", "occurred", "occurs", "open", "opens", "overview", "owner", "place", "play", "plays", "point", "population", "population of", "president", "pretend", "prevent", "prevents", "prime", "produce", "produced", "produces", "queen", "read", "reads", "reason", "reasons", "remain", "remains", "said", "sang", "say", "says", "show", "shows", "sing", "sings", "some", "something", "sort", "sorts", "start", "starts", "stay", "stays", "stop", "stops", "studies", "study", "stuff", "sudden", "suddenly", "summary", "sung", "suppose", "supposing", "switch", "switched", "switches", "take", "takes", "taught", "teach", "teaches", "tell", "tells", "term", "termed", "thing", "things", "think", "thinks", "took", "train", "trained", "trains", "turn", "turned", "turns", "type", "types", "understand", "understands", "use", "used", "uses", "want", "wants", "way", "ways", "went", "winner", "worse", "worst", "write", "writes", "wrote", "year", "years"],
    "conditional_frame": ["a", "an", "and", "are", "assume", "assuming", "be", "being", "but", "can", "ceased", "could", "day", "days", "disappeared", "disappears", "for", "gone", "happen", "happened", "happens", "if", "immediately", "instantly", "is", "minute", "minutes", "moment", "occur", "occurred", "occurs", "of", "off", "on", "one", "or", "removed", "second", "seconds", "shut", "stopped", "stops", "sudden", "suddenly", "suppose", "supposing", "switch", "switched", "the", "then", "to", "turn", "turned", "turns", "vanished", "was", "were", "what", "would"],
    "attr_words": ["age", "area", "author", "birthday", "capital", "ceo", "color", "colour", "creator", "currency", "depth", "director", "founder", "height", "inventor", "language", "length", "mass", "meaning", "population", "president", "religion", "symbol", "weight", "width"],
    "grammatical_concepts": ["a", "about", "above", "accordingly", "across", "after", "again", "ago", "all", "already", "also", "although", "am", "among", "an", "and", "another", "any", "are", "around", "as", "away", "back", "backward", "be", "because", "been", "before", "being", "below", "besides", "between", "both", "but", "bye", "can", "certain", "consequently", "could", "dare", "did", "different", "do", "does", "doing", "down", "during", "each", "either", "enough", "even", "every", "far", "few", "first", "for", "forward", "furthermore", "goodbye", "had", "has", "have", "he", "hello", "hence", "her", "here", "hers", "herself", "hey", "hi", "him", "himself", "his", "how", "however", "i", "if", "in", "inside", "instead", "is", "its", "itself", "just", "last", "later", "least", "less", "like", "little", "many", "may", "maybe", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "much", "must", "my", "myself", "near", "need", "neither", "nevertheless", "no", "nonetheless", "nor", "not", "now", "off", "on", "once", "one", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "outside", "over", "own", "particular", "perhaps", "please", "possibly", "probably", "same", "second", "several", "shall", "she", "should", "since", "so", "some", "soon", "sorry", "still", "subsequently", "such", "than", "thank", "thanks", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "these", "they", "this", "those", "though", "three", "through", "thus", "too", "twice", "two", "under", "unless", "until", "up", "us", "very", "was", "we", "were", "what", "when", "where", "whether", "which", "while", "who", "whole", "whom", "whose", "why", "will", "would", "yes", "yet", "you", "your", "yours", "yourself", "yourselves"],
}


class FunctionalLexicon:
    """Single source of truth for closed-class functional lexicons.

    Every accessor returns a ``set`` (seed-fallback per key) so a partial fit
    file never breaks a consumer.
    """

    def __init__(self, values: Optional[Dict[str, list]] = None) -> None:
        self._v: Dict[str, Set[str]] = {
            k: set(v) for k, v in _SEED.items()}
        if values:
            for k, v in values.items():
                if k in _SEED:
                    self._v[k] = set(v)

    @property
    def polarity_increase(self) -> Set[str]:
        return self._v.get("polarity_increase", set(_SEED["polarity_increase"]))

    @property
    def polarity_decrease(self) -> Set[str]:
        return self._v.get("polarity_decrease", set(_SEED["polarity_decrease"]))

    @property
    def polarity_remove(self) -> Set[str]:
        return self._v.get("polarity_remove", set(_SEED["polarity_remove"]))

    @property
    def moral_markers(self) -> Set[str]:
        return self._v.get("moral_markers", set(_SEED["moral_markers"]))

    @property
    def moral_ambiguous(self) -> Set[str]:
        return self._v.get("moral_ambiguous", set(_SEED["moral_ambiguous"]))

    @property
    def framing(self) -> Set[str]:
        return self._v.get("framing", set(_SEED["framing"]))

    @property
    def leading_modifiers(self) -> Set[str]:
        return self._v.get("leading_modifiers",
                           set(_SEED["leading_modifiers"]))

    @property
    def trailing_modifiers(self) -> Set[str]:
        return self._v.get("trailing_modifiers",
                           set(_SEED["trailing_modifiers"]))

    # ── Stage 5b-iii (P1-H): closed-class lists consolidated from the
    # 6 hand-crafted engine sets. Each accessor returns the fit-file
    # value when present, else the seed (the original hand list). The
    # engine routes its usage sites through CognitiveChatEngine._closed_class.
    @property
    def common_words(self) -> Set[str]:
        return self._v.get("common_words", set(_SEED["common_words"]))

    @property
    def topic_skip(self) -> Set[str]:
        return self._v.get("topic_skip", set(_SEED["topic_skip"]))

    @property
    def subject_context(self) -> Set[str]:
        return self._v.get("subject_context", set(_SEED["subject_context"]))

    @property
    def conditional_frame(self) -> Set[str]:
        return self._v.get("conditional_frame", set(_SEED["conditional_frame"]))

    @property
    def attr_words(self) -> Set[str]:
        return self._v.get("attr_words", set(_SEED["attr_words"]))

    @property
    def grammatical_concepts(self) -> Set[str]:
        # `.get` on the seed: unlike the other accessors there is no
        # grammatical_concepts seed (the canonical set lives on the engine as
        # _GRAMMATICAL_CONCEPTS), so a missing key must yield an empty set
        # rather than raise KeyError -- which _closed_class would swallow,
        # silently disabling the function-word safety net.
        return self._v.get("grammatical_concepts",
                           set(_SEED.get("grammatical_concepts", [])))

    @classmethod
    def load(cls) -> Optional["FunctionalLexicon"]:
        if not os.path.exists(_FIT_PATH):
            return None
        try:
            with open(_FIT_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return cls(values=d)
        except Exception:
            return None

    def save(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        out = {k: sorted(v) for k, v in self._v.items()}
        with open(_FIT_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    def to_dict(self) -> Dict[str, list]:
        return {k: sorted(v) for k, v in self._v.items()}


def default_lexicon() -> "FunctionalLexicon":
    """Loaded fit file if present, else the seed lexicon (fail-open)."""
    loaded = FunctionalLexicon.load()
    return loaded if loaded is not None else FunctionalLexicon()
