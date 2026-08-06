import os
import re
import pickle
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from .models import CorrectionType
from .personal_fact_store import PersonalFactStore, UserStanceStore
from . import pet_slots as _pet_slots

# ── Dedicated user-model store ───────────────────────────────────────────────
# The per-user model used to be pickled *inside* the engine weight snapshot,
# which coupled it to the cognitive-graph lifecycle. It now lives in its own
# <repo>/user_models/ directory so user profiles are independent of (and
# outlive) any single weight checkpoint.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
USER_MODELS_DIR = os.path.join(_REPO_ROOT, "user_models")


# Correction detection patterns — ACC conflict detection (Error-Related Negativity)
_CORRECTION_DIRECT_PATTERNS = [
    r"\bno[!,.]", r"that'?s wrong", r"that'?s not right", r"that'?s incorrect",
    r"you'?re wrong", r"you are wrong", r"you'?re incorrect", r"you are incorrect",
    r"not correct", r"actually[!,]", r"not true", r"that'?s false",
    r"\bwrong[!.]", r"\bincorrect[!.]", r"\bmistake[!.]", r"\berror[!.]",
    r"no[!,]\s+that", r"no[!,]\s+it", r"wait[!,]\s+",
    r"that'?s not what", r"that is not what", r"that isn'?t what",
    r"hold on[!,]",
]

# Patterns that explicitly supply a corrected fact
_CORRECTION_FACT_PATTERNS = [
    # "it's X, not Y"
    r"it'?s\s+(\w+)[,.]*\s+not\s+(\w+)",
    r"it is\s+(\w+)[,.]*\s+not\s+(\w+)",
    # "X is Y, not Z"
    r"([\w\s]+?)\s+is\s+(\w+)[,.]*\s+not\s+(\w+)",
    r"([\w\s]+?)\s+are\s+(\w+)[,.]*\s+not\s+(\w+)",
]

# D3 (round v3): explicit correction shape "X's name is not Y, it's Z" /
# "X is not Y, it's Z" where the CORRECTED value is the token after "it's"
# (never the negation word). Handled separately because its group order is
# (subject_attr, correct_value) which the generic 2-group branch would misread.
_CORRECTION_NAME_FACT_PATTERN = (
    r"(?:my\s+)?([\w'-]+?)(?:'s)?\s+(?:name\s+)?is\s+not\s+[\w'-]+"
    r"[,.]*\s+it'?s\s+([\w'-]+)"
)


# Stance-reversal cues (C4): universal retraction phrasings that recode the
# user's valuation of a topic to the opposite pole. These are STRUCTURAL
# language seeds, same status as the correction/opinion cue lists above — a
# small closed set of retraction verbs/idioms, NOT a per-topic table. Group
# placement matters: the topic is the clause AFTER the cue ("i take back what
# i said — plastic bans..."), so the match's .end() marks the topic span.
_RETRACTION_CUES = (
    # D4 fix (round v-aug06): allow an optional pronoun/adverb between "take"
    # and "back" — "i take it back", "i take this back", "i take that back" are
    # all the same retraction speech act. The prior pattern only allowed "that",
    # so "i take it back" (the most common spoken form) failed to match and the
    # contradiction was silently dropped, leaving the stale positive stance.
    r"\bi\s+take\s+(?:it|this|that|things|them|what\s+i\s+said)?\s*back",
    r"\bi\s+retract",
    r"\bi\s+(?:have|'ve)?\s*changed\s+my\s+mind",
    r"\bi\s+(?:am|'m)?\s*changing\s+my\s+mind",
    r"\bi\s+no\s+longer\s+(?:think|believe|feel|support|care)\b",
    r"\bi\s+was\s+wrong\s+(?:about|on|there)\b",
    # allow an adverb before "wrong" (e.g. "i was completely wrong about X")
    # or between "wrong" and the topic preposition. The common spoken form is
    # "<adverb> wrong about", which the original "wrong\s+(about...)" missed,
    # silently leaving the stale stance un-reversed.
    r"\bi\s+was\s+(?:\w+\s+)?wrong\s+(?:about|on|there)\b",
    r"\bi\s+was\s+wrong\s+\w+\s+(?:about|on|there)\b",
    r"\bi\s+(?:was|am)\s+too\s+hasty\b",
    r"\bi\s+(?:was|am|'m)\s+(?:mistaken|in\s+error)\b",
    r"\b(?:they're|it's|that's|wasn't|isn't|not)\s+(?:not\s+that\s+bad|not\s+so\s+bad|fine|okay|ok|acceptable|good\s+after\s+all)\b",
    r"\bi\s+take\s+(?:it|that|this|things|them|what\s+i\s+said|my\s+words)\s*back",
    r"\bscratch\s+that\b",
)

# Softening retraction cues: the user is walking a stance BACK toward neutral,
# not flipping to the opposite conviction ("olives? not that bad, i was too
# hasty"). These drive a PARTIAL reversal (relax toward neutral) instead of a
# full 180°. Kept as a separate tuple so mine_stance_reversal can test
# `cue is _SOFTENING_CUES` by object identity — it is a sub-set of
# _RETRACTION_CUES. No topic is named here, so this is seed structure, not an
# answer table; a hard recant lives in the rest of _RETRACTION_CUES.
_SOFTENING_CUES = (
    r"\bi\s+(?:was|am)\s+too\s+hasty\b",
    r"\bi\s+(?:was|am|'m)\s+(?:mistaken|in\s+error)\b",
    r"\b(?:they're|it's|that's|wasn't|isn't|not)\s+(?:not\s+that\s+bad|not\s+so\s+bad|fine|okay|ok|acceptable|good\s+after\s+all)\b",
)

# Conjoined multi-pet disclosure pattern: "i have a ferret named pim and a
# parrot called coco". One regex captures the whole chain; the miner expands it
# into per-animal "species named name" pairs (see possession-mining block below).
# Defined ONCE at module scope so the miner identifies it by object identity,
# not by a fragile `startswith` of its literal (a prior attempt compared a
# different literal and the branch never fired). Escapes are plain articles
# (a|an|the|some...), NOT \a/\t/\s — those are regex metachars that silently
# fail to match the spoken words.
_CONJOINED_PET_PAT = (
    r"\bi\s+have\s+(?:\d+\s+|(?:a|an|the|some|several|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"((?:(?:a|an|the|my|our|their|his|her)?\s*[\w\'-]+\s+(?:named|called)\s+[\w\'-]+"
    r"\s*(?:,?\s*(?:and|&|,)\s*(?:a|an|the)?\s*)?)+)"
)
@dataclass
class UserModel:
    edge_reactivations: Dict[Tuple[str, str], int] = field(default_factory=dict)
    query_concepts: Set[str] = field(default_factory=set)
    user_name: str = ""
    user_location: str = ""      # "I live in X" / "I am from X"
    user_background: str = ""     # free biographical note (e.g. "born in Paris")
    preferences: Dict[str, Any] = field(default_factory=dict)
    # Learned personal-fact store (brain-faithful: confidence x recency x decay,
    # improves over time via confirm/contradict). Seeded from the high-precision
    # regex below AND from any _mine_episodic_facts hit, so "my cat is Pixel"
    # becomes a gradeable, correctable fact rather than a frozen field.
    personal_facts: PersonalFactStore = field(default_factory=PersonalFactStore)
    # Learned opinion store (C): the user's value judgments (for/against a
    # topic), kept SEPARATE from biographical facts. Opinions decay faster than
    # facts (malleable attitudes), per OFC/vmPFC vs hippocampal circuit split.
    opinions: UserStanceStore = field(default_factory=UserStanceStore)

    knowledge_model: Dict[str, float] = field(default_factory=dict)
    learning_goals: Dict[str, int] = field(default_factory=dict)
    emotional_rapport: Dict[str, float] = field(default_factory=dict)
    cognitive_style: str = "balanced"
    engagement_level: float = 0.5
    conversation_depth: float = 0.0
    interaction_count: int = 0
    relationship_depth: float = 0.0
    goals: List[str] = field(default_factory=list)
    last_goal: str = "EXPLORING"

    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        'valence': 0.0, 'arousal': 0.3, 'dominance': 0.5,
    })
    belief_state: Dict[str, Dict] = field(default_factory=dict)
    interaction_history: List[Dict] = field(default_factory=list)
    _emotion_detector: Any = None

    topic_interaction_count: Dict[str, int] = field(default_factory=dict)
    topic_followup_count: Dict[str, int] = field(default_factory=dict)
    last_topic: str = ""
    turn_since_topic_change: int = 0

    # ── Phase: Correction pattern tracking (ACC/ERN analog) ──
    detected_correction: bool = False
    detected_correction_type: Optional[CorrectionType] = None
    correction_severity: float = 0.0
    correction_subject: str = ""
    detected_correction_fact: Optional[Tuple[str, str, str]] = None
    _last_user_valence_before_response: float = 0.0
    _last_response_for_correction: str = ""
    _last_response_strategy_for_correction: str = ""
    _previous_user_query: str = ""

    def observe_chain(self, hops: List[Tuple[str, str]], is_user_query: bool = False):
        for from_label, to_label in hops:
            key = (from_label.lower(), to_label.lower())
            self.edge_reactivations[key] = self.edge_reactivations.get(key, 0) + 1
        if is_user_query:
            for from_label, to_label in hops:
                self.query_concepts.add(from_label.lower())
                self.query_concepts.add(to_label.lower())
                self.knowledge_model[from_label.lower()] = min(1.0, self.knowledge_model.get(from_label.lower(), 0.0) + 0.1)
                self.learning_goals[to_label.lower()] = self.learning_goals.get(to_label.lower(), 0) + 1

    def mine_personal_facts(self, text: str,
                             run_correction: bool = True) -> None:
        """High-precision personal-fact miner (B3 / A5). Seeded from explicit
        "my X is Y" / "I have a X named Y" / name / location patterns into the
        learned PersonalFactStore. Called both from observe_user_query (full
        ToM pass) and directly by the identity gate for same-turn capture, so
        it must take ONLY raw text (no subject / valence dependency).

        Facts seeded here are gradeable + correctable; they are NOT frozen
        regex buckets — the store learns the rest from confirm/contradict.
        """
        q_clean = re.sub(r"\s+", " ", text).strip()
        # Correction cue (B4 wiring, investigation Gap 1): when the user is
        # correcting us ("no, my cat is milo", "actually i live in paris"),
        # a mined fact whose attribute already holds a DIFFERENT active value
        # must supersede it via contradict() — the user is ground truth for
        # their own profile. Without this the correction loop is dead: the
        # store has contradict() but nothing ever called it.
        _corrective = bool(re.search(
            r"^\s*no\b|\bactually\b|\bthat'?s\s+(?:wrong|not\s+right|incorrect)\b"
            r"|\bi\s+(?:said|told\s+you)\b|\bnot\s+[\w'-]+\s*,?\s*(?:it'?s|it\s+is)\b",
            q_clean, re.IGNORECASE))


        _NEG_WORDS = {"not", "no", "never", "none", "nil", "n't", "dont",
                       "don't", "false", "wrong", "incorrect"}
        _VALUE_STOP = _NEG_WORDS | {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "am", "be", "been", "being", "to", "of", "in", "on", "at", "for",
            "with", "my", "your", "i", "you", "it", "this", "that", "me"}

        # D3 (round v4): name-correction detection MUST run here, not only in
        # UserModel._extract_correction_fact (which observe_user_query calls at
        # engine.py:4193 — AFTER the self-disclosure early-return at :3392). A
        # correction phrased as a disclosure ("my sister's name is not meena,
        # it's priya") returns at :3392 before late detection ever sets the
        # flag, so the contradict() block at :3408 saw None and the correction
        # was silently lost. mine_personal_facts runs at :2549 (pre-early-return),
        # so detecting here lets the engine persist the corrected fact online.
        # Seeds the same flags the full circuit uses; the :3408 block does the
        # actual contradict() (no retrain, no authored text). _VALUE_STOP is
        # defined above, so the negation guard can reject junk values.
        _nm = re.search(_CORRECTION_NAME_FACT_PATTERN, q_clean, re.IGNORECASE)
        if _nm:
            _subj_attr = _nm.group(1).strip().removesuffix("'s")
            _correct_val = _nm.group(2).strip()
            if _correct_val and _correct_val not in _VALUE_STOP:
                self.detected_correction = True
                self.detected_correction_fact = ("i", _subj_attr, _correct_val)
                self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                self.correction_severity = max(self.correction_severity, 0.8)

        def _put_fact(attr: str, val: str, conf: float) -> None:
            # D3 (round v3): never store a closed-class / negation token as a
            # fact value. The old miner matched "my sister's name is not meena,
            # it's priya" and stored value="not" (the word after "is"). Values
            # that are not real content are rejected so they can't pollute the
            # personal-fact store; the real corrected value arrives via the
            # correction circuit (detected_correction_fact) instead.
            _val = (val or "").strip().strip(" .,!?;:'\"").lower()
            if not _val or _val in _VALUE_STOP:
                return
            existing = self.personal_facts.get("i", attr)
            if (_corrective and existing is not None
                    and existing.value.lower() != _val):
                self.personal_facts.contradict("i", attr, _val)
            else:
                self.personal_facts.assert_fact("i", attr, _val,
                                                confidence=conf,
                                                source="seed_regex")

        m_name = re.search(
            r"\b(?:my\s+name\s+is|i\s+am\s+called|call\s+me)\s+"
            r"([^.,!?]+)",
            q_clean, re.IGNORECASE)
        if not m_name:
            m_name = re.search(r"\b(?:do\s+you\s+know\s+my\s+name|know\s+my\s+name|is\s+my\s+name)\s+is\s+(.+)", q_clean, re.IGNORECASE)
        m_loc = re.search(
            r"\bi\s+(?:live|lives|am|was|were|grew\s+up)\s+(?:in|near|at|from)\s+"
            r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,7})",
            q_clean, re.IGNORECASE)
        # FIX (round v-aug06b): when the location clause NAMES a place via
        # "called/named" (e.g. "i live in a small town called hollow creek"),
        # the real toponym is the named phrase, not the filler leading up to
        # it. Extract the named toponym and prefer it over the raw capture so
        # "hollow creek" is stored instead of "a small town called hollow".
        # FIX (round v-aug06d): extended the lexicon to natural features
        # (valley/dale/glen/cove/bay/fjord/island/peninsula/canyon/hollow) so
        # "a converted mill in a valley called ashcombe" resolves to "ashcombe"
        # (previously "valley" was absent and the greedy capture grabbed
        # "a converted mill in a"). No per-toponym table.
        _named_loc = re.search(
            r"\b(?:in|near|at|from)\s+(?:a|an|the|my|our|their|his|her)?\s*"
            r"(?:small\s+)?(?:town|city|village|settlement|place|hamlet|"
            r"community|suburb|borough|region|area|country|state|province|"
            r"valley|dale|glen|cove|bay|fjord|island|peninsula|canyon|hollow|"
            r"estuary|inlet|harbor|harbour|beach|shore)\b"
            r"(?:\s+(?:called|named|spelled))\s+"
            r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3})",
            q_clean, re.IGNORECASE)
        if _named_loc:
            m_loc = type("_Loc", (), {"group": lambda self, n: _named_loc.group(1)})()
            # store the named toponym directly (reuse m_loc handling below)
            _loc = _named_loc.group(1).strip().strip(" .,!")
            if _loc and len(_loc.split()) <= 4:
                self.user_location = _loc
                _put_fact("location", _loc, 0.6)
        elif m_loc:
            _loc = m_loc.group(1).strip().strip(" .,!")
            _loc = re.split(r"\s+(?:and|but|,|\.)\s*", _loc)[0].strip()
            # FIX (round v-aug06c): a location clause like "a small apartment
            # near the river in porto" caps the 5-word capture at "a small
            # apartment near the" and silently drops the real toponym "porto".
            # A proper noun after a trailing "in/near/at <Place>" is the actual
            # place the user lives — prefer it. Generic: matches any capitalized
            # word led by a place preposition, no per-city table.
            # FIX (round v-aug06d): also catch a toponym introduced by
            # "called/named" ANYWHERE in the clause (not only end-of-string),
            # e.g. "a converted mill in a valley called ashcombe" where the
            # named-toponym lexicon missed the feature word — prefer the proper
            # noun after called/named over the filler leading up to it.
            _trailing = re.search(
                r"\b(?:in|near|at|from)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,2})\s*$",
                m_loc.group(1), re.IGNORECASE)
            if not _trailing:
                _trailing = re.search(
                    r"\b(?:called|named|spelled)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,3})",
                    m_loc.group(1), re.IGNORECASE)
            if _trailing:
                _loc = _trailing.group(1).strip()
            if _loc and len(_loc.split()) <= 5:
                self.user_location = _loc
                _put_fact("location", _loc, 0.6)
        if m_name:
            name_cand = m_name.group(1).strip()
            name_cand = re.split(r"\s+(?:and|but|,|\.)\s*", name_cand)[0].strip()
            name_words = name_cand.split()
            if name_words and name_words[0].lower() in ("is", "are", "was", "were"):
                name_words = name_words[1:]
            name_cand = " ".join(name_words)
            if name_cand and name_cand.lower() not in ("happy", "sad", "tired", "busy", "fine", "good", "what", "who", "why", "how"):
                name_cap = " ".join(w.capitalize() for w in name_cand.split())
                self.user_name = name_cap
                _put_fact("name", name_cap, 0.6)
        for _pat in (
            # D2 (round v2): tolerate an optional "name" word between the
            # relation and "is" so "my daughter name is ingrid" stores
            # daughter=ingrid (the old pattern required exactly one token
            # before "is" and missed the "name" form).
            # FIX (round v-aug06d): generalize to a MULTI-WORD attribute so
            # "my favorite color is ochre" stores attr="favorite color",
            # val="ochre" (the old single-token attr grabbed attr="favorite",
            # val="color" and lost the real value). Attribute capped at 4 words,
            # value at 4 words; this is structural (no per-topic attribute
            # list) — any "<attr phrase> is <value phrase>" is captured.
            r"\bmy\s+([\w'-]+(?:\s+[\w'-]+){0,3})\s+(?:is|are)\s+([\w'-]+(?:\s+[\w'-]+){0,3})",
            # FIX (round v-aug06d): "i work as a <role>" / "i work for
            # <org>" self-descriptions are identity facts, not throwaway
            # activities. The old activity miner only caught the verb "work"
            # inside the generic activity loop and stored junk (does=s).
            # Capture the role as a durable 'work' fact. Generic: any noun
            # after "work as/for", no occupation list.
            r"\bi\s+work\s+(?:as|for)\s+(?:a\s+|an\s+|the\s+)?([\w'-]+)",
            r"\bi\s+(?:have|keep)\s+(?:a|an|the)\s+([\w'-]+)\s+(?:named|called)\s+([\w'-]+)",
            # FIX (round v-aug06b): conjoined multi-pet disclosures
            # ("i have a ferret named pim and a parrot called coco"). The single
            # pattern above only captures the FIRST animal; the rest are lost.
            # This captures each "a/an <species> named/called <name>" segment
            # inside a "have ... and ..." chain so every pet is stored. Generic
            # (matches any species word, no per-animal table). The pattern is the
            # module-level _CONJOINED_PET_PAT constant; the miner identifies it by
            # object identity (_pat is _CONJOINED_PET_PAT), not a fragile
            # startswith of its literal (which previously never matched and left
            # the second animal unstored).
            _CONJOINED_PET_PAT,
            # C-fix (round v-aug04): quantified / multi-name possessions
            # ("i have two cats named biscuit and gravy", "i have 3 dogs
            # called rex, spot and max"). The old first pattern required a
            # single article + one name, so "two cats named biscuit and gravy"
            # never matched and the pets were lost. This matches an optional
            # number/quantifier word then a noun, then one or more names joined
            # by "and"/commas, and stores EACH name under its own entity slot
            # so a later "what are my cats called" recall finds them.
            r"\bi\s+(?:have|keep)\s+(?:\d+\s+|(?:a|an|the|some|several|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
            r"([\w'-]+)\s+(?:named|called|named\s+called)\s+"
            r"([\w'-]+(?:\s+(?:and|,|&)\s*[\w'-]+)*)",
            r"\bmy\s+([\w'-]+)\s+(?:named|called)\s+([\w'-]+)",
            # D2: "i am a/an <noun>" self-descriptions (vegetarian, pilot,
            # teacher, ...) captured as a durable identity/role fact. Generic
            # structural capture — the noun becomes the attribute value, no
            # per-topic list. A small universal stop-set excludes non-facts
            # like "person"/"human"/"child" which are not stored attributes.
            r"\bi\s+am\s+(?:a|an)\s+([\w'-]+)",
            r"\bi\s+am\s+allergic\s+to\s+([\w'-]+)",
        ):
            for _m in re.finditer(_pat, q_clean, re.IGNORECASE):
                _attr, _val = None, None
                # FIX (round v-aug06b): the conjoined-pet pattern captures a
                # chain like "ferret named pim and a parrot called coco".
                # Expand it into individual "species named name" pairs and
                # store each via the same slot logic below. Identified by object
                # identity (_pat IS the module constant), which is exact — the
                # previous startswith(guard) comparison against a DIFFERENT
                # literal never matched, so the branch was dead and the second
                # animal was silently dropped.
                if _pat is _CONJOINED_PET_PAT:
                    _segs = re.findall(
                        r"([\w'-]+)\s+(?:named|called)\s+([\w'-]+)",
                        _m.group(1), re.IGNORECASE)
                    for _sp, _nm in _segs:
                        _sp, _nm = _sp.strip().lower(), _nm.strip().strip(".,!?")
                        if not _sp or not _nm:
                            continue
                        _species = _pet_slots.species_of(_sp)
                        if _species is None and _sp.isalpha():
                            _species = _pet_slots.learn_species(_sp)
                        else:
                            _species = _pet_slots.species_of(_sp) or _sp
                        if _species is not None:
                            # count existing slots for this species to append _2, _3
                            _i = 1
                            while _pet_slots.slot_for(_species, _i) in self.personal_facts.facts:
                                _i += 1
                            _put_fact(_pet_slots.slot_for(_species, _i), _nm, 0.6)
                    continue
                if _m.lastindex is not None and _m.lastindex >= 2:
                    _attr, _val = _m.group(1).strip().lower(), _m.group(2).strip()
                elif _m.lastindex == 1:
                    # Single-group patterns: "i am a/an <noun>" / "i am
                    # allergic to <noun>" — the one group is the VALUE; the
                    # attribute is inferred from the pattern.
                    _val = _m.group(1).strip().lower()
                    if _pat.startswith(r"\bi\s+am\s+(?:a|an)"):
                        _attr = "role"
                        if _val in ("person", "human", "woman", "man", "child",
                                    "kid", "adult", "student", "robot", "machine",
                                    "ai", "thing", "people", "boy", "girl"):
                            continue
                    elif _pat.startswith(r"\bi\s+work\s+(?:as|for)"):
                        _attr = "work"
                    elif _pat.startswith(r"\bi\s+am\s+allergic\s+to"):
                        _attr = "allergy"
                if _attr and _val and _attr not in ("name", "location"):
                    # A possession disclosure may name several animals ("i have
                    # two cats named biscuit and gravy"). Split the value on
                    # "and"/"," and store EACH name in its own slot. The slot
                    # KEEPS the species (cat, cat_2, dog) via pet_slots so a
                    # user can own more than one kind of animal, a cued recall
                    # can ask about one species, and a correction ("no, my cat
                    # is milo") finds the prior value under the same key the
                    # user's own word resolves to.
                    _names = re.split(r"\s+(?:and|,|&)\s*", _val)
                    _species = _pet_slots.species_of(_attr)
                    if _species is None and len(_names) >= 1 and _attr.isalpha():
                        # An unknown animal word in a "named/called" possession
                        # frame is a species RAVANA has not met yet — learn it
                        # so later recalls address the same slot.
                        if re.search(r"\b(?:named|called)\b", _m.group(0), re.IGNORECASE):
                            _species = _pet_slots.learn_species(_attr)
                    if _species is not None:
                        for _i, _nm in enumerate(_names, 1):
                            _nm = _nm.strip().strip(".,!?")
                            if _nm:
                                _put_fact(_pet_slots.slot_for(_species, _i),
                                          _nm, 0.6)
                    elif len(_names) > 1:
                        for _i, _nm in enumerate(_names, 1):
                            _nm = _nm.strip().strip(".,!?")
                            if _nm:
                                _put_fact(f"{_attr}_{_i}", _nm, 0.6)
                    else:
                        _put_fact(_attr, _val, 0.6)

        # D3 (round v3): capture self-disclosed ACTIVITIES / possessions that the
        # existing "my X is Y" / "i am a role" miners miss — e.g. "i run a chai
        # stall near the mysore palace", "i play the tabla when the stall is
        # closed", "i've been watching the night sky for twelve years". These are
        # first-person disclosures of what the user DOES / has, and must land in
        # the personal-fact store so a later "what have you learned about me"
        # summary and cued recall can surface them (D-D bug: only 'likes' was
        # recalled because activity facts were never mined). Structural: a small
        # closed VERB set (not a per-topic list) + the resolved content HEAD of
        # the object phrase (via _opinion_topic, which drops closed-class words),
        # so the stored value is a real concept ("chai stall", "tabla", "night
        # sky"), never a function word. This is seed structure RAVANA expands from
        # experience — it adds to the same PersonalFactStore the user can correct.
        for _verb in ("run", "own", "operate", "play", "teach", "study",
                       "manage", "drive", "build", "make", "sell",
                       "restore", "grow", "watch", "raise", "tend", "brew",
                       "bake", "write", "read", "learn", "practice", "collect",
                       "fix", "paint", "code", "design", "craft", "volunteer",
                       "cook", "fish", "hike", "garden", "farm", "lead", "organize"):
            _m = re.search(
                r"\bi\s+(?:also\s+|really\s+|even\s+|just\s+|now\s+|still\s+"
                r"|often\s+|sometimes\s+|usually\s+)?"
                r"(?:have\s+been\s+)?(?:been\s+)?" + _verb +
                r"\s+(?:a|an|the\s+)?(.+?)(?:\bfor\b|\bwhen\b|\bbut\b|"
                r"\bbecause\b|\band\b|\.|\!|\?|$|,)",
                q_clean, re.IGNORECASE)
            if _m:
                _obj = self._opinion_topic(_m.group(1).strip().lower())
                if _obj and len(_obj.split()) <= 5:
                    _put_fact("does", _obj, 0.55)
        # "i've been <verb>-ing <object> for <duration>" (ongoing activity)
        _cont = re.search(
            r"\bi(?:'ve| have)\s+been\s+(\w+ing)\s+(.+?)(?:\bfor\b|\bsince\b|\.|\!|\?|$|,)",
            q_clean, re.IGNORECASE)
        if _cont:
            _obj = self._opinion_topic(_cont.group(2).strip().lower())
            if _obj and len(_obj.split()) <= 5:
                _put_fact("does", _obj, 0.55)

        # Opinion mining (C2): capture the user's value judgments alongside
        # facts. Runs in the miner (not only observe_user_query) so opinions are
        # captured even when process_turn early-returns before Step 5b (e.g. a
        # bare "i really like cats" hits a preference handler). Polarity from
        # explicit cues + VAD signal already inferred for this turn.
        _vad = self._infer_user_emotion(text)
        _v, _a, _d = _vad
        # A first-person opinion can ONLY be mined from a DECLARATIVE
        # self-report, never from a question ("do you think i love X?" is the
        # user asking about RAVANA's stance, not stating their own). Mining a
        # question creates garbage stances keyed on trailing clause fragments
        # (e.g. "letterpress given" from "do you still think i love letterpress
        # given the wrist thing?"). Structural guard: any interrogative is
        # skipped before the stance loop.
        _is_question = (q_clean.rstrip().endswith("?")
                        or bool(re.match(
                            r"^(what|who|when|where|why|how|which|is|are|do|"
                            r"does|did|can|could|would|should|will|may|might|"
                            r"am|have|has|had)\b", q_clean)))
        if _is_question:
            return
        for _pat, _pol, _conf in (
            # D2 (round v2): capture the FULL object phrase after the verb,
            # not just the first token — "small talk" / "the solitude of the
            # lighthouse" must be one topic, not the function word "small"/"the".
            # The salient CONTENT HEAD is resolved by _opinion_topic (skips
            # closed-class words), so a stance lands on "talk"/"solitude", a
            # real concept, never on "the"/"how"/"small".
            (r"\bi\s+(?:really\s+)?(?:like|love|enjoy|prefer|adore|care\s+for)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", 0.8, 0.6),
            (r"\bi\s+(?:really\s+)?(?:hate|dislike|detest|can't\s+stand)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", -0.8, 0.6),
            # FIX (round v-aug06b): word-boundary-guarded sentiment adjectives.
            # Without \b, "bad" matched the prefix of "badly" ("is badly
            # underrated" -> parsed as "is bad"), inverting a POSITIVE
            # endorsement into a negative stance. "underrated" is a positive
            # endorsement (deserves more recognition) and is now recognized as
            # such; an optional \w+ly adverb slot ("seriously/truly/badly
            # underrated") is tolerated so the adverb never blocks the adjective.
            # Generic sentiment-lexicon expansion (no per-topic rule).
            (r"\bi\s+think\s+(.+?)\s+(?:is|are)\s+(?:\w+ly\s+)?(?:good\b|great|awesome|nice|wonderful|amazing|the\s+future|essential|important|right\b|crucial|vital|underrated|under-rated|underappreciated|under-valued|undervalued)", 0.8, 0.5),
            (r"\bi\s+think\s+(.+?)\s+(?:is|are)\s+(?:\w+ly\s+)?(?:bad\b|terrible|awful|overrated|over-rated|horrible|poor\b|a\s+mistake|harmful|wrong\b|useless)", -0.8, 0.5),
            (r"\bi\s+believe\s+(.+?)\s+(?:is|are)\s+(?:\w+ly\s+)?(?:good\b|great|awesome|nice|wonderful|amazing|the\s+future|essential|important|right\b|crucial|vital|underrated|under-rated|underappreciated|under-valued|undervalued)", 0.8, 0.5),
            (r"\bi\s+believe\s+(.+?)\s+(?:is|are)\s+(?:\w+ly\s+)?(?:bad\b|terrible|awful|overrated|over-rated|horrible|poor\b|a\s+mistake|harmful|wrong\b|useless)", -0.8, 0.5),
            # "i believe we must/should protect/save/ban <X>" -> positive stance on X.
            (r"\bi\s+believe\s+we\s+(?:must|should)\s+(?:protect|save|preserve|defend|fund|support)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", 0.8, 0.55),
            (r"\bi\s+believe\s+we\s+(?:must|should)\s+(?:ban|cut|end|stop|reduce)\s+(.+?)(?:\.|\band\b|\bbut\b|$|,)", -0.8, 0.55),
            (r"\b(.+?)\s+is\s+my\s+favorite\b", 1.0, 0.7),
            (r"\bi\s+believe\s+([\w'-]+)\s+beats\s+([\w'-]+)", 0.7, 0.4),
        ):
            for _m in re.finditer(_pat, q_clean, re.IGNORECASE):
                _raw = _m.group(_m.lastindex).strip().lower()
                if not _raw:
                    continue
                # Resolve the content head so stances never land on a
                # closed-class word (the/a/how/small/...). Returns None when
                # the phrase has no usable content noun -> skip (don't seed a
                # garbage stance).
                _topic = self._opinion_topic(_raw)
                if not _topic:
                    continue
                # Sign-PRESERVING affect blend (D3 fix, round v-aug06).
                # Bug: the prior code used the RUNNING emotional buffer
                # (_v = EMA of ALL prior turns) to modulate polarity via
                # `max(0.3, _p + 0.2)` — so a clearly-negative lexical cue
                # ("i can't stand cilantro", _pol=-0.8) spoken right after a
                # happy stretch (_v>0.2) was flipped to +0.3, because the
                # buffer leak beat the explicit attitude. That is backwards:
                # the USER's stated verb is the ground-truth attitude signal;
                # affect should only REINFORCE it, never reverse it.
                # Fix: VAD now only strengthens the SAME signed pole (or
                # widens a neutral cue), and can never cross the sign line.
                # This matches vmPFC value integration: the lexical attitude
                # is the delta-rule target; affect is a gain, not a sign.
                _p = float(_pol)
                if _p < -0.05 and _v < -0.1:
                    _p = min(_p, _p - 0.15)          # more negative, same sign
                elif _p > 0.05 and _v > 0.1:
                    _p = max(_p, _p + 0.15)          # more positive, same sign
                # neutral lexical cue (_p==0) may be steered by a strong signal
                elif abs(_p) <= 0.05 and abs(_v) >= 0.3:
                    _p = 0.6 if _v > 0 else -0.6
                _p = max(-1.0, min(1.0, _p))
                self.opinions.express_stance(_topic, polarity=_p, confidence=_conf,
                                            valence=_v, arousal=_a)

        # Stance-reversal mining: "i take back X" / "i changed my mind about X" /
        # "i retract my stance on X" recodes the user's valuation of the topic to
        # the opposite pole (vmPFC re-evaluation), LINKED to the PRIOR stance the
        # store already holds — this is an attitude-change operator, not a fresh
        # opinion. Runs last so it can see (and reverse) any stance just mined.
        self.mine_stance_reversal(text)

    def _stance_key_in_text(self, text: str):
        """Return the held stance key whose TOPIC appears in `text`, else None.

        Generic resolver for retraction/idiom fallbacks: instead of requiring
        the FULL multiword key as a substring (which fails when the user names
        only part of it, e.g. recants "letterpress" but the key is "letterpress
        printing"), it matches when ANY whitespace-delimited token of the key
        appears as a whole word in `text`. This is content-driven (resolves
        against the live stance store) and generalizes to any topic — no
        per-topic table. Prefers the LONGEST key match so a more specific
        stance wins over a generic one.
        """
        if not text:
            return None
        _words = set(re.findall(r"[a-z']+", text.lower()))
        if not _words:
            return None
        _best = None
        _best_score = 0
        for _k in self.opinions.stances:
            if not _k:
                continue
            _ktoks = [t for t in re.findall(r"[a-z']+", _k.lower()) if t]
            if not _ktoks:
                continue
            # A partial recant names only part of a multiword key (e.g.
            # "letterpress" for the stored "letterpress printing"); match when
            # ANY key token appears as a whole word, and prefer the key with
            # the most tokens present (most specific match wins).
            _matched = sum(1 for t in _ktoks if t in _words)
            if _matched == 0:
                continue
            if _matched > _best_score:
                _best_score = _matched
                _best = _k
        return _best

    def mine_stance_reversal(self, text: str) -> None:
        """Detect a stance-reversal/retraction and recode the stored stance.

        Attitude change is a valuation RECODE, not a fresh opinion merge: when
        the user retracts a position ("i take back X", "i changed my mind about
        X"), the previously-held stance is flipped toward the opposite pole.
        Crucially this is LINKED to the stance the store already holds for that
        topic — a benign acknowledgment with no store mutation would leave the
        recorded opinion stale (the "take back not linked" gap).
        """
        q = re.sub(r"\s+", " ", (text or "").lower().strip())
        if not q:
            return
        cue_end = None
        _matched_cue = None
        for pat in _RETRACTION_CUES:
            m = re.search(pat, q)
            if m:
                cue_end = m.end()
                _matched_cue = pat
                break
        if cue_end is None:
            return
        # A retraction cue is either a HARD recant ("i was wrong about X",
        # "i take it all back" — flip decisively) or a SOFTENING ("x isn't that
        # bad, i was too hasty", "i came around a bit" — relax toward neutral,
        # never invert). The softening cues are exactly the phrase set that
        # resolves to a partial reversal; everything else is a hard recant.
        # This drives reverse_stance's blend magnitude, so the same code path
        # produces opposite-hemisphere vs near-neutral recodes from the
        # utterance itself — no per-topic rule, no hardcoding of the topic.
        _soft = _matched_cue in _SOFTENING_CUES
        # The topic lives in the clause after the retraction cue. Strip leading
        # connectors/prepositional frames that carry no content.
        tail = q[cue_end:].strip(" \t.,!?;:'\"\u2014-")
        tail = re.sub(
            r"^(?:what\s+i\s+(?:said|think|thought|meant)"
            r"|my\s+(?:stance|opinion|view)\s+on"
            r"|my\s+mind\s+(?:about|on)"
            r"|about|on|regarding|of|to|my\s+opinion\s+on)\s+", "", tail)
        topic = self._opinion_topic(tail)
        # FIX (round v-aug06b): end-anchored retraction idioms ("... olives
        # ... they're not that bad, i was too hasty") leave an EMPTY tail, so
        # the normal topic scan finds nothing. Before giving up, scan the
        # whole utterance for a held stance whose key appears as a content
        # word and reverse THAT. Generic: resolves against the live store,
        # no per-topic table. Only when no topic is resolvable from the tail.
        # FIX (round v-aug06d): match by TOKEN CONTAINMENT, not whole-key
        # substring. The user recants "letterpress" but the stored stance key
        # is "letterpress printing"; the old " letterpress printing " in q
        # check failed because the utterance only contains "letterpress". Now
        # any token of the key appearing as a whole word in q resolves it
        # (generic: works for any multiword stance key, no per-topic rule).
        if not topic:
            _target = self._stance_key_in_text(q)
            if _target is None:
                _whole = self._opinion_topic(q)
                if _whole:
                    _target = self.opinions.resolve_topic(_whole)
            if _target is None:
                return
            try:
                self.opinions._soft_reversal = _soft
                self.opinions.reverse_stance(_target)
            except Exception:
                pass
            return
        # D4 fix (round v-aug06): the retraction's real topic is often buried
        # INSIDE a follow-on clause, not in the head of the tail. E.g.
        # "i take it back — i've grown to hate running" → tail="i've grown to
        # hate running", and _opinion_topic returns "i've grown" (a pronoun +
        # auxiliary), which resolves to NO prior stance, so the contradiction
        # is silently dropped and the stale positive stance persists. Fix:
        # when the head is non-content (pronoun/auxiliary-led) but the tail
        # contains an explicit attitude verb + object ("hate running", "like
        # x"), extract the object of THAT verb as the topic. This reads the
        # attitude actually being retracted, not the filler grammar. Generic,
        # verb-driven (no per-topic table) — generalizes to any retracted
        # attitude the user names.
        if topic.split()[0] in ("ive", "i've", "i", "im", "i'm", "ive", "he",
                                 "she", "they", "we", "you") or topic in (
                "grown", "changed", "wrong", "back", "corrected"):
            _rev_m = re.search(
                r"\b(?:hate|dislike|detest|can't\s+stand|love|like|enjoy|"
                r"prefer|care\s+for|believe|think)\b\s+(?:that\s+|to\s+)?"
                r"(.+?)(?:\.|\band\b|\bbut\b|$|,)", tail)
            if _rev_m:
                _rev_topic = self._opinion_topic(_rev_m.group(1).strip())
                if _rev_topic:
                    topic = _rev_topic
        target = self.opinions.resolve_topic(topic) or self.opinions.resolve_topic(tail)
        # FIX (round v-aug06b): some retraction idioms are END-anchored — the
        # topic the user is recanting sits BEFORE the cue, not after it
        # ("... olives ... they're not that bad, i was too hasty"). The tail
        # after such a cue is empty, so the scan above finds nothing and the
        # stale stance persists. Fallback: scan the whole utterance for the
        # single held stance whose key appears as a content word, and reverse
        # THAT. Generic — no per-topic table; resolves against the live store.
        # FIX (round v-aug06d): use token-containment matching (see above) so a
        # recant of "letterpress" reverses the stored "letterpress printing".
        if target is None:
            target = self._stance_key_in_text(q)
            # also try resolving the whole-utterance content head
            if target is None:
                _whole = self._opinion_topic(q)
                if _whole:
                    target = self.opinions.resolve_topic(_whole)
        if target is None:
            return
        try:
            self.opinions._soft_reversal = _soft
            self.opinions.reverse_stance(target)
        except Exception:
            pass

    def observe_user_query(self, query: str, subject: str, valence: float):
        subject_lower = subject.lower()
        self.topic_interaction_count[subject_lower] = self.topic_interaction_count.get(subject_lower, 0) + 1
        self.learning_goals[subject_lower] = self.learning_goals.get(subject_lower, 0) + 1
        if subject_lower:
            self.knowledge_model[subject_lower] = min(
                1.0, self.knowledge_model.get(subject_lower, 0.0) + 0.1)
        current_rapport = self.emotional_rapport.get(subject_lower, 0.0)
        self.emotional_rapport[subject_lower] = current_rapport + 0.2 * (valence - current_rapport)

        q_clean = query.lower().strip(" ?!.")
        m_like = re.search(r"\bi\s+(?:like|love)\s+(.+)", q_clean, re.IGNORECASE)
        if m_like:
            thing = m_like.group(1).strip(" .!?")
            if thing and thing not in ("you", "it", "that", "this", "them", "him", "her", "me", "something", "everything", "anything"):
                if "likes" not in self.preferences:
                    self.preferences["likes"] = []
                if thing not in self.preferences["likes"]:
                    self.preferences["likes"].append(thing)

        m_interest = re.search(r"\bi\s+(?:want\s+to\s+learn\s+about|am\s+interested\s+in|'m\s+interested\s+in)\s+(.+)", q_clean, re.IGNORECASE)
        if m_interest:
            thing = m_interest.group(1).strip(" .!?")
            if thing and thing not in ("you", "it", "that", "this", "them", "him", "her", "me"):
                if "interests" not in self.preferences:
                    self.preferences["interests"] = []
                if thing not in self.preferences["interests"]:
                    self.preferences["interests"].append(thing)

        m_fav = re.search(r"\bmy\s+favorite\s+(.+?)\s+is\s+(.+)", q_clean, re.IGNORECASE)
        if m_fav:
            category = m_fav.group(1).strip(" .!?")
            val = m_fav.group(2).strip(" .!?")
            if category and val:
                if "favorites" not in self.preferences:
                    self.preferences["favorites"] = {}
                self.preferences["favorites"][category] = val

        m_name = re.search(
            r"\b(?:my\s+name\s+is|i\s+am\s+called|call\s+me)\s+"
            r"([^.,!?]+?)(?:\s+(?:and|but|,|\.|$))",
            q_clean, re.IGNORECASE)
        if not m_name:
            m_name = re.search(r"\b(?:do\s+you\s+know\s+my\s+name|know\s+my\s+name|is\s+my\s+name)\s+is\s+(.+)", q_clean, re.IGNORECASE)
        # Biographical location / origin: "I live in X" / "I am from X" /
        # "I was born in X". Captured into user_location / user_background
        # so a later "where do I live?" / "where are you from?" recalls it.
        # Mine biographical + general personal facts into the learned store.
        # Extracted so the same-turn identity gate can call it with only the
        # raw text (subject isn't assigned yet in process_turn there).
        self.mine_personal_facts(query)

        self._update_cognitive_style(query)
        if subject_lower != self.last_topic and self.last_topic:
            self.topic_followup_count[self.last_topic] = max(0, self.topic_followup_count.get(self.last_topic, 0) - 1)
            self.turn_since_topic_change = 0
        else:
            self.turn_since_topic_change += 1
            if self.last_topic:
                self.topic_followup_count[self.last_topic] = self.topic_followup_count.get(self.last_topic, 0) + 1
        self.last_topic = subject_lower

        total_interactions = sum(self.topic_interaction_count.values())
        total_followups = sum(self.topic_followup_count.values())
        self.conversation_depth = total_followups / max(1, len(self.topic_interaction_count))
        self.engagement_level = min(1.0, 0.3 + 0.7 * (total_followups / max(1, total_interactions)))

        self.interaction_count += 1
        self.personal_facts.advance_turn()
        self.opinions.advance_turn()
        self.relationship_depth = min(1.0, self.interaction_count / 20.0)

        inferred = self.infer_user_goal(query)
        self.last_goal = inferred
        self.goals.append(inferred)
        if len(self.goals) > 50:
            self.goals = self.goals[-50:]

        emotion_vad = self._infer_user_emotion(query)
        self._record_interaction(query, subject, emotion_vad)

        # ── ACC analog: Detect correction patterns ──
        self._detect_correction(query, subject, valence)

    def _detect_correction(self, query: str, subject: str, valence: float):
        """ACC conflict detection: detect that the user is correcting RAVANA.
        
        Three detection streams:
        1. Direct: explicit "no", "that's wrong", etc.
        2. Sentiment drop: valence drops significantly after response
        3. Re-ask: user repeats similar query within 3 turns
        """
        self.detected_correction = False
        self.detected_correction_type = None
        self.correction_severity = 0.0
        self.correction_subject = subject
        self.detected_correction_fact = None

        q_clean = query.lower().strip()

        # Stream 1: Direct correction patterns
        for pattern in _CORRECTION_DIRECT_PATTERNS:
            if re.search(pattern, q_clean, re.IGNORECASE):
                self.detected_correction = True
                self.detected_correction_type = CorrectionType.DIRECT
                self.correction_severity = max(self.correction_severity, 0.5)
                break

        # Stream 2: Sentiment drop — valence drops significantly from previous turn
        prev_valence = self._last_user_valence_before_response
        if prev_valence > 0 and (prev_valence - valence) > 0.4:
            self.detected_correction = True
            if self.detected_correction_type != CorrectionType.DIRECT:
                self.detected_correction_type = CorrectionType.SENTIMENT_DROP
            self.correction_severity = max(self.correction_severity, 
                                             min(0.8, (prev_valence - valence) * 1.5))

        # Stream 3: Re-ask — similar query within 3 turns
        if self._previous_user_query:
            prev_words = set(self._previous_user_query.lower().split())
            curr_words = set(q_clean.split())
            overlap = len(prev_words & curr_words) / max(1, len(prev_words | curr_words))
            if overlap > 0.6 and len(prev_words) >= 3:
                self.detected_correction = True
                if self.detected_correction_type not in (CorrectionType.DIRECT, CorrectionType.SENTIMENT_DROP):
                    self.detected_correction_type = CorrectionType.INDIRECT_REASK
                self.correction_severity = max(self.correction_severity, 0.3)

        # Extract corrected fact if user provides one. D3 (round v4): run
        # unconditionally, not only after a direct/sentiment/reask signal.
        # A name/value correction ("my sister's name is not meena, it's priya")
        # has NO direct-correction cue word, so gating on detected_correction
        # made the name extractor dead — the fact was never extracted. The
        # extractor self-signals via the name pattern, so let it set the flag.
        self._extract_correction_fact(query, subject)

        self._previous_user_query = q_clean

    def _extract_correction_fact(self, query: str, subject: str):
        """Extract (subject, relation, correct_value) from correction sentence.
        E.g. \"2+2 is 4, not 5\" → (\"2+2\", \"is\", \"4\")
        """
        q_clean = query.lower().strip()
        # D3 (round v3): "X's name is not Y, it's Z" / "X is not Y, it's Z".
        # The corrected value is the token after "it's" (group 2); the subject
        # attribute is group 1 (e.g. "sister's"). The negation word "not" is
        # NEVER stored as a value — that was the D-A bug (corrected value was
        # "not"). Handled before the generic loop because its group order
        # differs.
        # D3 (round v4): "X is not Y, it is/it's Z" — the NEGATION-FIRST shape
        # ("my dog is not max, it is rocky"). The corrected value is the token
        # after "it is"/"it's" (last group); the subject attribute is the noun
        # before "is not". The existing _CORRECTION_FACT_PATTERNS only handle
        # "X is Y, not Z" (negation LAST), so this dominant natural shape was
        # never extracted — corrections about pets/things/places were lost.
        # The negation word is NEVER stored as a value. Handled before the
        # generic loop because its group order (attr, correct_val) differs.
        _notfirst = re.search(
            r"(?:my|the|a|an)?\s*([\w'-]+?)'\s*is\s+not\s+[\w'-]+"
            r"[,.]*\s+(?:it'?s|it\s+is)\s+([\w'-]+)", q_clean, re.IGNORECASE)
        if _notfirst is None:
            _notfirst = re.search(
                r"([\w'-]+)\s+is\s+not\s+[\w'-]+[,.]*\s+"
                r"(?:it'?s|it\s+is)\s+([\w'-]+)", q_clean, re.IGNORECASE)
        if _notfirst:
            _subj_attr = _notfirst.group(1).strip().removesuffix("'s")
            _correct_val = _notfirst.group(2).strip()
            if _correct_val:
                self.detected_correction = True
                self.detected_correction_fact = ("i", _subj_attr, _correct_val)
                self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                self.correction_severity = max(self.correction_severity, 0.8)
                return
        _nm = re.search(_CORRECTION_NAME_FACT_PATTERN, q_clean, re.IGNORECASE)
        if _nm:
            _subj_attr = _nm.group(1).strip()
            _correct_val = _nm.group(2).strip()
            self.detected_correction_fact = ("i", _subj_attr, _correct_val)
            self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
            self.correction_severity = max(self.correction_severity, 0.8)
            return
        for pattern in _CORRECTION_FACT_PATTERNS:
            m = re.search(pattern, q_clean, re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 2:
                    # it's X, not Y
                    correct_val = groups[0]
                    wrong_val = groups[1]
                    self.detected_correction_fact = (subject, "is", correct_val)
                    self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                    self.correction_severity = max(self.correction_severity, 0.7)
                elif len(groups) == 3:
                    # X is Y, not Z
                    fact_subject = groups[0].strip()
                    correct_val = groups[1]
                    wrong_val = groups[2]
                    self.detected_correction_fact = (fact_subject, "is", correct_val)
                    self.detected_correction_type = CorrectionType.CORRECTION_WITH_FACT
                    self.correction_severity = max(self.correction_severity, 0.8)

    def store_response_for_correction(self, response: str, strategy: str, valence: float):
        """Store the last response so correction detection can reference it."""
        self._last_response_for_correction = response
        self._last_response_strategy_for_correction = strategy
        self._last_user_valence_before_response = valence

    def reset_correction_flags(self):
        """Reset correction flags for next turn."""
        self.detected_correction = False
        self.detected_correction_type = None
        self.correction_severity = 0.0
        self.correction_subject = ""
        self.detected_correction_fact = None

    # D2 (round v2): closed-class / function words that can never own a stance.
    # A stance must land on a real CONTENT concept ("talk", "solitude"),
    # never on a determiner/adverb ("the", "small", "how"). This is a minimal
    # universal seed (closed-class set is language-universal and tiny), NOT a
    # per-topic table — it generalizes to any topic the user names.
    _OPINION_STOP = {
        "the", "a", "an", "my", "your", "our", "their", "his", "her", "its",
        "this", "that", "these", "those", "some", "any", "no", "all", "every",
        "i", "you", "he", "she", "we", "they", "me", "him", "us", "them",
        "and", "but", "or", "so", "if", "when", "while", "because", "of",
        "to", "in", "on", "at", "for", "with", "from", "by", "as", "into",
        "about", "over", "under", "how", "what", "why", "who", "where",
        "really", "very", "just", "only", "also", "too", "quite", "more",
        "most", "much", "many", "such", "own", "same", "other", "another",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "not", "don't", "dont", "do", "does", "did", "can", "cannot", "cant",
        "it", "they're", "im", "i'm", "you're", "we're", "there",
    }

    def _opinion_topic(self, phrase: str) -> Optional[str]:
        """Resolve the salient CONTENT HEAD of an opinion-object phrase.

        Strips leading determiners and trailing modifiers and CUTS at the
        first internal closed-class word (preposition/conjunction) so:
          "the solitude of the lighthouse" -> "solitude"
          "small talk at the village market" -> "small talk"
          "accordion when the wind dies down" -> "accordion"
          "how whales communicate" -> "whales"
        Returns the joined head phrase (one or more content nouns), or None
        if the phrase is all closed-class words (so we never seed a garbage
        stance on a function word like "the"/"how"/"small").
        """
        toks = [t for t in re.findall(r"[a-z'][a-z']*", phrase.lower())]
        if not toks:
            return None
        # Drop leading closed-class words (determiners/prepositions).
        while toks and toks[0] in self._OPINION_STOP:
            toks.pop(0)
        if not toks:
            return None
        # Cut at the first internal closed-class word so a compound head
        # ("small talk") is kept whole but trailing prepositional spans
        # ("of the lighthouse", "at the market") are dropped.
        head = []
        for t in toks:
            if t in self._OPINION_STOP:
                break
            head.append(t)
        if not head:
            return None
        # Drop trailing closed-class/modifier words as a final safety.
        while len(head) > 1 and head[-1] in self._OPINION_STOP:
            head.pop()
        if not head:
            return None
        return " ".join(head)

    def _ensure_emotion_detector(self):
        if self._emotion_detector is None or not hasattr(self._emotion_detector, '_vad_matrix'):
            from ravana.core import UserEmotionDetector
            self._emotion_detector = UserEmotionDetector()

    def _infer_user_emotion(self, text: str) -> Tuple[float, float, float]:
        self._ensure_emotion_detector()
        v, a, d = self._emotion_detector.detect(text)
        rate = 0.35
        prev = self.emotional_state
        self.emotional_state = {
            'valence': prev['valence'] + rate * (v - prev['valence']),
            'arousal': prev['arousal'] + rate * (a - prev['arousal']),
            'dominance': prev['dominance'] + rate * (d - prev['dominance']),
        }
        return (v, a, d)

    def _record_interaction(self, text: str, subject: str,
                            emotion_vad: Tuple[float, float, float]):
        self.interaction_history.append({
            'text': text[:200],
            'subject': subject,
            'valence': emotion_vad[0],
            'arousal': emotion_vad[1],
            'dominance': emotion_vad[2],
            'turn': len(self.interaction_history),
        })
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

    def infer_user_goal(self, query: str) -> str:
        q = query.lower().strip()
        debug_markers = ('broken', "doesn't work", "doesn't work", 'error', 'fail',
                         'bug', 'crash', 'wrong', 'stuck', 'issue', 'fix', 'not working',
                         "isn't working", 'exception', 'traceback')
        if any(m in q for m in debug_markers) or q.startswith('why is') and any(
            m in q for m in ('broken', 'error', 'fail', 'wrong', 'crash')):
            return "DEBUGGING"
        learn_markers = ('how does', 'how do', 'how is', 'how are', 'what is', 'what are',
                         'explain', 'how come', 'why does', 'why do')
        if any(q.startswith(m) or (' ' + m) in q for m in learn_markers):
            return "LEARNING"
        explore_markers = ('tell me about', "let's talk about", 'i want to know',
                           'i wonder', 'teach me', 'show me', 'describe')
        if any(m in q for m in explore_markers):
            return "EXPLORING"
        return "EXPLORING"

    def _update_cognitive_style(self, query: str):
        q_lower = query.lower()
        style_scores = {
            'curious': sum(1 for w in ['why', 'how', 'what', 'explain', 'understand', 'curious', 'wonder'] if w in q_lower),
            'skeptical': sum(1 for w in ['really', 'actually', 'prove', 'evidence', 'doubt', 'sure', 'fake', 'lie'] if w in q_lower),
            'practical': sum(1 for w in ['how to', 'build', 'make', 'create', 'step', 'guide', 'tutorial', 'implement'] if w in q_lower),
        }
        if style_scores:
            top_style = max(style_scores, key=style_scores.get)
            if style_scores[top_style] > 0:
                self.cognitive_style = top_style

    def infer_topic_interest(self, topic: str) -> float:
        t = topic.lower()
        goal_strength = min(1.0, self.learning_goals.get(t, 0) * 0.2)
        rapport = (self.emotional_rapport.get(t, 0.0) + 1.0) / 2.0
        interaction = min(1.0, self.topic_interaction_count.get(t, 0) * 0.1)
        return (goal_strength * 0.4 + rapport * 0.4 + interaction * 0.2)

    def infer_user_knows(self, concept: str) -> float:
        return self.knowledge_model.get(concept.lower(), 0.0)

    def infer_user_wants_to_learn(self, concept: str) -> float:
        t = concept.lower()
        goal = min(1.0, self.learning_goals.get(t, 0) * 0.15)
        rapport = (self.emotional_rapport.get(t, 0.0) + 1.0) / 2.0
        return max(0.0, goal * 0.6 + rapport * 0.4 - self.knowledge_model.get(t, 0.0) * 0.5)

    def get_preferred_relation_types(self) -> List[str]:
        rel_counts = {}
        for (f, t), count in self.edge_reactivations.items():
            rel = 'semantic'
            rel_counts[rel] = rel_counts.get(rel, 0) + count
        return sorted(rel_counts, key=rel_counts.get, reverse=True)[:3]

    def inferred_preferences(self, threshold: int = 2) -> Dict[Tuple[str, str], int]:
        return {(f, t): c for (f, t), c in self.edge_reactivations.items()
                if c >= threshold}

    def activation_boost_for(self, concept: str) -> Dict[str, float]:
        boost: Dict[str, float] = {}
        cl = concept.lower()
        for (from_c, to_c), count in self.edge_reactivations.items():
            if from_c == cl:
                boost[to_c] = 1.0 + (count / (count + 1.0)) * 0.3
        return boost

    def get_state(self) -> Dict:
        return {
            'edge_reactivations': {str(k): v for k, v in self.edge_reactivations.items()},
            'query_concepts': list(self.query_concepts),
            'knowledge_model': self.knowledge_model,
            'learning_goals': self.learning_goals,
            'emotional_rapport': self.emotional_rapport,
            'cognitive_style': self.cognitive_style,
            'engagement_level': self.engagement_level,
            'conversation_depth': self.conversation_depth,
            'topic_interaction_count': self.topic_interaction_count,
            'topic_followup_count': self.topic_followup_count,
            'last_topic': self.last_topic,
            'turn_since_topic_change': self.turn_since_topic_change,
            'interaction_count': self.interaction_count,
            'relationship_depth': self.relationship_depth,
            'goals': self.goals,
            'last_goal': self.last_goal,
            'user_name': self.user_name,
            'user_location': self.user_location,
            'user_background': self.user_background,
            'preferences': self.preferences,
            'personal_facts': self.personal_facts.get_state(),
            'opinions': self.opinions.get_state(),
            'emotional_state': self.emotional_state,
            'belief_state': self.belief_state,
            'interaction_history': self.interaction_history,
        }

    def set_state(self, state: Dict):
        self.edge_reactivations = {eval(k): v for k, v in state.get('edge_reactivations', {}).items()}
        self.query_concepts = set(state.get('query_concepts', []))
        self.knowledge_model = state.get('knowledge_model', {})
        self.learning_goals = state.get('learning_goals', {})
        self.emotional_rapport = state.get('emotional_rapport', {})
        self.cognitive_style = state.get('cognitive_style', 'balanced')
        self.engagement_level = state.get('engagement_level', 0.5)
        self.conversation_depth = state.get('conversation_depth', 0.0)
        self.topic_interaction_count = state.get('topic_interaction_count', {})
        self.topic_followup_count = state.get('topic_followup_count', {})
        self.last_topic = state.get('last_topic', '')
        self.turn_since_topic_change = state.get('turn_since_topic_change', 0)
        self.interaction_count = state.get('interaction_count', 0)
        self.relationship_depth = state.get('relationship_depth', 0.0)
        self.goals = state.get('goals', [])
        self.last_goal = state.get('last_goal', 'EXPLORING')
        self.user_name = state.get('user_name', '')
        self.user_location = state.get('user_location', '')
        self.user_background = state.get('user_background', '')
        self.preferences = state.get('preferences', {})
        _pf = state.get('personal_facts')
        if _pf:
            self.personal_facts.set_state(_pf)
        _op = state.get('opinions')
        if _op:
            self.opinions.set_state(_op)
        self.emotional_state = state.get('emotional_state',
            {'valence': 0.0, 'arousal': 0.3, 'dominance': 0.5})
        self.belief_state = state.get('belief_state', {})
        self.interaction_history = state.get('interaction_history', [])


# ── Module-level persistence helpers ───────────────────────────────────────────
def _user_model_path(user_suffix: str = "") -> str:
    """Path for a per-user-model store file under <repo>/user_models/."""
    os.makedirs(USER_MODELS_DIR, exist_ok=True)
    return os.path.join(USER_MODELS_DIR, f"ravana_usermodel{user_suffix}.pkl")


def save_user_model(user_model: "UserModel", user_suffix: str = "") -> str:
    """Persist a UserModel to its own dedicated file. Returns the path."""
    path = _user_model_path(user_suffix)
    with open(path, "wb") as f:
        pickle.dump(user_model, f)
    return path


def load_user_model(user_suffix: str = "") -> "UserModel":
    """Load a UserModel from its dedicated file, or return a fresh one."""
    path = _user_model_path(user_suffix)
    if not os.path.exists(path):
        return UserModel()
    with open(path, "rb") as f:
        um = pickle.load(f)
    if not hasattr(um, "personal_facts"):
        from .personal_fact_store import PersonalFactStore, UserStanceStore
        um.personal_facts = PersonalFactStore()
        um.opinions = UserStanceStore()
    return um
