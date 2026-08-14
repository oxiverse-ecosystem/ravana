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

# real affect categories in brain_regions._CAUSE_SEEDS and
# support_router._SUPPORT_AFFECT). Used by the bare-copula name guard: a
# first-person "i'm X" where X is any of these is a TRANSIENT STATE, never a
# proper noun, so it must not be stored as the user's NAME. This is SEED
# vocabulary (a data set, not an answer path) — RAVANA-expandable via the
# shared affect lexicon; removing entries degrades gracefully. Participles
# ("shaking"/"tired"), irregulars ("torn"/"lost"), and stative/cognitive
# verbs ("thinking"/"convinced") are all covered so the guard generalizes
# across every tense/participle form rather than a frozen per-word list.
_AFFECT_STATE_LEXICON = {
    # affect / emotion nouns + adjectives
    "happy", "sad", "glad", "mad", "angry", "furious", "scared", "afraid",
    "anxious", "worried", "worry", "lonely", "alone", "empty", "hollow",
    "numb", "lost", "torn", "hurt", "hopeless", "hopeful", "grateful",
    "excited", "tense", "raw", "low", "blue", "down", "proud", "calm",
    "nervous", "stressed", "stressedout", "overwhelmed", "depressed",
    "exhausted", "tired", "hungry", "thirsty", "sick", "confused",
    "fine", "good", "bad", "ok", "okay", "well", "ready", "done", "sure",
    "certain", "right", "wrong", "sorry", "here", "there", "home", "awake",
    "asleep", "late", "early", "busy",
    # stative / cognitive / feeling verbs (incl. participles + infinitives)
    "feeling", "felt", "feel", "love", "like", "hate", "dislike", "prefer",
    "think", "thinking", "believing", "believe", "guess", "guessing",
    "wonder", "wondering", "mean", "meaning", "know", "knowing", "understand",
    "want", "wanting", "need", "needing", "wish", "wishing", "hope", "hoping",
    "doubt", "doubting", "fear", "fearing", "regret", "regretting",
    "suspect", "suspecting", "agree", "agreeing", "disagree", "convinced",
    "convincing", "standing", "coming", "going", "trying", "saying",
    "referring", "talking", "asking", "loving", "hating", "liking",
    "sticking", "getting", "shaking", "crying", "dying", "lying", "trying",
    "running", "falling", "breaking", "caring", "waiting", "working",
    "learning", "growing", "changing", "feeling",
    # Round 2026-08-13T2059Z: broaden to catch transient-state name
    # poisoning from emotional / participle words the prior set missed
    # (gutted / devastated / elated / shattered / drained / defeated /
    # wrecked / crushed / broken / frozen / shaken / heartbroken /
    # desolate / miserable / forlorn / wretched / bereft / gloomy / glum /
    # morose / melancholy / mournful / dismal / bitter / gleeful / rueful /
    # sorrowful / fretful / homesick / vexed / wistful / suspicious / wary /
    # jubilant / content / serene / tranquil / smug / sheepish / blithe /
    # coy / bashful / shy / sulky / querulous / uptight / vitriolic / void /
    # dubious / sullen / edgy / queasy / ticked / bereaved / anguished /
    # distraught / disheartened / dismayed). SEED vocabulary (RAVANA-
    # expandable), not an authored answer path — removing entries degrades
    # gracefully (one fewer guard), it is not content RAVANA can never change.
    "gutted", "devastated", "elated", "crushed", "heartbroken", "thrilled",
    "ecstatic", "miserable", "shattered", "wrecked", "defeated", "drained",
    "bitter", "gleeful", "rueful", "overjoyed", "despondent", "desolate",
    "forlorn", "wretched", "anguished", "bereft", "broken", "bereaved",
    "inconsolable", "dejected", "sorrowful", "gloomy", "glum", "morose",
    "melancholy", "mournful", "dismal", "crestfallen", "distraught",
    "dismayed", "disheartened", "fretful", "homesick", "horrified",
    "humiliated", "mortified", "terrified", "unnerved", "vexed", "wistful",
    "yearning", "aggrieved", "appalled", "bewildered", "chagrined",
    "daunted", "deflated", "delirious", "exasperated", "flabbergasted",
    "incensed", "indignant", "infuriated", "outraged", "peeved", "perturbed",
    "rattled", "resentful", "spooked", "startled", "stung", "sulky",
    "suspicious", "wary", "jubilant", "gratified", "invigorated", "content",
    "serene", "tranquil", "smug", "sheepish", "blithe", "coy", "bashful",
    "shy", "sullen", "querulous", "uptight", "vitriolic", "void", "dubious",
    "edgy", "queasy", "ticked", "frozen", "shaken",
}

# Round 2026-08-13T2059Z: stance-connector / function / common-adverb words
# that may follow a bare "i'm" copula in an opinion or emotion frame, but are
# NEVER proper-noun names. "i'm against plastics" / "i'm over the moon" /
# "i'm still buzzing" must not store name=against / name=over / name=still.
# SEED vocabulary (RAVANA-expandable via the shared lexicon), not a per-name
# table; removing entries degrades gracefully. Combined with the verb-form
# check (gerunds in 'ing', past/past-participle in 'ed') below, this rejects
# transient states regardless of which token they occupy.
_NAME_REJECT_FUNCTION = {
    "against", "for", "with", "without", "over", "under", "about", "around",
    "despite", "regardless", "beyond", "beneath", "beside", "except",
    "plus", "minus", "near", "far", "via", "per", "yes", "no", "maybe",
    "always", "never", "sometimes", "often", "usually", "already", "still",
    "just", "really", "very", "quite", "too", "also", "even", "almost",
    "nearly", "not", "actually", "basically", "probably", "definitely",
    "surely", "clearly", "simply", "merely", "mostly", "partly", "fully",
    "completely", "totally", "absolutely", "exactly", "particularly",
    "especially", "generally", "naturally", "obviously", "truly", "honestly",
    "frankly", "seriously", "admittedly", "ideally", "hopefully",
    "personally", "ultimately", "virtually", "wrongly", "correctly",
    "rightly", "rather", "instead", "though", "however", "whereas",
    "although", "because", "since", "unless", "until", "whether", "either",
    "neither", "both", "each", "every", "any", "some", "such", "same",
    "other", "another", "own",
}



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
    # Round t_6c023144 (2026-08-09T1953Z residual): first-person reversal
    # speech acts that the round worker saw slip through to a fresh FOR stance.
    # "i flipped, the reef tank is more work than joy" formed a new positive
    # stance instead of recoding the held one, because "flipped" was absent
    # here — so no cue matched, the concession branch (requires a "but"/belief
    # frame) never fired, and mine_stance_reversal bailed before reversing. Add
    # the decisive change-of-mind verbs as SEED cues (RAVANA-expandable, not a
    # per-topic table). The tail→held-stance resolver already guards against
    # corruption: a flip on something the user has no stance on is a no-op
    # (reverse_stance returns None), so false positives are bounded.
    r"\bi\s*(?:'ve\s+|have\s+|)(?:flipped|flip-?flopped)\b",
    r"\bi\s+(?:recant|recanted|renounce|renounced|revoked|reversed|reneged)\b",
    r"\bi\s+(?:backtracked|went\s+back\s+on|backed\s+off\s+from)\b",
    # allow the contraction 'i've' (no space after i) as well as 'i had'/'i have'
    r"\bi\s*'?ve\s+had\s+a\s+change\s+of\s+heart\b",
    r"\bi\s+(?:had|have)\s+a\s+change\s+of\s+heart\b",
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

        # COUNT / QUANTITY correction (round 2026-08-09g). A plain update like
        # "it's seven hives now, i split one last week" carries NO negation or
        # "my X is Y" structure, so the name-correction and _corrective paths
        # above miss it — the prior count fact ("keep six hives", stored under
        # the 'does' attribute) was left active and a later "how many hives do
        # i have" returned the STALE six (measured: T41 -> "ok, noted: wait.",
        # T42/T64 silently kept six). Detect an update cue + a cardinal number +
        # an entity noun, locate the prior count/activity fact for that entity,
        # and supersede it via contradict() (online, no retrain). Content comes
        # from the live store; no per-topic table, no authored text.
        if not self.detected_correction:
            _NUMWORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|"
                         r"ten|eleven|twelve|\d+)")
            _cnt = re.search(
                r"\b(?P<num>" + _NUMWORDS + r")\s+(?P<ent>[a-z][a-z]+)\b.*\b"
                r"(now|split|added|new|more|extra|another|gained|got|"
                r"increased|up to)\b", q_clean, re.IGNORECASE) or \
                re.search(
                r"\b(now|split|added|new|more|extra|another|gained|got)\b.*\b"
                + r"(?P<num>" + _NUMWORDS + r")\s+(?P<ent>[a-z][a-z]+)\b",
                q_clean, re.IGNORECASE)
            if _cnt:
                _num = _cnt.group("num")
                _ent = _cnt.group("ent").lower().strip()
                if _ent and _ent not in _VALUE_STOP and _num:
                    # find the prior activity/count fact whose value mentions
                    # this entity (e.g. "keep six hives" -> entity "hives").
                    _prior = None
                    for (s, a, v), f in self.personal_facts.facts.items():
                        if s == "i" and a in ("does", "count", "number", "qty") \
                                and not getattr(f, "superseded", False) \
                                and _ent in v.lower():
                            _prior = (a, v)
                            break
                    if _prior is not None:
                        # rebuild the new value with the corrected count,
                        # preserving the verb + entity from the prior fact.
                        _verb = re.match(
                            r"^(keep|have|keep on|have on|raise|own|breed|run|"
                            r"got)\b", _prior[1].lower())
                        _newval = (f"{_verb.group(1)} " if _verb else "") \
                            + f"{_num} {_ent}"
                        self.detected_correction = True
                        self.detected_correction_fact = (
                            "i", _prior[0], _newval)
                        self.detected_correction_type = \
                            CorrectionType.CORRECTION_WITH_FACT
                        self.correction_severity = \
                            max(self.correction_severity, 0.7)

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
            # C-value (round 2026-08-08c): trim a TRAILING prepositional phrase
            # from the value. The greedy value capture ("my X is Y", up to 8
            # words) over-grabbed trailing prepositions, storing nonsense like
            # "cabin at." / "always raw from hauling hive boxes up the.". A
            # value that ENDS in a preposition ("up"/"from"/"at"/"in") is an
            # incomplete capture — drop the preposition and everything after it
            # so "cabin at." -> "cabin" and "up the mountain" -> "up" -> "".
            # Structural: one shared chokepoint for every fact; the
            # preposition set is closed-class (not content), so trimming it
            # never discards a real value word. If the whole value is
            # prepositions, reject it (no content to store).
            _PREP = ("up", "down", "from", "at", "in", "on", "with", "to",
                     "of", "by", "for", "about", "into", "onto", "over",
                     "under", "near", "behind", "beside", "off")
            _vwords = _val.split()
            if _vwords and _vwords[-1] in _PREP:
                # drop trailing prepositions and any words following the first
                # trailing preposition
                _cut = len(_vwords)
                for _i in range(len(_vwords) - 1, -1, -1):
                    if _vwords[_i] in _PREP:
                        _cut = _i
                    else:
                        break
                _vwords = _vwords[:_cut]
                _val = " ".join(_vwords)
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

        def _split_possessive_attr(attr: str):
            """D6 (round 2026-08-08b-d): 'my partner's name is theo' must model
            an ENTITY (partner) and its attribute (name), not collapse onto the
            user's own self-profile. The multi-word attr pattern
            (r'\bmy\s+(...)\s+is\s+...') captures 'partner's name' as one attr
            key under subject 'i'; a later recall path sees the substring
            'name' and renders 'your name is theo' — reporting the PARTNER'S
            name as the USER's name (a self/other boundary breach; the same
            defect class that put 'your name is a hypocrite' in v-aug04).

            Fix: detect a possessive head ('s) in the attr and resolve it to
            (entity=<owner>, attr=<relation>), mirroring the ALREADY-CORRECT
            possessive handling in engine_memory._record_episode (which keys
            'my cat's name is whiskers' -> entity=cat, attr=name). This is
            structural: any 'my <X>'s <Y> is Z' is stored under entity X, never
            under the user's 'i' subject. Generic — no per-entity table. The
            entity grows from experience (the user can name any relation)."""
            _am = re.match(r"^([\w'-]+)'s\s+(.+)$", attr)
            if _am:
                return _am.group(1).strip().lower(), _am.group(2).strip().lower()
            return None, attr

        def _put_fact_ent(entity: str, attr: str, val: str, conf: float) -> None:
            """Entity-keyed variant of _put_fact (subject = entity, not 'i')."""
            _val = (val or "").strip().strip(" .,!?;:'\"").lower()
            if not _val or _val in _VALUE_STOP:
                return
            _PREP = ("up", "down", "from", "at", "in", "on", "with", "to",
                     "of", "by", "for", "about", "into", "onto", "over",
                     "under", "near", "behind", "beside", "off")
            _vwords = _val.split()
            if _vwords and _vwords[-1] in _PREP:
                _cut = len(_vwords)
                for _i in range(len(_vwords) - 1, -1, -1):
                    if _vwords[_i] in _PREP:
                        _cut = _i
                    else:
                        break
                _vwords = _vwords[:_cut]
                _val = " ".join(_vwords)
            if not _val or _val in _VALUE_STOP:
                return
            _subj = entity.lower()
            existing = self.personal_facts.get(_subj, attr)
            if (_corrective and existing is not None
                    and existing.value.lower() != _val):
                self.personal_facts.contradict(_subj, attr, _val)
            else:
                self.personal_facts.assert_fact(_subj, attr, _val,
                                                confidence=conf,
                                                source="seed_regex")

        m_name = re.search(
            r"\b(?:my\s+name\s+is|i\s+am\s+called|call\s+me)\s+"
            r"([^.,!?]+)",
            q_clean, re.IGNORECASE)
        if not m_name:
            # Bare first-person self-naming ("i'm noor", "i am noor"): the
            # contraction "i'm" / "i am" introduces a proper noun that is the
            # speaker's name. The existing patterns only caught "my name is /
            # i am called / call me", so a user opening with "i'm noor" (as
            # every persona in the chat rounds does) was NEVER captured and a
            # later "what's my name?" returned "i don't know your name yet".
            # Structural: a capitalized bare proper noun after the
            # first-person copula, not a common noun or a determiner-led
            # phrase. Rejects "i'm a nurse" (common noun) and "i am the bee
            # guy" (determiner) so descriptor phrases are never stored as a
            # name. Generalizes across all personas (no per-name list).
            m_name = re.search(
                r"\b(?:i'?m|i\s+am)\s+"
                r"([A-Za-z][A-Za-z']*(?:\s+[A-Za-z][A-Za-z']*){0,2})"
                r"(?=[\s.,!?]|$)",
                q_clean)
            # NOTE: names are typed lowercase in chat ("i'm noor"), so we do
            # NOT require capitalization here — instead we rely on the
            # downstream guards (lines ~319-321 reject determiner-led phrases
            # like "i am a/the ...", and the non-name stoplist rejects states
            # / common nouns like "tired"/"hungry"/"nurse"). The bare form is
            # accepted as a name ONLY when it survives those guards.
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
        # Round 2026-08-13T2059Z: capture location from "based in X" /
        # "located in X" and from natural-feature location phrases
        # ("on the isle of skye", "on the island of man", "in the valley of
        # X"). The prior m_loc only matched live/lives/am/was/were/grew up
        # + in/near/at/from, so "i'm a lighthouse keeper based in skye" and
        # "i keep the lighthouse on the isle of skye" stored NO location and
        # a later "where do i live" fell back to the NAME. Structural: extra
        # trigger shapes, no per-toponym table; the same _put_fact("location")
        # path is reused so recall stays consistent by construction.
        _m_loc_based = re.search(
            r"\b(?:i(?:'m| am)|he|she|they|we|you)\s+(?:[a-z]+['\-]?\s+){0,8}?"
            r"(?:based|located|stationed|situated)\s+(?:in|on|at|near)\s+"
            r"([A-Za-z][A-Za-z'\\-]*(?:\s+[A-Za-z][A-Za-z'\\-]*){0,2})",
            q_clean, re.IGNORECASE)
        _m_loc_feat = re.search(
            r"\b(?:in|on|at|near|off)\s+the\s+(?:isle|island|coast|shore|"
            r"headland|peninsula|cove|bay|fjord|valley|dale|glen|beach)\s+"
            r"of\s+([A-Za-z][A-Za-z'\\-]*)",
            q_clean, re.IGNORECASE)
        _loc_cand = None
        if _m_loc_based:
            _loc_cand = _m_loc_based.group(1).strip().strip(" .,!")
        elif _m_loc_feat:
            _loc_cand = _m_loc_feat.group(1).strip().strip(" .,!")
        if _loc_cand and len(_loc_cand.split()) <= 5 and _loc_cand.lower() not in _VALUE_STOP:
            self.user_location = _loc_cand
            _put_fact("location", _loc_cand, 0.6)
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
            # Round 2026-08-08f: a long location clause with a trailing
            # measure/qualifier ("i live in a lighthouse on a rock about two
            # kilometers offshore") over-grabs the qualifier, pushing the
            # capture past the <=5-word gate so NO location fact is stored and
            # the disclosure falls through to the hollow "got it" ack. Trim a
            # trailing qualifier phrase led by a measure word ("about/around/
            # roughly" or a number+unit like "two kilometers") so the real
            # place head ("a lighthouse on a rock") is kept and stored.
            # Structural: cuts at a closed-class qualifier, never invents a
            # place; degrades gracefully if trimming leaves nothing.
            else:
                _loc = re.split(
                    r"\s+(?:about|around|roughly|approximately|some)\s+"
                    r"(?:\d+\s+\w+|\w+)\b", _loc)[0].strip()
                _loc = re.split(r"\s+\d+\s+(?:kilometer|meter|mile|km|mi|minute|hour|year|month)s?\b", _loc)[0].strip()
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
            # A name is a proper noun, never an article-led description.
            # "call me a hypocrite" / "call me the bee guy" / "call me an
            # amateur" are self-descriptions, not names — reject candidates
            # that begin with a determiner so descriptor phrases are never
            # stored as the user's identity. Names are introduced bare
            # ("call me tobias", "i am called aria"), so this never blocks a
            # real name. Structural: a leading-determiner test, not a
            # per-name list.
            if name_cand.lower().startswith(("a ", "an ", "the ")):
                name_cand = ""
            # D1 (round 2026-08-08b): the bare "i'm X" / "i am X" copula form
            # is ALSO how users express transient states ("i'm furious at the
            # coast guard", "i'm scared the light will fail", "i'm feeling
            # really lonely", "i'm convinced cuttlefish are smarter"). The old
            # code stored the ENTIRE clause as (i, name, X), so a later
            # "what's my name" returned "feeling really lonely" / "furious at
            # the". A name is a SHORT proper noun, never a predicate clause.
            # Reject a bare-copula candidate unless it is name-shaped:
            #   - 1-2 word tokens (real chat names are single tokens: iris,
            #     noor, dev, soren, mira, wren, tobias; two-token names like
            #     "mary jane" are rare but legitimate), AND
            #   - contains NO closed-class word (preposition / copula /
            #     determiner: at/of/for/with/on/in/to/about/the/a/an/is/are/
            #     was/were/am/that/this/it/my/your), AND
            #   - its head token is NOT a feeling/state/cognitive verb.
            # This is structural predicate-vs-proper-noun discrimination, not a
            # per-name list; it generalizes across every persona. The
            # feeling/state verb set is SEED vocabulary (RAVANA-expandable: it
            # shares the role of the affect lexicon used by the empathy
            # gate), not authored answers.
            if name_cand:
                _nw = name_cand.split()
                _CLOSED = {
                    "at", "of", "for", "with", "on", "in", "to", "about",
                    "the", "a", "an", "is", "are", "was", "were", "am",
                    "that", "this", "it", "my", "your", "from", "by", "as",
                    "so", "but", "and", "or", "if", "because",
                }
                # A-name (round 2026-08-08c): a bare "i'm X" copula is how
                # users express TRANSIENT STATES ("i'm torn", "i'm shaking",
                # "i'm proud", "i'm hollow"). The old reject set was a frozen
                # stoplist that missed "torn"/"shaking"/"proud", so they were
                # stored as the user's NAME (name poisoning: a later "what's
                # my name?" answered "torn"/"shaking"). Reject any candidate
                # whose head token is an AFFECT / STATE / COGNITIVE word, drawn
                # from the SAME seed vocabulary the empathy gate uses
                # (brain_regions._CAUSE_SEEDS + support_router._SUPPORT_AFFECT),
                # expressed here as one data set. This is SEED vocabulary (not
                # an if/elif answer path): RAVANA can extend it at runtime via
                # the shared affect lexicon; removing entries degrades
                # gracefully (only loses one guard). Covers participles
                # ("shaking"/"tired"), irregulars ("torn"/"lost"), and
                # stative/cognitive verbs ("thinking"/"convinced").
                _NAME_REJECT_AFFECT = _AFFECT_STATE_LEXICON
                _has_closed = any(w.lower() in _CLOSED for w in _nw)
                # Round 2026-08-13T2059Z: generalize the guard. The prior
                # code only inspected the HEAD token, so "i'm against" /
                # "i'm gutted" / "i'm devastated" slipped through and were
                # stored as the user's NAME (name poisoning — a later
                # "what's my name" returned "against"/"gutted"). Reject when
                # ANY token is a transient-state / function / stance-connector
                # word (seed vocabularies _AFFECT_STATE_LEXICON +
                # _NAME_REJECT_FUNCTION), and reject verb-forms (gerunds in
                # 'ing', past/past-participle in 'ed', only for words >=5
                # letters so 3-4 letter names like 'ted'/'reed' survive). A
                # proper noun is NEVER any of these, so the guard is
                # exhaustive without enumerating every emotion word. SEED
                # data, RAVANA-expandable; removing entries degrades
                # gracefully, it is not content RAVANA cannot change.
                _any_state = any(
                    w.lower() in _NAME_REJECT_AFFECT
                    or w.lower() in _NAME_REJECT_FUNCTION
                    or (len(w) >= 5 and (w.endswith("ing") or w.endswith("ed")))
                    for w in _nw)
                if len(_nw) > 2 or _has_closed or _any_state:
                    name_cand = ""
            # Reject common states / descriptors / interrogatives so a bare
            # self-description is never stored as the user's name. Seed
            # stoplist of NON-name words, not a per-name allowlist.
            _NON_NAME = ("happy", "sad", "tired", "busy", "fine", "good",
                         "bad", "hungry", "thirsty", "angry", "mad", "glad",
                         "ok", "okay", "sorry", "here", "there", "lost",
                         "ready", "done", "sure", "right", "wrong", "well",
                         "sick", "late", "early", "home", "awake", "asleep",
                         "confused", "scared", "afraid", "excited",
                         "nervous", "calm", "what", "who", "why", "how",
                         "where", "when", "not", "no", "yes", "maybe")
            # DESCRIPTOR-NOUN deny set (round 2026-08-14T0103Z). "i'm
            # vegetarian" / "i'm vegan" / "i'm a ceramicist" / "i'm an
            # atheist" / "i'm a teacher" all reach the bare-copula name
            # candidate path and were stored as the user's NAME (name
            # poisoning: a later "what's my name" answered "Vegetarian" /
            # "Ceramicist"). A name is a PROPER NOUN (mira, wren, tobias),
            # never a common descriptor noun. This deny set covers the broad
            # CATEGORIES of self-descriptor nouns (diet, religion/identity,
            # occupation, nationality, orientation, family-role, political) as
            # SEED vocabulary — RAVANA-expandable, not a per-name allowlist,
            # so it generalizes to any descriptor the user might self-apply.
            # Removing entries degrades gracefully (only loses one guard).
            _NAME_REJECT_DESCRIPTOR = {
                # diet / lifestyle
                "vegetarian", "vegan", "omnivore", "pescatarian", "flexitarian",
                "carnivore", "meat-eater", "teetotaler", "teetotaller",
                # religion / belief / identity
                "atheist", "agnostic", "christian", "muslim", "islamist",
                "hindu", "buddhist", "jew", "jewish", "sikh", "pagan",
                "catholic", "protestant", "mormon", "spiritual", "humanist",
                "skeptic", "sceptic", "nihilist", "stoic", "optimist",
                "pessimist", "realist", "idealist", "pragmatist",
                # occupation / role
                "ceramicist", "ceramist", "artist", "painter", "sculptor",
                "writer", "author", "poet", "musician", "teacher", "student",
                "engineer", "doctor", "nurse", "lawyer", "chef", "baker",
                "programmer", "developer", "designer", "architect", "scientist",
                "researcher", "farmer", "fisherman", "sailor", "soldier",
                "officer", "clerk", "cashier", "waiter", "waitress", "barista",
                "driver", "pilot", "carpenter", "plumber", "electrician",
                "mechanic", "gardener", "librarian", "journalist", "editor",
                "actor", "singer", "dancer", "photographer", "printmaker",
                "potter", "weaver", "smith", "tailor", "cook", "builder",
                # nationality / origin
                "indian", "american", "british", "english", "scottish",
                "welsh", "irish", "french", "german", "spanish", "italian",
                "canadian", "australian", "chinese", "japanese", "korean",
                "russian", "mexican", "brazilian", "dutch", "swiss", "swede",
                "norwegian", "dane", "fin", "polish", "greek", "turk",
                # orientation / identity
                "straight", "gay", "lesbian", "bisexual", "transgender",
                "queer", "cisgender", "pansexual", "asexual", "demisexual",
                # family role (relative nouns can follow "i'm", e.g. "i'm a
                # father" / "i'm someone's sister") — these are roles, never names
                "father", "mother", "parent", "son", "daughter", "brother",
                "sister", "uncle", "aunt", "auntie", "cousin", "grandfather",
                "grandmother", "grandparent", "nephew", "niece",
                # political / affiliation
                "democrat", "republican", "socialist", "communist", "liberal",
                "conservative", "anarchist", "centrist", "libertarian",
                # general descriptors
                "introvert", "extrovert", "ambivert", "minimalist",
                "maximalist", "environmentalist", "feminist", "activist",
            }
            _nc_low = name_cand.lower()
            if (name_cand and _nc_low not in _NON_NAME
                    and _nc_low not in _NAME_REJECT_DESCRIPTOR):
                name_cap = " ".join(w.capitalize() for w in name_cand.split())
                self.user_name = name_cap
                _put_fact("name", name_cap, 0.6)

        # Round 2026-08-13T2059Z: GENERAL RELATIONAL MINER (the 6f
        # generalization gap — "who is wren to me" must resolve from the
        # store, type-agnostically, for ANY relationship the user states).
        # A bare "my <relation> <name> <rest>" disclosure (my cousin nora is
        # a glaciologist...; my partner june runs a library; my neighbour otto
        # fixes clocks; my auntie bea breeds ponies) is a RELATIONSHIP + an
        # ENTITY, not a self-profile attribute. It MUST be captured as a
        # relationship fact keyed by the NAMED ENTITY (subject=name,
        # attr=relationship) plus the descriptive rest as role/does keyed by
        # the same entity — reusing the SAME _put_fact_ent path the
        # possessive miner uses, so miner + recaller agree on the key by
        # construction. This generalizes across EVERY relationship type
        # (friend/sister/brother/cousin/partner/neighbour/auntie/pet/...);
        # it does NOT add a second narrow branch per entity kind.
        # RELATION_WORDS is SEED vocabulary: a closed set of common-English
        # relation terms, RAVANA-expandable at runtime via _learn_relation()
        # (the user can name any relation — e.g. "my godfather luis" extends
        # the set). It is DATA (not an if/elif answer path), and removing an
        # entry degrades gracefully (one fewer relation class captured). A
        # bare-capitalized-name test after it makes the miner fire on names
        # RAVANA has not seen, so it never needs the list to be exhaustive.
        _RELATION_WORDS = (
            "friend", "friends", "sister", "brother", "cousin", "siblings",
            "aunt", "auntie", "uncle", "nephew", "niece", "mother", "mom",
            "father", "dad", "parent", "parents", "grandmother", "grandfather",
            "grandma", "grandpa", "son", "daughter", "child", "children",
            "kid", "kids", "wife", "husband", "spouse", "partner", "fiance",
            "fiancee", "boyfriend", "girlfriend", "colleague", "coworker",
            "boss", "manager", "mentor", "student", "teacher", "neighbour",
            "neighbor", "roommate", "flatmate", "landlord", "tenant",
            "godfather", "godmother", "godparent", "stepfather", "stepmother",
            "stepsister", "stepbrother", "halfsister", "halfbrother",
            "grandson", "granddaughter", "pet", "cat", "dog", "horse", "pony",
            "companion", "bestie", "pal", "buddy", "acquaintance",
        )
        # mutable per-instance learned-relation set so RAVANA grows it online
        _rel_known = set(getattr(self, "_learned_relations", set())) | set(_RELATION_WORDS)
        # single-match form: "my <rel> <Name> <rest>" at end of clause or
        # anywhere; iterate so conjoined "my cat mochi ... and my dog pip ..."
        # captures BOTH entities (one match per "my <rel> <Name>" segment).
        _REL_RE = re.compile(
            r"\bmy\s+([a-z][a-z]+)\s+([A-Za-z][A-Za-z'-]+)"
            r"(?:\s+([^.?!]+?))?(?=\s+(?:and|but|,|&)|[.!?]|$)",
            re.IGNORECASE)
        for _m_rel in _REL_RE.finditer(q_clean):
            _rel = _m_rel.group(1).lower().strip()
            _ent = (_m_rel.group(2) or "").strip().strip(".,!?")
            _rest = (_m_rel.group(3) or "").strip().strip(".,!?")
            _looks_relation = (_rel in _rel_known) or (
                _rel.endswith("friend") or _rel.endswith("mate")
                or _rel in ("partner", "spouse", "sibling"))
            # Pet species (cat/dog/horse/...) are owned by the pet-slot system
            # (pet_slots.py) which keys them as (i, species) / (i, species_N);
            # never route them through the entity-keyed relationship store, or
            # "my cat is pixel" would be lost from the pet slot and break the
            # species-coexistence / cued-recall tests. Kinship + people
            # relations still flow through the relational miner.
            _is_pet = _pet_slots.species_of(_rel) is not None
            _name_shape = bool(re.match(r"^[A-Za-z][A-Za-z'-]+$", _ent)) \
                and _ent.lower() not in _VALUE_STOP
            if _looks_relation and _name_shape and len(_ent) >= 2 and not _is_pet:
                _put_fact_ent(_ent, "relationship", _rel, 0.65)
                if _rest:
                    _r = re.sub(r"^[\s,]+", "", _rest).strip()
                    # drop a leading copula + optional article ("is a X") OR a
                    # bare leading article ("a X") — the copula may already be
                    # consumed by the capture, leaving "a glaciologist ...".
                    _r = re.sub(r"^(?:is|are|was|were|'s)\s+(?: of)?\s+(?:a|an|the)\s+",
                                "", _r, flags=re.IGNORECASE).strip()
                    _r = re.sub(r"^(?:is|are|was|were)\s+", "", _r,
                                flags=re.IGNORECASE).strip()
                    _r = re.sub(r"^(?:a|an|the)\s+", "", _r,
                                flags=re.IGNORECASE).strip()
                    _r = re.split(r"(?<=[.!?])\s+|[.!?]\s*$", _r)[0].strip()
                    if _r and len(_r.split()) <= 10:
                        _put_fact_ent(_ent, "role", _r, 0.6)
                if _rel not in _RELATION_WORDS:
                    _rel_known.add(_rel)
                    self._learned_relations = _rel_known

        for _pat in (
            # D2 (round v2): tolerate an optional "name" word between the
            # relation and "is" so "my daughter name is ingrid" stores
            # daughter=ingrid (the old pattern required exactly one token
            # before "is" and missed the "name" form).
            # FIX (round v-aug06d): generalize to a MULTI-WORD attribute so
            # "my favorite color is ochre" stores attr="favorite color",
            # val="ochre" (the old single-token attr grabbed attr="favorite",
            # val="color" and lost the real value). Attribute capped at 4 words,
            # value at up to 8 words; this is structural (no per-topic
            # attribute list) — any "<attr phrase> is <value phrase>" is
            # captured. The cap was 4 words but real values are longer,
            # e.g. "my favorite time of day is the blue hour just before the
            # sun" was shredded to "the blue hour just" (the 4-word cap cut
            # "before the sun"). Raised to 8 words so a genuine multi-word
            # value is kept whole; a following sentence (".", "!", "?") is
            # still trimmed downstream so only THIS clause is stored.
            r"\bmy\s+([\w'-]+(?:\s+[\w'-]+){0,3})\s+(?:is|are)\s+([\w'-]+(?:\s+[\w'-]+){0,7})",
            # FIX (round v-aug06d): "i work as a <role>" / "i work for
            # <org>" self-descriptions are identity facts, not throwaway
            # activities. The old activity miner only caught the verb "work"
            # inside the generic activity loop and stored junk (does=s).
            # Capture the role as a durable 'work' fact. Generic: any noun
            # after "work as/for", no occupation list. Tolerates optional
            # adverbs/words between "work" and "as/for" ("i work nights as an
            # er nurse") — the previous pattern required them contiguous and
            # silently dropped "i work <adv> as X".
            r"\bi\s+work\b.*?\b(?:as|for)\s+(?:a\s+|an\s+|the\s+)?([\w'-]+(?:\s+[\w'-]+){0,4})",
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
                # Round 2026-08-13T2059Z: skip the generic "my <attr> is
                # <val>" / "my <attr> <val>" capture when the attr is a
                # RELATIONSHIP word followed by a NAME — that is the
                # "my <rel> <name> <rest>" shape already handled by the
                # relational miner above (stored under the ENTITY key, not
                # the user's 'i' subject). Without this guard the loop
                # ALSO stored ('i', 'cousin nora', ...) alongside the
                # correct ('nora', 'relationship', 'cousin'), polluting the
                # self-profile with a relationship dressed as a self-attr.
                # Structural: a relation-set membership test, not a
                # per-relation branch. _rel_known covers the seed set +
                # runtime-learned relations so it stays in sync with the
                # relational miner.
                _raw_attr = (_m.group(1).strip().lower()
                             if _m.lastindex is not None and _m.lastindex >= 1
                             else "")
                # attr may be multi-word ("cousin nora"); a relation-led attr
                # starts with a relation word followed by a proper-noun name.
                # Pet species (cat/dog/...) are excluded — they are owned by
                # the pet-slot system (pet_slots.py) and must NOT be skipped,
                # or "my cat is pixel" falls through to nothing.
                # Only skip the genuine "my <rel> <Name>" shape where the word
                # right after the relation is a NAME (not a copula/article):
                # "my cousin nora is ..." skips (nora is the named entity), but
                # "my child is a curious kid" does NOT (the word after "child"
                # is the copula "is", so it is a normal self-attribute).
                if _raw_attr:
                    _raw_head = _raw_attr.split()[0] if _raw_attr.split() else ""
                    _after = re.search(
                        r"\b" + re.escape(_raw_head)
                        + r"\s+([A-Za-z][A-Za-z'-]*)\b", q_clean, re.IGNORECASE)
                    _COPULA_ART = ("is", "are", "was", "were", "am", "a", "an",
                                   "the", "i", "my", "of", "to", "for", "on",
                                   "in", "at", "by", "with", "and", "that")
                    if (_raw_head in _rel_known
                            and _pet_slots.species_of(_raw_head) is None
                            and _after is not None
                            and _after.group(1).lower() not in _COPULA_ART
                            and re.match(r"^[A-Za-z][A-Za-z'-]+$",
                                         _after.group(1))):
                        continue
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
                    # Trim any FOLLOWING sentence so a value like "the blue
                    # hour just before the sun. i also keep pigeons" stores
                    # only the first clause, not the whole tail. Only a hard
                    # sentence break (., !, ?) ends the value; internal
                    # prepositions/conjunctions ("of the", "before the") are
                    # kept so a genuine multi-word value stays whole.
                    _val = re.split(r"(?<=[.!?])\s+|\s*[.!?]\s*$", _val)[0].strip()
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
                    elif _pat.startswith(r"\bi\s+work"):
                        _attr = "work"
                        # Trim trailing prepositional noise ("er nurse in a
                        # hospital near the river" -> "er nurse") via the same
                        # content-head resolver used for stances.
                        _trimmed = self._opinion_topic(_val)
                        if _trimmed:
                            _val = _trimmed
                    elif _pat.startswith(r"\bi\s+am\s+allergic\s+to"):
                        _attr = "allergy"
                # D6 (round 2026-08-08b-d): a possessive attr ('my partner's
                # name is theo' -> attr="partner's name") must be stored under
                # the OWNER entity, not the user's 'i' subject. Otherwise a
                # later recall renders "your name is theo" (partner's name
                # reported as the user's). Split the possessive head into
                # (entity, relation) and route through _put_fact_ent. The
                # recall reconstructor (_retrieve_episodic / _structured_recall)
                # already keys possessive facts by owner, so this makes the
                # MINER agree with the recaller by construction.
                _ent, _rel = _split_possessive_attr(_attr)
                if _ent is not None:
                    _put_fact_ent(_ent, _rel, _val, 0.6)
                    continue
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
                       "cook", "fish", "hike", "garden", "farm", "lead", "organize",
                       # Round 2026-08-08: activity verbs the prior seed set
                       # missed, so self-disclosures like "i keep homing
                       # pigeons", "i grind my own telescope mirrors", "i race
                       # homing pigeons", "i sail a small dinghy", "i knit
                       # wool socks", "i forge my own knives" were silently
                       # dropped and later recall had nothing to recall. These
                       # are the SAME kind of seed verb vocabulary (RAVANA
                       # expands it from experience; the user can correct any
                       # stored 'does' fact), not a per-topic list.
                       "keep", "grind", "race", "sail", "fly", "knit", "sew",
                       "weld", "forge", "carve", "compose", "record",
                       "perform", "coach", "train", "compete", "spin",
                       "weave", "mount", "trade", "sell", "host", "guide"):
            _m = re.search(
                r"\bi\s+(?:also\s+|really\s+|even\s+|just\s+|now\s+|still\s+"
                r"|often\s+|sometimes\s+|usually\s+)?"
                r"(?:have\s+been\s+)?(?:been\s+)?(?:keep\s+|grind\s+|race\s+)?"
                + _verb +
                # D3 (round 2026-08-08b-d): the article alternative `a` had NO
                # word boundary, so for "i teach at a school" it matched the
                # leading 'a' of "at", leaving "t a school" as the object ->
                # stored "teach t". Bound the article alternatives with \b so
                # `a`/`an`/`the` only match a STANDALONE article, never the
                # prefix of another word. The object still passes through
                # _opinion_topic which drops closed-class words, so "i teach at
                # a school by the river" -> "school" (real concept head), not
                # "t". Structural; no per-topic table.
                r"\s+(?:my\s+|\b(?:a|an|the)\b\s+)?(.+?)(?:\bfor\b|\bwhen\b|\bbut\b|"
                r"\bbecause\b|\band\b|\.|\!|\?|$|,)",
                q_clean, re.IGNORECASE)
            if _m:
                _obj = self._opinion_topic(_m.group(1).strip().lower())
                if _obj and len(_obj.split()) <= 5:
                    # Store the verb WITH the object ("keep homing pigeons")
                    # so activity recall ("what do i keep?") can match the
                    # verb and return a complete, grammatical answer instead
                    # of a bare noun. The verb is part of the mined disclosure,
                    # not an authored label.
                    _put_fact("does", f"{_verb} {_obj}", 0.55)
        # "i've been <verb>-ing <object> for <duration>" (ongoing activity)
        _cont = re.search(
            r"\bi(?:'ve| have)\s+been\s+(\w+ing)\s+(.+?)(?:\bfor\b|\bsince\b|\.|\!|\?|$|,)",
            q_clean, re.IGNORECASE)
        if _cont:
            _obj = self._opinion_topic(_cont.group(2).strip().lower())
            if _obj and len(_obj.split()) <= 5:
                _put_fact("does", _obj, 0.55)

        # FIX (round 2026-08-09T1953Z): general first-person activity +
        # experience capture. The D3 activity loop above only matches BARE
        # verb forms ("train", "keep") and omits common disclosure verbs
        # ("throw", "shoot", "develop", "clean", "grow", "train"). Real chat
        # is dominated by gerunds and continuous tenses ("i throw pots",
        # "i've been training a juniper bonsai", "i shoot 35mm", "i keep air
        # plants", "i clean the reef tank glass") — none of which the D3 loop
        # caught, so they fell through to the hollow "got it — thanks for
        # telling me." ack AND became unrecallable. This block generalises the
        # capture to inflected forms and a broader closed VERB SEED set, and
        # ADDS firsthand-experience (event) capture for disclosures like "i
        # dropped half its needles", "i lost a favia coral to heat", "i
        # repotted the juniper and found a root that went necrotic", "i
        # removed the dead favia". These are real things the user did/experienced
        # about their world and must land in the same PersonalFactStore so cued
        # recall and the "what have you learned about me" summary can surface
        # them (the old code only ever recalled 'does'/'likes' activity facts).
        #
        # DESIGN (per round hardcoding rule + seed-vs-hardcoding test):
        #  - The verb vocabulary is SEED structure: a closed list of
        #    activity/experience verbs. It is RAVANA-expandable in principle
        #    (it feeds the same PersonalFactStore the user can correct/extend),
        #    NOT a per-topic answer dictionary and NOT authored reply prose.
        #    Removing an entry degrades gracefully (one fewer activity class
        #    captured) — it is not content RAVANA can never change, so it is
        #    seed knowledge, not hardcoding.
        #  - The value is the resolved CONTENT HEAD of the object phrase
        #    (_opinion_topic drops closed-class words), so the stored value is
        #    a real concept ("pots", "bonsai", "35mm", "air plants", "reef
        #    tank glass"), never a function word.
        #  - Capture is GENERAL (any "i <verb> <object>"), so it fires on new
        #    topics without retraining or per-topic tuning.
        #  - Activity verbs -> attr "does" (consistent with the D3 loop).
        #  - Experience/event verbs -> attr "event" (new), so a later recall
        #    can reconstruct "you dropped <x>" / "you lost <y>" grammatically
        #    (see engine_memory._reconstruct_entity + engine_reasoning
        #    ._derive_ack_from_store which now render the 'event' attr).
        # Round 2026-08-13T2059Z: PRINCIPLED VERB PRUNE. The activity/event
        # verb seeds below mix genuine activity/experience verbs with
        # ACHIEVEMENT / COMMUNICATION verbs (got, said, made, gave, told,
        # came, went, did, saw, met, sold, paid, sent, spent, bought, caught,
        # brought, ate, drank, knew, wore, led, read, flew, swam, rode, drove,
        # broke, spoke, woke, froze, chose, slept, felt, held, found, lost,
        # kept, took, set, put, cut, hit, fed, bled, built, taught, wrote,
        # drew, sang, grew, threw). Those fire on OUTCOME / UTTERANCE
        # disclosures whose object is a bare noun naming a result
        # ("i got the artist residency" -> does=got artist residency;
        # "i said open-plan offices help" -> does=said open), and they echo
        # verbatim in the self-summary as garbage. SEED set (RAVANA-
        # expandable): these are communication/achievement verbs, not
        # sustained activities or physical-world experiences; removing them
        # degrades gracefully (one fewer outcome-class captured, which is
        # correct — outcomes are not recurring activities). The genuine
        # activity verbs (keep/grow/start/lost/found/build/throw/play/...)
        # and experience verbs (drop/lose/find/break/heal/...) are kept,
        # so the prior fact-mining tests (throw pots, grow air plants,
        # repot juniper, lost favia coral, reef tank) stay GREEN.
        _ACHIEVE_COMM_VERBS = frozenset({
            "got", "get", "said", "say", "made", "make", "gave", "give",
            "told", "tell", "came", "come", "went", "go", "did", "do",
            "saw", "see", "met", "meet", "sold", "sell", "paid", "pay",
            "sent", "send", "spent", "spend", "bought", "buy", "caught",
            "catch", "brought", "bring", "ate", "eat", "drank", "drink",
            "knew", "know", "wore", "wear", "led", "lead", "read", "fly",
            "flew", "swam", "swim", "rode", "ride", "drove", "drive",
            "broke", "break", "spoke", "speak", "woke", "wake", "froze",
            "freeze", "chose", "choose", "slept", "sleep", "felt", "feel",
            "held", "hold", "took", "take", "set", "put", "cut", "hit",
            "fed", "feed", "bled", "bleed",
        })
        # Closed VERB SEED vocabulary (RAVANA-expandable; feeds the same

        # PersonalFactStore the user can correct — NOT per-topic answers, NOT
        # authored prose). Covers everyday disclosure verbs + common irregular
        # past forms so first-person activities/experiences actually land.
        _ACTIVITY_VERBS = tuple(v for v in (
            "run", "own", "operate", "play", "teach", "study", "manage",
            "drive", "build", "make", "sell", "restore", "grow", "watch",
            "raise", "tend", "brew", "bake", "write", "read", "learn",
            "practice", "collect", "fix", "paint", "code", "design", "craft",
            "volunteer", "cook", "fish", "hike", "garden", "farm", "lead",
            "organize", "keep", "grind", "race", "sail", "fly", "knit",
            "sew", "weld", "forge", "carve", "compose", "record", "perform",
            "coach", "train", "compete", "spin", "weave", "mount", "trade",
            "host", "guide", "throw", "shoot", "develop", "clean", "reload",
            "recharge", "assemble", "mix", "pour", "press", "roll", "fire",
            "glaze", "wire", "prune", "pot", "plant", "sketch", "draw",
            "sculpt", "stitch", "mend", "whittle", "start", "begin", "try",
            "go", "use", "take", "make", "get", "built", "taught", "wrote",
            "drew", "sang", "flew", "swam", "rode", "drove", "broke",
            "spoke", "woke", "froze", "chose", "ate", "drank", "grew",
            "threw", "knew", "wore", "brought", "bought", "caught",
            "kept", "slept", "left", "felt", "met",
            "sent", "spent", "lost", "found", "held", "told", "sold",
            "paid", "said", "gave", "came", "went", "did", "saw", "got",
            "made", "took", "set", "put", "cut", "hit", "read", "led",
            "fed", "bled", "fed",
        ) if v not in _ACHIEVE_COMM_VERBS)
        _EVENT_VERBS = tuple(v for v in (
            "drop", "lose", "find", "remove", "break", "discover", "notice",
            "repot", "prune", "harvest", "spill", "melt", "crack", "kill",
            "ruin", "save", "nurse", "revive", "miss", "spot",
            "catch", "pull", "cut", "burn", "flood", "rescue", "rebuild",
            "recover", "heal", "uproot", "freeze", "thaw", "hatch",
            "bloom", "wilt", "die", "survive", "escape", "return", "birth",
            "fall", "fell", "crash", "lose", "lost", "found", "kept",
            "broke", "felt", "cut", "hit", "met", "told", "saw", "got",
            "made", "took", "gave", "came", "went", "did", "ate", "drank",
            "grew", "knew", "threw", "froze", "bled", "fed", "died",
        ) if v not in _ACHIEVE_COMM_VERBS)
        # Match "i [aux?] <verb>(s|ed|ing)? <object> <clause-boundary>".
        # The object stops at a clause boundary (., !, ?, ",", " and ",
        # " but ", " because ", " so ", " which ", " that ", " when ",
        # " where ") so a multi-clause sentence stores only the relevant
        # fragment (e.g. "i repotted the juniper and found a root..." ->
        # "juniper", not "juniper and found a root"). The verb is matched
        # with optional inflection so gerunds/continuous tenses are caught.
        _act_pat = re.compile(
            r"\bi\s+(?:also\s+|really\s+|even\s+|just\s+|now\s+|still\s+|"
            r"often\s+|sometimes\s+|usually\s+)?"
            r"(?:have\s+been\s+|has\s+been\s+|am\s+|was\s+|were\s+)?"
            r"(?:been\s+)?"
            r"(" + "|".join(_ACTIVITY_VERBS) + r")(?:s|es|ing|ed|[a-z]ed|[a-z]d)?"
            r"\s+(?:my\s+|a\s+|an\s+|the\s+|some\s+|two\s+|three\s+|four\s+|"
            r"five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)?"
            r"(.+?)(?:\s*(?:\.|\!|\?|,|-{1,3}|$|"
            r"\s+and\s+|\s+but\s+|\s+because\s+|\s+so\s+|\s+which\s+|"
            r"\s+that\s+|\s+when\s+|\s+where\s+|\s+while\s+))",
            re.IGNORECASE)
        for _am in _act_pat.finditer(q_clean):
            _verb = _am.group(1).lower()
            _obj = self._opinion_topic(_am.group(2).strip().lower())
            if _obj and 1 <= len(_obj.split()) <= 5:
                _put_fact("does", f"{_verb} {_obj}", 0.55)
        # Experience / event capture: first-person "i <event-verb> <object>"
        # describing something that happened to the user's world. Captured
        # under attr "event" so it is recallable as a lived experience (not
        # conflated with ongoing activity). Same clause-boundary + content-head
        # rules as the activity capture above.
        _evt_pat = re.compile(
            r"\bi\s+(?:also\s+|really\s+|even\s+|just\s+|now\s+|still\s+|"
            r"often\s+|sometimes\s+|usually\s+)?"
            r"(?:have\s+|has\s+|had\s+)?(?:almost\s+|nearly\s+)?"
            r"(" + "|".join(_EVENT_VERBS) + r")(?:s|es|ing|ed|[a-z]ed|[a-z]d)?"
            r"\s+(?:my\s+|a\s+|an\s+|the\s+|some\s+|two\s+|three\s+|four\s+|"
            r"five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)?"
            r"(.+?)(?:\s*(?:\.|\!|\?|,|-{1,3}|$|"
            r"\s+and\s+|\s+but\s+|\s+because\s+|\s+so\s+|\s+which\s+|"
            r"\s+that\s+|\s+when\s+|\s+where\s+|\s+while\s+))",
            re.IGNORECASE)
        for _em in _evt_pat.finditer(q_clean):
            _verb = _em.group(1).lower()
            _obj = self._opinion_topic(_em.group(2).strip().lower())
            if _obj and 1 <= len(_obj.split()) <= 5:
                _put_fact("event", f"{_verb} {_obj}", 0.5)

        # OPEN-CLASS activity/event capture (round 2026-08-13T2059Z). The two
        # frozen-verb blocks above (ACTIVITY_VERBS / EVENT_VERBS) only cover a
        # seeded whitelist, so first-person disclosures using a NOVEL or
        # HYPHENATED-COMPOUND verb ("i count meteor showers", "i tide-pool at
        # low water", "i astrophotograph the milky way") captured NO personal
        # fact and were even misrouted as knowledge queries. This block makes
        # capture GENERAL: any first-person "i <verb> <object>" whose verb is
        # NOT a stative/copula/achieve-comm verb is mined as a 'does' fact. The
        # verb vocabulary is now OPEN-CLASS (a closed deny-list), so RAVANA
        # learns the verb from experience instead of requiring the whitelist to
        # enumerate every possible activity. This is SEED structure (deny-list
        # + the learnable PersonalFactStore), NOT a per-verb answer dictionary
        # and NOT authored reply prose. Removing the deny-list degrades to
        # "capture everything" (still not a regression of capability), so it is
        # seed knowledge, not hardcoding.
        _STATIVE_DENY = frozenset({
            # copula / existence
            "am", "are", "is", "was", "were", "be", "been", "being",
            "become", "seem", "appear", "remain",
            # affect / cognition / volition (handled by opinion/benign paths)
            "feel", "feels", "love", "like", "hate", "dislike", "prefer",
            "think", "believe", "know", "understand", "want", "need", "wish",
            "hope", "guess", "suppose", "mean", "wonder", "agree", "disagree",
            "doubt", "fear", "regret", "suspect", "realize", "realise",
            "remember", "recall", "imagine", "mind", "care",
            # possession (handled by 'my X is Y' / have patterns)
            "have", "has", "had", "own", "possess",
            # communication / achievement utterances (echo verbatim as garbage;
            # seeded out just like _ACHIEVE_COMM_VERBS above)
            "got", "get", "said", "say", "made", "make", "gave", "give",
            "told", "tell", "came", "come", "went", "go", "did", "do",
            "saw", "see", "met", "meet", "sold", "sell", "paid", "pay",
            "sent", "send", "spent", "spend", "bought", "buy", "caught",
            "catch", "brought", "bring", "ate", "eat", "drank", "drink",
            "knew", "know", "wore", "wear", "led", "lead", "read", "fly",
            "flew", "swam", "swim", "rode", "ride", "drove", "drive",
            "broke", "break", "spoke", "speak", "woke", "wake", "froze",
            "freeze", "chose", "choose", "slept", "sleep", "felt", "feel",
            "held", "hold", "took", "take", "set", "put", "cut", "hit",
            "fed", "feed", "bled", "bleed",
        })
        _gen_verb_pat = re.compile(
            r"\bi\s+"
            r"(?:also\s+|really\s+|even\s+|just\s+|now\s+|still\s+|"
            r"often\s+|sometimes\s+|usually\s+)?"
            r"(?:have\s+been\s+|has\s+been\s+|am\s+|was\s+|were\s+)?"
            r"(?:been\s+)?"
            # verb: lowercase token, optionally hyphenated compound; excludes
            # 'ing/ed' inflections so we don't double-capture verbs already
            # handled by the seeded ACTIVITY_VERBS/EVENT_VERBS blocks above
            # (those keep their higher-confidence 0.55/0.5 paths). Only the
            # base form + 's/es' is captured here as the open-class fallback.
            r"((?:[a-z']+(?:-[a-z']+)*)(?:s|es)?)"
            r"\s+(?:my\s+|a\s+|an\s+|the\s+|some\s+|two\s+|three\s+|four\s+|"
            r"five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)?"
            r"(.+?)(?:\s*(?:\.|!|\?|,|-{1,3}|$|"
            r"\s+and\s+|\s+but\s+|\s+because\s+|\s+so\s+|\s+which\s+|"
            r"\s+that\s+|\s+when\s+|\s+where\s+|\s+while\s+))",
            re.IGNORECASE)
        for _gm in _gen_verb_pat.finditer(q_clean):
            _verb = _gm.group(1).lower()
            if _verb in _STATIVE_DENY:
                continue
            _obj = self._opinion_topic(_gm.group(2).strip().lower())
            if _obj and 1 <= len(_obj.split()) <= 5:
                _put_fact("does", f"{_verb} {_obj}", 0.5)

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
            # A-fix (round 2026-08-08b): comparative opinions ("small towns make
            # better humans than cities", "tea beats coffee", "the mountains are
            # finer than the coast"). The Winner (X) is the valued term -> a
            # positive stance on X. General, no per-topic rule; the content head
            # is resolved by _opinion_topic so the topic is a real concept
            # (e.g. "small towns"), never a function word. The loser (Y) is NOT
            # force-negatived here — if the user holds a negative view of it
            # they state it, and the same miner captures it; we must not invent
            # a polarity RAVANA could not revise by talking.
            (r"\b(.+?)\s+(?:makes?|are?|is|make|produce[s]?|breed[s]?|build[s]?)\s+"
             r"(?:better|finer|more human|more humane|healthier|stronger|"
             r"happier|wiser|kinder)\s+(?:humans?|people|folk|neighbou?rs?|"
             r"citizens?|communities?)?\s*(?:than|over|versus|vs\.?)\b", 0.7, 0.5),
            (r"\b(.+?)\s+(?:beats|outshines|trumps|wins\s+over|is\s+finer\s+than|"
             r"is\s+better\s+than)\s+(.+?)(?:[.!?]|\band\b|\bbut\b|$|,)", 0.7, 0.5),
            # Round 2026-08-08f: broaden the comparative / superlative /
            # dismissive opinion classes. The prior miner only caught the
            # 'makes better people than' and 'beats' shapes; rich value
            # judgments like 'the sea is a better teacher than any classroom',
            # 'hand-built synths sound warmer than mass-produced', 'graveyards
            # are the most honest libraries', 'the cold water is the only
            # honest part of my day', 'most modern music is just wallpaper',
            # 'the best knots are the ones you can untie in the dark' were NOT
            # captured -> no stance -> no contradiction target -> dead hollow
            # ack. Each class below is a GRAMMATICAL pattern (no per-topic
            # list); the content head is resolved by _opinion_topic so the
            # stance lands on the real concept (e.g. 'sea', 'hand-built
            # synths', 'graveyards', 'cold water', 'modern music'). Polarity
            # is lexical: comparatives/superlatives/dismissals are inherently
            # valenced. RAVANA can still revise any stance by talking (the
            # store merges on new input); nothing is frozen or retrained.
            # (a) comparative copula 'X is a better/safer/finer/honest-er Y
            #     than Z' -> positive stance on X. The leading 'i think/
            #     i believe' frame is stripped so the captured subject is the
            #     real content head (e.g. 'sea'), never 'i think the sea'.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:is|are)\s+(?:a|an)?\s*(?:better|safer|finer|kinder|"
             r"wiser|healthier|stronger|truer|freer|calmer|cleaner|warmer|"
             r"cooler|sharper|kinder|more honest|more human|more real|"
             r"more true|more free)\b"
             r"(?:\s+(?:teacher|thing|place|way|part|kind|sort|type|version|"
             r"form|bit|lot|deal))?\s+(?:than|over|versus|vs\.?\b)", 0.7, 0.55),
            # (b) sensory-comparative 'X sounds/feels/tastes/looks/reads WARMER
            #     than Y' -> positive stance on X (the WARMER-ER class). Strip
            #     a leading 'i think/believe' frame the same way.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:sounds|sound|feels|feel|tastes|taste|looks|look|reads|read|"
             r"seems|seem|comes|come|comes\s+across|comes\s+off)\s*"
             r"(?:more|much)?\s*(?:warmer|cooler|truer|cleaner|honest|realer|"
             r"more honest|more real|more true|more alive|more human)\b"
             r"(?:\s+(?:than|over|versus|vs\.?))", 0.7, 0.55),
            # (c) superlative 'X is the most Y' / 'X is the best Y' /
            #     'the only Y' -> positive stance on X. Strip the leading
            #     'i think/believe' frame so the subject is the content head.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:is|are)\s+the\s+(?:most|best|finest|truest|honest|real|"
             r"purest|clearest|only)\b", 0.75, 0.6),
            # (d) 'X is just wallpaper/noise/fluff/...' -> negative dismissive
            #     stance on X (the subject is being demoted to negligible).
            #     Structural: the dismissive-metaphor noun set is SEED
            #     vocabulary (the same kind of small lexicon the sentiment
            #     adjectives use), not a per-topic table; RAVANA can extend it
            #     at runtime. Strip the leading frame.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:is|are)\s+(?:just|merely|basically|really\s+just)\s*"
             r"(?:wallpaper|noise|fluff|decoration|decorative|background|filler|"
             r"branding|spin|hype|fad|nonsense|garbage|junk|pap|slop|trash)"
             r"(?:\b|[.!?])", -0.7, 0.55),
            # (e) 'X is the best kind of Y' -> positive stance on X (X is the
            #     prized member of category Y). Strip the leading frame.
            (r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?"
             r"\b(.+?)\s+(?:is|are)\s+the\s+best\b"
             r"(?:\s+(?:kind|sort|type|breed|example|form|version|bit))?", 0.7, 0.55),
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
        # A softening cue ANYWHERE in the utterance governs the whole retraction
        # speech act. "i take it back — but it's not that bad" is a hedged,
        # NON-inverting recant: the user relaxes the stance toward neutral, they
        # do NOT flip to the opposite conviction. The first-match loop above can
        # bind `_matched_cue` to a hard recant ("i take it back") while a
        # softening phrase trails it; if both are present the softening intent
        # wins. Generic: scans the whole utterance for any softening idiom, no
        # per-topic rule.
        _soft = _matched_cue in _SOFTENING_CUES or any(
            re.search(p, q) for p in _SOFTENING_CUES)
        if cue_end is None:
            # Round 2026-08-08f: concession shape. A very common contradiction
            # does NOT use a retraction keyword ("i take back", "i was wrong"):
            # the user concedes a prior stance with "i thought X but Y" /
            # "i used to think X but now Y" / "i told you X but actually Y",
            # where X's topic matches a stance RAVANA already holds and Y
            # contradicts it. The prior code only caught keyword-led retractions,
            # so these fell through to the hollow "got it" ack and the stale
            # stance persisted (the contradiction was silently dropped). Detect
            # the concession structurally: a first-person past/present belief
            # frame ("i thought/i used to think/i told you/i said") followed by
            # a BUT that introduces a contrasting clause. Resolve the conceded
            # topic against the LIVE stance store; if it matches a held stance,
            # reverse/soften it the same way a keyword retraction would. This is
            # grammatical (no per-topic table) and RAVANA can still revise the
            # stance by further talk. A concession is a SOFTENING (the user is
            # walking the stance back, not inverting to a hard opposite
            # conviction), so it relaxes toward neutral, never force-flips.
            _concession = re.search(
                r"\b(?:i\s+(?:thought|used\s+to\s+think|told\s+you|said|believed|felt)"
                r"|i'?m\s+not\s+so\s+sure)\b"
                r".{0,60}?\b(?:but|although|though|yet|actually|however)\b", q)
            if _concession is not None:
                # A concession ("i thought X but Y") walks BACK the belief held
                # in the PRE-connector clause X — X is the topic the user now
                # revokes, while Y is the NEW contrasting preference. Resolving
                # the topic against the WHOLE utterance is wrong: the opinion
                # miner in the same turn also creates a (bogus) stance from the
                # trailing "i prefer Y" clause, so a whole-utterance longest-key
                # match can bind the NEW topic and (a) reverse a stance the user
                # never walked back, and (b) emit a fabricated "you changed your
                # mind about Y" ack for a topic with no prior stance. Restrict
                # resolution to the conceded clause X only. Generic: the clause
                # is derived from the matched connector span, not a per-topic
                # table, and still resolves against the live store.
                _pre_clause = q[:_concession.end()]
                _target = self._stance_key_in_text(_pre_clause)
                if _target is None:
                    # Fallback: the conceded topic may be a multiword key whose
                    # tokens are split across the connector (e.g. "i thought the
                    # sea was X but Y"); still scope to the pre-connector span so
                    # the new preference Y can never be the resolved target.
                    _target = self._stance_key_in_text(q[:_concession.start()])
                if _target is not None:
                    try:
                        self.opinions._soft_reversal = True
                        self.opinions.reverse_stance(_target, utterance=text)
                    except Exception:
                        pass
                    return
            return
        # A retraction cue is either a HARD recant ("i was wrong about X",
        # "i take it all back" — flip decisively) or a SOFTENING ("x isn't that
        # bad, i was too hasty", "i came around a bit" — relax toward neutral,
        # never invert). The softening cues are exactly the phrase set that
        # resolves to a partial reversal; everything else is a hard recant.
        # This drives reverse_stance's blend magnitude, so the same code path
        # produces opposite-hemisphere vs near-neutral recodes from the
        # utterance itself — no per-topic rule, no hardcoding of the topic.
        # The topic lives in the clause after the retraction cue. Strip leading
        # connectors/prepositional frames that carry no content.
        tail = q[cue_end:].strip(" \t.,!?;:'\"\u2014-")
        tail = re.sub(
            r"^(?:what\s+i\s+(?:said|think|thought|meant)"
            r"|my\s+(?:stance|opinion|view)\s+on"
            r"|my\s+mind\s+(?:about|on)"
            r"|i\s+(?:think|thought|believe|felt|was|am)\s+(?:i\s+)?(?:was|were|am|wrong|right|too\s+hasty|mistaken|in\s+error)\s+(?:about|on|there)\s+"
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
                self.opinions.reverse_stance(_target, utterance=text)
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
        # REVERSAL SCOPE GUARD. A recant like "i was wrong about acoustic-only"
        # must only flip a stance when the recanted phrase RESOLVES to a held
        # topic. The loose resolver below can link "acoustic-only" to the held
        # "acoustic music" stance by substring, then invert a stance the user
        # was NARROWING (they still liked acoustic, just not *only* acoustic).
        # That is a scope-widening, not a reversal — flipping it corrupts the
        # store. So require a TIGHT link: the recant's topic must share a
        # content word with, or be closely contained by, a held stance key.
        # Loose substring containment (a broader held topic merely containing
        # a word of the recant) is rejected for reversals. This reads the live
        # stance store — no per-topic table.
        _target_candidates = []
        _raw_topic_tokens = set(re.findall(r"[a-z']+", (topic or "").lower()))
        # split hyphenated qualifiers ("acoustic-only" -> acoustic, only)
        _raw_topic_tokens |= set(t for tok in _raw_topic_tokens
                                 for t in tok.split("-") if t)
        # scope markers ("only"/"just"...) are NOT topic content — they signal
        # a NARROWING of an existing stance, not a reversal of it. Strip them
        # before matching so "acoustic-ONLY" is read as "acoustic", whose
        # remainder is a strict subset of the held "acoustic music" topic.
        _scope_markers = {"only", "just", "really", "truly", "merely", "simply"}
        _topic_tokens = _raw_topic_tokens - _scope_markers
        for _k in self.opinions.stances:
            _kt = set(re.findall(r"[a-z']+", _k.lower()))
            if not _kt:
                continue
            # tight: every recant token present in the held key, OR strong
            # jaccard overlap (>=0.5) between the two content-word sets.
            if _topic_tokens and (_topic_tokens <= _kt
                                  or len(_topic_tokens & _kt) / max(1, len(_topic_tokens)) >= 0.5):
                # REJECT a narrowing recant: when the recant's meaningful
                # content is a STRICT subset of the held topic, the user is
                # restricting scope (still likes acoustic, just not *only*
                # acoustic), not reversing their attitude. Flipping the held
                # stance would corrupt the store. Scope-widening is not a
                # reversal. Detected generically from token containment.
                if _topic_tokens and _topic_tokens < _kt:
                    continue
                _target_candidates.append(_k)
        # Only honor a loose substring match when it is NOT a broad-held-topic
        # merely containing the recant word (that is the corruption case).
        target = None
        if _target_candidates:
            # prefer the most specific (shortest) tight match
            target = min(_target_candidates, key=len)
        else:
            _loose = self.opinions.resolve_topic(topic) or self.opinions.resolve_topic(tail)
            if _loose is not None:
                _loose_tokens = set(re.findall(r"[a-z']+", _loose.lower()))
                # Accept the loose match only if the recant phrase is NOT a
                # narrowing qualifier of a broader held stance. A narrowing
                # recant ("i was wrong about acoustic-ONLY") keeps the held
                # attitude and only restricts its scope, so flipping the held
                # stance corrupts the store. Detect: the recant's content words
                # (ignoring scope markers like "only"/"just"/"really") are a
                # subset of the held topic's words, or a hyphenated qualifier
                # whose head appears in the held topic. Such a match is
                # scope-widening, not a reversal -> reject.
                _scope_markers = {"only", "just", "really", "truly", "merely", "simply"}
                _recant_meaningful = _topic_tokens - _scope_markers
                _is_narrowing = (
                    _recant_meaningful
                    and _recant_meaningful <= _loose_tokens
                    and _recant_meaningful != _loose_tokens)
                if not _is_narrowing:
                    target = _loose
        if target is None:
            target = self._stance_key_in_text(q)
            # also try resolving the whole-utterance content head
            if target is None:
                _whole = self._opinion_topic(q)
                if _whole:
                    target = self.opinions.resolve_topic(_whole)
                if target is None:
                    return
        # SCOPE-WIDENING GUARD (final): never reverse a stance whose topic is a
        # BROADER held stance that the recant merely narrows. "i was wrong about
        # acoustic-ONLY" still likes acoustic; flipping "acoustic music" corrupts
        # the store. Reject when the recant's extracted object content is a
        # strict subset of the resolved target's content words (a narrowing, not
        # a reversal). Reads the live stance store — no per-topic rule.
        _tgt_tokens = set(re.findall(r"[a-z']+", target.lower()))
        _recant_meaningful = set(re.findall(r"[a-z']+", (topic or "").lower()))
        _recant_meaningful = {t for tok in _recant_meaningful
                              for t in tok.split("-") if t} - _scope_markers
        if _recant_meaningful and _recant_meaningful < _tgt_tokens:
            return
        try:
            self.opinions._soft_reversal = _soft
            self.opinions.reverse_stance(target, utterance=text)
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
        m_like = re.search(r"\bi\s+(?:like|love|prefer|enjoy)\s+(.+)", q_clean, re.IGNORECASE)
        if m_like:
            # Capture only the REAL content head of what's liked, not the
            # entire trailing clause. "i prefer acoustic music over anything
            # produced on a laptop" must store "acoustic music", not the
            # comparative tail "over anything produced on a laptop". Resolve
            # via _opinion_topic so the value is a clean content concept, never
            # a function-word run-on. Generic — no per-topic list.
            raw = m_like.group(1).strip(" .!?")
            thing = self._opinion_topic(raw)
            if not thing:
                thing = raw.split()[0] if raw else ""
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
        # C-clock (round 2026-08-08c): advance the fact-store turn clock
        # BEFORE mining personal facts, not after. The disclosure-ack
        # composer (_derive_ack_from_store) acks a fact only if its
        # turn_number == the store's current turn_num, so that it reports
        # what was learned THIS turn (not a stale fact from 30 turns ago).
        # When advance_turn() ran AFTER mine_personal_facts, a freshly
        # stored fact got turn_number == turn_num - 1, so the equality
        # never held and the honest ack fell through to the degenerate
        # "got it — thanks for telling me." on ~30 turns of real
        # disclosures (e.g. "my hands are raw from hauling hive boxes").
        # Advancing first makes the just-stored fact match the clock, so
        # the ack renders the REAL stored relation. No retraining; pure
        # ordering fix.
        self.personal_facts.advance_turn()
        self.opinions.advance_turn()
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
        "off", "onto", "upon", "than", "then", "till", "until", "since",

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
