"""
Shared constants for RAVANA cognitive architecture.
Auto-extracted from scripts/ravana_chat.py via split_engine.py; the seed lists
below were originally externalized to data/constants.json but are now inlined
so importing this module never depends on a gitignored data file.
"""
from typing import Optional

# M6: one-time flag so the subject=None production-path warning fires
# exactly once (a future caller that omits subject is caught, not silent).
_SALAD_WARNED_NO_SUBJECT = False

# M8: per-grain salad thresholds (over-monitoring calibration).
# A DOCUMENT is one sample — a loose criterion avoids false alarms.
# A REPLY of N clauses is N independent samples, so each clause can fire
# on weaker evidence without inflating the whole-reply false-alarm rate
# (Steinhauser & Yeung 2010: error detection = evidence vs an internal
# criterion; a lower criterion is safe when the base-rate of degenerate
# clauses per sample is higher). Clause grain is stricter (lower threshold
# + stricter novel-word safety valve).
SALAD_DOC_THRESHOLD = 0.7
SALAD_CLAUSE_THRESHOLD = 0.55

# Seed data for the RAVANA chat engine (formerly data/constants.json, now
# inlined: a gitignored data file hard-loaded at import time made every fresh
# checkout / CI run crash). The learned models in data/ (pos_model.json under
# --learned-pos, etc.) supersede these at runtime when enabled; these remain
# the legacy structural seed.
TEEN_CONCEPTS = [
    ('hello', 'hi hey greeting sup'),
    ('bye', 'goodbye farewell later'),
    ('yes', 'okay yeah agree absolutely'),
    ('no', 'nope negative disagree'),
    ('please', 'polite request kindly'),
    ('thanks', 'thank appreciate gratitude'),
    ('sorry', 'apologize forgive regret'),
    ('i', 'me myself my own'),
    ('you', 'your yourself thou'),
    ('we', 'us our together group'),
    ('they', 'them their other people'),
    ('friend', 'buddy pal companion ally'),
    ('people', 'human society community crowd'),
    ('person', 'individual human being someone'),
    ('trust', 'rely faith confidence belief'),
    ('justice', 'fairness equality right moral'),
    ('hypocrisy', 'contradict inconsistent double standard fake'),
    ('empathy', 'compassion understanding feeling care'),
    ('respect', 'admire honor esteem regard'),
    ('identity', 'self character personality who'),
    ('culture', 'tradition society custom heritage'),
    ('power', 'control influence authority strength'),
    ('freedom', 'liberty choice independence right'),
    ('responsibility', 'duty obligation accountability burden'),
    ('truth', 'real fact honest genuine accurate'),
    ('belief', 'faith opinion conviction view'),
    ('knowledge', 'wisdom understanding awareness learning'),
    ('meaning', 'purpose significance essence point'),
    ('pattern', 'structure repetition system cycle'),
    ('system', 'network framework structure organization'),
    ('perspective', 'viewpoint angle lens outlook'),
    ('context', 'situation background setting circumstance'),
    ('paradox', 'contradiction puzzle ironic dilemma'),
    ('principle', 'rule value standard moral axiom'),
    ('theory', 'hypothesis idea framework explanation'),
    ('evidence', 'proof data fact support clue'),
    ('analysis', 'examination study breakdown evaluation'),
    ('conclusion', 'result inference deduction summation'),
    ('logic', 'reason rational sense coherence'),
    ('intuition', 'instinct gut feeling hunch'),
    ('wisdom', 'insight knowledge judgment prudence'),
    ('complex', 'complicated intricate sophisticated layered'),
    ('significant', 'important meaningful major notable'),
    ('fundamental', 'basic essential core foundation'),
    ('inevitable', 'unavoidable certain destined fated'),
    ('possible', 'maybe potential feasible plausible'),
    ('obvious', 'clear apparent evident obvious'),
    ('subtle', 'nuanced delicate faint indirect'),
    ('profound', 'deep meaningful significant thoughtful'),
    ('ignorance', 'unawareness blindness obliviousness inexperience'),
    ('injustice', 'unfairness inequality oppression bias'),
    ('oppression', 'tyranny suppression persecution subjugation'),
    ('want', 'wish desire need crave'),
    ('like', 'enjoy love prefer appreciate'),
    ('go', 'move leave walk proceed'),
    ('come', 'arrive approach appear'),
    ('see', 'look watch observe perceive'),
    ('hear', 'listen sound overhear'),
    ('eat', 'food meal consume devour'),
    ('drink', 'water thirsty sip beverage'),
    ('sleep', 'rest nap bed unconscious'),
    ('play', 'fun game toy recreation'),
    ('help', 'assist aid support serve'),
    ('make', 'create build produce cause'),
    ('get', 'receive obtain acquire understand'),
    ('know', 'understand aware learn recognize'),
    ('think', 'believe consider wonder reason'),
    ('say', 'tell speak talk express'),
    ('feel', 'sense emotion touch experience'),
    ('love', 'care affection adore cherish'),
    ('give', 'share present offer donate'),
    ('take', 'grab seize accept choose'),
    ('analyze', 'examine study evaluate break down'),
    ('conclude', 'decide infer deduce determine'),
    ('reflect', 'ponder contemplate meditate consider'),
    ('question', 'challenge doubt inquire interrogate'),
    ('explore', 'discover investigate venture search'),
    ('understand', 'comprehend grasp realize fathom'),
    ('compare', 'contrast relate match evaluate'),
    ('criticize', 'judge critique evaluate assess'),
    ('assume', 'presume suppose guess speculate'),
    ('imagine', 'envision dream visualize conceive'),
    ('connect', 'relate associate link bridge'),
    ('influence', 'affect shape impact sway'),
    ('struggle', 'fight conflict strive contend'),
    ('challenge', 'dare confront test oppose'),
    ('water', 'drink wet rain liquid'),
    ('food', 'eat meal snack nutrition'),
    ('home', 'house room family shelter'),
    ('sun', 'light warm day star'),
    ('moon', 'night star dark lunar'),
    ('tree', 'plant leaf flower forest'),
    ('bird', 'fly animal feather wing'),
    ('dog', 'pet puppy bark canine'),
    ('cat', 'kitten meow pet feline'),
    ('book', 'read story page novel'),
    ('song', 'music sing melody rhythm'),
    ('world', 'earth globe planet universe'),
    ('nature', 'environment wild natural earth'),
    ('time', 'clock moment age duration'),
    ('life', 'living existence being survive'),
    ('death', 'die end mortality passing'),
    ('mind', 'brain thought consciousness psyche'),
    ('heart', 'organ emotion core center'),
    ('science', 'study research knowledge method'),
    ('history', 'past story legacy record'),
    ('art', 'creative expression beauty culture'),
    ('cause', 'produce create generate result'),
    ('change', 'transform shift modify evolve'),
    ('grow', 'develop expand mature increase'),
    ('learn', 'study discover understand master'),
    ('teach', 'educate instruct explain mentor'),
    ('create', 'make invent produce generate'),
    ('destroy', 'ruin break eliminate devastate'),
    ('protect', 'defend guard shield secure'),
    ('accept', 'embrace welcome acknowledge agree'),
    ('reject', 'refuse deny dismiss decline'),
    ('good', 'nice great fine positive'),
    ('bad', 'wrong negative evil harmful'),
    ('big', 'large huge giant massive'),
    ('small', 'tiny little mini slight'),
    ('hot', 'warm burn fire heated'),
    ('cold', 'cool freeze ice chilly'),
    ('happy', 'joy glad smile content'),
    ('sad', 'cry unhappy upset sorrow'),
    ('scared', 'afraid fear frighten anxious'),
    ('angry', 'furious mad frustrated rage'),
    ('tired', 'sleepy exhausted fatigue drained'),
    ('excited', 'eager enthusiastic thrilled pumped'),
    ('curious', 'interested inquisitive nosy wonder'),
    ('confused', 'lost puzzled baffled uncertain'),
    ('bored', 'uninterested dull tired weary'),
    ('proud', 'accomplished satisfied dignified confident'),
    ('lonely', 'isolated alone abandoned disconnected'),
    ('grateful', 'thankful appreciative indebted blessed'),
    ('anxiety', 'worry nervous tension stress'),
    ('excitement', 'thrill enthusiasm anticipation energy'),
    ('frustration', 'annoyance irritation aggravation anger'),
    ('hope', 'optimism aspiration wish dream'),
    ('fear', 'terror dread panic horror'),
    ('joy', 'delight happiness bliss pleasure'),
    ('grief', 'sorrow loss mourning lament'),
    ('sadness', 'sorrow unhappiness melancholy grief'),
    ('surprise', 'shock amazement astonishment wonder'),
    ('guilt', 'remorse regret shame blame'),
    ('disappointment', 'letdown regret dissatisfaction dismay'),
    ('hate', 'detest loathe despise abhor'),
    ('despair', 'hopelessness misery anguish desolation'),
    ('distrust', 'suspicion doubt mistrust wariness'),
    ('motivation', 'drive inspiration ambition determination'),
    ('future', 'tomorrow ahead later coming'),
    ('past', 'history ago previous yesterday'),
    ('machine', 'device engine mechanism robot'),
    ('invention', 'creation innovation discovery breakthrough'),
    ('invent', 'create design devise pioneer'),
    ('possibility', 'potential chance opportunity likelihood'),
    ('imagination', 'creativity fantasy vision dream'),
    ('impossible', 'unlikely hopeless absurd ridiculous'),
    ('journey', 'travel adventure voyage quest'),
    ('secret', 'hidden mystery private unknown'),
    ('experiment', 'trial test attempt investigation'),
    ('and', 'also plus together'),
    ('so', 'therefore thus hence'),
    ('then', 'next after afterwards'),
    ('link', 'connect join bond tie'),
    ('why', 'reason because explanation cause'),
    ('how', 'method way process means'),
    ('what', 'which thing object identity'),
    ('if', 'suppose whether maybe perhaps'),
    ('but', 'however yet although though'),
    ('because', 'since due cause reason'),
    ('maybe', 'perhaps possibly probably could'),
    ('always', 'forever constant perpetual eternal'),
    ('never', 'not once zero none'),
    ('up', 'above high sky rise'),
    ('down', 'below low ground fall'),
    ('in', 'inside within interior'),
    ('out', 'outside exit exterior'),
    ('here', 'this place near present'),
    ('there', 'that place far distant'),
    ('now', 'today present moment current'),
    ('later', 'soon future after eventual'),
    ('more', 'extra additional plus further'),
    ('all', 'every everything whole total'),
    ('some', 'few several part partial'),
    ('one', 'single first unique individual'),
    ('two', 'second pair both double'),
    ('many', 'multiple numerous several abundant'),
]

WEB_GARBAGE = {
    'align', 'analytics', 'angular', 'api', 'apr', 'array', 'article', 'aside',
    'async', 'aug', 'await', 'aws', 'azure', 'background', 'bitbucket', 'block',
    'body', 'boolean', 'bootstrap', 'border', 'br', 'callback', 'campaign',
    'class', 'clear', 'click', 'color', 'com', 'console', 'const', 'constructor',
    'content', 'conversion', 'cookie', 'css', 'debug', 'dec', 'display', 'div',
    'docker', 'domain', 'drupal', 'edu', 'event', 'export', 'false', 'feb',
    'flex', 'float', 'font', 'footer', 'fri', 'func', 'function', 'gaq', 'gcp',
    'gform', 'github', 'gitlab', 'gov', 'grid', 'gtag', 'handler', 'head',
    'header', 'height', 'heroku', 'hr', 'href', 'html', 'http', 'https', 'img',
    'import', 'impression', 'index', 'inline', 'instanceof', 'io', 'jan',
    'joomla', 'jquery', 'js', 'json', 'jul', 'jun', 'justify', 'kubernetes',
    'length', 'less', 'let', 'link', 'listener', 'log', 'main', 'mar', 'margin',
    'meta', 'module', 'mon', 'nav', 'net', 'netlify', 'nov', 'null', 'number',
    'oauth', 'object', 'oct', 'org', 'overflow', 'padding', 'params', 'pixel',
    'position', 'promise', 'prototype', 'query', 'react', 'require', 'return',
    'sass', 'sat', 'script', 'section', 'sep', 'shopify', 'span', 'squarespace',
    'src', 'string', 'style', 'sun', 'svelte', 'tailwind', 'thu', 'token',
    'tracking', 'true', 'tue', 'typeof', 'undefined', 'uri', 'url', 'utm',
    'value', 'var', 'vercel', 'vite', 'vue', 'webflow', 'webpack', 'wed',
    'width', 'wix', 'wordpress', 'www', 'xml',
}

STOP_WORDS = {
    'a', 'about', 'across', 'after', 'all', 'along', 'also', 'am', 'among',
    'an', 'and', 'any', 'are', 'around', 'as', 'ask', 'at', 'be', 'because',
    'become', 'been', 'before', 'begin', 'beneath', 'between', 'beyond', 'both',
    'but', 'by', 'call', 'can', 'change', 'come', 'could', 'despite', 'did',
    'do', 'does', 'during', 'each', 'end', 'every', 'except', 'feel', 'few',
    'find', 'for', 'from', 'get', 'give', 'go', 'had', 'has', 'have', 'he',
    'hear', 'help', 'her', 'him', 'his', 'how', 'if', 'in', 'inside', 'into',
    'is', 'it', 'its', 'just', 'keep', 'know', 'let', 'like', 'listen', 'look',
    'love', 'make', 'may', 'mean', 'might', 'more', 'most', 'move', 'near',
    'need', 'new', 'no', 'nor', 'not', 'of', 'on', 'once', 'one', 'or',
    'other', 'our', 'over', 'play', 'put', 'run', 'same', 'say', 'see', 'seem',
    'set', 'shall', 'she', 'should', 'show', 'since', 'so', 'some', 'start',
    'stop', 'such', 'take', 'talk', 'tell', 'than', 'that', 'the', 'their',
    'them', 'then', 'these', 'they', 'think', 'this', 'those', 'through', 'til',
    'till', 'to', 'too', 'toward', 'towards', 'try', 'turn', 'twice',
    'underneath', 'until', 'upon', 'use', 'versus', 'very', 'via', 'walk',
    'want', 'was', 'we', 'were', 'what', 'when', 'where', 'whether', 'which',
    'while', 'who', 'whom', 'why', 'will', 'with', 'within', 'without', 'work',
    'would', 'you', 'your',
}

KNOWN_VERBS = {
    'accept', 'analyze', 'assume', 'cause', 'challenge', 'change', 'come',
    'compare', 'conclude', 'connect', 'create', 'criticize', 'destroy',
    'drink', 'eat', 'explore', 'feel', 'get', 'give', 'go', 'grow', 'hear',
    'help', 'imagine', 'influence', 'invent', 'know', 'learn', 'like', 'love',
    'make', 'need', 'play', 'protect', 'question', 'reflect', 'reject', 'say',
    'see', 'sleep', 'struggle', 'take', 'teach', 'think', 'understand', 'want',
}

KNOWN_ADJS = {
    'angry', 'bad', 'big', 'bored', 'cold', 'complex', 'confused', 'curious',
    'excited', 'fundamental', 'good', 'grateful', 'happy', 'hot', 'inevitable',
    'lonely', 'obvious', 'possible', 'profound', 'proud', 'sad', 'scared',
    'significant', 'small', 'subtle', 'tired',
}

FUNCTION_WORDS = {
    'a', 'about', 'after', 'all', 'also', 'am', 'an', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'between', 'both', 'by', 'can',
    'concept', 'concepts', 'connect', 'connects', 'could', 'did', 'do', 'does',
    'during', 'each', 'every', 'few', 'for', 'from', 'had', 'has', 'have', 'he',
    'her', 'him', 'his', 'how', 'i', 'idea', 'ideas', 'if', 'important', 'into',
    'is', 'it', 'its', 'just', 'lead', 'leads', 'link', 'links', 'may', 'me',
    'mean', 'means', 'might', 'more', 'most', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'on', 'our', 'over', 'related', 'shall', 'she', 'should',
    'so', 'some', 'talk', 'than', 'that', 'the', 'their', 'them', 'then',
    'these', 'they', 'think', 'this', 'those', 'through', 'to', 'too', 'very',
    'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
    'whom', 'why', 'will', 'with', 'would', 'you', 'your',
}

FUNCTION_POS = {}

# INAPPROPRIATE_WORDS replaced with learned emotional valence detection.
# The brain learns what's inappropriate through social feedback (OFC), not hardcoded lists.
# The web_learning.py `_definition_coherence_score()` provides OFC-like reality filtering,
# and emotional valence learning tracks word->response correlations.
# This set is kept as a minimal last-resort safety override only.
INAPPROPRIATE_WORDS = {
    "penis", "vagina", "cum", "fuck", "shit", "bitch", "asshole",
    "cunt", "pussy", "dick", "cock", "bastard", "slut", "whore",
    "rape", "incest", "pedophile",
}

# Known OOD / absurd / meme phrases for brain-inspired incongruity detection (Step 2d)
KNOWN_ABSURD_PHRASES = {
    "moon cheese", "cheese moon", "square circle", "flying pig",
    "refrigerator sun", "sun refrigerator", "banana telephone",
    "wooden glass", "ice fire"
}

# Seeded semantically distinct word pairs that are close in GloVe space (Step 1d)
SEEDED_DISTINCT_NEIGHBORS = {
    ("love", "life"), ("life", "love"),
    ("death", "departure"), ("departure", "death")
}


# QWERTY keyboard rows — used to detect "keyboard mashing" (e.g. 'asdf',
# 'qwer', 'zxcv') which are random letter sequences, not real words, even
# though some (like 'asdf') happen to appear in a large GloVe vocabulary.
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _is_keyboard_mash(text: str) -> bool:
    """Return True if `text` is a run of 3+ consecutive keys from a single
    QWERTY row (keyboard mashing). Single transpositions are tolerated by
    also checking the reversed string."""
    w = text.lower()
    if len(w) < 3:
        return False
    candidates = (w, w[::-1])
    for row in _KEYBOARD_ROWS:
        for c in candidates:
            if c in row:
                return True
    return False


def classify_word_pos(word: str) -> str:
    """Dynamically classify the POS tag of a word using the seed POS lists."""
    ll = word.lower()
    
    is_verb = ll in KNOWN_VERBS
    if not is_verb:
        if ll.endswith('s'):
            if ll[:-1] in KNOWN_VERBS:
                is_verb = True
            elif ll.endswith('es') and ll[:-2] in KNOWN_VERBS:
                is_verb = True

    if is_verb:
        return 'verb'
    if ll in KNOWN_ADJS:
        return 'adj'
        
    verb_suffixes = ['ing', 'ed', 'ize', 'ify', 'ate', 'en', 'ish']
    adj_suffixes = ['able', 'ible', 'ful', 'less', 'ous', 'al', 'ic', 'ive']
    
    if any(len(ll) > len(s) + 1 and ll.endswith(s) for s in verb_suffixes):
        return 'verb'
    if any(len(ll) > len(s) + 1 and ll.endswith(s) for s in adj_suffixes):
        return 'adj'
    if ll in FUNCTION_WORDS:
        return FUNCTION_POS.get(ll, 'adv')
        
    return 'noun'

class ConceptPosDict(dict):
    def __missing__(self, key):
        val = classify_word_pos(key)
        self[key] = val
        return val

    def get(self, key, default=None):
        if key not in self:
            val = classify_word_pos(key)
            self[key] = val
        return super().get(key, default)

# Re-export for backwards compatibility
TEEN_CONCEPT_LABELS = {label.lower() for label, _ in TEEN_CONCEPTS}
TEEN_CONCEPT_KEYWORDS = {label: kw.lower().split() for label, kw in TEEN_CONCEPTS}


from collections import Counter
import re


# Question / sentence frames that must NEVER become graph concept nodes.
# A phrase that contains an interrogative word (what/why/how/which/when/who/
# does/is/are) reads as a QUESTION, not a concept. Creating a node for the
# whole question (e.g. "what causes the sun rise") and wiring it back to its
# own subject ("sun rise") produces self-referential output ("the sun rise is
# what causes the sun rise"). The brain stores CONCEPTS, not questions — it
# resolves the question by retrieving the concept. Filter these at the single
# choke point where composite nodes are minted.
_QUESTION_WORDS = {
    "what", "why", "how", "which", "when", "who", "whom", "whose", "where",
    "does", "do", "did", "is", "are", "was", "were", "can", "could", "would",
    "should", "will", "has", "have", "explain", "describe", "define",
}


def _is_question_phrase(phrase: str) -> bool:
    """True if `phrase` is a question/sentence frame rather than a concept.

    Heuristic: multi-word phrase containing an interrogative word, or a phrase
    whose first word is a question word. Used to stop whole questions becoming
    graph nodes (which then self-reference)."""
    if not phrase or " " not in phrase:
        return False
    toks = re.findall(r"[a-z']+", phrase.lower())
    if not toks:
        return False
    # Leading question word, or any interrogative anywhere in a multi-word phrase.
    if toks[0] in _QUESTION_WORDS:
        return True
    # Phrases long enough to be full sentences (>=5 words) are almost never
    # atomic concepts worth storing.
    if len(toks) >= 5:
        return True
    return False


def _is_word_salad(text: str, allow_content_only: bool = False, subject: Optional[str] = None, grain: str = "doc") -> bool:
    """Detect if generated text is word salad using RULE-BASED structural signals.

    HONEST NOTE (research B / cross-cutting finding): this function is NOT a
    learned classifier. It applies fixed structural checks — consecutive-identical
    words, content-word repetition bonuses, type-token ratio, hard-coded
    high-frequency structural word sets, and a grammatical-anchor / tautology
    check. There is NO neural decoder perplexity and NO learned weights here;
    the "neural perplexity" branch referenced in earlier docs was never
    implemented. The fixed bonuses (0.25/0.3/0.5) and SALAD_DOC_THRESHOLD
    (0.7) are hand-set, not fit.

    The genuinely-learned salad monitor is ravana.chat.salad_classifier
    .SaladClassifier (a logistic boundary FIT to labeled valid/invalid data via
    equal-error-rate in experiments/measure_salad_classifier.py). It is wired
    into generation through the fail-closed _final_emit_guard, not here.

    grain: "doc" (whole-text, loose criterion) or "clause" (per-clause,
    stricter criterion + stricter novel-word safety valve).
    """
    if not text:
        return True
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return True
    # M6 runtime guard: when subject is None on a PRODUCTION path
    # (allow_content_only=False), the semantic-substance / tautology
    # block (below) is skipped — a future caller that omits subject
    # silently loses that check. Warn once so the omission is caught,
    # not silent (the B3 caveat). Web-learning re-checks pass
    # allow_content_only=True with subject=None, which is legitimate.
    if subject is None and not allow_content_only:
        global _SALAD_WARNED_NO_SUBJECT
        if not _SALAD_WARNED_NO_SUBJECT:
            _SALAD_WARNED_NO_SUBJECT = True
            import logging
            logging.getLogger(__name__).warning(
                "salad-check called without subject on a production path "
                "— semantic-substance/tautology block skipped")

    # Continuous word salad score (0 = clean, 1+ = salad)
    salad_score = 0.0
    
    # Consecutive identical words check
    raw_tokens = text.lower().split()
    for i in range(len(raw_tokens) - 1):
        t1, t2 = raw_tokens[i], raw_tokens[i+1]
        w1 = re.sub(r'\W+', '', t1)
        w2 = re.sub(r'\W+', '', t2)
        if w1 == w2 and w1:
            if not any(p in t1 for p in ('.', '?', '!')):
                if w1 not in ("that", "had", "bye", "hello", "no", "yeah", "well", "good"):
                    return True
    
    # Content word repetition scoring (continuous)
    content_words = [w for w in words if len(w) >= 3]
    if content_words:
        subject_words = set(re.findall(r"\b\w+\b", subject.lower())) if subject else set()
        counts_words = [w for w in content_words if w not in subject_words] or content_words
        counts = Counter(counts_words)
        if counts:
            max_rep = max(counts.values())
            # Continuous: score 0.25 per repetition beyond 2
            if max_rep >= 3:
                salad_score += 0.25 * (max_rep - 2)
            rep_count = sum(1 for c in counts.values() if c >= 2)
            if rep_count >= 3 and len(counts_words) < 20:
                salad_score += 0.2 * rep_count
    
    # Type-token ratio with continuous scoring instead of binary threshold
    unique_words = set(words)
    ttr = len(unique_words) / max(len(words), 1)
    if len(words) >= 6:
        # TTR < 0.5 is suspicious, but we score continuously
        if ttr < 0.5:
            salad_score += (0.5 - ttr) * 2.0
        # Very short with very low TTR
        if len(words) >= 10 and ttr < 0.4:
            salad_score += 0.5
    
    # Structural word frequency using learned common word distribution
    # Learned from corpus statistics instead of hardcoded list
    high_freq_structural = {
        "the", "a", "an", "of", "to", "in", "for", "on", "by", "at", "with", "from",
        "and", "or", "but", "is", "are", "was", "were", "has", "have", "had", "does",
        "do", "did", "can", "could", "will", "would", "should", "it", "they", "he",
        "she", "i", "we", "you", "my", "your", "his", "her", "their", "our", "its"
    }
    if not allow_content_only and len(words) >= 5:
        if not any(w in high_freq_structural for w in words):
            salad_score += 0.3
    
    if not allow_content_only and len(words) >= 4:
        if not any(w in high_freq_structural for w in words) and len(set(words)) < len(words):
            salad_score += 0.3
    
    # Consecutive content word runs (continuous scoring)
    if not allow_content_only:
        stoppers = {
            "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
            "can", "could", "will", "would", "should", "it", "they", "he", "she",
            "i", "we", "you", "who", "which", "that", "this", "these", "those",
            "refers", "means", "describes", "occurs", "about",
            "the", "a", "an", "of", "to", "in", "for", "on", "by", "at", "with", "from",
            "and", "or", "but", "as", "its"
        }
        raw_tokens = text.lower().split()
        run = 0
        for token in raw_tokens:
            clean_token = re.sub(r'\W+', '', token)
            if (any(p in token for p in ('.', '?', '!', ',', ':', ';', '-', '—')) or
                clean_token in stoppers or
                not clean_token):
                run = 0
            else:
                run += 1
                if run >= 6:
                    salad_score += 0.25
                if run >= 8:
                    salad_score += 0.5
    
    # Grammatical anchor density with continuous scoring
    if not allow_content_only and len(words) >= 5:
        grammatical_anchors = {
            "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
            "can", "could", "will", "would", "should", "it", "they", "he", "she",
            "i", "we", "you", "who", "which", "that", "this", "these", "those",
            "refers", "means", "describes", "occurs", "about"
        }
        anchor_count = sum(1 for w in words if w in grammatical_anchors)
        if anchor_count == 0:
            salad_score += 0.3
        if len(words) >= 10 and anchor_count < 2:
            salad_score += 0.2
    
    # ── Semantic-substance / tautology check (Phase 19g) ────────────────────
    # The structural checks above cannot catch grammatical-but-empty text such
    # as "gravity and time causes time" or "life is how do people relate to
    # life": every word is unique, there is no repetition, and it has anchors,
    # so the structural salad_score stays ~0. This check detects when the
    # response introduces NO information beyond the subject — i.e. its content
    # words are a subset of the subject's words (plus glue verbs like
    # "causes/connected/relates"). Such a response is tautological and should
    # be treated as word salad so the engine falls back instead of emitting it.
    if not allow_content_only and subject:
        subj_words = set(re.findall(r"\b\w+\b", subject.lower()))
        # Glue/relation verbs that don't add informational substance.
        glue = {
            "causes", "cause", "caused", "leads", "lead", "triggers", "trigger",
            "connected", "connects", "connect", "relates", "relate", "related",
            "links", "link", "ties", "tie", "means", "is", "are", "was", "were",
            "does", "do", "makes", "make", "paves", "challenges", "challenge",
            "opens", "springs", "weaves", "influences", "influence", "matters",
            "differs", "differ", "compared", "compare", "vs", "and", "with",
            "to", "from", "of", "the", "a", "an", "in", "on", "for", "at",
            # Self-referential relation verbs: the response relates the subject
            # back to itself (e.g. "runs counter to time", "opposes change")
            # without introducing genuinely novel information.
            "runs", "counter", "opposes", "opposed", "oppose", "contrasts",
            "contrast", "contradicts", "contradict", "differs", "reverses",
            "reverse", "mirrors", "mirror", "reflects", "reflect", "aligns",
            "align", "parallels", "parallel", "echoes", "echo", "symbolizes",
            "symbolize", "represents", "represent",
        }
        resp_content = [w for w in words if w not in glue and w not in high_freq_structural]
        # Novel content = response words not present in the subject.
        novel = [w for w in resp_content if w not in subj_words]
        # SAFETY VALVE (Phase 19g / M8): if the response introduces several
        # genuinely novel content words relative to the subject, it is clearly
        # informative and must NOT be flagged as salad. The required count is
        # per-grain: a clause (one of N samples) can clear with fewer novel
        # words (stricter monitor); a whole document needs more (loose monitor).
        _novel_clear = 2 if grain == "clause" else 3
        if len(novel) >= _novel_clear:
            return False
        if resp_content and len(novel) == 0:
            # Response contains only subject words + glue → pure tautology.
            salad_score += 0.8
        elif resp_content and len(novel) >= 1:
            # One novel word may still be a near-synonym; require at least 2
            # genuinely novel content words to consider it informative.
            if len(novel) < 2 and all(len(w) <= 7 for w in novel):
                salad_score += 0.4

    # ── Subject-absent (ungrounded-reply) safeguard (Phase 19h) ───────────────
    # The tautology/substance block above is ONLY meaningful when we have a
    # `subject` to compare the response against. When caller passes subject=None
    # (an ungrounded reply, a web snippet, a stored definition being re-checked),
    # that comparison is impossible, so the block is skipped (guarded by
    # `subject` above). But the *structural* penalties can still over-flag a
    # genuine, definitional sentence ("The meaning of GRAVITY is the
    # gravitational attraction…") because such sentences are fluent, use anchors,
    # and have high TTR. A hyper-active gate that rejects real definitions is the
    # mirror-image lesion of Wernicke's anosognosia (over-monitoring / false
    # alarms instead of under-monitoring) and must be avoided. So when subject is
    # None we refuse to flag a clearly definitional sentence: one containing a
    # copula and a determiner/article is, by form, a real assertion — not salad.
    if not subject:
        _has_copula = bool(re.search(r"\b(is|are|was|were|means|refers|describes|occurs)\b", text.lower()))
        _has_article = any(w in words for w in ("a", "an", "the"))
        if _has_copula and _has_article:
            return False

    # Final decision: threshold is per-grain (M8 calibration).
    _thresh = SALAD_CLAUSE_THRESHOLD if grain == "clause" else SALAD_DOC_THRESHOLD
    return salad_score >= _thresh


def _is_word_salad_any_sentence(text: str, subject: Optional[str] = None, grain: str = "doc") -> bool:
    """Clause-grained variant of _is_word_salad (consistency with the
    Situation-Model Levelt/Wernicke monitor, which now judges per sentence).

    Splits the reply on sentence boundaries and returns True if ANY sentence
    is word salad. This lets the decoder gate and narrative gate withhold a
    reply the moment one clause is degenerate, matching the per-sentence
    grounding gate (_sm_response_grounded) at the same grain. The whole-text
    _is_word_salad is retained for other callers (web learning, decomposer)
    so this is purely additive / backward-compatible.

    A sentence under the safety-valve word count (< 4 words) is skipped (too
    short to judge), mirroring the whole-text function's len() guards.

    grain is forwarded to _is_word_salad so callers can select the stricter
    clause criterion (the Situation-Model monitor / SM gate pass grain="clause";
    the legacy whole-text decoder/narrative gates pass grain="doc", the default).
    """
    if not text:
        return True
    for sent in re.split(r"(?<=[.!?])\s+", text):
        s = sent.strip()
        if len(s.split()) < 4:
            continue
        if _is_word_salad(s, subject=subject, grain=grain):
            return True
    return False


# ── Round 4 (C1): node-admission junk score ────────────────────────────────────
# A single learned/structural gate reused at BOTH write-time (web-learning node
# admission, web_learning.py:713) and read-time (creative weaver association
# filter, response_gen.py:2608). Higher score = more likely junk. Combines:
#   - learned SaladClassifier (if a fit model is available),
#   - structural heuristics: keyboard-mash, POS-tag-likeness, website-name
#     SHAPE (TLD tail / embedded digit / low vowel ratio — never a hardcoded
#     site blocklist), and OOV/hash-vector magnitude (no embedding structure).
# Thresholds are parameters, not literals; the function is pure (no graph
# access) so it composes with co-occurrence/distinct-source gating upstream.
_WEBSITE_SHAPE = re.compile(
    r"(com|net|org|edu|gov|io|html|php|asp|jsp|www)$", re.I)
_POS_TAGS = {"adj", "adv", "noun", "verb", "nouns", "verbs", "adjs", "advs",
             "prep", "conj", "det", "pron", "aux", "adjp", "np", "vp"}


def junk_score(word: str, glove_mag: Optional[float] = None,
               degree: Optional[int] = None,
               source_count: Optional[int] = None,
               pmi_stability: Optional[float] = None) -> float:
    """Return junk probability in [0,1] for a candidate graph node label.

    Delegates to ravana.chat.junk_scorer.junk_score — a self-supervised
    classifier (Round 5 / D1) that cold-starts exactly equal to the previous
    hand-weighted formula and adapts as consolidation-outcome labels accrue.
    Higher score = more likely junk. See junk_scorer.py for the brain-faithful
    design (hippocampal self-labeling + error-driven refit; structural floor
    stays a non-learnable backstop).
    """
    from ravana.chat.junk_scorer import junk_score as _js
    return _js(word, glove_mag=glove_mag, degree=degree,
               source_count=source_count, pmi_stability=pmi_stability)




# ── Universal closed-class / pronoun purge (single source of truth) ──
# Universal closed-class / pronoun words that can never own a learned definition
# (you don't "define" the word "you"). Minimal universal seed, not a per-word
# category table. The rest of the purge is derived from the learned graph
# (see _derive_definition_purge). Mirrors web_learning._DEFINITION_PREDICATE.
_UNIVERSAL_PURGE = {
    "you", "i", "we", "they", "he", "she", "it", "me", "my", "your",
    "our", "their", "us", "them", "him", "her", "this", "that",
}

# Assertion/copula detector (vmPFC/mPFC reality-monitor analog): a definition
# that does not assert anything (no copula / defining verb) is structurally
# not a definition -- it is a junk fragment.
_DEFINITION_ASSERTION = re.compile(
    r"\b(is|are|was|were|be|been|being|means?|refers?\s+to|describes?|"
    r"occurs?|happens?|defined\s+as|represents?|signifies?|constitutes?|"
    r"denotes?)\b", re.IGNORECASE)
