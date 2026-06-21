"""
Emotional Mirror Engine — Mirror Neuron System for RAVANA.

Neuroscience basis:
- Gallese & Goldman (1998): "Mirror neurons and the simulation theory
  of mind-reading" — mirror neurons enable experiential understanding
  of others' actions and emotions via embodied simulation.
- Rizzolatti & Craighero (2004): "The mirror-neuron system" — a
  cortical system matching observation and execution of actions.
- Gallese, Keysers & Rizzolatti (2004): "A unifying view of the basis
  of social cognition" — shared neural substrates for first- and
  third-person experience of actions AND emotions (insula/amygdala
  for emotional mirroring).
- Hatfield, Cacioppo & Rapson (1994): "Emotional contagion" —
  automatic mimicry → afferent feedback → emotional convergence.
- Wicker et al. (2003): "Both of us disgusted in my insula" —
  shared insula activation for experiencing and observing disgust.

RAVANA mapping:
  User text → Emotion Detection → Mirror Neuron Response →
  Modulation of temperature, concept breadth, verbosity

The mirror neuron system in humans (inferior frontal gyrus + inferior
parietal lobule) maps observed actions onto one's own motor repertoire.
For emotions, a similar viscero-motor mapping occurs via insula and
amygdala. This engine implements this mapping computationally: user
emotional state is detected from text, mirrored into RAVANA's VAD state,
and the mirrored state modulates response generation parameters.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Lexicon: word → (valence, arousal, dominance) perturbations
# Built from ANEW (Affective Norms for English Words) + Warriner et al. (2013)
# Valence: -1 (unpleasant) to +1 (pleasant)
# Arousal: 0 (calm) to 1 (excited)
# Dominance: -1 (submissive) to +1 (in control)
_VAD_LEXICON: Dict[str, Tuple[float, float, float]] = {
    # High-arousal positive
    "excited":        (0.80, 0.85, 0.60),
    "amazing":        (0.90, 0.80, 0.70),
    "awesome":        (0.85, 0.75, 0.65),
    "fantastic":      (0.90, 0.80, 0.70),
    "thrilled":       (0.85, 0.90, 0.65),
    "ecstatic":       (0.95, 0.90, 0.60),
    "passionate":     (0.75, 0.85, 0.70),
    "energetic":      (0.65, 0.80, 0.60),
    "inspired":       (0.80, 0.70, 0.55),
    "motivated":      (0.70, 0.75, 0.65),
    "enthusiastic":   (0.80, 0.85, 0.60),
    "glad":           (0.70, 0.50, 0.45),
    "cheerful":       (0.75, 0.65, 0.50),
    "delighted":      (0.85, 0.70, 0.60),
    "wonderful":      (0.85, 0.65, 0.60),
    "joy":            (0.85, 0.70, 0.55),
    "joyful":         (0.85, 0.70, 0.55),
    "happy":          (0.75, 0.60, 0.50),
    "love":           (0.85, 0.65, 0.45),
    "beautiful":      (0.80, 0.50, 0.45),
    "grateful":       (0.75, 0.40, 0.40),
    "proud":          (0.70, 0.55, 0.70),
    "hopeful":        (0.65, 0.50, 0.40),
    "great":          (0.70, 0.55, 0.50),
    "good":           (0.60, 0.35, 0.45),
    "nice":           (0.60, 0.30, 0.40),
    "fun":            (0.70, 0.70, 0.50),
    "cool":           (0.55, 0.40, 0.55),
    "interesting":    (0.50, 0.55, 0.40),
    "curious":        (0.45, 0.60, 0.40),
    "surprised":      (0.30, 0.80, 0.30),

    # High-arousal negative
    "angry":          (-0.80, 0.85, 0.20),
    "furious":        (-0.85, 0.95, 0.30),
    "outraged":       (-0.80, 0.90, 0.25),
    "frustrated":     (-0.65, 0.75, -0.20),
    "annoyed":        (-0.50, 0.60, -0.10),
    "irritated":      (-0.55, 0.65, -0.15),
    "fear":           (-0.75, 0.85, -0.50),
    "scared":         (-0.75, 0.85, -0.50),
    "afraid":         (-0.70, 0.80, -0.45),
    "terrified":      (-0.85, 0.95, -0.60),
    "anxious":        (-0.50, 0.80, -0.35),
    "worried":        (-0.45, 0.70, -0.30),
    "nervous":        (-0.40, 0.75, -0.25),
    "stressed":       (-0.55, 0.80, -0.30),
    "panic":          (-0.75, 0.90, -0.50),
    "hate":           (-0.80, 0.75, 0.30),
    "terrible":       (-0.80, 0.70, -0.30),
    "awful":          (-0.75, 0.65, -0.30),
    "horrible":       (-0.80, 0.75, -0.35),
    "upset":          (-0.60, 0.65, -0.25),
    "hurt":           (-0.65, 0.55, -0.40),
    "pain":           (-0.70, 0.60, -0.50),
    "cry":            (-0.65, 0.60, -0.45),
    "depressed":      (-0.80, 0.20, -0.50),
    "miserable":      (-0.85, 0.25, -0.55),

    # Low-arousal negative
    "sad":            (-0.65, 0.30, -0.35),
    "sorrow":         (-0.70, 0.25, -0.40),
    "lonely":         (-0.60, 0.20, -0.40),
    "disappointed":   (-0.55, 0.35, -0.30),
    "guilty":         (-0.60, 0.45, -0.35),
    "ashamed":        (-0.55, 0.40, -0.40),
    "bored":          (-0.40, 0.10, -0.20),
    "tired":          (-0.25, 0.15, -0.10),
    "confused":       (-0.30, 0.55, -0.35),
    "lost":           (-0.40, 0.40, -0.50),

    # Low-arousal positive
    "calm":           (0.50, 0.10, 0.50),
    "peaceful":       (0.65, 0.10, 0.55),
    "relaxed":        (0.60, 0.15, 0.50),
    "content":        (0.60, 0.20, 0.45),
    "serene":         (0.65, 0.10, 0.50),
    "comfortable":    (0.55, 0.15, 0.50),
    "safe":           (0.60, 0.15, 0.60),
    "satisfied":      (0.60, 0.25, 0.50),
    "thoughtful":     (0.40, 0.30, 0.35),
    "reflective":     (0.35, 0.25, 0.30),

    # Neutral / cognitive state
    "wonder":         (0.40, 0.55, 0.30),
    "doubt":          (-0.20, 0.40, -0.25),
    "skeptical":      (-0.15, 0.45, 0.15),
    "uncertain":      (-0.20, 0.50, -0.30),
    "puzzled":        (-0.10, 0.50, -0.20),
    "thoughtful":     (0.35, 0.30, 0.35),
    "insightful":     (0.60, 0.40, 0.60),
}

# Intensifiers multiply arousal
_INTENSIFIERS = {
    "very": 1.4, "really": 1.3, "extremely": 1.6, "incredibly": 1.5,
    "so": 1.2, "totally": 1.4, "absolutely": 1.5, "completely": 1.3,
    "deeply": 1.4, "highly": 1.3, "quite": 1.2, "too": 1.1,
    "super": 1.4, "ultra": 1.5,
}

# Negation reverses valence (limited set of closed-class negators)
_NEGATORS = {"not", "no", "never", "neither", "nor", "cannot", "can't",
             "don't", "doesn't", "didn't", "won't", "wouldn't",
             "couldn't", "shouldn't", "isn't", "aren't", "wasn't",
             "weren't", "haven't", "hasn't", "hadn't"}


@dataclass
class MirrorConfig:
    """Configuration for emotional mirroring dynamics.

    Neuroscience basis:
    - Mirror strength: degree of "mirror neuron" coupling between
      observed and executed emotion (Gallese & Goldman 1998).
    - Contagion rate: Hatfield et al.'s (1994) emotional contagion
      speed — how quickly one "catches" another's emotion.
    - Rapport bias: tendency toward positive valence in sustained
      interaction (mere-exposure effect; Zajonc 1968).
    - Empathy threshold: minimum user arousal before mirroring
      engages — prevents over-mirroring of neutral statements.
    """
    mirror_strength: float = 0.55
    contagion_rate: float = 0.45
    empathy_threshold: float = 0.15
    rapport_bias: float = 0.05
    dominance_inertia: float = 0.85


@dataclass
class MirrorState:
    """Current mirroring state tracking."""
    user_valence: float = 0.0
    user_arousal: float = 0.3
    user_dominance: float = 0.5
    mirror_engagement: float = 0.0
    contagion_history: List[Tuple[float, float, float]] = field(default_factory=list)
    rapport_level: float = 0.0

    def to_dict(self) -> dict:
        return {
            'user_valence': self.user_valence,
            'user_arousal': self.user_arousal,
            'user_dominance': self.user_dominance,
            'mirror_engagement': self.mirror_engagement,
            'rapport_level': self.rapport_level,
        }

    def set_state(self, state: dict):
        self.user_valence = state.get('user_valence', 0.0)
        self.user_arousal = state.get('user_arousal', 0.3)
        self.user_dominance = state.get('user_dominance', 0.5)
        self.mirror_engagement = state.get('mirror_engagement', 0.0)
        self.rapport_level = state.get('rapport_level', 0.0)


# Stem lookup: maps common variants to their lexicon stems
_STEM_MAP = {
    "frustrating": "frustrated", "frustrates": "frustrated",
    "confusing": "confused", "confuses": "confused",
    "exciting": "excited", "excites": "excited",
    "amazes": "amazing", "amazed": "amazing",
    "annoys": "annoyed", "annoying": "annoyed",
    "irritates": "irritated", "irritating": "irritated",
    "scares": "scared", "scaring": "scared",
    "terrifies": "terrified", "terrifying": "terrified",
    "worries": "worried", "worrying": "worried",
    "depresses": "depressed", "depressing": "depressed",
    "disappoints": "disappointed", "disappointing": "disappointed",
    "inspires": "inspired", "inspiring": "inspired",
    "thrills": "thrilled", "thrilling": "thrilled",
    "relaxes": "relaxed", "relaxing": "relaxed",
    "calms": "calm",
    "hurts": "hurt", "hurting": "hurt",
    "excites": "excited",
}

# Fallback keyword sets for when no lexicon match is found
_FALLBACK_POSITIVE = {"good", "great", "nice", "happy", "love", "fun",
                       "cool", "best", "better", "win", "winning", "like"}
_FALLBACK_NEGATIVE = {"bad", "sad", "mad", "wrong", "hard", "tough",
                       "hate", "ugly", "stupid", "boring", "lost", "cry",
                       "sick", "pain", "fail", "failed", "lose"}
_FALLBACK_HIGH_AROUSAL = {"crazy", "wild", "insane", "amazing", "terrible",
                          "shocking", "intense", "incredible", "extreme"}


class UserEmotionDetector:
    """Detects user emotional state from natural language text.

    Uses a VAD lexicon (based on ANEW norms) with intensifier/negation
    modulation to infer the user's valence, arousal, and dominance from
    their word choices. Falls back to simple keyword matching when no
    lexicon words are found.

    Neuroscience basis: semantic grounding of emotion concepts in
    somato-visceral experience (Barsalou 1999; Glenberg 1997).
    """

    def __init__(self, lexicon: Optional[Dict[str, Tuple[float, float, float]]] = None):
        self._lexicon = lexicon or _VAD_LEXICON
        self._stem_map = _STEM_MAP
        self._positive = _FALLBACK_POSITIVE
        self._negative = _FALLBACK_NEGATIVE
        self._high_arousal = _FALLBACK_HIGH_AROUSAL
        self._intensifiers = _INTENSIFIERS
        self._negators = _NEGATORS

    def _lookup_word(self, word: str) -> Optional[Tuple[float, float, float]]:
        """Look up word in lexicon with stem fallback."""
        if word in self._lexicon:
            return self._lexicon[word]
        if word in self._stem_map:
            stem = self._stem_map[word]
            return self._lexicon.get(stem)
        return None

    def detect(self, text: str) -> Tuple[float, float, float]:
        """Detect VAD from text. Returns (valence, arousal, dominance)."""
        if not text or not text.strip():
            return (0.0, 0.3, 0.5)

        words = text.lower().split()
        if not words:
            return (0.0, 0.3, 0.5)

        valence_acc, arousal_acc, dominance_acc = 0.0, 0.0, 0.0
        weight_sum = 0.0
        negated = False
        lex_match_found = False

        for i, w in enumerate(words):
            w_clean = w.strip(".,!?;:\"'()[]")

            if w_clean in self._negators:
                negated = True
                continue

            if w_clean in self._intensifiers:
                continue

            entry = self._lookup_word(w_clean)
            if entry is None:
                negated = False
                continue

            lex_match_found = True
            v, a, d = entry

            intensity = 1.0
            if i > 0 and words[i - 1] in self._intensifiers:
                intensity = self._intensifiers[words[i - 1]]

            if negated:
                v = -v * 0.6
                a = a * 0.8
                negated = False

            v *= intensity
            a = np.clip(a * intensity, 0.0, 1.0)

            weight = 1.0 + a * 0.5
            valence_acc += v * weight
            arousal_acc += a * weight
            dominance_acc += d * weight
            weight_sum += weight

        if lex_match_found and weight_sum > 0:
            valence = np.clip(valence_acc / weight_sum, -1.0, 1.0)
            arousal = np.clip(arousal_acc / weight_sum, 0.0, 1.0)
            dominance = np.clip(dominance_acc / weight_sum, -1.0, 1.0)
            return (valence, arousal, dominance)

        # Fallback: simple keyword matching
        word_set = set(w.strip(".,!?;:\"'()[]") for w in words)
        sv, sa = 0.0, 0.3
        if word_set & self._positive:
            sv += 0.35
            sa += 0.15
        if word_set & self._negative:
            sv -= 0.35
            sa += 0.15
        if word_set & self._high_arousal:
            sa += 0.25
        return (np.clip(sv, -1.0, 1.0), np.clip(sa, 0.0, 1.0), 0.5)


class EmotionalMirrorEngine:
    """Mirror neuron system for emotional rapport.

    The engine implements the three-stage emotional contagion process
    described by Hatfield et al. (1994):
      1. Mimicry: detect user's emotional state from text
      2. Feedback: update system's own VAD state toward user's
      3. Contagion: modulate response parameters accordingly

    Also tracks rapport level — accumulated positive mirroring that
    deepens with sustained interaction (relationship depth correlate).
    """

    def __init__(self, config: Optional[MirrorConfig] = None):
        self.config = config or MirrorConfig()
        self.state = MirrorState()
        self.detector = UserEmotionDetector()

    def detect_user_emotion(self, text: str) -> Tuple[float, float, float]:
        """Detect user's emotional VAD from text."""
        uv, ua, ud = self.detector.detect(text)
        self.state.user_valence = uv
        self.state.user_arousal = ua
        self.state.user_dominance = ud
        return (uv, ua, ud)

    def mirror(self, vad_engine, user_text: str, turn_count: int = 0) -> None:
        """Mirror detected user emotion into the system's VAD engine.

        Three-stage emotional contagion:
        1. Detect user VAD from text (mimicry)
        2. Compute mirror stimulus toward user's VAD
        3. Update VAD engine with mirror stimulus

        Mirror strength is modulated by:
        - Empathy threshold: ignore below-arousal-threshold queries
        - Rapport level: deepens mirroring over sustained interaction
        - Dominance inertia: system retains own dominance more stubbornly
        """
        uv, ua, ud = self.detect_user_emotion(user_text)

        cfg = self.config

        # Skip mirroring if user arousal is below empathy threshold
        if ua < cfg.empathy_threshold:
            self.state.mirror_engagement *= 0.8
            if self.state.mirror_engagement < 0.01:
                self.state.mirror_engagement = 0.0
            return

        # Compute mirror stimulus: how much to move toward user's state
        sv = uv * cfg.mirror_strength + cfg.rapport_bias
        sa = ua * cfg.mirror_strength
        sd = ud * cfg.mirror_strength * cfg.dominance_inertia

        # Update VAD engine with mirror stimulus
        vad_engine.update(
            stimulus_valence=sv * cfg.contagion_rate,
            stimulus_arousal=sa * cfg.contagion_rate,
            stimulus_dominance=sd * cfg.contagion_rate,
            uncertainty=0.0,
            dt=1.0,
        )

        # Update mirror state
        self.state.mirror_engagement = min(1.0,
            self.state.mirror_engagement + 0.05 * ua)
        self.state.contagion_history.append((uv, ua, ud))
        if len(self.state.contagion_history) > 100:
            self.state.contagion_history = self.state.contagion_history[-100:]

        self.state.rapport_level = min(1.0,
            self.state.rapport_level + 0.01 * max(0.0, uv))

    def get_modulation(self, vad_state) -> Dict[str, float]:
        """Get response modulation parameters from current mirror state.

        Returns:
            temperature_mult: 0.5 (subdued) to 2.0 (excited)
            breadth_mult: 0.5 (focused) to 2.0 (exploratory)
            verbosity_mult: 0.5 (terse) to 2.0 (verbose)
        """
        ua = self.state.user_arousal
        uv = self.state.user_valence
        engagement = self.state.mirror_engagement

        if engagement < 0.05:
            return {
                'temperature_mult': 1.0,
                'breadth_mult': 1.0,
                'verbosity_mult': 1.0,
            }

        # Arousal drives temperature and breadth
        # High arousal → more varied language, broader exploration
        temperature_mult = 0.7 + ua * 1.0
        breadth_mult = 0.6 + ua * 1.2

        # Valence drives verbosity
        # Positive → more talkative, negative → quieter
        if uv >= 0:
            verbosity_mult = 0.8 + abs(uv) * 0.8
        else:
            verbosity_mult = 0.5 + (1.0 - abs(uv)) * 0.5

        # Engagement scales everything
        base = 0.5 + engagement * 0.5
        temperature_mult = max(0.5, min(2.0, temperature_mult * base))
        breadth_mult = max(0.5, min(2.5, breadth_mult * base))
        verbosity_mult = max(0.5, min(2.0, verbosity_mult * base))

        return {
            'temperature_mult': temperature_mult,
            'breadth_mult': breadth_mult,
            'verbosity_mult': verbosity_mult,
        }

    def get_emotional_label(self) -> str:
        """Label for the mirrored user emotional state."""
        v, a, d = self.state.user_valence, self.state.user_arousal, self.state.user_dominance
        if a > 0.6:
            if v > 0.3:
                return "excitement" if d > 0 else "joy"
            elif v < -0.3:
                return "anger" if d > 0 else "fear"
            else:
                return "surprise"
        elif a < 0.3:
            if v > 0.3:
                return "contentment"
            elif v < -0.3:
                return "sadness"
            else:
                return "calm"
        else:
            if v > 0.3:
                return "interest" if d > 0 else "hope"
            elif v < -0.3:
                return "frustration" if d > 0 else "anxiety"
            else:
                return "neutral"
