"""
Implicature Detector — Pragmatic Inference and Theory-of-Mind
==============================================================
Detects when user input implies a question or hidden meaning (pragmatic
implicature) rather than literal semantic content.

Neuroscience grounding:
- mPFC and TPJ process pragmatic inference
- Implicit meaning processed as social-cognitive task: "What does the
  speaker intend me to understand?"

Design:
- Detects statements that carry pragmatic implicature
- Routes through belief store / social epistemology for response
- Signals contradiction when pragmatic mismatch is detected
"""
import re
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class ImplicatureResult:
    """Result of implicature analysis."""
    is_implicature: bool = False
    implied_question: str = ""
    detected_type: str = ""  # "personal_state", "obvious_statement", "social_cue", "information_request"
    urgency: float = 0.0
    suggested_question_type: str = "acknowledge"
    confidence: float = 0.5


class ImplicatureDetector:
    """Detects pragmatic implicature in user statements."""

    # Personal state patterns: "I [verb] [noun]" → implies "what should I do?"
    PERSONAL_STATE_PATTERNS = [
        r"\bi\s+(ate|drank|had|finished|completed|did|made|bought|got|found|saw|watched|read|wrote)\s+(.+)",
        r"\bi\s+(went|came|left|arrived|moved|traveled|visited)\s+(.+)",
        r"\bi\s+(started|began|stopped|quit|continued|kept)\s+(.+)",
        r"\bi\s+just\s+(ate|drank|had|finished|did|made|bought|got|found|saw|watched|read|wrote)\s+(.+)",
    ]

    # Obvious statement patterns: states the obvious, often prompting "no duh"
    OBVIOUS_STATEMENT_PATTERNS = [
        r"\bwater\s+is\s+wet\b",
        r"\b(sky|sun)\s+is\s+(blue|hot|bright)\b",
        r"\b(fire|ice)\s+is\s+(hot|cold)\b",
        r"\b(earth|world)\s+is\s+(round|big)\b",
        r"\bpeople\s+(need|require|must)\s+(water|food|air|sleep)\b",
    ]

    # Social cue patterns: information-sharing without explicit question
    SOCIAL_CUE_PATTERNS = [
        r"\byou\s+(know|remember|recall)\s+(.+?)\?$",
        r"\bi\s+(think|feel|believe|suspect|wonder)\s+(.+?)$",
        r"\bi?t\s+(seems|appears|looks|sounds)\s+(like|as if)\s+(.+?)$",
        r"\b(did you|have you|do you)\s+(.+?)$",
    ]

    # Information request: user is asking for an opinion/response implicitly
    INFO_REQUEST_PATTERNS = [
        r"\b(what|how)\s+(?:about|of|with)\s+(.+)",
        r"\b(what|how)\s+(?:do|does|did)\s+(.+?)\s+(?:think|feel|mean)\s+",
        r"\bcan\s+you\s+(explain|tell|describe|elaborate)\s+(.+)",
    ]

    def __init__(self):
        self._personal_compiled = [re.compile(p, re.IGNORECASE) for p in self.PERSONAL_STATE_PATTERNS]
        self._obvious_compiled = [re.compile(p, re.IGNORECASE) for p in self.OBVIOUS_STATEMENT_PATTERNS]
        self._social_compiled = [re.compile(p, re.IGNORECASE) for p in self.SOCIAL_CUE_PATTERNS]
        self._info_compiled = [re.compile(p, re.IGNORECASE) for p in self.INFO_REQUEST_PATTERNS]

    def analyze(self, text: str) -> ImplicatureResult:
        """Analyze user input for pragmatic implicature.

        Returns ImplicatureResult with detection findings.
        """
        text_lower = text.lower().strip()
        result = ImplicatureResult()

        # 1. Check for personal state statements
        for pattern in self._personal_compiled:
            m = pattern.match(text_lower)
            if m:
                verb = m.group(1)
                obj = m.group(2).strip()
                result.is_implicature = True
                result.detected_type = "personal_state"
                result.confidence = 0.7
                result.urgency = 0.5

                # Generate implied question based on context
                if verb in ("ate", "drank", "had", "finished"):
                    result.implied_question = f"tell me about your experience with {obj}"
                else:
                    result.implied_question = f"what do you think about {obj}"

                result.suggested_question_type = "acknowledge"
                return result

        # 2. Check for obvious statements
        for pattern in self._obvious_compiled:
            if pattern.search(text_lower):
                result.is_implicature = True
                result.detected_type = "obvious_statement"
                result.confidence = 0.8
                result.urgency = 0.3
                # Signal free energy spike for contradiction signaling
                result.suggested_question_type = "acknowledge"
                return result

        # 3. Check for social cues
        for pattern in self._social_compiled:
            m = pattern.search(text_lower)
            if m:
                result.is_implicature = True
                result.detected_type = "social_cue"
                result.confidence = 0.65
                result.urgency = 0.4
                result.suggested_question_type = "acknowledge"
                return result

        # 4. Check for implicit information requests
        for pattern in self._info_compiled:
            m = pattern.match(text_lower)
            if m:
                result.is_implicature = True
                result.detected_type = "information_request"
                result.confidence = 0.6
                result.urgency = 0.6
                result.suggested_question_type = "general"
                return result

        # 5. Check if input is a statement (not a question) that implies a follow-up
        is_question = any(text_lower.strip().endswith(q) for q in ['?', '.?', '!?'])
        if not is_question and len(text_lower.split()) >= 3:
            # Statement about a personal state → likely implicature
            personal_pronouns = {"i", "my", "me", "we", "our", "us"}
            words = set(text_lower.split())
            if words & personal_pronouns:
                result.is_implicature = True
                result.detected_type = "personal_state"
                result.confidence = 0.5
                result.urgency = 0.4
                result.suggested_question_type = "acknowledge"

        return result

    def should_acknowledge(self, text: str) -> bool:
        """Check if input is a statement that should be acknowledged rather than explained."""
        result = self.analyze(text)
        return result.is_implicature and result.suggested_question_type == "acknowledge"

    def get_free_energy_spike(self, text: str) -> float:
        """Get predicted free energy spike for contradiction signaling.

        When an obvious statement is detected, spike free energy to signal
        confusion and prompt clarifying question.
        """
        result = self.analyze(text)
        if result.detected_type == "obvious_statement":
            return 0.6  # High free energy spike
        elif result.detected_type == "personal_state":
            return 0.3  # Moderate uncertainty
        return 0.0
