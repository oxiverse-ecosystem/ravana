"""
Proposition Parser — Nested Proposition Detection and Splitting
================================================================
Parses complex inputs with nested propositions into multiple independent
proposition triples for Global Workspace-based competition.

Neuroscience grounding:
- RLPFC performs relational integration — holding multiple relations simultaneously
- Global Workspace (Baars/Dehaene) provides limited-capacity buffer for competition

Design:
- Detects complex inputs containing multiple propositions
- Splits into independent (subject, predicate, object) triples
- Each proposition submits as a separate GW bid
- Dual-process routing when multiple propositions detected
"""
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class Proposition:
    """A single proposition triple extracted from input."""
    subject: str
    predicate: str
    object: str = ""
    confidence: float = 0.6
    raw_text: str = ""
    is_contrastive: bool = False  # "both right when disagree"
    is_relational: bool = False   # A > B relation


class PropositionParser:
    """Parses complex inputs into multiple proposition triples."""

    # Patterns that indicate multi-proposition inputs
    COMPOUND_MARKERS = [
        r"\bwhen\b", r"\bwhile\b", r"\balthough\b", r"\bthough\b",
        r"\bbut\b", r"\bhowever\b", r"\bwhereas\b", r"\bmeanwhile\b",
        r"\band\b", r"\bif\b", r"\bthen\b",
    ]

    # Relationship patterns for extracting triples
    RELATION_PATTERNS = [
        # "X is Y" / "X are Y"
        (r"(.+?)\s+is\s+(.+)", "is"),
        (r"(.+?)\s+are\s+(.+)", "are"),
        # "X has Y" / "X have Y"
        (r"(.+?)\s+has\s+(.+)", "has"),
        (r"(.+?)\s+have\s+(.+)", "have"),
        # "X means Y"
        (r"(.+?)\s+means\s+(.+)", "means"),
        # "X does Y"
        (r"(.+?)\s+does\s+(.+)", "does"),
        # "X can Y"
        (r"(.+?)\s+can\s+(.+)", "can"),
        # "X will Y"
        (r"(.+?)\s+will\s+(.+)", "will"),
        # "X when Y" (conditional)
        (r"(.+?)\s+when\s+(.+)", "when"),
        # "X if Y"
        (r"(.+?)\s+if\s+(.+)", "if"),
        # "X and Y are Z" → two propositions
        (r"(.+?)\s+and\s+(.+?)\s+(?:is|are)\s+(.+)", "compound_and"),
    ]

    def __init__(self):
        self._compiled_markers = [re.compile(m, re.IGNORECASE) for m in self.COMPOUND_MARKERS]

    def detect_multi_proposition(self, text: str) -> bool:
        """Detect if input contains multiple propositions."""
        text_lower = text.lower()
        marker_count = sum(1 for m in self._compiled_markers if m.search(text_lower))
        # 2+ markers or specific complex patterns → multi-proposition
        if marker_count >= 2:
            return True
        # Check for specific complex patterns
        complex_patterns = [
            r"both\s+.+\s+(?:right|correct|wrong|true|false)",
            r"(?:two|both)\s+(?:people|person|sides?|views?|perspectives?)",
            r"when\s+.+\s+(?:and|but)\s+.+",
            r"if\s+.+\s+then\s+.+",
        ]
        for pat in complex_patterns:
            if re.search(pat, text_lower):
                return True
        return False

    def extract_propositions(self, text: str) -> List[Proposition]:
        """Split complex input into multiple propositions."""
        propositions = []
        seen = set()
        text_lower = text.lower().strip(" ?!.")

        # 1. Try to find coordination patterns and split
        # "X and Y are Z" → "X is Z", "Y is Z"
        for pattern, pred_type in self.RELATION_PATTERNS:
            m = re.match(pattern, text_lower)
            if not m:
                continue

            if pred_type == "compound_and":
                subj_a = m.group(1).strip()
                subj_b = m.group(2).strip()
                obj = m.group(3).strip()
                for s in [subj_a, subj_b]:
                    if s not in seen:
                        propositions.append(Proposition(
                            subject=s,
                            predicate="is",
                            object=obj,
                            confidence=0.7,
                            raw_text=text,
                        ))
                        seen.add(s)
                if propositions:
                    return propositions

            elif pred_type in ("when", "if"):
                # "X when Y" → "X happens", "Y is condition for X"
                clause_a = m.group(1).strip()
                clause_b = m.group(2).strip()
                if clause_a not in seen:
                    propositions.append(Proposition(
                        subject=clause_a,
                        predicate=pred_type,
                        object=clause_b,
                        confidence=0.6,
                        raw_text=text,
                    ))
                    seen.add(clause_a)
            else:
                # Simple relation
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                if subj not in seen:
                    propositions.append(Proposition(
                        subject=subj,
                        predicate=pred_type,
                        object=obj,
                        confidence=0.7,
                        raw_text=text,
                    ))
                    seen.add(subj)

        # 2. Detect contrastive patterns: "both X when Y"
        contrastive_match = re.search(
            r"both\s+(.+?)\s+(?:right|correct|true)\s+when\s+(.+)", text_lower
        )
        if contrastive_match:
            actors = contrastive_match.group(1).strip()
            condition = contrastive_match.group(2).strip()
            # "two people both right when disagree" → 
            #   proposition: "two people disagree" (state)
            #   proposition: "both are right" (judgment)
            if "agree" not in condition:
                propositions.append(Proposition(
                    subject=actors,
                    predicate="disagree",
                    object="",
                    confidence=0.7,
                    raw_text=text,
                    is_contrastive=True,
                ))
            propositions.append(Proposition(
                subject=actors,
                predicate="are",
                object="correct",
                confidence=0.65,
                raw_text=text,
                is_contrastive=True,
            ))
            for p in propositions:
                seen.add(p.subject)

        # 3. If no patterns matched, check for implicit multi-proposition
        if not propositions:
            # Look for key markers that indicate complex thought
            if "both" in text_lower and "when" in text_lower:
                # Generic "both X when Y" — split on "when"
                parts = re.split(r"\bwhen\b", text_lower, maxsplit=1)
                if len(parts) == 2:
                    pre_when = parts[0].strip()
                    post_when = parts[1].strip()
                    # Extract the both-clause
                    both_match = re.search(r"both\s+(.+?)$", pre_when)
                    if both_match:
                        actors = both_match.group(1)
                        propositions.append(Proposition(
                            subject=actors,
                            predicate="related_to",
                            object=post_when,
                            confidence=0.5,
                            raw_text=text,
                            is_relational=True,
                        ))

        return propositions

    def has_nested_propositions(self, text: str) -> bool:
        """Quick check if input needs multi-proposition handling."""
        if self.detect_multi_proposition(text):
            return True
        text_lower = text.lower()
        nested_indicators = [
            r"(?:both|two|several)\s+.+\s+(?:right|wrong|agree|disagree)",
            r"if\s+.+\s+then\s+.+",
            r"when\s+.+\s+but\s+.+",
            r"what\s+.+\s+means?\s+when\s+",
            r"(?:while|although)\s+.+\s*,",
        ]
        for pat in nested_indicators:
            if re.search(pat, text_lower):
                return True
        return False

    def get_surface_structure(self, propositions: List[Proposition]) -> str:
        """Determine the surface structure for response framing.

        Returns: "simple", "contrastive", "conditional", "multi_perspective"
        """
        if len(propositions) <= 1:
            return "simple"

        has_contrastive = any(p.is_contrastive for p in propositions)
        has_conditional = any(p.predicate in ("when", "if") for p in propositions)

        if has_contrastive:
            return "contrastive"
        elif has_conditional:
            return "conditional"
        else:
            return "multi_perspective"
