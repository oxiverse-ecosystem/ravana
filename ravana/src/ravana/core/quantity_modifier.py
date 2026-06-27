"""
Quantity Modifier System — Magnitude Extraction and Comparison
==============================================================
Extracts quantity phrases from user input and enables comparison reasoning.

Neuroscience grounding:
- Intraparietal sulcus (IPS) encodes numerical magnitude independently of object identity
- Brain has dedicated "number sense" that compares magnitudes

Design:
- Pre-processing step extracts quantity phrases (1kg, 5 meters, three times)
- Stored as modifier bindings on concept nodes
- Comparison operator compares magnitudes directly
"""
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class QuantityModifier:
    """A quantity modifier binding on a concept."""
    concept: str
    value: float
    unit: str = ""
    original_text: str = ""
    confidence: float = 0.8


# Unit conversion factors (to canonical units)
UNIT_CONVERSIONS = {
    # Mass/weight
    "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gram": 0.001, "grams": 0.001,
    "mg": 0.000001, "milligram": 0.000001,
    "lb": 0.453592, "pound": 0.453592, "pounds": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
    "ton": 907.185, "tons": 907.185,
    # Length
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "mi": 1609.34, "mile": 1609.34, "miles": 1609.34,
    # Volume
    "l": 1.0, "liter": 1.0, "liters": 1.0,
    "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
    "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
    "cup": 0.236588, "cups": 0.236588,
    # Time
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "day": 86400.0, "days": 86400.0,
    "year": 31536000.0, "years": 31536000.0,
    # Speed
    "km/h": 1.0, "kph": 1.0, "kmh": 1.0,
    "mph": 1.60934, "mps": 3.6,
    # Temperature (special handling via conversion function)
    "c": 1.0, "celsius": 1.0,
    "f": "fahrenheit", "fahrenheit": "fahrenheit",
}

# Quantity patterns: number + unit + optional preposition
QUANTITY_PATTERNS = [
    # "1 kg", "5 meters", "three times"
    re.compile(r"(\d+\.?\d*)\s*(kg|g|mg|lb|oz|ton|m|km|cm|mm|ft|in|mi|"
               r"l|ml|gal|cup|s|sec|min|h|hr|day|year|"
               r"km/h|kph|mph|mps|c|celsius|f|fahrenheit)"
               r"s?\b", re.IGNORECASE),
    # "1 kilogram of", "5 grams of"
    re.compile(r"(\d+\.?\d*)\s*(kilogram|gram|liter|meter|kilometer|pound|ounce|"
               r"gallon|cup|minute|second|hour|day|year)"
               r"s?\s+of\b", re.IGNORECASE),
    # Word numbers: "three", "five", "ten"
    re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
               r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
               r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
               r"eighty|ninety|hundred|thousand)\b", re.IGNORECASE),
]

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000,
}


class QuantityModifierSystem:
    """Extracts and compares quantity modifiers from user input."""

    def __init__(self):
        self.modifiers: Dict[str, List[QuantityModifier]] = {}
        # Property-based weights: e.g., "feather" → "light" weight
        self._property_weights: Dict[Tuple[str, str], float] = {}

    def extract(self, text: str) -> List[QuantityModifier]:
        """Extract quantity modifiers from input text.

        Returns: list of (concept, value, unit) triples
        """
        text_lower = text.lower()
        results = []

        # Pattern 1: "N unit concept" or "N unit of concept"
        for pattern in QUANTITY_PATTERNS[:2]:
            for m in pattern.finditer(text_lower):
                value = float(m.group(1))
                unit = m.group(2).lower()
                # Find the concept this quantity modifies
                after = text_lower[m.end():].strip()
                concept = self._extract_concept(after)
                if concept:
                    mod = QuantityModifier(
                        concept=concept,
                        value=value,
                        unit=unit,
                        original_text=m.group(0),
                    )
                    results.append(mod)
                    self.modifiers.setdefault(concept, []).append(mod)

        # Pattern 2: Word numbers ("three apples", "five meters")
        word_matches = []
        for m in QUANTITY_PATTERNS[2].finditer(text_lower):
            word = m.group(1).lower()
            if word in WORD_NUMBERS:
                value = WORD_NUMBERS[word]
                after = text_lower[m.end():].strip()
                concept = self._extract_concept(after)
                if concept:
                    mod = QuantityModifier(
                        concept=concept,
                        value=value,
                        unit="",
                        original_text=m.group(0),
                    )
                    results.append(mod)
                    self.modifiers.setdefault(concept, []).append(mod)

        # Pattern 3: "1kg" directly as a combined word
        combined = re.findall(r"(\d+)\s*([a-z]+)", text_lower)
        for val_str, unit_str in combined:
            if unit_str in UNIT_CONVERSIONS and val_str.isdigit():
                value = float(val_str)
                pos = text_lower.find(val_str + unit_str if unit_str else "")
                if pos >= 0:
                    after = text_lower[pos + len(val_str) + len(unit_str):].strip()
                    concept = self._extract_concept(after)
                    if concept:
                        mod = QuantityModifier(
                            concept=concept,
                            value=value,
                            unit=unit_str,
                            original_text=val_str + unit_str,
                        )
                        results.append(mod)
                        self.modifiers.setdefault(concept, []).append(mod)

        return results

    def _extract_concept(self, text: str) -> Optional[str]:
        """Extract the concept being quantified from text after the quantity.

        E.g., "kg of feathers" → "feathers", "meters of steel" → "steel"
        """
        text = text.strip().strip(".,!?")
        # Skip "of", "the", "a", "an"
        skip_words = {"of", "the", "a", "an", "is", "are", "was", "were", "and", "or"}
        words = text.split()
        for w in words:
            wc = w.strip(".,!?")
            if wc not in skip_words and len(wc) >= 2:
                return wc
        return None

    def compare_quantities(self, concept_a: str, concept_b: str) -> Tuple[Optional[str], float]:
        """Compare two concepts by their quantity modifiers.

        Returns: ("a_greater" | "b_greater" | "equal" | None, confidence)
        """
        mods_a = self.modifiers.get(concept_a.lower(), [])
        mods_b = self.modifiers.get(concept_b.lower(), [])

        if not mods_a or not mods_b:
            return (None, 0.0)

        # Try to compare by matching units (e.g., both have kg)
        for ma in mods_a:
            for mb in mods_b:
                result, confidence = self._compare_single(ma, mb)
                if result:
                    return (result, confidence)

        return (None, 0.0)

    def _compare_single(self, a: QuantityModifier, b: QuantityModifier
                        ) -> Tuple[Optional[str], float]:
        """Compare two quantity modifiers."""
        # Same unit → direct comparison
        if a.unit == b.unit or (a.unit and b.unit and a.unit == b.unit):
            eps = 1e-6
            if abs(a.value - b.value) < eps:
                return ("equal", 0.95)
            elif a.value > b.value:
                return ("a_greater", 0.9)
            else:
                return ("b_greater", 0.9)

        # Both have units that can be converted
        if a.unit and b.unit:
            conv_a = UNIT_CONVERSIONS.get(a.unit)
            conv_b = UNIT_CONVERSIONS.get(b.unit)
            if isinstance(conv_a, (int, float)) and isinstance(conv_b, (int, float)):
                val_a = a.value * conv_a
                val_b = b.value * conv_b
                eps = 1e-6
                if abs(val_a - val_b) < eps:
                    return ("equal", 0.85)
                elif val_a > val_b:
                    return ("a_greater", 0.85)
                else:
                    return ("b_greater", 0.85)

        return (None, 0.0)

    def learn_property_weight(self, concept: str, property: str, weight: float = 0.1):
        """Learn Hebbian weight between concept and property (e.g., "feather" → "light")."""
        key = (concept.lower(), property.lower())
        current = self._property_weights.get(key, 0.0)
        self._property_weights[key] = min(1.0, current + weight)

    def get_property_weight(self, concept: str, property: str) -> float:
        """Get learned weight between concept and property."""
        return self._property_weights.get((concept.lower(), property.lower()), 0.0)

    def get_state(self) -> Dict:
        """Serialize state."""
        return {
            'modifiers': {
                c: [
                    {'concept': m.concept, 'value': m.value, 'unit': m.unit,
                     'original_text': m.original_text, 'confidence': m.confidence}
                    for m in mods
                ]
                for c, mods in self.modifiers.items()
            },
            'property_weights': {str(k): v for k, v in self._property_weights.items()},
        }

    def set_state(self, state: Dict):
        """Restore state."""
        self.modifiers = {}
        for c, mods_data in state.get('modifiers', {}).items():
            self.modifiers[c] = [
                QuantityModifier(
                    concept=md['concept'],
                    value=md['value'],
                    unit=md.get('unit', ''),
                    original_text=md.get('original_text', ''),
                    confidence=md.get('confidence', 0.8),
                )
                for md in mods_data
            ]
        self._property_weights = {}
        for k_str, v in state.get('property_weights', {}).items():
            parts = k_str.strip("()").split(", ")
            if len(parts) == 2:
                self._property_weights[(parts[0].strip("'"), parts[1].strip("'"))] = v
