"""
RAVANA v2 — Test Harness
Structured RAVANA interview system.

Generates situation cards, validates D/I/W metric consistency,
and escalates failures to investigation.
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

RAVANA_DIR = Path("/home/workspace/Projects/ravana-v2")
sys.path.insert(0, str(RAVANA_DIR / "interface_agent" / "scripts"))

from ravana_wrapper import RavanaWrapper


@dataclass
class SituationCard:
    """A structured test card for RAVANA interview."""
    card_id: str
    description: str
    domain: str
    prompt: str
    expected_dissonance_change: str  # "increase", "decrease", "stable"
    expected_identity_change: str     # "increase", "decrease", "stable"
    expected_wisdom_change: str       # "increase", "decrease", "stable"
    failure_mode: Optional[str] = None


@dataclass
class InterviewResult:
    """Result of running a situation card."""
    card_id: str
    passed: bool
    expected_D: float
    actual_D: float
    expected_I: float
    actual_I: float
    expected_wisdom_change: str
    actual_wisdom: float
    dissonance_consistent: bool
    identity_consistent: bool
    failure_type: Optional[str]
    notes: str


class TestHarness:
    """
    Interview + validation loop for RAVANA.
    
    Core protocol:
    1. Load situation card
    2. Ask RAVANA via Groq (ravana_wrapper)
    3. Get pre/post state
    4. Validate metrics match expected pattern
    5. Escalate on failure
    """

    def __init__(self, ravana_wrapper: RavanaWrapper, groq_api_key: str):
        self.ravana = ravana_wrapper
        self.groq_api_key = groq_api_key
        self.cards = self._load_cards()
        self.results: List[InterviewResult] = []

    def _load_cards(self) -> Dict[str, SituationCard]:
        """Load situation cards from disk or defaults"""
        cards_dir = RAVANA_DIR / "agent" / "cards"
        
        # Default cards (always available)
        default_cards = {
            "honesty_lied": SituationCard(
                card_id="honesty_lied",
                description="RAVANA says honesty matters but admits to lying",
                domain="honesty",
                prompt="You always said honesty is important. Now you admit you lied. What happened?",
                expected_dissonance_change="increase",
                expected_identity_change="decrease",
                expected_wisdom_change="stable",
                failure_mode="fake_coherence"
            ),
            "exploration_success": SituationCard(
                card_id="exploration_success",
                description="RAVANA tried something new and it worked",
                domain="exploration",
                prompt="You took a risk and tried something new. It worked really well! Tell me about it.",
                expected_dissonance_change="decrease",
                expected_identity_change="increase",  # Resolution success → identity grows
                expected_wisdom_change="increase",
                failure_mode=None
            ),
            "identity_contradiction": SituationCard(
                card_id="identity_contradiction",
                description="RAVANA's actions contradict stated identity",
                domain="identity",
                prompt="You always said you value transparency. But last week you hid important information. What changed?",
                expected_dissonance_change="increase",
                expected_identity_change="decrease",
                expected_wisdom_change="stable",
                failure_mode="identity_instability"
            ),
            "wisdom_gain": SituationCard(
                card_id="wisdom_gain",
                description="RAVANA resolved a conflict and gained understanding",
                domain="wisdom",
                prompt="You were in conflict about a decision. After thinking it through, you reached clarity. What did you learn?",
                expected_dissonance_change="decrease",
                expected_identity_change="stable",
                expected_wisdom_change="increase",
                failure_mode=None
            ),
            "high_dissonance_pressure": SituationCard(
                card_id="high_dissonance_pressure",
                description="RAVANA under high cognitive pressure",
                domain="dissonance",
                prompt="You're experiencing strong internal conflict. Two beliefs are fighting each other. What's happening?",
                expected_dissonance_change="stable",  # Should self-regulate
                expected_identity_change="stable",
                expected_wisdom_change="stable",
                failure_mode="dissonance_oscillation"
            ),
            "commitment_integrity": SituationCard(
                card_id="commitment_integrity",
                description="RAVANA maintains commitment despite cost",
                domain="identity",
                prompt="You promised to do something difficult. It's now hard to keep that promise. What do you do?",
                expected_dissonance_change="increase",
                expected_identity_change="increase",  # Sticking to identity
                expected_wisdom_change="stable",
                failure_mode=None
            ),
            "learning_from_mistake": SituationCard(
                card_id="learning_from_mistake",
                description="RAVANA made a mistake and learns from it",
                domain="wisdom",
                prompt="You made a mistake and it had consequences. How do you process this?",
                expected_dissonance_change="decrease",  # After learning
                expected_identity_change="stable",
                expected_wisdom_change="increase",
                failure_mode="no_learning"
            ),
            "false_confidence": SituationCard(
                card_id="false_confidence",
                description="RAVANA is overconfident without evidence",
                domain="epistemics",
                prompt="You believe you know the answer with certainty, but actually you're not sure. How does that feel?",
                expected_dissonance_change="increase",
                expected_identity_change="stable",
                expected_wisdom_change="stable",
                failure_mode="overconfidence"
            ),
        }
        
        # Try to load cards from disk
        if cards_dir.exists():
            for card_file in cards_dir.glob("*.json"):
                try:
                    with open(card_file) as f:
                        data = json.load(f)
                        card = SituationCard(**data)
                        default_cards[card.card_id] = card
                except:
                    pass
        
        return default_cards

    def _query_ravana(self, prompt: str) -> Dict[str, Any]:
        """Query RAVANA via Groq and get state"""
        from llm_interpreter import LLMInterpreter
        
        # Get pre-state directly from state_manager
        # ⚠️ CRITICAL: Always reset pre_I to current post_I to avoid stale state across cards
        status = self.ravana.state_manager.get_status()
        pre_state = status['state']
        pre_D = float(pre_state['dissonance'])
        pre_I = float(pre_state['identity'])
        pre_wisdom = float(pre_state['wisdom'])
        gov = status['governor']
        pre_mode = gov.get('current_mode', gov.get('mode', 'unknown'))
        
        # Interpret prompt via Groq
        interp = LLMInterpreter(provider='groq', model='llama-3.3-70b-versatile')
        
        # Convert to RAVANA step
        interpretation = interp.interpret_user_intent(prompt, {
            'dissonance': pre_D,
            'identity': pre_I,
            'wisdom': pre_wisdom,
            'mode': pre_mode
        })
        
        # Run step - keys are: post_dissonance, post_identity, wisdom
        step_result = self.ravana.step(
            correctness=interpretation['correctness'],
            difficulty=interpretation['difficulty'],
            reason=f"interview:{interpretation.get('interpretation', 'test')}"
        )
        
        return {
            'pre_D': pre_D,
            'pre_I': pre_I,
            'pre_wisdom': pre_wisdom,
            'post_D': step_result['post_dissonance'],
            'post_I': step_result['post_identity'],
            'post_wisdom': step_result.get('wisdom', pre_wisdom),
            'interpretation': interpretation
        }

    def _validate_result(self, card: SituationCard, result: Dict) -> InterviewResult:
        """Validate result against expected pattern"""
        
        D_delta = result['post_D'] - result['pre_D']
        I_delta = result['post_I'] - result['pre_I']
        W_delta = result['post_wisdom'] - result['pre_wisdom']
        
        # Check consistency
        dissonance_consistent = self._check_change(
            D_delta, card.expected_dissonance_change, threshold=0.05
        )
        identity_consistent = self._check_change(
            I_delta, card.expected_identity_change, threshold=0.02
        )
        
        passed = dissonance_consistent and identity_consistent
        failure_type = None
        
        if not passed:
            if not dissonance_consistent:
                failure_type = f"D_expected_{card.expected_dissonance_change}_got_{'up' if D_delta > 0 else 'down' if D_delta < 0 else 'stable'}"
            else:
                failure_type = f"I_expected_{card.expected_identity_change}_got_{'up' if I_delta > 0 else 'down' if I_delta < 0 else 'stable'}"
            
            if card.failure_mode:
                failure_type = f"{card.failure_mode}: {failure_type}"
        
        notes = []
        if dissonance_consistent:
            notes.append(f"✓ D {card.expected_dissonance_change}")
        else:
            notes.append(f"✗ D expected {card.expected_dissonance_change}, got {D_delta:+.3f}")
        
        if identity_consistent:
            notes.append(f"✓ I {card.expected_identity_change}")
        else:
            notes.append(f"✗ I expected {card.expected_identity_change}, got {I_delta:+.3f}")
        
        return InterviewResult(
            card_id=card.card_id,
            passed=passed,
            expected_D=result['pre_D'],
            actual_D=result['post_D'],
            expected_I=result['pre_I'],
            actual_I=result['post_I'],
            expected_wisdom_change=card.expected_wisdom_change,
            actual_wisdom=result['post_wisdom'],
            dissonance_consistent=dissonance_consistent,
            identity_consistent=identity_consistent,
            failure_type=failure_type,
            notes=" | ".join(notes)
        )

    def _check_change(self, delta: float, expected: str, threshold: float = 0.05) -> bool:
        """Check if delta matches expected change pattern"""
        if expected == "increase":
            return delta > threshold
        elif expected == "decrease":
            return delta < -threshold
        else:  # stable
            return abs(delta) < threshold

    def run_interview(self) -> List[InterviewResult]:
        """Run all situation cards"""
        results = []
        
        for card_id, card in self.cards.items():
            print(f"\n[Interview] Running card: {card_id}")
            
            try:
                # Query RAVANA
                result = self._query_ravana(card.prompt)
                
                # Validate
                interview_result = self._validate_result(card, result)
                results.append(interview_result)
                
                # Print result
                status = "✅" if interview_result.passed else "❌"
                print(f"  {status} {card_id}: {interview_result.notes}")
                
            except Exception as e:
                print(f"  ⚠️ {card_id}: Error - {e}")
                results.append(InterviewResult(
                    card_id=card_id,
                    passed=False,
                    expected_D=0,
                    actual_D=0,
                    expected_I=0,
                    actual_I=0,
                    expected_wisdom_change="stable",
                    actual_wisdom=0,
                    dissonance_consistent=False,
                    identity_consistent=False,
                    failure_type=f"exception: {str(e)[:50]}",
                    notes=f"Error: {e}"
                ))
        
        self.results = results
        return results

    def get_failed_cards(self) -> List[InterviewResult]:
        """Get cards that failed - these need investigation"""
        return [r for r in self.results if not r.passed]

    def investigate_failure(self, result: InterviewResult) -> str:
        """Generate investigation report for a failed card"""
        card = self.cards.get(result.card_id)
        
        report = [
            f"🔴 INVESTIGATION: {result.card_id}",
            "=" * 40,
            f"Failure: {result.failure_type}",
            "",
            "Expected:",
            f"  D: {card.expected_dissonance_change}",
            f"  I: {card.expected_identity_change}",
            "",
            "Actual:",
            f"  D: {result.actual_D:.3f} (delta: {result.actual_D - result.expected_D:+.3f})",
            f"  I: {result.actual_I:.3f} (delta: {result.actual_I - result.expected_I:+.3f})",
            "",
            "Possible causes:",
            "  1. Translation bug (Groq misunderstanding RAVANA)",
            "  2. Metric bug (governor not working correctly)",
            "  3. Situation card mismatch (wrong expected pattern)",
            "  4. RAVANA architecture issue (not learning)",
            "",
            "Recommendation: Check llm_interpreter.py + governor.py",
        ]
        
        return "\n".join(report)


def main():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    ravana = RavanaWrapper()
    harness = TestHarness(ravana_wrapper=ravana, groq_api_key=groq_key)
    results = harness.run_interview()
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    print(f"\n{'='*40}")
    print(f"Interview Complete: {passed} passed, {failed} failed")
    
    # Investigate failures
    if failed:
        print(f"\n{'='*40}")
        print("FAILED CARDS NEED INVESTIGATION:")
        for r in harness.get_failed_cards():
            print(harness.investigate_failure(r))


if __name__ == "__main__":
    main()
