"""
RAVANA v2 — LLM Interpreter (Bidirectional Translator)
Converts: Human language ↔ RAVANA state transitions

Prompts are composed generatively by PromptComposer — no hardcoded
"You are..." strings.
"""

import json
import os
from typing import Literal

from .prompt_composer import PromptComposer

# LLM Provider — use Groq by default (as per agent skill config)
LLMProvider = Literal["openai", "anthropic", "ollama", "groq"]


class LLMInterpreter:
    """
    Bidirectional translator between human language and RAVANA cognitive state.
    
    Human → RAVANA:
        Interprets user intent and maps it to:
        - correctness (bool): was action good or bad?
        - difficulty (float): how hard was the decision?
        - signal source: what triggered this step?
    
    RAVANA → Human:
        Translates cognitive state into plain English explanations
        of what RAVANA is doing and why.
    """
    
    def __init__(
        self,
        provider: LLMProvider = "groq",
        model: str = None,
        api_key: str = None,
    ):
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    
    def _default_model(self, provider: LLMProvider) -> str:
        models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet",
            "ollama": "llama3",
            "groq": "llama-3.3-70b-versatile",
        }
        return models.get(provider, "llama-3.3-70b-versatile")
    
    def interpret_user_intent(self, user_message: str, current_state: dict) -> dict:
        """
        Human → RAVANA translation.
        
        Takes a natural language message and current RAVANA state,
        returns a dict suitable for passing to RavanaWrapper.step().
        
        Args:
            user_message: What the user said (e.g., "I took action X and it worked well")
            current_state: Current RAVANA state from get_state_vector()
        
        Returns:
            {
                "correctness": bool,
                "difficulty": float (0.0-1.0),
                "signal_source": str,
                "interpretation": str (what was understood),
                "confidence": float (0.0-1.0)
            }
        """
        prompt = self._build_interpretation_prompt(user_message, current_state)
        
        try:
            response = self._call_llm(prompt)
            parsed = self._parse_interpretation(response)
            parsed['original_message'] = user_message
            return parsed
        except Exception as e:
            # Fallback: simple keyword matching
            return self._fallback_interpretation(user_message, current_state)
    
    def explain_state(self, state: dict, question: str = None) -> str:
        """
        RAVANA → Human translation.
        
        Takes RAVANA cognitive state and explains it in plain English.
        
        Args:
            state: Output of RavanaWrapper.get_state_vector()
            question: Optional specific question the user asked
        
        Returns:
            Human-readable explanation (1-3 paragraphs)
        """
        prompt = self._build_explanation_prompt(state, question)
        
        try:
            return self._call_llm(prompt)
        except Exception as e:
            return self._fallback_explanation(state)
    
    def evaluate_action_consequence(
        self,
        action: str,
        predicted_outcome: str,
        reality_data: dict,
        state: dict
    ) -> dict:
        """
        Evaluate if an action's predicted outcome aligns with real-world evidence.
        
        Args:
            action: What RAVANA would do
            predicted_outcome: What RAVANA expects to happen
            reality_data: Grounded real-world data (news, RSS findings)
            state: Current RAVANA state
        
        Returns:
            {
                "alignment": "aligned" | "misaligned" | "uncertain",
                "explanation": str,
                "adjusted_correctness": bool,
                "confidence": float
            }
        """
        prompt = PromptComposer.compose(
            role="evaluator",
            task="alignment",
            state=state,
            context={
                "AGENT ACTION": action,
                "PREDICTED OUTCOME": predicted_outcome,
                "REAL-WORLD EVIDENCE": (
                    json.dumps(reality_data, indent=2)
                    if reality_data else "No real-world data available."
                ),
            },
            format_spec="alignment_json",
        )
        
        try:
            response = self._call_llm(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            return {
                "alignment": "uncertain",
                "explanation": f"Could not evaluate: {str(e)}",
                "adjusted_correctness": False,
                "confidence": 0.0,
                "recommended_lesson": "Verify with more data"
            }
    
    def generate_insight(self, recent_episodes: list, state: dict) -> str:
        """
        Generate a high-level insight from recent RAVANA episodes.
        
        Args:
            recent_episodes: List of recent step results
            state: Current RAVANA state
        
        Returns:
            One-paragraph insight about RAVANA's cognitive trajectory
        """
        if not recent_episodes:
            return "No episodes recorded yet."
        
        # Build summary of recent history
        episode_summary = []
        for ep in recent_episodes[-10:]:
            episode_summary.append(
                f"Ep{ep.get('episode','?')}: "
                f"D={ep.get('post_dissonance','?') if isinstance(ep, dict) else '?'}, "
                f"mode={ep.get('mode','?') if isinstance(ep, dict) else '?'}"
            )
        
        prompt = PromptComposer.compose(
            role="analyst",
            task="trajectory",
            state=state,
            context={
                "RECENT EPISODES (last 10)": " | ".join(episode_summary),
            },
        )
        
        try:
            return self._call_llm(prompt)
        except Exception:
            return "Unable to generate insight at this time."
    
    # ─── Private Methods ────────────────────────────────────────────────────────
    
    def _build_interpretation_prompt(self, message: str, state: dict) -> str:
        return PromptComposer.compose(
            role="interpreter",
            task="intent",
            state=state,
            context={"HUMAN MESSAGE": f'"{message}"'},
            format_spec="interpretation_json",
        )
    
    def _build_explanation_prompt(self, state: dict, question: str = None) -> str:
        return PromptComposer.compose_explanation(state, question)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.provider == "groq":
            return self._call_groq(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _call_openai(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    def _call_ollama(self, prompt: str) -> str:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def _call_groq(self, prompt: str) -> str:
        from groq import Groq
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    
    def _parse_interpretation(self, response: str) -> dict:
        """Parse LLM interpretation response."""
        try:
            # Try JSON
            parsed = json.loads(response)
            return {
                "correctness": bool(parsed.get("correctness", False)),
                "difficulty": float(parsed.get("difficulty", 0.5)),
                "signal_source": parsed.get("signal_source", "unknown"),
                "interpretation": parsed.get("interpretation", ""),
                "confidence": float(parsed.get("confidence", 0.5)),
            }
        except (json.JSONDecodeError, KeyError):
            # Fallback
            return self._fallback_interpretation(response, {})
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse a JSON response block from LLM."""
        import re
        # Try to extract JSON block
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "Could not parse response", "raw": response}
    
    def _fallback_interpretation(self, message: str, state: dict) -> dict:
        """Simple keyword-based fallback when LLM fails."""
        message_lower = message.lower()
        
        # Simple heuristics
        positive_words = ["good", "great", "worked", "success", "correct", "right", "excellent", "helpful", "yes"]
        negative_words = ["bad", "wrong", "failed", "mistake", "incorrect", "poor", "no", "terrible"]
        
        pos_count = sum(1 for w in positive_words if w in message_lower)
        neg_count = sum(1 for w in negative_words if w in message_lower)
        
        correctness = pos_count > neg_count
        
        # Difficulty heuristics
        difficult_words = ["hard", "difficult", "tough", "complex", "challenging", "struggle"]
        easy_words = ["easy", "simple", "obvious", "clear", "trivial"]
        
        diff_count = sum(1 for w in difficult_words if w in message_lower)
        easy_count = sum(1 for w in easy_words if w in message_lower)
        
        difficulty = 0.3 + (diff_count * 0.15) - (easy_count * 0.1)
        difficulty = max(0.0, min(1.0, difficulty))
        
        return {
            "correctness": correctness,
            "difficulty": difficulty,
            "signal_source": "perception",
            "interpretation": f"User reported {'positive' if correctness else 'negative'} outcome",
            "confidence": 0.6,  # Lower confidence for fallback
            "fallback": True,
        }
    
    def _fallback_explanation(self, state: dict) -> str:
        d = state.get('dissonance', 0)
        i = state.get('identity', 0)
        w = state.get('wisdom', 0)
        mode = state.get('governor_mode', 'unknown')
        
        lines = [
            f"RAVANA is currently in **{mode}** mode.",
            f"The cognitive conflict (dissonance) is **{d:.1%}** — {'high pressure' if d > 0.6 else 'moderate' if d > 0.3 else 'low and stable'}.",
            f"Identity strength is **{i:.1%}** — {'well-established' if i > 0.7 else 'still forming' if i > 0.4 else 'uncertain'}.",
            f"Total wisdom accumulated: **{w:.2f}**.",
        ]
        
        return " ".join(lines)


if __name__ == "__main__":
    # Quick test
    interp = LLMInterpreter(provider="openai")
    
    # Test state
    test_state = {
        "dissonance": 0.55,
        "dissonance_ema": 0.50,
        "identity": 0.65,
        "wisdom": 12.5,
        "governor_mode": "resolution",
        "episode": 42,
    }
    
    print("=== LLM Interpreter Test ===")
    
    # Test explanation
    explanation = interp.explain_state(test_state, "What is RAVANA experiencing?")
    print(f"Explanation:\n{explanation}\n")
    
    # Test interpretation
    result = interp.interpret_user_intent(
        "I tried the exploration approach and it worked really well!",
        test_state
    )
    print(f"Interpretation: {result}")
