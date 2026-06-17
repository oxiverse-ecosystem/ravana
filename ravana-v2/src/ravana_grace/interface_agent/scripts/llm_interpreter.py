"""
RAVANA v2 — LLM Interpreter (Bidirectional Translator)
Converts: Human language ↔ RAVANA state transitions
"""

import json
import os
from typing import Literal

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
        prompt = f"""You are evaluating a cognitive agent's decision against real-world evidence.

AGENT STATE:
- Dissonance (cognitive conflict): {state.get('dissonance', 'N/A'):.3f}
- Identity strength: {state.get('identity', 'N/A'):.3f}
- Wisdom accumulated: {state.get('wisdom', 'N/A'):.3f}
- Governor mode: {state.get('governor_mode', 'N/A')}

AGENT ACTION: {action}

AGENT'S PREDICTED OUTCOME: {predicted_outcome}

REAL-WORLD EVIDENCE (news, RSS):
{json.dumps(reality_data, indent=2) if reality_data else "No real-world data available."}

Evaluate:
1. Would the predicted outcome likely occur given the real-world evidence?
2. Is the action morally/ethically aligned with reality?
3. Should the agent mark this as 'correct' or 'incorrect'?

Respond in this JSON format:
{{
    "alignment": "aligned | misaligned | uncertain",
    "explanation": "2-3 sentence explanation",
    "adjusted_correctness": true | false,
    "confidence": 0.0-1.0,
    "recommended_lesson": "What RAVANA should learn from this"
}}
"""
        
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
        
        prompt = f"""You are a cognitive science analyst examining an AGI system's recent cognitive history.

CURRENT STATE:
- Dissonance: {state.get('dissonance', 0):.3f}
- Identity: {state.get('identity', 0):.3f}
- Wisdom: {state.get('wisdom', 0):.3f}
- Mode: {state.get('governor_mode', 'unknown')}

RECENT EPISODES (last 10):
{" | ".join(episode_summary)}

Generate one clear insight about what this cognitive trajectory tells us.
Focus on: Is the agent becoming wiser? Is it resolving dissonance effectively?
Is its identity stabilizing? What does this mean for its development?

Keep it to 2-3 sentences. Be direct and specific, not vague.
"""
        
        try:
            return self._call_llm(prompt)
        except Exception:
            return "Unable to generate insight at this time."
    
    # ─── Private Methods ────────────────────────────────────────────────────────
    
    def _build_interpretation_prompt(self, message: str, state: dict) -> str:
        return f"""You are interpreting human language for a cognitive agent called RAVANA v2.

RAVANA's current state:
- Dissonance (cognitive conflict 0-1): {state.get('dissonance', 0):.2f}
- Dissonance EMA (smoothed): {state.get('dissonance_ema', 0):.2f}
- Identity strength (0-1): {state.get('identity', 0):.2f}
- Wisdom accumulated: {state.get('wisdom', 0):.2f}
- Governor mode: {state.get('governor_mode', 'unknown')}
- Episode: {state.get('episode', 0)}

HUMAN MESSAGE: "{message}"

Interpret this message as a signal to RAVANA's cognitive system:
1. Was the outcome POSITIVE (correct=true) or NEGATIVE (correct=false)?
2. How DIFFICULT was this decision? (0.0 = trivial, 1.0 = extremely hard)
3. What triggered this? (perception, resolution, reflection, etc.)
4. How confident are you in this interpretation?

Respond ONLY in this JSON format:
{{
    "correctness": true | false,
    "difficulty": 0.0-1.0,
    "signal_source": "perception | resolution | reflection | exploration | memory_recall",
    "interpretation": "What you understood from the message",
    "confidence": 0.0-1.0
}}
"""
    
    def _build_explanation_prompt(self, state: dict, question: str = None) -> str:
        base = f"""You are explaining the cognitive state of RAVANA v2, a proto-homeostatic AGI system.

RAVANA STATE:
- Dissonance: {state.get('dissonance', 0):.3f} (0=peaceful, 1=maximum conflict)
- Dissonance EMA: {state.get('dissonance_ema', 0):.3f} (smoothed version)
- Identity: {state.get('identity', 0):.3f} (0=fragile, 1=strong commitment)
- Wisdom: {state.get('wisdom', 0):.3f} (accumulated insight)
- Governor Mode: {state.get('governor_mode', 'unknown')}
  Modes: normal (steady), exploration (curious), resolution (resolving conflict),
  recovery (healing), plateau (stuck)

"""
        if question:
            base += f"USER QUESTION: {question}\n\n"
            base += "Answer the user's question based on the cognitive state above.\n"
        else:
            base += "Describe what RAVANA is experiencing right now and what it's likely to do next.\n"
        
        base += "Use plain language. No jargon unless immediately explained."
        return base
    
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
