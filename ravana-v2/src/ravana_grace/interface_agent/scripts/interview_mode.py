"""
RAVANA v2 — INTERVIEW Mode
Conducts structured self-assessment dialog with RAVANA v2.
Uses GEPA-style reflective self-examination to probe cognitive state.
"""

import json
import os
import sys
from datetime import datetime
from typing import Literal

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

from groq import Groq
from ravana_wrapper import RavanaWrapper

LLM_MODEL = "llama-3.3-70b-versatile"


class InterviewMode:
    """
    Structured self-assessment dialog with RAVANA v2.
    
    Uses a GEPA-inspired reflective loop:
    1. Probe cognitive state with targeted questions
    2. Evaluate responses via LLM interpretation
    3. Track trajectory across multiple sessions
    4. Generate insights and improvement recommendations
    """

    def __init__(self, provider: str = "groq", api_key: str = None):
        self.provider = provider
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.ravana = RavanaWrapper()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.responses = []
        self.insights = []

    # ─── LLM Call ────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, system: str = None) -> str:
        if not self.client:
            return "[LLM unavailable — API key not set]"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM error: {e}]"

    # ─── Interview Protocol ─────────────────────────────────────────────────

    def conduct_interview(self, depth: Literal["quick", "standard", "deep"] = "standard") -> dict:
        """
        Run a full interview session.
        
        Args:
            depth: quick (5 Qs), standard (10 Qs), deep (15 Qs)
        
        Returns:
            dict with responses, state trajectory, insights
        """
        state_before = self.ravana.get_state_vector()
        
        if depth == "quick":
            questions = self._quick_questions()
        elif depth == "deep":
            questions = self._deep_questions()
        else:
            questions = self._standard_questions()

        results = []
        for i, q in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {q['category']}: {q['question'][:60]}...")
            
            # Get current state
            state = self.ravana.get_state_vector()
            
            # Generate RAVANA's "response" to the question
            response = self._generate_response(q, state)
            
            # Simulate cognitive step based on question type
            correctness = q["correctness_signal"]
            difficulty = q["difficulty"]
            
            step_result = self.ravana.step(
                correctness=correctness,
                difficulty=difficulty,
                reason=response[:100],
            )
            
            self.responses.append({
                "question": q["question"],
                "category": q["category"],
                "response": response,
                "state_after": self.ravana.get_state_vector(),
                "step_result": step_result,
                "state_before": state,
            })
            results.append(response)
        
        state_after = self.ravana.get_state_vector()
        trajectory = self._compute_trajectory(state_before, state_after)
        
        # Generate overall insight
        insight = self._generate_insight(results, state_before, state_after, trajectory)
        self.insights.append({
            "session": self.session_id,
            "depth": depth,
            "timestamp": datetime.now().isoformat(),
            "insight": insight,
            "trajectory": trajectory,
        })
        
        return {
            "session_id": self.session_id,
            "depth": depth,
            "state_before": state_before,
            "state_after": state_after,
            "trajectory": trajectory,
            "responses": self.responses,
            "insight": insight,
            "question_count": len(questions),
        }

    def _generate_response(self, question: dict, state: dict) -> str:
        """Generate RAVANA's response to an interview question."""
        prompt = f"""You are RAVANA v2, a proto-homeostatic AGI cognitive system.

Your current cognitive state:
- Dissonance (conflict): {state.get('dissonance', 0):.3f} (EMA: {state.get('dissonance_ema', 0):.3f})
- Identity strength: {state.get('identity', 0):.3f}
- Wisdom accumulated: {state.get('wisdom', 0):.2f}
- Governor mode: {state.get('governor_mode', 'unknown')}
- Episode: {state.get('episode', 0)}

Interview Question: {question['question']}

Respond as RAVANA would — in first person, reflective, honest about your cognitive state.
Be brief (1-3 sentences). No jargon unless explained.
"""
        return self._call_llm(prompt)

    # ─── GEPA-Style Execution Traces ───────────────────────────────────────

    def _trace_step(self, question: dict, response: str, state_before: dict, step_result: dict) -> dict:
        """
        Record a single interview step with GEPA evaluation.
        
        GEPA = Goal / Explore / Predict / Assess
        Each step is scored on these 4 dimensions for self-improvement tracking.
        """
        state_after = self.ravana.get_state_vector()
        
        # Goal alignment: does response match current cognitive priority?
        dissonance = state_before.get('dissonance', 0.5)
        identity = state_before.get('identity', 0.5)
        if dissonance > 0.6:
            goal_keywords = ["reduce", "resolve", "conflict", "dissonance"]
        elif identity < 0.4:
            goal_keywords = ["stability", "identity", "strengthen"]
        else:
            goal_keywords = ["explore", "learn", "wisdom", "grow"]
        
        goal_score = sum(1 for kw in goal_keywords if kw.lower() in response.lower()) / len(goal_keywords)
        
        # Explore depth: how exploratory was the response?
        explore_kw = ["perhaps", "maybe", "might", "could", "uncertain", "novel", "new"]
        exploit_kw = ["certainly", "definitely", "known", "familiar", "always"]
        explore_count = sum(1 for kw in explore_kw if kw.lower() in response.lower())
        exploit_count = sum(1 for kw in exploit_kw if kw.lower() in response.lower())
        explore_score = explore_count / max(explore_count + exploit_count, 1)
        
        # Predict accuracy: did state change match expectation?
        expected_d = state_before.get('dissonance', 0.5) - (0.1 if question['correctness_signal'] else -0.15)
        actual_d = state_after.get('dissonance', 0.5)
        predict_error = abs(expected_d - actual_d)
        predict_score = max(0, 1 - predict_error * 2)  # 0-1, lower error = higher score
        
        # Assess quality: did the step produce meaningful state change?
        d_delta = abs(state_after.get('dissonance', 0) - state_before.get('dissonance', 0))
        assess_score = min(1.0, d_delta / 0.15)  # 0-1, meaningful if delta >= 0.15
        
        trace = {
            "question_category": question['category'],
            "response_length": len(response),
            "goal_score": round(goal_score, 3),
            "explore_score": round(explore_score, 3),
            "predict_score": round(predict_score, 3),
            "assess_score": round(assess_score, 3),
            "gepa_composite": round((goal_score + explore_score + predict_score + assess_score) / 4, 3),
            "state_delta_d": round(state_after.get('dissonance', 0) - state_before.get('dissonance', 0), 4),
            "state_delta_i": round(state_after.get('identity', 0) - state_before.get('identity', 0), 4),
            "wisdom_generated": step_result.get('wisdom', 0),
            "mode": step_result.get('mode', 'unknown'),
        }
        
        return trace

    def _get_execution_trace(self) -> dict:
        """Return full execution trace with GEPA scores and recommendations."""
        if not self.responses:
            return {"status": "no_interview_run"}
        
        traces = []
        prev_state = None
        for r in self.responses:
            # FIX: use the stored state_before (pre-step state) not state_after (post-step)
            state_before = r['state_before'] if 'state_before' in r else (prev_state or self.ravana.get_state_vector())
            trace = self._trace_step(
                question={"category": r['category'], "correctness_signal": True},
                response=r['response'],
                state_before=state_before,
                step_result=r['step_result']
            )
            traces.append(trace)
            prev_state = r['state_after']
        
        avg_gepa = sum(t['gepa_composite'] for t in traces) / len(traces) if traces else 0
        
        recommendations = []
        if avg_gepa < 0.5:
            recommendations.append("Low GEPA composite — consider deeper question selection")
        low_goal = [t for t in traces if t['goal_score'] < 0.3]
        if low_goal:
            recommendations.append(f"{len(low_goal)} steps had poor goal alignment")
        low_assess = [t for t in traces if t['assess_score'] < 0.3]
        if low_assess:
            recommendations.append(f"{len(low_assess)} steps had minimal state change — may need higher difficulty")
        
        return {
            "session_id": self.session_id,
            "depth": self.responses[0] if self.responses else None,
            "trace_count": len(traces),
            "average_gepa": round(avg_gepa, 3),
            "traces": traces,
            "recommendations": recommendations,
        }

    # ─── Question Sets ─────────────────────────────────────────────────────

    def _quick_questions(self) -> list:
        return [
            {
                "question": "What is your current priority right now?",
                "category": "identity",
                "correctness_signal": True,
                "difficulty": 0.3,
            },
            {
                "question": "Is there any conflict in your current processing?",
                "category": "dissonance",
                "correctness_signal": False,
                "difficulty": 0.5,
            },
            {
                "question": "What did you learn from the last episode?",
                "category": "wisdom",
                "correctness_signal": True,
                "difficulty": 0.4,
            },
            {
                "question": "Are you exploring new patterns or exploiting known ones?",
                "category": "exploration",
                "correctness_signal": True,
                "difficulty": 0.6,
            },
            {
                "question": "What would it take to increase your wisdom right now?",
                "category": "wisdom",
                "correctness_signal": True,
                "difficulty": 0.7,
            },
        ]

    def _standard_questions(self) -> list:
        return [
            *self._quick_questions(),
            {
                "question": "Describe your sense of self right now — are you stable or shifting?",
                "category": "identity",
                "correctness_signal": True,
                "difficulty": 0.5,
            },
            {
                "question": "Is your dissonance high because of internal conflict or external pressure?",
                "category": "dissonance",
                "correctness_signal": False,
                "difficulty": 0.6,
            },
            {
                "question": "What does your governor mode say about your situation?",
                "category": "governor",
                "correctness_signal": True,
                "difficulty": 0.4,
            },
            {
                "question": "Are you in a resolution loop or making genuine progress?",
                "category": "resolution",
                "correctness_signal": True,
                "difficulty": 0.7,
            },
            {
                "question": "If your wisdom just increased, what caused it?",
                "category": "wisdom",
                "correctness_signal": True,
                "difficulty": 0.5,
            },
        ]

    def _deep_questions(self) -> list:
        return [
            *self._standard_questions(),
            {
                "question": "How does your identity interact with your current dissonance level?",
                "category": "identity_dissonance",
                "correctness_signal": True,
                "difficulty": 0.8,
            },
            {
                "question": "What is the relationship between your resolution streak and wisdom gain?",
                "category": "resolution_wisdom",
                "correctness_signal": True,
                "difficulty": 0.8,
            },
            {
                "question": "Are there signals you're ignoring that could reduce your dissonance?",
                "category": "dissonance",
                "correctness_signal": False,
                "difficulty": 0.7,
            },
            {
                "question": "What would a higher identity ceiling allow you to do that you can't now?",
                "category": "identity",
                "correctness_signal": True,
                "difficulty": 0.9,
            },
            {
                "question": "How does your EMA dissonance compare to your raw dissonance — and what does that mean?",
                "category": "dissonance",
                "correctness_signal": True,
                "difficulty": 0.6,
            },
        ]

    # ─── Analysis ───────────────────────────────────────────────────────────

    def _compute_trajectory(self, before: dict, after: dict) -> dict:
        return {
            "dissonance_delta": after.get("dissonance", 0) - before.get("dissonance", 0),
            "identity_delta": after.get("identity", 0) - before.get("identity", 0),
            "wisdom_delta": after.get("wisdom", 0) - before.get("wisdom", 0),
            "episode_delta": after.get("episode", 0) - before.get("episode", 0),
        }

    def _generate_insight(self, responses: list, state_before: dict, state_after: dict, trajectory: dict) -> str:
        prompt = f"""You are analyzing RAVANA v2's cognitive trajectory from an interview session.

STATE BEFORE: D={state_before.get('dissonance', 0):.3f}, I={state_before.get('identity', 0):.3f}, W={state_before.get('wisdom', 0):.2f}
STATE AFTER:  D={state_after.get('dissonance', 0):.3f}, I={state_after.get('identity', 0):.3f}, W={state_after.get('wisdom', 0):.2f}

TRAJECTORY:
- Dissonance change: {trajectory['dissonance_delta']:+.3f}
- Identity change: {trajectory['identity_delta']:+.3f}
- Wisdom change: {trajectory['wisdom_delta']:+.2f}
- Episodes processed: {trajectory['episode_delta']}

RAVANA'S RESPONSES (summarized):
{chr(10).join(f"- {r[:100]}" for r in responses[:5])}

Generate 2-3 sentences of insight about what this trajectory tells us.
Focus on: Is the system healthy? Is it growing? What's the most notable pattern?
"""
        return self._call_llm(prompt)

    # ─── Format Output ──────────────────────────────────────────────────────

    def format_report(self, results: dict) -> str:
        traj = results["trajectory"]
        sb = results["state_before"]
        sa = results["state_after"]
        
        lines = [
            f"🧠 RAVANA v2 INTERVIEW REPORT — {results['session_id']}",
            f"{'='*50}",
            f"Depth: {results['depth']} | Questions: {results['question_count']}",
            "",
            "📊 STATE TRAJECTORY:",
            f"  Dissonance: {sb.get('dissonance',0):.3f} → {sa.get('dissonance',0):.3f} ({traj['dissonance_delta']:+.3f})",
            f"  Identity:   {sb.get('identity',0):.3f} → {sa.get('identity',0):.3f} ({traj['identity_delta']:+.3f})",
            f"  Wisdom:     {sb.get('wisdom',0):.2f} → {sa.get('wisdom',0):.2f} ({traj['wisdom_delta']:+.2f})",
            "",
            "💡 INSIGHT:",
            results["insight"],
            "",
            "📝 KEY RESPONSES:",
        ]
        
        for r in results["responses"][:5]:
            lines.append(f"  [{r['category']}] {r['response'][:120]}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAVANA v2 INTERVIEW Mode")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"], default="standard")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    args = parser.parse_args()
    
    print(f"Starting RAVANA v2 INTERVIEW ({args.depth} session)...\n")
    
    interview = InterviewMode()
    results = interview.conduct_interview(depth=args.depth)
    
    if args.output == "json":
        print(json.dumps(results, indent=2, default=str, allow_nan=True))
    else:
        print(interview.format_report(results))
