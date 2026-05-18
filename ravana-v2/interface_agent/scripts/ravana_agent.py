#!/usr/bin/env python3
"""
RAVANA v2 — Interface Agent (Main Orchestrator)

Coordinates all components:
- RAVANA core (via wrapper)
- LLM interpreter (bidirectional translation)
- Reality grounding (web research, RSS)
- Telegram reporter (delivery)
- Memory learner (persistent learning)

Usage:
    python3 ravana_agent.py --task "question or action"
    python3 ravana_agent.py --interactive
    python3 ravana_agent.py --diagnose
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import components
from ravana_wrapper import RavanaWrapper
from llm_interpreter import LLMInterpreter
from reality_grounding import RealityGrounding
from telegram_reporter import TelegramReporter
from memory_learner import MemoryLearner


class RavanaAgent:
    """
    Main orchestrator for the RAVANA Interface Agent.
    
    Coordinates:
    1. User input → LLM interpreter → RAVANA state transition
    2. RAVANA output → LLM interpreter → Human explanation
    3. Reality grounding (news, RSS) for world-knowledge
    4. Telegram delivery for reports and alerts
    5. Memory learning for persistent improvement
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        telegram_enabled: bool = True,
        grounding_interval: int = 5,
    ):
        self._init_components(llm_provider, telegram_enabled)
        self.grounding_interval = grounding_interval
        self.episode_count = 0
        self.recent_episodes: list = []
    
    def _init_components(self, llm_provider: str, telegram_enabled: bool):
        """Initialize all sub-components."""
        print("  [Agent] Initializing RAVANA Interface Agent...")
        
        # Core RAVANA wrapper
        self.ravana = RavanaWrapper()
        print("  [Agent] ✓ RAVANA v2 core loaded")
        
        # LLM interpreter
        self.interpreter = LLMInterpreter(provider=llm_provider)
        print(f"  [Agent] ✓ LLM interpreter ready ({llm_provider})")
        
        # Reality grounding
        self.grounding = RealityGrounding()
        print("  [Agent] ✓ Reality grounding active (RSS + news)")
        
        # Telegram reporter
        if telegram_enabled:
            try:
                from zo_py import send_telegram_message
                self._telegram_send = lambda msg: send_telegram_message(message=msg)
                reporter = TelegramReporter(send_fn=self._telegram_send)
            except ImportError:
                reporter = TelegramReporter(send_fn=None)
        else:
            reporter = TelegramReporter(send_fn=None)
        self.reporter = reporter
        print(f"  [Agent] ✓ Telegram reporter ready ({'enabled' if telegram_enabled else 'disabled'})")
        
        # Memory learner
        self.learner = MemoryLearner()
        print("  [Agent] ✓ Memory learner active")
        
        print("  [Agent] All systems initialized.\n")
    
    def run_interactive(self):
        """Run interactive mode — user types commands, agent responds."""
        print("=" * 50)
        print("  RAVANA v2 Interface Agent — Interactive Mode")
        print("=" * 50)
        print("Type a message to RAVANA, or 'quit' to exit.\n")
        
        state = self.ravana.get_state_vector()
        self.reporter.send_status_card(state, episode=self.episode_count)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'status':
                self._show_status()
                continue
            
            if user_input.lower() == 'diagnose':
                self._run_diagnosis()
                continue
            
            if user_input.lower() == 'news':
                self._show_news()
                continue
            
            if user_input.lower() == 'learn':
                self._show_lessons()
                continue
            
            # Process the message
            response = self.process_message(user_input)
            print(f"\nRAVANA: {response}\n")
    
    def process_message(self, message: str) -> str:
        """
        Process a user message end-to-end.
        
        Pipeline:
        1. Interpret user intent → RAVANA signals
        2. Step RAVANA with those signals
        3. Optionally ground in reality (news/RSS)
        4. Learn from the episode
        5. Explain what happened to the user
        """
        # 1. Get current state
        current_state = self.ravana.get_state_vector()
        
        # 2. Interpret user intent
        interpretation = self.interpreter.interpret_user_intent(message, current_state)
        print(f"  [Interpret] correctness={interpretation['correctness']}, "
              f"difficulty={interpretation['difficulty']:.2f}, "
              f"confidence={interpretation['confidence']:.0%}")
        
        # 3. Step RAVANA
        result = self.ravana.step(
            correctness=interpretation['correctness'],
            difficulty=interpretation['difficulty'],
            reason=interpretation.get('interpretation', message),
        )
        self.episode_count += 1
        result['episode'] = self.episode_count
        
        self.recent_episodes.append(result)
        if len(self.recent_episodes) > 20:
            self.recent_episodes.pop(0)
        
        # 4. Reality grounding (every N episodes)
        reality_result = None
        if self.episode_count % self.grounding_interval == 0:
            news = self.grounding.fetch_rss_feeds()
            if news:
                # Find relevant articles
                relevant = news[:5]
                belief = interpretation.get('interpretation', message)
                reality_result = self.grounding.evaluate_belief_alignment(
                    belief=belief,
                    action=message,
                    news_items=relevant,
                )
                print(f"  [Ground] Alignment: {reality_result['verdict']} "
                      f"(score: {reality_result['alignment_score']:.2f})")
        
        # 5. Learn from episode
        lesson = self.learner.learn_from_episode(
            episode_data=result,
            reality_result=reality_result,
            user_feedback=message,
        )
        print(f"  [Learn] Lesson: {lesson.lesson[:80]}...")
        
        # 6. Send episode summary
        self.reporter.send_episode_summary(result)
        
        # 7. Generate insight
        if self.episode_count % 10 == 0:
            insight = self.interpreter.generate_insight(self.recent_episodes, current_state)
            self.reporter.send_report("insight", {"insight": insight, "state": current_state})
        
        # 8. Explain to user
        explanation = self.interpreter.explain_state(
            self.ravana.get_state_vector(),
            question=message,
        )
        
        return explanation
    
    def run_task(self, task: str) -> dict:
        """
        Run a single task and return results.
        
        Args:
            task: Natural language task description
        
        Returns:
            Dict with results, state, and reports
        """
        print(f"  [Agent] Processing task: {task}")
        
        # Process message
        self.process_message(task)
        
        # Get final state
        state = self.ravana.get_state_vector()
        diagnosis = self.ravana.get_diagnosis()
        clamp_metrics = self.ravana.governor.get_clamp_metrics()
        
        return {
            "task": task,
            "episode": self.episode_count,
            "final_state": state,
            "diagnosis": diagnosis,
            "clamp_metrics": clamp_metrics,
            "lessons_learned": len(self.learner.lessons),
        }
    
    def run_diagnosis(self) -> str:
        """Run full system diagnosis."""
        return self.reporter.send_diagnostic_report(self.ravana)
    
    def get_grounding_report(self) -> str:
        """Get current reality grounding report."""
        return self.grounding.get_grounding_report(
            self.ravana.get_state_vector(),
            self.recent_episodes,
        )
    
    # ─── Private Methods ────────────────────────────────────────────────────────
    
    def _show_status(self):
        state = self.ravana.get_state_vector()
        self.reporter.send_status_card(state, episode=self.episode_count)
        print()
        print(self.ravana.get_diagnosis())
    
    def _run_diagnosis(self):
        result = self.run_diagnosis()
        print(result)
    
    def _show_news(self):
        report = self.get_grounding_report()
        print(report)
    
    def _show_lessons(self):
        summary = self.learner.get_lesson_summary()
        print(summary)
        
        state = self.ravana.get_state_vector()
        rec = self.learner.get_recommendation(state)
        print(f"\nCurrent recommendation: {rec}")


def main():
    parser = argparse.ArgumentParser(description="RAVANA v2 Interface Agent")
    parser.add_argument("--task", type=str, help="Single task to process")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--diagnose", action="store_true", help="Run system diagnosis")
    parser.add_argument("--grounding-interval", type=int, default=5,
                        help="Episodes between reality grounding (default: 5)")
    parser.add_argument("--llm-provider", type=str, default="openai",
                        choices=["openai", "anthropic", "ollama"],
                        help="LLM provider for interpretation")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram reporting")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = RavanaAgent(
        llm_provider=args.llm_provider,
        telegram_enabled=not args.no_telegram,
        grounding_interval=args.grounding_interval,
    )
    
    if args.task:
        # Run single task
        result = agent.run_task(args.task)
        print("\n=== RESULT ===")
        print(json.dumps(result, indent=2, default=str))
    
    elif args.interactive:
        # Interactive mode
        agent.run_interactive()
    
    elif args.diagnose:
        # Diagnosis mode
        result = agent.run_diagnosis()
        print(result)
    
    else:
        # Default: show status
        agent._show_status()


if __name__ == "__main__":
    main()