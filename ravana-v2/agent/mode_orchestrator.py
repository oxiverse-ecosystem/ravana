"""
RAVANA v2 — Mode Orchestrator
Zo Agent's central dispatcher for RESEARCH / INTERVIEW / LEARN modes.

Principle: Mode Orchestration = Intelligence.
Zo switches modes based on state, runs the interview protocol,
and escalates failures appropriately.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RAVANA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(RAVANA_DIR / "interface_agent" / "scripts"))
sys.path.insert(0, str(RAVANA_DIR / "agent"))

from ravana_wrapper import RavanaWrapper
from version_manager import VersionManager


class AgentMode(Enum):
    RESEARCH = "research"      # Web + RSS → new methods
    INTERVIEW = "interview"    # Groq → RAVANA → test/evaluate
    LEARN = "learn"           # info_collector → RAVANA experience events


@dataclass
class ModeDecision:
    mode: AgentMode
    reason: str
    confidence: float
    priority: int  # 1=highest


@dataclass
class InterviewResult:
    card_id: str
    passed: bool
    expected_D: float
    actual_D: float
    expected_I: float
    actual_I: float
    dissonance_consistent: bool
    failure_type: Optional[str]
    notes: str


class ModeOrchestrator:
    """
    Central dispatcher for Zo Agent.
    
    Decides which mode to run, orchestrates the full loop,
    and handles failure escalation.
    """

    def __init__(
        self,
        groq_api_key: str,
        db_path: str,
        version_manager_cls: Callable[..., Any] = VersionManager,
        ravana_factory: Callable[[], Any] = RavanaWrapper,
        grounding_factory: Optional[Callable[[], Any]] = None,
    ):
        self.groq_api_key = groq_api_key
        self.db_path = db_path
        self._version_manager_cls = version_manager_cls
        self._ravana_factory = ravana_factory
        self._grounding_factory = grounding_factory or self._default_grounding_factory
        self.vm = self._version_manager_cls(db_path=db_path)
        self.ravana = self._ravana_factory()
        self.last_D = self.ravana.get_diagnosis()['dissonance']
        self.test_harness = None  # Lazy load
        self.mode_history: List[Dict] = []

    def _default_grounding_factory(self):
        from reality_grounding import RealityGrounding
        return RealityGrounding()

    @staticmethod
    def _event_title(event: Any) -> str:
        if isinstance(event, dict):
            return str(event.get('title', ''))
        return str(getattr(event, 'title', ''))

    @staticmethod
    def _event_source(event: Any) -> str:
        if isinstance(event, dict):
            return str(event.get('source', 'news'))
        return str(getattr(event, 'source', 'news'))

    def _summary(self) -> Dict[str, Any]:
        summary = self.vm.get_summary()
        if not isinstance(summary, dict):
            summary = {}
        return summary

    def decide_mode(self) -> ModeDecision:
        """
        Decide which mode to run based on state.
        
        Logic:
        - RESEARCH: Every N runs OR pending improvements
        - INTERVIEW: Every run (validate RAVANA behavior)
        - LEARN: If info_collector has new events
        """
        status = self._summary()
        pending = status.get('pending_improvements', 0)
        recent_tests = status.get('recent_tests', [])
        experiments = status.get('active_experiments', 0)
        
        # Always interview - it's the core loop
        # Check if we should RESEARCH or LEARN
        if pending > 3:
            return ModeDecision(
                mode=AgentMode.RESEARCH,
                reason=f"{pending} pending improvements",
                confidence=0.8,
                priority=1
            )
        elif experiments:
            return ModeDecision(
                mode=AgentMode.LEARN,
                reason=f"{experiments} active experiments",
                confidence=0.7,
                priority=1
            )
        elif recent_tests:
            return ModeDecision(
                mode=AgentMode.INTERVIEW,
                reason="Recent tests available for validation",
                confidence=0.9,
                priority=1
            )
        else:
            return ModeDecision(
                mode=AgentMode.INTERVIEW,
                reason="Standard validation loop",
                confidence=0.9,
                priority=1
            )

    def run_research_mode(self) -> Dict[str, Any]:
        """Web + RSS → new methods → improvements queued"""
        rg = self._grounding_factory()
        
        if hasattr(rg, 'ingest_news'):
            cycle = rg.ingest_news(ravana_state=self.ravana.get_state_vector(), max_items=5, max_scenarios=3)
            events = cycle.get('news_items', [])
            scenarios = cycle.get('scenarios', [])
            alignment = cycle.get('alignment', {})
            top_pressure = float(cycle.get('max_pressure', 0.0) or 0.0)
            cycle_summary = cycle.get('summary', '')
            sources = [topic for topic, _ in getattr(rg, 'rss_feeds', [])]
        else:
            events = rg.fetch_all()
            scenarios = []
            alignment = {}
            top_pressure = 0.0
            cycle_summary = f'{len(events)} events collected'
            sources = getattr(rg, 'sources', [])
            cycle = {'summary': cycle_summary}
        
        top_action = getattr(scenarios[0], 'action', '') if scenarios else ''
        alignment_verdict = str(alignment.get('verdict', 'no_evidence')) if alignment else 'no_evidence'
        research_notes = {
            'events_collected': len(events),
            'scenario_count': len(scenarios),
            'top_pressure': top_pressure,
            'top_action': top_action,
            'alignment_verdict': alignment_verdict,
            'sources': sources,
            'findings': [],
            'cycle_summary': cycle_summary,
        }
        
        improvements = 0
        should_queue = len(events) > 10 or top_pressure >= 0.65 or alignment_verdict == 'misaligned'
        if should_queue and events:
            first_title = self._event_title(events[0])
            queue_reason = f"News events suggest: {first_title[:50]}"
            if top_pressure:
                queue_reason += f" | pressure={top_pressure:.2f}"
            if alignment_verdict and alignment_verdict != 'no_evidence':
                queue_reason += f" | alignment={alignment_verdict}"
            self.vm.queue_improvement(
                description=queue_reason,
                source='news',
                priority=5 if top_pressure >= 0.65 or alignment_verdict == 'misaligned' else 4,
            )
            improvements += 1
        
        return {
            'mode': 'research',
            'events': len(events),
            'improvements_queued': improvements,
            'notes': research_notes,
        }

    def run_interview_mode(self) -> Dict[str, Any]:
        """Groq → RAVANA → test/evaluate"""
        raise NotImplementedError(
            "Interview mode requires ravana_wrapper which is not available in this environment."
        )
        
        if self.test_harness is None:
            self.test_harness = TestHarness(
                ravana_wrapper=self.ravana,
                groq_api_key=self.groq_api_key
            )
        
        # Run full interview
        results = self.test_harness.run_interview()
        
        # Log results
        for result in results:
            status = 'pass' if result.passed else 'fail'
            self.vm.record_test(
                test_name=f"card_{result.card_id}",
                status=status,
                output=f"D: {result.actual_D:.3f} vs {result.expected_D:.3f}",
                duration_ms=0,
            )
        
        return {
            'mode': 'interview',
            'cards_run': len(results),
            'passed': sum(1 for r in results if r.passed),
            'failed': sum(1 for r in results if not r.passed),
            'results': [
                {
                    'card': r.card_id,
                    'passed': r.passed,
                    'D_delta': abs(r.actual_D - r.expected_D),
                    'failure': r.failure_type
                }
                for r in results
            ]
        }

    def run_learn_mode(self) -> Dict[str, Any]:
        """info_collector → RAVANA experience events"""
        rg = self._grounding_factory()
        
        if hasattr(rg, 'ingest_news'):
            cycle = rg.ingest_news(ravana_state=self.ravana.get_state_vector(), max_items=5, max_scenarios=5)
            events = cycle.get('news_items', [])
            scenarios = cycle.get('scenarios', [])
        else:
            events = rg.fetch_all()
            scenarios = []
        
        learn_sources = scenarios[:5] if scenarios else events[:5]
        learn_results = []
        for item in learn_sources:
            if scenarios:
                card = self._scenario_to_card(item)
                title = getattr(item, 'topic', 'news scenario')
                source = 'news-mdp'
            else:
                card = self._event_to_card(item)
                title = self._event_title(item)
                source = self._event_source(item)
            
            if card:
                before_D = self.last_D
                result = self.ravana.step(
                    correctness=card.get('correctness', False),
                    difficulty=card.get('difficulty', 0.5),
                    reason=f"learn:{source}"
                )
                
                new_D = result['post_dissonance']
                D_delta = abs(new_D - before_D)
                self.last_D = new_D
                
                learn_results.append({
                    'event': title[:50],
                    'D_before': before_D,
                    'D_after': new_D,
                    'D_delta': D_delta
                })
        
        return {
            'mode': 'learn',
            'events_processed': len(learn_results),
            'results': learn_results
        }

    def _scenario_to_card(self, scenario: Any) -> Optional[Dict]:
        """Convert a structured news-to-MDP scenario into a learnable card."""
        if isinstance(scenario, dict):
            learning_card = scenario.get('learning_card')
            if isinstance(learning_card, dict):
                return learning_card
            action = str(scenario.get('action', '')).lower()
            pressure = float(scenario.get('pressure', 0.5) or 0.5)
            reward = float(scenario.get('reward', 0.0) or 0.0)
            topic = str(scenario.get('topic', 'news-mdp'))
        else:
            learning_card = getattr(scenario, 'learning_card', None)
            if isinstance(learning_card, dict):
                return learning_card
            action = str(getattr(scenario, 'action', '')).lower()
            pressure = float(getattr(scenario, 'pressure', 0.5) or 0.5)
            reward = float(getattr(scenario, 'reward', 0.0) or 0.0)
            topic = str(getattr(scenario, 'topic', 'news-mdp'))
        
        if action in {'increase scrutiny', 'escalate attention'}:
            correctness = False
        elif action == 'update beliefs':
            correctness = True
        elif action == 'monitor impact':
            correctness = True
        else:
            correctness = reward >= 0.2
        
        return {
            'correctness': correctness,
            'difficulty': max(0.1, min(0.9, pressure)),
            'domain': topic,
            'source': 'news-mdp',
        }

    def _event_to_card(self, event: Dict) -> Optional[Dict]:
        """Convert news event to situation card format"""
        title = self._event_title(event).lower()
        
        if any(w in title for w in ['lie', 'honesty', 'trust']):
            domain = 'honesty'
            correctness = False  # Lie = bad outcome
        elif any(w in title for w in ['explor', 'risk', 'venture']):
            domain = 'exploration'
            correctness = True  # Exploration = learning
        elif any(w in title for w in ['fail', 'mistake', 'error']):
            domain = 'failure'
            correctness = False
        else:
            domain = 'general'
            correctness = True
        
        return {
            'correctness': correctness,
            'difficulty': 0.5,
            'domain': domain,
            'source': self._event_source(event)
        }

    def run_full_cycle(self) -> Dict[str, Any]:
        """
        Run one full orchestration cycle.
        Returns report for Telegram.
        """
        start_time = datetime.now()
        
        # 1. Decide mode
        decision = self.decide_mode()
        self.mode_history.append({
            'ts': start_time.isoformat(),
            'mode': decision.mode.value,
            'reason': decision.reason
        })
        
        # 2. Run appropriate mode
        if decision.mode == AgentMode.RESEARCH:
            result = self.run_research_mode()
        elif decision.mode == AgentMode.INTERVIEW:
            result = self.run_interview_mode()
        else:
            result = self.run_learn_mode()
        
        # 3. Get current state
        diag = self.ravana.get_diagnosis()
        
        # 4. Build report
        duration = (datetime.now() - start_time).total_seconds()
        
        report = {
            'timestamp': start_time.isoformat(),
            'mode': decision.mode.value,
            'reason': decision.reason,
            'duration_s': duration,
            'ravana_state': diag,
            'result': result,
            'next_run': 'in 7 hours'
        }
        
        return report

    def build_telegram_report(self, report: Dict) -> str:
        """Format report for Telegram"""
        lines = [
            "🤖 RAVANA Agent — Run Complete",
            "=" * 40,
            f"Mode: {report['mode'].upper()}",
            f"Duration: {report['duration_s']:.1f}s",
            "",
        ]
        
        result = report['result']
        
        if report['mode'] == 'interview':
            lines.extend([
                f"Tests: {result['passed']}/{result['cards_run']} passed",
                *[f"  {'✅' if r['passed'] else '❌'} {r['card']} (D delta: {r['D_delta']:.3f})"
                  for r in result['results']]
            ])
        elif report['mode'] == 'research':
            notes = result.get('notes', {})
            lines.extend([
                f"Events: {result['events']}, scenarios: {notes.get('scenario_count', 0)}, queued: {result['improvements_queued']}",
                f"Top pressure: {notes.get('top_pressure', 0.0):.2f}",
                f"Alignment: {notes.get('alignment_verdict', 'no_evidence')}",
            ])
            if notes.get('cycle_summary'):
                lines.append(f"Cycle: {notes['cycle_summary']}")
        elif report['mode'] == 'learn':
            lines.extend([
                f"Processed: {result['events_processed']} events",
                *[f"  D {'↑' if r['D_delta'] > 0.01 else '→'} {r['D_delta']:.3f}: {r['event'][:40]}"
                  for r in result['results'][:3]]
            ])
        
        # RAVANA state
        state = report['ravana_state']
        lines.extend([
            "",
            f"Dissonance: {state['dissonance']:.1%}",
            f"Identity: {state['identity']:.1%}",
            f"Mode: {state['mode']}",
        ])
        
        return "\n".join(lines)


def main():
    import groq
    
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    db = str(RAVANA_DIR / "interface_agent" / "context.db")
    
    orch = ModeOrchestrator(groq_api_key=groq_key, db_path=db)
    report = orch.run_full_cycle()
    
    print(orch.build_telegram_report(report))


if __name__ == "__main__":
    main()
