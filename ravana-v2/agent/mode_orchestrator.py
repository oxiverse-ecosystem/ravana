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
from typing import Dict, Any, List, Optional
from datetime import datetime

# Paths
RAVANA_DIR = Path("/home/workspace/Projects/ravana-v2")
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

    def __init__(self, groq_api_key: str, db_path: str):
        self.groq_api_key = groq_api_key
        self.db_path = db_path
        self.vm = VersionManager(db_path=db_path)
        self.ravana = RavanaWrapper()
        self.last_D = self.ravana.get_diagnosis()['dissonance']
        self.test_harness = None  # Lazy load
        self.mode_history: List[Dict] = []

    def decide_mode(self) -> ModeDecision:
        """
        Decide which mode to run based on state.
        
        Logic:
        - RESEARCH: Every N runs OR pending improvements
        - INTERVIEW: Every run (validate RAVANA behavior)
        - LEARN: If info_collector has new events
        """
        status = self.vm.get_status()
        pending = status.get('pending_improvements', 0)
        recent_tests = status.get('recent_tests', [])
        experiments = status.get('active_experiments', [])
        
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
                reason=f"{len(experiments)} active experiments",
                confidence=0.7,
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
        from reality_grounding import RealityGrounding
        
        rg = RealityGrounding()
        events = rg.fetch_all()
        
        # Analyze events for relevant patterns
        research_notes = {
            'events_collected': len(events),
            'sources': rg.sources,
            'findings': []
        }
        
        # Queue improvements based on research
        improvements = 0
        if len(events) > 10:
            self.vm.add_improvement(
                category='info_collector',
                description=f"News events suggest: {events[0]['title'][:50]}",
                source='news',
                priority='medium'
            )
            improvements += 1
        
        return {
            'mode': 'research',
            'events': len(events),
            'improvements_queued': improvements,
            'notes': research_notes
        }

    def run_interview_mode(self) -> Dict[str, Any]:
        """Groq → RAVANA → test/evaluate"""
        from test_harness import TestHarness
        
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
            self.vm.add_test(
                test_name=f"card_{result.card_id}",
                status=status,
                output=f"D: {result.actual_D:.3f} vs {result.expected_D:.3f}",
                duration_ms=0,
                notes=result.notes
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
        from reality_grounding import RealityGrounding
        
        rg = RealityGrounding()
        events = rg.fetch_all()
        
        learn_results = []
        for event in events[:5]:  # Process top 5
            # Convert to situation card
            card = self._event_to_card(event)
            
            if card:
                # Run as RAVANA step
                result = self.ravana.step(
                    correctness=card.get('correctness', False),
                    difficulty=card.get('difficulty', 0.5),
                    reason=f"learn:{event.get('source', 'news')}"
                )
                
                # Track D change
                new_D = result['post_dissonance']
                D_delta = abs(new_D - self.last_D)
                self.last_D = new_D
                
                learn_results.append({
                    'event': event.get('title', '')[:50],
                    'D_before': self.last_D,
                    'D_after': new_D,
                    'D_delta': D_delta
                })
        
        return {
            'mode': 'learn',
            'events_processed': len(learn_results),
            'results': learn_results
        }

    def _event_to_card(self, event: Dict) -> Optional[Dict]:
        """Convert news event to situation card format"""
        # Extract domain from event
        title = event.get('title', '').lower()
        
        # Map to RAVANA domain
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
            'source': event.get('source', 'news')
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
            lines.append(f"Events: {result['events']}, queued: {result['improvements_queued']}")
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
    db = "/home/workspace/Projects/ravana-v2/interface_agent/context.db"
    
    orch = ModeOrchestrator(groq_api_key=groq_key, db_path=db)
    report = orch.run_full_cycle()
    
    print(orch.build_telegram_report(report))


if __name__ == "__main__":
    main()
