"""
RAVANA v2 — Memory Learner
Learns from interactions and stores lessons in persistent memory.
Uses the human-memory skill for continuity.
Enriches lessons with procedural memory compounding and reentry acceleration.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add human-memory skill to path
# Path: scripts/ → interface_agent/ → ravana-v2/ → Projects/ → workspace/ (5 levels)
skill_base = Path(__file__).parent.parent.parent.parent.parent
human_memory_path = skill_base / "Skills" / "human-memory" / "scripts"
sys.path.insert(0, str(human_memory_path))


@dataclass
class LearnedLesson:
    """A single lesson learned from an interaction."""
    episode: int
    situation: str  # e.g., "high_dissonance_action"
    action_taken: str
    outcome: str  # e.g., "dissonance_reduced"
    reality_check: str  # e.g., "aligned with news consensus"
    lesson: str  # The learned insight
    confidence: float  # How confident we are in this lesson
    timestamp: str
    tags: List[str]
    state_signature: Optional[str] = None  # For reentry detection
    gamma: float = 1.0  # Reentry gain multiplier (from FH-RL research)


@dataclass
class ProceduralSequence:
    """A reusable action sequence learned from successful episodes."""
    id: str
    name: str
    steps: List[str]  # Action descriptions
    situation_pattern: str  # When this applies
    success_count: int = 0
    failure_count: int = 0
    avg_reward: float = 0.0
    last_used_episode: int = 0
    gamma: float = 1.0  # Acceleration factor from FH-RL reentry gain


class MemoryLearner:
    """
    Learns from RAVANA interactions and stores lessons.
    
    Integrates with the human-memory skill for persistent storage.
    Updates a lesson index for future reference.
    
    Features (from latest research):
    - Procedural memory compounding (Hermes Agent v0.8.0 GEPA-inspired)
    - Reentry gain γ (Meta-RL for Homeostatic Regulation)
    - Adaptive αhomeo for situation-specific learning rates
    - Structural plasticity triggers (Stress-Reg)
    """
    
    def __init__(self):
        self.lessons: List[LearnedLesson] = []
        self.lesson_index: Dict[str, List[int]] = {}  # topic → lesson indices
        self._use_human_memory = self._check_human_memory()
        
        # Procedural memory compounding (from Hermes Agent)
        self.procedures: Dict[str, ProceduralSequence] = {}
        self._procedure_counter = 0
        
        # Reentry detection: tracks visited state signatures for γ acceleration
        self._state_visits: Dict[str, List[Tuple[int, float]]] = defaultdict(list)  # signature → [(episode, reward)]
        
        # Adaptive αhomeo: per-situation learning rate adjustments
        self._situation_counts: Dict[str, int] = defaultdict(int)
        self._alpha_homeo: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Structural plasticity: track when to add new memory traces
        self._stress_threshold = 0.75
        self._plasticity_events: List[dict] = []
        
        if self._use_human_memory:
            try:
                from skill import memorize, context as get_context
                self._memorize = memorize
                self._context = get_context
                print("  [Memory] Human-memory skill available — using for persistence")
            except ImportError as e:
                print(f"  [Memory] Human-memory skill not available: {e}")
                self._use_human_memory = False
    
    def _check_human_memory(self) -> bool:
        """Check if human-memory skill is available."""
        skill_base = Path(__file__).parent.parent.parent.parent.parent
        skill_path = skill_base / "Skills" / "human-memory"
        return skill_path.exists() and (skill_path / "scripts").exists()
    
    def learn_from_episode(
        self,
        episode_data: dict,
        reality_result: dict = None,
        user_feedback: str = None,
    ) -> LearnedLesson:
        """
        Extract and store a lesson from a completed episode.
        
        Args:
            episode_data: Output from RavanaWrapper.step()
            reality_result: Optional reality grounding result
            user_feedback: Optional user feedback on the action
        
        Returns:
            The newly created LearnedLesson
        """
        state = episode_data
        episode = state.get('episode', 0)
        post_d = state.get('post_dissonance', 0)
        pre_d = state.get('pre_dissonance', 0)
        post_i = state.get('post_identity', 0)
        mode = state.get('mode', 'unknown')
        
        # Determine situation
        d_change = post_d - pre_d
        if post_d > 0.7:
            situation = "high_dissonance"
        elif post_d < 0.25:
            situation = "low_dissonance_stagnation"
        elif abs(d_change) > 0.1:
            situation = "rapid_change"
        else:
            situation = "stable_operation"
        
        # Determine outcome
        if d_change < -0.05:
            outcome = "dissonance_reduced"
        elif d_change > 0.05:
            outcome = "dissonance_increased"
        else:
            outcome = "dissonance_stable"
        
        if post_i > state.get('pre_identity', 0) + 0.05:
            outcome += "_identity_grew"
        elif post_i < state.get('pre_identity', 0) - 0.05:
            outcome += "_identity_shrank"
        
        # Determine action taken
        action = f"governor_mode_{mode}"
        if state.get('resolution', {}).get('full_resolution'):
            action += "_full_resolution"
        
        # Reality check
        reality_check = "not_evaluated"
        if reality_result:
            alignment = reality_result.get('verdict', 'no_evidence')
            score = reality_result.get('alignment_score', 0.5)
            reality_check = f"{alignment} (score: {score:.2f})"
        
        # Extract lesson
        lesson_text = self._extract_lesson(state, situation, outcome, mode, d_change)
        
        # Compute confidence
        confidence = min(0.95, 0.5 + state.get('wisdom', 0) * 0.02)
        if reality_result:
            confidence = min(0.95, confidence + reality_result.get('confidence', 0) * 0.1)
        if user_feedback:
            confidence = min(0.95, confidence + 0.1)
        
        lesson = LearnedLesson(
            episode=episode,
            situation=situation,
            action_taken=action,
            outcome=outcome,
            reality_check=reality_check,
            lesson=lesson_text,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            tags=self._generate_tags(situation, mode, outcome),
        )
        
        self.lessons.append(lesson)
        self._update_index(lesson)
        
        # Store in human-memory if available
        if self._use_human_memory:
            self._store_in_human_memory(lesson)
        
        return lesson
    
    def learn_from_grounding_cycle(self, grounding_cycle: dict) -> List[LearnedLesson]:
        """Store a batch of news-derived grounding examples as lessons."""
        batch = grounding_cycle.get("learning_batch", []) or []
        learned: List[LearnedLesson] = []

        for idx, example in enumerate(batch, start=1):
            topic = str(example.get("topic", "news")).strip() or "news"
            action = str(example.get("action", "hold position")).strip()
            reward = float(example.get("reward", 0.0) or 0.0)
            pressure = float(example.get("pressure", 0.0) or 0.0)
            confidence = float(example.get("confidence", 0.5) or 0.5)
            rationale = str(example.get("rationale", "")).strip()
            state = example.get("state", {}) if isinstance(example.get("state", {}), dict) else {}
            next_state = example.get("next_state", {}) if isinstance(example.get("next_state", {}), dict) else {}

            if reward >= 0.25:
                outcome = "grounding_reward_positive"
            elif reward <= 0.0:
                outcome = "grounding_reward_negative"
            else:
                outcome = "grounding_reward_mixed"

            if pressure >= 0.7:
                situation = f"news_high_pressure_{topic.replace(' ', '_').lower()}"
            elif pressure >= 0.35:
                situation = f"news_medium_pressure_{topic.replace(' ', '_').lower()}"
            else:
                situation = f"news_low_pressure_{topic.replace(' ', '_').lower()}"

            lesson_text = (
                f"News grounding: {action} for {topic} "
                f"(reward={reward:+.2f}, pressure={pressure:.2f}, confidence={confidence:.2f})"
            )
            if rationale:
                lesson_text += f" — {rationale}"

            lesson = LearnedLesson(
                episode=int(example.get("episode", len(self.lessons) + idx)),
                situation=situation,
                action_taken=action,
                outcome=outcome,
                reality_check=str(example.get("source_url", "news-mdp")),
                lesson=lesson_text,
                confidence=min(0.95, max(0.2, confidence)),
                timestamp=datetime.now().isoformat(),
                tags=["news", "grounding", "mdp", topic.replace(" ", "_").lower()],
            )

            self.lessons.append(lesson)
            self._update_index(lesson)
            if self._use_human_memory:
                self._store_in_human_memory(lesson)

            if isinstance(state, dict) and isinstance(next_state, dict):
                procedure = self.compound_procedure(
                    episode_data={
                        "episode": lesson.episode,
                        "dissonance": float(state.get("dissonance", 0.5) or 0.5),
                        "identity": float(state.get("identity", 0.5) or 0.5),
                        "mode": "grounding",
                    },
                    action_sequence=[action, rationale] if rationale else [action],
                    reward=reward,
                )
                if procedure is not None:
                    self.record_state_visit(
                        state_signature=f"news:{topic}:{action}",
                        episode=lesson.episode,
                        reward=reward,
                    )

            learned.append(lesson)

        return learned
    
    def get_relevant_lessons(self, situation: str, limit: int = 3) -> List[LearnedLesson]:
        """
        Get lessons relevant to the current situation.
        
        Args:
            situation: Current situation type (e.g., "high_dissonance")
            limit: Max lessons to return
        
        Returns:
            List of most relevant lessons
        """
        # Direct lookup
        relevant = self.lesson_index.get(situation, [])
        direct = [self.lessons[i] for i in relevant[-limit:] if i < len(self.lessons)]
        
        # Fallback: find similar situations
        if not direct:
            similar_situations = {
                "high_dissonance": ["rapid_change", "dissonance_increased"],
                "low_dissonance_stagnation": ["stable_operation"],
                "rapid_change": ["high_dissonance"],
            }
            for sim in similar_situations.get(situation, []):
                idx_list = self.lesson_index.get(sim, [])
                for i in idx_list[-limit:]:
                    if i < len(self.lessons) and self.lessons[i] not in direct:
                        direct.append(self.lessons[i])
        
        return direct[:limit]
    
    def get_lesson_summary(self) -> str:
        """Get a human-readable summary of learned lessons."""
        if not self.lessons:
            return "No lessons learned yet."
        
        lines = [
            "=== LEARNED LESSONS ===",
            f"Total lessons: {len(self.lessons)}",
            "",
        ]
        
        # Group by situation
        by_situation = {}
        for lesson in self.lessons:
            if lesson.situation not in by_situation:
                by_situation[lesson.situation] = []
            by_situation[lesson.situation].append(lesson)
        
        for situation, lessons in by_situation.items():
            lines.append(f"{situation.upper()} ({len(lessons)} lessons):")
            for lesson in lessons[-2:]:  # Show last 2 per situation
                lines.append(f"  • Ep{lesson.episode}: {lesson.lesson[:80]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_recommendation(self, state: dict) -> str:
        """
        Get a recommendation based on learned lessons.
        
        Args:
            state: Current RAVANA state
        
        Returns:
            Recommended action based on past learning
        """
        d = state.get('dissonance', 0)
        mode = state.get('governor_mode', 'normal')
        
        # Determine situation
        if d > 0.7:
            situation = "high_dissonance"
        elif d < 0.25:
            situation = "low_dissonance_stagnation"
        else:
            situation = "stable_operation"
        
        # Get relevant lessons
        lessons = self.get_relevant_lessons(situation)
        
        if not lessons:
            # Default recommendations
            if d > 0.7:
                return "Prioritize resolution. High dissonance detected — examine conflicting beliefs."
            elif d < 0.25:
                return "Seek exploration. Low dissonance may indicate stagnation."
            else:
                return "Continue current trajectory. System is healthy."
        
        # Use highest confidence lesson
        best = max(lessons, key=lambda l: l.confidence)
        
        return f"Based on Ep{best.episode}: {best.lesson[:100]}... (confidence: {best.confidence:.0%})"
    
    def export_lessons(self, path: str = None) -> List[dict]:
        """Export lessons as serializable dicts."""
        return [asdict(l) for l in self.lessons]
    
    # ─── Private Methods ────────────────────────────────────────────────────────
    
    def _extract_lesson(
        self,
        state: dict,
        situation: str,
        outcome: str,
        mode: str,
        d_change: float,
    ) -> str:
        """Extract a lesson from episode data."""
        if situation == "high_dissonance" and d_change < 0:
            return "When D>0.7, resolution mode effectively reduces dissonance. Continue active conflict examination."
        elif situation == "high_dissonance" and d_change >= 0:
            return "High dissonance persists despite action. Consider deeper reappraisal or external grounding."
        elif situation == "low_dissonance_stagnation":
            return "Low dissonance may lead to stagnation. Explore novel patterns to prevent complacency."
        elif d_change < -0.1:
            return f"Governor mode {mode} successfully reduced dissonance. Record as effective strategy."
        elif d_change > 0.1:
            return f"Governor mode {mode} increased dissonance. Monitor for constitutional drift."
        else:
            return f"Stable operation in {mode} mode. Maintain current approach."
    
    def _generate_tags(self, situation: str, mode: str, outcome: str) -> List[str]:
        """Generate tags for a lesson."""
        tags = [situation, mode]
        if "reduced" in outcome:
            tags.append("success")
        elif "increased" in outcome:
            tags.append("warning")
        return tags
    
    def _update_index(self, lesson: LearnedLesson):
        """Update the lesson index."""
        for tag in lesson.tags:
            if tag not in self.lesson_index:
                self.lesson_index[tag] = []
            self.lesson_index[tag].append(len(self.lessons) - 1)
    
    def _store_in_human_memory(self, lesson: LearnedLesson):
        """Store a lesson in the human-memory skill."""
        try:
            content = (
                f"RAVANA lesson: {lesson.lesson} "
                f"(Ep{lesson.episode}, situation={lesson.situation}, "
                f"confidence={lesson.confidence:.0%})"
            )
            self._memorize(
                content=content,
                tags=",".join(["ravana", "lesson"] + lesson.tags),
                importance=lesson.confidence,
                emotional=0.3,
            )
        except Exception as e:
            print(f"  [Memory] Failed to store in human-memory: {e}")

    # ══════════════════════════════════════════════════════════════════════════════
    # PROCEDURAL MEMORY COMPOUNDING (from Hermes Agent v0.8.0 GEPA system)
    # ══════════════════════════════════════════════════════════════════════════════

    def compound_procedure(
        self,
        episode_data: dict,
        action_sequence: List[str],
        reward: float,
    ) -> Optional[ProceduralSequence]:
        """
        Create or update a reusable procedural sequence from successful episodes.
        
        Compounding: successful action sequences become reusable "skills" that 
        compound in value over time (like Hermes Agent's GEPA).
        
        Args:
            episode_data: Current episode state
            action_sequence: List of action descriptions taken
            reward: Outcome reward (positive = success)
        
        Returns:
            New or updated ProceduralSequence, or None if not yet compoundable
        """
        situation = self._classify_situation(episode_data)
        self._situation_counts[situation] += 1
        
        # Adaptive αhomeo: situations seen fewer times learn faster
        # TCH (Trend-Conserving Homeostasis) principle
        n = self._situation_counts[situation]
        self._alpha_homeo[situation] = min(2.0, 1.0 + 0.1 * n)
        
        # Only compound if we have enough evidence (≥3 reps) OR high reward
        MIN_REPS = 3
        if n < MIN_REPS and reward < 0.7:
            return None
        
        # Find existing procedure for this situation
        existing = self._find_procedure(situation)
        
        if existing:
            # Update existing procedure with new evidence
            self._update_procedure(existing, action_sequence, reward)
            return existing
        else:
            # Create new procedure
            self._procedure_counter += 1
            proc = ProceduralSequence(
                id=f"proc_{self._procedure_counter:04d}",
                name=f"{situation}_{action_sequence[0][:20]}" if action_sequence else f"proc_{self._procedure_counter}",
                steps=action_sequence,
                situation_pattern=situation,
                success_count=1 if reward > 0 else 0,
                failure_count=1 if reward <= 0 else 0,
                avg_reward=reward,
                last_used_episode=episode_data.get('episode', 0),
                gamma=1.0,
            )
            self.procedures[proc.id] = proc
            return proc
    
    def get_procedure(self, situation: str) -> Optional[ProceduralSequence]:
        """Get the best-performing procedure for a situation."""
        candidates = [p for p in self.procedures.values() 
                      if p.situation_pattern == situation and p.success_count > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.avg_reward * p.gamma)
    
    def get_procedure_summary(self) -> str:
        """Get human-readable summary of all procedures."""
        if not self.procedures:
            return "No procedures learned yet."
        lines = ["=== PROCEDURAL MEMORY ===", ""]
        for proc in sorted(self.procedures.values(), key=lambda p: p.avg_reward * p.gamma, reverse=True):
            lines.append(
                f"  {proc.id}: {proc.situation_pattern} "
                f"(success={proc.success_count}, fail={proc.failure_count}, "
                f"γ={proc.gamma:.2f}, reward={proc.avg_reward:.3f})"
            )
        return "\n".join(lines)
    
    def _classify_situation(self, state: dict) -> str:
        """Classify the current situation from state."""
        d = state.get('dissonance', 0.5)
        i = state.get('identity', 0.5)
        mode = state.get('mode', 'unknown')
        return f"D{int(d*100)}_I{int(i*100)}_{mode}"
    
    def _find_procedure(self, situation: str) -> Optional[ProceduralSequence]:
        """Find procedure matching situation pattern."""
        for proc in self.procedures.values():
            if proc.situation_pattern == situation:
                return proc
        return None
    
    def _update_procedure(self, proc: ProceduralSequence, 
                          new_steps: List[str], reward: float):
        """Update an existing procedure with new evidence."""
        # Extend sequence if steps are novel
        for step in new_steps:
            if step not in proc.steps:
                proc.steps.append(step)
        
        # Update statistics with reentry-aware weighting
        # FH-RL reentry gain: γ accumulates on revisits
        visit_key = f"{proc.situation_pattern}_last"
        prior_visits = len([v for v in self._state_visits.values() 
                           if any(visit_key in str(v) for v in _)])
        proc.gamma = min(3.0, 1.0 + 0.2 * prior_visits)
        
        # Incremental average with compound weighting
        if reward > 0:
            proc.success_count += 1
        else:
            proc.failure_count += 1
        
        alpha = 0.3 * proc.gamma
        proc.avg_reward = (1 - alpha) * proc.avg_reward + alpha * reward

    # ══════════════════════════════════════════════════════════════════════════════
    # REENTRY GAIN γ (from Meta-RL for Homeostatic Regulation research)
    # ══════════════════════════════════════════════════════════════════════════════
    
    def compute_reentry_gamma(self, state_signature: str, 
                               current_episode: int) -> float:
        """
        Compute reentry gain γ for a state signature.
        
        γ > 1.0 when re-entering a previously visited state, accelerating
        learning on familiar territory (like synaptic reentry gain).
        
        Args:
            state_signature: Unique hash of the cognitive state
            current_episode: Current episode number
        
        Returns:
            γ multiplier (1.0 = first visit, >1.0 = reentry)
        """
        visits = self._state_visits.get(state_signature, [])
        
        if not visits:
            # First visit — seed the tracker
            self._state_visits[state_signature].append((current_episode, 0.0))
            return 1.0
        
        # Compute γ based on recency and frequency
        last_ep, _ = visits[-1]
        episodes_since = current_episode - last_ep
        
        # γ increases with frequency and decreases with recency gap
        base_gamma = 1.0 + 0.15 * len(visits)
        recency_penalty = 0.02 * episodes_since
        gamma = max(1.0, base_gamma - recency_penalty)
        
        # Cap at 2.5
        gamma = min(2.5, gamma)
        
        return gamma
    
    def record_state_visit(self, state_signature: str, 
                           episode: int, reward: float):
        """Record a visit to a state for reentry tracking."""
        self._state_visits[state_signature].append((episode, reward))
        # Keep only last 10 visits per signature
        if len(self._state_visits[state_signature]) > 10:
            self._state_visits[state_signature] = self._state_visits[state_signature][-10:]

    # ══════════════════════════════════════════════════════════════════════════════
    # STRUCTURAL PLASTICITY TRIGGERS (Stress-Reg inspired)
    # ══════════════════════════════════════════════════════════════════════════════
    
    def should_trigger_plasticity(self, dissonance: float, 
                                   identity: float,
                                   clamp_rate: float) -> bool:
        """
        Determine if structural plasticity should be triggered.
        
        Structural plasticity = adding new memory traces rather than 
        modifying existing ones. Triggered by stress (high dissonance 
        + identity threat).
        
        Args:
            dissonance: Current dissonance level (0-1)
            identity: Current identity strength (0-1)
            clamp_rate: Current clamp rate (fraction of clamped steps)
        
        Returns:
            True if new memory traces should be created
        """
        stress = dissonance * (1.0 - identity) + clamp_rate * 0.5
        self._plasticity_events.append({
            'stress': stress,
            'dissonance': dissonance,
            'identity': identity,
            'clamp_rate': clamp_rate,
        })
        
        # Keep last 100 plasticity events
        if len(self._plasticity_events) > 100:
            self._plasticity_events = self._plasticity_events[-100:]
        
        return stress > self._stress_threshold
    
    def get_plasticity_report(self) -> str:
        """Get a report on structural plasticity activation."""
        if not self._plasticity_events:
            return "No plasticity events recorded."
        
        recent = self._plasticity_events[-20:]
        avg_stress = sum(e['stress'] for e in recent) / len(recent)
        triggers = sum(1 for e in recent if e['stress'] > self._stress_threshold)
        
        return (
            f"Plasticity report: {triggers}/{len(recent)} triggers in last 20 events, "
            f"avg stress={avg_stress:.3f}, threshold={self._stress_threshold}"
        )
    
    # ══════════════════════════════════════════════════════════════════════════════
    # ADAPTIVE αhomeo (Trend-Conserving Homeostasis)
    # ══════════════════════════════════════════════════════════════════════════════
    
    def get_adaptive_alpha(self, situation: str) -> float:
        """
        Get adaptive learning rate αhomeo for a situation.
        
        Situations seen more often get lower α (already well-learned).
        Novel situations get higher α (need rapid updating).
        
        Args:
            situation: The situation tag
        
        Returns:
            αhomeo multiplier (typically 0.5-2.0)
        """
        return self._alpha_homeo.get(situation, 1.0)
    
    def get_alphahomeo_summary(self) -> str:
        """Get summary of αhomeo values per situation."""
        if not self._alpha_homeo:
            return "No αhomeo data yet."
        lines = ["=== ADAPTIVE αhomeo ===", ""]
        for sit, alpha in sorted(self._alpha_homeo.items(), 
                                  key=lambda x: x[1], reverse=True):
            n = self._situation_counts.get(sit, 0)
            lines.append(f"  {sit}: α={alpha:.2f} (n={n})")
        return "\n".join(lines)


if __name__ == "__main__":
    learner = MemoryLearner()
    
    print("=== Memory Learner Test ===\n")
    
    # Simulate lessons
    test_episodes = [
        {
            'episode': 1,
            'pre_dissonance': 0.7,
            'post_dissonance': 0.6,
            'pre_identity': 0.5,
            'post_identity': 0.55,
            'mode': 'resolution',
            'wisdom': 0.1,
            'resolution': {'full_resolution': True},
        },
        {
            'episode': 2,
            'pre_dissonance': 0.6,
            'post_dissonance': 0.65,
            'pre_identity': 0.55,
            'post_identity': 0.55,
            'mode': 'resolution',
            'wisdom': 0.05,
            'resolution': {'full_resolution': False},
        },
        {
            'episode': 3,
            'pre_dissonance': 0.2,
            'post_dissonance': 0.22,
            'pre_identity': 0.6,
            'post_identity': 0.6,
            'mode': 'exploration',
            'wisdom': 0.0,
            'resolution': {'full_resolution': False},
        },
    ]
    
    for ep in test_episodes:
        lesson = learner.learn_from_episode(ep)
        print(f"Learned: {lesson.lesson[:80]}...")
    
    print()
    print(learner.get_lesson_summary())
    print()
    
    # Test recommendation
    state = {'dissonance': 0.72, 'governor_mode': 'resolution'}
    rec = learner.get_recommendation(state)
    print(f"Recommendation for D=0.72: {rec}")