"""
RAVANA v2 — COGNITIVE STATE
Unified state container with governor-gated updates.

PRINCIPLE: State is immutable except through official channels.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class CognitivePhase(Enum):
    """Cognitive processing phases."""
    PERCEPTION = "perception"
    RESOLUTION = "resolution"
    INTEGRATION = "integration"


@dataclass
class CognitiveState:
    """
    Immutable cognitive state container.
    
    All modifications go through StateManager.update() which applies governor.
    """
    # Core metrics
    dissonance: float = 0.5
    identity: float = 0.5
    dissonance_ema: float = 0.5  # Smoothed dissonance for regulation
    
    # Episode tracking
    episode: int = 0
    cycle: int = 0
    
    # Learning accumulators
    accumulated_wisdom: float = 0.0
    resolution_streak: int = 0
    
    # Extended accumulators
    accumulated_meaning: float = 0.0
    sleep_cycles_completed: int = 0
    
    # Debug
    last_update_reason: str = "initial"
    constraint_activated: bool = False
    processing_route: str = "system1_fast"  # system1_fast or system2_slow
    
    def snapshot(self) -> Dict[str, float]:
        """Return serializable snapshot."""
        return {
            "dissonance": self.dissonance,
            "dissonance_ema": self.dissonance_ema,
            "identity": self.identity,
            "episode": self.episode,
            "cycle": self.cycle,
            "wisdom": self.accumulated_wisdom,
            "meaning": self.accumulated_meaning,
            "sleep_cycles": self.sleep_cycles_completed,
            "processing_route": self.processing_route,
        }


class StateManager:
    """
    Central state manager with governor integration.
    
    ALL state modifications flow through here.
    Includes integrated emotion, sleep, dual-process, and meaning engines.
    """
    
    def __init__(
        self,
        governor,
        resolution_engine,
        identity_engine,
        smoothing_alpha: float = 0.2,
        emotion_engine: Optional[Any] = None,
        sleep_engine: Optional[Any] = None,
        dual_process: Optional[Any] = None,
        meaning_engine: Optional[Any] = None,
        global_workspace: Optional[Any] = None,
        human_memory: Optional[Any] = None,
    ):
        from .memory import RavanaMemorySystem
        self.state = CognitiveState()
        self.governor = governor
        self.resolution = resolution_engine
        self.identity = identity_engine
        self.memory = RavanaMemorySystem()
        self.smoothing_alpha = smoothing_alpha

        # Integrated cognitive engines (optional)
        self.emotion = emotion_engine
        self.sleep = sleep_engine
        self.dual_process = dual_process
        self.meaning = meaning_engine
        self.gw = global_workspace
        self.human_memory = human_memory

        # History for analysis
        self.history: list = []
        
    def step(
        self,
        correctness: bool,
        difficulty: float = 0.5,
        debug: bool = False,
        novelty: float = 0.0,
        stakes: float = 0.0,
        effort: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute one cognitive step with full governor regulation.
        
        This is the ONLY way state changes. No exceptions.
        
        Extended with integrated emotion, dual-process, meaning, and sleep.
        """
        # Capture pre-update state
        pre_d = self.state.dissonance
        pre_i = self.state.identity
        
        # === DUAL-PROCESS: Select processing route ===
        route = "system1_fast"
        if self.dual_process is not None:
            confidence = 1.0 - pre_d  # Higher dissonance = lower confidence
            route_decision = self.dual_process.decide_route(
                confidence=confidence,
                novelty=novelty,
                stakes=stakes,
            )
            route = route_decision.route.value
        
        # === VAD EMOTION: Update emotional state ===
        if self.emotion is not None:
            # Map correctness to stimulus valence/arousal
            stim_valence = 0.3 if correctness else -0.5
            stim_arousal = 0.2 + difficulty * 0.5  # Harder = more arousing
            stim_dominance = 0.6 if correctness else 0.3
            
            self.emotion.update(
                stimulus_valence=stim_valence,
                stimulus_arousal=stim_arousal,
                stimulus_dominance=stim_dominance,
                uncertainty=1.0 - confidence if self.dual_process is None else 1.0 - route_decision.confidence,
            )
        
        # FIX: Sync identity engine to current state at START of step.
        self.identity.state.strength = pre_i
        self.identity.last_delta = 0.0
        
        # 1. ESTIMATE: Predict desired delta from outcome
        desired_d_delta = -0.1 if correctness else 0.15
        
        # 2. IDENTITY: First pass - use placeholder for governor's future delta
        saved_last_delta = self.identity.last_delta
        
        est_res_success = (desired_d_delta < 0 and correctness)
        desired_identity = self.identity.compute_update(
            resolution_delta=abs(desired_d_delta),
            resolution_success=est_res_success,
            regulated_identity_delta=0.0,
            current_dissonance=pre_d,
            resolution_streak=self.state.resolution_streak,
            correctness=correctness
        )
        identity_delta = desired_identity - pre_i
        
        self.identity.last_delta = saved_last_delta
        
        # 3. GOVERNOR: Regulate all changes
        from .governor import CognitiveSignals
        
        # Emotion-influenced exploration drive
        emotion_drive = 0.0
        if self.emotion is not None:
            vad = self.emotion.state
            emotion_drive = abs(vad.valence) * 0.3 + vad.arousal * 0.2
        
        signals = CognitiveSignals(
            dissonance_delta=desired_d_delta,
            identity_delta=identity_delta,
            exploration_drive=emotion_drive,
            resolution_potential=0.1 if correctness else 0.0,
            source="state_step"
        )
        
        regulated = self.governor.regulate(
            current_dissonance=pre_d,
            current_identity=pre_i,
            signals=signals,
            episode=self.state.episode
        )
        
        # 4. APPLY: Governor-approved changes
        new_dissonance = np.clip(
            pre_d + regulated.dissonance_delta,
            self.governor.config.min_dissonance,
            self.governor.config.max_dissonance
        )
        
        new_ema = (1 - self.smoothing_alpha) * self.state.dissonance_ema + self.smoothing_alpha * new_dissonance
        
        # Clear last_delta for clean identity update
        self.identity.last_delta = 0.0
        
        regulated_identity = self.identity.compute_update(
            resolution_delta=abs(regulated.dissonance_delta),
            resolution_success=(regulated.dissonance_delta < 0 and correctness),
            regulated_identity_delta=regulated.identity_delta,
            current_dissonance=pre_d,
            resolution_streak=self.state.resolution_streak,
            correctness=correctness
        )
        new_identity = np.clip(
            regulated_identity,
            self.governor.config.min_identity,
            self.governor.config.max_identity
        )
        
        self.identity.apply_update(new_identity)
        
        # 5. RESOLUTION: Compute what ACTUALLY happened
        resolution_result = self.resolution.compute(
            episode=self.state.episode,
            prev_dissonance=pre_d,
            current_dissonance=new_dissonance,
            correctness=correctness,
            difficulty=difficulty,
            source="episode_step"
        )
        
        # 6. WISDOM: Check for generation
        wisdom_generated = resolution_result["wisdom_generated"]
        
        # 7. MEANING: Compute meaning from this step
        meaning_generated = 0.0
        if self.meaning is not None:
            predictive_gain = 1.0 if correctness else 0.0  # Simplified
            meaning_record = self.meaning.compute_meaning(
                episode=self.state.episode,
                pre_dissonance=pre_d,
                post_dissonance=new_dissonance,
                pre_identity=pre_i,
                post_identity=new_identity,
                predictive_gain=predictive_gain,
                effort=effort,
            )
            meaning_generated = meaning_record.effective_meaning
        
        # 8. GLOBAL WORKSPACE: Submit bids and compete
        gw_broadcast = None
        gw_context = None
        if self.gw is not None:
            # Emotion bids
            if self.emotion is not None:
                emotion_bid = self.emotion.compute_gw_bid()
                self.gw.submit_bid(
                    source="emotion",
                    payload=self.emotion.state.snapshot(),
                    urgency=emotion_bid,
                    valence=self.emotion.state.valence,
                    episode=self.state.episode,
                )

            # Meaning bids (if meaningful event)
            if self.meaning is not None and abs(meaning_generated) > 0.1:
                self.gw.submit_bid(
                    source="meaning",
                    payload={"meaning": meaning_generated},
                    urgency=min(1.0, abs(meaning_generated) * 2.0),
                    valence=0.1 if meaning_generated > 0 else -0.1,
                    episode=self.state.episode,
                )

            # Competition: select winning bid
            gw_broadcast = self.gw.compete()
            gw_context = self.gw.get_context_vector()

            # Accumulate pressure from competition intensity
            if gw_broadcast is not None:
                self.gw.accumulate_pressure(0.05)

        # 9. SLEEP: Accumulate pressure and check for consolidation
        sleep_triggered = False
        # Accumulate GW pressure to sleep engine
        if self.gw is not None and self.sleep is not None and self.gw.should_sleep():
            self.sleep.accumulate_pressure(self.gw.get_pressure() * 0.1)

        if self.sleep is not None:
            # Pressure from dissonance + prediction error
            pressure_delta = abs(new_dissonance - pre_d) * 0.5
            if not correctness:
                pressure_delta += 0.1
            self.sleep.accumulate_pressure(pressure_delta)
            
            if self.sleep.should_sleep():
                sleep_record = self.sleep.execute_sleep_cycle(
                    episode=self.state.episode,
                    state_snapshot=self.state.snapshot(),
                    episodic_memories=self.memory.episodic.traces if hasattr(self.memory, 'episodic') else None,
                    emotion_engine=self.emotion,
                    coherence_fn=lambda s: 1.0 - s.get("dissonance", 0.5),
                )
                sleep_triggered = True
        
        # 9. UPDATE STATE
        self.state = CognitiveState(
            dissonance=new_dissonance,
            identity=new_identity,
            dissonance_ema=new_ema,
            episode=self.state.episode + 1,
            cycle=self.state.cycle + 1,
            accumulated_wisdom=self.state.accumulated_wisdom + wisdom_generated,
            accumulated_meaning=self.state.accumulated_meaning + meaning_generated,
            sleep_cycles_completed=self.state.sleep_cycles_completed + (1 if sleep_triggered else 0),
            resolution_streak=resolution_result["streak"],
            last_update_reason=regulated.reason,
            constraint_activated=regulated.capped or regulated.dampened,
            processing_route=route,
        )
        
        # Track history
        step_record = {
            "episode": self.state.episode,
            "pre_dissonance": pre_d,
            "post_dissonance": new_dissonance,
            "pre_identity": pre_i,
            "post_identity": new_identity,
            "resolution": resolution_result,
            "mode": regulated.mode.value,
            "wisdom": wisdom_generated,
            "meaning": meaning_generated,
            "processing_route": route,
            "sleep_triggered": sleep_triggered,
            "reason": regulated.reason,
        }
        
        # Add VAD snapshot if available
        if self.emotion is not None:
            step_record["vad"] = self.emotion.state.snapshot()
            step_record["emotional_label"] = self.emotion.get_emotional_label()

        # Add GW broadcast if available
        if gw_broadcast is not None:
            step_record["gw_broadcast"] = {
                "source": gw_broadcast.source,
                "urgency": gw_broadcast.urgency,
                "valence": gw_broadcast.valence,
            }
        if gw_context is not None:
            step_record["gw_context"] = gw_context.tolist()
        
        self.history.append(step_record)
        
        # 10. MEMORY: Integrate new data
        memory_kwargs = {
            "workspace_context": gw_context,
            "workspace_broadcast": (
                {
                    "source": gw_broadcast.source,
                    "payload": gw_broadcast.payload,
                    "urgency": gw_broadcast.urgency,
                    "valence": gw_broadcast.valence,
                    "episode": gw_broadcast.episode,
                }
                if gw_broadcast is not None else None
            ),
        }
        self.memory.process_step(
            episode_data=step_record,
            state_snapshot=self.state.snapshot(),
            **memory_kwargs,
        )

        # 10b. HUMAN MEMORY: Persistent episodic storage with decay
        if self.human_memory is not None:
            self.human_memory.process_step(
                episode_data=step_record,
                state_snapshot=self.state.snapshot(),
                **memory_kwargs,
            )

        # Degradation is natural: Ebbinghaus decay runs every cycle in
        # process_step(). Without sleep consolidation, decay accumulates.
        # No separate tracking needed — the memory system degrades on its own.

        if debug:
            self._log_step(step_record)
        
        return step_record
    
    def _log_step(self, record: Dict[str, Any]):
        """Debug logging."""
        route_info = f" R:{record['processing_route'][:7]}"
        emotion_info = f" V:{record.get('emotional_label', 'N/A'):.6s}" if 'emotional_label' in record else ""
        sleep_info = " ZzZ" if record.get('sleep_triggered') else ""
        print(f"  [EP{record['episode']:04d}] "
              f"D:{record['pre_dissonance']:.3f}→{record['post_dissonance']:.3f} "
              f"I:{record['pre_identity']:.3f}→{record['post_identity']:.3f} "
              f"M:{record['mode'][:4]} "
              f"W:{record['wisdom']:.3f} "
              f"M:{record['meaning']:.3f}"
              f"{route_info}{emotion_info}{sleep_info} "
              f"Res:{'✓' if record['resolution']['full_resolution'] else '·'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Full system status."""
        status = {
            "state": self.state.snapshot(),
            "governor": self.governor.get_status(),
            "identity": self.identity.get_status(),
            "resolution": self.resolution.get_memory_status(),
            "total_steps": len(self.history),
        }
        if self.emotion is not None:
            status["emotion"] = self.emotion.get_status()
        if self.sleep is not None:
            status["sleep"] = self.sleep.get_status()
        if self.dual_process is not None:
            status["dual_process"] = self.dual_process.get_status()
        if self.meaning is not None:
            status["meaning"] = self.meaning.get_status()
        if self.gw is not None:
            status["global_workspace"] = self.gw.get_status()
        if self.human_memory is not None:
            status["human_memory"] = self.human_memory.get_status()
        return status
