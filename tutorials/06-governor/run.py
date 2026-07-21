"""Tutorial 06: Governor module demo — instantiate GRACE cognitive modules directly.

Layer 3 of 3 added on top of the mini-system. Shows how the chat engine's
cognitive regulation works internally (no chat engine needed).

Usage:
    python tutorials/06-governor/run.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ravana-v2", "src"))

from ravana_grace.core.emotion import VADEmotionEngine, VADConfig, VADState
from ravana_grace.core.identity import IdentityEngine, IdentityState
from ravana_grace.core.meaning import MeaningEngine, MeaningConfig
from ravana_grace.core.dual_process import DualProcessController, DualProcessConfig
from ravana_grace.core.global_workspace import GlobalWorkspace, GWConfig
from ravana_grace.core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode


def main() -> None:
    print("=== GRACE Governor Modules Demo ===\n")

    # 1. VAD Emotion Engine — 3D affective state
    # Note: update() takes stimulus_valence, stimulus_arousal, stimulus_dominance
    emotion = VADEmotionEngine(VADConfig())
    emotion.update(stimulus_valence=0.6, stimulus_arousal=0.3, stimulus_dominance=0.7)
    s = emotion.state
    print(f"  VAD Emotion:      V={s.valence:.2f}  A={s.arousal:.2f}  D={s.dominance:.2f}")

    # 2. Identity Engine — self-concept stabilization
    # Note: compute_update() returns new strength, use apply_update() to commit
    identity = IdentityEngine(initial_strength=0.25, momentum_factor=0.3)
    new_strength = identity.compute_update(
        resolution_delta=0.1, resolution_success=True,
        regulated_identity_delta=0.05, current_dissonance=0.3,
        resolution_streak=2, correctness=True
    )
    identity.apply_update(new_strength)
    print(f"  Identity:         strength={identity.state.strength:.2f}")

    # 3. Meaning Engine — intrinsic motivation
    # Note: compute_meaning() takes pre/post dissonance and identity
    meaning = MeaningEngine(MeaningConfig())
    mr = meaning.compute_meaning(
        episode=1,
        pre_dissonance=0.7, post_dissonance=0.3,
        pre_identity=0.5, post_identity=0.6,
        predictive_gain=0.2, effort=0.5
    )
    print(f"  Meaning:          raw={mr.raw_meaning:.3f}  effective={mr.effective_meaning:.3f}")

    # 4. Dual-Process Controller — System 1 / System 2 routing
    dual = DualProcessController(DualProcessConfig())
    route = dual.decide_route(confidence=0.3, novelty=0.6)
    print(f"  Dual-Process:     route={route.route.value}")

    # 5. Global Workspace — conscious broadcast bottleneck
    gw = GlobalWorkspace(GWConfig(capacity=7))
    gw.submit_bid(source="belief", payload={"concept": "trust"}, urgency=0.7, valence=0.5)
    gw.submit_bid(source="emotion", payload={"feeling": "honesty"}, urgency=0.5, valence=0.6)
    gw.submit_bid(source="belief", payload={"concept": "betrayal"}, urgency=0.3, valence=-0.4)
    winner = gw.compete()
    recent = gw.get_recent(3)
    print(f"  Global Workspace: winner={winner.source if winner else 'none'}, "
          f"buffer_size={len(recent)}")

    # 6. MetaCognition — monitoring own reasoning
    meta = MetaCognition(MetaCognitiveConfig())
    modes = [mode.value for mode in list(EpistemicMode)[:5]]
    print(f"  MetaCognition:    epistemic_modes={modes}")

    print("\n  [OK] All 6 GRACE modules instantiated successfully")
    print("\n  Note: The CognitiveChatEngine uses all these internally.")


if __name__ == "__main__":
    main()
