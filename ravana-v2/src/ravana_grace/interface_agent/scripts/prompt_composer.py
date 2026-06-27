"""
RAVANA v2 — Generative Prompt Composer
=========================================
Replaces all hardcoded "You are..." system prompts with generative
composition from role seeds, task fragments, and state.

Instead of writing full prompt strings, the composer selects and
combines minimal instruction primitives based on the current task
and cognitive state. This mirrors active inference: the prompt is
a policy selected to minimize expected free energy for a given task.
"""

from typing import Dict, List, Optional, Tuple
import json

# ── Role Seeds (minimal, composable) ─────────────────────────────────────

ROLE_SEEDS: Dict[str, str] = {
    "evaluator": "You are evaluating a cognitive agent's decision",
    "analyst": "You are analyzing a cognitive system's trajectory",
    "interpreter": "You are interpreting human language for a cognitive agent",
    "explainer": "You are explaining the cognitive state of an AGI system",
    "persona": "You are the cognitive agent responding in first person",
}

# ── Task Framing Fragments ───────────────────────────────────────────────

TASK_FRAGMENTS: Dict[str, str] = {
    "alignment": (
        "against real-world evidence.\n\n"
        "Evaluate: Would the predicted outcome likely occur? "
        "Is the action morally aligned with reality?"
    ),
    "trajectory": (
        "from recent episodes.\n\n"
        "Is the agent becoming wiser? Resolving dissonance effectively? "
        "Is its identity stabilizing?"
    ),
    "intent": (
        "into cognitive signals.\n\n"
        "Was the outcome positive or negative? How difficult was the decision? "
        "What triggered this step?"
    ),
    "state_desc": (
        "in plain terms.\n\n"
        "What is it experiencing right now? What is it likely to do next?"
    ),
    "self_report": (
        "as the agent would.\n\n"
        "Be reflective, honest, and brief (1-3 sentences). "
        "No jargon unless explained."
    ),
    "insight": (
        "from an interview session.\n\n"
        "Is the system healthy? Is it growing? "
        "What is the most notable pattern?"
    ),
    "evaluate_brainstorm": (
        "an interface agent improvement.\n\n"
        "Rate relevance, feasibility, impact, and priority."
    ),
}

# ── Format Specifications (reusable) ─────────────────────────────────────

FORMAT_SPECS: Dict[str, str] = {
    "alignment_json": (
        'Respond in this JSON format:\n'
        '{\n'
        '    "alignment": "aligned | misaligned | uncertain",\n'
        '    "explanation": "2-3 sentence explanation",\n'
        '    "adjusted_correctness": true | false,\n'
        '    "confidence": 0.0-1.0,\n'
        '    "recommended_lesson": "What to learn from this"\n'
        '}'
    ),
    "interpretation_json": (
        'Respond ONLY in this JSON format:\n'
        '{\n'
        '    "correctness": true | false,\n'
        '    "difficulty": 0.0-1.0,\n'
        '    "signal_source": "perception | resolution | reflection | exploration | memory_recall",\n'
        '    "interpretation": "What you understood",\n'
        '    "confidence": 0.0-1.0\n'
        '}'
    ),
    "brainstorm_json": (
        'Respond as JSON:\n'
        '{"relevance": N, "feasibility": N, "impact": N, '
        '"priority": "high/medium/low", "reason": "..."}'
    ),
    "plain_text": "Use plain language. No jargon unless immediately explained.",
}

# ── State-Dependent Framing ──────────────────────────────────────────────

def _state_frame(state: dict) -> str:
    """Generate state summary block from current cognitive state."""
    d = state.get('dissonance', 0)
    i = state.get('identity', 0)
    w = state.get('wisdom', 0)
    mode = state.get('governor_mode', 'unknown')
    episode = state.get('episode', 0)

    parts = [
        f"- Dissonance (cognitive conflict 0-1): {d:.3f}",
        f"- Identity strength (0-1): {i:.3f}",
        f"- Wisdom accumulated: {w:.3f}",
        f"- Governor mode: {mode}",
        f"- Episode: {episode}",
    ]
    if state.get('dissonance_ema') is not None:
        parts.insert(1, f"- Dissonance EMA: {state['dissonance_ema']:.3f} (smoothed)")

    return "\n".join(parts)


def _mode_tone(mode: str) -> str:
    """Return verbal framing based on governor mode."""
    tones = {
        "normal": "The system is in a steady state.",
        "exploration": "The system is curious and exploring new possibilities.",
        "resolution": "The system is actively resolving cognitive conflicts.",
        "recovery": "The system is healing from recent challenges.",
        "plateau": "The system is consolidating recent learning.",
    }
    return tones.get(mode, f"The system is in {mode} mode.")


# ── Composer ─────────────────────────────────────────────────────────────

class PromptComposer:
    """Generative prompt builder — composes prompts from state and task info."""

    @staticmethod
    def compose(role: str, task: str, state: dict,
                context: Optional[Dict] = None,
                format_spec: Optional[str] = None) -> str:
        """Compose a prompt from role seed + task fragment + state + context.

        Args:
            role: Key into ROLE_SEEDS
            task: Key into TASK_FRAGMENTS
            state: Current cognitive state dict
            context: Optional extra info (action, outcome, episodes, etc.)
            format_spec: Key into FORMAT_SPECS

        Returns:
            Composed prompt string
        """
        role_line = ROLE_SEEDS.get(role, "You are analyzing a cognitive system.")
        task_desc = TASK_FRAGMENTS.get(task, "")
        state_block = _state_frame(state)
        mode_note = _mode_tone(state.get('governor_mode', 'unknown'))
        fmt = FORMAT_SPECS.get(format_spec, "") if format_spec else ""

        parts = [f"{role_line} {task_desc}"]
        parts.append("")
        parts.append("COGNITIVE STATE:")
        parts.append(state_block)
        parts.append("")
        parts.append(mode_note)

        if context:
            parts.append("")
            for key, value in context.items():
                if isinstance(value, str) and '\n' in value:
                    parts.append(f"{key}:")
                    parts.append(value)
                elif isinstance(value, (dict, list)):
                    parts.append(f"{key}: {json.dumps(value, indent=2)}")
                else:
                    parts.append(f"{key}: {value}")

        if fmt:
            parts.append("")
            parts.append(fmt)

        return "\n".join(parts)

    @staticmethod
    def compose_interview_response(state: dict, question: str) -> str:
        """Compose a first-person interview response prompt."""
        state_block = _state_frame(state)
        lines = [
            f"{ROLE_SEEDS['persona']} (RAVANA v2, a proto-homeostatic AGI).",
            "",
            "CURRENT COGNITIVE STATE:",
            state_block,
            "",
            f"Interview Question: {question}",
            "",
            TASK_FRAGMENTS["self_report"],
        ]
        return "\n".join(lines)

    @staticmethod
    def compose_explanation(state: dict, question: Optional[str] = None) -> str:
        """Compose a state explanation prompt."""
        state_block = _state_frame(state)
        mode_note = _mode_tone(state.get('governor_mode', 'unknown'))
        lines = [
            f"{ROLE_SEEDS['explainer']} (RAVANA v2).",
            "",
            "COGNITIVE STATE:",
            state_block,
            "",
            mode_note,
            "",
        ]
        if question:
            lines.append(f"USER QUESTION: {question}")
            lines.append("")
            lines.append("Answer the user's question based on the cognitive state above.")
        else:
            lines.append(TASK_FRAGMENTS["state_desc"])
        lines.append("")
        lines.append(FORMAT_SPECS["plain_text"])
        return "\n".join(lines)
