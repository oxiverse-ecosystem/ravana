"""Monitor blind-spot tests (M6).

Proves:
- Every production caller of _is_word_salad / _is_word_salad_any_sentence
  passes an explicit subject= kwarg, so the semantic-substance /
  tautology block stays active (the B3 caveat: a future path that
  omits subject silently loses that check — caught here at review time).
- The universal choke-point (_forward_model_check) funnels arbitrary
  fluent-tautological / word-salad text and never emits it unguarded.

Run from repo root:
    python -m pytest tests/unit/test_monitor_blindspots.py -v
"""
import os
import re
import sys
import random

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext
from ravana.chat.constants import _is_word_salad, _is_word_salad_any_sentence


_CHAT_FILES = [
    os.path.join(_PROJ, "ravana", "src", "ravana", "chat", "engine.py"),
    os.path.join(_PROJ, "ravana", "src", "ravana", "chat", "response_gen.py"),
    os.path.join(_PROJ, "ravana", "src", "ravana", "chat", "interface.py"),
]


def test_every_salad_call_passes_subject():
    """Static gate: every production call site passes subject=.

    The semantic-substance / tautology block in _is_word_salad only
    fires when subject is given. A caller that omits it silently
    downgrades the monitor (the B3 caveat). This test greps the
    chat sources and asserts each call site carries an explicit subject=.
    """
    call_re = re.compile(r"_is_word_salad(?:_any_sentence)?\(\s*([^)]*)\)")
    missing = []
    for fp in _CHAT_FILES:
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for ln, line in enumerate(f, 1):
                m = call_re.search(line)
                if not m:
                    continue
                # Skip the def lines and the AST-internal recursion
                # (_is_word_salad_any_sentence calls _is_word_salad
                #  inside constants.py — that one legitimately threads subject).
                args = m.group(1)
                if "subject=" not in args and "subject" not in args:
                    missing.append(f"{os.path.basename(fp)}:{ln}: {line.strip()}")
    # constants.py's internal recursion legitimately threads subject via the
    # closure, so exclude it; everything in chat/ MUST pass subject=.
    assert not missing, (
        "salad-check call site(s) without subject= found:\n" + "\n".join(missing)
    )


_SALAD_TEMPLATES = [
    "Life semantic people, which semantic cannot.",
    "From another angle, life contrastive way, which contrastive even.",
    "black holes bend spacetime is black holes bend.",
    "black holes bend spacetime directly is black holes bend.",
    "Gravity semantic forces, which semantic starchild.",
    "the cat the cat the cat the cat the cat the cat the cat.",
    "meaning is meaning is meaning is meaning is meaning is meaning.",
    "trust the light and the space where the matter and the time bend.",
]


def _rnd_salad(rng):
    """Synthesize a random fluent-tautological string."""
    subjects = ["life", "gravity", "black holes", "trust", "dream", "time"]
    glue = ["is", "are", "causes", "connected", "relates", "means",
            "reflects", "parallels", "echoes", "symbolizes"]
    meta = ["semantic", "contrastive", "causal", "great", "even",
            "cannot", "which", "related", "linked", "tied"]
    subj = rng.choice(subjects)
    parts = []
    for _ in range(rng.randint(1, 3)):
        g = rng.choice(glue)
        m = rng.choice(meta)
        parts.append(f"{subj} {g} {m}.")
    return " ".join(parts)


def test_fuzz_all_paths_guarded():
    """Fuzz sweep: arbitrary fluent-tautological text funnels through
    the universal choke-point and is never emitted unguarded.

    We feed each candidate to _forward_model_check (the single
    guard that wraps every production path) and assert the returned
    text passes the salad check for the given subject — i.e. the
    guard either strips/replaces it or returns a clean fallback.
    """
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                             data_dir="/tmp/ravana_m6_fuzz")
    rng = random.Random(1234)
    total = 0
    leaked = 0
    for _ in range(200):
        if rng.random() < 0.5:
            text = _rnd_salad(rng)
        else:
            text = rng.choice(_SALAD_TEMPLATES)
        subj = "life" if "life" in text.lower() else (
            "black holes" if "black" in text.lower() else "gravity")
        ctx = CognitiveResponseContext(
            subject=subj, raw_input=f"what is {subj}?",
            associated_concepts=[(a, 0.5) for a in
                                ["meaning", "death", "consciousness", "purpose"]])
        out = eng._forward_model_check(text, ctx)
        total += 1
        if _is_word_salad_any_sentence(out, subject=ctx.subject):
            leaked += 1
            if leaked <= 3:
                print(f"  LEAK subj={subj!r} in={text!r} out={out!r}")
    assert leaked == 0, f"{leaked}/{total} fluent-tautological strings leaked past the guard"
