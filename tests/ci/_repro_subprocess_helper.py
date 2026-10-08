#!/usr/bin/env python3
"""Subprocess helper for cross-process determinism testing.

Initializes a CognitiveChatEngine with a fixed seed, runs a probe sequence,
and prints the reproducibility fingerprint as JSON on the last line of stdout.

Usage: python _repro_subprocess_helper.py <seed>
"""
import json
import os
import sys

# Set deterministic env BEFORE any imports
os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

# Add repo source roots to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (
    os.path.join(_REPO_ROOT, "ravana", "src"),
    os.path.join(_REPO_ROOT, "ravana_ml", "src"),
    os.path.join(_REPO_ROOT, "ravana-v2", "src"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.reproducibility import (
    SpikeLog,
    full_reproducibility_fingerprint,
)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    eng = CognitiveChatEngine(dim=64, seed=seed, baby_mode=True,
                              user_suffix="repro_helper")
    spike_log = SpikeLog()

    # Run a deterministic probe sequence
    probe_turns = [
        "i like open source software",
        "my cat is named milo",
        "i live in berlin",
        "privacy matters to me",
        "i think board games are a great way to bond with friends",
        "my dog had surgery last month",
        "i started to hate mondays",
        "my sister is a marine biologist",
        "i don't really have a solid grasp on privacy so far",
        "my favorite programming language is python",
    ]

    for turn in probe_turns:
        eng.process_turn(turn)
        spike_log.record(eng.turn_count, "probe", {
            "turn": eng.turn_count,
            "input": turn,
        })

    # Compute fingerprints
    fp = full_reproducibility_fingerprint(eng, spike_log)

    # Print JSON on the last line
    print(json.dumps(fp))


if __name__ == "__main__":
    main()
