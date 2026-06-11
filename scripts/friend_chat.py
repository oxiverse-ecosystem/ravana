#!/usr/bin/env python3
"""
RAVANA Friend Chat - 40 flowing queries, real conversation, not batch.
"""

import subprocess
import sys
import os
import time

# -- A Real Friend Conversation (each turn flows from the last) ---------------
QUERIES = [
    # ── 1-5: Greetings & catching up ──
    "hey whats up",
    "im good just thinking about stuff",
    "what have you been learning lately",
    "thats cool tell me more about that",
    "you know i was thinking about trust earlier",

    # ── 6-10: Sharing a personal thought ──
    "why do people find it so hard to trust each other",
    "yeah that makes sense ive been hurt before too",
    "do you think its possible to fully trust someone again",
    "thats a good point  i never thought of it like that",
    "what about honesty  is it always the best policy",

    # ── 11-15: Going deeper on the theme ──
    "but sometimes being honest can hurt people right",
    "so where do you draw the line between honesty and kindness",
    "i guess it depends on the situation",
    "speaking of difficult things  what do you think about change",
    "do you think people can really change who they are",

    # ── 16-20: Creative/imaginative shift ──
    "thats deep  let me ask you something fun",
    "if you could invent anything what would it be",
    "a time machine  how would you even build something like that",
    "imagine if you could travel to any point in time",
    "id probably go see the dinosaurs  what about you",

    # ── 21-25: Exploring the time travel idea ──
    "but if you changed the past wouldnt that mess up the future",
    "thats a good way to think about it",
    "ok what if instead of time travel we could read minds",
    "would that be a good thing or a bad thing for society",
    "i think it would destroy trust completely",

    # ── 26-30: Emotional / personal ──
    "what do you think is the most important thing in life",
    "why do you think people spend so much time chasing money then",
    "i guess society kind of forces us to",
    "do you ever feel lonely",
    "even though you have all this knowledge in your head",

    # ── 31-35: Future / hope ──
    "what gives you hope about the future",
    "thats beautiful  i feel the same way",
    "if you could change one thing about the world what would it be",
    "how would you even start solving a problem that big",
    "one step at a time  i like that",

    # ── 36-40: Wrapping up ──
    "this conversation was really nice",
    "i learned a lot talking to you today",
    "youre pretty cool for an ai",
    "lets talk again soon okay",
    "goodbye friend",
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chat_script = os.path.join(script_dir, "ravana_chat.py")

    chat_string = "|".join(QUERIES)

    cmd = [
        sys.executable, chat_script,
        "--reset",
        "--chat", chat_string,
        "--strategy",
    ]

    print("=" * 70)
    print("  RAVANA FRIEND CHAT - 40 flowing conversation turns")
    print("  Each turn builds on the last - like talking to a real friend")
    print("=" * 70)
    print()

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0
    output = result.stdout

    if result.returncode != 0:
        print(f"  [ERROR] Process crashed (code {result.returncode})!")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:2000]}")
        return

    lines = output.split("\n")

    qa_pairs = []
    current_q = ""
    current_a = ""

    for line in lines:
        line = line.rstrip()
        if line.startswith("Q"):
            if current_q:
                qa_pairs.append((current_q, current_a))
            colon_idx = line.find(": ")
            if colon_idx > 0:
                current_q = line[colon_idx + 2:]
            else:
                current_q = line
            current_a = ""
        elif line.startswith("A") and ": " in line:
            colon_idx = line.find(": ")
            if colon_idx > 0:
                current_a = line[colon_idx + 2:]
            else:
                current_a = line
        elif line.startswith("     [") and current_a:
            pass
        elif line.startswith("  [saved") or line.startswith("  [Stats") or line.startswith("  [Reset"):
            pass

    if current_q:
        qa_pairs.append((current_q, current_a))

    # Print flowing conversation
    print()
    print("=" * 70)
    print("  THE CONVERSATION")
    print("=" * 70)
    print()

    strategy_count = {}
    for i, (q, a) in enumerate(qa_pairs):
        strat = ""
        resp_text = a
        if a and " [" in a:
            bracket_idx = a.rfind(" [")
            if bracket_idx > 0:
                strat = a[bracket_idx + 2:-1]
                resp_text = a[:bracket_idx]

        if strat:
            strategy_count[strat] = strategy_count.get(strat, 0) + 1

        print(f"  [{i+1:2d}] You: {q}")
        print(f"      RAVANA: {resp_text}")
        if strat:
            print(f"               [strategy: {strat}]")
        print()

    # Stats
    print("-" * 70)
    for line in lines:
        if line.startswith("  [Stats"):
            print(line)
        if line.startswith("  [saved"):
            print(line)

    print()
    print(f"  Total turns: {len(qa_pairs)}")
    print(f"  Total time: {elapsed:.1f}s")

    if strategy_count:
        total = sum(strategy_count.values())
        print(f"\n  Strategy Distribution ({total} total):")
        for s, c in sorted(strategy_count.items(), key=lambda x: -x[1]):
            pct = c / max(total, 1) * 100
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"    {s:20s} {bar} {c:2d} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
