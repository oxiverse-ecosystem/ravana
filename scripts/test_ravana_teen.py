#!/usr/bin/env python3
"""
RAVANA Teenager CI Stress Test -- 40 queries, all strategies, deep conversation.
Runs via: python scripts/test_ravana_teen.py
"""

import subprocess
import sys
import os
import time

# -- 40 Diverse Queries -------------------------------------------------------
QUERIES = [
    # 1-5: Greetings & Getting Started
    "hello",
    "hey whats up",
    "im good how are you",
    "nice to meet you",
    "what do you think about",

    # 6-12: Abstract Social Concepts
    "what is trust",
    "why do people break trust",
    "do you think honesty is always the best policy",
    "what is justice",
    "is the world fair",
    "what does freedom mean to you",
    "what is empathy",

    # 13-18: Deep Philosophical
    "what is the meaning of life",
    "do you think people can change",
    "why do bad things happen",
    "what happens after we die",
    "is there such a thing as truth",
    "what is the difference between knowledge and wisdom",

    # 19-24: Causal & Scientific
    "why is the sky blue",
    "does learning change your brain",
    "what causes people to be angry",
    "why do people feel sad sometimes",
    "does practice really make you better at things",
    "what makes a person happy",

    # 25-30: Counterfactual & "What If"
    "what if humans could fly",
    "what if there was no such thing as money",
    "what if everyone told the truth all the time",
    "what would happen if the sun disappeared",
    "imagine a world without sadness",
    "what if you could learn anything instantly",

    # 31-35: Debate & Multiple Perspectives
    "is technology good or bad for society",
    "some people say money cant buy happiness what do you think",
    "is it better to be logical or emotional",
    "should we always follow the rules",
    "is social media good or bad",

    # 36-38: Meta-Cognitive / Self-Reflection
    "do you know what you dont know",
    "have you learned anything new today",
    "what do you think about yourself",

    # 39-40: Closing
    "thanks for talking to me",
    "goodbye",
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(script_dir)
    chat_script = os.path.join(script_dir, "ravana_chat.py")

    chat_string = "|".join(QUERIES)

    cmd = [
        sys.executable, chat_script,
        "--reset",
        "--chat", chat_string,
        "--strategy",
    ]

    print("=" * 65)
    print("  RAVANA TEENAGER CI STRESS TEST - 40 QUERIES")
    print("=" * 65)
    print(f"  Starting with --reset (fresh teenage brain)")
    print(f"  Query count: {len(QUERIES)}")
    print()

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0

    output = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"  [ERROR] Process crashed (code {result.returncode})!")
        if stderr:
            print(f"  STDERR: {stderr[:2000]}")
        return

    # Parse the output
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
            pass  # timing info -- skip
        elif line.startswith("  [saved") or line.startswith("  [Stats") or line.startswith("  [Reset"):
            pass  # stats/footer lines -- don't overwrite answer

    if current_q:
        qa_pairs.append((current_q, current_a))

    # Print all Q/A pairs
    print(f"  Completed in {elapsed:.1f}s")
    print()
    print("-" * 65)

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

        print(f"  [{i+1:2d}] {q}")
        print(f"       => {resp_text}")
        if strat:
            print(f"         [strategy: {strat}]")
        print()

    # Print end stats from ravana_chat.py output
    for line in lines:
        if line.startswith("  [Stats"):
            print(line)
        if line.startswith("  [saved"):
            print(line)

    print()
    print(f"  Total queries: {len(qa_pairs)}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Avg response: {elapsed / max(len(qa_pairs), 1):.2f}s")

    # Strategy distribution
    if strategy_count:
        total = sum(strategy_count.values())
        print(f"\n  Strategy Distribution ({total} total):")
        for s, c in sorted(strategy_count.items(), key=lambda x: -x[1]):
            pct = c / max(total, 1) * 100
            bar_len = int(pct / 5)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"    {s:20s} {bar} {c:2d} ({pct:.0f}%)")
        print()


if __name__ == "__main__":
    main()
