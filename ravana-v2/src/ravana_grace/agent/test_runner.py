#!/usr/bin/env python3
"""
RAVANA Agent — Quick Test Runner
Tests all interface agent scripts and reports results.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

SKILL_DIR = Path("/home/workspace/Skills/ravana-interface")
SCRIPTS_DIR = SKILL_DIR / "scripts"


def run_test(name: str, cmd: str) -> dict:
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
            cwd=SKILL_DIR
        )
        duration_ms = int((time.time() - start) * 1000)
        status = "pass" if result.returncode == 0 else "fail"
        output = result.stdout[:300] if result.returncode == 0 else result.stderr[:300]
        return {"name": name, "status": status, "duration_ms": duration_ms, "output": output}
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "error", "duration_ms": 60000, "output": "Timeout"}
    except Exception as e:
        return {"name": name, "status": "error", "duration_ms": 0, "output": str(e)[:100]}


def main():
    print("=== RAVANA Agent — Test Runner ===\n")
    
    tests = [
        ("ravana_wrapper", "python3 scripts/ravana_wrapper.py"),
        ("llm_interpreter", "python3 scripts/llm_interpreter.py"),
        ("reality_grounding", "python3 scripts/reality_grounding.py"),
        ("memory_learner", "python3 scripts/memory_learner.py"),
        ("telegram_reporter", "python3 scripts/telegram_reporter.py"),
        ("ravana_agent", "python3 scripts/ravana_agent.py --diagnose"),
    ]
    
    results = []
    for name, cmd in tests:
        print(f"Running {name}...", end=" ")
        r = run_test(name, cmd)
        results.append(r)
        icon = "✅" if r['status'] == 'pass' else "❌" if r['status'] == 'fail' else "⚠️"
        print(f"{icon} {r['status']} ({r['duration_ms']}ms)")
    
    passed = sum(1 for r in results if r['status'] == 'pass')
    print(f"\nResults: {passed}/{len(results)} passed")
    
    return results


if __name__ == "__main__":
    main()
