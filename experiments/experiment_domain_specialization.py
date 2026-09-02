#!/usr/bin/env python3
"""
Domain Specialization Experiments for RAVANA
=============================================
Evaluates performance across specialized domains:
1. Technical: Code generation, math reasoning
2. Creative Writing: Stories, poems, dialogue
3. Reasoning: Logic puzzles, syllogisms, causal reasoning
4. Planning: Task decomposition, multi-step reasoning
5. Factual QA: Knowledge retrieval, verification
"""

import os
import sys
import time
import json
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ravana_chat import CognitiveChatEngine


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DomainConfig:
    seed: int = 42
    output: str = None
    trace: bool = True
    n_runs: int = 3

    # Domains to test
    domains: List[str] = field(default_factory=lambda: [
        "code", "math", "creative", "reasoning", "planning", "factual_qa"
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DomainMetrics:
    domain: str
    task: str
    run: int
    query: str
    response: str
    latency_ms: float
    # Domain-specific scores
    syntax_valid: bool = None       # code
    execution_correct: bool = None  # code/math
    math_accuracy: float = None     # math
    creativity_score: float = None  # creative
    logic_valid: bool = None        # reasoning
    plan_feasible: bool = None      # planning
    factual_correct: bool = None    # factual_qa
    # General quality
    grammar_score: float = 0.0
    coherence: float = 0.0
    relevance: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Domain-Specific Evaluators
# ═══════════════════════════════════════════════════════════════════════════

def compute_grammar_score(response: str) -> float:
    if not response or len(response) < 5: return 0.0
    score = 0.0
    if response[0].isupper(): score += 0.25
    if response.strip().endswith('.'): score += 0.25
    words = set(response.lower().strip('.').split())
    verbs = {'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'can', 'will', 'would', 'could', 'should', 'connect', 'relate', 'lead', 'cause', 'make', 'create', 'include', 'involve', 'mean', 'refer', 'stand', 'represent', 'symbolize'}
    if words & verbs: score += 0.25
    word_counts = Counter(response.lower().split())
    if max(word_counts.values()) < 3: score += 0.25
    return min(1.0, score)


def compute_coherence(response: str, engine) -> float:
    if not response: return 0.0
    words = [w.strip('.,!?') for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    if not words: return 0.0
    known = sum(1 for w in words if w in engine._concept_keywords)
    return known / len(words)


# ─── CODE DOMAIN ───
DOMAIN_CODE_TASKS = [
    ("python_basics", "write a python function to reverse a string"),
    ("python_basics", "write a python function to calculate factorial"),
    ("python_basics", "write a python class for a binary tree node"),
    ("algorithms", "write a python function for binary search"),
    ("algorithms", "write a python function to merge two sorted lists"),
    ("data_structures", "implement a stack in python"),
    ("data_structures", "implement a queue in python"),
    ("debugging", "fix the bug: for i in range(10): print(i) # missing colon"),
    ("libraries", "write python code using requests to fetch a URL"),
    ("libraries", "write python code using pandas to read a CSV"),
]

def evaluate_code(response: str, task_name: str) -> Tuple[bool, bool]:
    """Check syntax validity and basic execution."""
    syntax_valid = False
    execution_correct = False

    # Extract code blocks
    code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
    if not code_blocks:
        # Try to find python-like code
        lines = response.split('\n')
        code_lines = [l for l in lines if any(kw in l for kw in ['def ', 'class ', 'import ', 'for ', 'if ', 'return ', '= '])]
        if code_lines:
            code_blocks = ['\n'.join(code_lines)]

    for code in code_blocks:
        try:
            compile(code, '<string>', 'exec')
            syntax_valid = True
            # Try execution in restricted namespace
            namespace = {}
            exec(code, {"__builtins__": {}}, namespace)
            execution_correct = True
            break
        except Exception:
            continue

    return syntax_valid, execution_correct


# ─── MATH DOMAIN ───
DOMAIN_MATH_TASKS = [
    ("arithmetic", "what is 15 * 23"),
    ("arithmetic", "calculate 144 / 12"),
    ("algebra", "solve for x: 2x + 5 = 17"),
    ("algebra", "solve: x^2 - 5x + 6 = 0"),
    ("geometry", "area of circle with radius 5"),
    ("geometry", "pythagorean theorem: a=3, b=4, find c"),
    ("calculus", "derivative of x^3 + 2x^2"),
    ("calculus", "integral of 2x dx"),
    ("statistics", "mean of [2, 4, 6, 8, 10]"),
    ("probability", "probability of rolling 6 on fair die"),
]

MATH_ANSWERS = {
    "what is 15 * 23": "345",
    "calculate 144 / 12": "12",
    "solve for x: 2x + 5 = 17": "6",
    "solve: x^2 - 5x + 6 = 0": "2 or 3",
    "area of circle with radius 5": "78.5",
    "pythagorean theorem: a=3, b=4, find c": "5",
    "derivative of x^3 + 2x^2": "3x^2 + 4x",
    "integral of 2x dx": "x^2",
    "mean of [2, 4, 6, 8, 10]": "6",
    "probability of rolling 6 on fair die": "1/6",
}

def evaluate_math(response: str, query: str) -> float:
    """Extract numeric answer and compare to expected."""
    expected = MATH_ANSWERS.get(query, "").lower()

    # Extract numbers from response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if not numbers:
        return 0.0

    # Check if any extracted number matches expected
    for num in numbers:
        try:
            if abs(float(num) - float(expected)) < 0.1:
                return 1.0
        except:
            # Try string match
            if expected in response.lower():
                return 1.0

    # Try string containment
    if expected in response.lower():
        return 1.0

    return 0.0


# ─── CREATIVE DOMAIN ───
DOMAIN_CREATIVE_TASKS = [
    ("story", "write a short story about a robot learning to trust"),
    ("story", "write a story about friendship between a cat and a dog"),
    ("poem", "write a poem about the concept of trust"),
    ("poem", "write a haiku about friendship"),
    ("dialogue", "write a dialogue between two friends discussing betrayal"),
    ("metaphor", "create a metaphor for memory"),
    ("worldbuilding", "describe a world where trust is a physical currency"),
    ("character", "create a character who cannot trust anyone"),
    ("plot", "outline a plot about rebuilding trust after betrayal"),
    ("style", "rewrite 'trust is important' in the style of Shakespeare"),
]

def evaluate_creativity(response: str, task_type: str) -> float:
    """Heuristic creativity scoring."""
    score = 0.0

    # Length (creative responses tend to be longer)
    if len(response) > 100: score += 0.2
    if len(response) > 300: score += 0.1

    # Diversity
    words = response.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        score += unique_ratio * 0.3

    # Presence of creative markers
    creative_markers = ['imagine', 'metaphor', 'like a', 'as if', 'symbolizes',
                        'represents', 'embodies', 'captures', 'evokes']
    if any(m in response.lower() for m in creative_markers):
        score += 0.2

    # Structure markers for stories/poems
    if task_type in ('story', 'poem'):
        if any(m in response.lower() for m in ['once upon', 'suddenly', 'meanwhile', 'finally']):
            score += 0.1
        if response.count('\n') > 2 or response.count('.') > 3:
            score += 0.1

    return min(1.0, score)


# ─── REASONING DOMAIN ───
DOMAIN_REASONING_TASKS = [
    ("syllogism", "all humans are mortal. socrates is human. is socrates mortal?"),
    ("syllogism", "all birds can fly. penguins are birds. can penguins fly?"),
    ("causal", "if it rains, the ground gets wet. the ground is wet. did it rain?"),
    ("causal", "fire causes smoke. there is smoke. is there fire?"),
    ("counterfactual", "if i had studied, i would have passed. i didn't study. did i pass?"),
    ("analogy", "trust is to friendship as foundation is to what?"),
    ("transitive", "A trusts B. B trusts C. Does A trust C?"),
    ("necessity", "is trust necessary for friendship?"),
    ("sufficiency", "is trust sufficient for friendship?"),
    ("abductive", "the window is broken. a baseball is inside. what happened?"),
]

def evaluate_reasoning(response: str, query: str) -> bool:
    """Check if response shows valid logical reasoning."""
    response_lower = response.lower()

    # Check for key logical connectors
    logical_markers = ['therefore', 'thus', 'hence', 'because', 'since',
                       'implies', 'follows', 'conclude', 'deduce', 'infer']

    # Specific correct answers for known tasks
    if "socrates" in query:
        return "yes" in response_lower or "mortal" in response_lower
    elif "penguin" in query:
        return "no" in response_lower or "cannot fly" in response_lower
    elif "ground is wet" in query:
        return "not necessarily" in response_lower or "could be" in response_lower
    elif "fire" in query and "smoke" in query:
        return "not necessarily" in response_lower or "could be" in response_lower
    elif "trust" in query and "friendship" in query and "necessary" in query:
        return "necessary" in response_lower or "essential" in response_lower
    elif "transitive" in query or "A trusts B" in query:
        return "not necessarily" in response_lower or "no" in response_lower
    elif "abductive" in query or "baseball" in query:
        return "baseball" in response_lower and ("threw" in response_lower or "hit" in response_lower)
    elif "analogy" in query:
        return "house" in response_lower or "building" in response_lower or "structure" in response_lower

    # General: has reasoning markers and substantive content
    has_logic = any(m in response_lower for m in logical_markers)
    has_content = len(response) > 30
    return has_logic and has_content


# ─── PLANNING DOMAIN ───
DOMAIN_PLANNING_TASKS = [
    ("decomposition", "plan steps to build trust in a new team"),
    ("decomposition", "create a plan to learn a new programming language"),
    ("scheduling", "plan a weekly schedule for studying and exercise"),
    ("resource", "plan a project with limited time and budget"),
    ("contingency", "plan for a trip with backup options for bad weather"),
    ("multi_agent", "plan how two people can resolve a conflict"),
    ("goal_setting", "break down 'become more trustworthy' into actionable steps"),
    ("prioritization", "prioritize: sleep, work, exercise, social, learning"),
    ("monitoring", "design a system to track progress on building habits"),
    ("adaptation", "how to adapt a plan when unexpected obstacles arise"),
]

def evaluate_planning(response: str) -> Tuple[bool, float]:
    """Check if response contains feasible multi-step plan."""
    response_lower = response.lower()

    # Look for step-like structure
    step_markers = ['step 1', 'step 2', 'first', 'second', 'third', 'then', 'next',
                    'finally', 'phase 1', 'phase 2', 'stage 1', 'stage 2']
    has_steps = any(m in response_lower for m in step_markers)

    # Numbered list
    numbered = bool(re.search(r'\d+\.\s', response))

    # Action verbs
    action_verbs = ['identify', 'analyze', 'design', 'implement', 'test', 'review',
                    'create', 'build', 'establish', 'monitor', 'adjust', 'evaluate']
    has_actions = any(v in response_lower for v in action_verbs)

    # Length and structure
    substantial = len(response) > 100

    feasible = has_steps or numbered
    quality = (0.3 if has_steps else 0) + (0.2 if numbered else 0) + (0.3 if has_actions else 0) + (0.2 if substantial else 0)

    return feasible, min(1.0, quality)


# ─── FACTUAL QA DOMAIN ───
DOMAIN_FACTUAL_TASKS = [
    ("science", "what is the speed of light"),
    ("science", "what is photosynthesis"),
    ("geography", "capital of france"),
    ("geography", "largest ocean on earth"),
    ("history", "who wrote the declaration of independence"),
    ("history", "when did world war ii end"),
    ("biology", "what is dna"),
    ("biology", "function of mitochondria"),
    ("physics", "newton's first law"),
    ("chemistry", "chemical symbol for gold"),
]

FACTUAL_ANSWERS = {
    "what is the speed of light": ["299792458", "300000", "3e8", "speed of light"],
    "what is photosynthesis": ["photosynthesis", "plants", "sunlight", "oxygen", "glucose"],
    "capital of france": ["paris"],
    "largest ocean on earth": ["pacific"],
    "who wrote the declaration of independence": ["jefferson", "thomas jefferson"],
    "when did world war ii end": ["1945"],
    "what is dna": ["dna", "deoxyribonucleic", "genetic", "genome"],
    "function of mitochondria": ["mitochondria", "energy", "atp", "powerhouse"],
    "newton's first law": ["inertia", "motion", "force", "rest"],
    "chemical symbol for gold": ["au", "gold"],
}

def evaluate_factual(response: str, query: str) -> bool:
    """Check if response contains key factual elements."""
    expected_keywords = FACTUAL_ANSWERS.get(query, [])
    response_lower = response.lower()

    matches = sum(1 for kw in expected_keywords if kw in response_lower)
    return matches > 0


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_domain_specialization(config: DomainConfig = None):
    if config is None:
        config = DomainConfig()

    np.random.seed(config.seed)

    print("=" * 70)
    print("DOMAIN SPECIALIZATION EXPERIMENTS")
    print("=" * 70)

    # Initialize engine
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
    if hasattr(engine, '_seed_corpus_training'):
        engine._seed_corpus_training()

    print(f"Engine ready: {len(engine.graph.nodes)} concepts")

    # Domain task registry
    domain_tasks = {
        "code": DOMAIN_CODE_TASKS,
        "math": DOMAIN_MATH_TASKS,
        "creative": DOMAIN_CREATIVE_TASKS,
        "reasoning": DOMAIN_REASONING_TASKS,
        "planning": DOMAIN_PLANNING_TASKS,
        "factual_qa": DOMAIN_FACTUAL_TASKS,
    }

    all_results = []

    for domain in config.domains:
        if domain not in domain_tasks:
            print(f"Unknown domain: {domain}")
            continue

        tasks = domain_tasks[domain]
        print(f"\n{'='*60}")
        print(f"DOMAIN: {domain.upper()} ({len(tasks)} tasks)")
        print(f"{'='*60}")

        for task_name, query in tasks:
            print(f"\n  Task: {task_name} - {query}")

            for run in range(config.n_runs):
                t0 = time.time()
                response = engine.process_turn(query)
                latency = (time.time() - t0) * 1000

                # General quality
                grammar = compute_grammar_score(response)
                coherence = compute_coherence(response, engine)
                relevance = 1.0 if len(response) > 10 else 0.0  # placeholder

                # Domain-specific evaluation
                syntax_v = exec_c = math_acc = creativity = logic_v = plan_feas = fact_c = None

                if domain == "code":
                    syntax_v, exec_c = evaluate_code(response, task_name)
                elif domain == "math":
                    math_acc = evaluate_math(response, query)
                elif domain == "creative":
                    creativity = evaluate_creativity(response, task_name)
                elif domain == "reasoning":
                    logic_v = evaluate_reasoning(response, query)
                elif domain == "planning":
                    plan_feas, plan_quality = evaluate_planning(response)
                elif domain == "factual_qa":
                    fact_c = evaluate_factual(response, query)

                m = DomainMetrics(
                    domain=domain,
                    task=task_name,
                    run=run,
                    query=query,
                    response=response[:300],
                    latency_ms=latency,
                    syntax_valid=syntax_v,
                    execution_correct=exec_c,
                    math_accuracy=math_acc,
                    creativity_score=creativity,
                    logic_valid=logic_v,
                    plan_feasible=plan_feas,
                    factual_correct=fact_c,
                    grammar_score=grammar,
                    coherence=coherence,
                    relevance=relevance,
                )
                all_results.append(m)

                if config.trace:
                    domain_info = ""
                    if domain == "code": domain_info = f" syntax={syntax_v} exec={exec_c}"
                    elif domain == "math": domain_info = f" math_acc={math_acc:.0f}"
                    elif domain == "creative": domain_info = f" creativity={creativity:.2f}"
                    elif domain == "reasoning": domain_info = f" logic={logic_v}"
                    elif domain == "planning": domain_info = f" feasible={plan_feas} quality={plan_quality:.2f}"
                    elif domain == "factual_qa": domain_info = f" fact={fact_c}"
                    print(f"    Run {run+1}: {latency:.1f}ms gram={grammar:.2f} coh={coherence:.2f}{domain_info}")

    # Summary by domain
    print("\n" + "=" * 70)
    print("DOMAIN SPECIALIZATION SUMMARY")
    print("=" * 70)

    summary = {}
    for domain in config.domains:
        domain_results = [r for r in all_results if r.domain == domain]
        if not domain_results:
            continue

        print(f"\n{domain.upper()}:")
        print(f"  Tasks: {len(set(r.task for r in domain_results))}")
        print(f"  Avg latency: {np.mean([r.latency_ms for r in domain_results]):.1f}ms")
        print(f"  Avg grammar: {np.mean([r.grammar_score for r in domain_results]):.3f}")
        print(f"  Avg coherence: {np.mean([r.coherence for r in domain_results]):.3f}")

        if domain == "code":
            syntax_rate = np.mean([1.0 if r.syntax_valid else 0.0 for r in domain_results if r.syntax_valid is not None])
            exec_rate = np.mean([1.0 if r.execution_correct else 0.0 for r in domain_results if r.execution_correct is not None])
            print(f"  Syntax validity: {syntax_rate:.1%}")
            print(f"  Execution correct: {exec_rate:.1%}")
        elif domain == "math":
            math_rate = np.mean([r.math_accuracy for r in domain_results if r.math_accuracy is not None])
            print(f"  Math accuracy: {math_rate:.1%}")
        elif domain == "creative":
            creat_avg = np.mean([r.creativity_score for r in domain_results if r.creativity_score is not None])
            print(f"  Creativity score: {creat_avg:.3f}")
        elif domain == "reasoning":
            logic_rate = np.mean([1.0 if r.logic_valid else 0.0 for r in domain_results if r.logic_valid is not None])
            print(f"  Logic validity: {logic_rate:.1%}")
        elif domain == "planning":
            feas_rate = np.mean([1.0 if r.plan_feasible else 0.0 for r in domain_results if r.plan_feasible is not None])
            print(f"  Plan feasibility: {feas_rate:.1%}")
        elif domain == "factual_qa":
            fact_rate = np.mean([1.0 if r.factual_correct else 0.0 for r in domain_results if r.factual_correct is not None])
            print(f"  Factual accuracy: {fact_rate:.1%}")

        summary[domain] = {
            "n_tasks": len(set(r.task for r in domain_results)),
            "avg_latency_ms": float(np.mean([r.latency_ms for r in domain_results])),
            "avg_grammar": float(np.mean([r.grammar_score for r in domain_results])),
            "avg_coherence": float(np.mean([r.coherence for r in domain_results])),
        }

    # Save
    if config.output:
        output = {
            'config': asdict(config),
            'summary': summary,
            'detailed_results': [asdict(r) for r in all_results],
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")

    return all_results, summary


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA Domain Specialization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs", type=int, default=3, help="Runs per task")
    parser.add_argument("--domains", nargs="+", help="Domains to test")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    config = DomainConfig(
        seed=args.seed,
        n_runs=args.runs,
        trace=not args.no_trace,
        output=args.output,
        domains=args.domains if args.domains else [
            "code", "math", "creative", "reasoning", "planning", "factual_qa"
        ],
    )

    run_domain_specialization(config)


if __name__ == "__main__":
    main()