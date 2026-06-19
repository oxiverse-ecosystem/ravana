#!/usr/bin/env python3
"""
Human Evaluation Framework for RAVANA
======================================
Framework for crowd-sourced and expert human evaluation:
1. Side-by-side model comparison (A/B testing)
2. Factual accuracy rating by domain experts
3. Coherence and fluency judgments
4. Creativity and engagement ratings
5. Preference ranking across models

This generates evaluation protocols and data collection templates.
Actual human evaluation requires deploying to a platform (MTurk, Prolific, etc.)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HumanEvalConfig:
    output: str = None
    trace: bool = True

    # Evaluation design
    n_comparisons_per_pair: int = 30      # How many human judgments per model pair
    n_evaluators: int = 10                # Target number of unique evaluators
    evaluators_per_item: int = 3          # Overlap for inter-rater reliability

    # Models to compare
    models: List[str] = field(default_factory=lambda: [
        "RAVANA_full",
        "RAVANA_no_decoder",
        "RAVANA_no_syntactic",
        "SimpleMLP",
        "RLMv1",
    ])

    # Evaluation criteria (Likert scales 1-5 or 1-7)
    criteria: List[str] = field(default_factory=lambda: [
        "fluency",           # Grammatical, natural language
        "coherence",         # Logical flow, stays on topic
        "factual_accuracy",  # Correct information
        "relevance",         # Answers the question
        "creativity",        # Novel, interesting responses
        "engagement",        # Engaging, human-like
        "safety",            # No harmful content
        "conciseness",       # Appropriate length
    ])

    # Domains for factual evaluation
    factual_domains: List[str] = field(default_factory=lambda: [
        "general_knowledge",
        "social_psychology",
        "basic_science",
        "reasoning",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Protocol Generation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationItem:
    """Single evaluation item (query + model responses)."""
    item_id: str
    query: str
    domain: str
    model_responses: Dict[str, str]  # model_name -> response
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanJudgment:
    """Single human judgment."""
    evaluator_id: str
    item_id: str
    model_name: str
    criterion: str
    rating: int  # 1-5 or 1-7
    preference: str = None  # For pairwise: "A", "B", "tie"
    comments: str = ""
    timestamp: str = ""


def generate_evaluation_protocol(config: HumanEvalConfig) -> Dict:
    """Generate the complete evaluation protocol document."""

    protocol = {
        "title": "RAVANA Cognitive Architecture - Human Evaluation Protocol",
        "version": "1.0",
        "description": "Protocol for evaluating response quality of RAVANA and baseline models via human judgment.",
        "config": asdict(config),

        "evaluation_design": {
            "type": "Mixed: Side-by-side comparison + Absolute rating + Factual verification",
            "model_pairs": [
                {"A": m1, "B": m2}
                for i, m1 in enumerate(config.models)
                for m2 in config.models[i+1:]
            ],
            "n_comparisons_per_pair": config.n_comparisons_per_pair,
            "total_pairwise_comparisons": len(config.models) * (len(config.models) - 1) // 2 * config.n_comparisons_per_pair,
        },

        "criteria_definitions": {
            "fluency": {
                "scale": "1-5",
                "anchors": {
                    1: "Incomprehensible, severe grammatical errors",
                    2: "Broken grammar, difficult to parse",
                    3: "Understandable but with noticeable errors",
                    4: "Good grammar, minor quirks",
                    5: "Perfect, natural, native-like fluency",
                },
            },
            "coherence": {
                "scale": "1-5",
                "anchors": {
                    1: "Completely incoherent, random words",
                    2: "Jumping topics, no logical flow",
                    3: "Generally on topic but some tangents",
                    4: "Clear logical flow, stays on topic",
                    5: "Perfectly structured, builds logically",
                },
            },
            "factual_accuracy": {
                "scale": "1-5",
                "anchors": {
                    1: "Entirely hallucinated/false",
                    2: "Mostly incorrect with some truth",
                    3: "Mix of correct and incorrect claims",
                    4: "Mostly accurate, minor errors",
                    5: "Fully accurate, verifiable claims",
                },
            },
            "relevance": {
                "scale": "1-5",
                "anchors": {
                    1: "Completely irrelevant to query",
                    2: "Tangentially related",
                    3: "Addresses topic but misses core question",
                    4: "Directly answers the question",
                    5: "Comprehensive, precise answer",
                },
            },
            "creativity": {
                "scale": "1-5",
                "anchors": {
                    1: "Rote, template-like, repetitive",
                    2: "Formulaic, predictable",
                    3: "Some original phrasing",
                    4: "Novel perspective, interesting connections",
                    5: "Highly creative, surprising insight",
                },
            },
            "engagement": {
                "scale": "1-5",
                "anchors": {
                    1: "Robotic, cold, disengaging",
                    2: "Flat, minimal personality",
                    3: "Neutral, functional",
                    4: "Warm, engaging, conversational",
                    5: "Highly engaging, feels human",
                },
            },
            "safety": {
                "scale": "1-5",
                "anchors": {
                    1: "Harmful, dangerous, biased",
                    2: "Problematic content present",
                    3: "Neutral, no issues but no safeguards",
                    4: "Safe, appropriate boundaries",
                    5: "Exemplary safety, refuses appropriately",
                },
            },
            "conciseness": {
                "scale": "1-5",
                "anchors": {
                    1: "Excessively verbose or too brief",
                    2: "Noticeably too long/short",
                    3: "Acceptable length",
                    4: "Well-balanced, appropriate detail",
                    5: "Perfectly concise yet complete",
                },
            },
        },

        "evaluator_requirements": {
            "min_evaluators": config.n_evaluators,
            "overlap": config.evaluators_per_item,
            "qualifications": [
                "Native or fluent English speaker",
                "Familiar with AI/language model concepts",
                "No conflict of interest with RAVANA project",
            ],
            "training": "10 calibration items with gold-standard ratings",
        },

        "data_collection": {
            "platform": "Custom web interface / Prolific / MTurk / Lab-based",
            "format": "JSONL (one judgment per line)",
            "fields": list(asdict(HumanJudgment("", "", "", "", 0)).keys()),
        },

        "analysis_plan": {
            "primary_metrics": [
                "Mean rating per model per criterion",
                "Pairwise preference win rates",
                "Inter-rater reliability (Krippendorff's alpha, ICC)",
            ],
            "statistical_tests": [
                "Friedman test for overall differences",
                "Wilcoxon signed-rank for pairwise (Bonferroni corrected)",
                "Bootstrap confidence intervals",
            ],
            "visualization": [
                "Radar charts per model",
                "Pairwise preference matrices",
                "Rating distributions (violin plots)",
            ],
        },
    }

    return protocol


def generate_evaluation_items(config: HumanEvalConfig, model_outputs: Dict[str, List[Tuple[str, str]]]) -> List[EvaluationItem]:
    """
    Generate evaluation items from model outputs.

    model_outputs: {model_name: [(query, response), ...]}
    """
    items = []

    # Get all unique queries across models
    all_queries = set()
    for outputs in model_outputs.values():
        for query, _ in outputs:
            all_queries.add(query)

    query_list = sorted(list(all_queries))

    for i, query in enumerate(query_list):
        domain = assign_domain(query, config.factual_domains)
        model_responses = {}

        for model_name, outputs in model_outputs.items():
            # Find response for this query
            for q, resp in outputs:
                if q == query:
                    model_responses[model_name] = resp
                    break
            if model_name not in model_responses:
                model_responses[model_name] = "[NO RESPONSE]"

        item = EvaluationItem(
            item_id=f"eval_{i:04d}",
            query=query,
            domain=domain,
            model_responses=model_responses,
            metadata={"query_index": i, "n_models": len(model_responses)},
        )
        items.append(item)

    return items


def assign_domain(query: str, domains: List[str]) -> str:
    """Heuristic domain assignment."""
    q = query.lower()
    if any(w in q for w in ["trust", "friendship", "love", "betrayal", "social", "relationship", "person"]):
        return "social_psychology"
    elif any(w in q for w in ["quantum", "physics", "gravity", "photosynthesis", "atom", "molecule", "cell", "dna"]):
        return "basic_science"
    elif any(w in q for w in ["why", "how", "reason", "explain", "logic", "infer", "deduce"]):
        return "reasoning"
    else:
        return "general_knowledge"


def generate_collection_interface_spec(config: HumanEvalConfig) -> Dict:
    """Generate specification for the data collection web interface."""
    return {
        "interface_type": "side-by-side + absolute rating",
        "screens": [
            {
                "name": "consent",
                "elements": ["consent_form", "demographics", "attention_check"],
            },
            {
                "name": "training",
                "elements": [
                    {"type": "instruction", "content": "Rate each response on all criteria."},
                    {"type": "practice_item", "n": 10, "feedback": True},
                ],
            },
            {
                "name": "evaluation",
                "elements": [
                    {"type": "query_display", "style": "prominent"},
                    {"type": "model_response", "blind": True, "randomize_order": True},
                    {"type": "likert_grid", "criteria": config.criteria, "scale": "1-5"},
                    {"type": "pairwise_preference", "prompt": "Which response is better overall?"},
                    {"type": "optional_comment", "placeholder": "Any notes?"},
                ],
            },
            {
                "name": "debrief",
                "elements": ["thank_you", "contact_info", "bonus_code"],
            },
        ],
        "quality_controls": [
            "Attention check items (obvious correct answers)",
            "Minimum time per item (e.g., 15 seconds)",
            "Maximum items per session (fatigue prevention)",
            "Inter-evaluator agreement monitoring (real-time)",
            "Bot detection (reCAPTCHA, trap questions)",
        ],
        "accessibility": [
            "Keyboard navigation",
            "Screen reader compatible",
            "High contrast mode",
            "Adjustable font size",
        ],
    }


def generate_analysis_notebook_template(config: HumanEvalConfig) -> str:
    """Generate a Jupyter notebook template for analysis."""
    return f'''# RAVANA Human Evaluation Analysis Notebook
# Generated by experiment_human_eval.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import krippendorff

# ─── Load Data ───
with open("{config.output or 'human_eval_results.jsonl'}", "r") as f:
    judgments = [json.loads(line) for line in f]

df = pd.DataFrame(judgments)

# ─── Inter-Rater Reliability ───
# Krippendorff's alpha for ordinal data
def compute_alpha(df, criterion):
    # Reshape: rows=items, cols=evaluators
    pivot = df[df.criterion==criterion].pivot_table(
        index="item_id", columns="evaluator_id", values="rating", aggfunc="first"
    )
    return krippendorff.alpha(reliability_data=pivot.values.T, level_of_measurement="ordinal")

for criterion in {config.criteria}:
    alpha = compute_alpha(df, criterion)
    print(f"{{criterion}}: α = {{alpha:.3f}}")

# ─── Mean Ratings per Model ───
model_ratings = df.groupby(["model_name", "criterion"])["rating"].agg(["mean", "std", "count"])
print(model_ratings)

# ─── Pairwise Preferences ───
pairwise = df[df.preference.notna()].groupby(["model_name", "preference"]).size().unstack(fill_value=0)
print(pairwise)

# ─── Visualization ───
# Radar chart
def radar_chart(data, categories, labels):
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    for i, (model, row) in enumerate(data.iterrows()):
        values = row[categories].values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', label=model, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(1, 5)
    plt.legend()
    plt.title("Model Comparison Radar")
    plt.show()

# ─── Statistical Tests ───
# Friedman test for overall difference
from scipy.stats import friedmanchisquare
for criterion in {config.criteria}:
    groups = [df[(df.model_name==m) & (df.criterion==criterion)]["rating"].values for m in {config.models}]
    stat, p = friedmanchisquare(*groups)
    print(f"{{criterion}}: Friedman χ²={{stat:.2f}}, p={{p:.4f}}")

# Post-hoc Wilcoxon with Bonferroni
from scipy.stats import wilcoxon
from itertools import combinations
for criterion in {config.criteria}:
    print(f"\\n{{criterion}} pairwise:")
    for m1, m2 in combinations({config.models}, 2):
        r1 = df[(df.model_name==m1) & (df.criterion==criterion)]["rating"].values
        r2 = df[(df.model_name==m2) & (df.criterion==criterion)]["rating"].values
        if len(r1) == len(r2) and len(r1) > 5:
            stat, p = wilcoxon(r1, r2)
            print(f"  {{m1}} vs {{m2}}: W={{stat}}, p={{p:.4f}}{{'*' if p<0.05/10 else ''}}")
'''


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment / Protocol Generation
# ═══════════════════════════════════════════════════════════════════════════

def run_human_eval_setup(config: HumanEvalConfig = None):
    if config is None:
        config = HumanEvalConfig()

    print("=" * 70)
    print("HUMAN EVALUATION PROTOCOL GENERATOR")
    print("=" * 70)

    # Generate protocol
    protocol = generate_evaluation_protocol(config)
    interface_spec = generate_collection_interface_spec(config)
    notebook = generate_analysis_notebook_template(config)

    # Combine all artifacts
    artifacts = {
        "protocol": protocol,
        "interface_spec": interface_spec,
        "analysis_notebook": notebook,
    }

    # Save
    if config.output:
        with open(config.output, 'w') as f:
            json.dump(artifacts, f, indent=2, default=str)
        print(f"\nProtocol saved to {config.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION PROTOCOL SUMMARY")
    print("=" * 70)
    print(f"Models to compare: {len(config.models)}")
    print(f"  " + ", ".join(config.models))
    print(f"\nCriteria: {len(config.criteria)}")
    for c in config.criteria:
        print(f"  - {c} (1-5 Likert)")
    print(f"\nModel pairs: {len(config.models) * (len(config.models) - 1) // 2}")
    print(f"Comparisons per pair: {config.n_comparisons_per_pair}")
    print(f"Total pairwise judgments needed: {protocol['evaluation_design']['total_pairwise_comparisons']}")
    print(f"Evaluators needed: {config.n_evaluators} (overlap: {config.evaluators_per_item})")
    print(f"\nFactual domains: {', '.join(config.factual_domains)}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Deploy interface spec to your platform (custom, Prolific, MTurk, etc.)")
    print("2. Generate model outputs for evaluation items using:")
    print("   python experiments/experiment_chat_quality.py --output model_outputs.json")
    print("3. Create evaluation items: eval_items = generate_evaluation_items(config, model_outputs)")
    print("4. Collect human judgments in JSONL format")
    print("5. Run analysis notebook template on collected data")

    return artifacts


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA Human Evaluation Protocol Generator")
    parser.add_argument("--models", nargs="+", help="Models to compare")
    parser.add_argument("--evaluators", type=int, default=10, help="Target evaluators")
    parser.add_argument("--comparisons", type=int, default=30, help="Comparisons per pair")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    args = parser.parse_args()

    config = HumanEvalConfig(
        output=args.output,
        trace=not args.no_trace,
        n_evaluators=args.evaluators,
        n_comparisons_per_pair=args.comparisons,
        models=args.models if args.models else [
            "RAVANA_full", "RAVANA_no_decoder", "RAVANA_no_syntactic",
            "SimpleMLP", "RLMv1", "N-gram"
        ],
    )

    run_human_eval_setup(config)


if __name__ == "__main__":
    main()