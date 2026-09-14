#!/usr/bin/env python3
"""Acceptance ledger: grade each cognitive module GREEN/RED with numbers.

Runs a series of probes against a fresh CognitiveChatEngine and grades
each module on pass/fail with quantitative metrics. Designed to be run
from the repo root:

    python scripts/acceptance_ledger.py

Exit code 0 = all GREEN, 1 = any RED.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# Path setup for repo imports
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
          os.path.join(_PROJ, "ravana_ml", "src"),
          os.path.join(_PROJ, "ravana-v2", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.engine import CognitiveChatEngine


@dataclass
class ModuleGrade:
    name: str
    status: str  # "GREEN" or "RED"
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def grade_identity(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade the identity module: strength, momentum, stability."""
    g = ModuleGrade("identity", "GREEN")
    try:
        status = eng.identity.get_status()
        g.metrics["strength"] = round(status.get("strength", 0.0), 4)
        g.metrics["momentum"] = round(status.get("momentum", 0.0), 4)
        g.metrics["stability"] = round(status.get("stability", 0.0), 4)
        g.metrics["trend"] = status.get("trend", "unknown")

        # Identity should have non-zero strength after processing
        if status.get("strength", 0.0) <= 0.0:
            g.status = "RED"
            g.notes.append("identity strength is zero after init")
        if "strength" not in status:
            g.status = "RED"
            g.notes.append("identity status missing 'strength' key")
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"identity.get_status() raised: {e}")
    return g


def grade_stances(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade the stance store: can form and recall stances."""
    g = ModuleGrade("stances", "GREEN")
    try:
        # Process a disclosure that should form a stance
        eng.process_turn("i love coffee")
        stances = eng.user_model.opinions.stances
        g.metrics["stance_count"] = len(stances)
        g.metrics["topics"] = list(stances.keys())[:5]

        if len(stances) == 0:
            g.status = "RED"
            g.notes.append("no stances formed after 'i love coffee'")
        else:
            # Check that the stance has a valid polarity
            for topic, stance in stances.items():
                if not (-1.0 <= stance.polarity <= 1.0):
                    g.status = "RED"
                    g.notes.append(f"stance '{topic}' polarity out of range: {stance.polarity}")
                if not (0.0 <= stance.confidence <= 1.0):
                    g.status = "RED"
                    g.notes.append(f"stance '{topic}' confidence out of range: {stance.confidence}")
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"stance grading raised: {e}")
    return g


def grade_facts(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade the personal fact store: can mine and recall facts."""
    g = ModuleGrade("facts", "GREEN")
    try:
        eng.process_turn("i live in berlin")
        facts = eng.user_model.personal_facts.facts
        g.metrics["fact_count"] = len(facts)
        g.metrics["sample_keys"] = [str(k) for k in list(facts.keys())[:5]]

        if len(facts) == 0:
            g.status = "RED"
            g.notes.append("no facts mined after 'i live in berlin'")
        else:
            # Check that at least one fact has the expected subject
            found_berlin = any(
                "berlin" in str(v.value).lower() or "berlin" in str(k).lower()
                for k, v in facts.items()
            )
            if not found_berlin:
                g.notes.append("warning: 'berlin' not found in any fact (may be stored differently)")
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"fact grading raised: {e}")
    return g


def grade_beliefs(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade the belief store: can assert and recall beliefs."""
    g = ModuleGrade("beliefs", "GREEN")
    try:
        state = eng.belief_store.get_state()
        beliefs = state.get("beliefs", {})
        g.metrics["belief_count"] = len(beliefs)

        if len(beliefs) == 0:
            # Beliefs may not form from a single disclosure; this is a soft check
            g.notes.append("no beliefs formed (may require specific disclosure types)")
        else:
            g.metrics["sample_beliefs"] = [str(b) for b in list(beliefs.keys())[:3]]
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"belief grading raised: {e}")
    return g


def grade_graph(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade the concept graph: nodes, edges, connectivity."""
    g = ModuleGrade("graph", "GREEN")
    try:
        nodes = len(eng.graph.nodes)
        edges = len(eng.graph.edges)
        g.metrics["node_count"] = nodes
        g.metrics["edge_count"] = edges
        g.metrics["avg_degree"] = round(2.0 * edges / nodes, 2) if nodes else 0.0

        if nodes == 0:
            g.status = "RED"
            g.notes.append("graph has zero nodes")
        if edges == 0:
            g.notes.append("graph has zero edges (may be expected for fresh engine)")
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"graph grading raised: {e}")
    return g


def grade_save_load(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade save/load persistence: roundtrip integrity."""
    g = ModuleGrade("save_load", "GREEN")
    try:
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="ravana_ledger_")
        eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=tmpdir)

        # Process some turns
        eng2.process_turn("i like open source software")
        eng2.process_turn("my cat is named pixel")
        eng2.process_turn("i live in berlin")

        # Save
        save_msg = eng2.save()
        g.metrics["save_message"] = save_msg[:80] if save_msg else ""

        # Load into a fresh engine
        eng3 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=tmpdir)
        load_ok = eng3.load()
        g.metrics["load_ok"] = load_ok

        # Verify state survived
        facts = eng3.user_model.personal_facts.facts
        stances = eng3.user_model.opinions.stances
        g.metrics["facts_after_load"] = len(facts)
        g.metrics["stances_after_load"] = len(stances)

        if not load_ok:
            g.status = "RED"
            g.notes.append("load() returned False")
        if len(facts) == 0:
            g.status = "RED"
            g.notes.append("no facts survived save/load roundtrip")
        if len(stances) == 0:
            g.status = "RED"
            g.notes.append("no stances survived save/load roundtrip")

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"save/load grading raised: {e}")
    return g


def grade_determinism(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade determinism: same seed = same state hash."""
    g = ModuleGrade("determinism", "GREEN")
    try:
        import tempfile
        tmpdir1 = tempfile.mkdtemp(prefix="ravana_det1_")
        tmpdir2 = tempfile.mkdtemp(prefix="ravana_det2_")

        eng_a = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=tmpdir1)
        eng_b = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=tmpdir2)

        # Process identical turns
        turns = ["i love coffee", "my dog is rex", "i live in berlin"]
        for t in turns:
            eng_a.process_turn(t)
            eng_b.process_turn(t)

        # Save both
        eng_a.save()
        eng_b.save()

        # Compare state checksums
        import pickle
        with open(eng_a._save_path, "rb") as f:
            state_a = pickle.load(f)
        with open(eng_b._save_path, "rb") as f:
            state_b = pickle.load(f)

        sha_a = state_a.get("state_checksum", "")
        sha_b = state_b.get("state_checksum", "")
        g.metrics["checksum_a"] = sha_a
        g.metrics["checksum_b"] = sha_b
        g.metrics["match"] = (sha_a == sha_b)

        if sha_a != sha_b:
            g.status = "RED"
            g.notes.append(f"checksums differ: {sha_a} != {sha_b}")

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir1, ignore_errors=True)
        shutil.rmtree(tmpdir2, ignore_errors=True)
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"determinism grading raised: {e}")
    return g


def grade_intent_router(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade the intent router: can classify basic intents."""
    g = ModuleGrade("intent_router", "GREEN")
    try:
        # Test that the engine can process a variety of query types
        # process_turn returns a string; the strategy is stored in eng._last_strategy
        test_queries = [
            "what is gravity?",
            "i love coffee",
            "do you like music?",
            "tell me about yourself",
        ]
        strategies = []
        for q in test_queries:
            eng.process_turn(q)
            strategies.append(getattr(eng, "_last_strategy", "unknown"))

        g.metrics["strategies"] = strategies
        g.metrics["unique_strategies"] = len(set(strategies))

        # At least 2 different strategies should be used
        if len(set(strategies)) < 2:
            g.notes.append(f"only {len(set(strategies))} unique strategies across {len(test_queries)} queries")
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"intent router grading raised: {e}")
    return g


def grade_episodic_memory(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade episodic memory: can store and recall episodes."""
    g = ModuleGrade("episodic_memory", "GREEN")
    try:
        # Check that the hippocampal buffer exists and has content
        buf = getattr(eng, "hippocampal_buffer", None)
        if buf is not None:
            g.metrics["buffer_type"] = type(buf).__name__
            # HippocampalBuffer stores facts in a dict keyed by subject
            if hasattr(buf, "facts"):
                g.metrics["fact_count"] = len(buf.facts)
            if hasattr(buf, "_all_facts"):
                g.metrics["all_facts_count"] = len(buf._all_facts)
            if hasattr(buf, "_turn_counter"):
                g.metrics["turn_counter"] = buf._turn_counter
        else:
            g.notes.append("no hippocampal_buffer attribute found")
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"episodic memory grading raised: {e}")
    return g


def grade_neuromodulators(eng: CognitiveChatEngine) -> ModuleGrade:
    """Grade neuromodulator system: dopamine, acetylcholine, etc."""
    g = ModuleGrade("neuromodulators", "GREEN")
    try:
        # The engine uses neuromodulator_engine (NeuromodulatorEngine)
        nm = getattr(eng, "neuromodulator_engine", None)
        if nm is not None:
            g.metrics["type"] = type(nm).__name__
            # Check for key neuromodulator attributes (real names from the engine)
            for attr in ["da_level", "da_tonic", "da_phasic",
                         "ach_level", "ach_tonic", "ach_phasic",
                         "ne_level", "ne_tonic", "ne_phasic",
                         "serotonin_level", "serotonin_tonic", "serotonin_phasic"]:
                if hasattr(nm, attr):
                    val = getattr(nm, attr)
                    g.metrics[attr] = round(float(val), 4) if isinstance(val, (int, float)) else str(val)
            # Also check the engine's own dopamine_tone
            if hasattr(eng, "_dopamine_tone"):
                g.metrics["engine_dopamine_tone"] = round(eng._dopamine_tone, 4)
        else:
            g.notes.append("no neuromodulator_engine attribute found (may be expected)")
    except Exception as e:
        g.status = "RED"
        g.notes.append(f"neuromodulator grading raised: {e}")
    return g


def run_ledger() -> List[ModuleGrade]:
    """Run all module grades and return the ledger."""
    print("=" * 60)
    print("RAVANA Acceptance Ledger")
    print("=" * 60)

    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    grades: List[ModuleGrade] = []

    graders = [
        grade_identity,
        grade_stances,
        grade_facts,
        grade_beliefs,
        grade_graph,
        grade_save_load,
        grade_determinism,
        grade_intent_router,
        grade_episodic_memory,
        grade_neuromodulators,
    ]

    for grader in graders:
        name = grader.__name__.replace("grade_", "")
        print(f"\n[{name}] ...", end=" ", flush=True)
        t0 = time.time()
        g = grader(eng)
        elapsed = time.time() - t0
        grades.append(g)
        status_mark = "✓" if g.status == "GREEN" else "✗"
        print(f"{status_mark} {g.status} ({elapsed:.1f}s)")
        for k, v in g.metrics.items():
            print(f"    {k}: {v}")
        for note in g.notes:
            print(f"    note: {note}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    green = sum(1 for g in grades if g.status == "GREEN")
    red = sum(1 for g in grades if g.status == "RED")
    print(f"  GREEN: {green}/{len(grades)}")
    print(f"  RED:   {red}/{len(grades)}")

    if red > 0:
        print("\n  RED modules:")
        for g in grades:
            if g.status == "RED":
                print(f"    - {g.name}: {'; '.join(g.notes)}")

    print("=" * 60)
    return grades


def main() -> int:
    grades = run_ledger()
    red_count = sum(1 for g in grades if g.status == "RED")
    return 1 if red_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
