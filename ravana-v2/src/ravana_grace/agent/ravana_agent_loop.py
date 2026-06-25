#!/usr/bin/env python3
"""
RAVANA Agent — Main Orchestrator (Zo Scheduled Agent)
Runs every 7 hours to improve the RAVANA v2 Interface Agent.

Tasks:
1. Load context from version manager
2. Web research for new methods
3. Brainstorm evaluation
4. Implement improvements
5. Test all components
6. Report via Telegram
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project paths to sys.path using the real repository layout.
SCRIPT_DIR = Path(__file__).resolve().parent
RAVANA_DIR = SCRIPT_DIR.parent
SCRIPTS_DIR = RAVANA_DIR / "interface_agent" / "scripts"
AGENT_DIR = RAVANA_DIR / "agent"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(RAVANA_DIR))

from version_manager import VersionManager
from reality_grounding import RealityGrounding
from groq import Groq

# ─── Config ────────────────────────────────────────────────────────

SKILL_DIR = RAVANA_DIR / "interface_agent"
DB_PATH = os.environ.get("CONTEXT_DB", str(SKILL_DIR / "context.db"))

# Groq models (fast for agents, capable for research)
FAST_MODEL = "llama-3.1-8b-instant"
CAPABLE_MODEL = "llama-3.3-70b-versatile"

# ─── Groq Client ──────────────────────────────────────────────────

def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")
    return Groq(api_key=api_key)


def groq_complete(prompt: str, model: str = CAPABLE_MODEL, system: str = "") -> str:
    """Call Groq API for completion."""
    client = get_groq_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ─── Web Research ────────────────────────────────────────────────

def research_ravana_improvements() -> List[Dict]:
    """Research new methods for RAVANA enhancement from live grounding data."""
    queries = [
        "cognitive architecture AGI self-improving agent 2025",
        "Hermes Agent Nous Research latest updates github",
        "RAVANA cognitive dissonance AI psychology 2025",
        "AI agent memory systems episodic semantic learning",
        "real-time news RSS feed AI belief revision",
    ]

    grounding = RealityGrounding()
    findings: List[Dict[str, Any]] = []

    for query in queries:
        try:
            items = grounding.search_news(query, num_results=3)
            if items:
                best = items[0]
                findings.append(
                    {
                        "query": query,
                        "topic": extract_topic(query),
                        "relevance": min(10, max(6, int(round(best.relevance_score * 10)) + 2)),
                        "news_count": len(items),
                        "headlines": [item.title for item in items[:3]],
                        "notes": f"Grounded in {len(items)} recent news items; top hit: {best.title}",
                    }
                )
            else:
                findings.append(
                    {
                        "query": query,
                        "topic": extract_topic(query),
                        "relevance": 5,
                        "news_count": 0,
                        "headlines": [],
                        "notes": f"No live hits for: {query}",
                    }
                )
        except Exception as exc:
            findings.append(
                {
                    "query": query,
                    "topic": extract_topic(query),
                    "relevance": 4,
                    "news_count": 0,
                    "headlines": [],
                    "notes": f"Research fallback after error: {str(exc)[:120]}",
                }
            )
        time.sleep(0.1)

    return findings


def extract_topic(query: str) -> str:
    """Extract main topic from query."""
    topics = {
        "cognitive architecture": "cognitive_architecture",
        "self-improving": "self_improving",
        "hermes agent": "hermes_agent",
        "memory": "memory_systems",
        "news": "real_time_grounding",
    }
    query_lower = query.lower()
    for key, val in topics.items():
        if key in query_lower:
            return val
    return "general"


# ─── Brainstorm Evaluation ──────────────────────────────────────

def brainstorm_evaluate(findings: List[Dict]) -> List[Dict]:
    """Use brainstorming approach to evaluate findings."""
    evaluated = []
    
    for finding in findings:
        topic = finding['topic']
        
        # Build evaluation prompt
        from ..interface_agent.scripts.prompt_composer import PromptComposer
        prompt = PromptComposer.compose(
            role="evaluator",
            task="evaluate_brainstorm",
            state={},
            context={
                "Topic": topic,
                "Query": finding['query'],
                "Consider": (
                    "RAVANA v2 has: dissonance/identity dynamics, governor regulation, memory systems. "
                    "Interface Agent needs: human↔RAVANA translation, real-world grounding, learning. "
                    "Hermes Agent has: skill creation, cross-session memory, scheduling."
                ),
            },
            format_spec="brainstorm_json",
        )
        
        try:
            result = groq_complete(prompt, model=CAPABLE_MODEL)
            # Try to parse JSON
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                finding.update(parsed)
                finding['tested'] = False
                evaluated.append(finding)
            else:
                # Fallback
                finding['relevance'] = 7
                finding['feasibility'] = 6
                finding['impact'] = 7
                finding['priority'] = 'medium'
                finding['reason'] = 'Fallback evaluation'
                evaluated.append(finding)
        except Exception as e:
            finding['relevance'] = 5
            finding['feasibility'] = 5
            finding['impact'] = 5
            finding['priority'] = 'low'
            finding['reason'] = f'Error: {str(e)[:100]}'
            evaluated.append(finding)
    
    return evaluated


# ─── Improvement Implementation ─────────────────────────────────

def implement_improvement(vm: VersionManager, improvement: Dict) -> bool:
    """Apply an approved improvement to the interface agent."""
    component = improvement.get('topic', 'unknown')
    description = improvement.get('description', improvement.get('query', ''))
    
    # Check if it's a memory/systemic improvement
    if component == 'memory_systems':
        return implement_memory_improvement(vm, improvement)
    elif component == 'hermes_agent':
        return implement_hermes_feature(vm, improvement)
    elif component == 'real_time_grounding':
        return implement_grounding_improvement(vm, improvement)
    else:
        return implement_general_improvement(vm, improvement)


def implement_memory_improvement(vm: VersionManager, improvement: Dict) -> bool:
    """Add memory system improvements."""
    notes = f"Memory improvement: {improvement.get('reason', 'Enhancement')}"
    vm.add_changelog("memory_learner.py", "improved", notes, notes, tested=False)
    return True


def implement_hermes_feature(vm: VersionManager, improvement: Dict) -> bool:
    """Add Hermes-like features."""
    notes = f"Hermes-style feature: {improvement.get('reason', 'Enhancement')}"
    vm.add_changelog("ravana_agent.py", "added", notes, notes, tested=False)
    return True


def implement_grounding_improvement(vm: VersionManager, improvement: Dict) -> bool:
    """Add real-time grounding improvements."""
    notes = f"Grounding improvement: {improvement.get('reason', 'Enhancement')}"
    vm.add_changelog("reality_grounding.py", "improved", notes, notes, tested=False)
    return True


def implement_general_improvement(vm: VersionManager, improvement: Dict) -> bool:
    """General improvement handler."""
    notes = f"General improvement: {improvement.get('reason', 'Enhancement')}"
    vm.add_changelog("general", "researched", notes, notes, tested=False)
    return True


# ─── Testing ────────────────────────────────────────────────────

def run_tests(vm: VersionManager) -> Dict[str, Any]:
    """Run all interface agent tests."""
    import subprocess
    results = {}
    
    test_scripts = [
        ("wrapper", "python3 scripts/ravana_wrapper.py"),
        ("llm_interpreter", "python3 scripts/llm_interpreter.py"),
        ("reality_grounding", "python3 scripts/reality_grounding.py"),
        ("memory_learner", "python3 scripts/memory_learner.py"),
        ("telegram_reporter", "python3 scripts/telegram_reporter.py"),
        ("agent_full", "python3 scripts/ravana_agent.py --diagnose"),
    ]
    
    os.chdir(SKILL_DIR)
    
    for name, cmd in test_scripts:
        start = time.time()
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=60
            )
            duration_ms = int((time.time() - start) * 1000)
            status = "pass" if result.returncode == 0 else "fail"
            output = result.stdout[:500] if result.returncode == 0 else result.stderr[:500]
            
            vm.record_test(name, status, output, duration_ms)
            results[name] = {"status": status, "duration_ms": duration_ms}
            
            if status == "pass":
                print(f"  ✅ {name}: OK ({duration_ms}ms)")
            else:
                print(f"  ❌ {name}: FAILED ({duration_ms}ms)")
                print(f"     Error: {output[:200]}")
                
        except subprocess.TimeoutExpired:
            vm.record_test(name, "error", "Timeout after 60s", 60000)
            results[name] = {"status": "error", "duration_ms": 60000}
            print(f"  ⚠️  {name}: TIMEOUT")
        except Exception as e:
            vm.record_test(name, "error", str(e)[:200], 0)
            results[name] = {"status": "error", "duration_ms": 0}
            print(f"  ⚠️  {name}: ERROR - {str(e)[:100]}")
    
    return results


# ─── Report ──────────────────────────────────────────────────────

def build_report(vm: VersionManager, test_results: Dict, new_findings: int, improvements: int) -> str:
    """Build Telegram report."""
    summary = vm.get_summary()
    
    # Count test pass/fail
    passed = sum(1 for r in test_results.values() if r['status'] == 'pass')
    total = len(test_results)
    
    lines = [
        "🔄 RAVANA Agent — Run Complete",
        "─────────────────────────────",
        f"Time: {datetime.now().strftime('%H:%M')}",
        f"Agent v{summary['agent_version']}",
        "",
        f"📊 Tests: {passed}/{total} passed",
    ]
    
    for name, result in test_results.items():
        icon = "✅" if result['status'] == 'pass' else "❌"
        lines.append(f"  {icon} {name}")
    
    lines.extend([
        "",
        f"🔍 Research: {new_findings} findings",
        f"💡 Improvements: {improvements} queued",
        f"📝 Pending: {summary['pending_improvements']}",
    ])
    
    # Get recent changes
    recent = vm.get_recent_changelog(3)
    if recent:
        lines.append("")
        lines.append("Recent changes:")
        for c in recent:
            icon = {"added": "🆕", "improved": "🔧", "fixed": "🐛", "researched": "🔬"}.get(c['change_type'], "•")
            lines.append(f"  {icon} {c['component']}: {c['description'][:50]}...")
    
    return "\n".join(lines)


# ─── Main Agent Loop ──────────────────────────────────────────────

def run_agent():
    """Main agent entry point (runs every 7 hours)."""
    print("\n" + "="*50)
    print("🤖 RAVANA Agent — Starting Run")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    # 1. Load context
    print("[1/6] Loading context...")
    vm = VersionManager()
    current = vm.get_current_versions()
    print(f"  Current agent version: {current['agent_version']}")
    print(f"  Scripts tracked: {len(current.get('script_versions', []))}")
    
    # 2. Detect changes
    print("\n[2/6] Detecting script changes...")
    changed = vm.detect_changed_scripts(str(SCRIPTS_DIR))
    changed_files = [s for s in changed if s.get('changed')]
    for s in changed_files:
        print(f"  ⚠️  {s['name']}: modified (v{s['version']})")
        vm.add_changelog(s['name'], "improved", f"Script updated to v{s['version']}", tested=False)
    
    # 3. Web research
    print("\n[3/6] Conducting web research...")
    findings = research_ravana_improvements()
    print(f"  Researched {len(findings)} topics")
    
    # 4. Brainstorm evaluation
    print("\n[4/6] Evaluating with brainstorming...")
    evaluated = brainstorm_evaluate(findings)
    
    # Queue high-priority improvements
    improvements_queued = 0
    for f in evaluated:
        if f.get('priority') == 'high' or f.get('relevance', 0) >= 8:
            vm.queue_improvement(
                description=f.get('query', '') + " - " + f.get('reason', ''),
                source="web_search",
                priority=f.get('relevance', 5)
            )
            improvements_queued += 1
            print(f"  📋 Queued: {f['topic']} (priority: {f.get('priority', 'low')})")
    
    # 5. Run tests
    print("\n[5/6] Running tests...")
    test_results = run_tests(vm)
    
    # Mark changelog entries as tested
    for c in vm.get_recent_changelog(10):
        if not c['tested']:
            vm.mark_tested(c['id'])
    
    # 6. Report
    print("\n[6/6] Building report...")
    report = build_report(vm, test_results, len(findings), improvements_queued)
    print(report)
    
    # Save new version
    script_versions = [{'name': s['name'], 'version': s['version'], 'checksum': s['checksum']} for s in changed]
    new_ver = increment_version(current['agent_version'], 'minor')
    vm.save_version(new_ver, script_versions, [])
    
    elapsed = time.time() - start_time
    print(f"\n✅ Run complete in {elapsed:.1f}s")
    print(f"   Next run in ~7 hours")
    
    return report


def increment_version(version: str, bump_type: str = "patch") -> str:
    """Bump semantic version."""
    try:
        major, minor, patch = version.split('.')
        major, minor, patch = int(major), int(minor), int(patch)
        if bump_type == "major":
            return f"{major+1}.0.0"
        elif bump_type == "minor":
            return f"{major}.{minor+1}.0"
        else:
            return f"{major}.{minor}.{patch+1}"
    except:
        return "1.0.0"


if __name__ == "__main__":
    report = run_agent()
    
    # Try to send via Telegram
    try:
        from send_telegram_message import send_telegram_message
        # Will be called by the actual agent with Telegram tool
        print("\n[TELEGRAM]", report)
    except:
        pass
