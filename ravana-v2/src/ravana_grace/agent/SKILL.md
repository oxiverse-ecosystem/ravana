---
name: ravana-agent
description: Zo agent skill for improving RAVANA v2 Interface Agent. Runs every 7 hours, researches new methods via web search, uses brainstorming skill for design, manages version/context system, and tests the interface agent. Uses Groq API (GROQ_API_KEY) for LLM calls.
metadata:
  author: itxlikhith.zo.computer
  version: 1.0.0
---

# RAVANA Agent Skill

Self-improving agent that maintains and enhances the RAVANA v2 Interface Agent.

## Environment

- **Groq API Key**: `GROQ_API_KEY` env var (set in Zo Settings > Advanced > Secrets)
- **RAVANA v2 Path**: `/home/workspace/Projects/ravana-v2`
- **Interface Agent Path**: `/home/workspace/Skills/ravana-interface`
- **Context DB**: `/home/workspace/Skills/ravana-interface/context.db`

## Version & Context System

Maintains a SQLite database (`context.db`) tracking:
- `versions`: agent_version, script_versions, last_updated, changelog
- `context`: current_state, active_experiments, pending_improvements
- `changelog`: timestamp, component, change_type, description, tested

## Workflow (Every 7 Hours)

### 1. Pre-flight: Load Context
```python
# Read context.db to understand current state
# Check what's changed since last run
# Identify pending improvements
```

### 2. Web Research
- Search for new AGI/cognitive architecture methods
- Find updates to Hermes Agent, RAVANA, relevant open source
- Research AI safety, cognitive science, behavioral economics news

### 3. Brainstorm (Use brainstorming skill)
- For each new finding, use brainstorming skill to evaluate
- Assess: relevance to RAVANA, feasibility, expected impact
- Generate improvement candidates

### 4. Implement Improvements
- Apply approved changes to interface agent scripts
- Update version in context DB
- Document in changelog

### 5. Test
```bash
cd /home/workspace/Skills/ravana-interface
python3 scripts/ravana_wrapper.py
python3 scripts/llm_interpreter.py
python3 scripts/reality_grounding.py
python3 scripts/memory_learner.py
python3 scripts/ravana_agent.py --diagnose
```

### 6. Report
- Send Telegram message with results
- Update context DB with new state

## Groq Integration

```python
from groq import Groq
import os

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "..."}]
)
```

## Key Files

- `scripts/ravana_wrapper.py` - RAVANA v2 core wrapper
- `scripts/llm_interpreter.py` - Human ↔ RAVANA translator  
- `scripts/reality_grounding.py` - RSS + Google News
- `scripts/telegram_reporter.py` - Telegram delivery
- `scripts/memory_learner.py` - Persistent learning
- `scripts/ravana_agent.py` - Main orchestrator
- `scripts/version_manager.py` - Version/context system
