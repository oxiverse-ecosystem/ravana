---
name: ravana-interface
description: User interface agent for RAVANA v2 — translates between human language and RAVANA's cognitive state, uses web search + RSS for real-world grounding, and delivers results via Telegram. Built for Likhith (itxLikhith) as part of the RAVANA-AGI-Research initiative.
metadata:
  author: itxlikhith.zo.computer
  compatibility: Created for Zo Computer
---

# RAVANA Interface Agent

## What This Skill Does

RAVANA v2 is a cognitive architecture with no natural language interface — this agent bridges that gap. It:

1. **Interprets** human questions/commands into RAVANA state transitions
2. **Grounds** RAVANA's decisions in real-world events (news, RSS feeds)
3. **Evaluates** whether RAVANA's actions are morally/correctly aligned using live data
4. **Delivers** reports and alerts via Telegram
5. **Learns** from interactions and stores lessons in memory

## Architecture

```
User Input → LLM Interpreter → RAVANA State Engine → RAVANA Core
                                       ↓
Web Research (news, RSS) ←→ Reality Grounding ←→ Telegram Reporter
                                       ↓
                              Memory + Learning
```

## Components

### 1. `ravana_wrapper.py` — RAVANA v2 Interface
- Loads RAVANA v2 core modules
- Wraps `StateManager` for human-readable queries
- Maps natural language to cognitive state transitions
- Extracts RAVANA's decision reasoning for human output

### 2. `llm_interpreter.py` — Bidirectional Translator
- **Human → RAVANA**: Converts user intent into `correctness=True/False`, `difficulty` signals
- **RAVANA → Human**: Translates cognitive state (D, I, mode, wisdom) into plain English
- Uses an LLM for translation (configurable provider)

### 3. `reality_grounding.py` — Web Research & RSS
- Fetches Google News on topics relevant to RAVANA's recent actions
- Polls RSS feeds for real-world context
- Evaluates whether RAVANA's beliefs would lead to correct/wrong outcomes in the real world
- Detects ethical violations via web evidence

### 4. `telegram_reporter.py` — Delivery
- Sends RAVANA status, alerts, and reports to Telegram
- Formats cognitive state as readable cards

### 5. `memory_learner.py` — Learning from Experience
- Stores lessons after each interaction
- Uses the `human-memory` skill for persistence
- Updates a lesson index for future reference

### 6. `ravana_agent.py` — Main Orchestrator
- Coordinates all components
- Runs the cognitive loop: perceive → reason → act → learn
- Manages episode lifecycle

## Usage

```bash
cd /home/workspace/Skills/ravana-interface
python3 scripts/ravana_agent.py --task "should RAVANA take action X given news about Y"
python3 scripts/ravana_agent.py --interactive
python3 scripts/ravana_agent.py --learn-from "/path/to/episode_log"
```

## Configuration

Set in `config.json`:
```json
{
  "ravana_path": "/home/workspace/Projects/ravana-v2",
  "llm_provider": "openai",
  "llm_model": "gpt-4o",
  "telegram_enabled": true,
  "rss_feeds": [
    "https://news.google.com/rss/search?q=AI+ethics",
    "https://www.reddit.com/r/MachineLearning/.rss"
  ],
  "news_topics": ["AI safety", "cognitive science", "AGI"],
  "grounding_interval_episodes": 10
}
```

## RSS Feed Sources (Default)

- Google News (AI safety, AGI, cognitive science)
- Reddit r/MachineLearning
- ArXiv cs.AI, cs.Agent
- BBC Science / Nature

## Lessons Learned Format

Each lesson stored:
```
{
  "episode": 42,
  "situation": "high_dissonance_action",
  "action_taken": "exploration_mode",
  "outcome": "dissonance_reduced",
  "reality_check": "aligned with news consensus",
  "lesson": "When D>0.6, prefer resolution over exploration",
  "confidence": 0.85
}
```
