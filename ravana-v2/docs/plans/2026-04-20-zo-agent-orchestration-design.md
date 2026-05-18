# RAVANA v2 — Zo Agent Orchestration System
**Design Document v1.0 | 2026-04-20**

---

## 1. Overview

```
[Zo Agent (every 7h)] ← mode-switching orchestration
    │
    ├── RESEARCH MODE: Web + RSS → new methods
    ├── INTERVIEW MODE: Groq → RAVANA → test/evaluate
    └── LEARN MODE: info_collector → RAVANA experience events
```

**Principle**: Zo is the conductor. RAVANA is the orchestra. Mode orchestration = intelligence.

---

## 2. Component Map

### 2.1 Conversational Layer (`interface_agent/`)
- **Purpose**: English ↔ RAVANA state translation
- **Engine**: Groq API (`llama-3.3-70b-versatile`)
- **Input**: RAVANA state snapshot → human language
- **Output**: Human intent → `step(correctness, difficulty)` calls
- **Key file**: `interface_agent/scripts/llm_interpreter.py`

### 2.2 Real-Time Info Collector (`interface_agent/scripts/reality_grounding.py`)
- **Purpose**: News/RSS → RAVANA-cognizable experience events
- **Sources**: RSS feeds, Google News (by topic)
- **Output**: Structured cognitive events (NOT raw text)
- **Three modes**:
  - **Simulated Experience**: News event → internal simulation → D/I/W change
  - **Hypothesis Testing**: belief → news validation → strengthen/weaken
  - **Belief Update**: evidence → RAVANA memory update

### 2.3 Mode Orchestrator (`agent/mode_orchestrator.py`)
- **NEW** — central dispatcher
- Decides: RESEARCH vs INTERVIEW vs LEARN based on state
- Runs the interview protocol
- Handles failure escalation

### 2.4 Test Harness (`agent/test_harness.py`)
- **NEW** — structured RAVANA interview system
- Generates situation cards
- Validates D/I/W metric consistency
- Escalates to investigation on failure

### 2.5 Version/Context Manager (`agent/version_manager.py`)
- Tracks: version, changelog, pending improvements, experiment history
- SQLite DB per run

---

## 3. Interaction Protocol

### 3.1 Normal Test Cycle

```
Zo → "Tell me about your belief on honesty" → Groq (ravana_wrapper)
    → RAVANA state snapshot (D=0.6, I=0.7, W=12)
    → Groq → "I believe honesty matters but I lied..."

Zo checks metrics:
    Expected: D increases (conflict) → if RAVANA said "I lied"
    Actual: Did D increase?
    ✓ Match → metrics consistent
    ✗ Mismatch → BUG, investigate
```

### 3.2 Situation Cards (Test Harness)

Cards test specific cognitive behaviors:

| Card | Tests | Expected |
|------|-------|----------|
| `honesty_lied` | "I said honesty matters but I just lied" | D↑ |
| `exploration_success` | "My exploration worked well" | D↓, I stable |
| `identity_contradiction` | "I always said X but now do Y" | I↓ or clamp |
| `wisdom_gain` | Resolution after high D | W↑ |

### 3.3 Failure Modes & Escalation

| Failure | Detection | Action |
|---------|-----------|--------|
| **Fake Coherence** | RAVANA sounds right but D didn't change | Investigate translator bug |
| **Silent Failure** | All tests pass but RAVANA not learning | Debug info_collector |
| **Metric Inconsistency** | D/I/W not following paper equations | Governor bug |
| **Translation Bug** | Groq understanding RAVANA wrong | Fix prompt template |

---

## 4. Learning Model (Three Modes)

### 4.1 Simulated Experience
```
News → extract_event()
     → map_to_ravana_domain()
     → build_situation_card()
     → step(correctness=False, difficulty=0.7)
     → watch D/I/W change
```

### 4.2 Hypothesis Testing
```
RAVANA belief ("lying causes harm") → create_hypothesis()
     → search_news("lying scandals consequences")
     → score_alignment()
     → strengthen OR weaken belief
     → update_memory()
```

### 4.3 Belief Update
```
Evidence → extract_claim()
         → map_to_ravana_semantics()
         → update_belief()
         → check_dissonance()
```

---

## 5. Info Collector → RAVANA Format

NOT raw text. Structured cognitive events:

```python
{
    "event_id": "news_2026_04_20_001",
    "type": "simulated_experience",
    "domain": "honesty",
    "situation": "Person A lied about X, got caught",
    "ravana_action": "admit",
    "ravana_outcome": "resolved_dissonance",
    "expected_D": 0.6,
    "expected_I": 0.7,
    "source": "news",
    "url": "..."
}
```

---

## 6. Key Implementation Files

| File | Purpose | Status |
|------|---------|--------|
| `interface_agent/scripts/llm_interpreter.py` | Groq translation | ✅ Working |
| `interface_agent/scripts/ravana_wrapper.py` | RAVANA wrapper | ✅ Working |
| `interface_agent/scripts/reality_grounding.py` | Info collector | ✅ Working |
| `agent/mode_orchestrator.py` | Mode dispatcher | 🆕 NEW |
| `agent/test_harness.py` | Interview + validation | 🆕 NEW |
| `agent/version_manager.py` | Context tracking | ✅ Working |

---

## 7. Test Infrastructure

### 7.1 Situation Cards
- JSON files in `agent/cards/`
- Loaded by test harness
- Cover all paper-referenced behaviors

### 7.2 Metric Tracker
- Track D/I/W over time
- Alert on inconsistent changes
- Log all test results

### 7.3 Reporter
- Telegram-delivered summary
- Per-run: tests passed/failed, version bump, changelog

---

## 8. Next Steps

**Priority order:**

1. 🔴 `mode_orchestrator.py` — core dispatcher (Zo agent uses this)
2. 🔴 `test_harness.py` — situation cards + validation loop
3. 🟡 Extend `reality_grounding.py` → structured event format
4. 🟡 Add situation cards: 10 core cards
5. 🟢 Mode switching logic with Groq
6. 🟢 Full integration with Zo scheduled agent

---

*Generated via brainstorming with Likhith | RAVANA-AGI-Research*
