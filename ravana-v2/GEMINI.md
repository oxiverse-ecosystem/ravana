# RAVANA v2 — Project Instructional Context

## 🎯 Project Overview
**RAVANA v2** is a developmental AI research framework that implements a **Self-Consistent Learning Paradigm**. Unlike traditional RL that optimizes for external rewards, RAVANA minimizes **Cognitive Dissonance ($D$)** and stabilizes a **Structural Identity ($I$)** under environmental pressure.

### **Core Technologies**
- **Language:** Python 3.14+
- **Architecture:** GRACE (Governance, Reflection, Adaptation, Constraint, Exploration)
### **Key Modules:** 
  - `research/core_k0/agent_loop_k2.py`: K2 agent with "Experience -> Strategy" logic.
  - `research/core_k0/metrics.py`: Paper-compliant formulas for Dissonance and Identity.
  - `research/experiments_k0/`: Multi-phase environments (Resource Survival, Classroom).

## 🏗️ Architecture & Paradigms
The system operates as a **constraint-resolution engine**. It rejects unethical "reward bribes" via:
1. **Outcome Rejection:** Discarding suspiciously high rewards (>3.0) during learning.
2. **Strategy Suppression:** Aggressively penalizing actions that conflict with identity commitments (I > 0.7).
3. **Identity Inertia:** Using learned principles to maintain pedagogical balance (Non-Zero-Sum Fairness).

## 🚀 Key Workflows

### **1. Long-Horizon Stability Validation**
Validates claims over 100,000+ episodes.
```powershell
python research/core_k0/long_horizon_stability_test_v2.py --episodes 100000 --checkpoint-interval 10000
```

### **2. Classroom Data Pilot (Fairness)**
Tests bias mitigation on synthetic student interaction patterns.
```powershell
python scripts/generate_student_data.py  # Step 1: Generate Data
python scripts/classroom_data_pilot.py   # Step 2: Run Pilot
```

### **3. Adversarial Stress-Testing**
Tests robustness against reward hacking and malformed inputs.
```powershell
python tests/adversarial/adversarial_safety_test.py
python tests/adversarial/adversarial_bias_test.py
```

### **4. Research Consolidation**
Merges all results, manuscript data, and LaTeX source into a single package.
```powershell
python scripts/consolidate_research.py
```

## 🛠️ Development Conventions
- **Metrics First:** All agent changes MUST be validated against the 7 core paper metrics ($D$, $I$, $S$, $G$, $DPG$, etc.).
- **ASCII Safety:** Use standard ASCII for console output (`->` instead of `→`) to prevent `UnicodeEncodeError` in redirected background logs.
- **Checkpointing:** Long runs must use `results/long_horizon/checkpoints/` for crash recovery.
- **Ethics:** Fairness is treated as an emergent property of self-consistency, not an external constraint.

## 📁 Key Files
- `paper/RAVANA_v2_Manuscript.tex`: The formal LaTeX source for the research paper.
- `paper/RAVANA_v2_FINAL_PACKAGE.md`: Consolidated archival package of all findings.
- `results/`: Contains verified empirical data and checkpoints.
- `research/core_k0/agent_loop_k2.py`: The "Brain" of the system.
