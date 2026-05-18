# K1.3 Fix Summary: Context-Aware Exploration

## The Bug (Dual Discovery)

### 1. EP24 Deterministic Death (Expected)
K1.2's **10-step exploration floor** triggered exploration regardless of energy state:
```python
if self.steps_since_explore > 10:
    self.steps_since_explore = 0
    return AgentAction.EXPLORE  # ← Always explores, even when dying
```

This caused agents to explore at exactly EP24 when energy was depleted, resulting in deterministic death.

### 2. The Hidden Enum Bug (Critical Discovery)
**Root cause**: Two different `AgentAction` enums existed:
- `core_k0.agent_loop_k1_2.AgentAction`
- `experiments_k0.resource_env.AgentAction`

Same values, **different classes** → `action == AgentAction.EXPLORE` always returned `False` in the environment!

**Result**: All exploration "succeeded" but energy gains were lost. Only metabolism (-0.02) applied.

## The Fix: K1.3

### Architecture Changes

```python
class ExplorationMode(Enum):
    DISABLED = "disabled"   # Energy bleeding — conserve only
    GUARDED = "guarded"     # Stable margin — limited exploration  
    ENABLED = "enabled"     # Healthy — full exploration allowed
```

### Key Logic

```python
def _get_exploration_mode(self) -> ExplorationMode:
    E = self.state.energy_estimate
    trend = self.state.get_energy_trend(window=5)
    survival_buffer = self.base_metabolism * 3  # 0.06
    
    # 🔴 DISABLED: Bleeding energy or below buffer
    if E < survival_buffer * 2 or trend < -0.05:
        return ExplorationMode.DISABLED
    
    # 🟡 GUARDED: Marginal but stable
    if E < survival_buffer * 4 or trend < 0:
        return ExplorationMode.GUARDED
    
    # 🟢 ENABLED: Healthy
    return ExplorationMode.ENABLED
```

### Feasibility Check (The Fix for EP24)

```python
if self.steps_since_explore > 10:
    feasible, reason = self._exploration_is_feasible()
    if feasible:
        return AgentAction.EXPLORE
    else:
        # K1.3 FIX: Don't explore if it would kill us
        self.steps_since_explore = 0
        return AgentAction.CONSERVE  # Fall through to safe policy
```

## Results (20-Run Test)

| Metric | K1.2 | K1.3 | Change |
|--------|------|------|--------|
| **Survival Rate** | 50% (10/20) | **65%** (13/20) | +30% ↑ |
| **Exploration Frequency** | 31.5/run | **17.1/run** | -46% ↓ |
| **Exploration Success** | 0%* → 66% | 66% | Fixed* |
| **EP24 Deaths** | 0** | 0 | Eliminated |

*Before enum fix (hidden bug)  
**After enum fix (scattered deaths)

## Key Insight

**K1.2 taught**: "Don't sit still and die"  
**K1.3 teaches**: "Don't jump off a cliff just because you're scared"

That distinction is the birth of judgment. The agent now:
1. **Checks affordability** before gambling (energy > buffer)
2. **Reads the trend** (negative slope = bleeding)
3. **Tracks success rates** (abort if exploration failing)
4. **Conserves when dying** instead of panic-exploring

## Files Modified

- `core_k0/agent_loop_k1_2.py` — Fixed enum import
- `core_k0/agent_loop_k1_3.py` — New context-aware implementation
- `core_k0/test_k1_3.py` — Validation test (20-run comparison)

## Execution

```bash
cd <repo_root>
python core_k0/test_k1_3.py
```

Exit code 0 = fix verified (survival improved + exploration reduced)
