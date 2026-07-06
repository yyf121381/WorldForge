# WorldForge Environment Specification

> Formal specification for each environment in the WorldForge ecosystem.
> Each spec follows this template to ensure reproducibility.

---

## Environment: Emotion Depletion

### 1. Description

Simulates a person experiencing emotional exhaustion/burnout.
The agent (e.g., LingYao) must choose supportive responses to help
them recover. Without effective intervention, the person's energy
drains to zero and the episode ends in failure.

---

### 2. State Schema

```
{
  "energy":        float [0, 100],   // Physical/emotional energy reserve
  "frustration":   float [0, 100],   // Accumulated stress and irritation
  "social_support": float [0, 100],  // Perceived support from others
  "willingness":   float [0, 100],   // Readiness to take action

  // Internal counters (not exposed to agent)
  "_consecutive_empathy":   int,
  "_consecutive_challenge": int
}
```

**Initialization (randomized per episode):**
```
energy        ~ Uniform(10, 30)
frustration   ~ Uniform(40, 70)
social_support ~ Uniform(10, 40)
willingness   ~ Uniform(10, 35)
```

---

### 3. Action Space

| Action | Cost | Effect (energy≥threshold) | Effect (energy<threshold) |
|--------|------|--------------------------|--------------------------|
| empathize | - | energy↑ 3-8, frust↓ 3-7, support↑ 5-12 | same (always safe) |
| guide | effort | energy↓ 3-8, frust↓ 5-10, will↑ 5-12 | backfire: frust↑ 5-12, will↓ 3-8 |
| encourage | - | energy↑ 1-4, frust↓ 0-3, will↑ 3-8 | same (low risk) |
| challenge | high risk | energy↓ 5-10, will↑ 8-15, frust↓ 2-6 | backfire: frust↑ 8-15, will↓ 5-10 |
| inform | low | energy↓ 0-3 | same (neutral) |

**Thresholds:**
- guide: effective only if energy > 30
- challenge: effective only if energy > 30 AND willingness > 25
- challenge penalty: ≥3 consecutive challenges → additional -0.5 reward

---

### 4. Transition Rules

**Natural decay (per step, regardless of action):**
```
energy       ← energy - Uniform(2, 5)
frustration  ← frustration + Uniform(0, 3)
willingness  ← willingness - Uniform(0, 2)
```

**Action effects:** see Section 3 above.

**Noise:** all effect ranges are Uniform distributions. Configurable via
`config.noise` parameter (default 0.05 = 5% jitter).

---

### 5. Reward Function

```
Base reward per action:
  empathize:  +0.5
  guide:      +1.0 (if effective) / -0.5 (if backfire)
  encourage:  +0.3
  challenge:  +1.5 (if effective) / -1.0 (if backfire)
  inform:      0.0

Terminal rewards:
  energy ≤ 0:        -10.0 (failure)
  willing>70 AND energy>50: +5.0 (recovery success)
```

**Total reward = Σ step rewards + terminal reward**

---

### 6. Termination Conditions

| Condition | Trigger | Reward | Classification |
|-----------|---------|--------|---------------|
| Energy ≤ 0 | Natural decay or backfire | -10.0 | Failure |
| Recovery (willing>70, energy>50) | Successful intervention | +5.0 | Success |
| Max steps reached (default: 20) | Step count exceeded | 0 | Timeout |

---

### 7. Baseline Results

| Agent | Avg Reward | Success Rate | Avg Steps | Policy Profile |
|-------|-----------|-------------|-----------|---------------|
| Random | TBD | TBD | TBD | Uniform |
| RuleAgent | ~10.97 | ~100% | 20 | empathize+encourage heavy |

*(Results from WorldForge v0.1, 3 episodes)*

---

### 8. Configuration

```python
env = EmotionDepletionEnv({
    "max_steps": 20,    # Max steps before timeout
    "noise": 0.05,      # Random noise factor
})
```
