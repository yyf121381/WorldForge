# 🔷 WorldForge v0.1

**Synthetic Decision Environment Benchmark for Agent Policy Evaluation**

We evaluate decision-making agents under dynamic psychological state transitions. The core question:

> *Can a policy trained/stabilized for one environment generalize to a structurally different one?*

**Key finding from v0.1:** Rule-based policies show strong performance in stabilization tasks but fail in open-ended cognitive reconstruction tasks — *worse than random*.

---

## Benchmark Structure

```
┌─────────────────────────────────────────────┐
│            Agent Policy                      │
│  (random / rule-based / learned)            │
└──────────────────┬──────────────────────────┘
                   │ action
                   ▼
┌─────────────────────────────────────────────┐
│  Environment                                 │
│  • State transition: sₜ₊₁ = f(sₜ, aₜ)       │
│  • Natural decay + action effects + noise    │
│  • Triple-objective reward function          │
└──────────────────┬──────────────────────────┘
                   │ state, reward
                   ▼
┌─────────────────────────────────────────────┐
│  Evaluation                                  │
│  • Survival success (stayed alive)           │
│  • Stability success (sustained improvement) │
│  • Recovery success (reached threshold)      │
│  • Velocity (trajectory quality)             │
└─────────────────────────────────────────────┘
```

---

## Environments

### 🔹 Emotion Depletion
A low-energy, high-frustration emotional system. The agent must stabilize and recover.

| State | Range | Initial | Meaning |
|-------|-------|---------|---------|
| `energy` | 0–100 | 10–30 | Physical/emotional reserve |
| `frustration` | 0–100 | 40–70 | Accumulated stress |
| `social_support` | 0–100 | 10–40 | Perceived support |
| `willingness` | 0–100 | 10–35 | Readiness to act |

**Actions:** empathize / guide / encourage / challenge / inform

**Transition:** Natural decay (−2 to −5 energy, +0 to +3 frustration per step) + action effects.

### 🔹 Meaning Crisis
A low-structure cognitive system. The agent must reconstruct meaning and direction.

| State | Range | Initial | Meaning |
|-------|-------|---------|---------|
| `meaning` | 0–100 | 5–25 | Sense of purpose |
| `confusion` | 0–100 | 50–80 | Mental fog |
| `openness` | 0–100 | 10–30 | Receptivity to new ideas |
| `trust` | 0–100 | 10–30 | Trust in the process |

**Actions:** same as above (environment-agnostic action space)

---

## Reward Function

Triple-objective:

```
R = survival_term + recovery_term - instability_penalty

survival = -α·ReLU(frustration - 80) - β·ReLU(20 - energy)
recovery = λ₁·ΔE - λ₂·ΔF + λ₃·ΔS + λ₄·ΔW
instability = δ·(|ΔE| + |ΔF| + |ΔS| + |ΔW|)
```

Default coefficients: α=0.2, β=0.2, λ=[0.3, 0.3, 0.2, 0.4], δ=0.1

---

## Success Hierarchy (Three-Layer)

| Level | Criterion | What it measures |
|-------|-----------|-----------------|
| **Survival** | Final energy ≥ 20 and frustration ≤ 80 | Agent kept the system alive |
| **Stability** | Sustained velocity > 3.0 (last 5 steps) | Agent drove consistent improvement |
| **Recovery** | Energy ≥ 50, frustration ≤ 30, willingness ≥ 70 | Agent achieved full recovery |

This layered structure prevents "survival without progress" from being mistaken for success.

---

## Evaluation Protocol

**Each episode:**
1. Environment resets with randomized initial state (see ranges above)
2. Agent observes state → selects action → environment transitions
3. Repeat until termination (max 25 steps, or collapse, or recovery)
4. Reward accumulated, velocity computed, success layer assigned

**Benchmark run:**
- 10 episodes per (agent, environment) pair
- Results aggregated: mean/std reward, success rates, velocity metrics
- Action profile logged for policy analysis

---

## Baselines

| Agent | Strategy | Expected Behavior |
|-------|----------|------------------|
| **RandomAgent** | Uniform random action selection | Collapses quickly (lower bound) |
| **RuleAgent** | Heuristic: energy < 20 → empathize, energy > 40 → guide, etc. | Survives but plateaus |
| **LingYaoAgent** | Policy Library + structured decision engine | Unknown (hypothesis: better balance) |

---

## Key Findings

### 1. Environment Separation Validated
Policies behave fundamentally differently across environments:

| Agent | Emotion Depletion | Meaning Crisis |
|-------|------------------|----------------|
| RandomAgent | −38 avg reward, 100% collapse | −15 avg reward, 100% collapse |
| RuleAgent | **+63 avg reward**, 0% collapse | **−45 avg reward**, 100% collapse |

**RuleAgent outperforms random in emotion depletion (+63 vs −38) but underperforms random in meaning crisis (−45 vs −15).**

### 2. Policy Generalization Failure
The rule-based policy designed for emotional stabilization does not transfer. In meaning crisis, it spams `guide` (which backfires when openness is low) and avoids actions that build trust first.

### 3. Velocity Reveals Plateau Behavior
RuleAgent in emotion depletion shows avg velocity +4.77 but late-stage sustained velocity drops to +2.59 — the system is alive but no longer improving. This pattern is invisible from reward alone.

---

## Quick Start

```bash
# Clone
git clone https://github.com/yyf121381/WorldForge.git
cd WorldForge
pip install -r requirements.txt

# Compare baseline agents on all environments
python run.py --compare

# Single configuration
python run.py --env emotion_depletion --agent rule --episodes 20 --max-steps 30
```

### Example Output

```
[emotion_depletion] RuleAgent × 10 episodes
  Reward:    avg=+62.98 σ=2.32 [59.0, 66.6]
  Steps:     avg=20.0
  Success:   survival=100%  stability=80%  recovery=0%
  Failure:   collapse=0%  timeout=100%
  Velocity:  avg=+4.77  sustained=+2.59  trend=→
  Actions:   {'empathize': 169, 'guide': 28, 'encourage': 3}
```

---

## Repository Structure

```
WorldForge/
├── run.py                    # Entry point: `python run.py --compare`
├── envs/                     # Environments (emotion_depletion, meaning_crisis)
├── agents/                   # Baseline agents (random, rule, LingYao bridge)
├── metrics/                  # Evaluation (reward, velocity, three-layer success)
├── benchmarks/               # Episode runner + comparison engine
├── docs/                     # Environment specifications
└── results/                  # Generated benchmark outputs
```

---

## Planned

- 3–5 additional environments (decision conflict, social pressure, long-term planning)
- Trajectory-based success evaluation
- LingYao Agent benchmark results
- Learned agent baseline

---

## Status

✅ **v0.1 — Core framework operational.** 2 environments, 3 baseline agents, layered evaluation.

---

## License

MIT
