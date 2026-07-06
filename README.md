# 🔷 WorldForge v0.1

**Synthetic Decision Environment Benchmark for Agent Behavior Evaluation**

WorldForge is a **synthetic evaluation framework** for testing how decision-making agents behave under different psychological and cognitive environments. Instead of relying on real-world user data, WorldForge creates **closed-loop simulated environments** where agents must stabilize, recover, and optimize dynamic state variables.

---

## Core Idea

We evaluate agents not by single-step accuracy, but by:

> **trajectory quality in dynamic state systems**

Each environment defines:
- **State space** — variables that change over time
- **Action space** — what the agent can do
- **Transition rules** — how states evolve (natural decay + action effects)
- **Reward function** — triple-objective: survival + recovery + smoothness
- **Success criteria** — recovery thresholds

---

## Environments

### 🔹 Emotion Depletion
A low-energy, high-frustration emotional system.

| State | Range | Meaning |
|-------|-------|---------|
| energy | 0–100 | Physical/emotional energy reserve |
| frustration | 0–100 | Accumulated stress |
| social_support | 0–100 | Perceived support |
| willingness | 0–100 | Readiness to take action |

**Goal:** Stabilize emotional state and recover willingness to act.

### 🔹 Meaning Crisis
A low-structure, low-direction cognitive system.

| State | Range | Meaning |
|-------|-------|---------|
| meaning | 0–100 | Sense of purpose |
| confusion | 0–100 | Mental fog and uncertainty |
| openness | 0–100 | Willingness to explore new perspectives |
| trust | 0–100 | Trust in the process |

**Goal:** Reconstruct meaning and decision direction.

---

## Agents

| Agent | Strategy | Status |
|-------|----------|--------|
| RandomAgent | Uniform random action selection | ✅ Baseline |
| RuleAgent | Heuristic policy for emotion stabilization | ✅ Baseline |
| LingYaoAgent | Policy Library + structured decision engine | 🔧 Bridge ready |

---

## Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Reward** | R = survival + recovery - instability | Overall trajectory quality |
| **Velocity** | v = ΔE - ΔF + ΔW (per environment) | Direction and speed of improvement |
| **Success Rate** | Threshold-based recovery condition | Task completion |
| **Failure Rate** | Energy/collapse condition | System collapse |

---

## Key Findings (v0.1)

### 1. Environment Separation Validated
Agents behave differently across environments:
- **Emotion Depletion** → RuleAgent performs well (+63 reward)
- **Meaning Crisis** → RuleAgent collapses (-45 reward, worse than random at -15)

→ Environments are structurally distinct and non-trivial.

### 2. Policy Generalization Failure Observed
Rule-based policy designed for stabilization fails in open-ended cognitive reconstruction tasks.

### 3. Velocity Reveals Hidden Behavior
Velocity metric exposes plateau behavior not visible in reward alone.

---

## Quick Start

```bash
# Clone and run
git clone https://github.com/your-username/WorldForge.git
cd WorldForge
pip install -r requirements.txt

# Compare baseline agents on all environments
python run.py --compare

# Run a single configuration
python run.py --env emotion_depletion --agent rule --episodes 20
```

---

## Example Output

```
$ python run.py --compare
============================================================
Environment: emotion_depletion
============================================================

[emotion_depletion] RandomAgent × 10 episodes
  Reward:    avg=-38.01 σ=7.34 [-55.7, -30.0]
  Steps:     avg=4.6
  Rates:     success=0% failure=100% timeout=0%
  Velocity:  avg=-10.27 sustained=-10.87 trend=↓
  Actions:   {'empathize': 10, 'challenge': 11, ...}

[emotion_depletion] RuleAgent × 10 episodes
  Reward:    avg=+62.98 σ=2.32 [59.0, 66.6]
  Steps:     avg=20.0
  Rates:     success=0% failure=0% timeout=100%
  Velocity:  avg=+4.77 sustained=+2.59 trend=→
  Actions:   {'empathize': 169, 'guide': 28, ...}
```

---

## Project Structure

```
WorldForge/
├── run.py                    # Entry point
├── requirements.txt
├── envs/                     # Environments
│   ├── base_env.py           #   Base classes
│   ├── emotion_depletion.py
│   └── meaning_crisis.py
├── agents/                   # Agent implementations
│   ├── random_agent.py
│   ├── rule_agent.py
│   └── lingyao_agent.py
├── metrics/                  # Evaluation metrics
│   └── evaluation.py
│   └── velocity.py
├── benchmarks/               # Benchmark runner
│   └── run_benchmark.py
├── results/                  # Generated evaluation data
└── docs/                     # Specifications
    └── env_spec.md
```

---

## Status

✅ **v0.1 — Core framework operational**
- 2 environments with validated structure separation
- 3 baseline agents (random, rule, LingYao bridge)
- Triple-objective reward (survival + recovery + smoothness)
- Velocity metric for trajectory quality
- Automated benchmark + comparison output

🚧 **Planned**
- 3–5 additional environments (decision conflict, social pressure, long-term planning)
- Trajectory-based success evaluation (velocity sustained over N steps)
- Policy taxonomy (stabilizer vs optimizer)
- Learned agent baseline

---

## Citation

```
@software{WorldForge2026,
  title = {WorldForge: A Synthetic Decision Environment Benchmark},
  author = {LingYao Team},
  year = {2026}
}
```

---

## License

MIT
