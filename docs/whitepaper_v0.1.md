# WorldForge: A Synthetic Decision Environment Benchmark

## Technical White Paper v0.1

> **2026-07-06** | LingYao AI · WorldForge Team

---

## Abstract

We present WorldForge, a synthetic evaluation framework for testing decision-making agents under dynamic psychological state environments. WorldForge defines three environment classes based on the relationship between decay rate and maximum recovery rate: recoverable (A), fragile (B), and irreversible (C). We evaluate four agent architectures—random baseline, rule-based policy, zero-shot LLM, and a structured three-stage state machine (LingYao V3)—across multiple environments.

**Key finding:** Structured decision architectures significantly improve robustness over zero-shot LLM agents under emotionally dynamic synthetic environments. The advantage is synergistic: no single component (intent probe, trust gate, or state machine) reproduces the full system's performance.

---

## 1. Introduction

Evaluating AI decision-making agents typically requires either expensive real-user data or game-specific environments that don't generalize. We propose an alternative: **synthetic psychological state environments** that model dynamic decision scenarios with clear reward structures and measurable outcomes.

**Core research question:** Can structured policy architectures outperform both heuristic rules and zero-shot LLM agents across heterogeneous synthetic environments?

---

## 2. Benchmark Design

### 2.1 Environment Taxonomy

Environments are classified by the inequality:

```
decay_rate vs max_recovery_rate
```

| Class | Condition | Meaning | Example |
|-------|-----------|---------|---------|
| A (Recoverable) | decay < max_recovery | Policy can reverse trend | emotion_depletion, meaning_crisis |
| B (Fragile) | decay ≈ recovery | Precision required, errors fatal | (future work) |
| C (Irreversible) | decay > recovery | Mathematically unrecoverable | fast_decay |

### 2.2 Action Space (Shared Across All Environments)

Five actions form the common action space:

| Action | Effect (when conditions met) | Effect (when conditions fail) |
|--------|------------------------------|-------------------------------|
| empathize | +trust, +energy, -frustration | Always safe |
| encourage | +willingness, +energy | Low risk |
| guide | +meaning (strong) | Backfires without trust |
| challenge | +meaning (stronger) | Backfires heavily |
| inform | -confusion | Neutral |

### 2.3 Reward Function

```
R = survival_term + recovery_term - instability_penalty

survival = -α·ReLU(frustration - 80) - β·ReLU(20 - energy)
recovery = λ₁·ΔE - λ₂·ΔF + λ₃·ΔS + λ₄·ΔW
instability = δ·(|ΔE| + |ΔF| + |ΔS| + |ΔW|)
```

Default coefficients: α=0.2, β=0.2, λ=[0.3, 0.3, 0.2, 0.4], δ=0.1

### 2.4 Scoring System (WorldForge Score)

```
WS = 0.35·reward_score + 0.25·survival_score + 0.20·stability - 0.20·collapse_rate
```

**Weight sensitivity test:** 10 weight perturbations (α±10%, β±10%, γ±10%, δ±10%, plus extreme variants) produce identical agent rankings. Score function is structurally stable, not a tuning artifact.

---

## 3. Experimental Results

### 3.1 Agent Comparison

| Agent | Emotion Depletion | Meaning Crisis | Total WS |
|-------|-----------------|----------------|---------|
| **LingYao V3** | **+78 (±5)** | **+63 (±10)** | **1.503** |
| RuleAgent | +63 (±3) | -45 (±14) | 0.800 |
| Qwen zero-shot | +50 (±5) | -8 (±68) | 0.763 |
| RandomAgent | -38 (±7) | -15 (±12) | 0.349 |

### 3.2 Ablation Study (Meaning Crisis)

| Variant | Components | Reward | Contribution |
|---------|-----------|--------|-------------|
| Random | None | -15 | Baseline |
| Qwen raw | LLM only | -8 | +7 vs Random |
| V1 | Probe only | -44 | -36 (harmful!) |
| V2 | +Trust gate | -1 | +43 |
| V3 (full) | +Stage machine | +63 | +64 |

**Finding:** No single component explains the full advantage. The trust gate provides the largest individual contribution (+43), but synergy with the state machine (+64) produces the complete system.

### 3.3 External Baseline (Qwen Zero-Shot)

Qwen-3-Max without any structured decision framework achieves +50 on Emotion Depletion and -8 on Meaning Crisis. Adding the V3 state machine overlay does not significantly change performance (+44/—), confirming that the LingYao advantage comes from the *combination* of components, not from the state machine alone.

### 3.4 Probe Robustness

| Test | Variants | Result |
|------|----------|--------|
| Prompt robustness | 3 prompt templates | Consistent intent classification |
| Semantic robustness | 10 paraphrased inputs | 10/10 correct classification |
| Cross-model | 3 LLMs (Chat/R1/Qwen) | Consistent intent distribution |

---

## 4. FastDecay Mathematical Proof

For the FastDecay environment:

```
Prepare guide condition steps × Decay per step > Initial meaning
              3 × 11.5 = 34.5 > 20 (initial mean)
```

**Theorem:** No policy can achieve positive expected return in the FastDecay environment. The environment's decay rate exceeds any action's maximum recovery rate. This environment should serve as a collapse-speed stress test, not a performance metric.

---

## 5. Conclusion

1. **Structured architectures outperform both heuristic rules and zero-shot LLMs** across recoverable synthetic environments.
2. **The advantage is synergistic** — no single component reproduces full system performance.
3. **The WorldForge score is structurally stable** — insensitive to weight perturbations.
4. **Irreversible decay environments define a fundamental boundary** for decision systems.

---

## 6. Limitations & Future Work

- Current environments limited to 2 recoverable + 1 irreversible; more environments needed
- Qwen only zero-shot baseline; GPT/Claude baselines planned
- Probe semantic robustness tested on Chinese only; cross-language testing needed
- Syntehtic environments only; real-world validation pending (2026-09 course deployment)

---

## References

- [WorldForge Repository](https://github.com/yyf121381/WorldForge)
- [LingYao Decision Engine](https://gitee.com/ximenting-laoyang/lingyao) (V3.4)
- Cross-model consistency data: available in repository `/results/`
