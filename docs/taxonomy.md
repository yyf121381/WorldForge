# WorldForge Environment Taxonomy

> **分类标准：decay_rate vs recovery_rate 的关系**

---

## 一、分类系统

### A类：可恢复（Recoverable）
```
decay_rate < max_recovery_rate
```
策略可以逆转状态趋势。存在 equilibrium policy。

| 环境 | Decay/步 | Max Recovery/步 | 结论 |
|------|---------|----------------|------|
| emotion_depletion | E: -2~-5, F: +0~+3 | empathize: +3~+8E, -3~-7F | ✅ 可恢复 |
| meaning_crisis | M: -0~-3, C: +1~+4 | guide(有效): +5~+12M | ✅ 可恢复 |

### B类：临界（Fragile）
```
decay_rate ≈ max_recovery_rate
```
策略必须精确执行。微小错误会导致不可逆崩溃。

| 环境 | 特征 |
|------|------|
| — | 尚无对应环境，可设计 | 

### C类：不可恢复（Irreversible Decay）
```
decay_rate > max_recovery_rate
```
任何策略都无法逆转下降趋势。只能延缓崩溃。

| 环境 | Decay/步 | Max Recovery/步 | Net Drift | 预期存活步数 |
|------|---------|----------------|-----------|------------|
| fast_decay | M: -8~-15 | guide(有效): +10~+20 | -5~+12 (期望负值) | ~3-7步 |

---

## 二、实验结果汇总

```
                      情绪耗竭   意义危机   快速衰减
                      ───────   ───────   ──────
RandomAgent            -38       -15        —
RuleAgent              +63       -45        —
V1 (直映射)             —         -43        —
V2 (trust门)           —         -0.7      -21.0
V3 (状态机)            +78       +63.5      -7.3
V4 (紧急通道)           —          —        -8.6
A类✅                  ✅✅      ✅         —
B类⚠️                  —          —          —
C类❌                  —          —          ❌
```

---

## 三、结论

1. **V3（三阶段状态机）在A类环境中表现最优** — 情绪耗竭+78，意义危机+63.5
2. **all agents在C类环境中必然崩溃** — 数学上限，非策略问题
3. **FastDecay应作为"崩溃速度测试"，非"成功率测试"**
4. **V4紧急通道在C类无效** — 因为没有动作能对抗净衰减
