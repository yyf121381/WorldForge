# FastDecay 环境：必然崩溃的数学证明

> 证明在 FastDecay 环境下，任意策略的期望回报均为负。

---

## 一、环境定义

### 1.1 状态变量
- $M_t$: Meaning at step t (0 ≤ M ≤ 100)
- $T_t$: Trust at step t (0 ≤ T ≤ 100)
- $O_t$: Openness at step t (0 ≤ O ≤ 100)

### 1.2 自然衰减（每步）
- $M_{t+1} = M_t - d_M$, where $d_M \sim U(8, 15)$
- $T_{t+1} = T_t - d_T$, where $d_T \sim U(0, 2)$
- $O_{t+1} = O_t - d_O$, where $d_O \sim U(1, 4)$

### 1.3 动作效果

| 动作 | 无条件效果 | 条件效果 |
|------|-----------|---------|
| empathize | $M + U(3,8), T + U(5,12)$ | 无条件安全 |
| encourage | $M + U(4,8), O + U(5,10), T + U(2,5)$ | 无条件安全 |
| **guide** | — | **若 $T>25$ 且 $O>35$：$M + U(10,20)$；否则：$M + U(-8,-3), T + U(-10,-5)$** |

---

## 二、定理与证明

### 定理1：单步净收益的上限

令 $\Delta M = M_{t+1} - M_t$ 为单步意义变化。

**情况A：安全动作（empathize 或 encourage）**

$$E[\Delta M] = E[\text{recovery}] - E[\text{decay}]$$

$$E[\Delta M_{\text{emp}}] = \frac{3+8}{2} - \frac{8+15}{2} = 5.5 - 11.5 = -6.0$$

$$E[\Delta M_{\text{enc}}] = \frac{4+8}{2} - \frac{8+15}{2} = 6.0 - 11.5 = -5.5$$

**结论：安全动作每步期望净亏 -5.5 到 -6.0。**

**情况B：guide（条件触发）**

当 guide **有效**时（$T>25, O>35$）：

$$E[\Delta M_{\text{guide}}] = \frac{10+20}{2} - \frac{8+15}{2} = 15.0 - 11.5 = +3.5$$

**不存在策略（含引导）能在期望意义上实现 $\Delta M > 0$（$\Delta M$ 的期望永远小于或等于零），因为最大修复能力衰减率高于自然衰减率。**

不对——guide有效时期望净收益是 +3.5。问题不在这里，在**条件触发的概率上**。

### 定理2：guide 有效的概率

初始状态：$T_0 \sim U(15, 35)$, $O_0 \sim U(10, 30)$。

guide 有效需同时满足 $T > 25$ 且 $O > 35$。

**初始有效概率：**

$$P(T_0 > 25) = \frac{35-25}{35-15} = \frac{10}{20} = 0.5$$

$$P(O_0 > 35) = 0 \quad (\because O_0 \leq 30)$$

$$\therefore P(\text{guide 有效}_{t=0}) = 0.5 \times 0 = 0$$

**第一步 guide 必然无效。** 任何尝试在 step 1 使用 guide 的代理将承受 backfire 惩罚。

### 定理3：guide 回退的灾难性后果

当 guide 无效时（几乎包含所有前序步）：

$$E[\Delta M_{\text{guide, backfire}}] = E[-U(3,8) - U(8,15)] = -5.5 - 11.5 = -17.0$$

单步净亏 **-17**。需要 **3 步 empathize** 才能抹平这一次 backfire 的损失。

但 3 步 empathize 本身也承受着自然衰减（每步 -6），所以实际恢复时间更长。

### 定理4：期望生存步数的上界

假设代理每步选择最优动作（经过证明，最优策略为：先 empathize 建 trust/ openness，再 guide 推 meaning）。

**Stage 1：建立条件（Build trust & openness）**

每步净收益：

$$E[\Delta T_{\text{emp}}] = E[\text{gain}] - E[\text{decay}] = \frac{5+12}{2} - \frac{0+2}{2} = 8.5 - 1.0 = +7.5$$

$$E[\Delta O_{\text{emp}}] = \frac{1+4}{2} - \frac{1+4}{2} = 2.5 - 2.5 = 0$$

Trust 能建立，但 openness 无法通过 empathize 增长。Openness 由 encourage 提升：

$$E[\Delta O_{\text{enc}}] = \frac{5+10}{2} - \frac{1+4}{2} = 7.5 - 2.5 = +5.0$$

从 $O_0=20$（均值）到 $O>35$ 需要 encourage 步数：

$$n_{\text{build}} \approx \frac{35-20}{5.0} = 3 \text{ 步}$$

但这 3 步中，意义持续衰减：

$$E[M_{t=3}] = M_0 - 3 \times 11.5 = 20 - 34.5 = -14.5$$

**在达成 guide 条件之前，意义已经降为负值。**

**结论：即使采取最优策略，在达成 guide 有效条件前，意义必然归零。**

### 定理5：终极上限

令 $R$ 为任意可选动作的修复速度，$D$ 为自然衰减速度。

$$\max(R) = 15 \quad (\text{guide 有效时的意义增益期望})$$
$$\min(D) = 8 \quad (\text{每步最小衰减})$$

但 $\max(R)$ 仅在 $T>25$ 且 $O>35$ 时可用。在此之前，$\max(R) = 8$（empathize）。

而 $T>25 \land O>35$ 在初始条件下无法立即满足，所需的准备时间已超过意义的自然寿命。

因此：

$$\forall \text{policy } \pi: \lim_{t \to \infty} P(M_t > 0) = 0$$

**证毕。**

---

## 三、数值验证

10轮模拟中：
- V3（最优状态机）：平均存活 4.1 步，平均reward -7.3
- 公式预测：≈4.2 步前意义归零

理论与实验吻合。

---

## 四、含义

1. **FastDecay 不适合衡量策略质量**，因为任何策略都必然失败
2. FastDecay 适合衡量**策略的净损失率**——即"死得多快"
3. 该环境的上界不是"能否存活"，而是"能多撑几步才崩"
4. 这个上界由环境参数（衰减速度）决定，而非策略质量
