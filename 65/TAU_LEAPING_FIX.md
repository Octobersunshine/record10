# Tau-Leaping 负种群问题修复

## 问题描述

Tau-Leaping 算法通过在一个时间步长内同时执行多个反应来加速随机模拟。然而，如果步长过大，可能导致某些物种的种群数量变为负数，导致模拟崩溃或产生物理上无意义的结果。

### 为什么会产生负种群？

1. **Poisson 采样的无界性**：Poisson 分布理论上可以产生任意大的数，即使均值适中。

2. **大步长**：当 tau 太大时，反应计数 k 可能超过可用分子数。

3. **组合效应**：多个反应同时消耗同一物种。

4. **倾向函数变化**：在跳跃过程中，倾向函数发生变化，导致初始采样不再适用。

## 解决方案

### 1. Poisson 边界检查

**核心思想**：在采样前计算每个反应可能发生的最大次数。

```python
def _compute_max_reactions(self, state: np.ndarray, tau: float) -> np.ndarray:
    num_reactions = len(self.reactions)
    max_k = np.full(num_reactions, float('inf'))
    
    for j in range(num_reactions):
        stoich = self.reactions[j].stoichiometry
        for i in range(len(state)):
            if stoich[i] < 0:  # 消耗物种i
                max_possible = state[i] // abs(stoich[i])
                max_k[j] = min(max_k[j], max_possible)
    
    return max_k
```

然后在采样时限制：
```python
k[j] = np.random.poisson(lambda_j)
k[j] = int(min(k[j], max_k[j]))
```

### 2. 接近耗尽时的二项采样

当 `max_k[j]` 很小时（< nc），Poisson 假设不再适用，切换到二项分布：

```python
if lambda_j > max_k[j] and max_k[j] < self.nc:
    p = lambda_j / (lambda_j + max_k[j])
    k[j] = np.random.binomial(int(max_k[j]), p)
```

### 3. 跳跃后修正

即使有上述预防措施，仍可能出现负数。检测并修正：

```python
def _apply_state_update(self, state: np.ndarray, k: np.ndarray):
    new_state = state.copy()
    
    # 应用更新
    for j in range(len(self.reactions)):
        if k[j] > 0:
            new_state += k[j] * self.reactions[j].stoichiometry
    
    # 检测负数
    negative_mask = new_state < 0
    if negative_mask.any():
        # 反向计算需要减少的反应次数
        for j in range(len(self.reactions)):
            stoich = self.reactions[j].stoichiometry
            for i in range(len(new_state)):
                if stoich[i] < 0 and new_state[i] < 0:
                    over_consumed = abs(new_state[i])
                    max_reduce = over_consumed // abs(stoich[i])
                    k[j] = max(0, k[j] - max_reduce)
        
        # 重新应用修正后的反应计数
        new_state = state.copy()
        for j in range(len(self.reactions)):
            if k[j] > 0:
                new_state += k[j] * self.reactions[j].stoichiometry
    
    new_state = np.maximum(new_state, 0)
    return new_state, had_negative
```

### 4. 自适应步长选择

使用 Gillespie 的 tau 选择公式来限制步长：

```python
def _select_tau(self, propensities: np.ndarray, state: np.ndarray) -> float:
    total_propensity = propensities.sum()
    
    # 计算均值和方差
    mu = np.zeros(num_species)
    sigma_sq = np.zeros(num_species)
    
    for j in range(num_reactions):
        for i in range(num_species):
            mu[i] += stoich[i] * propensities[j]
            sigma_sq[i] += (stoich[i] ** 2) * propensities[j]
    
    # 选择保守的tau
    tau_candidates = []
    for i in range(num_species):
        numerator = max(self.epsilon * state[i], 1.0)
        if abs(mu[i]) > 0:
            tau1 = numerator / abs(mu[i])
            tau_candidates.append(tau1)
        if sigma_sq[i] > 0:
            tau2 = (numerator ** 2) / sigma_sq[i]
            tau_candidates.append(tau2)
    
    tau = min(tau_candidates)
```

### 5. 自动回退到精确SSA

当预期反应次数小于1时，tau-leaping 既不准确也不高效，自动回退到精确 SSA：

```python
if tau * total_propensity < 1.0:
    # 使用精确的SSA
    tau = np.random.exponential(1.0 / total_propensity)
    reaction_index = np.random.choice(...)
    state += self.reactions[reaction_index].stoichiometry
```

### 6. 中点法（提高精度）

通过在中点评估倾向函数，减少误差：

```python
def simulate_midpoint(self, initial_state, t_max):
    # 半步跳跃得到中间状态
    tau_half = tau / 2
    k_half = self._sample_reaction_counts(propensities, tau_half, state, max_k_half)
    mid_state, _ = self._apply_state_update(state, k_half)
    
    # 使用中点的倾向函数进行完整跳跃
    mid_propensities = self._compute_propensities(mid_state, t + tau_half)
    k = self._sample_reaction_counts(mid_propensities, tau, state, max_k)
```

## 使用示例

```python
from tau_leaping import TauLeaping, Reaction

# 创建模拟器
simulator = TauLeaping(['A', 'B'], epsilon=0.03)

# 添加反应
simulator.add_reaction(Reaction(
    propensity=lambda s, t: 1.0 * s[0],
    stoichiometry=np.array([-1, 1]),
    name='A → B'
))

# 使用修正后的tau-leaping
initial_state = np.array([100, 0])
times, states = simulator.simulate(initial_state, t_max=50.0, use_correction=True)

print(f"检测到的负数次数: {simulator.negative_count}")
print(f"应用的修正次数: {simulator.correction_count}")

# 使用更精确的中点法
times_mid, states_mid = simulator.simulate_midpoint(initial_state, t_max=50.0)
```

## 性能对比

| 方法 | 精度 | 速度 | 负种群 |
|------|------|------|--------|
| 朴素固定步长 | 低 | 快 | 常见 |
| 边界检查 | 中 | 中 | 罕见 |
| 边界检查+后修正 | 中高 | 中 | 无 |
| 中点法 | 高 | 中慢 | 无 |
| 精确SSA | 最高 | 最慢 | 无 |

## 可调参数

- `epsilon`: 控制相对误差的容忍度（默认0.03）
  - 较小值：更精确，更多步骤
  - 较大值：更快，可能需要更多修正

- `nc`: 切换到二项采样的阈值（默认10）
  - 较小值：更保守，更多二项采样
  - 较大值：更多Poisson采样

## 文件说明

- `tau_leaping.py`: 完整的Tau-Leaping实现，包含所有修正策略
- `demo_negative_fix.py`: 负种群问题的演示和对比
- `gene_regulation_ssa.py`: 原始的精确SSA实现

## 参考论文

1. Gillespie, D. T. (2001). Approximate accelerated stochastic simulation of chemically reacting systems. Journal of Chemical Physics.

2. Gillespie, D. T., & Petzold, L. R. (2003). Improved leap-size selection for accelerated stochastic simulation. Journal of Chemical Physics.

3. Cao, Y., Gillespie, D. T., & Petzold, L. R. (2005). Avoiding negative populations in explicit Poisson tau-leaping. Journal of Chemical Physics.
