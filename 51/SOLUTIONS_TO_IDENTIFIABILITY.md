# 贝叶斯变点检测：变点数目可辨识性问题的解决方案

## 问题描述

在贝叶斯变点检测中，一个常见的统计问题是**变点数目的可辨识性问题**：

1. **多峰后验分布**：不同数量的变点可能都有较高的后验概率
2. **过拟合倾向**：添加更多变点总是会增加似然性（因为有更多参数）
3. **模型选择不确定性**：很难确定"正确"的变点数量

## 解决方案

我们实现了以下策略来解决这个问题：

### 1. 使用边际似然替代条件似然

**原来的问题**：使用极大似然估计时，添加更多参数总是会增加似然值

**解决方案**：使用**贝叶斯边际似然**（对参数积分），自然包含奥卡姆剃刀效应

```python
def _log_marginal_likelihood(self, changepoints):
    # 使用正态-逆Gamma共轭先验计算边际似然
    # 自动惩罚过复杂的模型
```

边际似然公式：
```
p(y | 变点) = ∫ p(y | 变点, 参数) p(参数) d参数
```

### 2. 正则化先验分布

提供三种先验选择：

| 先验类型 | 说明 | 适用场景 |
|---------|------|---------|
| `regularized_poisson` | 泊松先验 + 变点间距惩罚 | 最常用，平衡效果 |
| `negative_binomial` | 负二项分布先验 | 更保守 |
| `strongly_regularized` | 强正则化：泊松先验 + 二次惩罚项 | 严格控制过拟合 |

```python
# 正则化泊松先验：惩罚太近的变点
for cp in changepoints:
    for other_cp in changepoints:
        if cp != other_cp:
            dist = abs(cp - other_cp)
            if dist < 10:
                log_prior -= (10 - dist) * 0.1 * penalty_strength
```

### 3. 贝叶斯模型平均 (BMA)

**不再选择单一模型**，而是考虑所有可能模型的加权平均

```python
# 获取BMA估计：选择后验概率 >= 阈值的所有变点
def get_bma_estimate(self, threshold=0.5):
    bma_cp = [i for i, p in enumerate(self.posterior_probs) if p >= threshold]
    return sorted(bma_cp)
```

### 4. 分层模型展示

提供不同模型大小下的变点概率：

```python
# 获取按变点数量分层的结果
models = bcd.get_model_averaged_changepoints(n_models=3)
```

## 使用示例

```python
import numpy as np
from improved_mcmc_changepoint import ImprovedMCMCChangepointDetection

# 生成测试数据
data = np.concatenate([
    np.random.normal(0, 1, 50),
    np.random.normal(5, 1.5, 50),
    np.random.normal(2, 1, 50)
])

# 创建模型（使用强正则化先验）
bcd = ImprovedMCMCChangepointDetection(
    data,
    max_changepoints=8,
    prior_type='regularized_poisson',  # 选择先验类型
    penalty_strength=2.0                # 调整惩罚强度
)

# 运行MCMC
bcd.run_mcmc(n_iterations=25000, burn_in=5000)

# 计算后验概率
posterior_probs = bcd.compute_posterior_probs()

# 查看变点数量的后验分布（检查是否有多峰）
bcd.print_summary()

# 获取MAP估计（单一模型）
map_cp = bcd.get_map_estimate()

# 获取BMA估计（模型平均，更稳健）
bma_cp = bcd.get_bma_estimate(threshold=0.5)

# 查看不同变点数量下的结果
models = bcd.get_model_averaged_changepoints(n_models=3)
for model in models:
    print(f"{model['n_changepoints']} 个变点, 概率={model['posterior_prob']:.3f}")
```

## 参数调优建议

| 数据特征 | 建议设置 |
|---------|---------|
| 噪声大、变化不明显 | `prior_type='strongly_regularized'`, `penalty_strength=3.0` |
| 信号清晰、变化明显 | `prior_type='regularized_poisson'`, `penalty_strength=1.0` |
| 预期变点很少 | `prior_type='negative_binomial'`, `max_changepoints=3` |
| 不确定变点数量 | 使用BMA，设置 `threshold=0.5~0.7` |

## 结果解读

### 好的结果特征：
- ✓ 变点数量后验分布有**单一明显峰值**
- ✓ 真实变点位置有**高后验概率**（>0.7）
- ✓ 虚假变点有**低后验概率**（<0.3）

### 需要调整的信号：
- ✗ 变点数量后验分布有**多个峰值** → 增加惩罚强度
- ✗ 很多位置概率在 0.3-0.7 之间 → 增加迭代次数或调整先验
- ✗ 明显虚假变点概率很高 → 使用更强的正则化先验

## 可视化说明

改进后的可视化包含4个子图：

1. **原始数据 + 变点估计**：显示MAP和BMA两种估计
2. **后验概率曲线**：每个位置是变点的概率（BMA结果）
3. **变点数量后验分布**：直观检查是否有多峰问题
4. **分层模型比较**：不同变点数量下的概率曲线
