# 皮尔逊III型分布拟合 - 稳健L-Moment与非平稳频率分析

## 功能概述

本工具包实现了完整的皮尔逊III型(P-III)分布频率分析功能，包括：

1. **平稳P-III分布拟合** - 传统矩法、多种L-矩法
2. **稳健L-Moment估计** - 处理异常值/特大洪水
3. **非平稳频率分析** - 引入协变量(气候指标等)到P-III参数

---

## 问题背景1: 异常值对L-Moment的影响

在水文频率分析中，当年最大洪水序列包含异常值（如特大洪水）时，传统的L-Moment估计会被"拉伸"，导致偏态系数(Cs)估计偏高，进而影响设计洪水的计算精度。

---

## 问题背景2: 变化环境下的非平稳性

受气候变化和人类活动影响，洪水序列往往呈现非平稳特征（均值/方差随时间变化）。传统平稳频率分析假设"独立同分布"，在变化环境下不再适用，需要引入协变量进行非平稳频率分析。

## 解决方案

本实现提供了多种稳健的L-Moment估计方法，有效处理异常值问题：

### 1. 传统矩法 (method='moments')
- 基于样本均值、方差、偏度的经典估计
- 对异常值最敏感

### 2. 传统L-矩法 (method='l_moments', robust=False)
- 基于次序统计量的线性组合
- 比传统矩法更稳健，但仍受极端值影响

### 3. 稳健L-矩法 (method='l_moments', robust=True)
- **核心修复方法**：对称截断两端5%的数据后计算L-矩
- 有效去除两端异常值的影响
- 可通过`trim_percent`参数调整截断比例

### 4. 截断L-矩法 (method='trimmed_l_moments')
- 非对称截断，可对左右尾设置不同截断比例
- 特别适合洪水数据（右尾有特大洪水）
- 参数：`trim_left`(左截断), `trim_right`(右截断)

### 5. 加权L-矩法 (method='weighted_l_moments')
- 根据数据偏离中位数的程度自动分配权重
- 支持Huber权重和Tukey双平方权重
- 异常值权重降低但不完全剔除

## 使用方法

### 基本使用
```python
from pearson3 import PearsonIIIFitter

# 年最大洪水序列
flood_data = [820, 960, 1050, 780, 1200, 920, 1100, ...]

# 使用稳健L-矩法拟合
fitter = PearsonIIIFitter(flood_data)
params = fitter.fit(method='l_moments', robust=True)

# 计算100年一遇设计洪水
design_100 = fitter.design_flood(100)
print(f"100年一遇设计洪水: {design_100:.2f} m³/s")
```

### 截断L-矩法（推荐用于特大洪水）
```python
# 左截断5%，右截断10%（针对右尾特大洪水）
params = fitter.fit(method='trimmed_l_moments', 
                    trim_left=0.05, 
                    trim_right=0.1)
```

### 加权L-矩法
```python
# Huber权重
params = fitter.fit(method='weighted_l_moments', 
                    weight_type='huber', 
                    c=1.5)

# Tukey双平方权重
params = fitter.fit(method='weighted_l_moments', 
                    weight_type='tukey', 
                    c=4.685)
```

## 方法对比

| 方法 | 异常值敏感度 | 优点 | 适用场景 |
|------|-------------|------|---------|
| 传统矩法 | 高 | 计算简单 | 数据质量好，无异常值 |
| 传统L-矩法 | 中 | 无偏性好 | 轻度异常值的数据 |
| 稳健L-矩法 | 低 | 实现简单 | 存在少量特大洪水 |
| 截断L-矩法 | 很低 | 可非对称截断 | 右尾有特大洪水 |
| 加权L-矩法 | 很低 | 自动权重调整 | 异常值程度不明 |

## 推荐实践

1. **首选方法**：对于可能存在特大洪水的洪水序列，推荐使用：
   ```python
   method='trimmed_l_moments', trim_left=0.05, trim_right=0.1
   ```

2. **参数验证**：
   - 比较不同方法的Cs值，差异过大时检查数据
   - 观察设计洪水值的稳定性

3. **异常值处理**：
   - 特大洪水应慎重考虑是否属于真实历史事件
   - 可结合水文专业知识进行判断

## 文件说明

- `pearson3.py` - 核心实现模块
- `example_usage.py` - 使用示例
- `README.md` - 本文档

## 运行示例

```bash
python pearson3.py
python example_usage.py
```

---

## 非平稳频率分析 (Nonstationary Frequency Analysis)

### 核心思想

将P-III分布的参数（位置、尺度）表示为协变量的函数：

- **位置参数**: μ(t) = θ₀ + θ₁·Z(t)  [线性函数]
- **尺度参数**: σ(t) = exp[γ₀ + γ₁·Z(t)]  [指数函数，保证为正]
- **形状参数**: α = 常数  [通常假设平稳]

其中 Z(t) 为协变量（时间趋势、气候指标等）

### 类结构

**1. NonstationaryPearsonIII - 基础非平稳模型**

```python
from pearson3 import NonstationaryPearsonIII

# 洪水序列 + 协变量（如时间趋势、气候指标）
model = NonstationaryPearsonIII(flood_data, covariates)

# 拟合模型（极大似然估计）
params = model.fit(method='MLE')

# 获取时变参数
tv_params = model.get_time_varying_params(covariate_values)

# 计算时变设计洪水
design_100 = model.design_flood(T=100, covariate_value=current_climate)
```

**主要方法:**
- `fit()` - 使用极大似然估计拟合模型
- `get_time_varying_params()` - 获取时变分布参数
- `design_flood(T, cov)` - 计算特定协变量下的设计洪水
- `trend_test()` - 检验趋势显著性
- `plot_trend()` - 绘制参数变化趋势

**2. NonstationaryP3Advanced - 高级模型选择**

```python
from pearson3 import NonstationaryP3Advanced

adv_model = NonstationaryP3Advanced(flood_data, covariates)

# 拟合多个模型并选择最优
results, best_model = adv_model.fit_multi_model()
```

**自动比较4种模型:**
- `stationary` - 平稳模型（无协变量）
- `mu_trend` - 仅位置参数有趋势
- `sigma_trend` - 仅尺度参数有趋势
- `both_trend` - 位置和尺度参数都有趋势

**选择准则:** AIC (Akaike Information Criterion)、BIC (Bayesian Information Criterion)

### 使用示例

```python
import numpy as np
from pearson3 import NonstationaryPearsonIII, NonstationaryP3Advanced

# 1. 准备数据
n_years = 50
time = np.arange(n_years)
climate_index = 0.05 * time  # 模拟气候指标

# 模拟具有趋势的洪水数据
mu = 1000 + 300 * climate_index
sigma = np.exp(np.log(150) + 0.2 * climate_index)
alpha = 3.0

from scipy.stats import gamma
flood_data = mu - sigma + gamma.rvs(a=alpha, scale=sigma/alpha, size=n_years)

# 2. 拟合非平稳模型
model = NonstationaryPearsonIII(flood_data, climate_index)
params = model.fit()

# 3. 查看结果
if params['converged']:
    print(f"位置参数: μ = {params['theta'][0]:.2f} + {params['theta'][1]:.2f} × 气候指标")
    print(f"尺度参数: log(σ) = {params['gamma'][0]:.3f} + {params['gamma'][1]:.3f} × 气候指标")
    print(f"形状参数: α = {params['alpha']:.4f}")

# 4. 计算不同气候条件下的设计洪水
for climate in [-0.5, 0.0, 0.5]:
    df = model.design_flood(T=100, covariate_value=climate)
    print(f"气候={climate:.1f}, 100年一遇={df:.0f} m³/s")

# 5. 多模型比较选择
adv_model = NonstationaryP3Advanced(flood_data, climate_index.reshape(-1, 1))
model_results, best_model = adv_model.fit_multi_model()
print(f"最优模型: {best_model}")
```

### 常用协变量类型

| 协变量类型 | 示例 | 说明 |
|-----------|------|------|
| **时间趋势** | 年序、标准化时间 | 捕捉长期变化趋势 |
| **气候指标** | SOI, PDO, AMO, NAO | 捕捉气候振荡影响 |
| **人类活动** | 水库蓄水量、城市化率 | 捕捉下垫面变化 |
| **多变量组合** | 时间 + 气候指标 | 多因子联合分析 |

### 模型诊断与检验

1. **收敛性检验**: 检查优化是否收敛
2. **参数显著性**: 查看趋势项的大小
3. **残差分析**: 检验模型拟合优度
4. **模型比较**: 使用AIC/BIC选择最优模型

### 非平稳设计洪水的应用

1. **考虑气候变化的设计** - 基于未来气候情景计算设计洪水
2. **更新现有设计标准** - 针对已变化的水文条件重新评估
3. **风险动态评估** - 评估洪水风险随时间的变化
4. **适应性管理** - 为水利工程适应性运行提供依据

### 注意事项

1. **形状参数假设**: 通常假设形状参数α为常数，若需放宽需更多数据
2. **协变量选择**: 优先选择物理意义明确的协变量
3. **外推谨慎**: 协变量超出观测范围时，预测不确定性增大
4. **样本量**: 非平稳模型需要更长的序列（建议≥40年）
