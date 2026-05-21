# 贝叶斯热传导反问题 - MCMC不确定性分析

## 概述

本模块实现了基于马尔可夫链蒙特卡洛(MCMC)方法的贝叶斯热传导反问题求解，能够量化边界热流估计的不确定性，提供概率意义下的置信区间。

## 贝叶斯推断理论

### 贝叶斯定理

后验概率 ∝ 似然 × 先验

```
p(q | T_meas) ∝ p(T_meas | q) × p(q)
```

其中:
- **p(q | T_meas)**: 待求的后验概率分布
- **p(T_meas | q)**: 似然函数（测量数据的概率）
- **p(q)**: 先验概率分布（关于热流的先验知识）

### 似然函数

假设测量噪声服从独立高斯分布:

```
p(T_meas | q) = exp(-0.5 × ||T_pred(q) - T_meas||² / σ²)
```

### 先验分布

使用高斯先验:

```
p(q) = exp(-0.5 × ||q - μ_prior||² / σ_prior²)
```

## MCMC采样算法

### 1. 标准Metropolis-Hastings (MH)

- 对称随机游走建议分布
- 接受率判断: α = min(1, p(q*)/p(q))
- 自适应调整建议尺度

### 2. DRAM (延迟拒绝自适应Metropolis)

- **多阶段建议**: 第一阶段大跳跃，后续阶段小跳跃
- **延迟拒绝**: 第一阶段被拒后尝试更小的建议
- **自适应**: 燃烧期自动调整建议尺度
- **优势**: 高维参数空间采样效率更高

## 模块架构

```
bayesian_inverse_mcmc.py
├── solve_heat_direct_cn()          # 正问题求解器(Crank-Nicolson)
├── compute_likelihood()            # 似然函数计算
├── compute_prior()                 # 先验概率计算
├── compute_posterior()             # 后验概率计算
├── MCMCSampler                     # 基础MH采样器
│   ├── propose()                   # 生成建议样本
│   ├── sample()                    # 执行采样
│   ├── get_stats()                 # 后验统计量
│   └── geweke_test()               # 收敛性诊断
├── DRAMSampler                     # DRAM高级采样器（继承MCMCSampler）
├── generate_test_data()            # 生成测试数据
└── run_bayesian_inversion()        # 主函数
```

## 使用方法

### 基本用法

```python
from bayesian_inverse_mcmc import *

# 生成测试数据
L, T_total, alpha, x_measured, t_measured, T_measured, q_true, t, sigma_meas = generate_test_data()

# 定义后验概率函数
def log_posterior(q):
    return compute_posterior(q, L, T_total, alpha, Nx, Nt, 
                             T_measured, x_measured, t_measured,
                             sigma_meas, prior_mean, prior_std)

# 创建DRAM采样器
sampler = DRAMSampler(log_posterior, dim=Nt, proposal_scale=5.0, n_stages=2)

# 执行采样
samples, log_posteriors = sampler.sample(
    initial_q, 
    n_samples=2000, 
    burn_in=1000, 
    thin=2, 
    adapt=True
)

# 获取后验统计
stats = sampler.get_stats()
q_mean = stats['mean']       # 后验均值
q_std = stats['std']         # 后验标准差
q_ci_low = stats['ci_low']   # 95%置信区间下界
q_ci_high = stats['ci_high'] # 95%置信区间上界
```

## 输出结果说明

### 1. 后验统计量

| 统计量 | 说明 |
|--------|------|
| `mean` | 后验均值（点估计） |
| `median` | 后验中位数（鲁棒点估计） |
| `std` | 后验标准差（不确定性度量） |
| `ci_low` / `ci_high` | 95%置信区间 |
| `q25` / `q75` | 50%置信区间（四分位距） |

### 2. 收敛诊断

- **接受率**: 理想范围 0.15-0.5
- **Geweke检验**: z-score < 2 表示收敛
- **轨迹图**: 平稳分布表示收敛

### 3. 可视化结果

程序生成 `bayesian_mcmc_results.png`，包含6张子图:

| 子图 | 内容 | 用途 |
|------|------|------|
| 1 | 热流后验估计+置信区间 | 展示不确定性范围 |
| 2 | 后验概率轨迹 | 检查收敛性 |
| 3 | 单个参数轨迹 | 诊断特定时刻的采样质量 |
| 4 | 边际后验分布直方图 | 查看参数分布形态 |
| 5 | 后验标准差随时间变化 | 识别高不确定性时段 |
| 6 | 后验样本集合 | 直观展示解的多样性 |

另外生成 `posterior_correlation.png` 展示参数间相关性。

## 参数调优指南

### 采样参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_samples` | 2000-10000 | 总采样数，越多越精确但越慢 |
| `burn_in` | n_samples × 0.5 | 燃烧期，丢弃初始非平稳样本 |
| `thin` | 2-5 | 抽样间隔，减少样本自相关 |
| `proposal_scale` | 1-10 | 初始建议尺度，自适应调整 |

### 先验参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `prior_mean` | 50 | 先验均值，基于物理知识 |
| `prior_std` | 20 | 先验标准差，反映先验不确定性 |

### 收敛判据

1. ✅ 接受率在 0.15-0.5 之间
2. ✅ Geweke检验 |z| < 2
3. ✅ 轨迹图呈现平稳分布
4. ✅ 多个链结果一致（Gelman-Rubin检验）

## 性能优化建议

### 1. 降维技术

**问题**: 100时间步 = 100维参数，采样效率低

**解决方案**:
- 基函数展开（如傅里叶基、PCA）
- 样条插值表示热流
- 降维后参数维度可降至10-20

```python
# 示例：傅里叶基展开
def q_from_coeffs(coeffs, t):
    q = coeffs[0]  # 常数项
    for i in range(1, len(coeffs)//2 + 1):
        q += coeffs[2*i-1] * np.sin(2*np.pi*i*t/T_total)
        q += coeffs[2*i] * np.cos(2*np.pi*i*t/T_total)
    return q
```

### 2. 并行计算

- 多条马尔可夫链并行采样
- 似然计算向量化
- 使用GPU加速正问题求解

### 3. 代理模型

- 用神经网络或GPE代理正问题
- 训练后MCMC采样速度提升100-1000倍

## 与确定性方法对比

| 特性 | 共轭梯度法(CGM) | 贝叶斯MCMC |
|------|----------------|-----------|
| 输出 | 单个最优解 | 完整后验分布 |
| 不确定性 | 需额外计算 | 自然提供 |
| 计算量 | O(10)正问题求解 | O(1000+)正问题求解 |
| 收敛速度 | 快 | 慢 |
| 先验信息 | 难以引入 | 自然融合 |
| 适用场景 | 快速点估计 | 不确定性量化、风险分析 |

## 实际应用建议

### 工作流程

1. **先用CGM**快速获得最优解估计
2. **再用MCMC**在最优解附近采样（减少燃烧期）
3. **分析后验**获得置信区间
4. **收敛诊断**确保结果可靠

### 报告不确定性

在工程报告中应同时提供:
- 点估计值（后验均值/中位数）
- 95%置信区间
- 超出安全阈值的概率

示例报告:
```
边界热流估计:
  均值: 52.3 ± 4.1 W/m²
  95%置信区间: [44.5, 60.1] W/m²
  超过60 W/m²的概率: 3.2%
```

## 参考文献

1. Kaipio, J., & Somersalo, E. (2005). Statistical and Computational Inverse Problems. Springer.
2. Gamerman, D., & Lopes, H. F. (2006). Markov Chain Monte Carlo: Stochastic Simulation for Bayesian Inference.
3. Haario, H., et al. (2006). DRAM: Efficient adaptive MCMC. Statistics and Computing.
4. Beck, J. V., & Woodbury, K. A. (2016). Inverse Heat Conduction. Wiley.
