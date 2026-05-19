# 随机基因调控网络模拟工具集

## 项目概述

这是一个完整的Python工具集，用于模拟和分析随机基因调控网络。包含三种核心方法：精确的Gillespie SSA、加速的Tau-Leaping、以及矩封闭近似（用于参数估计）。

## 工具列表

### 📦 核心模块

| 文件 | 用途 | 关键特性 |
|------|------|---------|
| **gene_regulation_ssa.py** | Gillespie随机模拟算法 | 精确SSA、内置网络模板、可视化 |
| **tau_leaping.py** | Tau-Leaping近似算法 | 负种群修正、多策略、自适应步长 |
| **moment_closure.py** | 矩封闭近似 | 正态/对数正态封闭、参数估计 |

### 📊 演示/比较脚本

| 文件 | 用途 |
|------|------|
| **demo_negative_fix.py** | Tau-Leaping负种群问题演示 |
| **compare_methods.py** | 三种方法的性能/精度对比 |
| **verify_code.py** | 代码语法验证工具 |

### 📖 文档

| 文件 | 内容 |
|------|------|
| **MOMENT_CLOSURE_GUIDE.md** | 矩封闭方法详细指南和API文档 |
| **TAU_LEAPING_FIX.md** | Tau-Leaping负种群问题修复说明 |
| **README.md** | 基础使用指南 |

## 核心功能详解

### 1. Gillespie SSA (gene_regulation_ssa.py)

**GillespieSSA类**：
- 精确的随机模拟算法
- 支持任意反应网络
- 内置两个网络模板：
  - 基本基因表达网络
  - 抑制振荡网络

**使用示例**：
```python
from gene_regulation_ssa import GillespieSSA, Reaction

ssa = GillespieSSA(['mRNA', 'Protein'])
ssa.add_reaction(Reaction(
    propensity=lambda s, t: 5.0,
    stoichiometry=np.array([1, 0]),
    name='Transcription'
))
times, states = ssa.simulate(initial_state, t_max=50)
ssa.plot_results()
```

### 2. Tau-Leaping (tau_leaping.py)

**TauLeaping类**：
- 多种负种群修正策略
- 自适应步长选择
- 自动切换到精确SSA
- 中点法提高精度

**三种采样策略**：
1. **Poisson边界检查**：限制样本不超过最大可能值
2. **二项式采样**：适用于接近耗尽的反应
3. **后跳跃修正**：检测并修正负数

**使用示例**：
```python
from tau_leaping import TauLeaping, Reaction

tl = TauLeaping(['A', 'B'], epsilon=0.05)
# 添加反应...
times, states = tl.simulate(initial_state, t_max=50, use_correction=True)
```

### 3. 矩封闭近似 (moment_closure.py)

**MomentClosure类**：
- 正态封闭（Normal）
- 对数正态封闭（LogNormal）
- 导数匹配

**MomentEstimator类**：
- 基于矩的参数估计
- 支持加权最小二乘
- 优化历史跟踪

**使用示例**：
```python
from moment_closure import MomentClosure, MomentEstimator

# 矩模拟
mc = MomentClosure(species_names, propensities, stoichs, closure_type='normal')
times, means, covs = mc.simulate(initial_means, initial_cov, t_span)

# 参数估计
estimator = MomentEstimator(species_names, prop_template, stoichs, param_names)
estimated_params = estimator.estimate(initial_params, target_times, target_means)
```

## 方法选择指南

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| **精确分布计算** | Gillespie SSA | 金标准，精确但慢 |
| **快速近似模拟** | Tau-Leaping | 速度快，支持修正 |
| **参数估计** | 矩封闭 | O(1000x)加速，可计算梯度 |
| **低拷贝数系统** | SSA或对数正态封闭 | 正态封闭精度下降 |
| **高拷贝数系统** | 任何方法 | 所有方法都表现良好 |
| **不确定性量化** | 矩封闭或SSA | 同时给出均值和方差 |

## 性能对比

### 典型运行时间

| 方法 | 单次运行 | 200次统计 |
|------|---------|----------|
| **SSA** | ~10ms | ~2秒 |
| **Tau-Leaping** | ~1ms | ~0.2秒 |
| **矩封闭** | ~0.1ms | ~0.02秒 |

### 加速比（相对于SSA）

- Tau-Leaping: ~10x
- 矩封闭: **~100x**

对于参数估计（需要O(1000)次评估）：
- SSA: ~30分钟
- 矩封闭: **~2秒**

## 快速开始

### 1. 运行基因表达网络SSA模拟
```bash
python gene_regulation_ssa.py
```

### 2. 演示Tau-Leaping负种群修正
```bash
python demo_negative_fix.py
```

### 3. 参数估计演示
```bash
python moment_closure.py
```

### 4. 三种方法对比
```bash
python compare_methods.py
```

## 数学基础

### SSA
基于化学主方程的精确随机采样：
- 反应时间 ~ Exp(总倾向)
- 反应选择 ~ 多项分布(倾向)

### Tau-Leaping
在时间τ内同时发生多个反应：
- 每个反应 ~ Poisson(倾向×τ)
- τ由期望状态变化限制

### 矩封闭
通过ODE求解均值和协方差：
```
d⟨X⟩/dt = Σ ν·⟨a(X)⟩
dCov/dt = Σ νν'·⟨a(X)⟩ + 交叉项
```
通过分布假设计算更高阶矩来封闭方程。

## 扩展开发

### 添加新的反应网络
```python
def create_my_network(params):
    species_names = ['A', 'B', 'C']
    propensities = [...]
    stoichiometries = [...]
    return species_names, propensities, stoichiometries
```

### 自定义矩封闭方法
```python
class MyClosure(MomentClosure):
    def _compute_moment_equations(self, t, y):
        dydt = super()._compute_moment_equations(t, y)
        # 添加自定义修正
        return dydt
```

### 新的参数估计目标函数
```python
class MyEstimator(MomentEstimator):
    def _compute_objective(self, params, ...):
        # 自定义目标函数
        return custom_loss
```

## 依赖要求

```
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.4
```

安装依赖：
```bash
pip install numpy scipy matplotlib
```

## 测试建议

1. **验证SSA正确性**：与解析解（如基因表达稳态）对比
2. **Tau-Leaping验证**：确保修正后无负种群
3. **矩封闭验证**：与SSA统计量对比
4. **参数估计验证**：已知真值的合成数据测试

## 引用与参考

如在研究中使用此工具集，请参考：

1. Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions.
2. Gillespie, D. T. (2001). Approximate accelerated stochastic simulation.
3. Cao, Y., et al. (2005). Avoiding negative populations in explicit Poisson tau-leaping.

## 许可证

MIT License - 可自由用于学术和商业目的。

---

**祝您的建模愉快！** 🧬🔬
