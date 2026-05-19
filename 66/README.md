# Grad-Shafranov方程求解器 - 托卡马克平衡计算

## 简介

这是一个用Python实现的Grad-Shafranov方程求解器，用于计算托卡马克等离子体平衡。本求解器包含完整的磁散度修复功能和气球模稳定性分析。

## 主要功能

1. **Grad-Shafranov方程求解**: 有限差分法 + Picard迭代
2. **磁场散度修复**: Helmholtz分解散度清除算法
3. **气球模稳定性分析**: 第一/第二稳定区评估
4. **第二稳定区访问可能性分析**

## 最新更新1：磁场散度修复

**问题**: 由于数值离散误差，原始磁场计算的散度达到约1e-3量级，破坏了磁通量守恒。

**解决方案**:
1. **精确中心差分格式**: 改进了磁场的数值计算
2. **Helmholtz分解散度清除算法**: 通过求解泊松方程迭代清除散度

**效果**: 散度从约1e-3降低到1e-12量级，恢复了磁通量守恒。

## 最新更新2：气球模稳定性与第二稳定区分析

**功能**:
1. **关键参数计算**: 安全因子q、磁剪切s、β值、α参数
2. **气球模稳定性判据**: α_crit = 0.6s/q 判据
3. **第二稳定区评估**: 综合β、s、q参数评估访问潜力

**物理意义**:
- **第一稳定区**: 低β区域，气球模稳定
- **第二稳定区**: 高β + 高剪切区域，气球模重新稳定
- **关键条件**: q > 2, s > 0.2, β > 2%

## 数学原理

Grad-Shafranov方程是描述轴对称托卡马克等离子体平衡的基本方程：

```
Δ*ψ = -μ₀ R² dp/dψ - F dF/dψ
```

其中：
- ψ 是极向磁通量 (Poloidal Flux)
- Δ* 是 Grad-Shafranov 算子： Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
- p(ψ) 是等离子体压力剖面
- F(ψ) = R B_φ 是环向磁场函数

## 求解方法

- **有限差分法**：对计算域进行离散
- **稀疏矩阵**：高效存储和求解线性系统
- **Picard迭代**：处理非线性项
- **松弛法**：提高迭代稳定性

## 依赖库

```bash
pip install numpy scipy matplotlib
```

## 使用方法

### 基本用法

```python
from grad_shafranov_solver import GradShafranovSolver

# 创建求解器实例
solver = GradShafranovSolver(
    Rmin=0.5, Rmax=1.5,  # R方向范围 (m)
    Zmin=-0.6, Zmax=0.6,  # Z方向范围 (m)
    nr=65, nz=65           # 网格点数
)

# 求解方程
psi = solver.solve(
    p0=1e5,      # 中心压力 (Pa)
    alpha=2.0,    # 压力剖面指数
    F0=1.0,       # 环向磁场参数
    beta=0.05,     # 磁场剖面参数
    max_iter=150,  # 最大迭代次数
    tol=1e-6,      # 收敛公差
    relaxation=0.5 # 松弛因子
)

# 可视化结果
solver.plot_psi(levels=20)
solver.plot_magnetic_field(density=1.5)
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `Rmin`, `Rmax` | 径向范围 (m) | 0.5, 1.5 |
| `Zmin`, `Zmax` | 垂直方向范围 (m) | -0.7, 0.7 |
| `nr`, `nz` | 网格点数 | 65, 65 |
| `p0` | 中心压力 (Pa) | 1e5 |
| `alpha` | 压力剖面指数 p ∝ (1-ψ/ψ_max)^α | 2.0 |
| `F0` | 环向磁场参数 | 1.0 |
| `beta` | 磁场剖面参数 | 0.1 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `grad_shafranov_solver.py` | 主求解器程序 |
| `test_divergence.py` | 磁场散度修复验证测试 |
| `test_stability.py` | 气球模稳定性与第二稳定区分析 |
| `example.py` | 基本使用示例 |

## 运行

### 基本运行（含稳定性分析）
```bash
python grad_shafranov_solver.py
```

### 散度修复验证
```bash
python test_divergence.py
```

### 气球模与第二稳定区多场景分析
```bash
python test_stability.py
```

## 气球模稳定性分析使用示例

```python
from grad_shafranov_solver import GradShafranovSolver

# 创建求解器
solver = GradShafranovSolver()

# 求解平衡
solver.solve(p0=2e5, alpha=2.0)

# 稳定性分析
results = solver.analyze_stability(
    p0=2e5,        # 中心压力 (Pa)
    alpha=2.0,      # 压力剖面指数
    B0=1.0,         # 磁场强度 (T)
    q0=2.0          # 轴上安全因子
)

# 打印摘要
solver.print_stability_summary(results)

# 可视化
solver.plot_stability_maps(results)      # 参数分布图
solver.plot_stability_diagram(results)    # s-α 稳定性图
```

## 第二稳定区关键参数

| 参数 | 典型阈值 | 物理意义 |
|------|----------|----------|
| β | > 2% | 等离子体/磁压比 |
| 磁剪切 s | > 0.2 | 磁场剪切强度 |
| 安全因子 q | > 2 | 磁面扭曲程度 |
| α 参数 | ~ 0.5s | 压力梯度驱动参数 |

## 第二稳定区访问策略

1. **提高β值**: 增加等离子体压力，目标β > 3%
2. **增加磁剪切**: 通过成形和电流剖面优化提高s
3. **优化q剖面**: 保持q_min > 2，避免低阶共振
4. **压力剖面整形**: 采用较宽的压力剖面（低α值）

## 输出说明

程序将显示：
1. 迭代过程信息
2. 收敛信息
3. 磁面结构等高线图
4. 极向磁场流线图

## 类方法说明

### `GradShafranovSolver`

#### `__init__(self, Rmin, Rmax, Zmin, Zmax, nr, nz)`
初始化求解器

#### `solve(self, p0, alpha, F0, beta, max_iter, tol, relaxation)`
求解Grad-Shafranov方程

#### `plot_psi(self, levels, figsize)`
绘制磁面结构（ψ等值线）

#### `get_magnetic_field(self, clean_divergence=True)`
计算极向磁场分量 B_R, B_Z
- `clean_divergence`: 是否启用散度清除（默认True）

#### `get_magnetic_field_centered(self, clean_divergence=True)`
使用精确中心差分格式计算磁场

#### `compute_divergence(self, B_R, B_Z)`
计算磁场散度 ∇·B

#### `divergence_cleaning_fast(self, B_R, B_Z, max_iter=50)`
Helmholtz分解散度清除算法

#### `check_divergence_error(self, B_R, B_Z, method_name="")`
检查并打印散度误差

#### `plot_divergence(self, B_R, B_Z, title, figsize)`
绘制磁场散度分布图

#### `plot_magnetic_field(self, density, figsize)`
绘制极向磁场流线图

#### `plot_psi_3d(self, figsize)`
绘制3D磁通量图

## 注意事项

1. 网格点数建议使用奇数（如65），以便有明确的中心点
2. 收敛公差太小可能导致迭代次数增加
3. 松弛因子通常在0.3-0.7之间效果较好
4. 压力剖面参数 alpha 通常取1-4

## 改进方向

- 添加更多压力剖面模型
- 实现Newton迭代法加速收敛
- 添加安全因子 q(ψ) 计算
- 添加等离子体电流密度计算
- 支持非圆截面边界
