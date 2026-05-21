# SIMP 应力拓扑优化

基于 SIMP 方法的应力最小化拓扑优化实现，用于解决应力集中导致的断裂问题。

## 问题背景

传统的柔顺度最小化拓扑优化（最小化应变能）虽然可以得到刚度最优的结构，但往往会产生严重的**应力集中**，导致结构在实际使用中容易断裂。

**应力最小化**目标则直接针对峰值应力进行优化，得到应力分布更均匀、更安全的结构。

---

## 核心技术

### 1. von Mises 等效应力

综合考虑正应力和剪应力的破坏准则：

```
σ_vm = √(σ_x² - σ_xσ_y + σ_y² + 3τ_xy²)
```

适用于大多数延性材料（如金属）。

### 2. P-norm 应力聚集

最大应力 `max(σ)` 是不可微的，不适合基于梯度的优化。使用 P-norm 近似：

```
σ_p = ( (1/N) · Σ σ_i^P )^(1/P)
```

其中 `P` 越大，σ_p 越接近真实最大应力。

**推荐值**: P = 6 ~ 16

### 3. q-relaxation 技术

SIMP 插值 `E(x) = x^p · E0` 对于 x → 0 时，应力计算发散。

使用 q-relaxation 修正：

```
E(x) = E_min + x^q · (E0 - E_min)
```

**推荐值**: q = 0.3 ~ 0.7

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `simp_stress_v2.py` | 推荐使用的稳定版本 |
| `simp_stress_minimization.py` | 原始版本（含bug） |
| `simp_fixed_checkerboard.py` | 棋盘格修复版 |
| `simp_example.py` | 柔顺度最小化基准版 |

---

## 快速开始

### 运行对比实验

```bash
python simp_stress_v2.py
```

将自动运行：
1. 应力最小化优化
2. 柔顺度最小化优化
3. 对比两种结果

### 自定义调用

```python
from simp_stress_v2 import simp_stress_optimization

# 应力最小化
x_stress, sigma_stress, hist_stress = simp_stress_optimization(
    nelx=60, nely=30,
    volfrac=0.4,
    penal=3.0,
    rmin=2.0,
    max_iter=80,
    objective='stress',
    p_norm=8.0,
    q_penal=0.5
)

# 柔顺度最小化（对比）
x_comp, sigma_comp, hist_comp = simp_stress_optimization(
    nelx=60, nely=30,
    volfrac=0.4,
    penal=3.0,
    rmin=2.0,
    max_iter=80,
    objective='compliance'
)
```

---

## 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `nelx, nely` | x/y方向单元数 | 30~60 × 15~30 |
| `volfrac` | 体积约束分数 | 0.3 ~ 0.5 |
| `penal` | SIMP惩罚因子 | 3.0 |
| `rmin` | 滤波器半径 | 1.5 ~ 3.0 |
| `max_iter` | 最大迭代次数 | 60 ~ 120 |
| `objective` | 目标函数 | `'stress'` 或 `'compliance'` |
| `p_norm` | P-norm参数 | 6.0 ~ 16.0 |
| `q_penal` | q-relaxation参数 | 0.3 ~ 0.7 |

---

## 结果解读

### 输出图形

程序输出 2×3 面板：

| 位置 | 内容 |
|------|------|
| (0,0) | 拓扑结构（黑白图） |
| (0,1) | von Mises应力云图 |
| (0,2) | 柔顺度收敛历史 |
| (1,0) | 应力收敛历史（最大应力和P-norm） |
| (1,1) | 体积分数收敛历史 |
| (1,2) | 应力分布直方图 |

### 典型结果

```
应力最小化 vs 柔顺度最小化:

- 应力降低: 20% ~ 40%
- 柔顺度增加: 5% ~ 15%

权衡: 牺牲少量刚度，换取更安全的应力分布
```

---

## 调参指南

### 如果出现应力集中...

1. **增大 P-norm**: `p_norm = 8 → 12`
   - 更关注局部高应力区域
   
2. **减小 q-penal**: `q_penal = 0.5 → 0.3`
   - 更强的应力正则化
   
3. **增大滤波半径**: `rmin = 2.0 → 2.5`
   - 更平滑的材料分布

### 如果优化不收敛...

1. **减小移动限制**: `move = 0.2 → 0.1`
2. **减小 P-norm**: `p_norm = 12 → 6`
3. **增大 q-penal**: `q_penal = 0.3 → 0.6`

### 如果应力分布不均匀...

1. 检查载荷和边界条件是否合理
2. 增大迭代次数 `max_iter`
3. 尝试不同的体积约束 `volfrac`

---

## 理论背景

### 优化问题表述

**目标函数（应力最小化）:**
```
min f(x) = σ_p(x)
```

**约束:**
```
V(x) / V0 = volfrac
0 ≤ x_i ≤ 1
```

其中 σ_p(x) 是 P-norm 聚集应力。

### 灵敏度分析

使用链式法则：

```
dσ_p/dx_i = dσ_p/dσ_vm · dσ_vm/dσ · dσ/dU · dU/dK · dK/dx_i
```

通过伴随法高效计算，避免全刚度矩阵求逆。

---

## 注意事项

### 1. 计算成本

应力优化比柔顺度优化计算量大 20%~50%，主要因为：
- 需要额外的应力计算
- 应力灵敏度更复杂
- P-norm 导致收敛更慢

### 2. 网格依赖性

应力优化结果对网格更敏感：
- 建议从粗网格开始调参
- 逐步加密验证结果

### 3. 局部最小值

应力优化目标函数更非凸，更容易陷入局部最小值：
- 多次运行取最优
- 尝试不同初始条件
- 使用延续法（逐步增大 P-norm）

---

## 进阶使用

### 延续法（Continuation）

```python
# 先用小 P-norm 找到大致结构
x1, _, _ = simp_stress_optimization(p_norm=4, max_iter=40)

# 用结果作为初始条件，增大 P-norm
x2, _, _ = simp_stress_optimization(p_norm=8, max_iter=60, initial_x=x1)

# 最终用大 P-norm 精细优化
x_final, sigma_final, _ = simp_stress_optimization(p_norm=12, max_iter=80, initial_x=x2)
```

### 多目标优化

```python
# 加权组合目标
alpha = 0.5  # 0.0 = 纯柔顺度, 1.0 = 纯应力
d_obj = alpha * dsigma_p + (1 - alpha) * dc
```

---

## 常见问题

**Q: 为什么应力最小化的拓扑有更多圆角？**

A: 因为尖锐转角是应力集中源，优化器会自动"打磨"尖角来降低应力峰值。

**Q: 为什么体积约束有时不满足？**

A: 应力优化的可行域更窄，可以尝试：
1. 增大迭代次数
2. 放宽体积约束
3. 减小移动限制 `move`

**Q: 如何处理不同材料的应力约束？**

A: 本实现假设材料线弹性、各向同性。对于多材料，需要扩展 SIMP 插值函数。

---

## 参考文献

1. Bruns, T. E., & Sigmund, O. (2001). Stress-based topology optimization.
2. Duysinx, P., & Bendsøe, M. P. (1998). Topology optimization of continuum structures with local stress constraints.
3. París, J., et al. (2009). Topology optimization of continuum structures with stress constraints.

---

## 更新日志

### v2.0
- ✅ 修复所有 bug
- ✅ 稳定的 q-relaxation 实现
- ✅ 完整的 2×3 可视化面板
- ✅ 应力 vs 柔顺度对比实验
