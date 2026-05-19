# 欧拉-拉格朗日方程数值解 - 最优控制求解器

本项目实现了用Python求解**带不等式约束**的最优控制问题，支持直接法和间接法两种求解方式。

## ✨ 最新功能：不等式约束处理

### 三种约束处理方法

| 方法 | 适用场景 | 特点 |
|------|---------|------|
| **边界法 (SLSQP)** | 简单控制量约束 | 快速、内置优化器 |
| **增广拉格朗日方法** | 一般不等式约束 | 鲁棒、可处理复杂约束 |
| **庞特里亚金最大值原理** | 理论分析 | 精确的Bang-Bang和饱和控制 |

---

## 📋 功能特性总览

### 1. 直接法求解器 (`ConstrainedEulerLagrangeSolver`)
- ✅ 支持终端代价函数：J = Φ(x(T)) + ∫₀ᵀ L(t, x, u) dt
- ✅ 支持不等式约束：u_min ≤ u(t) ≤ u_max
- ✅ 两种求解模式：边界法 / 增广拉格朗日
- ✅ 灵活的边界条件

### 2. 间接法求解器 (`ConstrainedIndirectSolver`)
- ✅ 基于庞特里亚金最大值原理
- ✅ 自动处理控制量饱和
- ✅ 支持Bang-Bang控制
- ✅ 横截条件处理

### 3. 增广拉格朗日类 (`AugmentedLagrangian`)
- ✅ 可处理任意不等式约束
- ✅ 自适应罚参数更新
- ✅ 拉格朗日乘子更新

---

## 🔧 安装依赖

```bash
pip install numpy scipy matplotlib
```

---

## 💡 使用方法

### 方法1：使用边界法处理约束

```python
from euler_lagrange_optimal_control import ConstrainedEulerLagrangeSolver

# 定义问题
def L(t, x, u):
    return 0.5 * u**2  # 积分代价

def f(t, x, u):
    return u  # 状态方程

# 创建求解器 (带约束 |u| ≤ 0.5)
solver = ConstrainedEulerLagrangeSolver(
    L=L,
    f=f,
    t_span=(0, 1),
    x0=0.0,
    xf=1.0,
    u_min=-0.5,
    u_max=0.5,
    n_points=100
)

# 求解
x_opt, u_opt = solver.solve_with_bounds()
```

### 方法2：使用增广拉格朗日方法

```python
# 使用增广拉格朗日求解
x_opt, u_opt, constraint_violation = solver.solve_augmented_lagrangian()
print(f"约束违反程度: {constraint_violation:.2e}")
```

### 方法3：间接法 (庞特里亚金最大值原理)

```python
from euler_lagrange_optimal_control import ConstrainedIndirectSolver

def H(t, x, u, lam):
    return 0.5 * u**2 + lam * (-x + u)

def dH_dx(t, x, u, lam):
    return -lam

def dH_dlam(t, x, u, lam):
    return -x + u

def optimal_u_unconstrained(t, x, lam):
    return -lam  # 无约束最优控制

# 创建求解器 (带饱和约束)
solver = ConstrainedIndirectSolver(
    H=H,
    dH_dx=dH_dx,
    dH_dlam=dH_dlam,
    optimal_u_unconstrained=optimal_u_unconstrained,
    t_span=(0, 2),
    x0=1.0,
    u_min=-0.3,
    u_max=0.3
)

t, x_opt, u_opt, lam_opt = solver.solve()
```

---

## 🎯 增广拉格朗日方法详解

### 数学原理

对于优化问题：
```
minimize   f(u)
subject to g_i(u) ≤ 0, i = 1, ..., m
```

增广拉格朗日函数为：
```
L_a(u, λ, c) = f(u) + (1/(2c)) * Σ [max(λ_i + c*g_i(u), 0)² - λ_i²]
```

### 算法流程

1. **初始化**：λ = 0, c = c₀
2. **无约束优化**：min L_a(u, λ, c)
3. **检查收敛**：若约束违反 < tol，停止
4. **更新乘子**：λ_i = max(λ_i + c*g_i(u), 0)
5. **更新罚参数**：c = β * c (β > 1)
6. **返回步骤2**

---

## 📊 运行示例

```bash
python euler_lagrange_optimal_control.py
```

### 示例1：Bang-Bang控制
- **问题**：最小时间控制，|u| ≤ 1
- **现象**：控制量在边界上（Bang-Bang）
- **文件**：`example_bang_bang.png`

### 示例2：带约束的最小能量
- **问题**：最小能量控制，|u| ≤ 0.5
- **现象**：控制量被钳位在边界
- **文件**：`example_constrained_energy.png`

### 示例3：饱和控制（间接法）
- **问题**：带横截条件的饱和控制
- **现象**：控制量在初始阶段饱和，后期释放
- **文件**：`example_saturated_control.png`

### 示例4：约束对比
- **问题**：有约束 vs 无约束对比
- **现象**：约束导致能量增加
- **文件**：`example_constraint_comparison.png`

---

## 📐 理论背景

### 庞特里亚金最小值原理

对于带约束 u ∈ U 的最优控制问题：
```
minimize J = Φ(x(T)) + ∫₀ᵀ L(t, x, u) dt
```

最优控制满足：
```
u*(t) = argmin_{u ∈ U} H(t, x*(t), u, λ*(t))
```

其中哈密顿函数：
```
H(t, x, u, λ) = L(t, x, u) + λ·f(t, x, u)
```

### Bang-Bang控制

当哈密顿函数关于 u 线性时：
```
H(t, x, u, λ) = a(t, x, λ) + b(t, x, λ)·u
```

最优控制为：
```
u*(t) = u_max  若 b(t, x, λ) < 0
u*(t) = u_min  若 b(t, x, λ) > 0
```

### 饱和控制

当哈密顿函数关于 u 二次时：
```
H = (1/2)u² + b·u
```

无约束最优：u* = -b
带约束最优：u* = clamp(-b, u_min, u_max)

---

## 📈 边界条件总结

| 边界类型 | 条件 | 控制特性 |
|---------|------|---------|
| 固定端点 | x(T) = xf | 可能不连续（拉格朗日乘子跳跃） |
| 自由端点 | λ(T) = ∂Φ/∂x | 控制量通常连续 |
| 控制约束 | u_min ≤ u ≤ u_max | Bang-Bang或饱和控制 |

---

## 📁 文件结构

```
.
├── euler_lagrange_optimal_control.py  # 主程序
├── README.md                           # 详细文档
├── example_bang_bang.png              # Bang-Bang控制结果
├── example_constrained_energy.png     # 带约束的能量最小化
├── example_saturated_control.png      # 饱和控制结果
└── example_constraint_comparison.png  # 约束对比
```

---

## 📝 重要提示

1. **增广拉格朗日 vs 边界法**：
   - 边界法更快，适用于简单约束
   - 增广拉格朗日更鲁棒，适用于复杂约束

2. **控制约束的影响**：
   - 约束通常会增加最优代价
   - 约束可能导致控制量饱和
   - 约束可能导致奇异弧

3. **数值稳定性**：
   - 初始猜测对收敛性很重要
   - 增广拉格朗日需适当选择初始罚参数

## 许可证

MIT License
