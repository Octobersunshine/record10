# 分数阶微分方程数值求解器

本项目实现了多种分数阶微分方程的数值解法，包括线性和非线性方程。**已修复α接近1时的数值震荡问题，并增加了预测-校正法求解非线性方程。**

## 📁 文件列表

| 文件 | 功能 |
|------|------|
| `fractional_ode_solver.py` | 线性方程求解器（分数阶弛豫方程） |
| `nonlinear_fractional_solver.py` | **非线性方程求解器（预测-校正法）** ⭐ |
| `improved_solver.py` | 改进版求解器（三种方法对比） |
| `simple_test.py` | 简化测试程序 |

---

## 🔧 第一部分：线性分数阶方程

### 稳定性改进

当分数阶数α接近1时，传统的显式格式容易出现数值震荡。已实现三种稳定的数值方法：

| 方法 | 稳定性 | 精度 | 推荐场景 |
|------|--------|------|----------|
| 显式格式 | 条件稳定 | 一般 | α较小（<0.8）时使用 |
| **隐式格式** | **无条件稳定** | 较好 | ⭐ 推荐使用，适用于所有α值 |
| L1格式 | 无条件稳定 | 高 | 需要高精度的科研计算 |

### 数学模型：分数阶弛豫方程

```
D^α f(t) = -λ f(t),  f(0) = f0
```

解析解用Mittag-Leffler函数表示：
```
f(t) = f0 * E_α(-λ t^α)
```

---

## 🌀 第二部分：非线性分数阶方程 ⭐

### 求解方法：预测-校正法 (Predictor-Corrector)

- **预测步**：Adams-Bashforth，用显式格式预估下一时刻值
- **校正步**：Adams-Moulton，用隐式格式修正预测值
- 二阶精度，无条件稳定，适合非线性系统

### 已实现的非线性模型

#### 1. 分数阶Lotka-Volterra捕食者-食饵模型

```
D^α1 x = a x - b x y
D^α2 y = -c y + d x y
```

其中：
- `x`: 食饵种群数量
- `y`: 捕食者种群数量
- `a`: 食饵内禀增长率
- `b`: 捕食率
- `c`: 捕食者死亡率
- `d`: 转化率

**动力学行为**：
- α=1时：周期振荡，振幅恒定
- α<1时：振幅逐渐衰减，振荡周期变长

#### 2. 分数阶Van der Pol振子

```
D^α1 x = y
D^α2 y = μ (1 - x²) y - x
```

**动力学行为**：
- 极限环振荡
- 分数阶导致特殊的分岔行为

#### 3. 分数阶Lorenz混沌系统

```
D^α1 x = σ (y - x)
D^α2 y = x (ρ - z) - y
D^α3 z = x y - β z
```

**动力学行为**：
- 混沌吸引子（蝴蝶效应）
- 分数阶改变混沌阈值

---

## 🚀 使用方法

### 安装依赖

```bash
pip install numpy matplotlib scipy
```

### 运行程序

```bash
# 线性方程求解器
python fractional_ode_solver.py

# 非线性方程求解器（推荐）
python nonlinear_fractional_solver.py
```

### 使用示例

#### 求解分数阶Lotka-Volterra模型

```python
from nonlinear_fractional_solver import NonlinearFractionalSolver, lotka_volterra

# 创建求解器
alpha = [0.9, 0.9]  # 两个变量可以有不同的阶数
solver = NonlinearFractionalSolver(alpha, t_final=50.0, h=0.01, dim=2)

# 设置参数和初始条件
params = [1.0, 1.0, 1.0, 1.0]  # [a, b, c, d]
x0 = [0.5, 0.5]

# 使用预测-校正法求解
t, x = solver.predictor_corrector(lotka_volterra, x0, params)
```

#### 求解分数阶Lorenz系统

```python
from nonlinear_fractional_solver import NonlinearFractionalSolver, lorenz

alpha = [0.99, 0.99, 0.99]
solver = NonlinearFractionalSolver(alpha, t_final=50.0, h=0.01, dim=3)

params = [10.0, 28.0, 8.0/3.0]  # [σ, ρ, β]
x0 = [1.0, 1.0, 1.0]

t, x = solver.predictor_corrector(lorenz, x0, params)
```

---

## 📊 输出文件

### 线性方程（弛豫模型）

| 文件 | 内容 |
|------|------|
| `fractional_relaxation.png` | 数值解与解析解对比 |
| `stability_comparison.png` | 不同数值方法稳定性对比 |
| `different_alphas.png` | 不同阶数α的解对比 |

### 非线性方程

| 文件 | 模型 | 内容 |
|------|------|------|
| `lotka_volterra.png` | Lotka-Volterra | 时间演化 + 相空间轨迹 |
| `lv_different_alphas.png` | Lotka-Volterra | 不同α值对比 |
| `lv_frac_vs_int.png` | Lotka-Volterra | 分数阶 vs 整数阶 |
| `van_der_pol.png` | Van der Pol | 极限环振荡 |
| `lorenz.png` | Lorenz | 混沌吸引子（2D + 3D） |

---

## 💡 算法特点

### 预测-校正法的优势

1. **无条件稳定**：解决了α接近1时的震荡问题
2. **二阶精度**：比显式格式精度更高
3. **通用性强**：适用于任意非线性系统
4. **多变量支持**：每个变量可以有不同的分数阶数

### 计算复杂度

- 时间复杂度：O(N²)，其中N为时间步数
- 空间复杂度：O(N)

---

## 🔬 分数阶动力学的特点

1. **记忆效应**：系统演化依赖于全部历史，而非仅当前状态
2. **慢衰减**：解的衰减速度比整数阶慢
3. **长程关联**：时间序列具有长程相关性
4. **分岔延迟**：分岔点随α值变化
5. **混沌阈值改变**：分数阶系统的混沌参数区间不同

---

## 📚 扩展说明

### 如何添加新的非线性方程

1. 定义右端函数：
```python
def my_ode(t, x, params):
    dx = np.zeros_like(x)
    dx[0] = ...  # 第一个方程
    dx[1] = ...  # 第二个方程
    return dx
```

2. 调用求解器：
```python
solver = NonlinearFractionalSolver(alpha, t_final, h, dim=n)
t, x = solver.predictor_corrector(my_ode, x0, params)
```

### 已测试的参数范围

- α: 0.7 ~ 1.0（稳定范围）
- h: 0.005 ~ 0.02（推荐步长）
- t_final: 根据系统特性调整

---

## 📝 验证结果

当α=0.95时：
- 显式格式：出现数值震荡 ⚠️
- 预测-校正法：无震荡，稳定收敛 ✓ ⭐

分数阶Lotka-Volterra与整数阶对比：
- 整数阶：恒定振幅周期振荡
- 分数阶：振幅逐渐衰减，振荡周期变长
