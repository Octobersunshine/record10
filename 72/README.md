# 复制动态方程与网络演化博弈

## 简介

本项目用Python实现了演化博弈论中的两大核心模型：
1. **复制动态方程（Replicator Dynamics）** - 均匀混合种群的演化
2. **网络演化博弈（Network Evolutionary Game）** - 结构化种群的演化，个体仅与邻居交互

用于求解对称博弈（如鹰鸽博弈）的演化稳定策略（Evolutionarily Stable Strategy, ESS），并分析空间结构对策略演化的影响。

## 功能特性

### 均匀混合种群（复制动态）
1. **复制动态方程实现** - 描述策略频率随时间演化的微分方程
2. **演化稳定策略求解** - 自动计算博弈的ESS
3. **鹰鸽博弈示例** - 经典的对称博弈模型
4. **可视化功能** - 策略演化轨迹图和相图
5. **数值稳定性修复** - 处理边界情况的数值问题

### 网络演化博弈（空间结构）
1. **多种网络拓扑** - 随机图、小世界网络、二维网格网络
2. **两种更新规则** - Fermi概率规则、最优响应规则
3. **邻居交互机制** - 个体仅与网络邻居进行博弈
4. **空间分布可视化** - 策略在网络上的空间分布
5. **演化动画生成** - 策略演化过程的动态展示
6. **拓扑比较分析** - 不同网络拓扑对演化结果的影响

## 依赖库

```
numpy
matplotlib
scipy
networkx
```

安装依赖：
```bash
pip install numpy matplotlib scipy networkx
```

## 核心函数说明

### 1. replicator_dynamics(x, t, payoff_matrix)

复制动态方程的核心实现（带数值稳定性修复）：

```python
dx/dt = x_i * (f_i(x) - φ(x))
```

其中：
- `f_i(x)` = 策略i的适应度
- `φ(x)` = 种群平均适应度

**数值稳定性修复**：
- 使用 `EPS = 1e-10` 避免除零和数值不稳定
- 对输入进行裁剪，确保频率在 `[EPS, 1-EPS]` 范围内
- 边界约束：当频率接近0时，只允许非负变化率；当频率接近1时，只允许非正变化率

### 2. solve_replicator_dynamics(payoff_matrix, initial_x, t_span)

使用`scipy.integrate.odeint`数值求解复制动态方程。

### 3. hawk_dove_game(V, C)

创建鹰鸽博弈的支付矩阵：

|        | 鹰    | 鸽    |
|--------|-------|-------|
| **鹰** | (V-C)/2 | V     |
| **鸽** | 0     | V/2   |

### 4. find_ess(payoff_matrix)

寻找2×2博弈的演化稳定策略：

- 纯策略ESS：检查占优策略
- 混合策略ESS：求解均衡点p = (d - b) / (a - b - c + d)

## 使用示例

```python
import numpy as np
from replicator_dynamics import *

# 1. 定义鹰鸽博弈参数
V = 2  # 资源价值
C = 4  # 争斗成本

# 2. 创建支付矩阵
payoff_matrix = hawk_dove_game(V, C)
print("支付矩阵:")
print(payoff_matrix)

# 3. 求解ESS
ess = find_ess(payoff_matrix)
print(f"\nESS: 鹰={ess[0]:.2f}, 鸽={ess[1]:.2f}")

# 4. 模拟演化过程
initial_x = np.array([0.3, 0.7])  # 初始策略频率
t_span = [0, 50]
t, x = solve_replicator_dynamics(payoff_matrix, initial_x, t_span)

# 5. 可视化结果
plot_replicator_dynamics(t, x, ['鹰策略', '鸽策略'])
plot_phase_portrait(payoff_matrix)
```

## 理论背景

### 复制动态方程

复制动态方程描述了种群中各策略频率的变化率。适应度高于平均水平的策略频率会增加，反之则减少。

### 演化稳定策略 (ESS)

一个策略是ESS，如果它能够抵御任何小的突变策略入侵。数学上，ESS满足：

1. u(s, s) ≥ u(s', s)
2. 如果u(s, s) = u(s', s)，则u(s, s') > u(s', s')

### 鹰鸽博弈分析

在鹰鸽博弈中：
- 当V > C时：鹰策略是ESS
- 当V < C时：混合策略是ESS，p = V/C
- 当V = C时：中性稳定

### 网络演化博弈

在网络演化博弈中，个体仅与其网络邻居交互，而非与整个种群随机交互。这导致：

1. **空间效应**：策略可能形成空间聚类或图案
2. **拓扑效应**：不同网络结构（度分布、聚类系数）影响演化结果
3. **协同演化**：网络结构和策略可能共同演化

**常见网络拓扑特性**：
- **随机图**：均匀度分布，低聚类系数
- **小世界网络**：短平均路径长度，高聚类系数
- **网格网络**：规则结构，局部交互强

**策略更新规则**：
- **Fermi规则**：概率性更新，存在随机性
- **最优响应**：确定性更新，模仿最优邻居

## 文件结构

```
.
├── replicator_dynamics.py    # 主程序
└── README.md                 # 说明文档
```

## 数值稳定性修复说明

### 修复的问题

**问题：当策略频率为0或接近边界时，模拟可能崩溃**

1. **除零错误**：当频率接近0时，数值计算可能产生除零或NaN
2. **边界溢出**：数值求解器可能产生超出[0,1]范围的频率
3. **归一化失效**：数值误差累积导致频率和不为1

### 修复方案

#### 1. 复制动态方程层面 (`replicator_dynamics` 函数)

```python
EPS = 1e-10
x_clipped = np.clip(x, EPS, 1 - EPS)  # 避免极端值
near_zero = x < EPS
near_one = x > 1 - EPS
dxdt[near_zero] = np.maximum(dxdt[near_zero], 0)  # 边界约束
dxdt[near_one] = np.minimum(dxdt[near_one], 0)
```

#### 2. 求解器层面 (`solve_replicator_dynamics` 函数)

```python
# 初始条件预处理
initial_x = np.clip(initial_x, EPS, 1 - EPS)
initial_x = initial_x / np.sum(initial_x)

# 严格的积分参数
odeint(..., rtol=1e-8, atol=1e-8, mxstep=5000)

# 后处理确保有效性
x = np.clip(x, 0, 1)
x = x / np.sum(x, axis=1, keepdims=True)
```

#### 3. 边界情况测试

程序内置4个测试案例验证稳定性：
- 正常初始条件: [0.3, 0.7]
- 纯鸽策略边界: [0.0, 1.0]
- 纯鹰策略边界: [1.0, 0.0]
- 接近边界的初始条件: [0.0001, 0.9999]

## 运行程序

```bash
python replicator_dynamics.py
```

## 扩展应用

本代码框架可以轻松扩展到其他对称博弈：
- 囚徒困境
- 雪堆博弈
- 协调博弈
- 石头-剪刀-布

只需修改支付矩阵即可分析不同的博弈模型。
