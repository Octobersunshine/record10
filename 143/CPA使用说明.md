# CPA (Cubic-Plus-Association) 状态方程使用说明

## 概述

CPA状态方程结合了立方型状态方程（Peng-Robinson）和统计缔合流体理论（SAFT），能够准确描述含氢键组分（如水、醇类）的热力学性质和相平衡行为。

## CPA状态方程

### 压力表达式

$$P = P_{\text{physical}} + P_{\text{association}}$$

其中：
- **物理项**（PR方程）：
  $$P_{\text{physical}} = \frac{RT}{V - b} - \frac{a}{V(V + b) + b(V - b)}$$

- **缔合项**：
  $$P_{\text{association}} = -RT\rho \sum_i x_i \sum_A M_{iA} \left(\ln X_{iA} + \frac{1 - X_{iA}}{2}\right)$$

### 单体分数方程

$$X_{iA} = \frac{1}{1 + \rho \sum_j x_j \sum_B M_{jB} X_{jB} \Delta_{ijAB}}$$

### 缔合强度

$$\Delta_{ijAB} = g(\rho) \left[\exp\left(\frac{\varepsilon_{ijAB}}{RT}\right) - 1\right] \beta_{ijAB} b$$

## 安装依赖

```bash
pip install numpy scipy
```

## 快速开始

### 基本使用示例

```python
from cpa_eos import CPAEOS
import numpy as np

# 创建CPA对象
cpa = CPAEOS(R=8.314)

# 设置组分（使用内置参数库）
cpa.set_components(['water', 'methanol'])

# 设置二元相互作用参数
cpa.set_binary_interaction(0, 1, k_ij=-0.05)

# 计算泡点压力
T = 323.15
x = np.array([0.5, 0.5])
P_bubble, y = cpa.bubble_pressure(x, T)

print(f"泡点压力: {P_bubble/1e5:.2f} bar")
print(f"气相组成: y_water={y[0]:.4f}, y_methanol={y[1]:.4f}")
```

## 内置组分参数

目前支持以下组分的内置参数：

| 组分 | 英文名称 | 缔合位点 | 氢键类型 |
|------|----------|----------|----------|
| 水 | water | 4 | 2给体 + 2受体 |
| 甲醇 | methanol | 2 | 1给体 + 1受体 |
| 乙醇 | ethanol | 2 | 1给体 + 1受体 |
| 1-丙醇 | 1-propanol | 2 | 1给体 + 1受体 |
| 甲烷 | methane | 0 | 非缔合 |
| 乙烷 | ethane | 0 | 非缔合 |
| 丙烷 | propane | 0 | 非缔合 |
| 正丁烷 | n-butane | 0 | 非缔合 |
| 二氧化碳 | co2 | 0 | 非缔合 |
| 氮气 | n2 | 0 | 非缔合 |
| 苯 | benzene | 0 | 非缔合 |

### 添加自定义组分

```python
custom_comp = {
    'name': 'custom',
    'Tc': 500.0,           # 临界温度 (K)
    'Pc': 5e6,             # 临界压力 (Pa)
    'omega': 0.3,          # 偏心因子
    'a': 2.0,              # PR方程参数a
    'b': 4e-5,             # PR方程参数b
    'kappa': 0.5,          # alpha函数参数
    'assoc_sites': 2,      # 缔合位点数
    'assoc_sites_dict': {'H': 1, 'e': 1},  # 给体/受体
    'eps_assoc': 12000.0,  # 缔合能 (J/mol)
    'beta_assoc': 0.08     # 缔合体积参数
}

cpa.set_components([custom_comp, 'water'])
```

## 相平衡计算

### 1. 气液平衡（VLE）

#### 泡点压力计算

```python
# 给定液相组成和温度，计算泡点压力和气相组成
T = 323.15
x = np.array([0.3, 0.7])  # 液相摩尔分率
P_bubble, y = cpa.bubble_pressure(x, T, P_guess=1e5)

print(f"泡点压力: {P_bubble/1e5:.3f} bar")
print(f"气相组成: {y}")
```

#### 露点压力计算

```python
# 给定气相组成和温度，计算露点压力和液相组成
T = 323.15
y = np.array([0.6, 0.4])  # 气相摩尔分率
P_dew, x = cpa.dew_pressure(y, T, P_guess=1e5)

print(f"露点压力: {P_dew/1e5:.3f} bar")
print(f"液相组成: {x}")
```

### 2. 液液平衡（LLE）

```python
# 给定进料组成、温度和压力，计算液液分相
T = 298.15
P = 1e5
z = np.array([0.5, 0.5])  # 进料摩尔分率

beta, x1, x2 = cpa.lle_flash(z, T, P)

print(f"相1摩尔分率: {beta:.4f}")
print(f"相1组成: {x1}")
print(f"相2组成: {x2}")
```

## 二元相互作用参数设置

```python
# 设置van der Waals单流体混合规则的k_ij
cpa.set_binary_interaction(
    i=0, j=1, 
    k_ij=-0.05,    # 能量参数修正
    l_ij=0.0,      # 体积参数修正（可选）
    beta_ij=1.0,   # 交叉缔合beta修正（可选）
    gamma_ij=1.0   # 交叉缔合能量修正（可选）
)
```

### 常用体系的k_ij参考值

| 体系 | k_ij |
|------|------|
| 水-甲醇 | -0.05 ~ 0.0 |
| 水-乙醇 | -0.03 ~ 0.02 |
| 水-CO2 | 0.15 ~ 0.25 |
| 甲醇-CO2 | 0.05 ~ 0.1 |

## 热力学性质计算

### 逸度系数

```python
# 计算各组分的逸度系数
x = np.array([0.5, 0.5])
T = 323.15
P = 1e5

ln_phi = cpa.ln_phi(x, T, P)
phi = np.exp(ln_phi)

for i, comp in enumerate(['water', 'methanol']):
    print(f"{comp}: ln(phi) = {ln_phi[i]:.4f}, phi = {phi[i]:.4f}")
```

### 压力计算

```python
# 给定组成、温度和体积，计算压力
x = np.array([0.5, 0.5])
T = 323.15
V = 0.001  # m³/mol

P = cpa.pressure(x, T, V)
print(f"压力: {P/1e5:.3f} bar")
```

### 体积求解

```python
# 给定组成、温度和压力，求解体积
V_liquid = cpa.solve_volume(x, T, P, phase='liquid')
V_vapor = cpa.solve_volume(x, T, P, phase='vapor')

print(f"液相体积: {V_liquid:.6e} m³/mol")
print(f"气相体积: {V_vapor:.6e} m³/mol")
```

## 完整示例：水-乙醇体系VLE计算

```python
from cpa_eos import CPAEOS
import numpy as np

cpa = CPAEOS()
cpa.set_components(['water', 'ethanol'])
cpa.set_binary_interaction(0, 1, k_ij=0.0)

T = 298.15
x_water = np.linspace(0.1, 0.9, 9)

print("=" * 70)
print(f"水-乙醇体系VLE (T = {T} K)")
print("=" * 70)
print(f"{'x_water':>10} {'x_ethanol':>12} {'P (bar)':>12} {'y_water':>12} {'y_ethanol':>12}")
print("-" * 70)

for xw in x_water:
    x = np.array([xw, 1 - xw])
    P, y = cpa.bubble_pressure(x, T)
    print(f"{xw:10.4f} {1-xw:12.4f} {P/1e5:12.4f} {y[0]:12.4f} {y[1]:12.4f}")

print("=" * 70)
```

## 常见问题与注意事项

### 1. 收敛性问题

**现象**：计算不收敛或出现NaN

**解决方法**：
```python
# 1. 提供更好的初始猜测
P_bubble, y = cpa.bubble_pressure(x, T, P_guess=2e5)

# 2. 调整二元相互作用参数
cpa.set_binary_interaction(0, 1, k_ij=-0.02)

# 3. 检查组分参数是否合理
print(cpa.components)
```

### 2. 液液平衡计算

**注意事项**：
- LLE计算对二元参数k_ij非常敏感
- 通常需要较大的正k_ij才能产生分相
- 建议使用实验数据回归k_ij

### 3. 缔合参数

- 水：4个缔合位点（2给体H + 2受体e）
- 醇类：2个缔合位点（1给体H + 1受体e）
- 非极性组分：0个缔合位点

### 4. 单位一致性

请确保使用一致的单位：
- 温度：开尔文 (K)
- 压力：帕斯卡 (Pa)
- 体积：立方米每摩尔 (m³/mol)
- 气体常数R = 8.314 J/(mol·K)

## 性能优化

### 避免重复计算

```python
# 预先计算混合物参数
a_mix, b_mix = cpa.calculate_mixture_ab(x, T)

# 预先计算单体分数
X = cpa.calculate_monmer_fractions(x, T, V, b_mix)
```

### 参数回归

对于特定体系，建议使用实验数据回归二元相互作用参数k_ij：

```python
def objective(k_ij):
    cpa.set_binary_interaction(0, 1, k_ij=k_ij[0])
    error = 0.0
    for x_exp, P_exp in zip(x_experimental, P_experimental):
        P_calc, _ = cpa.bubble_pressure(x_exp, T)
        error += ((P_calc - P_exp) / P_exp) ** 2
    return error

from scipy.optimize import minimize
result = minimize(objective, [0.0], bounds=[[-0.2, 0.2]])
k_ij_opt = result.x[0]
```

## 参考文献

1. Kontogeorgis, G. M., et al. (1996). "Cubic-Plus-Association (CPA) Equation of State for Associating Fluids."
2. Michelsen, M. L., & Mollerup, J. M. (2007). "Thermodynamic Models: Fundamentals & Computational Aspects."
