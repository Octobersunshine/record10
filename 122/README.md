# 声子玻尔兹曼输运方程（BTE）求解器 - 晶格热导率计算

本项目实现了基于弛豫时间近似（RTA）的声子玻尔兹曼输运方程求解器，用于计算纳米传热中的晶格热导率。

## 物理模型

### 1. 声子BTE与弛豫时间近似

在弛豫时间近似下，声子玻尔兹曼方程的解为：

$$ f = f_0 + \tau \left( -\frac{\partial f_0}{\partial T} \right) \mathbf{v}_g \cdot \nabla T $$

其中：
- $f_0$ 是玻色-爱因斯坦平衡分布
- $\tau$ 是弛豫时间
- $\mathbf{v}_g$ 是声子群速度

### 2. 晶格热导率公式

基于动能理论，晶格热导率为：

$$ \kappa = \frac{1}{3} \sum_{\mathbf{q},s} C_{\mathbf{q},s} v_{g,\mathbf{q},s}^2 \tau_{\mathbf{q},s} $$

连续形式：

$$ \kappa = \frac{1}{3} \int_0^{\omega_D} C(\omega) v_g^2(\omega) \tau(\omega) g(\omega) d\omega $$

### 3. 散射机制

本代码包含以下弛豫时间散射机制：

#### (1) Umklapp散射（三声子散射）
$$ \tau_U^{-1} = A \omega^2 T \exp(-\Theta_D / 3T) $$

#### (2) 边界散射（Casimir极限）
$$ \tau_B^{-1} = \frac{v}{L} $$

#### (3) 杂质散射（Rayleigh散射）
$$ \tau_I^{-1} = B \omega^4 $$

总弛豫时间遵循Matthiessen定则：
$$ \frac{1}{\tau_{total}} = \frac{1}{\tau_U} + \frac{1}{\tau_B} + \frac{1}{\tau_I} $$

## 文件结构

```
.
├── phonon_bte.py          # 核心BTE求解器类
├── example_thermal_conductivity.py  # 示例计算脚本
└── README.md              # 本说明文档
```

## 核心功能

### PhononBTE类

#### 初始化
```python
bte = PhononBTE(material='Si', L=None)
```
- `material`: 材料类型 ('Si', 'Ge', 'GaAs', 或其他)
- `L`: 特征尺寸（用于边界散射，单位：m）

#### 主要方法

| 方法 | 功能 |
|------|------|
| `thermal_conductivity(T)` | 计算温度T时的晶格热导率 |
| `spectral_thermal_conductivity(omega, T)` | 计算频谱热导率 |
| `cumulative_thermal_conductivity(T, max_lambda)` | 计算累积热导率 |
| `size_effect_thermal_conductivity(T, L_array)` | 计算尺寸效应 |
| `relaxation_time(omega, T)` | 计算总弛豫时间 |
| `mean_free_path(omega, T)` | 计算声子平均自由程 |

## 支持的材料参数

| 材料 | 密度 (kg/m³) | 纵向声速 (m/s) | 横向声速 (m/s) | 德拜温度 (K) |
|------|-------------|----------------|----------------|-------------|
| Si | 2330 | 8433 | 5845 | 640 |
| Ge | 5323 | 5410 | 3350 | 374 |
| GaAs | 5317 | 4730 | 3340 | 360 |

## 使用示例

### 1. 基本热导率计算
```python
from phonon_bte import PhononBTE

# 创建Si的BTE求解器
bte = PhononBTE(material='Si', L=None)

# 计算300K时的热导率
T = 300
kappa = bte.thermal_conductivity(T)
print(f"κ({T}K) = {kappa:.2f} W/mK")
```

### 2. 纳米尺度尺寸效应
```python
import numpy as np

# 计算不同尺寸下的热导率
L_array = np.logspace(-9, -6, 20)  # 1nm to 1μm
kappas = bte.size_effect_thermal_conductivity(300, L_array)
```

### 3. 不同材料对比
```python
materials = ['Si', 'Ge', 'GaAs']
for mat in materials:
    bte = PhononBTE(material=mat)
    kappa = bte.thermal_conductivity(300)
    print(f"{mat}: κ = {kappa:.2f} W/mK")
```

## 运行示例脚本

```bash
# 运行所有示例
python example_thermal_conductivity.py
```

脚本将生成以下结果和图表：
1. `thermal_conductivity_vs_T_Si.png` - 体硅热导率随温度变化
2. `thermal_conductivity_comparison.png` - 不同材料热导率对比
3. `size_effect_thermal_conductivity.png` - 纳米尺寸效应
4. `spectral_thermal_conductivity.png` - 频谱热导率分布
5. `cumulative_mfp.png` - 累积热导率与平均自由程
6. `relaxation_time.png` - 各散射机制弛豫时间对比

## 依赖库

- numpy
- scipy
- matplotlib

安装依赖：
```bash
pip install numpy scipy matplotlib
```

## 理论背景

### 德拜模型
代码使用德拜近似描述声子色散关系：
$$ \omega = v_s k $$

声子态密度：
$$ g(\omega) = \frac{3\omega^2}{2\pi^2 v_s^3} \quad (\omega \leq \omega_D) $$

### 模式热容
$$ C(\omega, T) = k_B \left( \frac{\hbar\omega}{k_B T} \right)^2 \frac{\exp(\hbar\omega/k_B T)}{[\exp(\hbar\omega/k_B T) - 1]^2} $$

## 纳米传热应用

本代码特别适用于：
1. 纳米线、纳米薄膜的热导率计算
2. 声子工程设计
3. 热管理材料的性能预测
4. 纳米尺度能量输运研究

## 参考文献

1. Chen, G. (2005). Nanoscale Energy Transport and Conversion. Oxford University Press.
2. Ziman, J. M. (1960). Electrons and Phonons. Oxford University Press.
3. Callaway, J. (1959). Model for Lattice Thermal Conductivity at Low Temperatures. Physical Review, 113(4), 1046.

## 作者信息

本代码用于教学和研究目的，基于经典的声子输运理论。
