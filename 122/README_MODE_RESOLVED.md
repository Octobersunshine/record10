# 模式分辨声子BTE求解器 - 增强版

本项目对原始灰体近似（单弛豫时间）模型进行了重大改进，实现了**模式分辨弛豫时间**模型，大幅提高了计算精度。

## 问题：灰体近似的局限性

灰体近似（Gray Body Approximation）假设所有声子模式具有相同的弛豫时间，这在实际情况中存在显著误差，因为：

1. **不同声子支（LA/TA）的声速不同**
   - 纵波声速 > 横波声速
   - 对热导率贡献差异显著

2. **不同频率声子的散射强度不同**
   - 低频声子：边界散射主导，弛豫时间长
   - 高频声子：Umklapp散射主导，弛豫时间短

3. **实验观测**：硅在300K时，TA声子贡献约50-60%热导率，LA声子贡献约40-50%

## 改进方案：模式分辨模型

### 1. 声子支分辨（Phonon Branch Resolution）

| 声子支 | 偏振 | 声速（Si） | 系数 |
|--------|------|-----------|------|
| LA | 纵波 | 8433 m/s | B_U_LA, B_N_LA |
| TA1 | 横波1 | 5845 m/s | B_U_TA, B_N_TA |
| TA2 | 横波2 | 5845 m/s | B_U_TA, B_N_TA |

各支独立计算热导率：
$$ \kappa_{total} = \sum_{branch} \kappa_{branch} $$

### 2. 增强的散射机制

#### (1) 正常过程（N过程）散射
$$ \tau_N^{-1} = B_N \omega^2 T^3 $$

N过程虽然不直接产生热阻，但影响声子分布，对总热导率有重要作用。

#### (2) Umklapp过程（U过程）散射
低温区（T < Θ_D/4）：
$$ \tau_U^{-1} = B_U \omega^2 T \exp(-\Theta_D / 3T) $$

高温区（T ≥ Θ_D/4）：
$$ \tau_U^{-1} = B_U \omega^2 T $$

#### (3) 同位素散射（Rayleigh型）
$$ \tau_{iso}^{-1} = A_{iso} \omega^4 $$

#### (4) 边界散射（Casimir极限）
$$ \tau_B^{-1} = \frac{v_g(k, branch)}{L} $$

### 3. 色散关系修正

线性德拜模型：ω = v_s k

修正色散（考虑周期性边界）：
$$ \omega(k) = (1-\alpha) v_s k + \alpha \cdot \omega_{max} \sin\left(\frac{\pi k}{2 k_D}\right) $$

其中α ≈ 0.3为修正系数。

### 4. 第一性原理数据接口

支持直接加载第一性原理计算结果：
```python
bte.load_first_principles_data(
    omega_array,    # 频率点
    tau_dict,       # 各支弛豫时间 {'LA': array, 'TA1': array, ...}
    v_g_dict,       # 各支群速度
    dos_dict        # 各支态密度
)
```

## 文件结构

```
.
├── phonon_bte.py              # 原始版本（灰体近似）
├── phonon_bte_enhanced.py     # 增强版本（模式分辨）
│   ├── PhononBranch          # 声子支类
│   └── EnhancedPhononBTE     # 增强BTE求解器
├── example_thermal_conductivity.py  # 原始示例
├── example_mode_resolved.py         # 模式分辨示例
├── README.md                        # 原始文档
└── README_MODE_RESOLVED.md          # 本文档
```

## 核心增强功能

### EnhancedPhononBTE类

#### 初始化参数
```python
bte = EnhancedPhononBTE(
    material='Si',              # 材料: 'Si', 'Ge', 'GaAs'
    L=None,                     # 特征尺寸（边界散射）
    use_branch_resolved=True,   # 是否使用支分辨
    use_dispersion_correction=True  # 是否使用色散修正
)
```

#### 主要新方法

| 方法 | 功能 |
|------|------|
| `thermal_conductivity_branch(T, branch)` | 单支热导率 |
| `get_branch_contributions(T)` | 各支贡献统计 |
| `compare_models(T)` | 模式分辨vs灰体对比 |
| `dispersion_relation(k, branch)` | 色散关系 |
| `group_velocity(k, branch)` | 群速度 |
| `load_first_principles_data(...)` | 加载第一性原理数据 |

## 使用示例

### 1. 模型对比
```python
from phonon_bte_enhanced import EnhancedPhononBTE

bte = EnhancedPhononBTE(material='Si')
result = bte.compare_models(T=300)
print(f"模式分辨: {result['mode_resolved']:.2f} W/mK")
print(f"灰体近似: {result['gray_body']:.2f} W/mK")
print(f"差异: {result['difference']:+.1f}%")
```

### 2. 声子支贡献分析
```python
contributions = bte.get_branch_contributions(T=300)
for branch, kappa in contributions.items():
    print(f"{branch}: {kappa:.2f} W/mK")
```

### 3. 使用第一性原理数据
```python
omega = np.logspace(12, 14, 100)
tau_LA = 1e-12 * (omega / 1e13)**(-1.5)
tau_TA = 1e-12 * (omega / 1e13)**(-2.0)

tau_dict = {'LA': tau_LA, 'TA1': tau_TA, 'TA2': tau_TA}
vg_dict = {'LA': vg_LA, 'TA1': vg_TA, 'TA2': vg_TA}
dos_dict = {'LA': dos_LA, 'TA1': dos_TA, 'TA2': dos_TA}

bte.load_first_principles_data(omega, tau_dict, vg_dict, dos_dict)
kappa = bte.thermal_conductivity(300)
```

## 运行示例脚本

```bash
python example_mode_resolved.py
```

生成的图表：
1. `model_comparison.png` - 灰体vs模式分辨模型对比
2. `branch_contributions.png` - 各声子支热导率贡献
3. `tau_branch_comparison.png` - 不同声子支弛豫时间对比
4. `scattering_mechanisms_TA1.png` - TA支各散射机制分析
5. `dispersion_correction.png` - 色散关系修正对比
6. `size_effect_mode_resolved.png` - 模式分辨下的尺寸效应

## 精度改进分析

### 理论预期差异

| 温度范围 | 灰体近似误差 | 原因 |
|---------|------------|------|
| 低温 (<200K) | 低估10-20% | 边界散射对长波声子影响被平均 |
| 室温 (300K) | 差异5-15% | TA支弛豫时间长于LA支，灰体平均后偏差 |
| 高温 (>500K) | 差异减小 | U过程主导，各支弛豫时间差异减小 |

### 硅的典型结果（300K）

| 模型 | 热导率 (W/mK) | 说明 |
|------|--------------|------|
| 灰体近似 | ~120-140 | 单弛豫时间平均 |
| 模式分辨 | ~140-160 | LA+TA独立计算 |
| 实验值 | ~150 | 文献值 |

## 纳米传热应用

模式分辨模型在纳米尺度的优势：

1. **尺寸效应更准确**
   - 不同MFP声子被边界散射的程度不同
   - TA声子MFP通常更长，对尺寸更敏感

2. **声子工程设计**
   - 针对性抑制特定声子支的输运
   - 优化热电材料ZT值

3. **跨尺度模拟**
   - 可与第一性原理结果直接耦合
   - 支持多尺度材料设计

## 参考文献

1. Chen, G. (2005). Nanoscale Energy Transport and Conversion.
2. Ziman, J. M. (1960). Electrons and Phonons.
3. Srivastava, G. P. (1990). The Physics of Phonons.
4. Broido, D. A., et al. (2007). Intrinsic lattice thermal conductivity of semiconductors from first principles. Applied Physics Letters, 91(23), 231922.
5. Ward, A., et al. (2009). Ab initio theory of the lattice thermal conductivity in diamond. Physical Review B, 80(12), 125203.

## 技术细节

### 数值积分
- 使用Simpson积分（k空间）
- 默认200个k点，可调节精度
- 自动跳过光学支（热导率贡献小）

### 数值稳定性
- ω→0时的奇异性处理
- 高温/低温下的指数溢出保护
- 各散射率的数值上下限截断

### 可扩展性
- 支持自定义材料参数
- 支持添加新的散射机制
- 支持自定义色散关系
