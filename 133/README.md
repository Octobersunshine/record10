# AGN综合光谱模型 - 黑洞吸积盘多波段辐射谱（v3.0）
=================================================================

## 简介

本项目实现了完整的活动星系核（AGN）光谱模型，包含从光学到X射线的多波段辐射计算和黑洞自旋参数提取功能。

## 版本历史

### v3.0 (2026年5月) - 重大增强
- ✅ **相对论性铁Kα线展宽模型**：基于Laor (1991)模型，包含引力红移、多普勒效应、相对论性聚束
- ✅ **盘风/喷流模型**：宽线区(BLR)发射线、蓝移吸收线、外流特征
- ✅ **X射线综合模型**：幂律连续谱、康普顿反射峰、热连续谱
- ✅ **黑洞自旋拟合器**：通过拟合X射线能谱提取自旋参数
- ✅ **模拟观测数据生成器**：用于测试拟合算法

### v2.0 (2026年5月) - 边界条件修正
- ✅ **正确实现ISCO处零力矩边界条件**：F(R_isco) = 0
- ✅ **修正辐射通量计算公式**：使用Page & Thorne (1974)方程(12)的积分形式
- ✅ **能量守恒验证**：积分总光度与辐射效率预测一致（误差<5%）
- ✅ **添加边界条件验证功能**：自动检查ISCO处通量和能量守恒

## 物理模型

Novikov-Thorne模型是相对论性的薄吸积盘模型，基于以下关键假设：

1. **几何薄盘**：盘的厚度远小于半径 (H << R)
2. **光学厚**：盘在垂直方向光学厚，辐射为热辐射
3. **稳态吸积**：吸积率不随时间变化
4. **开普勒旋转**：物质近似沿开普勒轨道旋转

## 核心公式

### 1. ISCO半径（最内稳定圆轨道）

对于Kerr黑洞的ISCO半径为：

```
R_isco = f(a*) * R_g
```

其中 R_g = GM/c² 是引力半径，a*是无量纲自旋参数。

### 2. 辐射流量剖面（Novikov-Thorne公式）

```
F(r) = - (M_dot * Ω * dΩ/dr) / (4π E²) * (L - L_isco * E/E_isco)
```

其中：
- Ω(r) 是角速度
- E(r) 是单位质量的比能量
- L(r) 是单位质量的比角动量

### 3. 有效温度

```
T(r) = (F(r) / σ_SB)^(1/4)
```

### 4. 多波段谱

通过积分每个环带的黑体辐射得到总谱：

```
L_ν = cos(i) * ∫[R_isco, R_out] 4π r * B_ν(T(r)) dr
```

其中 B_ν(T) 是普朗克函数。

## 安装依赖

```bash
pip install numpy scipy matplotlib
```

## 使用方法

### 基本用法

```python
from novikov_thorne_disk import NovikovThorneDisk

# 创建吸积盘模型
disk = NovikovThorneDisk(
    M_BH=1e8,        # 黑洞质量 (M_sun)
    M_dot=1.0,       # 吸积率 (M_sun/yr)
    a_star=0.7,      # 自旋参数 (-1 ~ 1)
    inclination=30.0, # 倾角 (度)
    distance=1e6      # 距离 (pc)
)

# 打印盘的性质
disk.print_disk_properties()

# 计算光谱
import numpy as np
nu_array = np.logspace(14, 19, 100)  # 频率范围 (Hz)
L_nu, F_nu = disk.compute_spectrum(nu_array)

# 计算 nu*L_nu
nuL_nu = disk.compute_nuLnu(nu_array)
```

### 绘制图形

```python
import matplotlib.pyplot as plt

# 温度剖面
fig, ax = plt.subplots()
disk.plot_temperature_profile(ax=ax)
plt.show()

# 辐射谱
fig, ax = plt.subplots()
disk.plot_spectrum(ax=ax)
plt.show()
```

### 获取各波段流量

```python
band_fluxes = disk.get_band_fluxes()
for band, data in band_fluxes.items():
    print(f"{band}: {data['L_sun_units']:.4e} L_sun")
```

### 验证边界条件和能量守恒

```python
# 验证零力矩边界条件
bc = disk.verify_boundary_conditions()
print(f"ISCO附近通量: {bc['F_near_ISCO']:.2e} erg/cm²/s")
print(f"能量守恒比: {bc['energy_conservation_ratio']:.4f}")

# 计算总光度（通过积分流量）
L_bol = disk.compute_total_luminosity()
print(f"积分总光度: {L_bol:.4e} erg/s")
```

## 运行示例

```bash
python novikov_thorne_disk.py
```

## 参数说明

| 参数 | 说明 | 单位 | 典型值 |
|------|------|------|--------|
| M_BH | 黑洞质量 | M_sun | 10^6 ~ 10^10 (AGN)<br>10 ~ 100 (X射线双星) |
| M_dot | 吸积率 | M_sun/yr | 0.01 ~ 10 (AGN)<br>1e-10 ~ 1e-7 (X射线双星) |
| a_star | 自旋参数 | 无量纲 | -1 ~ 1 |
| inclination | 倾角 | 度 | 0 ~ 90 |
| distance | 距离 | pc | 1e3 ~ 1e9 |

## 输出波段

- **光学 (V)**: ~5500 Å
- **紫外 (UV)**: 1000 ~ 3000 Å
- **软X射线**: 0.1 ~ 2 keV
- **硬X射线**: 2 ~ 10 keV

## 新增功能：相对论性铁线与自旋测量（v3.0）

### 1. 相对论性铁Kα线模型

```python
from agn_spectral_model import RelativisticIronLine

# 创建铁线模型
iron_line = RelativisticIronLine(
    M_BH=1e8,
    a_star=0.7,          # 自旋参数
    inclination=30,      # 倾角
    line_energy=6.4,     # 静止系能量 (keV)
    emissivity_index=3.0  # ε ∝ r^-q
)

# 计算线轮廓
import numpy as np
energy_grid = np.linspace(4.0, 8.0, 200)
profile = iron_line.compute_profile_fast(energy_grid)
```

**相对论效应包括：**
- 引力红移：靠近黑洞的光子被红移
- 多普勒红移/蓝移：旋转盘的两侧分别被红移和蓝移
- 相对论性聚束：接近侧的辐射被增强
- 横向多普勒效应：运动时钟变慢

### 2. 盘风/喷流模型

```python
from agn_spectral_model import DiskWindModel

wind = DiskWindModel()

# 添加自定义宽线
wind.add_broad_line('FeII', 4570.0, fwhm_km_s=4000, strength=0.8)

# 添加自定义吸收线
wind.add_absorption_line('SiIV', 1394.0, velocity_km_s=-3000, depth=0.25)

# 计算盘风贡献
wavelength = np.linspace(1000, 7000, 300)
emission, absorption = wind.compute_wind_contribution(wavelength)
```

### 3. AGN综合光谱模型

```python
from agn_spectral_model import AGNSpectralModel

# 创建完整AGN模型
agn = AGNSpectralModel(
    M_BH=1e8,
    M_dot=0.5,
    a_star=0.7,
    inclination=30
)

# 计算X射线能谱
energy_keV = np.logspace(0, 2, 200)
xray_spec, components = agn.compute_xray_spectrum(energy_keV)

# components包含:
#   'power_law': 冕区幂律连续谱
#   'iron_line': 相对论性铁Kα线
#   'compton_hump': 康普顿反射峰
#   'disk_thermal': 热盘贡献

# 计算紫外-光学光谱
wavelength = np.linspace(1000, 7000, 300)
uv_spec, uv_components = agn.compute_uv_optical_spectrum(wavelength)
```

### 4. 黑洞自旋拟合

```python
from agn_spectral_model import AGNSpectralModel, MockObservation, SpinFitter
import numpy as np

# 创建模型（已知自旋）
true_a_star = 0.7
model = AGNSpectralModel(M_BH=1e8, M_dot=0.5, a_star=true_a_star)

# 生成模拟观测数据
obs = MockObservation(model)
energy, data, error = obs.generate_xray_data(
    energy_min=4.0, 
    energy_max=8.0,
    n_points=80, 
    noise_level=0.05
)

# 拟合自旋
fitter = SpinFitter(obs, true_a_star=true_a_star)
result = fitter.fit_spin(a_guess=0.5)

# 输出结果
print(f"测量自旋: {result['a_star']:.3f} ± {result['a_star_error']:.3f}")
print(f"真实自旋: {true_a_star:.3f}")
print(f"约化χ²: {result['red_chi2']:.2f}")

# 绘制拟合结果
fig = fitter.plot_fit_result()
fig.savefig('spin_fit.png', dpi=150)
```

### 5. 运行完整示例

```bash
python agn_spectral_model.py
```

将生成三个图像：
- `iron_line_spin_comparison.png`: 不同自旋的铁线轮廓比较
- `full_agn_spectrum.png`: 完整多波段光谱（紫外-光学 + X射线）
- `spin_fitting_result.png`: 自旋拟合结果

## 物理模型组件

| 组件 | 说明 | 典型参数 |
|------|------|----------|
| 热连续谱 | Novikov-Thorne薄盘 | T ∝ r^-3/4 |
| 幂律连续谱 | 冕区康普顿化 | Γ = 1.7-2.0 |
| 铁Kα线 | 相对论性展宽 | 6.4 keV (中性) / 6.7 keV (电离) |
| 康普顿峰 | 反射谱特征 | ~20-30 keV |
| 宽发射线 | 盘风/BLR | FWHM = 1000-10000 km/s |
| 蓝移吸收线 | 外流吸收 | v = -1000 ~ -10000 km/s |

## 自旋测量原理

黑洞自旋通过相对论性铁Kα线的轮廓进行测量：

1. **高自旋黑洞 (a* ~ 1)**: ISCO半径小 (R_isco ~ 1.2 R_g)，内盘更靠近黑洞，引力红移更强，线轮廓更宽更红
2. **低自旋黑洞 (a* ~ 0)**: ISCO半径大 (R_isco = 6 R_g)，线轮廓相对较窄

## 参考

### Novikov-Thorne盘模型
1. Novikov, I. D. & Thorne, K. S. (1973), Black Holes, p. 343
2. Page, D. N. & Thorne, K. S. (1974), ApJ, 191, 499
3. Frank, J., King, A., & Raine, D. (2002), Accretion Power in Astrophysics

### 相对论性铁线
4. Laor, A. (1991), ApJ, 376, 90
5. Fabian, A. C. et al. (2000), PASP, 112, 1145

### 盘风/AGN光谱
6. Proga, D. (2007), arXiv:0706.4063
7. Marziani, P. et al. (2026), CAOSP, 56, 87
