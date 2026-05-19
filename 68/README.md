# 一维能量平衡模型 (1D Energy Balance Model)

## 模型概述

这是一个一维能量平衡模型（纬度带，经向平均），用于求解温度分布 T(φ,t)。模型包含以下物理过程：

1. **短波辐射吸收** - 太阳辐射的纬度分布和反照率效应
2. **长波辐射发射** - 依据Budyko-Sellers参数化的红外辐射冷却（含温室气体效应）
3. **云反馈参数化** - 云量随温度变化及其对辐射的影响
4. **CO₂辐射强迫** - 基于RCMIP场景的温室气体强迫
5. **经向热扩散** - 模拟大气和海洋的热量输送

---

## 新增功能：云反馈参数化与RCMIP敏感性实验

### 1. 云反馈参数化

#### 云量参数化方案

云量计算基于纬度分布和温度依赖性：

```python
def compute_cloud_cover(self, T, step):
    T_anomaly = T - 288.0
    c = self.c_base + self.c_lat + self.c_T * T_anomaly
    c = np.clip(c, 0.2, 0.9)
    return c
```

其中：
- `c_base = 0.65` - 基准云量
- `c_lat` - 纬度依赖项（极地云量较低）
- `c_T = -0.015 K⁻¹` - 温度敏感性（变暖减少云量，正反馈）

#### 云辐射强迫

云对辐射有双重效应：

```python
def compute_cloud_forcing(self, T, c, step):
    # 短波反照率效应（冷却）
    sw_forcing = -c_sw_eff * sw_incident * c_anomaly
    
    # 长波温室效应（变暖）
    lw_forcing = c_lw_eff * sigma_T4 * c_anomaly
    
    return sw_forcing, lw_forcing
```

参数：
- `c_sw_eff = 0.22` - 短波强迫效率
- `c_lw_eff = 0.15` - 长波强迫效率

#### 净云反馈

默认配置下，云的净反馈为**正反馈**：
- 变暖 → 云量减少 → 短波强迫减弱（减少冷却）
- 主导效应为正反馈，增强全球变暖

---

### 2. CO₂辐射强迫（RCMIP标准）

使用对数关系计算CO₂辐射强迫：

```python
def compute_co2_forcing(self, co2_ppm=None):
    co2_ref = 284.0  # 工业化前水平
    forcing = 5.35 * np.log(co2_ppm / co2_ref)
    return forcing
```

**CO₂倍增强迫**（284 → 568 ppm）：
- 理论值：ΔF = 5.35 × ln(2) ≈ 3.7 W/m²

#### RCMIP场景支持

内置5种RCMIP SSP场景：
| 场景 | CO₂增长率 (ppm/yr) | 峰值年份 | 描述 |
|------|---------------------|----------|------|
| SSP1-1.9 | 0.1 | 2050 | 极低排放，1.5°C目标 |
| SSP1-2.6 | 0.3 | 2070 | 低排放，2°C目标 |
| SSP2-4.5 | 0.8 | 2100 | 中等排放 |
| SSP3-7.0 | 1.5 | 2150 | 高排放 |
| SSP5-8.5 | 2.5 | 2200 | 极高排放 |

---

### 3. CO₂倍增敏感性实验

模型内置完整的敏感性实验框架：

```python
results = model.run_co2_doubling_experiment(feedback_strength=None)
```

实验包含三部分：
1. **控制实验** - 284 ppm CO₂（工业化前）
2. **无云反馈实验** - 568 ppm CO₂，云量固定
3. **有云反馈实验** - 568 ppm CO₂，启用云反馈

#### 输出指标

- **平衡气候敏感性（ECS）** - CO₂加倍导致的全球平均变暖
- **反馈参数** - λ = ΔF / ΔT
- **云反馈贡献** - 有无云反馈的变暖差值

#### 可视化功能

```python
EnergyBalanceModel1D.plot_sensitivity_experiment(results)
model.plot_cloud_feedback()
```

生成：
- 温度分布对比图
- 变暖响应纬度分布图
- 云量分布图
- 气候敏感性对比柱状图

---

## 控制方程

C ∂T/∂t = (Q₀/4) s(φ) (1 - α) + F_CO₂ + F_cloud_SW - [OLR(T) + F_cloud_LW] + ∇ · (D C cosφ ∇T)

其中：
- F_CO₂ = CO₂辐射强迫
- F_cloud_SW = 云短波辐射强迫
- F_cloud_LW = 云长波辐射强迫

## 安装依赖

```bash
pip install numpy matplotlib
```

## 使用方法

### 基本运行（含云反馈）

```python
from ebm_1d import EnergyBalanceModel1D

# 创建模型实例
model = EnergyBalanceModel1D(
    n_lat=36,      
    dt=86400.0 * 5,  # 5天时间步长（加速计算）
    n_years=80       
)

# 启用云反馈
model.enable_cloud_feedback = True

# 设置CO₂浓度（当前水平）
model.co2_ppm = 415.0
model.forcing_co2 = model.compute_co2_forcing()

# 运行模型
model.run()

# 可视化
model.plot_equilibrium()
model.plot_cloud_feedback()
model.plot_time_evolution()
```

### CO₂倍增敏感性实验

```python
# 创建模型实例
model = EnergyBalanceModel1D(n_lat=36, dt=86400.0*5, n_years=80)

# 运行敏感性实验
results = model.run_co2_doubling_experiment(
    feedback_strength=None  # None使用默认值，可自定义如-0.02
)

# 绘制实验结果
EnergyBalanceModel1D.plot_sensitivity_experiment(results)
```

### 云反馈强度调节

```python
# 弱正反馈（接近CMIP6平均）
model.c_T = -0.01  # 每变暖1K，云量减少1%

# 强正反馈（高敏感性情景）
model.c_T = -0.02

# 零反馈（中性）
model.c_T = 0.0

# 负反馈（不太可能）
model.c_T = 0.01
```

### RCMIP瞬变模拟

```python
# 沿SSP5-8.5场景模拟
model.run(co2_scenarios='ssp585')

# 沿SSP1-2.6场景模拟
model.run(co2_scenarios='ssp126')
```

---

### 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| n_lat | 纬度带数量 | 36 |
| dt | 时间步长（秒） | 86400.0 |
| n_years | 模拟年数 | 100 |
| C | 热容量 (J m⁻² K⁻¹) | 2.08e8 |
| D | 扩散系数 | 0.65 |
| Q0 | 太阳常数 (W m⁻²) | 1368.0 |
| sigma | Stefan-Boltzmann常数 | 5.67e-8 |
| A | 长波辐射截距 (W m⁻²) | 203.3 |
| B | 长波辐射斜率 (W m⁻² K⁻¹) | 2.09 |
| enable_cloud_feedback | 启用云反馈 | True |
| c_base | 基准云量 | 0.65 |
| c_T | 云量温度敏感性 (K⁻¹) | -0.015 |
| c_sw_eff | 云短波强迫效率 | 0.22 |
| c_lw_eff | 云长波强迫效率 | 0.15 |
| co2_ppm | CO₂浓度 (ppm) | 415.0 |
| co2_forcing_scale | 强迫系数 | 5.35 |

---

## 模型特点

1. **云反馈参数化** - 温度依赖的云量变化，包含短波和长波强迫
2. **RCMIP CO₂场景** - 5种SSP排放情景支持
3. **气候敏感性实验** - 内置CO₂加倍敏感性实验框架
4. **多种长波辐射模式** - Budyko-Sellers参数化或物理基模型
5. **完整诊断输出** - 能量平衡验证、反馈参数计算
6. **高级可视化** - 云反馈分析、敏感性实验对比图

---

## 文件说明

- `ebm_1d.py` - 主程序文件，包含完整模型实现
- `test_ebm.py` - 测试和验证脚本
- `README.md` - 本文档

---

## 运行示例

运行主程序（含云反馈和敏感性实验）：

```bash
python ebm_1d.py
```

运行测试脚本：

```bash
python test_ebm.py
```

---

## 预期典型结果

### 控制实验（284 ppm CO₂）
- 全球平均温度：~14-16°C
- 赤道温度：~25-28°C
- 极地温度：~-18 到 -22°C

### CO₂倍增实验（568 ppm）

| 情景 | 变暖幅度 (°C) | 说明 |
|------|---------------|------|
| 无云反馈 | ~1.8-2.2 | 仅Planck和 lapse rate反馈 |
| 有云反馈（默认） | ~2.8-3.5 | 含正云反馈，接近CMIP6平均 |
| 强云反馈 | ~4.0-5.0 | 高敏感性情景 |

### 云反馈诊断
- 云短波强迫：~-15 到 -25 W/m²（冷却）
- 云长波强迫：~+10 到 +15 W/m²（变暖）
- 净云强迫：~-5 到 -10 W/m²（净冷却）
- 反馈参数：~+0.3 到 +0.5 W/m²/K（正反馈）

---

## 参考资料

- Budyko (1969) - 能量平衡模型开创工作
- Sellers (1969) - 独立提出的EBM
- IPCC AR6 - 气候敏感性评估
- Meinshausen et al. (2020) - RCMIP情景数据库
- Zelinka et al. (2020) - CMIP6云反馈评估
