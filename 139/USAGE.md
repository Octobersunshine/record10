# 蛋白质折叠粗粒化模拟 (Go模型 + REMD)

## 概述

本项目实现了蛋白质折叠的多尺度模拟框架：
- **Go模型**: 基于天然态接触的粗粒化力场
- **AMBER混合力场**: 结合Go模型和全原子键/角/二面角势能
- **REMD**: 副本交换分子动力学，加速能量势垒跨越
- **路径分析**: 折叠路径、接触形成顺序、状态聚类

## 目录结构

```
.
├── go_model.py          # 核心Go模型模拟代码
├── amber_forcefield.py  # AMBER力场和Go-AMBER混合模型
├── remd.py              # 副本交换MD和折叠路径分析
├── units.py             # 单位转换和摩擦系数校准
├── main.py              # 主程序入口和示例
├── requirements.txt     # 依赖包列表
└── USAGE.md             # 使用说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

或者单独安装：
```bash
pip install numpy scipy matplotlib tqdm
```

## 使用方法

### 1. 单轨迹模拟

运行单个折叠轨迹，观察Q值变化和最终结构：

```bash
python main.py --mode single
```

输出：
- `go_model_results.png`: 包含Q值时间序列、Q值分布、最终结构图像

### 2. MFPT计算

计算不同温度下的折叠时间分布：

```bash
python main.py --mode mfpt
```

输出：
- `mfpt_results.png`: 包含折叠时间直方图和多条折叠轨迹
- 控制台输出各温度下的MFPT值及统计误差

### 3. 自定义接触图

使用用户定义的天然态接触图进行模拟：

```bash
python main.py --mode custom
```

## 核心类和函数

### GoModel 类

主要参数：
- `num_beads`: 珠子数量
- `native_contacts`: 天然态接触对 (N, 2) 数组
- `native_distances`: 对应接触对的天然距离
- `temperature`: 温度 (约化单位)
- `gamma`: 摩擦系数
- `dt`: 时间步长
- `epsilon`: 能量强度

主要方法：
- `langevin_step()`: 执行一步Langevin动力学
- `compute_native_contact_fraction()`: 计算天然接触分数Q
- `simulate(n_steps)`: 运行模拟并返回轨迹

### 工具函数

- `generate_native_structure(num_beads, type)`: 生成理想结构(helix/hairpin/linear)
- `generate_contact_map(positions, cutoff, min_sep)`: 从结构生成接触图
- `calculate_mfpt(num_trajs, sim, threshold)`: 计算MFPT
- `plot_results(...)`: 绘制结果图像

## Go模型原理

### 能量函数

Go模型只考虑天然态接触的相互作用：

1. **键势能**: 相邻珠子的简谐势
   ```
   V_bond = (1/2) * k_bond * (r - r0)^2
   ```

2. **天然接触势能**: Lennard-Jones势
   ```
   V_native = 4 * ε * [(σ/r)^12 - (σ/r)^6]
   ```
   其中σ = r_native / 2^(1/6)

3. **非天然排斥**: 仅排斥项，防止原子重叠
   ```
   V_repulsive = 4 * ε * (σ/r)^12,  r < σ*2^(1/6)
   ```

### Langevin动力学

运动方程：
```
m dv/dt = F - γ v + ξ(t)
```
其中ξ(t)是高斯白噪声，满足涨落耗散定理。

## 折叠判断标准

- **Q值**: 形成的天然接触数 / 总天然接触数
- 折叠阈值: Q >= 0.8 (可配置)
- 接触形成条件: r < 1.2 * r_native

## 摩擦系数校准 (重要!)

### 问题背景

**原问题**: 之前的摩擦系数设置过高 (γ=0.5)，导致扩散太慢，折叠时间被高估。

**解决方案**: 使用Stokes-Einstein关系计算物理上合理的摩擦系数，并通过实验折叠速率进一步校正。

### Stokes-Einstein关系

摩擦系数由溶剂粘度和粒子半径决定：

```
γ_SI = 6 π η R
```

其中:
- η = 0.001 Pa·s (水在25°C的粘度)
- R = 蛋白质流体力学半径

### 单位转换系统

约化单位与真实单位的对应关系：

| 物理量 | 约化单位 | 真实单位 | 转换因子 |
|--------|----------|----------|----------|
| 质量 | m = 1 | 110 Da | ~1.83e-25 kg |
| 长度 | σ = 1 | 3.8 Å | 3.8e-10 m |
| 能量 | ε = 1 | 1 kcal/mol | ~6.94e-21 J |
| 时间 | τ = 1 | ~3.8 ps | ~3.8e-12 s |

### 摩擦系数模型

1. **Stokes-Einstein模型** (推荐): 根据蛋白质大小物理计算
   ```python
   gamma = get_optimal_gamma(num_beads, model='stokes_einstein')
   ```

2. **经验模型**: 基于文献的经验公式
   ```python
   gamma = get_optimal_gamma(num_beads, model='empirical')
   ```

3. **文献值**: 常用的标准值 (γ≈0.15)
   ```python
   gamma = get_optimal_gamma(num_beads, model='literature')
   ```

### 实验速率校准

使用已知的实验折叠速率校正模拟结果：

```python
from units import FoldingRateCalibrator

calibrator = FoldingRateCalibrator(num_beads=56)
results = calibrator.get_calibrated_gamma(
    simulated_mfpt=50000,  # 模拟得到的MFPT (步数)
    dt=0.005,              # 时间步长
    target_kf=1.5e4        # 实验折叠速率 (s^-1)
)

corrected_gamma = results['corrected_gamma']
```

### 典型摩擦系数值

| 蛋白质大小 | Stokes-Einstein γ | 旧值 (高估) | 相对扩散 |
|------------|-------------------|-------------|----------|
| 20残基 | ~0.08 | 0.5 | 6.25x |
| 30残基 | ~0.10 | 0.5 | 5.0x |
| 50残基 | ~0.12 | 0.5 | 4.2x |
| 100残基 | ~0.15 | 0.5 | 3.3x |

**注意**: 摩擦系数减小会使扩散加快，折叠时间缩短，更接近真实实验值。

## 参数调优建议

### 模拟稳定性
- 时间步长 dt = 0.001 ~ 0.01
- 摩擦系数 gamma = 0.05 ~ 0.3 (修正后)

### 折叠温度范围
- 低于折叠温度 Tf: 快速折叠
- 接近 Tf: 双稳态，MFPT最长
- 高于 Tf: 无法折叠

### MFPT统计精度
- 轨迹数: >= 20
- 误差估计: std / sqrt(N)

## 自定义扩展

### 1. 使用PDB结构

```python
from go_model import GoModel, generate_contact_map

# 从PDB读取Cα坐标 (需自行实现PDB解析)
native_positions = read_pdb_ca('protein.pdb')

# 生成接触图
contacts, distances = generate_contact_map(
    native_positions, 
    cutoff=7.0,  # 7埃
    min_sequence_separation=4
)

# 创建模拟
sim = GoModel(
    num_beads=len(native_positions),
    native_contacts=contacts,
    native_distances=distances,
    temperature=0.9
)
```

### 2. 温度扫描

```python
temperatures = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
mfpts = []

for T in temperatures:
    sim = GoModel(..., temperature=T)
    mfpt, err, _, _ = calculate_mfpt(30, sim, ...)
    mfpts.append((T, mfpt, err))
```

## 注意事项

1. **计算资源**: MFPT计算需要大量轨迹，建议使用多核并行
2. **单位**: 所有物理量使用约化单位
3. **收敛性**: MFPT计算需确保轨迹充分采样
4. **内存**: 长轨迹保存需注意内存使用

## 参考文献

1. Go, N. (1983). Theoretical studies of protein folding.
2. Clementi, C., Nymeyer, H., & Onuchic, J. N. (2000). 
   Topological and energetic factors: what determines the structural details 
   of the transition state ensemble and "en-route" intermediates for protein folding?
