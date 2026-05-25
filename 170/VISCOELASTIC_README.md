# AFM 粘弹性（蠕变）分析指南

## 概述

本工具包专为**生物材料（细胞、水凝胶等）**的AFM粘弹性分析设计，支持：

- **标准线性固体模型（SLS）**：简单的三参数粘弹性模型
- **广义Maxwell模型**：多松弛时间谱提取
- **蠕变柔量分析**：从压痕-时间曲线提取材料特性
- **松弛时间谱**：揭示材料内部不同尺度的动力学过程

## 粘弹性理论基础

### 标准线性固体（SLS）模型

SLS模型由一个弹簧和一个Maxwell单元并联组成：

```
      _______
     |       |
  ---|  E∞   |---
     |_______|
         |
     ____|____
    |         |
    |---E₀----|
    |         |
    |---η-----|
    |_________|
```

**蠕变响应（恒定力下）：**
```
δ(t) = δ₀ * [1 + (E₀/E∞ - 1) * (1 - exp(-t/τ))]
```

其中：
- **E₀**：瞬时模量（短时间响应）
- **E∞**：平衡模量（长时间响应）
- **τ**：特征松弛时间 = η/E₀
- **E∞/E₀**：弹性比，衡量材料的"弹性程度"

### 松弛时间谱

对于复杂生物材料，单一松弛时间不够，需要广义Maxwell模型：

```
G(t) = G_e + Σ G_i * exp(-t/τ_i)
```

其中：
- **G_e**：平衡模量
- **G_i**：第i个松弛模式的模量
- **τ_i**：第i个松弛模式的特征时间

## 典型生物材料参数范围

| 材料类型 | 瞬时模量 E₀ | 平衡模量 E∞ | 松弛时间 τ | 弹性比 E∞/E₀ |
|---------|------------|------------|-----------|-------------|
| 软细胞（神经元） | 100-1000 Pa | 50-500 Pa | 1-10 s | 30-60% |
| 成纤维细胞 | 500-5000 Pa | 200-2000 Pa | 0.5-3 s | 40-70% |
| 硬细胞（骨细胞） | 1000-10000 Pa | 500-5000 Pa | 0.1-1 s | 50-80% |
| 软水凝胶 | 100-5000 Pa | 50-2000 Pa | 0.1-100 s | 20-70% |
| 硬水凝胶 | 1-100 kPa | 0.5-50 kPa | 0.01-10 s | 40-90% |

## 使用指南

### 1. 基础SLS拟合

```python
from afm_viscoelastic import ViscoelasticAFM

# 创建分析器（细胞常用半径20-50 nm）
ve_afm = ViscoelasticAFM(radius=30e-9, nu=0.5)

# 生成合成测试数据（或加载实验数据）
t, delta_noisy, delta_true = ve_afm.generate_creep_data(
    t_total=30.0,      # 总时间 30秒
    n_points=500,      # 数据点数
    E0=1500,           # 瞬时模量 1500 Pa
    Einf=600,          # 平衡模量 600 Pa
    tau=1.5,           # 松弛时间 1.5 s
    delta0=80e-9,      # 初始压痕 80 nm
    noise=0.02         # 2% 噪声
)

# 拟合SLS模型
ve_afm.fit_sls_creep(t, delta_noisy, p0=[1000, 400, 1.0, 60e-9])

# 打印结果
ve_afm.print_viscoelastic_results()

# 绘制蠕变曲线
ve_afm.plot_creep_curve(t, delta_noisy, delta_true, save_path='creep_curve.png')
```

### 2. 提取松弛时间谱

```python
# 从应力松弛数据提取多模式松弛谱
tau_fit, Gi_fit, Ge = ve_afm.calculate_stress_relaxation_spectrum(
    t, G_t,           # 时间和模量时间序列
    n_modes=3,        # 松弛模式数量
    tau_min=1e-3,     # 最小松弛时间
    tau_max=1e2       # 最大松弛时间
)

# 绘制松弛谱
ve_afm.plot_relaxation_spectrum(save_path='spectrum.png')
```

### 3. 加载实验数据

CSV数据格式（两列：时间, 压痕）：
```csv
time_s,indentation_m
0.0,5.0e-08
0.1,5.23e-08
0.5,5.87e-08
1.0,6.32e-08
...
```

加载并拟合：
```python
from afm_viscoelastic import load_creep_data

t, delta = load_creep_data('experiment_data.csv')

ve_afm = ViscoelasticAFM(radius=20e-9, nu=0.5)
ve_afm.fit_sls_creep(t, delta, p0=[1000, 500, 1.0, 40e-9])
ve_afm.print_viscoelastic_results()
```

### 4. 对比不同细胞类型

```python
cell_types = {
    'Neuron': {'E0': 500, 'Einf': 200, 'tau': 3.0},
    'Fibroblast': {'E0': 2000, 'Einf': 800, 'tau': 1.0},
    'Bone cell': {'E0': 8000, 'Einf': 4000, 'tau': 0.3}
}

for cell_type, params in cell_types.items():
    t, delta, _ = ve_afm.generate_creep_data(
        E0=params['E0'], Einf=params['Einf'], tau=params['tau']
    )
    ve_afm.fit_sls_creep(t, delta)
    print(f"\n{cell_type}:")
    ve_afm.print_viscoelastic_results()
```

## 实验设计建议

### AFM蠕变实验步骤

1. **定位细胞**：在光学显微镜下找到目标细胞
2. **接近表面**：以低速接近细胞表面
3. **施加恒定力**：快速达到目标力并保持
4. **记录蠕变**：记录压痕随时间的变化（建议10-60秒）
5. **撤力**：缓慢撤力，记录恢复曲线（可选）

### 关键参数选择

| 参数 | 推荐范围 | 说明 |
|-----|---------|------|
| 针尖半径 | 10-50 nm | 细胞常用20-30 nm |
| 接触力 | 0.5-5 nN | 根据细胞硬度调整 |
| 采样频率 | 10-100 Hz | 捕捉快速和慢速过程 |
| 蠕变时间 | 10-60 s | 至少3倍最长松弛时间 |
| 泊松比 | 0.45-0.5 | 细胞通常取0.5 |

### 数据预处理建议

1. **基线校正**：扣除接触前的基线漂移
2. **接触点确定**：准确确定零压痕位置
3. **平滑处理**：使用滑动平均或Savitzky-Golay滤波
4. **异常值去除**：移除明显的噪声尖峰

## 参数解释

### SLS模型参数

| 参数 | 物理意义 | 生物意义 |
|-----|---------|---------|
| **E₀** | 瞬时弹性模量 | 细胞骨架的即时刚度 |
| **E∞** | 平衡弹性模量 | 细胞的长期刚度 |
| **τ** | 特征松弛时间 | 细胞内部重排的时间尺度 |
| **E∞/E₀** | 弹性比 | 细胞的"弹性/粘性"比例 |

### 松弛时间谱的生物学解释

| 松弛时间范围 | 可能的生物学过程 |
|------------|----------------|
| **< 0.1 s** | 细胞膜形变、水分子流动 |
| **0.1-1 s** | 皮质肌动蛋白重排 |
| **1-10 s** | 应力纤维重组、细胞骨架整体形变 |
| **> 10 s** | 细胞核形变、长期细胞响应 |

## 常见问题

### Q1: 拟合不收敛怎么办？

**解决方案：**
1. 调整初始猜测值 `p0`
2. 增加 `maxfev` 参数
3. 检查数据质量（是否有异常值）
4. 尝试缩小参数边界

### Q2: 如何选择松弛模式数量？

**指导原则：**
- 简单材料（均质水凝胶）：2-3个模式
- 复杂材料（细胞）：3-5个模式
- 使用贝叶斯信息准则（BIC）选择最优数量
- 避免过拟合（模式数量不宜过多）

### Q3: 不同针尖半径结果如何比较？

**建议：**
1. 使用相同半径的针尖进行对比实验
2. 使用接触力学模型转换到固有材料属性
3. 报告具体的实验条件（半径、力等）

### Q4: 数据噪声大怎么办？

**处理方法：**
1. 增加平均次数
2. 延长采样时间
3. 使用更软的悬臂梁（提高力分辨率）
4. 应用适当的滤波算法

## 文件说明

| 文件 | 功能 |
|-----|------|
| `afm_viscoelastic.py` | 粘弹性分析核心模块 |
| `example_viscoelastic.py` | 完整使用示例 |
| `sample_creep_data.csv` | 样例实验数据 |

## 参考文献

1. **Radmacher, M. et al.** (1996). Measuring the viscoelastic properties of human platelets with the atomic force microscope. Biophysical Journal.

2. **Dimitriadis, E. K. et al.** (2002). Determination of elastic moduli of thin layers of soft material using the atomic force microscope. Biophysical Journal.

3. **Mahaffy, R. E. et al.** (2004). Viscoelasticity of human neuroblastoma cells observed with atomic force microscope. Physical Review E.

4. **Guo, M. et al.** (2014). Probing the mechanical properties of cells with atomic force microscopy. Methods in Molecular Biology.
