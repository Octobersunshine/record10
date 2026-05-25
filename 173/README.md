# 复合材料层板层间应力计算与渐进损伤分析

## 功能概述

本程序使用Python实现了复合材料层板在弯曲载荷下的层间应力计算和渐进损伤分析，提供三个版本：

### 版本1: 基础版本 (`composite_laminates.py`)
1. **经典层板理论 (CLT)** - 计算面内应力和层间剪应力
2. **一阶剪切变形理论 (FSDT)** - 考虑剪切变形影响的层间剪应力计算
3. **层间剥离应力** - 基于梁理论的近似计算

### 版本2: 三维精确版本 (`composite_laminates_3d.py`)
1. **三维弹性解 (Pagano方法)** - 傅里叶级数展开的精确解
2. **边界层模型** - 修正自由边界处的应力奇异性
3. **精确积分方法** - 基于平衡方程的数值积分
4. **边界效应分析** - 评估边界应力放大系数

### 版本3: 渐进损伤分析 (`composite_damage_analysis.py`) ⭐新增
1. **Hashin失效准则** - 6种失效模式的判定
2. **渐进损伤演化** - 刚度折减模型
3. **分层扩展模型** - 基于能量释放率的混合模式准则
4. **极限载荷预测** - 初始失效和极限载荷分析
5. **破坏模式识别** - 统计各失效模式发生情况

## 文件说明

| 文件 | 说明 |
|------|------|
| `composite_laminates.py` | 基础版本（CLT + FSDT）
| `composite_laminates_3d.py` | 三维精确版本（Pagano + 边界层模型）
| `composite_damage_analysis.py` | **渐进损伤分析（Hashin准则 + 刚度折减）**
| `composite_laminates.ipynb` | Jupyter Notebook交互式版本
| `README.md` | 使用说明

## 核心理论

### 经典层板理论 (CLT)

- **ABD矩阵**：计算层板的面内刚度(A)、耦合刚度(B)和弯曲刚度(D)
- **应力应变关系**：σ = Q̄(ε₀ + zκ)
- **层间剪应力**：通过面内应力的梯度积分得到

### 一阶剪切变形理论 (FSDT)

- 考虑横向剪切变形
- 引入剪切修正系数 k = 5/6
- 剪应力呈抛物线分布

### 三维弹性解 (Pagano方法)

- 基于傅里叶级数展开
- 求解三维弹性方程
- 考虑层间应力的精确分布
- 修正边界奇异性

### 边界层模型

- 指数衰减函数描述边界效应
- 边界层厚度 λ ≈ 3/h
- 应力放大系数评估

## 使用方法

### 基础版本示例

```python
from composite_laminates import CompositeLaminate, plot_stress_distribution

# 1. 定义材料属性 (T300/环氧)
ply_properties = {
    'E1': 138e9,      # 纵向弹性模量 (Pa)
    'E2': 8.96e9,     # 横向弹性模量 (Pa)
    'G12': 7.1e9,     # 面内剪切模量 (Pa)
    'G13': 7.1e9,     # 横向剪切模量 (Pa)
    'G23': 3.9e9,     # 横向剪切模量 (Pa)
    'nu12': 0.3       # 泊松比
}

# 2. 定义铺层和厚度
layup = [0, 45, -45, 90, 90, -45, 45, 0]  # 对称铺层
ply_thickness = 0.125e-3  # 单层厚度 (m)

# 3. 创建层板对象
laminate = CompositeLaminate(ply_properties, layup, ply_thickness)

# 4. 应用弯曲载荷
Mx = 100.0  # x方向弯矩 (N·m/m)
My = 50.0   # y方向弯矩 (N·m/m)
Mxy = 0.0   # 扭矩 (N·m/m)

laminate.apply_bending_load(Mx=Mx, My=My, Mxy=Mxy)

# 5. 打印结果摘要
laminate.print_summary()

# 6. 计算层间应力
z_clt, tau_xz_clt, tau_yz_clt = laminate.calculate_interlaminar_shear_stress()
z_peel, sigma_z = laminate.calculate_peeling_stress()

# 7. FSDT计算
Qx = 1000  # x方向剪力 (N/m)
Qy = 500   # y方向剪力 (N/m)
z_fsdm, tau_xz_fsdm, tau_yz_fsdm = laminate.calculate_fsdm_shear_stress(Qx, Qy)

# 8. 绘制应力分布
fig = plot_stress_distribution(laminate, title_prefix="[0/45/-45/90]s")
fig.savefig('composite_stresses.png', dpi=300)
```

### 三维精确版本示例

```python
from composite_laminates_3d import CompositeLaminate3D, plot_3d_stress_distribution, plot_comparison

# 1. 定义材料属性 (T300/环氧)
ply_properties = {
    'E1': 138e9,      # 纵向弹性模量 (Pa)
    'E2': 8.96e9,     # 横向弹性模量 (Pa)
    'E3': 8.96e9,     # 厚度方向弹性模量 (Pa)
    'G12': 7.1e9,     # 面内剪切模量 (Pa)
    'G13': 7.1e9,     # 横向剪切模量 (Pa)
    'G23': 3.9e9,     # 横向剪切模量 (Pa)
    'nu12': 0.3,       # 泊松比
    'nu13': 0.3,       # 泊松比
    'nu23': 0.3        # 泊松比
}

# 2. 定义铺层和厚度
layup = [0, 45, -45, 90, 90, -45, 45, 0]  # 对称铺层
ply_thickness = 0.125e-3  # 单层厚度 (m)
plate_width = 0.01  # 层板宽度 (m)

# 3. 创建层板对象
laminate = CompositeLaminate3D(ply_properties, layup, ply_thickness, plate_width)

# 4. 应用弯曲载荷
Mx = 100.0  # x方向弯矩 (N·m/m)
My = 50.0   # y方向弯矩 (N·m/m)
Mxy = 0.0   # 扭矩 (N·m/m)

laminate.apply_bending_load(Mx=Mx, My=My, Mxy=Mxy)

# 5. 方法1: 三维弹性解 (Pagano方法)
y_arr, z_arr, tau_xz_3d, tau_yz_3d, sigma_z_3d = laminate.calculate_3d_pagano_full()

# 6. 方法2: 精确积分方法
z_int, tau_xz_int, tau_yz_int, sigma_z_int = laminate.calculate_interlaminar_stress_integral()

# 7. 方法3: FSDT
Qx = 1000  # x方向剪力 (N/m)
Qy = 500   # y方向剪力 (N/m)
z_fsdm, tau_xz_fsdm, tau_yz_fsdm = laminate._calculate_fsdm_shear(Qx, Qy)

# 8. 绘制三维应力分布
fig1 = plot_3d_stress_distribution(laminate, title_prefix="[0/45/-45/90]s ")
fig1.savefig('composite_stresses_3d.png', dpi=300)

# 9. 绘制方法对比
fig2 = plot_comparison(laminate, title_prefix="[0/45/-45/90]s ")
fig2.savefig('composite_stresses_comparison.png', dpi=300)
```

## 主要API

### CompositeLaminate3D 类

| 方法 | 说明 |
|------|------|
| `__init__(ply_properties, layup, ply_thickness, plate_width)` | 初始化层板 |
| `apply_bending_load(Mx, My, Mxy)` | 应用弯曲载荷 |
| `calculate_3d_pagano_full(n_terms)` | 三维弹性解 (Pagano方法) |
| `calculate_interlaminar_stress_integral()` | 精确积分方法 |
| `calculate_interlaminar_shear_exact(y_points)` | 精确剪应力计算 |
| `_calculate_fsdm_shear(Qx, Qy)` | FSDT剪应力计算 |
| `get_peak_stresses(method)` | 获取峰值应力 |
| `print_summary()` | 打印计算摘要 |

## 运行程序

```bash
# 确保已安装依赖
pip install numpy scipy matplotlib

# 运行基础版本
python composite_laminates.py

# 运行三维精确版本
python composite_laminates_3d.py
```

## 输出说明

### 基础版本输出
1. 层板基本信息（铺层、厚度）
2. ABD矩阵
3. 中面应变和曲率
4. 各层的顶面和底面应力
5. 最大层间剪应力（CLT和FSDT方法）
6. 最大剥离应力
7. 应力分布图

### 三维精确版本输出
1. 层板基本信息（铺层、厚度、宽度）
2. ABD矩阵
3. 中面应变和曲率
4. 各层的顶面和底面应力
5. 三种方法的峰值应力对比
6. 边界效应分析（边界放大系数）
7. 三维应力分布图
8. 方法对比图

## 技术细节

### 边界奇异性修复

原基础版本在自由边界处存在应力奇异性问题，三维版本通过以下方法解决：

1. **Pagano方法**：使用傅里叶级数展开求解三维弹性方程
2. **边界层模型**：引入指数衰减函数描述边界效应
3. **精确积分**：基于平衡方程的数值积分
4. **应力归一化**：确保应力分布满足边界条件

### 边界层模型

边界层厚度：λ = 3/h

应力衰减：σ(y) = σ₀ · exp(-λ·|y - y_edge|)

## 渐进损伤分析使用方法

### 基本示例

```python
from composite_damage_analysis import ProgressiveDamageAnalysis, plot_damage_progression

# 1. 定义材料属性
ply_properties = {
    'E1': 138e9,      # 纵向弹性模量 (Pa)
    'E2': 8.96e9,     # 横向弹性模量 (Pa)
    'G12': 7.1e9,     # 面内剪切模量 (Pa)
    'G13': 7.1e9,     # 横向剪切模量 (Pa)
    'G23': 3.9e9,     # 横向剪切模量 (Pa)
    'nu12': 0.3       # 泊松比
}

# 2. 定义强度属性
strength_properties = {
    'Xt': 1500e6,     # 纤维拉伸强度 (Pa)
    'Xc': 1200e6,     # 纤维压缩强度 (Pa)
    'Yt': 50e6,       # 基体拉伸强度 (Pa)
    'Yc': 200e6,      # 基体压缩强度 (Pa)
    'S12': 100e6,     # 面内剪切强度 (Pa)
    'S13': 100e6,     # 横向剪切强度 (Pa)
    'S23': 60e6       # 层间剪切强度 (Pa)
}

# 3. 定义断裂韧性（用于分层分析）
fracture_properties = {
    'GIC': 200.0,     # I型断裂韧性 (J/m²)
    'GIIC': 800.0,    # II型断裂韧性 (J/m²)
    'GIIIC': 800.0    # III型断裂韧性 (J/m²)
}

# 4. 定义铺层和厚度
layup = [0, 45, -45, 90, 90, -45, 45, 0]
ply_thickness = 0.125e-3

# 5. 创建损伤分析对象
analysis = ProgressiveDamageAnalysis(
    ply_properties, strength_properties, layup, ply_thickness, fracture_properties
)

# 6. 预测极限载荷
Mx_target = 400.0
n_increments = 80

result = analysis.predict_ultimate_load(Mx_max=Mx_target, n_increments=n_increments)

# 7. 输出结果
print(f"初始失效载荷: {result['first_failure']['Mx']:.2f} N·m/m")
print(f"失效模式: {result['first_failure']['dominant_mode']}")
print(f"极限载荷: {result['ultimate_load']['Mx']:.2f} N·m/m")

# 8. 绘制损伤演化
fig = plot_damage_progression(result, title="[0/45/-45/90]s 渐进损伤分析")
fig.savefig('damage_progression.png', dpi=300)
```

### Hashin失效准则

| 失效模式 | 判定准则 |
|---------|---------|
| **纤维拉伸** | (σ₁/Xt)² + (τ₁₂² + τ₁₃²)/S₁₂² ≥ 1 (σ₁ ≥ 0) |
| **纤维压缩** | (-σ₁/Xc)² ≥ 1 (σ₁ < 0) |
| **基体拉伸** | ((σ₂+σ₃)/Yt)² + (τ₁₂² + τ₂₃²)/S₁₂² ≥ 1 (σ₂+σ₃ ≥ 0) |
| **基体压缩** | 混合模式准则 (σ₂+σ₃ < 0) |
| **分层拉伸** | (σ₃/Yt)² + (τ₁₃² + τ₂₃²)/S₁₃² ≥ 1 (σ₃ ≥ 0) |
| **分层压缩** | (τ₁₃² + τ₂₃²)/S₁₃² ≥ 1 (σ₃ < 0) |

### 损伤演化模型

- **纤维损伤 (d_f)**: 控制E1和ν12的折减
- **基体损伤 (d_m)**: 控制E2和ν12的折减
- **分层损伤 (d_s)**: 控制G12的折减
- **损伤增长**: 与失效指标成比例增长

### 主要API

#### HashinFailureCriteria 类

| 方法 | 说明 |
|------|------|
| `__init__(strength_properties)` | 初始化强度参数 |
| `fiber_tension(sigma1, tau12, tau13)` | 纤维拉伸失效指标 |
| `fiber_compression(sigma1)` | 纤维压缩失效指标 |
| `matrix_tension(sigma2, sigma3, tau12, tau23)` | 基体拉伸失效指标 |
| `matrix_compression(sigma2, sigma3, tau12, tau23)` | 基体压缩失效指标 |
| `delamination_tension(sigma3, tau13, tau23)` | 分层拉伸失效指标 |
| `delamination_compression(sigma3, tau13, tau23)` | 分层压缩失效指标 |
| `check_failure(sigma)` | 综合失效检查 |

#### DelaminationModel 类

| 方法 | 说明 |
|------|------|
| `__init__(fracture_properties)` | 初始化断裂韧性 |
| `calculate_energy_release_rate(stresses)` | 计算能量释放率 |
| `check_delamination(stresses)` | 检查分层扩展 |

#### ProgressiveDamageAnalysis 类

| 方法 | 说明 |
|------|------|
| `__init__(ply_properties, strength_properties, layup, ply_thickness, ...)` | 初始化分析 |
| `apply_load_incremental(Mx_target, My_target, n_increments)` | 增量加载分析 |
| `predict_ultimate_load(Mx_max, My_max, n_increments)` | 预测极限载荷 |
| `_analyze_damage_progression()` | 分析损伤演化过程 |

## 运行程序

```bash
# 确保已安装依赖
pip install numpy scipy matplotlib

# 运行基础版本
python composite_laminates.py

# 运行三维精确版本
python composite_laminates_3d.py

# 运行渐进损伤分析
python composite_damage_analysis.py
```

## 输出说明

### 渐进损伤分析输出
1. 材料强度属性
2. 初始失效载荷和失效模式
3. 极限载荷预测
4. 损伤演化过程摘要
5. 各层最终损伤状态
6. 损伤演化图 (`damage_progression.png`)
7. 失效模式对比图 (`failure_modes_comparison.png`)

## 注意事项

1. 三维版本需要安装 scipy 库用于数值积分
2. Pagano方法的精度取决于傅里叶级数项数
3. 边界层模型参数可根据实际材料调整
4. 材料属性需根据实际复合材料进行调整
5. 铺层角度单位为度
6. 损伤演化速度参数可根据实验数据校准
7. Hashin准则适用于连续纤维增强复合材料
