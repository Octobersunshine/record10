# 各向异性介质射线追踪 - VTI裂缝介质

## 概述

本项目实现了**垂直横向各向同性(VTI)介质**的射线追踪，使用**Christoffel方程**计算相速度和群速度。VTI模型常用于描述具有垂直接触的裂缝介质，是地震勘探中最常用的各向异性模型。

## 理论基础

### Christoffel方程

在各向异性介质中，平面波的传播由Christoffel方程描述：

```
|c_ijkl n_j n_l - ρ v² δ_ik| = 0
```

其中:
- `c_ijkl` 是弹性刚度张量 (4阶)
- `n_j` 是波法向单位矢量 (相方向)
- `ρ` 是密度
- `v` 是相速度

对于VTI介质，刚度张量简化为5个独立参数。

### Thomsen参数

Thomsen参数是描述VTI各向异性的标准参数：

| 参数 | 物理意义 | 典型值范围 |
|------|----------|------------|
| vp0 | 垂直方向P波速度 | 2000-5000 m/s |
| vs0 | 垂直方向S波速度 | 1000-3000 m/s |
| ε | P波各向异性强度 | 0-0.5 |
| δ | 近轴P波各向异性 | -0.2-0.3 |
| γ | S波各向异性强度 | 0-0.3 |

### 波的类型

VTI介质中存在三种体波：

1. **qP波** (准纵波): 偏振方向近似于波传播方向
2. **qSV波** (准横波SV): 偏振方向在对称轴平面内
3. **qSH波** (准横波SH): 偏振方向垂直于对称轴平面

### 相速度与群速度

- **相速度 (Phase velocity)**: 波前法向的传播速度
- **群速度 (Group velocity)**: 能量传播的实际速度和方向

在各向异性介质中，相速度方向与群速度方向通常不同！

## 文件说明

| 文件 | 说明 |
|------|------|
| `anisotropic_shooting.py` | 各向异性射线追踪主程序 |
| `shooting_method.py` | 各向同性射线追踪 (含层状介质折射盲区修复) |
| `ANISOTROPIC_README.md` | 本文档 |
| `README.md` | 各向同性版本说明 |

## 核心类说明

### VTIModel类

VTI介质模型类，负责计算刚度张量、相速度、群速度等。

```python
from anisotropic_shooting import VTIModel

# 创建VTI介质模型
vti = VTIModel(
    vp0=3000,      # 垂直P波速度 (m/s)
    vs0=1500,      # 垂直S波速度 (m/s)
    epsilon=0.25,  # Thomsen ε参数
    delta=0.15,    # Thomsen δ参数
    gamma=0.2,     # Thomsen γ参数
    rho=2500       # 密度 (kg/m³)
)
```

主要方法：
- `christoffel_phase_velocity(theta, wave_type)`: 计算指定相角的相速度
- `group_velocity(theta, wave_type)`: 计算群速度大小和群角
- `phase_angle_from_group_angle(phi, wave_type)`: 从群角反算相角
- `get_slowness_vector(phi, wave_type)`: 获取慢度矢量

### AnisotropicShooting类

各向异性打靶法射线追踪类。

```python
from anisotropic_shooting import AnisotropicShooting

# 创建射线追踪器
shooter = AnisotropicShooting(
    vti,                 # VTI介质模型
    wave_type='qP',      # 波型: 'qP', 'qSV', 'qSH'
    debug=True           # 是否输出调试信息
)

# 发射一条射线
x, z, travel_time = shooter.shoot_ray(
    x0=0, z0=0,          # 震源位置
    takeoff_angle=45,    # 出射群角 (度)
    max_s=20000          # 最大路径长度
)

# 自动寻找到接收点的射线
rays = shooter.compute_rays(
    source=(0, 0),
    receivers=[(2000, 1000), (4000, 2000)]
)
```

## 使用示例

### 1. 基本速度计算

```python
import numpy as np
from anisotropic_shooting import VTIModel

vti = VTIModel(vp0=3000, vs0=1500, epsilon=0.2, delta=0.1, gamma=0.15)

# 计算不同角度的qP波速度
for angle in [0, 30, 45, 60, 90]:
    theta = np.radians(angle)
    vp = vti.christoffel_phase_velocity(theta, 'qP')
    vg, phi = vti.group_velocity(theta, 'qP')
    print(f"相角={angle}°, 相速度={vp:.1f}m/s, 群速度={vg:.1f}m/s, 群角={np.degrees(phi):.2f}°")
```

### 2. 完整的射线追踪

```python
from anisotropic_shooting import VTIModel, AnisotropicShooting, plot_anisotropic_rays

# 创建VTI模型
vti = VTIModel(
    vp0=3500,
    vs0=1800,
    epsilon=0.2,
    delta=0.1,
    gamma=0.12
)

# 追踪qP波
shooter_qP = AnisotropicShooting(vti, wave_type='qP', debug=True)

source = (0, 0)
receivers = [(3000, 1500), (5000, 2500)]

rays = shooter_qP.compute_rays(source, receivers)

# 绘制结果
plot_anisotropic_rays(rays, source, vti, 'qP', title='Example')
```

### 3. 各向同性 vs 各向异性对比

```python
vti = VTIModel(vp0=3000, epsilon=0.0, delta=0.0, gamma=0.0)  # 各向同性
vti_aniso = VTIModel(vp0=3000, epsilon=0.25, delta=0.15, gamma=0.2)  # 各向异性

# 对比45度方向的速度
theta = np.radians(45)
vp_iso = vti.christoffel_phase_velocity(theta, 'qP')
vp_aniso = vti_aniso.christoffel_phase_velocity(theta, 'qP')

print(f"各向同性相速度: {vp_iso:.1f} m/s")
print(f"各向异性相速度: {vp_aniso:.1f} m/s")
```

## 运行演示

直接运行主程序：

```bash
python anisotropic_shooting.py
```

程序将自动执行：

1. 创建VTI介质模型并显示参数
2. 绘制速度随角度变化图 (`anisotropic_velocity_comparison.png`)
3. 绘制群速度面图 (`group_velocity_surface.png`)
4. 演示不同角度的相速度与群速度对比
5. 对qP波和qSV波进行射线追踪
6. 绘制射线路径图 (`anisotropic_rays_qP.png`, `anisotropic_rays_qSV.png`)
7. 进行各向同性与各向异性走时对比

## 输出图形说明

### 1. 速度对比图 (`anisotropic_velocity_comparison.png`)

包含三个子图，分别显示qP、qSV、qSH三种波的：
- 蓝色实线: 相速度 vs 相角
- 红色虚线: 群速度 vs 群角

### 2. 群速度面图 (`group_velocity_surface.png`)

极坐标下的群速度分布，即波前形状。各向异性越强，形状偏离圆越明显。

### 3. 射线路径图 (`anisotropic_rays_*.png`)

显示从震源到各接收点的射线路径，标注走时和出射角。

## 数值方法

### Christoffel方程求解

对于VTI介质，Christoffel方程可以解析求解：

**qP波和qSV波:**
```
A = (c11 sin²θ + c33 cos²θ) / ρ
B = (c13 + c44)² sin²θ cos²θ / ρ²
C = (c11 - c44)(c33 - c44) sin²θ cos²θ / ρ²
v² = 0.5 [A ± sqrt(A² - 4(C - B))]
```
(+号对应qP波，-号对应qSV波)

**qSH波:**
```
v² = (c66 sin²θ + c44 cos²θ) / ρ
```

### 群速度计算

使用数值微分计算相速度对角度的导数，然后通过以下关系得到群速度：

```
v_g = 1/v * sqrt[(v dP/dθ)² + (d(vQ)/dθ)²]
```
其中 P = sinθ, Q = cosθ。

### 射线追踪

使用常微分方程描述射线传播：
```
dx/ds = v_g sin(φ)
dz/ds = v_g cos(φ)
dpx/ds = 0
dpz/ds = 0
```
使用RK45方法数值求解。

## 常见问题

### Q1: qSV波为什么会有速度反向？

A: 对于某些各向异性参数，qSV波的群速度会出现"三瓣"形状，群角与相角关系非单调，这是正常的物理现象。

### Q2: 为什么群角和相角不同？

A: 这是各向异性介质的基本特征，能量传播方向（群方向）与波前法向（相方向）不同，差异大小取决于各向异性强度。

### Q3: 如何选择Thomsen参数值？

A: 典型地层的各向异性参数：
- 页岩: ε≈0.1-0.3, δ≈0.05-0.2, γ≈0.1-0.25
- 砂岩: ε≈0.05-0.15, δ≈0-0.1, γ≈0.05-0.15
- 裂缝介质: ε可达0.4以上

### Q4: 射线追踪收敛慢怎么办？

A: 可以尝试：
- 增加初始角度采样点数 (默认60个)
- 增大局部优化搜索范围 (默认8°)
- 对于强各向异性介质，缩小角度搜索范围

## 参考资料

1. Thomsen, L. (1986). Weak elastic anisotropy. Geophysics, 51(10), 1954-1966.
2. Tsvankin, I. (2001). Seismic signatures and analysis of reflection data in anisotropic media.
3. Červený, V. (2001). Seismic ray theory. Cambridge university press.

## 更新历史

- v1.0 (2024): 初始版本，实现VTI介质Christoffel方程求解和打靶法射线追踪
