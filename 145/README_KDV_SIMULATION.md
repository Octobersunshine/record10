# KdV方程海洋内波模拟器

基于Korteweg-de Vries (KdV)方程的海洋内波数值模拟工具，使用伪谱方法求解。

## 项目概述

KdV方程是描述弱非线性、弱色散波动的经典模型，广泛应用于海洋内波、浅水波、等离子体物理等领域。

### KdV方程形式

标准KdV方程：
```
∂u/∂t + 6u∂u/∂x + ∂³u/∂x³ = 0
```

考虑地形效应的变系数KdV方程：
```
∂u/∂t + (6/h)u∂u/∂x + (h_x/h)u + ∂³u/∂x³ = 0
```
其中h(x)为水深。

## 主要功能

1. **数值求解**：使用傅里叶伪谱方法 + 四阶龙格-库塔(RK4)时间积分
2. **初始波形**：
   - 解析孤立波解 (Soliton)
   - 高斯波包
3. **地形模型**：
   - 平坦地形
   - 大陆架 (Continental Shelf)
   - 海岭/海脊 (Submarine Ridge)
   - 海沟 (Trench)
   - 自定义复杂地形
4. **可视化**：
   - 时间演化曲线
   - 时空热力图
   - GIF动画

## 文件结构

```
.
├── kdv_internal_wave.py      # 主求解器类
├── example_custom_simulation.py  # 自定义示例脚本
├── README_KDV_SIMULATION.md  # 本文档
└── output/                   # 输出图像和动画
```

## 快速开始

### 环境要求

- Python 3.7+
- NumPy
- Matplotlib
- Pillow (用于GIF动画)

### 安装依赖

```bash
pip install numpy matplotlib pillow
```

### 运行主程序

```bash
python kdv_internal_wave.py
```

### 运行自定义示例

```bash
python example_custom_simulation.py
```

## 类和方法说明

### KdVSolver 类

#### 初始化参数

```python
solver = KdVSolver(
    L=50,        # 计算域长度
    N=1024,      # 空间网格点数
    dt=0.001,    # 时间步长
    T_max=10     # 总模拟时间
)
```

#### 初始波形生成

**孤立波解**：
```python
u0 = solver.soliton_solution(
    x,           # 空间坐标
    x0=0,        # 初始位置
    c=1,         # 波速
    A=1          # 振幅系数
)
```

**高斯波包**：
```python
u0 = solver.gaussian_wave(
    x,           # 空间坐标
    x0=0,        # 中心位置
    sigma=2,     # 宽度
    amp=1        # 振幅
)
```

#### 地形类型

| 地形类型 | 参数 | 说明 |
|---------|------|------|
| `flat` | - | 平坦地形 |
| `shelf` | h1, h2, x_trans, width | 大陆架 |
| `ridge` | h0, height, x0, width | 海岭 |
| `trench` | h0, depth, x0, width | 海沟 |

#### 求解与可视化

```python
# 求解方程
u_history, h_x = solver.solve(
    u0,                     # 初始条件
    terrain_type='shelf',   # 地形类型
    **terrain_kwargs        # 地形参数
)

# 绘制时间演化
solver.plot_evolution(u_history, h_x, 'output.png')

# 绘制时空热力图
solver.plot_spacetime(u_history, 'spacetime.png')

# 创建动画
solver.create_animation(u_history, h_x, 'animation.gif', fps=30)
```

## 自定义模拟示例

### 示例1: 单个孤立波传播

```python
from kdv_internal_wave import KdVSolver

solver = KdVSolver(L=80, N=512, dt=0.001, T_max=3)
u0 = solver.soliton_solution(solver.x, x0=-25, c=1.5, A=1)
u_history, h = solver.solve(u0, terrain_type='flat')
solver.plot_evolution(u_history, h, 'soliton_flat.png')
```

### 示例2: 双孤立波相互作用

```python
solver = KdVSolver(L=100, N=512, dt=0.0005, T_max=6)
u1 = solver.soliton_solution(solver.x, x0=-30, c=3, A=1)
u2 = solver.soliton_solution(solver.x, x0=-10, c=1, A=0.5)
u0 = u1 + u2
u_history, h = solver.solve(u0, terrain_type='flat')
```

### 示例3: 大陆架地形

```python
solver = KdVSolver(L=120, N=1024, dt=0.001, T_max=8)
u0 = solver.soliton_solution(solver.x, x0=-40, c=2, A=1)
u_history, h = solver.solve(
    u0,
    terrain_type='shelf',
    h1=1.0,      # 左侧水深
    h2=0.3,      # 右侧水深
    x_trans=10,  # 过渡中心位置
    width=4      # 过渡宽度
)
```

### 示例4: 自定义复杂地形

```python
def custom_terrain(x):
    h = np.ones_like(x)
    h -= 0.3 * np.exp(-((x + 10) ** 2) / 32)
    h += 0.2 * np.exp(-((x - 20) ** 2) / 18)
    return h

h_x = custom_terrain(solver.x)
solver.terrain_function = lambda *args, **kwargs: h_x
u_history, _ = solver.solve(u0, terrain_type='flat')
```

## 数值方法说明

### 伪谱方法 (Pseudo-Spectral Method)

1. **空间导数**：在傅里叶空间计算，利用FFT实现谱精度
   - ∂u/∂x ↔ ikû
   - ∂³u/∂x³ ↔ -ik³û

2. **时间积分**：四阶龙格-库塔(RK4)方法
   - 稳定性好
   - 精度高

### 算法流程

```
初始化:
    x = 空间网格
    k = 波数网格
    u0 = 初始条件

时间迭代:
    对每个时间步:
        1. 计算非线性项 (物理空间)
        2. FFT到谱空间
        3. 计算线性项 (谱空间)
        4. RK4时间推进
        5. IFFT回到物理空间
        6. 保存结果
```

## 海洋内波物理背景

### 密度层结流体

海洋中由于温度、盐度分布不均匀形成密度层结。在两层流体模型中：

```
ρ₁ (上层)
----- 温跃层/密度跃层
ρ₂ (下层)
```

内波在密度界面传播，满足的KdV方程系数与上下层密度差有关。

### 地形效应

1. **浅化效应**：水深减小时，波速减小，振幅增大
2. **波破碎**：非线性效应增强，可能导致波破碎
3. **反射与透射**：地形突变处产生反射波和透射波

## 输出文件说明

运行程序后将生成以下类型的文件：

- `*_evolution.png`：不同时刻的波形演化图
- `*_spacetime.png`：时空演化热力图
- `*_animation.gif`：波传播动画

## 参数调整建议

| 参数 | 作用 | 建议范围 |
|------|------|----------|
| L | 计算域大小 | 50-200 |
| N | 空间分辨率 | 256-2048 (2的幂次) |
| dt | 时间步长 | 0.0001-0.005 |
| T_max | 总时间 | 3-20 |

**注意**：为保证FFT效率，N建议取2的幂次。

## 常见问题

### Q: 模拟出现数值振荡？
A: 减小时间步长dt，或增加空间分辨率N。

### Q: 波形不按预期传播？
A: 检查初始条件参数，确保孤立波解的波速与振幅匹配。

### Q: 动画生成失败？
A: 确保安装了Pillow库：`pip install pillow`

## 扩展功能建议

1. 添加更多初始条件（正弦波、随机扰动等）
2. 实现三维或二维KdV方程（如KP方程）
3. 添加更多物理效应（耗散、外力驱动）
4. 实现并行计算以加速大尺度模拟
5. 添加数据保存与加载功能（HDF5、NetCDF）

## 参考文献

1. Korteweg, D. J., & de Vries, G. (1895). On the change of form of long waves advancing in a rectangular canal, and on a new type of long stationary waves.
2. Osborne, A. R., & Burch, T. L. (1980). Internal solitons in the Andaman Sea.
3. Helfrich, K. R., & Melville, W. K. (2006). Long nonlinear internal waves.
