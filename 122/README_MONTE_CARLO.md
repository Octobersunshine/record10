# 蒙特卡洛声子BTE求解器 - 直接模拟

本模块实现了基于**直接模拟蒙特卡洛（DSMC）**方法的声子玻尔兹曼输运方程求解器，专门用于处理复杂纳米结构（如超晶格、纳米薄膜）的热输运问题。

## 蒙特卡洛方法原理

### 1. 直接模拟蒙特卡洛（DSMC）

与解析BTE不同，蒙特卡洛方法通过**追踪大量声子的运动**来求解热输运：

1. **初始化**：根据玻色-爱因斯坦分布生成N个声子
2. **自由飞行**：声子以群速度直线运动
3. **散射事件**：到达散射时间后发生散射，重新选择方向
4. **边界/界面处理**：声子到达边界或界面时发生反射/透射
5. **统计热流**：统计穿过计算域的能量，计算热导率

### 2. 主要优势

| 特性 | 解析BTE | 蒙特卡洛BTE |
|------|--------|------------|
| 复杂几何 | 困难 | 容易 |
| 界面效应 | 近似处理 | 精确模拟 |
| 非稳态输运 | 困难 | 直接支持 |
| 计算成本 | 低 | 高（需大量声子） |
| 统计误差 | 无 | 有（需收敛分析） |

## 文件结构

```
.
├── phonon_monte_carlo.py          # 蒙特卡洛核心模块
│   ├── PhononPacket              # 声子包数据类
│   ├── Layer                     # 材料层类
│   ├── Superlattice              # 超晶格类
│   └── PhononMonteCarlo          # 主求解器
├── example_monte_carlo.py         # 蒙特卡洛示例脚本
└── README_MONTE_CARLO.md          # 本文档
```

## 核心类说明

### 1. PhononPacket（声子包）

```python
@dataclass
class PhononPacket:
    phonon_id: int           # 声子ID
    branch: str              # 声子支 ('LA', 'TA1', 'TA2')
    omega: float             # 角频率 (rad/s)
    k_magnitude: float       # 波矢大小
    position: np.ndarray     # 位置向量 (x, y, z)
    velocity: np.ndarray     # 速度向量
    energy: float            # 声子能量
    active: bool             # 是否活跃
    time_to_scatter: float   # 下次散射剩余时间
    scatter_count: int       # 散射次数统计
```

### 2. Layer（材料层）

用于定义多层结构：
```python
@dataclass
class Layer:
    thickness: float           # 层厚 (m)
    material: str              # 材料类型
    interface_transmission: float  # 界面透射率 (0-1)
```

### 3. Superlattice（超晶格）

支持周期性超晶格结构：
```python
class Superlattice:
    def __init__(self, layers: List[Layer], periodic: bool = True)
    def get_layer_at(self, z: float) -> Tuple[int, Layer]
    def check_interface_crossing(self, z_old, z_new)
```

### 4. PhononMonteCarlo（主求解器）

```python
class PhononMonteCarlo:
    def __init__(self, material='Si', structure='bulk', **kwargs)
```

**初始化参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `material` | str | 材料: 'Si', 'Ge', 'Si/Ge' |
| `structure` | str | 结构: 'bulk', 'thin_film', 'superlattice' |
| `L` | float | 特征尺寸（薄膜厚度） |
| `T` | float | 平均温度 (K) |
| `dT` | float | 温度差 (K) |
| `seed` | int | 随机数种子 |

## 核心算法

### 1. 声子采样

**频率采样**（拒绝采样法）：
```python
def sample_phonon_frequency(self, branch='TA'):
    f_max = self.theta_D * k / hbar
    while True:
        omega = self.rng.uniform(0, f_max)
        p_accept = (omega / f_max)**2
        if self.rng.random() < p_accept:
            return omega
```

**分支采样**：
- LA: 1/3 概率
- TA1: 1/3 概率
- TA2: 1/3 概率

**方向采样**（各向同性）：
```python
theta = np.arccos(self.rng.uniform(-1, 1))
phi = self.rng.uniform(0, 2 * pi)
```

### 2. 散射事件处理

**散射时间采样**：
$$ t_{scatter} = -\tau \ln(\xi) $$
其中 $\xi \in [0,1)$ 为随机数。

**散射类型**：
- **动量弛豫**（70%概率）：随机重选方向
- **能量弛豫**（30%概率）：仅内部状态变化

### 3. 边界处理

**薄膜边界**：
- 镜面反射：90%概率（弹性反射）
- 漫反射：10%概率（随机重选方向）

**界面处理**：
- 透射：概率由界面透射率决定
- 反射：弹性反射，保留在原层内

### 4. 热导率计算

通过统计冷热端的能量交换：
$$ \kappa = \frac{Q \cdot L}{A \cdot \Delta T \cdot t_{sim}} $$

其中：
- $Q$：净能量传输
- $L$：样品厚度
- $A$：横截面积
- $\Delta T$：温度差
- $t_{sim}$：模拟时间

## 使用示例

### 1. 基础蒙特卡洛计算

```python
from phonon_monte_carlo import PhononMonteCarlo

# 创建薄膜模拟
mc = PhononMonteCarlo(
    material='Si',
    structure='thin_film',
    L=100e-9,  # 100nm厚
    seed=42
)

# 运行模拟
result = mc.run_simulation(
    n_phonons=5000,    # 声子包数量
    n_steps=500,       # 时间步数
    dt=1e-13,          # 时间步长
    T_hot=310,         # 热端温度
    T_cold=290         # 冷端温度
)

print(f"热导率: {result['thermal_conductivity']:.2f} W/mK")
```

### 2. 尺寸效应分析

```python
mc = PhononMonteCarlo(material='Si', structure='thin_film')

# 分析不同厚度下的热导率
L_array = np.logspace(-8, -6, 7)  # 10nm to 1μm
kappas, kappa_bulk = mc.size_effect_analysis(
    L_array,
    T=300,
    n_phonons=2000,
    n_steps=300
)
```

### 3. Si/Ge超晶格模拟

```python
from phonon_monte_carlo import Layer, Superlattice

# 定义超晶格层
layers = [
    Layer(5e-9, 'Si', 0.8),   # 5nm Si，透射率80%
    Layer(5e-9, 'Ge', 0.8),   # 5nm Ge，透射率80%
]
superlattice = Superlattice(layers, periodic=True)

mc = PhononMonteCarlo(
    material='Si/Ge',
    structure='superlattice',
    superlattice=superlattice,
    L=100e-9
)

kappa = mc.thermal_conductivity_mc(
    T=300,
    n_phonons=3000,
    n_steps=400
)
```

### 4. 界面透射率影响

```python
mc = PhononMonteCarlo(material='Si/Ge', structure='superlattice')

period = 10e-9
for trans in [0.4, 0.6, 0.8, 0.95]:
    layers = [
        Layer(period/2, 'Si', trans),
        Layer(period/2, 'Ge', trans),
    ]
    mc.superlattice = Superlattice(layers)
    kappa = mc.thermal_conductivity_mc(T=300, n_phonons=2000, n_steps=300)
    print(f"透射率={trans}, κ={kappa:.2f}")
```

## 运行示例脚本

```bash
python example_monte_carlo.py
```

**生成的图表：**

| 图表文件 | 说明 |
|---------|------|
| `size_effect_monte_carlo.png` | 薄膜热导率尺寸效应 |
| `superlattice_thermal_conductivity.png` | Si/Ge超晶格热导率 |
| `phonon_trajectories.png` | 声子运动轨迹可视化 |
| `interface_transmission_effect.png` | 界面透射率影响 |
| `temperature_dependence_mc.png` | 温度依赖性分析 |
| `method_comparison.png` | 蒙特卡洛vs解析BTE对比 |

## 收敛性分析

### 声子数收敛

为获得精确结果，需确保：
1. **n_phonons ≥ 2000**：统计噪声可接受
2. **n_steps ≥ 300**：声子充分扩散
3. **多重模拟平均**：减少随机误差

### 计算时间估计

| n_phonons | n_steps | 时间 (s) | 相对误差 |
|-----------|---------|----------|---------|
| 500 | 200 | ~5 | ~20% |
| 2000 | 300 | ~30 | ~10% |
| 5000 | 500 | ~120 | ~5% |
| 10000 | 1000 | ~500 | ~2% |

## 纳米传热应用

### 1. 纳米薄膜热导率

蒙特卡洛方法精确捕捉声子边界散射的尺寸效应：
- **Casimir极限**：当尺寸 << 声子MFP时，热导率线性减小
- **过渡区**：尺寸与MFP相当时，热导率随尺寸非线性变化

### 2. 超晶格热导调控

超晶格热导率低于体材料的原因：
1. **界面散射**：声子在异质结界面的反射
2. **相干效应**：周期性势场导致的声子色散改变
3. **能带折叠**：布里渊区折叠增加声子态密度

### 3. 声子工程策略

通过蒙特卡洛模拟优化：
- **界面粗糙度**：调整漫反射比例
- **层厚优化**：寻找最小热导率的周期
- **材料组合**：寻找最佳声学失配

## 技术细节

### 数值稳定性

1. **时间步长**：$dt \ll \tau_{min}$，确保散射事件正确
2. **边界检查**：精确的界面穿越检测
3. **能量守恒**：散射/反射不改变声子能量

### 并行计算支持

可通过以下方式加速：
```python
# 独立运行多个模拟后平均
seeds = [42, 123, 456, 789]
results = []
for seed in seeds:
    mc = PhononMonteCarlo(seed=seed)
    results.append(mc.thermal_conductivity_mc(...))
kappa_avg = np.mean(results)
```

## 与解析BTE对比

| 特性 | 解析BTE（RTA） | 蒙特卡洛BTE |
|------|---------------|------------|
| 体材料热导率 | ✓ 高效精确 | ✓ 但计算量大 |
| 简单薄膜 | ✓ 近似可用 | ✓ 精确 |
| 超晶格 | ✗ 难以处理 | ✓ 直接模拟 |
| 复杂界面 | ✗ 需模型 | ✓ 精确处理 |
| 瞬态热输运 | ✗ 困难 | ✓ 自然支持 |

## 参考文献

1. Mazumder, S., & Majumdar, A. (2001). Monte Carlo study of phonon transport in thin silicon films. Journal of Heat Transfer, 123(4), 749-759.

2. Landry, E. S., & McGaughey, A. J. H. (2009). Phonon transport in periodic silicon nanoporous films. Physical Review B, 79(11), 115432.

3. Chen, G. (2005). Nanoscale Energy Transport and Conversion. Oxford University Press.

4. Katcho, N. A., et al. (2016). Monte Carlo simulation of phonon thermal transport in superlattices. Journal of Applied Physics, 119(12), 125105.

## 常见问题

**Q: 蒙特卡洛模拟需要多少声子？**
A: 快速验证用500-1000个，精确计算用5000-10000个。

**Q: 如何减少统计噪声？**
A: 1) 增加声子数；2) 增加时间步数；3) 多次模拟取平均。

**Q: 超晶格模拟的最小周期是多少？**
A: 建议周期 ≥ 2nm，过小会出现明显的量子效应。

**Q: 界面透射率如何确定？**
A: 可通过声子失配模型（AMM）或扩散失配模型（DMM）计算，典型值0.6-0.9。
