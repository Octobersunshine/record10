# 微重力下液滴动力学数值模拟

本项目提供多种数值方法模拟微重力环境下的液滴动力学，包括自由振荡、电润湿效应、接触角迟滞等，用于空间流体管理（燃料箱、热控系统等）。

**特别说明**：相场法（Phase Field Method）已被实现以解决边界元法在拓扑变化（液滴合并/破裂）时的表面重构失败问题，并扩展支持电润湿效应和接触角迟滞。

## 方法对比

| 方法 | 拓扑变化 | 电润湿 | 接触角迟滞 | 计算效率 | 数值稳定性 | 适用场景 |
|------|---------|--------|-----------|----------|-----------|---------|
| **边界元法 (BEM)** | ❌ 失败 | ❌ 不支持 | ❌ 不支持 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 单液滴、小变形 |
| **相场法 (Cahn-Hilliard)** | ✅ 自然处理 | ✅ 完整支持 | ✅ 完整支持 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 电润湿、空间流体管理 |
| **格子玻尔兹曼 (LBM)** | ✅ 自然处理 | ⚠️ 部分支持 | ⚠️ 部分支持 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 并行计算、复杂流场 |
| **电润湿相场法** | ✅ 自然处理 | ✅ 完整支持 | ✅ 完整支持 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 微流控、热控系统 |

## 物理背景

### 1. 液滴自由振荡 (Rayleigh模式)

在微重力环境中，表面张力是液滴形状演化的主导因素。Rayleigh（1879）推导了无粘性不可压缩液滴的振荡频率公式：

```
ωₙ² = n(n-1)(n+2)γ/(ρR₀³)
```

其中：
- `ωₙ` 是第n阶模态的角频率
- `n` 是振荡模态数（n=2为基模）
- `γ` 是表面张力系数
- `ρ` 是液体密度
- `R₀` 是液滴平衡半径

### 2. 电润湿效应 (Electrowetting)

Lippmann-Young方程描述了接触角与施加电压的关系：

```
cosθ(V) = cosθ_Y + (ε₀ε_r / (2γ d)) V²
```

其中：
- `θ_Y` 是Young接触角（零电压）
- `ε₀` 是真空介电常数
- `ε_r` 是介电层相对介电常数
- `d` 是介电层厚度
- `V` 是施加电压

### 3. 接触角迟滞 (Contact Angle Hysteresis)

实际表面存在粗糙度和化学非均匀性，导致：
- **前进角 (θ_A)**：液滴扩展时的接触角
- **后退角 (θ_R)**：液滴收缩时的接触角
- **迟滞值**：Δθ = θ_A - θ_R

接触角迟滞会导致液滴钉扎（pinning）效应，这对空间流体管理至关重要。

## 文件说明

### 电润湿相场法（空间流体管理专用）⭐⭐⭐推荐

#### 1. `electrowetting_phase_field.py` ⭐最新
**Cahn-Hilliard相场法 + Navier-Stokes + 电场求解**，完整实现：
- ✅ 电场拉普拉斯方程求解（有限差分法）
- ✅ Lippmann-Young电润湿接触角模型
- ✅ Maxwell应力张量计算介电力
- ✅ 接触角迟滞模型（前进/后退角）
- ✅ 液滴钉扎效应模拟
- ✅ 多电极阵列液滴操控
- ✅ 空间流体管理应用示例

### 相场法（基础版本）

#### 2. `phase_field_droplet.py`
**Cahn-Hilliard相场法 + Navier-Stokes求解器**：
- ✅ 隐式界面追踪，自然处理拓扑变化
- ✅ Cahn-Hilliard方程求解相场演化
- ✅ 投影法求解Navier-Stokes方程
- ✅ 单液滴振荡模拟
- ✅ 双液滴合并模拟

#### 3. `lattice_boltzmann_droplet.py`
**格子玻尔兹曼方法 + 相场耦合**：
- ✅ LBM D2Q9模型求解流体流动
- ✅ 相场序参数输运方程
- ✅ 液滴合并/破裂模拟

### 边界元法（仅用于单液滴）

#### 4. `droplet_oscillation_bem.py`
完整的边界元法实现：
- ⚠️ **不支持拓扑变化和电润湿**

#### 5. `simple_bem_droplet.py`
简化版本的BEM实现

#### 6. `analytical_validation.py`
解析解验证程序

## 运行方法

### 电润湿模拟（空间流体管理）

```bash
# 电润湿液滴振荡模拟
python electrowetting_phase_field.py

# 电润湿液滴驱动（多电极阵列）- 空间流体管理演示
python electrowetting_phase_field.py actuate

# 接触角迟滞演示
python electrowetting_phase_field.py hysteresis
```

### 基础相场法

```bash
# 单液滴振荡模拟
python phase_field_droplet.py

# 双液滴合并模拟
python phase_field_droplet.py merge

# 格子玻尔兹曼方法
python lattice_boltzmann_droplet.py
python lattice_boltzmann_droplet.py merge
```

## 电润湿相场法核心实现

### 1. 电场求解 (`electrowetting_phase_field.py:113-157`)

```python
def solve_electric_field(self):
    # 介电常数随相场变化
    epsilon = self.compute_permittivity()
    
    # 有限差分法求解拉普拉斯方程 ∇·(ε∇φ_e) = 0
    for _ in range(200):
        # 五点差分格式
        phi_e_new[i, j] = (eps_w * phi_e[i+1, j] + 
                           eps_e * phi_e[i-1, j] + 
                           eps_n * phi_e[i, j+1] + 
                           eps_s * phi_e[i, j-1]) / coeff
```

### 2. 电润湿力计算 (`electrowetting_phase_field.py:159-168`)

```python
def compute_electrowetting_force(self):
    E_x, E_y = self.solve_electric_field()
    
    # Maxwell应力张量产生的介电力
    f_ew_x = 0.5 * (E_x² - E_y²) * (ε_l - ε_g) * ε₀ * ∇φ_x
    f_ew_y = E_x * E_y * (ε_l - ε_g) * ε₀ * ∇φ_y
```

### 3. 接触角模型 (`electrowetting_phase_field.py:179-207`)

```python
def compute_wetting_force(self):
    # Lippmann-Young方程
    cos_theta_e = cos_theta_Y + 0.5 * ε₀(ε_l - 1)V² / (γ d)
    
    # 接触角迟滞
    if advancing:
        theta = min(theta_e, theta_A)
    elif receding:
        theta = max(theta_e, theta_R)
```

### 4. 多电极液滴驱动 (`electrowetting_phase_field.py:498-567`)

```python
def voltage_schedule(t, sim_obj):
    # 顺序激活电极，实现液滴输运
    if t < 0.005:
        sim_obj.set_electrode_voltage(1, 20.0)
    elif t < 0.01:
        sim_obj.set_electrode_voltage(2, 20.0)
    # ... 依次激活后续电极
```

## 空间流体管理应用

### 1. 燃料箱液位控制

- 电润湿电极阵列控制燃料分布
- 接触角迟滞防止燃料晃动
- 微重力环境下的位置保持

### 2. 热控系统（环路热管）

- 电润湿驱动工质循环
- 精确控制液滴/气泡运动
- 主动热管理

### 3. 微流控芯片

- 液滴分离与合并
- 试剂混合
- 生物样本处理

## 典型参数

### 电润湿模拟参数
- 液滴半径：R₀ = 0.005-0.01 m
- 表面张力：γ = 0.072 N/m
- 液体介电常数：ε_l = 80
- 气体介电常数：ε_g = 1.0
- 工作电压：10-50 V
- Young接触角：45°
- 前进角：70°，后退角：40°
- 电极阵列：6个电极，间距10mm

### 数值参数
- 计算域：0.06m × 0.02m
- 网格：192 × 64
- 时间步长：5e-7 s

## 依赖项

- numpy
- matplotlib
- scipy (可选)

## 输出文件

电润湿模拟运行后生成：
- `electrowetting_results.png` - 相场、密度、速度、电场、压力分布图
- `electrowetting_actuation.png` - 液滴驱动过程
- `actuation_position.png` - 液滴位置随时间变化
- `hysteresis_results.png` - 接触角迟滞效应

## 为什么选择相场法进行空间流体管理？

### 边界元法的局限性
1. ❌ 显式界面追踪 - 拓扑变化困难
2. ❌ 无法处理电润湿效应 - 需要体网格
3. ❌ 难以模拟接触角迟滞
4. ❌ 不适合复杂固体边界

### 相场法的优势
1. ✅ 隐式界面 - 合并/分裂自然处理
2. ✅ 多物理场耦合 - 流体+电场+传热
3. ✅ 接触角迟滞 - 固体表面效应
4. ✅ 拓扑鲁棒性 - 液滴操控稳定
5. ✅ 易于扩展 - 添加相变、磁流体等

## 技术路线图

- [x] 相场法基础实现（Cahn-Hilliard + NS）
- [x] 拓扑变化处理（液滴合并）
- [x] 电场求解（拉普拉斯方程）
- [x] 电润湿效应（Lippmann-Young）
- [x] 接触角迟滞模型
- [x] 多电极液滴驱动
- [ ] 三维扩展
- [ ] 传热耦合（蒸发/冷凝）
- [ ] 实验验证

## 作者

相场法求解器用于空间流体管理和电润湿微流控研究
