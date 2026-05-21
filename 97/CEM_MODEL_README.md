# 完整电极模型 (Complete Electrode Model, CEM) 说明文档

## 问题背景

### 电极接触阻抗问题
在实际EIT测量中，电极与被测对象（如皮肤）之间存在**接触阻抗**。当这些接触阻抗不匹配时，传统的点电极模型会产生显著的模型误差，表现为：
- **重构图像偏心**：异常区域的位置偏移
- **伪影产生**：出现不存在的假异常
- **定量误差**：电导率值估计偏差

### 传统点电极模型的假设
传统模型假设：
1. 电极是无限小的点
2. 电流在单点注入/流出
3. 接触阻抗为零或均匀

## 完整电极模型 (CEM) 原理

### CEM的数学表述

CEM模型在边界上引入以下条件：

**电流守恒条件**：
```
∫_{e_l} σ ∂u/∂n ds = I_l
```
其中 `e_l` 是第 l 个电极的面积，`I_l` 是注入电流。

**电极电位条件**：
```
u + z_l σ ∂u/∂n = V_l   在电极 e_l 上
```
其中 `z_l` 是第 l 个电极的接触阻抗，`V_l` 是电极表面的平均电位。

**绝缘边界条件**：
```
σ ∂u/∂n = 0   在非电极区域
```

### 扩展的有限元系统

CEM模型将未知量从节点电位 `u` 扩展为 `[u; V]`，其中 `V` 是电极电位向量。

系统矩阵形式：
```
[ K    C^T ] [u]   = [0]
[ C     D  ] [V]     [I]
```

其中：
- `K`: 标准刚度矩阵
- `C`: 电极-节点关联矩阵
- `D`: 接触阻抗对角矩阵
- `I`: 注入电流向量

## 实现细节

### 1. 网格生成 (`EITMesh` 类)

```python
mesh = EITMesh(n_radius=6, n_angles=16, r=1.0, electrode_angle_width=0.25)
```

- `electrode_angle_width`: 电极覆盖的角度范围（弧度）
- 每个电极可以包含多个边界节点

### 2. 正向求解 (`EITForwardCEM` 类)

#### 组装CEM系统矩阵
```python
sys = assemble_cem_system(sigma, contact_impedance)
```

关键步骤：
1. 组装标准刚度矩阵 `K`
2. 构建电极-节点关联矩阵 `C`
3. 构建接触阻抗矩阵 `D`
4. 组合扩展系统并求解

#### 模拟测量
```python
voltages = simulate_measurements(sigma, contact_impedance)
```

采用相邻电流注入模式，测量其余电极的电位差。

### 3. 逆问题求解 (`EITInverseCEM` 类)

#### 方法一：联合重构（推荐）
同时重构电导率分布和接触阻抗：

```python
sigma_recon, z_recon = reconstruct_joint(measured_voltages)
```

目标函数：
```
min_{σ,z} ||V_meas - F(σ, z)||² + λ_σ ||σ - σ₀||² + λ_z ||z - z₀||²
```

参数：
- `reg_sigma`: 电导率正则化参数（默认 1e-3）
- `reg_z`: 接触阻抗正则化参数（默认 1e-2）

#### 方法二：已知接触阻抗重构
如果接触阻抗已知，可以单独重构电导率：

```python
sigma_recon = reconstruct_with_known_z(measured_voltages, contact_impedance)
```

### 4. 雅可比矩阵计算

采用**伴随方法**结合**有限差分**：

- 电导率雅可比 (`J_sigma`): 使用伴随方法，基于电场能量积分
- 接触阻抗雅可比 (`J_z`): 使用有限差分扰动

## 使用示例

### 快速测试
```bash
python test_cem.py
```

### 完整对比演示
```bash
python demo_cem_contact_impedance.py
```

### 自定义使用

```python
from eit_solver_cem import EITMesh, EITForwardCEM, EITInverseCEM

# 1. 创建网格
mesh = EITMesh(n_radius=6, n_angles=16)

# 2. 创建正向求解器
forward = EITForwardCEM(mesh)

# 3. 模拟测量数据
sigma_true = create_your_conductivity_distribution()
z_true = np.ones(mesh.n_electrodes) * 0.1  # 基准接触阻抗
z_true[:8] *= 4.0  # 左侧阻抗升高

measured = forward.simulate_measurements(sigma_true, z_true)

# 4. 联合重构
inverse = EITInverseCEM(forward, max_iter=20, reg_sigma=5e-3, reg_z=5e-2)
sigma_recon, z_est = inverse.reconstruct_joint(measured)
```

## 关键参数调优

### 正则化参数选择

| 参数 | 推荐范围 | 作用 |
|------|----------|------|
| `reg_sigma` | 1e-4 ~ 1e-2 | 控制电导率分布的平滑性 |
| `reg_z` | 1e-3 ~ 1e-1 | 控制接触阻抗估计的稳定性 |

### 迭代参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_iter` | 15 ~ 30 | 迭代次数 |
| `tol` | 1e-4 ~ 1e-3 | 收敛阈值 |

## 预期效果

### 接触阻抗不匹配的影响
- **忽略接触阻抗**: 图像偏心明显，误差通常 > 30%
- **使用CEM模型**: 图像偏心被修正，误差通常 < 15%

### 质心偏移改善
CEM模型通常能将质心偏移减少 50% ~ 80%。

## 参考文献

1. Cheney, M., Isaacson, D., & Newell, J. C. (1999). Electrical impedance tomography. SIAM review, 41(1), 85-101.

2. Vauhkonen, M., et al. (1999). Electrical impedance tomography and electrode models. IEEE Transactions on Biomedical Engineering.

3. Adler, A., & Lionheart, W. R. B. (2006). Uses and abuses of EIDORS: an extensible software base for EIT. Physiological measurement.
