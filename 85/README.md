# 7自由度机械臂逆运动学求解器

基于Python实现的完整7DOF机械臂逆运动学（IK）解决方案，包含**数值IK**和**解析IK**两种方法，支持臂型角参数化、多解控制、解空间分析等高级功能。

---

## 🌟 核心功能

### 数值逆运动学 (`ik_solver.py`)
解决多解和解跳变问题：
- **零空间投影**: 在不影响末端位姿的前提下优化次要目标
- **肘部姿态偏好**: 通过代价函数引导解向期望位形收敛
- **关节中心正则化**: 避免关节运动到限位极端位置
- **连续性约束**: 轨迹跟踪时保持解的连续性

### 解析逆运动学 (`analytical_ik.py`)
基于臂型角参数化的完整解空间控制：
- **臂型角参数化**: 用单一参数（臂型角ψ）描述所有逆运动学解
- **肘部位置控制**: 直接控制肘部在垂直于肩-腕连线平面内的位置
- **解空间分析**: 采样和分析整个可行解空间
- **多目标优化**: 支持肘部向上/向下、关节中心、可操作度等优化目标
- **冗余自由度利用**: 末端保持不动的同时控制肘部运动

---

## 📐 臂型角参数化理论

### 冗余自由度的本质
7DOF机械臂比完成6DOF位姿控制所需的自由度多1个，这个冗余自由度可以用**臂型角（Arm Angle, ψ）**来参数化。

### 几何意义
臂型角描述肘部在垂直于"肩关节-腕关节连线"平面内的旋转角度：
```
  肩部 (S) ---- 肘部 (E) ---- 腕部 (W)
  
  肘部可以在垂直于SW连线的平面内旋转360°（受关节限位限制）
  
  ψ = 0:   肘部处于参考位置
  ψ = π/2: 肘部旋转90°
  ...
```

### 数学表达
给定目标末端位姿T，所有逆运动学解可以表示为：
```
q(ψ) = 关节角度关于臂型角ψ的函数
ψ ∈ [ψ_min, ψ_max]  （受关节限位限制）
```

其中肘部位置随臂型角变化：
```
E(ψ) = S + d_proj * u1 + d_perp * (cosψ * u2 + sinψ * u3)
```
- `u1`: SW连线方向单位向量
- `u2, u3`: 垂直平面内的正交基向量

---

## 🔧 快速开始

### 安装依赖
```bash
pip install numpy scipy
```

## 安装依赖

```bash
pip install numpy scipy
```

## 使用方法

### 基础稳定求解（推荐）

```python
import numpy as np
from ik_solver import SevenDOFArm

arm = SevenDOFArm()

target_pos = np.array([0.5, 0.0, 0.5])

# 使用稳定版求解器
joint_angles, success = arm.inverse_kinematics_stable(
    target_pos,
    max_iterations=3000,
    tolerance=1e-4,
    step_size=0.05,
    damping=0.01,
    elbow_preference_weight=1.0,    # 肘部向上偏好
    center_weight=0.05,             # 关节中心正则化
    null_space_weight=0.1           # 零空间投影权重
)

if success:
    print(f"求解成功! 关节角度: {joint_angles}")
    pos_verify, _, _ = arm.forward_kinematics(joint_angles)
    print(f"位置误差: {np.linalg.norm(target_pos - pos_verify)}")
```

### 轨迹跟踪（连续性约束）

```python
# 生成轨迹
num_points = 30
trajectory = []
for t in np.linspace(0, 2 * np.pi, num_points):
    x = 0.5 + 0.2 * np.cos(t)
    y = 0.2 * np.sin(t)
    z = 0.5
    trajectory.append(np.array([x, y, z]))

# 轨迹跟踪，自动保持连续性
arm.reset_previous_solution()
solutions, successes = arm.inverse_kinematics_track(
    trajectory,
    max_iterations=2000,
    tolerance=1e-4,
    step_size=0.05,
    continuity_weight=1.0,       # 强连续性约束
    elbow_preference_weight=1.0,
    center_weight=0.05
)

# 检查关节跳变
for i in range(1, len(solutions)):
    jump = np.max(np.abs(solutions[i] - solutions[i-1]))
    print(f"点{i} 最大关节跳变: {jump:.4f} rad")
```

### 带姿态控制

```python
from scipy.spatial.transform import Rotation

target_pos = np.array([0.5, 0.0, 0.5])
target_rot = Rotation.from_euler('z', 90, degrees=True).as_matrix()

joint_angles, success = arm.inverse_kinematics_stable(
    target_pos,
    target_rot=target_rot,
    elbow_preference_weight=0.5,
    center_weight=0.05
)
```

## 参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `step_size` | 0.03 - 0.1 | 迭代步长，越大收敛越快但可能不稳定 |
| `damping` | 0.01 - 0.05 | 阻尼系数，越大越稳定但收敛越慢 |
| `elbow_preference_weight` | 0.5 - 2.0 | 肘部姿态偏好权重，越大越偏向目标姿态 |
| `center_weight` | 0.01 - 0.1 | 关节中心正则化权重，避免关节到限位 |
| `continuity_weight` | 0.5 - 2.0 | 连续性约束权重，轨迹跟踪时使用 |
| `null_space_weight` | 0.1 - 0.5 | 零空间投影权重 |
| `tolerance` | 1e-5 - 1e-3 | 收敛阈值 |

## 推荐配置

### 单点求解
```python
joint_angles, success = arm.inverse_kinematics_stable(
    target_pos,
    max_iterations=3000,
    tolerance=1e-4,
    step_size=0.05,
    damping=0.01,
    elbow_preference_weight=1.0,
    center_weight=0.05,
    null_space_weight=0.1
)
```

### 轨迹跟踪
```python
solutions, successes = arm.inverse_kinematics_track(
    trajectory,
    max_iterations=2000,
    tolerance=1e-4,
    step_size=0.05,
    continuity_weight=1.0,
    elbow_preference_weight=1.0,
    center_weight=0.05,
    null_space_weight=0.1
)
```

## 算法原理

### 零空间投影
利用7DOF机械臂的冗余性，在保证末端位姿精度的同时优化次要目标：

```
Δθ = J⁺·e + N·(-∇C)
```

其中:
- `J⁺·e`: 主任务 - 减小末端位置误差
- `N·(-∇C)`: 零空间任务 - 优化次要目标（肘部姿态、关节中心等）
- `N = I - J⁺J`: 零空间投影矩阵

### 代价函数

1. **肘部姿态代价**:
   ```
   C_elbow = (θ_elbow - θ_target)²
   ```

2. **关节中心代价**:
   ```
   C_center = Σ(θ_i - θ_center_i)²
   ```

3. **连续性代价**（轨迹跟踪时）:
   ```
   C_continuity = Σ(θ_i - θ_prev_i)²
   ```

## 📁 文件结构

```
.
├── ik_solver.py              # 数值逆运动学求解器
│   ├── SevenDOFArm          # 主类，包含正运动学、数值IK
│   ├── inverse_kinematics_stable   # 稳定版数值IK
│   └── inverse_kinematics_track    # 轨迹跟踪专用
│
├── analytical_ik.py          # 解析逆运动学求解器
│   ├── Analytical7DOF       # 臂型角参数化解空间控制
│   │   ├── inverse_kinematics_single_psi   # 指定臂型角求解
│   │   ├── inverse_kinematics_optimize_psi # 臂型角优化
│   │   └── inverse_kinematics_all_solutions # 完整解空间采样
│   └── AnalyticalIKWrapper  # 统一接口，自动回退数值方法
│
├── ik_example.py            # 数值IK对比演示
├── demo_analytical_ik.py    # 解析IK完整演示
└── README.md               # 本文档
```

## ▶️ 运行演示

### 1. 解析IK演示（推荐）
```bash
python demo_analytical_ik.py
```

演示内容包括：
- 臂型角参数化效果 - 同一目标不同肘部位置
- 解空间特性分析 - 不同位置的可行臂型角范围
- 多目标优化对比 - 肘部向上/向下、关节中心、可操作度
- 可操作度随臂型角的变化
- 冗余自由度利用 - 末端不动，肘部做圆周运动
- 关节限位对解空间的影响

### 2. 数值IK演示
```bash
python ik_solver.py
```

### 3. 数值IK稳定性对比
```bash
python ik_example.py
```

演示内容包括：
- 多解问题演示
- 肘部姿态偏好效果对比
- 连续性约束效果对比
- 关节中心正则化效果对比
- 奇异位形附近稳定性对比

## 🤖 机械臂参数

本实现基于Franka Emika Panda机械臂的DH参数：

| 关节 | α (扭转角) | a (连杆长度) | d (连杆偏移) | 角度范围 |
|------|------------|--------------|--------------|----------|
| 1 | π/2 | 0 | 0.333 | ±166° |
| 2 | -π/2 | 0 | 0 | ±101° |
| 3 | π/2 | 0 | 0.316 | ±166° |
| 4 | -π/2 | 0.0825 | 0 | -176° ~ -4° |
| 5 | π/2 | 0 | 0.384 | ±166° |
| 6 | -π/2 | -0.0825 | 0 | 1° ~ 215° |
| 7 | 0 | 0 | 0.2575 | ±166° |

**连杆长度**:
- 上臂长度: ~0.327 m
- 前臂长度: ~0.393 m
- 最大工作半径: ~0.85 m

## ⚠️ 注意事项

### 解析IK
1. **臂型角范围**: 实际可行范围受关节限位限制，通常小于[-π, π]
2. **工作空间**: 目标位置超出可达范围时无解，需检查肩-腕距离
3. **简化模型**: 当前实现为几何简化版，完整解析IK需要更精确的关节映射

### 数值IK
1. **肘部目标姿态**: 肘部向上目标为 ~-1.0 rad (~-57°)，向下为 ~-2.5 rad (~-143°)
2. **连续性约束**: 需要调用 `reset_previous_solution()` 重置前一解记忆
3. **权重调优**: 根据具体应用场景调整各权重，过大权重可能影响主任务精度
4. **奇异性**: 在工作空间边界附近，收敛速度和精度会下降

## 🚀 扩展功能建议

- ✅ 臂型角参数化解析逆运动学
- ✅ 多目标优化（肘部姿态、关节中心、可操作度）
- ✅ 解空间分析和可视化接口
- ⬜ 完整解析IK - 精确的7关节角度映射
- ⬜ 实时避障 - 在解空间搜索无碰撞的臂型角
- ⬜ 可操作度椭球可视化
- ⬜ 3D机械臂可视化界面
- ⬜ 轨迹跟踪中的臂型角连续控制
- ⬜ 多优先级任务分层控制
