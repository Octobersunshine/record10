# 无人机姿态PID控制器仿真

## 项目简介
本项目实现了一个基于Python的PID控制器仿真，用于无人机姿态控制。通过给定角度误差，计算控制力矩输出，并仿真系统的动态响应。

**核心特性：**
- ✅ 标准PID控制算法
- ✅ **抗积分饱和(Anti-Windup)机制**（反向计算法）
- ✅ **自动调参功能**（继电反馈法 + Ziegler-Nichols公式）
- ✅ 多种整定方法支持

## 文件结构
- `pid_drone_attitude.py` - 主程序文件，包含所有功能实现

## 核心组件

### 1. PIDController 类
PID控制器实现，包含以下功能：
- 比例(P)、积分(I)、微分(D)三项计算
- 输出限幅
- **抗积分饱和(Anti-Windup)机制**（反向计算法）
- 状态重置

**参数：**
- `kp`: 比例系数
- `ki`: 积分系数
- `kd`: 微分系数
- `setpoint`: 目标设定值
- `output_limits`: 输出限制范围
- `anti_windup`: 是否启用抗积分饱和（默认True）
- `anti_windup_gain`: 抗积分饱和增益（默认0.5，推荐0.5~2.0）

### 2. RelayFeedbackTuner 类
**自动PID参数整定器**，使用继电反馈法获取系统临界参数：

- 自动进行继电反馈测试
- 检测自持振荡，获取临界增益和临界周期
- 支持多种整定方法计算PID参数

**参数：**
- `plant`: 被控对象（如DroneAttitudeDynamics）
- `relay_amplitude`: 继电输出幅值
- `hysteresis`: 滞环宽度
- `simulation_time`: 仿真时间
- `dt`: 时间步长

**支持的整定方法：**

| 方法 | Kp | Ki | Kd | 特点 |
|------|-----|-----|-----|------|
| ziegler-nichols | 0.6K_crit | 2Kp/T_crit | Kp*T_crit/8 | 经典方法，响应快但有超调 |
| pessen-integral | 0.7K_crit | 2.5Kp/T_crit | 0.15Kp*T_crit | 积分误差最小，超调适中 |
| some-overshoot | 0.33K_crit | 2Kp/T_crit | Kp*T_crit/3 | 允许少量超调，平衡性能 |
| no-overshoot | 0.2K_crit | 2Kp/T_crit | 0.5Kp*T_crit | 追求无超调，响应较慢 |

**继电反馈法原理：**
```python
# 临界增益计算
K_crit = 4 * d / (π * a)
# d: 继电输出幅值, a: 振荡幅值
```

### 3. DroneAttitudeDynamics 类
无人机姿态动力学模型，模拟角度和角速度的变化：
- 转动惯量 `inertia`
- 阻尼系数 `damping`
- 基于力矩输入更新姿态

### 4. 仿真函数
- `simulate_pid()`: 运行PID控制仿真，支持抗积分饱和参数配置
- `plot_results()`: 绘制响应曲线（角度、力矩、积分项）
- `plot_comparison()`: 绘制有/无抗积分饱和的对比图
- `plot_relay_test()`: 绘制继电反馈测试图
- `calculate_performance_metrics()`: 计算性能指标
- `print_auto_tune_results()`: 格式化输出自动整定结果

## 性能指标
- **稳态误差**: 最终角度与目标角度的差值
- **超调量**: 最大超出目标值的百分比
- **上升时间**: 达到目标值90%所需的时间
- **调节时间**: 进入目标值±2%误差带所需的时间

## 使用方法

### 环境要求
```bash
pip install numpy matplotlib
```

### 运行仿真
```bash
python pid_drone_attitude.py
```

### 手动使用自动调参功能

```python
# 创建被控对象
drone = DroneAttitudeDynamics()

# 创建自动整定器
tuner = RelayFeedbackTuner(
    plant=drone,
    relay_amplitude=2.0,
    hysteresis=0.3,
    simulation_time=15.0,
    dt=0.01
)

# 自动获取PID参数
params, time_array, output_history, input_history = tuner.auto_tune(
    setpoint=10.0, method='ziegler-nichols'
)

# 使用整定的参数进行仿真
kp_auto = params['kp']
ki_auto = params['ki']
kd_auto = params['kd']
```

### 调整参数
在 `main()` 函数中修改：
```python
# 目标参数
target_angle = 30.0
anti_windup_gain = 1.0

# 继电反馈测试参数
relay_amplitude = 2.0
hysteresis = 0.3
simulation_time = 15.0

# 选择整定方法
best_method = 'ziegler-nichols'  # 或其他方法
```

## 参数调优指南

### 自动调参 vs 手动调参

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **自动调参** | 无需经验，快速获取初始参数 | 参数可能非最优 | 首次调试、无经验用户 |
| **手动调参** | 可精细优化到最佳性能 | 需要经验，耗时 | 追求最佳性能、已有初始参数 |

**推荐流程：**
1. ✅ **先用继电反馈法自动获取初始PID参数**
2. 根据实际需求选择整定方法
3. 启用抗积分饱和机制改善大阶跃响应
4. 必要时微调参数以获得更佳性能

### PID基础参数

| 参数 | 增加效果 | 减少效果 |
|------|---------|---------|
| Kp | 加快响应，减小稳态误差 | 响应变慢，稳态误差增大 |
| Ki | 消除稳态误差 | 增加超调和调节时间 |
| Kd | 抑制超调，提高稳定性 | 对噪声更敏感 |

### 抗积分饱和参数

| 参数 | 值范围 | 效果 |
|------|--------|------|
| anti_windup_gain | 0.5 ~ 2.0 | 值越大，抗饱和效果越强，但可能导致响应变慢 |

### 继电反馈调参参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| relay_amplitude | 1.0 ~ 3.0 | 继电输出幅值，太小可能不振荡 |
| hysteresis | 0.1 ~ 0.5 | 滞环宽度，太小可能产生高频振荡 |
| simulation_time | 10 ~ 20 s | 需足够长以建立稳定振荡 |

## 积分饱和问题说明

**什么是积分饱和？**
- 当系统有大的阶跃输入时，积分项会持续累积
- 即使输出已达到饱和限制，积分项仍在增加
- 当误差反向时，积分项需要很长时间才能消退
- 导致超调剧烈、调节时间延长，甚至系统发散

**抗积分饱和的作用：**
- 检测输出饱和状态
- 反向调整积分项，防止过度累积
- 显著改善大阶跃响应的性能

## 输出结果
运行程序后会执行三个步骤：

**步骤1: 继电反馈自动调参**
- 控制台打印4种整定方法的PID参数
- 保存: `relay_feedback_test.png`

**步骤2: 自动整定参数仿真验证**
- 控制台打印性能指标
- 保存: `pid_response_自动整定+抗积分饱和.png`

**步骤3: 抗积分饱和效果验证**
- 对比有/无抗积分饱和的性能
- 保存: `pid_response_自动整定-无抗积分饱和.png`
- 保存: `pid_anti_windup_comparison.png`

## 扩展功能建议
- 添加噪声模拟
- 实现多轴姿态控制
- 添加干扰力矩测试
- 实现自适应PID控制
- 增加其他抗积分饱和算法（如条件积分法、Tracking Back-Calculation）
