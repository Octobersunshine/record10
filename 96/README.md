# 热传导反问题求解 - 共轭梯度法(CGM)

## 项目概述

本项目实现了使用共轭梯度法(Conjugate Gradient Method, CGM)求解一维热传导反问题。通过已知的内部温度历史数据，估算边界热流密度。

## 数学背景

### 热传导正问题

一维瞬态热传导方程：

```
∂T/∂t = α ∂²T/∂x²
```

边界条件：
- x=0: -k ∂T/∂x = q(t) （热流边界）
- x=L: T(L,t) = T_right （温度边界）

初始条件：T(x,0) = T0(x)

### 热传导反问题

反问题是病态问题(ill-posed)，需要使用正则化方法。目标函数：

```
J(q) = ||T_calc(q) - T_measured||²
```

使用共轭梯度法最小化目标函数，通过伴随问题(adjoint problem)计算梯度。

## 代码结构

### 1. `solve_heat_direct()`
- 求解热传导正问题
- 使用隐式有限差分法（无条件稳定）
- 返回温度场 T(x,t)

### 2. `compute_gradient()`
- 计算目标函数关于热流q的梯度
- 求解伴随问题得到拉格朗日乘子
- 返回梯度和目标函数值

### 3. `cgm_inverse()`
- 共轭梯度法主算法
- 支持Fletcher-Reeves和Polak-Ribiere两种beta计算方式
- 内置线搜索确定步长
- 返回优化后的热流、收敛历史

### 4. `generate_test_data()`
- 生成合成测试数据
- 已知精确热流，生成温度测量值
- 添加高斯噪声模拟实际测量

## 依赖库

```
numpy >= 1.18
scipy >= 1.5
matplotlib >= 3.3
```

安装命令：
```bash
pip install numpy scipy matplotlib
```

## 使用方法

### 直接运行示例：

```bash
python heat_inverse_cgm.py
```

### 自定义使用：

```python
from heat_inverse_cgm import solve_heat_direct, cgm_inverse, generate_test_data

# 生成测试数据
L, T_total, alpha, Nx, Nt, x_measured, t_measured, T_measured, q_true, t = generate_test_data()

# 初始猜测
q_initial = 50 * np.ones_like(t)

# 运行共轭梯度法
q_opt, cost_history, q_history = cgm_inverse(
    L, T_total, alpha, Nx, Nt,
    T_measured, x_measured, t_measured,
    q_initial, max_iter=50, tol=1e-4, beta_type='FR'
)
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| L | 空间域长度 | 1.0 m |
| T_total | 总时间 | 10.0 s |
| alpha | 热扩散系数 | 0.01 m²/s |
| Nx | 空间网格数 | 50 |
| Nt | 时间步数 | 100 |
| max_iter | 最大迭代次数 | 100 |
| tol | 收敛容差 | 1e-6 |
| beta_type | beta公式 | 'FR' |

## 输出结果

程序运行后会生成：

1. **边界热流对比图** - 精确热流 vs 反演热流
2. **目标函数收敛历史** - 对数坐标显示收敛过程
3. **热流反演误差** - 绝对误差随时间变化
4. **温度场云图** - 反演热流下的温度分布

结果图片保存为：`heat_inverse_cgm_results.png`

## 算法特点

1. **共轭梯度法** - 比最速下降法收敛更快
2. **伴随方法** - 高效计算梯度，复杂度与正问题相同
3. **线搜索** - 保证每步迭代目标函数下降
4. **正则化效果** - CGM本身具有正则化特性，迭代停止即正则化

## 扩展功能

可扩展的方向：

- 二维/三维热传导问题
- 多参数同时反演
- 不同正则化方法（Tikhonov, TV）
- 不确定性分析
- 并行计算加速

## 参考文献

1. Alifanov, O. M. (1994). Inverse Heat Transfer Problems. Springer.
2. Beck, J. V., Blackwell, B., & St. Clair, C. R. (1985). Inverse Heat Conduction: Ill-Posed Problems. Wiley.
3. Fletcher, R., & Reeves, C. M. (1964). Function minimization by conjugate gradients.
