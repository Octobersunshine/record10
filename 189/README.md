# 电力系统低频振荡模式辨识 - Prony分析工具

## 功能概述

本工具实现了基于Prony分析（指数拟合）的电力系统低频振荡模式辨识算法，支持从功角或功率曲线中提取振荡频率和阻尼比。

## 技术特点

- **两种分析方法**：
  - 矩阵束方法（Matrix Pencil Method）：抗噪性强，数值稳定性好（默认推荐）
  - 经典Prony方法：传统算法实现

- **信号预处理**：
  - 线性去趋势
  - 带通滤波（0.1-2Hz）

- **自动模式筛选**：
  - 自动去除共轭复数对
  - 按振幅排序显示主要模式
  - 有效频率范围：0.1-2Hz（低频振荡典型范围）

## 使用方法

### 基本使用示例

```python
import numpy as np
from prony_analysis import PronyAnalyzer, generate_test_signal, print_results, plot_results

# 1. 初始化分析器
analyzer = PronyAnalyzer(
    fs=100.0,              # 采样频率 (Hz)
    freq_range=(0.1, 2.0)  # 关注的频率范围
)

# 2. 生成或加载信号
# 示例：生成测试信号
true_modes = [
    {'freq': 0.4, 'damping': 0.05, 'amp': 1.0, 'phase': 0.0},
    {'freq': 1.2, 'damping': 0.08, 'amp': 0.6, 'phase': 0.3},
]
t, signal = generate_test_signal(fs=100.0, duration=20.0, modes=true_modes)

# 3. 执行Prony分析
result = analyzer.analyze(
    signal,
    method='matrix_pencil',  # 使用矩阵束方法
    order=30                 # 铅笔参数/模型阶数
)

# 4. 显示结果
print_results(result, true_modes=true_modes)

# 5. 绘制结果
plot_results(result, save_path='result_plot.png')
```

### 加载实测数据示例

```python
import numpy as np
from prony_analysis import PronyAnalyzer, print_results, plot_results

# 加载数据（假设CSV格式：时间, 功率/功角）
data = np.loadtxt('your_data.csv', delimiter=',', skiprows=1)
time = data[:, 0]
signal = data[:, 1]

# 计算采样频率
fs = 1.0 / (time[1] - time[0])

# 初始化分析器并执行分析
analyzer = PronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))
result = analyzer.analyze(signal, method='matrix_pencil', order=40)

# 输出结果
print_results(result)
plot_results(result)
```

## 结果字段说明

分析结果返回一个字典，包含以下字段：

| 字段 | 说明 |
|------|------|
| `freq` | 振荡频率数组 (Hz) |
| `damping_ratio` | 阻尼比数组 |
| `sigma` | 衰减系数数组 |
| `amplitude` | 振荡幅值数组 |
| `phase` | 初始相位数组 (rad) |
| `reconstructed` | 重构信号 |
| `processed` | 预处理后的信号 |
| `original` | 原始信号 |
| `time` | 时间序列 |
| `order` | 使用的模型阶数 |

## 参数调优指南

### 采样频率 (fs)
- 建议：至少为最高分析频率的5-10倍
- 对于2Hz的振荡，建议采样频率 ≥ 20Hz

### 模型阶数/铅笔参数 (order)
- 建议：数据长度的1/4 ~ 1/3
- 阶数过小：可能丢失弱阻尼模式
- 阶数过大：引入虚假模式，增加计算量
- 典型值：20-50

### 频率范围 (freq_range)
- 电力系统低频振荡典型范围：0.1-2Hz
- 区域振荡模式：0.1-0.8Hz
- 本地振荡模式：0.8-2.0Hz

## 运行测试

直接运行主程序进行测试：

```bash
python prony_analysis.py
```

测试信号包含3个预设的振荡模式：
- Mode 1: 0.40 Hz, 阻尼比 = 0.050
- Mode 2: 1.20 Hz, 阻尼比 = 0.080
- Mode 3: 0.80 Hz, 阻尼比 = 0.030

## 理论背景

### Prony分析原理

Prony方法将信号表示为一组衰减正弦分量的线性组合：

$$
x(t) = \sum_{k=1}^{M} A_k e^{\sigma_k t} \cos(2\pi f_k t + \phi_k)
$$

其中：
- $A_k$：第k个模式的幅值
- $\sigma_k$：第k个模式的衰减系数
- $f_k$：第k个模式的振荡频率
- $\phi_k$：第k个模式的初始相位

### 阻尼比计算

阻尼比 $\zeta$ 与衰减系数 $\sigma$ 和频率 $f$ 的关系：

$$
\zeta = \frac{-\sigma}{\sqrt{\sigma^2 + (2\pi f)^2}}
$$

## 文件说明

- `prony_analysis.py`：主程序，包含PronyAnalyzer类和所有功能函数
- `prony_analysis_plot.png`：运行测试后生成的可视化结果图

## 依赖库

- numpy
- scipy
- matplotlib

安装依赖：

```bash
pip install numpy scipy matplotlib
```
