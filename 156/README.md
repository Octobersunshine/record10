# 表面码量子纠错模拟器

这是一个用Python实现的表面码（Surface Code）量子纠错模拟器，支持错误模型模拟、MWPM解码器和逻辑错误率计算。

## 功能特性

- **表面码数据结构**: 支持任意码距的表面码实现
- **错误模型**: 
  - 独立比特翻转错误
  - 独立相位翻转错误
  - 去极化错误模型
- **MWPM解码器**: 最小权完美匹配解码器（使用贪心匹配算法）
- **逻辑错误率计算**: 蒙特卡洛模拟估计逻辑错误率

## 文件结构

```
e:\temp\record10\156\
├── surface_code.py      # 表面码核心实现
├── mwpm_decoder.py      # MWPM解码器实现
├── error_model.py       # 错误模型实现
├── simulator.py         # 模拟框架
├── main.py              # 主程序入口
├── test_simple.py       # 简单测试脚本
├── requirements.txt     # 依赖包列表
└── README.md            # 说明文档
```

## 安装依赖

```bash
pip install numpy scipy
```

或者：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 演示模式

运行单次表面码纠错试验演示：

```bash
python main.py --mode demo
```

### 2. 估计逻辑错误率

估计给定物理错误率下的逻辑错误率：

```bash
python main.py --mode estimate
```

### 3. 阈值扫描

扫描不同码距和物理错误率下的逻辑错误率：

```bash
python main.py --mode threshold
```

## 模块说明

### surface_code.py - SurfaceCode类

- `__init__(distance)`: 初始化指定码距的表面码
- `apply_bit_flip(qubit_idx)`: 应用比特翻转错误
- `apply_phase_flip(qubit_idx)`: 应用相位翻转错误
- `measure_stabilizers()`: 测量所有稳定子
- `get_defects(stab_type)`: 获取缺陷位置
- `get_logical_error()`: 检测逻辑错误
- `reset()`: 重置所有错误

### mwpm_decoder.py - MWPMDecoder类

- `__init__(surface_code)`: 初始化解码器
- `decode(stab_type)`: 对缺陷进行最小权完美匹配
- `apply_correction(matching, stab_type)`: 应用纠错操作

### error_model.py - 错误模型

- `ErrorModel(p_bit_flip, p_phase_flip)`: 独立错误模型
- `DepolarizingErrorModel(p)`: 去极化错误模型

### simulator.py - SurfaceCodeSimulator类

- `run_single_trial()`: 运行单次纠错试验
- `estimate_logical_error_rate(n_trials)`: 估计逻辑错误率
- `threshold_scan(distances, error_rates, n_trials)`: 阈值扫描

## 示例代码

```python
from surface_code import SurfaceCode
from mwpm_decoder import MWPMDecoder
from error_model import ErrorModel

# 创建码距为3的表面码
sc = SurfaceCode(3)

# 创建错误模型
error_model = ErrorModel(p_bit_flip=0.05, p_phase_flip=0.05)

# 应用错误
error_model.apply_errors(sc)

# 测量稳定子
x_stabs, z_stabs = sc.measure_stabilizers()

# 解码
decoder = MWPMDecoder(sc)
x_matching = decoder.decode('x')
z_matching = decoder.decode('z')

# 应用纠错
decoder.apply_correction(x_matching, 'x')
decoder.apply_correction(z_matching, 'z')

# 检测逻辑错误
x_logical, z_logical = sc.get_logical_error()
print(f"逻辑错误: X={x_logical}, Z={z_logical}")
```

## 理论背景

表面码是一种拓扑量子纠错码，具有以下特点：
- 二维格点结构，只需要近邻相互作用
- 阈值较高（约1%的物理错误率）
- 使用稳定子测量进行错误检测
- MWPM解码器用于缺陷配对和纠错

## 注意事项

- 当前实现使用贪心匹配算法近似MWPM，对于精确的最小权完美匹配可以使用blossom V算法
- 较大的码距和较多的试验次数会增加计算时间
- 逻辑错误率估计的统计误差随试验次数增加而减小
