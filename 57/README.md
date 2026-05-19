# EEG 互相关分析 - 功能连接分析工具

基于 Python + MNE 的多通道 EEG 信号互相关函数计算工具，用于脑功能连接分析。

## 更新日志

### v1.2 - 动态功能连接 (2026-05-19)
- **新增**: 滑动窗口动态功能连接分析
- **新增**: `compute_dynamic_functional_connectivity()` 函数
- **新增**: 三种动态连接可视化函数
- **支持**: 自定义窗口大小、步长、最大滞后参数

### v1.1 - 归一化修复 (2026-05-19)
- **修复**: 互相关计算中的归一化错误
- **问题**: 有偏估计导致零延迟自相关值 > 1
- **解决**: 将分母从 `N - |lag|` (无偏估计) 改为 `N` (有偏但归一化正确)
- **验证**: 标准化数据的零延迟自相关值现在为 ~1.0，相关值范围在 [-1, 1] 内

## 功能特性

- **数据加载**: 支持多种 EEG 数据格式 (.fif, .edf, .set)
- **信号预处理**: 滤波、去噪、epoch 提取
- **互相关计算**: 多通道间互相关函数计算
- **静态功能连接**: 三种连接强度计算方法 (max_abs, zero_lag, max_pos)
- **动态功能连接**: 滑动窗口分析，捕捉连接随时间的变化
- **可视化**: 
  - 互相关曲线
  - 功能连接矩阵热力图
  - 脑电拓扑图
  - 动态连接时间序列
  - 动态连接热力图
  - 动态拓扑图关键帧动画

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 运行完整演示（使用MNE示例数据）

```bash
python eeg_cross_correlation.py
```

### 运行基础示例（使用模拟数据）

```bash
python example_usage.py
```

## 核心函数说明

### 1. compute_cross_correlation()

计算多通道 EEG 数据的互相关函数

```python
corr_matrix, lags_ms = compute_cross_correlation(
    data,           # EEG数据 (n_channels, n_times)
    channel_names,  # 通道名称列表
    max_lag=100,    # 最大滞后（样本点）
    fs=1000         # 采样频率（Hz）
)
```

**算法细节**:
- 数据先经过 z-score 标准化 (均值=0, 标准差=1)
- 使用 `scipy.signal.correlate` 计算互相关
- 归一化: `corr / N` (N为样本数)，确保:
  - 自相关在零延迟时 ≈ 1.0
  - 所有相关值在 [-1, 1] 范围内

### 2. compute_functional_connectivity()

基于互相关计算静态功能连接矩阵

### 3. compute_dynamic_functional_connectivity()

滑动窗口计算动态功能连接矩阵序列

```python
conn_series, window_times = compute_dynamic_functional_connectivity(
    data,            # EEG数据 (n_channels, n_times)
    channel_names,   # 通道名称列表
    fs=1000,         # 采样频率（Hz）
    window_size=1000,# 窗口大小（样本点）
    step_size=500,   # 步长（样本点）
    max_lag=100,     # 互相关最大滞后（样本点）
    method='max_abs' # 连接强度计算方法
)
```

**返回值**:
- `conn_series`: 形状为 `(n_windows, n_channels, n_channels)` 的连接矩阵序列
- `window_times`: 每个窗口的中心时间点（秒）

### 4. 动态功能连接可视化函数

- `plot_connectivity_timeseries()`: 绘制多组通道对的连接强度时间序列
- `plot_dynamic_connectivity_heatmap()`: 绘制动态功能连接的热力图
- `plot_dynamic_topomap_animation()`: 绘制动态连接拓扑图的关键帧

```python
conn_matrix = compute_functional_connectivity(
    corr_matrix,   # 互相关矩阵
    lags_ms,       # 滞后时间数组
    method='max_abs'  # 计算方法
)
```

支持的计算方法：
- `'max_abs'`: 最大绝对值
- `'zero_lag'`: 零滞后相关
- `'max_pos'`: 最大正相关

### 3. 可视化函数

- `plot_cross_correlation()`: 绘制通道对的互相关函数
- `plot_functional_connectivity()`: 绘制功能连接矩阵热力图
- `plot_topomap()`: 绘制连接强度拓扑图

## 使用自定义数据

```python
from eeg_cross_correlation import load_custom_eeg_data, compute_cross_correlation

# 加载数据
raw = load_custom_eeg_data('your_data.edf')

# 提取数据
data = raw.get_data()
channel_names = raw.ch_names
fs = raw.info['sfreq']

# 计算互相关
corr_matrix, lags_ms = compute_cross_correlation(data, channel_names, max_lag=100, fs=fs)
```

## 文件说明

- `eeg_cross_correlation.py`: 主程序，包含完整分析流程
- `example_usage.py`: 使用示例和演示
- `requirements.txt`: Python 依赖包
