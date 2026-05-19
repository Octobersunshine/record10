# 在线贝叶斯变点检测使用指南

## 概述

在线贝叶斯变点检测（Online Bayesian Changepoint Detection）是一种用于实时流数据的变点检测算法。基于 Adams & MacKay (2007) 提出的算法，使用贝叶斯因子进行序列更新。

## 核心特性

- ✅ **逐点处理**: 无需存储完整历史数据，适合实时流数据
- ✅ **贝叶斯因子**: 量化变点证据强度，便于决策
- ✅ **自适应参数**: 根据数据特性自动调整危险率
- ✅ **内存高效**: 使用滑动窗口和充分统计量
- ✅ **并行支持**: 可同时监控多个数据流

## 快速开始

### 基本使用

```python
import numpy as np
from online_changepoint_detection import OnlineBayesianChangepointDetection

# 初始化检测器
detector = OnlineBayesianChangepointDetection(
    hazard=0.01,      # 危险率参数
    threshold=0.5     # 变点告警阈值
)

# 模拟流数据
data_stream = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(5, 1.2, 100)
])

# 逐点处理（模拟实时流）
for t, x in enumerate(data_stream):
    # 更新检测器，返回当前时刻是变点的概率
    cp_prob = detector.update(x)
    
    # 获取贝叶斯因子
    bf = detector.get_bayes_factor()
    
    # 检测告警
    if cp_prob > 0.5:
        print(f"t={t}: 检测到变点! 概率={cp_prob:.4f}, BF={bf:.2f}")

# 打印摘要
detector.print_summary()
```

### 批量处理

```python
# 如果已经有完整数据，可以批量处理
detector.update_batch(data)
```

## 核心类

### 1. OnlineBayesianChangepointDetection

标准在线变点检测器。

#### 初始化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hazard | 0.01 | 危险率（任意时刻发生变点的先验概率） |
| mu0 | 0.0 | 均值先验的均值 |
| kappa0 | 1.0 | 均值先验的精度 |
| alpha0 | 1.0 | 方差先验的形状参数 |
| beta0 | 1.0 | 方差先验的尺度参数 |
| max_run_length | 1000 | 最大运行长度（控制内存使用） |
| threshold | 0.5 | 变点告警阈值 |

#### 主要方法

| 方法 | 说明 |
|------|------|
| `update(x)` | 处理新数据点，返回变点概率 |
| `update_batch(data)` | 批量处理数据 |
| `get_bayes_factor()` | 获取贝叶斯因子 |
| `get_current_segment_stats()` | 获取当前段统计 |
| `print_summary()` | 打印检测摘要 |
| `plot_results()` | 可视化结果 |
| `reset()` | 重置检测器状态 |

### 2. AdaptiveOnlineChangepointDetection

自适应在线变点检测器，根据数据统计特性自动调整危险率。

#### 额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hazard_min | 0.001 | 危险率下界 |
| hazard_max | 0.1 | 危险率上界 |
| adaptation_rate | 0.01 | 适应速率 |

## 贝叶斯因子解读

贝叶斯因子（Bayes Factor, BF）用于量化变点的证据强度：

| 贝叶斯因子 | 证据强度 | 解读 |
|-----------|---------|------|
| BF < 1 | - | 支持无变点 |
| 1 < BF < 3 | 微弱 | 微弱支持变点 |
| 3 < BF < 10 | 实质性 | 实质性支持变点 |
| 10 < BF < 100 | 强烈 | 强烈支持变点 |
| BF > 100 | 决定性 | 决定性支持变点 |

使用示例：
```python
bf = detector.get_bayes_factor()

if bf > 10:
    print("强烈证据表明发生了变点")
elif bf > 3:
    print("有实质性证据表明可能发生了变点")
elif bf > 1:
    print("有微弱证据，建议继续观察")
else:
    print("没有足够证据表明发生了变点")
```

## 参数调优指南

### 危险率 (hazard)

- **hazard = 0.001**: 适用于变化稀少的稳定系统（预期每1000点变点一次）
- **hazard = 0.01**: 默认值，平衡场景（预期每100点变点一次）
- **hazard = 0.05**: 适用于频繁变化的动态系统

### 阈值 (threshold)

- **threshold = 0.3**: 高灵敏度，可能产生更多误报
- **threshold = 0.5**: 平衡，推荐值
- **threshold = 0.7**: 低灵敏度，更保守

## 使用场景示例

### 场景1: 工业传感器监控

```python
from online_changepoint_detection import OnlineBayesianChangepointDetection

class TemperatureSensorMonitor:
    def __init__(self):
        self.detector = OnlineBayesianChangepointDetection(
            hazard=0.005,
            threshold=0.6
        )
        self.alert_log = []
    
    def process_reading(self, value, timestamp):
        cp_prob = self.detector.update(value)
        bf = self.detector.get_bayes_factor()
        
        if cp_prob > 0.6:
            self.alert_log.append({
                'timestamp': timestamp,
                'value': value,
                'probability': cp_prob,
                'bayes_factor': bf
            })
            print(f"温度异常告警 @ {timestamp}: 值={value:.1f}, 概率={cp_prob:.3f}")
        
        return cp_prob, bf

# 使用
monitor = TemperatureSensorMonitor()

# 模拟实时数据流
for t in range(1000):
    if t < 300:
        val = np.random.normal(25, 0.5)  # 正常温度
    else:
        val = np.random.normal(45, 2.0)  # 异常高温
    
    monitor.process_reading(val, t)
```

### 场景2: 金融时间序列监控

```python
from online_changepoint_detection import AdaptiveOnlineChangepointDetection

detector = AdaptiveOnlineChangepointDetection(
    hazard=0.02,
    hazard_min=0.005,
    hazard_max=0.1,
    threshold=0.5
)

# 处理价格数据
for price in price_stream:
    prob = detector.update(price)
    bf = detector.get_bayes_factor()
    
    if bf > 10:
        send_trading_alert(f"市场结构发生变化，BF={bf:.1f}")
```

### 场景3: 多传感器并行监控

```python
sensors = {}
for sensor_id in ['temp', 'pressure', 'vibration']:
    sensors[sensor_id] = OnlineBayesianChangepointDetection(
        hazard=0.01,
        threshold=0.5
    )

# 实时处理
while True:
    readings = get_sensor_readings()
    for sensor_id, value in readings.items():
        prob = sensors[sensor_id].update(value)
        if prob > 0.7:
            print(f"{sensor_id}: 检测到异常变化!")
```

## 与离线MCMC的比较

| 特性 | 在线变点检测 | 离线MCMC |
|------|-------------|---------|
| 处理方式 | 逐点，实时 | 批量，后处理 |
| 内存使用 | O(K)，可控制 | O(N)，随数据增长 |
| 延迟 | 即时 | 需等待全部数据 |
| 变点数量后验 | 近似 | 精确MCMC采样 |
| 计算复杂度 | 每步O(K) | 整体O(M*N)，M为迭代次数 |
| 适用场景 | 实时监控、流数据 | 历史数据分析 |

**选择指南:**
- 需要实时告警 → 使用在线检测
- 分析历史数据 → 使用离线MCMC
- 数据量大 → 使用在线检测
- 需要精确后验分布 → 使用离线MCMC

## 常见问题

### Q: 如何选择 hazard 参数？

A: 可以根据先验知识估计变点发生频率。如果预期每 T 个点发生一次变点，设置 hazard ≈ 1/T。

### Q: 检测器产生太多误报怎么办？

A: 
1. 降低 hazard 参数
2. 提高 threshold 阈值
3. 使用自适应检测器，它会根据数据自动调整
4. 考虑加入后处理，例如要求连续多个点概率都高

### Q: 检测延迟怎么办？

A: 贝叶斯方法天然有一定延迟，因为需要积累足够证据。可以：
1. 降低 threshold 以提高灵敏度
2. 适当提高 hazard
3. 接受一定延迟以换取准确性

### Q: 如何处理非正态分布数据？

A: 当前实现假设正态分布。对于其他分布，需要修改 `_pred_prob` 和 `_update_sufficient_stats` 方法中的预测分布和充分统计量更新。

## 参考文献

- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742
