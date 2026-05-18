# 遗传算法选择压力问题修复说明

## 问题分析

原始实现存在以下问题导致选择压力过小，进化缓慢：

1. **缺少适应度尺度变换**：直接使用原始目标函数值作为适应度，当种群差异较小时，选择区分度不足
2. **选择方法单一**：仅使用锦标赛选择，压力不可调节
3. **未分离原始适应度与缩放适应度**：无法进行灵活的适应度变换

## 修复内容

### 1. 新增多种选择方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `TournamentSelection` | 锦标赛选择 | 通用场景 |
| `RouletteWheelSelection` | 轮盘赌选择 | 适应度分布均匀时 |
| `RankSelection` | 排名选择（默认） | 选择压力可控，推荐使用 |
| `StochasticUniversalSampling` | 随机通用采样 | 减少采样偏差 |

### 2. 新增多种适应度尺度变换方法

| 方法 | 说明 | 特点 |
|------|------|------|
| `NoScaling` | 无变换 | 选择压力小，不推荐 |
| `LinearScaling` | 线性尺度变换 | 简单有效，可调节放大倍数 |
| `SigmaTruncationScaling` | Sigma截断变换（默认） | 鲁棒性好，抑制超级个体 |
| `PowerLawScaling` | 幂律变换 | 非线性变换 |

### 3. 新增配置参数

- `SelectionMethod`: 选择方法
- `FitnessScaling`: 适应度缩放方法
- `ScalingFactor`: 缩放系数（默认 2.0）
- `PressureFactor`: 排名选择压力系数（1.0~2.0，默认 1.7）

### 4. 新增预设配置

```go
// 默认配置：平衡探索与利用
genetic.DefaultConfig()

// 高压力配置：快速收敛
genetic.HighPressureConfig()
```

## 使用示例

### 基本使用（默认高选择压力）
```go
config := genetic.DefaultConfig()
ga := genetic.NewGA(objective, varRanges, nil, config)
solution, fitness, err := ga.Run()
```

### 自定义选择压力
```go
config := genetic.DefaultConfig()
config.SelectionMethod = genetic.RankSelection    // 排名选择
config.FitnessScaling = genetic.LinearScaling    // 线性缩放
config.PressureFactor = 1.9                     // 较高选择压力
config.ScalingFactor = 3.0                       // 缩放系数
```

### 配置对比
| 参数 | 默认配置 | 高压力配置 |
|------|----------|------------|
| 种群大小 | 100 | 150 |
| 精英数量 | 5 | 10 |
| 选择方法 | 排名选择 | 排名选择 |
| 缩放方法 | Sigma截断 | 线性缩放 |
| 压力系数 | 1.7 | 1.9 |
| 缩放系数 | 2.0 | 3.0 |

## 效果提升

1. **收敛速度提升**：排名选择 + 适应度缩放使优秀个体被选中概率显著增加
2. **鲁棒性增强**：Sigma截断有效抑制"超级个体"主导种群，避免早熟收敛
3. **适应性提高**：多种选择和缩放方法可根据问题特性灵活配置
4. **选择压力可控**：通过 `PressureFactor` 在 1.0~2.0 之间平滑调节选择压力
