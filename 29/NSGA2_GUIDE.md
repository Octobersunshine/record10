# NSGA-II 多目标优化使用指南

## 概述

NSGA-II（Non-dominated Sorting Genetic Algorithm II）是一种经典的多目标优化算法，能够找到Pareto最优解集。

## 核心功能

### 数据结构

```go
type MultiObjectiveResult struct {
    Solutions  [][]float64  // Pareto最优解
    Objectives [][]float64  // 对应的目标函数值
}

type NSGAIIIndividual struct {
    Genes            []float64  // 决策变量
    Objectives       []float64  // 目标函数值
    Rank             int        // Pareto等级（0为最优）
    CrowdingDistance float64    // 拥挤度
    Viable           bool       // 是否满足约束
}
```

## 使用方法

### 基本使用步骤

1. **定义目标函数**：需要最小化的目标函数
2. **定义变量范围**：每个决策变量的上下界
3. **定义约束条件**（可选）：约束函数，返回true表示满足
4. **配置算法参数**：种群大小、迭代次数等
5. **运行优化**：获取Pareto前沿

### 示例代码

```go
package main

import (
    "fmt"
    "genetic"
)

// 目标函数1：最小化 f1 = x
func objective1(x []float64) float64 {
    return x[0]
}

// 目标函数2：最小化 f2 = (x-2)^2
func objective2(x []float64) float64 {
    return (x[0] - 2) * (x[0] - 2)
}

func main() {
    // 变量范围
    varRanges := []genetic.VariableRange{
        {Min: -10, Max: 10},
    }
    
    // 目标函数列表
    objectives := []genetic.ObjectiveFunc{objective1, objective2}
    
    // 约束条件（可选）
    constraints := []genetic.ConstraintFunc{
        func(x []float64) bool { return x[0] >= 0 },
    }
    
    // 算法配置
    config := genetic.DefaultNSGAIIConfig()
    config.PopulationSize = 100
    config.MaxGenerations = 300
    
    // 创建并运行NSGA-II
    nsga := genetic.NewNSGAII(objectives, varRanges, constraints, config)
    result, err := nsga.Run()
    
    if err != nil {
        fmt.Println("错误:", err)
        return
    }
    
    // 输出结果
    fmt.Printf("找到 %d 个Pareto最优解\n", len(result.Solutions))
    for i := range result.Solutions {
        fmt.Printf("解 %d: x=%.4f, f1=%.4f, f2=%.4f\n",
            i, result.Solutions[i][0],
            result.Objectives[i][0],
            result.Objectives[i][1])
    }
}
```

## 算法核心机制

### 1. 快速非支配排序 (Fast Non-dominated Sort)

将种群按Pareto支配关系分层：
- **Rank 0**：非支配解（Pareto前沿）
- **Rank 1**：仅被Rank 0支配的解
- 以此类推...

### 2. 拥挤度计算 (Crowding Distance)

衡量解周围的密度，保持种群多样性：
- 边界解拥挤度为无穷大
- 内部解根据目标空间距离计算

### 3. 拥挤锦标赛选择

优先选择：
1. Rank更低的解
2. Rank相同时，拥挤度更大的解

### 4. 精英保留策略

父代和子代合并后选择最优个体。

## 测试问题

### Schaffer N1
- 变量：1维
- 目标：f1 = x², f2 = (x-2)²
- 真实Pareto前沿：x ∈ [0, 2]

### ZDT1
- 变量：30维
- 凸Pareto前沿
- 常用于测试收敛性

### ZDT2
- 变量：30维
- 非凸Pareto前沿
- 更具挑战性

## 参数配置建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| PopulationSize | 50-200 | 目标越多需要越大种群 |
| MaxGenerations | 200-1000 | 根据问题复杂度调整 |
| CrossoverRate | 0.8-0.9 | 较高的交叉率 |
| MutationRate | 0.05-0.15 | 根据变量数调整 |
| TournamentSize | 2 | 二元锦标赛 |

## 结果分析

### 运行示例

```bash
cd example
go run nsga2_example.go
```

### 如何判断收敛

1. **Pareto前沿形状**：与理论前沿对比
2. **解的分布**：均匀分布表示多样性好
3. **解的数量**：通常应接近种群大小

## 高级用法

### 带约束的多目标优化

```go
constraints := []genetic.ConstraintFunc{
    func(x []float64) bool { return x[0] + x[1] >= 1 },
    func(x []float64) bool { return x[0]*x[0] + x[1]*x[1] <= 25 },
}
```

### 三目标优化

```go
objectives := []genetic.ObjectiveFunc{f1, f2, f3}
```

### 处理最大化问题

将最大化转化为最小化：
```go
// 最大化 f(x) 等价于最小化 -f(x)
func maxToMin(x []float64) float64 {
    return -originalFunction(x)
}
```

## 常见问题

### Q1: 如何选择合适的种群大小？
- 2目标：50-100
- 3目标：100-200
- 更多目标：需要更大种群

### Q2: 解的分布不均匀怎么办？
- 增加迭代次数
- 增大种群规模
- 检查目标函数尺度

### Q3: 如何验证结果正确性？
- 与已知理论前沿对比
- 检查解之间的非支配关系
- 多次运行观察稳定性

### Q4: 约束处理效果不佳？
- 确保约束函数正确实现
- 考虑使用约束违反度排序
- 增大种群多样性

## 性能优化

1. **目标函数缓存**：如果计算代价高
2. **并行评估**：独立个体可并行计算
3. **早停策略**：前沿稳定时提前终止

## 参考文献

- Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
