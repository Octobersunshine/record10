# 共轭梯度法 (CG) 求解器

使用 Go + gonum 实现的共轭梯度法求解器，用于求解对称正定线性方程组 Ax = b。

## 特性

- 基于 gonum 矩阵库实现
- 支持自定义最大迭代次数和收敛容差
- **数值稳定性修复**：定期重新计算残差以避免舍入误差累积
- **自动预条件子选择**：根据矩阵特性智能选择最优预条件子
- 链式调用 API 设计
- 返回解向量和实际迭代次数

## 项目结构

```
.
├── go.mod              # Go 模块定义
├── cgsolver/
│   └── cg_solver.go   # CG 求解器实现
├── main.go             # 使用示例
└── README.md           # 说明文档
```

## API 说明

### 创建求解器

```go
cg := cgsolver.New()
```

### 配置求解器（链式调用）

```go
cg := cgsolver.New().
    WithMaxIter(1000).                  // 设置最大迭代次数
    WithTolerance(1e-10).               // 设置收敛容差
    WithReorthogonalizeFreq(10).        // 设置残差重计算频率（数值稳定性）
    WithAutoPreconditioner(true).       // 启用自动预条件子选择（默认开启）
    WithPreconditioner(precond)         // 手动指定预条件子
```

### 求解方程组

```go
x, iterations, err := cg.Solve(A, b)
```

**参数：**
- `A mat.Matrix`: 对称正定矩阵
- `b mat.Vector`: 右端项向量

**返回值：**
- `x *mat.VecDense`: 解向量
- `iterations int`: 实际迭代次数
- `err error`: 错误信息

### 矩阵分析

```go
analysis := cgsolver.AnalyzeMatrix(A)
fmt.Printf("对角占优: %v\n", analysis.IsDiagonallyDominant)
fmt.Printf("对角强度: %.2f%%\n", analysis.DiagonalStrength*100)
fmt.Printf("稀疏度: %.2f%%\n", analysis.Sparsity*100)
fmt.Printf("条件数估计: %.2f\n", analysis.ConditionEstimate)
```

### 预条件子

**内置预条件子：**

1. **Jacobi (对角) 预条件子**：
   - 适用于对角占优矩阵
   - 计算代价低：O(n)
   - 使用：`cgsolver.NewJacobiPreconditioner(A)`

2. **ILU (不完全LU分解) 预条件子**：
   - 适用于一般稠密或带状矩阵
   - 计算代价较高：O(n²)
   - 使用：`cgsolver.NewILUPreconditioner(A)`

3. **无预条件**：
   - 标准 CG 算法
   - 使用：`&cgsolver.NoPreconditioner{}`

**自定义预条件子：**

实现 `Preconditioner` 接口即可：
```go
type Preconditioner interface {
    Apply(z, r mat.Vector)  // z = M⁻¹ * r
}
```

## 数值稳定性修复

### 问题

标准共轭梯度法中，残差使用递推公式更新：
```
r = r - α * Ap
```

由于浮点数舍入误差的累积，残差会逐渐失去正交性，导致：
- 收敛速度变慢
- 甚至不收敛
- 解的精度下降

### 解决方案

定期重新计算精确残差：
```
r = b - Ax
```

通过 `WithReorthogonalizeFreq(n)` 设置每 n 次迭代重新计算一次残差，默认每 10 次迭代重计算一次。

### 效果

- 显著提高数值稳定性
- 保证残差的正交性
- 确保可靠收敛

## 自动预条件子选择

### 选择策略

求解器会根据以下矩阵特性自动选择最优预条件子：

1. **对角占优检测**：如果矩阵对角占优或对角强度 > 70%，使用 Jacobi 预条件子
2. **规模检测**：如果 n ≤ 200 或稀疏度 > 30%，使用 ILU 预条件子
3. **默认**：对于大型稀疏矩阵，默认使用 Jacobi 预条件子

### 预条件效果对比

| 预条件 | 迭代次数减少 | 额外计算量 | 适用场景 |
|--------|-------------|-----------|---------|
| 无     | 0%          | 0         | 小规模良态矩阵 |
| Jacobi | 30-50%      | O(n)      | 对角占优矩阵 |
| ILU    | 60-80%      | O(n²)     | 一般稠密/病态矩阵 |

## 使用示例

```go
package main

import (
    "fmt"
    "cg_solver/cgsolver"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // 创建对称正定矩阵 A
    n := 10
    data := make([]float64, n*n)
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            if i == j {
                data[i*n+j] = 4.0
            } else if abs(i-j) == 1 {
                data[i*n+j] = 1.0
            }
        }
    }
    A := mat.NewSymDense(n, data)

    // 创建右端项 b
    bData := make([]float64, n)
    for i := 0; i < n; i++ {
        bData[i] = float64(i + 1)
    }
    b := mat.NewVecDense(n, bData)

    // 分析矩阵特性
    analysis := cgsolver.AnalyzeMatrix(A)
    fmt.Printf("对角占优: %v\n", analysis.IsDiagonallyDominant)
    fmt.Printf("对角强度: %.2f%%\n", analysis.DiagonalStrength*100)

    // 创建求解器并求解（自动选择预条件子）
    cg := cgsolver.New().
        WithMaxIter(1000).
        WithTolerance(1e-10).
        WithReorthogonalizeFreq(10)

    x, iter, err := cg.Solve(A, b)

    if err != nil {
        fmt.Printf("求解错误: %v\n", err)
    } else {
        fmt.Printf("迭代次数: %d\n", iter)
        fmt.Printf("使用预条件: %s\n", cg.SelectedPrecond)
    }
}
```

## 运行程序

```bash
go mod tidy
go run main.go
```

## 算法原理

预条件共轭梯度法 (PCG) 通过引入预条件子 M，求解等价系统：
```
M⁻¹Ax = M⁻¹b
```

使得新的系数矩阵 M⁻¹A 具有更好的条件数，从而加速收敛。

**算法步骤：**
1. 初始残差 r₀ = b - Ax₀
2. 预条件残差 z₀ = M⁻¹r₀
3. 搜索方向 p₀ = z₀
4. 对于 k = 0, 1, 2, ...:
   - αₖ = (rₖ, zₖ) / (pₖ, Apₖ)
   - xₖ₊₁ = xₖ + αₖpₖ
   - rₖ₊₁ = rₖ - αₖApₖ
   - （定期重算: rₖ₊₁ = b - Axₖ₊₁）
   - zₖ₊₁ = M⁻¹rₖ₊₁
   - βₖ = (rₖ₊₁, zₖ₊₁) / (rₖ, zₖ)
   - pₖ₊₁ = zₖ₊₁ + βₖpₖ
   - 收敛检查: ||rₖ₊₁|| < tol
