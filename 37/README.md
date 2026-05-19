# 一维热传导方程 FTCS 求解器（改进版）

用Go语言实现的显式有限差分法（FTCS）求解一维热传导方程，包含**二阶精度 Neumann 边界条件**和**热流守恒验证**。

## 修复内容

### 问题分析

原实现存在的问题：
1. 仅支持 Dirichlet 边界条件（固定温度）
2. Neumann 边界一阶近似精度为 O(dx)，导致热流不守恒
3. 缺少热流守恒验证机制

### 解决方案

**二阶精度 Neumann 边界条件（镜像点法）**：
- 左边界：`u[-1] = u[1] - 2*dx*q_left`
- 右边界：`u[N] = u[N-2] + 2*dx*q_right`
- 精度：O(dx²)，优于一阶近似 O(dx)

## 数学模型

一维热传导方程：
```
∂u/∂t = α ∂²u/∂x²
```

### 边界条件类型

1. **Dirichlet（第一类边界条件）**：固定温度
   - `u(0, t) = T_left`
   - `u(L, t) = T_right`

2. **Neumann（第二类边界条件）**：固定热流
   - `∂u/∂x(0, t) = q_left`
   - `∂u/∂x(L, t) = q_right`
   - 绝热边界：`q = 0`

## FTCS 差分格式

```
u(i, n+1) = u(i, n) + r * [u(i+1, n) - 2u(i, n) + u(i-1, n)]
```

其中傅里叶数 `r = α * dt / dx²`

**稳定性条件**: r ≤ 0.5

## 代码结构

### 数据结构

```go
type BoundaryType int

const (
    Dirichlet BoundaryType = iota
    Neumann
)

type BoundaryCondition struct {
    Type       BoundaryType  // 边界类型: Dirichlet 或 Neumann
    LeftValue  float64       // 左边界值
    RightValue float64       // 右边界值
}

type SimulationParams struct {
    Length      float64          // 杆长 (m)
    Time        float64          // 总模拟时间 (s)
    Dx          float64          // 空间步长 (m)
    Dt          float64          // 时间步长 (s)
    Alpha       float64          // 热扩散系数 (m²/s)
    Boundary    BoundaryCondition // 边界条件
    InitialTemp []float64        // 初始温度分布
}
```

### 核心函数

```go
func FTCS(params SimulationParams) ([][]float64, error)
```

- 支持 Dirichlet 和 Neumann 两种边界条件
- Neumann 边界采用二阶精度镜像点法
- 自动检查稳定性条件（r ≤ 0.5）
- 参数验证（初始条件长度匹配）

### 辅助函数

```go
// 计算总热量
func CalculateTotalHeat(temperature []float64, dx float64) float64

// 计算边界热流
func CalculateBoundaryHeatFlux(temperature []float64, dx, alpha float64, boundary BoundaryCondition) (float64, float64)

// 验证热流守恒
func VerifyHeatConservation(temperature [][]float64, params SimulationParams) []float64
```

### 初始条件函数

- `InitialConditionUniform(x, L)` - 均匀温度分布
- `InitialConditionSin(x, L)` - 正弦分布
- `InitialConditionStep(x, L)` - 阶跃分布
- `InitialConditionGaussian(x, L)` - 高斯分布

## 使用方法

### 1. 安装Go环境

下载并安装 Go 1.16+：https://golang.org/dl/

### 2. 运行程序

```bash
go run ftcs.go
```

### 3. 自定义边界条件

#### Dirichlet 边界（固定温度）

```go
params := SimulationParams{
    // ...
    Boundary: BoundaryCondition{
        Type:       Dirichlet,
        LeftValue:  0,    // 左边界温度 0°C
        RightValue: 0,    // 右边界温度 0°C
    },
    // ...
}
```

#### Neumann 边界（固定热流）

```go
params := SimulationParams{
    // ...
    Boundary: BoundaryCondition{
        Type:       Neumann,
        LeftValue:  0,    // 左边界热流 0（绝热）
        RightValue: 0,    // 右边界热流 0（绝热）
    },
    // ...
}
```

## 输出示例

```
=== 一维热传导方程 FTCS 求解器（改进版）===
包含二阶精度 Neumann 边界条件和热流守恒验证

测试1: Neumann 边界条件（绝热边界，热流应为0）
边界条件：左右热流 = 0

参数: L=1.00m, T=0.0500s, dx=0.0100m, dt=0.000050s, alpha=0.0100 m²/s
空间点数: 101, 时间步数: 1001

热流守恒验证 (绝热边界下总热量应恒定):
  初始总热量: 25.066161
  最终总热量: 25.066161
  相对误差:   0.000000%

温度分布:
  时刻 t = 0.0000s:
    x=0.00m: 1.3888°C
    x=0.10m: 13.5335°C
    x=0.20m: 60.6531°C
    ...
```

## Neumann 边界实现细节

### 一阶近似（旧方法，精度 O(dx)）

```go
// 左边界一阶近似（错误，精度不足）
temperature[n+1][0] = temperature[n][1] + dx * q_left
```

问题：
- 精度只有一阶 O(dx)
- 边界处热流误差累积
- 绝热边界下总热量不守恒

### 二阶精度镜像点法（新方法，精度 O(dx²)）

```go
// 左边界：引入镜像点 u[-1]
uLeftGhost := temperature[n][1] - 2*dx*q_left
temperature[n+1][0] = temperature[n][0] +
    r*(temperature[n][1] - 2*temperature[n][0] + uLeftGhost)

// 右边界：引入镜像点 u[N]
uRightGhost := temperature[n][nx-2] + 2*dx*q_right
temperature[n+1][nx-1] = temperature[n][nx-1] +
    r*(uRightGhost - 2*temperature[n][nx-1] + temperature[n][nx-2])
```

优势：
- 精度达到二阶 O(dx²)
- 与内部点格式一致
- 绝热边界下热流严格守恒
- 相对误差 < 0.001%

## 热流守恒验证

绝热边界条件下（q=0），系统总热量应保持恒定：

```go
heatHistory := VerifyHeatConservation(temperature, params)
initialHeat := heatHistory[0]
finalHeat := heatHistory[len(heatHistory)-1]
relativeError := math.Abs(finalHeat - initialHeat) / initialHeat * 100
```

验证标准：相对误差 < 0.1% 表明热流守恒良好

## 注意事项

1. **稳定性条件**：确保 `r = α*dt/dx² ≤ 0.5`
2. **Neumann 边界单位**：热流值为 `∂u/∂x`（°C/m），不是热流密度
3. **镜像点法**：Neumann 边界需要在边界外引入虚拟点，与内部格式统一
4. **热流符号**：
   - 正热流：热量流入（温度梯度为正）
   - 负热流：热量流出（温度梯度为负）

## 自适应网格细化 (AMR)

新增文件 `ftcs_amr.go` 实现了完整的自适应网格细化功能。

### 核心数据结构

```go
type Cell struct {
    x           float64  // 位置
    dx          float64  // 网格步长
    temperature float64  // 温度
    level       int      // 网格层次 (0=最粗)
    marked      bool     // 细化标记
}

type AMRGrid struct {
    cells         []Cell
    maxLevel      int     // 最大细化层次
    refineThresh  float64 // 细化阈值
    coarsenThresh float64 // 粗化阈值
    alpha         float64 // 热扩散系数
}
```

### AMR 工作流程

**1. 梯度检测 (ComputeGradients)**
- 计算每个单元格的温度梯度 |∂u/∂x|
- 中心差分格式：`(u[i+1] - u[i-1]) / (2*dx)`

**2. 标记算法 (MarkForRefinement)**
```go
if 梯度 > 细化阈值 && level < maxLevel:
    标记为需要细化
elif 梯度 < 粗化阈值 && level > 0:
    标记为可以粗化
```

**3. 网格细化 (RefineGrid)**
- 标记单元格一分为二
- 新节点温度用线性插值计算
- 网格层次 level+1
- 步长 dx = dx_old / 2

**4. 网格粗化 (CoarsenGrid)**
- 相邻两个同层次单元格合并
- 温度取平均值
- 网格层次 level-1
- 步长 dx = dx_old * 2

**5. 时间步长自适应 (ComputeTimeStep)**
```go
dt = 0.4 * min(dx)² / α
```
- 自动根据最细网格调整
- 满足稳定性条件 r ≤ 0.5

### 使用方法

```go
params := AMRParams{
    Length:        1.0,       // 杆长
    Time:          0.02,      // 总时间
    InitialDx:     0.05,      // 初始步长
    MaxLevel:      3,         // 最大细化层次 (0-3)
    Alpha:         0.01,      // 热扩散系数
    RefineThresh:  80.0,      // 细化阈值 (°C/m)
    CoarsenThresh: 20.0,      // 粗化阈值
    Boundary: BoundaryCondition{
        Type: Dirichlet,
        LeftValue: 0,
        RightValue: 0,
    },
    InitialFunc: func(x, L float64) float64 {
        // 自定义初始条件
        if x > 0.3 && x < 0.7 {
            return 100.0
        }
        return 0.0
    },
}

grid, err := FTCS_AMR(params)
```

### 输出示例

```
t=0.0000s | 单元格数: 21 | 网格层次: L0:21 | dx范围: 0.050000m
t=0.0013s | 单元格数: 33 | 网格层次: L0:9 L1:24 | dx范围: 0.025000-0.050000m
t=0.0025s | 单元格数: 41 | 网格层次: L0:5 L1:16 L2:20 | dx范围: 0.012500-0.050000m
```

### 性能优势

| 方案 | 网格数 (均匀L3) | 网格数 (AMR) | 节省比例 |
|------|-----------------|--------------|----------|
| 方波 | 161             | ~40-60       | 60-75%   |
| 高斯 | 161             | ~30-50       | 70-80%   |

AMR 自动在梯度大区域加密（如方波边缘、高斯脉冲边缘），在梯度小区域保持粗网格，显著节省计算资源。

### 非均匀网格差分格式

对于相邻单元格步长不同的情况，使用守恒型差分：
```go
dxLeft = x[i] - x[i-1]
dxRight = x[i+1] - x[i]
dxAvg = (dxLeft + dxRight) / 2

gradRight = (u[i+1] - u[i]) / dxRight
gradLeft = (u[i] - u[i-1]) / dxLeft

∂²u/∂x² = (gradRight - gradLeft) / dxAvg
u_new = u + α * dt * ∂²u/∂x²
```

## 扩展功能建议

- [x] 二阶精度 Neumann 边界条件
- [x] 热流守恒验证
- [x] 自适应网格细化 (AMR)
- [ ] Robin 边界条件（对流换热）
- [ ] Crank-Nicolson 隐式格式
- [ ] CSV 数据导出
- [ ] 可变热扩散系数
- [ ] 解析解对比验证
- [ ] 并行计算优化
- [ ] 更高阶插值（二次/三次）
- [ ] 通量守恒修正
- [ ] 网格缓冲区域处理
