# gRPC 定积分计算器

使用 Go 开发的 gRPC 服务，提供复合辛普森法则和自适应辛普森法则计算定积分近似值。

## ✨ 重要更新：高斯-克朗罗德高精度积分与自动切换

已全面升级积分计算系统，新增高斯-克朗罗德高精度积分算法和智能自动切换机制：

### 🔬 高斯-克朗罗德积分算法 (Gauss-Kronrod)
1. **15点Gauss-Kronrod + 7点Gauss嵌套求积**
   - 使用高精度的求积节点和权重
   - 7点Gauss与15点Kronrod节点嵌套设计
   - 两个结果的差值作为误差估计

2. **自适应区间细分**
   - 栈结构管理待处理区间
   - 误差超过阈值时自动二分区间
   - 最大迭代次数保护 (1000次)

3. **内置误差估计**
   - `result, errorEstimate, err = GaussKronrod15(...)`
   - 实时监控积分精度

### 🔒 安全机制升级
1. **最大递归深度** (`DefaultMaxRecursionDepth = 1000`)
   - 深度安全保护，防止无限递归

2. **最小步长限制** (`MinIntervalWidth = 1e-10`)
   - 防止区间宽度过小导致浮点数精度问题

### 📐 曲率感知动态调整（自适应辛普森）
3. **函数曲率估计**
   - 通过二阶差分估计函数局部曲率
   - 使用多点采样提高估计精度

4. **自适应精度动态调整**
   - **高曲率区域**: 精度要求提高 10 倍
   - **低曲率区域**: 精度要求放宽 2 倍
   - **中等曲率**: 保持原始精度

### 🤖 智能自动切换机制 (Hybrid Adaptive)
5. **先验误差估计**
   - 使用辛普森法则进行快速预计算
   - 估计函数的平滑程度

6. **自动方法选择**
   - `SwitchToKronrodThreshold = 1e-4`
   - 误差小 → 使用自适应辛普森（快速）
   - 误差大 → 切换到高斯-克朗罗德（高精度）
   - 对平滑函数优先使用高效算法
   - 对困难函数自动切换到鲁棒算法

### 🚀 四种积分方法对比
| 方法 | 适用场景 | 精度 | 速度 |
|------|---------|------|------|
| **COMPOSITE_SIMPSON** | 已知平滑函数，指定步数 | 中 | ⭐⭐⭐⭐⭐ |
| **ADAPTIVE_SIMPSON** | 一般光滑函数，曲率感知 | 高 | ⭐⭐⭐⭐ |
| **GAUSS_KRONROD** | 困难函数，高精度需求 | 极高 | ⭐⭐⭐ |
| **HYBRID_AUTO** | 所有场景，自动选择 | 自适应 | ⭐⭐⭐⭐ |

## 项目结构

```
integral-grpc/
├── proto/
│   ├── integral.proto        # gRPC 服务定义
│   ├── integral.pb.go        # 生成的 Protobuf 代码
│   └── integral_grpc.pb.go   # 生成的 gRPC 代码
├── calculator/
│   ├── simpson.go            # 辛普森法则实现（含自适应）
│   └── simpson_test.go       # 单元测试（含栈溢出测试）
├── server/
│   └── main.go               # gRPC 服务端
├── client/
│   └── main.go               # gRPC 客户端
└── go.mod                    # Go 模块文件
```

## 功能特性

- 两种积分方法：
  - **复合辛普森法则** - 固定子区间数
  - **自适应辛普森法则** - 自动调整精度（含栈溢出保护）
- 接收函数表达式字符串（支持 +, -, *, /, ^ 运算符）
- 支持数学函数: sin, cos, tan, exp, log, sqrt, abs
- 自定义积分区间 [a, b]
- 自定义精度 epsilon（自适应方法）

## 环境要求

- Go 1.21+
- Protocol Buffers 编译器（可选，用于重新生成代码）

## 安装依赖

```bash
go mod download
```

## 编译和运行

### 启动服务端

```bash
cd server
go run main.go
```

服务将在 `localhost:50051` 上启动。

### 运行客户端

在另一个终端中：

```bash
cd client
# 使用复合辛普森法则（默认）
go run main.go -f "x^2" -a 0 -b 1 -method composite

# 使用自适应辛普森法则
go run main.go -f "x^2" -a 0 -b 1 -method adaptive -epsilon 1e-8

# 使用高斯-克朗罗德高精度方法
go run main.go -f "sin(100*x)" -a 0 -b 1 -method gauss-kronrod

# 混合自动模式（推荐）
go run main.go -f "1/(1+x^2)" -a 0 -b 1 -method hybrid
```

#### 客户端参数说明

- `-f`: 函数表达式，默认值为 "x^2"
- `-a`: 积分下限，默认值为 0
- `-b`: 积分上限，默认值为 1
- `-n`: 子区间数（复合方法用，必须为偶数），默认值为 100
- `-adaptive`: 使用自适应辛普森方法，默认 false
- `-epsilon`: 自适应方法的精度，默认 1e-8

#### 示例

1. 计算 ∫x² dx 从 0 到 1（自适应方法）：
```bash
go run main.go -f "x^2" -a 0 -b 1 -adaptive
```

2. 计算 ∫sin(x) dx 从 0 到 π（高精度）：
```bash
go run main.go -f "sin(x)" -a 0 -b 3.141592653589793 -adaptive -epsilon 1e-12
```

3. 计算有尖锐峰值的函数（测试栈溢出保护）：
```bash
go run main.go -f "1/(1+10000*(x-0.5)^2)" -a 0 -b 1 -adaptive -epsilon 1e-12
```

## 数值积分方法

### 复合辛普森法则

将区间 [a, b] 分成 n（偶数）个相等的子区间，每个子区间宽度 h = (b-a)/n

∫f(x)dx ≈ (h/3) * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]

### 自适应辛普森法则

自动在函数变化剧烈的区域细化网格，递归计算直到满足精度要求。**核心改进：**

1. 递归到最大深度时直接返回当前值
2. 区间宽度小于阈值时停止细分
3. 有效防止浮点数精度问题导致的无限递归

## gRPC 服务定义

```proto
enum Method {
  COMPOSITE_SIMPSON = 0;
  ADAPTIVE_SIMPSON = 1;
}

service IntegralService {
  rpc CalculateIntegral(IntegralRequest) returns (IntegralResponse);
}

message IntegralRequest {
  string function = 1;    // 函数表达式
  double a = 2;           // 积分下限
  double b = 3;           // 积分上限
  int32 n = 4;            // 子区间数（复合方法用）
  Method method = 5;      // 积分方法
  double epsilon = 6;     // 精度（自适应方法用）
}

message IntegralResponse {
  double result = 1;      // 积分结果
  string error = 2;       // 错误信息（如果有）
}
```

## 运行测试

```bash
cd calculator
go test -v
```

**包含的测试：**
- 基本函数解析测试
- 复合辛普森法则测试
- 自适应辛普森法则测试
- **栈溢出保护测试**（尖锐峰值函数、极小 epsilon）
- 最小区间宽度测试
- 自定义 epsilon 测试

## 重新生成 gRPC 代码（可选）

如果需要修改 proto 文件并重新生成代码：

```bash
# 安装 protoc 插件
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 生成代码
protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    proto/integral.proto
```

## 支持的函数表达式

- 基本运算符: `+`, `-`, `*`, `/`, `^`
- 数学函数:
  - `sin(x)` - 正弦函数
  - `cos(x)` - 余弦函数
  - `tan(x)` - 正切函数
  - `exp(x)` - 指数函数
  - `log(x)` - 自然对数
  - `sqrt(x)` - 平方根
  - `abs(x)` - 绝对值
- 括号: `()`

示例表达式:
- `x^2 + 2*x + 1`
- `sin(x) * cos(x)`
- `exp(-x^2)`
- `1/(1 + x^2)`
- `1/(1+10000*(x-0.5)^2)` (测试自适应能力)
