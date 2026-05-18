# Go + gonum PCA降维实现

使用Go语言和gonum库实现的主成分分析(PCA)和增量PCA(IPCA)降维功能。

## 功能特性

- **批量PCA**: 接收数据矩阵执行标准PCA降维
- **增量PCA (IPCA)**: 支持在线数据更新，无需重新计算全部数据
- 数值稳定性优化：协方差矩阵强制对称 + SVD分解 + Gram-Schmidt正交化
- 返回降维后的数据矩阵和解释方差比

## 安装依赖

```bash
go mod tidy
```

## 使用方法

### 批量PCA

```go
import "gonum.org/v1/gonum/mat"

// 创建数据矩阵
data := mat.NewDense(rows, cols, dataSlice)

// 执行PCA降维到2维
result, err := PCA(data, 2)
if err != nil {
    // 处理错误
}

// 访问降维后的数据
reducedData := result.Data

// 访问解释方差比
varianceRatios := result.VarianceRatio
```

### 增量PCA (IPCA)

```go
// 创建IPCA实例，目标维度2
ipca := NewIPCA(2)

// 第一批数据
err := ipca.PartialFit(batch1)
reduced1, _ := ipca.Transform(batch1)

// 第二批数据（无需重新计算第一批）
err = ipca.PartialFit(batch2)
reduced2, _ := ipca.Transform(batch2)

// 获取模型信息
nSamples := ipca.NSamples()       // 已处理样本数
components := ipca.Components()   // 主成分向量
mean := ipca.Mean()               // 均值向量
ratio := ipca.ExplainedVarianceRatio()  // 解释方差比
```

## API说明

### PCA函数

```go
func PCA(data *mat.Dense, targetDim int) (*PCAResult, error)
```

**参数:**
- `data`: 输入数据矩阵，每行一个样本，每列一个特征
- `targetDim`: 目标降维维度，必须大于0且小于等于原数据维度

**返回值:**
- `PCAResult.Data`: 降维后的数据矩阵
- `PCAResult.VarianceRatio`: 各主成分的解释方差比数组

### IPCA 方法

```go
func NewIPCA(nComponents int) *IPCA              // 创建IPCA实例
func (ipca *IPCA) PartialFit(X *mat.Dense) error // 增量拟合
func (ipca *IPCA) Transform(X *mat.Dense) (*mat.Dense, error) // 数据转换
func (ipca *IPCA) FitTransform(X *mat.Dense) (*mat.Dense, error) // 拟合并转换
func (ipca *IPCA) ExplainedVarianceRatio() []float64 // 解释方差比
func (ipca *IPCA) NSamples() int                 // 获取已处理样本数
func (ipca *IPCA) Components() *mat.Dense        // 获取主成分向量
func (ipca *IPCA) Mean() []float64               // 获取均值向量
```

## 运行示例

```bash
go run .
```

## 实现原理

### 批量PCA

1. **数据中心化**: 减去各特征的均值
2. **计算协方差矩阵**
3. **SVD分解**: 替代特征值分解，数值更稳定
4. **选择主成分**: 选取前d个奇异向量
5. **正交化修正**: Gram-Schmidt确保特征向量正交性
6. **投影**: 将原始数据投影到新的子空间

### 增量PCA (IPCA)

1. **增量均值更新**: Welford算法在线更新均值
2. **残差计算**: 计算新数据在当前子空间的投影残差
3. **子空间更新**: 将残差向量加入基向量并重新正交化
4. **方差缩放**: 按样本数比例缩放解释方差

## 数值稳定性优化

- 协方差矩阵强制对称：`forceSymmetric()`
- SVD分解替代特征值分解
- Gram-Schmidt正交化：`orthonormalizeColumns()`
- 浮点精度阈值：`epsilon = 1e-12`
