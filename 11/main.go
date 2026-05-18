package main

import (
	"fmt"
	"math"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func main() {
	data := mat.NewDense(5, 3, []float64{
		2.5, 2.4, 1.2,
		0.5, 0.7, 0.8,
		2.2, 2.9, 1.5,
		1.9, 2.2, 1.1,
		3.1, 3.0, 1.8,
	})

	fmt.Println("=== 数值稳定性验证 ===")
	verifySymmetry(data)

	fmt.Println("\n=== PCA降维结果 ===")
	fmt.Println("原始数据 (5x3):")
	PrintMatrix(data)

	result, err := PCA(data, 2)
	if err != nil {
		fmt.Printf("PCA执行失败: %v\n", err)
		return
	}

	fmt.Println("\n降维后的数据 (5x2):")
	PrintMatrix(result.Data)

	fmt.Println("\n解释方差比:")
	for i, ratio := range result.VarianceRatio {
		fmt.Printf("主成分%d: %.4f (%.2f%%)\n", i+1, ratio, ratio*100)
	}

	totalRatio := 0.0
	for _, ratio := range result.VarianceRatio {
		totalRatio += ratio
	}
	fmt.Printf("\n累计解释方差: %.2f%%\n", totalRatio*100)

	fmt.Println("\n=== 增量PCA (IPCA) 示例 ===")
	ipcaDemo()
}

func ipcaDemo() {
	batch1 := mat.NewDense(3, 3, []float64{
		2.5, 2.4, 1.2,
		0.5, 0.7, 0.8,
		2.2, 2.9, 1.5,
	})
	batch2 := mat.NewDense(2, 3, []float64{
		1.9, 2.2, 1.1,
		3.1, 3.0, 1.8,
	})

	ipca := NewIPCA(2)

	fmt.Println("第一批数据 (3个样本):")
	PrintMatrix(batch1)
	err := ipca.PartialFit(batch1)
	if err != nil {
		fmt.Printf("IPCA第一批拟合失败: %v\n", err)
		return
	}
	fmt.Printf("已处理样本数: %d\n", ipca.NSamples())
	reduced1, _ := ipca.Transform(batch1)
	fmt.Println("第一批降维结果:")
	PrintMatrix(reduced1)

	fmt.Println("\n第二批数据 (2个样本):")
	PrintMatrix(batch2)
	err = ipca.PartialFit(batch2)
	if err != nil {
		fmt.Printf("IPCA第二批拟合失败: %v\n", err)
		return
	}
	fmt.Printf("已处理样本数: %d\n", ipca.NSamples())

	allData := mat.NewDense(5, 3, []float64{
		2.5, 2.4, 1.2,
		0.5, 0.7, 0.8,
		2.2, 2.9, 1.5,
		1.9, 2.2, 1.1,
		3.1, 3.0, 1.8,
	})
	reducedAll, _ := ipca.Transform(allData)
	fmt.Println("\n全部数据降维结果 (增量更新后):")
	PrintMatrix(reducedAll)

	ratio := ipca.ExplainedVarianceRatio()
	fmt.Println("\n解释方差比 (增量更新后):")
	for i, r := range ratio {
		fmt.Printf("主成分%d: %.4f (%.2f%%)\n", i+1, r, r*100)
	}
}

func verifySymmetry(data *mat.Dense) {
	rows, cols := data.Dims()

	means := make([]float64, cols)
	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += data.At(i, j)
		}
		means[j] = sum / float64(rows)
	}

	centered := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			centered.Set(i, j, data.At(i, j)-means[j])
		}
	}

	covMat := mat.NewSymDense(cols, nil)
	stat.CovarianceMatrix(covMat, centered, nil)

	maxDiff := 0.0
	for i := 0; i < cols; i++ {
		for j := i + 1; j < cols; j++ {
			diff := math.Abs(covMat.At(i, j) - covMat.At(j, i))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	fmt.Printf("协方差矩阵最大不对称误差: %.2e\n", maxDiff)

	forceSymmetric(covMat)

	maxDiffAfter := 0.0
	for i := 0; i < cols; i++ {
		for j := i + 1; j < cols; j++ {
			diff := math.Abs(covMat.At(i, j) - covMat.At(j, i))
			if diff > maxDiffAfter {
				maxDiffAfter = diff
			}
		}
	}
	fmt.Printf("强制对称后不对称误差: %.2e\n", maxDiffAfter)

	var svd mat.SVD
	svd.Factorize(centered, mat.SVDThinV)
	V := mat.NewDense(cols, cols, nil)
	svd.VTo(V)

	maxOrthogError := 0.0
	for i := 0; i < cols; i++ {
		for j := i + 1; j < cols; j++ {
			dot := 0.0
			for k := 0; k < cols; k++ {
				dot += V.At(k, i) * V.At(k, j)
			}
			if math.Abs(dot) > maxOrthogError {
				maxOrthogError = math.Abs(dot)
			}
		}
	}
	fmt.Printf("SVD特征向量最大正交性误差: %.2e\n", maxOrthogError)

	orthonormalizeColumns(V)

	maxOrthogErrorAfter := 0.0
	for i := 0; i < cols; i++ {
		for j := i + 1; j < cols; j++ {
			dot := 0.0
			for k := 0; k < cols; k++ {
				dot += V.At(k, i) * V.At(k, j)
			}
			if math.Abs(dot) > maxOrthogErrorAfter {
				maxOrthogErrorAfter = math.Abs(dot)
			}
		}
	}
	fmt.Printf("Gram-Schmidt正交化后误差: %.2e\n", maxOrthogErrorAfter)
}

func PrintMatrix(m *mat.Dense) {
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Printf("%8.4f", m.At(i, j))
		}
		fmt.Println()
	}
}
