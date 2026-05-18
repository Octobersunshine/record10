package main

import (
	"fmt"

	"cg_solver/cgsolver"

	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== 共轭梯度法求解器 (CG) 示例 ===")
	fmt.Println()

	n := 10
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				data[i*n+j] = 4.0
			} else if abs(i-j) == 1 {
				data[i*n+j] = 1.0
			} else {
				data[i*n+j] = 0.0
			}
		}
	}
	A := mat.NewSymDense(n, data)

	bData := make([]float64, n)
	for i := 0; i < n; i++ {
		bData[i] = float64(i + 1)
	}
	b := mat.NewVecDense(n, bData)

	analysis := cgsolver.AnalyzeMatrix(A)
	fmt.Println("=== 矩阵分析 ===")
	fmt.Printf("矩阵规模: %d x %d\n", n, n)
	fmt.Printf("对角占优: %v\n", analysis.IsDiagonallyDominant)
	fmt.Printf("对角强度: %.2f%%\n", analysis.DiagonalStrength*100)
	fmt.Printf("稀疏度: %.2f%%\n", analysis.Sparsity*100)
	fmt.Printf("条件数估计: %.2f\n", analysis.ConditionEstimate)
	fmt.Println()

	fmt.Println("=== 自动选择预条件子求解 ===")
	cg := cgsolver.New().
		WithMaxIter(1000).
		WithTolerance(1e-10).
		WithReorthogonalizeFreq(10)

	x, iter, err := cg.Solve(A, b)

	if err != nil {
		fmt.Printf("求解错误: %v\n", err)
	} else {
		fmt.Printf("✓ 成功收敛！\n")
		fmt.Printf("使用预条件: %s\n", cg.SelectedPrecond)
		fmt.Printf("迭代次数: %d\n", iter)

		Ax := mat.NewVecDense(n, nil)
		Ax.MulVec(A, x)
		residual := mat.NewVecDense(n, nil)
		residual.SubVec(Ax, b)
		residualNorm := mat.Norm(residual, 2)
		fmt.Printf("残差范数 ||Ax - b||: %.2e\n", residualNorm)
	}
	fmt.Println()

	fmt.Println("=== 不同预条件子对比测试 ===")
	testPreconditioners(A, b)
	fmt.Println()

	fmt.Println("=== 不同规模矩阵测试 ===")
	testSizes := []int{10, 50, 100}
	for _, size := range testSizes {
		testLargeMatrix(size)
	}
	fmt.Println()

	fmt.Println("=== 病态矩阵测试 (非对角占优) ===")
	testIllConditionedMatrix()
}

func testPreconditioners(A mat.Matrix, b mat.Vector) {
	n, _ := A.Dims()
	
	preconds := []struct {
		name string
		p    cgsolver.Preconditioner
	}{
		{"无预条件", &cgsolver.NoPreconditioner{}},
		{"Jacobi", cgsolver.NewJacobiPreconditioner(A)},
		{"ILU", cgsolver.NewILUPreconditioner(A)},
	}

	fmt.Printf("%-15s %-10s %-15s\n", "预条件", "迭代次数", "残差范数")
	fmt.Println("----------------------------------------")
	
	for _, pc := range preconds {
		cg := cgsolver.New().
			WithMaxIter(1000).
			WithTolerance(1e-10).
			WithPreconditioner(pc.p)
		
		x, iter, err := cg.Solve(A, b)
		if err != nil {
			fmt.Printf("%-15s %-10s %-15s\n", pc.name, "未收敛", "-")
		} else {
			Ax := mat.NewVecDense(n, nil)
			Ax.MulVec(A, x)
			residual := mat.NewVecDense(n, nil)
			residual.SubVec(Ax, b)
			residualNorm := mat.Norm(residual, 2)
			fmt.Printf("%-15s %-10d %-15.2e\n", pc.name, iter, residualNorm)
		}
	}
}

func testLargeMatrix(n int) {
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				data[i*n+j] = float64(n)
			} else if abs(i-j) <= 2 {
				data[i*n+j] = 1.0
			} else {
				data[i*n+j] = 0.0
			}
		}
	}
	A := mat.NewSymDense(n, data)

	bData := make([]float64, n)
	for i := 0; i < n; i++ {
		bData[i] = float64(i + 1)
	}
	b := mat.NewVecDense(n, bData)

	cg := cgsolver.New()
	x, iter, err := cg.Solve(A, b)

	if err != nil {
		fmt.Printf("n=%4d: 错误 - %v\n", n, err)
	} else {
		Ax := mat.NewVecDense(n, nil)
		Ax.MulVec(A, x)
		residual := mat.NewVecDense(n, nil)
		residual.SubVec(Ax, b)
		residualNorm := mat.Norm(residual, 2)
		fmt.Printf("n=%4d: 迭代 %3d 次, 预条件: %s, 残差 %.2e\n", 
			n, iter, cg.SelectedPrecond, residualNorm)
	}
}

func testIllConditionedMatrix() {
	n := 20
	data := make([]float64, n*n)
	
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				data[i*n+j] = 100.0
			} else {
				data[i*n+j] = 99.0
			}
		}
	}
	A := mat.NewSymDense(n, data)

	analysis := cgsolver.AnalyzeMatrix(A)
	fmt.Printf("矩阵规模: %d x %d\n", n, n)
	fmt.Printf("对角占优: %v\n", analysis.IsDiagonallyDominant)
	fmt.Printf("对角强度: %.2f%%\n", analysis.DiagonalStrength*100)
	fmt.Printf("条件数估计: %.2f\n", analysis.ConditionEstimate)

	bData := make([]float64, n)
	for i := 0; i < n; i++ {
		bData[i] = float64(i + 1)
	}
	b := mat.NewVecDense(n, bData)

	cg := cgsolver.New()
	x, iter, err := cg.Solve(A, b)

	if err != nil {
		fmt.Printf("求解错误: %v\n", err)
	} else {
		Ax := mat.NewVecDense(n, nil)
		Ax.MulVec(A, x)
		residual := mat.NewVecDense(n, nil)
		residual.SubVec(Ax, b)
		residualNorm := mat.Norm(residual, 2)
		fmt.Printf("自动选择: %s, 迭代 %d 次, 残差 %.2e\n", 
			cg.SelectedPrecond, iter, residualNorm)
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
