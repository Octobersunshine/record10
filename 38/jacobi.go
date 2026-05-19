package main

import (
	"fmt"
	"math"
)

type MethodType string

const (
	MethodJacobi      MethodType = "Jacobi"
	MethodGaussSeidel MethodType = "Gauss-Seidel"
	MethodSOR         MethodType = "SOR"
)

type MatrixProperties struct {
	IsDiagonallyDominant bool
	IsSymmetric          bool
	IsSymmetricPositive  bool
	Sparsity             float64
	RecommendedMethod    MethodType
	Omega                float64
}

func IsDiagonallyDominant(A [][]float64) bool {
	n := len(A)
	for i := 0; i < n; i++ {
		diag := math.Abs(A[i][i])
		sum := 0.0
		for j := 0; j < n; j++ {
			if i != j {
				sum += math.Abs(A[i][j])
			}
		}
		if diag < sum {
			return false
		}
	}
	return true
}

func IsSymmetric(A [][]float64) bool {
	n := len(A)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if math.Abs(A[i][j]-A[j][i]) > 1e-10 {
				return false
			}
		}
	}
	return true
}

func CalculateSparsity(A [][]float64) float64 {
	n := len(A)
	total := n * n
	nonZero := 0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if math.Abs(A[i][j]) > 1e-10 {
				nonZero++
			}
		}
	}
	return 1.0 - float64(nonZero)/float64(total)
}

func IsSymmetricPositiveDefinite(A [][]float64) bool {
	if !IsSymmetric(A) {
		return false
	}

	n := len(A)
	L := make([][]float64, n)
	for i := range L {
		L[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := 0.0
			for k := 0; k < j; k++ {
				sum += L[i][k] * L[j][k]
			}
			if i == j {
				diag := A[i][i] - sum
				if diag <= 0 {
					return false
				}
				L[i][j] = math.Sqrt(diag)
			} else {
				L[i][j] = (A[i][j] - sum) / L[j][j]
			}
		}
	}
	return true
}

func EstimateOmega(A [][]float64) float64 {
	if !IsSymmetric(A) {
		return 1.4
	}
	if IsSymmetricPositiveDefinite(A) {
		if IsDiagonallyDominant(A) {
			return 1.2
		}
		return 1.5
	}
	if IsDiagonallyDominant(A) {
		return 1.1
	}
	return 1.0
}

func AnalyzeMatrix(A [][]float64) MatrixProperties {
	props := MatrixProperties{}
	props.IsDiagonallyDominant = IsDiagonallyDominant(A)
	props.IsSymmetric = IsSymmetric(A)
	props.IsSymmetricPositive = IsSymmetricPositiveDefinite(A)
	props.Sparsity = CalculateSparsity(A)

	if props.IsSymmetricPositive {
		props.RecommendedMethod = MethodSOR
		props.Omega = EstimateOmega(A)
	} else if props.IsDiagonallyDominant {
		if props.IsSymmetric {
			props.RecommendedMethod = MethodSOR
			props.Omega = EstimateOmega(A)
		} else {
			props.RecommendedMethod = MethodGaussSeidel
		}
	} else if props.Sparsity > 0.6 {
		props.RecommendedMethod = MethodJacobi
	} else {
		props.RecommendedMethod = MethodGaussSeidel
	}

	return props
}

func Jacobi(A [][]float64, b []float64, tol float64, maxIter int) ([]float64, int, error) {
	n := len(b)
	if len(A) != n || len(A[0]) != n {
		return nil, 0, fmt.Errorf("matrix dimensions do not match vector size")
	}

	x := make([]float64, n)
	xNew := make([]float64, n)

	bNorm := 0.0
	for i := 0; i < n; i++ {
		bNorm += b[i] * b[i]
	}
	bNorm = math.Sqrt(bNorm)

	for iter := 0; iter < maxIter; iter++ {
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				if i != j {
					sum += A[i][j] * x[j]
				}
			}
			xNew[i] = (b[i] - sum) / A[i][i]
		}

		copy(x, xNew)

		residualNorm := 0.0
		for i := 0; i < n; i++ {
			Ax := 0.0
			for j := 0; j < n; j++ {
				Ax += A[i][j] * x[j]
			}
			residual := b[i] - Ax
			residualNorm += residual * residual
		}
		residualNorm = math.Sqrt(residualNorm)

		relativeResidual := residualNorm / bNorm

		if relativeResidual < tol {
			return x, iter + 1, nil
		}
	}

	return x, maxIter, fmt.Errorf("did not converge within %d iterations", maxIter)
}

func GaussSeidel(A [][]float64, b []float64, tol float64, maxIter int) ([]float64, int, error) {
	n := len(b)
	if len(A) != n || len(A[0]) != n {
		return nil, 0, fmt.Errorf("matrix dimensions do not match vector size")
	}

	x := make([]float64, n)

	bNorm := 0.0
	for i := 0; i < n; i++ {
		bNorm += b[i] * b[i]
	}
	bNorm = math.Sqrt(bNorm)

	for iter := 0; iter < maxIter; iter++ {
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < i; j++ {
				sum += A[i][j] * x[j]
			}
			for j := i + 1; j < n; j++ {
				sum += A[i][j] * x[j]
			}
			x[i] = (b[i] - sum) / A[i][i]
		}

		residualNorm := 0.0
		for i := 0; i < n; i++ {
			Ax := 0.0
			for j := 0; j < n; j++ {
				Ax += A[i][j] * x[j]
			}
			residual := b[i] - Ax
			residualNorm += residual * residual
		}
		residualNorm = math.Sqrt(residualNorm)

		relativeResidual := residualNorm / bNorm

		if relativeResidual < tol {
			return x, iter + 1, nil
		}
	}

	return x, maxIter, fmt.Errorf("did not converge within %d iterations", maxIter)
}

func SOR(A [][]float64, b []float64, omega float64, tol float64, maxIter int) ([]float64, int, error) {
	n := len(b)
	if len(A) != n || len(A[0]) != n {
		return nil, 0, fmt.Errorf("matrix dimensions do not match vector size")
	}

	x := make([]float64, n)

	bNorm := 0.0
	for i := 0; i < n; i++ {
		bNorm += b[i] * b[i]
	}
	bNorm = math.Sqrt(bNorm)

	for iter := 0; iter < maxIter; iter++ {
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < i; j++ {
				sum += A[i][j] * x[j]
			}
			for j := i + 1; j < n; j++ {
				sum += A[i][j] * x[j]
			}
			x[i] = (1-omega)*x[i] + omega*(b[i]-sum)/A[i][i]
		}

		residualNorm := 0.0
		for i := 0; i < n; i++ {
			Ax := 0.0
			for j := 0; j < n; j++ {
				Ax += A[i][j] * x[j]
			}
			residual := b[i] - Ax
			residualNorm += residual * residual
		}
		residualNorm = math.Sqrt(residualNorm)

		relativeResidual := residualNorm / bNorm

		if relativeResidual < tol {
			return x, iter + 1, nil
		}
	}

	return x, maxIter, fmt.Errorf("did not converge within %d iterations", maxIter)
}

func Solve(A [][]float64, b []float64, tol float64, maxIter int) ([]float64, int, MethodType, error) {
	props := AnalyzeMatrix(A)

	var x []float64
	var iter int
	var err error

	switch props.RecommendedMethod {
	case MethodSOR:
		x, iter, err = SOR(A, b, props.Omega, tol, maxIter)
	case MethodGaussSeidel:
		x, iter, err = GaussSeidel(A, b, tol, maxIter)
	default:
		x, iter, err = Jacobi(A, b, tol, maxIter)
	}

	return x, iter, props.RecommendedMethod, err
}

func printMatrixProperties(props MatrixProperties) {
	fmt.Println("=== 矩阵性质分析 ===")
	fmt.Printf("  对角占优: %v\n", props.IsDiagonallyDominant)
	fmt.Printf("  对称: %v\n", props.IsSymmetric)
	fmt.Printf("  对称正定: %v\n", props.IsSymmetricPositive)
	fmt.Printf("  稀疏度: %.2f%%\n", props.Sparsity*100)
	fmt.Printf("  推荐方法: %s\n", props.RecommendedMethod)
	if props.RecommendedMethod == MethodSOR {
		fmt.Printf("  SOR松弛因子 ω: %.2f\n", props.Omega)
	}
	fmt.Println()
}

func main() {
	A := [][]float64{
		{4, 1, 1},
		{1, 5, 2},
		{1, 2, 4},
	}

	b := []float64{6, 9, 11}

	tol := 1e-6
	maxIter := 100

	props := AnalyzeMatrix(A)
	printMatrixProperties(props)

	x, iter, method, err := Solve(A, b, tol, maxIter)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("使用 %s 方法收敛于 %d 次迭代\n", method, iter)
	fmt.Printf("Solution: x = %.6f\n", x)

	fmt.Println("\n验证解 Ax = b:")
	for i := 0; i < len(A); i++ {
		Ax := 0.0
		for j := 0; j < len(A[i]); j++ {
			Ax += A[i][j] * x[j]
		}
		fmt.Printf("  Equation %d: Ax = %.6f, b = %.6f, residual = %.6e\n", i+1, Ax, b[i], b[i]-Ax)
	}

	fmt.Println("\n=== 三种方法对比测试 ===")
	testMatrices := []struct {
		name string
		A    [][]float64
		b    []float64
	}{
		{
			name: "对称正定矩阵",
			A: [][]float64{
				{4, 1, 1},
				{1, 5, 2},
				{1, 2, 4},
			},
			b: []float64{6, 9, 11},
		},
		{
			name: "对角占优非对称矩阵",
			A: [][]float64{
				{10, 1, 2},
				{3, 10, 4},
				{1, 2, 10},
			},
			b: []float64{13, 17, 13},
		},
		{
			name: "三对角稀疏矩阵",
			A: [][]float64{
				{4, 1, 0, 0},
				{1, 4, 1, 0},
				{0, 1, 4, 1},
				{0, 0, 1, 4},
			},
			b: []float64{5, 6, 6, 5},
		},
		{
			name: "非对角占优稠密矩阵",
			A: [][]float64{
				{5, 3, 2},
				{3, 6, 2},
				{2, 2, 5},
			},
			b: []float64{10, 11, 9},
		},
	}

	for _, test := range testMatrices {
		fmt.Printf("\n--- %s ---\n", test.name)
		props := AnalyzeMatrix(test.A)
		printMatrixProperties(props)

		methods := []struct {
			name string
			fn   func([][]float64, []float64, float64, int) ([]float64, int, error)
		}{
			{"Jacobi", Jacobi},
			{"Gauss-Seidel", GaussSeidel},
		}

		for _, m := range methods {
			x, iter, err := m.fn(test.A, test.b, tol, maxIter)
			if err != nil {
				fmt.Printf("  %-12s: %v\n", m.name, err)
			} else {
				fmt.Printf("  %-12s: %3d 次迭代, x = %.4f\n", m.name, iter, x)
			}
		}

		xSOR, iterSOR, errSOR := SOR(test.A, test.b, 1.2, tol, maxIter)
		if errSOR != nil {
			fmt.Printf("  %-12s: %v\n", "SOR(ω=1.2)", errSOR)
		} else {
			fmt.Printf("  %-12s: %3d 次迭代, x = %.4f\n", "SOR(ω=1.2)", iterSOR, xSOR)
		}
	}
}

}
