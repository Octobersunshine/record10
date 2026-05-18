package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"
)

const (
	defaultEpsilon = 1e-10
	defaultMaxIter = 10000
	defaultKrylovDim = 50
)

type SparseEntry struct {
	Row int
	Col int
	Val float64
}

type SparseMatrix struct {
	N        int
	Entries  []SparseEntry
	RowPtr   []int
	ColInd   []int
	Values   []float64
}

type Config struct {
	Matrix       [][]float64 `json:"matrix,omitempty"`
	SparseMatrix *SparseConfig `json:"sparse_matrix,omitempty"`
	InitialDist  []float64   `json:"initial_dist,omitempty"`
	Epsilon      float64     `json:"epsilon,omitempty"`
	MaxIter      int         `json:"max_iter,omitempty"`
	KrylovDim    int         `json:"krylov_dim,omitempty"`
}

type SparseConfig struct {
	N       int           `json:"n"`
	Entries []SparseEntry `json:"entries"`
}

func (sm *SparseMatrix) Multiply(v []float64) []float64 {
	n := sm.N
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := sm.RowPtr[i]; j < sm.RowPtr[i+1]; j++ {
			col := sm.ColInd[j]
			result[col] += v[i] * sm.Values[j]
		}
	}
	return result
}

func NewSparseMatrix(n int, entries []SparseEntry) *SparseMatrix {
	sm := &SparseMatrix{
		N:       n,
		Entries: entries,
	}
	sm.RowPtr = make([]int, n+1)
	for _, e := range entries {
		sm.RowPtr[e.Row+1]++
	}
	for i := 1; i <= n; i++ {
		sm.RowPtr[i] += sm.RowPtr[i-1]
	}
	sm.ColInd = make([]int, len(entries))
	sm.Values = make([]float64, len(entries))
	next := make([]int, n)
	copy(next, sm.RowPtr[:n])
	for _, e := range entries {
		pos := next[e.Row]
		sm.ColInd[pos] = e.Col
		sm.Values[pos] = e.Val
		next[e.Row]++
	}
	return sm
}

func DenseToSparse(matrix [][]float64) *SparseMatrix {
	n := len(matrix)
	var entries []SparseEntry
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if math.Abs(matrix[i][j]) > 1e-15 {
				entries = append(entries, SparseEntry{i, j, matrix[i][j]})
			}
		}
	}
	return NewSparseMatrix(n, entries)
}

func normalize(v []float64) float64 {
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm > 1e-15 {
		for i := range v {
			v[i] /= norm
		}
	}
	return norm
}

func dot(a, b []float64) float64 {
	res := 0.0
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func ArnoldiIteration(sm *SparseMatrix, v0 []float64, k int) ([][]float64, [][]float64) {
	n := sm.N
	V := make([][]float64, k+1)
	H := make([][]float64, k+1)
	for i := range H {
		H[i] = make([]float64, k)
	}
	V[0] = make([]float64, n)
	copy(V[0], v0)
	normalize(V[0])
	for m := 0; m < k; m++ {
		w := sm.Multiply(V[m])
		for j := 0; j <= m; j++ {
			H[j][m] = dot(w, V[j])
			for i := 0; i < n; i++ {
				w[i] -= H[j][m] * V[j][i]
			}
		}
		H[m+1][m] = normalize(w)
		if H[m+1][m] < 1e-12 {
			return V[:m+1], H[:m+1]
		}
		V[m+1] = w
	}
	return V, H
}

func solveEigenvalue(H [][]float64) ([]float64, []float64, error) {
	k := len(H) - 1
	if k <= 0 {
		return nil, nil, fmt.Errorf("dimension too small")
	}
	Hsmall := make([][]float64, k)
	for i := 0; i < k; i++ {
		Hsmall[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			if i < len(H) && j < len(H[i]) {
				Hsmall[i][j] = H[i][j]
			}
		}
	}
	eigenvector := powerIterationSmall(Hsmall, 100)
	eigenvalues := make([]float64, k)
	for i := 0; i < k; i++ {
		eigenvalues[i] = Hsmall[i][i]
	}
	return eigenvalues, eigenvector, nil
}

func powerIterationSmall(A [][]float64, maxIter int) []float64 {
	k := len(A)
	v := make([]float64, k)
	for i := range v {
		v[i] = 1.0 / float64(k)
	}
	for iter := 0; iter < maxIter; iter++ {
		next := make([]float64, k)
		for i := 0; i < k; i++ {
			for j := 0; j < k; j++ {
				next[i] += A[j][i] * v[j]
			}
		}
		norm := normalize(next)
		if norm < 1e-15 {
			break
		}
		v = next
	}
	return v
}

func KrylovSteadyState(sm *SparseMatrix, initialDist []float64, epsilon float64, maxIter int, krylovDim int) ([]float64, int, time.Duration, error) {
	start := time.Now()
	n := sm.N
	var v0 []float64
	if initialDist != nil && len(initialDist) == n {
		v0 = make([]float64, n)
		copy(v0, initialDist)
	} else {
		v0 = make([]float64, n)
		for i := 0; i < n; i++ {
			v0[i] = 1.0 / float64(n)
		}
	}
	if krylovDim <= 0 || krylovDim > n {
		krylovDim = defaultKrylovDim
		if krylovDim > n {
			krylovDim = n
		}
	}
	V, H := ArnoldiIteration(sm, v0, krylovDim)
	_, y, _ := solveEigenvalue(H)
	pi := make([]float64, n)
	k := len(V) - 1
	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			pi[i] += V[j][i] * y[j]
		}
	}
	sum := 0.0
	for _, v := range pi {
		sum += math.Abs(v)
	}
	if sum > 1e-15 {
		for i := range pi {
			pi[i] = math.Abs(pi[i]) / sum
		}
	}
	for iter := 0; iter < 10; iter++ {
		nextPi := sm.Multiply(pi)
		diff := 0.0
		for i := 0; i < n; i++ {
			diff += math.Abs(nextPi[i] - pi[i])
		}
		pi = nextPi
		if diff < epsilon {
			return pi, iter + 1, time.Since(start), nil
		}
	}
	return pi, 10, time.Since(start), nil
}

func detectAbsorbingStatesSparse(sm *SparseMatrix) []int {
	var absorbing []int
	n := sm.N
	rowVals := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := sm.RowPtr[i]; j < sm.RowPtr[i+1]; j++ {
			col := sm.ColInd[j]
			if col == i {
				rowVals[i] = sm.Values[j]
			}
		}
	}
	for i := 0; i < n; i++ {
		if math.Abs(rowVals[i]-1.0) < 1e-9 {
			isAbsorbing := true
			for j := sm.RowPtr[i]; j < sm.RowPtr[i+1]; j++ {
				col := sm.ColInd[j]
				if col != i && sm.Values[j] > 1e-9 {
					isAbsorbing = false
					break
				}
			}
			if isAbsorbing {
				absorbing = append(absorbing, i)
			}
		}
	}
	return absorbing
}

func PowerIterationSparse(sm *SparseMatrix, initialDist []float64, epsilon float64, maxIter int) ([]float64, int, time.Duration, error) {
	start := time.Now()
	n := sm.N
	var pi []float64
	if initialDist != nil && len(initialDist) == n {
		pi = make([]float64, n)
		copy(pi, initialDist)
	} else {
		pi = make([]float64, n)
		for i := 0; i < n; i++ {
			pi[i] = 1.0 / float64(n)
		}
	}
	for iter := 0; iter < maxIter; iter++ {
		nextPi := sm.Multiply(pi)
		diff := 0.0
		for i := 0; i < n; i++ {
			diff += math.Abs(nextPi[i] - pi[i])
		}
		pi = nextPi
		if diff < epsilon {
			return pi, iter + 1, time.Since(start), nil
		}
	}
	return pi, maxIter, time.Since(start), fmt.Errorf("did not converge within %d iterations", maxIter)
}

func LoadConfig(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	return &config, nil
}

func main() {
	var config Config
	var sm *SparseMatrix
	if len(os.Args) > 1 {
		filename := os.Args[1]
		fmt.Printf("从文件加载配置: %s\n", filename)
		loaded, err := LoadConfig(filename)
		if err != nil {
			fmt.Printf("加载文件失败: %v\n", err)
			fmt.Println("使用默认示例矩阵...")
			config = Config{
				Matrix: [][]float64{
					{0.7, 0.2, 0.1},
					{0.3, 0.5, 0.2},
					{0.1, 0.3, 0.6},
				},
				Epsilon:   defaultEpsilon,
				MaxIter:   defaultMaxIter,
				KrylovDim: defaultKrylovDim,
			}
			sm = DenseToSparse(config.Matrix)
		} else {
			config = *loaded
			if config.SparseMatrix != nil {
				sm = NewSparseMatrix(config.SparseMatrix.N, config.SparseMatrix.Entries)
			} else if config.Matrix != nil {
				sm = DenseToSparse(config.Matrix)
			} else {
				fmt.Println("配置文件中未找到矩阵，使用默认矩阵...")
				config.Matrix = [][]float64{
					{0.7, 0.2, 0.1},
					{0.3, 0.5, 0.2},
					{0.1, 0.3, 0.6},
				}
				sm = DenseToSparse(config.Matrix)
			}
		}
	} else {
		fmt.Println("使用默认示例矩阵")
		config = Config{
			Matrix: [][]float64{
				{0.7, 0.2, 0.1},
				{0.3, 0.5, 0.2},
				{0.1, 0.3, 0.6},
			},
			Epsilon:   defaultEpsilon,
			MaxIter:   defaultMaxIter,
			KrylovDim: 10,
		}
		sm = DenseToSparse(config.Matrix)
	}
	if config.Epsilon <= 0 {
		config.Epsilon = defaultEpsilon
	}
	if config.MaxIter <= 0 {
		config.MaxIter = defaultMaxIter
	}
	if config.KrylovDim <= 0 {
		config.KrylovDim = defaultKrylovDim
	}
	fmt.Printf("\n矩阵大小: %d × %d\n", sm.N, sm.N)
	fmt.Printf("非零元素数: %d (稀疏度: %.2f%%)\n", len(sm.Values), 100*float64(len(sm.Values))/float64(sm.N*sm.N))
	absorbingStates := detectAbsorbingStatesSparse(sm)
	if len(absorbingStates) > 0 {
		fmt.Printf("检测到吸收态: %v\n", absorbingStates)
		fmt.Println("警告: 对于吸收链，极限分布依赖于初始分布！")
	}
	if config.InitialDist != nil {
		fmt.Println("\n初始分布已指定")
	}
	fmt.Println("\n========== 幂迭代法 (Power Iteration) ==========")
	piPower, iterPower, timePower, errPower := PowerIterationSparse(sm, config.InitialDist, config.Epsilon, config.MaxIter)
	if errPower != nil {
		fmt.Printf("警告: %v\n", errPower)
	}
	fmt.Printf("迭代次数: %d\n", iterPower)
	fmt.Printf("计算时间: %v\n", timePower)
	fmt.Println("\n极限分布:")
	for i := 0; i < min(10, sm.N); i++ {
		fmt.Printf("  π[%d] = %.8f\n", i, piPower[i])
	}
	if sm.N > 10 {
		fmt.Printf("  ... (仅显示前10个状态)\n")
	}
	fmt.Println("\n========== Krylov子空间法 (Arnoldi迭代) ==========")
	fmt.Printf("Krylov子空间维度: %d\n", config.KrylovDim)
	piKrylov, iterKrylov, timeKrylov, errKrylov := KrylovSteadyState(sm, config.InitialDist, config.Epsilon, config.MaxIter, config.KrylovDim)
	if errKrylov != nil {
		fmt.Printf("警告: %v\n", errKrylov)
	}
	fmt.Printf("迭代次数: %d\n", iterKrylov)
	fmt.Printf("计算时间: %v\n", timeKrylov)
	fmt.Println("\n极限分布:")
	for i := 0; i < min(10, sm.N); i++ {
		fmt.Printf("  π[%d] = %.8f\n", i, piKrylov[i])
	}
	if sm.N > 10 {
		fmt.Printf("  ... (仅显示前10个状态)\n")
	}
	fmt.Println("\n========== 性能对比 ==========")
	fmt.Printf("时间加速比: %.2fx\n", float64(timePower)/float64(timeKrylov))
	fmt.Printf("迭代次数比: %.2fx\n", float64(iterPower)/float64(iterKrylov))
	maxDiff := 0.0
	for i := 0; i < sm.N; i++ {
		diff := math.Abs(piKrylov[i] - piPower[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	fmt.Printf("结果最大差异: %.2e\n", maxDiff)
	sum := 0.0
	for _, v := range piKrylov {
		sum += v
	}
	fmt.Printf("Σπ = %.8f (归一化验证)\n", sum)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
