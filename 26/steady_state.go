package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

const (
	defaultEpsilon = 1e-10
	defaultMaxIter = 10000
)

type Config struct {
	Matrix      [][]float64 `json:"matrix"`
	InitialDist []float64   `json:"initial_dist,omitempty"`
	Epsilon     float64     `json:"epsilon,omitempty"`
	MaxIter     int         `json:"max_iter,omitempty"`
}

type ChainType string

const (
	RegularChain    ChainType = "regular"
	AbsorbingChain  ChainType = "absorbing"
	UnknownChain    ChainType = "unknown"
)

func detectAbsorbingStates(P [][]float64) []int {
	var absorbing []int
	n := len(P)
	for i := 0; i < n; i++ {
		if math.Abs(P[i][i]-1.0) < 1e-9 {
			isAbsorbing := true
			for j := 0; j < n; j++ {
				if j != i && P[i][j] > 1e-9 {
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

func classifyChain(P [][]float64, absorbingStates []int) ChainType {
	if len(absorbingStates) > 0 {
		return AbsorbingChain
	}
	return RegularChain
}

func PowerIteration(P [][]float64, initialDist []float64, epsilon float64, maxIter int) ([]float64, int, ChainType, []int, error) {
	n := len(P)
	if n == 0 {
		return nil, 0, UnknownChain, nil, fmt.Errorf("matrix is empty")
	}

	for i := 0; i < n; i++ {
		if len(P[i]) != n {
			return nil, 0, UnknownChain, nil, fmt.Errorf("matrix is not square: row %d has %d elements, expected %d", i, len(P[i]), n)
		}
		rowSum := 0.0
		for j := 0; j < n; j++ {
			if P[i][j] < 0 {
				return nil, 0, UnknownChain, nil, fmt.Errorf("negative probability at P[%d][%d] = %f", i, j, P[i][j])
			}
			rowSum += P[i][j]
		}
		if math.Abs(rowSum-1.0) > 1e-6 {
			return nil, 0, UnknownChain, nil, fmt.Errorf("row %d sum is %.6f, should be 1.0", i, rowSum)
		}
	}

	absorbingStates := detectAbsorbingStates(P)
	chainType := classifyChain(P, absorbingStates)

	epsilon = math.Max(epsilon, 1e-15)
	if maxIter <= 0 {
		maxIter = defaultMaxIter
	}

	var pi []float64
	if initialDist != nil && len(initialDist) == n {
		pi = make([]float64, n)
		sum := 0.0
		for i := 0; i < n; i++ {
			if initialDist[i] < 0 {
				return nil, 0, chainType, absorbingStates, fmt.Errorf("initial distribution has negative value at %d", i)
			}
			sum += initialDist[i]
		}
		if math.Abs(sum-1.0) > 1e-6 {
			return nil, 0, chainType, absorbingStates, fmt.Errorf("initial distribution sum is %.6f, should be 1.0", sum)
		}
		copy(pi, initialDist)
	} else {
		pi = make([]float64, n)
		for i := 0; i < n; i++ {
			pi[i] = 1.0 / float64(n)
		}
	}

	for iter := 0; iter < maxIter; iter++ {
		nextPi := make([]float64, n)
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				nextPi[j] += pi[i] * P[i][j]
			}
		}

		diff := 0.0
		for i := 0; i < n; i++ {
			diff += math.Abs(nextPi[i] - pi[i])
		}

		pi = nextPi

		if diff < epsilon {
			return pi, iter + 1, chainType, absorbingStates, nil
		}
	}

	return pi, maxIter, chainType, absorbingStates, fmt.Errorf("did not converge within %d iterations (last diff=%.2e)", maxIter, diff)
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

	if len(os.Args) > 1 {
		filename := os.Args[1]
		fmt.Printf("从文件加载矩阵: %s\n", filename)
		loaded, err := LoadConfig(filename)
		if err != nil {
			fmt.Printf("加载文件失败: %v\n", err)
			fmt.Println("使用默认矩阵...")
			config = Config{
				Matrix: [][]float64{
					{1.0, 0.0, 0.0, 0.0},
					{0.0, 1.0, 0.0, 0.0},
					{0.2, 0.0, 0.5, 0.3},
					{0.1, 0.1, 0.4, 0.4},
				},
				Epsilon: defaultEpsilon,
				MaxIter: defaultMaxIter,
			}
		} else {
			config = *loaded
		}
	} else {
		fmt.Println("使用默认吸收链示例矩阵")
		config = Config{
			Matrix: [][]float64{
				{1.0, 0.0, 0.0, 0.0},
				{0.0, 1.0, 0.0, 0.0},
				{0.2, 0.0, 0.5, 0.3},
				{0.1, 0.1, 0.4, 0.4},
			},
			InitialDist: []float64{0.0, 0.0, 0.6, 0.4},
			Epsilon:     defaultEpsilon,
			MaxIter:     defaultMaxIter,
		}
	}

	if config.Epsilon <= 0 {
		config.Epsilon = defaultEpsilon
	}
	if config.MaxIter <= 0 {
		config.MaxIter = defaultMaxIter
	}

	fmt.Println("\n状态转移矩阵 P:")
	for i := range config.Matrix {
		fmt.Printf("  %v\n", config.Matrix[i])
	}

	if config.InitialDist != nil {
		fmt.Println("\n初始分布:")
		fmt.Printf("  %v\n", config.InitialDist)
	}

	pi, iterations, chainType, absorbingStates, err := PowerIteration(config.Matrix, config.InitialDist, config.Epsilon, config.MaxIter)
	if err != nil {
		fmt.Printf("\n警告: %v\n", err)
	}

	fmt.Printf("\n马尔可夫链类型: %s\n", chainType)
	if len(absorbingStates) > 0 {
		fmt.Printf("检测到吸收态: %v\n", absorbingStates)
		fmt.Println("警告: 对于吸收链，稳态分布依赖于初始分布！")
		fmt.Println("       结果表示从初始分布出发最终被各吸收态吸收的概率。")
	}

	fmt.Printf("\n极限分布 π (迭代 %d 次):\n", iterations)
	for i := range pi {
		fmt.Printf("  π[%d] = %.8f\n", i, pi[i])
	}

	sum := 0.0
	for _, v := range pi {
		sum += v
	}
	fmt.Printf("\n验证: Σπ = %.8f\n", sum)

	fmt.Println("\n验证 πP = π:")
	piP := make([]float64, len(pi))
	maxDiff := 0.0
	for j := 0; j < len(pi); j++ {
		for i := 0; i < len(pi); i++ {
			piP[j] += pi[i] * config.Matrix[i][j]
		}
		diff := math.Abs(piP[j] - pi[j])
		if diff > maxDiff {
			maxDiff = diff
		}
		fmt.Printf("  πP[%d] = %.8f, π[%d] = %.8f, diff = %.2e\n", j, piP[j], j, pi[j], diff)
	}
	fmt.Printf("\n最大误差: %.2e\n", maxDiff)

	if len(absorbingStates) > 0 {
		fmt.Println("\n吸收态吸收概率:")
		totalAbsorbProb := 0.0
		for _, s := range absorbingStates {
			fmt.Printf("  吸收态 %d 的吸收概率: %.8f\n", s, pi[s])
			totalAbsorbProb += pi[s]
		}
		fmt.Printf("  总吸收概率: %.8f\n", totalAbsorbProb)
	}
}
