package main

import (
	"fmt"
	"math"
)

const (
	epsilon    = 1e-10
	maxIter    = 10000
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

func powerIteration(P [][]float64, initialDist []float64) ([]float64, int, []int) {
	n := len(P)

	absorbingStates := detectAbsorbingStates(P)

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
			return pi, iter + 1, absorbingStates
		}
	}

	return pi, maxIter, absorbingStates
}

func main() {
	P := [][]float64{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.2, 0.0, 0.5, 0.3},
		{0.1, 0.1, 0.4, 0.4},
	}

	initialDist := []float64{0.0, 0.0, 0.6, 0.4}

	fmt.Println("状态转移矩阵 P:")
	for i := range P {
		fmt.Printf("  %.4f\n", P[i])
	}

	fmt.Println("\n初始分布:")
	fmt.Printf("  %.4f\n", initialDist)

	pi, iterations, absorbingStates := powerIteration(P, initialDist)

	if len(absorbingStates) > 0 {
		fmt.Printf("\n检测到吸收态: %v\n", absorbingStates)
		fmt.Println("警告: 对于吸收链，极限分布依赖于初始分布！")
		fmt.Println("       结果表示最终被各吸收态吸收的概率。")
	}

	fmt.Printf("\n极限分布 π (迭代 %d 次后收敛):\n", iterations)
	for i := range pi {
		fmt.Printf("  π[%d] = %.6f\n", i, pi[i])
	}

	sum := 0.0
	for _, v := range pi {
		sum += v
	}
	fmt.Printf("\n验证: Σπ = %.6f\n", sum)

	fmt.Println("\n验证 πP = π:")
	piP := make([]float64, len(pi))
	maxDiff := 0.0
	for j := 0; j < len(pi); j++ {
		for i := 0; i < len(pi); i++ {
			piP[j] += pi[i] * P[i][j]
		}
		diff := math.Abs(piP[j] - pi[j])
		if diff > maxDiff {
			maxDiff = diff
		}
		fmt.Printf("  πP[%d] = %.6f, π[%d] = %.6f, diff = %.2e\n", j, piP[j], j, pi[j], diff)
	}
	fmt.Printf("\n最大误差: %.2e\n", maxDiff)

	if len(absorbingStates) > 0 {
		fmt.Println("\n吸收概率:")
		total := 0.0
		for _, s := range absorbingStates {
			fmt.Printf("  吸收态 %d: %.6f\n", s, pi[s])
			total += pi[s]
		}
		fmt.Printf("  总吸收概率: %.6f\n", total)
	}
}
