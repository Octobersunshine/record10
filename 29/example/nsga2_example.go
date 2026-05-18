package main

import (
	"fmt"
	"math"
	"sort"
	"genetic"
)

func schafferN1(x []float64) float64 {
	return x[0] * x[0]
}

func schafferN2(x []float64) float64 {
	return math.Pow(x[0]-2, 2)
}

func zdt1F1(x []float64) float64 {
	return x[0]
}

func zdt1F2(x []float64) float64 {
	n := len(x)
	sum := 0.0
	for i := 1; i < n; i++ {
		sum += x[i]
	}
	g := 1 + 9.0*sum/float64(n-1)
	h := 1 - math.Sqrt(x[0]/g)
	return g * h
}

func zdt2F1(x []float64) float64 {
	return x[0]
}

func zdt2F2(x []float64) float64 {
	n := len(x)
	sum := 0.0
	for i := 1; i < n; i++ {
		sum += x[i]
	}
	g := 1 + 9.0*sum/float64(n-1)
	h := 1 - math.Pow(x[0]/g, 2)
	return g * h
}

func main() {
	fmt.Println("=== NSGA-II 多目标优化示例 ===")
	fmt.Println()
	
	fmt.Println("1. Schaffer N1 测试问题 (2维)")
	testSchafferN1()
	fmt.Println()
	
	fmt.Println("2. ZDT1 测试问题 (凸Pareto前沿)")
	testZDT1()
	fmt.Println()
	
	fmt.Println("3. ZDT2 测试问题 (非凸Pareto前沿)")
	testZDT2()
	fmt.Println()
	
	fmt.Println("4. 带约束的多目标优化")
	testConstrained()
}

func testSchafferN1() {
	varRanges := []genetic.VariableRange{
		{Min: -1000, Max: 1000},
	}
	
	objectives := []genetic.ObjectiveFunc{schafferN1, schafferN2}
	
	config := genetic.DefaultNSGAIIConfig()
	config.PopulationSize = 100
	config.MaxGenerations = 200
	
	nsga := genetic.NewNSGAII(objectives, varRanges, nil, config)
	result, err := nsga.Run()
	
	if err != nil {
		fmt.Println("错误:", err)
		return
	}
	
	fmt.Printf("Pareto前沿解数量: %d\n", len(result.Solutions))
	
	sort.Slice(result.Objectives, func(i, j int) bool {
		return result.Objectives[i][0] < result.Objectives[j][0]
	})
	
	fmt.Println("部分Pareto最优解:")
	for i := 0; i < 5 && i < len(result.Solutions); i++ {
		idx := i * len(result.Solutions) / 5
		fmt.Printf("  x=%.4f, f1=%.6f, f2=%.6f\n", 
			result.Solutions[idx][0],
			result.Objectives[idx][0],
			result.Objectives[idx][1])
	}
}

func testZDT1() {
	n := 30
	varRanges := make([]genetic.VariableRange, n)
	for i := range varRanges {
		varRanges[i] = genetic.VariableRange{Min: 0, Max: 1}
	}
	
	objectives := []genetic.ObjectiveFunc{zdt1F1, zdt1F2}
	
	config := genetic.DefaultNSGAIIConfig()
	config.PopulationSize = 100
	config.MaxGenerations = 250
	
	nsga := genetic.NewNSGAII(objectives, varRanges, nil, config)
	result, err := nsga.Run()
	
	if err != nil {
		fmt.Println("错误:", err)
		return
	}
	
	fmt.Printf("Pareto前沿解数量: %d\n", len(result.Solutions))
	
	sort.Slice(result.Objectives, func(i, j int) bool {
		return result.Objectives[i][0] < result.Objectives[j][0]
	})
	
	fmt.Println("Pareto前沿采样点:")
	for i := 0; i < 5 && i < len(result.Solutions); i++ {
		idx := i * len(result.Solutions) / 5
		fmt.Printf("  f1=%.6f, f2=%.6f\n", 
			result.Objectives[idx][0],
			result.Objectives[idx][1])
	}
}

func testZDT2() {
	n := 30
	varRanges := make([]genetic.VariableRange, n)
	for i := range varRanges {
		varRanges[i] = genetic.VariableRange{Min: 0, Max: 1}
	}
	
	objectives := []genetic.ObjectiveFunc{zdt2F1, zdt2F2}
	
	config := genetic.DefaultNSGAIIConfig()
	config.PopulationSize = 100
	config.MaxGenerations = 250
	
	nsga := genetic.NewNSGAII(objectives, varRanges, nil, config)
	result, err := nsga.Run()
	
	if err != nil {
		fmt.Println("错误:", err)
		return
	}
	
	fmt.Printf("Pareto前沿解数量: %d\n", len(result.Solutions))
	
	sort.Slice(result.Objectives, func(i, j int) bool {
		return result.Objectives[i][0] < result.Objectives[j][0]
	})
	
	fmt.Println("Pareto前沿采样点:")
	for i := 0; i < 5 && i < len(result.Solutions); i++ {
		idx := i * len(result.Solutions) / 5
		fmt.Printf("  f1=%.6f, f2=%.6f\n", 
			result.Objectives[idx][0],
			result.Objectives[idx][1])
	}
}

func testConstrained() {
	varRanges := []genetic.VariableRange{
		{Min: 0, Max: 10},
		{Min: 0, Max: 10},
	}
	
	objectives := []genetic.ObjectiveFunc{
		func(x []float64) float64 { return x[0] },
		func(x []float64) float64 { return x[1] },
	}
	
	constraints := []genetic.ConstraintFunc{
		func(x []float64) bool { return x[0]*x[0] + x[1]*x[1] >= 1 },
		func(x []float64) bool { return x[0] + x[1] <= 5 },
	}
	
	config := genetic.DefaultNSGAIIConfig()
	config.PopulationSize = 100
	config.MaxGenerations = 300
	
	nsga := genetic.NewNSGAII(objectives, varRanges, constraints, config)
	result, err := nsga.Run()
	
	if err != nil {
		fmt.Println("错误:", err)
		return
	}
	
	fmt.Printf("Pareto前沿解数量: %d\n", len(result.Solutions))
	
	sort.Slice(result.Objectives, func(i, j int) bool {
		return result.Objectives[i][0] < result.Objectives[j][0]
	})
	
	fmt.Println("部分Pareto最优解 (x, y | f1, f2):")
	for i := 0; i < 6 && i < len(result.Solutions); i++ {
		idx := i * len(result.Solutions) / 6
		fmt.Printf("  (%.4f, %.4f) | (%.4f, %.4f)\n", 
			result.Solutions[idx][0],
			result.Solutions[idx][1],
			result.Objectives[idx][0],
			result.Objectives[idx][1])
	}
}
