package main

import (
	"fmt"
	"math"
	"genetic"
)

func sphere(x []float64) float64 {
	sum := 0.0
	for _, xi := range x {
		sum += xi * xi
	}
	return sum
}

func rastrigin(x []float64) float64 {
	n := float64(len(x))
	sum := 10 * n
	for _, xi := range x {
		sum += xi*xi - 10*math.Cos(2*math.Pi*xi)
	}
	return sum
}

func rosenbrock(x []float64) float64 {
	sum := 0.0
	for i := 0; i < len(x)-1; i++ {
		sum += 100*math.Pow(x[i+1]-x[i]*x[i], 2) + math.Pow(1-x[i], 2)
	}
	return sum
}

func main() {
	fmt.Println("=== 遗传算法优化示例 (修复选择压力问题) ===")
	fmt.Println()
	
	fmt.Println("【配置说明】")
	fmt.Println("- 默认配置: 排名选择 + Sigma截断尺度变换")
	fmt.Println("- 高压力配置: 排名选择 + 线性尺度变换")
	fmt.Println()
	
	runTests()
}

func runTests() {
	configs := map[string]genetic.Config{
		"默认配置(高选择压力)":    genetic.DefaultConfig(),
		"高压力配置(更快收敛)": genetic.HighPressureConfig(),
	}
	
	testFuncs := []struct {
		name     string
		objective genetic.ObjectiveFunc
		varRanges []genetic.VariableRange
	}{
		{
			name:     "Sphere函数 (简单单峰)",
			objective: sphere,
			varRanges: []genetic.VariableRange{
				{Min: -5.12, Max: 5.12},
				{Min: -5.12, Max: 5.12},
				{Min: -5.12, Max: 5.12},
			},
		},
		{
			name:     "Rastrigin函数 (多峰困难)",
			objective: rastrigin,
			varRanges: []genetic.VariableRange{
				{Min: -5.12, Max: 5.12},
				{Min: -5.12, Max: 5.12},
			},
		},
		{
			name:     "Rosenbrock函数 (窄谷)",
			objective: rosenbrock,
			varRanges: []genetic.VariableRange{
				{Min: -2.048, Max: 2.048},
				{Min: -2.048, Max: 2.048},
			},
		},
	}
	
	for _, tf := range testFuncs {
		fmt.Printf("===== %s =====\n", tf.name)
		for cfgName, cfg := range configs {
			cfg.MaxGenerations = 200
			cfg.PopulationSize = 100
			
			ga := genetic.NewGA(tf.objective, tf.varRanges, nil, cfg)
			solution, fitness, err := ga.Run()
			
			if err != nil {
				fmt.Printf("%s: 错误 - %v\n", cfgName, err)
			} else {
				fmt.Printf("%s:\n", cfgName)
				fmt.Printf("  解: %v\n", solution)
				fmt.Printf("  适应度: %.8f\n", fitness)
			}
		}
		fmt.Println()
	}
	
	fmt.Println("===== 带约束的优化问题 =====")
	constraints := []genetic.ConstraintFunc{
		func(x []float64) bool {
			return x[0]+x[1] >= 1
		},
		func(x []float64) bool {
			return x[0] >= 0 && x[1] >= 0
		},
	}
	
	varRanges := []genetic.VariableRange{
		{Min: -1, Max: 2},
		{Min: -1, Max: 2},
	}
	
	cfg := genetic.DefaultConfig()
	cfg.SelectionMethod = genetic.RankSelection
	cfg.FitnessScaling = genetic.SigmaTruncationScaling
	cfg.PressureFactor = 1.8
	
	ga := genetic.NewGA(sphere, varRanges, constraints, cfg)
	solution, fitness, err := ga.Run()
	
	if err != nil {
		fmt.Printf("错误: %v\n", err)
	} else {
		fmt.Printf("目标函数: f(x,y) = x² + y²\n")
		fmt.Printf("约束: x + y >= 1, x >= 0, y >= 0\n")
		fmt.Printf("理论最优: x=0.5, y=0.5, f=0.5\n")
		fmt.Printf("计算解: %v\n", solution)
		fmt.Printf("计算值: %.6f\n", fitness)
		fmt.Printf("约束验证: x+y = %.4f >= 1\n", solution[0]+solution[1])
	}
	
	fmt.Println()
	fmt.Println("【选择方法说明】")
	fmt.Println("1. TournamentSelection: 锦标赛选择")
	fmt.Println("2. RouletteWheelSelection: 轮盘赌选择")
	fmt.Println("3. RankSelection: 排名选择 (推荐，选择压力可控)")
	fmt.Println("4. StochasticUniversalSampling: 随机通用采样")
	fmt.Println()
	fmt.Println("【适应度尺度变换说明】")
	fmt.Println("1. NoScaling: 无变换 (选择压力小，不推荐)")
	fmt.Println("2. LinearScaling: 线性尺度变换")
	fmt.Println("3. SigmaTruncationScaling: Sigma截断变换 (默认，鲁棒性好)")
	fmt.Println("4. PowerLawScaling: 幂律变换")
}
