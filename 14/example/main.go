package main

import (
	"fmt"
	"differential_evolution"
)

func main() {
	dim := 10

	fmt.Println("=== 微分进化算法完整演示 ===")
	fmt.Println("\n=== 第一部分: 边界处理策略 ===")
	fmt.Println("4种边界处理策略:")
	fmt.Println("1. Clip (截断) - 传统方法, 可能降低多样性")
	fmt.Println("2. Reflection (反射) - 超出边界时反射回来")
	fmt.Println("3. RandomReset (随机重置) - 超出时随机重新生成")
	fmt.Println("4. Mixed (混合) - 默认策略, 50%概率反射或重置\n")

	fmt.Println("=== 对比不同边界策略效果 ===")

	config := differential_evolution.DefaultConfig()
	config.MaxIterations = 500
	config.AdaptiveStrategy = differential_evolution.NoAdaptive

	boundaryStrategies := []struct {
		name     string
		strategy differential_evolution.BoundaryStrategy
	}{
		{"Clip (截断)", differential_evolution.ClipStrategy},
		{"Reflection (反射)", differential_evolution.ReflectionStrategy},
		{"RandomReset (随机重置)", differential_evolution.RandomResetStrategy},
		{"Mixed (混合-默认)", differential_evolution.MixedStrategy},
	}

	for _, s := range boundaryStrategies {
		config.BoundaryStrategy = s.strategy
		result, err := differential_evolution.MinimizeWithConfig("rastrigin", dim, -5.12, 5.12, config)
		if err != nil {
			fmt.Printf("%s: 错误 - %v\n", s.name, err)
			continue
		}
		fmt.Printf("%-25s 最优值: %.8f\n", s.name, result.BestValue)
	}

	fmt.Println("\n=== 第二部分: 自适应参数控制 (JADE算法) ===")
	fmt.Println("2种参数控制策略:")
	fmt.Println("1. NoAdaptive - 固定F和CR值 (传统DE)")
	fmt.Println("2. JadeAdaptive - 根据成功历史自适应调整 (默认)\n")

	fmt.Println("=== 对比固定参数 vs JADE自适应 ===")
	functions := []string{"sphere", "rastrigin", "rosenbrock"}
	for _, fn := range functions {
		config.MaxIterations = 800

		config.AdaptiveStrategy = differential_evolution.NoAdaptive
		result1, _ := differential_evolution.MinimizeWithConfig(fn, dim, -5.12, 5.12, config)

		config.AdaptiveStrategy = differential_evolution.JadeAdaptive
		result2, _ := differential_evolution.MinimizeWithConfig(fn, dim, -5.12, 5.12, config)

		fmt.Printf("%-12s 固定参数: %12.8f  JADE自适应: %12.8f\n", fn, result1.BestValue, result2.BestValue)
	}

	fmt.Println("\n=== 第三部分: 综合使用示例 ===")
	fmt.Println("1. 使用默认配置 (混合边界 + JADE自适应):")
	result1, _ := differential_evolution.Minimize("sphere", 5, -5, 5)
	fmt.Printf("   结果: %.6f\n", result1.BestValue)

	fmt.Println("\n2. 禁用自适应，使用固定参数:")
	customConfig := differential_evolution.DefaultConfig()
	customConfig.AdaptiveStrategy = differential_evolution.NoAdaptive
	customConfig.F = 0.6
	customConfig.CR = 0.8
	result2, _ := differential_evolution.MinimizeWithConfig("sphere", 5, -5, 5, customConfig)
	fmt.Printf("   固定参数(F=0.6, CR=0.8)结果: %.6f\n", result2.BestValue)

	fmt.Println("\n3. 完全自定义 (反射边界 + JADE自适应):")
	customDim := 2
	lowerBound := []float64{-10, -10}
	upperBound := []float64{10, 10}

	customFunc := func(x []float64) float64 {
		return (x[0]-2)*(x[0]-2) + (x[1]-3)*(x[1]-3) + 5
	}

	de := differential_evolution.NewDE(100, 0.9, 0.5, 500, customDim, lowerBound, upperBound, customFunc)
	de.SetBoundaryStrategy(differential_evolution.ReflectionStrategy)
	de.SetAdaptiveStrategy(differential_evolution.JadeAdaptive)
	result3 := de.Optimize()

	fmt.Printf("   理论最优解: (2.0, 3.0), 理论最优值: 5.0\n")
	fmt.Printf("   计算最优解: (%.4f, %.4f)\n", result3.BestSolution[0], result3.BestSolution[1])
	fmt.Printf("   计算最优值: %.6f\n", result3.BestValue)

	fmt.Println("\n=== JADE自适应参数原理 ===")
	fmt.Println("• F值: 柯西分布采样，长尾特性允许更大步长探索")
	fmt.Println("• CR值: 正态分布采样，控制交叉概率")
	fmt.Println("• 更新机制: 根据改进幅度加权平均成功参数")
	fmt.Println("• 记忆池: 存储历史成功参数，定期取中位数平滑")
	fmt.Println("• 学习率 c=0.1: 控制新旧信息的平衡")
}
