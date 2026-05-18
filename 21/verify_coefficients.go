package main

import (
	"fmt"
	"math"
	"newton-cotes-integration/integration"
)

func main() {
	nc := &integration.NewtonCotes{}
	
	fmt.Println("=== 伯努利数验证 ===")
	nc.PrintBernoulliNumbers(10)
	
	fmt.Println("\n=== 牛顿-柯特斯系数验证 ===")
	for n := 1; n <= 6; n++ {
		nc.VerifyCoefficients(n)
	}
	
	fmt.Println("\n=== 龙贝格积分验证 ===")
	romberg := &integration.Romberg{
		Tolerance: 1e-10,
		MaxLevels: 10,
	}
	
	testCases := []struct {
		name     string
		function string
		a, b     float64
		exact    float64
	}{
		{"f(x)=x²", "x^2", 0, 1, 1.0 / 3.0},
		{"f(x)=x³", "x^3", 0, 1, 0.25},
		{"f(x)=sin(x)", "sin(x)", 0, math.Pi, 2.0},
		{"f(x)=exp(x)", "exp(x)", 0, 1, math.E - 1},
	}
	
	for _, tc := range testCases {
		fmt.Printf("\n%s 在 [%.2f, %.2f] 积分:\n", tc.name, tc.a, tc.b)
		fmt.Printf("精确值: %.12f\n", tc.exact)
		
		result, table, _ := romberg.IntegrateExpr(tc.function, tc.a, tc.b)
		integration.PrintRombergTable(table)
		
		errRomberg := math.Abs(result - tc.exact)
		fmt.Printf("龙贝格结果: %.12f, 绝对误差: %e\n", result, errRomberg)
		
		fmt.Println("牛顿-柯特斯对比 (N=10):")
		for n := 1; n <= 4; n++ {
			resultNC, _ := nc.Integrate(tc.function, tc.a, tc.b, n, 10)
			errNC := math.Abs(resultNC - tc.exact)
			fmt.Printf("  n=%d: %.12f, 绝对误差: %e\n", n, resultNC, errNC)
		}
	}
	
	fmt.Println("\n=== 龙贝格积分收敛速度测试 ===")
	f := func(x float64) float64 { return math.Sin(x) }
	resultFast, tableFast, _ := romberg.Integrate(f, 0, math.Pi)
	fmt.Printf("∫ sin(x) dx [0,π] = %.15f (精确值=2.0)\n", resultFast)
	integration.PrintRombergTable(tableFast)
}
