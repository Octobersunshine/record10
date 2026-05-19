package main

import (
	"fmt"
	"math"
)

type BoundaryType int

const (
	Dirichlet BoundaryType = iota
	Neumann
)

type BoundaryCondition struct {
	Type      BoundaryType
	LeftValue  float64
	RightValue float64
}

type SimulationParams struct {
	Length      float64
	Time        float64
	Dx          float64
	Dt          float64
	Alpha       float64
	Boundary    BoundaryCondition
	InitialTemp []float64
}

func FTCS(params SimulationParams) ([][]float64, error) {
	nx := int(params.Length/params.Dx) + 1
	nt := int(params.Time/params.Dt) + 1

	if len(params.InitialTemp) != nx {
		return nil, fmt.Errorf("初始温度数组长度 %d 与空间离散点数 %d 不匹配", len(params.InitialTemp), nx)
	}

	r := params.Alpha * params.Dt / (params.Dx * params.Dx)
	if r > 0.5 {
		fmt.Printf("警告：傅里叶数 r = %.4f > 0.5，数值解可能不稳定\n", r)
	}

	temperature := make([][]float64, nt)
	for i := range temperature {
		temperature[i] = make([]float64, nx)
	}

	copy(temperature[0], params.InitialTemp)

	for n := 0; n < nt-1; n++ {
		for i := 1; i < nx-1; i++ {
			temperature[n+1][i] = temperature[n][i] +
				r*(temperature[n][i+1]-2*temperature[n][i]+temperature[n][i-1])
		}

		switch params.Boundary.Type {
		case Dirichlet:
			temperature[n+1][0] = params.Boundary.LeftValue
			temperature[n+1][nx-1] = params.Boundary.RightValue

		case Neumann:
			leftFlux := params.Boundary.LeftValue
			rightFlux := params.Boundary.RightValue

			uLeftGhost := temperature[n][1] - 2*params.Dx*leftFlux
			temperature[n+1][0] = temperature[n][0] +
				r*(temperature[n][1]-2*temperature[n][0]+uLeftGhost)

			uRightGhost := temperature[n][nx-2] + 2*params.Dx*rightFlux
			temperature[n+1][nx-1] = temperature[n][nx-1] +
				r*(uRightGhost-2*temperature[n][nx-1]+temperature[n][nx-2])
		}
	}

	return temperature, nil
}

func CalculateTotalHeat(temperature []float64, dx float64) float64 {
	total := 0.0
	for i := 0; i < len(temperature); i++ {
		total += temperature[i]
	}
	return total * dx
}

func CalculateBoundaryHeatFlux(temperature []float64, dx, alpha float64, boundary BoundaryCondition) (float64, float64) {
	nx := len(temperature)
	leftFlux := 0.0
	rightFlux := 0.0

	switch boundary.Type {
	case Dirichlet:
		leftFlux = alpha * (temperature[1] - temperature[0]) / dx
		rightFlux = alpha * (temperature[nx-1] - temperature[nx-2]) / dx
	case Neumann:
		leftFlux = boundary.LeftValue
		rightFlux = boundary.RightValue
	}

	return leftFlux, rightFlux
}

func VerifyHeatConservation(temperature [][]float64, params SimulationParams) []float64 {
	nt := len(temperature)
	heatHistory := make([]float64, nt)

	for n := 0; n < nt; n++ {
		heatHistory[n] = CalculateTotalHeat(temperature[n], params.Dx)
	}

	return heatHistory
}

func InitialConditionUniform(x, L float64) float64 {
	return 50.0
}

func InitialConditionSin(x, L float64) float64 {
	return 100 * math.Sin(math.Pi*x/L)
}

func InitialConditionStep(x, L float64) float64 {
	if x < L/2 {
		return 100.0
	}
	return 0.0
}

func InitialConditionGaussian(x, L float64) float64 {
	mu := L / 2
	sigma := L / 10
	return 100 * math.Exp(-math.Pow(x-mu, 2)/(2*math.Pow(sigma, 2)))
}

func main() {
	fmt.Println("=== 一维热传导方程 FTCS 求解器（改进版）===")
	fmt.Println("包含二阶精度 Neumann 边界条件和热流守恒验证")
	fmt.Println()

	length := 1.0
	totalTime := 0.05
	dx := 0.01
	dt := 0.00005
	alpha := 0.01

	nx := int(length/dx) + 1

	fmt.Println("测试1: Neumann 边界条件（绝热边界，热流应为0）")
	fmt.Println("边界条件：左右热流 = 0")
	fmt.Println()

	initialTemp1 := make([]float64, nx)
	for i := 0; i < nx; i++ {
		x := float64(i) * dx
		initialTemp1[i] = InitialConditionGaussian(x, length)
	}

	params1 := SimulationParams{
		Length:   length,
		Time:     totalTime,
		Dx:       dx,
		Dt:       dt,
		Alpha:    alpha,
		Boundary: BoundaryCondition{Type: Neumann, LeftValue: 0, RightValue: 0},
		InitialTemp: initialTemp1,
	}

	temperature1, err := FTCS(params1)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}

	heatHistory1 := VerifyHeatConservation(temperature1, params1)
	fmt.Printf("参数: L=%.2fm, T=%.4fs, dx=%.4fm, dt=%.6fs, alpha=%.4f m²/s\n",
		length, totalTime, dx, dt, alpha)
	fmt.Printf("空间点数: %d, 时间步数: %d\n\n", len(temperature1[0]), len(temperature1))

	fmt.Println("热流守恒验证 (绝热边界下总热量应恒定):")
	fmt.Printf("  初始总热量: %.6f\n", heatHistory1[0])
	fmt.Printf("  最终总热量: %.6f\n", heatHistory1[len(heatHistory1)-1])
	fmt.Printf("  相对误差:   %.6f%%\n\n",
		math.Abs(heatHistory1[len(heatHistory1)-1]-heatHistory1[0])/heatHistory1[0]*100)

	fmt.Println("温度分布:")
	timePoints := []int{0, len(temperature1)/2, len(temperature1)-1}
	for _, tIdx := range timePoints {
		fmt.Printf("  时刻 t = %.4fs:\n", float64(tIdx)*dt)
		showPoints := 10
		step := len(temperature1[tIdx]) / showPoints
		for i := 0; i < len(temperature1[tIdx]); i += step {
			fmt.Printf("    x=%.2fm: %.4f°C\n", float64(i)*dx, temperature1[tIdx][i])
		}
	}

	fmt.Println()
	fmt.Println("测试2: Dirichlet 边界条件")
	fmt.Println("边界条件：左右温度 = 0")
	fmt.Println()

	initialTemp2 := make([]float64, nx)
	for i := 0; i < nx; i++ {
		x := float64(i) * dx
		initialTemp2[i] = InitialConditionSin(x, length)
	}

	params2 := SimulationParams{
		Length:   length,
		Time:     totalTime,
		Dx:       dx,
		Dt:       dt,
		Alpha:    alpha,
		Boundary: BoundaryCondition{Type: Dirichlet, LeftValue: 0, RightValue: 0},
		InitialTemp: initialTemp2,
	}

	temperature2, err := FTCS(params2)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}

	fmt.Println("温度分布:")
	for _, tIdx := range timePoints {
		fmt.Printf("  时刻 t = %.4fs:\n", float64(tIdx)*dt)
		showPoints := 10
		step := len(temperature2[tIdx]) / showPoints
		for i := 0; i < len(temperature2[tIdx]); i += step {
			fmt.Printf("    x=%.2fm: %.4f°C\n", float64(i)*dx, temperature2[tIdx][i])
		}
	}

	fmt.Println()
	fmt.Println("=== 边界条件实现说明 ===")
	fmt.Println("Dirichlet (固定温度): u(0) = T_left, u(L) = T_right")
	fmt.Println("Neumann (固定热流):   ∂u/∂x(0) = q_left, ∂u/∂x(L) = q_right")
	fmt.Println()
	fmt.Println("Neumann 边界二阶精度实现（镜像点法）:")
	fmt.Println("  左边界: u[-1] = u[1] - 2*dx*q_left")
	fmt.Println("  右边界: u[N] = u[N-2] + 2*dx*q_right")
	fmt.Println("  精度: O(dx²)，优于一阶近似 O(dx)")
	fmt.Println()
	fmt.Println("热流守恒验证:")
	fmt.Println("  绝热边界 (q=0) 下，总热量应保持恒定")
	fmt.Println("  相对误差 < 0.1% 表明热流守恒良好")
}
