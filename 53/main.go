package main

import (
	"fmt"
	"math"
	"euler_maruyama/sde"
)

func main() {
	fmt.Println("=== 随机微分方程求解器 ===")
	fmt.Println()

	fmt.Println("1. Euler-Maruyama 方法（强收敛阶 0.5，弱收敛阶 1.0）")
	fmt.Println("   示例：几何布朗运动")
	demoGBM1D()
	fmt.Println()

	fmt.Println("2. 随机Runge-Kutta SRI1 方法（弱收敛阶 1.0）")
	fmt.Println("   示例：几何布朗运动")
	demoSRI1GBM()
	fmt.Println()

	fmt.Println("3. 随机Runge-Kutta SRI2 方法（弱收敛阶 2.0）")
	fmt.Println("   示例：几何布朗运动")
	demoSRI2GBM()
	fmt.Println()

	fmt.Println("4. 收敛阶对比分析")
	fmt.Println("   比较 Euler-Maruyama, SRI1, SRI2 的弱误差")
	demoConvergenceComparison()
	fmt.Println()

	fmt.Println("5. 二维SDE：相关几何布朗运动（耦合噪声 + SRI2方法）")
	fmt.Println("   dX1_t = μ1X1_t dt + σ1X1_t dW1_t")
	fmt.Println("   dX2_t = μ2X2_t dt + σ2X2_t dW2_t")
	fmt.Println("   Corr(dW1_t, dW2_t) = ρ")
	demoCorrelatedGBMWithSRI2()
}

func demoGBM1D() {
	mu := 0.05
	sigma := 0.2

	drift := func(x, t float64) float64 {
		return mu * x
	}
	diffusion := func(x, t float64) float64 {
		return sigma * x
	}

	x0 := 100.0
	t0 := 0.0
	T := 1.0
	numSteps := 1000

	path := sde.EulerMaruyama1DWithSeed(drift, diffusion, x0, t0, T, numSteps, 42)

	fmt.Printf("   初始值: %.2f, 终止时间: %.1f, 步数: %d\n", x0, T, numSteps)
	fmt.Printf("   最终值: %.4f\n", path.Values[numSteps][0])
	fmt.Printf("   理论期望: %.4f\n", x0*math.Exp(mu*T))
}

func demoCorrelatedGBM() {
	mu1, mu2 := 0.05, 0.08
	sigma1, sigma2 := 0.2, 0.3
	rho := 0.7

	corrMatrix := sde.Matrix{
		{1.0, rho},
		{rho, 1.0},
	}

	drift := sde.DriftFunc(func(x sde.Vector, t float64) sde.Vector {
		return sde.Vector{mu1 * x[0], mu2 * x[1]}
	})

	diffusion := sde.DiffusionFunc(func(x sde.Vector, t float64) sde.Matrix {
		return sde.Matrix{
			{sigma1 * x[0], 0.0},
			{0.0, sigma2 * x[1]},
		}
	})

	x0 := sde.Vector{100.0, 100.0}
	t0 := 0.0
	T := 1.0
	numSteps := 1000

	config := sde.SDEConfig{
		Drift:      drift,
		Diffusion:  diffusion,
		X0:         x0,
		T0:         t0,
		T:          T,
		NumSteps:   numSteps,
		CorrMatrix: corrMatrix,
		Seed:       12345,
		UseSeed:    true,
	}

	path, err := sde.EulerMaruyama(config)
	if err != nil {
		fmt.Printf("   错误: %v\n", err)
		return
	}

	fmt.Printf("   相关系数 ρ = %.2f\n", rho)
	fmt.Printf("   初始值: [%.2f, %.2f], 终止时间: %.1f, 步数: %d\n", x0[0], x0[1], T, numSteps)
	fmt.Printf("   最终值: [%.4f, %.4f]\n", path.Values[numSteps][0], path.Values[numSteps][1])
	fmt.Printf("   理论期望: [%.4f, %.4f]\n", x0[0]*math.Exp(mu1*T), x0[1]*math.Exp(mu2*T))

	fmt.Println()
	fmt.Println("   噪声相关性验证（计算路径相关系数）:")
	corr := calculateCorrelation(path)
	fmt.Printf("   模拟路径相关系数: %.4f\n", corr)
	fmt.Printf("   目标相关系数:     %.4f\n", rho)
}

func demoOU2D() {
	theta := sde.Vector{1.0, 1.5}
	mu := sde.Vector{0.0, 0.0}
	sigma := sde.Vector{0.1, 0.15}
	rho := -0.5

	corrMatrix := sde.Matrix{
		{1.0, rho},
		{rho, 1.0},
	}

	drift := sde.DriftFunc(func(x sde.Vector, t float64) sde.Vector {
		return sde.Vector{
			theta[0] * (mu[0] - x[0]),
			theta[1] * (mu[1] - x[1]),
		}
	})

	diffusion := sde.DiffusionFunc(func(x sde.Vector, t float64) sde.Matrix {
		return sde.Matrix{
			{sigma[0], 0.0},
			{0.0, sigma[1]},
		}
	})

	x0 := sde.Vector{1.0, -1.0}
	t0 := 0.0
	T := 10.0
	numSteps := 2000

	config := sde.SDEConfig{
		Drift:      drift,
		Diffusion:  diffusion,
		X0:         x0,
		T0:         t0,
		T:          T,
		NumSteps:   numSteps,
		CorrMatrix: corrMatrix,
		Seed:       67890,
		UseSeed:    true,
	}

	path, err := sde.EulerMaruyama(config)
	if err != nil {
		fmt.Printf("   错误: %v\n", err)
		return
	}

	fmt.Printf("   相关系数 ρ = %.2f\n", rho)
	fmt.Printf("   初始值: [%.2f, %.2f], 终止时间: %.1f, 步数: %d\n", x0[0], x0[1], T, numSteps)
	fmt.Printf("   最终值: [%.4f, %.4f]\n", path.Values[numSteps][0], path.Values[numSteps][1])
	fmt.Printf("   理论均值回复到: [%.1f, %.1f]\n", mu[0], mu[1])
}

func demoSRI1GBM() {
	mu := 0.05
	sigma := 0.2

	drift := func(x, t float64) float64 {
		return mu * x
	}
	diffusion := func(x, t float64) float64 {
		return sigma * x
	}

	x0 := 100.0
	t0 := 0.0
	T := 1.0
	numSteps := 1000

	config := sde.SRIConfig{
		Drift:     drift,
		Diffusion: diffusion,
		X0:        x0,
		T0:        t0,
		T:         T,
		NumSteps:  numSteps,
		Seed:      42,
		UseSeed:   true,
		Order:     1,
	}

	path := sde.SRI1(config)

	fmt.Printf("   初始值: %.2f, 终止时间: %.1f, 步数: %d\n", x0, T, numSteps)
	fmt.Printf("   最终值: %.4f\n", path.Values[numSteps][0])
	fmt.Printf("   理论期望: %.4f\n", x0*math.Exp(mu*T))
}

func demoSRI2GBM() {
	mu := 0.05
	sigma := 0.2

	drift := func(x, t float64) float64 {
		return mu * x
	}
	diffusion := func(x, t float64) float64 {
		return sigma * x
	}

	x0 := 100.0
	t0 := 0.0
	T := 1.0
	numSteps := 1000

	config := sde.SRIConfig{
		Drift:     drift,
		Diffusion: diffusion,
		X0:        x0,
		T0:        t0,
		T:         T,
		NumSteps:  numSteps,
		Seed:      42,
		UseSeed:   true,
		Order:     2,
	}

	path := sde.SRI2(config)

	fmt.Printf("   初始值: %.2f, 终止时间: %.1f, 步数: %d\n", x0, T, numSteps)
	fmt.Printf("   最终值: %.4f\n", path.Values[numSteps][0])
	fmt.Printf("   理论期望: %.4f\n", x0*math.Exp(mu*T))
}

func demoConvergenceComparison() {
	mu := 0.05
	sigma := 0.2
	x0 := 100.0
	T := 1.0
	numPaths := 1000

	drift := func(x, t float64) float64 {
		return mu * x
	}
	diffusion := func(x, t float64) float64 {
		return sigma * x
	}

	stepSizes := []int{10, 20, 50, 100, 200}
	exactMean := x0 * math.Exp(mu*T)

	fmt.Println("   步长   | Euler误差 |  SRI1误差  |  SRI2误差")
	fmt.Println("   -------|-----------|------------|-----------")

	for _, numSteps := range stepSizes {
		dt := T / float64(numSteps)

		eulerSum := 0.0
		sri1Sum := 0.0
		sri2Sum := 0.0

		for seed := int64(0); seed < int64(numPaths); seed++ {
			eulerPath := sde.EulerMaruyama1DWithSeed(drift, diffusion, x0, 0, T, numSteps, seed)
			eulerSum += eulerPath.Values[numSteps][0]

			sri1Config := sde.SRIConfig{
				Drift:     drift,
				Diffusion: diffusion,
				X0:        x0,
				T0:        0,
				T:         T,
				NumSteps:  numSteps,
				Seed:      seed,
				UseSeed:   true,
				Order:     1,
			}
			sri1Path := sde.SRI1(sri1Config)
			sri1Sum += sri1Path.Values[numSteps][0]

			sri2Config := sde.SRIConfig{
				Drift:     drift,
				Diffusion: diffusion,
				X0:        x0,
				T0:        0,
				T:         T,
				NumSteps:  numSteps,
				Seed:      seed,
				UseSeed:   true,
				Order:     2,
			}
			sri2Path := sde.SRI2(sri2Config)
			sri2Sum += sri2Path.Values[numSteps][0]
		}

		eulerError := math.Abs(eulerSum/float64(numPaths) - exactMean)
		sri1Error := math.Abs(sri1Sum/float64(numPaths) - exactMean)
		sri2Error := math.Abs(sri2Sum/float64(numPaths) - exactMean)

		fmt.Printf("   dt=%.4f|  %8.6f |  %9.6f |  %8.6f\n",
			dt, eulerError, sri1Error, sri2Error)
	}

	fmt.Println()
	fmt.Println("   理论收敛阶:")
	fmt.Println("   - Euler-Maruyama: O(Δt)  (弱一阶)")
	fmt.Println("   - SRI1:           O(Δt)  (弱一阶)")
	fmt.Println("   - SRI2:           O(Δt²) (弱二阶)")
}

func demoCorrelatedGBMWithSRI2() {
	mu1, mu2 := 0.05, 0.08
	sigma1, sigma2 := 0.2, 0.3
	rho := 0.7

	corrMatrix := sde.Matrix{
		{1.0, rho},
		{rho, 1.0},
	}

	drift := sde.DriftFunc(func(x sde.Vector, t float64) sde.Vector {
		return sde.Vector{mu1 * x[0], mu2 * x[1]}
	})

	diffusion := sde.DiffusionFunc(func(x sde.Vector, t float64) sde.Matrix {
		return sde.Matrix{
			{sigma1 * x[0], 0.0},
			{0.0, sigma2 * x[1]},
		}
	})

	x0 := sde.Vector{100.0, 100.0}
	t0 := 0.0
	T := 1.0
	numSteps := 1000

	config := sde.SRIMultiConfig{
		Drift:      drift,
		Diffusion:  diffusion,
		X0:         x0,
		T0:         t0,
		T:          T,
		NumSteps:   numSteps,
		CorrMatrix: corrMatrix,
		Seed:       12345,
		UseSeed:    true,
		Order:      2,
	}

	path, err := sde.SRI2Multi(config)
	if err != nil {
		fmt.Printf("   错误: %v\n", err)
		return
	}

	fmt.Printf("   相关系数 ρ = %.2f\n", rho)
	fmt.Printf("   初始值: [%.2f, %.2f], 终止时间: %.1f, 步数: %d\n", x0[0], x0[1], T, numSteps)
	fmt.Printf("   最终值: [%.4f, %.4f]\n", path.Values[numSteps][0], path.Values[numSteps][1])
	fmt.Printf("   理论期望: [%.4f, %.4f]\n", x0[0]*math.Exp(mu1*T), x0[1]*math.Exp(mu2*T))
}

func calculateCorrelation(path *sde.Path) float64 {
	n := len(path.Values)
	sum1, sum2, sum12 := 0.0, 0.0, 0.0
	sum1Sq, sum2Sq := 0.0, 0.0

	for _, v := range path.Values {
		sum1 += v[0]
		sum2 += v[1]
		sum12 += v[0] * v[1]
		sum1Sq += v[0] * v[0]
		sum2Sq += v[1] * v[1]
	}

	cov := sum12/float64(n) - (sum1/float64(n))*(sum2/float64(n))
	std1 := math.Sqrt(sum1Sq/float64(n) - (sum1/float64(n))*(sum1/float64(n)))
	std2 := math.Sqrt(sum2Sq/float64(n) - (sum2/float64(n))*(sum2/float64(n)))

	return cov / (std1 * std2)
}
