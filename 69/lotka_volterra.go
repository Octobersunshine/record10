package main

import (
	"fmt"
	"math"
)

type Params struct {
	Alpha float64
	Beta  float64
	Gamma float64
	Delta float64
}

type DiffusionParams struct {
	DPrey     float64
	DPredator float64
	Dx        float64
	Nx        int
}

type State struct {
	Prey     float64
	Predator float64
}

type LogState struct {
	LogPrey     float64
	LogPredator float64
}

type SpatialState struct {
	Prey     []float64
	Predator []float64
}

func NewSpatialState(nx int) SpatialState {
	return SpatialState{
		Prey:     make([]float64, nx),
		Predator: make([]float64, nx),
	}
}

func toLogState(s State) LogState {
	const minPop = 1e-15
	return LogState{
		LogPrey:     math.Log(math.Max(minPop, s.Prey)),
		LogPredator: math.Log(math.Max(minPop, s.Predator)),
	}
}

func fromLogState(ls LogState) State {
	return State{
		Prey:     math.Exp(ls.LogPrey),
		Predator: math.Exp(ls.LogPredator),
	}
}

func LotkaVolterraLog(t float64, ls LogState, p Params) LogState {
	prey := math.Exp(ls.LogPrey)
	predator := math.Exp(ls.LogPredator)
	
	dPreyDt := p.Alpha - p.Beta*predator
	dPredatorDt := p.Delta*prey - p.Gamma
	
	return LogState{
		LogPrey:     dPreyDt,
		LogPredator: dPredatorDt,
	}
}

func RK4Log(t float64, ls LogState, dt float64, p Params) LogState {
	k1 := LotkaVolterraLog(t, ls, p)
	k2 := LotkaVolterraLog(t+dt/2, LogState{
		LogPrey:     ls.LogPrey + dt*k1.LogPrey/2,
		LogPredator: ls.LogPredator + dt*k1.LogPredator/2,
	}, p)
	k3 := LotkaVolterraLog(t+dt/2, LogState{
		LogPrey:     ls.LogPrey + dt*k2.LogPrey/2,
		LogPredator: ls.LogPredator + dt*k2.LogPredator/2,
	}, p)
	k4 := LotkaVolterraLog(t+dt, LogState{
		LogPrey:     ls.LogPrey + dt*k3.LogPrey,
		LogPredator: ls.LogPredator + dt*k3.LogPredator,
	}, p)

	return LogState{
		LogPrey:     ls.LogPrey + dt*(k1.LogPrey+2*k2.LogPrey+2*k3.LogPrey+k4.LogPrey)/6,
		LogPredator: ls.LogPredator + dt*(k1.LogPredator+2*k2.LogPredator+2*k3.LogPredator+k4.LogPredator)/6,
	}
}

func LotkaVolterra(t float64, s State, p Params, dt float64) State {
	const epsilon = 1e-10
	const softZero = 1e-100
	
	prey := math.Max(softZero, s.Prey)
	predator := math.Max(softZero, s.Predator)
	
	preyRate := p.Alpha - p.Beta*predator
	predatorRate := p.Delta*prey - p.Gamma
	
	preyDt := preyRate * prey
	predatorDt := predatorRate * predator
	
	if prey < epsilon && preyDt < 0 {
		preyDt = -prey / math.Max(dt, 1e-6) * 0.5
	}
	if predator < epsilon && predatorDt < 0 {
		predatorDt = -predator / math.Max(dt, 1e-6) * 0.5
	}
	
	return State{
		Prey:     preyDt,
		Predator: predatorDt,
	}
}

func RK4(t float64, s State, dt float64, p Params) State {
	const softZero = 1e-100
	
	safeState := State{
		Prey:     math.Max(softZero, s.Prey),
		Predator: math.Max(softZero, s.Predator),
	}
	
	k1 := LotkaVolterra(t, safeState, p, dt)
	k2 := LotkaVolterra(t+dt/2, State{
		Prey:     math.Max(softZero, safeState.Prey + dt*k1.Prey/2),
		Predator: math.Max(softZero, safeState.Predator + dt*k1.Predator/2),
	}, p, dt)
	k3 := LotkaVolterra(t+dt/2, State{
		Prey:     math.Max(softZero, safeState.Prey + dt*k2.Prey/2),
		Predator: math.Max(softZero, safeState.Predator + dt*k2.Predator/2),
	}, p, dt)
	k4 := LotkaVolterra(t+dt, State{
		Prey:     math.Max(softZero, safeState.Prey + dt*k3.Prey),
		Predator: math.Max(softZero, safeState.Predator + dt*k3.Predator),
	}, p, dt)

	result := State{
		Prey:     safeState.Prey + dt*(k1.Prey+2*k2.Prey+2*k3.Prey+k4.Prey)/6,
		Predator: safeState.Predator + dt*(k1.Predator+2*k2.Predator+2*k3.Predator+k4.Predator)/6,
	}
	
	return State{
		Prey:     math.Max(softZero, result.Prey),
		Predator: math.Max(softZero, result.Predator),
	}
}

func Solve(initial State, p Params, dt float64, steps int) ([]float64, []float64, []float64) {
	time := make([]float64, steps+1)
	prey := make([]float64, steps+1)
	predator := make([]float64, steps+1)

	time[0] = 0.0
	prey[0] = initial.Prey
	predator[0] = initial.Predator

	currentLog := toLogState(initial)
	for i := 0; i < steps; i++ {
		currentLog = RK4Log(time[i], currentLog, dt, p)
		time[i+1] = time[i] + dt
		current := fromLogState(currentLog)
		prey[i+1] = current.Prey
		predator[i+1] = current.Predator
	}

	return time, prey, predator
}

func diffusion1D(u []float64, dx float64) []float64 {
	n := len(u)
	laplacian := make([]float64, n)
	dx2 := dx * dx

	for i := 1; i < n-1; i++ {
		laplacian[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx2
	}

	laplacian[0] = (u[1] - 2*u[0] + u[0]) / dx2
	laplacian[n-1] = (u[n-1] - 2*u[n-1] + u[n-2]) / dx2

	return laplacian
}

func reactionTerm(s SpatialState, p Params, i int) (float64, float64) {
	const softZero = 1e-100
	prey := math.Max(softZero, s.Prey[i])
	predator := math.Max(softZero, s.Predator[i])

	dPreyDt := p.Alpha*prey - p.Beta*prey*predator
	dPredatorDt := p.Delta*prey*predator - p.Gamma*predator

	return dPreyDt, dPredatorDt
}

func SolvePDE(initial SpatialState, p Params, dp DiffusionParams, dt float64, steps int) ([]float64, []SpatialState) {
	nx := dp.Nx
	time := make([]float64, steps+1)
	history := make([]SpatialState, steps+1)

	history[0] = NewSpatialState(nx)
	for i := 0; i < nx; i++ {
		history[0].Prey[i] = initial.Prey[i]
		history[0].Predator[i] = initial.Predator[i]
	}

	current := NewSpatialState(nx)
	for i := 0; i < nx; i++ {
		current.Prey[i] = initial.Prey[i]
		current.Predator[i] = initial.Predator[i]
	}

	for t := 0; t < steps; t++ {
		preyLaplacian := diffusion1D(current.Prey, dp.Dx)
		predLaplacian := diffusion1D(current.Predator, dp.Dx)

		next := NewSpatialState(nx)
		for i := 0; i < nx; i++ {
			rPrey, rPred := reactionTerm(current, p, i)

			next.Prey[i] = current.Prey[i] + dt*(rPrey+dp.DPrey*preyLaplacian[i])
			next.Predator[i] = current.Predator[i] + dt*(rPred+dp.DPredator*predLaplacian[i])

			const softZero = 1e-100
			next.Prey[i] = math.Max(softZero, next.Prey[i])
			next.Predator[i] = math.Max(softZero, next.Predator[i])
		}

		current = next
		time[t+1] = float64(t+1) * dt
		history[t+1] = NewSpatialState(nx)
		for i := 0; i < nx; i++ {
			history[t+1].Prey[i] = current.Prey[i]
			history[t+1].Predator[i] = current.Predator[i]
		}
	}

	return time, history
}

func gaussian(x, x0, sigma float64) float64 {
	return math.Exp(-(x-x0)*(x-x0)/(2*sigma*sigma))
}

func main() {
	fmt.Println("=== 敏感性测试: 低增长率 (Alpha=0.1) ===")
	params1 := Params{
		Alpha: 0.1,
		Beta:  0.1,
		Gamma: 1.5,
		Delta: 0.075,
	}
	initial1 := State{Prey: 5.0, Predator: 20.0}
	dt := 0.01
	steps := 2000

	_, prey1, pred1 := Solve(initial1, params1, dt, steps)
	
	hasNegative := false
	minPrey1 := math.Inf(1)
	minPred1 := math.Inf(1)
	for i := 0; i <= steps; i++ {
		if prey1[i] < minPrey1 {
			minPrey1 = prey1[i]
		}
		if pred1[i] < minPred1 {
			minPred1 = pred1[i]
		}
		if prey1[i] < 0 || pred1[i] < 0 {
			hasNegative = true
		}
	}
	fmt.Printf("最小猎物: %.6f, 最小捕食者: %.6f, 有负值: %v\n", minPrey1, minPred1, hasNegative)
	if hasNegative {
		fmt.Println("警告: 检测到负值种群！")
	} else {
		fmt.Println("✓ 数值稳定，无负值种群")
	}

	fmt.Println("\n=== 极端灭绝测试 ===")
	params2 := Params{Alpha: 0.01, Beta: 0.5, Gamma: 0.1, Delta: 0.01}
	initial2 := State{Prey: 1.0, Predator: 100.0}
	_, prey2, pred2 := Solve(initial2, params2, 0.01, 5000)
	
	minPrey2 := math.Inf(1)
	minPred2 := math.Inf(1)
	hasNegative2 := false
	for i := 0; i <= 5000; i++ {
		if prey2[i] < minPrey2 {
			minPrey2 = prey2[i]
		}
		if pred2[i] < minPred2 {
			minPred2 = pred2[i]
		}
		if prey2[i] < 0 || pred2[i] < 0 {
			hasNegative2 = true
		}
	}
	fmt.Printf("最小猎物: %.6e, 最小捕食者: %.6e, 有负值: %v\n", minPrey2, minPred2, hasNegative2)
	if hasNegative2 {
		fmt.Println("警告: 检测到负值种群！")
	} else {
		fmt.Println("✓ 数值稳定，无负值种群")
	}

	fmt.Println("\n=== 正常参数演示 ===")
	params3 := Params{Alpha: 1.0, Beta: 0.1, Gamma: 1.5, Delta: 0.075}
	initial3 := State{Prey: 40.0, Predator: 9.0}
	time3, prey3, pred3 := Solve(initial3, params3, 0.01, 1000)
	
	fmt.Println("Time\tPrey\tPredator")
	for i := 0; i <= 1000; i += 100 {
		fmt.Printf("%.2f\t%.2f\t%.2f\n", time3[i], prey3[i], pred3[i])
	}
	
	maxPrey := 0.0
	minPrey := math.Inf(1)
	maxPred := 0.0
	minPred := math.Inf(1)
	for i := 0; i <= 1000; i++ {
		if prey3[i] > maxPrey {
			maxPrey = prey3[i]
		}
		if prey3[i] < minPrey {
			minPrey = prey3[i]
		}
		if pred3[i] > maxPred {
			maxPred = pred3[i]
		}
		if pred3[i] < minPred {
			minPred = pred3[i]
		}
	}
	fmt.Printf("\n种群范围:\n")
	fmt.Printf("猎物: %.2f ~ %.2f\n", minPrey, maxPrey)
	fmt.Printf("捕食者: %.2f ~ %.2f\n", minPred, maxPred)

	fmt.Println("\n=== 反应-扩散PDE: 行波入侵测试 ===")
	nx := 200
	dx := 0.5
	dtPDE := 0.001
	stepsPDE := 5000

	paramsPDE := Params{Alpha: 1.0, Beta: 0.1, Gamma: 0.5, Delta: 0.1}
	diffParams := DiffusionParams{DPrey: 1.0, DPredator: 0.5, Dx: dx, Nx: nx}

	initialSpatial := NewSpatialState(nx)
	for i := 0; i < nx; i++ {
		x := float64(i) * dx
		initialSpatial.Prey[i] = 10.0 * gaussian(x, 25.0, 5.0)
		initialSpatial.Predator[i] = 2.0 * gaussian(x, 25.0, 5.0)
	}

	_, history := SolvePDE(initialSpatial, paramsPDE, diffParams, dtPDE, stepsPDE)

	t0 := history[0]
	tMid := history[stepsPDE/2]
	tEnd := history[stepsPDE]

	fmt.Println("波前位置（猎物峰值）:")
	fmt.Printf("  t=0:    x=%.1f\n", findPeakPosition(t0.Prey, dx))
	fmt.Printf("  t=%.1f: x=%.1f\n", float64(stepsPDE/2)*dtPDE, findPeakPosition(tMid.Prey, dx))
	fmt.Printf("  t=%.1f: x=%.1f\n", float64(stepsPDE)*dtPDE, findPeakPosition(tEnd.Prey, dx))

	peak0 := findPeakPosition(t0.Prey, dx)
	peakEnd := findPeakPosition(tEnd.Prey, dx)
	waveSpeed := (peakEnd - peak0) / (float64(stepsPDE) * dtPDE)
	fmt.Printf("波速估计: %.3f 单位/时间\n", waveSpeed)

	fmt.Println("\n空间分布（x位置:猎物,捕食者）:")
	for i := 0; i < nx; i += 20 {
		fmt.Printf("  x=%.1f: %.2f, %.2f\n", float64(i)*dx, tEnd.Prey[i], tEnd.Predator[i])
	}
}

func findPeakPosition(u []float64, dx float64) float64 {
	maxIdx := 0
	maxVal := u[0]
	for i := 1; i < len(u); i++ {
		if u[i] > maxVal {
			maxVal = u[i]
			maxIdx = i
		}
	}
	return float64(maxIdx) * dx
}
