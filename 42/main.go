package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

type IsingModel struct {
	L     int
	N     int
	Spin  [][]int
	Rand  *rand.Rand
}

func NewIsingModel(L int) *IsingModel {
	im := &IsingModel{
		L:    L,
		N:    L * L,
		Spin: make([][]int, L),
		Rand: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	for i := range im.Spin {
		im.Spin[i] = make([]int, L)
		for j := range im.Spin[i] {
			if im.Rand.Float64() < 0.5 {
				im.Spin[i][j] = 1
			} else {
				im.Spin[i][j] = -1
			}
		}
	}
	return im
}

func (im *IsingModel) idxUp(i int) int {
	return (i - 1 + im.L) % im.L
}

func (im *IsingModel) idxDown(i int) int {
	return (i + 1) % im.L
}

func (im *IsingModel) idxLeft(j int) int {
	return (j - 1 + im.L) % im.L
}

func (im *IsingModel) idxRight(j int) int {
	return (j + 1) % im.L
}

func (im *IsingModel) Magnetization() float64 {
	sum := 0
	for i := 0; i < im.L; i++ {
		for j := 0; j < im.L; j++ {
			sum += im.Spin[i][j]
		}
	}
	return float64(sum) / float64(im.N)
}

func (im *IsingModel) MagnetizationAbs() float64 {
	return math.Abs(im.Magnetization())
}

func (im *IsingModel) Energy() float64 {
	sum := 0
	for i := 0; i < im.L; i++ {
		for j := 0; j < im.L; j++ {
			up := im.idxUp(i)
			down := im.idxDown(i)
			left := im.idxLeft(j)
			right := im.idxRight(j)
			sum += -im.Spin[i][j] * (im.Spin[up][j] + im.Spin[down][j] + im.Spin[i][left] + im.Spin[i][right])
		}
	}
	return float64(sum) / float64(2*im.N)
}

func (im *IsingModel) deltaEnergy(i, j int) float64 {
	up := im.idxUp(i)
	down := im.idxDown(i)
	left := im.idxLeft(j)
	right := im.idxRight(j)

	neighbors := im.Spin[up][j] + im.Spin[down][j] + im.Spin[i][left] + im.Spin[i][right]
	return float64(2 * im.Spin[i][j] * neighbors)
}

func (im *IsingModel) MetropolisStep(beta float64) {
	i := im.Rand.Intn(im.L)
	j := im.Rand.Intn(im.L)

	dE := im.deltaEnergy(i, j)

	if dE <= 0 || im.Rand.Float64() < math.Exp(-beta*dE) {
		im.Spin[i][j] *= -1
	}
}

func (im *IsingModel) Sweep(beta float64) {
	for k := 0; k < im.N; k++ {
		im.MetropolisStep(beta)
	}
}

func (im *IsingModel) VerifyEnergyConservation(i, j int) (bool, float64, float64, float64) {
	E1 := im.Energy() * float64(im.N)
	dE := im.deltaEnergy(i, j)
	im.Spin[i][j] *= -1
	E2 := im.Energy() * float64(im.N)
	im.Spin[i][j] *= -1
	
	actualDelta := E2 - E1
	return math.Abs(actualDelta-dE) < 1e-10, dE, actualDelta, E1
}

type Result struct {
	Temperature     float64
	Magnetization   float64
	Magnetization2  float64
	Magnetization4  float64
	Energy          float64
	Energy2         float64
	Chi             float64
	SpecificHeat    float64
}

type FSSData struct {
	L         int
	T         float64
	M         float64
	Chi       float64
}

func Simulate(L int, temperatures []float64, thermalSweeps int, measureSweeps int, verbose bool) []Result {
	results := make([]Result, len(temperatures))
	N := float64(L * L)

	for idx, T := range temperatures {
		im := NewIsingModel(L)
		beta := 1.0 / T

		for i := 0; i < thermalSweeps; i++ {
			im.Sweep(beta)
		}

		var magSum, mag2Sum, mag4Sum, energySum, energy2Sum float64
		for i := 0; i < measureSweeps; i++ {
			im.Sweep(beta)
			m := im.Magnetization()
			e := im.Energy()
			magSum += math.Abs(m)
			mag2Sum += m * m
			mag4Sum += m * m * m * m
			energySum += e
			energy2Sum += e * e
		}

		nSweeps := float64(measureSweeps)
		M_avg := magSum / nSweeps
		M2_avg := mag2Sum / nSweeps
		M4_avg := mag4Sum / nSweeps
		E_avg := energySum / nSweeps
		E2_avg := energy2Sum / nSweeps

		chi := N * M2_avg / T
		cv := N * (E2_avg - E_avg*E_avg) / (T * T)

		results[idx] = Result{
			Temperature:     T,
			Magnetization:   M_avg,
			Magnetization2:  M2_avg,
			Magnetization4:  M4_avg,
			Energy:          E_avg,
			Energy2:         E2_avg,
			Chi:             chi,
			SpecificHeat:    cv,
		}

		if verbose {
			fmt.Printf("L=%d, T=%.3f, M=%.4f, χ=%.4f, E=%.4f, Cv=%.4f\n", 
				L, results[idx].Temperature, results[idx].Magnetization, 
				results[idx].Chi, results[idx].Energy, results[idx].SpecificHeat)
		}
	}

	return results
}

func linearRegression(x, y []float64) (slope, intercept, r2 float64) {
	n := len(x)
	if n != len(y) || n < 2 {
		return 0, 0, 0
	}

	var sumX, sumY, sumXY, sumX2, sumY2 float64
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}

	slope = (float64(n)*sumXY - sumX*sumY) / (float64(n)*sumX2 - sumX*sumX)
	intercept = (sumY - slope*sumX) / float64(n)

	yMean := sumY / float64(n)
	var ssTot, ssRes float64
	for i := 0; i < n; i++ {
		ssTot += (y[i] - yMean) * (y[i] - yMean)
		yPred := slope*x[i] + intercept
		ssRes += (y[i] - yPred) * (y[i] - yPred)
	}
	r2 = 1 - ssRes/ssTot

	return slope, intercept, r2
}

func findPeak(results []Result) (Tpeak float64, peakVal float64) {
	maxChi := -1.0
	Tpeak = results[0].Temperature
	for _, r := range results {
		if r.Chi > maxChi {
			maxChi = r.Chi
			Tpeak = r.Temperature
		}
	}
	return Tpeak, maxChi
}

func finiteSizeScaling(Ls []int, thermalSweeps, measureSweeps int) {
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("有限尺寸标度分析 (Finite Size Scaling)")
	fmt.Println(strings.Repeat("=", 70))
	
	Tc_exact := 2.0 / math.Log(1.0+math.Sqrt(2.0))
	beta_nu_exact := 1.0 / 8.0
	gamma_nu_exact := 7.0 / 4.0
	
	fmt.Printf("\n理论值 (2D Ising): Tc=%.4f, β/ν=%.4f, γ/ν=%.4f\n", 
		Tc_exact, beta_nu_exact, gamma_nu_exact)
	
	Tmin := 2.0
	Tmax := 2.6
	dT := 0.02
	
	temperatures := make([]float64, 0)
	for T := Tmin; T <= Tmax; T += dT {
		temperatures = append(temperatures, math.Round(T*100)/100)
	}
	
	allResults := make([][]Result, len(Ls))
	Tpeaks := make([]float64, len(Ls))
	ChiPeaks := make([]float64, len(Ls))
	
	for i, L := range Ls {
		fmt.Printf("\n正在模拟 L=%d...\n", L)
		allResults[i] = Simulate(L, temperatures, thermalSweeps, measureSweeps, false)
		Tpeaks[i], ChiPeaks[i] = findPeak(allResults[i])
		fmt.Printf("  L=%d: χ峰值位置 T=%.4f, χ_max=%.4f\n", L, Tpeaks[i], ChiPeaks[i])
	}
	
	fmt.Println("\n" + strings.Repeat("-", 70))
	fmt.Println("临界指数估计:")
	fmt.Println(strings.Repeat("-", 70))
	
	logL := make([]float64, len(Ls))
	logM_Tc := make([]float64, len(Ls))
	logChiPeak := make([]float64, len(Ls))
	
	for i, L := range Ls {
		logL[i] = math.Log(float64(L))
		
		var M_Tc float64
		minDiff := 1000.0
		for _, r := range allResults[i] {
			if math.Abs(r.Temperature-Tc_exact) < minDiff {
				minDiff = math.Abs(r.Temperature - Tc_exact)
				M_Tc = r.Magnetization
			}
		}
		logM_Tc[i] = math.Log(M_Tc)
		logChiPeak[i] = math.Log(ChiPeaks[i])
	}
	
	slope_M, _, r2_M := linearRegression(logL, logM_Tc)
	beta_nu := -slope_M
	
	slope_Chi, _, r2_Chi := linearRegression(logL, logChiPeak)
	gamma_nu := slope_Chi
	
	fmt.Printf("\nβ/ν 估计 (从 M(Tc) ~ L^(-β/ν)):\n")
	fmt.Printf("  斜率 = %.4f, β/ν = %.4f (R²=%.4f)\n", slope_M, beta_nu, r2_M)
	fmt.Printf("  理论值 β/ν = %.4f, 偏差 = %.2f%%\n", beta_nu_exact, 
		math.Abs(beta_nu-beta_nu_exact)/beta_nu_exact*100)
	
	fmt.Printf("\nγ/ν 估计 (从 χ_max ~ L^(γ/ν)):\n")
	fmt.Printf("  斜率 = %.4f, γ/ν = %.4f (R²=%.4f)\n", slope_Chi, gamma_nu, r2_Chi)
	fmt.Printf("  理论值 γ/ν = %.4f, 偏差 = %.2f%%\n", gamma_nu_exact, 
		math.Abs(gamma_nu-gamma_nu_exact)/gamma_nu_exact*100)
	
	fmt.Println("\n" + strings.Repeat("-", 70))
	fmt.Println("数据汇总表:")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("%6s | %10s | %10s | %10s | %10s\n", 
		"L", "T_peak(χ)", "χ_max", "M(Tc)", "log(L)")
	fmt.Println(strings.Repeat("-", 70))
	for i, L := range Ls {
		var M_Tc float64
		minDiff := 1000.0
		for _, r := range allResults[i] {
			if math.Abs(r.Temperature-Tc_exact) < minDiff {
				minDiff = math.Abs(r.Temperature - Tc_exact)
				M_Tc = r.Magnetization
			}
		}
		fmt.Printf("%6d | %10.4f | %10.4f | %10.4f | %10.4f\n", 
			L, Tpeaks[i], ChiPeaks[i], M_Tc, logL[i])
	}
}

func verifyIndices(L int) bool {
	im := NewIsingModel(L)
	ok := true
	
	for i := 0; i < L; i++ {
		if im.idxUp(i) != (i-1+L)%L {
			fmt.Printf("idxUp(%d) 错误: 预期%d, 实际%d\n", i, (i-1+L)%L, im.idxUp(i))
			ok = false
		}
		if im.idxDown(i) != (i+1)%L {
			fmt.Printf("idxDown(%d) 错误: 预期%d, 实际%d\n", i, (i+1)%L, im.idxDown(i))
			ok = false
		}
		if im.idxLeft(i) != (i-1+L)%L {
			fmt.Printf("idxLeft(%d) 错误: 预期%d, 实际%d\n", i, (i-1+L)%L, im.idxLeft(i))
			ok = false
		}
		if im.idxRight(i) != (i+1)%L {
			fmt.Printf("idxRight(%d) 错误: 预期%d, 实际%d\n", i, (i+1)%L, im.idxRight(i))
			ok = false
		}
	}
	
	if ok {
		fmt.Println("✓ 所有周期性边界索引计算正确!")
	}
	return ok
}

func main() {
	L := 5
	
	fmt.Println("验证周期性边界索引计算:")
	idxOk := verifyIndices(L)
	if !idxOk {
		fmt.Println("✗ 索引计算错误!")
		return
	}
	fmt.Println()
	
	im := NewIsingModel(L)
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			im.Spin[i][j] = 1
		}
	}
	
	fmt.Println("测试边界自旋能量守恒 (全向上自旋):")
	allPassed := true
	
	testPositions := [][]int{
		{0, 0}, {0, L - 1}, {L - 1, 0}, {L - 1, L - 1},
		{0, L / 2}, {L / 2, 0}, {L - 1, L / 2}, {L / 2, L - 1},
		{L / 2, L / 2},
	}
	
	for _, pos := range testPositions {
		i, j := pos[0], pos[1]
		passed, dE, actual, E1 := im.VerifyEnergyConservation(i, j)
		status := "✓ PASS"
		if !passed {
			status = "✗ FAIL"
			allPassed = false
		}
		fmt.Printf("  位置 (%d,%d): E=%.0f, deltaE计算=%.0f, 实际delta=%.0f, %s\n", i, j, E1, dE, actual, status)
	}
	
	if allPassed {
		fmt.Println("✓ 所有位置能量守恒测试通过!")
	} else {
		fmt.Println("✗ 发现能量不守恒问题!")
	}
	fmt.Println()

	Ls := []int{8, 16, 32, 64}
	thermalSweeps := 2000
	measureSweeps := 2000

	fmt.Println("二维Ising模型 - 有限尺寸标度分析")
	fmt.Printf("晶格尺寸: %v\n", Ls)
	fmt.Printf("热化步数: %d, 测量步数: %d\n", thermalSweeps, measureSweeps)

	start := time.Now()
	finiteSizeScaling(Ls, thermalSweeps, measureSweeps)
	fmt.Printf("\n" + strings.Repeat("=", 70) + "\n")
	fmt.Printf("总计算时间: %v\n", time.Since(start))
}
