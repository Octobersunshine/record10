package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type ThermostatType int

const (
	ThermostatNone ThermostatType = iota
	ThermostatBerendsen
	ThermostatLangevin
)

type UmbrellaPotential struct {
	Active   bool
	AtomA    int
	AtomB    int
	Kspring  float64
	R0       float64
	MinDist  float64
	MaxDist  float64
}

type System struct {
	N              int
	Dim            int
	Box            []float64
	Pos            [][]float64
	Vel            [][]float64
	Force          [][]float64
	Mass           float64
	Epsilon        float64
	Sigma          float64
	Rcut           float64
	DT             float64
	Time           float64
	KBTarget       float64
	Thermostat     ThermostatType
	Tau            float64
	Gamma          float64
	RandGen        *rand.Rand
	Umbrella       UmbrellaPotential
	TrajRc         []float64
	VelHistory     [][]float64
	VelHistorySize int
}

type Trajectory struct {
	Time     []float64
	Temp     []float64
	Pressure []float64
	Rc       []float64
	VACF     []float64
}

type WHAMResult struct {
	Bins    []float64
	PMF     []float64
	Rho     []float64
	Weights []float64
}

func NewSystem(n, dim int, box []float64, mass, epsilon, sigma, rcut, dt, kbTarget float64) *System {
	sys := &System{
		N:              n,
		Dim:            dim,
		Box:            make([]float64, dim),
		Pos:            make([][]float64, n),
		Vel:            make([][]float64, n),
		Force:          make([][]float64, n),
		Mass:           mass,
		Epsilon:        epsilon,
		Sigma:          sigma,
		Rcut:           rcut,
		DT:             dt,
		Time:           0.0,
		KBTarget:       kbTarget,
		Thermostat:     ThermostatNone,
		Tau:            100.0 * dt,
		Gamma:          0.1,
		RandGen:        rand.New(rand.NewSource(time.Now().UnixNano())),
		Umbrella:       UmbrellaPotential{Active: false},
		VelHistorySize: 1000,
		VelHistory:     make([][]float64, 0, 1000),
		TrajRc:         make([]float64, 0),
	}
	copy(sys.Box, box)
	for i := 0; i < n; i++ {
		sys.Pos[i] = make([]float64, dim)
		sys.Vel[i] = make([]float64, dim)
		sys.Force[i] = make([]float64, dim)
	}
	return sys
}

func (sys *System) SetBerendsenThermostat(tau float64) {
	sys.Thermostat = ThermostatBerendsen
	sys.Tau = tau
}

func (sys *System) SetLangevinThermostat(gamma float64) {
	sys.Thermostat = ThermostatLangevin
	sys.Gamma = gamma
}

func (sys *System) SetUmbrellaPotential(atomA, atomB int, kspring, r0, minDist, maxDist float64) {
	sys.Umbrella = UmbrellaPotential{
		Active:   true,
		AtomA:    atomA,
		AtomB:    atomB,
		Kspring:  kspring,
		R0:       r0,
		MinDist:  minDist,
		MaxDist:  maxDist,
	}
}

func (sys *System) RemoveUmbrellaPotential() {
	sys.Umbrella.Active = false
}

func (sys *System) GetReactionCoordinate() float64 {
	if !sys.Umbrella.Active {
		return 0.0
	}
	dr := make([]float64, sys.Dim)
	r2 := 0.0
	for d := 0; d < sys.Dim; d++ {
		dr[d] = sys.Pos[sys.Umbrella.AtomA][d] - sys.Pos[sys.Umbrella.AtomB][d]
		dr[d] -= sys.Box[d] * math.Round(dr[d]/sys.Box[d])
		r2 += dr[d] * dr[d]
	}
	return math.Sqrt(r2)
}

func (sys *System) InitPositionsLigandProtein(spacing float64, ligandDist float64) {
	nProtein := sys.N - 1
	nPerDim := int(math.Ceil(math.Pow(float64(nProtein), 1.0/float64(sys.Dim))))
	idx := 0
	center := make([]float64, sys.Dim)
	for d := 0; d < sys.Dim; d++ {
		center[d] = sys.Box[d] / 2.0
	}
	for i := 0; i < nPerDim && idx < nProtein; i++ {
		for j := 0; j < nPerDim && idx < nProtein; j++ {
			if sys.Dim == 2 {
				sys.Pos[idx][0] = center[0] + (float64(i)-float64(nPerDim)/2.0)*spacing
				sys.Pos[idx][1] = center[1] + (float64(j)-float64(nPerDim)/2.0)*spacing
				idx++
			} else {
				for k := 0; k < nPerDim && idx < nProtein; k++ {
					sys.Pos[idx][0] = center[0] + (float64(i)-float64(nPerDim)/2.0)*spacing
					sys.Pos[idx][1] = center[1] + (float64(j)-float64(nPerDim)/2.0)*spacing
					sys.Pos[idx][2] = center[2] + (float64(k)-float64(nPerDim)/2.0)*spacing
					idx++
				}
			}
		}
	}
	ligandIdx := sys.N - 1
	for d := 0; d < sys.Dim; d++ {
		sys.Pos[ligandIdx][d] = center[d]
	}
	sys.Pos[ligandIdx][0] += ligandDist
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			sys.Pos[i][d] -= sys.Box[d] * math.Floor(sys.Pos[i][d]/sys.Box[d])
		}
	}
}

func (sys *System) InitPositions(spacing float64) {
	nPerDim := int(math.Ceil(math.Pow(float64(sys.N), 1.0/float64(sys.Dim))))
	idx := 0
	for i := 0; i < nPerDim && idx < sys.N; i++ {
		for j := 0; j < nPerDim && idx < sys.N; j++ {
			if sys.Dim == 2 {
				sys.Pos[idx][0] = (float64(i) + 0.5) * spacing
				sys.Pos[idx][1] = (float64(j) + 0.5) * spacing
				idx++
			} else {
				for k := 0; k < nPerDim && idx < sys.N; k++ {
					sys.Pos[idx][0] = (float64(i) + 0.5) * spacing
					sys.Pos[idx][1] = (float64(j) + 0.5) * spacing
					sys.Pos[idx][2] = (float64(k) + 0.5) * spacing
					idx++
				}
			}
		}
	}
}

func (sys *System) InitVelocities(seed int64) {
	sys.RandGen = rand.New(rand.NewSource(seed))
	cmVel := make([]float64, sys.Dim)
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			sys.Vel[i][d] = sys.RandGen.Float64() - 0.5
			cmVel[d] += sys.Vel[i][d]
		}
	}
	for d := 0; d < sys.Dim; d++ {
		cmVel[d] /= float64(sys.N)
	}
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			sys.Vel[i][d] -= cmVel[d]
		}
	}
	currentKb := sys.ComputeTemperature()
	scale := math.Sqrt(sys.KBTarget / currentKb)
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			sys.Vel[i][d] *= scale
		}
	}
}

func (sys *System) ComputeForces() (float64, float64, float64) {
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			sys.Force[i][d] = 0.0
		}
	}
	potential := 0.0
	virial := 0.0
	rcut2 := sys.Rcut * sys.Rcut
	for i := 0; i < sys.N; i++ {
		for j := i + 1; j < sys.N; j++ {
			dr := make([]float64, sys.Dim)
			r2 := 0.0
			for d := 0; d < sys.Dim; d++ {
				dr[d] = sys.Pos[i][d] - sys.Pos[j][d]
				dr[d] -= sys.Box[d] * math.Round(dr[d]/sys.Box[d])
				r2 += dr[d] * dr[d]
			}
			if r2 < rcut2 {
				r := math.Sqrt(r2)
				r6 := math.Pow(sys.Sigma/r, 6)
				r12 := r6 * r6
				pairPotential := 4.0 * sys.Epsilon * (r12 - r6)
				potential += pairPotential
				fMag := 24.0 * sys.Epsilon * (2.0*r12 - r6) / r2
				virial += fMag * r2
				for d := 0; d < sys.Dim; d++ {
					f := fMag * dr[d]
					sys.Force[i][d] += f
					sys.Force[j][d] -= f
				}
			}
		}
	}
	umbrellaEnergy := 0.0
	if sys.Umbrella.Active {
		dr := make([]float64, sys.Dim)
		r2 := 0.0
		for d := 0; d < sys.Dim; d++ {
			dr[d] = sys.Pos[sys.Umbrella.AtomA][d] - sys.Pos[sys.Umbrella.AtomB][d]
			dr[d] -= sys.Box[d] * math.Round(dr[d]/sys.Box[d])
			r2 += dr[d] * dr[d]
		}
		r := math.Sqrt(r2)
		if r >= sys.Umbrella.MinDist && r <= sys.Umbrella.MaxDist {
			umbrellaEnergy = 0.5 * sys.Umbrella.Kspring * (r - sys.Umbrella.R0) * (r - sys.Umbrella.R0)
			if r > 1e-10 {
				fMag := sys.Umbrella.Kspring * (r - sys.Umbrella.R0) / r
				for d := 0; d < sys.Dim; d++ {
					f := fMag * dr[d]
					sys.Force[sys.Umbrella.AtomA][d] += f
					sys.Force[sys.Umbrella.AtomB][d] -= f
				}
			}
		}
	}
	return potential, virial, umbrellaEnergy
}

func (sys *System) ComputeTemperature() float64 {
	ke := 0.0
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			ke += 0.5 * sys.Mass * sys.Vel[i][d] * sys.Vel[i][d]
		}
	}
	dof := float64(sys.N*sys.Dim - sys.Dim)
	return 2.0 * ke / dof
}

func (sys *System) ComputePressure(virial float64, temp float64) float64 {
	volume := 1.0
	for d := 0; d < sys.Dim; d++ {
		volume *= sys.Box[d]
	}
	return (float64(sys.N)*temp + virial/float64(sys.Dim)) / volume
}

func (sys *System) applyThermostat() {
	switch sys.Thermostat {
	case ThermostatBerendsen:
		currentTemp := sys.ComputeTemperature()
		if currentTemp > 1e-10 {
			lambda := math.Sqrt(1.0 + (sys.DT/sys.Tau)*(sys.KBTarget/currentTemp-1.0))
			for i := 0; i < sys.N; i++ {
				for d := 0; d < sys.Dim; d++ {
					sys.Vel[i][d] *= lambda
				}
			}
		}
	case ThermostatLangevin:
		sigmaLangevin := math.Sqrt(2.0 * sys.Gamma * sys.KBTarget / sys.Mass)
		coeff1 := math.Exp(-sys.Gamma * sys.DT)
		coeff2 := sigmaLangevin * math.Sqrt((1.0 - coeff1*coeff1) / (2.0 * sys.Gamma))
		for i := 0; i < sys.N; i++ {
			for d := 0; d < sys.Dim; d++ {
				xi := sys.RandGen.NormFloat64()
				sys.Vel[i][d] = coeff1*sys.Vel[i][d] + coeff2*xi
			}
		}
	}
}

func (sys *System) saveVelocities() {
	velSnapshot := make([]float64, sys.N*sys.Dim)
	idx := 0
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			velSnapshot[idx] = sys.Vel[i][d]
			idx++
		}
	}
	if len(sys.VelHistory) >= sys.VelHistorySize {
		sys.VelHistory = sys.VelHistory[1:]
	}
	sys.VelHistory = append(sys.VelHistory, velSnapshot)
}

func (sys *System) computeVACF() []float64 {
	if len(sys.VelHistory) < 2 {
		return nil
	}
	maxTau := len(sys.VelHistory) / 2
	vacf := make([]float64, maxTau)
	v0 := sys.VelHistory[0]
	for tau := 0; tau < maxTau; tau++ {
		vt := sys.VelHistory[tau]
		corr := 0.0
		for i := range v0 {
			corr += v0[i] * vt[i]
		}
		vacf[tau] = corr / float64(len(v0))
	}
	norm := vacf[0]
	if norm > 1e-10 {
		for tau := range vacf {
			vacf[tau] /= norm
		}
	}
	return vacf
}

func (sys *System) VerletStep() float64 {
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			sys.Vel[i][d] += 0.5 * sys.DT * sys.Force[i][d] / sys.Mass
			sys.Pos[i][d] += sys.DT * sys.Vel[i][d]
			sys.Pos[i][d] -= sys.Box[d] * math.Floor(sys.Pos[i][d]/sys.Box[d])
		}
	}
	sys.applyThermostat()
	_, _, umbrellaEnergy := sys.ComputeForces()
	for i := 0; i < sys.N; i++ {
		for d := 0; d < sys.Dim; d++ {
			sys.Vel[i][d] += 0.5 * sys.DT * sys.Force[i][d] / sys.Mass
		}
	}
	sys.Time += sys.DT
	return umbrellaEnergy
}

func (sys *System) Run(nSteps int, sampleInterval int, saveRc bool) *Trajectory {
	traj := &Trajectory{
		Time:     make([]float64, 0),
		Temp:     make([]float64, 0),
		Pressure: make([]float64, 0),
		Rc:       make([]float64, 0),
		VACF:     nil,
	}
	sys.VelHistory = sys.VelHistory[:0]
	sys.TrajRc = sys.TrajRc[:0]
	sys.ComputeForces()
	for step := 0; step < nSteps; step++ {
		sys.VerletStep()
		if step%sampleInterval == 0 {
			temp := sys.ComputeTemperature()
			_, virial, _ := sys.ComputeForces()
			pressure := sys.ComputePressure(virial, temp)
			traj.Time = append(traj.Time, sys.Time)
			traj.Temp = append(traj.Temp, temp)
			traj.Pressure = append(traj.Pressure, pressure)
			if saveRc && sys.Umbrella.Active {
				rc := sys.GetReactionCoordinate()
				traj.Rc = append(traj.Rc, rc)
				sys.TrajRc = append(sys.TrajRc, rc)
			}
			sys.saveVelocities()
		}
	}
	traj.VACF = sys.computeVACF()
	return traj
}

func RunUmbrellaSampling(nWindows int, r0Min, r0Max, kspring float64, nStepsEq, nStepsProd int) [][]float64 {
	nParticles := 10
	dim := 3
	boxSize := 15.0
	box := make([]float64, dim)
	for d := 0; d < dim; d++ {
		box[d] = boxSize
	}
	mass := 1.0
	epsilon := 1.0
	sigma := 1.0
	rcut := 2.5 * sigma
	dt := 0.005
	kbTarget := 1.0
	allRc := make([][]float64, nWindows)
	dr0 := (r0Max - r0Min) / float64(nWindows-1)
	for w := 0; w < nWindows; w++ {
		r0 := r0Min + float64(w)*dr0
		sys := NewSystem(nParticles, dim, box, mass, epsilon, sigma, rcut, dt, kbTarget)
		spacing := 1.2 * sigma
		sys.InitPositionsLigandProtein(spacing, r0)
		seed := int64(42 + w)
		sys.InitVelocities(seed)
		sys.SetLangevinThermostat(1.0)
		atomA := 0
		atomB := nParticles - 1
		sys.SetUmbrellaPotential(atomA, atomB, kspring, r0, r0Min-0.5, r0Max+0.5)
		sys.Run(nStepsEq, 100, false)
		traj := sys.Run(nStepsProd, 10, true)
		allRc[w] = make([]float64, len(traj.Rc))
		copy(allRc[w], traj.Rc)
		fmt.Printf("窗口 %d/%d: r0=%.2f, 采样点=%d\n", w+1, nWindows, r0, len(traj.Rc))
	}
	return allRc
}

func WHAM(allRc [][]float64, kspring float64, r0Min, r0Max float64, nBins int, tol float64, maxIter int) *WHAMResult {
	nWindows := len(allRc)
	if nWindows == 0 {
		return nil
	}
	dr0 := (r0Max - r0Min) / float64(nWindows-1)
	r0List := make([]float64, nWindows)
	for w := 0; w < nWindows; w++ {
		r0List[w] = r0Min + float64(w)*dr0
	}
	binWidth := (r0Max - r0Min) / float64(nBins)
	bins := make([]float64, nBins)
	for i := 0; i < nBins; i++ {
		bins[i] = r0Min + float64(i)*binWidth + binWidth/2.0
	}
	histograms := make([][]int, nWindows)
	for w := 0; w < nWindows; w++ {
		histograms[w] = make([]int, nBins)
		for _, r := range allRc[w] {
			binIdx := int(math.Floor((r - r0Min) / binWidth))
			if binIdx >= 0 && binIdx < nBins {
				histograms[w][binIdx]++
			}
		}
	}
	nPointsPerWindow := make([]int, nWindows)
	for w := 0; w < nWindows; w++ {
		nPointsPerWindow[w] = len(allRc[w])
	}
	F := make([]float64, nWindows)
	rho := make([]float64, nBins)
	pmf := make([]float64, nBins)
	for iter := 0; iter < maxIter; iter++ {
		for i := 0; i < nBins; i++ {
			r := bins[i]
			numerator := 0.0
			denominator := 0.0
			for w := 0; w < nWindows; w++ {
				if nPointsPerWindow[w] == 0 {
					continue
				}
				bias := 0.5 * kspring * (r - r0List[w]) * (r - r0List[w])
				numerator += float64(histograms[w][i])
				denominator += float64(nPointsPerWindow[w]) * math.Exp(F[w]-bias)
			}
			if denominator > 1e-10 {
				rho[i] = numerator / denominator
			} else {
				rho[i] = 0.0
			}
		}
		oldF := make([]float64, nWindows)
		copy(oldF, F)
		for w := 0; w < nWindows; w++ {
			if nPointsPerWindow[w] == 0 {
				continue
			}
			sum := 0.0
			for i := 0; i < nBins; i++ {
				if rho[i] > 1e-10 {
					r := bins[i]
					bias := 0.5 * kspring * (r - r0List[w]) * (r - r0List[w])
					sum += rho[i] * math.Exp(-bias)
				}
			}
			if sum > 1e-10 {
				F[w] = -math.Log(sum / binWidth)
			}
		}
		maxDiff := 0.0
		for w := 0; w < nWindows; w++ {
			diff := math.Abs(F[w] - oldF[w])
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if maxDiff < tol {
			fmt.Printf("WHAM收敛于 %d 次迭代，最大误差: %.2e\n", iter+1, maxDiff)
			break
		}
	}
	for i := 0; i < nBins; i++ {
		if rho[i] > 1e-10 {
			pmf[i] = -math.Log(rho[i])
		} else {
			pmf[i] = math.Inf(1)
		}
	}
	minPmf := math.Inf(1)
	for i := 0; i < nBins; i++ {
		if pmf[i] < minPmf {
			minPmf = pmf[i]
		}
	}
	for i := 0; i < nBins; i++ {
		pmf[i] -= minPmf
	}
	return &WHAMResult{
		Bins:    bins,
		PMF:     pmf,
		Rho:     rho,
		Weights: F,
	}
}

func main() {
	fmt.Println("=== 分子动力学：自由能计算（伞形采样 + WHAM）===")
	fmt.Println()
	fmt.Println("模拟系统：类蛋白团簇 + 配体粒子")
	fmt.Println("反应坐标：配体与团簇中心粒子的距离")
	fmt.Println()
	nWindows := 15
	r0Min := 1.0
	r0Max := 8.0
	kspring := 20.0
	nStepsEq := 1000
	nStepsProd := 2000
	fmt.Printf("伞形采样参数:\n")
	fmt.Printf("  窗口数: %d\n", nWindows)
	fmt.Printf("  反应坐标范围: [%.1f, %.1f]\n", r0Min, r0Max)
	fmt.Printf("  伞形势能常数: %.1f\n", kspring)
	fmt.Printf("  平衡步数: %d\n", nStepsEq)
	fmt.Printf("  生产步数: %d\n", nStepsProd)
	fmt.Println()
	fmt.Println("开始伞形采样...")
	allRc := RunUmbrellaSampling(nWindows, r0Min, r0Max, kspring, nStepsEq, nStepsProd)
	fmt.Println()
	fmt.Println("开始WHAM分析...")
	nBins := 50
	tol := 1e-6
	maxIter := 10000
	whamResult := WHAM(allRc, kspring, r0Min, r0Max, nBins, tol, maxIter)
	if whamResult != nil {
		fmt.Println()
		fmt.Println("=== 势能平均力(PMF)结果 ===")
		fmt.Println("距离 r    自由能 A(r)")
		fmt.Println("------------------------")
		step := 5
		for i := 0; i < nBins; i += step {
			if !math.IsInf(whamResult.PMF[i], 1) {
				fmt.Printf("%.2f      %.4f\n", whamResult.Bins[i], whamResult.PMF[i])
			}
		}
		minIdx := 0
		minVal := math.Inf(1)
		for i := 0; i < nBins; i++ {
			if whamResult.PMF[i] < minVal && whamResult.Bins[i] < 3.0 {
				minVal = whamResult.PMF[i]
				minIdx = i
			}
		}
		boundFreeEnergy := 0.0
		for i := nBins - 1; i >= 0; i-- {
			if whamResult.Bins[i] > 6.0 && !math.IsInf(whamResult.PMF[i], 1) {
				boundFreeEnergy = whamResult.PMF[i] - whamResult.PMF[minIdx]
				break
			}
		}
		fmt.Println()
		fmt.Printf("结合自由能估计: %.4f kBT\n", boundFreeEnergy)
		fmt.Println()
		fmt.Println("=== 解释 ===")
		fmt.Println("1. PMF最小值对应配体-蛋白结合态")
		fmt.Println("2. r增大方向对应解离过程")
		fmt.Println("3. 结合自由能 = 解离态自由能 - 结合态自由能")
		fmt.Println("4. 正值表示结合是有利的（稳定态）")
	}
}
