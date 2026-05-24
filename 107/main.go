package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

const (
	L         = 16
	Steps     = 2000
	EquiSteps = 500
)

type UnionFind struct {
	parent []int
	size   []int
}

func NewUnionFind(n int) *UnionFind {
	parent := make([]int, n)
	size := make([]int, n)
	for i := range parent {
		parent[i] = i
		size[i] = 1
	}
	return &UnionFind{parent: parent, size: size}
}

func (uf *UnionFind) Find(x int) int {
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) {
	xRoot := uf.Find(x)
	yRoot := uf.Find(y)
	if xRoot == yRoot {
		return
	}
	if uf.size[xRoot] < uf.size[yRoot] {
		xRoot, yRoot = yRoot, xRoot
	}
	uf.parent[yRoot] = xRoot
	uf.size[xRoot] += uf.size[yRoot]
}

type IsingModel struct {
	spins [][]int
	size  int
	rand  *rand.Rand
}

func NewIsingModel(size int, seed int64) *IsingModel {
	src := rand.NewSource(seed)
	r := rand.New(src)

	spins := make([][]int, size)
	for i := range spins {
		spins[i] = make([]int, size)
		for j := range spins[i] {
			if r.Float64() < 0.5 {
				spins[i][j] = 1
			} else {
				spins[i][j] = -1
			}
		}
	}
	return &IsingModel{spins: spins, size: size, rand: r}
}

func (l *IsingModel) idx(i, j int) int {
	return i*l.size + j
}

func (l *IsingModel) Energy() float64 {
	var E int
	N := l.size
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			down := (i + 1) % N
			right := (j + 1) % N
			E -= l.spins[i][j] * l.spins[down][j]
			E -= l.spins[i][j] * l.spins[i][right]
		}
	}
	return float64(E)
}

func (l *IsingModel) Magnetization() float64 {
	var M int
	for i := 0; i < l.size; i++ {
		for j := 0; j < l.size; j++ {
			M += l.spins[i][j]
		}
	}
	return float64(M)
}

func (l *IsingModel) SwendsenWangStep(T float64) {
	N := l.size
	beta := 1.0 / T
	pBond := 1.0 - math.Exp(-2.0*beta)

	uf := NewUnionFind(N * N)

	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			if l.spins[i][j] == l.spins[(i+1)%N][j] && l.rand.Float64() < pBond {
				uf.Union(l.idx(i, j), l.idx((i+1)%N, j))
			}
			if l.spins[i][j] == l.spins[i][(j+1)%N] && l.rand.Float64() < pBond {
				uf.Union(l.idx(i, j), l.idx(i, (j+1)%N))
			}
		}
	}

	clusters := make(map[int][]int)
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			root := uf.Find(l.idx(i, j))
			clusters[root] = append(clusters[root], l.idx(i, j))
		}
	}

	for _, indices := range clusters {
		if l.rand.Float64() < 0.5 {
			for _, idx := range indices {
				i := idx / N
				j := idx % N
				l.spins[i][j] *= -1
			}
		}
	}
}

type PureResult struct {
	T            float64
	Energy       float64
	Magnet       float64
	SpecificC    float64
	Suscept      float64
	BinderRatio  float64
	EnergyErr    float64
	MagnetErr    float64
	SpecificCErr float64
	SusceptErr   float64
}

func simulatePure(T float64, seed int64) PureResult {
	lattice := NewIsingModel(L, seed)

	for i := 0; i < EquiSteps; i++ {
		lattice.SwendsenWangStep(T)
	}

	nSamples := Steps
	E_samples := make([]float64, nSamples)
	M_samples := make([]float64, nSamples)
	E2_samples := make([]float64, nSamples)
	M2_samples := make([]float64, nSamples)
	M4_samples := make([]float64, nSamples)

	for i := 0; i < Steps; i++ {
		lattice.SwendsenWangStep(T)
		E := lattice.Energy()
		M := math.Abs(lattice.Magnetization())
		E_samples[i] = E
		M_samples[i] = M
		E2_samples[i] = E * E
		M2_samples[i] = M * M
		M4_samples[i] = M * M * M * M
	}

	N := float64(L * L)
	E_mean := mean(E_samples)
	M_mean := mean(M_samples)
	E2_mean := mean(E2_samples)
	M2_mean := mean(M2_samples)
	M4_mean := mean(M4_samples)

	Cv := (E2_mean - E_mean*E_mean) / (T * T) / N
	Chi := (M2_mean - M_mean*M_mean) / T / N
	U4 := 1.0 - M4_mean/(3.0*M2_mean*M2_mean)

	E_err := stdErr(E_samples)
	M_err := stdErr(M_samples)
	Cv_err := bootstrapError(E_samples, func(s []float64) float64 {
		em := mean(s)
		e2m := meanSquare(s)
		return (e2m - em*em) / (T * T) / N
	})
	Chi_err := bootstrapError(M_samples, func(s []float64) float64 {
		mm := meanAbs(s)
		m2m := meanSquareAbs(s)
		return (m2m - mm*mm) / T / N
	})

	return PureResult{
		T:            T,
		Energy:       E_mean / N,
		Magnet:       M_mean / N,
		SpecificC:    Cv,
		Suscept:      Chi,
		BinderRatio:  U4,
		EnergyErr:    E_err / N,
		MagnetErr:    M_err / N,
		SpecificCErr: Cv_err,
		SusceptErr:   Chi_err,
	}
}

func runPureIsing() {
	rand.Seed(time.Now().UnixNano())

	Tc := 2.0 / math.Log(1.0+math.Sqrt(2.0))
	fmt.Printf("纯净二维伊辛模型 - Swendsen-Wang 簇算法\n")
	fmt.Printf("========================================\n")
	fmt.Printf("晶格尺寸: %d x %d\n", L, L)
	fmt.Printf("热化步数: %d\n", EquiSteps)
	fmt.Printf("测量步数: %d\n", Steps)
	fmt.Printf("临界温度 Tc ≈ %.6f\n\n", Tc)

	temperatures := []float64{
		1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.26, 2.27, 2.28, 2.29,
		2.30, 2.31, 2.32, 2.34, 2.36, 2.38, 2.40, 2.45,
		2.5, 2.6, 2.8, 3.0, 3.5, 4.0,
	}

	file, err := os.Create("ising_pure_results.txt")
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	fmt.Fprintf(file, "# T\tE\tE_err\tM\tM_err\tCv\tCv_err\tChi\tChi_err\tU4\n")
	fmt.Printf("%-8s %-12s %-12s %-12s %-12s %-12s\n", "T", "E", "|M|", "Cv", "Chi", "U4")
	fmt.Println("------------------------------------------------------------------------")

	for idx, T := range temperatures {
		seed := time.Now().UnixNano() + int64(idx)*1000
		result := simulatePure(T, seed)

		fmt.Printf("%-8.4f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n",
			result.T, result.Energy, result.Magnet, result.SpecificC, result.Suscept, result.BinderRatio)

		fmt.Fprintf(file, "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
			result.T,
			result.Energy, result.EnergyErr,
			result.Magnet, result.MagnetErr,
			result.SpecificC, result.SpecificCErr,
			result.Suscept, result.SusceptErr,
			result.BinderRatio)
	}

	fmt.Println("\n结果已保存到 ising_pure_results.txt")
}

func runSpinGlass(hMean float64, hWidth float64, nDisorder int) {
	fmt.Printf("随机横场伊辛模型 (自旋玻璃) - 平行回火算法\n")
	fmt.Printf("==========================================\n")
	fmt.Printf("晶格尺寸: %d x %d\n", L, L)
	fmt.Printf("横场强度 hMean = %.2f, hWidth = %.2f\n", hMean, hWidth)
	fmt.Printf("无序样本数: %d\n", nDisorder)

	temperatures := makeTemperatureRange(0.2, 4.0, 12)

	fmt.Printf("\n温度点: ")
	for _, T := range temperatures {
		fmt.Printf("%.2f ", T)
	}
	fmt.Printf("\n\n")

	seed := time.Now().UnixNano()
	results := SimulateSpinGlass(L, hMean, hWidth, temperatures, nDisorder, 200, 500, seed)

	file, err := os.Create("spinglass_results.txt")
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	fmt.Fprintf(file, "# T\tE\tM_z\tq_EA\tCv\tChi\n")
	fmt.Printf("%-8s %-12s %-12s %-12s %-12s %-12s\n", "T", "E", "|M_z|", "q_EA", "Cv", "Chi")
	fmt.Println("------------------------------------------------------------------------")

	for _, r := range results {
		fmt.Printf("%-8.4f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n",
			r.T, r.Energy, r.MagnetZ, r.EdwardsAnderson, r.SpecificC, r.Susceptibility)

		fmt.Fprintf(file, "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
			r.T, r.Energy, r.MagnetZ, r.EdwardsAnderson, r.SpecificC, r.Susceptibility)
	}

	fmt.Println("\n结果已保存到 spinglass_results.txt")
	fmt.Println("\n自旋玻璃相分析:")
	fmt.Println("- q_EA > 0: 自旋玻璃相 (Edwards-Anderson序参量非零)")
	fmt.Println("- 低温下能量低，q_EA高 -> 玻璃态冻结")
	fmt.Println("- 高温下q_EA -> 0 -> 顺磁相")
}

func runPhaseDiagram(nDisorder int) {
	fmt.Printf("自旋玻璃相图计算 - 平行回火\n")
	fmt.Printf("================================\n")
	fmt.Printf("晶格尺寸: %d x %d\n", L, L)
	fmt.Printf("无序样本数: %d\n", nDisorder)

	hValues := []float64{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
	temperatures := makeTemperatureRange(0.5, 3.5, 8)

	file, err := os.Create("spinglass_phase_diagram.txt")
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	fmt.Fprintf(file, "# h\tT\tq_EA\tM_z\n")
	fmt.Printf("\n计算中... (这可能需要一些时间)\n\n")

	for _, h := range hValues {
		fmt.Printf("h = %.1f: ", h)
		seed := time.Now().UnixNano()
		results := SimulateSpinGlass(L, h, 0.0, temperatures, nDisorder, 150, 400, seed)

		for _, r := range results {
			fmt.Fprintf(file, "%.2f\t%.4f\t%.6f\t%.6f\n",
				h, r.T, r.EdwardsAnderson, r.MagnetZ)
		}
		fmt.Printf("完成\n")
	}

	fmt.Println("\n相图数据已保存到 spinglass_phase_diagram.txt")
	fmt.Println("\n相图说明:")
	fmt.Println("- h=0: 纯净伊辛模型，T_c≈2.27")
	fmt.Println("- h增大: 量子涨落增强，相变温度降低")
	fmt.Println("- h足够大时: 量子顺磁态")
}

func mean(s []float64) float64 {
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}

func meanAbs(s []float64) float64 {
	sum := 0.0
	for _, v := range s {
		sum += math.Abs(v)
	}
	return sum / float64(len(s))
}

func meanSquare(s []float64) float64 {
	sum := 0.0
	for _, v := range s {
		sum += v * v
	}
	return sum / float64(len(s))
}

func meanSquareAbs(s []float64) float64 {
	sum := 0.0
	for _, v := range s {
		absV := math.Abs(v)
		sum += absV * absV
	}
	return sum / float64(len(s))
}

func stdErr(s []float64) float64 {
	m := mean(s)
	variance := 0.0
	for _, v := range s {
		variance += (v - m) * (v - m)
	}
	variance /= float64(len(s) - 1)
	return math.Sqrt(variance / float64(len(s)))
}

func bootstrapError(s []float64, estimator func([]float64) float64) float64 {
	nBootstrap := 100
	estimates := make([]float64, nBootstrap)
	n := len(s)

	for b := 0; b < nBootstrap; b++ {
		sample := make([]float64, n)
		for i := 0; i < n; i++ {
			idx := rand.Intn(n)
			sample[i] = s[idx]
		}
		estimates[b] = estimator(sample)
	}

	m := mean(estimates)
	variance := 0.0
	for _, e := range estimates {
		variance += (e - m) * (e - m)
	}
	return math.Sqrt(variance / float64(nBootstrap))
}

func main() {
	mode := flag.String("mode", "pure", "运行模式: pure(纯净), glass(自旋玻璃), diagram(相图)")
	hMean := flag.Float64("h", 1.0, "自旋玻璃: 平均横场强度")
	hWidth := flag.Float64("hw", 0.5, "自旋玻璃: 横场无序宽度")
	nDisorder := flag.Int("n", 3, "自旋玻璃: 无序样本数")
	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	switch *mode {
	case "pure":
		runPureIsing()
	case "glass":
		runSpinGlass(*hMean, *hWidth, *nDisorder)
	case "diagram":
		runPhaseDiagram(*nDisorder)
	default:
		fmt.Println("未知模式。使用 -mode=pure|glass|diagram")
		flag.Usage()
	}
}
