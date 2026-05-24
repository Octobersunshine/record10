package main

import (
	"math"
	"math/rand"
	"time"
)

type RTFIM struct {
	spins [][]float64
	size  int
	J     float64
	h     [][]float64
	rand  *rand.Rand
}

func NewRTFIM(size int, hMean float64, hWidth float64, seed int64) *RTFIM {
	src := rand.NewSource(seed)
	r := rand.New(src)

	spins := make([][]float64, size)
	for i := range spins {
		spins[i] = make([]float64, size)
		for j := range spins[i] {
			theta := r.Float64() * math.Pi
			spins[i][j] = math.Cos(theta)
		}
	}

	h := make([][]float64, size)
	for i := range h {
		h[i] = make([]float64, size)
		for j := range h[i] {
			h[i][j] = hMean + hWidth*(2.0*r.Float64()-1.0)
			if h[i][j] < 0 {
				h[i][j] = 0
			}
		}
	}

	return &RTFIM{
		spins: spins,
		size:  size,
		J:     1.0,
		h:     h,
		rand:  r,
	}
}

func (m *RTFIM) Energy() float64 {
	var E float64
	N := m.size
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			sz1 := m.spins[i][j]
			down := (i + 1) % N
			right := (j + 1) % N
			E -= m.J * sz1 * m.spins[down][j]
			E -= m.J * sz1 * m.spins[i][right]
			E -= m.h[i][j] * math.Sqrt(1.0-sz1*sz1)
		}
	}
	return E
}

func (m *RTFIM) MagnetizationZ() float64 {
	var M float64
	for i := 0; i < m.size; i++ {
		for j := 0; j < m.size; j++ {
			M += m.spins[i][j]
		}
	}
	return M / float64(m.size*m.size)
}

func (m *RTFIM) MetropolisStep(T float64) {
	N := m.size
	i := m.rand.Intn(N)
	j := m.rand.Intn(N)

	szOld := m.spins[i][j]
	theta := m.rand.Float64() * math.Pi
	szNew := math.Cos(theta)

	up := (i - 1 + N) % N
	down := (i + 1) % N
	left := (j - 1 + N) % N
	right := (j + 1) % N

	neighborsSZ := m.spins[up][j] + m.spins[down][j] + m.spins[i][left] + m.spins[i][right]
	deltaE := -m.J * (szNew - szOld) * neighborsSZ
	deltaE += -m.h[i][j] * (math.Sqrt(1.0-szNew*szNew) - math.Sqrt(1.0-szOld*szOld))

	if deltaE <= 0 || m.rand.Float64() < math.Exp(-deltaE/T) {
		m.spins[i][j] = szNew
	}
}

func (m *RTFIM) Sweep(T float64) {
	N := m.size * m.size
	for k := 0; k < N; k++ {
		m.MetropolisStep(T)
	}
}

func (m *RTFIM) GetSpins() [][]float64 {
	spinsCopy := make([][]float64, m.size)
	for i := range spinsCopy {
		spinsCopy[i] = make([]float64, m.size)
		copy(spinsCopy[i], m.spins[i])
	}
	return spinsCopy
}

type Replica struct {
	model    *RTFIM
	T        float64
	energy   float64
	magnet   float64
}

type ParallelTempering struct {
	replicas []*Replica
	nSwap    int
	rand     *rand.Rand
}

func NewParallelTempering(size int, hMean float64, hWidth float64, temperatures []float64, disorderSeed int64) *ParallelTempering {
	replicaRand := rand.New(rand.NewSource(time.Now().UnixNano()))

	replicas := make([]*Replica, len(temperatures))
	for i, T := range temperatures {
		seed := disorderSeed + int64(i)*1000
		model := NewRTFIM(size, hMean, hWidth, seed)
		replicas[i] = &Replica{
			model:  model,
			T:      T,
			energy: model.Energy(),
		}
	}

	return &ParallelTempering{
		replicas: replicas,
		nSwap:    0,
		rand:     replicaRand,
	}
}

func (pt *ParallelTempering) RunSweeps(nSweeps int) {
	for r := range pt.replicas {
		for s := 0; s < nSweeps; s++ {
			pt.replicas[r].model.Sweep(pt.replicas[r].T)
		}
		pt.replicas[r].energy = pt.replicas[r].model.Energy()
		pt.replicas[r].magnet = pt.replicas[r].model.MagnetizationZ()
	}
}

func (pt *ParallelTempering) SwapAttempt() int {
	accepted := 0
	nReps := len(pt.replicas)

	for i := 0; i < nReps-1; i++ {
		beta1 := 1.0 / pt.replicas[i].T
		beta2 := 1.0 / pt.replicas[i+1].T
		delta := (beta2 - beta1) * (pt.replicas[i].energy - pt.replicas[i+1].energy)

		if delta <= 0 || pt.rand.Float64() < math.Exp(delta) {
			pt.replicas[i], pt.replicas[i+1] = pt.replicas[i+1], pt.replicas[i]
			pt.replicas[i].T, pt.replicas[i+1].T = pt.replicas[i+1].T, pt.replicas[i].T
			accepted++
		}
	}
	pt.nSwap++
	return accepted
}

type SpinGlassResult struct {
	T                float64
	Energy           float64
	MagnetZ          float64
	SpinGlassOrder   float64
	SpecificC        float64
	Susceptibility   float64
	EdwardsAnderson  float64
}

func makeTemperatureRange(Tmin, Tmax float64, nTemps int) []float64 {
	temps := make([]float64, nTemps)
	for i := range temps {
		ratio := float64(i) / float64(nTemps-1)
		temps[i] = Tmin + ratio*(Tmax-Tmin)
	}
	return temps
}

func SimulateSpinGlass(size int, hMean float64, hWidth float64, temperatures []float64,
	nDisorder int, equiSteps int, measureSteps int, seed int64) []SpinGlassResult {

	results := make([]SpinGlassResult, len(temperatures))
	for i := range results {
		results[i].T = temperatures[i]
	}

	for d := 0; d < nDisorder; d++ {
		disorderSeed := seed + int64(d)*100000
		pt := NewParallelTempering(size, hMean, hWidth, temperatures, disorderSeed)

		for e := 0; e < equiSteps/10; e++ {
			pt.RunSweeps(10)
			pt.SwapAttempt()
		}

		nSamples := measureSteps / 10
		E_samples := make([][]float64, len(temperatures))
		M_samples := make([][]float64, len(temperatures))
		SZ_samples := make([][][]float64, len(temperatures))

		for i := range temperatures {
			E_samples[i] = make([]float64, nSamples)
			M_samples[i] = make([]float64, nSamples)
			SZ_samples[i] = make([][]float64, nSamples)
		}

		idx := 0
		for s := 0; s < measureSteps; s++ {
			pt.RunSweeps(10)
			pt.SwapAttempt()

			if s%10 == 0 {
				for r, rep := range pt.replicas {
					tIdx := 0
					for ti, t := range temperatures {
						if math.Abs(rep.T-t) < 1e-6 {
							tIdx = ti
							break
						}
					}
					E_samples[tIdx][idx] = rep.energy
					M_samples[tIdx][idx] = rep.magnet
					SZ_samples[tIdx][idx] = flattenSpins(rep.model.GetSpins())
				}
				idx++
			}
		}

		N := float64(size * size)
		for ti := range temperatures {
			E_mean := mean(E_samples[ti])
			M_mean := meanAbs(M_samples[ti])
			E2_mean := meanSquare(E_samples[ti])
			M2_mean := meanSquareAbs(M_samples[ti])

			q := computeSpinGlassOrder(SZ_samples[ti])

			Cv := (E2_mean - E_mean*E_mean) / (temperatures[ti] * temperatures[ti]) / N
			Chi := (M2_mean - M_mean*M_mean) / temperatures[ti] / N

			results[ti].Energy += E_mean / N
			results[ti].MagnetZ += math.Abs(M_mean)
			results[ti].SpinGlassOrder += q
			results[ti].SpecificC += Cv
			results[ti].Susceptibility += Chi
			results[ti].EdwardsAnderson += q
		}
	}

	nDisorderF := float64(nDisorder)
	for i := range results {
		results[i].Energy /= nDisorderF
		results[i].MagnetZ /= nDisorderF
		results[i].SpinGlassOrder /= nDisorderF
		results[i].SpecificC /= nDisorderF
		results[i].Susceptibility /= nDisorderF
		results[i].EdwardsAnderson /= nDisorderF
	}

	return results
}

func flattenSpins(spins [][]float64) []float64 {
	N := len(spins)
	flat := make([]float64, N*N)
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			flat[i*N+j] = spins[i][j]
		}
	}
	return flat
}

func computeSpinGlassOrder(samples [][]float64) float64 {
	if len(samples) < 2 {
		return 0
	}
	N := float64(len(samples[0]))
	var q float64
	nPairs := 0

	for i := 0; i < len(samples); i++ {
		for j := i + 1; j < len(samples); j++ {
			var overlap float64
			for k := range samples[i] {
				overlap += samples[i][k] * samples[j][k]
			}
			q += (overlap / N) * (overlap / N)
			nPairs++
		}
	}
	return q / float64(nPairs)
}
