package differential_evolution

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

type BoundaryStrategy int
type AdaptiveStrategy int

const (
	ClipStrategy BoundaryStrategy = iota
	ReflectionStrategy
	RandomResetStrategy
	MixedStrategy
)

const (
	NoAdaptive AdaptiveStrategy = iota
	JadeAdaptive
)

const (
	defaultMemorySize = 100
	c                 = 0.1
)

type individual struct {
	params []float64
	f      float64
	cr     float64
}

type Result struct {
	BestSolution []float64
	BestValue    float64
	Iterations   int
}

type DE struct {
	PopulationSize   int
	CR               float64
	F                float64
	MaxIterations    int
	Dim              int
	LowerBound       []float64
	UpperBound       []float64
	ObjectiveFunc    func([]float64) float64
	BoundaryStrategy BoundaryStrategy
	AdaptiveStrategy AdaptiveStrategy

	muF      float64
	muCR     float64
	memoryF  []float64
	memoryCR []float64
	memIdx   int
}

func NewDE(populationSize int, cr, f float64, maxIterations int, dim int, lowerBound, upperBound []float64, objectiveFunc func([]float64) float64) *DE {
	return &DE{
		PopulationSize:   populationSize,
		CR:               cr,
		F:                f,
		MaxIterations:    maxIterations,
		Dim:              dim,
		LowerBound:       lowerBound,
		UpperBound:       upperBound,
		ObjectiveFunc:    objectiveFunc,
		BoundaryStrategy: MixedStrategy,
		AdaptiveStrategy: JadeAdaptive,
		muF:              0.5,
		muCR:             0.5,
		memoryF:          make([]float64, defaultMemorySize),
		memoryCR:         make([]float64, defaultMemorySize),
		memIdx:           0,
	}
}

func (de *DE) SetBoundaryStrategy(strategy BoundaryStrategy) {
	de.BoundaryStrategy = strategy
}

func (de *DE) SetAdaptiveStrategy(strategy AdaptiveStrategy) {
	de.AdaptiveStrategy = strategy
}

func (de *DE) initAdaptiveMemory() {
	for i := range de.memoryF {
		de.memoryF[i] = 0.5
		de.memoryCR[i] = 0.5
	}
}

func (de *DE) Optimize() *Result {
	rand.Seed(time.Now().UnixNano())

	if de.AdaptiveStrategy == JadeAdaptive {
		de.initAdaptiveMemory()
	}

	population := make([][]float64, de.PopulationSize)
	for i := range population {
		population[i] = make([]float64, de.Dim)
		for j := 0; j < de.Dim; j++ {
			population[i][j] = de.LowerBound[j] + rand.Float64()*(de.UpperBound[j]-de.LowerBound[j])
		}
	}

	fitness := make([]float64, de.PopulationSize)
	for i := range population {
		fitness[i] = de.ObjectiveFunc(population[i])
	}

	bestIndex := 0
	for i := 1; i < de.PopulationSize; i++ {
		if fitness[i] < fitness[bestIndex] {
			bestIndex = i
		}
	}

	for iter := 0; iter < de.MaxIterations; iter++ {
		successfulF := make([]float64, 0)
		successfulCR := make([]float64, 0)
		improvements := make([]float64, 0)

		for i := 0; i < de.PopulationSize; i++ {
			var f, cr float64

			if de.AdaptiveStrategy == JadeAdaptive {
				f = de.generateF()
				cr = de.generateCR()
			} else {
				f = de.F
				cr = de.CR
			}

			a, b, c := de.selectThreeIndices(i)

			target := population[i]
			mutant := de.mutateWithF(population[a], population[b], population[c], f)
			trial := de.crossoverWithCR(target, mutant, cr)
			de.handleBoundary(trial)

			trialFitness := de.ObjectiveFunc(trial)

			if trialFitness < fitness[i] {
				delta := fitness[i] - trialFitness
				improvements = append(improvements, delta)

				if de.AdaptiveStrategy == JadeAdaptive {
					successfulF = append(successfulF, f)
					successfulCR = append(successfulCR, cr)
				}

				population[i] = trial
				fitness[i] = trialFitness

				if trialFitness < fitness[bestIndex] {
					bestIndex = i
				}
			}
		}

		if de.AdaptiveStrategy == JadeAdaptive && len(successfulF) > 0 {
			de.updateAdaptiveParams(successfulF, successfulCR, improvements)
		}
	}

	return &Result{
		BestSolution: population[bestIndex],
		BestValue:    fitness[bestIndex],
		Iterations:   de.MaxIterations,
	}
}

func (de *DE) selectThreeIndices(exclude int) (int, int, int) {
	indices := make([]int, 0, 3)
	for len(indices) < 3 {
		idx := rand.Intn(de.PopulationSize)
		if idx != exclude && !contains(indices, idx) {
			indices = append(indices, idx)
		}
	}
	return indices[0], indices[1], indices[2]
}

func contains(slice []int, val int) bool {
	for _, item := range slice {
		if item == val {
			return true
		}
	}
	return false
}

func (de *DE) mutate(a, b, c []float64) []float64 {
	return de.mutateWithF(a, b, c, de.F)
}

func (de *DE) mutateWithF(a, b, c []float64, f float64) []float64 {
	mutant := make([]float64, de.Dim)
	for i := 0; i < de.Dim; i++ {
		mutant[i] = a[i] + f*(b[i]-c[i])
	}
	return mutant
}

func (de *DE) crossover(target, mutant []float64) []float64 {
	return de.crossoverWithCR(target, mutant, de.CR)
}

func (de *DE) crossoverWithCR(target, mutant []float64, cr float64) []float64 {
	trial := make([]float64, de.Dim)
	jRand := rand.Intn(de.Dim)
	for j := 0; j < de.Dim; j++ {
		if rand.Float64() < cr || j == jRand {
			trial[j] = mutant[j]
		} else {
			trial[j] = target[j]
		}
	}
	return trial
}

func (de *DE) generateF() float64 {
	f := de.cauchyRandom(de.muF, 0.1)
	for f <= 0 || f > 1 {
		f = de.cauchyRandom(de.muF, 0.1)
	}
	return f
}

func (de *DE) generateCR() float64 {
	cr := de.normalRandom(de.muCR, 0.1)
	cr = math.Max(0, math.Min(1, cr))
	return cr
}

func (de *DE) cauchyRandom(mu, sigma float64) float64 {
	u := rand.Float64() - 0.5
	return mu + sigma*math.Tan(math.Pi*u)
}

func (de *DE) normalRandom(mu, sigma float64) float64 {
	u1 := rand.Float64()
	u2 := rand.Float64()
	z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	return mu + sigma*z
}

func (de *DE) updateAdaptiveParams(successfulF, successfulCR, improvements []float64) {
	totalImprovement := 0.0
	for _, imp := range improvements {
		totalImprovement += imp
	}

	if totalImprovement == 0 {
		return
	}

	weightedMeanF := 0.0
	weightedMeanCR := 0.0
	for i := range successfulF {
		weight := improvements[i] / totalImprovement
		weightedMeanF += weight * successfulF[i] * successfulF[i]
		weightedMeanCR += weight * successfulCR[i]
	}

	de.muF = (1-c)*de.muF + c*weightedMeanF
	de.muCR = (1-c)*de.muCR + c*weightedMeanCR

	de.memoryF[de.memIdx] = weightedMeanF
	de.memoryCR[de.memIdx] = weightedMeanCR
	de.memIdx = (de.memIdx + 1) % len(de.memoryF)

	if de.memIdx == 0 {
		sort.Float64s(de.memoryF)
		sort.Float64s(de.memoryCR)
		de.muF = de.memoryF[len(de.memoryF)/2]
		de.muCR = de.memoryCR[len(de.memoryCR)/2]
	}
}

func (de *DE) handleBoundary(vec []float64) {
	switch de.BoundaryStrategy {
	case ClipStrategy:
		de.clip(vec)
	case ReflectionStrategy:
		de.reflect(vec)
	case RandomResetStrategy:
		de.randomReset(vec)
	case MixedStrategy:
		de.mixedHandle(vec)
	}
}

func (de *DE) clip(vec []float64) {
	for i := 0; i < de.Dim; i++ {
		if vec[i] < de.LowerBound[i] {
			vec[i] = de.LowerBound[i]
		}
		if vec[i] > de.UpperBound[i] {
			vec[i] = de.UpperBound[i]
		}
	}
}

func (de *DE) reflect(vec []float64) {
	for i := 0; i < de.Dim; i++ {
		lower := de.LowerBound[i]
		upper := de.UpperBound[i]
		rangeSize := upper - lower

		for vec[i] < lower || vec[i] > upper {
			if vec[i] < lower {
				vec[i] = lower + (lower - vec[i])
			}
			if vec[i] > upper {
				vec[i] = upper - (vec[i] - upper)
			}
		}
	}
}

func (de *DE) randomReset(vec []float64) {
	for i := 0; i < de.Dim; i++ {
		if vec[i] < de.LowerBound[i] || vec[i] > de.UpperBound[i] {
			vec[i] = de.LowerBound[i] + rand.Float64()*(de.UpperBound[i]-de.LowerBound[i])
		}
	}
}

func (de *DE) mixedHandle(vec []float64) {
	for i := 0; i < de.Dim; i++ {
		lower := de.LowerBound[i]
		upper := de.UpperBound[i]

		if vec[i] < lower || vec[i] > upper {
			if rand.Float64() < 0.5 {
				rangeSize := upper - lower
				for vec[i] < lower || vec[i] > upper {
					if vec[i] < lower {
						vec[i] = lower + (lower - vec[i])
					}
					if vec[i] > upper {
						vec[i] = upper - (vec[i] - upper)
					}
				}
			} else {
				vec[i] = lower + rand.Float64()*(upper-lower)
			}
		}
	}
}

func Optimize(funcStr string, dim int, lowerBound, upperBound []float64, populationSize int, cr, f float64, maxIterations int) (*Result, error) {
	objectiveFunc, err := parseFunction(funcStr, dim)
	if err != nil {
		return nil, fmt.Errorf("failed to parse function: %v", err)
	}

	de := NewDE(populationSize, cr, f, maxIterations, dim, lowerBound, upperBound, objectiveFunc)
	return de.Optimize(), nil
}

func parseFunction(funcStr string, dim int) (func([]float64) float64, error) {
	switch funcStr {
	case "sphere":
		return func(x []float64) float64 {
			sum := 0.0
			for _, v := range x {
				sum += v * v
			}
			return sum
		}, nil
	case "rastrigin":
		return func(x []float64) float64 {
			sum := 10.0 * float64(dim)
			for _, v := range x {
				sum += v*v - 10*math.Cos(2*math.Pi*v)
			}
			return sum
		}, nil
	case "rosenbrock":
		return func(x []float64) float64 {
			sum := 0.0
			for i := 0; i < dim-1; i++ {
				sum += 100*math.Pow(x[i+1]-x[i]*x[i], 2) + math.Pow(1-x[i], 2)
			}
			return sum
		}, nil
	case "ackley":
		return func(x []float64) float64 {
			sum1 := 0.0
			sum2 := 0.0
			for _, v := range x {
				sum1 += v * v
				sum2 += math.Cos(2 * math.Pi * v)
			}
			return -20*math.Exp(-0.2*math.Sqrt(sum1/float64(dim))) - math.Exp(sum2/float64(dim)) + 20 + math.E
		}, nil
	case "griewank":
		return func(x []float64) float64 {
			sum := 0.0
			product := 1.0
			for i, v := range x {
				sum += v * v / 4000.0
				product *= math.Cos(v / math.Sqrt(float64(i+1)))
			}
			return sum - product + 1
		}, nil
	default:
		return nil, fmt.Errorf("unknown function: %s. Available functions: sphere, rastrigin, rosenbrock, ackley, griewank", funcStr)
	}
}
