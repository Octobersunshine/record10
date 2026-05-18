package differential_evolution

import (
	"math"
	"testing"
)

func TestBoundaryStrategies(t *testing.T) {
	strategies := []struct {
		name     string
		strategy BoundaryStrategy
	}{
		{"Clip", ClipStrategy},
		{"Reflection", ReflectionStrategy},
		{"RandomReset", RandomResetStrategy},
		{"Mixed", MixedStrategy},
	}

	for _, s := range strategies {
		t.Run(s.name, func(t *testing.T) {
			dim := 5
			lowerBound := []float64{-5, -5, -5, -5, -5}
			upperBound := []float64{5, 5, 5, 5, 5}

			result, err := testStrategy(s.strategy, dim, lowerBound, upperBound)
			if err != nil {
				t.Fatalf("优化失败: %v", err)
			}

			if result.BestValue > 1e-2 {
				t.Errorf("策略 %s 最优值应该接近0, 得到: %f", s.name, result.BestValue)
			}

			for i, v := range result.BestSolution {
				if v < lowerBound[i] || v > upperBound[i] {
					t.Errorf("策略 %s 解超出边界, 索引%d: %f 不在[%.2f, %.2f]", s.name, i, v, lowerBound[i], upperBound[i])
				}
			}
		})
	}
}

func testStrategy(strategy BoundaryStrategy, dim int, lowerBound, upperBound []float64) (*Result, error) {
	customFunc := func(x []float64) float64 {
		sum := 0.0
		for _, v := range x {
			sum += v * v
		}
		return sum
	}

	config := DefaultConfig()
	config.MaxIterations = 800
	config.BoundaryStrategy = strategy

	de := NewDE(config.PopulationSize, config.CR, config.F, config.MaxIterations, dim, lowerBound, upperBound, customFunc)
	de.SetBoundaryStrategy(strategy)
	return de.Optimize(), nil
}

func TestPopulationDiversity(t *testing.T) {
	dim := 10
	lowerBound := make([]float64, dim)
	upperBound := make([]float64, dim)
	for i := range lowerBound {
		lowerBound[i] = -1
		upperBound[i] = 1
	}

	customFunc := func(x []float64) float64 {
		sum := 0.0
		for _, v := range x {
			sum += v * v
		}
		return sum
	}

	strategies := []BoundaryStrategy{ClipStrategy, MixedStrategy}
	results := make([]float64, len(strategies))

	for i, strategy := range strategies {
		diversity := calculateDiversity(strategy, dim, lowerBound, upperBound, customFunc)
		results[i] = diversity
	}

	if results[0] >= results[1] {
		t.Logf("注意: Clip策略多样性 %.4f, Mixed策略多样性 %.4f", results[0], results[1])
	} else {
		t.Logf("验证通过: Clip策略多样性 %.4f, Mixed策略多样性 %.4f (Mixed更高)", results[0], results[1])
	}
}

func calculateDiversity(strategy BoundaryStrategy, dim int, lowerBound, upperBound []float64, objFunc func([]float64) float64) float64 {
	rand.Seed(42)

	population := make([][]float64, 50)
	for i := range population {
		population[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			population[i][j] = lowerBound[j] + rand.Float64()*(upperBound[j]-lowerBound[j])*4
		}
	}

	de := NewDE(50, 0.9, 0.5, 1, dim, lowerBound, upperBound, objFunc)
	de.SetBoundaryStrategy(strategy)

	for i := range population {
		de.handleBoundary(population[i])
	}

	mean := make([]float64, dim)
	for i := 0; i < dim; i++ {
		sum := 0.0
		for j := range population {
			sum += population[j][i]
		}
		mean[i] = sum / float64(len(population))
	}

	varianceSum := 0.0
	for i := 0; i < dim; i++ {
		for j := range population {
			varianceSum += math.Pow(population[j][i]-mean[i], 2)
		}
	}

	return varianceSum / float64(dim*len(population))
}

func TestReflection(t *testing.T) {
	de := &DE{
		Dim:              3,
		LowerBound:       []float64{0, 0, 0},
		UpperBound:       []float64{10, 10, 10},
		BoundaryStrategy: ReflectionStrategy,
	}

	vec := []float64{-2, 12, 5}
	de.reflect(vec)

	if math.Abs(vec[0]-2) > 1e-10 {
		t.Errorf("反射后应该是2, 得到: %f", vec[0])
	}
	if math.Abs(vec[1]-8) > 1e-10 {
		t.Errorf("反射后应该是8, 得到: %f", vec[1])
	}
	if math.Abs(vec[2]-5) > 1e-10 {
		t.Errorf("应该保持5不变, 得到: %f", vec[2])
	}
}

func TestMultipleReflection(t *testing.T) {
	de := &DE{
		Dim:              1,
		LowerBound:       []float64{0},
		UpperBound:       []float64{10},
		BoundaryStrategy: ReflectionStrategy,
	}

	vec := []float64{35}
	de.reflect(vec)

	if vec[0] < 0 || vec[0] > 10 {
		t.Errorf("多次反射后应该在[0,10]范围内, 得到: %f", vec[0])
	}
}
