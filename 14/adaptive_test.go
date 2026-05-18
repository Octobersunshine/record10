package differential_evolution

import (
	"testing"
)

func TestAdaptiveStrategies(t *testing.T) {
	strategies := []struct {
		name     string
		strategy AdaptiveStrategy
	}{
		{"NoAdaptive", NoAdaptive},
		{"JadeAdaptive", JadeAdaptive},
	}

	for _, s := range strategies {
		t.Run(s.name, func(t *testing.T) {
			dim := 10
			config := DefaultConfig()
			config.MaxIterations = 500
			config.AdaptiveStrategy = s.strategy

			result, err := MinimizeWithConfig("sphere", dim, -5.12, 5.12, config)
			if err != nil {
				t.Fatalf("优化失败: %v", err)
			}

			t.Logf("%s 最优值: %.8f", s.name, result.BestValue)
		})
	}
}

func TestJADEPerformance(t *testing.T) {
	dim := 10
	config := DefaultConfig()
	config.MaxIterations = 500

	functions := []string{"sphere", "rastrigin", "rosenbrock"}

	for _, fn := range functions {
		t.Run(fn, func(t *testing.T) {
			config.AdaptiveStrategy = NoAdaptive
			result1, _ := MinimizeWithConfig(fn, dim, -5.12, 5.12, config)

			config.AdaptiveStrategy = JadeAdaptive
			result2, _ := MinimizeWithConfig(fn, dim, -5.12, 5.12, config)

			t.Logf("%s - 固定参数: %.8f, JADE自适应: %.8f", fn, result1.BestValue, result2.BestValue)
		})
	}
}

func TestParameterGeneration(t *testing.T) {
	de := NewDE(50, 0.9, 0.5, 100, 2, []float64{-5, -5}, []float64{5, 5}, nil)

	de.muF = 0.5
	f := de.generateF()
	if f <= 0 || f > 1 {
		t.Errorf("生成的F应该在(0,1]范围内，得到: %f", f)
	}

	de.muCR = 0.5
	cr := de.generateCR()
	if cr < 0 || cr > 1 {
		t.Errorf("生成的CR应该在[0,1]范围内，得到: %f", cr)
	}
}

func TestCauchyRandom(t *testing.T) {
	de := &DE{}
	count := 0
	for i := 0; i < 1000; i++ {
		val := de.cauchyRandom(0.5, 0.1)
		if val < 0 || val > 1 {
			count++
		}
	}
	t.Logf("柯西分布超出[0,1]的比例: %.2f%%", float64(count)/1000*100)
}

func TestNormalRandom(t *testing.T) {
	de := &DE{}
	sum := 0.0
	n := 10000
	for i := 0; i < n; i++ {
		sum += de.normalRandom(0.5, 0.1)
	}
	mean := sum / float64(n)
	if mean < 0.45 || mean > 0.55 {
		t.Errorf("正态分布均值应该接近0.5，得到: %f", mean)
	}
	t.Logf("正态分布均值: %.4f", mean)
}

func TestAdaptiveUpdate(t *testing.T) {
	de := &DE{
		muF:      0.5,
		muCR:     0.5,
		memoryF:  make([]float64, defaultMemorySize),
		memoryCR: make([]float64, defaultMemorySize),
		memIdx:   0,
	}

	successfulF := []float64{0.4, 0.5, 0.6}
	successfulCR := []float64{0.8, 0.9, 0.85}
	improvements := []float64{1.0, 2.0, 1.0}

	oldMuF := de.muF
	oldMuCR := de.muCR

	de.updateAdaptiveParams(successfulF, successfulCR, improvements)

	if de.muF == oldMuF {
		t.Error("muF应该被更新")
	}
	if de.muCR == oldMuCR {
		t.Error("muCR应该被更新")
	}

	t.Logf("muF: %.4f -> %.4f", oldMuF, de.muF)
	t.Logf("muCR: %.4f -> %.4f", oldMuCR, de.muCR)
}

func TestBoundaryAndAdaptiveCombination(t *testing.T) {
	boundaryStrategies := []BoundaryStrategy{ClipStrategy, ReflectionStrategy, MixedStrategy}
	adaptiveStrategies := []AdaptiveStrategy{NoAdaptive, JadeAdaptive}

	for _, bs := range boundaryStrategies {
		for _, as := range adaptiveStrategies {
			config := DefaultConfig()
			config.MaxIterations = 200
			config.BoundaryStrategy = bs
			config.AdaptiveStrategy = as

			result, _ := MinimizeWithConfig("rastrigin", 5, -5.12, 5.12, config)
			t.Logf("边界=%v, 自适应=%v, 结果=%.6f", bs, as, result.BestValue)
		}
	}
}
