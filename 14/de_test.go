package differential_evolution

import (
	"math"
	"testing"
)

func TestSphereFunction(t *testing.T) {
	dim := 5
	result, err := Minimize("sphere", dim, -5.12, 5.12)
	if err != nil {
		t.Fatalf("优化失败: %v", err)
	}

	if result.BestValue > 1e-4 {
		t.Errorf("Sphere函数最优值应该接近0, 得到: %f", result.BestValue)
	}

	for _, v := range result.BestSolution {
		if math.Abs(v) > 1e-2 {
			t.Errorf("Sphere函数最优解应该接近0, 得到: %f", v)
		}
	}
}

func TestRastriginFunction(t *testing.T) {
	dim := 5
	result, err := Minimize("rastrigin", dim, -5.12, 5.12)
	if err != nil {
		t.Fatalf("优化失败: %v", err)
	}

	if result.BestValue > 1.0 {
		t.Errorf("Rastrigin函数最优值应该接近0, 得到: %f", result.BestValue)
	}
}

func TestUnknownFunction(t *testing.T) {
	_, err := Minimize("unknown", 2, -1, 1)
	if err == nil {
		t.Error("未知函数应该返回错误")
	}
}

func TestCustomObjective(t *testing.T) {
	dim := 2
	lowerBound := []float64{-10, -10}
	upperBound := []float64{10, 10}

	customFunc := func(x []float64) float64 {
		return x[0]*x[0] + x[1]*x[1]
	}

	de := NewDE(50, 0.9, 0.5, 500, dim, lowerBound, upperBound, customFunc)
	result := de.Optimize()

	if result.BestValue > 1e-4 {
		t.Errorf("自定义函数最优值应该接近0, 得到: %f", result.BestValue)
	}
}

func TestBounds(t *testing.T) {
	dim := 3
	lowerBound := []float64{0, 0, 0}
	upperBound := []float64{1, 1, 1}

	result, err := MinimizeWithBounds("sphere", dim, lowerBound, upperBound)
	if err != nil {
		t.Fatalf("优化失败: %v", err)
	}

	for i, v := range result.BestSolution {
		if v < lowerBound[i] || v > upperBound[i] {
			t.Errorf("解超出边界, 索引%d: %f 不在[%.2f, %.2f]", i, v, lowerBound[i], upperBound[i])
		}
	}
}
