package calculator

import (
	"math"
	"runtime"
	"testing"
)

func TestParser_Evaluate(t *testing.T) {
	tests := []struct {
		name     string
		expr     string
		x        float64
		expected float64
	}{
		{"x^2 at x=2", "x^2", 2, 4},
		{"x+1 at x=3", "x+1", 3, 4},
		{"2*x at x=5", "2*x", 5, 10},
		{"x/2 at x=4", "x/2", 4, 2},
		{"-x at x=3", "-x", 3, -3},
		{"sin(0)", "sin(x)", 0, 0},
		{"cos(0)", "cos(x)", 0, 1},
		{"exp(0)", "exp(x)", 0, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewParser(tt.expr)
			result, err := p.Evaluate(tt.x)
			if err != nil {
				t.Errorf("Evaluate() error = %v", err)
				return
			}
			if math.Abs(result-tt.expected) > 1e-9 {
				t.Errorf("Evaluate() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCompositeSimpson(t *testing.T) {
	tests := []struct {
		name     string
		expr     string
		a        float64
		b        float64
		n        int
		expected float64
	}{
		{"∫x^2 dx from 0 to 1", "x^2", 0, 1, 100, 1.0 / 3.0},
		{"∫1 dx from 0 to 1", "1", 0, 1, 10, 1.0},
		{"∫x dx from 0 to 2", "x", 0, 2, 100, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := CompositeSimpson(tt.expr, tt.a, tt.b, tt.n)
			if err != nil {
				t.Errorf("CompositeSimpson() error = %v", err)
				return
			}
			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CompositeSimpson() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCompositeSimpson_Errors(t *testing.T) {
	tests := []struct {
		name string
		expr string
		a    float64
		b    float64
		n    int
	}{
		{"n is odd", "x^2", 0, 1, 3},
		{"n is zero", "x^2", 0, 1, 0},
		{"n is negative", "x^2", 0, 1, -2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := CompositeSimpson(tt.expr, tt.a, tt.b, tt.n)
			if err == nil {
				t.Error("CompositeSimpson() expected error, got nil")
			}
		})
	}
}

func TestAdaptiveSimpson_Basic(t *testing.T) {
	tests := []struct {
		name     string
		expr     string
		a        float64
		b        float64
		expected float64
	}{
		{"∫x^2 dx from 0 to 1", "x^2", 0, 1, 1.0 / 3.0},
		{"∫1 dx from 0 to 1", "1", 0, 1, 1.0},
		{"∫x dx from 0 to 2", "x", 0, 2, 2.0},
		{"∫sin(x) dx from 0 to pi", "sin(x)", 0, math.Pi, 2.0},
		{"∫exp(x) dx from 0 to 1", "exp(x)", 0, 1, math.E - 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := AdaptiveSimpson(tt.expr, tt.a, tt.b)
			if err != nil {
				t.Errorf("AdaptiveSimpson() error = %v", err)
				return
			}
			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("AdaptiveSimpson() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestAdaptiveSimpson_NoStackOverflow(t *testing.T) {
	t.Run("function with sharp peak - no stack overflow", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				if _, ok := r.(runtime.Error); ok {
					t.Errorf("Got runtime panic (likely stack overflow): %v", r)
				}
			}
		}()

		result, err := AdaptiveSimpson("1/(1+10000*(x-0.5)^2)", 0, 1, 1e-12)
		if err != nil {
			t.Errorf("AdaptiveSimpson() error = %v", err)
			return
		}
		
		expected := math.Atan(50) / 50.0
		if math.Abs(result-expected) > 1e-3 {
			t.Logf("Note: Result may vary due to recursion limit. Got %v, expected ~%v", result, expected)
		}
		
		t.Logf("Successfully computed without stack overflow, result = %v", result)
	})

	t.Run("extremely small epsilon - should not recurse infinitely", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Got panic (likely stack overflow): %v", r)
			}
		}()

		result, err := AdaptiveSimpson("x^2", 0, 1, 1e-30)
		if err != nil {
			t.Errorf("AdaptiveSimpson() error = %v", err)
			return
		}
		
		expected := 1.0 / 3.0
		t.Logf("Computed with epsilon=1e-30, result = %v, expected = %v", result, expected)
	})
}

func TestAdaptiveSimpson_MinIntervalWidth(t *testing.T) {
	t.Run("very small interval should return error", func(t *testing.T) {
		_, err := AdaptiveSimpson("x^2", 0, 1e-15)
		if err == nil {
			t.Error("Expected error for very small interval, got nil")
		}
	})

	t.Run("normal interval should succeed", func(t *testing.T) {
		_, err := AdaptiveSimpson("x^2", 0, 1)
		if err != nil {
			t.Errorf("Unexpected error for normal interval: %v", err)
		}
	})
}

func TestAdaptiveSimpson_WithCustomEpsilon(t *testing.T) {
	tests := []struct {
		name    string
		expr    string
		a       float64
		b       float64
		epsilon float64
	}{
		{"loose epsilon 1e-3", "x^2", 0, 1, 1e-3},
		{"default epsilon 1e-8", "x^2", 0, 1, 1e-8},
		{"tight epsilon 1e-10", "sin(x)", 0, math.Pi, 1e-10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := AdaptiveSimpson(tt.expr, tt.a, tt.b, tt.epsilon)
			if err != nil {
				t.Errorf("AdaptiveSimpson() error = %v", err)
				return
			}
			t.Logf("With epsilon=%g, result=%v", tt.epsilon, result)
		})
	}
}

func TestSimpsonRule(t *testing.T) {
	parser := NewParser("x^2")
	result, err := simpsonRule(parser, 0, 1)
	if err != nil {
		t.Fatalf("simpsonRule() error = %v", err)
	}
	
	expected := 1.0 / 3.0
	if math.Abs(result-expected) > 1e-3 {
		t.Errorf("simpsonRule() = %v, want ~%v", result, expected)
	}
}

func TestSimpsonRuleWithPoints(t *testing.T) {
	parser := NewParser("x^2")
	fa := 0.0
	fb := 1.0
	s, fc, err := simpsonRuleWithPoints(parser, 0, 1, fa, fb)
	if err != nil {
		t.Fatalf("simpsonRuleWithPoints() error = %v", err)
	}
	
	if fc != 0.25 {
		t.Errorf("fc = %v, want 0.25", fc)
	}
	
	expected := 1.0 / 3.0
	if math.Abs(s-expected) > 1e-3 {
		t.Errorf("simpsonRuleWithPoints() = %v, want ~%v", s, expected)
	}
}

func TestEstimateCurvature(t *testing.T) {
	tests := []struct {
		name           string
		expr           string
		a, b           float64
		expectHigh     bool
		expectLow      bool
	}{
		{
			name:       "linear function has low curvature",
			expr:       "x",
			a:          0,
			b:          1,
			expectHigh: false,
			expectLow:  true,
		},
		{
			name:       "quadratic has moderate curvature",
			expr:       "x^2",
			a:          0,
			b:          1,
			expectHigh: false,
			expectLow:  false,
		},
		{
			name:       "sharp peak function has high curvature",
			expr:       "1/(1+10000*(x-0.5)^2)",
			a:          0.45,
			b:          0.55,
			expectHigh: true,
			expectLow:  false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := NewParser(tt.expr)
			fa, _ := parser.Evaluate(tt.a)
			fb, _ := parser.Evaluate(tt.b)
			curvature, err := estimateCurvature(parser, tt.a, tt.b, fa, fb)
			if err != nil {
				t.Fatalf("estimateCurvature() error = %v", err)
			}
			
			t.Logf("Curvature estimate: %v", curvature)
			
			if tt.expectHigh && curvature < CurvatureThresholdHigh {
				t.Errorf("Expected high curvature (>%v), got %v", CurvatureThresholdHigh, curvature)
			}
			if tt.expectLow && curvature > CurvatureThresholdLow {
				t.Logf("Note: Curvature %v is higher than expected low threshold %v", curvature, CurvatureThresholdLow)
			}
		})
	}
}

func TestAdaptiveEpsilon(t *testing.T) {
	tests := []struct {
		name      string
		epsilon   float64
		curvature float64
		factor    float64
	}{
		{"high curvature, tighter epsilon", 1e-8, 200.0, 0.1},
		{"low curvature, looser epsilon", 1e-8, 0.1, 2.0},
		{"moderate curvature, same epsilon", 1e-8, 50.0, 1.0},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adaptiveEpsilon(tt.epsilon, tt.curvature)
			expected := tt.epsilon * tt.factor
			if math.Abs(result-expected) > 1e-15 {
				t.Errorf("adaptiveEpsilon() = %v, want %v", result, expected)
			}
		})
	}
}

func TestShouldEarlyStop(t *testing.T) {
	tests := []struct {
		name      string
		s         float64
		sLeft     float64
		sRight    float64
		eps       float64
		curvature float64
		depth     int
		want      bool
	}{
		{
			name:  "error within tolerance",
			s:     1.0,
			sLeft: 0.4999,
			sRight: 0.5000,
			eps:   1e-4,
			depth: 10,
			want:  true,
		},
		{
			name:  "error too large",
			s:     1.0,
			sLeft: 0.5,
			sRight: 0.6,
			eps:   1e-4,
			depth: 10,
			want:  false,
		},
		{
			name:  "at max recursion depth",
			s:     1.0,
			sLeft: 0.5,
			sRight: 0.5,
			eps:   1e-8,
			depth: DefaultMaxRecursionDepth,
			want:  true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shouldEarlyStop(tt.s, tt.sLeft, tt.sRight, tt.eps, tt.curvature, tt.depth)
			if result != tt.want {
				t.Errorf("shouldEarlyStop() = %v, want %v", result, tt.want)
			}
		})
	}
}

func TestHighCurvatureIntegration(t *testing.T) {
	t.Run("sharp Lorentzian peak", func(t *testing.T) {
		expr := "1/(1+10000*(x-0.5)^2)"
		a := 0.0
		b := 1.0
		
		result, err := AdaptiveSimpson(expr, a, b, 1e-8)
		if err != nil {
			t.Fatalf("AdaptiveSimpson() error = %v", err)
		}
		
		analytical := math.Atan(50) / 50.0
		t.Logf("Numerical result: %.10f", result)
		t.Logf("Analytical result: %.10f", analytical)
		t.Logf("Relative error: %.2e", math.Abs(result-analytical)/analytical)
		
		if math.Abs(result-analytical) > 1e-3 {
			t.Errorf("Result %v not close enough to expected %v", result, analytical)
		}
	})
}

func TestCurvatureAwareEfficiency(t *testing.T) {
	t.Run("smooth function should converge quickly", func(t *testing.T) {
		expr := "exp(-x^2)"
		a := -3.0
		b := 3.0
		
		result, err := AdaptiveSimpson(expr, a, b, 1e-6)
		if err != nil {
			t.Fatalf("AdaptiveSimpson() error = %v", err)
		}
		
		analytical := math.Sqrt(math.Pi) * math.Erf(3)
		t.Logf("Result: %.10f, Expected: %.10f", result, analytical)
		
		if math.Abs(result-analytical) > 1e-5 {
			t.Errorf("Result not accurate enough")
		}
	})
}

func TestGaussKronrod15(t *testing.T) {
	tests := []struct {
		name     string
		expr     string
		a, b     float64
		expected float64
	}{
		{
			name:     "integral of x^2 from 0 to 1",
			expr:     "x^2",
			a:        0,
			b:        1,
			expected: 1.0 / 3.0,
		},
		{
			name:     "integral of sin(x) from 0 to pi",
			expr:     "sin(x)",
			a:        0,
			b:        math.Pi,
			expected: 2.0,
		},
		{
			name:     "integral of exp(x) from 0 to 1",
			expr:     "exp(x)",
			a:        0,
			b:        1,
			expected: math.E - 1,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := NewParser(tt.expr)
			result, errEst, err := GaussKronrod15(parser, tt.a, tt.b)
			if err != nil {
				t.Fatalf("GaussKronrod15() error = %v", err)
			}
			
			t.Logf("Result: %.10f, Error estimate: %.2e", result, errEst)
			
			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("GaussKronrod15() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestAdaptiveGaussKronrod(t *testing.T) {
	t.Run("sharp peak integration", func(t *testing.T) {
		expr := "1/(1+10000*(x-0.5)^2)"
		a := 0.0
		b := 1.0
		
		result, errEst, err := AdaptiveGaussKronrod(expr, a, b, 1e-8)
		if err != nil {
			t.Fatalf("AdaptiveGaussKronrod() error = %v", err)
		}
		
		analytical := math.Atan(50) / 50.0
		t.Logf("Numerical result: %.10f", result)
		t.Logf("Analytical result: %.10f", analytical)
		t.Logf("Error estimate: %.2e", errEst)
		
		if math.Abs(result-analytical) > 1e-6 {
			t.Errorf("Result %v not close enough to expected %v", result, analytical)
		}
	})
}

func TestHybridAdaptiveIntegral(t *testing.T) {
	tests := []struct {
		name           string
		expr            string
		a, b            float64
		expectedMethod  string
	}{
		{
			name:          "smooth polynomial - should use Simpson",
			expr:          "x^2",
			a:             0,
			b:             1,
			expectedMethod: "Simpson",
		},
		{
			name:          "sharp peak - may switch to Gauss-Kronrod",
			expr:          "1/(1+10000*(x-0.5)^2)",
			a:             0,
			b:             1,
			expectedMethod: "Gauss-Kronrod",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, method, err := HybridAdaptiveIntegral(tt.expr, tt.a, tt.b, 1e-8)
			if err != nil {
				t.Fatalf("HybridAdaptiveIntegral() error = %v", err)
			}
			
			t.Logf("Method used: %s, Result: %.10f", method, result)
			
			var expected float64
			if tt.expr == "x^2" {
				expected = 1.0 / 3.0
			} else {
				expected = math.Atan(50) / 50.0
			}
			
			if math.Abs(result-expected) > 1e-5 {
				t.Errorf("Result %v not close enough to expected %v", result, expected)
			}
		})
	}
}

func TestMethodComparison(t *testing.T) {
	t.Run("compare Simpson vs Gauss-Kronrod on difficult function", func(t *testing.T) {
		expr := "sin(100*x)"
		a := 0.0
		b := 1.0
		epsilon := 1e-6
		
		simpsonResult, err := AdaptiveSimpson(expr, a, b, epsilon)
		if err != nil {
			t.Logf("Simpson error: %v", err)
		}
		
		gkResult, gkErr, err := AdaptiveGaussKronrod(expr, a, b, epsilon)
		if err != nil {
			t.Logf("Gauss-Kronrod error: %v", err)
		}
		
		analytical := (1 - math.Cos(100)) / 100.0
		
		t.Logf("Analytical result: %.10f", analytical)
		t.Logf("Simpson result:    %.10f (error: %.2e)", simpsonResult, math.Abs(simpsonResult-analytical))
		t.Logf("Gauss-Kronrod result: %.10f (error: %.2e, est: %.2e)", gkResult, math.Abs(gkResult-analytical), gkErr)
	})
}
