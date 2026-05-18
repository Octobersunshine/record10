package solver

import (
	"fmt"
	"math"
	"math/cmplx"
	"newton_raphson_grpc/pkg/mathutil"
)

type RootType int

const (
	RealRoot RootType = iota
	ComplexRoot
)

type Result struct {
	Root         float64
	RootComplex  complex128
	RootType     RootType
	Iterations   int
	Converged    bool
	Error        string
	Multiplicity int
	UsedFallback bool
	FallbackInfo string
	Validation   mathutil.ValidationResult
	Derivative   string
}

func estimateMultiplicity(fx, dfx, d2fx float64) int {
	if math.Abs(fx) < 1e-10 {
		return 1
	}
	
	numerator := dfx * dfx
	denominator := dfx*dfx - fx*d2fx
	
	if math.Abs(denominator) < 1e-12 {
		return 1
	}
	
	mEstimate := numerator / denominator
	m := math.Round(mEstimate)
	
	if m >= 1 && m <= 10 {
		return int(m)
	}
	return 1
}

func NewtonRaphsonComplex(expr string, initialGuess complex128, precision float64, maxIterations int) Result {
	x := initialGuess
	
	derivStr, err := mathutil.SymbolicDerivative(expr)
	if err != nil {
		return Result{
			RootType: ComplexRoot,
			Error:    "Failed to compute symbolic derivative: " + err.Error(),
		}
	}
	
	for i := 0; i < maxIterations; i++ {
		fx, err := mathutil.EvaluateComplex(expr, x)
		if err != nil {
			return Result{
				RootType:    ComplexRoot,
				RootComplex: x,
				Error:       "Failed to evaluate complex expression: " + err.Error(),
				Derivative:  derivStr,
			}
		}

		if cmplx.Abs(fx) < precision {
			return Result{
				RootType:    ComplexRoot,
				RootComplex: x,
				Iterations:  i + 1,
				Converged:   true,
				Derivative:  derivStr,
			}
		}

		dfx, err := mathutil.EvaluateDerivativeComplex(expr, x)
		if err != nil {
			return Result{
				RootType:    ComplexRoot,
				RootComplex: x,
				Error:       "Failed to compute symbolic complex derivative: " + err.Error(),
				Derivative:  derivStr,
			}
		}

		if cmplx.Abs(dfx) < 1e-12 {
			return Result{
				RootType:    ComplexRoot,
				RootComplex: x,
				Error:       "Complex derivative is too close to zero, cannot continue",
				Derivative:  derivStr,
			}
		}

		deltaX := fx / dfx
		x = x - deltaX

		if cmplx.Abs(deltaX) < precision {
			return Result{
				RootType:    ComplexRoot,
				RootComplex: x,
				Iterations:  i + 1,
				Converged:   true,
				Derivative:  derivStr,
			}
		}
	}

	return Result{
		RootType:    ComplexRoot,
		RootComplex: x,
		Iterations:  maxIterations,
		Converged:   false,
		Error:       "Maximum iterations reached without convergence in complex mode",
		Derivative:  derivStr,
	}
}

func NewtonRaphson(expr string, initialGuess float64, precision float64, maxIterations int) Result {
	validation := mathutil.ValidateInitialValue(expr, initialGuess)
	
	derivStr, derivErr := mathutil.SymbolicDerivative(expr)
	
	if !validation.IsValid {
		return Result{
			Error:      validation.Message,
			Validation: validation,
			Derivative: derivStr,
		}
	}

	x := initialGuess
	if !validation.IsInDomain {
		x = validation.Suggestion
	}

	m := 1
	consecutiveSlowConvergence := 0
	prevDeltaX := 0.0
	prevFx := 0.0
	
	for i := 0; i < maxIterations; i++ {
		fx, err := mathutil.Evaluate(expr, x)
		if err != nil {
			return Result{
				Error:      "Failed to evaluate expression: " + err.Error(),
				Validation: validation,
				Derivative: derivStr,
			}
		}

		if math.Abs(fx) < precision {
			return Result{
				Root:         x,
				RootType:     RealRoot,
				Iterations:   i + 1,
				Converged:    true,
				Multiplicity: m,
				Validation:   validation,
				Derivative:   derivStr,
			}
		}

		if i > 3 && mathutil.IsDiverging(fx, prevFx) {
			complexGuess := complex(x, 0.1)
			complexResult := NewtonRaphsonComplex(expr, complexGuess, precision, maxIterations)
			complexResult.UsedFallback = true
			complexResult.FallbackInfo = fmt.Sprintf("Real iteration diverged at x=%.6f, switched to complex mode with initial guess %.3f%+.3fi", x, real(complexGuess), imag(complexGuess))
			complexResult.Validation = validation
			return complexResult
		}
		prevFx = fx

		dfx, err := mathutil.EvaluateDerivative(expr, x)
		if err != nil {
			if derivErr != nil {
				return Result{
					Error:      "Failed to compute symbolic derivative: " + derivErr.Error(),
					Validation: validation,
					Derivative: derivStr,
				}
			}
			return Result{
				Error:      "Failed to evaluate symbolic derivative: " + err.Error(),
				Validation: validation,
				Derivative: derivStr,
			}
		}

		if math.Abs(dfx) < 1e-12 {
			dfxNode, err := mathutil.ParseExpression(derivStr)
			if err == nil {
				d2fxNode := dfxNode.Differentiate().Simplify()
				d2fx := d2fxNode.Evaluate(x)
				if math.Abs(d2fx) > 1e-12 {
					deltaX := math.Sqrt(2 * math.Abs(fx) / math.Abs(d2fx))
					if fx*d2fx < 0 {
						x = x + deltaX
					} else {
						x = x - deltaX
					}
					continue
				}
			}
			
			complexGuess := complex(x, 0.1)
			complexResult := NewtonRaphsonComplex(expr, complexGuess, precision, maxIterations-i)
			complexResult.Iterations += i
			complexResult.UsedFallback = true
			complexResult.FallbackInfo = fmt.Sprintf("Real derivative vanished at x=%.6f, switched to complex mode", x)
			complexResult.Validation = validation
			return complexResult
		}

		deltaX := fx / dfx
		
		if i > 2 && math.Abs(deltaX) > 1e-10 && math.Abs(prevDeltaX) > 1e-10 {
			ratio := math.Abs(deltaX) / math.Abs(prevDeltaX)
			if ratio > 0.5 && ratio < 1.0 {
				consecutiveSlowConvergence++
			} else {
				consecutiveSlowConvergence = 0
			}
			
			if consecutiveSlowConvergence >= 2 {
				dfxNode, err := mathutil.ParseExpression(derivStr)
				if err == nil {
					d2fxNode := dfxNode.Differentiate().Simplify()
					d2fx := d2fxNode.Evaluate(x)
					if math.Abs(d2fx) > 1e-12 {
						m = estimateMultiplicity(fx, dfx, d2fx)
						deltaX = float64(m) * fx / dfx
					}
				}
			}
		}
		
		prevDeltaX = deltaX
		x = x - deltaX

		if math.Abs(deltaX) < precision {
			return Result{
				Root:         x,
				RootType:     RealRoot,
				Iterations:   i + 1,
				Converged:    true,
				Multiplicity: m,
				Validation:   validation,
				Derivative:   derivStr,
			}
		}
	}

	complexGuess := complex(x, 0.1)
	complexResult := NewtonRaphsonComplex(expr, complexGuess, precision, maxIterations/2)
	complexResult.UsedFallback = true
	complexResult.FallbackInfo = "Real iteration did not converge within max iterations, switched to complex mode"
	complexResult.Validation = validation
	return complexResult
}
