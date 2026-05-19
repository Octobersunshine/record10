package sde

import (
	"math"
)

type Matrix [][]float64
type Vector []float64

func NewMatrix(rows, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

func NewVector(size int) Vector {
	return make(Vector, size)
}

func Cholesky(A Matrix) (Matrix, error) {
	n := len(A)
	L := NewMatrix(n, n)

	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := 0.0
			for k := 0; k < j; k++ {
				sum += L[i][k] * L[j][k]
			}
			if i == j {
				val := A[i][i] - sum
				if val <= 0 {
					return nil, &CholeskyError{msg: "matrix is not positive definite"}
				}
				L[i][j] = math.Sqrt(val)
			} else {
				L[i][j] = (A[i][j] - sum) / L[j][j]
			}
		}
	}
	return L, nil
}

type CholeskyError struct {
	msg string
}

func (e *CholeskyError) Error() string {
	return e.msg
}

func MatVecMul(L Matrix, v Vector) Vector {
	n := len(L)
	result := NewVector(n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j <= i; j++ {
			sum += L[i][j] * v[j]
		}
		result[i] = sum
	}
	return result
}

func VecAdd(a, b Vector) Vector {
	n := len(a)
	result := NewVector(n)
	for i := 0; i < n; i++ {
		result[i] = a[i] + b[i]
	}
	return result
}

func VecScale(v Vector, s float64) Vector {
	n := len(v)
	result := NewVector(n)
	for i := 0; i < n; i++ {
		result[i] = v[i] * s
	}
	return result
}

func CorrelationToCovariance(corr Matrix, sigmas Vector) Matrix {
	n := len(corr)
	cov := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			cov[i][j] = corr[i][j] * sigmas[i] * sigmas[j]
		}
	}
	return cov
}
