package cgsolver

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type Preconditioner interface {
	Apply(z, r mat.Vector)
}

type JacobiPreconditioner struct {
	invD *mat.VecDense
}

func NewJacobiPreconditioner(A mat.Matrix) *JacobiPreconditioner {
	n, _ := A.Dims()
	invD := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		invD.SetVec(i, 1.0/A.At(i, i))
	}
	return &JacobiPreconditioner{invD: invD}
}

func (j *JacobiPreconditioner) Apply(z, r mat.Vector) {
	zVec := z.(*mat.VecDense)
	rVec := r.(*mat.VecDense)
	zVec.MulElemVec(rVec, j.invD)
}

type ILUPreconditioner struct {
	L *mat.TriDense
	U *mat.TriDense
}

func NewILUPreconditioner(A mat.Matrix) *ILUPreconditioner {
	n, _ := A.Dims()
	
	L := mat.NewTriDense(n, mat.Lower, nil)
	U := mat.NewTriDense(n, mat.Upper, nil)
	
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := 0.0
			for k := 0; k < j; k++ {
				sum += L.At(i, k) * U.At(k, j)
			}
			if i == j {
				U.Set(i, j, A.At(i, j)-sum)
				L.Set(i, j, 1.0)
			} else {
				L.Set(i, j, (A.At(i, j)-sum)/U.At(j, j))
			}
		}
		for j := i + 1; j < n; j++ {
			sum := 0.0
			for k := 0; k < i; k++ {
				sum += L.At(i, k) * U.At(k, j)
			}
			U.Set(i, j, A.At(i, j)-sum)
		}
	}
	
	return &ILUPreconditioner{L: L, U: U}
}

func (ilu *ILUPreconditioner) Apply(z, r mat.Vector) {
	n := r.Len()
	y := mat.NewVecDense(n, nil)
	
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < i; j++ {
			sum += ilu.L.At(i, j) * y.At(j, 0)
		}
		y.SetVec(i, r.At(i, 0)-sum)
	}
	
	zVec := z.(*mat.VecDense)
	for i := n - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < n; j++ {
			sum += ilu.U.At(i, j) * zVec.At(j, 0)
		}
		zVec.SetVec(i, (y.At(i, 0)-sum)/ilu.U.At(i, i))
	}
}

type NoPreconditioner struct{}

func (n *NoPreconditioner) Apply(z, r mat.Vector) {
	z.(*mat.VecDense).CloneFromVec(r)
}

type MatrixAnalysis struct {
	IsDiagonallyDominant bool
	DiagonalStrength     float64
	Sparsity             float64
	ConditionEstimate    float64
}

func AnalyzeMatrix(A mat.Matrix) *MatrixAnalysis {
	n, _ := A.Dims()
	
	ddCount := 0
	totalDiag := 0.0
	totalOffDiag := 0.0
	nonZeroCount := 0
	
	for i := 0; i < n; i++ {
		diag := math.Abs(A.At(i, i))
		totalDiag += diag
		rowSum := 0.0
		
		for j := 0; j < n; j++ {
			val := A.At(i, j)
			if math.Abs(val) > 1e-15 {
				nonZeroCount++
			}
			if i != j {
				rowSum += math.Abs(val)
				totalOffDiag += math.Abs(val)
			}
		}
		
		if diag >= rowSum-1e-15 {
			ddCount++
		}
	}
	
	return &MatrixAnalysis{
		IsDiagonallyDominant: ddCount == n,
		DiagonalStrength:     totalDiag / (totalDiag + totalOffDiag),
		Sparsity:             float64(nonZeroCount) / float64(n*n),
		ConditionEstimate:    1.0 + totalOffDiag/totalDiag,
	}
}

func SelectPreconditioner(A mat.Matrix) Preconditioner {
	analysis := AnalyzeMatrix(A)
	n, _ := A.Dims()
	
	if analysis.IsDiagonallyDominant || analysis.DiagonalStrength > 0.7 {
		return NewJacobiPreconditioner(A)
	}
	
	if n <= 200 || analysis.Sparsity > 0.3 {
		return NewILUPreconditioner(A)
	}
	
	return NewJacobiPreconditioner(A)
}

type CGSolver struct {
	MaxIter             int
	Tol                 float64
	ReorthogonalizeFreq int
	Preconditioner      Preconditioner
	AutoPrecondition    bool
	SelectedPrecond     string
}

func New() *CGSolver {
	return &CGSolver{
		MaxIter:             1000,
		Tol:                 1e-10,
		ReorthogonalizeFreq: 10,
		AutoPrecondition:    true,
	}
}

func (cg *CGSolver) WithMaxIter(maxIter int) *CGSolver {
	cg.MaxIter = maxIter
	return cg
}

func (cg *CGSolver) WithTolerance(tol float64) *CGSolver {
	cg.Tol = tol
	return cg
}

func (cg *CGSolver) WithReorthogonalizeFreq(freq int) *CGSolver {
	cg.ReorthogonalizeFreq = freq
	return cg
}

func (cg *CGSolver) WithPreconditioner(p Preconditioner) *CGSolver {
	cg.Preconditioner = p
	cg.AutoPrecondition = false
	return cg
}

func (cg *CGSolver) WithAutoPreconditioner(enable bool) *CGSolver {
	cg.AutoPrecondition = enable
	return cg
}

func GetPreconditionerName(p Preconditioner) string {
	switch p.(type) {
	case *JacobiPreconditioner:
		return "Jacobi (对角)"
	case *ILUPreconditioner:
		return "ILU (不完全LU)"
	case *NoPreconditioner:
		return "无预条件"
	default:
		return "自定义"
	}
}

func (cg *CGSolver) Solve(A mat.Matrix, b mat.Vector) (x *mat.VecDense, iterations int, err error) {
	n, m := A.Dims()
	if n != m {
		return nil, 0, fmt.Errorf("matrix A must be square, got %dx%d", n, m)
	}

	bLen := b.Len()
	if bLen != n {
		return nil, 0, fmt.Errorf("vector b length %d does not match matrix size %d", bLen, n)
	}

	var M Preconditioner
	if cg.AutoPrecondition {
		M = SelectPreconditioner(A)
	} else if cg.Preconditioner != nil {
		M = cg.Preconditioner
	} else {
		M = &NoPreconditioner{}
	}
	cg.SelectedPrecond = GetPreconditionerName(M)

	x = mat.NewVecDense(n, nil)

	r := mat.NewVecDense(n, nil)
	r.MulVec(A, x)
	r.SubVec(b, r)

	z := mat.NewVecDense(n, nil)
	M.Apply(z, r)

	p := mat.NewVecDense(n, nil)
	p.CloneFromVec(z)

	rsold := mat.Dot(r, z)

	for k := 0; k < cg.MaxIter; k++ {
		Ap := mat.NewVecDense(n, nil)
		Ap.MulVec(A, p)

		pAp := mat.Dot(p, Ap)

		if math.Abs(pAp) < 1e-15 {
			return x, k + 1, fmt.Errorf("encountered numerical instability: p^T A p is too small")
		}

		alpha := rsold / pAp

		x.AddScaledVec(x, alpha, p)
		r.AddScaledVec(r, -alpha, Ap)

		if cg.ReorthogonalizeFreq > 0 && (k+1)%cg.ReorthogonalizeFreq == 0 {
			r.MulVec(A, x)
			r.SubVec(b, r)
		}

		M.Apply(z, r)
		rsnew := mat.Dot(r, z)

		if math.Sqrt(mat.Dot(r, r)) < cg.Tol {
			return x, k + 1, nil
		}

		beta := rsnew / rsold
		p.AddScaledVec(z, beta, p)

		rsold = rsnew
	}

	return x, cg.MaxIter, fmt.Errorf("did not converge within %d iterations, residual norm: %.2e", cg.MaxIter, math.Sqrt(mat.Dot(r, r)))
}
