package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Vector3 struct {
	X, Y, Z float64
}

type CurvePoint struct {
	Point       Vector3
	Tangent     Vector3
	Normal      Vector3
	Binormal    Vector3
	Curvature   float64
	Torsion     float64
	ArcLength   float64
}

type BSplineCurve struct {
	Degree     int
	Knots      []float64
	ControlPts []Vector3
}

func (v Vector3) Add(u Vector3) Vector3 {
	return Vector3{v.X + u.X, v.Y + u.Y, v.Z + u.Z}
}

func (v Vector3) Sub(u Vector3) Vector3 {
	return Vector3{v.X - u.X, v.Y - u.Y, v.Z - u.Z}
}

func (v Vector3) Mul(s float64) Vector3 {
	return Vector3{v.X * s, v.Y * s, v.Z * s}
}

func (v Vector3) Dot(u Vector3) float64 {
	return v.X*u.X + v.Y*u.Y + v.Z*u.Z
}

func (v Vector3) Cross(u Vector3) Vector3 {
	return Vector3{
		v.Y*u.Z - v.Z*u.Y,
		v.Z*u.X - v.X*u.Z,
		v.X*u.Y - v.Y*u.X,
	}
}

func (v Vector3) Norm() float64 {
	return math.Sqrt(v.Dot(v))
}

func (v Vector3) Normalize() Vector3 {
	norm := v.Norm()
	if norm < 1e-10 {
		return Vector3{0, 0, 0}
	}
	return v.Mul(1.0 / norm)
}

func computeCumulativeChordLength(points []Vector3) []float64 {
	n := len(points)
	s := make([]float64, n)
	s[0] = 0.0
	for i := 1; i < n; i++ {
		s[i] = s[i-1] + points[i].Sub(points[i-1]).Norm()
	}
	return s
}

func bsplineBasis(t float64, i, degree int, knots []float64) float64 {
	if degree == 0 {
		if t >= knots[i] && t < knots[i+1] {
			if i+2 == len(knots)-1 && t == knots[i+1] {
				return 1.0
			}
			return 1.0
		}
		return 0.0
	}

	left := 0.0
	if knots[i+degree] != knots[i] {
		left = (t - knots[i]) / (knots[i+degree] - knots[i]) * bsplineBasis(t, i, degree-1, knots)
	}

	right := 0.0
	if knots[i+degree+1] != knots[i+1] {
		right = (knots[i+degree+1] - t) / (knots[i+degree+1] - knots[i+1]) * bsplineBasis(t, i+1, degree-1, knots)
	}

	return left + right
}

func bsplineBasisDeriv(t float64, i, degree int, knots []float64) float64 {
	if degree == 0 {
		return 0.0
	}

	left := 0.0
	if knots[i+degree] != knots[i] {
		left = float64(degree) / (knots[i+degree] - knots[i]) * bsplineBasis(t, i, degree-1, knots)
	}

	right := 0.0
	if knots[i+degree+1] != knots[i+1] {
		right = float64(degree) / (knots[i+degree+1] - knots[i+1]) * bsplineBasis(t, i+1, degree-1, knots)
	}

	return left - right
}

func generateKnots(params []float64, degree, numControlPts int) []float64 {
	n := len(params)
	knots := make([]float64, numControlPts+degree+1)

	for i := 0; i <= degree; i++ {
		knots[i] = params[0]
	}

	step := float64(n-1) / float64(numControlPts-degree)
	for i := degree + 1; i < numControlPts; i++ {
		idx := int(float64(i-degree) * step)
		if idx >= n-1 {
			idx = n-2
		}
		knots[i] = params[idx]
	}

	for i := numControlPts; i <= numControlPts+degree; i++ {
		knots[i] = params[n-1]
	}

	return knots
}

func buildDesignMatrix(params []float64, knots []float64, degree int) [][]float64 {
	n := len(params)
	m := len(knots) - degree - 1
	Phi := make([][]float64, n)
	for i := range Phi {
		Phi[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			Phi[i][j] = bsplineBasis(params[i], j, degree, knots)
		}
	}
	return Phi
}

func buildPenaltyMatrix(knots []float64, degree int) [][]float64 {
	m := len(knots) - degree - 1
	K := make([][]float64, m)
	for i := range K {
		K[i] = make([]float64, m)
	}

	for i := 0; i < m; i++ {
		if i+2 < m {
			K[i][i] += 1
			K[i][i+1] -= 2
			K[i][i+2] += 1
			K[i+1][i] -= 2
			K[i+1][i+1] += 4
			K[i+1][i+2] -= 2
			K[i+2][i] += 1
			K[i+2][i+1] -= 2
			K[i+2][i+2] += 1
		}
	}

	return K
}

func solveLinearSystem(A [][]float64, b []float64) []float64 {
	n := len(A)
	Ab := make([][]float64, n)
	for i := range Ab {
		Ab[i] = make([]float64, n+1)
		copy(Ab[i], A[i])
		Ab[i][n] = b[i]
	}

	for i := 0; i < n; i++ {
		pivot := i
		for j := i + 1; j < n; j++ {
			if math.Abs(Ab[j][i]) > math.Abs(Ab[pivot][i]) {
				pivot = j
			}
		}
		Ab[i], Ab[pivot] = Ab[pivot], Ab[i]

		if math.Abs(Ab[i][i]) < 1e-10 {
			return nil
		}

		for j := i + 1; j < n; j++ {
			factor := Ab[j][i] / Ab[i][i]
			for k := i; k <= n; k++ {
				Ab[j][k] -= factor * Ab[i][k]
			}
		}
	}

	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = Ab[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= Ab[i][j] * x[j]
		}
		x[i] /= Ab[i][i]
	}

	return x
}

func matMul(A [][]float64, B [][]float64) [][]float64 {
	m := len(A)
	n := len(B[0])
	p := len(B)
	C := make([][]float64, m)
	for i := range C {
		C[i] = make([]float64, n)
		for k := 0; k < p; k++ {
			for j := 0; j < n; j++ {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return C
}

func matTranspose(A [][]float64) [][]float64 {
	m := len(A)
	n := len(A[0])
	AT := make([][]float64, n)
	for i := range AT {
		AT[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			AT[i][j] = A[j][i]
		}
	}
	return AT
}

func matVecMul(A [][]float64, v []float64) []float64 {
	m := len(A)
	n := len(v)
	result := make([]float64, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result[i] += A[i][j] * v[j]
		}
	}
	return result
}

func computeGCV(Phi [][]float64, y []float64, K [][]float64, lambda float64) float64 {
	n := len(Phi)
	m := len(Phi[0])

	PhiT := matTranspose(Phi)
	PhiTPhi := matMul(PhiT, Phi)

	A := make([][]float64, m)
	for i := range A {
		A[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			A[i][j] = PhiTPhi[i][j] + lambda*K[i][j]
		}
	}

	c := make([]float64, m)
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			c[i] += PhiT[i][j] * y[j]
		}
	}

	x := solveLinearSystem(A, c)
	if x == nil {
		return math.Inf(1)
	}

	yHat := matVecMul(Phi, x)

	rss := 0.0
	for i := 0; i < n; i++ {
		rss += (y[i] - yHat[i]) * (y[i] - yHat[i])
	}

	trace := 0.0
	invA := make([][]float64, m)
	for i := range invA {
		invA[i] = make([]float64, m)
		ei := make([]float64, m)
		ei[i] = 1.0
		col := solveLinearSystem(A, ei)
		if col == nil {
			return math.Inf(1)
		}
		for j := 0; j < m; j++ {
			invA[j][i] = col[j]
		}
	}

	S := matMul(Phi, invA)
	S = matMul(S, PhiT)

	for i := 0; i < n; i++ {
		trace += S[i][i]
	}

	denom := float64(n) * (1.0 - trace/float64(n)) * (1.0 - trace/float64(n))
	if denom < 1e-10 {
		return math.Inf(1)
	}

	return rss / denom
}

func findBestLambda(Phi [][]float64, y []float64, K [][]float64) float64 {
	bestLambda := 0.0
	bestGCV := math.Inf(1)

	for exp := -6; exp <= 6; exp++ {
		lambda := math.Pow(10.0, float64(exp))
		gcv := computeGCV(Phi, y, K, lambda)
		if gcv < bestGCV {
			bestGCV = gcv
			bestLambda = lambda
		}
	}

	low := math.Log10(bestLambda) - 1.0
	high := math.Log10(bestLambda) + 1.0
	for i := 0; i < 20; i++ {
		mid1 := low + (high-low)/3.0
		mid2 := high - (high-low)/3.0
		gcv1 := computeGCV(Phi, y, K, math.Pow(10.0, mid1))
		gcv2 := computeGCV(Phi, y, K, math.Pow(10.0, mid2))
		if gcv1 < gcv2 {
			high = mid2
		} else {
			low = mid1
		}
	}

	return math.Pow(10.0, (low+high)/2.0)
}

func BSplineSmooth(points []Vector3, degree, numControlPts int, lambda float64) (*BSplineCurve, float64) {
	n := len(points)
	if n < degree+1 {
		return nil, 0
	}

	params := computeCumulativeChordLength(points)
	knots := generateKnots(params, degree, numControlPts)
	Phi := buildDesignMatrix(params, knots, degree)
	K := buildPenaltyMatrix(knots, degree)

	if lambda < 0 {
		xs := make([]float64, n)
		for i := 0; i < n; i++ {
			xs[i] = points[i].X
		}
		lambda = findBestLambda(Phi, xs, K)
	}

	PhiT := matTranspose(Phi)
	PhiTPhi := matMul(PhiT, Phi)
	m := len(Phi[0])

	A := make([][]float64, m)
	for i := range A {
		A[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			A[i][j] = PhiTPhi[i][j] + lambda*K[i][j]
		}
	}

	solveCoord := func(coords []float64) []float64 {
		b := make([]float64, m)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				b[i] += PhiT[i][j] * coords[j]
			}
		}
		return solveLinearSystem(A, b)
	}

	xs := make([]float64, n)
	ys := make([]float64, n)
	zs := make([]float64, n)
	for i := 0; i < n; i++ {
		xs[i] = points[i].X
		ys[i] = points[i].Y
		zs[i] = points[i].Z
	}

	cx := solveCoord(xs)
	cy := solveCoord(ys)
	cz := solveCoord(zs)

	if cx == nil || cy == nil || cz == nil {
		return nil, lambda
	}

	controlPts := make([]Vector3, m)
	for i := 0; i < m; i++ {
		controlPts[i] = Vector3{cx[i], cy[i], cz[i]}
	}

	return &BSplineCurve{
		Degree:     degree,
		Knots:      knots,
		ControlPts: controlPts,
	}, lambda
}

func (curve *BSplineCurve) Evaluate(t float64) Vector3 {
	m := len(curve.ControlPts)
	result := Vector3{0, 0, 0}
	for i := 0; i < m; i++ {
		basis := bsplineBasis(t, i, curve.Degree, curve.Knots)
		result = result.Add(curve.ControlPts[i].Mul(basis))
	}
	return result
}

func (curve *BSplineCurve) FirstDerivative(t float64) Vector3 {
	m := len(curve.ControlPts)
	result := Vector3{0, 0, 0}
	for i := 0; i < m; i++ {
		deriv := bsplineBasisDeriv(t, i, curve.Degree, curve.Knots)
		result = result.Add(curve.ControlPts[i].Mul(deriv))
	}
	return result
}

func (curve *BSplineCurve) SecondDerivative(t float64) Vector3 {
	if curve.Degree < 2 {
		return Vector3{0, 0, 0}
	}

	degree := curve.Degree - 1
	m := len(curve.ControlPts) - 1

	ctrlPts := make([]Vector3, m)
	for i := 0; i < m; i++ {
		dt := curve.Knots[i+curve.Degree+1] - curve.Knots[i+1]
		if dt > 1e-10 {
			ctrlPts[i] = curve.ControlPts[i+1].Sub(curve.ControlPts[i]).Mul(float64(curve.Degree) / dt)
		}
	}

	result := Vector3{0, 0, 0}
	for i := 0; i < m-1; i++ {
		deriv := bsplineBasisDeriv(t, i, degree, curve.Knots[1:len(curve.Knots)-1])
		dt := curve.Knots[i+degree+1] - curve.Knots[i+1]
		if dt > 1e-10 {
			deriv *= float64(degree) / dt
		}
		result = result.Add(ctrlPts[i].Mul(deriv))
	}
	return result
}

func ComputeFrenetSerretBSpline(points []Vector3, lambda float64) []CurvePoint {
	n := len(points)
	if n < 4 {
		return nil
	}

	degree := 3
	numControlPts := int(math.Min(float64(n), float64(n*2/3)))
	if numControlPts < degree+1 {
		numControlPts = degree + 1
	}

	spline, usedLambda := BSplineSmooth(points, degree, numControlPts, lambda)
	if spline == nil {
		return ComputeFrenetSerretSimple(points)
	}

	fmt.Printf("使用的平滑参数 lambda = %.6e\n", usedLambda)

	params := computeCumulativeChordLength(points)
	result := make([]CurvePoint, n)

	for i := 0; i < n; i++ {
		t := params[i]

		r1 := spline.FirstDerivative(t)
		r2 := spline.SecondDerivative(t)

		var T, N, B Vector3
		var kappa, tau float64

		r1Norm := r1.Norm()
		if r1Norm > 1e-10 {
			T = r1.Normalize()

			r1CrossR2 := r1.Cross(r2)
			r1CrossR2Norm := r1CrossR2.Norm()

			if r1CrossR2Norm > 1e-10 {
				B = r1CrossR2.Normalize()
				N = B.Cross(T)
				kappa = r1CrossR2Norm / (r1Norm * r1Norm * r1Norm)
				tau = 0
			} else {
				perp := Vector3{-r1.Y, r1.X, 0}
				if perp.Norm() < 1e-10 {
					perp = Vector3{0, -r1.Z, r1.Y}
				}
				N = perp.Normalize()
				B = T.Cross(N)
				kappa = 0
				tau = 0
			}
		} else {
			T = Vector3{1, 0, 0}
			N = Vector3{0, 1, 0}
			B = Vector3{0, 0, 1}
			kappa = 0
			tau = 0
		}

		smoothedPoint := spline.Evaluate(t)

		result[i] = CurvePoint{
			Point:     smoothedPoint,
			Tangent:   T,
			Normal:    N,
			Binormal:  B,
			Curvature: kappa,
			Torsion:   tau,
			ArcLength: t,
		}
	}

	return result
}

func ComputeFrenetSerretSimple(points []Vector3) []CurvePoint {
	n := len(points)
	if n < 3 {
		return nil
	}

	arcLengths := computeCumulativeChordLength(points)
	result := make([]CurvePoint, n)

	for i := 0; i < n; i++ {
		var r1 Vector3

		if i == 0 {
			ds := arcLengths[2] - arcLengths[0]
			r1 = points[2].Sub(points[0]).Mul(1.0 / ds)
		} else if i == n-1 {
			ds := arcLengths[n-1] - arcLengths[n-3]
			r1 = points[n-1].Sub(points[n-3]).Mul(1.0 / ds)
		} else {
			ds := arcLengths[i+1] - arcLengths[i-1]
			r1 = points[i+1].Sub(points[i-1]).Mul(1.0 / ds)
		}

		var T, N, B Vector3
		var kappa, tau float64

		r1Norm := r1.Norm()
		if r1Norm > 1e-10 {
			T = r1.Normalize()

			if i > 0 && i < n-1 {
				T_prev := points[i].Sub(points[i-1]).Normalize()
				T_next := points[i+1].Sub(points[i]).Normalize()
				crossDir := T_prev.Cross(T_next)
				crossNorm := crossDir.Norm()
				if crossNorm > 1e-10 {
					ds := arcLengths[i+1] - arcLengths[i-1]
					kappa = 2 * crossNorm / ds
					B = crossDir.Normalize()
					N = B.Cross(T)
				} else {
					perp := Vector3{-T.Y, T.X, 0}
					if perp.Norm() < 1e-10 {
						perp = Vector3{0, -T.Z, T.Y}
					}
					N = perp.Normalize()
					B = T.Cross(N)
					kappa = 0
				}
			} else {
				perp := Vector3{-T.Y, T.X, 0}
				if perp.Norm() < 1e-10 {
					perp = Vector3{0, -T.Z, T.Y}
				}
				N = perp.Normalize()
				B = T.Cross(N)
				kappa = 0
			}
			tau = 0
		} else {
			T = Vector3{1, 0, 0}
			N = Vector3{0, 1, 0}
			B = Vector3{0, 0, 1}
			kappa = 0
			tau = 0
		}

		result[i] = CurvePoint{
			Point:     points[i],
			Tangent:   T,
			Normal:    N,
			Binormal:  B,
			Curvature: kappa,
			Torsion:   tau,
			ArcLength: arcLengths[i],
		}
	}

	return result
}

func ComputeFrenetSerret(points []Vector3) []CurvePoint {
	return ComputeFrenetSerretBSpline(points, -1.0)
}

func GenerateHelix(radius, pitch, turns float64, numPoints int) []Vector3 {
	points := make([]Vector3, numPoints)
	for i := 0; i < numPoints; i++ {
		t := float64(i) / float64(numPoints-1) * 2 * math.Pi * turns
		points[i] = Vector3{
			X: radius * math.Cos(t),
			Y: radius * math.Sin(t),
			Z: pitch * t / (2 * math.Pi),
		}
	}
	return points
}

func GenerateNoisyHelix(radius, pitch, turns, noiseLevel float64, numPoints int) []Vector3 {
	points := make([]Vector3, numPoints)
	for i := 0; i < numPoints; i++ {
		t := float64(i) / float64(numPoints-1) * 2 * math.Pi * turns
		points[i] = Vector3{
			X: radius * math.Cos(t)  + (rand.Float64()-0.5)*noiseLevel,
			Y: radius * math.Sin(t) + (rand.Float64()-0.5)*noiseLevel,
			Z: pitch * t / (2 * math.Pi) + (rand.Float64()-0.5)*noiseLevel,
		}
	}
	return points
}

func GenerateCircle(radius float64, numPoints int) []Vector3 {
	points := make([]Vector3, numPoints)
	for i := 0; i < numPoints; i++ {
		t := float64(i) / float64(numPoints-1) * 2 * math.Pi
		points[i] = Vector3{
			X: radius * math.Cos(t),
			Y: radius * math.Sin(t),
			Z: 0,
		}
	}
	return points
}

func GenerateNoisyCircle(radius, noiseLevel float64, numPoints int) []Vector3 {
	points := make([]Vector3, numPoints)
	for i := 0; i < numPoints; i++ {
		t := float64(i) / float64(numPoints-1) * 2 * math.Pi
		points[i] = Vector3{
			X: radius * math.Cos(t)  + (rand.Float64()-0.5)*noiseLevel,
			Y: radius * math.Sin(t) + (rand.Float64()-0.5)*noiseLevel,
			Z: 0 + (rand.Float64()-0.5)*noiseLevel,
		}
	}
	return points
}

func main() {
	rand.Seed(42)
	fmt.Println("=== Frenet-Serret 公式计算 (B样条平滑 + GCV) ===")
	fmt.Println()

	fmt.Println("1. 带噪声的螺旋线测试 (B样条平滑效果")
	fmt.Println("------------------------------------")
	noisyHelix := GenerateNoisyHelix(1.0, 1.0, 2.0, 0.05, 50)
	smoothedResults := ComputeFrenetSerretBSpline(noisyHelix, -1.0)
	rawResults := ComputeFrenetSerretSimple(noisyHelix)

	fmt.Println()
	fmt.Printf("理论曲率 κ = 0.5, 理论挠率 τ = 0.5")
	fmt.Println()
	fmt.Printf("%-8s %-12s %-12s %-12s %-12s\n", "点索引", "原始曲率", "平滑后曲率", "原始挠率", "平滑后挠率")
	fmt.Println("--------------------------------------------------------")
	for i := 10; i <= 40; i += 10 {
		fmt.Printf("%-8d %-12.6f %-12.6f %-12.6f %-12.6f\n",
			i,
			rawResults[i].Curvature,
			smoothedResults[i].Curvature,
			rawResults[i].Torsion,
			smoothedResults[i].Torsion)
	}
	fmt.Println()

	fmt.Println("2. 带噪声的平面圆测试 (半径=2, 噪声=0.05)")
	fmt.Println("----------------------------------------")
	noisyCircle := GenerateNoisyCircle(2.0, 0.05, 40)
	smoothedCircle := ComputeFrenetSerretBSpline(noisyCircle, -1.0)
	rawCircle := ComputeFrenetSerretSimple(noisyCircle)

	fmt.Println()
	fmt.Printf("理论曲率 κ = 0.5 (1/2)")
	fmt.Println()
	fmt.Printf("%-8s %-12s %-12s\n", "点索引", "原始曲率", "平滑后曲率")
	fmt.Println("--------------------------------")
	for i := 5; i <= 35; i += 10 {
		fmt.Printf("%-8d %-12.6f %-12.6f\n",
			i,
			rawCircle[i].Curvature,
			smoothedCircle[i].Curvature)
	}
	fmt.Println()

	fmt.Println("3. 平滑效果统计")
	fmt.Println("-------------")
	maxRaw := 0.0
	maxSmooth := 0.0
	meanRaw := 0.0
	meanSmooth := 0.0
	count := 0
	for i := 5; i < len(noisyHelix)-5; i++ {
		maxRaw = math.Max(maxRaw, math.Abs(rawResults[i].Curvature-0.5))
		maxSmooth = math.Max(maxSmooth, math.Abs(smoothedResults[i].Curvature-0.5))
		meanRaw += math.Abs(rawResults[i].Curvature - 0.5)
		meanSmooth += math.Abs(smoothedResults[i].Curvature - 0.5)
		count++
	}
	meanRaw /= float64(count)
	meanSmooth /= float64(count)

	fmt.Printf("曲率误差 (与理论值0.5比较:")
	fmt.Printf("  原始数据 - 最大误差: %.6f, 平均误差: %.6f\n", maxRaw, meanRaw)
	fmt.Printf("  平滑后 - 最大误差: %.6f, 平均误差: %.6f\n", maxSmooth, meanSmooth)
	fmt.Printf("  平滑效果提升: %.1f%% (平均误差降低)\n", (meanRaw-meanSmooth)/meanRaw*100)
}
