package integration

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

type NewtonCotes struct{}

type Romberg struct {
	Tolerance float64
	MaxLevels int
}

func bernoulliNumber(n int) float64 {
	B := make([]float64, n+1)
	B[0] = 1.0
	for m := 1; m <= n; m++ {
		B[m] = 0.0
		for k := 0; k < m; k++ {
			comb := binomial(m+1, k)
			B[m] -= comb * B[k]
		}
		B[m] /= float64(m + 1)
	}
	return B[n]
}

func binomial(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	k = min(k, n-k)
	res := 1.0
	for i := 1; i <= k; i++ {
		res *= float64(n - k + i)
		res /= float64(i)
	}
	return res
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func lagrangeBasis(j int, points []float64, t float64) float64 {
	result := 1.0
	n := len(points)
	for m := 0; m < n; m++ {
		if m != j {
			result *= (t - points[m]) / (points[j] - points[m])
		}
	}
	return result
}

func integrateLagrange(j int, n int) float64 {
	points := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		points[i] = float64(i)
	}

	N := 10000
	h := 1.0 / float64(N)
	sum := 0.0

	for i := 0; i < N; i++ {
		x := float64(i)*h + h/2
		sum += lagrangeBasis(j, points, x) * h
	}

	return sum
}

func (nc *NewtonCotes) computeCotesCoefficients(n int) ([]float64, float64) {
	weights := make([]float64, n+1)
	
	for j := 0; j <= n; j++ {
		weights[j] = integrateLagrange(j, n)
	}

	sum := 0.0
	for _, w := range weights {
		sum += w
	}

	return weights, 1.0
}

func (nc *NewtonCotes) getWeights(n int) []float64 {
	switch n {
	case 1:
		return []float64{1, 1}
	case 2:
		return []float64{1, 4, 1}
	case 3:
		return []float64{1, 3, 3, 1}
	case 4:
		return []float64{7, 32, 12, 32, 7}
	case 5:
		return []float64{19, 75, 50, 50, 75, 19}
	case 6:
		return []float64{41, 216, 27, 272, 27, 216, 41}
	default:
		weights, _ := nc.computeCotesCoefficients(n)
		return weights
	}
}

func (nc *NewtonCotes) getCoefficient(n int) float64 {
	switch n {
	case 1:
		return 1.0 / 2.0
	case 2:
		return 1.0 / 3.0
	case 3:
		return 3.0 / 8.0
	case 4:
		return 2.0 / 45.0
	case 5:
		return 5.0 / 288.0
	case 6:
		return 1.0 / 140.0
	default:
		return 1.0
	}
}

func (nc *NewtonCotes) parseFunction(expr string) (func(float64) float64, error) {
	return func(x float64) float64 {
		return evaluateExpr(expr, x)
	}, nil
}

func evaluateExpr(expr string, x float64) float64 {
	expr = strings.ReplaceAll(expr, " ", "")
	expr = strings.ReplaceAll(expr, "pi", fmt.Sprintf("%f", math.Pi))
	expr = strings.ReplaceAll(expr, "e", fmt.Sprintf("%f", math.E))
	
	return parseExpr(expr, x)
}

func parseExpr(expr string, x float64) float64 {
	expr = strings.ReplaceAll(expr, "x", fmt.Sprintf("%f", x))
	
	expr = replaceFunctions(expr)
	
	result, _ := strconv.ParseFloat(evaluateSimple(expr), 64)
	return result
}

func replaceFunctions(expr string) string {
	functions := []string{"sin", "cos", "tan", "sqrt", "log", "exp", "asin", "acos", "atan"}
	
	for _, fn := range functions {
		for strings.Contains(expr, fn+"(") {
			start := strings.Index(expr, fn+"(")
			if start == -1 {
				break
			}
			parenCount := 1
			end := start + len(fn) + 1
			for end < len(expr) && parenCount > 0 {
				if expr[end] == '(' {
					parenCount++
				} else if expr[end] == ')' {
					parenCount--
				}
				end++
			}
			inner := expr[start+len(fn)+1 : end-1]
			innerVal := evaluateSimple(inner)
			var result float64
			switch fn {
			case "sin":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Sin(val)
			case "cos":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Cos(val)
			case "tan":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Tan(val)
			case "sqrt":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Sqrt(val)
			case "log":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Log(val)
			case "exp":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Exp(val)
			case "asin":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Asin(val)
			case "acos":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Acos(val)
			case "atan":
				val, _ := strconv.ParseFloat(innerVal, 64)
				result = math.Atan(val)
			}
			expr = expr[:start] + fmt.Sprintf("%f", result) + expr[end:]
		}
	}
	return expr
}

func evaluateSimple(expr string) string {
	for strings.Contains(expr, "(") {
		start := strings.LastIndex(expr, "(")
		parenCount := 1
		end := start + 1
		for end < len(expr) && parenCount > 0 {
			if expr[end] == '(' {
				parenCount++
			} else if expr[end] == ')' {
				parenCount--
			}
			end++
		}
		inner := expr[start+1 : end-1]
		innerVal := evaluateBasic(inner)
		expr = expr[:start] + innerVal + expr[end:]
	}
	return evaluateBasic(expr)
}

func evaluateBasic(expr string) string {
	for strings.Contains(expr, "^") {
		idx := strings.Index(expr, "^")
		leftStart := idx - 1
		for leftStart >= 0 && (isDigit(expr[leftStart]) || expr[leftStart] == '.' || expr[leftStart] == '-') {
			leftStart--
		}
		leftStart++
		
		rightEnd := idx + 1
		for rightEnd < len(expr) && (isDigit(expr[rightEnd]) || expr[rightEnd] == '.') {
			rightEnd++
		}
		
		left, _ := strconv.ParseFloat(expr[leftStart:idx], 64)
		right, _ := strconv.ParseFloat(expr[idx+1:rightEnd], 64)
		result := math.Pow(left, right)
		
		expr = expr[:leftStart] + fmt.Sprintf("%f", result) + expr[rightEnd:]
	}
	
	for strings.Contains(expr, "*") || strings.Contains(expr, "/") {
		multIdx := strings.Index(expr, "*")
		divIdx := strings.Index(expr, "/")
		
		var idx int
		var isMult bool
		if multIdx != -1 && (divIdx == -1 || multIdx < divIdx) {
			idx = multIdx
			isMult = true
		} else if divIdx != -1 {
			idx = divIdx
			isMult = false
		} else {
			break
		}
		
		leftStart := idx - 1
		for leftStart >= 0 && (isDigit(expr[leftStart]) || expr[leftStart] == '.' || expr[leftStart] == '-') {
			leftStart--
		}
		leftStart++
		
		rightEnd := idx + 1
		for rightEnd < len(expr) && (isDigit(expr[rightEnd]) || expr[rightEnd] == '.') {
			rightEnd++
		}
		
		left, _ := strconv.ParseFloat(expr[leftStart:idx], 64)
		right, _ := strconv.ParseFloat(expr[idx+1:rightEnd], 64)
		var result float64
		if isMult {
			result = left * right
		} else {
			result = left / right
		}
		
		expr = expr[:leftStart] + fmt.Sprintf("%f", result) + expr[rightEnd:]
	}
	
	for strings.Contains(expr, "+") || strings.Contains(expr, "-") {
		addIdx := strings.Index(expr, "+")
		subIdx := strings.Index(expr[1:], "-") + 1
		if subIdx == 0 {
			subIdx = -1
		}
		
		var idx int
		var isAdd bool
		if addIdx != -1 && (subIdx == -1 || addIdx < subIdx) {
			idx = addIdx
			isAdd = true
		} else if subIdx != -1 {
			idx = subIdx
			isAdd = false
		} else {
			break
		}
		
		leftStart := idx - 1
		for leftStart >= 0 && (isDigit(expr[leftStart]) || expr[leftStart] == '.') {
			leftStart--
		}
		leftStart++
		
		rightEnd := idx + 1
		for rightEnd < len(expr) && (isDigit(expr[rightEnd]) || expr[rightEnd] == '.' || expr[rightEnd] == '-') {
			rightEnd++
		}
		
		left, _ := strconv.ParseFloat(expr[leftStart:idx], 64)
		right, _ := strconv.ParseFloat(expr[idx+1:rightEnd], 64)
		var result float64
		if isAdd {
			result = left + right
		} else {
			result = left - right
		}
		
		expr = expr[:leftStart] + fmt.Sprintf("%f", result) + expr[rightEnd:]
	}
	
	return expr
}

func isDigit(c byte) bool {
	return c >= '0' && c <= '9'
}

func (nc *NewtonCotes) Integrate(expr string, a, b float64, n, N int) (float64, error) {
	if n < 1 {
		return 0, fmt.Errorf("n must be at least 1")
	}
	if N < 1 {
		return 0, fmt.Errorf("N must be at least 1")
	}
	if a >= b {
		return 0, fmt.Errorf("a must be less than b")
	}

	f, err := nc.parseFunction(expr)
	if err != nil {
		return 0, err
	}

	h := (b - a) / float64(N)
	weights := nc.getWeights(n)
	coeff := nc.getCoefficient(n)
	subH := h / float64(n)

	var total float64

	for i := 0; i < N; i++ {
		startX := a + float64(i)*h
		var sum float64
		for j := 0; j <= n; j++ {
			x := startX + float64(j)*subH
			sum += weights[j] * f(x)
		}
		total += coeff * subH * sum
	}

	return total, nil
}

func (nc *NewtonCotes) PrintBernoulliNumbers(max int) {
	fmt.Println("伯努利数:")
	for i := 0; i <= max; i++ {
		fmt.Printf("B%d = %.10f\n", i, bernoulliNumber(i))
	}
}

func (nc *NewtonCotes) VerifyCoefficients(n int) {
	weights, coeff := nc.computeCotesCoefficients(n)
	fmt.Printf("\n牛顿-柯特斯系数 (n=%d):\n", n)
	fmt.Printf("数值积分权重: ")
	for _, w := range weights {
		fmt.Printf("%.6f ", w)
	}
	fmt.Println()
	
	standardWeights := nc.getWeights(n)
	standardCoeff := nc.getCoefficient(n)
	fmt.Printf("标准权重:     ")
	for _, w := range standardWeights {
		fmt.Printf("%.6f ", w)
	}
	fmt.Println()
	fmt.Printf("系数: coeff=%.6f, standardCoeff=%.6f\n", coeff, standardCoeff)
	
	fmt.Print("归一化标准权重 * coeff: ")
	sumStandard := 0.0
	for _, w := range standardWeights {
		sumStandard += w
	}
	for _, w := range standardWeights {
		fmt.Printf("%.6f ", w*standardCoeff)
	}
	fmt.Println()
}

func (r *Romberg) trapezoidal(f func(float64) float64, a, b float64, n int) float64 {
	h := (b - a) / float64(n)
	sum := 0.5 * (f(a) + f(b))
	for i := 1; i < n; i++ {
		sum += f(a + float64(i)*h)
	}
	return sum * h
}

func (r *Romberg) richardson(R [][]float64, i, j int) float64 {
	factor := math.Pow(4, float64(j))
	return (factor*R[i][j-1] - R[i-1][j-1]) / (factor - 1)
}

func (r *Romberg) Integrate(f func(float64) float64, a, b float64) (float64, [][]float64, error) {
	if r.Tolerance <= 0 {
		r.Tolerance = 1e-10
	}
	if r.MaxLevels <= 0 {
		r.MaxLevels = 10
	}
	
	R := make([][]float64, r.MaxLevels)
	for i := 0; i < r.MaxLevels; i++ {
		R[i] = make([]float64, i+1)
		n := int(math.Pow(2, float64(i)))
		R[i][0] = r.trapezoidal(f, a, b, n)
		
		for j := 1; j <= i; j++ {
			R[i][j] = r.richardson(R, i, j)
		}
		
		if i > 0 {
			err := math.Abs(R[i][i] - R[i-1][i-1])
			if err < r.Tolerance {
				return R[i][i], R[:i+1], nil
			}
		}
	}
	
	return R[r.MaxLevels-1][r.MaxLevels-1], R, nil
}

func (r *Romberg) IntegrateExpr(expr string, a, b float64) (float64, [][]float64, error) {
	nc := &NewtonCotes{}
	f, err := nc.parseFunction(expr)
	if err != nil {
		return 0, nil, err
	}
	return r.Integrate(f, a, b)
}

func PrintRombergTable(R [][]float64) {
	fmt.Println("\n龙贝格积分表 (Richardson Extrapolation):")
	for i := 0; i < len(R); i++ {
		fmt.Printf("R[%d]: ", i)
		for j := 0; j <= i; j++ {
			fmt.Printf("%.10f ", R[i][j])
		}
		fmt.Println()
	}
}
