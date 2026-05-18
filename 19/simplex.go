package main

import (
	"fmt"
	"math"
)

const (
	epsilon        = 1e-9
	maxIterations  = 10000
)

type SimplexTableau struct {
	tableau   [][]float64
	basis     []int
	numRows   int
	numCols   int
	numVars   int
	numSlacks int
}

func NewSimplexTableau(A [][]float64, b, c []float64) *SimplexTableau {
	m := len(A)
	n := len(c)
	numCols := n + m + 1
	
	tableau := make([][]float64, m+1)
	for i := range tableau {
		tableau[i] = make([]float64, numCols)
	}
	
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			tableau[i][j] = A[i][j]
		}
		tableau[i][n+i] = 1.0
		tableau[i][numCols-1] = b[i]
	}
	
	for j := 0; j < n; j++ {
		tableau[m][j] = -c[j]
	}
	
	basis := make([]int, m)
	for i := 0; i < m; i++ {
		basis[i] = n + i
	}
	
	return &SimplexTableau{
		tableau:   tableau,
		basis:     basis,
		numRows:   m + 1,
		numCols:   numCols,
		numVars:   n,
		numSlacks: m,
	}
}

func (st *SimplexTableau) pivot(pivotRow, pivotCol int) {
	pivotVal := st.tableau[pivotRow][pivotCol]
	for j := 0; j < st.numCols; j++ {
		st.tableau[pivotRow][j] /= pivotVal
	}
	
	for i := 0; i < st.numRows; i++ {
		if i != pivotRow {
			factor := st.tableau[i][pivotCol]
			for j := 0; j < st.numCols; j++ {
				st.tableau[i][j] -= factor * st.tableau[pivotRow][j]
			}
		}
	}
	
	st.basis[pivotRow] = pivotCol
}

func (st *SimplexTableau) findEnteringVariable() int {
	lastRow := st.numRows - 1
	for j := 0; j < st.numCols-1; j++ {
		if st.tableau[lastRow][j] < -epsilon {
			return j
		}
	}
	return -1
}

func (st *SimplexTableau) findLeavingVariable(enteringCol int) int {
	leavingRow := -1
	minRatio := math.Inf(1)
	minBasisIdx := math.MaxInt32
	
	for i := 0; i < st.numRows-1; i++ {
		if st.tableau[i][enteringCol] > epsilon {
			ratio := st.tableau[i][st.numCols-1] / st.tableau[i][enteringCol]
			if ratio < minRatio-epsilon {
				minRatio = ratio
				leavingRow = i
				minBasisIdx = st.basis[i]
			} else if math.Abs(ratio-minRatio) <= epsilon {
				if st.basis[i] < minBasisIdx {
					minBasisIdx = st.basis[i]
					leavingRow = i
				}
			}
		}
	}
	return leavingRow
}

func (st *SimplexTableau) optimize() bool {
	for iter := 0; iter < maxIterations; iter++ {
		enteringCol := st.findEnteringVariable()
		if enteringCol == -1 {
			return true
		}
		
		leavingRow := st.findLeavingVariable(enteringCol)
		if leavingRow == -1 {
			return false
		}
		
		st.pivot(leavingRow, enteringCol)
	}
	return true
}

func (st *SimplexTableau) getSolution() []float64 {
	solution := make([]float64, st.numVars)
	for i := 0; i < st.numRows-1; i++ {
		if st.basis[i] < st.numVars {
			solution[st.basis[i]] = st.tableau[i][st.numCols-1]
		}
	}
	return solution
}

func (st *SimplexTableau) getObjectiveValue() float64 {
	return st.tableau[st.numRows-1][st.numCols-1]
}

func (st *SimplexTableau) isDualFeasible() bool {
	lastRow := st.numRows - 1
	for j := 0; j < st.numCols-1; j++ {
		if st.tableau[lastRow][j] < -epsilon {
			return false
		}
	}
	return true
}

func (st *SimplexTableau) findLeavingVariableDual() int {
	leavingRow := -1
	minBasisIdx := math.MaxInt32
	
	for i := 0; i < st.numRows-1; i++ {
		if st.tableau[i][st.numCols-1] < -epsilon {
			if st.basis[i] < minBasisIdx {
				minBasisIdx = st.basis[i]
				leavingRow = i
			}
		}
	}
	return leavingRow
}

func (st *SimplexTableau) findEnteringVariableDual(leavingRow int) int {
	enteringCol := -1
	minRatio := math.Inf(1)
	lastRow := st.numRows - 1
	
	for j := 0; j < st.numCols-1; j++ {
		if st.tableau[leavingRow][j] < -epsilon {
			ratio := math.Abs(st.tableau[lastRow][j] / st.tableau[leavingRow][j])
			if ratio < minRatio-epsilon {
				minRatio = ratio
				enteringCol = j
			} else if math.Abs(ratio-minRatio) <= epsilon {
				if j < enteringCol {
					enteringCol = j
				}
			}
		}
	}
	return enteringCol
}

func (st *SimplexTableau) dualOptimize() bool {
	for iter := 0; iter < maxIterations; iter++ {
		leavingRow := st.findLeavingVariableDual()
		if leavingRow == -1 {
			return true
		}
		
		enteringCol := st.findEnteringVariableDual(leavingRow)
		if enteringCol == -1 {
			return false
		}
		
		st.pivot(leavingRow, enteringCol)
	}
	return true
}

func SolveDualLP(A [][]float64, b, c []float64) (solution []float64, objVal float64, status string) {
	m := len(A)
	n := len(c)
	
	tableau := NewSimplexTableau(A, b, c)
	
	lastRow := tableau.numRows - 1
	for i := 0; i < m; i++ {
		col := tableau.basis[i]
		factor := tableau.tableau[lastRow][col]
		for j := 0; j < tableau.numCols; j++ {
			tableau.tableau[lastRow][j] -= factor * tableau.tableau[i][j]
		}
	}
	
	unbounded := tableau.dualOptimize()
	if !unbounded {
		return nil, math.Inf(1), "对偶无界（原问题无可行解）"
	}
	
	return tableau.getSolution(), tableau.getObjectiveValue(), "最优解"
}

func SolveDualFromCurrent(tableau *SimplexTableau, c []float64) (solution []float64, objVal float64, status string) {
	m := tableau.numRows - 1
	lastRow := tableau.numRows - 1
	
	for j := 0; j < len(c); j++ {
		tableau.tableau[lastRow][j] = -c[j]
	}
	for j := len(c); j < tableau.numCols-1; j++ {
		tableau.tableau[lastRow][j] = 0
	}
	tableau.tableau[lastRow][tableau.numCols-1] = 0
	
	for i := 0; i < m; i++ {
		col := tableau.basis[i]
		factor := tableau.tableau[lastRow][col]
		for j := 0; j < tableau.numCols; j++ {
			tableau.tableau[lastRow][j] -= factor * tableau.tableau[i][j]
		}
	}
	
	unbounded := tableau.dualOptimize()
	if !unbounded {
		return nil, math.Inf(1), "对偶无界（原问题无可行解）"
	}
	
	return tableau.getSolution(), tableau.getObjectiveValue(), "最优解"
}

func AddCutAndReoptimize(tableau *SimplexTableau, cutCoeffs []float64, cutRHS float64) (solution []float64, objVal float64, status string) {
	m := tableau.numRows - 1
	n := tableau.numVars
	
	newRow := make([]float64, tableau.numCols)
	for j := 0; j < n; j++ {
		newRow[j] = cutCoeffs[j]
	}
	newRow[tableau.numCols-2] = 1
	newRow[tableau.numCols-1] = cutRHS
	
	for i := 0; i < m; i++ {
		rowFactor := newRow[tableau.basis[i]]
		if math.Abs(rowFactor) > epsilon {
			for j := 0; j < tableau.numCols; j++ {
				newRow[j] -= rowFactor * tableau.tableau[i][j]
			}
		}
	}
	
	newTableau := make([][]float64, tableau.numRows+1)
	for i := 0; i < tableau.numRows; i++ {
		newTableau[i] = make([]float64, tableau.numCols+1)
		for j := 0; j < tableau.numCols-1; j++ {
			newTableau[i][j] = tableau.tableau[i][j]
		}
		newTableau[i][tableau.numCols-1] = 0
		newTableau[i][tableau.numCols] = tableau.tableau[i][tableau.numCols-1]
	}
	newTableau[tableau.numRows-1][tableau.numCols-1] = 0
	
	newTableau[tableau.numRows-1] = make([]float64, tableau.numCols+1)
	for j := 0; j < tableau.numCols-1; j++ {
		newTableau[tableau.numRows-1][j] = newRow[j]
	}
	newTableau[tableau.numRows-1][tableau.numCols-1] = 1
	newTableau[tableau.numRows-1][tableau.numCols] = newRow[tableau.numCols-1]
	
	newTableau[tableau.numRows] = tableau.tableau[tableau.numRows-1]
	
	newBasis := make([]int, len(tableau.basis)+1)
	copy(newBasis, tableau.basis)
	newBasis[len(tableau.basis)] = tableau.numCols - 1
	
	newST := &SimplexTableau{
		tableau:   newTableau,
		basis:     newBasis,
		numRows:   tableau.numRows + 1,
		numCols:   tableau.numCols + 1,
		numVars:   tableau.numVars,
		numSlacks: tableau.numSlacks + 1,
	}
	
	unbounded := newST.dualOptimize()
	if !unbounded {
		return nil, math.Inf(1), "无可行解"
	}
	
	return newST.getSolution(), newST.getObjectiveValue(), "最优解"
}

func PhaseOne(A [][]float64, b []float64) (*SimplexTableau, bool) {
	m := len(A)
	n := len(A[0])
	
	artificialN := n + m
	artificialA := make([][]float64, m)
	artificialB := make([]float64, m)
	
	for i := range artificialA {
		artificialA[i] = make([]float64, artificialN)
		for j := 0; j < n; j++ {
			artificialA[i][j] = A[i][j]
		}
		artificialA[i][n+i] = 1.0
		artificialB[i] = b[i]
		
		if b[i] < -epsilon {
			for j := 0; j < artificialN; j++ {
				artificialA[i][j] *= -1
			}
			artificialB[i] *= -1
		}
	}
	
	artificialC := make([]float64, artificialN)
	for i := n; i < artificialN; i++ {
		artificialC[i] = -1.0
	}
	
	tableau := NewSimplexTableau(artificialA, artificialB, artificialC)
	
	lastRow := tableau.numRows - 1
	for i := 0; i < m; i++ {
		for j := 0; j < tableau.numCols; j++ {
			tableau.tableau[lastRow][j] += tableau.tableau[i][j]
		}
	}
	
	tableau.optimize()
	
	if tableau.getObjectiveValue() < -epsilon {
		return nil, false
	}
	
	originalTableau := NewSimplexTableau(A, b, make([]float64, n))
	
	for i := 0; i < m; i++ {
		if tableau.basis[i] < n {
			originalTableau.basis[i] = tableau.basis[i]
			for j := 0; j < originalTableau.numCols; j++ {
				originalTableau.tableau[i][j] = tableau.tableau[i][j]
			}
		} else if tableau.basis[i] < artificialN {
			for j := 0; j < n; j++ {
				if math.Abs(tableau.tableau[i][j]) > epsilon {
					tableau.pivot(i, j)
					originalTableau.basis[i] = j
					for k := 0; k < originalTableau.numCols; k++ {
						originalTableau.tableau[i][k] = tableau.tableau[i][k]
					}
					break
				}
			}
		}
	}
	
	return originalTableau, true
}

func SolveLP(A [][]float64, b, c []float64) (solution []float64, objVal float64, status string) {
	tableau, feasible := PhaseOne(A, b)
	if !feasible {
		return nil, 0, "无可行解"
	}
	
	for j := 0; j < len(c); j++ {
		tableau.tableau[tableau.numRows-1][j] = -c[j]
	}
	
	lastRow := tableau.numRows - 1
	for i := 0; i < tableau.numRows-1; i++ {
		col := tableau.basis[i]
		factor := tableau.tableau[lastRow][col]
		for j := 0; j < tableau.numCols; j++ {
			tableau.tableau[lastRow][j] -= factor * tableau.tableau[i][j]
		}
	}
	
	unbounded := tableau.optimize()
	if !unbounded {
		return tableau.getSolution(), math.Inf(1), "无界解"
	}
	
	return tableau.getSolution(), tableau.getObjectiveValue(), "最优解"
}

func main() {
	fmt.Println("=== 测试用例1: 标准线性规划问题 ===")
	testCase1()
	fmt.Println()
	
	fmt.Println("=== 测试用例2: 无可行解问题 ===")
	testCase2()
	fmt.Println()
	
	fmt.Println("=== 测试用例3: 无界解问题 ===")
	testCase3()
	fmt.Println()
	
	fmt.Println("=== 测试用例4: 三变量线性规划 ===")
	testCase4()
	fmt.Println()
	
	fmt.Println("=== 测试用例5: Beale退化循环例子 ===")
	testCase5()
	fmt.Println()
	
	fmt.Println("=== 测试用例6: 摄动法验证 ===")
	testCase6()
	fmt.Println()
	
	fmt.Println("=== 测试用例7: 对偶单纯形法 - >=约束 ===")
	testCase7()
	fmt.Println()
	
	fmt.Println("=== 测试用例8: 对偶单纯形法 - 整数规划松弛 ===")
	testCase8()
	fmt.Println()
	
	fmt.Println("=== 测试用例9: 对偶单纯形法 - 对偶可行 ===")
	testCase9()
}

func testCase1() {
	A := [][]float64{
		{1, 2},
		{4, 0},
		{0, 4},
	}
	b := []float64{8, 16, 12}
	c := []float64{2, 3}
	
	fmt.Println("max z = 2x1 + 3x2")
	fmt.Println("约束:")
	fmt.Println("x1 + 2x2 <= 8")
	fmt.Println("4x1 <= 16")
	fmt.Println("4x2 <= 12")
	fmt.Println("x1, x2 >= 0")
	
	solution, objVal, status := SolveLP(A, b, c)
	
	fmt.Printf("状态: %s\n", status)
	if status == "最优解" {
		fmt.Printf("最优解: x1 = %.4f, x2 = %.4f\n", solution[0], solution[1])
		fmt.Printf("最优目标值: %.4f\n", objVal)
		fmt.Println("期望: x1=4, x2=2, z=14")
	}
}

func testCase2() {
	A := [][]float64{
		{1, 1},
		{1, -1},
	}
	b := []float64{1, -3}
	c := []float64{2, 1}
	
	fmt.Println("max z = 2x1 + x2")
	fmt.Println("约束:")
	fmt.Println("x1 + x2 <= 1")
	fmt.Println("x1 - x2 <= -3")
	fmt.Println("x1, x2 >= 0")
	
	solution, objVal, status := SolveLP(A, b, c)
	
	fmt.Printf("状态: %s\n", status)
	fmt.Println("期望: 无可行解")
	_, _ = solution, objVal
}

func testCase3() {
	A := [][]float64{
		{-1, 1},
		{2, -1},
	}
	b := []float64{2, 4}
	c := []float64{1, 1}
	
	fmt.Println("max z = x1 + x2")
	fmt.Println("约束:")
	fmt.Println("-x1 + x2 <= 2")
	fmt.Println("2x1 - x2 <= 4")
	fmt.Println("x1, x2 >= 0")
	
	solution, objVal, status := SolveLP(A, b, c)
	
	fmt.Printf("状态: %s\n", status)
	if status == "无界解" {
		fmt.Println("期望: 无界解")
	}
	_, _ = solution, objVal
}

func testCase4() {
	A := [][]float64{
		{1, 1, 1},
		{2, 1, 0},
		{0, 1, 3},
	}
	b := []float64{100, 80, 90}
	c := []float64{3, 2, 5}
	
	fmt.Println("max z = 3x1 + 2x2 + 5x3")
	fmt.Println("约束:")
	fmt.Println("x1 + x2 + x3 <= 100")
	fmt.Println("2x1 + x2 <= 80")
	fmt.Println("x2 + 3x3 <= 90")
	fmt.Println("x1, x2, x3 >= 0")
	
	solution, objVal, status := SolveLP(A, b, c)
	
	fmt.Printf("状态: %s\n", status)
	if status == "最优解" {
		fmt.Printf("最优解: x1 = %.4f, x2 = %.4f, x3 = %.4f\n", 
			solution[0], solution[1], solution[2])
		fmt.Printf("最优目标值: %.4f\n", objVal)
	}
}

func testCase5() {
	A := [][]float64{
		{1, -2, -2, 1, 0},
		{-4, 1, -1, 0, 1},
		{-2, 0, 1, 0, 0},
	}
	b := []float64{0, 0, 1}
	c := []float64{1, -3, -1, 0, 0}
	
	fmt.Println("Beale退化例子: 不使用Bland法则会循环")
	fmt.Println("max z = x1 - 3x2 - x3")
	fmt.Println("约束:")
	fmt.Println("x1 - 2x2 - 2x3 + x4 = 0")
	fmt.Println("-4x1 + x2 - x3 + x5 = 0")
	fmt.Println("-2x1 + x3 = 1")
	fmt.Println("x1, x2, x3, x4, x5 >= 0")
	
	solution, objVal, status := SolveLP(A, b, c)
	
	fmt.Printf("状态: %s\n", status)
	if status == "最优解" {
		fmt.Printf("最优目标值: %.4f\n", objVal)
		fmt.Println("期望最优值: -2 (使用Bland法则不会循环)")
	}
}

func testCase6() {
	A := [][]float64{
		{1, 1, 1, 0},
		{-1, 1, 0, 1},
	}
	b := []float64{1, 0}
	c := []float64{-1, -1, 0, 0}
	
	fmt.Println("退化例子: b有零元素")
	fmt.Println("min z = x1 + x2")
	fmt.Println("约束:")
	fmt.Println("x1 + x2 + x3 = 1")
	fmt.Println("-x1 + x2 + x4 = 0")
	fmt.Println("x1, x2, x3, x4 >= 0")
	
	solution, objVal, status := SolveLP(A, b, c)
	
	fmt.Printf("状态: %s\n", status)
	if status == "最优解" {
		fmt.Printf("最优解: x1 = %.4f, x2 = %.4f\n", solution[0], solution[1])
		fmt.Printf("最优目标值: %.4f\n", objVal)
		fmt.Println("期望: z=0.5 (x1=0.5, x2=0.5)")
	}
}

func testCase7() {
	A := [][]float64{
		{-1, -2, 1, 0},
		{-4, -1, 0, 1},
	}
	b := []float64{-4, -3}
	c := []float64{-2, -3, 0, 0}
	
	fmt.Println("对偶单纯形法: 右端项有负数（>=约束）")
	fmt.Println("min z = 2x1 + 3x2")
	fmt.Println("约束:")
	fmt.Println("x1 + 2x2 >= 4")
	fmt.Println("4x1 + x2 >= 3")
	fmt.Println("x1, x2 >= 0")
	
	solution, objVal, status := SolveDualLP(A, b, c)
	
	fmt.Printf("状态: %s\n", status)
	if status == "最优解" {
		fmt.Printf("最优解: x1 = %.4f, x2 = %.4f\n", solution[0], solution[1])
		fmt.Printf("最优目标值: %.4f\n", objVal)
		fmt.Println("期望: x1=0.4, x2=1.8, z=6.2")
	}
}

func testCase8() {
	A := [][]float64{
		{1, 1, 1, 0},
		{2, 1, 0, 1},
	}
	b := []float64{5, 8}
	c := []float64{-3, -2, 0, 0}
	
	fmt.Println("整数规划松弛: 分支定界场景")
	fmt.Println("min z = 3x1 + 2x2")
	fmt.Println("约束:")
	fmt.Println("x1 + x2 <= 5")
	fmt.Println("2x1 + x2 <= 8")
	fmt.Println("x1, x2 >= 0")
	
	tableau := NewSimplexTableau(A, b, c)
	solution, objVal, status := SolveDualFromCurrent(tableau, c)
	
	fmt.Printf("状态: %s\n", status)
	if status == "最优解" {
		fmt.Printf("松弛最优解: x1 = %.4f, x2 = %.4f\n", solution[0], solution[1])
		fmt.Printf("松弛最优目标值: %.4f\n", objVal)
		fmt.Println("期望: x1=3, x2=2, z=13")
		
		if math.Abs(solution[0]-3.0) < epsilon {
			fmt.Println("\n模拟分支定界: x1=3 是整数，继续求解...")
		}
	}
}

func testCase9() {
	A := [][]float64{
		{1, 0, 1, 0},
		{0, 1, 0, 1},
	}
	b := []float64{6, 6}
	c := []float64{-1, -1, 0, 0}
	
	fmt.Println("对偶可行问题: 检验数全部非负")
	fmt.Println("min z = x1 + x2")
	fmt.Println("约束:")
	fmt.Println("x1 <= 6")
	fmt.Println("x2 <= 6")
	fmt.Println("x1, x2 >= 0")
	
	solution, objVal, status := SolveDualLP(A, b, c)
	
	fmt.Printf("状态: %s\n", status)
	if status == "最优解" {
		fmt.Printf("最优解: x1 = %.4f, x2 = %.4f\n", solution[0], solution[1])
		fmt.Printf("最优目标值: %.4f\n", objVal)
		fmt.Println("期望: x1=0, x2=0, z=0")
	}
}
