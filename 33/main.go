package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"strconv"
)

type Dual struct {
	Val float64
	D   float64
}

func dualConst(c float64) Dual {
	return Dual{Val: c, D: 0}
}

func dualVar(x float64) Dual {
	return Dual{Val: x, D: 1}
}

func (d Dual) Add(other Dual) Dual {
	return Dual{
		Val: d.Val + other.Val,
		D:   d.D + other.D,
	}
}

func (d Dual) Sub(other Dual) Dual {
	return Dual{
		Val: d.Val - other.Val,
		D:   d.D - other.D,
	}
}

func (d Dual) Mul(other Dual) Dual {
	return Dual{
		Val: d.Val * other.Val,
		D:   d.D*other.Val + d.Val*other.D,
	}
}

func (d Dual) Div(other Dual) Dual {
	return Dual{
		Val: d.Val / other.Val,
		D:   (d.D*other.Val - d.Val*other.D) / (other.Val * other.Val),
	}
}

func (d Dual) Neg() Dual {
	return Dual{
		Val: -d.Val,
		D:   -d.D,
	}
}

func dualSin(d Dual) Dual {
	return Dual{
		Val: math.Sin(d.Val),
		D:   math.Cos(d.Val) * d.D,
	}
}

func dualCos(d Dual) Dual {
	return Dual{
		Val: math.Cos(d.Val),
		D:   -math.Sin(d.Val) * d.D,
	}
}

func dualTan(d Dual) Dual {
	cosVal := math.Cos(d.Val)
	return Dual{
		Val: math.Tan(d.Val),
		D:   d.D / (cosVal * cosVal),
	}
}

func dualExp(d Dual) Dual {
	expVal := math.Exp(d.Val)
	return Dual{
		Val: expVal,
		D:   expVal * d.D,
	}
}

func dualLog(d Dual) Dual {
	return Dual{
		Val: math.Log(d.Val),
		D:   d.D / d.Val,
	}
}

func dualSqrt(d Dual) Dual {
	sqrtVal := math.Sqrt(d.Val)
	return Dual{
		Val: sqrtVal,
		D:   d.D / (2 * sqrtVal),
	}
}

func dualAbs(d Dual) Dual {
	absVal := math.Abs(d.Val)
	var deriv float64
	if d.Val > 0 {
		deriv = 1
	} else if d.Val < 0 {
		deriv = -1
	} else {
		deriv = math.NaN()
	}
	return Dual{
		Val: absVal,
		D:   deriv * d.D,
	}
}

func centralDifference(f func(float64) float64, x0, h float64) float64 {
	return (f(x0+h) - f(x0-h)) / (2 * h)
}

func checkContinuity(f func(float64) float64, x0, h float64) (bool, string) {
	fLeft := f(x0 - h)
	fRight := f(x0 + h)
	fMid := f(x0)

	if math.IsNaN(fLeft) || math.IsNaN(fRight) || math.IsNaN(fMid) {
		return false, "函数在该点或附近无定义"
	}

	jumpAtMid := math.Abs(fRight - fLeft)
	avgMag := (math.Abs(fLeft) + math.Abs(fRight)) / 2

	if avgMag > 1e-10 {
		relativeJump := jumpAtMid / avgMag
		if relativeJump > 0.1 {
			return false, fmt.Sprintf("检测到函数不连续（相对跳跃量: %.4f）", relativeJump)
		}
	} else if jumpAtMid > 1e-6 {
		return false, fmt.Sprintf("检测到函数不连续（绝对跳跃量: %.4e）", jumpAtMid)
	}

	leftDeriv := (fMid - f(x0 - h/2)) / (h / 2)
	rightDeriv := (f(x0 + h/2) - fMid) / (h / 2)
	sideDerivDiff := math.Abs(leftDeriv - rightDeriv)
	avgSideDeriv := (math.Abs(leftDeriv) + math.Abs(rightDeriv)) / 2

	if avgSideDeriv > 1e-10 {
		relativeSideDiff := sideDerivDiff / avgSideDeriv
		if relativeSideDiff > 0.01 {
			return false, fmt.Sprintf("左右导数不相等（相对差异: %.4f）", relativeSideDiff)
		}
	} else if sideDerivDiff > 1e-6 {
		return false, fmt.Sprintf("左右导数不相等（绝对差异: %.4e）", sideDerivDiff)
	}

	h2 := h / 2
	d1 := centralDifference(f, x0, h)
	d2 := centralDifference(f, x0, h2)
	derivDiff := math.Abs(d1 - d2)
	avgDeriv := (math.Abs(d1) + math.Abs(d2)) / 2

	if avgDeriv > 1e-10 {
		relativeDerivDiff := derivDiff / avgDeriv
		if relativeDerivDiff > 0.5 {
			return false, fmt.Sprintf("导数不收敛（步长减半后相对变化: %.4f）", relativeDerivDiff)
		}
	} else if derivDiff > 1e-6 {
		return false, fmt.Sprintf("导数不收敛（步长减半后绝对变化: %.4e）", derivDiff)
	}

	return true, ""
}

func parseAndEval(expr string, x float64) (float64, error) {
	exprStr := "package main; func _() float64 { return " + expr + " }"
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", exprStr, 0)
	if err != nil {
		return 0, fmt.Errorf("解析表达式失败: %v", err)
	}

	if len(f.Decls) == 0 {
		return 0, fmt.Errorf("没有找到函数声明")
	}

	funcDecl, ok := f.Decls[0].(*ast.FuncDecl)
	if !ok || len(funcDecl.Body.List) == 0 {
		return 0, fmt.Errorf("函数体为空")
	}

	returnStmt, ok := funcDecl.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(returnStmt.Results) == 0 {
		return 0, fmt.Errorf("没有返回语句")
	}

	return evalExpr(returnStmt.Results[0], x)
}

func evalExpr(expr ast.Expr, x float64) (float64, error) {
	switch e := expr.(type) {
	case *ast.BasicLit:
		if e.Kind == token.FLOAT || e.Kind == token.INT {
			val, err := strconv.ParseFloat(e.Value, 64)
			if err != nil {
				return 0, err
			}
			return val, nil
		}
	case *ast.Ident:
		if e.Name == "x" {
			return x, nil
		}
		if e.Name == "pi" || e.Name == "Pi" || e.Name == "PI" {
			return math.Pi, nil
		}
		if e.Name == "e" {
			return math.E, nil
		}
	case *ast.BinaryExpr:
		left, err := evalExpr(e.X, x)
		if err != nil {
			return 0, err
		}
		right, err := evalExpr(e.Y, x)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.ADD:
			return left + right, nil
		case token.SUB:
			return left - right, nil
		case token.MUL:
			return left * right, nil
		case token.QUO:
			if right == 0 {
				return 0, fmt.Errorf("除零错误")
			}
			return left / right, nil
		}
	case *ast.ParenExpr:
		return evalExpr(e.X, x)
	case *ast.UnaryExpr:
		val, err := evalExpr(e.X, x)
		if err != nil {
			return 0, err
		}
		if e.Op == token.SUB {
			return -val, nil
		}
		return val, nil
	case *ast.CallExpr:
		if ident, ok := e.Fun.(*ast.Ident); ok {
			if len(e.Args) != 1 {
				return 0, fmt.Errorf("函数 %s 需要1个参数", ident.Name)
			}
			arg, err := evalExpr(e.Args[0], x)
			if err != nil {
				return 0, err
			}
			switch ident.Name {
			case "sin", "Sin":
				return math.Sin(arg), nil
			case "cos", "Cos":
				return math.Cos(arg), nil
			case "tan", "Tan":
				return math.Tan(arg), nil
			case "exp", "Exp":
				return math.Exp(arg), nil
			case "log", "Log":
				return math.Log(arg), nil
			case "sqrt", "Sqrt":
				return math.Sqrt(arg), nil
			case "abs", "Abs":
				return math.Abs(arg), nil
			default:
				return 0, fmt.Errorf("不支持的函数: %s", ident.Name)
			}
		}
	}
	return 0, fmt.Errorf("不支持的表达式类型")
}

func parseAndEvalDual(expr string, x float64) (Dual, error) {
	exprStr := "package main; func _() float64 { return " + expr + " }"
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", exprStr, 0)
	if err != nil {
		return Dual{}, fmt.Errorf("解析表达式失败: %v", err)
	}

	if len(f.Decls) == 0 {
		return Dual{}, fmt.Errorf("没有找到函数声明")
	}

	funcDecl, ok := f.Decls[0].(*ast.FuncDecl)
	if !ok || len(funcDecl.Body.List) == 0 {
		return Dual{}, fmt.Errorf("函数体为空")
	}

	returnStmt, ok := funcDecl.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(returnStmt.Results) == 0 {
		return Dual{}, fmt.Errorf("没有返回语句")
	}

	return evalExprDual(returnStmt.Results[0], x)
}

func evalExprDual(expr ast.Expr, x float64) (Dual, error) {
	switch e := expr.(type) {
	case *ast.BasicLit:
		if e.Kind == token.FLOAT || e.Kind == token.INT {
			val, err := strconv.ParseFloat(e.Value, 64)
			if err != nil {
				return Dual{}, err
			}
			return dualConst(val), nil
		}
	case *ast.Ident:
		if e.Name == "x" {
			return dualVar(x), nil
		}
		if e.Name == "pi" || e.Name == "Pi" || e.Name == "PI" {
			return dualConst(math.Pi), nil
		}
		if e.Name == "e" {
			return dualConst(math.E), nil
		}
	case *ast.BinaryExpr:
		left, err := evalExprDual(e.X, x)
		if err != nil {
			return Dual{}, err
		}
		right, err := evalExprDual(e.Y, x)
		if err != nil {
			return Dual{}, err
		}
		switch e.Op {
		case token.ADD:
			return left.Add(right), nil
		case token.SUB:
			return left.Sub(right), nil
		case token.MUL:
			return left.Mul(right), nil
		case token.QUO:
			if right.Val == 0 {
				return Dual{}, fmt.Errorf("除零错误")
			}
			return left.Div(right), nil
		}
	case *ast.ParenExpr:
		return evalExprDual(e.X, x)
	case *ast.UnaryExpr:
		val, err := evalExprDual(e.X, x)
		if err != nil {
			return Dual{}, err
		}
		if e.Op == token.SUB {
			return val.Neg(), nil
		}
		return val, nil
	case *ast.CallExpr:
		if ident, ok := e.Fun.(*ast.Ident); ok {
			if len(e.Args) != 1 {
				return Dual{}, fmt.Errorf("函数 %s 需要1个参数", ident.Name)
			}
			arg, err := evalExprDual(e.Args[0], x)
			if err != nil {
				return Dual{}, err
			}
			switch ident.Name {
			case "sin", "Sin":
				return dualSin(arg), nil
			case "cos", "Cos":
				return dualCos(arg), nil
			case "tan", "Tan":
				return dualTan(arg), nil
			case "exp", "Exp":
				return dualExp(arg), nil
			case "log", "Log":
				return dualLog(arg), nil
			case "sqrt", "Sqrt":
				return dualSqrt(arg), nil
			case "abs", "Abs":
				return dualAbs(arg), nil
			default:
				return Dual{}, fmt.Errorf("不支持的函数: %s", ident.Name)
			}
		}
	}
	return Dual{}, fmt.Errorf("不支持的表达式类型")
}

func main() {
	expr := flag.String("f", "x*x", "函数表达式，例如: -f=\"x*x + 2*x\"")
	x0 := flag.Float64("x", 0, "求导点x0")
	h := flag.Float64("h", 0.001, "步长h（仅用于数值微分）")
	method := flag.String("method", "auto", "微分方法: auto-自动选择, num-数值微分, ad-自动微分")
	flag.Parse()

	f := func(x float64) float64 {
		val, err := parseAndEval(*expr, x)
		if err != nil {
			fmt.Printf("表达式求值错误: %v\n", err)
			return math.NaN()
		}
		return val
	}

	fmt.Printf("函数表达式: f(x) = %s\n", *expr)
	fmt.Printf("求导点 x0 = %.6f\n", *x0)

	var derivative float64
	var methodUsed string

	useAD := false
	if *method == "ad" {
		useAD = true
	} else if *method == "num" {
		useAD = false
	} else {
		isContinuous, _ := checkContinuity(f, *x0, *h)
		useAD = isContinuous
	}

	if useAD {
		dualResult, err := parseAndEvalDual(*expr, *x0)
		if err != nil {
			fmt.Printf("自动微分错误: %v\n", err)
			return
		}
		derivative = dualResult.D
		methodUsed = "自动微分（前向模式）"
		if math.IsNaN(derivative) {
			fmt.Println("警告: 自动微分检测到该点不可导（如 abs 在 x=0 处）")
		}
	} else {
		fmt.Printf("步长 h = %.10f\n", *h)
		isContinuous, reason := checkContinuity(f, *x0, *h)
		if !isContinuous {
			fmt.Printf("警告: %s\n", reason)
			fmt.Println("该点可能是不连续点或不可导点，导数结果不可靠")
		}
		derivative = centralDifference(f, *x0, *h)
		methodUsed = "中心差分（数值微分）"
	}

	funcVal := f(*x0)
	fmt.Printf("函数值 f(%g) = %.10f\n", *x0, funcVal)
	fmt.Printf("使用方法: %s\n", methodUsed)
	fmt.Printf("一阶导数 f'(%g) = %.15f\n", *x0, derivative)
}
