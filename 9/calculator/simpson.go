package calculator

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

type Parser struct {
	expr string
	pos  int
}

func (p *Parser) skipWhitespace() {
	for p.pos < len(p.expr) && (p.expr[p.pos] == ' ' || p.expr[p.pos] == '\t') {
		p.pos++
	}
}

func (p *Parser) peek() byte {
	p.skipWhitespace()
	if p.pos >= len(p.expr) {
		return 0
	}
	return p.expr[p.pos]
}

func (p *Parser) consume() byte {
	ch := p.peek()
	if ch != 0 {
		p.pos++
	}
	return ch
}

func (p *Parser) parseNumber() (float64, error) {
	start := p.pos
	for p.pos < len(p.expr) && (isDigit(p.expr[p.pos]) || p.expr[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		return 0, fmt.Errorf("expected number")
	}
	num, err := strconv.ParseFloat(p.expr[start:p.pos], 64)
	if err != nil {
		return 0, err
	}
	return num, nil
}

func isDigit(ch byte) bool {
	return ch >= '0' && ch <= '9'
}

func (p *Parser) parseAtom(x float64) (float64, error) {
	ch := p.peek()
	if ch == 'x' {
		p.consume()
		return x, nil
	}
	if ch == '(' {
		p.consume()
		val, err := p.parseExpression(x)
		if err != nil {
			return 0, err
		}
		if p.consume() != ')' {
			return 0, fmt.Errorf("expected )")
		}
		return val, nil
	}
	if isDigit(ch) || ch == '.' {
		return p.parseNumber()
	}
	if ch == '-' {
		p.consume()
		val, err := p.parseAtom(x)
		if err != nil {
			return 0, err
		}
		return -val, nil
	}
	if ch == '+' {
		p.consume()
		return p.parseAtom(x)
	}
	if isLetter(ch) {
		return p.parseFunction(x)
	}
	return 0, fmt.Errorf("unexpected character: %c", ch)
}

func isLetter(ch byte) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
}

func (p *Parser) parseFunction(x float64) (float64, error) {
	start := p.pos
	for p.pos < len(p.expr) && isLetter(p.expr[p.pos]) {
		p.pos++
	}
	name := strings.ToLower(p.expr[start:p.pos])
	
	if p.consume() != '(' {
		return 0, fmt.Errorf("expected ( after function name")
	}
	
	arg, err := p.parseExpression(x)
	if err != nil {
		return 0, err
	}
	
	if p.consume() != ')' {
		return 0, fmt.Errorf("expected )")
	}
	
	switch name {
	case "sin":
		return math.Sin(arg), nil
	case "cos":
		return math.Cos(arg), nil
	case "tan":
		return math.Tan(arg), nil
	case "exp":
		return math.Exp(arg), nil
	case "log":
		return math.Log(arg), nil
	case "sqrt":
		return math.Sqrt(arg), nil
	case "abs":
		return math.Abs(arg), nil
	default:
		return 0, fmt.Errorf("unknown function: %s", name)
	}
}

func (p *Parser) parsePower(x float64) (float64, error) {
	base, err := p.parseAtom(x)
	if err != nil {
		return 0, err
	}
	if p.peek() == '^' {
		p.consume()
		exp, err := p.parsePower(x)
		if err != nil {
			return 0, err
		}
		return math.Pow(base, exp), nil
	}
	return base, nil
}

func (p *Parser) parseMultiplyDivide(x float64) (float64, error) {
	left, err := p.parsePower(x)
	if err != nil {
		return 0, err
	}
	for {
		op := p.peek()
		if op != '*' && op != '/' {
			break
		}
		p.consume()
		right, err := p.parsePower(x)
		if err != nil {
			return 0, err
		}
		if op == '*' {
			left *= right
		} else {
			left /= right
		}
	}
	return left, nil
}

func (p *Parser) parseAddSubtract(x float64) (float64, error) {
	left, err := p.parseMultiplyDivide(x)
	if err != nil {
		return 0, err
	}
	for {
		op := p.peek()
		if op != '+' && op != '-' {
			break
		}
		p.consume()
		right, err := p.parseMultiplyDivide(x)
		if err != nil {
			return 0, err
		}
		if op == '+' {
			left += right
		} else {
			left -= right
		}
	}
	return left, nil
}

func (p *Parser) parseExpression(x float64) (float64, error) {
	return p.parseAddSubtract(x)
}

func (p *Parser) Evaluate(x float64) (float64, error) {
	p.pos = 0
	val, err := p.parseExpression(x)
	if err != nil {
		return 0, err
	}
	if p.peek() != 0 {
		return 0, fmt.Errorf("unexpected token at end")
	}
	return val, nil
}

func NewParser(expr string) *Parser {
	return &Parser{expr: expr, pos: 0}
}

func CompositeSimpson(expr string, a, b float64, n int) (float64, error) {
	if n%2 != 0 {
		return 0, fmt.Errorf("n must be even for Simpson's rule")
	}
	if n <= 0 {
		return 0, fmt.Errorf("n must be positive")
	}

	parser := NewParser(expr)
	h := (b - a) / float64(n)

	sum, err := parser.Evaluate(a)
	if err != nil {
		return 0, fmt.Errorf("error evaluating function at x=%f: %v", a, err)
	}

	fb, err := parser.Evaluate(b)
	if err != nil {
		return 0, fmt.Errorf("error evaluating function at x=%f: %v", b, err)
	}
	sum += fb

	for i := 1; i < n; i++ {
		x := a + float64(i)*h
		fx, err := parser.Evaluate(x)
		if err != nil {
			return 0, fmt.Errorf("error evaluating function at x=%f: %v", x, err)
		}
		if i%2 == 1 {
			sum += 4 * fx
		} else {
			sum += 2 * fx
		}
	}

	result := sum * h / 3.0
	return result, nil
}

const (
	DefaultEpsilon              = 1e-8
	DefaultMaxRecursionDepth    = 1000
	MinIntervalWidth            = 1e-10
	CurvatureThresholdHigh      = 100.0
	CurvatureThresholdLow       = 1.0
	AdaptiveEpsilonFactorHigh   = 0.1
	AdaptiveEpsilonFactorLow    = 2.0
	GaussKronrodErrorThreshold  = 1e-6
	SwitchToKronrodThreshold    = 1e-4
)

type EvalPoint struct {
	x float64
	f float64
}

func simpsonRule(parser *Parser, a, b float64) (float64, error) {
	c := (a + b) / 2.0
	h := b - a
	fa, err := parser.Evaluate(a)
	if err != nil {
		return 0, err
	}
	fc, err := parser.Evaluate(c)
	if err != nil {
		return 0, err
	}
	fb, err := parser.Evaluate(b)
	if err != nil {
		return 0, err
	}
	return h * (fa + 4*fc + fb) / 6.0, nil
}

func simpsonRuleWithPoints(parser *Parser, a, b float64, fa, fb float64) (s float64, fc float64, err error) {
	c := (a + b) / 2.0
	h := b - a
	fc, err = parser.Evaluate(c)
	if err != nil {
		return 0, 0, err
	}
	s = h * (fa + 4*fc + fb) / 6.0
	return s, fc, nil
}

func estimateCurvature(parser *Parser, a, b, fa, fb float64) (float64, error) {
	c := (a + b) / 2.0
	d := (a + c) / 2.0
	e := (c + b) / 2.0
	
	fc, err := parser.Evaluate(c)
	if err != nil {
		return 0, err
	}
	fd, err := parser.Evaluate(d)
	if err != nil {
		return 0, err
	}
	fe, err := parser.Evaluate(e)
	if err != nil {
		return 0, err
	}
	
	h2 := (b - a) * (b - a) / 16.0
	
	secondDeriv1 := (fd - 2*fc + fe) / h2
	secondDeriv2 := (fa - 2*fc + fb) / (4 * h2)
	
	secondDeriv := (math.Abs(secondDeriv1) + math.Abs(secondDeriv2)) / 2.0
	
	return secondDeriv, nil
}

func adaptiveEpsilon(epsilon float64, curvature float64) float64 {
	if curvature > CurvatureThresholdHigh {
		return epsilon * AdaptiveEpsilonFactorHigh
	} else if curvature < CurvatureThresholdLow {
		return epsilon * AdaptiveEpsilonFactorLow
	}
	return epsilon
}

func shouldEarlyStop(s, sLeft, sRight, eps float64, curvature float64, depth int) bool {
	if depth >= DefaultMaxRecursionDepth {
		return true
	}
	
	errorEstimate := math.Abs(sLeft + sRight - s)
	if errorEstimate <= 15*eps {
		return true
	}
	
	if curvature > CurvatureThresholdHigh && depth > DefaultMaxRecursionDepth/2 {
		if errorEstimate <= 150*eps {
			return true
		}
	}
	
	return false
}

func adaptiveSimpsonRecursive(parser *Parser, a, b, eps, s float64, fa, fb float64, depth int) (float64, error) {
	c := (a + b) / 2.0
	h := b - a
	
	if h < MinIntervalWidth {
		return s, nil
	}
	
	sLeft, fc, err := simpsonRuleWithPoints(parser, a, c, fa, fb)
	if err != nil {
		return 0, err
	}
	sRight, _, err := simpsonRuleWithPoints(parser, c, b, fc, fb)
	if err != nil {
		return 0, err
	}
	s2 := sLeft + sRight
	
	curvature, err := estimateCurvature(parser, a, b, fa, fb)
	if err != nil {
		return 0, err
	}
	
	adaptiveEps := adaptiveEpsilon(eps, curvature)
	
	if shouldEarlyStop(s, sLeft, sRight, adaptiveEps, curvature, depth) {
		return s2 + (s2-s)/15.0, nil
	}
	
	leftResult, err := adaptiveSimpsonRecursive(parser, a, c, eps/2.0, sLeft, fa, fc, depth+1)
	if err != nil {
		return 0, err
	}
	rightResult, err := adaptiveSimpsonRecursive(parser, c, b, eps/2.0, sRight, fc, fb, depth+1)
	if err != nil {
		return 0, err
	}
	return leftResult + rightResult, nil
}

func AdaptiveSimpson(expr string, a, b float64, eps ...float64) (float64, error) {
	epsilon := DefaultEpsilon
	if len(eps) > 0 && eps[0] > 0 {
		epsilon = eps[0]
	}

	if math.Abs(b-a) < MinIntervalWidth {
		return 0, fmt.Errorf("interval width is too small")
	}

	parser := NewParser(expr)
	
	fa, err := parser.Evaluate(a)
	if err != nil {
		return 0, err
	}
	fb, err := parser.Evaluate(b)
	if err != nil {
		return 0, err
	}
	
	s, fc, err := simpsonRuleWithPoints(parser, a, b, fa, fb)
	if err != nil {
		return 0, err
	}

	result, err := adaptiveSimpsonRecursive(parser, a, b, epsilon, s, fa, fb, 0)
	if err != nil {
		return 0, err
	}
	return result, nil
}

var gaussKronrod15Nodes = [...]float64{
	-0.991455371120812639206854697526329,
	-0.949107912342758524526189684047851,
	-0.864864423359769072789712788640926,
	-0.741531185599394439863864773280788,
	-0.586087235467691130294144838258730,
	-0.405845151377397166906606412076961,
	-0.207784955007898467600689403773245,
	0.0,
	0.207784955007898467600689403773245,
	0.405845151377397166906606412076961,
	0.586087235467691130294144838258730,
	0.741531185599394439863864773280788,
	0.864864423359769072789712788640926,
	0.949107912342758524526189684047851,
	0.991455371120812639206854697526329,
}

var gaussKronrod15Weights = [...]float64{
	0.022935322010529224963732008058970,
	0.063092092629978553290700663189204,
	0.104790010322250183839876322541518,
	0.140653259715525918745189590510238,
	0.169004726639267902826583426598550,
	0.190350578064785409913256402421014,
	0.204432940075298892414161999234649,
	0.209482141084727828012999174891714,
	0.204432940075298892414161999234649,
	0.190350578064785409913256402421014,
	0.169004726639267902826583426598550,
	0.140653259715525918745189590510238,
	0.104790010322250183839876322541518,
	0.063092092629978553290700663189204,
	0.022935322010529224963732008058970,
}

var gauss7Weights = [...]float64{
	0.000000000000000000000000000000000,
	0.129484966168869693270611432679082,
	0.000000000000000000000000000000000,
	0.279705391489276667901467771423780,
	0.000000000000000000000000000000000,
	0.381830050505118944950369775488975,
	0.000000000000000000000000000000000,
	0.417959183673469387755102040816327,
	0.000000000000000000000000000000000,
	0.381830050505118944950369775488975,
	0.000000000000000000000000000000000,
	0.279705391489276667901467771423780,
	0.000000000000000000000000000000000,
	0.129484966168869693270611432679082,
	0.000000000000000000000000000000000,
}

func GaussKronrod15(parser *Parser, a, b float64) (result float64, errorEstimate float64, err error) {
	c := (a + b) / 2.0
	halfLen := (b - a) / 2.0
	
	var gaussSum, kronrodSum float64
	
	for i := 0; i < 15; i++ {
		x := c + halfLen * gaussKronrod15Nodes[i]
		fx, err := parser.Evaluate(x)
		if err != nil {
			return 0, 0, err
		}
		kronrodSum += gaussKronrod15Weights[i] * fx
		gaussSum += gauss7Weights[i] * fx
	}
	
	result = kronrodSum * halfLen
	gaussResult := gaussSum * halfLen
	errorEstimate = math.Abs(result - gaussResult)
	
	return result, errorEstimate, nil
}

func AdaptiveGaussKronrod(expr string, a, b float64, eps ...float64) (float64, float64, error) {
	epsilon := DefaultEpsilon
	if len(eps) > 0 && eps[0] > 0 {
		epsilon = eps[0]
	}
	
	if math.Abs(b-a) < MinIntervalWidth {
		return 0, 0, fmt.Errorf("interval width is too small")
	}
	
	parser := NewParser(expr)
	
	type interval struct {
		a, b float64
	}
	
	stack := []interval{{a, b}}
	totalResult := 0.0
	totalError := 0.0
	maxIterations := 1000
	iterations := 0
	
	for len(stack) > 0 && iterations < maxIterations {
		iterations++
		last := len(stack) - 1
		ival := stack[last]
		stack = stack[:last]
		
		result, errEst, err := GaussKronrod15(parser, ival.a, ival.b)
		if err != nil {
			return 0, 0, err
		}
		
		if errEst <= epsilon*(ival.b-ival.a)/(b-a) {
			totalResult += result
			totalError += errEst
		} else {
			mid := (ival.a + ival.b) / 2.0
			stack = append(stack, interval{mid, ival.b})
			stack = append(stack, interval{ival.a, mid})
		}
	}
	
	return totalResult, totalError, nil
}

func HybridAdaptiveIntegral(expr string, a, b float64, eps ...float64) (float64, string, error) {
	epsilon := DefaultEpsilon
	if len(eps) > 0 && eps[0] > 0 {
		epsilon = eps[0]
	}
	
	parser := NewParser(expr)
	
	fa, _ := parser.Evaluate(a)
	fb, _ := parser.Evaluate(b)
	
	s, fc, err := simpsonRuleWithPoints(parser, a, b, fa, fb)
	if err != nil {
		return 0, "", err
	}
	
	c := (a + b) / 2.0
	sLeft, _, err := simpsonRuleWithPoints(parser, a, c, fa, fc)
	if err != nil {
		return 0, "", err
	}
	sRight, _, err := simpsonRuleWithPoints(parser, c, b, fc, fb)
	if err != nil {
		return 0, "", err
	}
	
	simpsonError := math.Abs(sLeft + sRight - s)
	
	if simpsonError <= SwitchToKronrodThreshold*epsilon {
		result, err := AdaptiveSimpson(expr, a, b, epsilon)
		if err != nil {
			return 0, "", err
		}
		return result, "Simpson", nil
	}
	
	result, _, err := AdaptiveGaussKronrod(expr, a, b, epsilon)
	if err != nil {
		return 0, "", err
	}
	return result, "Gauss-Kronrod", nil
}
