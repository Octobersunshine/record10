package mathutil

import (
	"fmt"
	"math"
	"math/cmplx"
	"strconv"
	"strings"
	"unicode"
)

type TokenType int

const (
	TokenNumber TokenType = iota
	TokenVariable
	TokenOperator
	TokenLeftParen
	TokenRightParen
	TokenEOF
)

type Token struct {
	Type  TokenType
	Value string
}

type Node interface {
	Evaluate(x float64) float64
	EvaluateComplex(x complex128) complex128
	Differentiate() Node
	Simplify() Node
	String() string
}

type NumberNode struct {
	Value float64
}

func (n *NumberNode) Evaluate(x float64) float64 {
	return n.Value
}

func (n *NumberNode) EvaluateComplex(x complex128) complex128 {
	return complex(n.Value, 0)
}

func (n *NumberNode) Differentiate() Node {
	return &NumberNode{Value: 0}
}

func (n *NumberNode) Simplify() Node {
	return n
}

func (n *NumberNode) String() string {
	if math.Abs(n.Value-math.Round(n.Value)) < 1e-10 {
		return fmt.Sprintf("%d", int(math.Round(n.Value)))
	}
	return fmt.Sprintf("%.6g", n.Value)
}

type VariableNode struct{}

func (v *VariableNode) Evaluate(x float64) float64 {
	return x
}

func (v *VariableNode) EvaluateComplex(x complex128) complex128 {
	return x
}

func (v *VariableNode) Differentiate() Node {
	return &NumberNode{Value: 1}
}

func (v *VariableNode) Simplify() Node {
	return v
}

func (v *VariableNode) String() string {
	return "x"
}

type BinaryOpNode struct {
	Left     Node
	Right    Node
	Operator rune
}

func (b *BinaryOpNode) Evaluate(x float64) float64 {
	leftVal := b.Left.Evaluate(x)
	rightVal := b.Right.Evaluate(x)
	switch b.Operator {
	case '+':
		return leftVal + rightVal
	case '-':
		return leftVal - rightVal
	case '*':
		return leftVal * rightVal
	case '/':
		return leftVal / rightVal
	case '^':
		return math.Pow(leftVal, rightVal)
	default:
		return 0
	}
}

func (b *BinaryOpNode) EvaluateComplex(x complex128) complex128 {
	leftVal := b.Left.EvaluateComplex(x)
	rightVal := b.Right.EvaluateComplex(x)
	switch b.Operator {
	case '+':
		return leftVal + rightVal
	case '-':
		return leftVal - rightVal
	case '*':
		return leftVal * rightVal
	case '/':
		return leftVal / rightVal
	case '^':
		return cmplx.Pow(leftVal, rightVal)
	default:
		return 0
	}
}

func (b *BinaryOpNode) Differentiate() Node {
	u := b.Left
	v := b.Right
	du := u.Differentiate()
	dv := v.Differentiate()

	switch b.Operator {
	case '+':
		return &BinaryOpNode{Left: du, Right: dv, Operator: '+'}
	case '-':
		return &BinaryOpNode{Left: du, Right: dv, Operator: '-'}
	case '*':
		term1 := &BinaryOpNode{Left: du, Right: v, Operator: '*'}
		term2 := &BinaryOpNode{Left: u, Right: dv, Operator: '*'}
		return &BinaryOpNode{Left: term1, Right: term2, Operator: '+'}
	case '/':
		numer1 := &BinaryOpNode{Left: du, Right: v, Operator: '*'}
		numer2 := &BinaryOpNode{Left: u, Right: dv, Operator: '*'}
		numer := &BinaryOpNode{Left: numer1, Right: numer2, Operator: '-'}
		denom := &BinaryOpNode{Left: v, Right: &NumberNode{Value: 2}, Operator: '^'}
		return &BinaryOpNode{Left: numer, Right: denom, Operator: '/'}
	case '^':
		if _, ok := v.(*NumberNode); ok {
			n := v.(*NumberNode).Value
			term1 := &NumberNode{Value: n}
			term2 := &BinaryOpNode{Left: u, Right: &NumberNode{Value: n - 1}, Operator: '^'}
			term3 := du
			part1 := &BinaryOpNode{Left: term1, Right: term2, Operator: '*'}
			return &BinaryOpNode{Left: part1, Right: term3, Operator: '*'}
		}
		lnu := &UnaryOpNode{Right: u, Operator: 'l'}
		term1 := &BinaryOpNode{Left: dv, Right: lnu, Operator: '*'}
		term2 := &BinaryOpNode{Left: v, Right: du, Operator: '*'}
		term3 := &BinaryOpNode{Left: term2, Right: u, Operator: '/'}
		part := &BinaryOpNode{Left: term1, Right: term3, Operator: '+'}
		return &BinaryOpNode{Left: b, Right: part, Operator: '*'}
	default:
		return &NumberNode{Value: 0}
	}
}

func (b *BinaryOpNode) Simplify() Node {
	left := b.Left.Simplify()
	right := b.Right.Simplify()

	leftNum, leftIsNum := left.(*NumberNode)
	rightNum, rightIsNum := right.(*NumberNode)

	switch b.Operator {
	case '+':
		if leftIsNum && rightIsNum {
			return &NumberNode{Value: leftNum.Value + rightNum.Value}
		}
		if leftIsNum && leftNum.Value == 0 {
			return right
		}
		if rightIsNum && rightNum.Value == 0 {
			return left
		}
	case '-':
		if leftIsNum && rightIsNum {
			return &NumberNode{Value: leftNum.Value - rightNum.Value}
		}
		if rightIsNum && rightNum.Value == 0 {
			return left
		}
	case '*':
		if leftIsNum && rightIsNum {
			return &NumberNode{Value: leftNum.Value * rightNum.Value}
		}
		if leftIsNum && leftNum.Value == 0 {
			return &NumberNode{Value: 0}
		}
		if rightIsNum && rightNum.Value == 0 {
			return &NumberNode{Value: 0}
		}
		if leftIsNum && leftNum.Value == 1 {
			return right
		}
		if rightIsNum && rightNum.Value == 1 {
			return left
		}
	case '/':
		if leftIsNum && rightIsNum {
			return &NumberNode{Value: leftNum.Value / rightNum.Value}
		}
		if leftIsNum && leftNum.Value == 0 {
			return &NumberNode{Value: 0}
		}
		if rightIsNum && rightNum.Value == 1 {
			return left
		}
	case '^':
		if leftIsNum && rightIsNum {
			return &NumberNode{Value: math.Pow(leftNum.Value, rightNum.Value)}
		}
		if rightIsNum && rightNum.Value == 0 {
			return &NumberNode{Value: 1}
		}
		if rightIsNum && rightNum.Value == 1 {
			return left
		}
		if leftIsNum && leftNum.Value == 1 {
			return &NumberNode{Value: 1}
		}
	}
	return &BinaryOpNode{Left: left, Right: right, Operator: b.Operator}
}

func (b *BinaryOpNode) String() string {
	leftStr := b.Left.String()
	rightStr := b.Right.String()

	if _, ok := b.Left.(*BinaryOpNode); ok {
		leftStr = "(" + leftStr + ")"
	}
	if _, ok := b.Right.(*BinaryOpNode); ok {
		if b.Operator == '^' || b.Operator == '*' || b.Operator == '/' {
			if _, isAddSub := b.Right.(*BinaryOpNode); isAddSub && (b.Right.(*BinaryOpNode).Operator == '+' || b.Right.(*BinaryOpNode).Operator == '-') {
				rightStr = "(" + rightStr + ")"
			}
		}
	}
	return leftStr + string(b.Operator) + rightStr
}

type UnaryOpNode struct {
	Right    Node
	Operator rune
}

func (u *UnaryOpNode) Evaluate(x float64) float64 {
	rightVal := u.Right.Evaluate(x)
	switch u.Operator {
	case '-':
		return -rightVal
	case '+':
		return rightVal
	case 'l':
		return math.Log(rightVal)
	default:
		return rightVal
	}
}

func (u *UnaryOpNode) EvaluateComplex(x complex128) complex128 {
	rightVal := u.Right.EvaluateComplex(x)
	switch u.Operator {
	case '-':
		return -rightVal
	case '+':
		return rightVal
	case 'l':
		return cmplx.Log(rightVal)
	default:
		return rightVal
	}
}

func (u *UnaryOpNode) Differentiate() Node {
	dr := u.Right.Differentiate()
	switch u.Operator {
	case '-':
		return &UnaryOpNode{Right: dr, Operator: '-'}
	case '+':
		return dr
	case 'l':
		return &BinaryOpNode{Left: dr, Right: u.Right, Operator: '/'}
	default:
		return dr
	}
}

func (u *UnaryOpNode) Simplify() Node {
	right := u.Right.Simplify()
	if rightNum, ok := right.(*NumberNode); ok {
		switch u.Operator {
		case '-':
			return &NumberNode{Value: -rightNum.Value}
		case '+':
			return right
		case 'l':
			return &NumberNode{Value: math.Log(rightNum.Value)}
		}
	}
	if u.Operator == '+' {
		return right
	}
	return &UnaryOpNode{Right: right, Operator: u.Operator}
}

func (u *UnaryOpNode) String() string {
	rightStr := u.Right.String()
	if _, ok := u.Right.(*BinaryOpNode); ok {
		rightStr = "(" + rightStr + ")"
	}
	switch u.Operator {
	case '-':
		return "-" + rightStr
	case '+':
		return "+" + rightStr
	case 'l':
		return "ln(" + rightStr + ")"
	default:
		return rightStr
	}
}

type Parser struct {
	tokens  []Token
	pos     int
	current Token
}

func tokenize(expr string) []Token {
	var tokens []Token
	expr = strings.ReplaceAll(expr, " ", "")
	var buf []rune

	for i := 0; i < len(expr); i++ {
		c := rune(expr[i])
		switch {
		case unicode.IsDigit(c) || c == '.':
			buf = append(buf, c)
		case c == 'x':
			if len(buf) > 0 {
				tokens = append(tokens, Token{TokenNumber, string(buf)})
				buf = nil
			}
			tokens = append(tokens, Token{TokenVariable, "x"})
		case strings.ContainsRune("+-*/^", c):
			if len(buf) > 0 {
				tokens = append(tokens, Token{TokenNumber, string(buf)})
				buf = nil
			}
			tokens = append(tokens, Token{TokenOperator, string(c)})
		case c == '(':
			if len(buf) > 0 {
				tokens = append(tokens, Token{TokenNumber, string(buf)})
				buf = nil
			}
			tokens = append(tokens, Token{TokenLeftParen, "("})
		case c == ')':
			if len(buf) > 0 {
				tokens = append(tokens, Token{TokenNumber, string(buf)})
				buf = nil
			}
			tokens = append(tokens, Token{TokenRightParen, ")"})
		default:
			panic(fmt.Sprintf("unexpected character: %c", c))
		}
	}
	if len(buf) > 0 {
		tokens = append(tokens, Token{TokenNumber, string(buf)})
	}
	tokens = append(tokens, Token{TokenEOF, ""})
	return tokens
}

func NewParser(expr string) *Parser {
	tokens := tokenize(expr)
	return &Parser{
		tokens:  tokens,
		pos:     0,
		current: tokens[0],
	}
}

func (p *Parser) advance() {
	p.pos++
	if p.pos < len(p.tokens) {
		p.current = p.tokens[p.pos]
	} else {
		p.current = Token{TokenEOF, ""}
	}
}

func (p *Parser) parsePrimary() Node {
	switch p.current.Type {
	case TokenNumber:
		val, _ := strconv.ParseFloat(p.current.Value, 64)
		p.advance()
		return &NumberNode{Value: val}
	case TokenVariable:
		p.advance()
		return &VariableNode{}
	case TokenLeftParen:
		p.advance()
		node := p.parseExpression()
		if p.current.Type != TokenRightParen {
			panic("missing closing parenthesis")
		}
		p.advance()
		return node
	case TokenOperator:
		if p.current.Value == "-" || p.current.Value == "+" {
			op := rune(p.current.Value[0])
			p.advance()
			return &UnaryOpNode{Right: p.parsePrimary(), Operator: op}
		}
		fallthrough
	default:
		panic(fmt.Sprintf("unexpected token: %v", p.current))
	}
}

func (p *Parser) parsePower() Node {
	node := p.parsePrimary()
	for p.current.Type == TokenOperator && p.current.Value == "^" {
		op := rune(p.current.Value[0])
		p.advance()
		node = &BinaryOpNode{Left: node, Right: p.parsePrimary(), Operator: op}
	}
	return node
}

func (p *Parser) parseMulDiv() Node {
	node := p.parsePower()
	for p.current.Type == TokenOperator && (p.current.Value == "*" || p.current.Value == "/") {
		op := rune(p.current.Value[0])
		p.advance()
		node = &BinaryOpNode{Left: node, Right: p.parsePower(), Operator: op}
	}
	return node
}

func (p *Parser) parseAddSub() Node {
	node := p.parseMulDiv()
	for p.current.Type == TokenOperator && (p.current.Value == "+" || p.current.Value == "-") {
		op := rune(p.current.Value[0])
		p.advance()
		node = &BinaryOpNode{Left: node, Right: p.parseMulDiv(), Operator: op}
	}
	return node
}

func (p *Parser) parseExpression() Node {
	return p.parseAddSub()
}

func (p *Parser) Parse() (Node, error) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	node := p.parseExpression()
	if p.current.Type != TokenEOF {
		return nil, fmt.Errorf("unexpected token at end: %v", p.current)
	}
	return node, nil
}

func Evaluate(expr string, x float64) (float64, error) {
	parser := NewParser(expr)
	node, err := parser.Parse()
	if err != nil {
		return 0, err
	}
	return node.Evaluate(x), nil
}

func EvaluateComplex(expr string, x complex128) (complex128, error) {
	parser := NewParser(expr)
	node, err := parser.Parse()
	if err != nil {
		return 0, err
	}
	return node.EvaluateComplex(x), nil
}

func Derivative(expr string, x float64, h float64) (float64, error) {
	fxh, err := Evaluate(expr, x+h)
	if err != nil {
		return 0, err
	}
	fxnh, err := Evaluate(expr, x-h)
	if err != nil {
		return 0, err
	}
	return (fxh - fxnh) / (2 * h), nil
}

func SecondDerivative(expr string, x float64, h float64) (float64, error) {
	fxh, err := Evaluate(expr, x+h)
	if err != nil {
		return 0, err
	}
	fx, err := Evaluate(expr, x)
	if err != nil {
		return 0, err
	}
	fxnh, err := Evaluate(expr, x-h)
	if err != nil {
		return 0, err
	}
	return (fxh - 2*fx + fxnh) / (h * h), nil
}

func DerivativeComplex(expr string, x complex128, h complex128) (complex128, error) {
	fxh, err := EvaluateComplex(expr, x+h)
	if err != nil {
		return 0, err
	}
	fxnh, err := EvaluateComplex(expr, x-h)
	if err != nil {
		return 0, err
	}
	return (fxh - fxnh) / (2 * h), nil
}

type ValidationResult struct {
	IsValid    bool
	IsInDomain bool
	Suggestion float64
	Message    string
}

func ValidateInitialValue(expr string, x0 float64) ValidationResult {
	result := ValidationResult{
		IsValid:    true,
		IsInDomain: true,
		Suggestion: x0,
	}

	if math.IsInf(x0, 0) || math.IsNaN(x0) {
		result.IsValid = false
		result.IsInDomain = false
		result.Message = "初始值为无穷大或NaN"
		return result
	}

	fx, err := Evaluate(expr, x0)
	if err != nil {
		result.IsValid = false
		result.IsInDomain = false
		result.Message = "表达式解析错误: " + err.Error()
		return result
	}

	if math.IsInf(fx, 0) || math.IsNaN(fx) {
		result.IsInDomain = false
		result.Message = "初始值导致表达式求值为无穷或NaN"
		for _, try := range []float64{0, 1, -1, 2, -2, 0.5, -0.5, 10, -10} {
			if try == x0 {
				continue
			}
			fTry, err := Evaluate(expr, try)
			if err == nil && !math.IsInf(fTry, 0) && !math.IsNaN(fTry) {
				result.Suggestion = try
				result.Message += fmt.Sprintf(", 建议尝试: %g", try)
				break
			}
		}
		return result
	}

	dfx, err := Derivative(expr, x0, 1e-6)
	if err != nil || math.IsInf(dfx, 0) || math.IsNaN(dfx) {
		result.Message = "警告：在初始值处导数异常"
	}

	return result
}

func IsDiverging(fx float64, prevFx float64) bool {
	if math.IsInf(fx, 0) || math.IsNaN(fx) {
		return true
	}
	return math.Abs(fx) > 1e6 * math.Max(1.0, math.Abs(prevFx))
}

func IsDivergingComplex(fx complex128, prevFx complex128) bool {
	if cmplx.IsInf(fx) || cmplx.IsNaN(fx) {
		return true
	}
	return cmplx.Abs(fx) > 1e6 * math.Max(1.0, cmplx.Abs(prevFx))
}

func ParseExpression(expr string) (Node, error) {
	parser := NewParser(expr)
	return parser.Parse()
}

func SymbolicDerivative(expr string) (string, error) {
	node, err := ParseExpression(expr)
	if err != nil {
		return "", err
	}
	derivative := node.Differentiate()
	simplified := derivative.Simplify()
	return simplified.String(), nil
}

func EvaluateDerivative(expr string, x float64) (float64, error) {
	node, err := ParseExpression(expr)
	if err != nil {
		return 0, err
	}
	derivative := node.Differentiate()
	simplified := derivative.Simplify()
	return simplified.Evaluate(x), nil
}

func EvaluateDerivativeComplex(expr string, x complex128) (complex128, error) {
	node, err := ParseExpression(expr)
	if err != nil {
		return 0, err
	}
	derivative := node.Differentiate()
	simplified := derivative.Simplify()
	return simplified.EvaluateComplex(x), nil
}
