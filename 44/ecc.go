package main

import (
	"fmt"
	"math/big"
	"math"
)

type Point struct {
	X *big.Int
	Y *big.Int
}

type EllipticCurve struct {
	A *big.Int
	B *big.Int
	P *big.Int
	N *big.Int
	G Point
}

func NewPoint(x, y *big.Int) Point {
	return Point{X: new(big.Int).Set(x), Y: new(big.Int).Set(y)}
}

func (p Point) IsInfinity() bool {
	return p.X == nil && p.Y == nil
}

func Infinity() Point {
	return Point{X: nil, Y: nil}
}

func (ec *EllipticCurve) IsOnCurve(p Point) bool {
	if p.IsInfinity() {
		return true
	}
	
	y2 := new(big.Int).Mul(p.Y, p.Y)
	y2.Mod(y2, ec.P)
	
	x3 := new(big.Int).Mul(p.X, p.X)
	x3.Mul(x3, p.X)
	
	ax := new(big.Int).Mul(ec.A, p.X)
	
	rhs := new(big.Int).Add(x3, ax)
	rhs.Add(rhs, ec.B)
	rhs.Mod(rhs, ec.P)
	
	return y2.Cmp(rhs) == 0
}

func (ec *EllipticCurve) PointAdd(p1, p2 Point) Point {
	if p1.IsInfinity() {
		return p2
	}
	if p2.IsInfinity() {
		return p1
	}
	
	negP2Y := new(big.Int).Neg(p2.Y)
	negP2Y.Mod(negP2Y, ec.P)
	if p1.X.Cmp(p2.X) == 0 && p1.Y.Cmp(negP2Y) == 0 {
		return Infinity()
	}
	
	m := new(big.Int)
	
	if p1.X.Cmp(p2.X) == 0 && p1.Y.Cmp(p2.Y) == 0 {
		threeX2 := new(big.Int).Mul(big.NewInt(3), new(big.Int).Mul(p1.X, p1.X))
		threeX2a := new(big.Int).Add(threeX2, ec.A)
		twoY := new(big.Int).Mul(big.NewInt(2), p1.Y)
		invTwoY := new(big.Int).ModInverse(twoY, ec.P)
		m.Mul(threeX2a, invTwoY)
	} else {
		dy := new(big.Int).Sub(p2.Y, p1.Y)
		dx := new(big.Int).Sub(p2.X, p1.X)
		invDx := new(big.Int).ModInverse(dx, ec.P)
		m.Mul(dy, invDx)
	}
	m.Mod(m, ec.P)
	
	x3 := new(big.Int).Mul(m, m)
	x3.Sub(x3, p1.X)
	x3.Sub(x3, p2.X)
	x3.Mod(x3, ec.P)
	
	y3 := new(big.Int).Sub(x3, p1.X)
	y3.Mul(y3, m)
	y3.Add(y3, p1.Y)
	y3.Neg(y3)
	y3.Mod(y3, ec.P)
	
	return NewPoint(x3, y3)
}

func (ec *EllipticCurve) ScalarMult(k *big.Int, p Point) Point {
	result := Infinity()
	current := p
	
	kCopy := new(big.Int).Set(k)
	one := big.NewInt(1)
	
	for kCopy.Sign() > 0 {
		if new(big.Int).And(kCopy, one).Cmp(one) == 0 {
			result = ec.PointAdd(result, current)
		}
		current = ec.PointAdd(current, current)
		kCopy.Rsh(kCopy, 1)
	}
	
	return result
}

func (ec *EllipticCurve) BabyStepGiantStep(P, Q Point, maxK *big.Int) (*big.Int, error) {
	mFloat := math.Ceil(math.Sqrt(float64(maxK.Int64())))
	m := big.NewInt(int64(mFloat))
	
	babySteps := make(map[string]*big.Int)
	
	current := Infinity()
	for j := big.NewInt(0); j.Cmp(m) < 0; j.Add(j, big.NewInt(1)) {
		key := ""
		if !current.IsInfinity() {
			key = fmt.Sprintf("%x,%x", current.X, current.Y)
		} else {
			key = "inf"
		}
		if _, exists := babySteps[key]; !exists {
			babySteps[key] = new(big.Int).Set(j)
		}
		current = ec.PointAdd(current, P)
	}
	
	mP := ec.ScalarMult(m, P)
	negMP := NewPoint(mP.X, new(big.Int).Neg(mP.Y).Mod(new(big.Int).Neg(mP.Y), ec.P))
	
	currentQ := Q
	for i := big.NewInt(0); i.Cmp(m) < 0; i.Add(i, big.NewInt(1)) {
		key := ""
		if !currentQ.IsInfinity() {
			key = fmt.Sprintf("%x,%x", currentQ.X, currentQ.Y)
		} else {
			key = "inf"
		}
		if j, exists := babySteps[key]; exists {
			k := new(big.Int).Mul(i, m)
			k.Add(k, j)
			if k.Cmp(maxK) <= 0 {
				return k, nil
			}
		}
		currentQ = ec.PointAdd(currentQ, negMP)
	}
	
	return nil, fmt.Errorf("discrete logarithm not found within range")
}

func (ec *EllipticCurve) ComputeEmbeddingDegree(maxK int) (int, error) {
	one := big.NewInt(1)
	pk := new(big.Int).Set(ec.P)
	
	for k := 1; k <= maxK; k++ {
		pkMinus1 := new(big.Int).Sub(pk, one)
		if new(big.Int).Mod(pkMinus1, ec.N).Sign() == 0 {
			return k, nil
		}
		pk.Mul(pk, ec.P)
		pk.Mod(pk, ec.N)
	}
	
	return 0, fmt.Errorf("embedding degree not found within %d iterations", maxK)
}

func (ec *EllipticCurve) SecurityLevel() map[string]interface{} {
	result := make(map[string]interface{})
	
	nBits := ec.N.BitLen()
	result["order_bits"] = nBits
	
	babyStepGiantStepBits := nBits / 2
	result["baby_step_giant_step_bits"] = babyStepGiantStepBits
	
	pBits := ec.P.BitLen()
	result["field_size_bits"] = pBits
	
	squareRootBits := pBits / 2
	result["generic_dlp_security_bits"] = squareRootBits
	
	embeddingDegree, err := ec.ComputeEmbeddingDegree(100)
	if err == nil {
		result["embedding_degree"] = embeddingDegree
		result["mov_attack_risk"] = embeddingDegree <= 6
		
		if embeddingDegree <= 6 {
			result["mov_security_bits"] = pBits / (embeddingDegree * 2)
		} else {
			result["mov_security_bits"] = squareRootBits
		}
	} else {
		result["embedding_degree"] = ">100"
		result["mov_attack_risk"] = false
		result["mov_security_bits"] = squareRootBits
	}
	
	discriminant := new(big.Int).Mul(big.NewInt(4), new(big.Int).Exp(ec.A, big.NewInt(3), nil))
	discriminant.Add(discriminant, new(big.Int).Mul(big.NewInt(27), new(big.Int).Exp(ec.B, big.NewInt(2), nil))))
	discriminant.Mod(discriminant, ec.P)
	result["discriminant_nonzero"] = discriminant.Sign() != 0
	result["is_singular"] = discriminant.Sign() == 0
	
	minBits := babyStepGiantStepBits
	if movBits, ok := result["mov_security_bits"].(int); ok && movBits < minBits {
		minBits = movBits
	}
	result["effective_security_bits"] = minBits
	
	return result
}

func main() {
	p := big.NewInt(17)
	a := big.NewInt(2)
	b := big.NewInt(2)
	n := big.NewInt(19)
	
	gx := big.NewInt(5)
	gy := big.NewInt(1)
	G := NewPoint(gx, gy)
	
	ec := &EllipticCurve{
		A: a,
		B: b,
		P: p,
		N: n,
		G: G,
	}
	
	fmt.Println("=== 椭圆曲线参数 ===")
	fmt.Printf("曲线方程: y² = x³ + %dx + %d mod %d\n", a, b, p)
	fmt.Printf("基点 G: (%d, %d)\n", gx, gy)
	fmt.Printf("阶 N: %d\n\n", n)
	
	fmt.Println("=== 安全性评估 ===")
	security := ec.SecurityLevel()
	fmt.Printf("域大小 (p 的比特数): %d bits\n", security["field_size_bits"])
	fmt.Printf("子群阶 (n 的比特数): %d bits\n", security["order_bits"])
	fmt.Printf("通用 DLP 安全性 (平方根): %d bits\n", security["generic_dlp_security_bits"])
	fmt.Printf("Baby-step Giant-step 复杂度: %d bits\n", security["baby_step_giant_step_bits"])
	fmt.Printf("嵌入度 (Embedding Degree): %v\n", security["embedding_degree"])
	fmt.Printf("MOV 攻击风险: %v\n", security["mov_attack_risk"])
	if security["mov_attack_risk"] == true {
		fmt.Printf("  警告: 嵌入度 <= 6，易受MOV攻击！\n")
		fmt.Printf("  MOV 攻击后安全级别: %d bits\n", security["mov_security_bits"])
	}
	fmt.Printf("曲线非奇异 (判别式 ≠ 0): %v\n", security["discriminant_nonzero"])
	fmt.Printf("实际有效安全级别: %d bits\n", security["effective_security_bits"])
	fmt.Println()
	
	fmt.Println("=== 更大曲线示例 ===")
	p2, _ := new(big.Int).SetString("115792089237316195423570985008687907853269984665640564039457584007908834671663", 10)
	n2, _ := new(big.Int).SetString("115792089237316195423570985008687907852837564279074904382605163141518161494337", 10)
	ec256 := &EllipticCurve{
		A: big.NewInt(0),
		B: big.NewInt(7),
		P: p2,
		N: n2,
		G: NewPoint(big.NewInt(0), big.NewInt(0)),
	}
	security256 := ec256.SecurityLevel()
	fmt.Printf("secp256k1 类曲线安全评估:\n")
	fmt.Printf("  域大小: %d bits\n", security256["field_size_bits"])
	fmt.Printf("  子群阶: %d bits\n", security256["order_bits"])
	fmt.Printf("  通用 DLP 安全性: %d bits\n", security256["generic_dlp_security_bits"])
	fmt.Printf("  嵌入度: %v\n", security256["embedding_degree"])
	fmt.Printf("  MOV 攻击风险: %v\n", security256["mov_attack_risk"])
	fmt.Printf("  有效安全级别: %d bits\n", security256["effective_security_bits"])
	fmt.Println()
	
	fmt.Println("=== 点加倍运算验证 (G + G) ===")
	G2 := ec.PointAdd(G, G)
	fmt.Printf("G + G = (%d, %d)\n", G2.X, G2.Y)
	fmt.Printf("验证在曲线上: %v\n", ec.IsOnCurve(G2))
	
	fmt.Println("\n=== 验证 y² = x³ + ax + b ===")
	x := G2.X.Int64()
	y := G2.Y.Int64()
	y2 := (y * y) % 17
	rhs := (x*x*x + 2*x + 2) % 17
	fmt.Printf("y² = %d² = %d mod 17\n", y, y2)
	fmt.Printf("x³ + ax + b = %d³ + 2*%d + 2 = %d mod 17\n", x, x, rhs)
	fmt.Printf("相等: %v\n\n", y2 == rhs)
	
	fmt.Println("=== 点加运算测试 ===")
	P1 := NewPoint(big.NewInt(5), big.NewInt(1))
	P2 := NewPoint(big.NewInt(6), big.NewInt(3))
	P3 := ec.PointAdd(P1, P2)
	fmt.Printf("P1: (%d, %d)\n", P1.X, P1.Y)
	fmt.Printf("P2: (%d, %d)\n", P2.X, P2.Y)
	fmt.Printf("P1 + P2: (%d, %d)\n", P3.X, P3.Y)
	fmt.Printf("验证在曲线上: %v\n\n", ec.IsOnCurve(P3))
	
	fmt.Println("=== 点乘运算测试 ===")
	kValues := []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	for _, k := range kValues {
		kBig := big.NewInt(k)
		Q := ec.ScalarMult(kBig, G)
		fmt.Printf("%2d * G = ", k)
		if Q.IsInfinity() {
			fmt.Printf("∞ (无穷远点)")
		} else {
			fmt.Printf("(%2d, %2d)", Q.X, Q.Y)
		}
		fmt.Printf("  在曲线上: %v\n", ec.IsOnCurve(Q))
	}
	fmt.Println()
	
	fmt.Println("=== 验证 19*G = ∞ (阶验证) ===")
	G19 := ec.ScalarMult(big.NewInt(19), G)
	fmt.Printf("19 * G = ∞: %v\n\n", G19.IsInfinity())
	
	fmt.Println("=== Baby-step Giant-step 离散对数求解 ===")
	testK := big.NewInt(11)
	Q := ec.ScalarMult(testK, G)
	fmt.Printf("已知 P = G = (%d, %d)\n", G.X, G.Y)
	fmt.Printf("已知 Q = %d*G = (%d, %d)\n", testK, Q.X, Q.Y)
	fmt.Println("求解 k 使得 Q = k*G...")
	
	foundK, err := ec.BabyStepGiantStep(G, Q, n)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
	} else {
		fmt.Printf("找到 k = %d\n", foundK)
		verifyQ := ec.ScalarMult(foundK, G)
		fmt.Printf("验证: %d * G = (%d, %d)\n", foundK, verifyQ.X, verifyQ.Y)
		fmt.Printf("匹配: %v\n", verifyQ.X.Cmp(Q.X) == 0 && verifyQ.Y.Cmp(Q.Y) == 0)
	}
}
