package main

import (
	"crypto/rand"
	"math/big"
)

type CurveParams struct {
	P       *big.Int
	A       *big.Int
	B       *big.Int
	Gx      *big.Int
	Gy      *big.Int
	N       *big.Int
	BitSize int
}

type Point struct {
	X, Y *big.Int
}

func NewPoint() *Point {
	return &Point{
		X: new(big.Int),
		Y: new(big.Int),
	}
}

func (p *Point) Set(x, y *big.Int) *Point {
	p.X.Set(x)
	p.Y.Set(y)
	return p
}

func (p *Point) IsZero() bool {
	return p.X.Sign() == 0 && p.Y.Sign() == 0
}

func (p *Point) Copy() *Point {
	return &Point{
		X: new(big.Int).Set(p.X),
		Y: new(big.Int).Set(p.Y),
	}
}

func cswapBigInt(a, b *big.Int, mask big.Word) {
	wordsA := a.Bits()
	wordsB := b.Bits()

	maxLen := len(wordsA)
	if len(wordsB) > maxLen {
		maxLen = len(wordsB)
	}

	extA := make([]big.Word, maxLen)
	extB := make([]big.Word, maxLen)
	copy(extA, wordsA)
	copy(extB, wordsB)

	for i := 0; i < maxLen; i++ {
		diff := mask & (extA[i] ^ extB[i])
		extA[i] ^= diff
		extB[i] ^= diff
	}

	a.SetBits(extA)
	b.SetBits(extB)
}

func cswapPoint(a, b *Point, bit uint) {
	mask := big.Word(bit)
	mask = -mask

	cswapBigInt(a.X, b.X, mask)
	cswapBigInt(a.Y, b.Y, mask)
}

func newIntFromString(s string) *big.Int {
	i, _ := new(big.Int).SetString(s, 16)
	return i
}

func Secp256k1() *CurveParams {
	return &CurveParams{
		P:       newIntFromString("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"),
		A:       new(big.Int),
		B:       big.NewInt(7),
		Gx:      newIntFromString("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"),
		Gy:      newIntFromString("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"),
		N:       newIntFromString("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"),
		BitSize: 256,
	}
}

func (curve *CurveParams) modAdd(a, b *big.Int) *big.Int {
	result := new(big.Int).Add(a, b)
	return result.Mod(result, curve.P)
}

func (curve *CurveParams) modSub(a, b *big.Int) *big.Int {
	result := new(big.Int).Sub(a, b)
	return result.Mod(result, curve.P)
}

func (curve *CurveParams) modMul(a, b *big.Int) *big.Int {
	result := new(big.Int).Mul(a, b)
	return result.Mod(result, curve.P)
}

func (curve *CurveParams) modDiv(a, b *big.Int) *big.Int {
	inv := new(big.Int).ModInverse(b, curve.P)
	if inv == nil {
		return big.NewInt(0)
	}
	return curve.modMul(a, inv)
}

func (curve *CurveParams) modSquare(a *big.Int) *big.Int {
	return curve.modMul(a, a)
}

func (curve *CurveParams) IsOnCurve(p *Point) bool {
	if p.IsZero() {
		return true
	}

	y2 := curve.modSquare(p.Y)
	x3axb := curve.modMul(curve.modAdd(curve.modSquare(p.X), curve.A), p.X)
	x3axb = curve.modAdd(x3axb, curve.B)

	return y2.Cmp(x3axb) == 0
}

func (curve *CurveParams) Add(p1, p2 *Point) *Point {
	if p1.IsZero() {
		return p2.Copy()
	}
	if p2.IsZero() {
		return p1.Copy()
	}

	var m *big.Int
	if p1.X.Cmp(p2.X) == 0 {
		if p1.Y.Cmp(p2.Y) != 0 {
			return NewPoint()
		}
		return curve.Double(p1)
	}

	numerator := curve.modSub(p2.Y, p1.Y)
	denominator := curve.modSub(p2.X, p1.X)
	m = curve.modDiv(numerator, denominator)

	x3 := curve.modSub(curve.modSub(curve.modSquare(m), p1.X), p2.X)
	y3 := curve.modSub(curve.modMul(m, curve.modSub(p1.X, x3)), p1.Y)

	return &Point{X: x3, Y: y3}
}

func (curve *CurveParams) Double(p *Point) *Point {
	if p.IsZero() {
		return NewPoint()
	}

	threeX2 := curve.modMul(big.NewInt(3), curve.modSquare(p.X))
	numerator := curve.modAdd(threeX2, curve.A)
	denominator := curve.modMul(big.NewInt(2), p.Y)
	m := curve.modDiv(numerator, denominator)

	x3 := curve.modSub(curve.modSub(curve.modSquare(m), p.X), p.X)
	y3 := curve.modSub(curve.modMul(m, curve.modSub(p.X, x3)), p.Y)

	return &Point{X: x3, Y: y3}
}

func (curve *CurveParams) ScalarMult(k *big.Int, p *Point) *Point {
	result := NewPoint()
	current := p.Copy()
	kMod := new(big.Int).Mod(k, curve.N)

	for i := 0; i < kMod.BitLen(); i++ {
		if kMod.Bit(i) == 1 {
			result = curve.Add(result, current)
		}
		current = curve.Double(current)
	}

	return result
}

func (curve *CurveParams) MontgomeryLadder(k *big.Int, p *Point) *Point {
	kMod := new(big.Int).Mod(k, curve.N)

	isZeroK := uint(1)
	if kMod.Sign() != 0 {
		isZeroK = 0
	}

	isZeroP := uint(1)
	if !p.IsZero() {
		isZeroP = 0
	}

	R0 := NewPoint()
	R1 := p.Copy()

	for i := curve.BitSize - 1; i >= 0; i-- {
		bit := kMod.Bit(i)

		cswapPoint(R0, R1, bit^1)

		sum := curve.Add(R0, R1)
		doubleR0 := curve.Double(R0)

		R0.Set(doubleR0.X, doubleR0.Y)
		R1.Set(sum.X, sum.Y)

		cswapPoint(R0, R1, bit^1)
	}

	zeroPoint := NewPoint()
	cswapPoint(R0, zeroPoint, isZeroK|isZeroP)

	return R0
}

type ECDH struct {
	curve *CurveParams
}

func NewECDH(curve *CurveParams) *ECDH {
	return &ECDH{curve: curve}
}

func (e *ECDH) GenerateKeyPair() (*big.Int, *Point, error) {
	k, err := rand.Int(rand.Reader, e.curve.N)
	if err != nil {
		return nil, nil, err
	}

	k.Add(k, big.NewInt(1))

	G := &Point{X: e.curve.Gx, Y: e.curve.Gy}
	publicKey := e.curve.MontgomeryLadder(k, G)

	return k, publicKey, nil
}

func (e *ECDH) ComputeSharedSecret(privateKey *big.Int, peerPublicKey *Point) (*big.Int, error) {
	if !e.curve.IsOnCurve(peerPublicKey) {
		return nil, nil
	}

	sharedPoint := e.curve.MontgomeryLadder(privateKey, peerPublicKey)
	return sharedPoint.X, nil
}

type EdwardsPoint struct {
	X, Y, Z, T *big.Int
}

type EdwardsCurve struct {
	P       *big.Int
	D       *big.Int
	Gx      *big.Int
	Gy      *big.Int
	N       *big.Int
	BitSize int
}

func NewEdwardsPoint() *EdwardsPoint {
	return &EdwardsPoint{
		X: new(big.Int),
		Y: new(big.Int),
		Z: new(big.Int),
		T: new(big.Int),
	}
}

func (p *EdwardsPoint) Set(x, y, z, t *big.Int) *EdwardsPoint {
	p.X.Set(x)
	p.Y.Set(y)
	p.Z.Set(z)
	p.T.Set(t)
	return p
}

func (p *EdwardsPoint) Copy() *EdwardsPoint {
	return &EdwardsPoint{
		X: new(big.Int).Set(p.X),
		Y: new(big.Int).Set(p.Y),
		Z: new(big.Int).Set(p.Z),
		T: new(big.Int).Set(p.T),
	}
}

func Ed25519() *EdwardsCurve {
	p, _ := new(big.Int).SetString("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED", 16)
	d, _ := new(big.Int).SetString("52036CEE2B6FFE738CC740797779E89800700A4D4141D8AB75EB4DCA135978A3", 16)
	gx, _ := new(big.Int).SetString("216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A", 16)
	gy, _ := new(big.Int).SetString("6666666666666666666666666666666666666666666666666666666666666658", 16)
	n, _ := new(big.Int).SetString("1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED", 16)

	return &EdwardsCurve{
		P:       p,
		D:       d,
		Gx:      gx,
		Gy:      gy,
		N:       n,
		BitSize: 253,
	}
}

func (curve *EdwardsCurve) modAdd(a, b *big.Int) *big.Int {
	result := new(big.Int).Add(a, b)
	return result.Mod(result, curve.P)
}

func (curve *EdwardsCurve) modSub(a, b *big.Int) *big.Int {
	result := new(big.Int).Sub(a, b)
	return result.Mod(result, curve.P)
}

func (curve *EdwardsCurve) modMul(a, b *big.Int) *big.Int {
	result := new(big.Int).Mul(a, b)
	return result.Mod(result, curve.P)
}

func (curve *EdwardsCurve) modInv(a *big.Int) *big.Int {
	return new(big.Int).ModInverse(a, curve.P)
}

func (curve *EdwardsCurve) modNeg(a *big.Int) *big.Int {
	return new(big.Int).Neg(a).Mod(new(big.Int).Neg(a), curve.P)
}

func (curve *EdwardsCurve) AddEdwards(p1, p2 *EdwardsPoint) *EdwardsPoint {
	A := curve.modMul(p1.Y, p2.Y)
	B := curve.modMul(p1.X, p2.X)
	C := curve.modMul(p1.T, p2.T)
	C = curve.modMul(C, curve.D)
	D := curve.modMul(p1.Z, p2.Z)
	E := curve.modSub(
		curve.modMul(curve.modAdd(p1.X, p1.Y), curve.modAdd(p2.X, p2.Y)),
		curve.modAdd(A, B),
	)
	F := curve.modSub(D, C)
	G := curve.modAdd(D, C)
	H := curve.modSub(A, B)

	X3 := curve.modMul(E, F)
	Y3 := curve.modMul(G, H)
	T3 := curve.modMul(E, H)
	Z3 := curve.modMul(F, G)

	return &EdwardsPoint{X: X3, Y: Y3, Z: Z3, T: T3}
}

func (curve *EdwardsCurve) DoubleEdwards(p *EdwardsPoint) *EdwardsPoint {
	A := curve.modMul(p.X, p.X)
	B := curve.modMul(p.Y, p.Y)
	C := curve.modMul(big.NewInt(2), curve.modMul(p.Z, p.Z))
	D := curve.modAdd(A, B)
	E := curve.modSub(
		curve.modSub(D, curve.modMul(p.X, p.Y)),
		curve.modMul(p.X, p.Y),
	)
	G := curve.modAdd(A, B)
	H := curve.modSub(C, D)

	X3 := curve.modMul(E, H)
	Y3 := curve.modMul(G, E)
	T3 := curve.modMul(E, H)
	Z3 := curve.modMul(G, H)

	return &EdwardsPoint{X: X3, Y: Y3, Z: Z3, T: T3}
}

func (curve *EdwardsCurve) Negate(p *EdwardsPoint) *EdwardsPoint {
	return &EdwardsPoint{
		X: curve.modNeg(p.X),
		Y: new(big.Int).Set(p.Y),
		Z: new(big.Int).Set(p.Z),
		T: curve.modNeg(p.T),
	}
}

func cswapBigIntEdwards(a, b *big.Int, mask big.Word) {
	wordsA := a.Bits()
	wordsB := b.Bits()

	maxLen := len(wordsA)
	if len(wordsB) > maxLen {
		maxLen = len(wordsB)
	}

	extA := make([]big.Word, maxLen)
	extB := make([]big.Word, maxLen)
	copy(extA, wordsA)
	copy(extB, wordsB)

	for i := 0; i < maxLen; i++ {
		diff := mask & (extA[i] ^ extB[i])
		extA[i] ^= diff
		extB[i] ^= diff
	}

	a.SetBits(extA)
	b.SetBits(extB)
}

func (curve *EdwardsCurve) cswapEdwardsPoint(a, b *EdwardsPoint, bit uint) {
	mask := big.Word(bit)
	mask = -mask

	cswapBigIntEdwards(a.X, b.X, mask)
	cswapBigIntEdwards(a.Y, b.Y, mask)
	cswapBigIntEdwards(a.Z, b.Z, mask)
	cswapBigIntEdwards(a.T, b.T, mask)
}

func (curve *EdwardsCurve) ScalarMultEdwards(k *big.Int, p *EdwardsPoint) *EdwardsPoint {
	kMod := new(big.Int).Mod(k, curve.N)

	R0 := &EdwardsPoint{
		X: big.NewInt(0),
		Y: big.NewInt(1),
		Z: big.NewInt(1),
		T: big.NewInt(0),
	}
	R1 := p.Copy()

	for i := curve.BitSize - 1; i >= 0; i-- {
		bit := kMod.Bit(i)

		curve.cswapEdwardsPoint(R0, R1, bit^1)

		sum := curve.AddEdwards(R0, R1)
		doubleR0 := curve.DoubleEdwards(R0)

		R0.Set(doubleR0.X, doubleR0.Y, doubleR0.Z, doubleR0.T)
		R1.Set(sum.X, sum.Y, sum.Z, sum.T)

		curve.cswapEdwardsPoint(R0, R1, bit^1)
	}

	return R0
}

func (curve *EdwardsCurve) ToAffine(p *EdwardsPoint) *Point {
	zInv := curve.modInv(p.Z)
	x := curve.modMul(p.X, zInv)
	y := curve.modMul(p.Y, zInv)
	return &Point{X: x, Y: y}
}

func (curve *EdwardsCurve) Generator() *EdwardsPoint {
	return &EdwardsPoint{
		X: new(big.Int).Set(curve.Gx),
		Y: new(big.Int).Set(curve.Gy),
		Z: big.NewInt(1),
		T: curve.modMul(curve.Gx, curve.Gy),
	}
}

type MultiScalar struct {
	curveWeierstrass *CurveParams
	curveEdwards     *EdwardsCurve
}

func NewMultiScalarWeierstrass(curve *CurveParams) *MultiScalar {
	return &MultiScalar{curveWeierstrass: curve}
}

func NewMultiScalarEdwards(curve *EdwardsCurve) *MultiScalar {
	return &MultiScalar{curveEdwards: curve}
}

func (ms *MultiScalar) DualScalarMult(k, l *big.Int, P, Q *Point) *Point {
	curve := ms.curveWeierstrass

	kMod := new(big.Int).Mod(k, curve.N)
	lMod := new(big.Int).Mod(l, curve.N)

	result := NewPoint()

	maxBits := 256

	R0 := NewPoint()
	R1 := P.Copy()
	R2 := Q.Copy()
	R3 := curve.Add(P, Q)

	for i := maxBits - 1; i >= 0; i-- {
		result = curve.Double(result)

		kb := kMod.Bit(i)
		lb := lMod.Bit(i)

		idx := (kb << 1) | lb

		var addPoint *Point
		switch idx {
		case 1:
			addPoint = R2
		case 2:
			addPoint = R1
		case 3:
			addPoint = R3
		default:
			addPoint = R0
		}

		result = curve.Add(result, addPoint)
	}

	return result
}

func (ms *MultiScalar) DualScalarMultStraus(k, l *big.Int, P, Q *Point) *Point {
	curve := ms.curveWeierstrass

	kMod := new(big.Int).Mod(k, curve.N)
	lMod := new(big.Int).Mod(l, curve.N)

	precomputed := [4]*Point{
		NewPoint(),
		Q.Copy(),
		P.Copy(),
		curve.Add(P, Q),
	}

	result := NewPoint()

	for i := 255; i >= 0; i-- {
		result = curve.Double(result)

		kb := kMod.Bit(i)
		lb := lMod.Bit(i)
		idx := (kb << 1) | lb

		result = curve.Add(result, precomputed[idx])
	}

	return result
}

func (ms *MultiScalar) DualScalarMultEdwards(k, l *big.Int, P, Q *EdwardsPoint) *EdwardsPoint {
	curve := ms.curveEdwards

	kMod := new(big.Int).Mod(k, curve.N)
	lMod := new(big.Int).Mod(l, curve.N)

	precomputed := [4]*EdwardsPoint{
		{X: big.NewInt(0), Y: big.NewInt(1), Z: big.NewInt(1), T: big.NewInt(0)},
		Q.Copy(),
		P.Copy(),
		curve.AddEdwards(P, Q),
	}

	result := &EdwardsPoint{
		X: big.NewInt(0),
		Y: big.NewInt(1),
		Z: big.NewInt(1),
		T: big.NewInt(0),
	}

	for i := curve.BitSize - 1; i >= 0; i-- {
		result = curve.DoubleEdwards(result)

		kb := kMod.Bit(i)
		lb := lMod.Bit(i)
		idx := (kb << 1) | lb

		result = curve.AddEdwards(result, precomputed[idx])
	}

	return result
}

func (curve *CurveParams) MultiScalarMult(scalars []*big.Int, points []*Point) *Point {
	n := len(scalars)
	if n != len(points) || n == 0 {
		return NewPoint()
	}

	windowSize := 4
	numWindows := (256 + windowSize - 1) / windowSize

	precomputed := make([][]*Point, n)
	for i := 0; i < n; i++ {
		precomputed[i] = make([]*Point, 1<<windowSize)
		precomputed[i][0] = NewPoint()
		for j := 1; j < 1<<windowSize; j++ {
			precomputed[i][j] = curve.Add(precomputed[i][j-1], points[i])
		}
	}

	result := NewPoint()

	for w := numWindows - 1; w >= 0; w-- {
		if w != numWindows-1 {
			for d := 0; d < windowSize; d++ {
				result = curve.Double(result)
			}
		}

		for i := 0; i < n; i++ {
			offset := w * windowSize
			idx := uint(0)
			for b := 0; b < windowSize && offset+b < 256; b++ {
				idx |= scalars[i].Bit(offset+b) << b
			}
			if idx > 0 {
				result = curve.Add(result, precomputed[i][idx])
			}
		}
	}

	return result
}

type BlindSignature struct {
	curve *CurveParams
}

func NewBlindSignature(curve *CurveParams) *BlindSignature {
	return &BlindSignature{curve: curve}
}

func (bs *BlindSignature) BlindingFactor() (*big.Int, *Point, error) {
	r, err := rand.Int(rand.Reader, bs.curve.N)
	if err != nil {
		return nil, nil, err
	}

	G := &Point{X: bs.curve.Gx, Y: bs.curve.Gy}
	R := bs.curve.MontgomeryLadder(r, G)

	return r, R, nil
}

func (bs *BlindSignature) BlindMessage(m *big.Int, blindingFactor *big.Int, R, pubKey *Point) *big.Int {
	G := &Point{X: bs.curve.Gx, Y: bs.curve.Gy}
	invBlinding := new(big.Int).ModInverse(blindingFactor, bs.curve.N)

	blindedM := new(big.Int).Mul(m, invBlinding)
	blindedM.Mod(blindedM, bs.curve.N)

	return blindedM
}

func (bs *BlindSignature) SignBlinded(blindedM *big.Int, privKey *big.Int) *big.Int {
	s := new(big.Int).Mul(privKey, blindedM)
	s.Mod(s, bs.curve.N)
	return s
}

func (bs *BlindSignature) UnblindSignature(blindedSig *big.Int, blindingFactor *big.Int) *big.Int {
	s := new(big.Int).Mul(blindedSig, blindingFactor)
	s.Mod(s, bs.curve.N)
	return s
}

func (bs *BlindSignature) Verify(s *big.Int, m *big.Int, pubKey *Point) bool {
	G := &Point{X: bs.curve.Gx, Y: bs.curve.Gy}

	sG := bs.curve.MontgomeryLadder(s, G)
	mP := bs.curve.MontgomeryLadder(m, pubKey)

	return sG.X.Cmp(mP.X) == 0 && sG.Y.Cmp(mP.Y) == 0
}

type AggregateSignature struct {
	curve *CurveParams
}

func NewAggregateSignature(curve *CurveParams) *AggregateSignature {
	return &AggregateSignature{curve: curve}
}

func (ag *AggregateSignature) AggregatePublicKeys(pubKeys []*Point) *Point {
	result := NewPoint()
	for _, pubKey := range pubKeys {
		result = ag.curve.Add(result, pubKey)
	}
	return result
}

func (ag *AggregateSignature) AggregateSignatures(signatures []*big.Int) *big.Int {
	result := big.NewInt(0)
	for _, sig := range signatures {
		result.Add(result, sig)
		result.Mod(result, ag.curve.N)
	}
	return result
}

func (ag *AggregateSignature) VerifyAggregate(aggSig *big.Int, message *big.Int, aggPubKey *Point) bool {
	G := &Point{X: ag.curve.Gx, Y: ag.curve.Gy}

	sG := ag.curve.MontgomeryLadder(aggSig, G)
	mX := ag.curve.MontgomeryLadder(message, aggPubKey)

	return sG.X.Cmp(mX.X) == 0
}

