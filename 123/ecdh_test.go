package main

import (
	"math/big"
	"testing"
)

func TestPointAddSamePoint(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	result := curve.Add(G, G)
	doubleG := curve.Double(G)

	if result.X.Cmp(doubleG.X) != 0 || result.Y.Cmp(doubleG.Y) != 0 {
		t.Error("Add(G, G) should equal Double(G)")
	}
}

func TestPointAddCommutative(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	twoG := curve.Double(G)
	threeG := curve.Add(twoG, G)
	threeG2 := curve.Add(G, twoG)

	if threeG.X.Cmp(threeG2.X) != 0 || threeG.Y.Cmp(threeG2.Y) != 0 {
		t.Error("Point addition should be commutative")
	}
}

func TestPointAddAssociative(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	twoG := curve.Double(G)
	threeG := curve.Add(twoG, G)
	fourG := curve.Add(threeG, G)

	fourG2 := curve.Add(twoG, twoG)

	if fourG.X.Cmp(fourG2.X) != 0 || fourG.Y.Cmp(fourG2.Y) != 0 {
		t.Error("Point addition should be associative")
	}
}

func TestScalarMultSmallValues(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	k1 := big.NewInt(1)
	result1 := curve.ScalarMult(k1, G)
	if result1.X.Cmp(G.X) != 0 || result1.Y.Cmp(G.Y) != 0 {
		t.Error("1*G should equal G")
	}

	k2 := big.NewInt(2)
	result2 := curve.ScalarMult(k2, G)
	doubleG := curve.Double(G)
	if result2.X.Cmp(doubleG.X) != 0 || result2.Y.Cmp(doubleG.Y) != 0 {
		t.Error("2*G should equal 2G")
	}
}

func TestMontgomeryLadderMatchesScalarMult(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	testValues := []int64{1, 2, 3, 10, 100, 12345, 999999}

	for _, k := range testValues {
		kBig := big.NewInt(k)
		resultLadder := curve.MontgomeryLadder(kBig, G)
		resultStandard := curve.ScalarMult(kBig, G)

		if resultLadder.X.Cmp(resultStandard.X) != 0 || resultLadder.Y.Cmp(resultStandard.Y) != 0 {
			t.Errorf("MontgomeryLadder(%d, G) != ScalarMult(%d, G)", k, k)
		}
	}
}

func TestMontgomeryLadderZero(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	k0 := big.NewInt(0)
	result := curve.MontgomeryLadder(k0, G)
	if !result.IsZero() {
		t.Error("0*G should be point at infinity")
	}
}

func TestMontgomeryLadderOrder(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	result := curve.MontgomeryLadder(curve.N, G)
	if !result.IsZero() {
		t.Error("n*G should be point at infinity (order of curve)")
	}
}

func TestECDHKeyExchange(t *testing.T) {
	curve := Secp256k1()
	ecdh := NewECDH(curve)

	alicePriv, alicePub, err := ecdh.GenerateKeyPair()
	if err != nil {
		t.Fatalf("Failed to generate Alice's keys: %v", err)
	}

	if !curve.IsOnCurve(alicePub) {
		t.Error("Alice's public key should be on curve")
	}

	bobPriv, bobPub, err := ecdh.GenerateKeyPair()
	if err != nil {
		t.Fatalf("Failed to generate Bob's keys: %v", err)
	}

	if !curve.IsOnCurve(bobPub) {
		t.Error("Bob's public key should be on curve")
	}

	aliceShared, err := ecdh.ComputeSharedSecret(alicePriv, bobPub)
	if err != nil {
		t.Fatalf("Failed to compute Alice's shared secret: %v", err)
	}

	bobShared, err := ecdh.ComputeSharedSecret(bobPriv, alicePub)
	if err != nil {
		t.Fatalf("Failed to compute Bob's shared secret: %v", err)
	}

	if aliceShared.Cmp(bobShared) != 0 {
		t.Error("Shared secrets should match")
	}
}

func TestIsOnCurve(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	if !curve.IsOnCurve(G) {
		t.Error("Generator point G should be on curve")
	}

	twoG := curve.Double(G)
	if !curve.IsOnCurve(twoG) {
		t.Error("2G should be on curve")
	}

	invalidPoint := &Point{X: big.NewInt(1), Y: big.NewInt(2)}
	if curve.IsOnCurve(invalidPoint) {
		t.Error("Invalid point should not be on curve")
	}
}

func TestZeroPointOperations(t *testing.T) {
	curve := Secp256k1()
	zero := NewPoint()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	result := curve.Add(zero, G)
	if result.X.Cmp(G.X) != 0 || result.Y.Cmp(G.Y) != 0 {
		t.Error("0 + G should equal G")
	}

	result = curve.Add(G, zero)
	if result.X.Cmp(G.X) != 0 || result.Y.Cmp(G.Y) != 0 {
		t.Error("G + 0 should equal G")
	}

	result = curve.Double(zero)
	if !result.IsZero() {
		t.Error("Double(0) should equal 0")
	}
}

func TestCswapBigInt(t *testing.T) {
	a := big.NewInt(12345)
	b := big.NewInt(67890)

	origA := new(big.Int).Set(a)
	origB := new(big.Int).Set(b)

	mask0 := big.Word(0)
	cswapBigInt(a, b, mask0)
	if a.Cmp(origA) != 0 || b.Cmp(origB) != 0 {
		t.Error("cswap with mask 0 should not swap")
	}

	mask1 := big.Word(1)
	mask1 = -mask1
	cswapBigInt(a, b, mask1)
	if a.Cmp(origB) != 0 || b.Cmp(origA) != 0 {
		t.Error("cswap with mask -1 should swap")
	}
}

func TestCswapPoint(t *testing.T) {
	curve := Secp256k1()
	p1 := &Point{X: big.NewInt(1), Y: big.NewInt(2)}
	p2 := &Point{X: curve.Gx, Y: curve.Gy}

	origP1 := p1.Copy()
	origP2 := p2.Copy()

	cswapPoint(p1, p2, 0)
	if p1.X.Cmp(origP1.X) != 0 || p1.Y.Cmp(origP1.Y) != 0 {
		t.Error("cswapPoint with bit 0 should not swap p1")
	}
	if p2.X.Cmp(origP2.X) != 0 || p2.Y.Cmp(origP2.Y) != 0 {
		t.Error("cswapPoint with bit 0 should not swap p2")
	}

	cswapPoint(p1, p2, 1)
	if p1.X.Cmp(origP2.X) != 0 || p1.Y.Cmp(origP2.Y) != 0 {
		t.Error("cswapPoint with bit 1 should swap p1")
	}
	if p2.X.Cmp(origP1.X) != 0 || p2.Y.Cmp(origP1.Y) != 0 {
		t.Error("cswapPoint with bit 1 should swap p2")
	}
}

func TestMontgomeryLadderConstantTime(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	testK := big.NewInt(42)
	resultCT := curve.MontgomeryLadder(testK, G)
	resultStandard := curve.ScalarMult(testK, G)

	if resultCT.X.Cmp(resultStandard.X) != 0 || resultCT.Y.Cmp(resultStandard.Y) != 0 {
		t.Error("Constant-time Montgomery Ladder result mismatch")
	}
}

func TestMontgomeryLadderFullBitSize(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}

	for k := int64(1); k <= 100; k++ {
		kBig := big.NewInt(k)
		resultCT := curve.MontgomeryLadder(kBig, G)
		resultStandard := curve.ScalarMult(kBig, G)

		if resultCT.X.Cmp(resultStandard.X) != 0 || resultCT.Y.Cmp(resultStandard.Y) != 0 {
			t.Errorf("Montgomery Ladder mismatch for k=%d", k)
		}
	}
}

func TestEdwardsPointAdd(t *testing.T) {
	curve := Ed25519()
	G := curve.Generator()

	twoG := curve.DoubleEdwards(G)
	threeG := curve.AddEdwards(twoG, G)
	threeG2 := curve.AddEdwards(G, twoG)

	affine1 := curve.ToAffine(threeG)
	affine2 := curve.ToAffine(threeG2)

	if affine1.X.Cmp(affine2.X) != 0 || affine1.Y.Cmp(affine2.Y) != 0 {
		t.Error("Edwards point addition should be commutative")
	}
}

func TestEdwardsScalarMult(t *testing.T) {
	curve := Ed25519()
	G := curve.Generator()

	k := big.NewInt(42)
	result := curve.ScalarMultEdwards(k, G)
	affine := curve.ToAffine(result)

	twoG := curve.DoubleEdwards(G)
	manual := curve.DoubleEdwards(twoG)
	for i := 0; i < 19; i++ {
		manual = curve.AddEdwards(manual, twoG)
	}
	manualAffine := curve.ToAffine(manual)

	if affine.X.Cmp(manualAffine.X) != 0 || affine.Y.Cmp(manualAffine.Y) != 0 {
		t.Error("Edwards scalar multiplication mismatch")
	}
}

func TestDualScalarMult(t *testing.T) {
	curve := Secp256k1()
	ms := NewMultiScalarWeierstrass(curve)
	G := &Point{X: curve.Gx, Y: curve.Gy}

	twoG := curve.Double(G)
	k := big.NewInt(3)
	l := big.NewInt(5)

	result := ms.DualScalarMult(k, l, G, twoG)

	kG := curve.ScalarMult(k, G)
	lQ := curve.ScalarMult(l, twoG)
	expected := curve.Add(kG, lQ)

	if result.X.Cmp(expected.X) != 0 || result.Y.Cmp(expected.Y) != 0 {
		t.Error("Dual scalar multiplication mismatch")
	}
}

func TestDualScalarMultStraus(t *testing.T) {
	curve := Secp256k1()
	ms := NewMultiScalarWeierstrass(curve)
	G := &Point{X: curve.Gx, Y: curve.Gy}

	twoG := curve.Double(G)
	k := big.NewInt(7)
	l := big.NewInt(11)

	result1 := ms.DualScalarMult(k, l, G, twoG)
	result2 := ms.DualScalarMultStraus(k, l, G, twoG)

	if result1.X.Cmp(result2.X) != 0 || result1.Y.Cmp(result2.Y) != 0 {
		t.Error("Dual scalar multiplication methods should agree")
	}
}

func TestMultiScalarMult(t *testing.T) {
	curve := Secp256k1()
	G := &Point{X: curve.Gx, Y: curve.Gy}
	twoG := curve.Double(G)
	threeG := curve.Add(twoG, G)

	scalars := []*big.Int{big.NewInt(2), big.NewInt(3), big.NewInt(4)}
	points := []*Point{G, twoG, threeG}

	result := curve.MultiScalarMult(scalars, points)

	expected := NewPoint()
	expected = curve.Add(expected, curve.ScalarMult(scalars[0], points[0]))
	expected = curve.Add(expected, curve.ScalarMult(scalars[1], points[1]))
	expected = curve.Add(expected, curve.ScalarMult(scalars[2], points[2]))

	if result.X.Cmp(expected.X) != 0 || result.Y.Cmp(expected.Y) != 0 {
		t.Error("Multi scalar multiplication mismatch")
	}
}

func TestAggregateSignature(t *testing.T) {
	curve := Secp256k1()
	ag := NewAggregateSignature(curve)
	ecdh := NewECDH(curve)

	priv1, pub1, _ := ecdh.GenerateKeyPair()
	priv2, pub2, _ := ecdh.GenerateKeyPair()
	priv3, pub3, _ := ecdh.GenerateKeyPair()

	message := big.NewInt(12345)

	sig1 := new(big.Int).Mul(priv1, message)
	sig1.Mod(sig1, curve.N)
	sig2 := new(big.Int).Mul(priv2, message)
	sig2.Mod(sig2, curve.N)
	sig3 := new(big.Int).Mul(priv3, message)
	sig3.Mod(sig3, curve.N)

	aggPubKey := ag.AggregatePublicKeys([]*Point{pub1, pub2, pub3})
	aggSig := ag.AggregateSignatures([]*big.Int{sig1, sig2, sig3})

	if !ag.VerifyAggregate(aggSig, message, aggPubKey) {
		t.Error("Aggregate signature verification failed")
	}
}

func TestBlindSignature(t *testing.T) {
	curve := Secp256k1()
	bs := NewBlindSignature(curve)
	ecdh := NewECDH(curve)

	privKey, pubKey, _ := ecdh.GenerateKeyPair()

	message := big.NewInt(67890)

	blindingFactor, R, err := bs.BlindingFactor()
	if err != nil {
		t.Fatalf("Failed to generate blinding factor: %v", err)
	}

	blindedMessage := bs.BlindMessage(message, blindingFactor, R, pubKey)
	blindedSig := bs.SignBlinded(blindedMessage, privKey)
	unblindedSig := bs.UnblindSignature(blindedSig, blindingFactor)

	if !bs.Verify(unblindedSig, message, pubKey) {
		t.Error("Blind signature verification failed")
	}
}
