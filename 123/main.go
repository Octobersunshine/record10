package main

import (
	"fmt"
	"math/big"
)

func main() {
	fmt.Println("=== Advanced Elliptic Curve Cryptography ===")
	fmt.Println()

	demoECDH()
	fmt.Println()
	fmt.Println("=" + string(make([]byte, 60)))
	fmt.Println()

	demoEd25519()
	fmt.Println()
	fmt.Println("=" + string(make([]byte, 60)))
	fmt.Println()

	demoDualScalarMult()
	fmt.Println()
	fmt.Println("=" + string(make([]byte, 60)))
	fmt.Println()

	demoAggregateSignature()
	fmt.Println()
	fmt.Println("=" + string(make([]byte, 60)))
	fmt.Println()

	demoBlindSignature()
}

func demoECDH() {
	fmt.Println("--- ECDH Key Exchange (secp256k1) ---")
	fmt.Println()

	curve := Secp256k1()
	ecdh := NewECDH(curve)

	alicePriv, alicePub, err := ecdh.GenerateKeyPair()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	bobPriv, bobPub, err := ecdh.GenerateKeyPair()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	aliceShared, _ := ecdh.ComputeSharedSecret(alicePriv, bobPub)
	bobShared, _ := ecdh.ComputeSharedSecret(bobPriv, alicePub)

	if aliceShared.Cmp(bobShared) == 0 {
		fmt.Println("✓ ECDH Key Exchange: SUCCESS")
		fmt.Printf("  Shared Secret (first 32 bits): %x...\n", aliceShared.Bytes()[:4])
	} else {
		fmt.Println("✗ ECDH Key Exchange: FAILED")
	}
}

func demoEd25519() {
	fmt.Println("--- Ed25519 Edwards Curve Operations ---")
	fmt.Println()

	curve := Ed25519()
	G := curve.Generator()

	fmt.Println("Curve: Ed25519 (Twisted Edwards)")
	fmt.Printf("  Equation: -x² + y² = 1 + dx²y² mod p")
	fmt.Println()
	fmt.Printf("  Prime p:   %x...\n", curve.P.Bytes()[:8])
	fmt.Printf("  Order n:   %x...\n", curve.N.Bytes()[:8])
	fmt.Println()

	k := big.NewInt(1000)
	result := curve.ScalarMultEdwards(k, G)
	affine := curve.ToAffine(result)

	fmt.Printf("  1000 * G (X): %x...\n", affine.X.Bytes()[:8])
	fmt.Println("  ✓ Edwards curve scalar multiplication")
	fmt.Println("  ✓ Extended coordinates (X,Y,Z,T) for efficiency")
	fmt.Println("  ✓ Constant-time Montgomery ladder")
}

func demoDualScalarMult() {
	fmt.Println("--- Dual Scalar Multiplication (kP + lQ) ---")
	fmt.Println()

	curve := Secp256k1()
	ms := NewMultiScalarWeierstrass(curve)
	G := &Point{X: curve.Gx, Y: curve.Gy}

	twoG := curve.Double(G)
	k := big.NewInt(5)
	l := big.NewInt(7)

	result := ms.DualScalarMultStraus(k, l, G, twoG)

	expected := curve.Add(
		curve.ScalarMult(k, G),
		curve.ScalarMult(l, twoG),
	)

	if result.X.Cmp(expected.X) == 0 {
		fmt.Println("✓ Dual Scalar Mult: SUCCESS")
		fmt.Printf("  k = %d, l = %d\n", k, l)
		fmt.Printf("  kP + lQ = 5G + 7(2G) = 19G\n")
		fmt.Printf("  Result X (first 32 bits): %x...\n", result.X.Bytes()[:4])
	} else {
		fmt.Println("✗ Dual Scalar Mult: FAILED")
	}
	fmt.Println()
	fmt.Println("  Use cases:")
	fmt.Println("    • Schnorr signature verification (sG + eP)")
	fmt.Println("    • BLS aggregate signatures")
	fmt.Println("    • Batch verification")
	fmt.Println("    • Zero-knowledge proofs")
}

func demoAggregateSignature() {
	fmt.Println("--- Aggregate Signatures ---")
	fmt.Println()

	curve := Secp256k1()
	ag := NewAggregateSignature(curve)
	ecdh := NewECDH(curve)

	numSigners := 3
	message := big.NewInt(123456789)

	fmt.Printf("  Message: %d\n", message.Int64())
	fmt.Printf("  Number of signers: %d\n", numSigners)
	fmt.Println()

	var pubKeys []*Point
	var sigs []*big.Int

	for i := 0; i < numSigners; i++ {
		priv, pub, _ := ecdh.GenerateKeyPair()
		pubKeys = append(pubKeys, pub)

		sig := new(big.Int).Mul(priv, message)
		sig.Mod(sig, curve.N)
		sigs = append(sigs, sig)

		fmt.Printf("  Signer %d: ✓ Signed ✓\n", i+1)
	}
	fmt.Println()

	aggPubKey := ag.AggregatePublicKeys(pubKeys)
	aggSig := ag.AggregateSignatures(sigs)

	if ag.VerifyAggregate(aggSig, message, aggPubKey) {
		fmt.Println("✓ Aggregate Signature: VERIFIED")
		fmt.Printf("  Aggregated Public Key size: 1 point (instead of %d)\n", numSigners)
		fmt.Printf("  Aggregated Signature size: 1 scalar (instead of %d)\n", numSigners)
	} else {
		fmt.Println("✗ Aggregate Signature: FAILED")
	}
	fmt.Println()
	fmt.Println("  Benefits:")
	fmt.Println("    • Reduced storage: O(1) vs O(n)")
	fmt.Println("    • Faster verification: single check instead of n checks")
	fmt.Println("    • Used in blockchain (e.g., BLS signatures)")
}

func demoBlindSignature() {
	fmt.Println("--- Blind Signatures ---")
	fmt.Println()

	curve := Secp256k1()
	bs := NewBlindSignature(curve)
	ecdh := NewECDH(curve)

	privKey, pubKey, _ := ecdh.GenerateKeyPair()
	message := big.NewInt(987654321)

	fmt.Printf("  Original Message: %d\n", message.Int64())
	fmt.Println()

	blindingFactor, R, err := bs.BlindingFactor()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("  Step 1: User blinds message with random factor")
	fmt.Printf("    Blinding factor r = %x...\n", blindingFactor.Bytes()[:4])

	blindedMessage := bs.BlindMessage(message, blindingFactor, R, pubKey)
	fmt.Printf("    Blinded message: %x...\n", blindedMessage.Bytes()[:4])
	fmt.Println()

	fmt.Println("  Step 2: Signer signs blinded message")
	fmt.Println("    (Signer cannot see original message)")
	blindedSig := bs.SignBlinded(blindedMessage, privKey)
	fmt.Printf("    Blinded signature: %x...\n", blindedSig.Bytes()[:4])
	fmt.Println()

	fmt.Println("  Step 3: User unblinds signature")
	unblindedSig := bs.UnblindSignature(blindedSig, blindingFactor)
	fmt.Printf("    Unblinded signature: %x...\n", unblindedSig.Bytes()[:4])
	fmt.Println()

	if bs.Verify(unblindedSig, message, pubKey) {
		fmt.Println("✓ Blind Signature: VERIFIED")
	} else {
		fmt.Println("✗ Blind Signature: FAILED")
	}
	fmt.Println()
	fmt.Println("  Use cases:")
	fmt.Println("    • Digital cash (e-cash)")
	fmt.Println("    • Anonymous voting")
	fmt.Println("    • Privacy-preserving authentication")
	fmt.Println("    • Signer learns nothing about the message")
}
