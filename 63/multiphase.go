package main

import (
	"fmt"
	"math"
	"math/rand"
)

type LBMShanChen struct {
	NX, NY int
	Tau    float64
	Omega  float64
	G      float64
	Rho0   float64
	F      [][][]float64
	Ftemp  [][][]float64
	Rho    [][]float64
	U      [][]float64
	V      [][]float64
	ForceX [][]float64
	ForceY [][]float64
}

var Wsc = [9]float64{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
	1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0}
var Cxsc = [9]int{0, 1, 0, -1, 0, 1, -1, -1, 1}
var Cysc = [9]int{0, 0, 1, 0, -1, 1, 1, -1, -1}

func NewLBMShanChen(nx, ny int, tau, g, rho0 float64) *LBMShanChen {
	lbm := &LBMShanChen{
		NX:   nx,
		NY:   ny,
		Tau:  tau,
		Omega: 1.0 / tau,
		G:    g,
		Rho0: rho0,
	}

	lbm.F = make([][][]float64, nx)
	lbm.Ftemp = make([][][]float64, nx)
	lbm.Rho = make([][]float64, nx)
	lbm.U = make([][]float64, nx)
	lbm.V = make([][]float64, nx)
	lbm.ForceX = make([][]float64, nx)
	lbm.ForceY = make([][]float64, nx)

	for i := range lbm.F {
		lbm.F[i] = make([][]float64, ny)
		lbm.Ftemp[i] = make([][]float64, ny)
		lbm.Rho[i] = make([]float64, ny)
		lbm.U[i] = make([]float64, ny)
		lbm.V[i] = make([]float64, ny)
		lbm.ForceX[i] = make([]float64, ny)
		lbm.ForceY[i] = make([]float64, ny)
		for j := range lbm.F[i] {
			lbm.F[i][j] = make([]float64, 9)
			lbm.Ftemp[i][j] = make([]float64, 9)
		}
	}

	return lbm
}

func (lbm *LBMShanChen) Psi(rho float64) float64 {
	return lbm.Rho0 * (1.0 - math.Exp(-rho/lbm.Rho0))
}

func (lbm *LBMShanChen) ComputeMacroscopic() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			rho := 0.0
			u := 0.0
			v := 0.0
			for k := 0; k < 9; k++ {
				f := lbm.F[i][j][k]
				rho += f
				u += f * float64(Cxsc[k])
				v += f * float64(Cysc[k])
			}
			lbm.Rho[i][j] = rho
			lbm.U[i][j] = u / rho
			lbm.V[i][j] = v / rho
		}
	}
}

func (lbm *LBMShanChen) ComputeInteractionForce() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			psi := lbm.Psi(lbm.Rho[i][j])
			fx := 0.0
			fy := 0.0
			for k := 1; k < 9; k++ {
				ip := (i + Cxsc[k] + lbm.NX) % lbm.NX
				jp := (j + Cysc[k] + lbm.NY) % lbm.NY
				psineigh := lbm.Psi(lbm.Rho[ip][jp])
				fx += Wsc[k] * psineigh * float64(Cxsc[k])
				fy += Wsc[k] * psineigh * float64(Cysc[k])
			}
			lbm.ForceX[i][j] = -lbm.G * psi * fx
			lbm.ForceY[i][j] = -lbm.G * psi * fy
		}
	}
}

func (lbm *LBMShanChen) ComputeFeq(i, j int, feq []float64) {
	rho := lbm.Rho[i][j]
	u := lbm.U[i][j] + 0.5*lbm.Tau*lbm.ForceX[i][j]/rho
	v := lbm.V[i][j] + 0.5*lbm.Tau*lbm.ForceY[i][j]/rho
	usq := u*u + v*v

	for k := 0; k < 9; k++ {
		cu := float64(Cxsc[k])*u + float64(Cysc[k])*v
		feq[k] = Wsc[k] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)
	}
}

func (lbm *LBMShanChen) Collision() {
	var feq [9]float64
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			rho := lbm.Rho[i][j]
			lbm.ComputeFeq(i, j, feq[:])
			
			for k := 0; k < 9; k++ {
				forceTerm := Wsc[k] * 3.0 * (
					(1.0 - 0.5*lbm.Omega) * (
						(float64(Cxsc[k]) - lbm.U[i][j]) * lbm.ForceX[i][j] / rho +
							(float64(Cysc[k]) - lbm.V[i][j]) * lbm.ForceY[i][j] / rho) +
						3.0*(float64(Cxsc[k])*lbm.U[i][j] + float64(Cysc[k])*lbm.V[i][j]) *
							(float64(Cxsc[k])*lbm.ForceX[i][j]/rho + float64(Cysc[k])*lbm.ForceY[i][j]/rho))
				
				lbm.F[i][j][k] = lbm.F[i][j][k]*(1.0-lbm.Omega) + feq[k]*lbm.Omega + forceTerm
			}
		}
	}
}

func (lbm *LBMShanChen) Streaming() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			for k := 0; k < 9; k++ {
				ip := (i - Cxsc[k] + lbm.NX) % lbm.NX
				jp := (j - Cysc[k] + lbm.NY) % lbm.NY
				lbm.Ftemp[i][j][k] = lbm.F[ip][jp][k]
			}
		}
	}

	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			copy(lbm.F[i][j], lbm.Ftemp[i][j])
		}
	}
}

func (lbm *LBMShanChen) Step() {
	lbm.ComputeMacroscopic()
	lbm.ComputeInteractionForce()
	lbm.Collision()
	lbm.Streaming()
}

func (lbm *LBMShanChen) InitializeRandom(seed int64) {
	r := rand.New(rand.NewSource(seed))
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			noise := 0.05 * (r.Float64() - 0.5)
			lbm.Rho[i][j] = lbm.Rho0 * (1.0 + noise)
			lbm.U[i][j] = 0.0
			lbm.V[i][j] = 0.0
			var feq [9]float64
			lbm.ComputeFeq(i, j, feq[:])
			copy(lbm.F[i][j], feq[:])
		}
	}
}

func (lbm *LBMShanChen) InitializeTwoLayers() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			if j < lbm.NY/2 {
				lbm.Rho[i][j] = lbm.Rho0 * 2.0
			} else {
				lbm.Rho[i][j] = lbm.Rho0 * 0.5
			}
			lbm.U[i][j] = 0.0
			lbm.V[i][j] = 0.0
			var feq [9]float64
			lbm.ComputeFeq(i, j, feq[:])
			copy(lbm.F[i][j], feq[:])
		}
	}
}

func (lbm *LBMShanChen) InitializeDroplet(cx, cy, radius int) {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			dx := i - cx
			dy := j - cy
			dist := math.Sqrt(float64(dx*dx + dy*dy))
			if dist < float64(radius) {
				lbm.Rho[i][j] = lbm.Rho0 * 2.0
			} else {
				lbm.Rho[i][j] = lbm.Rho0 * 0.5
			}
			lbm.U[i][j] = 0.0
			lbm.V[i][j] = 0.0
			var feq [9]float64
			lbm.ComputeFeq(i, j, feq[:])
			copy(lbm.F[i][j], feq[:])
		}
	}
}

func (lbm *LBMShanChen) ComputeTotalMass() float64 {
	totalMass := 0.0
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			totalMass += lbm.Rho[i][j]
		}
	}
	return totalMass
}

func (lbm *LBMShanChen) ComputeDensityStats() (float64, float64, float64) {
	minRho := math.Inf(1)
	maxRho := math.Inf(-1)
	meanRho := 0.0
	count := 0
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			rho := lbm.Rho[i][j]
			if rho < minRho {
				minRho = rho
			}
			if rho > maxRho {
				maxRho = rho
			}
			meanRho += rho
			count++
		}
	}
	return minRho, maxRho, meanRho / float64(count)
}

func (lbm *LBMShanChen) PrintDensityProfile() {
	fmt.Println("\nDensity profile at center line:")
	centerX := lbm.NX / 2
	for j := 0; j < lbm.NY; j += lbm.NY / 10 {
		fmt.Printf("y = %d, rho = %.6f\n", j, lbm.Rho[centerX][j])
	}
}

func (lbm *LBMShanChen) Run(steps int) {
	initialMass := lbm.ComputeTotalMass()
	fmt.Printf("Initial total mass: %.6f\n", initialMass)

	for step := 0; step < steps; step++ {
		lbm.Step()

		if step%1000 == 0 {
			currentMass := lbm.ComputeTotalMass()
			massError := (currentMass - initialMass) / initialMass
			minRho, maxRho, meanRho := lbm.ComputeDensityStats()
			fmt.Printf("Step %d: Mass error: %.3e%%, rho=[%.3f, %.3f], mean=%.3f\n",
				step, massError*100, minRho, maxRho, meanRho)
		}
	}

	finalMass := lbm.ComputeTotalMass()
	fmt.Printf("\nFinal total mass: %.6f\n", finalMass)
	fmt.Printf("Total mass error: %.3e%%\n", (finalMass-initialMass)/initialMass*100)
}

func mainMultiphase() {
	NX := 100
	NY := 100
	Tau := 1.0
	G := -5.0
	Rho0 := 200.0
	Steps := 20000

	fmt.Printf("Shan-Chen Pseudopotential Multiphase LBM\n")
	fmt.Printf("Two-Phase Separation Simulation\n")
	fmt.Printf("Grid: %d x %d\n", NX, NY)
	fmt.Printf("Tau: %.2f, G: %.2f, Rho0: %.1f\n", Tau, G, Rho0)
	fmt.Printf("Expected steps: %d\n\n", Steps)

	lbm := NewLBMShanChen(NX, NY, Tau, G, Rho0)
	lbm.InitializeRandom(42)

	fmt.Println("Starting simulation...")
	lbm.Run(Steps)

	lbm.PrintDensityProfile()

	minRho, maxRho, meanRho := lbm.ComputeDensityStats()
	fmt.Printf("\nFinal density statistics:\n")
	fmt.Printf("  Minimum: %.3f\n", minRho)
	fmt.Printf("  Maximum: %.3f\n", maxRho)
	fmt.Printf("  Mean:    %.3f\n", meanRho)
	fmt.Printf("  Density ratio: %.3f\n", maxRho/minRho)
}
