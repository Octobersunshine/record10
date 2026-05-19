package main

import (
	"fmt"
	"math"
)

type LBMPoiseuille struct {
	NX, NY int
	Tau    float64
	Omega  float64
	Rho0   float64
	F      [][][]float64
	Ftemp  [][][]float64
	Rho    [][]float64
	U      [][]float64
	V      [][]float64
}

var Wp = [9]float64{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
	1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0}
var Cxp = [9]int{0, 1, 0, -1, 0, 1, -1, -1, 1}
var Cyp = [9]int{0, 0, 1, 0, -1, 1, 1, -1, -1}

func NewLBMPoiseuille(nx, ny int, tau, rho0 float64) *LBMPoiseuille {
	lbm := &LBMPoiseuille{
		NX:    nx,
		NY:    ny,
		Tau:   tau,
		Omega: 1.0 / tau,
		Rho0:  rho0,
	}

	lbm.F = make([][][]float64, nx)
	lbm.Ftemp = make([][][]float64, nx)
	lbm.Rho = make([][]float64, nx)
	lbm.U = make([][]float64, nx)
	lbm.V = make([][]float64, nx)

	for i := range lbm.F {
		lbm.F[i] = make([][]float64, ny)
		lbm.Ftemp[i] = make([][]float64, ny)
		lbm.Rho[i] = make([]float64, ny)
		lbm.U[i] = make([]float64, ny)
		lbm.V[i] = make([]float64, ny)
		for j := range lbm.F[i] {
			lbm.F[i][j] = make([]float64, 9)
			lbm.Ftemp[i][j] = make([]float64, 9)
			lbm.Rho[i][j] = rho0
		}
	}

	lbm.InitEquilibrium()

	return lbm
}

func (lbm *LBMPoiseuille) InitEquilibrium() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			lbm.Rho[i][j] = lbm.Rho0
			lbm.U[i][j] = 0.0
			lbm.V[i][j] = 0.0
			lbm.ComputeFeq(i, j, lbm.F[i][j])
		}
	}
}

func (lbm *LBMPoiseuille) ComputeFeq(i, j int, feq []float64) {
	rho := lbm.Rho[i][j]
	u := lbm.U[i][j]
	v := lbm.V[i][j]
	usq := u*u + v*v

	for k := 0; k < 9; k++ {
		cu := float64(Cxp[k])*u + float64(Cyp[k])*v
		feq[k] = Wp[k] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)
	}
}

func (lbm *LBMPoiseuille) Collision() {
	var feq [9]float64
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			lbm.ComputeFeq(i, j, feq[:])
			for k := 0; k < 9; k++ {
				lbm.F[i][j][k] = lbm.F[i][j][k]*(1.0-lbm.Omega) + feq[k]*lbm.Omega
			}
		}
	}
}

func (lbm *LBMPoiseuille) Streaming() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			for k := 0; k < 9; k++ {
				ip := (i - Cxp[k] + lbm.NX) % lbm.NX
				jp := j - Cyp[k]
				if jp >= 0 && jp < lbm.NY {
					lbm.Ftemp[i][j][k] = lbm.F[ip][jp][k]
				}
			}
		}
	}

	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			copy(lbm.F[i][j], lbm.Ftemp[i][j])
		}
	}
}

func (lbm *LBMPoiseuille) ZouHeBottomWall() {
	j := 0
	for i := 0; i < lbm.NX; i++ {
		uWall := 0.0
		vWall := 0.0

		rho := (lbm.F[i][j][0] + lbm.F[i][j][1] + lbm.F[i][j][3] +
			2.0*(lbm.F[i][j][2]+lbm.F[i][j][5]+lbm.F[i][j][6])) / (1.0 + vWall)

		lbm.Rho[i][j] = rho
		lbm.U[i][j] = uWall
		lbm.V[i][j] = vWall

		lbm.F[i][j][4] = lbm.F[i][j][2] - (2.0/3.0)*rho*vWall
		lbm.F[i][j][7] = lbm.F[i][j][5] - 0.5*(lbm.F[i][j][1]-lbm.F[i][j][3]) -
			(1.0/6.0)*rho*vWall + 0.5*rho*uWall
		lbm.F[i][j][8] = lbm.F[i][j][6] + 0.5*(lbm.F[i][j][1]-lbm.F[i][j][3]) -
			(1.0/6.0)*rho*vWall - 0.5*rho*uWall
	}
}

func (lbm *LBMPoiseuille) ZouHeTopWall() {
	j := lbm.NY - 1
	for i := 0; i < lbm.NX; i++ {
		uWall := 0.0
		vWall := 0.0

		rho := (lbm.F[i][j][0] + lbm.F[i][j][1] + lbm.F[i][j][3] +
			2.0*(lbm.F[i][j][4]+lbm.F[i][j][7]+lbm.F[i][j][8])) / (1.0 - vWall)

		lbm.Rho[i][j] = rho
		lbm.U[i][j] = uWall
		lbm.V[i][j] = vWall

		lbm.F[i][j][2] = lbm.F[i][j][4] + (2.0/3.0)*rho*vWall
		lbm.F[i][j][5] = lbm.F[i][j][7] + 0.5*(lbm.F[i][j][3]-lbm.F[i][j][1]) +
			(1.0/6.0)*rho*vWall - 0.5*rho*uWall
		lbm.F[i][j][6] = lbm.F[i][j][8] + 0.5*(lbm.F[i][j][1]-lbm.F[i][j][3]) +
			(1.0/6.0)*rho*vWall + 0.5*rho*uWall
	}
}

func (lbm *LBMPoiseuille) ComputeMacroscopic() {
	for i := 0; i < lbm.NX; i++ {
		for j := 1; j < lbm.NY-1; j++ {
			rho := 0.0
			u := 0.0
			v := 0.0
			for k := 0; k < 9; k++ {
				f := lbm.F[i][j][k]
				rho += f
				u += f * float64(Cxp[k])
				v += f * float64(Cyp[k])
			}
			lbm.Rho[i][j] = rho
			lbm.U[i][j] = u / rho
			lbm.V[i][j] = v / rho
		}
	}
}

func (lbm *LBMPoiseuille) ComputeBodyForce() {
	G := 0.00001
	for i := 0; i < lbm.NX; i++ {
		for j := 1; j < lbm.NY-1; j++ {
			for k := 0; k < 9; k++ {
				cu := float64(Cxp[k])*lbm.U[i][j] + float64(Cyp[k])*lbm.V[i][j]
				force := Wp[k] * 3.0 * (float64(Cxp[k])*G - G*cu*float64(Cxp[k]))
				lbm.F[i][j][k] += force
			}
		}
	}
}

func (lbm *LBMPoiseuille) ComputeTotalMass() float64 {
	totalMass := 0.0
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			totalMass += lbm.Rho[i][j]
		}
	}
	return totalMass
}

func (lbm *LBMPoiseuille) ComputeMassFlowRate() (float64, float64) {
	inflow := 0.0
	outflow := 0.0
	j := lbm.NY / 2

	for i := 0; i < lbm.NX; i++ {
		u := lbm.U[i][j]
		rho := lbm.Rho[i][j]
		if u > 0 {
			outflow += rho * u
		} else {
			inflow += -rho * u
		}
	}
	return inflow, outflow
}

func (lbm *LBMPoiseuille) Step() {
	lbm.Collision()
	lbm.ComputeBodyForce()
	lbm.Streaming()
	lbm.ZouHeBottomWall()
	lbm.ZouHeTopWall()
	lbm.ComputeMacroscopic()
}

func (lbm *LBMPoiseuille) Run(steps int) {
	initialMass := lbm.ComputeTotalMass()
	fmt.Printf("Initial total mass: %.6f\n", initialMass)

	for step := 0; step < steps; step++ {
		lbm.Step()

		if step%1000 == 0 {
			currentMass := lbm.ComputeTotalMass()
			massError := (currentMass - initialMass) / initialMass
			inflow, outflow := lbm.ComputeMassFlowRate()
			netFlow := outflow - inflow

			fmt.Printf("Step %d: Mass error: %.3e%%, Net flow: %.3e\n",
				step, massError*100, netFlow)
		}
	}

	finalMass := lbm.ComputeTotalMass()
	fmt.Printf("\nFinal total mass: %.6f\n", finalMass)
	fmt.Printf("Total mass error: %.3e%%\n", (finalMass-initialMass)/initialMass*100)
}

func (lbm *LBMPoiseuille) PrintVelocityProfile() {
	fmt.Println("\nVelocity profile at x-center:")
	centerX := lbm.NX / 2
	for j := 0; j < lbm.NY; j++ {
		yNorm := float64(j) / float64(lbm.NY-1)
		fmt.Printf("y/H = %.3f, u = %.6f, v = %.6f, rho = %.6f\n",
			yNorm, lbm.U[centerX][j], lbm.V[centerX][j], lbm.Rho[centerX][j])
	}
}

func (lbm *LBMPoiseuille) CompareAnalytical() {
	G := 0.00001
	nu := (lbm.Tau - 0.5) / 3.0
	H := float64(lbm.NY - 1)

	fmt.Println("\nComparison with analytical Poiseuille solution:")
	fmt.Println("y/H\tNumerical u\tAnalytical u\tError (%)")
	centerX := lbm.NX / 2
	for j := 0; j < lbm.NY; j++ {
		y := float64(j)
		yNorm := y / H
		uAnalytical := G * y * (H - y) / (2.0 * nu)
		uNumerical := lbm.U[centerX][j]
		error := 0.0
		if uAnalytical != 0 {
			error = (uNumerical - uAnalytical) / uAnalytical * 100
		}
		fmt.Printf("%.3f\t%.6f\t%.6f\t%.3f\n", yNorm, uNumerical, uAnalytical, error)
	}
}

func mainPoiseuille() {
	NX := 100
	NY := 50
	Tau := 1.0
	Rho0 := 1.0
	Steps := 20000

	fmt.Printf("D2Q9 LBM Poiseuille Flow Simulation\n")
	fmt.Printf("Grid: %d x %d\n", NX, NY)
	fmt.Printf("Tau: %.2f, Rho0: %.2f\n", Tau, Rho0)
	fmt.Printf("Expected steps: %d\n\n", Steps)

	lbm := NewLBMPoiseuille(NX, NY, Tau, Rho0)
	nu := (Tau - 0.5) / 3.0
	fmt.Printf("Viscosity nu: %.6f\n\n", nu)

	fmt.Println("Starting simulation...")
	lbm.Run(Steps)

	lbm.PrintVelocityProfile()
	lbm.CompareAnalytical()
}
