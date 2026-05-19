package main

import (
	"fmt"
	"math"
)

type LBM struct {
	NX, NY int
	Tau    float64
	Omega  float64
	UW     float64
	F      [][][]float64
	Ftemp  [][][]float64
	Rho    [][]float64
	U      [][]float64
	V      [][]float64
}

var W = [9]float64{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
	1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0}
var Cx = [9]int{0, 1, 0, -1, 0, 1, -1, -1, 1}
var Cy = [9]int{0, 0, 1, 0, -1, 1, 1, -1, -1}

func NewLBM(nx, ny int, tau, uw float64) *LBM {
	lbm := &LBM{
		NX:    nx,
		NY:    ny,
		Tau:   tau,
		Omega: 1.0 / tau,
		UW:    uw,
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
			lbm.Rho[i][j] = 1.0
		}
	}

	lbm.InitEquilibrium()

	return lbm
}

func (lbm *LBM) InitEquilibrium() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			u := 0.0
			v := 0.0
			if j == lbm.NY-1 {
				u = lbm.UW
			}
			lbm.Rho[i][j] = 1.0
			lbm.U[i][j] = u
			lbm.V[i][j] = v
			lbm.ComputeFeq(i, j, lbm.F[i][j])
		}
	}
}

func (lbm *LBM) ComputeFeq(i, j int, feq []float64) {
	rho := lbm.Rho[i][j]
	u := lbm.U[i][j]
	v := lbm.V[i][j]
	usq := u*u + v*v

	for k := 0; k < 9; k++ {
		cu := float64(Cx[k])*u + float64(Cy[k])*v
		feq[k] = W[k] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)
	}
}

func (lbm *LBM) Collision() {
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

func (lbm *LBM) Streaming() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			for k := 0; k < 9; k++ {
				ip := i - Cx[k]
				jp := j - Cy[k]
				if ip >= 0 && ip < lbm.NX && jp >= 0 && jp < lbm.NY {
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

func (lbm *LBM) BoundaryConditions() {
	for i := 0; i < lbm.NX; i++ {
		lbm.F[i][lbm.NY-1][4] = lbm.F[i][lbm.NY-1][2]
		lbm.F[i][lbm.NY-1][7] = lbm.F[i][lbm.NY-1][5] + 6.0*W[7]*lbm.UW
		lbm.F[i][lbm.NY-1][8] = lbm.F[i][lbm.NY-1][6] - 6.0*W[8]*lbm.UW
	}

	for i := 0; i < lbm.NX; i++ {
		lbm.F[i][0][2] = lbm.F[i][0][4]
		lbm.F[i][0][5] = lbm.F[i][0][7]
		lbm.F[i][0][6] = lbm.F[i][0][8]
	}

	for j := 0; j < lbm.NY; j++ {
		lbm.F[0][j][1] = lbm.F[0][j][3]
		lbm.F[0][j][5] = lbm.F[0][j][7]
		lbm.F[0][j][8] = lbm.F[0][j][6]
	}

	for j := 0; j < lbm.NY; j++ {
		lbm.F[lbm.NX-1][j][3] = lbm.F[lbm.NX-1][j][1]
		lbm.F[lbm.NX-1][j][7] = lbm.F[lbm.NX-1][j][5]
		lbm.F[lbm.NX-1][j][6] = lbm.F[lbm.NX-1][j][8]
	}
}

func (lbm *LBM) ComputeMacroscopic() {
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			rho := 0.0
			u := 0.0
			v := 0.0
			for k := 0; k < 9; k++ {
				f := lbm.F[i][j][k]
				rho += f
				u += f * float64(Cx[k])
				v += f * float64(Cy[k])
			}
			lbm.Rho[i][j] = rho
			lbm.U[i][j] = u / rho
			lbm.V[i][j] = v / rho
		}
	}

	for i := 0; i < lbm.NX; i++ {
		lbm.U[i][lbm.NY-1] = lbm.UW
		lbm.V[i][lbm.NY-1] = 0.0
	}
}

func (lbm *LBM) Step() {
	lbm.Collision()
	lbm.Streaming()
	lbm.BoundaryConditions()
	lbm.ComputeMacroscopic()
}

func (lbm *LBM) Run(steps int) {
	for step := 0; step < steps; step++ {
		lbm.Step()
		if step%1000 == 0 {
			fmt.Printf("Step %d completed\n", step)
		}
	}
}

func (lbm *LBM) GetVelocityField() ([][]float64, [][]float64) {
	return lbm.U, lbm.V
}

func (lbm *LBM) PrintCenterlineVelocities() {
	fmt.Println("\nCenterline velocities (x-center, varying y):")
	centerX := lbm.NX / 2
	for j := 0; j < lbm.NY; j++ {
		yNorm := float64(j) / float64(lbm.NY-1)
		fmt.Printf("y/H = %.3f, u = %.6f, v = %.6f\n", yNorm, lbm.U[centerX][j], lbm.V[centerX][j])
	}

	fmt.Println("\nCenterline velocities (y-center, varying x):")
	centerY := lbm.NY / 2
	for i := 0; i < lbm.NX; i++ {
		xNorm := float64(i) / float64(lbm.NX-1)
		fmt.Printf("x/L = %.3f, u = %.6f, v = %.6f\n", xNorm, lbm.U[i][centerY], lbm.V[i][centerY])
	}
}

func (lbm *LBM) ComputeReynolds() float64 {
	nu := (lbm.Tau - 0.5) / 3.0
	return lbm.UW * float64(lbm.NX) / nu
}

func (lbm *LBM) ComputeTotalMass() float64 {
	totalMass := 0.0
	for i := 0; i < lbm.NX; i++ {
		for j := 0; j < lbm.NY; j++ {
			totalMass += lbm.Rho[i][j]
		}
	}
	return totalMass
}

func (lbm *LBM) BoundaryConditionsZouHe() {
	j := lbm.NY - 1
	for i := 0; i < lbm.NX; i++ {
		uWall := lbm.UW
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

	j = 0
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

	i := 0
	for j := 0; j < lbm.NY; j++ {
		uWall := 0.0
		vWall := 0.0

		rho := (lbm.F[i][j][0] + lbm.F[i][j][2] + lbm.F[i][j][4] +
			2.0*(lbm.F[i][j][3]+lbm.F[i][j][6]+lbm.F[i][j][7])) / (1.0 + uWall)

		lbm.Rho[i][j] = rho
		lbm.U[i][j] = uWall
		lbm.V[i][j] = vWall

		lbm.F[i][j][1] = lbm.F[i][j][3] + (2.0/3.0)*rho*uWall
		lbm.F[i][j][5] = lbm.F[i][j][7] + 0.5*(lbm.F[i][j][4]-lbm.F[i][j][2]) -
			(1.0/6.0)*rho*uWall + 0.5*rho*vWall
		lbm.F[i][j][8] = lbm.F[i][j][6] + 0.5*(lbm.F[i][j][2]-lbm.F[i][j][4]) -
			(1.0/6.0)*rho*uWall - 0.5*rho*vWall
	}

	i = lbm.NX - 1
	for j := 0; j < lbm.NY; j++ {
		uWall := 0.0
		vWall := 0.0

		rho := (lbm.F[i][j][0] + lbm.F[i][j][2] + lbm.F[i][j][4] +
			2.0*(lbm.F[i][j][1]+lbm.F[i][j][5]+lbm.F[i][j][8])) / (1.0 - uWall)

		lbm.Rho[i][j] = rho
		lbm.U[i][j] = uWall
		lbm.V[i][j] = vWall

		lbm.F[i][j][3] = lbm.F[i][j][1] - (2.0/3.0)*rho*uWall
		lbm.F[i][j][7] = lbm.F[i][j][5] + 0.5*(lbm.F[i][j][2]-lbm.F[i][j][4]) +
			(1.0/6.0)*rho*uWall - 0.5*rho*vWall
		lbm.F[i][j][6] = lbm.F[i][j][8] + 0.5*(lbm.F[i][j][4]-lbm.F[i][j][2]) +
			(1.0/6.0)*rho*uWall + 0.5*rho*vWall
	}
}

func (lbm *LBM) StepZouHe() {
	lbm.Collision()
	lbm.Streaming()
	lbm.BoundaryConditionsZouHe()
	lbm.ComputeMacroscopic()
}

func (lbm *LBM) RunZouHe(steps int) {
	initialMass := lbm.ComputeTotalMass()
	fmt.Printf("Initial total mass: %.6f\n", initialMass)

	for step := 0; step < steps; step++ {
		lbm.StepZouHe()

		if step%1000 == 0 {
			currentMass := lbm.ComputeTotalMass()
			massError := (currentMass - initialMass) / initialMass
			fmt.Printf("Step %d: Mass error: %.3e%%\n", step, massError*100)
		}
	}

	finalMass := lbm.ComputeTotalMass()
	fmt.Printf("\nFinal total mass: %.6f\n", finalMass)
	fmt.Printf("Total mass error: %.3e%%\n", (finalMass-initialMass)/initialMass*100)
}

func mainLidDriven() {
	NX := 100
	NY := 100
	Tau := 0.8
	UW := 0.1
	Steps := 10000

	fmt.Printf("D2Q9 LBM Lid-Driven Cavity Simulation (Zou-He Boundary)\n")
	fmt.Printf("Grid: %d x %d\n", NX, NY)
	fmt.Printf("Tau: %.2f, UW: %.3f\n", Tau, UW)
	fmt.Printf("Expected steps: %d\n\n", Steps)

	lbm := NewLBM(NX, NY, Tau, UW)
	fmt.Printf("Reynolds number: %.2f\n", lbm.ComputeReynolds())

	fmt.Println("\nStarting simulation with Zou-He boundary (mass conserving)...")
	lbm.RunZouHe(Steps)

	fmt.Println("\nSimulation completed!")
	lbm.PrintCenterlineVelocities()

	U, V := lbm.GetVelocityField()
	fmt.Printf("\nVelocity field statistics:\n")
	maxU, maxV := 0.0, 0.0
	minU, minV := 0.0, 0.0
	for i := 0; i < NX; i++ {
		for j := 0; j < NY; j++ {
			maxU = math.Max(maxU, U[i][j])
			maxV = math.Max(maxV, V[i][j])
			minU = math.Min(minU, U[i][j])
			minV = math.Min(minV, V[i][j])
		}
	}
	fmt.Printf("U range: [%.6f, %.6f]\n", minU, maxU)
	fmt.Printf("V range: [%.6f, %.6f]\n", minV, maxV)
}

func main() {
	fmt.Println("Choose simulation:")
	fmt.Println("1. Poiseuille Flow (Zou-He boundary)")
	fmt.Println("2. Lid-Driven Cavity (Zou-He boundary)")
	fmt.Println("3. Two-Phase Separation (Shan-Chen model)")
	fmt.Println("\nRunning multiphase simulation by default...\n")
	mainMultiphase()
}
