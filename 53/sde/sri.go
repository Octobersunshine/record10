package sde

import (
	"math"
	"math/rand"
	"time"
)

type SRIConfig struct {
	Drift      func(x, t float64) float64
	Diffusion  func(x, t float64) float64
	X0         float64
	T0         float64
	T          float64
	NumSteps   int
	Seed       int64
	UseSeed    bool
	Order      int
}

func SRI1(config SRIConfig) *Path {
	var r *rand.Rand
	if config.UseSeed {
		r = rand.New(rand.NewSource(config.Seed))
	} else {
		r = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	dt := (config.T - config.T0) / float64(config.NumSteps)
	sqrtDt := math.Sqrt(dt)

	path := NewPath(1, config.NumSteps)
	path.Times[0] = config.T0
	path.Values[0][0] = config.X0

	for i := 0; i < config.NumSteps; i++ {
		t := path.Times[i]
		x := path.Values[i][0]

		dW := r.NormFloat64() * sqrtDt
		dZ := 0.5 * (dW*dW - dt) / sqrtDt

		a1 := x + config.Drift(x, t)*dt + config.Diffusion(x, t)*(dW-sqrtDt)
		K1 := config.Drift(a1, t+dt)
		L1 := config.Diffusion(a1, t+dt)

		xNext := x + 0.5*(config.Drift(x, t)+K1)*dt +
			config.Diffusion(x, t)*dW +
			0.5*(L1-config.Diffusion(x, t))*dZ

		path.Times[i+1] = t + dt
		path.Values[i+1][0] = xNext
	}

	return path
}

func SRI2(config SRIConfig) *Path {
	var r *rand.Rand
	if config.UseSeed {
		r = rand.New(rand.NewSource(config.Seed))
	} else {
		r = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	dt := (config.T - config.T0) / float64(config.NumSteps)
	sqrtDt := math.Sqrt(dt)

	path := NewPath(1, config.NumSteps)
	path.Times[0] = config.T0
	path.Values[0][0] = config.X0

	for i := 0; i < config.NumSteps; i++ {
		t := path.Times[i]
		x := path.Values[i][0]

		dW := r.NormFloat64() * sqrtDt

		H1 := x + config.Drift(x, t)*dt + config.Diffusion(x, t)*sqrtDt
		H2 := x + config.Drift(x, t)*dt - config.Diffusion(x, t)*sqrtDt

		K1 := config.Drift(H1, t+dt)
		K2 := config.Drift(H2, t+dt)
		L1 := config.Diffusion(H1, t+dt)
		L2 := config.Diffusion(H2, t+dt)

		xNext := x + 0.25*(K1+K2+2*config.Drift(x, t))*dt +
			0.25*(L1+L2)*dW +
			0.25*(L1-L2)*(dW*dW-dt)/sqrtDt

		path.Times[i+1] = t + dt
		path.Values[i+1][0] = xNext
	}

	return path
}

type SRIMultiConfig struct {
	Drift      DriftFunc
	Diffusion  DiffusionFunc
	X0         Vector
	T0         float64
	T          float64
	NumSteps   int
	CorrMatrix Matrix
	Seed       int64
	UseSeed    bool
	Order      int
}

func SRI1Multi(config SRIMultiConfig) (*Path, error) {
	dim := len(config.X0)
	dt := (config.T - config.T0) / float64(config.NumSteps)
	sqrtDt := math.Sqrt(dt)

	var r *rand.Rand
	if config.UseSeed {
		r = rand.New(rand.NewSource(config.Seed))
	} else {
		r = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	var L Matrix
	var err error
	if config.CorrMatrix != nil {
		L, err = Cholesky(config.CorrMatrix)
		if err != nil {
			return nil, err
		}
	} else {
		L = identityMatrix(dim)
	}

	path := NewPath(dim, config.NumSteps)
	path.Times[0] = config.T0
	copy(path.Values[0], config.X0)

	for i := 0; i < config.NumSteps; i++ {
		t := path.Times[i]
		x := path.Values[i]

		dW := NewVector(dim)
		dZ := NewVector(dim)
		for j := 0; j < dim; j++ {
			xi := r.NormFloat64()
			dW[j] = xi * sqrtDt
			dZ[j] = 0.5 * (dW[j]*dW[j] - dt) / sqrtDt
		}

		if config.CorrMatrix != nil {
			dW = MatVecMul(L, dW)
			dZ = MatVecMul(L, dZ)
		}

		a1 := make(Vector, dim)
		for j := 0; j < dim; j++ {
			a1[j] = x[j] + config.Drift(x, t)[j]*dt + config.Diffusion(x, t)[j][j]*(dW[j]-sqrtDt)
		}
		K1 := config.Drift(a1, t+dt)
		L1 := config.Diffusion(a1, t+dt)

		xNext := make(Vector, dim)
		driftCurrent := config.Drift(x, t)
		diffusionCurrent := config.Diffusion(x, t)
		for j := 0; j < dim; j++ {
			driftTerm := 0.5 * (driftCurrent[j] + K1[j]) * dt
			diffusionTerm1 := diffusionCurrent[j][j] * dW[j]
			diffusionTerm2 := 0.5 * (L1[j][j] - diffusionCurrent[j][j]) * dZ[j]
			xNext[j] = x[j] + driftTerm + diffusionTerm1 + diffusionTerm2
		}

		path.Times[i+1] = t + dt
		path.Values[i+1] = xNext
	}

	return path, nil
}

func SRI2Multi(config SRIMultiConfig) (*Path, error) {
	dim := len(config.X0)
	dt := (config.T - config.T0) / float64(config.NumSteps)
	sqrtDt := math.Sqrt(dt)

	var r *rand.Rand
	if config.UseSeed {
		r = rand.New(rand.NewSource(config.Seed))
	} else {
		r = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	var L Matrix
	var err error
	if config.CorrMatrix != nil {
		L, err = Cholesky(config.CorrMatrix)
		if err != nil {
			return nil, err
		}
	} else {
		L = identityMatrix(dim)
	}

	path := NewPath(dim, config.NumSteps)
	path.Times[0] = config.T0
	copy(path.Values[0], config.X0)

	for i := 0; i < config.NumSteps; i++ {
		t := path.Times[i]
		x := path.Values[i]

		dW := NewVector(dim)
		for j := 0; j < dim; j++ {
			dW[j] = r.NormFloat64() * sqrtDt
		}

		if config.CorrMatrix != nil {
			dW = MatVecMul(L, dW)
		}

		H1 := make(Vector, dim)
		H2 := make(Vector, dim)
		diffusionCurrent := config.Diffusion(x, t)
		for j := 0; j < dim; j++ {
			H1[j] = x[j] + config.Drift(x, t)[j]*dt + diffusionCurrent[j][j]*sqrtDt
			H2[j] = x[j] + config.Drift(x, t)[j]*dt - diffusionCurrent[j][j]*sqrtDt
		}

		K1 := config.Drift(H1, t+dt)
		K2 := config.Drift(H2, t+dt)
		L1 := config.Diffusion(H1, t+dt)
		L2 := config.Diffusion(H2, t+dt)

		xNext := make(Vector, dim)
		driftCurrent := config.Drift(x, t)
		for j := 0; j < dim; j++ {
			driftTerm := 0.25 * (K1[j] + K2[j] + 2*driftCurrent[j]) * dt
			diffusionTerm1 := 0.25 * (L1[j][j] + L2[j][j]) * dW[j]
			diffusionTerm2 := 0.25 * (L1[j][j] - L2[j][j]) * (dW[j]*dW[j] - dt) / sqrtDt
			xNext[j] = x[j] + driftTerm + diffusionTerm1 + diffusionTerm2
		}

		path.Times[i+1] = t + dt
		path.Values[i+1] = xNext
	}

	return path, nil
}
