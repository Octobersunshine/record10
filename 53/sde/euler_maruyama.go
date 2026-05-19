package sde

import (
	"math"
	"math/rand"
	"time"
)

type DriftFunc func(x Vector, t float64) Vector
type DiffusionFunc func(x Vector, t float64) Matrix

type Path struct {
	Times  []float64
	Values []Vector
	Dim    int
}

func NewPath(dim, numSteps int) *Path {
	values := make([]Vector, numSteps+1)
	for i := range values {
		values[i] = NewVector(dim)
	}
	return &Path{
		Times:  make([]float64, numSteps+1),
		Values: values,
		Dim:    dim,
	}
}

type SDEConfig struct {
	Drift       DriftFunc
	Diffusion   DiffusionFunc
	X0          Vector
	T0          float64
	T           float64
	NumSteps    int
	CorrMatrix  Matrix
	Seed        int64
	UseSeed     bool
}

func EulerMaruyama(config SDEConfig) (*Path, error) {
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

		driftTerm := config.Drift(x, t)
		diffusionMatrix := config.Diffusion(x, t)
		diffusionTerm := MatVecMul(diffusionMatrix, dW)

		driftScaled := VecScale(driftTerm, dt)
		path.Times[i+1] = t + dt
		path.Values[i+1] = VecAdd(VecAdd(x, driftScaled), diffusionTerm)
	}

	return path, nil
}

func EulerMaruyama1D(
	drift func(x, t float64) float64,
	diffusion func(x, t float64) float64,
	x0 float64,
	t0 float64,
	T float64,
	numSteps int,
) *Path {
	driftMulti := func(x Vector, t float64) Vector {
		return Vector{drift(x[0], t)}
	}
	diffusionMulti := func(x Vector, t float64) Matrix {
		return Matrix{{diffusion(x[0], t)}}
	}

	config := SDEConfig{
		Drift:     driftMulti,
		Diffusion: diffusionMulti,
		X0:        Vector{x0},
		T0:        t0,
		T:         T,
		NumSteps:  numSteps,
		UseSeed:   false,
	}

	path, _ := EulerMaruyama(config)
	return path
}

func EulerMaruyama1DWithSeed(
	drift func(x, t float64) float64,
	diffusion func(x, t float64) float64,
	x0 float64,
	t0 float64,
	T float64,
	numSteps int,
	seed int64,
) *Path {
	driftMulti := func(x Vector, t float64) Vector {
		return Vector{drift(x[0], t)}
	}
	diffusionMulti := func(x Vector, t float64) Matrix {
		return Matrix{{diffusion(x[0], t)}}
	}

	config := SDEConfig{
		Drift:     driftMulti,
		Diffusion: diffusionMulti,
		X0:        Vector{x0},
		T0:        t0,
		T:         T,
		NumSteps:  numSteps,
		Seed:      seed,
		UseSeed:   true,
	}

	path, _ := EulerMaruyama(config)
	return path
}

func identityMatrix(n int) Matrix {
	m := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		m[i][i] = 1.0
	}
	return m
}

func SimulateMultiple(config SDEConfig, numPaths int) ([]*Path, error) {
	paths := make([]*Path, numPaths)
	for i := 0; i < numPaths; i++ {
		configI := config
		configI.UseSeed = false
		path, err := EulerMaruyama(configI)
		if err != nil {
			return nil, err
		}
		paths[i] = path
	}
	return paths, nil
}
