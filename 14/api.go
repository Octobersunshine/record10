package differential_evolution

import "fmt"

type Config struct {
	PopulationSize   int
	CR               float64
	F                float64
	MaxIterations    int
	BoundaryStrategy BoundaryStrategy
	AdaptiveStrategy AdaptiveStrategy
}

func DefaultConfig() Config {
	return Config{
		PopulationSize:   100,
		CR:               0.9,
		F:                0.5,
		MaxIterations:    1000,
		BoundaryStrategy: MixedStrategy,
		AdaptiveStrategy: JadeAdaptive,
	}
}

func Minimize(function string, dim int, lower, upper float64) (*Result, error) {
	config := DefaultConfig()
	return MinimizeWithConfig(function, dim, lower, upper, config)
}

func MinimizeWithBounds(function string, dim int, lowerBound, upperBound []float64) (*Result, error) {
	config := DefaultConfig()
	return MinimizeWithConfigAndBounds(function, dim, lowerBound, upperBound, config)
}

func MinimizeWithConfig(function string, dim int, lower, upper float64, config Config) (*Result, error) {
	lowerBound := make([]float64, dim)
	upperBound := make([]float64, dim)
	for i := 0; i < dim; i++ {
		lowerBound[i] = lower
		upperBound[i] = upper
	}
	objectiveFunc, err := parseFunction(function, dim)
	if err != nil {
		return nil, fmt.Errorf("failed to parse function: %v", err)
	}
	de := NewDE(config.PopulationSize, config.CR, config.F, config.MaxIterations, dim, lowerBound, upperBound, objectiveFunc)
	de.SetBoundaryStrategy(config.BoundaryStrategy)
	de.SetAdaptiveStrategy(config.AdaptiveStrategy)
	return de.Optimize(), nil
}

func MinimizeWithConfigAndBounds(function string, dim int, lowerBound, upperBound []float64, config Config) (*Result, error) {
	objectiveFunc, err := parseFunction(function, dim)
	if err != nil {
		return nil, fmt.Errorf("failed to parse function: %v", err)
	}
	de := NewDE(config.PopulationSize, config.CR, config.F, config.MaxIterations, dim, lowerBound, upperBound, objectiveFunc)
	de.SetBoundaryStrategy(config.BoundaryStrategy)
	de.SetAdaptiveStrategy(config.AdaptiveStrategy)
	return de.Optimize(), nil
}
