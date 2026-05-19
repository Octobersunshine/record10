package plume

import (
	"math"
)

type StabilityClass rune

const (
	ClassA StabilityClass = 'A'
	ClassB StabilityClass = 'B'
	ClassC StabilityClass = 'C'
	ClassD StabilityClass = 'D'
	ClassE StabilityClass = 'E'
	ClassF StabilityClass = 'F'
)

const (
	DefaultMixingHeight    = 1000.0
	DefaultReflectionCount = 10
	ConvergenceThreshold   = 1e-6
	MinTerrainAmplification = 1.0
	MaxTerrainAmplification = 3.0
)

type TerrainType int

const (
	TerrainFlat TerrainType = iota
	TerrainHilly
	TerrainMountainous
	TerrainValley
)

type TerrainParams struct {
	Type              TerrainType
	Slope             float64
	Elevation         float64
	Roughness         float64
	ValleyWidth       float64
	IsUpwind         bool
	AmplificationFactor float64
}

type SourceParams struct {
	Height         float64
	EmissionRate   float64
	MixingHeight   float64
	ReflectionCount int
	Terrain        TerrainParams
}

type WeatherParams struct {
	WindSpeed      float64
	StabilityClass StabilityClass
}

type Point struct {
	X float64
	Y float64
	Z float64
}

func terrainAmplificationFactorY(terrain TerrainParams, x float64, class StabilityClass) float64 {
	if terrain.AmplificationFactor > 0 {
		return math.Min(math.Max(terrain.AmplificationFactor, MinTerrainAmplification), MaxTerrainAmplification)
	}

	baseFactor := 1.0

	switch terrain.Type {
	case TerrainHilly:
		baseFactor = 1.3
	case TerrainMountainous:
		baseFactor = 1.6
	case TerrainValley:
		baseFactor = 1.4
	default:
		baseFactor = 1.0
	}

	if terrain.Slope > 0 {
		slopeFactor := 1.0 + 0.5*math.Sin(terrain.Slope*math.Pi/180.0)
		baseFactor *= slopeFactor
	}

	if terrain.Roughness > 0 {
		roughnessFactor := 1.0 + 0.3*math.Log10(1.0+terrain.Roughness)
		baseFactor *= roughnessFactor
	}

	if terrain.Type == TerrainValley && terrain.ValleyWidth > 0 {
		valleyFactor := 1.0 + 0.2*math.Exp(-x/(2*terrain.ValleyWidth))
		baseFactor *= valleyFactor
	}

	if terrain.IsUpwind {
		baseFactor *= 1.2
	}

	return math.Min(math.Max(baseFactor, MinTerrainAmplification), MaxTerrainAmplification)
}

func terrainAmplificationFactorZ(terrain TerrainParams, x float64, class StabilityClass) float64 {
	yFactor := terrainAmplificationFactorY(terrain, x, class)
	
	zFactor := yFactor * 0.8
	
	if class >= ClassE {
		zFactor *= 1.15
	}
	
	if terrain.Type == TerrainValley {
		zFactor *= 1.2
	}
	
	return math.Min(math.Max(zFactor, MinTerrainAmplification), MaxTerrainAmplification)
}

func sigmaY(x float64, class StabilityClass) float64 {
	switch class {
	case ClassA:
		return 0.22 * x * math.Pow(1+0.0001*x, -0.5)
	case ClassB:
		return 0.16 * x * math.Pow(1+0.0001*x, -0.5)
	case ClassC:
		return 0.11 * x * math.Pow(1+0.0001*x, -0.5)
	case ClassD:
		return 0.08 * x * math.Pow(1+0.0001*x, -0.5)
	case ClassE:
		return 0.06 * x * math.Pow(1+0.0001*x, -0.5)
	case ClassF:
		return 0.04 * x * math.Pow(1+0.0001*x, -0.5)
	default:
		return 0.08 * x * math.Pow(1+0.0001*x, -0.5)
	}
}

func sigmaZ(x float64, class StabilityClass) float64 {
	switch class {
	case ClassA:
		return 0.20 * x
	case ClassB:
		return 0.12 * x
	case ClassC:
		return 0.06 * x * math.Pow(1+0.0015*x, -0.5)
	case ClassD:
		return 0.03 * x * math.Pow(1+0.0003*x, -1)
	case ClassE:
		return 0.015 * x * math.Pow(1+0.0003*x, -1)
	case ClassF:
		return 0.008 * x * math.Pow(1+0.0003*x, -1)
	default:
		return 0.03 * x * math.Pow(1+0.0003*x, -1)
	}
}

func effectiveSourceHeight(source SourceParams, terrain TerrainParams) float64 {
	effectiveH := source.Height
	
	if terrain.Elevation > 0 {
		effectiveH += terrain.Elevation * 0.3
	}
	
	if terrain.Slope > 0 && !terrain.IsUpwind {
		effectiveH += terrain.Slope * 0.5
	}
	
	return effectiveH
}

func CalculateConcentration(source SourceParams, weather WeatherParams, point Point) float64 {
	if point.X <= 0 {
		return 0
	}

	amplificationY := terrainAmplificationFactorY(source.Terrain, point.X, weather.StabilityClass)
	amplificationZ := terrainAmplificationFactorZ(source.Terrain, point.X, weather.StabilityClass)

	sy := sigmaY(point.X, weather.StabilityClass) * amplificationY
	sz := sigmaZ(point.X, weather.StabilityClass) * amplificationZ

	if sy <= 0 || sz <= 0 || weather.WindSpeed <= 0 {
		return 0
	}

	mixingHeight := source.MixingHeight
	if mixingHeight <= 0 {
		mixingHeight = DefaultMixingHeight
	}

	reflectionCount := source.ReflectionCount
	if reflectionCount <= 0 {
		reflectionCount = DefaultReflectionCount
	}

	effectiveH := effectiveSourceHeight(source, source.Terrain)

	term1 := source.EmissionRate / (2 * math.Pi * weather.WindSpeed * sy * sz)
	term2 := math.Exp(-math.Pow(point.Y, 2) / (2 * math.Pow(sy, 2)))

	normZ := point.Z / sz
	normH := effectiveH / sz
	normL := mixingHeight / sz

	reflectionSum := 0.0
	prevSum := 0.0

	for n := 0; n <= reflectionCount; n++ {
		if n == 0 {
			reflectionSum += math.Exp(-math.Pow(normZ-normH, 2)/2) +
				math.Exp(-math.Pow(normZ+normH, 2)/2)
		} else {
			reflectionSum += math.Exp(-math.Pow(normZ-normH+2*float64(n)*normL, 2)/2) +
				math.Exp(-math.Pow(normZ+normH+2*float64(n)*normL, 2)/2) +
				math.Exp(-math.Pow(normZ+normH-2*float64(n)*normL, 2)/2) +
				math.Exp(-math.Pow(normZ-normH-2*float64(n)*normL, 2)/2)
		}

		if n > 0 && math.Abs(reflectionSum-prevSum) < ConvergenceThreshold*prevSum {
			break
		}
		prevSum = reflectionSum
	}

	return term1 * term2 * reflectionSum
}

func CalculateGroundLevelConcentration(source SourceParams, weather WeatherParams, x, y float64) float64 {
	return CalculateConcentration(source, weather, Point{X: x, Y: y, Z: 0})
}

func MaxGroundLevelConcentration(source SourceParams, weather WeatherParams) (float64, float64) {
	maxConc := 0.0
	maxX := 0.0

	for x := 10.0; x <= 10000; x += 10 {
		conc := CalculateGroundLevelConcentration(source, weather, x, 0)
		if conc > maxConc {
			maxConc = conc
			maxX = x
		}
	}

	return maxConc, maxX
}
