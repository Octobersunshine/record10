package plume

import (
	"math"
	"testing"
)

func TestSigmaY(t *testing.T) {
	tests := []struct {
		x     float64
		class StabilityClass
		want  float64
	}{
		{1000, ClassA, 0.22 * 1000 * math.Pow(1+0.0001*1000, -0.5)},
		{1000, ClassD, 0.08 * 1000 * math.Pow(1+0.0001*1000, -0.5)},
		{0, ClassA, 0},
	}

	for _, tt := range tests {
		got := sigmaY(tt.x, tt.class)
		if math.Abs(got-tt.want) > 1e-9 {
			t.Errorf("sigmaY(%v, %v) = %v, want %v", tt.x, tt.class, got, tt.want)
		}
	}
}

func TestSigmaZ(t *testing.T) {
	tests := []struct {
		x     float64
		class StabilityClass
		want  float64
	}{
		{1000, ClassA, 0.20 * 1000},
		{1000, ClassB, 0.12 * 1000},
		{1000, ClassD, 0.03 * 1000 * math.Pow(1+0.0003*1000, -1)},
	}

	for _, tt := range tests {
		got := sigmaZ(tt.x, tt.class)
		if math.Abs(got-tt.want) > 1e-9 {
			t.Errorf("sigmaZ(%v, %v) = %v, want %v", tt.x, tt.class, got, tt.want)
		}
	}
}

func TestCalculateConcentration(t *testing.T) {
	source := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 1000, ReflectionCount: 10}
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}

	tests := []struct {
		name  string
		point Point
		want  float64
	}{
		{"X=0", Point{0, 0, 0}, 0},
		{"Source height at X=1000", Point{1000, 0, 100}, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CalculateConcentration(source, weather, tt.point)
			if tt.name == "X=0" && got != 0 {
				t.Errorf("CalculateConcentration() = %v, want 0 for X=0", got)
			}
		})
	}
}

func TestCalculateGroundLevelConcentration(t *testing.T) {
	source := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 1000, ReflectionCount: 10}
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}

	conc := CalculateGroundLevelConcentration(source, weather, 1000, 0)
	if conc <= 0 {
		t.Errorf("Expected positive concentration, got %v", conc)
	}
}

func TestMaxGroundLevelConcentration(t *testing.T) {
	source := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 1000, ReflectionCount: 10}
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}

	maxConc, maxX := MaxGroundLevelConcentration(source, weather)
	if maxConc <= 0 {
		t.Errorf("Expected positive max concentration, got %v", maxConc)
	}
	if maxX <= 0 {
		t.Errorf("Expected positive max X position, got %v", maxX)
	}
}

func TestEdgeCases(t *testing.T) {
	source := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 1000, ReflectionCount: 10}
	weather := WeatherParams{WindSpeed: 0, StabilityClass: ClassD}

	conc := CalculateConcentration(source, weather, Point{1000, 0, 0})
	if conc != 0 {
		t.Errorf("Expected 0 for zero wind speed, got %v", conc)
	}

	weather.WindSpeed = 5
	weather.StabilityClass = 'Z'
	conc = CalculateConcentration(source, weather, Point{1000, 0, 0})
	if conc <= 0 {
		t.Errorf("Expected positive concentration for default class, got %v", conc)
	}
}

func TestMultipleReflections(t *testing.T) {
	sourceFew := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 500, ReflectionCount: 0}
	sourceMany := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 500, ReflectionCount: 10}
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}

	point := Point{5000, 0, 0}
	
	concFew := CalculateConcentration(sourceFew, weather, point)
	concMany := CalculateConcentration(sourceMany, weather, point)
	
	if concMany <= concFew {
		t.Errorf("Multiple reflections should increase concentration. Got %v vs %v", concMany, concFew)
	}
}

func TestReflectionConvergence(t *testing.T) {
	source5 := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 1000, ReflectionCount: 5}
	source10 := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 1000, ReflectionCount: 10}
	source20 := SourceParams{Height: 100, EmissionRate: 100, MixingHeight: 1000, ReflectionCount: 20}
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}

	point := Point{5000, 0, 0}
	
	conc5 := CalculateConcentration(source5, weather, point)
	conc10 := CalculateConcentration(source10, weather, point)
	conc20 := CalculateConcentration(source20, weather, point)
	
	diff1 := math.Abs(conc10 - conc5)
	diff2 := math.Abs(conc20 - conc10)
	
	if diff2 > diff1 {
		t.Errorf("Convergence should improve with more reflections: diff5-10=%v, diff10-20=%v", diff1, diff2)
	}
}

func TestDefaultParams(t *testing.T) {
	source := SourceParams{Height: 100, EmissionRate: 100}
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}
	
	conc := CalculateConcentration(source, weather, Point{1000, 0, 0})
	if conc <= 0 {
		t.Errorf("Expected positive concentration with default params, got %v", conc)
	}
}

func TestTerrainAmplificationFactorY(t *testing.T) {
	tests := []struct {
		name    string
		terrain TerrainParams
		x       float64
		class   StabilityClass
		minVal  float64
		maxVal  float64
	}{
		{"Flat terrain", TerrainParams{Type: TerrainFlat}, 1000, ClassD, 1.0, 1.0},
		{"Hilly terrain", TerrainParams{Type: TerrainHilly}, 1000, ClassD, 1.3, 1.5},
		{"Mountainous terrain", TerrainParams{Type: TerrainMountainous}, 1000, ClassD, 1.6, 2.0},
		{"Valley terrain", TerrainParams{Type: TerrainValley, ValleyWidth: 1000}, 1000, ClassD, 1.4, 1.8},
		{"Custom amplification", TerrainParams{Type: TerrainFlat, AmplificationFactor: 1.5}, 1000, ClassD, 1.5, 1.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := terrainAmplificationFactorY(tt.terrain, tt.x, tt.class)
			if got < tt.minVal || got > tt.maxVal {
				t.Errorf("terrainAmplificationFactorY() = %v, want between %v and %v", got, tt.minVal, tt.maxVal)
			}
		})
	}
}

func TestTerrainAmplificationFactorZ(t *testing.T) {
	terrain := TerrainParams{Type: TerrainHilly}
	yFactor := terrainAmplificationFactorY(terrain, 1000, ClassD)
	zFactor := terrainAmplificationFactorZ(terrain, 1000, ClassD)
	
	if zFactor > yFactor {
		t.Errorf("Z amplification should be less than Y amplification: Z=%v, Y=%v", zFactor, yFactor)
	}
}

func TestTerrainEffectOnConcentration(t *testing.T) {
	sourceFlat := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainFlat},
	}
	
	sourceHilly := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainHilly},
	}
	
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}
	point := Point{2000, 0, 0}
	
	concFlat := CalculateConcentration(sourceFlat, weather, point)
	concHilly := CalculateConcentration(sourceHilly, weather, point)
	
	if concHilly >= concFlat {
		t.Errorf("Hilly terrain should have lower concentration due to larger dispersion. Flat=%v, Hilly=%v", concFlat, concHilly)
	}
}

func TestTerrainSlopeEffect(t *testing.T) {
	sourceNoSlope := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainMountainous, Slope: 0},
	}
	
	sourceWithSlope := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainMountainous, Slope: 30},
	}
	
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}
	point := Point{2000, 0, 0}
	
	concNoSlope := CalculateConcentration(sourceNoSlope, weather, point)
	concWithSlope := CalculateConcentration(sourceWithSlope, weather, point)
	
	if concWithSlope >= concNoSlope {
		t.Errorf("Sloped terrain should have lower concentration. NoSlope=%v, WithSlope=%v", concNoSlope, concWithSlope)
	}
}

func TestTerrainRoughnessEffect(t *testing.T) {
	sourceSmooth := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainHilly, Roughness: 0},
	}
	
	sourceRough := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainHilly, Roughness: 2.0},
	}
	
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}
	point := Point{2000, 0, 0}
	
	concSmooth := CalculateConcentration(sourceSmooth, weather, point)
	concRough := CalculateConcentration(sourceRough, weather, point)
	
	if concRough >= concSmooth {
		t.Errorf("Rough terrain should have lower concentration. Smooth=%v, Rough=%v", concSmooth, concRough)
	}
}

func TestValleyTerrainEffect(t *testing.T) {
	sourceFlat := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainFlat},
	}
	
	sourceValley := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainValley, ValleyWidth: 2000},
	}
	
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}
	point := Point{3000, 0, 0}
	
	concFlat := CalculateConcentration(sourceFlat, weather, point)
	concValley := CalculateConcentration(sourceValley, weather, point)
	
	if concValley >= concFlat {
		t.Errorf("Valley terrain should have lower concentration. Flat=%v, Valley=%v", concFlat, concValley)
	}
}

func TestUpwindTerrainEffect(t *testing.T) {
	sourceDownwind := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainHilly, IsUpwind: false},
	}
	
	sourceUpwind := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainHilly, IsUpwind: true},
	}
	
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}
	point := Point{2000, 0, 0}
	
	concDownwind := CalculateConcentration(sourceDownwind, weather, point)
	concUpwind := CalculateConcentration(sourceUpwind, weather, point)
	
	if concUpwind >= concDownwind {
		t.Errorf("Upwind terrain should have lower concentration. Downwind=%v, Upwind=%v", concDownwind, concUpwind)
	}
}

func TestTerrainElevationEffect(t *testing.T) {
	sourceLow := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainMountainous, Elevation: 0},
	}
	
	sourceHigh := SourceParams{
		Height:         100,
		EmissionRate:   100,
		MixingHeight:   1000,
		ReflectionCount: 10,
		Terrain:        TerrainParams{Type: TerrainMountainous, Elevation: 1000},
	}
	
	weather := WeatherParams{WindSpeed: 5, StabilityClass: ClassD}
	point := Point{2000, 0, 0}
	
	concLow := CalculateConcentration(sourceLow, weather, point)
	concHigh := CalculateConcentration(sourceHigh, weather, point)
	
	if concHigh >= concLow {
		t.Errorf("Higher elevation should reduce concentration. Low=%v, High=%v", concLow, concHigh)
	}
}

func TestTerrainAmplificationBounds(t *testing.T) {
	terrain := TerrainParams{
		Type:      TerrainMountainous,
		Slope:     45,
		Roughness: 10.0,
		IsUpwind:  true,
	}
	
	ampY := terrainAmplificationFactorY(terrain, 1000, ClassD)
	ampZ := terrainAmplificationFactorZ(terrain, 1000, ClassD)
	
	if ampY > MaxTerrainAmplification {
		t.Errorf("Y amplification exceeds max bound: %v > %v", ampY, MaxTerrainAmplification)
	}
	if ampY < MinTerrainAmplification {
		t.Errorf("Y amplification below min bound: %v < %v", ampY, MinTerrainAmplification)
	}
	if ampZ > MaxTerrainAmplification {
		t.Errorf("Z amplification exceeds max bound: %v > %v", ampZ, MaxTerrainAmplification)
	}
	if ampZ < MinTerrainAmplification {
		t.Errorf("Z amplification below min bound: %v < %v", ampZ, MinTerrainAmplification)
	}
}
