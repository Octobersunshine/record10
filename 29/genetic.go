package genetic

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

type MultiObjectiveResult struct {
	Solutions [][]float64
	Objectives [][]float64
}

type NSGAIIIndividual struct {
	Genes          []float64
	Objectives     []float64
	Rank           int
	CrowdingDistance float64
	Viable         bool
	DominatedCount int
	Dominates      []int
}

type NSGAII struct {
	objectives  []ObjectiveFunc
	constraints []ConstraintFunc
	varRanges   []VariableRange
	config      Config
	population  []NSGAIIIndividual
	rng         *rand.Rand
}

type SelectionMethod int

const (
	TournamentSelection SelectionMethod = iota
	RouletteWheelSelection
	RankSelection
	StochasticUniversalSampling
)

type FitnessScalingMethod int

const (
	NoScaling FitnessScalingMethod = iota
	LinearScaling
	SigmaTruncationScaling
	PowerLawScaling
)

type ObjectiveFunc func([]float64) float64

type ConstraintFunc func([]float64) bool

type VariableRange struct {
	Min float64
	Max float64
}

type Config struct {
	PopulationSize       int
	MaxGenerations     int
	MutationRate       float64
	CrossoverRate      float64
	ElitismCount       int
	TournamentSize     int
	SelectionMethod    SelectionMethod
	FitnessScaling     FitnessScalingMethod
	ScalingFactor      float64
	PressureFactor     float64
}

type Individual struct {
	Genes         []float64
	RawFitness     float64
	ScaledFitness  float64
	Viable          bool
}

type GA struct {
	objective   ObjectiveFunc
	constraints []ConstraintFunc
	varRanges   []VariableRange
	config      Config
	population  []Individual
	best        Individual
	rng         *rand.Rand
}

func NewGA(objective ObjectiveFunc, varRanges []VariableRange, constraints []ConstraintFunc, config Config) *GA {
	return &GA{
		objective:   objective,
		constraints: constraints,
		varRanges:   varRanges,
		config:      config,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (ga *GA) initializePopulation() {
	ga.population = make([]Individual, ga.config.PopulationSize)
	for i := range ga.population {
		genes := make([]float64, len(ga.varRanges))
		for j, vr := range ga.varRanges {
			genes[j] = vr.Min + ga.rng.Float64()*(vr.Max-vr.Min)
		}
		ga.population[i] = Individual{
			Genes:         genes,
			RawFitness:     0,
			ScaledFitness:  0,
			Viable:          false,
		}
	}
}

func (ga *GA) evaluatePopulation() {
	for i := range ga.population {
		ga.evaluateIndividual(&ga.population[i])
	}
	ga.applyFitnessScaling()
}

func (ga *GA) evaluateIndividual(ind *Individual) {
	ind.Viable = true
	for _, constraint := range ga.constraints {
		if !constraint(ind.Genes) {
			ind.Viable = false
			break
		}
	}
	
	if ind.Viable {
		ind.RawFitness = ga.objective(ind.Genes)
	} else {
		ind.RawFitness = math.Inf(1)
	}
}

func (ga *GA) applyFitnessScaling() {
	viableCount := 0
	sum := 0.0
	minFit := math.Inf(1)
	maxFit := math.Inf(-1)
	
	for _, ind := range ga.population {
		if ind.Viable {
			viableCount++
			sum += ind.RawFitness
			if ind.RawFitness < minFit {
				minFit = ind.RawFitness
			}
			if ind.RawFitness > maxFit {
				maxFit = ind.RawFitness
			}
		}
	}
	
	if viableCount == 0 {
		return
	}
	
	avg := sum / float64(viableCount)
	
	switch ga.config.FitnessScaling {
	case NoScaling:
		for i := range ga.population {
			if ga.population[i].Viable {
				ga.population[i].ScaledFitness = ga.population[i].RawFitness
			}
		}
		
	case LinearScaling:
		a := ga.config.ScalingFactor
		if a <= 1.0 {
			a = 2.0
		}
		
		delta := maxFit - minFit
		if delta < 1e-10 {
			for i := range ga.population {
				if ga.population[i].Viable {
					ga.population[i].ScaledFitness = 1.0
				}
			}
			return
		}
		
		scale := (a - 1) * avg / delta
		offset := avg - scale*minFit
		
		for i := range ga.population {
			if ga.population[i].Viable {
				ga.population[i].ScaledFitness = scale*ga.population[i].RawFitness + offset
				if ga.population[i].ScaledFitness < 0 {
					ga.population[i].ScaledFitness = 0
				}
			}
		}
		
	case SigmaTruncationScaling:
		variance := 0.0
		for _, ind := range ga.population {
			if ind.Viable {
				diff := ind.RawFitness - avg
				variance += diff * diff
			}
		}
		stdDev := math.Sqrt(variance / float64(viableCount))
		
		c := ga.config.ScalingFactor
		if c <= 0 {
			c = 2.0
		}
		
		for i := range ga.population {
			if ga.population[i].Viable {
				ga.population[i].ScaledFitness = ga.population[i].RawFitness - (avg - c*stdDev)
				if ga.population[i].ScaledFitness < 0 {
					ga.population[i].ScaledFitness = 0
				}
			}
		}
		
	case PowerLawScaling:
		k := ga.config.ScalingFactor
		if k <= 0 {
			k = 1.5
		}
		for i := range ga.population {
			if ga.population[i].Viable {
				ga.population[i].ScaledFitness = math.Pow(ga.population[i].RawFitness, k)
			}
		}
	}
	
	for i := range ga.population {
		if ga.population[i].Viable && ga.population[i].ScaledFitness > 0 {
			ga.population[i].ScaledFitness = 1.0 / (1.0 + ga.population[i].ScaledFitness)
		}
	}
}

func (ga *GA) selectParent() Individual {
	switch ga.config.SelectionMethod {
	case TournamentSelection:
		return ga.tournamentSelect()
	case RouletteWheelSelection:
		return ga.rouletteWheelSelect()
	case RankSelection:
		return ga.rankSelect()
	case StochasticUniversalSampling:
		return ga.stochasticUniversalSamplingSelect()
	default:
		return ga.tournamentSelect()
	}
}

func (ga *GA) tournamentSelect() Individual {
	bestIdx := ga.rng.Intn(len(ga.population))
	for i := 1; i < ga.config.TournamentSize; i++ {
		idx := ga.rng.Intn(len(ga.population))
		if ga.population[idx].Viable && (!ga.population[bestIdx].Viable || 
			ga.population[idx].ScaledFitness > ga.population[bestIdx].ScaledFitness) {
			bestIdx = idx
		}
	}
	return ga.population[bestIdx]
}

func (ga *GA) rouletteWheelSelect() Individual {
	totalFitness := 0.0
	for _, ind := range ga.population {
		if ind.Viable {
			totalFitness += ind.ScaledFitness
		}
	}
	
	if totalFitness <= 0 {
		return ga.tournamentSelect()
	}
	
	point := ga.rng.Float64() * totalFitness
	cumulative := 0.0
	
	for _, ind := range ga.population {
		if ind.Viable {
			cumulative += ind.ScaledFitness
			if cumulative >= point {
				return ind
			}
		}
	}
	
	return ga.population[0]
}

func (ga *GA) rankSelect() Individual {
	viable := make([]int, 0, len(ga.population))
	for i, ind := range ga.population {
		if ind.Viable {
			viable = append(viable, i)
		}
	}
	
	if len(viable) == 0 {
		return ga.population[0]
	}
	
	sort.Slice(viable, func(i, j int) bool {
		return ga.population[viable[i]].ScaledFitness > ga.population[viable[j]].ScaledFitness
	})
	
	n := float64(len(viable))
	sp := ga.config.PressureFactor
	if sp <= 1.0 || sp > 2.0 {
		sp = 1.5
	}
	
	total := 0.0
	rankWeights := make([]float64, len(viable))
	for i := range viable {
		rank := float64(i + 1)
		weight := (2 - sp) + 2*(sp-1)*(n-rank)/(n-1)
		if weight < 0 {
			weight = 0
		}
		rankWeights[i] = weight
		total += weight
	}
	
	point := ga.rng.Float64() * total
	cumulative := 0.0
	
	for i, idx := range viable {
		cumulative += rankWeights[i]
		if cumulative >= point {
			return ga.population[idx]
		}
	}
	
	return ga.population[viable[0]]
}

func (ga *GA) stochasticUniversalSamplingSelect() Individual {
	viable := make([]int, 0, len(ga.population))
	for i, ind := range ga.population {
		if ind.Viable {
			viable = append(viable, i)
		}
	}
	
	if len(viable) < 2 {
		return ga.population[viable[0]]
	}
	
	totalFitness := 0.0
	for _, idx := range viable {
		totalFitness += ga.population[idx].ScaledFitness
	}
	
	if totalFitness <= 0 {
		return ga.tournamentSelect()
	}
	
	n := 2
	distance := totalFitness / float64(n)
	start := ga.rng.Float64() * distance
	
	points := make([]float64, n)
	for i := range points {
		points[i] = start + float64(i)*distance
	}
	
	selected := -1
	keep := ga.rng.Intn(n)
	
	cumulative := 0.0
	pointIdx := 0
	
	for _, idx := range viable {
		cumulative += ga.population[idx].ScaledFitness
		for pointIdx < n && cumulative >= points[pointIdx] {
			if pointIdx == keep {
				selected = idx
			}
			pointIdx++
		}
	}
	
	if selected >= 0 {
		return ga.population[selected]
	}
	
	return ga.population[viable[0]]
}

func (ga *GA) crossover(parent1, parent2 Individual) Individual {
	child := Individual{
		Genes: make([]float64, len(parent1.Genes)),
	}
	
	if ga.rng.Float64() < ga.config.CrossoverRate {
		point := ga.rng.Intn(len(parent1.Genes))
		for i := range child.Genes {
			if i < point {
				child.Genes[i] = parent1.Genes[i]
			} else {
				child.Genes[i] = parent2.Genes[i]
			}
		}
	} else {
		copy(child.Genes, parent1.Genes)
	}
	
	return child
}

func (ga *GA) mutate(ind *Individual) {
	for i := range ind.Genes {
		if ga.rng.Float64() < ga.config.MutationRate {
			vr := ga.varRanges[i]
			ind.Genes[i] = vr.Min + ga.rng.Float64()*(vr.Max-vr.Min)
		}
	}
}

func (ga *GA) sortPopulation() {
	sort.Slice(ga.population, func(i, j int) bool {
		a := ga.population[i]
		b := ga.population[j]
		if a.Viable && b.Viable {
			return a.RawFitness < b.RawFitness
		}
		return a.Viable
	})
}

func (ga *GA) createNewPopulation() {
	newPop := make([]Individual, 0, ga.config.PopulationSize)
	
	ga.sortPopulation()
	
	for i := 0; i < ga.config.ElitismCount && i < len(ga.population); i++ {
		newPop = append(newPop, ga.population[i])
	}
	
	for len(newPop) < ga.config.PopulationSize {
		parent1 := ga.selectParent()
		parent2 := ga.selectParent()
		child := ga.crossover(parent1, parent2)
		ga.mutate(&child)
		ga.evaluateIndividual(&child)
		newPop = append(newPop, child)
	}
	
	ga.population = newPop
	ga.applyFitnessScaling()
}

func (ga *GA) updateBest() {
	ga.sortPopulation()
	if ga.population[0].Viable {
		ga.best = ga.population[0]
	}
}

func (ga *GA) Run() ([]float64, float64, error) {
	if len(ga.varRanges) == 0 {
		return nil, 0, fmt.Errorf("variable ranges cannot be empty")
	}
	if ga.objective == nil {
		return nil, 0, fmt.Errorf("objective function cannot be nil")
	}
	
	ga.initializePopulation()
	ga.evaluatePopulation()
	ga.updateBest()
	
	for generation := 0; generation < ga.config.MaxGenerations; generation++ {
		ga.createNewPopulation()
		ga.updateBest()
	}
	
	if !ga.best.Viable {
		return nil, 0, fmt.Errorf("no viable solution found")
	}
	
	return ga.best.Genes, ga.best.RawFitness, nil
}

func DefaultConfig() Config {
	return Config{
		PopulationSize:    100,
		MaxGenerations:   1000,
		MutationRate:     0.1,
		CrossoverRate:    0.8,
		ElitismCount:     5,
		TournamentSize:   3,
		SelectionMethod:  RankSelection,
		FitnessScaling:   SigmaTruncationScaling,
		ScalingFactor:    2.0,
		PressureFactor:   1.7,
	}
}

func HighPressureConfig() Config {
	return Config{
		PopulationSize:    150,
		MaxGenerations:   800,
		MutationRate:     0.08,
		CrossoverRate:    0.9,
		ElitismCount:     10,
		TournamentSize:   5,
		SelectionMethod:  RankSelection,
		FitnessScaling:   LinearScaling,
		ScalingFactor:    3.0,
		PressureFactor:   1.9,
	}
}

func NewNSGAII(objectives []ObjectiveFunc, varRanges []VariableRange, constraints []ConstraintFunc, config Config) *NSGAII {
	return &NSGAII{
		objectives:  objectives,
		constraints: constraints,
		varRanges:   varRanges,
		config:      config,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (nsga *NSGAII) initializePopulation() {
	nsga.population = make([]NSGAIIIndividual, nsga.config.PopulationSize)
	for i := range nsga.population {
		genes := make([]float64, len(nsga.varRanges))
		for j, vr := range nsga.varRanges {
			genes[j] = vr.Min + nsga.rng.Float64()*(vr.Max-vr.Min)
		}
		nsga.population[i] = NSGAIIIndividual{
			Genes:          genes,
			Objectives:     make([]float64, len(nsga.objectives)),
			Rank:           0,
			CrowdingDistance: 0,
			Viable:         false,
			DominatedCount: 0,
			Dominates:      make([]int, 0),
		}
	}
}

func (nsga *NSGAII) evaluatePopulation() {
	for i := range nsga.population {
		nsga.evaluateIndividual(&nsga.population[i])
	}
}

func (nsga *NSGAII) evaluateIndividual(ind *NSGAIIIndividual) {
	ind.Viable = true
	for _, constraint := range nsga.constraints {
		if !constraint(ind.Genes) {
			ind.Viable = false
			break
		}
	}
	
	if ind.Viable {
		for j, obj := range nsga.objectives {
			ind.Objectives[j] = obj(ind.Genes)
		}
	} else {
		for j := range ind.Objectives {
			ind.Objectives[j] = math.Inf(1)
		}
	}
}

func (nsga *NSGAII) dominates(a, b *NSGAIIIndividual) bool {
	if !a.Viable || !b.Viable {
		return a.Viable && !b.Viable
	}
	
	atLeastOneBetter := false
	for i := range a.Objectives {
		if a.Objectives[i] > b.Objectives[i] {
			return false
		}
		if a.Objectives[i] < b.Objectives[i] {
			atLeastOneBetter = true
		}
	}
	return atLeastOneBetter
}

func (nsga *NSGAII) fastNonDominatedSort(pop []NSGAIIIndividual) [][]int {
	n := len(pop)
	fronts := make([][]int, 0)
	
	for i := range pop {
		pop[i].DominatedCount = 0
		pop[i].Dominates = make([]int, 0)
	}
	
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				if nsga.dominates(&pop[i], &pop[j]) {
					pop[i].Dominates = append(pop[i].Dominates, j)
				} else if nsga.dominates(&pop[j], &pop[i]) {
					pop[i].DominatedCount++
				}
			}
		}
	}
	
	front := make([]int, 0)
	for i := range pop {
		if pop[i].DominatedCount == 0 {
			pop[i].Rank = 0
			front = append(front, i)
		}
	}
	fronts = append(fronts, front)
	
	currentFront := 0
	for len(fronts[currentFront]) > 0 {
		nextFront := make([]int, 0)
		for _, i := range fronts[currentFront] {
			for _, j := range pop[i].Dominates {
				pop[j].DominatedCount--
				if pop[j].DominatedCount == 0 {
					pop[j].Rank = currentFront + 1
					nextFront = append(nextFront, j)
				}
			}
		}
		currentFront++
		if len(nextFront) > 0 {
			fronts = append(fronts, nextFront)
		}
	}
	
	return fronts
}

func (nsga *NSGAII) calculateCrowdingDistance(pop []NSGAIIIndividual, front []int) {
	n := len(front)
	if n == 0 {
		return
	}
	
	for _, idx := range front {
		pop[idx].CrowdingDistance = 0
	}
	
	numObj := len(nsga.objectives)
	for obj := 0; obj < numObj; obj++ {
		sort.Slice(front, func(i, j int) bool {
			return pop[front[i]].Objectives[obj] < pop[front[j]].Objectives[obj]
		})
		
		pop[front[0]].CrowdingDistance = math.Inf(1)
		pop[front[n-1]].CrowdingDistance = math.Inf(1)
		
		if n > 2 {
			min := pop[front[0]].Objectives[obj]
			max := pop[front[n-1]].Objectives[obj]
			rangeVal := max - min
			
			if rangeVal > 1e-10 {
				for i := 1; i < n-1; i++ {
					dist := (pop[front[i+1]].Objectives[obj] - pop[front[i-1]].Objectives[obj]) / rangeVal
					pop[front[i]].CrowdingDistance += dist
				}
			}
		}
	}
}

func (nsga *NSGAII) crowdedComparison(a, b *NSGAIIIndividual) bool {
	if a.Rank != b.Rank {
		return a.Rank < b.Rank
	}
	return a.CrowdingDistance > b.CrowdingDistance
}

func (nsga *NSGAII) tournamentSelect(pop []NSGAIIIndividual) NSGAIIIndividual {
	bestIdx := nsga.rng.Intn(len(pop))
	for i := 1; i < nsga.config.TournamentSize; i++ {
		idx := nsga.rng.Intn(len(pop))
		if nsga.crowdedComparison(&pop[idx], &pop[bestIdx]) {
			bestIdx = idx
		}
	}
	return pop[bestIdx]
}

func (nsga *NSGAII) crossover(parent1, parent2 NSGAIIIndividual) NSGAIIIndividual {
	child := NSGAIIIndividual{
		Genes:          make([]float64, len(parent1.Genes)),
		Objectives:     make([]float64, len(nsga.objectives)),
		Rank:           0,
		CrowdingDistance: 0,
		Viable:         false,
		DominatedCount: 0,
		Dominates:      make([]int, 0),
	}
	
	if nsga.rng.Float64() < nsga.config.CrossoverRate {
		point := nsga.rng.Intn(len(parent1.Genes))
		for i := range child.Genes {
			if i < point {
				child.Genes[i] = parent1.Genes[i]
			} else {
				child.Genes[i] = parent2.Genes[i]
			}
		}
	} else {
		copy(child.Genes, parent1.Genes)
	}
	
	return child
}

func (nsga *NSGAII) mutate(ind *NSGAIIIndividual) {
	for i := range ind.Genes {
		if nsga.rng.Float64() < nsga.config.MutationRate {
			vr := nsga.varRanges[i]
			ind.Genes[i] = vr.Min + nsga.rng.Float64()*(vr.Max-vr.Min)
		}
	}
}

func (nsga *NSGAII) createOffspring(pop []NSGAIIIndividual) []NSGAIIIndividual {
	offspring := make([]NSGAIIIndividual, 0, nsga.config.PopulationSize)
	
	for len(offspring) < nsga.config.PopulationSize {
		parent1 := nsga.tournamentSelect(pop)
		parent2 := nsga.tournamentSelect(pop)
		child := nsga.crossover(parent1, parent2)
		nsga.mutate(&child)
		nsga.evaluateIndividual(&child)
		offspring = append(offspring, child)
	}
	
	return offspring
}

func (nsga *NSGAII) selectNextGeneration(combined []NSGAIIIndividual) []NSGAIIIndividual {
	fronts := nsga.fastNonDominatedSort(combined)
	
	nextGen := make([]NSGAIIIndividual, 0, nsga.config.PopulationSize)
	frontIdx := 0
	
	for frontIdx < len(fronts) && len(nextGen)+len(fronts[frontIdx]) <= nsga.config.PopulationSize {
		nsga.calculateCrowdingDistance(combined, fronts[frontIdx])
		for _, idx := range fronts[frontIdx] {
			nextGen = append(nextGen, combined[idx])
		}
		frontIdx++
	}
	
	if frontIdx < len(fronts) && len(nextGen) < nsga.config.PopulationSize {
		nsga.calculateCrowdingDistance(combined, fronts[frontIdx])
		front := fronts[frontIdx]
		sort.Slice(front, func(i, j int) bool {
			return combined[front[i]].CrowdingDistance > combined[front[j]].CrowdingDistance
		})
		
		remaining := nsga.config.PopulationSize - len(nextGen)
		for i := 0; i < remaining && i < len(front); i++ {
			nextGen = append(nextGen, combined[front[i]])
		}
	}
	
	return nextGen
}

func (nsga *NSGAII) Run() (MultiObjectiveResult, error) {
	if len(nsga.varRanges) == 0 {
		return MultiObjectiveResult{}, fmt.Errorf("variable ranges cannot be empty")
	}
	if len(nsga.objectives) == 0 {
		return MultiObjectiveResult{}, fmt.Errorf("objective functions cannot be empty")
	}
	
	nsga.initializePopulation()
	nsga.evaluatePopulation()
	
	for generation := 0; generation < nsga.config.MaxGenerations; generation++ {
		offspring := nsga.createOffspring(nsga.population)
		
		combined := make([]NSGAIIIndividual, 0, len(nsga.population)+len(offspring))
		combined = append(combined, nsga.population...)
		combined = append(combined, offspring...)
		
		nsga.population = nsga.selectNextGeneration(combined)
	}
	
	result := nsga.getParetoFront()
	return result, nil
}

func (nsga *NSGAII) getParetoFront() MultiObjectiveResult {
	popCopy := make([]NSGAIIIndividual, len(nsga.population))
	copy(popCopy, nsga.population)
	
	fronts := nsga.fastNonDominatedSort(popCopy)
	
	result := MultiObjectiveResult{
		Solutions:  make([][]float64, 0),
		Objectives: make([][]float64, 0),
	}
	
	if len(fronts) > 0 {
		for _, idx := range fronts[0] {
			if popCopy[idx].Viable {
				sol := make([]float64, len(popCopy[idx].Genes))
				copy(sol, popCopy[idx].Genes)
				obj := make([]float64, len(popCopy[idx].Objectives))
				copy(obj, popCopy[idx].Objectives)
				result.Solutions = append(result.Solutions, sol)
				result.Objectives = append(result.Objectives, obj)
			}
		}
	}
	
	return result
}

func DefaultNSGAIIConfig() Config {
	return Config{
		PopulationSize:  100,
		MaxGenerations:  500,
		MutationRate:    0.1,
		CrossoverRate:   0.9,
		TournamentSize:  2,
	}
}
