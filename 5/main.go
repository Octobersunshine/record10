package main

import (
	"math"
	"math/rand"
	"net/http"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type City struct {
	ID        string  `json:"id"`
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

type StrategyStats struct {
	Swap    int `json:"swap"`
	Inverse int `json:"inverse"`
	Insert  int `json:"insert"`
	Slide   int `json:"slide"`
}

type ChainStats struct {
	StrategyStats StrategyStats `json:"strategyStats"`
	Iterations    int           `json:"iterations"`
	BestDistance  float64       `json:"bestDistance"`
	CurrentTemp   float64       `json:"currentTemp"`
}

type ParallelStats struct {
	ChainStats    []ChainStats `json:"chainStats"`
	ExchangeCount int          `json:"exchangeCount"`
	TotalTime     string       `json:"totalTime"`
}

type TSPSolution struct {
	Path          []string     `json:"path"`
	Distance      float64      `json:"distance"`
	Iteration     int          `json:"iteration"`
	Duration      string       `json:"duration"`
	StrategyStats StrategyStats `json:"strategyStats"`
	ParallelStats ParallelStats `json:"parallelStats,omitempty"`
}

type OptimizeRequest struct {
	Cities        []City  `json:"cities" binding:"required,min=2"`
	InitialTemp   float64 `json:"initialTemp"`
	CoolingRate   float64 `json:"coolingRate"`
	Iterations    int     `json:"iterations"`
	NumChains     int     `json:"numChains"`
	ExchangeInterval int  `json:"exchangeInterval"`
	EnableParallel bool   `json:"enableParallel"`
}

type OptimizeResponse struct {
	Success    bool       `json:"success"`
	Message    string     `json:"message"`
	Solution   TSPSolution `json:"solution"`
	Parameters map[string]interface{} `json:"parameters"`
}

func distance(c1, c2 City) float64 {
	const R = 6371.0
	lat1 := c1.Latitude * math.Pi / 180
	lon1 := c1.Longitude * math.Pi / 180
	lat2 := c2.Latitude * math.Pi / 180
	lon2 := c2.Longitude * math.Pi / 180

	dlat := lat2 - lat1
	dlon := lon2 - lon1

	a := math.Sin(dlat/2)*math.Sin(dlat/2) +
		math.Cos(lat1)*math.Cos(lat2)*
			math.Sin(dlon/2)*math.Sin(dlon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return R * c
}

func isValidSolution(order []int, n int) bool {
	if len(order) != n {
		return false
	}

	seen := make(map[int]bool, n)
	for _, v := range order {
		if v < 0 || v >= n {
			return false
		}
		if seen[v] {
			return false
		}
		seen[v] = true
	}

	return len(seen) == n
}

func repairSolution(order []int, n int) []int {
	if len(order) != n {
		result := make([]int, n)
		for i := range result {
			result[i] = i
		}
		return result
	}

	seen := make(map[int]bool, n)
	missing := make([]int, 0, n)
	duplicates := make([]int, 0, n)

	for i, v := range order {
		if v < 0 || v >= n {
			duplicates = append(duplicates, i)
			continue
		}
		if seen[v] {
			duplicates = append(duplicates, i)
		} else {
			seen[v] = true
		}
	}

	for i := 0; i < n; i++ {
		if !seen[i] {
			missing = append(missing, i)
		}
	}

	result := make([]int, n)
	copy(result, order)

	for i, dupIdx := range duplicates {
		if i < len(missing) {
			result[dupIdx] = missing[i]
		}
	}

	if !isValidSolution(result, n) {
		for i := range result {
			result[i] = i
		}
	}

	return result
}

func totalDistance(cities []City, order []int) float64 {
	total := 0.0
	n := len(order)
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		total += distance(cities[order[i]], cities[order[j]])
	}
	return total
}

func swapOperator(order []int) []int {
	n := len(order)
	neighbor := make([]int, n)
	copy(neighbor, order)

	i := rand.Intn(n)
	j := rand.Intn(n)
	for i == j {
		j = rand.Intn(n)
	}

	neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
	return neighbor
}

func inverseOperator(order []int) []int {
	n := len(order)
	neighbor := make([]int, n)
	copy(neighbor, order)

	i := rand.Intn(n)
	j := rand.Intn(n)
	for i == j {
		j = rand.Intn(n)
	}

	if i > j {
		i, j = j, i
	}

	for k := 0; k <= (j-i)/2; k++ {
		neighbor[i+k], neighbor[j-k] = neighbor[j-k], neighbor[i+k]
	}

	return neighbor
}

func insertOperator(order []int) []int {
	n := len(order)
	neighbor := make([]int, 0, n)

	i := rand.Intn(n)
	j := rand.Intn(n)
	for i == j {
		j = rand.Intn(n)
	}

	for k := 0; k < n; k++ {
		if k == i {
			continue
		}
		neighbor = append(neighbor, order[k])
		if k == j {
			neighbor = append(neighbor, order[i])
		}
	}

	return neighbor
}

func slideOperator(order []int) []int {
	n := len(order)
	if n < 4 {
		return swapOperator(order)
	}

	start := rand.Intn(n - 2)
	length := rand.Intn(n-start-2) + 2
	end := start + length

	insertPos := rand.Intn(n - length + 1)
	for insertPos >= start && insertPos <= end-length {
		insertPos = rand.Intn(n - length + 1)
	}

	subPath := make([]int, length)
	copy(subPath, order[start:end])

	result := make([]int, 0, n)
	for k := 0; k < n; k++ {
		if k >= start && k < end {
			continue
		}
		if k == insertPos {
			result = append(result, subPath...)
		}
		result = append(result, order[k])
	}

	if insertPos >= end || insertPos >= len(result) {
		result = append(result, subPath...)
	}

	if len(result) != n {
		return swapOperator(order)
	}

	return result
}

func generateNeighbor(order []int) ([]int, int) {
	n := len(order)
	maxAttempts := 5

	for attempt := 0; attempt < maxAttempts; attempt++ {
		strategy := rand.Intn(4)
		var neighbor []int
		var strat int

		switch strategy {
		case 0:
			neighbor = swapOperator(order)
			strat = 0
		case 1:
			neighbor = inverseOperator(order)
			strat = 1
		case 2:
			neighbor = insertOperator(order)
			strat = 2
		case 3:
			neighbor = slideOperator(order)
			strat = 3
		default:
			neighbor = swapOperator(order)
			strat = 0
		}

		if isValidSolution(neighbor, n) {
			return neighbor, strat
		}

		repaired := repairSolution(neighbor, n)
		if isValidSolution(repaired, n) {
			return repaired, strat
		}
	}

	return swapOperator(order), 0
}

type AnnealingChain struct {
	Current     []int
	Best        []int
	CurrentDist float64
	BestDist    float64
	Temp        float64
	InitialTemp float64
	CoolingRate float64
	Iteration   int
	Stats       StrategyStats
	ID          int
}

func NewAnnealingChain(cities []City, initialTemp, coolingRate float64, id int) *AnnealingChain {
	n := len(cities)
	if n <= 1 {
		return &AnnealingChain{}
	}

	current := make([]int, n)
	for i := range current {
		current[i] = i
	}
	rand.Shuffle(n, func(i, j int) {
		current[i], current[j] = current[j], current[i]
	})

	currentDist := totalDistance(cities, current)
	best := make([]int, n)
	copy(best, current)

	return &AnnealingChain{
		Current:     current,
		Best:        best,
		CurrentDist: currentDist,
		BestDist:    currentDist,
		Temp:        initialTemp,
		InitialTemp: initialTemp,
		CoolingRate: coolingRate,
		Iteration:   0,
		Stats:       StrategyStats{},
		ID:          id,
	}
}

func (chain *AnnealingChain) Step(cities []City, steps int) {
	n := len(cities)
	if n <= 1 {
		return
	}

	for i := 0; i < steps && chain.Temp > 1e-8; i++ {
		neighbor, strategy := generateNeighbor(chain.Current)

		switch strategy {
		case 0:
			chain.Stats.Swap++
		case 1:
			chain.Stats.Inverse++
		case 2:
			chain.Stats.Insert++
		case 3:
			chain.Stats.Slide++
		}

		neighborDist := totalDistance(cities, neighbor)
		delta := neighborDist - chain.CurrentDist

		if delta < 0 || rand.Float64() < math.Exp(-delta/chain.Temp) {
			copy(chain.Current, neighbor)
			chain.CurrentDist = neighborDist

			if chain.CurrentDist < chain.BestDist {
				copy(chain.Best, chain.Current)
				chain.BestDist = chain.CurrentDist
			}
		}

		chain.Temp *= chain.CoolingRate
		chain.Iteration++
	}
}

func (chain *AnnealingChain) InjectSolution(solution []int, dist float64) {
	n := len(solution)
	if !isValidSolution(solution, n) {
		return
	}

	copy(chain.Current, solution)
	chain.CurrentDist = dist

	if dist < chain.BestDist {
		copy(chain.Best, solution)
		chain.BestDist = dist
	}
}

func simulatedAnnealing(cities []City, initialTemp, coolingRate float64, maxIter int) ([]int, float64, int, StrategyStats) {
	chain := NewAnnealingChain(cities, initialTemp, coolingRate, 0)
	chain.Step(cities, maxIter)
	return chain.Best, chain.BestDist, chain.Iteration, chain.Stats
}

type ParallelAnnealingResult struct {
	BestSolution  []int
	BestDistance  float64
	TotalIter     int
	TotalStats    StrategyStats
	ParallelStats ParallelStats
}

type chainResult struct {
	chainID int
	best    []int
	bestDist float64
}

func ParallelAnnealing(cities []City, initialTemp, coolingRate float64, maxIter, numChains, exchangeInterval int) ParallelAnnealingResult {
	n := len(cities)
	if n <= 1 {
		return ParallelAnnealingResult{}
	}

	rand.Seed(time.Now().UnixNano())
	startTime := time.Now()

	chains := make([]*AnnealingChain, numChains)
	for i := range chains {
		chainTemp := initialTemp * (1.0 + float64(i)*0.2/float64(numChains))
		chainCooling := coolingRate * (1.0 - float64(i)*0.01/float64(numChains))
		chains[i] = NewAnnealingChain(cities, chainTemp, chainCooling, i)
	}

	totalIterations := 0
	exchangeCount := 0

	for totalIterations < maxIter {
		stepSize := exchangeInterval
		if remaining := maxIter - totalIterations; remaining < stepSize {
			stepSize = remaining
		}

		for _, chain := range chains {
			chain.Step(cities, stepSize)
		}
		totalIterations += stepSize

		globalBestIdx := 0
		globalBestDist := chains[0].BestDist
		for i := 1; i < numChains; i++ {
			if chains[i].BestDist < globalBestDist {
				globalBestDist = chains[i].BestDist
				globalBestIdx = i
			}
		}

		for i, chain := range chains {
			if i != globalBestIdx && rand.Float64() < 0.3 {
				chain.InjectSolution(chains[globalBestIdx].Best, globalBestDist)
			}
		}

		if numChains >= 2 && rand.Float64() < 0.5 {
			i := rand.Intn(numChains)
			j := rand.Intn(numChains)
			for i == j {
				j = rand.Intn(numChains)
			}

			if rand.Float64() < 0.5 {
				chains[i].InjectSolution(chains[j].Best, chains[j].BestDist)
			} else {
				chains[j].InjectSolution(chains[i].Best, chains[i].BestDist)
			}
		}

		exchangeCount++
	}

	globalBestIdx := 0
	globalBestDist := chains[0].BestDist
	globalBest := make([]int, n)
	copy(globalBest, chains[0].Best)
	for i := 1; i < numChains; i++ {
		if chains[i].BestDist < globalBestDist {
			globalBestDist = chains[i].BestDist
			globalBestIdx = i
			copy(globalBest, chains[i].Best)
		}
	}

	chainStats := make([]ChainStats, numChains)
	totalStats := StrategyStats{}
	for i, chain := range chains {
		chainStats[i] = ChainStats{
			StrategyStats: chain.Stats,
			Iterations:    chain.Iteration,
			BestDistance:  chain.BestDist,
			CurrentTemp:   chain.Temp,
		}
		totalStats.Swap += chain.Stats.Swap
		totalStats.Inverse += chain.Stats.Inverse
		totalStats.Insert += chain.Stats.Insert
		totalStats.Slide += chain.Stats.Slide
	}

	return ParallelAnnealingResult{
		BestSolution: globalBest,
		BestDistance: globalBestDist,
		TotalIter:    totalIterations,
		TotalStats:   totalStats,
		ParallelStats: ParallelStats{
			ChainStats:    chainStats,
			ExchangeCount: exchangeCount,
			TotalTime:     time.Since(startTime).String(),
		},
	}
}

func optimizeHandler(c *gin.Context) {
	var req OptimizeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, OptimizeResponse{
			Success: false,
			Message: "无效的请求参数: " + err.Error(),
		})
		return
	}

	initialTemp := req.InitialTemp
	if initialTemp <= 0 {
		initialTemp = 1000.0
	}

	coolingRate := req.CoolingRate
	if coolingRate <= 0 || coolingRate >= 1 {
		coolingRate = 0.995
	}

	iterations := req.Iterations
	if iterations <= 0 {
		iterations = 10000
	}

	numChains := req.NumChains
	if numChains <= 0 {
		numChains = 4
	}
	if numChains > 32 {
		numChains = 32
	}

	exchangeInterval := req.ExchangeInterval
	if exchangeInterval <= 0 {
		exchangeInterval = 1000
	}
	if exchangeInterval > iterations {
		exchangeInterval = iterations
	}

	enableParallel := req.EnableParallel

	startTime := time.Now()

	var bestOrder []int
	var bestDist float64
	var iter int
	var stats StrategyStats
	var parallelStats ParallelStats

	if enableParallel {
		result := ParallelAnnealing(req.Cities, initialTemp, coolingRate, iterations, numChains, exchangeInterval)
		bestOrder = result.BestSolution
		bestDist = result.BestDistance
		iter = result.TotalIter
		stats = result.TotalStats
		parallelStats = result.ParallelStats
	} else {
		bestOrder, bestDist, iter, stats = simulatedAnnealing(req.Cities, initialTemp, coolingRate, iterations)
	}

	duration := time.Since(startTime)

	path := make([]string, len(bestOrder))
	for i, idx := range bestOrder {
		path[i] = req.Cities[idx].ID
	}

	solution := TSPSolution{
		Path:          path,
		Distance:      bestDist,
		Iteration:     iter,
		Duration:      duration.String(),
		StrategyStats: stats,
		ParallelStats: parallelStats,
	}

	c.JSON(http.StatusOK, OptimizeResponse{
		Success: true,
		Message: "路径优化完成",
		Solution: solution,
		Parameters: map[string]interface{}{
			"initialTemp":      initialTemp,
			"coolingRate":      coolingRate,
			"iterations":       iterations,
			"enableParallel":   enableParallel,
			"numChains":        numChains,
			"exchangeInterval": exchangeInterval,
		},
	})
}

func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"time":   time.Now().Format(time.RFC3339),
	})
}

func main() {
	r := gin.Default()

	r.Use(cors.New(cors.Config{
		AllowAllOrigins:  true,
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Content-Type"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	r.GET("/health", healthHandler)
	r.POST("/api/optimize", optimizeHandler)

	r.Run(":8080")
}
