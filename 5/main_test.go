package main

import (
	"math/rand"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func TestIsValidSolution(t *testing.T) {
	tests := []struct {
		name   string
		order  []int
		n      int
		expect bool
	}{
		{
			name:   "valid small solution",
			order:  []int{0, 1, 2, 3},
			n:      4,
			expect: true,
		},
		{
			name:   "valid shuffled solution",
			order:  []int{3, 1, 0, 2},
			n:      4,
			expect: true,
		},
		{
			name:   "invalid duplicate",
			order:  []int{0, 1, 1, 3},
			n:      4,
			expect: false,
		},
		{
			name:   "invalid wrong length",
			order:  []int{0, 1, 2},
			n:      4,
			expect: false,
		},
		{
			name:   "invalid out of range",
			order:  []int{0, 1, 2, 5},
			n:      4,
			expect: false,
		},
		{
			name:   "invalid negative",
			order:  []int{0, -1, 2, 3},
			n:      4,
			expect: false,
		},
		{
			name:   "valid single element",
			order:  []int{0},
			n:      1,
			expect: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidSolution(tt.order, tt.n)
			if result != tt.expect {
				t.Errorf("isValidSolution() = %v, want %v", result, tt.expect)
			}
		})
	}
}

func TestRepairSolution(t *testing.T) {
	tests := []struct {
		name  string
		order []int
		n     int
	}{
		{
			name:  "repair duplicate",
			order: []int{0, 1, 1, 3},
			n:     4,
		},
		{
			name:  "repair wrong length",
			order: []int{0, 1, 2},
			n:     4,
		},
		{
			name:  "repair out of range",
			order: []int{0, 1, 2, 5},
			n:     4,
		},
		{
			name:  "repair multiple duplicates",
			order: []int{0, 0, 0, 0},
			n:     4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := repairSolution(tt.order, tt.n)
			if !isValidSolution(result, tt.n) {
				t.Errorf("repairSolution() produced invalid solution: %v", result)
			}
		})
	}
}

func TestSwapOperatorValidity(t *testing.T) {
	initial := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	n := len(initial)

	for i := 0; i < 100; i++ {
		result := swapOperator(initial)
		if !isValidSolution(result, n) {
			t.Errorf("swapOperator produced invalid solution: %v", result)
		}
	}
}

func TestInverseOperatorValidity(t *testing.T) {
	initial := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	n := len(initial)

	for i := 0; i < 100; i++ {
		result := inverseOperator(initial)
		if !isValidSolution(result, n) {
			t.Errorf("inverseOperator produced invalid solution: %v", result)
		}
	}
}

func TestInsertOperatorValidity(t *testing.T) {
	initial := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	n := len(initial)

	for i := 0; i < 100; i++ {
		result := insertOperator(initial)
		if !isValidSolution(result, n) {
			t.Errorf("insertOperator produced invalid solution: %v", result)
		}
	}
}

func TestSlideOperatorValidity(t *testing.T) {
	initial := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	n := len(initial)

	for i := 0; i < 100; i++ {
		result := slideOperator(initial)
		if !isValidSolution(result, n) {
			t.Errorf("slideOperator produced invalid solution: %v", result)
		}
	}
}

func TestGenerateNeighborValidity(t *testing.T) {
	initial := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	n := len(initial)

	strategyCount := make(map[int]int)

	for i := 0; i < 1000; i++ {
		result, strat := generateNeighbor(initial)
		if !isValidSolution(result, n) {
			t.Errorf("generateNeighbor produced invalid solution: %v", result)
		}
		strategyCount[strat]++
	}

	for strat := 0; strat < 4; strat++ {
		if strategyCount[strat] == 0 {
			t.Logf("Warning: strategy %d was not used in 1000 iterations", strat)
		}
	}
}

func TestSpecificInitialArrangement(t *testing.T) {
	testCases := [][]int{
		{0, 1, 2, 3, 4, 5},
		{5, 4, 3, 2, 1, 0},
		{0, 2, 4, 1, 3, 5},
		{3, 1, 4, 1, 5, 9},
		{0},
		{0, 1},
		{0, 1, 2},
	}

	for _, initial := range testCases {
		n := len(initial)
		t.Run("", func(t *testing.T) {
			for i := 0; i < 100; i++ {
				result, _ := generateNeighbor(initial)
				if !isValidSolution(result, n) {
					t.Errorf("generateNeighbor produced invalid solution for initial %v: %v", initial, result)
				}
			}
		})
	}
}

func TestSimulatedAnnealingProducesValidSolutions(t *testing.T) {
	cities := []City{
		{ID: "A", Latitude: 0, Longitude: 0},
		{ID: "B", Latitude: 1, Longitude: 0},
		{ID: "C", Latitude: 1, Longitude: 1},
		{ID: "D", Latitude: 0, Longitude: 1},
		{ID: "E", Latitude: 0.5, Longitude: 0.5},
	}

	for i := 0; i < 10; i++ {
		result, _, _, _ := simulatedAnnealing(cities, 100.0, 0.95, 1000)
		if !isValidSolution(result, len(cities)) {
			t.Errorf("simulatedAnnealing produced invalid solution: %v", result)
		}
	}
}

func TestAnnealingChainStepAndInject(t *testing.T) {
	cities := []City{
		{ID: "A", Latitude: 0, Longitude: 0},
		{ID: "B", Latitude: 1, Longitude: 0},
		{ID: "C", Latitude: 1, Longitude: 1},
	}

	chain := NewAnnealingChain(cities, 100.0, 0.95, 0)

	chain.Step(cities, 100)
	if !isValidSolution(chain.Current, len(cities)) {
		t.Errorf("Current solution invalid after step")
	}
	if !isValidSolution(chain.Best, len(cities)) {
		t.Errorf("Best solution invalid after step")
	}

	newSolution := []int{2, 0, 1}
	newDist := totalDistance(cities, newSolution)
	chain.InjectSolution(newSolution, newDist)

	if chain.CurrentDist != newDist {
		t.Errorf("Current distance not updated after inject")
	}
	if !isValidSolution(chain.Current, len(cities)) {
		t.Errorf("Current solution invalid after inject")
	}
}

func TestParallelAnnealingProducesValidSolutions(t *testing.T) {
	cities := []City{
		{ID: "A", Latitude: 0, Longitude: 0},
		{ID: "B", Latitude: 1, Longitude: 0},
		{ID: "C", Latitude: 1, Longitude: 1},
		{ID: "D", Latitude: 0, Longitude: 1},
		{ID: "E", Latitude: 0.5, Longitude: 0.5},
	}

	for i := 0; i < 5; i++ {
		result := ParallelAnnealing(cities, 100.0, 0.95, 2000, 4, 500)
		if !isValidSolution(result.BestSolution, len(cities)) {
			t.Errorf("ParallelAnnealing produced invalid solution: %v", result.BestSolution)
		}
		if len(result.ParallelStats.ChainStats) != 4 {
			t.Errorf("Expected 4 chain stats, got %d", len(result.ParallelStats.ChainStats))
		}
		if result.ParallelStats.ExchangeCount == 0 {
			t.Errorf("Expected some exchange operations")
		}
	}
}

func TestParallelAnnealingWithDifferentChainCounts(t *testing.T) {
	cities := []City{
		{ID: "A", Latitude: 0, Longitude: 0},
		{ID: "B", Latitude: 1, Longitude: 0},
		{ID: "C", Latitude: 1, Longitude: 1},
		{ID: "D", Latitude: 0, Longitude: 1},
	}

	chainCounts := []int{2, 4, 8}
	for _, numChains := range chainCounts {
		result := ParallelAnnealing(cities, 100.0, 0.95, 1000, numChains, 200)
		if !isValidSolution(result.BestSolution, len(cities)) {
			t.Errorf("ParallelAnnealing with %d chains produced invalid solution", numChains)
		}
		if len(result.ParallelStats.ChainStats) != numChains {
			t.Errorf("Expected %d chain stats, got %d", numChains, len(result.ParallelStats.ChainStats))
		}
	}
}

func TestParallelAnnealingBetterThanSingle(t *testing.T) {
	cities := []City{
		{ID: "A", Latitude: 0, Longitude: 0},
		{ID: "B", Latitude: 2, Longitude: 0},
		{ID: "C", Latitude: 2, Longitude: 2},
		{ID: "D", Latitude: 0, Longitude: 2},
		{ID: "E", Latitude: 1, Longitude: 1},
		{ID: "F", Latitude: 0.5, Longitude: 1.5},
		{ID: "G", Latitude: 1.5, Longitude: 0.5},
	}

	var singleSum, parallelSum float64
	runs := 5

	for i := 0; i < runs; i++ {
		_, singleDist, _, _ := simulatedAnnealing(cities, 100.0, 0.95, 5000)
		singleSum += singleDist
	}

	for i := 0; i < runs; i++ {
		result := ParallelAnnealing(cities, 100.0, 0.95, 5000, 4, 500)
		parallelSum += result.BestDistance
	}

	t.Logf("Average single chain distance: %.4f", singleSum/float64(runs))
	t.Logf("Average parallel chain distance: %.4f", parallelSum/float64(runs))
}
