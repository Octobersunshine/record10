package main

import (
	"container/heap"
	"math"
	"sync"
)

type WeightUpdate struct {
	From        int
	To          int
	NewWeight   int
	OldWeight   int
	Timestamp   int64
}

type CachedDistance struct {
	Distances    []int
	Version      int64
	AffectedFrom map[int]bool
	AffectedTo   map[int]bool
}

type DynamicGraph struct {
	*CompactGraph
	version        int64
	edgeWeights    map[[2]int]int
	cache          map[int]*CachedDistance
	cacheMutex     sync.RWMutex
	updateHistory  []WeightUpdate
	maxHistorySize int
}

func NewDynamicGraph() *DynamicGraph {
	return &DynamicGraph{
		CompactGraph:   NewCompactGraph(),
		edgeWeights:    make(map[[2]int]int),
		cache:          make(map[int]*CachedDistance),
		maxHistorySize: 1000,
	}
}

func (g *DynamicGraph) AddEdge(from, to, weight int) {
	g.CompactGraph.AddEdge(from, to, weight)
	key := [2]int{from, to}
	g.edgeWeights[key] = weight
}

func (g *DynamicGraph) AddUndirectedEdge(from, to, weight int) {
	g.AddEdge(from, to, weight)
	g.AddEdge(to, from, weight)
}

func (g *DynamicGraph) GetEdgeWeight(from, to int) (int, bool) {
	fromID, fromOk := g.GetInternalID(from)
	toID, toOk := g.GetInternalID(to)
	if !fromOk || !toOk {
		return 0, false
	}

	for _, edge := range g.adjList[fromID] {
		if edge.To == toID {
			return edge.Weight, true
		}
	}
	return 0, false
}

func (g *DynamicGraph) UpdateEdgeWeight(from, to, newWeight int) (WeightUpdate, bool) {
	fromID, fromOk := g.GetInternalID(from)
	toID, toOk := g.GetInternalID(to)
	if !fromOk || !toOk {
		return WeightUpdate{}, false
	}

	found := false
	oldWeight := 0
	for i, edge := range g.adjList[fromID] {
		if edge.To == toID {
			oldWeight = edge.Weight
			g.adjList[fromID][i].Weight = newWeight
			found = true
			break
		}
	}

	if !found {
		return WeightUpdate{}, false
	}

	g.version++
	update := WeightUpdate{
		From:      from,
		To:        to,
		NewWeight: newWeight,
		OldWeight: oldWeight,
		Timestamp: g.version,
	}

	g.updateHistory = append(g.updateHistory, update)
	if len(g.updateHistory) > g.maxHistorySize {
		g.updateHistory = g.updateHistory[1:]
	}

	g.invalidateCacheForUpdate(fromID, toID, oldWeight, newWeight)

	return update, true
}

func (g *DynamicGraph) invalidateCacheForUpdate(fromID, toID, oldWeight, newWeight int) {
	g.cacheMutex.Lock()
	defer g.cacheMutex.Unlock()

	weightDelta := newWeight - oldWeight

	for sourceID, cached := range cached {
		if weightDelta > 0 {
			oldPathWeight := cached.Distances[fromID] + oldWeight
			if cached.Distances[toID] == oldPathWeight {
				cached.AffectedTo[toID] = true
			}
		} else {
			if cached.Distances[fromID] != math.MaxInt32 {
				cached.AffectedFrom[fromID] = true
				cached.AffectedTo[toID] = true
			}
		}
	}
}

type IncrementalItem struct {
	node     int
	distance int
	index    int
}

type IncrementalPQ []*IncrementalItem

func (pq IncrementalPQ) Len() int { return len(pq) }

func (pq IncrementalPQ) Less(i, j int) bool {
	return pq[i].distance < pq[j].distance
}

func (pq IncrementalPQ) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *IncrementalPQ) Push(x interface{}) {
	n := len(*pq)
	item := x.(*IncrementalItem)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *IncrementalPQ) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

func (g *DynamicGraph) IncrementalDijkstra(source int) ([]int, bool) {
	sourceID, ok := g.GetInternalID(source)
	if !ok {
		return nil, false
	}

	g.cacheMutex.RLock()
	cached, exists := g.cache[sourceID]
	g.cacheMutex.RUnlock()

	if !exists {
		dist := g.fullDijkstra(sourceID)
		g.cacheMutex.Lock()
		g.cache[sourceID] = &CachedDistance{
			Distances:    dist,
			Version:      g.version,
			AffectedFrom: make(map[int]bool),
			AffectedTo:   make(map[int]bool),
		}
		g.cacheMutex.Unlock()
		return dist, true
	}

	if len(cached.AffectedFrom) > 0 || len(cached.AffectedTo) > 0 {
		updated := g.updateDistancesIncrementally(cached, sourceID)
		if updated {
			cached.Version = g.version
			cached.AffectedFrom = make(map[int]bool)
			cached.AffectedTo = make(map[int]bool)
		}
	}

	return cached.Distances, true
}

func (g *DynamicGraph) fullDijkstra(sourceID int) []int {
	dist := make([]int, g.nodeCount)
	for i := range dist {
		dist[i] = math.MaxInt32
	}
	dist[sourceID] = 0

	visited := make([]bool, g.nodeCount)
	pq := &IncrementalPQ{}
	heap.Init(pq)
	heap.Push(pq, &IncrementalItem{node: sourceID, distance: 0})

	for pq.Len() > 0 {
		current := heap.Pop(pq).(*IncrementalItem)
		u := current.node

		if visited[u] {
			continue
		}
		visited[u] = true

		for _, edge := range g.adjList[u] {
			v := edge.To
			if !visited[v] {
				newDist := dist[u] + edge.Weight
				if newDist < dist[v] {
					dist[v] = newDist
					heap.Push(pq, &IncrementalItem{node: v, distance: newDist})
				}
			}
		}
	}

	return dist
}

func (g *DynamicGraph) updateDistancesIncrementally(cached *CachedDistance, sourceID int) bool {
	if len(cached.AffectedTo) > 0 {
		return g.handleWeightIncrease(cached)
	}
	if len(cached.AffectedFrom) > 0 {
		return g.handleWeightDecrease(cached, sourceID)
	}
	return false
}

func (g *DynamicGraph) handleWeightIncrease(cached *CachedDistance) bool {
	inQueue := make([]bool, g.nodeCount)
	pq := &IncrementalPQ{}
	heap.Init(pq)

	for nodeID := range cached.AffectedTo {
		heap.Push(pq, &IncrementalItem{node: nodeID, distance: cached.Distances[nodeID]})
		inQueue[nodeID] = true
	}

	updated := false
	for pq.Len() > 0 {
		current := heap.Pop(pq).(*IncrementalItem)
		u := current.node
		inQueue[u] = false

		minDist := math.MaxInt32
		for _, edge := range g.adjList[u] {
			if cached.Distances[edge.To] != math.MaxInt32 {
				candidate := cached.Distances[edge.To] + edge.Weight
				if candidate < minDist {
					minDist = candidate
				}
			}
		}

		if minDist > cached.Distances[u] {
			cached.Distances[u] = minDist
			updated = true

			for _, edge := range g.adjList[u] {
				v := edge.To
				newDist := cached.Distances[u] + edge.Weight
				if newDist < cached.Distances[v] {
					cached.Distances[v] = newDist
					if !inQueue[v] {
						heap.Push(pq, &IncrementalItem{node: v, distance: newDist})
						inQueue[v] = true
					}
				}
			}
		}
	}

	return updated
}

func (g *DynamicGraph) handleWeightDecrease(cached *CachedDistance, sourceID int) bool {
	pq := &IncrementalPQ{}
	heap.Init(pq)

	for startID := range cached.AffectedFrom {
		if cached.Distances[startID] != math.MaxInt32 {
			for _, edge := range g.adjList[startID] {
				v := edge.To
				newDist := cached.Distances[startID] + edge.Weight
				if newDist < cached.Distances[v] {
					cached.Distances[v] = newDist
					heap.Push(pq, &IncrementalItem{node: v, distance: newDist})
				}
			}
		}
	}

	updated := pq.Len() > 0

	for pq.Len() > 0 {
		current := heap.Pop(pq).(*IncrementalItem)
		u := current.node

		if current.distance > cached.Distances[u] {
			continue
		}

		for _, edge := range g.adjList[u] {
			v := edge.To
			newDist := cached.Distances[u] + edge.Weight
			if newDist < cached.Distances[v] {
				cached.Distances[v] = newDist
				heap.Push(pq, &IncrementalItem{node: v, distance: newDist})
			}
		}
	}

	return updated
}

func (g *DynamicGraph) DynamicBatchShortestPaths(queries []CompactPathQuery, maxCacheSize int) []CompactPathResult {
	if g.nodeCount == 0 {
		return nil
	}

	results := make([]CompactPathResult, len(queries))
	cacheList := make([]int, 0, maxCacheSize)

	for i, query := range queries {
		srcID, srcExists := g.GetInternalID(query.Source)
		dstID, dstExists := g.GetInternalID(query.Destination)

		if !srcExists || !dstExists {
			results[i] = CompactPathResult{
				Source:      query.Source,
				Destination: query.Destination,
				Distance:    math.MaxInt32,
				Reachable:   false,
			}
			continue
		}

		dist, ok := g.IncrementalDijkstra(query.Source)
		if !ok {
			results[i] = CompactPathResult{
				Source:      query.Source,
				Destination: query.Destination,
				Distance:    math.MaxInt32,
				Reachable:   false,
			}
			continue
		}

		g.cacheMutex.RLock()
		_, inCache := g.cache[srcID]
		g.cacheMutex.RUnlock()

		if !inCache && maxCacheSize > 0 {
			g.cacheMutex.Lock()
			if len(g.cache) >= maxCacheSize {
				oldest := cacheList[0]
				delete(g.cache, oldest)
				cacheList = cacheList[1:]
			}
			g.cache[srcID] = &CachedDistance{
				Distances:    dist,
				Version:      g.version,
				AffectedFrom: make(map[int]bool),
				AffectedTo:   make(map[int]bool),
			}
			cacheList = append(cacheList, srcID)
			g.cacheMutex.Unlock()
		}

		distance := dist[dstID]
		results[i] = CompactPathResult{
			Source:      query.Source,
			Destination: query.Destination,
			Distance:    distance,
			Reachable:   distance != math.MaxInt32,
		}
	}

	return results
}

func (g *DynamicGraph) ClearCache() {
	g.cacheMutex.Lock()
	defer g.cacheMutex.Unlock()
	g.cache = make(map[int]*CachedDistance)
}

func (g *DynamicGraph) GetCacheSize() int {
	g.cacheMutex.RLock()
	defer g.cacheMutex.RUnlock()
	return len(g.cache)
}

func (g *DynamicGraph) GetVersion() int64 {
	return g.version
}

func (g *DynamicGraph) GetUpdateHistory() []WeightUpdate {
	return g.updateHistory
}

func BuildDynamicGraphFromData(nodes []int, edges [][3]int, isDirected bool) *DynamicGraph {
	g := NewDynamicGraph()

	for _, node := range nodes {
		g.AddNode(node)
	}

	for _, edge := range edges {
		if isDirected {
			g.AddEdge(edge[0], edge[1], edge[2])
		} else {
			g.AddUndirectedEdge(edge[0], edge[1], edge[2])
		}
	}

	g.SortEdges()
	return g
}
