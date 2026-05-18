package main

import (
	"container/heap"
	"math"
	"sort"
	"sync"
)

type Edge struct {
	To     int
	Weight int
}

type CompactGraph struct {
	nodeIDMap  map[int]int
	invIDMap   []int
	adjList    [][]Edge
	nodeCount  int
	edgeCount  int
}

func NewCompactGraph() *CompactGraph {
	return &CompactGraph{
		nodeIDMap: make(map[int]int),
		invIDMap:  make([]int, 0),
		adjList:   make([][]Edge, 0),
	}
}

func (g *CompactGraph) getOrCreateID(node int) int {
	if id, exists := g.nodeIDMap[node]; exists {
		return id
	}
	id := len(g.invIDMap)
	g.nodeIDMap[node] = id
	g.invIDMap = append(g.invIDMap, node)
	g.adjList = append(g.adjList, make([]Edge, 0, 8))
	g.nodeCount++
	return id
}

func (g *CompactGraph) AddNode(node int) {
	g.getOrCreateID(node)
}

func (g *CompactGraph) AddEdge(from, to, weight int) {
	fromID := g.getOrCreateID(from)
	toID := g.getOrCreateID(to)
	g.adjList[fromID] = append(g.adjList[fromID], Edge{To: toID, Weight: weight})
	g.edgeCount++
}

func (g *CompactGraph) AddUndirectedEdge(from, to, weight int) {
	g.AddEdge(from, to, weight)
	g.AddEdge(to, from, weight)
}

func (g *CompactGraph) SortEdges() {
	for i := range g.adjList {
		sort.Slice(g.adjList[i], func(a, b int) bool {
			return g.adjList[i][a].To < g.adjList[i][b].To
		})
	}
}

func (g *CompactGraph) GetInternalID(externalID int) (int, bool) {
	id, exists := g.nodeIDMap[externalID]
	return id, exists
}

func (g *CompactGraph) GetExternalID(internalID int) (int, bool) {
	if internalID < 0 || internalID >= len(g.invIDMap) {
		return 0, false
	}
	return g.invIDMap[internalID], true
}

type CompactItem struct {
	node     int
	distance int
	index    int
}

type CompactPriorityQueue []*CompactItem

func (pq CompactPriorityQueue) Len() int { return len(pq) }

func (pq CompactPriorityQueue) Less(i, j int) bool {
	return pq[i].distance < pq[j].distance
}

func (pq CompactPriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *CompactPriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*CompactItem)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *CompactPriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

type DistancePool struct {
	pool      sync.Pool
	nodeCount int
}

func NewDistancePool(nodeCount int) *DistancePool {
	return &DistancePool{
		nodeCount: nodeCount,
		pool: sync.Pool{
			New: func() interface{} {
				dist := make([]int, nodeCount)
				return &dist
			},
		},
	}
}

func (dp *DistancePool) Get() []int {
	distPtr := dp.pool.Get().(*[]int)
	dist := *distPtr
	for i := range dist {
		dist[i] = math.MaxInt32
	}
	return dist
}

func (dp *DistancePool) Put(dist []int) {
	dp.pool.Put(&dist)
}

type CompactPathQuery struct {
	Source      int
	Destination int
}

type CompactPathResult struct {
	Source      int
	Destination int
	Distance    int
	Reachable   bool
}

func (g *CompactGraph) DijkstraWithPool(sourceID int, distPool *DistancePool, visitedPool *sync.Pool) []int {
	dist := distPool.Get()
	dist[sourceID] = 0

	visitedPtr := visitedPool.Get().(*[]bool)
	visited := *visitedPtr
	for i := range visited {
		visited[i] = false
	}

	pq := &CompactPriorityQueue{}
	heap.Init(pq)
	heap.Push(pq, &CompactItem{node: sourceID, distance: 0})

	for pq.Len() > 0 {
		current := heap.Pop(pq).(*CompactItem)
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
					heap.Push(pq, &CompactItem{node: v, distance: newDist})
				}
			}
		}
	}

	visitedPool.Put(visitedPtr)
	return dist
}

func (g *CompactGraph) BatchShortestPaths(queries []CompactPathQuery, maxCacheSize int) []CompactPathResult {
	if g.nodeCount == 0 {
		return nil
	}

	distPool := NewDistancePool(g.nodeCount)
	visitedPool := &sync.Pool{
		New: func() interface{} {
			visited := make([]bool, g.nodeCount)
			return &visited
		},
	}

	cache := make(map[int][]int)
	cacheList := make([]int, 0, maxCacheSize)

	results := make([]CompactPathResult, len(queries))

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

		var dist []int
		if cachedDist, exists := cache[srcID]; exists {
			dist = cachedDist
		} else {
			dist = g.DijkstraWithPool(srcID, distPool, visitedPool)
			
			if len(cache) >= maxCacheSize && maxCacheSize > 0 {
				oldest := cacheList[0]
				distPool.Put(cache[oldest])
				delete(cache, oldest)
				cacheList = cacheList[1:]
			}
			
			if maxCacheSize > 0 {
				cache[srcID] = dist
				cacheList = append(cacheList, srcID)
			}
		}

		distance := dist[dstID]
		results[i] = CompactPathResult{
			Source:      query.Source,
			Destination: query.Destination,
			Distance:    distance,
			Reachable:   distance != math.MaxInt32,
		}

		if _, inCache := cache[srcID]; !inCache {
			distPool.Put(dist)
		}
	}

	for _, d := range cache {
		distPool.Put(d)
	}

	return results
}

func (g *CompactGraph) BatchShortestPathsNoCache(queries []CompactPathQuery) []CompactPathResult {
	return g.BatchShortestPaths(queries, 0)
}

func (g *CompactGraph) GetNodeCount() int {
	return g.nodeCount
}

func (g *CompactGraph) GetEdgeCount() int {
	return g.edgeCount
}

func (g *CompactGraph) EstimateMemoryUsage() uint64 {
	nodeMapMem := uint64(len(g.nodeIDMap)) * (8 + 8)
	invIDMapMem := uint64(len(g.invIDMap)) * 8
	adjListMem := uint64(g.edgeCount) * (8 + 4)
	return nodeMapMem + invIDMapMem + adjListMem
}

func BuildGraphFromData(nodes []int, edges [][3]int, isDirected bool) *CompactGraph {
	g := NewCompactGraph()
	
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
