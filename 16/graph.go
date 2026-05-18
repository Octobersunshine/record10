package main

import (
	"container/heap"
	"math"
)

type Graph struct {
	nodes map[int]bool
	edges map[int]map[int]int
}

func NewGraph() *Graph {
	return &Graph{
		nodes: make(map[int]bool),
		edges: make(map[int]map[int]int),
	}
}

func (g *Graph) AddNode(node int) {
	g.nodes[node] = true
	if _, exists := g.edges[node]; !exists {
		g.edges[node] = make(map[int]int)
	}
}

func (g *Graph) AddEdge(from, to, weight int) {
	g.AddNode(from)
	g.AddNode(to)
	g.edges[from][to] = weight
}

func (g *Graph) AddUndirectedEdge(from, to, weight int) {
	g.AddEdge(from, to, weight)
	g.AddEdge(to, from, weight)
}

type Item struct {
	node     int
	distance int
	index    int
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].distance < pq[j].distance
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*Item)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

func (g *Graph) Dijkstra(source int) map[int]int {
	distances := make(map[int]int)
	for node := range g.nodes {
		distances[node] = math.MaxInt32
	}
	distances[source] = 0

	pq := &PriorityQueue{}
	heap.Init(pq)
	heap.Push(pq, &Item{node: source, distance: 0})

	visited := make(map[int]bool)

	for pq.Len() > 0 {
		current := heap.Pop(pq).(*Item)
		currentNode := current.node

		if visited[currentNode] {
			continue
		}
		visited[currentNode] = true

		for neighbor, weight := range g.edges[currentNode] {
			if !visited[neighbor] {
				newDist := distances[currentNode] + weight
				if newDist < distances[neighbor] {
					distances[neighbor] = newDist
					heap.Push(pq, &Item{node: neighbor, distance: newDist})
				}
			}
		}
	}

	return distances
}

type PathQuery struct {
	Source      int
	Destination int
}

type PathResult struct {
	Source      int
	Destination int
	Distance    int
	Reachable   bool
}

func (g *Graph) BatchShortestPaths(queries []PathQuery) []PathResult {
	cache := make(map[int]map[int]int)
	results := make([]PathResult, len(queries))

	for i, query := range queries {
		if _, exists := cache[query.Source]; !exists {
			cache[query.Source] = g.Dijkstra(query.Source)
		}

		distances := cache[query.Source]
		distance := distances[query.Destination]

		results[i] = PathResult{
			Source:      query.Source,
			Destination: query.Destination,
			Distance:    distance,
			Reachable:   distance != math.MaxInt32,
		}
	}

	return results
}

func (g *Graph) GetNodes() []int {
	nodes := make([]int, 0, len(g.nodes))
	for node := range g.nodes {
		nodes = append(nodes, node)
	}
	return nodes
}

func (g *Graph) GetEdges() [][3]int {
	edges := make([][3]int, 0)
	for from, neighbors := range g.edges {
		for to, weight := range neighbors {
			edges = append(edges, [3]int{from, to, weight})
		}
	}
	return edges
}
