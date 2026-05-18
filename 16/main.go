package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"runtime"
	"time"
)

func main() {
	fmt.Println("=== 最短路径批量计算程序 - 动态权重版 ===")
	fmt.Println()

	DemoCompactGraph()
	fmt.Println()
	DemoDynamicGraph()
	fmt.Println()
	DemoLargeGraphSimulation()
}

func DemoCompactGraph() {
	fmt.Println("--- 使用紧凑邻接表的示例 ---")
	g := NewCompactGraph()

	g.AddUndirectedEdge(1, 2, 7)
	g.AddUndirectedEdge(1, 3, 9)
	g.AddUndirectedEdge(1, 6, 14)
	g.AddUndirectedEdge(2, 3, 10)
	g.AddUndirectedEdge(2, 4, 15)
	g.AddUndirectedEdge(3, 4, 11)
	g.AddUndirectedEdge(3, 6, 2)
	g.AddUndirectedEdge(4, 5, 6)
	g.AddUndirectedEdge(5, 6, 9)

	g.SortEdges()

	fmt.Printf("节点数: %d, 边数: %d\n", g.GetNodeCount(), g.GetEdgeCount())
	fmt.Printf("预估内存使用: ~%.2f KB\n", float64(g.EstimateMemoryUsage())/1024)
	fmt.Println()

	queries := []CompactPathQuery{
		{Source: 1, Destination: 5},
		{Source: 1, Destination: 4},
		{Source: 2, Destination: 5},
		{Source: 3, Destination: 1},
		{Source: 5, Destination: 2},
		{Source: 1, Destination: 100},
	}

	fmt.Println("查询列表:")
	for i, q := range queries {
		fmt.Printf("  %d: %d -> %d\n", i+1, q.Source, q.Destination)
	}
	fmt.Println()

	maxCacheSize := 10
	results := g.BatchShortestPaths(queries, maxCacheSize)

	fmt.Println("计算结果:")
	for i, r := range results {
		if r.Reachable {
			fmt.Printf("  %d: %d -> %d = %d\n", i+1, r.Source, r.Destination, r.Distance)
		} else {
			fmt.Printf("  %d: %d -> %d = 不可达\n", i+1, r.Source, r.Destination)
		}
	}
}

func DemoDynamicGraph() {
	fmt.Println("--- 动态权重更新与增量计算示例 ---")
	g := NewDynamicGraph()

	g.AddUndirectedEdge(1, 2, 7)
	g.AddUndirectedEdge(1, 3, 9)
	g.AddUndirectedEdge(1, 6, 14)
	g.AddUndirectedEdge(2, 3, 10)
	g.AddUndirectedEdge(2, 4, 15)
	g.AddUndirectedEdge(3, 4, 11)
	g.AddUndirectedEdge(3, 6, 2)
	g.AddUndirectedEdge(4, 5, 6)
	g.AddUndirectedEdge(5, 6, 9)

	g.SortEdges()

	fmt.Printf("初始状态 - 节点数: %d, 边数: %d, 版本: %d\n",
		g.GetNodeCount(), g.GetEdgeCount(), g.GetVersion())

	queries := []CompactPathQuery{
		{Source: 1, Destination: 5},
		{Source: 1, Destination: 4},
	}

	fmt.Println("\n第1次查询 (全量计算):")
	results := g.DynamicBatchShortestPaths(queries, 10)
	for i, r := range results {
		fmt.Printf("  查询%d: %d->%d = %d\n", i+1, r.Source, r.Destination, r.Distance)
	}
	fmt.Printf("  缓存大小: %d\n", g.GetCacheSize())

	fmt.Println("\n--- 更新边权重: 3->6 从 2 增加到 20 ---")
	update, ok := g.UpdateEdgeWeight(3, 6, 20)
	if ok {
		fmt.Printf("  更新成功: %d->%d, %d -> %d, 新版本: %d\n",
			update.From, update.To, update.OldWeight, update.NewWeight, g.GetVersion())
	}

	fmt.Println("\n第2次查询 (增量计算, 权重增加):")
	start := time.Now()
	results = g.DynamicBatchShortestPaths(queries, 10)
	elapsed := time.Since(start)
	for i, r := range results {
		fmt.Printf("  查询%d: %d->%d = %d\n", i+1, r.Source, r.Destination, r.Distance)
	}
	fmt.Printf("  耗时: %v, 缓存大小: %d\n", elapsed, g.GetCacheSize())

	fmt.Println("\n--- 更新边权重: 4->5 从 6 减少到 1 ---")
	update, ok = g.UpdateEdgeWeight(4, 5, 1)
	if ok {
		fmt.Printf("  更新成功: %d->%d, %d -> %d, 新版本: %d\n",
			update.From, update.To, update.OldWeight, update.NewWeight, g.GetVersion())
	}

	fmt.Println("\n第3次查询 (增量计算, 权重减少):")
	start = time.Now()
	results = g.DynamicBatchShortestPaths(queries, 10)
	elapsed = time.Since(start)
	for i, r := range results {
		fmt.Printf("  查询%d: %d->%d = %d\n", i+1, r.Source, r.Destination, r.Distance)
	}
	fmt.Printf("  耗时: %v, 缓存大小: %d\n", elapsed, g.GetCacheSize())

	fmt.Println("\n动态权重更新特性:")
	fmt.Println("  1. 权重增加: 只重新计算受影响的节点")
	fmt.Println("  2. 权重减少: 从受影响的边开始松弛传播")
	fmt.Println("  3. 版本追踪: 自动标记需要更新的缓存")
	fmt.Println("  4. 避免全量重跑: 显著提升更新后的查询速度")
}

func DemoLargeGraphSimulation() {
	fmt.Println("\n--- 大图模拟测试 (10万节点级别优化) ---")
	fmt.Println("注: 此演示展示内存优化架构，实际大图需更多内存")
	fmt.Println()

	nodeCount := 10000
	edgeCount := 50000
	
	fmt.Printf("正在生成模拟图: %d 节点, %d 边...\n", nodeCount, edgeCount)
	
	nodes := make([]int, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodes[i] = i*100 + 1
	}

	edges := make([][3]int, edgeCount)
	r := rand.New(rand.NewSource(42))
	for i := 0; i < edgeCount; i++ {
		from := nodes[r.Intn(nodeCount)]
		to := nodes[r.Intn(nodeCount)]
		for from == to {
			to = nodes[r.Intn(nodeCount)]
		}
		weight := r.Intn(100) + 1
		edges[i] = [3]int{from, to, weight}
	}

	start := time.Now()
	g := BuildGraphFromData(nodes, edges, false)
	buildTime := time.Since(start)

	var memStats runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memStats)
	allocMB := float64(memStats.Alloc) / 1024 / 1024

	fmt.Printf("图构建时间: %v\n", buildTime)
	fmt.Printf("节点数: %d, 边数: %d\n", g.GetNodeCount(), g.GetEdgeCount())
	fmt.Printf("预估图内存: ~%.2f KB\n", float64(g.EstimateMemoryUsage())/1024)
	fmt.Printf("实际分配内存: %.2f MB\n", allocMB)
	fmt.Println()

	queryCount := 100
	queries := make([]CompactPathQuery, queryCount)
	for i := 0; i < queryCount; i++ {
		src := nodes[r.Intn(nodeCount)]
		dst := nodes[r.Intn(nodeCount)]
		queries[i] = CompactPathQuery{Source: src, Destination: dst}
	}

	fmt.Printf("正在处理 %d 个批量查询...\n", queryCount)
	start = time.Now()
	results := g.BatchShortestPaths(queries, 50)
	queryTime := time.Since(start)

	reachableCount := 0
	for _, r := range results {
		if r.Reachable {
			reachableCount++
		}
	}

	fmt.Printf("查询处理时间: %v\n", queryTime)
	fmt.Printf("平均每个查询: %v\n", queryTime/time.Duration(queryCount))
	fmt.Printf("可达路径数: %d/%d\n", reachableCount, queryCount)
	fmt.Println()

	fmt.Println("内存优化要点:")
	fmt.Println("  1. 使用 [][]Edge 代替 map[int]map[int]int 存储邻接表")
	fmt.Println("  2. 节点ID压缩映射: 将外部ID映射到连续的内部ID")
	fmt.Println("  3. sync.Pool 复用距离数组和访问标记数组")
	fmt.Println("  4. LRU风格缓存管理: 限制缓存的源点数量")
	fmt.Println("  5. 边排序优化内存局部性")
}

func ExampleFromJSON() {
	type GraphInput struct {
		Nodes   []int              `json:"nodes"`
		Edges   [][3]int           `json:"edges"`
		Queries []CompactPathQuery `json:"queries"`
	}

	jsonStr := `{
		"nodes": [1, 2, 3, 4, 5],
		"edges": [
			[1, 2, 10],
			[2, 3, 5],
			[1, 3, 20],
			[3, 4, 1],
			[4, 5, 2]
		],
		"queries": [
			{"Source": 1, "Destination": 5},
			{"Source": 2, "Destination": 4}
		]
	}`

	var input GraphInput
	json.Unmarshal([]byte(jsonStr), &input)

	g := BuildGraphFromData(input.Nodes, input.Edges, true)
	results := g.BatchShortestPaths(input.Queries, 10)
	
	fmt.Println("\n=== JSON 输入示例结果 ===")
	for _, r := range results {
		fmt.Printf("%d->%d: %d (可达: %v)\n", r.Source, r.Destination, r.Distance, r.Reachable)
	}
}

func ComputeShortestPathsCompact(nodes []int, edges [][3]int, queries []CompactPathQuery, isDirected bool, maxCache int) []CompactPathResult {
	g := BuildGraphFromData(nodes, edges, isDirected)
	return g.BatchShortestPaths(queries, maxCache)
}
