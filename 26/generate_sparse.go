package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"time"
)

type SparseEntry struct {
	Row int     `json:"row"`
	Col int     `json:"col"`
	Val float64 `json:"val"`
}

type SparseConfig struct {
	N       int           `json:"n"`
	Entries []SparseEntry `json:"entries"`
}

type Config struct {
	SparseMatrix *SparseConfig `json:"sparse_matrix,omitempty"`
	Epsilon      float64       `json:"epsilon,omitempty"`
	MaxIter      int           `json:"max_iter,omitempty"`
	KrylovDim    int           `json:"krylov_dim,omitempty"`
}

func generateRandomWalk(n int, k int) []SparseEntry {
	var entries []SparseEntry
	for i := 0; i < n; i++ {
		neighbors := make(map[int]bool)
		neighbors[i] = true
		for len(neighbors) < k {
			j := rand.Intn(n)
			neighbors[j] = true
		}
		probs := make([]float64, 0, len(neighbors))
		total := 0.0
		for range neighbors {
			p := rand.Float64() + 0.1
			probs = append(probs, p)
			total += p
		}
		idx := 0
		for j := range neighbors {
			entries = append(entries, SparseEntry{i, j, probs[idx] / total})
			idx++
		}
	}
	return entries
}

func main() {
	rand.Seed(time.Now().UnixNano())
	n := 500
	k := 5
	fmt.Printf("生成 %d 状态的稀疏马尔可夫链，每个状态有 %d 个邻居\n", n, k)
	entries := generateRandomWalk(n, k)
	fmt.Printf("非零元素总数: %d (稀疏度: %.2f%%)\n", len(entries), 100*float64(len(entries))/float64(n*n))
	config := Config{
		SparseMatrix: &SparseConfig{
			N:       n,
			Entries: entries,
		},
		Epsilon:   1e-10,
		MaxIter:   10000,
		KrylovDim: 30,
	}
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		fmt.Printf("JSON编码错误: %v\n", err)
		return
	}
	filename := "random_walk.json"
	if err := os.WriteFile(filename, data, 0644); err != nil {
		fmt.Printf("写文件错误: %v\n", err)
		return
	}
	fmt.Printf("配置已保存到 %s\n", filename)
	fmt.Println("\n运行命令:")
	fmt.Printf("  go run krylov.go %s\n", filename)
}
