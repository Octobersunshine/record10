package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

type Cell struct {
	x          float64
	dx         float64
	temperature float64
	level      int
	marked     bool
}

type AMRGrid struct {
	cells       []Cell
	maxLevel    int
	minDx       float64
	refineThresh float64
	coarsenThresh float64
	alpha       float64
}

type AMRParams struct {
	Length        float64
	Time          float64
	InitialDx     float64
	MaxLevel      int
	Alpha         float64
	RefineThresh  float64
	CoarsenThresh float64
	Boundary      BoundaryCondition
	InitialFunc   func(x, L float64) float64
}

func NewAMRGrid(params AMRParams) *AMRGrid {
	nx := int(params.Length/params.InitialDx) + 1
	cells := make([]Cell, nx)

	for i := 0; i < nx; i++ {
		x := float64(i) * params.InitialDx
		cells[i] = Cell{
			x:          x,
			dx:         params.InitialDx,
			temperature: params.InitialFunc(x, params.Length),
			level:      0,
			marked:     false,
		}
	}

	return &AMRGrid{
		cells:        cells,
		maxLevel:     params.MaxLevel,
		minDx:        params.InitialDx / math.Pow(2, float64(params.MaxLevel)),
		refineThresh: params.RefineThresh,
		coarsenThresh: params.CoarsenThresh,
		alpha:        params.Alpha,
	}
}

func (g *AMRGrid) ComputeTimeStep() float64 {
	minDx := g.cells[0].dx
	for _, cell := range g.cells {
		if cell.dx < minDx {
			minDx = cell.dx
		}
	}
	r := 0.4
	return r * minDx * minDx / g.alpha
}

func (g *AMRGrid) ComputeGradients() []float64 {
	n := len(g.cells)
	gradients := make([]float64, n)

	for i := 1; i < n-1; i++ {
		dxAvg := (g.cells[i-1].dx + g.cells[i].dx + g.cells[i+1].dx) / 3.0
		gradients[i] = math.Abs(g.cells[i+1].temperature - g.cells[i-1].temperature) / (2.0 * dxAvg)
	}

	if n >= 2 {
		gradients[0] = math.Abs(g.cells[1].temperature - g.cells[0].temperature) / g.cells[0].dx
		gradients[n-1] = math.Abs(g.cells[n-1].temperature - g.cells[n-2].temperature) / g.cells[n-1].dx
	}

	return gradients
}

func (g *AMRGrid) MarkForRefinement() {
	gradients := g.ComputeGradients()

	for i := range g.cells {
		if gradients[i] > g.refineThresh && g.cells[i].level < g.maxLevel {
			g.cells[i].marked = true
		} else if gradients[i] < g.coarsenThresh && g.cells[i].level > 0 {
			g.cells[i].marked = false
		}
	}
}

func interpolateLinear(x0, x1, x, y0, y1 float64) float64 {
	t := (x - x0) / (x1 - x0)
	return y0*(1-t) + y1*t
}

func (g *AMRGrid) RefineGrid() {
	var newCells []Cell
	i := 0
	n := len(g.cells)

	for i < n {
		if !g.cells[i].marked {
			newCells = append(newCells, g.cells[i])
			i++
			continue
		}

		if i < n-1 && g.cells[i].marked {
			left := g.cells[i]
			right := g.cells[i+1]

			newDx := left.dx / 2.0
			newLevel := left.level + 1

			x1 := left.x
			x2 := left.x + newDx
			x3 := right.x

			temp1 := left.temperature
			temp2 := interpolateLinear(left.x, right.x, x2, left.temperature, right.temperature)
			temp3 := right.temperature

			newCells = append(newCells, Cell{
				x:          x1,
				dx:         newDx,
				temperature: temp1,
				level:      newLevel,
				marked:     false,
			})
			newCells = append(newCells, Cell{
				x:          x2,
				dx:         newDx,
				temperature: temp2,
				level:      newLevel,
				marked:     false,
			})

			i++
		} else {
			newCells = append(newCells, g.cells[i])
			i++
		}
	}

	g.cells = newCells
	sort.Slice(g.cells, func(i, j int) bool {
		return g.cells[i].x < g.cells[j].x
	})
}

func (g *AMRGrid) CoarsenGrid() {
	if len(g.cells) < 3 {
		return
	}

	var newCells []Cell
	i := 0
	n := len(g.cells)

	for i < n {
		if i < n-1 &&
			g.cells[i].level == g.cells[i+1].level &&
			g.cells[i].level > 0 &&
			!g.cells[i].marked &&
			!g.cells[i+1].marked {

			left := g.cells[i]
			right := g.cells[i+1]

			newDx := left.dx * 2.0
			newLevel := left.level - 1
			avgTemp := (left.temperature + right.temperature) / 2.0

			newCells = append(newCells, Cell{
				x:          left.x,
				dx:         newDx,
				temperature: avgTemp,
				level:      newLevel,
				marked:     false,
			})
			i += 2
		} else {
			newCells = append(newCells, g.cells[i])
			i++
		}
	}

	g.cells = newCells
}

func (g *AMRGrid) ApplyBoundaryConditions(boundary BoundaryCondition, dt float64) {
	n := len(g.cells)
	if n < 2 {
		return
	}

	r := g.alpha * dt / (g.cells[0].dx * g.cells[0].dx)

	switch boundary.Type {
	case Dirichlet:
		g.cells[0].temperature = boundary.LeftValue
		g.cells[n-1].temperature = boundary.RightValue

	case Neumann:
		leftFlux := boundary.LeftValue
		rightFlux := boundary.RightValue

		if n >= 2 {
			uLeftGhost := g.cells[1].temperature - 2*g.cells[0].dx*leftFlux
			g.cells[0].temperature = g.cells[0].temperature +
				r*(g.cells[1].temperature-2*g.cells[0].temperature+uLeftGhost)
		}

		if n >= 3 {
			uRightGhost := g.cells[n-2].temperature + 2*g.cells[n-1].dx*rightFlux
			g.cells[n-1].temperature = g.cells[n-1].temperature +
				r*(uRightGhost-2*g.cells[n-1].temperature+g.cells[n-2].temperature)
		}
	}
}

func (g *AMRGrid) TimeStep(dt float64, boundary BoundaryCondition) {
	n := len(g.cells)
	if n < 3 {
		return
	}

	newTemp := make([]float64, n)
	copy(newTemp, g.getTemperatures())

	for i := 1; i < n-1; i++ {
		dxLeft := g.cells[i].x - g.cells[i-1].x
		dxRight := g.cells[i+1].x - g.cells[i].x
		dxAvg := (dxLeft + dxRight) / 2.0

		r := g.alpha * dt / (dxAvg * dxAvg)

		gradRight := (g.cells[i+1].temperature - g.cells[i].temperature) / dxRight
		gradLeft := (g.cells[i].temperature - g.cells[i-1].temperature) / dxLeft

		diffusion := (gradRight - gradLeft) / dxAvg
		newTemp[i] = g.cells[i].temperature + g.alpha*dt*diffusion
	}

	for i := 1; i < n-1; i++ {
		g.cells[i].temperature = newTemp[i]
	}

	g.ApplyBoundaryConditions(boundary, dt)
}

func (g *AMRGrid) getTemperatures() []float64 {
	temps := make([]float64, len(g.cells))
	for i, cell := range g.cells {
		temps[i] = cell.temperature
	}
	return temps
}

func (g *AMRGrid) PrintStats(time float64) {
	n := len(g.cells)
	levelCount := make(map[int]int)

	for _, cell := range g.cells {
		levelCount[cell.level]++
	}

	minDx := g.cells[0].dx
	maxDx := g.cells[0].dx
	for _, cell := range g.cells {
		if cell.dx < minDx {
			minDx = cell.dx
		}
		if cell.dx > maxDx {
			maxDx = cell.dx
		}
	}

	fmt.Printf("t=%.4fs | 单元格数: %d | 网格层次: ", time, n)
	for l := 0; l <= g.maxLevel; l++ {
		if count, ok := levelCount[l]; ok {
			fmt.Printf("L%d:%d ", l, count)
		}
	}
	fmt.Printf("| dx范围: %.6f-%.6fm\n", minDx, maxDx)
}

func (g *AMRGrid) PrintTemperature() {
	fmt.Println("  x位置      温度        网格层次  dx")
	for _, cell := range g.cells {
		fmt.Printf("  %8.4f  %10.6f  L%d      %8.6f\n",
			cell.x, cell.temperature, cell.level, cell.dx)
	}
}

func FTCS_AMR(params AMRParams) (*AMRGrid, error) {
	grid := NewAMRGrid(params)

	dt := grid.ComputeTimeStep()
	totalSteps := int(params.Time / dt)
	adaptInterval := int(math.Max(10, float64(totalSteps)/20))

	fmt.Printf("AMR 模拟开始\n")
	fmt.Printf("参数: L=%.2fm, T=%.4fs, 初始dx=%.4fm, maxLevel=%d\n",
		params.Length, params.Time, params.InitialDx, params.MaxLevel)
	fmt.Printf("细化阈值: %.2f, 粗化阈值: %.2f\n", params.RefineThresh, params.CoarsenThresh)
	fmt.Printf("总时间步数: %d, 网格调整间隔: %d步\n\n", totalSteps, adaptInterval)

	grid.PrintStats(0)

	for step := 0; step < totalSteps; step++ {
		currentTime := float64(step) * dt

		if step > 0 && step%adaptInterval == 0 {
			grid.MarkForRefinement()
			grid.RefineGrid()
			grid.CoarsenGrid()
			dt = grid.ComputeTimeStep()
			grid.PrintStats(currentTime)
		}

		grid.TimeStep(dt, params.Boundary)
	}

	fmt.Println()
	grid.PrintStats(params.Time)

	return grid, nil
}

func main() {
	fmt.Println("=== 一维热传导方程 FTCS-AMR 求解器 ===")
	fmt.Println("自适应网格细化版本")
	fmt.Println()

	params := AMRParams{
		Length:        1.0,
		Time:          0.02,
		InitialDx:     0.05,
		MaxLevel:      3,
		Alpha:         0.01,
		RefineThresh:  80.0,
		CoarsenThresh: 20.0,
		Boundary: BoundaryCondition{
			Type:       Dirichlet,
			LeftValue:  0,
			RightValue: 0,
		},
		InitialFunc: func(x, L float64) float64 {
			if x > 0.3 && x < 0.7 {
				return 100.0
			}
			return 0.0
		},
	}

	fmt.Println("测试1: 方波初始条件 + Dirichlet 边界")
	fmt.Println("方波边缘温度梯度大，应触发网格细化\n")

	grid, err := FTCS_AMR(params)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}

	fmt.Println("\n最终温度分布:")
	grid.PrintTemperature()

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("\n测试2: 高斯脉冲初始条件 + Neumann 绝热边界")
	fmt.Println("高斯脉冲边缘梯度大，中心梯度逐渐减小\n")

	params2 := AMRParams{
		Length:        1.0,
		Time:          0.01,
		InitialDx:     0.05,
		MaxLevel:      3,
		Alpha:         0.01,
		RefineThresh:  100.0,
		CoarsenThresh: 30.0,
		Boundary: BoundaryCondition{
			Type:       Neumann,
			LeftValue:  0,
			RightValue: 0,
		},
		InitialFunc: func(x, L float64) float64 {
			mu := L / 2
			sigma := L / 15
			return 100 * math.Exp(-math.Pow(x-mu, 2)/(2*math.Pow(sigma, 2)))
		},
	}

	grid2, err := FTCS_AMR(params2)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}

	fmt.Println("\n最终温度分布:")
	grid2.PrintTemperature()

	fmt.Println()
	fmt.Println("=== AMR 技术说明 ===")
	fmt.Println("1. 梯度检测: 计算温度梯度 |∂u/∂x|")
	fmt.Println("   - 梯度 > 细化阈值: 标记为需要细化")
	fmt.Println("   - 梯度 < 粗化阈值: 标记为可以粗化")
	fmt.Println()
	fmt.Println("2. 网格细化: 单元格一分为二")
	fmt.Println("   - 线性插值计算新节点温度")
	fmt.Println("   - 网格层次 level+1")
	fmt.Println()
	fmt.Println("3. 网格粗化: 相邻两个单元格合并")
	fmt.Println("   - 温度取平均值")
	fmt.Println("   - 网格层次 level-1")
	fmt.Println()
	fmt.Println("4. 时间步长: 自动根据最细网格调整")
	fmt.Println("   - dt = r * min(dx)² / α, r=0.4")
	fmt.Println("   - 满足稳定性条件 r ≤ 0.5")
	fmt.Println()
	fmt.Println("5. 优势:")
	fmt.Println("   - 梯度大区域自动加密，提高精度")
	fmt.Println("   - 梯度小区域保持粗网格，节省计算量")
	fmt.Println("   - 总单元格数远小于均匀细网格")
}
