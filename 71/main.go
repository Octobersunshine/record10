package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

type EventType int

const (
	Arrival EventType = iota
	Departure
)

type ArrivalProcessType int

const (
	Poisson ArrivalProcessType = iota
	Hyperexponential
	Erlang
)

type MAPParameters struct {
	Type       ArrivalProcessType
	K          int
	Lambdas    []float64
	Weights    []float64
}

func NewPoissonMAP(lambda float64) MAPParameters {
	return MAPParameters{
		Type:    Poisson,
		K:       1,
		Lambdas:  []float64{lambda},
		Weights:  []float64{1.0},
	}
}

func NewHyperexponentialMAP(lambdas []float64, weights []float64) MAPParameters {
	return MAPParameters{
		Type:    Hyperexponential,
		K:       len(lambdas),
		Lambdas:  lambdas,
		Weights:  weights,
	}
}

func NewErlangMAP(k int, lambda float64) MAPParameters {
	return MAPParameters{
		Type:    Erlang,
		K:       k,
		Lambdas:  []float64{lambda},
		Weights:  []float64{1.0},
	}
}

func (m MAPParameters) Mean() float64 {
	switch m.Type {
	case Poisson:
		return 1.0 / m.Lambdas[0]
	case Hyperexponential:
		mean := 0.0
		for i := 0; i < m.K; i++ {
			mean += m.Weights[i] / m.Lambdas[i]
		}
		return mean
	case Erlang:
		return float64(m.K) / m.Lambdas[0]
	}
	return 0.0
}

func (m MAPParameters) SCV() float64 {
	switch m.Type {
	case Poisson:
		return 1.0
	case Hyperexponential:
		mean := m.Mean()
		secondMoment := 0.0
		for i := 0; i < m.K; i++ {
			secondMoment += 2.0 * m.Weights[i] / (m.Lambdas[i] * m.Lambdas[i])
		}
		variance := secondMoment - mean*mean
		return variance / (mean * mean)
	case Erlang:
		return 1.0 / float64(m.K)
	}
	return 1.0
}

type Event struct {
	Time      float64
	Type      EventType
	Priority  int
	SeqNum  uint64
}

type EventQueue []Event

func (eq EventQueue) Len() int { return len(eq) }
func (eq EventQueue) Less(i, j int) bool {
	if eq[i].Time != eq[j].Time {
		return eq[i].Time < eq[j].Time
	}
	if eq[i].Type == Departure && eq[j].Type == Arrival {
		return true
	}
	if eq[i].Type == Arrival && eq[j].Type == Departure {
		return false
	}
	if eq[i].Priority != eq[j].Priority {
		return eq[i].Priority < eq[j].Priority
	}
	return eq[i].SeqNum < eq[j].SeqNum
}
func (eq EventQueue) Swap(i, j int) { eq[i], eq[j] = eq[j], eq[i] }

func (eq *EventQueue) Push(x interface{}) {
	*eq = append(*eq, x.(Event))
}

func (eq *EventQueue) Pop() interface{} {
	old := *eq
	n := len(old)
	x := old[n-1]
	*eq = old[0 : n-1]
	return x
}

type Customer struct {
	ArrivalTime  float64
	ServiceTime  float64
	Priority     int
	SeqNum       uint64
}

type PriorityQueue []Customer

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool {
	if pq[i].Priority != pq[j].Priority {
		return pq[i].Priority < pq[j].Priority
	}
	return pq[i].SeqNum < pq[j].SeqNum
}
func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(Customer))
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	x := old[n-1]
	*pq = old[0 : n-1]
	return x
}

type SimulationResult struct {
	GlobalAvgWaitTime    float64
	PriorityAvgWaitTime  map[int]float64
	QueueLengthDist      map[int]float64
	PriorityQueueDist     map[int]map[int]float64
	Throughput          float64
}

type MM1Queue struct {
	arrivalParams     MAPParameters
	mu                float64
	numPriorities     int
	priorityWeights   []float64
	randGen           *rand.Rand
	eventQueue        *EventQueue
	queue             *PriorityQueue
	clock             float64
	serverBusy        bool
	currentCustomer     Customer
	seqCounter        uint64
	totalWaitTime     float64
	priorityWaitTime  map[int]float64
	totalCustomers    int
	priorityCustomers map[int]int
	lastEventTime     float64
	queueLengthDist   map[int]float64
	priorityQueueDist map[int]map[int]float64
	warmupComplete    bool
	totalServiceTime float64
}

func NewMM1Queue(arrivalParams MAPParameters, mu float64, numPriorities int, priorityWeights []float64, seed int64) *MM1Queue {
	r := rand.New(rand.NewSource(seed))
	eq := &EventQueue{}
	q := &PriorityQueue{}
	
	priorityWaitTime := make(map[int]float64)
	priorityCustomers := make(map[int]int)
	priorityQueueDist := make(map[int]map[int]float64)
	for i := 0; i < numPriorities; i++ {
		priorityWaitTime[i] = 0.0
		priorityCustomers[i] = 0
		priorityQueueDist[i] = make(map[int]float64)
	}

	return &MM1Queue{
		arrivalParams:     arrivalParams,
		mu:                mu,
		numPriorities:     numPriorities,
		priorityWeights:   priorityWeights,
		randGen:           r,
		eventQueue:        eq,
		queue:             q,
		priorityWaitTime:  priorityWaitTime,
		priorityCustomers: priorityCustomers,
		queueLengthDist:   make(map[int]float64),
		priorityQueueDist: priorityQueueDist,
		warmupComplete:    false,
	}
}

func (q *MM1Queue) exp(rate float64) float64 {
	return -math.Log(1.0 - q.randGen.Float64()) / rate
}

func (q *MM1Queue) generateInterarrivalTime() float64 {
	switch q.arrivalParams.Type {
	case Poisson:
		return q.exp(q.arrivalParams.Lambdas[0])
	
	case Hyperexponential:
		u := q.randGen.Float64()
		cumProb := 0.0
		for i := 0; i < q.arrivalParams.K; i++ {
			cumProb += q.arrivalParams.Weights[i]
			if u <= cumProb {
				return q.exp(q.arrivalParams.Lambdas[i])
			}
		}
		return q.exp(q.arrivalParams.Lambdas[q.arrivalParams.K-1])
	
	case Erlang:
		k := q.arrivalParams.K
		sum := 0.0
		for i := 0; i < k; i++ {
			sum += q.exp(q.arrivalParams.Lambdas[0])
		}
		return sum
	}
	return q.exp(1.0)
}

func (q *MM1Queue) generatePriority() int {
	q.seqCounter++
	if q.numPriorities == 1 {
		return 0
	}
	
	u := q.randGen.Float64()
	cumProb := 0.0
	for i := 0; i < q.numPriorities; i++ {
		cumProb += q.priorityWeights[i]
		if u <= cumProb {
			return i
		}
	}
	return q.numPriorities - 1
}

func (q *MM1Queue) scheduleArrival() {
	nextArrival := q.clock + q.generateInterarrivalTime()
	q.eventQueue.Push(Event{Time: nextArrival, Type: Arrival})
}

func (q *MM1Queue) scheduleDeparture() {
	serviceTime := q.exp(q.mu)
	departureTime := q.clock + serviceTime
	q.eventQueue.Push(Event{Time: departureTime, Type: Departure, SeqNum: q.seqCounter})
}

func (q *MM1Queue) updateQueueLengthStats() {
	timeDelta := q.clock - q.lastEventTime
	queueLen := q.queue.Len()
	q.queueLengthDist[queueLen] += timeDelta

	priorityCounts := make(map[int]int)
	for _, cust := range *q.queue {
		priorityCounts[cust.Priority]++
	}
	for p := 0; p < q.numPriorities; p++ {
		q.priorityQueueDist[p][priorityCounts[p]] += timeDelta
	}

	q.lastEventTime = q.clock
}

func (q *MM1Queue) Simulate(numCustomers int, warmupCustomers int) SimulationResult {
	q.clock = 0.0
	q.serverBusy = false
	q.seqCounter = 0
	q.totalWaitTime = 0.0
	q.totalCustomers = 0
	q.lastEventTime = 0.0
	q.queueLengthDist = make(map[int]float64)
	q.warmupComplete = false
	q.totalServiceTime = 0.0
	
	for p := 0; p < q.numPriorities; p++ {
		q.priorityWaitTime[p] = 0.0
		q.priorityCustomers[p] = 0
		q.priorityQueueDist[p] = make(map[int]float64)
	}
	
	*q.eventQueue = (*q.eventQueue)[:0]
	*q.queue = (*q.queue)[:0]

	q.scheduleArrival()

	startMeasurementTime := 0.0

	for q.totalCustomers < numCustomers+warmupCustomers {
		if q.eventQueue.Len() == 0 {
			break
		}

		sort.Sort(q.eventQueue)
		event := q.eventQueue.Pop().(Event)

		if q.warmupComplete {
			q.updateQueueLengthStats()
		}
		q.clock = event.Time

		switch event.Type {
		case Arrival:
			priority := q.generatePriority()
			customer := Customer{
				ArrivalTime: q.clock,
				ServiceTime: q.exp(q.mu),
				Priority:    priority,
				SeqNum:      q.seqCounter,
			}

			if !q.serverBusy {
				q.serverBusy = true
				q.currentCustomer = customer
				q.scheduleDeparture()
			} else {
				q.queue.Push(customer)
			}
			q.scheduleArrival()

		case Departure:
			q.totalCustomers++
			waitTime := q.clock - q.currentCustomer.ArrivalTime
			q.totalServiceTime += q.currentCustomer.ServiceTime

			if q.totalCustomers > warmupCustomers {
				if !q.warmupComplete {
					q.warmupComplete = true
					q.lastEventTime = q.clock
					startMeasurementTime = q.clock
				}
				q.totalWaitTime += waitTime
				q.priorityWaitTime[q.currentCustomer.Priority] += waitTime
				q.priorityCustomers[q.currentCustomer.Priority]++
			}

			if q.queue.Len() > 0 {
				sort.Sort(q.queue)
				nextCustomer := q.queue.Pop().(Customer)
				q.currentCustomer = nextCustomer
				q.scheduleDeparture()
			} else {
				q.serverBusy = false
			}
		}
	}

	if q.warmupComplete {
		q.updateQueueLengthStats()
	}

	globalAvgWaitTime := 0.0
	priorityAvgWaitTime := make(map[int]float64)
	
	actualCustomers := 0
	for p := 0; p < q.numPriorities; p++ {
		actualCustomers += q.priorityCustomers[p]
		if q.priorityCustomers[p] > 0 {
			priorityAvgWaitTime[p] = q.priorityWaitTime[p] / float64(q.priorityCustomers[p])
		}
	}
	
	if actualCustomers > 0 {
		globalAvgWaitTime = q.totalWaitTime / float64(actualCustomers)
	}

	totalTime := q.clock - startMeasurementTime
	queueDist := make(map[int]float64)
	if totalTime > 0 {
		for length, time := range q.queueLengthDist {
			queueDist[length] = time / totalTime
		}
	}

	priorityDist := make(map[int]map[int]float64)
	for p := 0; p < q.numPriorities; p++ {
		priorityDist[p] = make(map[int]float64)
		if totalTime > 0 {
			for length, time := range q.priorityQueueDist[p] {
				priorityDist[p][length] = time / totalTime
			}
		}
	}

	throughput := 0.0
	if totalTime > 0 {
		throughput = float64(actualCustomers) / totalTime
	}

	return SimulationResult{
		GlobalAvgWaitTime:    globalAvgWaitTime,
		PriorityAvgWaitTime:  priorityAvgWaitTime,
		QueueLengthDist:      queueDist,
		PriorityQueueDist:     priorityDist,
		Throughput:          throughput,
	}
}

func RunBatchSimulations(arrivalParams MAPParameters, mu float64, numPriorities int, priorityWeights []float64, numCustomers int, warmupCustomers int, numRuns int) (meanResult SimulationResult, stdDevWait float64) {
	results := make([]float64, numRuns)
	baseSeed := time.Now().UnixNano()

	aggResult := SimulationResult{
		PriorityAvgWaitTime:  make(map[int]float64),
		QueueLengthDist:      make(map[int]float64),
		PriorityQueueDist:     make(map[int]map[int]float64),
	}
	for p := 0; p < numPriorities; p++ {
		aggResult.PriorityAvgWaitTime[p] = 0.0
		aggResult.PriorityQueueDist[p] = make(map[int]float64)
	}

	for i := 0; i < numRuns; i++ {
		queue := NewMM1Queue(arrivalParams, mu, numPriorities, priorityWeights, baseSeed+int64(i*1000))
		result := queue.Simulate(numCustomers, warmupCustomers)
		results[i] = result.GlobalAvgWaitTime

		aggResult.GlobalAvgWaitTime += result.GlobalAvgWaitTime
		aggResult.Throughput += result.Throughput
		
		for p := 0; p < numPriorities; p++ {
			aggResult.PriorityAvgWaitTime[p] += result.PriorityAvgWaitTime[p]
		}
		
		for length, prob := range result.QueueLengthDist {
			aggResult.QueueLengthDist[length] += prob
		}
		
		for p := 0; p < numPriorities; p++ {
			for length, prob := range result.PriorityQueueDist[p] {
				aggResult.PriorityQueueDist[p][length] += prob
			}
		}
	}

	aggResult.GlobalAvgWaitTime /= float64(numRuns)
	aggResult.Throughput /= float64(numRuns)
	
	for p := 0; p < numPriorities; p++ {
		aggResult.PriorityAvgWaitTime[p] /= float64(numRuns)
	}
	
	for length := range aggResult.QueueLengthDist {
		aggResult.QueueLengthDist[length] /= float64(numRuns)
	}
	
	for p := 0; p < numPriorities; p++ {
		for length := range aggResult.PriorityQueueDist[p] {
			aggResult.PriorityQueueDist[p][length] /= float64(numRuns)
		}
	}

	sum := 0.0
	for _, r := range results {
		sum += r
	}
	mean := sum / float64(numRuns)

	variance := 0.0
	for _, r := range results {
		diff := r - mean
		variance += diff * diff
	}
	variance /= float64(numRuns - 1)
	stdDev := math.Sqrt(variance)

	return aggResult, stdDev
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║       M/M/1 队列离散事件模拟 - 增强版                    ║")
	fmt.Println("║    支持: 非泊松到达(MAP) & 优先级队列(非抢占)               ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")

	lambda := 2.0
	mu := 3.0
	numCustomers := 50000
	warmupCustomers := 5000
	numRuns := 20

	fmt.Printf("\n『配置参数:")
	fmt.Printf("\n  基础到达率 λ = %.2f", lambda)
	fmt.Printf("\n  服务率 μ = %.2f", mu)
	fmt.Printf("\n  流量强度 ρ = %.2f", lambda/mu)
	fmt.Printf("\n  单次模拟顾客数 = %d", numCustomers)
	fmt.Printf("\n  预热期顾客数 = %d", warmupCustomers)
	fmt.Printf("\n  独立模拟次数 = %d\n", numRuns)

	fmt.Println("\n══════════════════════════════════════════════════════════")
	fmt.Println("测试1: 泊松到达 + 单优先级 (标准M/M/1)")
	fmt.Println("══════════════════════════════════════════════════════════")
	
	poissonMAP := NewPoissonMAP(lambda)
	priorityWeights1 := []float64{1.0}
	result1, stdDev1 := RunBatchSimulations(poissonMAP, mu, 1, priorityWeights1, numCustomers, warmupCustomers, numRuns)
	
	theoreticalWaitTime := lambda / (mu * (mu - lambda))
	fmt.Printf("\n  理论平均等待时间: %.6f", theoreticalWaitTime)
	fmt.Printf("\n  模拟平均等待时间: %.6f (±%.4f)", result1.GlobalAvgWaitTime, stdDev1)
	fmt.Printf("\n  相对误差: %.2f%%", math.Abs(result1.GlobalAvgWaitTime-theoreticalWaitTime)/theoreticalWaitTime*100)
	fmt.Printf("\n  吞吐量: %.4f 顾客/单位时间", result1.Throughput)

	fmt.Println("\n\n══════════════════════════════════════════════════════════")
	fmt.Println("测试2: 超指数到达 + 单优先级")
	fmt.Println("══════════════════════════════════════════════════════════")
	
	hyperexponentialMAP := NewHyperexponentialMAP(
		[]float64{4.0, 4.0/3.0},
		[]float64{0.5, 0.5},
	)
	fmt.Printf("\n  超指数分布参数:")
	fmt.Printf("\n    - 阶段1: λ=%.2f, 权重=%.2f", hyperexponentialMAP.Lambdas[0], hyperexponentialMAP.Weights[0])
	fmt.Printf("\n    - 阶段2: λ=%.2f, 权重=%.2f", hyperexponentialMAP.Lambdas[1], hyperexponentialMAP.Weights[1])
	fmt.Printf("\n    - 平均间隔时间: %.4f", hyperexponentialMAP.Mean())
	fmt.Printf("\n    - 变异系数平方(C²): %.4f", hyperexponentialMAP.SCV())
	
	result2, stdDev2 := RunBatchSimulations(hyperexponentialMAP, mu, 1, priorityWeights1, numCustomers, warmupCustomers, numRuns)
	
	fmt.Printf("\n\n  模拟平均等待时间: %.6f (±%.4f)", result2.GlobalAvgWaitTime, stdDev2)
	fmt.Printf("\n  吞吐量: %.4f 顾客/单位时间", result2.Throughput)
	fmt.Printf("\n  与泊松相比等待时间增加: %.2f%%", (result2.GlobalAvgWaitTime-result1.GlobalAvgWaitTime)/result1.GlobalAvgWaitTime*100)

	fmt.Println("\n\n══════════════════════════════════════════════════════════")
	fmt.Println("测试3: 泊松到达 + 3级优先级队列")
	fmt.Println("══════════════════════════════════════════════════════════")
	
	numPriorities := 3
	priorityWeights3 := []float64{0.2, 0.3, 0.5}
	fmt.Printf("\n  优先级配置:")
	fmt.Printf("\n    - 优先级0(最高): 权重%.0f%%", priorityWeights3[0]*100)
	fmt.Printf("\n    - 优先级1: 权重%.0f%%", priorityWeights3[1]*100)
	fmt.Printf("\n    - 优先级2(最低): 权重%.0f%%", priorityWeights3[2]*100)
	
	result3, stdDev3 := RunBatchSimulations(poissonMAP, mu, numPriorities, priorityWeights3, numCustomers, warmupCustomers, numRuns)
	
	fmt.Printf("\n\n  全局平均等待时间: %.6f (±%.4f)", result3.GlobalAvgWaitTime, stdDev3)
	fmt.Printf("\n\n  各优先级等待时间:")
	for p := 0; p < numPriorities; p++ {
		fmt.Printf("\n    - 优先级%d: %.6f", p, result3.PriorityAvgWaitTime[p])
	}
	fmt.Printf("\n\n  优先级差异:")
	fmt.Printf("\n    - 最高优先级比最低优先级快 %.2f%%", 
		(result3.PriorityAvgWaitTime[2]-result3.PriorityAvgWaitTime[0])/result3.PriorityAvgWaitTime[2]*100)

	fmt.Println("\n\n══════════════════════════════════════════════════════════")
	fmt.Println("测试4: Erlang到达 + 3级优先级队列")
	fmt.Println("══════════════════════════════════════════════════════════")
	
	erlangK := 2
	erlangMAP := NewErlangMAP(erlangK, lambda*float64(erlangK))
	fmt.Printf("\n  Erlang分布参数: k=%d, λ=%.2f", erlangK, lambda*float64(erlangK))
	fmt.Printf("\n    - 平均间隔时间: %.4f", erlangMAP.Mean())
	fmt.Printf("\n    - 变异系数平方(C²): %.4f", erlangMAP.SCV())
	
	result4, stdDev4 := RunBatchSimulations(erlangMAP, mu, numPriorities, priorityWeights3, numCustomers, warmupCustomers, numRuns)
	
	fmt.Printf("\n\n  全局平均等待时间: %.6f (±%.4f)", result4.GlobalAvgWaitTime, stdDev4)
	fmt.Printf("\n\n  各优先级等待时间:")
	for p := 0; p < numPriorities; p++ {
		fmt.Printf("\n    - 优先级%d: %.6f", p, result4.PriorityAvgWaitTime[p])
	}

	fmt.Println("\n\n══════════════════════════════════════════════════════════")
	fmt.Println("队列长度分布 (测试3 - 优先级队列):")
	fmt.Println("══════════════════════════════════════════════════════════")
	
	var lengths []int
	for length := range result3.QueueLengthDist {
		lengths = append(lengths, length)
	}
	sort.Ints(lengths)
	
	rho := lambda / mu
	for i, length := range lengths {
		if i >= 6 {
			break
		}
		prob := result3.QueueLengthDist[length]
		theoreticalProb := (1 - rho) * math.Pow(rho, float64(length))
		if prob > 0.001 {
			fmt.Printf("\n  队列长度 %d: 模拟概率 %.4f, 理论概率 %.4f", length, prob, theoreticalProb)
		}
	}

	fmt.Println("\n\n══════════════════════════════════════════════════════════")
	fmt.Println("使用说明:")
	fmt.Println("══════════════════════════════════════════════════════════")
	fmt.Println("  到达过程类型:")
	fmt.Println("    - Poisson: 标准泊松到达 (C²=1)")
	fmt.Println("    - Hyperexponential: 超指数分布 (C²>1, 突发到达)")
	fmt.Println("    - Erlang: Erlang-k分布 (C²=1/k, 规则到达)")
	fmt.Println("\n  优先级调度:")
	fmt.Println("    - 非抢占式优先级调度")
	fmt.Println("    - 数字越小优先级越高 (0=最高)")
	fmt.Println("    - 同优先级FCFS")
	fmt.Println("\n  统计输出:")
	fmt.Println("    - 全局&各优先级平均等待时间")
	fmt.Println("    - 队列长度分布")
	fmt.Println("    - 系统吞吐量")
	fmt.Println("")
}
