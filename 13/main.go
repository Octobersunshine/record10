package main

import (
	crand "crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"runtime"
	"strconv"
	"sync"
	"time"
)

type PiRequest struct {
	N              int     `json:"n,omitempty"`
	TargetError    float64 `json:"target_error,omitempty"`
	MaxPoints      int     `json:"max_points,omitempty"`
}

type PiResponse struct {
	Pi             float64 `json:"pi"`
	Points         int     `json:"points"`
	Goroutines     int     `json:"goroutines"`
	Duration       string  `json:"duration"`
	Error          float64 `json:"error"`
	ConfidenceLow  float64 `json:"confidence_low"`
	ConfidenceHigh float64 `json:"confidence_high"`
	Iterations     int     `json:"iterations,omitempty"`
}

func newRandSource() rand.Source {
	var b [8]byte
	if _, err := crand.Read(b[:]); err != nil {
		return rand.NewSource(time.Now().UnixNano())
	}
	seed := int64(binary.LittleEndian.Uint64(b[:]))
	return rand.NewSource(seed)
}

func estimatePiChunk(points int, results chan<- int, wg *sync.WaitGroup) {
	defer wg.Done()
	circleCount := 0
	r := rand.New(newRandSource())

	for i := 0; i < points; i++ {
		x := r.Float64()*2 - 1
		y := r.Float64()*2 - 1
		if x*x+y*y <= 1 {
			circleCount++
		}
	}
	results <- circleCount
}

func estimatePiConcurrent(n int) (float64, int, int, time.Duration) {
	start := time.Now()
	numGoroutines := runtime.NumCPU()
	if numGoroutines > n {
		numGoroutines = n
	}

	pointsPerGoroutine := n / numGoroutines
	remainingPoints := n % numGoroutines

	results := make(chan int, numGoroutines)
	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {
		points := pointsPerGoroutine
		if i < remainingPoints {
			points++
		}
		wg.Add(1)
		go estimatePiChunk(points, results, &wg)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	totalCircle := 0
	for count := range results {
		totalCircle += count
	}

	pi := 4.0 * float64(totalCircle) / float64(n)
	duration := time.Since(start)

	return pi, totalCircle, numGoroutines, duration
}

func calculateError(pi float64, n int) float64 {
	p := pi / 4.0
	standardError := math.Sqrt(p * (1.0 - p) / float64(n))
	return 4.0 * 1.96 * standardError
}

func estimatePiAdaptive(targetError float64, maxPoints int) (float64, int, int, float64, int, time.Duration) {
	start := time.Now()
	initialPoints := 10000
	totalPoints := 0
	totalCircle := 0
	iterations := 0

	if maxPoints <= 0 {
		maxPoints = 1000000000
	}

	for {
		iterations++
		batchSize := initialPoints
		if totalPoints+batchSize > maxPoints {
			batchSize = maxPoints - totalPoints
		}
		if batchSize <= 0 {
			break
		}

		piBatch, circleBatch, _, _ := estimatePiConcurrent(batchSize)
		totalPoints += batchSize
		totalCircle += circleBatch

		pi := 4.0 * float64(totalCircle) / float64(totalPoints)
		currentError := calculateError(pi, totalPoints)

		if currentError <= targetError || totalPoints >= maxPoints {
			duration := time.Since(start)
			numGoroutines := runtime.NumCPU()
			return pi, totalPoints, numGoroutines, currentError, iterations, duration
		}

		initialPoints *= 2
	}

	pi := 4.0 * float64(totalCircle) / float64(totalPoints)
	finalError := calculateError(pi, totalPoints)
	duration := time.Since(start)
	numGoroutines := runtime.NumCPU()
	return pi, totalPoints, numGoroutines, finalError, iterations, duration
}

func piHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	if r.Method == "GET" {
		nStr := r.URL.Query().Get("n")
		targetErrorStr := r.URL.Query().Get("target_error")
		maxPointsStr := r.URL.Query().Get("max_points")

		if targetErrorStr != "" {
			targetError, err := strconv.ParseFloat(targetErrorStr, 64)
			if err != nil || targetError <= 0 {
				http.Error(w, `{"error": "Parameter 'target_error' must be a positive number"}`, http.StatusBadRequest)
				return
			}

			maxPoints := 0
			if maxPointsStr != "" {
				maxPoints, err = strconv.Atoi(maxPointsStr)
				if err != nil || maxPoints < 0 {
					http.Error(w, `{"error": "Parameter 'max_points' must be a non-negative integer"}`, http.StatusBadRequest)
					return
				}
			}

			pi, points, goroutines, errEst, iterations, duration := estimatePiAdaptive(targetError, maxPoints)
			response := PiResponse{
				Pi:             pi,
				Points:         points,
				Goroutines:     goroutines,
				Duration:       duration.String(),
				Error:          errEst,
				ConfidenceLow:  pi - errEst/2,
				ConfidenceHigh: pi + errEst/2,
				Iterations:     iterations,
			}

			json.NewEncoder(w).Encode(response)
			return
		}

		if nStr == "" {
			http.Error(w, `{"error": "Missing parameter 'n' or 'target_error'"}`, http.StatusBadRequest)
			return
		}

		n, err := strconv.Atoi(nStr)
		if err != nil || n <= 0 {
			http.Error(w, `{"error": "Parameter 'n' must be a positive integer"}`, http.StatusBadRequest)
			return
		}

		pi, _, goroutines, duration := estimatePiConcurrent(n)
		errEst := calculateError(pi, n)
		response := PiResponse{
			Pi:             pi,
			Points:         n,
			Goroutines:     goroutines,
			Duration:       duration.String(),
			Error:          errEst,
			ConfidenceLow:  pi - errEst/2,
			ConfidenceHigh: pi + errEst/2,
		}

		json.NewEncoder(w).Encode(response)
		return
	}

	if r.Method == "POST" {
		var req PiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error": "Invalid request body"}`, http.StatusBadRequest)
			return
		}

		if req.TargetError > 0 {
			pi, points, goroutines, errEst, iterations, duration := estimatePiAdaptive(req.TargetError, req.MaxPoints)
			response := PiResponse{
				Pi:             pi,
				Points:         points,
				Goroutines:     goroutines,
				Duration:       duration.String(),
				Error:          errEst,
				ConfidenceLow:  pi - errEst/2,
				ConfidenceHigh: pi + errEst/2,
				Iterations:     iterations,
			}
			json.NewEncoder(w).Encode(response)
			return
		}

		if req.N <= 0 {
			http.Error(w, `{"error": "Parameter 'n' must be a positive integer or 'target_error' must be positive"}`, http.StatusBadRequest)
			return
		}

		pi, _, goroutines, duration := estimatePiConcurrent(req.N)
		errEst := calculateError(pi, req.N)
		response := PiResponse{
			Pi:             pi,
			Points:         req.N,
			Goroutines:     goroutines,
			Duration:       duration.String(),
			Error:          errEst,
			ConfidenceLow:  pi - errEst/2,
			ConfidenceHigh: pi + errEst/2,
		}

		json.NewEncoder(w).Encode(response)
		return
	}

	http.Error(w, `{"error": "Method not allowed"}`, http.StatusMethodNotAllowed)
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprintf(w, `
	<!DOCTYPE html>
	<html>
	<head>
		<title>π Estimator</title>
		<style>
			body { font-family: Arial, sans-serif; max-width: 700px; margin: 50px auto; padding: 20px; }
			h1 { color: #333; }
			.tabs { margin: 20px 0; }
			.tab { padding: 10px 20px; background: #ddd; border: none; cursor: pointer; }
			.tab.active { background: #4CAF50; color: white; }
			.form { margin: 20px 0; }
			input { padding: 8px; width: 200px; margin-right: 10px; }
			button { padding: 8px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
			.result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
			.tab-content { display: none; }
			.tab-content.active { display: block; }
		</style>
	</head>
	<body>
		<h1>并发估算 π</h1>
		<div class="tabs">
			<button class="tab active" onclick="showTab('fixed')">固定投点数</button>
			<button class="tab" onclick="showTab('adaptive')">自适应抽样</button>
		</div>

		<div id="fixed" class="tab-content active">
			<div class="form">
				<input type="number" id="points" value="1000000" min="1000">
				<button onclick="estimateFixed()">计算 π</button>
			</div>
		</div>

		<div id="adaptive" class="tab-content">
			<div class="form">
				<label>目标误差: </label>
				<input type="number" id="targetError" value="0.001" step="0.0001" min="0.000001">
				<label>最大投点数: </label>
				<input type="number" id="maxPoints" value="100000000" min="1000">
				<button onclick="estimateAdaptive()">计算 π</button>
			</div>
		</div>

		<div id="result" class="result"></div>

		<script>
			function showTab(tabId) {
				document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
				document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
				event.target.classList.add('active');
				document.getElementById(tabId).classList.add('active');
			}

			async function estimateFixed() {
				const n = document.getElementById('points').value;
				const response = await fetch('/pi?n=' + n);
				const data = await response.json();
				displayResult(data);
			}

			async function estimateAdaptive() {
				const targetError = document.getElementById('targetError').value;
				const maxPoints = document.getElementById('maxPoints').value;
				const response = await fetch('/pi?target_error=' + targetError + '&max_points=' + maxPoints);
				const data = await response.json();
				displayResult(data);
			}

			function displayResult(data) {
				let html = '<strong>π ≈ ' + data.pi.toFixed(8) + '</strong><br>' +
					'投点数: ' + data.points + '<br>' +
					'并发数: ' + data.goroutines + '<br>' +
					'95%置信误差: ±' + (data.error/2).toFixed(8) + '<br>' +
					'置信区间: [' + data.confidence_low.toFixed(8) + ', ' + data.confidence_high.toFixed(8) + ']<br>' +
					'耗时: ' + data.duration;
				if (data.iterations) {
					html += '<br>迭代次数: ' + data.iterations;
				}
				document.getElementById('result').innerHTML = html;
			}
		</script>
	</body>
	</html>
	`)
}

func main() {
	http.HandleFunc("/", rootHandler)
	http.HandleFunc("/pi", piHandler)

	fmt.Println("π Estimator Server starting on :8080...")
	fmt.Println("访问 http://localhost:8080 使用Web界面")
	fmt.Println("")
	fmt.Println("API 使用方式:")
	fmt.Println("  固定投点数: GET /pi?n=1000000")
	fmt.Println("  自适应抽样: GET /pi?target_error=0.001&max_points=100000000")
	fmt.Println("  或 POST /pi with JSON body")
	fmt.Println("")

	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Printf("Server error: %v\n", err)
	}
}
