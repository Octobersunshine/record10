package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
)

type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

type InterpolationRequest struct {
	Points      []Point   `json:"points"`
	TargetXList []float64 `json:"target_x_list"`
	Method      string    `json:"method,omitempty"`
}

type InterpolationResponse struct {
	Success      bool      `json:"success"`
	Message      string    `json:"message,omitempty"`
	Results      []Point   `json:"results,omitempty"`
	Deduplicated int       `json:"deduplicated,omitempty"`
	Segmented    bool      `json:"segmented,omitempty"`
	MethodUsed   string    `json:"method_used,omitempty"`
}

type Spline struct {
	a, b, c, d []float64
	x          []float64
	n          int
}

const (
	epsilon          = 1e-8
	maxDegree        = 6
	pointsPerSegment = maxDegree + 1
	MethodLagrange   = "lagrange"
	MethodSpline     = "spline"
)

func deduplicatePoints(points []Point) []Point {
	if len(points) < 2 {
		return points
	}

	sort.Slice(points, func(i, j int) bool {
		return points[i].X < points[j].X
	})

	var result []Point
	prevX := points[0].X
	result = append(result, points[0])

	for i := 1; i < len(points); i++ {
		if math.Abs(points[i].X-prevX) >= epsilon {
			result = append(result, points[i])
			prevX = points[i].X
		}
	}

	return result
}

func findSegment(points []Point, x float64) []Point {
	n := len(points)

	if n <= pointsPerSegment {
		return points
	}

	idx := sort.Search(n, func(i int) bool {
		return points[i].X > x
	})

	start := idx - pointsPerSegment/2
	if start < 0 {
		start = 0
	}
	if start+pointsPerSegment > n {
		start = n - pointsPerSegment
	}

	return points[start : start+pointsPerSegment]
}

func lagrangeInterpolation(points []Point, x float64) (float64, error) {
	n := len(points)
	if n < 2 {
		return 0, errors.New("至少需要2个点才能进行插值")
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if math.Abs(points[i].X-points[j].X) < epsilon {
				return 0, fmt.Errorf("存在重复的x值: %v", points[i].X)
			}
		}
	}

	result := 0.0
	for i := 0; i < n; i++ {
		term := points[i].Y
		for j := 0; j < n; j++ {
			if i != j {
				denominator := points[i].X - points[j].X
				if math.Abs(denominator) < epsilon {
					return 0, fmt.Errorf("计算时发现重复的x值: %v", points[i].X)
				}
				term = term * (x - points[j].X) / denominator
			}
		}
		result += term
	}

	return result, nil
}

func buildNaturalSpline(points []Point) (*Spline, error) {
	n := len(points)
	if n < 2 {
		return nil, errors.New("至少需要2个点才能进行样条插值")
	}

	nIntervals := n - 1

	a := make([]float64, n)
	b := make([]float64, nIntervals)
	c := make([]float64, n)
	d := make([]float64, nIntervals)
	x := make([]float64, n)
	h := make([]float64, nIntervals)

	for i := 0; i < n; i++ {
		a[i] = points[i].Y
		x[i] = points[i].X
	}

	for i := 0; i < nIntervals; i++ {
		h[i] = x[i+1] - x[i]
		if h[i] < epsilon {
			return nil, fmt.Errorf("x值必须严格递增，发现重复或递减: x[%d]=%v, x[%d]=%v", i, x[i], i+1, x[i+1])
		}
	}

	if n == 2 {
		b[0] = (a[1] - a[0]) / h[0]
		return &Spline{a: a, b: b, c: c, d: d, x: x, n: n}, nil
	}

	alpha := make([]float64, n)
	for i := 1; i < nIntervals; i++ {
		alpha[i] = 3*(a[i+1]-a[i])/h[i] - 3*(a[i]-a[i-1])/h[i-1]
	}

	l := make([]float64, n)
	mu := make([]float64, n)
	z := make([]float64, n)

	l[0] = 1.0
	mu[0] = 0.0
	z[0] = 0.0

	for i := 1; i < nIntervals; i++ {
		l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
		if math.Abs(l[i]) < epsilon {
			return nil, errors.New("样条矩阵奇异，无法求解")
		}
		mu[i] = h[i] / l[i]
		z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
	}

	l[n-1] = 1.0
	z[n-1] = 0.0
	c[n-1] = 0.0

	for j := nIntervals - 1; j >= 0; j-- {
		c[j] = z[j] - mu[j]*c[j+1]
		b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
		d[j] = (c[j+1] - c[j]) / (3 * h[j])
	}

	return &Spline{a: a, b: b, c: c, d: d, x: x, n: n}, nil
}

func (s *Spline) interpolate(xVal float64) (float64, error) {
	if s.n < 2 {
		return 0, errors.New("样条数据不足")
	}

	if xVal < s.x[0]-epsilon || xVal > s.x[s.n-1]+epsilon {
		return 0, fmt.Errorf("x值 %.6f 超出插值范围 [%.6f, %.6f]", xVal, s.x[0], s.x[s.n-1])
	}

	idx := sort.Search(s.n, func(i int) bool {
		return s.x[i] > xVal
	})

	if idx > 0 {
		idx--
	}
	if idx >= s.n-1 {
		idx = s.n - 2
	}

	dx := xVal - s.x[idx]
	return s.a[idx] + s.b[idx]*dx + s.c[idx]*dx*dx + s.d[idx]*dx*dx*dx, nil
}

func interpolateHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(InterpolationResponse{
			Success: false,
			Message: "只支持POST请求",
		})
		return
	}

	var req InterpolationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(InterpolationResponse{
			Success: false,
			Message: "请求参数解析失败: " + err.Error(),
		})
		return
	}

	if len(req.Points) < 2 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(InterpolationResponse{
			Success: false,
			Message: "至少需要2个点才能进行插值",
		})
		return
	}

	if len(req.TargetXList) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(InterpolationResponse{
			Success: false,
			Message: "target_x_list不能为空",
		})
		return
	}

	method := strings.ToLower(strings.TrimSpace(req.Method))
	if method == "" {
		method = MethodSpline
	}
	if method != MethodLagrange && method != MethodSpline {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(InterpolationResponse{
			Success: false,
			Message: fmt.Sprintf("不支持的插值方法: %s，支持: lagrange, spline", req.Method),
		})
		return
	}

	originalCount := len(req.Points)
	uniquePoints := deduplicatePoints(req.Points)
	deduplicated := originalCount - len(uniquePoints)

	if len(uniquePoints) < 2 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(InterpolationResponse{
			Success: false,
			Message: "去重后点数不足2个，无法进行插值",
		})
		return
	}

	segmented := false
	var results []Point

	if method == MethodLagrange {
		segmented = len(uniquePoints) > pointsPerSegment
		for _, targetX := range req.TargetXList {
			segmentPoints := findSegment(uniquePoints, targetX)
			interpolatedY, err := lagrangeInterpolation(segmentPoints, targetX)
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				json.NewEncoder(w).Encode(InterpolationResponse{
					Success: false,
					Message: err.Error(),
				})
				return
			}
			results = append(results, Point{
				X: targetX,
				Y: interpolatedY,
			})
		}
	} else {
		spline, err := buildNaturalSpline(uniquePoints)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(InterpolationResponse{
				Success: false,
				Message: "样条构建失败: " + err.Error(),
			})
			return
		}

		for _, targetX := range req.TargetXList {
			interpolatedY, err := spline.interpolate(targetX)
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				json.NewEncoder(w).Encode(InterpolationResponse{
					Success: false,
					Message: err.Error(),
				})
				return
			}
			results = append(results, Point{
				X: targetX,
				Y: interpolatedY,
			})
		}
	}

	json.NewEncoder(w).Encode(InterpolationResponse{
		Success:      true,
		Results:      results,
		Deduplicated: deduplicated,
		Segmented:    segmented,
		MethodUsed:   method,
	})
}

func formatFloat(f float64) string {
	return fmt.Sprintf("%v", f)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":       "ok",
		"service":      "插值服务",
		"methods":      "lagrange, spline(默认)",
		"default":      "spline",
		"max_degree":   fmt.Sprintf("%d", maxDegree),
		"tolerance":    fmt.Sprintf("%.1e", epsilon),
		"spline_type":  "natural spline (自然边界)",
	})
}

func main() {
	http.HandleFunc("/interpolate", interpolateHandler)
	http.HandleFunc("/health", healthHandler)

	http.ListenAndServe(":8080", nil)
}
