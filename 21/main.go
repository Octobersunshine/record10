package main

import (
	"encoding/json"
	"log"
	"net/http"
	"newton-cotes-integration/integration"
)

type IntegrationRequest struct {
	Function  string  `json:"function"`
	A         float64 `json:"a"`
	B         float64 `json:"b"`
	N         int     `json:"N,omitempty"`
	NCotes    int     `json:"n,omitempty"`
	Method    string  `json:"method,omitempty"`
	Tolerance float64 `json:"tolerance,omitempty"`
	MaxLevels int     `json:"max_levels,omitempty"`
}

type IntegrationResponse struct {
	Result  float64   `json:"result"`
	Success bool      `json:"success"`
	Message string    `json:"message,omitempty"`
	Method  string    `json:"method,omitempty"`
	Table   [][]float64 `json:"table,omitempty"`
}

func integrateHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(IntegrationResponse{
			Success: false,
			Message: "Method not allowed, use POST",
		})
		return
	}

	var req IntegrationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(IntegrationResponse{
			Success: false,
			Message: "Invalid JSON: " + err.Error(),
		})
		return
	}

	if req.Function == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(IntegrationResponse{
			Success: false,
			Message: "Function expression is required",
		})
		return
	}

	if req.Method == "" {
		req.Method = "romberg"
	}

	var result float64
	var err error
	var table [][]float64

	switch req.Method {
	case "romberg":
		romberg := &integration.Romberg{
			Tolerance: req.Tolerance,
			MaxLevels: req.MaxLevels,
		}
		result, table, err = romberg.IntegrateExpr(req.Function, req.A, req.B)
		
	case "newton-cotes", "nc":
		if req.N == 0 {
			req.N = 10
		}
		if req.NCotes == 0 {
			req.NCotes = 2
		}
		nc := &integration.NewtonCotes{}
		result, err = nc.Integrate(req.Function, req.A, req.B, req.NCotes, req.N)
		
	default:
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(IntegrationResponse{
			Success: false,
			Message: "Unknown method: " + req.Method + ". Use 'romberg' or 'newton-cotes'",
		})
		return
	}

	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(IntegrationResponse{
			Success: false,
			Message: err.Error(),
		})
		return
	}

	json.NewEncoder(w).Encode(IntegrationResponse{
		Result:  result,
		Success: true,
		Method:  req.Method,
		Table:   table,
	})
}

func main() {
	nc := &integration.NewtonCotes{}
	
	log.Println("=== 牛顿-柯特斯系数验证 ===")
	for n := 1; n <= 6; n++ {
		nc.VerifyCoefficients(n)
	}
	
	http.HandleFunc("/integrate", integrateHandler)
	
	log.Println("\nServer starting on :8080...")
	log.Println("POST /integrate with JSON body:")
	log.Println("\n=== 龙贝格积分 (默认) ===")
	log.Println(`{
  "function": "x^2",
  "a": 0,
  "b": 1,
  "method": "romberg",
  "tolerance": 1e-10,
  "max_levels": 10
}`)
	log.Println("\n=== 牛顿-柯特斯积分 ===")
	log.Println(`{
  "function": "x^2",
  "a": 0,
  "b": 1,
  "method": "newton-cotes",
  "n": 2,
  "N": 10
}`)
	log.Println("\n支持的方法:")
	log.Println("  - romberg: 龙贝格积分 (理查德森外推加速)")
	log.Println("  - newton-cotes (或 nc): 牛顿-柯特斯积分")
	log.Println("\n牛顿-柯特斯阶数:")
	log.Println("  n=1: Trapezoidal rule (梯形)")
	log.Println("  n=2: Simpson's rule (辛普森)")
	log.Println("  n=3: Simpson's 3/8 rule (辛普森3/8)")
	log.Println("  n=4: Boole's rule (布尔)")
	log.Println("  n=5: Milne's rule")
	log.Println("  n=6: Weddle's rule")
	
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal("Server failed to start:", err)
	}
}
