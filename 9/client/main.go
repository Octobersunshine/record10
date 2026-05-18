package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"integral-grpc/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	function := flag.String("f", "x^2", "Function expression (e.g., 'x^2', 'sin(x)', 'exp(-x)')")
	a := flag.Float64("a", 0, "Lower bound of integration")
	b := flag.Float64("b", 1, "Upper bound of integration")
	n := flag.Int("n", 100, "Number of sub-intervals (for composite Simpson, must be even)")
	methodStr := flag.String("method", "composite", "Integration method: 'composite', 'adaptive', 'gauss-kronrod', 'hybrid'")
	epsilon := flag.Float64("epsilon", 1e-8, "Precision epsilon for adaptive methods")
	flag.Parse()

	conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := proto.NewIntegralServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("Integral Calculator gRPC Client")
	fmt.Println("================================")
	fmt.Printf("Function: %s\n", *function)
	fmt.Printf("Interval: [%.6f, %.6f]\n", *a, *b)

	var method proto.Method
	switch *methodStr {
	case "adaptive":
		method = proto.Method_ADAPTIVE_SIMPSON
		fmt.Printf("Method: ADAPTIVE_SIMPSON (curvature-aware)\n")
		fmt.Printf("Epsilon: %.e\n", *epsilon)
	case "gauss-kronrod":
		method = proto.Method_GAUSS_KRONROD
		fmt.Printf("Method: GAUSS_KRONROD (high-precision)\n")
		fmt.Printf("Epsilon: %.e\n", *epsilon)
	case "hybrid":
		method = proto.Method_HYBRID_AUTO
		fmt.Printf("Method: HYBRID_AUTO (auto-switch between Simpson and Gauss-Kronrod)\n")
		fmt.Printf("Epsilon: %.e\n", *epsilon)
	case "composite":
		fallthrough
	default:
		method = proto.Method_COMPOSITE_SIMPSON
		fmt.Printf("Method: COMPOSITE_SIMPSON\n")
		fmt.Printf("Sub-intervals: %d\n", *n)
	}
	fmt.Println()

	req := &proto.IntegralRequest{
		Function: *function,
		A:        *a,
		B:        *b,
		N:        int32(*n),
		Method:   method,
		Epsilon:  *epsilon,
	}

	resp, err := client.CalculateIntegral(ctx, req)
	if err != nil {
		log.Fatalf("Error calling CalculateIntegral: %v", err)
	}

	if resp.GetError() != "" {
		fmt.Printf("Error: %s\n", resp.GetError())
	} else {
		fmt.Printf("Result: ∫ %s dx from %.6f to %.6f = %.12f\n",
			*function, *a, *b, resp.GetResult())
	}
}
