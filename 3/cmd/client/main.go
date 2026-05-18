package main

import (
	"context"
	"fmt"
	"log"
	"newton_raphson_grpc/proto"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type TestCase struct {
	name       string
	expression string
	initial    float64
	precision  float64
	maxIter    int32
}

func printResult(resp *proto.RootResponse) {
	if resp.GetDerivative() != "" {
		fmt.Printf("  f'(x) = %s\n", resp.GetDerivative())
	}
	if resp.GetIsComplex() {
		fmt.Printf("  Result: root=%.10f %+.10fi\n", resp.GetRoot(), resp.GetRootImag())
	} else {
		fmt.Printf("  Result: root=%.10f\n", resp.GetRoot())
	}
	fmt.Printf("  Iterations: %d\n", resp.GetIterations())
	fmt.Printf("  Converged: %v\n", resp.GetConverged())
	if resp.GetMultiplicity() > 0 {
		fmt.Printf("  Multiplicity: %d\n", resp.GetMultiplicity())
	}
	if resp.GetUsedFallback() {
		fmt.Printf("  Used Fallback: YES - %s\n", resp.GetFallbackInfo())
	}
	if resp.GetValidationMsg() != "" {
		fmt.Printf("  Validation: %s\n", resp.GetValidationMsg())
		if resp.GetSuggestedInitial() != 0 {
			fmt.Printf("  Suggested Initial: %.6f\n", resp.GetSuggestedInitial())
		}
	}
	if resp.GetError() != "" {
		fmt.Printf("  Error message: %s\n", resp.GetError())
	}
}

func main() {
	conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := proto.NewNewtonRaphsonServiceClient(conn)

	fmt.Println("=== Part 1: Basic Root Finding ===")
	basicCases := []TestCase{
		{
			name:       "sqrt(2) - simple root",
			expression: "x*x - 2",
			initial:    1.5,
			precision:  1e-10,
			maxIter:    100,
		},
		{
			name:       "(x-1)^2 = 0 - double root",
			expression: "(x-1)^2",
			initial:    2.0,
			precision:  1e-10,
			maxIter:    100,
		},
		{
			name:       "(x-1)^3 = 0 - triple root",
			expression: "(x-1)^3",
			initial:    2.0,
			precision:  1e-10,
			maxIter:    100,
		},
	}

	for i, tc := range basicCases {
		fmt.Printf("\nTest %d: %s\n", i+1, tc.name)
		fmt.Printf("  Expression: %s\n", tc.expression)
		fmt.Printf("  Initial guess: %.6f\n", tc.initial)

		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()

		resp, err := client.FindRoot(ctx, &proto.RootRequest{
			Expression:    tc.expression,
			InitialGuess:  tc.initial,
			Precision:     tc.precision,
			MaxIterations: tc.maxIter,
		})

		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}
		printResult(resp)
	}

	fmt.Println("\n\n=== Part 2: Domain Validation Tests ===")
	validationCases := []TestCase{
		{
			name:       "1/x at x=0 - singularity point",
			expression: "1/x",
			initial:    0.0,
			precision:  1e-10,
			maxIter:    100,
		},
		{
			name:       "1/(x-2) at x=2 - singularity point",
			expression: "1/(x-2)",
			initial:    2.0,
			precision:  1e-10,
			maxIter:    100,
		},
		{
			name:       "Valid initial value near root",
			expression: "x^2 - 4",
			initial:    2.5,
			precision:  1e-10,
			maxIter:    100,
		},
	}

	for i, tc := range validationCases {
		fmt.Printf("\nTest %d: %s\n", i+1, tc.name)
		fmt.Printf("  Expression: %s\n", tc.expression)
		fmt.Printf("  Initial guess: %.6f\n", tc.initial)

		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()

		resp, err := client.FindRoot(ctx, &proto.RootRequest{
			Expression:    tc.expression,
			InitialGuess:  tc.initial,
			Precision:     tc.precision,
			MaxIterations: tc.maxIter,
		})

		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}
		printResult(resp)
	}

	fmt.Println("\n\n=== Part 3: Complex Root Auto-Fallback Tests ===")
	complexCases := []TestCase{
		{
			name:       "x^2 + 1 = 0 - complex roots only (i)",
			expression: "x^2 + 1",
			initial:    1.0,
			precision:  1e-10,
			maxIter:    200,
		},
		{
			name:       "x^2 + 2x + 2 = 0 - complex conjugate roots",
			expression: "x^2 + 2*x + 2",
			initial:    0.0,
			precision:  1e-10,
			maxIter:    200,
		},
		{
			name:       "x^4 - 16 = 0 - mixed real and complex roots",
			expression: "x^4 - 16",
			initial:    1.0,
			precision:  1e-10,
			maxIter:    200,
		},
	}

	for i, tc := range complexCases {
		fmt.Printf("\nTest %d: %s\n", i+1, tc.name)
		fmt.Printf("  Expression: %s\n", tc.expression)
		fmt.Printf("  Initial guess: %.6f\n", tc.initial)

		ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
		defer cancel()

		resp, err := client.FindRoot(ctx, &proto.RootRequest{
			Expression:    tc.expression,
			InitialGuess:  tc.initial,
			Precision:     tc.precision,
			MaxIterations: tc.maxIter,
		})

		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}
		printResult(resp)
	}

	fmt.Println("\n\n=== Part 4: Concurrent Requests Test ===")
	var wg sync.WaitGroup
	concurrentCount := 8

	fmt.Printf("Sending %d concurrent requests...\n", concurrentCount)

	allCases := append(basicCases, complexCases...)
	allCases = append(allCases, validationCases...)

	for i := 0; i < concurrentCount; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			tc := allCases[id%len(allCases)]
			ctx, cancel := context.WithTimeout(context.Background(), time.Second*15)
			defer cancel()

			start := time.Now()
			resp, err := client.FindRoot(ctx, &proto.RootRequest{
				Expression:    tc.expression,
				InitialGuess:  tc.initial + float64(id)*0.1,
				Precision:     tc.precision,
				MaxIterations: tc.maxIter,
			})
			elapsed := time.Since(start)

			if err != nil {
				fmt.Printf("Request %d failed: %v\n", id, err)
				return
			}

			rootStr := fmt.Sprintf("%.6f", resp.GetRoot())
			if resp.GetIsComplex() {
				rootStr = fmt.Sprintf("%.6f%+.6fi", resp.GetRoot(), resp.GetRootImag())
			}
			fmt.Printf("Request %d completed in %v: root=%s, iter=%d (converged=%v, fallback=%v)\n",
				id, elapsed, rootStr, resp.GetIterations(), resp.GetConverged(), resp.GetUsedFallback())
		}(i)
	}

	wg.Wait()
	fmt.Println("\n✅ All tests completed successfully!")
}
