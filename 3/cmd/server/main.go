package main

import (
	"context"
	"log"
	"math/cmplx"
	"net"
	"newton_raphson_grpc/pkg/solver"
	"newton_raphson_grpc/proto"
	"sync"

	"google.golang.org/grpc"
)

type Server struct {
	proto.UnimplementedNewtonRaphsonServiceServer
	requestCount int
	mu           sync.Mutex
}

func (s *Server) FindRoot(ctx context.Context, req *proto.RootRequest) (*proto.RootResponse, error) {
	s.mu.Lock()
	s.requestCount++
	count := s.requestCount
	s.mu.Unlock()

	log.Printf("Received request #%d: expression=%s, initial_guess=%.6f, precision=%.6e, max_iterations=%d",
		count, req.GetExpression(), req.GetInitialGuess(), req.GetPrecision(), req.GetMaxIterations())

	result := solver.NewtonRaphson(
		req.GetExpression(),
		req.GetInitialGuess(),
		req.GetPrecision(),
		int(req.GetMaxIterations()),
	)

	rootReal := result.Root
	rootImag := 0.0
	if result.RootType == solver.ComplexRoot {
		rootReal = real(result.RootComplex)
		rootImag = imag(result.RootComplex)
		log.Printf("Request #%d completed: root=%.10f%+.10fi, iterations=%d, converged=%v, complex=%v",
			count, rootReal, rootImag, result.Iterations, result.Converged, result.RootType == solver.ComplexRoot)
	} else {
		log.Printf("Request #%d completed: root=%.10f, iterations=%d, converged=%v, multiplicity=%d",
			count, result.Root, result.Iterations, result.Converged, result.Multiplicity)
	}

	if result.UsedFallback {
		log.Printf("Request #%d used fallback: %s", count, result.FallbackInfo)
	}

	if result.Validation.Message != "" {
		log.Printf("Request #%d validation: %s", count, result.Validation.Message)
	}

	if result.Derivative != "" {
		log.Printf("Request #%d symbolic derivative: f'(x) = %s", count, result.Derivative)
	}

	return &proto.RootResponse{
		Root:              rootReal,
		RootImag:          rootImag,
		Iterations:        int32(result.Iterations),
		Converged:         result.Converged,
		Error:             result.Error,
		Multiplicity:      int32(result.Multiplicity),
		IsComplex:         result.RootType == solver.ComplexRoot,
		UsedFallback:      result.UsedFallback,
		FallbackInfo:      result.FallbackInfo,
		InputValid:        result.Validation.IsValid,
		ValidationMsg:     result.Validation.Message,
		SuggestedInitial:  result.Validation.Suggestion,
		Derivative:        result.Derivative,
	}, nil
}

func main() {
	listener, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	s := grpc.NewServer(
		grpc.NumStreamWorkers(10),
	)
	proto.RegisterNewtonRaphsonServiceServer(s, &Server{})

	log.Println("Newton-Raphson gRPC server started on :50051")
	log.Println("Supports concurrent requests")

	if err := s.Serve(listener); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
