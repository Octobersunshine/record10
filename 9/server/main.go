package main

import (
	"context"
	"log"
	"net"

	"integral-grpc/calculator"
	"integral-grpc/proto"

	"google.golang.org/grpc"
)

type server struct {
	proto.UnimplementedIntegralServiceServer
}

func (s *server) CalculateIntegral(ctx context.Context, req *proto.IntegralRequest) (*proto.IntegralResponse, error) {
	log.Printf("Received request: function=%s, a=%f, b=%f, n=%d, method=%v",
		req.GetFunction(), req.GetA(), req.GetB(), req.GetN(), req.GetMethod())

	var (
		result float64
		err    error
		methodUsed string
	)

	switch req.GetMethod() {
	case proto.Method_ADAPTIVE_SIMPSON:
		epsilon := req.GetEpsilon()
		if epsilon <= 0 {
			epsilon = calculator.DefaultEpsilon
		}
		result, err = calculator.AdaptiveSimpson(
			req.GetFunction(),
			req.GetA(),
			req.GetB(),
			epsilon,
		)
		methodUsed = "Adaptive Simpson"
	case proto.Method_GAUSS_KRONROD:
		epsilon := req.GetEpsilon()
		if epsilon <= 0 {
			epsilon = calculator.DefaultEpsilon
		}
		result, _, err = calculator.AdaptiveGaussKronrod(
			req.GetFunction(),
			req.GetA(),
			req.GetB(),
			epsilon,
		)
		methodUsed = "Gauss-Kronrod"
	case proto.Method_HYBRID_AUTO:
		epsilon := req.GetEpsilon()
		if epsilon <= 0 {
			epsilon = calculator.DefaultEpsilon
		}
		result, methodUsed, err = calculator.HybridAdaptiveIntegral(
			req.GetFunction(),
			req.GetA(),
			req.GetB(),
			epsilon,
		)
	case proto.Method_COMPOSITE_SIMPSON:
		fallthrough
	default:
		n := int(req.GetN())
		if n <= 0 {
			n = 100
		}
		result, err = calculator.CompositeSimpson(
			req.GetFunction(),
			req.GetA(),
			req.GetB(),
			n,
		)
		methodUsed = "Composite Simpson"
	}

	if err != nil {
		log.Printf("Error calculating integral: %v", err)
		return &proto.IntegralResponse{
			Result: 0,
			Error:  err.Error(),
		}, nil
	}

	log.Printf("Calculation result: %.10f (method: %s)", result, methodUsed)
	return &proto.IntegralResponse{
		Result: result,
		Error:  "",
	}, nil
}

func main() {
	listener, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	s := grpc.NewServer()
	proto.RegisterIntegralServiceServer(s, &server{})

	log.Println("gRPC server started on :50051")
	log.Println("Integral Calculator Service is running...")
	log.Println("Supported methods: COMPOSITE_SIMPSON, ADAPTIVE_SIMPSON (with stack overflow protection)")

	if err := s.Serve(listener); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
