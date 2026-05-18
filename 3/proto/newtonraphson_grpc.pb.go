package proto

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

const _ = grpc.SupportPackageIsVersion7

type NewtonRaphsonServiceClient interface {
	FindRoot(ctx context.Context, in *RootRequest, opts ...grpc.CallOption) (*RootResponse, error)
}

type newtonRaphsonServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewNewtonRaphsonServiceClient(cc grpc.ClientConnInterface) NewtonRaphsonServiceClient {
	return &newtonRaphsonServiceClient{cc}
}

func (c *newtonRaphsonServiceClient) FindRoot(ctx context.Context, in *RootRequest, opts ...grpc.CallOption) (*RootResponse, error) {
	out := new(RootResponse)
	err := c.cc.Invoke(ctx, "/newtonraphson.NewtonRaphsonService/FindRoot", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

type NewtonRaphsonServiceServer interface {
	FindRoot(context.Context, *RootRequest) (*RootResponse, error)
	mustEmbedUnimplementedNewtonRaphsonServiceServer()
}

type UnimplementedNewtonRaphsonServiceServer struct{}

func (UnimplementedNewtonRaphsonServiceServer) FindRoot(context.Context, *RootRequest) (*RootResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method FindRoot not implemented")
}
func (UnimplementedNewtonRaphsonServiceServer) mustEmbedUnimplementedNewtonRaphsonServiceServer() {}

type UnsafeNewtonRaphsonServiceServer interface {
	mustEmbedUnimplementedNewtonRaphsonServiceServer()
}

func RegisterNewtonRaphsonServiceServer(s grpc.ServiceRegistrar, srv NewtonRaphsonServiceServer) {
	s.RegisterService(&NewtonRaphsonService_ServiceDesc, srv)
}

func _NewtonRaphsonService_FindRoot_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RootRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NewtonRaphsonServiceServer).FindRoot(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/newtonraphson.NewtonRaphsonService/FindRoot",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NewtonRaphsonServiceServer).FindRoot(ctx, req.(*RootRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var NewtonRaphsonService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "newtonraphson.NewtonRaphsonService",
	HandlerType: (*NewtonRaphsonServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "FindRoot",
			Handler:    _NewtonRaphsonService_FindRoot_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "proto/newtonraphson.proto",
}
