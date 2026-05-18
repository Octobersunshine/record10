package proto

import (
	reflect "reflect"
	sync "sync"

	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
)

const (
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type RootRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Expression    string  `protobuf:"bytes,1,opt,name=expression,proto3" json:"expression,omitempty"`
	InitialGuess  float64 `protobuf:"fixed64,2,opt,name=initial_guess,json=initialGuess,proto3" json:"initial_guess,omitempty"`
	Precision     float64 `protobuf:"fixed64,3,opt,name=precision,proto3" json:"precision,omitempty"`
	MaxIterations int32   `protobuf:"varint,4,opt,name=max_iterations,json=maxIterations,proto3" json:"max_iterations,omitempty"`
}

func (x *RootRequest) Reset() {
	*x = RootRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_proto_newtonraphson_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *RootRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RootRequest) ProtoMessage() {}

func (x *RootRequest) ProtoReflect() protoreflect.Message {
	mi := &file_proto_newtonraphson_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

func (*RootRequest) Descriptor() ([]byte, []int) {
	return file_proto_newtonraphson_proto_rawDescGZIP(), []int{0}
}

func (x *RootRequest) GetExpression() string {
	if x != nil {
		return x.Expression
	}
	return ""
}

func (x *RootRequest) GetInitialGuess() float64 {
	if x != nil {
		return x.InitialGuess
	}
	return 0
}

func (x *RootRequest) GetPrecision() float64 {
	if x != nil {
		return x.Precision
	}
	return 0
}

func (x *RootRequest) GetMaxIterations() int32 {
	if x != nil {
		return x.MaxIterations
	}
	return 0
}

type RootResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Root         float64 `protobuf:"fixed64,1,opt,name=root,proto3" json:"root,omitempty"`
	Iterations   int32   `protobuf:"varint,2,opt,name=iterations,proto3" json:"iterations,omitempty"`
	Converged    bool    `protobuf:"varint,3,opt,name=converged,proto3" json:"converged,omitempty"`
	Error        string  `protobuf:"bytes,4,opt,name=error,proto3" json:"error,omitempty"`
	Multiplicity int32   `protobuf:"varint,5,opt,name=multiplicity,proto3" json:"multiplicity,omitempty"`
}

func (x *RootResponse) Reset() {
	*x = RootResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_proto_newtonraphson_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *RootResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RootResponse) ProtoMessage() {}

func (x *RootResponse) ProtoReflect() protoreflect.Message {
	mi := &file_proto_newtonraphson_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

func (*RootResponse) Descriptor() ([]byte, []int) {
	return file_proto_newtonraphson_proto_rawDescGZIP(), []int{1}
}

func (x *RootResponse) GetRoot() float64 {
	if x != nil {
		return x.Root
	}
	return 0
}

func (x *RootResponse) GetIterations() int32 {
	if x != nil {
		return x.Iterations
	}
	return 0
}

func (x *RootResponse) GetConverged() bool {
	if x != nil {
		return x.Converged
	}
	return false
}

func (x *RootResponse) GetError() string {
	if x != nil {
		return x.Error
	}
	return ""
}

func (x *RootResponse) GetMultiplicity() int32 {
	if x != nil {
		return x.Multiplicity
	}
	return 0
}

var File_proto_newtonraphson_proto protoreflect.FileDescriptor

var file_proto_newtonraphson_proto_rawDesc = []byte{
	0x0a, 0x17, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x6e, 0x65, 0x77, 0x74, 0x6f, 0x6e, 0x72, 0x61,
	0x70, 0x68, 0x73, 0x6f, 0x6e, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0d, 0x6e, 0x65,
	0x77, 0x74, 0x6f, 0x6e, 0x72, 0x61, 0x70, 0x68, 0x73, 0x6f, 0x6e, 0x22, 0x8c, 0x01, 0x0a, 0x0b,
	0x52, 0x6f, 0x6f, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1e, 0x0a, 0x0a, 0x65,
	0x78, 0x70, 0x72, 0x65, 0x73, 0x73, 0x69, 0x6f, 0x6e, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52,
	0x0a, 0x65, 0x78, 0x70, 0x72, 0x65, 0x73, 0x73, 0x69, 0x6f, 0x6e, 0x12, 0x21, 0x0a, 0x0d,
	0x69, 0x6e, 0x69, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x67, 0x75, 0x65, 0x73, 0x73, 0x18, 0x02,
	0x20, 0x01, 0x28, 0x01, 0x52, 0x0c, 0x69, 0x6e, 0x69, 0x74, 0x69, 0x61, 0x6c, 0x47, 0x75,
	0x65, 0x73, 0x73, 0x12, 0x1c, 0x0a, 0x09, 0x70, 0x72, 0x65, 0x63, 0x69, 0x73, 0x69, 0x6f,
	0x6e, 0x18, 0x03, 0x20, 0x01, 0x28, 0x01, 0x52, 0x09, 0x70, 0x72, 0x65, 0x63, 0x69, 0x73, 0x69,
	0x6f, 0x6e, 0x12, 0x25, 0x0a, 0x0e, 0x6d, 0x61, 0x78, 0x5f, 0x69, 0x74, 0x65, 0x72, 0x61,
	0x74, 0x69, 0x6f, 0x6e, 0x73, 0x18, 0x04, 0x20, 0x01, 0x28, 0x05, 0x52, 0x0d, 0x6d, 0x61,
	0x78, 0x49, 0x74, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x22, 0x77, 0x0a, 0x0c,
	0x52, 0x6f, 0x6f, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x12, 0x0a,
	0x04, 0x72, 0x6f, 0x6f, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x01, 0x52, 0x04, 0x72, 0x6f, 0x6f,
	0x74, 0x12, 0x1e, 0x0a, 0x0a, 0x69, 0x74, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x0a, 0x69, 0x74, 0x65, 0x72, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x73, 0x12, 0x1c, 0x0a, 0x09, 0x63, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x67, 0x65,
	0x64, 0x18, 0x03, 0x20, 0x01, 0x28, 0x08, 0x52, 0x09, 0x63, 0x6f, 0x6e, 0x76, 0x65, 0x72,
	0x67, 0x65, 0x64, 0x12, 0x14, 0x0a, 0x05, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x18, 0x04, 0x20,
	0x01, 0x28, 0x09, 0x52, 0x05, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x32, 0x5d, 0x0a, 0x13, 0x4e,
	0x65, 0x77, 0x74, 0x6f, 0x6e, 0x52, 0x61, 0x70, 0x68, 0x73, 0x6f, 0x6e, 0x53, 0x65, 0x72,
	0x76, 0x69, 0x63, 0x65, 0x12, 0x46, 0x0a, 0x08, 0x46, 0x69, 0x6e, 0x64, 0x52, 0x6f, 0x6f,
	0x74, 0x12, 0x1a, 0x2e, 0x6e, 0x65, 0x77, 0x74, 0x6f, 0x6e, 0x72, 0x61, 0x70, 0x68, 0x73,
	0x6f, 0x6e, 0x2e, 0x52, 0x6f, 0x6f, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a,
	0x1b, 0x2e, 0x6e, 0x65, 0x77, 0x74, 0x6f, 0x6e, 0x72, 0x61, 0x70, 0x68, 0x73, 0x6f, 0x6e,
	0x2e, 0x52, 0x6f, 0x6f, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00,
	0x42, 0x1e, 0x5a, 0x1c, 0x6e, 0x65, 0x77, 0x74, 0x6f, 0x6e, 0x5f, 0x72, 0x61, 0x70, 0x68,
	0x73, 0x6f, 0x6e, 0x5f, 0x67, 0x72, 0x70, 0x63, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_proto_newtonraphson_proto_rawDescOnce sync.Once
	file_proto_newtonraphson_proto_rawDescData = file_proto_newtonraphson_proto_rawDesc
)

func file_proto_newtonraphson_proto_rawDescGZIP() []byte {
	file_proto_newtonraphson_proto_rawDescOnce.Do(func() {
		file_proto_newtonraphson_proto_rawDescData = protoimpl.X.CompressGZIP(file_proto_newtonraphson_proto_rawDescData)
	})
	return file_proto_newtonraphson_proto_rawDescData
}

var file_proto_newtonraphson_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_proto_newtonraphson_proto_goTypes = []interface{}{
	(*RootRequest)(nil),  // 0: newtonraphson.RootRequest
	(*RootResponse)(nil), // 1: newtonraphson.RootResponse
}
var file_proto_newtonraphson_proto_depIdxs = []int32{
	0, // 0: newtonraphson.NewtonRaphsonService.FindRoot:input_type -> newtonraphson.RootRequest
	1, // 1: newtonraphson.NewtonRaphsonService.FindRoot:output_type -> newtonraphson.RootResponse
	1, // [1:2] is the sub-list for method output_type
	0, // [0:1] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_proto_newtonraphson_proto_init() }
func file_proto_newtonraphson_proto_init() {
	if File_proto_newtonraphson_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_proto_newtonraphson_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*RootRequest) {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_proto_newtonraphson_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*RootResponse) {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_proto_newtonraphson_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   2,
			NumServices:   1,
		},
		GoTypes:           file_proto_newtonraphson_proto_goTypes,
		DependencyIndexes: file_proto_newtonraphson_proto_depIdxs,
		MessageInfos:      file_proto_newtonraphson_proto_msgTypes,
	}.Build()
	File_proto_newtonraphson_proto = out.File
	file_proto_newtonraphson_proto_rawDesc = nil
	file_proto_newtonraphson_proto_goTypes = nil
	file_proto_newtonraphson_proto_depIdxs = nil
}
