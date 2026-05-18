# Audio FFT Service

基于Go+Gin框架开发的音频频谱分析服务，支持批量文件上传分析和实时流式频谱分析。

## 功能特性

### 批量文件分析
- 接收WAV音频文件上传（PCM 16bit，单声道）

### 实时流式分析
- WebSocket实时音频流传输
- 滑动窗口FFT处理
- 动态配置（采样率、窗口大小、Hop大小）
- 增量返回前3个峰值频率
- 实时频谱图可视化
- 麦克风实时输入测试
- 延迟和性能指标监控

### 通用功能
- 使用gonum/fft库进行快速傅里叶变换
- 汉宁窗函数减少频谱泄漏
- 抛物线插值提高频率精度
- 自动检测并返回前N个峰值频率
- Prometheus指标监控（请求数、处理耗时、文件大小）

## 技术栈

- Go 1.21+
- Gin v1.9.1 - HTTP框架
- gonum/fft v0.14.0 - FFT计算
- Prometheus client_golang v1.17.0 - 指标监控

## 安装与运行

```bash
# 克隆项目
cd audio-fft-service

# 安装依赖
go mod download

# 运行服务
go run main.go
```

服务将在 `http://localhost:8080` 启动

## API接口

### 1. WebSocket流式频谱分析

**GET** `/api/v1/stream`

建立WebSocket连接进行实时音频流分析。

**二进制消息（客户端→服务端）：**
- 16-bit PCM 小端格式的单声道音频数据
- 建议每次发送256字节（128个样本）

**文本配置消息（客户端→服务端）：**
```json
{
  "type": "config",
  "sample_rate": 44100,
  "window_size": 1024,
  "hop_size": 128
}
```

**响应消息（服务端→客户端）：**
```json
{
  "peaks": [
    { "frequency": 440.5, "magnitude": 0.85, "change": 5.2 },
    { "frequency": 881.2, "magnitude": 0.42, "change": -2.1 },
    { "frequency": 1320.8, "magnitude": 0.21, "change": 1.5 }
  ],
  "seq": 42,
  "latency": 0.85
}
```

### 2. 批量音频分析接口

**POST** `/api/v1/analyze?n=5`

**参数：**
- `n` (可选) - 返回的峰值数量，默认5

**请求：**
- Content-Type: `multipart/form-data`
- 表单字段: `audio` - WAV音频文件

**响应示例：**
```json
{
  "sample_rate": 44100,
  "num_samples": 88200,
  "peaks": [
    { "frequency": 440.0, "magnitude": 0.85 },
    { "frequency": 880.0, "magnitude": 0.42 },
    { "frequency": 1320.0, "magnitude": 0.21 }
  ],
  "processed_at": "2024-01-15T10:30:00Z"
}
```

**使用curl测试：**
```bash
curl -X POST -F "audio=@test.wav" "http://localhost:8080/api/v1/analyze?n=10"
```

### 2. 健康检查

**GET** `/health`

**响应：**
```json
{
  "status": "ok",
  "time": "2024-01-15T10:30:00Z"
}
```

### 3. Prometheus指标

**GET** `/metrics`

暴露以下Prometheus指标：
- `http_requests_total` - HTTP请求总数（按method、endpoint、status标签）
- `http_request_duration_seconds` - HTTP请求耗时直方图
- `fft_processing_seconds` - FFT处理耗时直方图
- `upload_file_size_bytes` - 上传文件大小直方图

## 音频格式要求

- 格式：WAV
- 编码：PCM 16bit
- 声道：单声道（Mono）
- 采样率：任意（通常44100Hz或48000Hz）

## 演示页面

启动服务后访问 `http://localhost:8080` 可以使用Web演示页面：

- 实时WebSocket连接状态显示
- 麦克风实时音频输入
- 440Hz测试音播放
- 实时峰值频率显示（前3个）
- 频谱图可视化
- FPS、延迟、数据包统计
- 配置参数动态调整

## 项目结构

```
audio-fft-service/
├── main.go                 # 主程序入口
├── go.mod                  # Go模块配置
├── demo.html               # Web演示页面
├── pkg/
│   ├── audio/
│   │   ├── wav.go         # WAV文件解析
│   │   └── wav_test.go    # WAV解析单元测试
│   ├── fft/
│   │   ├── processor.go   # 批量FFT处理
│   │   ├── processor_test.go # 批量处理测试
│   │   ├── stream_processor.go # 流式FFT处理器
│   │   └── stream_processor_test.go # 流式处理测试
│   ├── metrics/
│   │   └── metrics.go     # Prometheus指标
│   └── websocket/
│       └── handler.go     # WebSocket处理器
└── README.md
```

## 许可证

MIT
