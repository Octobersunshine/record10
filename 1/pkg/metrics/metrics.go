package metrics

import (
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	requestCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	requestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	fftProcessingTime = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "fft_processing_seconds",
			Help:    "FFT processing time in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 10),
		},
	)

	fileSizeBytes = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "upload_file_size_bytes",
			Help:    "Size of uploaded WAV files in bytes",
			Buckets: prometheus.ExponentialBuckets(1024, 2, 10),
		},
	)
)

func PrometheusMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		c.Next()

		duration := time.Since(start).Seconds()
		status := strconv.Itoa(c.Writer.Status())

		requestCount.WithLabelValues(c.Request.Method, c.FullPath(), status).Inc()
		requestDuration.WithLabelValues(c.Request.Method, c.FullPath()).Observe(duration)
	}
}

func RecordFFTProcessingTime(duration time.Duration) {
	fftProcessingTime.Observe(duration.Seconds())
}

func RecordFileSize(size int64) {
	fileSizeBytes.Observe(float64(size))
}
