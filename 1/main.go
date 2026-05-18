package main

import (
	"net/http"
	"strconv"
	"time"

	"audio-fft-service/pkg/audio"
	"audio-fft-service/pkg/fft"
	"audio-fft-service/pkg/metrics"
	"audio-fft-service/pkg/websocket"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type ErrorResponse struct {
	Error string `json:"error"`
}

type FFTResponse struct {
	SampleRate int         `json:"sample_rate"`
	NumSamples int         `json:"num_samples"`
	Peaks      []fft.Peak  `json:"peaks"`
	Processed  string      `json:"processed_at"`
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()

	router.Use(metrics.PrometheusMiddleware())

	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "ok",
			"time":   time.Now().Format(time.RFC3339),
		})
	})

	api := router.Group("/api/v1")
	{
		api.POST("/analyze", handleAnalyze)
		api.GET("/stream", websocket.HandleWebSocket)
	}

	router.GET("/", func(c *gin.Context) {
		c.File("demo.html")
	})

	router.Run(":8080")
}

func handleAnalyze(c *gin.Context) {
	file, header, err := c.Request.FormFile("audio")
	if err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "Failed to get audio file: " + err.Error()})
		return
	}
	defer file.Close()

	metrics.RecordFileSize(header.Size)

	numPeaksStr := c.DefaultQuery("n", "5")
	numPeaks, err := strconv.Atoi(numPeaksStr)
	if err != nil || numPeaks < 1 {
		numPeaks = 5
	}

	wavData, err := audio.ParseWAV(file)
	if err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "Failed to parse WAV file: " + err.Error()})
		return
	}

	startTime := time.Now()
	result := fft.ProcessFFT(wavData.Samples, wavData.SampleRate, numPeaks)
	processingTime := time.Since(startTime)

	metrics.RecordFFTProcessingTime(processingTime)

	response := FFTResponse{
		SampleRate: wavData.SampleRate,
		NumSamples: len(wavData.Samples),
		Peaks:      result.Peaks,
		Processed:  time.Now().Format(time.RFC3339),
	}

	c.JSON(http.StatusOK, response)
}
