package websocket

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"audio-fft-service/pkg/fft"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type Client struct {
	conn      *websocket.Conn
	processor *fft.StreamProcessor
	send      chan []byte
	mu        sync.Mutex
	closed    bool
}

type AudioChunk struct {
	Data      []byte `json:"data"`
	Timestamp int64  `json:"timestamp,omitempty"`
}

type ConfigMessage struct {
	Type       string `json:"type"`
	SampleRate int    `json:"sample_rate,omitempty"`
	WindowSize int    `json:"window_size,omitempty"`
	HopSize    int    `json:"hop_size,omitempty"`
}

func (c *Client) readPump() {
	defer func() {
		c.mu.Lock()
		c.closed = true
		c.mu.Unlock()
		c.conn.Close()
		close(c.send)
	}()

	for {
		messageType, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		if messageType == websocket.BinaryMessage {
			startTime := time.Now()
			c.processor.Push16BitPCM(message)
			result := c.processor.Process()
			result.Latency = float64(time.Since(startTime).Microseconds()) / 1000.0

			response, _ := json.Marshal(result)
			c.mu.Lock()
			if !c.closed {
				select {
				case c.send <- response:
				default:
				}
			}
			c.mu.Unlock()
		} else if messageType == websocket.TextMessage {
			var config ConfigMessage
			if err := json.Unmarshal(message, &config); err == nil && config.Type == "config" {
				sampleRate := 44100
				windowSize := 1024
				hopSize := 128

				if config.SampleRate > 0 {
					sampleRate = config.SampleRate
				}
				if config.WindowSize > 0 {
					windowSize = config.WindowSize
				}
				if config.HopSize > 0 {
					hopSize = config.HopSize
				}

				c.processor = fft.NewStreamProcessor(sampleRate, windowSize, hopSize)
				log.Printf("Stream processor configured: sampleRate=%d, windowSize=%d, hopSize=%d",
					sampleRate, windowSize, hopSize)
			}
		}
	}
}

func (c *Client) writePump() {
	defer func() {
		c.conn.Close()
	}()

	for message := range c.send {
		c.conn.WriteMessage(websocket.TextMessage, message)
	}
}

func HandleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("Failed to upgrade WebSocket: %v", err)
		return
	}

	client := &Client{
		conn:      conn,
		processor: fft.NewStreamProcessor(44100, 1024, 128),
		send:      make(chan []byte, 256),
	}

	go client.writePump()
	go client.readPump()
}
