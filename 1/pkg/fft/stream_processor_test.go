package fft

import (
	"encoding/binary"
	"math"
	"testing"
)

func generateSineWave16Bit(frequency float64, sampleRate int, numSamples int) []byte {
	data := make([]byte, numSamples*2)
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		sample := int16(math.Sin(2*math.Pi*frequency*t) * 32767)
		binary.LittleEndian.PutUint16(data[i*2:], uint16(sample))
	}
	return data
}

func TestStreamProcessorPushPCM(t *testing.T) {
	sp := NewStreamProcessor(44100, 1024, 128)

	data := generateSineWave16Bit(440.0, 44100, 100)
	sp.Push16BitPCM(data)

	if sp.GetBufferLevel() != 100 {
		t.Errorf("Expected buffer level 100, got %d", sp.GetBufferLevel())
	}
}

func TestStreamProcessorProcess(t *testing.T) {
	sp := NewStreamProcessor(44100, 1024, 128)

	data := generateSineWave16Bit(440.0, 44100, 2048)
	sp.Push16BitPCM(data)

	result := sp.Process()
	if result.Seq != 1 {
		t.Errorf("Expected seq 1, got %d", result.Seq)
	}

	if len(result.Peaks) == 0 {
		t.Error("Expected at least one peak")
	}

	found := false
	for _, peak := range result.Peaks {
		if math.Abs(peak.Frequency-440.0) < 20.0 {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Expected 440Hz peak, got peaks: %v", result.Peaks)
	}
}

func TestStreamProcessorSlidingWindow(t *testing.T) {
	sp := NewStreamProcessor(44100, 1024, 128)

	data := generateSineWave16Bit(1000.0, 44100, 4096)
	sp.Push16BitPCM(data)

	result1 := sp.Process()
	result2 := sp.Process()

	if result1.Seq != 1 {
		t.Errorf("Expected seq 1 for first result, got %d", result1.Seq)
	}
	if result2.Seq != 2 {
		t.Errorf("Expected seq 2 for second result, got %d", result2.Seq)
	}
}

func TestStreamProcessorReset(t *testing.T) {
	sp := NewStreamProcessor(44100, 1024, 128)

	data := generateSineWave16Bit(440.0, 44100, 2048)
	sp.Push16BitPCM(data)
	sp.Process()

	sp.Reset()

	if sp.GetBufferLevel() != 0 {
		t.Errorf("Expected buffer level 0 after reset, got %d", sp.GetBufferLevel())
	}

	result := sp.Process()
	if result.Seq != 0 {
		t.Errorf("Expected seq 0 after reset, got %d", result.Seq)
	}
}

func TestStreamProcessorIncrementalProcessing(t *testing.T) {
	sp := NewStreamProcessor(44100, 512, 64)

	chunkSize := 128
	numChunks := 10
	totalPeaks := 0

	for i := 0; i < numChunks; i++ {
		data := generateSineWave16Bit(880.0, 44100, chunkSize)
		sp.Push16BitPCM(data)
		result := sp.Process()
		if len(result.Peaks) > 0 {
			totalPeaks++
		}
	}

	if totalPeaks == 0 {
		t.Error("Expected peaks to be detected in incremental processing")
	}
}

func TestStreamProcessorHopSize(t *testing.T) {
	sp := NewStreamProcessor(44100, 1024, 256)

	data := generateSineWave16Bit(440.0, 44100, 2048)
	sp.Push16BitPCM(data)

	initialLevel := sp.GetBufferLevel()
	sp.Process()
	levelAfterProcess := sp.GetBufferLevel()

	expectedHop := initialLevel - 256
	if levelAfterProcess > expectedHop {
		t.Errorf("Expected buffer to reduce by hop size, initial: %d, after: %d",
			initialLevel, levelAfterProcess)
	}
}

func TestStreamProcessorSmallBuffer(t *testing.T) {
	sp := NewStreamProcessor(44100, 1024, 128)

	data := generateSineWave16Bit(440.0, 44100, 500)
	sp.Push16BitPCM(data)

	result := sp.Process()

	if len(result.Peaks) != 0 {
		t.Error("Expected no peaks when buffer is smaller than window size")
	}
}
