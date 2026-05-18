package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func createTestWAV(sampleRate int, numSamples int, frequency float64) []byte {
	var buf bytes.Buffer

	binary.Write(&buf, binary.LittleEndian, [4]byte{'R', 'I', 'F', 'F'})
	chunkSize := uint32(36 + numSamples*2)
	binary.Write(&buf, binary.LittleEndian, chunkSize)
	binary.Write(&buf, binary.LittleEndian, [4]byte{'W', 'A', 'V', 'E'})

	binary.Write(&buf, binary.LittleEndian, [4]byte{'f', 'm', 't', ' '})
	binary.Write(&buf, binary.LittleEndian, uint32(16))
	binary.Write(&buf, binary.LittleEndian, uint16(1))
	binary.Write(&buf, binary.LittleEndian, uint16(1))
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate*2))
	binary.Write(&buf, binary.LittleEndian, uint16(2))
	binary.Write(&buf, binary.LittleEndian, uint16(16))

	binary.Write(&buf, binary.LittleEndian, [4]byte{'d', 'a', 't', 'a'})
	binary.Write(&buf, binary.LittleEndian, uint32(numSamples*2))

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		sample := int16(math.Sin(2*math.Pi*frequency*t) * 32767)
		binary.Write(&buf, binary.LittleEndian, sample)
	}

	return buf.Bytes()
}

func TestParseWAV(t *testing.T) {
	sampleRate := 44100
	numSamples := 1000
	frequency := 440.0

	wavData := createTestWAV(sampleRate, numSamples, frequency)
	reader := bytes.NewReader(wavData)

	result, err := ParseWAV(reader)
	if err != nil {
		t.Fatalf("Failed to parse WAV: %v", err)
	}

	if result.SampleRate != sampleRate {
		t.Errorf("Expected sample rate %d, got %d", sampleRate, result.SampleRate)
	}

	if len(result.Samples) != numSamples {
		t.Errorf("Expected %d samples, got %d", numSamples, len(result.Samples))
	}

	maxAmp := 0.0
	for _, s := range result.Samples {
		if math.Abs(s) > maxAmp {
			maxAmp = math.Abs(s)
		}
	}

	if maxAmp < 0.9 {
		t.Errorf("Expected amplitude close to 1.0, got max %.4f", maxAmp)
	}
}

func TestParseInvalidWAV(t *testing.T) {
	invalidData := []byte("not a wav file")
	reader := bytes.NewReader(invalidData)

	_, err := ParseWAV(reader)
	if err == nil {
		t.Error("Expected error for invalid WAV, got nil")
	}
}
