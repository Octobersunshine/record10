package audio

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
)

type WAVHeader struct {
	ChunkID       [4]byte
	ChunkSize     uint32
	Format        [4]byte
	Subchunk1ID   [4]byte
	Subchunk1Size uint32
	AudioFormat   uint16
	NumChannels   uint16
	SampleRate    uint32
	ByteRate      uint32
	BlockAlign    uint16
	BitsPerSample uint16
}

type WAVData struct {
	SampleRate int
	Samples    []float64
}

func ParseWAV(r io.Reader) (*WAVData, error) {
	header := &WAVHeader{}

	if err := binary.Read(r, binary.LittleEndian, header); err != nil {
		return nil, err
	}

	if string(header.ChunkID[:]) != "RIFF" {
		return nil, errors.New("invalid WAV file: missing RIFF chunk")
	}
	if string(header.Format[:]) != "WAVE" {
		return nil, errors.New("invalid WAV file: missing WAVE format")
	}
	if string(header.Subchunk1ID[:]) != "fmt " {
		return nil, errors.New("invalid WAV file: missing fmt chunk")
	}
	if header.AudioFormat != 1 {
		return nil, errors.New("only PCM format is supported")
	}
	if header.NumChannels != 1 {
		return nil, errors.New("only mono audio is supported")
	}
	if header.BitsPerSample != 16 {
		return nil, errors.New("only 16-bit PCM is supported")
	}

	var subchunk2ID [4]byte
	var subchunk2Size uint32

	for {
		if err := binary.Read(r, binary.LittleEndian, &subchunk2ID); err != nil {
			return nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &subchunk2Size); err != nil {
			return nil, err
		}
		if string(subchunk2ID[:]) == "data" {
			break
		}
		discard := make([]byte, subchunk2Size)
		if _, err := io.ReadFull(r, discard); err != nil {
			return nil, err
		}
	}

	numSamples := int(subchunk2Size) / 2
	samples := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		var sample int16
		if err := binary.Read(r, binary.LittleEndian, &sample); err != nil {
			return nil, err
		}
		samples[i] = float64(sample) / math.MaxInt16
	}

	return &WAVData{
		SampleRate: int(header.SampleRate),
		Samples:    samples,
	}, nil
}
