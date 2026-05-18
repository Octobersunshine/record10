package fft

import (
	"encoding/binary"
	"math"
	"sync"

	"gonum.org/v1/gonum/dsp/fourier"
)

type StreamPeak struct {
	Frequency float64 `json:"frequency"`
	Magnitude float64 `json:"magnitude"`
	Change    float64 `json:"change,omitempty"`
}

type StreamResult struct {
	Peaks   []StreamPeak `json:"peaks"`
	Seq     int64        `json:"seq"`
	Latency float64      `json:"latency_ms,omitempty"`
}

type StreamProcessor struct {
	sampleRate    int
	windowSize    int
	hopSize       int
	buffer        []float64
	prevPeaks     []StreamPeak
	prevMagnitude []float64
	mu            sync.Mutex
	fft           *fourier.FFT
	seq           int64
}

func NewStreamProcessor(sampleRate, windowSize, hopSize int) *StreamProcessor {
	fftSize := nextPowerOfTwo(windowSize)
	return &StreamProcessor{
		sampleRate:    sampleRate,
		windowSize:    windowSize,
		hopSize:       hopSize,
		buffer:        make([]float64, 0, windowSize*2),
		prevPeaks:     make([]StreamPeak, 0, 3),
		prevMagnitude: make([]float64, 0),
		fft:           fourier.NewFFT(fftSize),
	}
}

func (sp *StreamProcessor) Push16BitPCM(data []byte) {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	numSamples := len(data) / 2
	for i := 0; i < numSamples; i++ {
		if i*2+1 >= len(data) {
			break
		}
		sample := int16(binary.LittleEndian.Uint16(data[i*2:]))
		sp.buffer = append(sp.buffer, float64(sample)/math.MaxInt16)
	}

	if len(sp.buffer) > sp.windowSize*2 {
		sp.buffer = sp.buffer[len(sp.buffer)-sp.windowSize*2:]
	}
}

func (sp *StreamProcessor) PushFloatSamples(samples []float64) {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	sp.buffer = append(sp.buffer, samples...)

	if len(sp.buffer) > sp.windowSize*2 {
		sp.buffer = sp.buffer[len(sp.buffer)-sp.windowSize*2:]
	}
}

func (sp *StreamProcessor) Process() *StreamResult {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	if len(sp.buffer) < sp.windowSize {
		return &StreamResult{
			Peaks: []StreamPeak{},
			Seq:   sp.seq,
		}
	}

	sp.seq++

	analysisWindow := sp.buffer[len(sp.buffer)-sp.windowSize:]

	fftSize := nextPowerOfTwo(sp.windowSize)
	windowed := make([]float64, fftSize)
	for i := 0; i < sp.windowSize; i++ {
		window := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(sp.windowSize-1)))
		windowed[i] = analysisWindow[i] * window
	}

	complexFFT := sp.fft.Coefficients(nil, windowed)

	numBins := fftSize / 2
	magnitudes := make([]float64, numBins)
	binSize := float64(sp.sampleRate) / float64(fftSize)

	hannCorrection := 1.0 / 0.5
	for i := 0; i < numBins; i++ {
		real := real(complexFFT[i])
		imag := imag(complexFFT[i])
		magnitudes[i] = math.Sqrt(real*real+imag*imag) / float64(sp.windowSize) * 2 * hannCorrection
	}

	peakIndices := findPeaks(magnitudes)

	peaks := make([]StreamPeak, len(peakIndices))
	for i, idx := range peakIndices {
		interpIndex, interpMag := parabolicInterpolation(magnitudes, idx)
		var change float64
		if len(sp.prevMagnitude) > idx {
			change = (interpMag - sp.prevMagnitude[idx]) / (sp.prevMagnitude[idx] + 1e-10) * 100
		}
		peaks[i] = StreamPeak{
			Frequency: interpIndex * binSize,
			Magnitude: interpMag,
			Change:    change,
		}
	}

	sp.prevMagnitude = make([]float64, numBins)
	copy(sp.prevMagnitude, magnitudes)

	if len(peaks) > 3 {
		peaks = peaks[:3]
	}

	sp.prevPeaks = make([]StreamPeak, len(peaks))
	copy(sp.prevPeaks, peaks)

	if len(sp.buffer) > sp.hopSize {
		sp.buffer = sp.buffer[sp.hopSize:]
	}

	return &StreamResult{
		Peaks: peaks,
		Seq:   sp.seq,
	}
}

func (sp *StreamProcessor) Reset() {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	sp.buffer = sp.buffer[:0]
	sp.prevPeaks = sp.prevPeaks[:0]
	sp.prevMagnitude = sp.prevMagnitude[:0]
	sp.seq = 0
}

func (sp *StreamProcessor) GetBufferLevel() int {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return len(sp.buffer)
}
