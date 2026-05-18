package fft

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/dsp/fourier"
)

type Peak struct {
	Frequency float64 `json:"frequency"`
	Magnitude float64 `json:"magnitude"`
}

type FFTResult struct {
	Peaks []Peak `json:"peaks"`
}

func nextPowerOfTwo(n int) int {
	if n <= 0 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n++
	return n
}

func parabolicInterpolation(mag []float64, peakIndex int) (float64, float64) {
	if peakIndex <= 0 || peakIndex >= len(mag)-1 {
		return float64(peakIndex), mag[peakIndex]
	}

	y0 := mag[peakIndex-1]
	y1 := mag[peakIndex]
	y2 := mag[peakIndex+1]

	offset := (y2 - y0) / (2 * (2*y1 - y0 - y2))
	interpMag := y1 - 0.25*(y0-y2)*offset

	return float64(peakIndex) + offset, interpMag
}

func ProcessFFT(samples []float64, sampleRate int, numPeaks int) *FFTResult {
	n := len(samples)
	if n == 0 {
		return &FFTResult{Peaks: []Peak{}}
	}

	fftSize := nextPowerOfTwo(n)

	windowed := make([]float64, fftSize)
	for i := 0; i < n; i++ {
		window := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
		windowed[i] = samples[i] * window
	}

	fft := fourier.NewFFT(fftSize)
	complexFFT := fft.Coefficients(nil, windowed)

	numBins := fftSize / 2
	magnitudes := make([]float64, numBins)
	binSize := float64(sampleRate) / float64(fftSize)

	hannCorrection := 1.0 / 0.5
	for i := 0; i < numBins; i++ {
		real := real(complexFFT[i])
		imag := imag(complexFFT[i])
		magnitudes[i] = math.Sqrt(real*real+imag*imag) / float64(n) * 2 * hannCorrection
	}

	peakIndices := findPeaks(magnitudes)

	peaks := make([]Peak, len(peakIndices))
	for i, idx := range peakIndices {
		interpIndex, interpMag := parabolicInterpolation(magnitudes, idx)
		peaks[i] = Peak{
			Frequency: interpIndex * binSize,
			Magnitude: interpMag,
		}
	}

	sort.Slice(peaks, func(i, j int) bool {
		return peaks[i].Magnitude > peaks[j].Magnitude
	})

	if len(peaks) > numPeaks {
		peaks = peaks[:numPeaks]
	}

	return &FFTResult{Peaks: peaks}
}

func findPeaks(magnitudes []float64) []int {
	var peaks []int
	threshold := 0.01

	for i := 1; i < len(magnitudes)-1; i++ {
		if magnitudes[i] > threshold &&
			magnitudes[i] > magnitudes[i-1] &&
			magnitudes[i] > magnitudes[i+1] {
			peaks = append(peaks, i)
		}
	}

	return peaks
}
