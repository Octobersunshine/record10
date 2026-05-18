package fft

import (
	"math"
	"testing"
)

func generateSineWave(frequency float64, sampleRate int, numSamples int) []float64 {
	samples := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		samples[i] = math.Sin(2 * math.Pi * frequency * t)
	}
	return samples
}

func TestFFTFrequencyAccuracy(t *testing.T) {
	testCases := []struct {
		name      string
		frequency float64
		sampleRate int
		numSamples int
	}{
		{"440Hz_A4", 440.0, 44100, 4096},
		{"1000Hz_1kHz", 1000.0, 44100, 4096},
		{"8000Hz_8kHz", 8000.0, 44100, 4096},
		{"15000Hz_15kHz", 15000.0, 44100, 4096},
		{"440Hz_small_fft", 440.0, 44100, 1024},
		{"1000Hz_48k", 1000.0, 48000, 4096},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			samples := generateSineWave(tc.frequency, tc.sampleRate, tc.numSamples)
			result := ProcessFFT(samples, tc.sampleRate, 3)

			if len(result.Peaks) == 0 {
				t.Fatal("No peaks found")
			}

			peakFreq := result.Peaks[0].Frequency
			errorPercent := math.Abs(peakFreq-tc.frequency) / tc.frequency * 100

			t.Logf("Expected: %.2f Hz, Got: %.2f Hz, Error: %.4f%%",
				tc.frequency, peakFreq, errorPercent)

			if errorPercent > 0.5 {
				t.Errorf("Frequency error %.4f%% exceeds 0.5%% threshold", errorPercent)
			}
		})
	}
}

func TestParabolicInterpolation(t *testing.T) {
	mag := []float64{0.1, 0.5, 1.0, 0.5, 0.1}
	peakIndex := 2

	interpIndex, interpMag := parabolicInterpolation(mag, peakIndex)

	if math.Abs(interpIndex-float64(peakIndex)) > 0.01 {
		t.Errorf("Expected index ~%d, got %.4f", peakIndex, interpIndex)
	}

	if math.Abs(interpMag-1.0) > 0.01 {
		t.Errorf("Expected magnitude ~1.0, got %.4f", interpMag)
	}
}

func TestNextPowerOfTwo(t *testing.T) {
	testCases := []struct {
		input    int
		expected int
	}{
		{1, 1},
		{2, 2},
		{3, 4},
		{5, 8},
		{100, 128},
		{1023, 1024},
		{1024, 1024},
	}

	for _, tc := range testCases {
		result := nextPowerOfTwo(tc.input)
		if result != tc.expected {
			t.Errorf("nextPowerOfTwo(%d) = %d, expected %d", tc.input, result, tc.expected)
		}
	}
}

func TestEmptyInput(t *testing.T) {
	result := ProcessFFT([]float64{}, 44100, 5)
	if len(result.Peaks) != 0 {
		t.Error("Expected no peaks for empty input")
	}
}

func TestMultipleFrequencies(t *testing.T) {
	sampleRate := 44100
	numSamples := 4096
	samples := make([]float64, numSamples)

	freq1 := 440.0
	freq2 := 1000.0
	amp1 := 1.0
	amp2 := 0.5

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		samples[i] = amp1*math.Sin(2*math.Pi*freq1*t) +
			amp2*math.Sin(2*math.Pi*freq2*t)
	}

	result := ProcessFFT(samples, sampleRate, 5)

	if len(result.Peaks) < 2 {
		t.Fatalf("Expected at least 2 peaks, got %d", len(result.Peaks))
	}

	peak1Freq := result.Peaks[0].Frequency
	peak2Freq := result.Peaks[1].Frequency

	error1 := math.Abs(peak1Freq-freq1) / freq1 * 100
	error2 := math.Abs(peak2Freq-freq2) / freq2 * 100

	t.Logf("Peak 1: Expected %.2f Hz, Got %.2f Hz (%.4f%% error)", freq1, peak1Freq, error1)
	t.Logf("Peak 2: Expected %.2f Hz, Got %.2f Hz (%.4f%% error)", freq2, peak2Freq, error2)

	if error1 > 0.5 || error2 > 0.5 {
		t.Error("One or more frequency errors exceed 0.5%")
	}
}
