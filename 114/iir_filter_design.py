import numpy as np
from scipy import signal
from typing import Tuple, Optional


class IIRFilterDesigner:
    def __init__(self, fs: float):
        self.fs = fs
        self.nyquist = fs / 2.0

    def _normalize_freq(self, freq: float) -> float:
        return freq / self.nyquist

    def butterworth(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        wp = self._normalize_freq(np.array(passband_freq))
        ws = self._normalize_freq(np.array(stopband_freq))
        
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        N, Wn = signal.buttord(wp, ws, passband_ripple, stopband_attenuation)
        b, a = signal.butter(N, Wn, btype=btype, output='ba')
        
        return b, a, N

    def chebyshev1(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        wp = self._normalize_freq(np.array(passband_freq))
        ws = self._normalize_freq(np.array(stopband_freq))
        
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        N, Wn = signal.cheb1ord(wp, ws, passband_ripple, stopband_attenuation)
        b, a = signal.cheby1(N, passband_ripple, Wn, btype=btype, output='ba')
        
        return b, a, N

    def chebyshev2(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        wp = self._normalize_freq(np.array(passband_freq))
        ws = self._normalize_freq(np.array(stopband_freq))
        
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        N, Wn = signal.cheb2ord(wp, ws, passband_ripple, stopband_attenuation)
        b, a = signal.cheby2(N, stopband_attenuation, Wn, btype=btype, output='ba')
        
        return b, a, N

    def get_frequency_response(self, b: np.ndarray, a: np.ndarray, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        w, h = signal.freqz(b, a, worN=num_points, fs=self.fs)
        return w, 20 * np.log10(np.abs(h))

    def apply_filter(self, b: np.ndarray, a: np.ndarray, signal_data: np.ndarray) -> np.ndarray:
        return signal.lfilter(b, a, signal_data)

    def apply_filter_filtfilt(self, b: np.ndarray, a: np.ndarray, signal_data: np.ndarray) -> np.ndarray:
        return signal.filtfilt(b, a, signal_data)


def print_coefficients(b: np.ndarray, a: np.ndarray, filter_name: str, order: int):
    print(f"\n{'='*60}")
    print(f"{filter_name} 滤波器 (阶数: {order})")
    print('='*60)
    print(f"\n分子系数 (b):")
    for i, coeff in enumerate(b):
        print(f"  b[{i}] = {coeff:.12e}")
    print(f"\n分母系数 (a):")
    for i, coeff in enumerate(a):
        print(f"  a[{i}] = {coeff:.12e}")


def main():
    fs = 1000.0
    designer = IIRFilterDesigner(fs)
    
    print(f"采样频率: {fs} Hz")
    print(f"奈奎斯特频率: {designer.nyquist} Hz")

    print("\n" + "#"*60)
    print("# 示例 1: 低通滤波器")
    print("#"*60)
    
    filter_type = 'lowpass'
    passband_freq = 100.0
    stopband_freq = 150.0
    passband_ripple = 1.0
    stopband_attenuation = 40.0
    
    print(f"\n通带截止频率: {passband_freq} Hz")
    print(f"阻带截止频率: {stopband_freq} Hz")
    print(f"通带最大波纹: {passband_ripple} dB")
    print(f"阻带最小衰减: {stopband_attenuation} dB")

    b_butter, a_butter, N_butter = designer.butterworth(
        filter_type, passband_freq, stopband_freq, passband_ripple, stopband_attenuation
    )
    print_coefficients(b_butter, a_butter, "巴特沃斯", N_butter)

    b_cheb1, a_cheb1, N_cheb1 = designer.chebyshev1(
        filter_type, passband_freq, stopband_freq, passband_ripple, stopband_attenuation
    )
    print_coefficients(b_cheb1, a_cheb1, "切比雪夫I型", N_cheb1)

    b_cheb2, a_cheb2, N_cheb2 = designer.chebyshev2(
        filter_type, passband_freq, stopband_freq, passband_ripple, stopband_attenuation
    )
    print_coefficients(b_cheb2, a_cheb2, "切比雪夫II型", N_cheb2)

    print("\n" + "#"*60)
    print("# 示例 2: 带通滤波器")
    print("#"*60)
    
    filter_type = 'bandpass'
    passband_freq = (80.0, 120.0)
    stopband_freq = (60.0, 140.0)
    passband_ripple = 1.0
    stopband_attenuation = 40.0
    
    print(f"\n通带频率范围: {passband_freq[0]} - {passband_freq[1]} Hz")
    print(f"阻带频率范围: {stopband_freq[0]} - {stopband_freq[1]} Hz")
    print(f"通带最大波纹: {passband_ripple} dB")
    print(f"阻带最小衰减: {stopband_attenuation} dB")

    b_bp, a_bp, N_bp = designer.butterworth(
        filter_type, passband_freq, stopband_freq, passband_ripple, stopband_attenuation
    )
    print_coefficients(b_bp, a_bp, "巴特沃斯带通", N_bp)

    print("\n" + "#"*60)
    print("# 示例 3: 实时信号处理演示")
    print("#"*60)
    
    t = np.linspace(0, 1.0, int(fs), endpoint=False)
    signal_50hz = np.sin(2 * np.pi * 50 * t)
    signal_120hz = 0.5 * np.sin(2 * np.pi * 120 * t)
    signal_200hz = 0.3 * np.sin(2 * np.pi * 200 * t)
    noisy_signal = signal_50hz + signal_120hz + signal_200hz
    
    filtered_signal = designer.apply_filter(b_butter, a_butter, noisy_signal)
    
    print(f"\n原始信号包含: 50Hz + 120Hz + 200Hz")
    print(f"使用低通滤波器 (截止100Hz) 滤波后")
    print(f"滤波前信号能量: {np.sum(noisy_signal**2):.4f}")
    print(f"滤波后信号能量: {np.sum(filtered_signal**2):.4f}")
    print(f"主要保留了50Hz分量")


if __name__ == "__main__":
    main()
