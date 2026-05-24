import numpy as np
from scipy import signal
from typing import List, Tuple, Optional, Union
import warnings


class FIRFilter:
    def __init__(self, taps: np.ndarray, fs: float, filter_type: str, 
                 bands: List[float], desired: List[float], weights: List[float]):
        self.taps = taps
        self.fs = fs
        self.filter_type = filter_type
        self.bands = bands
        self.desired = desired
        self.weights = weights
        self.order = len(taps) - 1
        self._state = np.zeros(len(taps) - 1)
        
    def process_sample(self, x: float) -> float:
        y = self.taps[0] * x + np.dot(self.taps[1:], self._state)
        self._state[1:] = self._state[:-1]
        self._state[0] = x
        return y
    
    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=np.float64)
        for i, sample in enumerate(x):
            y[i] = self.process_sample(sample)
        return y
    
    def reset(self):
        self._state.fill(0.0)
    
    def apply(self, signal_data: np.ndarray) -> np.ndarray:
        return np.convolve(signal_data, self.taps, mode='same')
    
    def get_frequency_response(self, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        w, h = signal.freqz(self.taps, [1.0], worN=num_points, fs=self.fs)
        return w, 20 * np.log10(np.abs(h) + 1e-10)
    
    def get_phase_response(self, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        w, h = signal.freqz(self.taps, [1.0], worN=num_points, fs=self.fs)
        return w, np.unwrap(np.angle(h))
    
    def get_group_delay(self, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        w, gd = signal.group_delay((self.taps, [1.0]), worN=num_points, fs=self.fs)
        return w, gd
    
    def measure_ripple(self) -> Tuple[float, float]:
        w, mag_db = self.get_frequency_response(num_points=4096)
        bands_array = np.array(self.bands)
        
        passband_ripple = 0.0
        stopband_atten = 0.0
        
        for i in range(0, len(self.desired), 2):
            band_start = self.bands[i]
            band_end = self.bands[i + 1]
            desired_level = self.desired[i]
            
            mask = (w >= band_start) & (w <= band_end)
            if np.any(mask):
                band_mag = mag_db[mask]
                if desired_level > 0.1:
                    ripple = np.max(np.abs(band_mag - 0))
                    passband_ripple = max(passband_ripple, ripple)
                else:
                    atten = np.min(band_mag)
                    stopband_atten = min(stopband_atten, atten)
        
        return passband_ripple, abs(stopband_atten)
    
    def export_c_header(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f"// FIR Filter Coefficients (Parks-McClellan)\n")
            f.write(f"// Type: {self.filter_type}\n")
            f.write(f"// Order: {self.order}\n")
            f.write(f"// Number of taps: {len(self.taps)}\n")
            f.write(f"// Sampling Frequency: {self.fs} Hz\n\n")
            
            f.write(f"#define FIR_NUM_TAPS {len(self.taps)}\n")
            f.write(f"#define FIR_ORDER {self.order}\n")
            f.write(f"#define FIR_FS {self.fs}f\n\n")
            
            f.write(f"const float fir_taps[{len(self.taps)}] = {{\n")
            for i, coeff in enumerate(self.taps):
                f.write(f"    {coeff:.12e}f")
                if i < len(self.taps) - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n")
    
    def get_impulse_response(self) -> np.ndarray:
        return self.taps.copy()


class ParksMcClellanDesigner:
    def __init__(self, fs: float, verbose: bool = False):
        self.fs = fs
        self.nyquist = fs / 2.0
        self.verbose = verbose
    
    def _normalize_bands(self, bands: List[float]) -> List[float]:
        return [f / self.nyquist for f in bands]
    
    def estimate_order(self, bands: List[float], ripple_db: List[float]) -> int:
        if len(bands) < 4:
            raise ValueError("Need at least 2 bands (4 edge frequencies)")
        
        normalized_bands = self._normalize_bands(bands)
        
        min_transition = float('inf')
        for i in range(1, len(normalized_bands) - 1, 2):
            transition = normalized_bands[i + 1] - normalized_bands[i]
            min_transition = min(min_transition, transition)
        
        if min_transition <= 0:
            raise ValueError("Invalid band edges")
        
        max_ripple = min(ripple_db)
        if max_ripple < 0.01:
            max_ripple = 0.01
        
        N = int(np.ceil(-20 * np.log10(np.sqrt(10**(-max_ripple/10) / (10**(-max_ripple/10) + 1))) 
                      / (22 * min_transition)))
        
        N = max(N, 10)
        if N % 2 == 0:
            N += 1
        
        return N
    
    def design_lowpass(
        self,
        passband_freq: float,
        stopband_freq: float,
        passband_ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
        order: Optional[int] = None,
        maxiter: int = 10000
    ) -> FIRFilter:
        bands = [0, passband_freq, stopband_freq, self.nyquist]
        desired = [1.0, 1.0, 0.0, 0.0]
        
        delta_p = 1 - 10**(-passband_ripple_db / 20)
        delta_s = 10**(-stopband_atten_db / 20)
        weights = [1.0, delta_p / delta_s]
        
        if order is None:
            order = self.estimate_order(bands, [passband_ripple_db, stopband_atten_db])
        
        if self.verbose:
            self._print_design_info("Lowpass", bands, desired, weights, order)
        
        normalized_bands = self._normalize_bands(bands)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            taps = signal.remez(
                numtaps=order + 1,
                bands=normalized_bands,
                desired=desired,
                weight=weights,
                fs=2.0,
                maxiter=maxiter
            )
        
        return FIRFilter(taps, self.fs, "Parks-McClellan Lowpass", bands, desired, weights)
    
    def design_highpass(
        self,
        stopband_freq: float,
        passband_freq: float,
        passband_ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
        order: Optional[int] = None,
        maxiter: int = 10000
    ) -> FIRFilter:
        bands = [0, stopband_freq, passband_freq, self.nyquist]
        desired = [0.0, 0.0, 1.0, 1.0]
        
        delta_p = 1 - 10**(-passband_ripple_db / 20)
        delta_s = 10**(-stopband_atten_db / 20)
        weights = [delta_p / delta_s, 1.0]
        
        if order is None:
            order = self.estimate_order(bands, [stopband_atten_db, passband_ripple_db])
        
        if order % 2 == 0:
            order += 1
        
        if self.verbose:
            self._print_design_info("Highpass", bands, desired, weights, order)
        
        normalized_bands = self._normalize_bands(bands)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            taps = signal.remez(
                numtaps=order + 1,
                bands=normalized_bands,
                desired=desired,
                weight=weights,
                fs=2.0,
                maxiter=maxiter
            )
        
        return FIRFilter(taps, self.fs, "Parks-McClellan Highpass", bands, desired, weights)
    
    def design_bandpass(
        self,
        lower_stopband: float,
        lower_passband: float,
        upper_passband: float,
        upper_stopband: float,
        passband_ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
        order: Optional[int] = None,
        maxiter: int = 10000
    ) -> FIRFilter:
        bands = [0, lower_stopband, lower_passband, upper_passband, upper_stopband, self.nyquist]
        desired = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        
        delta_p = 1 - 10**(-passband_ripple_db / 20)
        delta_s = 10**(-stopband_atten_db / 20)
        weights = [delta_p / delta_s, 1.0, delta_p / delta_s]
        
        if order is None:
            order = self.estimate_order(bands, [stopband_atten_db, passband_ripple_db, stopband_atten_db])
        
        if self.verbose:
            self._print_design_info("Bandpass", bands, desired, weights, order)
        
        normalized_bands = self._normalize_bands(bands)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            taps = signal.remez(
                numtaps=order + 1,
                bands=normalized_bands,
                desired=desired,
                weight=weights,
                fs=2.0,
                maxiter=maxiter
            )
        
        return FIRFilter(taps, self.fs, "Parks-McClellan Bandpass", bands, desired, weights)
    
    def design_bandstop(
        self,
        lower_passband: float,
        lower_stopband: float,
        upper_stopband: float,
        upper_passband: float,
        passband_ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
        order: Optional[int] = None,
        maxiter: int = 10000
    ) -> FIRFilter:
        bands = [0, lower_passband, lower_stopband, upper_stopband, upper_passband, self.nyquist]
        desired = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        
        delta_p = 1 - 10**(-passband_ripple_db / 20)
        delta_s = 10**(-stopband_atten_db / 20)
        weights = [1.0, delta_p / delta_s, 1.0]
        
        if order is None:
            order = self.estimate_order(bands, [passband_ripple_db, stopband_atten_db, passband_ripple_db])
        
        if self.verbose:
            self._print_design_info("Bandstop", bands, desired, weights, order)
        
        normalized_bands = self._normalize_bands(bands)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            taps = signal.remez(
                numtaps=order + 1,
                bands=normalized_bands,
                desired=desired,
                weight=weights,
                fs=2.0,
                maxiter=maxiter
            )
        
        return FIRFilter(taps, self.fs, "Parks-McClellan Bandstop", bands, desired, weights)
    
    def design_multiband(
        self,
        bands: List[float],
        desired: List[float],
        weights: Optional[List[float]] = None,
        order: Optional[int] = None,
        filter_type_name: str = "Multiband",
        maxiter: int = 10000
    ) -> FIRFilter:
        if len(bands) % 2 != 0:
            raise ValueError("Bands must have even number of elements (pairs of edges)")
        if len(desired) != len(bands):
            raise ValueError("Desired must have same length as bands")
        
        if bands[-1] != self.nyquist:
            bands = bands.copy()
            bands.append(self.nyquist)
            desired.append(desired[-1])
        
        if weights is None:
            weights = [1.0] * (len(bands) // 2)
        
        if order is None:
            ripples = [3.0] * (len(bands) // 2)
            order = self.estimate_order(bands, ripples)
        
        if self.verbose:
            self._print_design_info(filter_type_name, bands, desired, weights, order)
        
        normalized_bands = self._normalize_bands(bands)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            taps = signal.remez(
                numtaps=order + 1,
                bands=normalized_bands,
                desired=desired,
                weight=weights,
                fs=2.0,
                maxiter=maxiter
            )
        
        return FIRFilter(taps, self.fs, f"Parks-McClellan {filter_type_name}", bands, desired, weights)
    
    def design_arbirary_response(
        self,
        bands: List[float],
        amplitude_response: callable,
        num_points_per_band: int = 10,
        **kwargs
    ) -> FIRFilter:
        band_edges = []
        desired = []
        
        for i in range(0, len(bands) - 1):
            band_start = bands[i]
            band_end = bands[i + 1]
            freqs = np.linspace(band_start, band_end, num_points_per_band)
            for f in freqs:
                band_edges.append(f)
                desired.append(amplitude_response(f))
        
        return self.design_multiband(band_edges, desired, **kwargs)
    
    def _print_design_info(self, name: str, bands: List[float], desired: List[float], 
                           weights: List[float], order: int):
        print(f"\n{'='*60}")
        print(f"Parks-McClellan {name} 滤波器设计")
        print(f"{'='*60}")
        print(f"采样频率: {self.fs} Hz")
        print(f"滤波器阶数: {order}")
        print(f"抽头数量: {order + 1}")
        print(f"\n频带定义:")
        
        for i in range(0, len(bands), 2):
            band_idx = i // 2
            print(f"  频带 {band_idx + 1}: [{bands[i]:.2f}, {bands[i+1]:.2f}] Hz")
            print(f"    期望幅度: {desired[i]:.4f}")
            if band_idx < len(weights):
                print(f"    权重: {weights[band_idx]:.4f}")
        print('='*60)


def print_filter_info(fir_filter: FIRFilter):
    print(f"\n{'='*60}")
    print(f"FIR 滤波器信息")
    print(f"{'='*60}")
    print(f"类型: {fir_filter.filter_type}")
    print(f"阶数: {fir_filter.order}")
    print(f"抽头数: {len(fir_filter.taps)}")
    print(f"采样频率: {fir_filter.fs} Hz")
    print(f"\n抽头系数 (前10个/后10个):")
    
    if len(fir_filter.taps) <= 20:
        for i, tap in enumerate(fir_filter.taps):
            print(f"  h[{i}] = {tap:.12e}")
    else:
        for i in range(10):
            print(f"  h[{i}] = {fir_filter.taps[i]:.12e}")
        print("  ...")
        for i in range(len(fir_filter.taps) - 10, len(fir_filter.taps)):
            print(f"  h[{i}] = {fir_filter.taps[i]:.12e}")
    
    ripple, atten = fir_filter.measure_ripple()
    print(f"\n实测性能:")
    print(f"  通带波纹: {ripple:.3f} dB")
    print(f"  阻带衰减: {atten:.1f} dB")


def demonstrate_parks_mcclellan():
    print("\n" + "#"*70)
    print("# Parks-McClellan 等波纹 FIR 滤波器演示")
    print("#"*70)
    
    fs = 1000.0
    designer = ParksMcClellanDesigner(fs, verbose=True)
    
    print("\n" + "="*70)
    print("示例 1: 低通滤波器")
    print("="*70)
    
    lp_filter = designer.design_lowpass(
        passband_freq=100.0,
        stopband_freq=150.0,
        passband_ripple_db=1.0,
        stopband_atten_db=40.0
    )
    print_filter_info(lp_filter)
    
    print("\n" + "="*70)
    print("示例 2: 带通滤波器")
    print("="*70)
    
    bp_filter = designer.design_bandpass(
        lower_stopband=60.0,
        lower_passband=80.0,
        upper_passband=120.0,
        upper_stopband=140.0,
        passband_ripple_db=1.0,
        stopband_atten_db=40.0
    )
    print_filter_info(bp_filter)
    
    print("\n" + "="*70)
    print("示例 3: 多频带滤波器 (三频带)")
    print("="*70)
    
    mb_designer = ParksMcClellanDesigner(fs, verbose=False)
    mb_filter = mb_designer.design_multiband(
        bands=[0, 50, 70, 130, 150, 200, 250, fs/2],
        desired=[0, 0, 1, 1, 0, 0, 0.5, 0.5],
        weights=[2.0, 1.0, 2.0, 1.0],
        order=100,
        filter_type_name="三频带滤波器"
    )
    print_filter_info(mb_filter)
    
    print("\n" + "="*70)
    print("示例 4: 实时信号处理演示")
    print("="*70)
    
    t = np.linspace(0, 0.02, 21, endpoint=False)
    signal_50hz = np.sin(2 * np.pi * 50 * t)
    signal_100hz = 0.5 * np.sin(2 * np.pi * 100 * t)
    signal_200hz = 0.3 * np.sin(2 * np.pi * 200 * t)
    test_signal = signal_50hz + signal_100hz + signal_200hz
    
    print(f"\n测试信号: 50Hz + 100Hz + 200Hz")
    print(f"滤波器: 低通 100Hz")
    
    lp_filter.reset()
    filtered = lp_filter.process_block(test_signal)
    
    print(f"\n逐样本处理:")
    print(f"{'样本':>6} {'输入':>12} {'输出':>12}")
    print("-" * 32)
    for i in range(len(test_signal)):
        print(f"{i:>6} {test_signal[i]:>12.6f} {filtered[i]:>12.6f}")
    
    print(f"\n✓ 演示完成!")
    print(f"  (100Hz 部分保留，200Hz 被衰减)")


if __name__ == "__main__":
    demonstrate_parks_mcclellan()
