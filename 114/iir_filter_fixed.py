import numpy as np
from scipy import signal
from typing import Tuple, List, Optional


class BilinearTransformer:
    @staticmethod
    def prewarp(freq_hz: float or np.ndarray, fs: float) -> float or np.ndarray:
        T = 1.0 / fs
        omega_d = 2 * np.pi * freq_hz
        omega_a = (2.0 / T) * np.tan(omega_d * T / 2.0)
        return omega_a / (2 * np.pi)
    
    @staticmethod
    def prewarp_angular(omega_d: float or np.ndarray, fs: float) -> float or np.ndarray:
        T = 1.0 / fs
        omega_a = (2.0 / T) * np.tan(omega_d * T / 2.0)
        return omega_a
    
    @staticmethod
    def digital_to_analog_freq(f_digital: float or np.ndarray, fs: float) -> float or np.ndarray:
        return BilinearTransformer.prewarp(f_digital, fs)
    
    @staticmethod
    def analog_to_digital_freq(f_analog: float or np.ndarray, fs: float) -> float or np.ndarray:
        T = 1.0 / fs
        omega_a = 2 * np.pi * f_analog
        omega_d = (2.0 / T) * np.arctan(omega_a * T / 2.0)
        return omega_d / (2 * np.pi)


class IIRFilter:
    def __init__(self, b: np.ndarray, a: np.ndarray, fs: float, filter_type: str, order: int):
        self.b = b
        self.a = a
        self.fs = fs
        self.filter_type = filter_type
        self.order = order
        self._state = np.zeros(max(len(b), len(a)) - 1)
        
    def process_sample(self, x: float) -> float:
        y = self.b[0] * x + np.dot(self.b[1:], self._state[:len(self.b)-1])
        y -= np.dot(self.a[1:], self._state[:len(self.a)-1])
        y /= self.a[0]
        
        self._state[1:] = self._state[:-1]
        self._state[0] = x
        
        return y
    
    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for i, sample in enumerate(x):
            y[i] = self.process_sample(sample)
        return y
    
    def reset(self):
        self._state.fill(0.0)
    
    def get_sos(self) -> np.ndarray:
        return signal.tf2sos(self.b, self.a)
    
    def get_frequency_response(self, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        w, h = signal.freqz(self.b, self.a, worN=num_points, fs=self.fs)
        return w, 20 * np.log10(np.abs(h) + 1e-10)
    
    def get_group_delay(self, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        w, gd = signal.group_delay((self.b, self.a), worN=num_points, fs=self.fs)
        return w, gd
    
    def find_cutoff_frequency(self, target_db: float = -3.0) -> float:
        w, mag_db = self.get_frequency_response(num_points=4096)
        idx = np.argmin(np.abs(mag_db - target_db))
        return w[idx]
    
    def export_c_header(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f"// IIR Filter Coefficients\n")
            f.write(f"// Type: {self.filter_type}\n")
            f.write(f"// Order: {self.order}\n")
            f.write(f"// Sampling Frequency: {self.fs} Hz\n\n")
            
            f.write(f"#define FILTER_ORDER {self.order}\n")
            f.write(f"#define FILTER_FS {self.fs}f\n\n")
            
            f.write(f"const float b[] = {{\n")
            for i, coeff in enumerate(self.b):
                f.write(f"    {coeff:.12e}f")
                if i < len(self.b) - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")
            
            f.write(f"const float a[] = {{\n")
            for i, coeff in enumerate(self.a):
                f.write(f"    {coeff:.12e}f")
                if i < len(self.a) - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n")


class SOSFilter:
    def __init__(self, sos: np.ndarray, fs: float, filter_type: str, order: int):
        self.sos = sos
        self.fs = fs
        self.filter_type = filter_type
        self.order = order
        self._states = np.zeros((len(sos), 2))
        
    def process_sample(self, x: float) -> float:
        y = x
        for i, section in enumerate(self.sos):
            b0, b1, b2, a0, a1, a2 = section
            y_new = (b0 * y + b1 * self._states[i, 0] + b2 * self._states[i, 1]) / a0
            y_new -= (a1 * self._states[i, 0] + a2 * self._states[i, 1]) / a0
            
            self._states[i, 1] = self._states[i, 0]
            self._states[i, 0] = y
            y = y_new
        return y
    
    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for i, sample in enumerate(x):
            y[i] = self.process_sample(sample)
        return y
    
    def reset(self):
        self._states.fill(0.0)
    
    def get_frequency_response(self, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        w, h = signal.sosfreqz(self.sos, worN=num_points, fs=self.fs)
        return w, 20 * np.log10(np.abs(h) + 1e-10)
    
    def find_cutoff_frequency(self, target_db: float = -3.0) -> float:
        w, mag_db = self.get_frequency_response(num_points=4096)
        idx = np.argmin(np.abs(mag_db - target_db))
        return w[idx]
    
    def export_c_header(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f"// IIR Filter - Second-Order Sections (SOS)\n")
            f.write(f"// Type: {self.filter_type}\n")
            f.write(f"// Order: {self.order}\n")
            f.write(f"// Sampling Frequency: {self.fs} Hz\n")
            f.write(f"// Number of sections: {len(self.sos)}\n\n")
            
            f.write(f"#define NUM_SECTIONS {len(self.sos)}\n")
            f.write(f"#define FILTER_ORDER {self.order}\n\n")
            
            f.write("// Each section: [b0, b1, b2, a0, a1, a2]\n")
            f.write(f"const float sos[{len(self.sos)}][6] = {{\n")
            for i, section in enumerate(self.sos):
                f.write("    {")
                for j, coeff in enumerate(section):
                    f.write(f"{coeff:.12e}f")
                    if j < 5:
                        f.write(", ")
                f.write("}")
                if i < len(self.sos) - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n")


class IIRFilterDesigner:
    def __init__(self, fs: float, verbose: bool = False):
        self.fs = fs
        self.nyquist = fs / 2.0
        self.verbose = verbose
        self.transformer = BilinearTransformer()
    
    def _normalize_freq(self, freq: float or Tuple[float, float]) -> np.ndarray:
        return np.array(freq) / self.nyquist
    
    def _prewarp_frequencies(
        self,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        wp = np.array(passband_freq)
        ws = np.array(stopband_freq)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"双线性变换频率预畸变")
            print(f"{'='*60}")
            print(f"采样频率 fs = {self.fs} Hz")
            print(f"采样周期 T = {1/self.fs:.6e} s")
            print(f"\n预畸变公式:")
            print(f"  Ω_a = (2/T) * tan(ω_d * T / 2)")
            print(f"  ω_d = 2π * f_d / fs")
            print(f"  f_a = Ω_a / (2π)")
            print(f"\n数字频率 → 模拟频率 (预畸变后):")
        
        wp_prewarped = self.transformer.prewarp(wp, self.fs)
        ws_prewarped = self.transformer.prewarp(ws, self.fs)
        
        if self.verbose:
            if wp.ndim == 0:
                print(f"  通带: {wp:.2f} Hz → {wp_prewarped:.2f} Hz")
            else:
                print(f"  通带: [{wp[0]:.2f}, {wp[1]:.2f}] Hz → [{wp_prewarped[0]:.2f}, {wp_prewarped[1]:.2f}] Hz")
            if ws.ndim == 0:
                print(f"  阻带: {ws:.2f} Hz → {ws_prewarped:.2f} Hz")
            else:
                print(f"  阻带: [{ws[0]:.2f}, {ws[1]:.2f}] Hz → [{ws_prewarped[0]:.2f}, {ws_prewarped[1]:.2f}] Hz")
        
        return wp_prewarped, ws_prewarped
    
    def _analog_butterworth(
        self,
        filter_type: str,
        wp: np.ndarray,
        ws: np.ndarray,
        passband_ripple: float,
        stopband_attenuation: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        N, Wn = signal.buttord(wp, ws, passband_ripple, stopband_attenuation, analog=True)
        b, a = signal.butter(N, Wn, btype=filter_type, analog=True, output='ba')
        return b, a, N
    
    def _analog_chebyshev1(
        self,
        filter_type: str,
        wp: np.ndarray,
        ws: np.ndarray,
        passband_ripple: float,
        stopband_attenuation: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        N, Wn = signal.cheb1ord(wp, ws, passband_ripple, stopband_attenuation, analog=True)
        b, a = signal.cheby1(N, passband_ripple, Wn, btype=filter_type, analog=True, output='ba')
        return b, a, N
    
    def _analog_chebyshev2(
        self,
        filter_type: str,
        wp: np.ndarray,
        ws: np.ndarray,
        passband_ripple: float,
        stopband_attenuation: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        N, Wn = signal.cheb2ord(wp, ws, passband_ripple, stopband_attenuation, analog=True)
        b, a = signal.cheby2(N, stopband_attenuation, Wn, btype=filter_type, analog=True, output='ba')
        return b, a, N
    
    def _bilinear_transform(
        self,
        b_analog: np.ndarray,
        a_analog: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        b_digital, a_digital = signal.bilinear(b_analog, a_analog, self.fs)
        return b_digital, a_digital
    
    def butterworth(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float,
        return_sos: bool = False,
        prewarp: bool = True
    ) -> IIRFilter or SOSFilter:
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        if prewarp:
            wp, ws = self._prewarp_frequencies(passband_freq, stopband_freq)
            b, a, N = self._analog_butterworth(btype, wp, ws, passband_ripple, stopband_attenuation)
            b, a = self._bilinear_transform(b, a)
        else:
            wp = self._normalize_freq(passband_freq)
            ws = self._normalize_freq(stopband_freq)
            N, Wn = signal.buttord(wp, ws, passband_ripple, stopband_attenuation)
            b, a = signal.butter(N, Wn, btype=btype, output='ba')
        
        if return_sos:
            sos = signal.tf2sos(b, a)
            return SOSFilter(sos, self.fs, f'Butterworth {filter_type}', N)
        else:
            return IIRFilter(b, a, self.fs, f'Butterworth {filter_type}', N)
    
    def chebyshev1(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float,
        return_sos: bool = False,
        prewarp: bool = True
    ) -> IIRFilter or SOSFilter:
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        if prewarp:
            wp, ws = self._prewarp_frequencies(passband_freq, stopband_freq)
            b, a, N = self._analog_chebyshev1(btype, wp, ws, passband_ripple, stopband_attenuation)
            b, a = self._bilinear_transform(b, a)
        else:
            wp = self._normalize_freq(passband_freq)
            ws = self._normalize_freq(stopband_freq)
            N, Wn = signal.cheb1ord(wp, ws, passband_ripple, stopband_attenuation)
            b, a = signal.cheby1(N, passband_ripple, Wn, btype=btype, output='ba')
        
        if return_sos:
            sos = signal.tf2sos(b, a)
            return SOSFilter(sos, self.fs, f'Chebyshev I {filter_type}', N)
        else:
            return IIRFilter(b, a, self.fs, f'Chebyshev I {filter_type}', N)
    
    def chebyshev2(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float,
        return_sos: bool = False,
        prewarp: bool = True
    ) -> IIRFilter or SOSFilter:
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        if prewarp:
            wp, ws = self._prewarp_frequencies(passband_freq, stopband_freq)
            b, a, N = self._analog_chebyshev2(btype, wp, ws, passband_ripple, stopband_attenuation)
            b, a = self._bilinear_transform(b, a)
        else:
            wp = self._normalize_freq(passband_freq)
            ws = self._normalize_freq(stopband_freq)
            N, Wn = signal.cheb2ord(wp, ws, passband_ripple, stopband_attenuation)
            b, a = signal.cheby2(N, stopband_attenuation, Wn, btype=btype, output='ba')
        
        if return_sos:
            sos = signal.tf2sos(b, a)
            return SOSFilter(sos, self.fs, f'Chebyshev II {filter_type}', N)
        else:
            return IIRFilter(b, a, self.fs, f'Chebyshev II {filter_type}', N)


def print_filter_info(filter_obj: IIRFilter or SOSFilter):
    print(f"\n{'='*60}")
    print(f"滤波器类型: {filter_obj.filter_type}")
    print(f"滤波器阶数: {filter_obj.order}")
    print(f"采样频率: {filter_obj.fs} Hz")
    print('='*60)
    
    if isinstance(filter_obj, IIRFilter):
        print(f"\n分子系数 (b):")
        for i, coeff in enumerate(filter_obj.b):
            print(f"  b[{i}] = {coeff:.12e}")
        print(f"\n分母系数 (a):")
        for i, coeff in enumerate(filter_obj.a):
            print(f"  a[{i}] = {coeff:.12e}")
    elif isinstance(filter_obj, SOSFilter):
        print(f"\n二阶节系数 (SOS):")
        for i, section in enumerate(filter_obj.sos):
            print(f"\n  Section {i+1}:")
            print(f"    b = [{section[0]:.12e}, {section[1]:.12e}, {section[2]:.12e}]")
            print(f"    a = [{section[3]:.12e}, {section[4]:.12e}, {section[5]:.12e}]")


def demonstrate_prewarp_effect():
    print("\n" + "#"*70)
    print("# 双线性变换频率预畸变演示")
    print("#"*70)
    
    fs = 1000.0
    designer_verbose = IIRFilterDesigner(fs, verbose=True)
    
    target_passband = 100.0
    target_stopband = 150.0
    
    print(f"\n设计目标:")
    print(f"  通带截止频率 (目标): {target_passband} Hz")
    print(f"  阻带截止频率 (目标): {target_stopband} Hz")
    print(f"  通带波纹: 1 dB")
    print(f"  阻带衰减: 40 dB")
    
    print(f"\n" + "="*70)
    print(f"方法 1: 无预畸变 (直接使用 scipy 数字滤波器设计)")
    print(f"="*70)
    designer_simple = IIRFilterDesigner(fs, verbose=False)
    filter_no_prewarp = designer_simple.butterworth(
        'lowpass', target_passband, target_stopband, 1.0, 40.0, prewarp=False
    )
    actual_cutoff_no_prewarp = filter_no_prewarp.find_cutoff_frequency(-3.0)
    print(f"  实际 -3dB 截止频率: {actual_cutoff_no_prewarp:.2f} Hz")
    print(f"  偏差: {actual_cutoff_no_prewarp - target_passband:+.2f} Hz")
    
    print(f"\n" + "="*70)
    print(f"方法 2: 有预畸变 (模拟→双线性变换)")
    print(f"="*70)
    filter_with_prewarp = designer_verbose.butterworth(
        'lowpass', target_passband, target_stopband, 1.0, 40.0, prewarp=True
    )
    actual_cutoff_with_prewarp = filter_with_prewarp.find_cutoff_frequency(-3.0)
    print(f"\n  实际 -3dB 截止频率: {actual_cutoff_with_prewarp:.2f} Hz")
    print(f"  偏差: {actual_cutoff_with_prewarp - target_passband:+.2f} Hz")
    
    print(f"\n" + "="*70)
    print(f"预畸变效果对比:")
    print(f"="*70)
    print(f"  目标截止频率: {target_passband} Hz")
    print(f"  无预畸变: {actual_cutoff_no_prewarp:.2f} Hz (偏差: {actual_cutoff_no_prewarp - target_passband:+.2f} Hz)")
    print(f"  有预畸变: {actual_cutoff_with_prewarp:.2f} Hz (偏差: {actual_cutoff_with_prewarp - target_passband:+.2f} Hz)")
    
    return filter_no_prewarp, filter_with_prewarp


def show_prewarp_table():
    print("\n" + "#"*70)
    print("# 不同频率下的预畸变效果 (fs=1000Hz)")
    print("#"*70)
    
    fs = 1000.0
    transformer = BilinearTransformer()
    
    test_freqs = [10, 50, 100, 200, 300, 400, 450, 480]
    
    print(f"\n{'数字频率 (Hz)':>15} {'预畸变后 (Hz)':>15} {'偏差 (Hz)':>15} {'相对偏差 (%)':>15}")
    print("-" * 65)
    
    for f_d in test_freqs:
        f_a = transformer.prewarp(f_d, fs)
        diff = f_a - f_d
        rel_diff = (diff / f_d) * 100
        print(f"{f_d:>15.1f} {f_a:>15.2f} {diff:>+15.2f} {rel_diff:>+14.2f}%")
    
    print(f"\n说明: 当频率接近奈奎斯特频率 (fs/2={fs/2}Hz) 时，预畸变效果更显著")


def real_time_processing_demo():
    print("\n" + "#"*70)
    print("# 实时信号处理演示 (带预畸变的滤波器)")
    print("#"*70)
    
    fs = 1000.0
    designer = IIRFilterDesigner(fs, verbose=False)
    
    lp_filter = designer.butterworth(
        'lowpass',
        passband_freq=100.0,
        stopband_freq=150.0,
        passband_ripple=1.0,
        stopband_attenuation=40.0,
        return_sos=True,
        prewarp=True
    )
    
    print_filter_info(lp_filter)
    
    t = np.linspace(0, 0.01, 11, endpoint=False)
    test_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)
    
    print(f"\n逐样本处理演示:")
    print(f"{'样本':>6} {'输入':>12} {'输出':>12}")
    print("-" * 32)
    
    lp_filter.reset()
    for i, x in enumerate(test_signal):
        y = lp_filter.process_sample(x)
        print(f"{i:>6} {x:>12.6f} {y:>12.6f}")
    
    print(f"\n说明: 200Hz 分量被显著衰减，50Hz 分量保留")


if __name__ == "__main__":
    show_prewarp_table()
    demonstrate_prewarp_effect()
    real_time_processing_demo()
