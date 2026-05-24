import numpy as np
from scipy import signal
from typing import Tuple, List, Optional


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
    def __init__(self, fs: float):
        self.fs = fs
        self.nyquist = fs / 2.0
    
    def _normalize_freq(self, freq: float or Tuple[float, float]) -> np.ndarray:
        return np.array(freq) / self.nyquist
    
    def butterworth(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float,
        return_sos: bool = False
    ) -> IIRFilter or SOSFilter:
        wp = self._normalize_freq(passband_freq)
        ws = self._normalize_freq(stopband_freq)
        
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        N, Wn = signal.buttord(wp, ws, passband_ripple, stopband_attenuation)
        
        if return_sos:
            sos = signal.butter(N, Wn, btype=btype, output='sos')
            return SOSFilter(sos, self.fs, f'Butterworth {filter_type}', N)
        else:
            b, a = signal.butter(N, Wn, btype=btype, output='ba')
            return IIRFilter(b, a, self.fs, f'Butterworth {filter_type}', N)
    
    def chebyshev1(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float,
        return_sos: bool = False
    ) -> IIRFilter or SOSFilter:
        wp = self._normalize_freq(passband_freq)
        ws = self._normalize_freq(stopband_freq)
        
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        N, Wn = signal.cheb1ord(wp, ws, passband_ripple, stopband_attenuation)
        
        if return_sos:
            sos = signal.cheby1(N, passband_ripple, Wn, btype=btype, output='sos')
            return SOSFilter(sos, self.fs, f'Chebyshev I {filter_type}', N)
        else:
            b, a = signal.cheby1(N, passband_ripple, Wn, btype=btype, output='ba')
            return IIRFilter(b, a, self.fs, f'Chebyshev I {filter_type}', N)
    
    def chebyshev2(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float,
        return_sos: bool = False
    ) -> IIRFilter or SOSFilter:
        wp = self._normalize_freq(passband_freq)
        ws = self._normalize_freq(stopband_freq)
        
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        N, Wn = signal.cheb2ord(wp, ws, passband_ripple, stopband_attenuation)
        
        if return_sos:
            sos = signal.cheby2(N, stopband_attenuation, Wn, btype=btype, output='sos')
            return SOSFilter(sos, self.fs, f'Chebyshev II {filter_type}', N)
        else:
            b, a = signal.cheby2(N, stopband_attenuation, Wn, btype=btype, output='ba')
            return IIRFilter(b, a, self.fs, f'Chebyshev II {filter_type}', N)
    
    def elliptic(
        self,
        filter_type: str,
        passband_freq: float or Tuple[float, float],
        stopband_freq: float or Tuple[float, float],
        passband_ripple: float,
        stopband_attenuation: float,
        return_sos: bool = False
    ) -> IIRFilter or SOSFilter:
        wp = self._normalize_freq(passband_freq)
        ws = self._normalize_freq(stopband_freq)
        
        btype = filter_type.lower()
        if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        N, Wn = signal.ellipord(wp, ws, passband_ripple, stopband_attenuation)
        
        if return_sos:
            sos = signal.ellip(N, passband_ripple, stopband_attenuation, Wn, btype=btype, output='sos')
            return SOSFilter(sos, self.fs, f'Elliptic {filter_type}', N)
        else:
            b, a = signal.ellip(N, passband_ripple, stopband_attenuation, Wn, btype=btype, output='ba')
            return IIRFilter(b, a, self.fs, f'Elliptic {filter_type}', N)


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


def demonstrate_real_time_processing():
    print("\n" + "#"*60)
    print("# 实时信号处理演示")
    print("#"*60)
    
    fs = 1000.0
    designer = IIRFilterDesigner(fs)
    
    lp_filter = designer.butterworth(
        'lowpass',
        passband_freq=100.0,
        stopband_freq=150.0,
        passband_ripple=1.0,
        stopband_attenuation=40.0,
        return_sos=True
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
    demonstrate_real_time_processing()
