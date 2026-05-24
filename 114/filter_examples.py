from iir_filter_design import IIRFilterDesigner, print_coefficients
import numpy as np


def example_ecg_filter():
    print("="*70)
    print("示例 1: ECG 心电信号滤波器设计")
    print("="*70)
    
    fs = 1000.0
    designer = IIRFilterDesigner(fs)
    
    print(f"\n采样频率: {fs} Hz")
    print("设计目标: 去除 50Hz 工频干扰和高频噪声")
    print("保留: 0.5Hz - 40Hz 的 ECG 信号")
    
    filter_type = 'bandpass'
    passband_freq = (0.5, 40.0)
    stopband_freq = (0.3, 50.0)
    passband_ripple = 1.0
    stopband_attenuation = 40.0
    
    print(f"\n通带: {passband_freq[0]} - {passband_freq[1]} Hz")
    print(f"阻带: < {stopband_freq[0]} Hz 和 > {stopband_freq[1]} Hz")
    print(f"通带波纹: {passband_ripple} dB")
    print(f"阻带衰减: {stopband_attenuation} dB")
    
    b, a, N = designer.butterworth(
        filter_type, passband_freq, stopband_freq, passband_ripple, stopband_attenuation
    )
    print_coefficients(b, a, "ECG 巴特沃斯带通", N)
    
    return b, a


def example_audio_lowpass():
    print("\n" + "="*70)
    print("示例 2: 音频低通滤波器 (切比雪夫I型)")
    print("="*70)
    
    fs = 44100.0
    designer = IIRFilterDesigner(fs)
    
    print(f"\n采样频率: {fs} Hz")
    print("设计目标: 8kHz 低通，去除高频噪声")
    
    filter_type = 'lowpass'
    passband_freq = 8000.0
    stopband_freq = 10000.0
    passband_ripple = 0.5
    stopband_attenuation = 60.0
    
    print(f"\n通带截止: {passband_freq} Hz")
    print(f"阻带截止: {stopband_freq} Hz")
    print(f"通带波纹: {passband_ripple} dB")
    print(f"阻带衰减: {stopband_attenuation} dB")
    
    b, a, N = designer.chebyshev1(
        filter_type, passband_freq, stopband_freq, passband_ripple, stopband_attenuation
    )
    print_coefficients(b, a, "音频切比雪夫I型低通", N)
    
    return b, a


def example_notch_filter():
    print("\n" + "="*70)
    print("示例 3: 50Hz 陷波滤波器 (带阻)")
    print("="*70)
    
    fs = 2000.0
    designer = IIRFilterDesigner(fs)
    
    print(f"\n采样频率: {fs} Hz")
    print("设计目标: 去除 50Hz 工频干扰")
    
    filter_type = 'bandstop'
    passband_freq = (45.0, 55.0)
    stopband_freq = (49.0, 51.0)
    passband_ripple = 1.0
    stopband_attenuation = 60.0
    
    print(f"\n通带: < {passband_freq[0]} Hz 和 > {passband_freq[1]} Hz")
    print(f"阻带: {stopband_freq[0]} - {stopband_freq[1]} Hz")
    print(f"通带波纹: {passband_ripple} dB")
    print(f"阻带衰减: {stopband_attenuation} dB")
    
    b, a, N = designer.chebyshev2(
        filter_type, passband_freq, stopband_freq, passband_ripple, stopband_attenuation
    )
    print_coefficients(b, a, "50Hz 陷波 (切比雪夫II型)", N)
    
    return b, a


def example_highpass_filter():
    print("\n" + "="*70)
    print("示例 4: 高通滤波器 (去除直流漂移)")
    print("="*70)
    
    fs = 500.0
    designer = IIRFilterDesigner(fs)
    
    print(f"\n采样频率: {fs} Hz")
    print("设计目标: 去除直流和低频漂移")
    
    filter_type = 'highpass'
    passband_freq = 1.0
    stopband_freq = 0.5
    passband_ripple = 1.0
    stopband_attenuation = 40.0
    
    print(f"\n通带截止: {passband_freq} Hz")
    print(f"阻带截止: {stopband_freq} Hz")
    print(f"通带波纹: {passband_ripple} dB")
    print(f"阻带衰减: {stopband_attenuation} dB")
    
    b, a, N = designer.butterworth(
        filter_type, passband_freq, stopband_freq, passband_ripple, stopband_attenuation
    )
    print_coefficients(b, a, "直流去除高通", N)
    
    return b, a


def real_time_processing_demo():
    print("\n" + "="*70)
    print("示例 5: 实时信号处理演示 (伪代码)")
    print("="*70)
    
    print("""
实时信号处理差分方程:
y[n] = (b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M] 
                 - a[1]*y[n-1] - ... - a[N]*y[n-N]) / a[0]

其中:
  x[n] - 当前输入样本
  y[n] - 当前输出样本
  b[] - 分子系数 (FIR 部分)
  a[] - 分母系数 (IIR 部分)
""")
    
    print("C语言实时处理示例代码:")
    print("""
// 二阶节实时处理 (推荐使用二阶节避免数值问题)
typedef struct {
    float b0, b1, b2;  // 分子系数
    float a1, a2;      // 分母系数 (a0 = 1)
    float x1, x2;      // 输入状态
    float y1, y2;      // 输出状态
} Biquad;

float biquad_process(Biquad *f, float x) {
    float y = f->b0 * x + f->b1 * f->x1 + f->b2 * f->x2
                        - f->a1 * f->y1 - f->a2 * f->y2;
    f->x2 = f->x1;
    f->x1 = x;
    f->y2 = f->y1;
    f->y1 = y;
    return y;
}
""")


def export_coefficients_for_c(b, a, filename):
    """导出滤波器系数为 C 语言头文件格式"""
    with open(filename, 'w') as f:
        f.write(f"// IIR 滤波器系数\n")
        f.write(f"// 阶数: {max(len(b)-1, len(a)-1)}\n\n")
        f.write(f"#define FILTER_ORDER {max(len(b)-1, len(a)-1)}\n\n")
        
        f.write(f"const float b[] = {{\n")
        for i, coeff in enumerate(b):
            f.write(f"    {coeff:.12e}f")
            if i < len(b) - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        f.write(f"const float a[] = {{\n")
        for i, coeff in enumerate(a):
            f.write(f"    {coeff:.12e}f")
            if i < len(a) - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n")
    print(f"系数已导出到: {filename}")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# IIR 数字滤波器设计 - 完整示例")
    print("#"*70)
    
    b_ecg, a_ecg = example_ecg_filter()
    b_audio, a_audio = example_audio_lowpass()
    b_notch, a_notch = example_notch_filter()
    b_hp, a_hp = example_highpass_filter()
    real_time_processing_demo()
    
    print("\n" + "="*70)
    print("滤波器选型指南:")
    print("="*70)
    print("""
巴特沃斯 (Butterworth):
  ✓ 通带和阻带都平坦 (最大平坦)
  ✓ 相位响应较好
  ✗ 需要较高阶数达到相同衰减
  → 适合: 通用滤波、音频、ECG

切比雪夫I型 (Chebyshev Type I):
  ✓ 通带有波纹，阻带单调
  ✓ 较低阶数达到高衰减
  ✗ 相位响应较差
  → 适合: 对相位要求不高，需要陡过渡带

切比雪夫II型 (Chebyshev Type II):
  ✓ 通带平坦，阻带有波纹
  ✓ 较低阶数达到高衰减
  → 适合: 陷波滤波器、希望通带平坦

实时处理注意事项:
1. 优先使用二阶节 (SOS) 结构，避免高阶滤波器数值不稳定
2. 系数量化时注意精度问题
3. 考虑使用定点数实现时的动态范围
4. 注意滤波器的群延迟
""")
