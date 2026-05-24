from fir_parks_mcclellan import ParksMcClellanDesigner, FIRFilter, print_filter_info
from iir_filter_fixed import IIRFilterDesigner
import numpy as np


def compare_filters():
    print("\n" + "#"*70)
    print("# FIR (Parks-McClellan) vs IIR 滤波器对比")
    print("#"*70)
    
    fs = 1000.0
    
    design_specs = {
        'filter_type': 'lowpass',
        'passband_freq': 100.0,
        'stopband_freq': 150.0,
        'passband_ripple': 1.0,
        'stopband_attenuation': 40.0
    }
    
    print(f"\n设计规格:")
    print(f"  采样频率: {fs} Hz")
    print(f"  通带截止: {design_specs['passband_freq']} Hz")
    print(f"  阻带截止: {design_specs['stopband_freq']} Hz")
    print(f"  通带波纹: {design_specs['passband_ripple']} dB")
    print(f"  阻带衰减: {design_specs['stopband_attenuation']} dB")
    
    print(f"\n" + "="*70)
    print(f"1. FIR 等波纹滤波器 (Parks-McClellan)")
    print(f"="*70)
    
    fir_designer = ParksMcClellanDesigner(fs, verbose=False)
    fir_filter = fir_designer.design_lowpass(
        passband_freq=design_specs['passband_freq'],
        stopband_freq=design_specs['stopband_freq'],
        passband_ripple_db=design_specs['passband_ripple'],
        stopband_atten_db=design_specs['stopband_attenuation']
    )
    
    w_fir, mag_fir = fir_filter.get_frequency_response(num_points=4096)
    fir_ripple, fir_atten = fir_filter.measure_ripple()
    _, fir_gd = fir_filter.get_group_delay(num_points=100)
    
    print(f"\n性能指标:")
    print(f"  滤波器阶数: {fir_filter.order}")
    print(f"  抽头数量: {len(fir_filter.taps)}")
    print(f"  实测通带波纹: {fir_ripple:.3f} dB")
    print(f"  实测阻带衰减: {fir_atten:.1f} dB")
    print(f"  群延迟 (常数): {np.mean(fir_gd):.2f} 样本")
    print(f"  线性相位: Yes (对称抽头)")
    print(f"  稳定性: 固有稳定 (FIR)")
    
    print(f"\n" + "="*70)
    print(f"2. IIR 巴特沃斯滤波器")
    print(f"="*70)
    
    iir_designer = IIRFilterDesigner(fs, verbose=False)
    iir_filter = iir_designer.butterworth(
        design_specs['filter_type'],
        design_specs['passband_freq'],
        design_specs['stopband_freq'],
        design_specs['passband_ripple'],
        design_specs['stopband_attenuation'],
        prewarp=True
    )
    
    w_iir, mag_iir = iir_filter.get_frequency_response(num_points=4096)
    _, iir_gd = iir_filter.get_group_delay(num_points=100)
    
    iir_ripple = np.max(np.abs(mag_iir[w_iir <= design_specs['passband_freq']]))
    iir_atten = np.min(mag_iir[w_iir >= design_specs['stopband_freq']])
    
    print(f"\n性能指标:")
    print(f"  滤波器阶数: {iir_filter.order}")
    print(f"  系数数量: {len(iir_filter.b) + len(iir_filter.a) - 1}")
    print(f"  实测通带波纹: {iir_ripple:.3f} dB")
    print(f"  实测阻带衰减: {abs(iir_atten):.1f} dB")
    print(f"  群延迟 (通带): {np.mean(iir_gd[w_iir <= design_specs['passband_freq']]):.2f} 样本")
    print(f"  线性相位: No (非线性相位)")
    print(f"  稳定性: 需要检查极点")
    
    print(f"\n" + "="*70)
    print(f"3. IIR 切比雪夫I型滤波器")
    print(f"="*70)
    
    cheb1_filter = iir_designer.chebyshev1(
        design_specs['filter_type'],
        design_specs['passband_freq'],
        design_specs['stopband_freq'],
        design_specs['passband_ripple'],
        design_specs['stopband_attenuation'],
        prewarp=True
    )
    
    w_cheb1, mag_cheb1 = cheb1_filter.get_frequency_response(num_points=4096)
    _, cheb1_gd = cheb1_filter.get_group_delay(num_points=100)
    
    cheb1_ripple = np.max(np.abs(mag_cheb1[w_cheb1 <= design_specs['passband_freq']]))
    cheb1_atten = np.min(mag_cheb1[w_cheb1 >= design_specs['stopband_freq']])
    
    print(f"\n性能指标:")
    print(f"  滤波器阶数: {cheb1_filter.order}")
    print(f"  系数数量: {len(cheb1_filter.b) + len(cheb1_filter.a) - 1}")
    print(f"  实测通带波纹: {cheb1_ripple:.3f} dB")
    print(f"  实测阻带衰减: {abs(cheb1_atten):.1f} dB")
    print(f"  群延迟 (通带): {np.mean(cheb1_gd[w_cheb1 <= design_specs['passband_freq']]):.2f} 样本")
    print(f"  线性相位: No (非线性相位)")
    print(f"  稳定性: 需要检查极点")
    
    print(f"\n" + "="*70)
    print(f"对比总结")
    print(f"="*70)
    
    print(f"\n{'特性':<25} {'FIR (Parks-McClellan)':<25} {'IIR Butterworth':<20} {'IIR Chebyshev':<20}")
    print("-" * 90)
    print(f"{'阶数':<25} {fir_filter.order:<25} {iir_filter.order:<20} {cheb1_filter.order:<20}")
    print(f"{'系数数量':<25} {len(fir_filter.taps):<25} {len(iir_filter.b)+len(iir_filter.a)-1:<20} {len(cheb1_filter.b)+len(cheb1_filter.a)-1:<20}")
    print(f"{'通带波纹 (dB)':<25} {fir_ripple:.3f:<25} {iir_ripple:.3f:<20} {cheb1_ripple:.3f:<20}")
    print(f"{'阻带衰减 (dB)':<25} {fir_atten:.1f:<25} {abs(iir_atten):.1f:<20} {abs(cheb1_atten):.1f:<20}")
    print(f"{'线性相位':<25} {'是':<25} {'否':<20} {'否':<20}")
    print(f"{'固有稳定':<25} {'是':<25} {'否':<20} {'否':<20}")
    print(f"{'等波纹':<25} {'是':<25} {'否':<20} {'通带':<20}")
    
    print(f"\n" + "="*70)
    print(f"选型建议")
    print(f"="*70)
    print(f"""
选择 FIR (Parks-McClellan) 当:
  ✓ 需要精确线性相位
  ✓ 需要保证稳定性
  ✓ 需要等波纹特性
  ✗ 计算资源充足

选择 IIR 当:
  ✓ 需要低阶数/少系数
  ✓ 计算资源受限
  ✗ 可以接受非线性相位
""")


def demonstrate_multiband():
    print("\n" + "#"*70)
    print("# 多频带滤波器演示 (Parks-McClellan)")
    print("#"*70)
    
    fs = 1000.0
    designer = ParksMcClellanDesigner(fs, verbose=True)
    
    print(f"\n设计一个三频带滤波器:")
    print(f"  频带1: 0-50 Hz    → 0 幅度 (阻带)")
    print(f"  频带2: 70-130 Hz  → 1 幅度 (通带)")
    print(f"  频带3: 150-200 Hz → 0 幅度 (阻带)")
    print(f"  频带4: 250-500 Hz → 0.5 幅度 (半增益)")
    
    mb_filter = designer.design_multiband(
        bands=[0, 50, 70, 130, 150, 200, 250, 500],
        desired=[0, 0, 1, 1, 0, 0, 0.5, 0.5],
        weights=[2.0, 1.0, 2.0, 1.0],
        order=80,
        filter_type_name="三频带滤波器"
    )
    
    print_filter_info(mb_filter)
    
    w, mag = mb_filter.get_frequency_response(num_points=4096)
    
    print(f"\n各频带实测响应:")
    for band_name, f1, f2, target in [
        ("阻带1", 0, 50, 0),
        ("通带", 70, 130, 1),
        ("阻带2", 150, 200, 0),
        ("半增益带", 250, 400, 0.5),
    ]:
        mask = (w >= f1) & (w <= f2)
        if np.any(mask):
            avg_mag = np.mean(mag[mask])
            target_db = 20 * np.log10(target + 1e-10) if target > 0 else -np.inf
            print(f"  {band_name}: {f1}-{f2} Hz, 平均衰减: {avg_mag:.1f} dB (目标: {target_db:.1f} dB)")


def demonstrate_arbitrary_response():
    print("\n" + "#"*70)
    print("# 任意幅度响应滤波器演示")
    print("#"*70)
    
    fs = 2000.0
    designer = ParksMcClellanDesigner(fs, verbose=True)
    
    def custom_response(f):
        if f < 100:
            return 1.0
        elif f < 200:
            return 1.0 - (f - 100) / 100 * 0.5
        elif f < 300:
            return 0.5
        elif f < 400:
            return 0.5 - (f - 300) / 100 * 0.5
        else:
            return 0.0
    
    print(f"\n自定义响应函数:")
    print(f"  0-100 Hz: 1.0")
    print(f"  100-200 Hz: 1.0 → 0.5 (斜坡)")
    print(f"  200-300 Hz: 0.5")
    print(f"  300-400 Hz: 0.5 → 0.0 (斜坡)")
    print(f"  >400 Hz: 0.0")
    
    arb_filter = designer.design_multiband(
        bands=[0, 100, 120, 180, 200, 300, 320, 380, 400, fs/2],
        desired=[1.0, 1.0, 0.9, 0.6, 0.5, 0.5, 0.4, 0.1, 0.0, 0.0],
        weights=[1.0, 2.0, 1.0, 2.0, 1.0],
        order=150,
        filter_type_name="自定义响应"
    )
    
    print_filter_info(arb_filter)


def realtime_comparison():
    print("\n" + "#"*70)
    print("# 实时处理性能对比")
    print("#"*70)
    
    fs = 1000.0
    
    print(f"\n生成测试信号...")
    t = np.linspace(0, 1.0, int(fs), endpoint=False)
    signal_50hz = np.sin(2 * np.pi * 50 * t)
    signal_120hz = 0.5 * np.sin(2 * np.pi * 120 * t)
    signal_200hz = 0.3 * np.sin(2 * np.pi * 200 * t)
    test_signal = signal_50hz + signal_120hz + signal_200hz
    
    print(f"测试信号包含: 50Hz (保留) + 120Hz (衰减) + 200Hz (衰减)")
    print(f"设计目标: 低通 100Hz")
    
    fir_designer = ParksMcClellanDesigner(fs, verbose=False)
    fir_filter = fir_designer.design_lowpass(100, 150, 1.0, 40.0)
    
    iir_designer = IIRFilterDesigner(fs, verbose=False)
    iir_filter = iir_designer.butterworth('lowpass', 100, 150, 1.0, 40.0, prewarp=True)
    
    import time
    
    fir_filter.reset()
    start = time.time()
    fir_output = fir_filter.process_block(test_signal)
    fir_time = time.time() - start
    
    iir_filter.reset()
    start = time.time()
    iir_output = iir_filter.process_block(test_signal)
    iir_time = time.time() - start
    
    input_energy = np.sum(test_signal**2)
    fir_energy = np.sum(fir_output**2)
    iir_energy = np.sum(iir_output**2)
    
    print(f"\n处理结果:")
    print(f"{'指标':<25} {'FIR':<20} {'IIR':<20}")
    print("-" * 65)
    print(f"{'阶数':<25} {fir_filter.order:<20} {iir_filter.order:<20}")
    print(f"{'处理时间 (ms)':<25} {fir_time*1000:.2f:<20} {iir_time*1000:.2f:<20}")
    print(f"{'输出能量':<25} {fir_energy:.4f:<20} {iir_energy:.4f:<20}")
    print(f"{'能量保留率 (%)':<25} {fir_energy/input_energy*100:.1f:<20} {iir_energy/input_energy*100:.1f:<20}")
    
    print(f"\n✓ 实时处理对比完成!")


def main():
    compare_filters()
    demonstrate_multiband()
    demonstrate_arbitrary_response()
    realtime_comparison()
    
    print("\n" + "#"*70)
    print("# 所有演示完成!")
    print("#"*70)


if __name__ == "__main__":
    main()
