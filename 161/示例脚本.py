import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wimp_recoil_simulation import (
    WIMPSimulation, 
    validate_smooth_cutoff,
    compare_spectra_smoothness
)

print("=" * 70)
print("示例 1: 验证速度分布平滑截断")
print("=" * 70)

validate_smooth_cutoff()

print("\n" + "=" * 70)
print("示例 2: 验证能谱平滑性（无伪尖峰）")
print("=" * 70)

compare_spectra_smoothness()

print("\n" + "=" * 70)
print("示例 3: 年度调制效应演示")
print("=" * 70)

sim_annual = WIMPSimulation(
    wimp_mass_gev=50,
    cross_section_cm2=1e-45,
    target_mass_amu=72.63
)

v_june = sim_annual.earth_velocity_day_of_year(152)
v_december = sim_annual.earth_velocity_day_of_year(355)

print(f"6月地球速度: {v_june/1000:.1f} km/s")
print(f"12月地球速度: {v_december/1000:.1f} km/s")
print(f"速度差: {(v_june - v_december)/1000:.1f} km/s")

days, rates = sim_annual.annual_modulation_curve(er_min=2, er_max=6)
rate_mean = rates.mean()
rate_amp = (rates.max() - rates.min()) / rate_mean * 100

print(f"\n2-6 keV 能区平均事例率: {rate_mean:.4f} 事例/kg/day")
print(f"年度调制幅度: {rate_amp:.2f}%")
print(f"峰值日期: 第 {days[rates.argmax()]:.0f} 天")

sim_annual.plot_annual_modulation(er_min=2, er_max=6)

print("\n" + "=" * 70)
print("示例 4: 方向性探测 - 核乳胶探测器")
print("=" * 70)

sim_emulsion = WIMPSimulation(
    wimp_mass_gev=50,
    cross_section_cm2=1e-45,
    detector_type='emulsion'
)

cos_theta, rate = sim_emulsion.angular_distribution(er_kev=20)
forward_rate = rate[cos_theta >= 0].mean()
backward_rate = rate[cos_theta < 0].mean()
fb_ratio = forward_rate / max(backward_rate, 1e-10)

print(f"20 keV 处平均前向事例率: {forward_rate:.2e} 事例/keV/kg/day")
print(f"20 keV 处平均后向事例率: {backward_rate:.2e} 事例/keV/kg/day")
print(f"前向后向比: {fb_ratio:.2f}")

sim_emulsion.plot_angular_distribution(er_kev=20)
sim_emulsion.plot_directional_events(n_events=2000, er_min=5, er_max=50)

print("\n" + "=" * 70)
print("示例 5: 不同探测器类型对比")
print("=" * 70)

sim_compare = WIMPSimulation(wimp_mass_gev=50, cross_section_cm2=1e-45)
sim_compare.compare_detector_types(er_kev=20)

print("\n探测器参数:")
for det_type, params in sim_compare.detector_params.items():
    print(f"  {det_type:10s}: {params['name']:15s}, "
          f"角分辨率 {np.degrees(params['angular_resolution']):.0f}°, "
          f"效率 {params['efficiency']*100:.0f}%")

print("\n" + "=" * 70)
print("示例 6: 完整参数输出")
print("=" * 70)

sim_full = WIMPSimulation(
    wimp_mass_gev=50,
    cross_section_cm2=1e-45,
    target_mass_amu=72.63,
    exposure_kg_day=1000,
    detector_type='gas'
)
sim_full.print_parameters()

print("\n" + "=" * 70)
print("所有示例运行完成! 请查看生成的图表窗口。")
print("=" * 70)

import matplotlib.pyplot as plt
plt.show()
