import numpy as np
import matplotlib.pyplot as plt
from multilayer_earth_model import (
    LayeredEarthModel, Layer, AdvancedGICCalculator,
    Substation, TransmissionLine, generate_storm_profile
)
from mt_inversion import MTInversion, MTData, create_sample_mt_data, save_mt_data, load_mt_data


def run_complete_workflow():
    print("=" * 80)
    print("完整GIC计算工作流程: MT数据反演 -> 地电阻率模型 -> GIC计算")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("步骤1: 准备MT测深数据")
    print("=" * 80)
    
    mt_data = create_sample_mt_data()
    save_mt_data(mt_data, 'mt_measurement_data.txt')
    print(f"已生成并保存MT数据到: mt_measurement_data.txt")
    print(f"频率范围: {mt_data.frequencies.min():.2e} - {mt_data.frequencies.max():.2e} Hz")
    print(f"数据点数: {len(mt_data.frequencies)}")
    
    print("\n" + "=" * 80)
    print("步骤2: MT数据反演 - 获取地电阻率模型")
    print("=" * 80)
    
    inversion = MTInversion(mt_data, n_layers=4)
    print("正在反演MT数据...")
    inverted_layers = inversion.invert()
    
    print("\n反演得到的地电阻率模型:")
    for i, layer in enumerate(inverted_layers):
        if layer.thickness == float('inf'):
            print(f"  层{i+1}: 半空间, 电阻率={layer.resistivity:.1f} Ω·m")
        else:
            print(f"  层{i+1}: 厚度={layer.thickness:.0f}m, 电阻率={layer.resistivity:.1f} Ω·m")
    
    print("\n" + "=" * 80)
    print("步骤3: 构建电网模型")
    print("=" * 80)
    
    earth_model = LayeredEarthModel(inverted_layers)
    calc = AdvancedGICCalculator(earth_model)
    
    substations_data = [
        (1, "变电站A", 0, 0, 0.5),
        (2, "变电站B", 100, 0, 0.3),
        (3, "变电站C", 50, 80, 0.4),
        (4, "变电站D", 150, 60, 0.6),
        (5, "变电站E", 200, 100, 0.45)
    ]
    
    for sid, name, x, y, r in substations_data:
        calc.add_substation(Substation(sid, name, x, y, r))
        print(f"  添加 {name}: 位置({x}, {y})km, 接地电阻{r}Ω")
    
    lines_data = [
        (1, 1, 2, 100, 0.05),
        (2, 1, 3, 95, 0.048),
        (3, 2, 3, 90, 0.045),
        (4, 2, 4, 70, 0.035),
        (5, 3, 4, 100, 0.05),
        (6, 4, 5, 80, 0.04),
        (7, 3, 5, 130, 0.065)
    ]
    
    for lid, f, t, length, res in lines_data:
        calc.add_line(TransmissionLine(lid, f, t, length, res))
    
    print(f"\n  变电站数量: {len(calc.substations)}")
    print(f"  输电线路数量: {len(calc.lines)}")
    
    print("\n" + "=" * 80)
    print("步骤4: 生成磁暴事件 (地磁场变化数据)")
    print("=" * 80)
    
    t, dB_dt = generate_storm_profile(duration_hours=48.0, dt_minutes=5.0)
    print(f"  磁暴持续时间: 48小时")
    print(f"  采样间隔: 5分钟")
    print(f"  数据点数: {len(t)}")
    print(f"  最大dB/dt: {np.max(np.abs(dB_dt)):.1f} nT/min")
    
    print("\n" + "=" * 80)
    print("步骤5: 计算大地电场 (使用分层介质解析解)")
    print("=" * 80)
    
    E_time = calc.calculate_time_domain_electric_field(t, dB_dt, use_fem=False)
    print(f"  最大地表电场: {np.max(np.abs(E_time)) * 1000:.2f} mV/km")
    print(f"  平均地表电场: {np.mean(np.abs(E_time)) * 1000:.2f} mV/km")
    
    print("\n" + "=" * 80)
    print("步骤6: 计算各变电站GIC")
    print("=" * 80)
    
    gic_results, _ = calc.calculate_gic(t, dB_dt, e_field_angle=45.0, use_fem=False)
    
    print("\n各变电站中性点GIC统计:")
    for sid, gic in gic_results.items():
        sub = calc.substations[sid]
        print(f"\n  {sub.name}:")
        print(f"    最大GIC: {np.max(np.abs(gic)):.2f} A")
        print(f"    最小GIC: {np.min(gic):.2f} A")
        print(f"    平均GIC: {np.mean(np.abs(gic)):.2f} A")
        print(f"    均方根GIC: {np.sqrt(np.mean(gic**2)):.2f} A")
    
    print("\n" + "=" * 80)
    print("步骤7: 结果可视化")
    print("=" * 80)
    
    fig = plt.figure(figsize=(16, 14))
    
    plt.subplot(3, 2, 1)
    rho_pred, phase_pred = inversion.compute_predicted_response(inverted_layers)
    plt.loglog(mt_data.frequencies, mt_data.apparent_resistivity, 
               'ko', label='观测数据', markersize=4, alpha=0.6)
    plt.loglog(mt_data.frequencies, rho_pred, 'r-', label='反演拟合', linewidth=2)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('视电阻率 (Ω·m)')
    plt.title('MT视电阻率曲线及拟合')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    plt.subplot(3, 2, 2)
    plt.semilogx(mt_data.frequencies, mt_data.phase, 
                 'ko', label='观测数据', markersize=4, alpha=0.6)
    plt.semilogx(mt_data.frequencies, phase_pred, 'r-', label='反演拟合', linewidth=2)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('相位 (度)')
    plt.title('MT相位曲线及拟合')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    ax3 = plt.subplot(3, 2, 3)
    depths = []
    resistivities = []
    current_depth = 0
    for i, layer in enumerate(inverted_layers):
        if layer.thickness != float('inf'):
            depths.extend([current_depth, current_depth + layer.thickness])
            resistivities.extend([layer.resistivity, layer.resistivity])
            current_depth += layer.thickness
        else:
            depths.extend([current_depth, current_depth + 100000])
            resistivities.extend([layer.resistivity, layer.resistivity])
    ax3.semilogx(resistivities, np.array(depths) / 1000, 'b-', linewidth=2.5)
    ax3.set_xlabel('电阻率 (Ω·m)')
    ax3.set_ylabel('深度 (km)')
    ax3.set_title('反演得到的地电阻率剖面')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.invert_yaxis()
    
    plt.subplot(3, 2, 4)
    plt.plot(t / 3600, dB_dt, 'b-', linewidth=1.5)
    plt.xlabel('时间 (小时)')
    plt.ylabel('dB/dt (nT/min)')
    plt.title('磁暴期间地磁场变化率')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 5)
    plt.plot(t / 3600, E_time * 1000, 'g-', linewidth=1.5)
    plt.xlabel('时间 (小时)')
    plt.ylabel('地表电场 (mV/km)')
    plt.title('感应地表电场强度')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    for sid, gic in gic_results.items():
        sub = calc.substations[sid]
        plt.plot(t / 3600, gic, label=f'{sub.name}', linewidth=1.5)
    plt.xlabel('时间 (小时)')
    plt.ylabel('GIC (A)')
    plt.title('各变电站中性点GIC')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_gic_workflow.png', dpi=150, bbox_inches='tight')
    print("  完整工作流结果图已保存为: complete_gic_workflow.png")
    
    fig2 = plt.figure(figsize=(10, 8))
    for sid, gic in gic_results.items():
        sub = calc.substations[sid]
        plt.plot(t / 3600, np.abs(gic), label=f'{sub.name}', linewidth=2)
    plt.xlabel('时间 (小时)')
    plt.ylabel('|GIC| (A)')
    plt.title('各变电站中性点GIC幅值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('gic_amplitude_log.png', dpi=150, bbox_inches='tight')
    print("  GIC幅值图已保存为: gic_amplitude_log.png")
    
    print("\n" + "=" * 80)
    print("GIC风险评估")
    print("=" * 80)
    
    risk_thresholds = [5, 20, 50, 100]
    risk_levels = ["低", "中", "高", "极高"]
    
    for sid, gic in gic_results.items():
        sub = calc.substations[sid]
        max_gic = np.max(np.abs(gic))
        risk_idx = np.digitize(max_gic, risk_thresholds) - 1
        risk_idx = min(risk_idx, len(risk_levels) - 1)
        print(f"\n  {sub.name}:")
        print(f"    最大GIC: {max_gic:.2f} A")
        print(f"    风险等级: {risk_levels[risk_idx]}")
    
    print("\n" + "=" * 80)
    print("工作流完成!")
    print("=" * 80)


def compare_earth_models():
    print("\n" + "=" * 80)
    print("对比分析: 均匀半空间 vs 多层地电阻率模型")
    print("=" * 80)
    
    uniform_model = LayeredEarthModel([
        Layer(thickness=float('inf'), resistivity=100)
    ])
    
    layered_model = LayeredEarthModel([
        Layer(thickness=500, resistivity=100),
        Layer(thickness=5000, resistivity=500),
        Layer(thickness=20000, resistivity=10),
        Layer(thickness=float('inf'), resistivity=1000)
    ])
    
    frequencies = np.logspace(-4, 2, 100)
    rho_uniform, phase_uniform = uniform_model.compute_apparent_resistivity(frequencies)
    rho_layered, phase_layered = layered_model.compute_apparent_resistivity(frequencies)
    
    calc_uniform = AdvancedGICCalculator(uniform_model)
    calc_layered = AdvancedGICCalculator(layered_model)
    
    for calc in [calc_uniform, calc_layered]:
        calc.add_substation(Substation(1, "变电站A", 0, 0, 0.5))
        calc.add_substation(Substation(2, "变电站B", 100, 0, 0.3))
        calc.add_line(TransmissionLine(1, 1, 2, 100, 0.05))
    
    t, dB_dt = generate_storm_profile(duration_hours=24.0, dt_minutes=5.0)
    
    gic_uniform, E_uniform = calc_uniform.calculate_gic(t, dB_dt, e_field_angle=45.0)
    gic_layered, E_layered = calc_layered.calculate_gic(t, dB_dt, e_field_angle=45.0)
    
    print("\n地表电场对比:")
    print(f"  均匀模型最大电场: {np.max(np.abs(E_uniform)) * 1000:.2f} mV/km")
    print(f"  多层模型最大电场: {np.max(np.abs(E_layered)) * 1000:.2f} mV/km")
    print(f"  相对差异: {(np.max(np.abs(E_layered)) - np.max(np.abs(E_uniform))) / np.max(np.abs(E_uniform)) * 100:.1f}%")
    
    print("\nGIC对比 (变电站A):")
    max_uniform = np.max(np.abs(gic_uniform[1]))
    max_layered = np.max(np.abs(gic_layered[1]))
    print(f"  均匀模型最大GIC: {max_uniform:.2f} A")
    print(f"  多层模型最大GIC: {max_layered:.2f} A")
    print(f"  相对差异: {(max_layered - max_uniform) / max_uniform * 100:.1f}%")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.loglog(frequencies, rho_uniform, 'b--', label='均匀半空间', linewidth=2)
    ax.loglog(frequencies, rho_layered, 'r-', label='四层模型', linewidth=2)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('视电阻率 (Ω·m)')
    ax.set_title('视电阻率对比')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    ax = axes[0, 1]
    ax.semilogx(frequencies, phase_uniform, 'b--', label='均匀半空间', linewidth=2)
    ax.semilogx(frequencies, phase_layered, 'r-', label='四层模型', linewidth=2)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('相位 (度)')
    ax.set_title('相位对比')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    ax = axes[1, 0]
    ax.plot(t / 3600, E_uniform * 1000, 'b--', label='均匀半空间', linewidth=1.5)
    ax.plot(t / 3600, E_layered * 1000, 'r-', label='四层模型', linewidth=1.5)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('地表电场 (mV/km)')
    ax.set_title('地表电场对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(t / 3600, gic_uniform[1], 'b--', label='均匀半空间', linewidth=1.5)
    ax.plot(t / 3600, gic_layered[1], 'r-', label='四层模型', linewidth=1.5)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('GIC (A)')
    ax.set_title('GIC对比 (变电站A)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  模型对比图已保存为: model_comparison.png")


if __name__ == "__main__":
    run_complete_workflow()
    compare_earth_models()
