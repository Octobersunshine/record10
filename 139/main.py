import numpy as np
from go_model import (
    GoModel,
    generate_native_structure,
    generate_contact_map,
    calculate_mfpt,
    plot_results,
    plot_mfpt_results,
    calibrate_friction_coefficient,
    create_calibrated_simulation,
    compare_gamma_models
)
from units import UnitConverter, FoldingRateCalibrator, folding_rate_from_mfpt
from amber_forcefield import AmberForceField, GoAmberHybrid
from remd import REMD, FoldingPathAnalyzer, compare_sampling_efficiency


def single_trajectory_demo():
    print("=" * 60)
    print("单轨迹模拟演示")
    print("=" * 60)
    
    num_beads = 30
    print(f"\n参数设置:")
    print(f"  珠子数: {num_beads}")
    print(f"  结构类型: alpha螺旋")
    
    native_positions = generate_native_structure(num_beads, structure_type='helix')
    contacts, contact_distances = generate_contact_map(
        native_positions, 
        cutoff=6.0, 
        min_sequence_separation=3
    )
    
    print(f"  天然态接触数: {len(contacts)}")
    
    simulation = GoModel(
        num_beads=num_beads,
        native_contacts=contacts,
        native_distances=contact_distances,
        temperature=0.8,
        gamma=1.0,
        dt=0.005
    )
    
    n_steps = 200000
    record_interval = 100
    print(f"\n开始模拟 {n_steps} 步...")
    
    trajectory, Q_values = simulation.simulate(
        n_steps=n_steps,
        record_interval=record_interval
    )
    
    final_Q = Q_values[-1]
    max_Q = np.max(Q_values)
    print(f"\n模拟结果:")
    print(f"  最终Q值: {final_Q:.3f}")
    print(f"  最高Q值: {max_Q:.3f}")
    
    plot_results(Q_values, trajectory, contacts)
    print("\n结果图已保存为: go_model_results.png")
    
    return simulation, native_positions, contacts


def mfpt_calculation_demo():
    print("\n" + "=" * 60)
    print("折叠时间 (MFPT) 计算")
    print("=" * 60)
    
    num_beads = 20
    print(f"\n参数设置:")
    print(f"  珠子数: {num_beads}")
    print(f"  结构类型: beta发夹")
    
    native_positions = generate_native_structure(num_beads, structure_type='hairpin')
    contacts, contact_distances = generate_contact_map(
        native_positions, 
        cutoff=5.0, 
        min_sequence_separation=3
    )
    
    print(f"  天然态接触数: {len(contacts)}")
    
    temperatures = [0.7, 0.8, 0.9, 1.0]
    num_trajectories = 20
    max_steps = 200000
    Q_threshold = 0.75
    
    print(f"  轨迹数: {num_trajectories}")
    print(f"  Q阈值: {Q_threshold}")
    print(f"  最大步数: {max_steps}")
    
    mfpt_results = []
    
    for T in temperatures:
        print(f"\n温度 T = {T}:")
        
        simulation = GoModel(
            num_beads=num_beads,
            native_contacts=contacts,
            native_distances=contact_distances,
            temperature=T,
            gamma=1.0,
            dt=0.005
        )
        
        mfpt, mfpt_error, folding_times, Q_trajectories = calculate_mfpt(
            num_trajectories=num_trajectories,
            simulation=simulation,
            native_positions=native_positions,
            Q_threshold=Q_threshold,
            max_steps=max_steps
        )
        
        mfpt_results.append((T, mfpt, mfpt_error))
        
        print(f"  MFPT = {mfpt:.0f} ± {mfpt_error:.0f} 步")
        print(f"  折叠概率: {sum(1 for t in folding_times if t < max_steps)}/{num_trajectories}")
        
        if T == temperatures[0]:
            plot_mfpt_results(folding_times, Q_trajectories, mfpt)
            print(f"  MFPT结果图已保存为: mfpt_results.png")
    
    print("\n" + "=" * 60)
    print("温度依赖的MFPT总结:")
    print("=" * 60)
    print(f"{'温度':>10} {'MFPT':>15} {'误差':>15}")
    print("-" * 42)
    for T, mfpt, err in mfpt_results:
        print(f"{T:>10.2f} {mfpt:>15.0f} {err:>15.0f}")
    
    return mfpt_results


def custom_contact_map_example():
    print("\n" + "=" * 60)
    print("使用自定义接触图")
    print("=" * 60)
    
    num_beads = 25
    
    contacts = np.array([
        (0, 24), (1, 23), (2, 22), (3, 21), (4, 20),
        (0, 20), (4, 24),
        (8, 16), (9, 15), (10, 14), (11, 13)
    ])
    
    contact_distances = np.array([
        4.0, 4.0, 4.0, 4.0, 4.0,
        5.0, 5.0,
        3.5, 3.5, 3.5, 3.5
    ])
    
    print(f"自定义接触数: {len(contacts)}")
    
    native_positions = generate_native_structure(num_beads, structure_type='helix')
    
    simulation = GoModel(
        num_beads=num_beads,
        native_contacts=contacts,
        native_distances=contact_distances,
        temperature=0.85,
        gamma=1.0,
        dt=0.005
    )
    
    print("\n开始模拟...")
    trajectory, Q_values = simulation.simulate(n_steps=150000, record_interval=100)
    
    print(f"最终Q值: {Q_values[-1]:.3f}")
    print(f"最高Q值: {np.max(Q_values):.3f}")
    
    return simulation, native_positions, contacts


def gamma_calibration_demo():
    print("\n" + "=" * 60)
    print("摩擦系数校准演示")
    print("=" * 60)
    
    num_beads = 30
    print(f"\n参数设置:")
    print(f"  珠子数: {num_beads}")
    print(f"  结构类型: alpha螺旋")
    
    native_positions = generate_native_structure(num_beads, structure_type='helix')
    contacts, contact_distances = generate_contact_map(
        native_positions, 
        cutoff=6.0, 
        min_sequence_separation=3
    )
    
    print(f"  天然态接触数: {len(contacts)}")
    
    calibration = calibrate_friction_coefficient(
        num_beads=num_beads,
        contacts=contacts,
        contact_distances=contact_distances,
        num_calibration_trajs=5,
        max_calibration_steps=80000,
        temperature=0.9
    )
    
    print("\n" + "=" * 60)
    print("使用校准后的摩擦系数进行模拟")
    print("=" * 60)
    
    sim_calibrated = GoModel(
        num_beads=num_beads,
        native_contacts=contacts,
        native_distances=contact_distances,
        temperature=0.9,
        gamma=calibration['corrected_gamma'],
        auto_gamma=False
    )
    
    trajectory, Q_values = sim_calibrated.simulate(
        n_steps=100000,
        record_interval=100
    )
    
    print(f"\n校准后模拟结果:")
    print(f"  最终Q值: {Q_values[-1]:.3f}")
    print(f"  最高Q值: {np.max(Q_values):.3f}")
    
    kf = folding_rate_from_mfpt(
        50000, 
        dt=sim_calibrated.dt,
        time_ps_per_tau=calibration['time_scale_ps']
    )
    print(f"  估计折叠速率: kf ≈ {kf:.2e} s^-1")
    
    return calibration


def gamma_comparison_demo():
    compare_gamma_models(num_beads=30, temperature=0.9)
    
    print("\n" + "=" * 60)
    print("摩擦系数对折叠的影响")
    print("=" * 60)
    
    num_beads = 20
    native_positions = generate_native_structure(num_beads, structure_type='hairpin')
    contacts, contact_distances = generate_contact_map(
        native_positions, 
        cutoff=5.0, 
        min_sequence_separation=3
    )
    
    gamma_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    
    print(f"\n{'γ':<10} {'估计kf (s^-1)':<20} {'相对速度':<15}")
    print("-" * 45)
    
    for gamma in gamma_values:
        sim = GoModel(
            num_beads=num_beads,
            native_contacts=contacts,
            native_distances=contact_distances,
            temperature=0.85,
            gamma=gamma,
            auto_gamma=False
        )
        
        kf_est = 1.0 / (gamma * 1e4)
        relative_speed = 1.0 / gamma
        
        print(f"{gamma:<10.2f} {kf_est:.2e}                 {relative_speed:<15.1f}")
    
    print("\n说明: 摩擦系数越小，扩散越快，折叠时间越短")
    print("=" * 60)


def hybrid_forcefield_demo():
    print("\n" + "=" * 60)
    print("Go-AMBER混合力场演示")
    print("=" * 60)
    
    num_beads = 25
    print(f"\n参数设置:")
    print(f"  珠子数: {num_beads}")
    print(f"  结构类型: alpha螺旋")
    
    native_positions = generate_native_structure(num_beads, structure_type='helix')
    contacts, contact_distances = generate_contact_map(
        native_positions, 
        cutoff=6.0, 
        min_sequence_separation=3
    )
    
    print(f"  天然态接触数: {len(contacts)}")
    print(f"  AMBER权重: 0.3, Go权重: 0.7")
    
    sim = GoAmberHybrid(
        num_beads=num_beads,
        native_contacts=contacts,
        native_distances=contact_distances,
        native_positions=native_positions,
        temperature=0.85,
        weight_amber=0.3,
        weight_go=0.7
    )
    
    print(f"\n开始模拟 (50000步)...")
    trajectory, energies, Q_values = sim.simulate(
        n_steps=50000,
        record_interval=100
    )
    
    print(f"\n模拟结果:")
    print(f"  最终Q值: {Q_values[-1]:.3f}")
    print(f"  最高Q值: {np.max(Q_values):.3f}")
    print(f"  最终能量: {energies[-1]:.2f}")
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(Q_values)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Q')
    axes[0].set_title('Native Contact Fraction')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(energies)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Energy vs Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_ff_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n结果图已保存为: hybrid_ff_results.png")
    
    return sim


def remd_demo():
    print("\n" + "=" * 60)
    print("副本交换分子动力学 (REMD) 演示")
    print("=" * 60)
    
    num_beads = 20
    print(f"\n参数设置:")
    print(f"  珠子数: {num_beads}")
    print(f"  结构类型: beta发夹")
    
    native_positions = generate_native_structure(num_beads, structure_type='hairpin')
    contacts, contact_distances = generate_contact_map(
        native_positions, 
        cutoff=5.0, 
        min_sequence_separation=3
    )
    
    print(f"  天然态接触数: {len(contacts)}")
    
    num_replicas = 6
    T_min = 0.6
    T_max = 1.4
    
    remd = REMD(
        num_replicas=num_replicas,
        T_min=T_min,
        T_max=T_max,
        num_beads=num_beads,
        native_contacts=contacts,
        native_distances=contact_distances,
        native_positions=native_positions,
        use_hybrid_ff=False,
        scheduler='geometric'
    )
    
    results = remd.run_remd(
        n_cycles=50,
        n_steps_per_cycle=1000,
        exchange_interval=1,
        record_interval=2
    )
    
    print(f"\nREMD结果总结:")
    print(f"{'温度':<10} {'平均Q':<10} {'Std Q':<10}")
    print("-" * 30)
    for i, T in enumerate(results['temperatures']):
        print(f"{T:<10.2f} {results['avg_Q'][i]:<10.3f} {results['std_Q'][i]:<10.3f}")
    
    exchange_rates = results['exchange_rates']
    if isinstance(exchange_rates, np.ndarray) and exchange_rates.ndim == 2:
        avg_exchange = np.mean(exchange_rates.diagonal(offset=1))
        print(f"\n平均交换接受率: {avg_exchange:.2%}")
    
    remd.plot_results('remd_results.png')
    print("\nREMD结果图已保存为: remd_results.png")
    
    return remd, results


def folding_path_analysis_demo():
    print("\n" + "=" * 60)
    print("折叠路径分析演示")
    print("=" * 60)
    
    num_beads = 25
    print(f"\n参数设置:")
    print(f"  珠子数: {num_beads}")
    print(f"  结构类型: alpha螺旋")
    
    native_positions = generate_native_structure(num_beads, structure_type='helix')
    contacts, contact_distances = generate_contact_map(
        native_positions, 
        cutoff=6.0, 
        min_sequence_separation=3
    )
    
    sim = GoModel(
        num_beads=num_beads,
        native_contacts=contacts,
        native_distances=contact_distances,
        temperature=0.9,
        auto_gamma=True,
        gamma_model='stokes_einstein'
    )
    
    print(f"\n生成长轨迹用于路径分析...")
    trajectory, Q_values = sim.simulate(
        n_steps=100000,
        record_interval=100
    )
    
    print(f"轨迹帧数: {len(trajectory)}")
    
    analyzer = FoldingPathAnalyzer(trajectory, native_positions)
    order_params = analyzer.compute_order_parameters()
    
    print(f"\n序参数统计:")
    print(f"  Q范围: {np.min(order_params['Q']):.3f} - {np.max(order_params['Q']):.3f}")
    print(f"  RMSD范围: {np.min(order_params['RMSD']):.2f} - {np.max(order_params['RMSD']):.2f}")
    
    transitions = analyzer.find_folding_transitions(Q_threshold=0.7)
    print(f"  检测到的折叠/去折叠事件: {len(transitions)}")
    
    contact_order = analyzer.compute_contact_order(contacts)
    print(f"  早期形成的接触数: {len(contact_order['early_contacts'])}")
    print(f"  晚期形成的接触数: {len(contact_order['late_contacts'])}")
    
    clusters = analyzer.cluster_states(n_clusters=4)
    print(f"  聚类数: 4")
    print(f"  各态布居: {clusters['populations']}")
    
    analyzer.plot_folding_path('folding_path_analysis.png')
    print("\n路径分析图已保存为: folding_path_analysis.png")
    
    return analyzer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='蛋白质折叠Go模型模拟')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'mfpt', 'custom', 'calibrate', 'compare', 
                                'hybrid', 'remd', 'path'],
                       help='运行模式: single=单轨迹, mfpt=MFPT计算, custom=自定义接触图, '
                            'calibrate=摩擦系数校准, compare=摩擦系数对比, '
                            'hybrid=混合力场, remd=副本交换MD, path=折叠路径分析')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        single_trajectory_demo()
    elif args.mode == 'mfpt':
        mfpt_calculation_demo()
    elif args.mode == 'custom':
        custom_contact_map_example()
    elif args.mode == 'calibrate':
        gamma_calibration_demo()
    elif args.mode == 'compare':
        gamma_comparison_demo()
    elif args.mode == 'hybrid':
        hybrid_forcefield_demo()
    elif args.mode == 'remd':
        remd_demo()
    elif args.mode == 'path':
        folding_path_analysis_demo()
    
    print("\n" + "=" * 60)
    print("模拟完成!")
    print("=" * 60)
