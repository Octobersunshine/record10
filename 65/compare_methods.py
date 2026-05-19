import numpy as np
import matplotlib.pyplot as plt
import time

from gene_regulation_ssa import GillespieSSA, Reaction
from tau_leaping import TauLeaping
from moment_closure import MomentClosure


def create_model():
    species_names = ['mRNA', 'Protein']
    
    k0 = 5.0    # 转录速率
    k1 = 0.3    # mRNA降解速率
    k2 = 2.0    # 翻译速率
    k3 = 0.1    # 蛋白质降解速率
    
    ssa_propensities = [
        lambda s, t: k0,
        lambda s, t: k1 * s[0],
        lambda s, t: k2 * s[0],
        lambda s, t: k3 * s[1],
    ]
    
    moment_propensities = [
        lambda m, t: k0 * np.ones_like(m[0]),
        lambda m, t: k1 * m[0],
        lambda m, t: k2 * m[0],
        lambda m, t: k3 * m[1],
    ]
    
    stoichiometries = [
        np.array([1, 0]),
        np.array([-1, 0]),
        np.array([0, 1]),
        np.array([0, -1]),
    ]
    
    return species_names, ssa_propensities, moment_propensities, stoichiometries


def run_ssa(species_names, propensities, stoichiometries,
            initial_state, t_span, num_runs=100):
    print(f"\n{'='*60}")
    print(f"Gillespie SSA: {num_runs}次模拟")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    all_times = []
    all_states = []
    
    for run in range(num_runs):
        ssa = GillespieSSA(species_names)
        for i, prop in enumerate(propensities):
            ssa.add_reaction(Reaction(prop, stoichiometries[i], f'Reaction_{i}'))
        
        times, states = ssa.simulate(initial_state, t_span[1])
        all_times.append(times)
        all_states.append(states)
    
    common_times = np.linspace(t_span[0], t_span[1], 200)
    mean_states = np.zeros((len(common_times), len(species_names)))
    std_states = np.zeros((len(common_times), len(species_names)))
    
    for t_idx, t in enumerate(common_times):
        values_at_t = []
        for run in range(num_runs):
            times = all_times[run]
            states = all_states[run]
            
            idx = np.searchsorted(times, t) - 1
            idx = max(0, min(idx, len(times) - 1))
            values_at_t.append(states[idx])
        
        values_array = np.array(values_at_t)
        mean_states[t_idx] = values_array.mean(axis=0)
        std_states[t_idx] = values_array.std(axis=0)
    
    elapsed = time.time() - start_time
    print(f"运行时间: {elapsed:.2f}秒")
    
    return common_times, mean_states, std_states, elapsed


def run_tau_leaping(species_names, propensities, stoichiometries,
                     initial_state, t_span):
    print(f"\n{'='*60}")
    print("Tau-Leaping")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    tl = TauLeaping(species_names, epsilon=0.05)
    for i, prop in enumerate(propensities):
        tl.add_reaction(Reaction(prop, stoichiometries[i], f'Reaction_{i}'))
    
    initial_means = initial_state.astype(float)
    initial_cov = np.eye(len(species_names)) * 0.1
    
    t_eval = np.linspace(t_span[0], t_span[1], 200)
    
    propensities_moment = [
        lambda m, t: propensities[0](m, t),
        lambda m, t: propensities[1](m, t),
        lambda m, t: propensities[2](m, t),
        lambda m, t: propensities[3](m, t),
    ]
    
    mc = MomentClosure(species_names, propensities_moment, stoichiometries,
                       closure_type='normal')
    times, means_history, covs_history = mc.simulate(
        initial_means, initial_cov, t_span, t_eval
    )
    
    stds_history = np.sqrt(np.array([np.diag(cov) for cov in covs_history]))
    
    elapsed = time.time() - start_time
    print(f"运行时间: {elapsed:.4f}秒")
    
    return times, means_history, stds_history, elapsed


def run_moment_closure(species_names, propensities, stoichiometries,
                       initial_state, t_span, closure_type='normal'):
    print(f"\n{'='*60}")
    print(f"Moment Closure: {closure_type}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    mc = MomentClosure(species_names, propensities, stoichiometries,
                       closure_type=closure_type)
    
    initial_means = initial_state.astype(float)
    initial_cov = np.eye(len(species_names)) * 0.1
    
    t_eval = np.linspace(t_span[0], t_span[1], 200)
    times, means_history, covs_history = mc.simulate(
        initial_means, initial_cov, t_span, t_eval
    )
    
    stds_history = np.sqrt(np.array([np.diag(cov) for cov in covs_history]))
    
    elapsed = time.time() - start_time
    print(f"运行时间: {elapsed:.4f}秒")
    print(f"最终均值: {means_history[-1]}")
    print(f"最终标准差: {stds_history[-1]}")
    
    return times, means_history, stds_history, elapsed


def compare_all_methods():
    species_names, ssa_props, moment_props, stoichs = create_model()
    
    initial_state = np.array([0, 0])
    t_span = (0.0, 50.0)
    
    print("\n" + "="*70)
    print("三种随机模拟方法对比: SSA vs Tau-Leaping vs Moment Closure")
    print("="*70)
    
    results = {}
    
    results['ssa'] = run_ssa(species_names, ssa_props, stoichs,
                             initial_state, t_span, num_runs=200)
    
    results['moment_normal'] = run_moment_closure(
        species_names, moment_props, stoichs, initial_state, t_span,
        closure_type='normal'
    )
    
    results['moment_lognormal'] = run_moment_closure(
        species_names, moment_props, stoichs, initial_state, t_span,
        closure_type='log_normal'
    )
    
    return results


def plot_comparison(results, species_names):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        'ssa': 'blue',
        'moment_normal': 'red',
        'moment_lognormal': 'green',
    }
    
    labels = {
        'ssa': 'SSA (200 runs)',
        'moment_normal': 'Moment (Normal)',
        'moment_lognormal': 'Moment (LogNormal)',
    }
    
    for method in ['ssa', 'moment_normal', 'moment_lognormal']:
        times, means, stds, _ = results[method]
        
        for i, name in enumerate(species_names):
            axes[0, i].plot(times, means[:, i], color=colors[method],
                           label=labels[method], linewidth=2)
            
            axes[0, i].fill_between(times,
                                    means[:, i] - 2 * stds[:, i],
                                    means[:, i] + 2 * stds[:, i],
                                    color=colors[method], alpha=0.15)
            
            axes[1, i].plot(times, stds[:, i], color=colors[method],
                           label=labels[method], linewidth=2)
    
    for i, name in enumerate(species_names):
        axes[0, i].set_xlabel('Time', fontsize=12)
        axes[0, i].set_ylabel('Mean Count', fontsize=12)
        axes[0, i].set_title(f'{name} - Mean ± 2σ', fontsize=14)
        axes[0, i].legend(fontsize=10)
        axes[0, i].grid(True, alpha=0.3)
        
        axes[1, i].set_xlabel('Time', fontsize=12)
        axes[1, i].set_ylabel('Standard Deviation', fontsize=12)
        axes[1, i].set_title(f'{name} - Standard Deviation', fontsize=14)
        axes[1, i].legend(fontsize=10)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比图已保存到 'method_comparison.png'")
    
    return fig


def performance_summary(results):
    print("\n" + "="*70)
    print("性能总结")
    print("="*70)
    
    methods = ['ssa', 'moment_normal', 'moment_lognormal']
    method_names = ['SSA (200 runs)', 'Moment (Normal)', 'Moment (LogNormal)']
    
    times = [results[m][3] for m in methods]
    
    for name, t in zip(method_names, times):
        print(f"{name:25s}: {t:.4f}秒")
    
    print("-"*70)
    print(f"加速比 (SSA vs Normal): {times[0]/times[1]:.1f}x")
    print(f"加速比 (SSA vs LogNormal): {times[0]/times[2]:.1f}x")
    
    return times


def accuracy_summary(results):
    print("\n" + "="*70)
    print("精度总结 (与SSA对比)")
    print("="*70)
    
    ssa_times, ssa_means, ssa_stds, _ = results['ssa']
    
    for method in ['moment_normal', 'moment_lognormal']:
        times, means, stds, _ = results[method]
        
        mean_error = np.mean(np.abs(means - ssa_means) / (ssa_means + 1e-6)) * 100
        std_error = np.mean(np.abs(stds - ssa_stds) / (ssa_stds + 1e-6)) * 100
        
        print(f"\n{method.upper()}:")
        print(f"  均值相对误差: {mean_error:.2f}%")
        print(f"  标准差相对误差: {std_error:.2f}%")


def main():
    np.random.seed(42)
    
    results = compare_all_methods()
    
    species_names = ['mRNA', 'Protein']
    
    plot_comparison(results, species_names)
    
    performance_summary(results)
    
    accuracy_summary(results)
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
1. **Gillespie SSA**:
   - 精确的随机模拟
   - 计算成本高（需要多次模拟求平均）
   - 作为金标准

2. **Moment Closure (正态近似)**:
   - 速度最快（比SSA快几个数量级）
   - 精度对于高拷贝数系统很好
   - 适合参数估计

3. **Moment Closure (对数正态近似)**:
   - 适合低拷贝数系统
   - 保持非负性
   - 数值稳定性稍差

**选择指南**:
- 需要精确分布 → SSA (或Tau-Leaping加速)
- 参数估计/快速探索 → Moment Closure
- 低拷贝数 → LogNormal封闭或SSA
    """)


if __name__ == "__main__":
    main()
