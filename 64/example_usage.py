#!/usr/bin/env python3
"""
变分蒙特卡洛（VMC）计算氢原子基态能量 - 使用示例

本脚本演示如何使用自适应步长、随机重配置和虚拟时间演化功能
"""

import numpy as np
from vmc_hydrogen import VMC_Hydrogen, optimize_alpha


def example_1_basic_vmc():
    """示例1: 基础VMC计算（固定α）"""
    print("=" * 60)
    print("示例1: 基础VMC计算（α = 1.0）")
    print("=" * 60)
    
    vmc = VMC_Hydrogen(alpha=1.0, n_walkers=2000, n_steps=1000, step_size=0.5)
    energies, accept_rate, _ = vmc.run(thermalization=200, adapt_step_size=True)
    
    avg_energy = np.mean(energies)
    error = np.std(energies) / np.sqrt(len(energies))
    
    print(f"平均接受率: {accept_rate:.2%}")
    print(f"基态能量: {avg_energy:.6f} ± {error:.6f} Hartree")
    print(f"精确值: -0.5 Hartree")
    print()


def example_2_grid_search():
    """示例2: 网格扫描优化α"""
    print("=" * 60)
    print("示例2: 网格扫描优化α")
    print("=" * 60)
    
    alpha_range = np.linspace(0.8, 1.2, 10)
    energies, variances, _, _ = optimize_alpha(
        alpha_range, n_walkers=1500, n_steps=800, initial_step_size=0.1
    )
    
    min_idx = np.argmin(energies)
    optimal_alpha = alpha_range[min_idx]
    min_energy = energies[min_idx]
    
    print(f"最优α: {optimal_alpha:.4f}")
    print(f"最小能量: {min_energy:.6f} Hartree")
    print(f"方差: {variances[min_idx]:.6f}")
    print()


def example_3_virtual_time_propagation():
    """示例3: 虚拟时间演化自动优化α"""
    print("=" * 60)
    print("示例3: 虚拟时间演化自动优化α")
    print("=" * 60)
    
    initial_alpha = 0.7  # 从远离最优值开始
    print(f"初始α: {initial_alpha}")
    
    vmc = VMC_Hydrogen(initial_alpha, n_walkers=2000, n_steps=500, step_size=0.1)
    alpha_hist, energy_hist, grad_hist = vmc.virtual_time_propagation(
        n_iterations=30, dt=0.05, thermalization=200, verbose=True
    )
    
    print(f"\n优化结果:")
    print(f"  最终α: {alpha_hist[-1]:.6f}")
    print(f"  最终能量: {energy_hist[-1]:.6f} Hartree")
    print(f"  最终梯度: {grad_hist[-1]:.6f}")
    print()


def example_4_compare_methods():
    """示例4: 比较不同初始α的收敛性"""
    print("=" * 60)
    print("示例4: 不同初始α的收敛性比较")
    print("=" * 60)
    
    initial_alphas = [0.6, 0.8, 1.0, 1.2, 1.4]
    
    results = []
    for alpha in initial_alphas:
        print(f"  优化初始α = {alpha}")
        vmc = VMC_Hydrogen(alpha, n_walkers=1500, n_steps=400, step_size=0.1)
        alpha_hist, energy_hist, grad_hist = vmc.virtual_time_propagation(
            n_iterations=25, dt=0.05, thermalization=150, verbose=False
        )
        results.append({
            'initial': alpha,
            'final_alpha': alpha_hist[-1],
            'final_energy': energy_hist[-1],
            'converged': abs(alpha_hist[-1] - 1.0) < 0.05
        })
    
    print("\n" + "-" * 60)
    print(f"{'初始α':^10} {'最终α':^12} {'最终能量':^15} {'收敛':^8}")
    print("-" * 60)
    for r in results:
        print(f"{r['initial']:^10.2f} {r['final_alpha']:^12.4f} "
              f"{r['final_energy']:^15.6f} {'是' if r['converged'] else '否':^8}")
    print()


if __name__ == "__main__":
    print("\n" + "%" * 60)
    print("%" + " " * 58 + "%")
    print("%" + "VMC - 氢原子基态能量计算演示".center(58) + "%")
    print("%" + " " * 58 + "%")
    print("%" * 60 + "\n")
    
    # 运行所有示例
    example_1_basic_vmc()
    example_2_grid_search()
    example_3_virtual_time_propagation()
    example_4_compare_methods()
    
    print("=" * 60)
    print("所有示例完成!")
    print("=" * 60)
