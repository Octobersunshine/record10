#!/usr/bin/env python
"""
气球模稳定性与第二稳定区分析脚本
"""

from grad_shafranov_solver import GradShafranovSolver
import numpy as np


def analyze_scenario(scenario_name, p0, alpha_p, q0):
    print("\n" + "="*70)
    print(f"场景: {scenario_name}")
    print(f"  p0 = {p0:.1e} Pa, alpha = {alpha_p:.1f}, q0 = {q0:.1f}")
    print("="*70)
    
    solver = GradShafranovSolver(
        Rmin=0.5, Rmax=1.5,
        Zmin=-0.6, Zmax=0.6,
        nr=65, nz=65
    )
    
    solver.solve(p0=p0, alpha=alpha_p, max_iter=100, relaxation=0.5)
    
    results = solver.analyze_stability(p0=p0, alpha=alpha_p, q0=q0)
    solver.print_stability_summary(results)
    
    return solver, results


def main():
    print("="*70)
    print("气球模稳定性与第二稳定区分析")
    print("="*70)
    
    scenarios = [
        ("低β常规托卡马克", 5e4, 2.0, 1.5),
        ("中等β第一稳定区", 1e5, 2.0, 2.0),
        ("高β接近第二稳定区", 3e5, 1.5, 2.5),
        ("高剪切高β工况", 2e5, 1.0, 3.0),
    ]
    
    all_results = []
    
    for name, p0, alpha_p, q0 in scenarios:
        solver, results = analyze_scenario(name, p0, alpha_p, q0)
        all_results.append((name, solver, results))
    
    print("\n" + "="*70)
    print("跨场景比较 - 第二稳定区访问潜力")
    print("="*70)
    
    print(f"\n{'场景':<25} {'β_max (%)':>12} {'s_mean':>10} {'访问潜力':>12}")
    print("-" * 60)
    
    for name, _, results in all_results:
        beta_max = np.max(results['beta']) * 100
        s_mean = np.mean(results['s'][1:-1, 1:-1])
        access = np.mean(results['access_potential'][1:-1, 1:-1])
        
        marker = "✓" if access > 0.3 else ("○" if access > 0.1 else "✗")
        print(f"{name:<25} {beta_max:>12.2f} {s_mean:>10.3f} {access:>12.3f} {marker}")
    
    print("\n" + "="*70)
    print("生成最后场景的稳定性可视化图...")
    print("="*70)
    
    last_solver = all_results[-1][1]
    last_results = all_results[-1][2]
    
    last_solver.plot_stability_maps(last_results)
    last_solver.plot_stability_diagram(last_results)
    
    print("\n✓ 分析完成！")
    print("\n关键物理结论:")
    print("  1. 增加β（等离子体压力）提高第二稳定区访问可能性")
    print("  2. 足够的磁剪切（s > 0.2）是进入第二稳定区的关键")
    print("  3. 安全因子 q > 2 有助于稳定气球模")
    print("  4. 高β + 高剪切构型最有可能访问第二稳定区")


if __name__ == "__main__":
    main()
