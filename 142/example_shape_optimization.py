import numpy as np
import sys
sys.path.insert(0, 'e:/temp/record10/142')

import matplotlib.pyplot as plt

from mesh import generate_sphere_mesh, generate_cylinder_mesh
from adjoint_sensitivity import (
    SensitivityAnalyzer, ObjectiveFunctions
)


def example_sphere_radius_sensitivity():
    print("=" * 70)
    print("示例1: 球体半径对目标强度的灵敏度分析")
    print("=" * 70)

    c = 1500
    rho = 1025
    frequency = 500
    k = 2 * np.pi * frequency / c

    print(f"\n参数设置:")
    print(f"  频率: {frequency} Hz")
    print(f"  波数: {k:.2f} m^-1")
    print(f"  声速: {c} m/s")

    radius = 1.0
    mesh = generate_sphere_mesh(radius=radius, n_theta=10, n_phi=20)

    print(f"\n网格信息:")
    print(f"  单元数: {mesh.num_elements}")
    print(f"  顶点数: {mesh.num_vertices}")

    analyzer = SensitivityAnalyzer(mesh, k, c=c, rho=rho)

    analyzer.param.add_sphere_radius(center=np.array([0.0, 0.0, 0.0]), name="radius")

    direction = np.array([1.0, 0.0, 0.0])
    p_incident = np.exp(-1j * k * np.dot(mesh.centroids, direction))

    print("\n求解原问题...", end='', flush=True)
    p_surface, v_surface = analyzer.solve_primal(
        p_incident, method='dirichlet', use_fmm=False
    )
    print("完成!")

    observation_dir = np.array([1.0, 0.0, 0.0])
    objective = ObjectiveFunctions.target_strength(
        observation_direction=observation_dir,
        reference_distance=1.0,
        r_observation=50.0
    )

    J = objective.compute(p_surface, v_surface, analyzer.solver)
    print(f"\n目标函数值 (目标强度): {J:.2f} dB")

    print("\n计算灵敏度 (伴随方法)...", end='', flush=True)
    dJ_dr_adjoint = analyzer.compute_sensitivity(
        objective, dv_name="radius", method='dirichlet', use_adjoint=True
    )
    print("完成!")
    print(f"  d(TS)/dr = {dJ_dr_adjoint:.4e} dB/m")

    print("\n计算灵敏度 (有限差分验证)...", end='', flush=True)
    dJ_dr_fd = analyzer.compute_sensitivity(
        objective, dv_name="radius", method='dirichlet', use_adjoint=False
    )
    print("完成!")
    print(f"  d(TS)/dr = {dJ_dr_fd:.4e} dB/m")

    rel_error = abs(dJ_dr_adjoint - dJ_dr_fd) / abs(dJ_dr_fd) * 100
    print(f"\n相对误差: {rel_error:.2f}%")

    return J, dJ_dr_adjoint, dJ_dr_fd


def example_multiple_objectives():
    print("\n" + "=" * 70)
    print("示例2: 多目标函数灵敏度分析")
    print("=" * 70)

    c = 343
    rho = 1.21
    frequency = 1000
    k = 2 * np.pi * frequency / c

    mesh = generate_sphere_mesh(radius=1.0, n_theta=8, n_phi=16)

    analyzer = SensitivityAnalyzer(mesh, k, c=c, rho=rho)
    analyzer.param.add_sphere_radius(center=np.array([0.0, 0.0, 0.0]), name="radius")

    direction = np.array([1.0, 0.0, 0.0])
    p_incident = np.exp(-1j * k * np.dot(mesh.centroids, direction))

    print("\n求解原问题...")
    p_surface, v_surface = analyzer.solve_primal(p_incident)

    objectives = [
        ("前向声压", ObjectiveFunctions.sound_pressure_at_point(np.array([5.0, 0.0, 0.0]))),
        ("侧向声压", ObjectiveFunctions.sound_pressure_at_point(np.array([0.0, 5.0, 0.0]))),
        ("辐射功率", ObjectiveFunctions.radiated_power()),
        ("表面声压范数", ObjectiveFunctions.surface_pressure_norm()),
    ]

    print("\n各目标函数灵敏度:")
    print(f"{'目标函数':<15} {'值':<20} {'dJ/dr':<20}")
    print("-" * 55)

    results = []
    for name, obj in objectives:
        J = obj.compute(p_surface, v_surface, analyzer.solver)
        dJ = analyzer.compute_sensitivity(obj, "radius", use_adjoint=True)
        results.append((name, J, dJ))
        print(f"{name:<15} {np.real(J):<20.4e} {dJ:<20.4e}")

    return results


def example_shape_optimization():
    print("\n" + "=" * 70)
    print("示例3: 简单形状优化 - 最小化前向散射")
    print("=" * 70)

    c = 1500
    rho = 1025
    frequency = 800
    k = 2 * np.pi * frequency / c

    radius_0 = 1.5
    mesh = generate_sphere_mesh(radius=radius_0, n_theta=8, n_phi=16)

    analyzer = SensitivityAnalyzer(mesh, k, c=c, rho=rho)
    analyzer.param.add_sphere_radius(center=np.array([0.0, 0.0, 0.0]), name="radius")

    observation_dir = np.array([1.0, 0.0, 0.0])
    objective = ObjectiveFunctions.target_strength(
        observation_direction=observation_dir, r_observation=50.0
    )

    direction = np.array([1.0, 0.0, 0.0])
    p_incident = np.exp(-1j * k * np.dot(mesh.centroids, direction))

    print("\n初始求解...")
    analyzer.solve_primal(p_incident)

    n_iterations = 5
    step_size = 0.05

    print(f"\n优化迭代 ({n_iterations}步, 步长={step_size}):")
    print(f"{'迭代':<8} {'半径(m)':<12} {'TS(dB)':<12} {'dJ/dr':<15}")
    print("-" * 50)

    history = []
    current_radius = radius_0

    for i in range(n_iterations):
        J = objective.compute(analyzer.p_surface, analyzer.v_surface, analyzer.solver)
        dJ = analyzer.compute_sensitivity(objective, "radius")

        history.append({
            'iteration': i,
            'radius': current_radius,
            'TS': J,
            'sensitivity': dJ
        })

        print(f"{i:<8} {current_radius:<12.3f} {J:<12.2f} {dJ:<15.4e}")

        dv_info = analyzer.param.design_variables["radius"]
        current_radius = current_radius - step_size * dJ
        new_mesh = dv_info['update'](current_radius)

        analyzer.mesh = new_mesh
        analyzer.solver.mesh = new_mesh
        p_incident_new = np.exp(-1j * k * np.dot(new_mesh.centroids, direction))
        analyzer.solve_primal(p_incident_new)

    print("\n优化完成!")

    return history


def plot_sensitivity_results(history1, history2=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if isinstance(history1, list) and len(history1) > 0 and 'iteration' in history1[0]:
        iterations = [h['iteration'] for h in history1]
        radii = [h['radius'] for h in history1]
        ts_values = [h['TS'] for h in history1]
        sensitivities = [h['sensitivity'] for h in history1]

        ax = axes[0, 0]
        ax.plot(iterations, radii, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('迭代')
        ax.set_ylabel('半径 (m)')
        ax.set_title('设计变量演化')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(iterations, ts_values, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('迭代')
        ax.set_ylabel('目标强度 (dB)')
        ax.set_title('目标函数演化')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(iterations, sensitivities, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('迭代')
        ax.set_ylabel('d(TS)/dr')
        ax.set_title('灵敏度演化')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(radii, ts_values, 'mo-', linewidth=2, markersize=8)
        ax.set_xlabel('半径 (m)')
        ax.set_ylabel('目标强度 (dB)')
        ax.set_title('设计空间')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('shape_optimization_results.png', dpi=150, bbox_inches='tight')
    print("\n结果图表已保存: shape_optimization_results.png")
    plt.close()


def main():
    print("\n" + "#" * 70)
    print("#  声学灵敏度分析与形状优化示例")
    print("#  使用伴随方法高效计算导数")
    print("#" * 70)

    J, dJ_adj, dJ_fd = example_sphere_radius_sensitivity()

    results = example_multiple_objectives()

    history = example_shape_optimization()

    plot_sensitivity_results(history)

    print("\n" + "=" * 70)
    print("总结:")
    print("=" * 70)
    print("✓ 伴随方法优势: 一次求解获得所有设计变量的导数")
    print("✓ 设计变量数: N_dv")
    print("✓ 传统方法: O(N_dv) 次求解")
    print("✓ 伴随方法: 2次求解 (原问题+伴随问题)")
    print("✓ 特别适合大尺度优化问题（汽车、潜艇等）")
    print("=" * 70)


if __name__ == '__main__':
    main()
