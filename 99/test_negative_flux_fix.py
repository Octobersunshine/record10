import numpy as np
from sn_solver import Mesh, SNQuadrature, SNSolver, Material, solve_sn_1d


def test_comparison():
    print("=" * 80)
    print("负通量修复方法对比测试")
    print("=" * 80)

    x_min, x_max = 0.0, 1.0
    n_cells = 5
    sn_order = 8
    sigma_t = 20.0
    sigma_s = 0.9 * sigma_t
    source = np.ones(n_cells)

    spatial_methods = ['dd', 'wdd', 'step', 'sc', 'edd']
    fix_methods = ['simple', 'conservative', 'redistribute', 'scale']

    print(f"\n测试条件: σ_t={sigma_t}, σ_s={sigma_s:.1f}, τ={sigma_t * (x_max-x_min)/n_cells:.2f}")
    print(f"SN{sn_order}, 网格数={n_cells}")
    print()

    print(f"{'方法':<15s} {'修复':<8s} {'min(φ)':>15s} {'min(ψ)':>15s} {'平均φ':>12s} {'粒子平衡':>12s}")
    print("-" * 80)

    for sm in spatial_methods:
        mesh = Mesh(x_min, x_max, n_cells)
        quad = SNQuadrature(sn_order)
        materials = [Material(sigma_t, sigma_s) for _ in range(n_cells)]

        solver = SNSolver(mesh, quad, materials, 'vacuum', 'vacuum',
                          fix_negative_flux=False, spatial_method=sm)
        phi, psi = solver.solve(source, max_iter=2000, tol=1e-8)

        mean_phi = np.mean(phi)
        balance = compute_balance(mesh, phi, psi, sigma_t, sigma_s, source, quad)

        print(f"{sm.upper():<15s} {'无':<8s} {np.min(phi):>15.6e} {np.min(psi):>15.6e} {mean_phi:>12.4f} {balance:>12.4%}")

        for fm in fix_methods:
            solver = SNSolver(mesh, quad, materials, 'vacuum', 'vacuum',
                              fix_negative_flux=True,
                              negative_flux_fix_method=fm,
                              spatial_method=sm)
            phi, psi = solver.solve(source, max_iter=2000, tol=1e-8)

            mean_phi = np.mean(phi)
            balance = compute_balance(mesh, phi, psi, sigma_t, sigma_s, source, quad)

            label = f"  +{fm}"
            print(f"{label:<15s} {'是':<8s} {np.min(phi):>15.6e} {np.min(psi):>15.6e} {mean_phi:>12.4f} {balance:>12.4%}")

        print()


def compute_balance(mesh, phi, psi, sigma_t, sigma_s, source, quad):
    dx = mesh.dx
    total_source = np.sum(source) * dx
    sigma_a = sigma_t - sigma_s
    total_absorption = np.sum(sigma_a * phi) * dx

    total_leakage = 0.0
    for n in range(len(quad.mus)):
        if quad.mus[n] > 0:
            total_leakage += 0.5 * quad.weights[n] * quad.mus[n] * psi[n, -1]
        else:
            total_leakage += 0.5 * quad.weights[n] * abs(quad.mus[n]) * psi[n, 0]

    balance = abs(total_source - total_absorption - total_leakage) / (total_source + 1e-10)
    return balance


def test_extreme_cases():
    print("\n" + "=" * 80)
    print("极端条件下的鲁棒性测试")
    print("=" * 80)

    x_min, x_max = 0.0, 1.0
    sn_order = 8

    test_cases = [
        ("σ_t=50, 粗网格", 50.0, 0.95, 5),
        ("σ_t=100, 粗网格", 100.0, 0.98, 5),
        ("材料界面突变", None, None, 10),
    ]

    for name, sigma_t, sigma_s_ratio, n_cells in test_cases:
        print(f"\n测试: {name}")
        print("-" * 60)

        if name == "材料界面突变":
            mesh = Mesh(x_min, x_max, n_cells)
            materials = []
            source = np.ones(n_cells)
            for i in range(n_cells):
                if i < n_cells // 2:
                    materials.append(Material(sigma_t=50.0, sigma_s=49.0))
                else:
                    materials.append(Material(sigma_t=1.0, sigma_s=0.5))
        else:
            mesh = Mesh(x_min, x_max, n_cells)
            sigma_s = sigma_t * sigma_s_ratio
            materials = [Material(sigma_t, sigma_s) for _ in range(n_cells)]
            source = np.ones(n_cells)

        quad = SNQuadrature(sn_order)

        methods = [
            ('DD + conservative', 'dd', 'conservative'),
            ('WDD + conservative', 'wdd', 'conservative'),
            ('EDD + conservative', 'edd', 'conservative'),
            ('SC + redistribute', 'sc', 'redistribute'),
        ]

        print(f"{'方法':<25s} {'min(φ)':>15s} {'min(ψ)':>15s} {'最大φ':>12s}")
        print("-" * 60)

        for label, sm, fm in methods:
            solver = SNSolver(mesh, quad, materials, 'vacuum', 'vacuum',
                              fix_negative_flux=True,
                              negative_flux_fix_method=fm,
                              spatial_method=sm)
            phi, psi = solver.solve(source, max_iter=3000, tol=1e-8)

            print(f"{label:<25s} {np.min(phi):>15.6e} {np.min(psi):>15.6e} {np.max(phi):>12.4f}")


def verify_physical_correctness():
    print("\n" + "=" * 80)
    print("物理正确性验证 - 纯吸收体解析解对比")
    print("=" * 80)

    x_min, x_max = 0.0, 5.0
    n_cells = 50
    sn_order = 16

    sigma_t = 1.0
    sigma_s = 0.0
    source = np.ones(n_cells)

    print(f"\n解析解: 纯吸收体 (σ_s=0)")
    print()

    methods = [
        ('DD + 无修复', 'dd', None, False),
        ('DD + conservative', 'dd', 'conservative', True),
        ('WDD + conservative', 'wdd', 'conservative', True),
        ('EDD + conservative', 'edd', 'conservative', True),
        ('SC + redistribute', 'sc', 'redistribute', True),
    ]

    x = Mesh(x_min, x_max, n_cells).x_centers
    dx = (x_max - x_min) / n_cells
    sigma_a = sigma_t - sigma_s

    phi_analytic = 1.0 / sigma_a * (1 - (np.sinh(sigma_a * (x_max - x)) + np.sinh(sigma_a * x)) / np.sinh(sigma_a * x_max))

    print(f"{'方法':<25s} {'L1误差':>12s} {'L2误差':>12s} {'最大误差':>12s} {'最小φ':>12s}")
    print("-" * 80)

    for label, sm, fm, fix_neg in methods:
        mesh, phi, _ = solve_sn_1d(x_min, x_max, n_cells, sn_order,
                                    sigma_t, sigma_s, source,
                                    spatial_method=sm,
                                    fix_negative_flux=fix_neg,
                                    negative_flux_fix_method=fm if fm else 'simple',
                                    tol=1e-8)

        l1_err = np.mean(np.abs(phi - phi_analytic)) / np.mean(np.abs(phi_analytic))
        l2_err = np.sqrt(np.mean((phi - phi_analytic) ** 2)) / np.sqrt(np.mean(phi_analytic ** 2))
        max_err = np.max(np.abs(phi - phi_analytic)) / np.max(phi_analytic)

        print(f"{label:<25s} {l1_err:>12.4%} {l2_err:>12.4%} {max_err:>12.4%} {np.min(phi):>12.6e}")


def main():
    test_comparison()
    test_extreme_cases()
    verify_physical_correctness()

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
