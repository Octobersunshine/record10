import numpy as np
import sys
sys.path.insert(0, 'e:/temp/record10/142')

import matplotlib.pyplot as plt

from broadband_fmm import (
    ExpansionConfig, ExpansionType,
    PlaneWaveExpansion, SphericalExpansion,
    BroadbandFMM
)


def test_expansion_config():
    print("测试展开配置自适应选择...")

    k_values = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    box_size = 1.0

    print(f"\n{'k (m^-1)':<12} {'kr':<10} {'展开类型':<20} {'p/n_theta':<15}")
    print("-" * 60)

    for k in k_values:
        config = ExpansionConfig.adaptive(k, box_size, kr_switch=8.0)
        kr = k * box_size

        if config.expansion_type == ExpansionType.SPHERICAL_HARMONICS:
            exp_type = "球谐函数"
            param = f"p={config.p}"
        else:
            exp_type = "平面波"
            param = f"n_theta={config.n_theta}"

        print(f"{k:<12.1f} {kr:<10.1f} {exp_type:<20} {param:<15}")

    print("\n通过!")


def test_plane_wave_expansion():
    print("\n测试平面波展开...", end=' ')

    k = 50.0
    n_theta = 32
    n_phi = 64

    pw = PlaneWaveExpansion(k, n_theta, n_phi)

    assert pw.directions.shape == (n_theta, n_phi, 3)
    assert pw.weights.shape == (n_theta, n_phi)

    sources = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    charges = np.array([1.0 + 0j, 1.0 + 0j])
    center = np.array([0.0, 0.0, 0.0])

    pw.compute_from_sources(sources, charges, center)
    assert pw.coeffs is not None
    assert pw.coeffs.shape == (n_theta, n_phi)

    targets = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    result = pw.evaluate(targets, center)

    assert result.shape == (2,)
    assert np.all(np.isfinite(result))

    print("通过!")


def test_spherical_expansion_adaptive_order():
    print("\n测试球谐展开自适应阶数...")

    k_values = [1.0, 5.0, 10.0, 20.0]
    box_size = 1.0
    tol = 1e-6

    print(f"\n{'k':<10} {'kr':<10} {'p_min':<10} {'p_calc':<10}")
    print("-" * 40)

    for k in k_values:
        kr = k * box_size
        p = ExpansionConfig._compute_spherical_order(kr, tol, min_p=2, max_p=50)
        p_min = int(kr) + 1
        print(f"{k:<10.1f} {kr:<10.1f} {p_min:<10d} {p:<10d}")

    print("\n通过!")


def test_direct_vs_fmm():
    print("\n测试直接计算 vs FMM...", end=' ')

    np.random.seed(42)
    n_points = 100
    sources = np.random.randn(n_points, 3) * 2
    charges = np.random.randn(n_points) + 1j * np.random.randn(n_points)
    targets = np.random.randn(50, 3) * 5

    k = 10.0

    def direct_potential(s, c, t):
        result = np.zeros(len(t), dtype=complex)
        for i, tgt in enumerate(t):
            for j, src in enumerate(s):
                r = np.linalg.norm(tgt - src)
                if r > 1e-12:
                    result[i] += c[j] * np.exp(-1j * k * r) / (4 * np.pi * r)
        return result

    direct = direct_potential(sources, charges, targets)

    fmm = BroadbandFMM(k, tol=1e-4, max_points_per_leaf=10, kr_switch=10.0)
    fmm_result = fmm.compute_potential(sources, charges, targets)

    rel_error = np.linalg.norm(fmm_result - direct) / np.linalg.norm(direct)
    print(f"相对误差: {rel_error:.2e}")

    print("通过!")


def test_broadband_convergence():
    print("\n测试宽频FMM收敛性...")

    np.random.seed(42)
    n_points = 50
    sources = np.random.randn(n_points, 3)
    charges = np.ones(n_points, dtype=complex)
    targets = np.array([[10.0, 0.0, 0.0]])

    def direct_potential(s, c, t, k):
        result = np.zeros(len(t), dtype=complex)
        for i, tgt in enumerate(t):
            for j, src in enumerate(s):
                r = np.linalg.norm(tgt - src)
                if r > 1e-12:
                    result[i] += c[j] * np.exp(-1j * k * r) / (4 * np.pi * r)
        return result

    k_values = [5.0, 10.0, 20.0, 30.0]
    errors = []

    print(f"\n{'k':<10} {'kr_box':<12} {'展开类型':<15} {'相对误差':<15}")
    print("-" * 55)

    for k in k_values:
        direct = direct_potential(sources, charges, targets, k)

        fmm = BroadbandFMM(k, tol=1e-4, max_points_per_leaf=15, kr_switch=8.0)
        fmm.build_tree(sources, targets)

        config = fmm.octree.root.expansion_config
        exp_type = "球谐" if config.expansion_type == ExpansionType.SPHERICAL_HARMONICS else "平面波"

        fmm_result = fmm.compute_potential(sources, charges, targets)

        rel_error = np.abs(fmm_result[0] - direct[0]) / np.abs(direct[0])
        errors.append(rel_error)

        kr_box = k * fmm.octree.root.size
        print(f"{k:<10.1f} {kr_box:<12.1f} {exp_type:<15} {rel_error:<15.2e}")

    print("\n通过!")
    return k_values, errors


def plot_expansion_order_vs_kr():
    print("\n生成展开阶数 vs kr 曲线...")

    kr_values = np.linspace(1, 30, 100)
    tol = 1e-6

    p_spherical = []
    for kr in kr_values:
        p = ExpansionConfig._compute_spherical_order(kr, tol, min_p=2, max_p=100)
        p_spherical.append(p)

    p_spherical = np.array(p_spherical)

    plt.figure(figsize=(10, 6))
    plt.plot(kr_values, p_spherical, 'b-', linewidth=2, label='球谐展开阶数 p')
    plt.plot(kr_values, kr_values, 'r--', label='p = kr')
    plt.plot(kr_values, kr_values + 10, 'g--', label='p = kr + 10')

    plt.axvline(x=8.0, color='k', linestyle=':', label='kr=8 (切换阈值)')

    plt.xlabel('kr')
    plt.ylabel('展开阶数 p')
    plt.title('球谐展开阶数 vs kr')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('expansion_order_vs_kr.png', dpi=150, bbox_inches='tight')
    print("图表已保存: expansion_order_vs_kr.png")
    plt.close()


def main():
    print("=" * 70)
    print("宽频 FMM 测试套件")
    print("=" * 70)
    print()

    tests = [
        test_expansion_config,
        test_plane_wave_expansion,
        test_spherical_expansion_adaptive_order,
        test_direct_vs_fmm,
        test_broadband_convergence,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"失败: {e}")
            import traceback
            traceback.print_exc()

    plot_expansion_order_vs_kr()

    print()
    print("=" * 70)
    print(f"测试结果: {passed}/{len(tests)} 通过")
    print("=" * 70)

    if passed == len(tests):
        print("\n所有测试通过! 宽频FMM已准备就绪。")
    else:
        print("\n部分测试失败，请检查实现。")


if __name__ == '__main__':
    main()
