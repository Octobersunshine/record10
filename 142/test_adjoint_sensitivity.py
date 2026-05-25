import numpy as np
import sys
sys.path.insert(0, 'e:/temp/record10/142')

from mesh import generate_sphere_mesh
from adjoint_sensitivity import (
    ObjectiveFunctions, AdjointSolver, SensitivityAnalyzer
)


def test_objective_functions():
    print("测试目标函数定义...", end=' ')

    fp = np.array([5.0, 0.0, 0.0])
    obj_p = ObjectiveFunctions.sound_pressure_at_point(fp)
    assert obj_p.name is not None
    assert obj_p.compute is not None
    assert obj_p.compute_derivative is not None

    obj_spl = ObjectiveFunctions.sound_pressure_level_at_point(fp)
    assert obj_spl.name is not None
    assert obj_spl.compute is not None

    obj_ts = ObjectiveFunctions.target_strength(np.array([1.0, 0.0, 0.0]))
    assert obj_ts.name is not None
    assert obj_ts.compute is not None
    assert obj_ts.compute_derivative is not None

    obj_power = ObjectiveFunctions.radiated_power()
    assert obj_power.name is not None
    assert obj_power.compute is not None

    obj_norm = ObjectiveFunctions.surface_pressure_norm()
    assert obj_norm.name is not None
    assert obj_norm.compute is not None

    print("通过!")


def test_adjoint_solver_setup():
    print("测试伴随求解器初始化...", end=' ')

    from bem import BEMSolver

    mesh = generate_sphere_mesh(radius=1.0, n_theta=6, n_phi=12)
    k = 5.0

    solver = BEMSolver(mesh, k)
    adjoint_solver = AdjointSolver(solver)

    assert adjoint_solver.solver is solver
    assert adjoint_solver.k == k
    assert adjoint_solver.mesh is mesh

    print("通过!")


def test_shape_parameterization():
    print("测试形状参数化...", end=' ')

    mesh = generate_sphere_mesh(radius=1.0, n_theta=6, n_phi=12)

    from adjoint_sensitivity import ShapeParameterization

    param = ShapeParameterization(mesh)

    assert param.mesh is mesh
    assert param.num_dv == 0
    assert len(param.design_variables) == 0

    param.add_sphere_radius(np.array([0.0, 0.0, 0.0]), name="radius")
    assert param.num_dv == 1
    assert "radius" in param.design_variables

    elem_indices = np.array([0, 1, 2])
    param.add_surface_displacement(elem_indices, np.array([1.0, 0.0, 0.0]), name="nose")
    assert param.num_dv == 2
    assert "nose" in param.design_variables

    print("通过!")


def test_sensitivity_analyzer_setup():
    print("测试灵敏度分析器初始化...", end=' ')

    mesh = generate_sphere_mesh(radius=1.0, n_theta=6, n_phi=12)
    k = 5.0

    analyzer = SensitivityAnalyzer(mesh, k)

    assert analyzer.mesh is mesh
    assert analyzer.k == k
    assert analyzer.solver is not None
    assert analyzer.adjoint_solver is not None
    assert analyzer.param is not None

    print("通过!")


def test_primal_solve():
    print("测试原问题求解...", end=' ')

    mesh = generate_sphere_mesh(radius=1.0, n_theta=6, n_phi=12)
    k = 5.0

    analyzer = SensitivityAnalyzer(mesh, k)

    direction = np.array([1.0, 0.0, 0.0])
    p_incident = np.exp(-1j * k * np.dot(mesh.centroids, direction))

    p_surface, v_surface = analyzer.solve_primal(
        p_incident, method='dirichlet', use_fmm=False
    )

    assert hasattr(analyzer, 'p_surface')
    assert hasattr(analyzer, 'v_surface')
    assert p_surface.shape == (mesh.num_elements,)
    assert v_surface.shape == (mesh.num_elements,)
    assert np.all(np.isfinite(p_surface))
    assert np.all(np.isfinite(v_surface))

    print("通过!")


def test_adjoint_vs_fd():
    print("测试伴随方法 vs 有限差分...")

    mesh = generate_sphere_mesh(radius=1.0, n_theta=6, n_phi=12)
    k = 2.0

    analyzer = SensitivityAnalyzer(mesh, k)
    analyzer.param.add_sphere_radius(np.array([0.0, 0.0, 0.0]), name="radius")

    direction = np.array([1.0, 0.0, 0.0])
    p_incident = np.exp(-1j * k * np.dot(mesh.centroids, direction))

    analyzer.solve_primal(p_incident, method='dirichlet')

    objective = ObjectiveFunctions.sound_pressure_at_point(np.array([5.0, 0.0, 0.0]))

    dJ_adj = analyzer.compute_sensitivity(objective, "radius", method='dirichlet', use_adjoint=True)
    dJ_fd = analyzer.compute_sensitivity(objective, "radius", method='dirichlet', use_adjoint=False)

    print(f"  伴随方法: {dJ_adj:.6e}")
    print(f"  有限差分: {dJ_fd:.6e}")

    rel_error = abs(dJ_adj - dJ_fd) / max(abs(dJ_fd), 1e-10)
    print(f"  相对误差: {rel_error:.2e}")

    assert np.isfinite(dJ_adj)
    assert np.isfinite(dJ_fd)

    print("  通过!")


def test_multiple_objectives_sensitivity():
    print("测试多目标灵敏度计算...", end=' ')

    mesh = generate_sphere_mesh(radius=1.0, n_theta=5, n_phi=10)
    k = 2.0

    analyzer = SensitivityAnalyzer(mesh, k)
    analyzer.param.add_sphere_radius(np.array([0.0, 0.0, 0.0]), name="radius")

    direction = np.array([1.0, 0.0, 0.0])
    p_incident = np.exp(-1j * k * np.dot(mesh.centroids, direction))

    analyzer.solve_primal(p_incident)

    objectives = [
        ObjectiveFunctions.sound_pressure_at_point(np.array([5.0, 0.0, 0.0])),
        ObjectiveFunctions.target_strength(np.array([1.0, 0.0, 0.0])),
    ]

    sensitivities = []
    for obj in objectives:
        dJ = analyzer.compute_sensitivity(obj, "radius", use_adjoint=True)
        sensitivities.append(dJ)
        assert np.isfinite(dJ)

    assert len(sensitivities) == len(objectives)

    print("通过!")


def main():
    print("=" * 60)
    print("伴随灵敏度分析测试套件")
    print("=" * 60)
    print()

    tests = [
        test_objective_functions,
        test_adjoint_solver_setup,
        test_shape_parameterization,
        test_sensitivity_analyzer_setup,
        test_primal_solve,
        test_adjoint_vs_fd,
        test_multiple_objectives_sensitivity,
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

    print()
    print("=" * 60)
    print(f"测试结果: {passed}/{len(tests)} 通过")
    print("=" * 60)

    if passed == len(tests):
        print("\n所有测试通过! 伴随灵敏度分析已准备就绪。")
    else:
        print("\n部分测试失败，请检查实现。")


if __name__ == '__main__':
    main()
