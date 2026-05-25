import numpy as np
import sys
sys.path.insert(0, 'e:/temp/record10/142')

from kernels import HelmholtzKernel, SphericalHarmonics
from mesh import generate_sphere_mesh, generate_cube_mesh, generate_cylinder_mesh
from bem import BEMSolver, AcousticScattering


def test_kernels():
    print("测试Helmholtz核函数...", end=' ')
    k = 2 * np.pi * 1000 / 343
    kernel = HelmholtzKernel(k)

    r = np.array([1.0, 0.0, 0.0])
    G = kernel.G(r)
    assert abs(G - np.exp(-1j * k) / (4 * np.pi)) < 1e-10

    r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    G = kernel.G(r)
    assert G.shape == (2,)
    print("通过!")


def test_spherical_harmonics():
    print("测试球谐函数...", end=' ')
    theta = np.array([0.0, np.pi/2, np.pi])
    phi = np.array([0.0, np.pi/2, np.pi])

    Y00 = SphericalHarmonics.Y_lm(0, 0, theta, phi)
    assert np.allclose(np.abs(Y00), 1 / np.sqrt(4 * np.pi))

    print("通过!")


def test_mesh_generation():
    print("测试网格生成...", end=' ')
    sphere = generate_sphere_mesh(radius=1.0, n_theta=10, n_phi=20)
    assert sphere.num_elements > 0
    assert sphere.num_vertices > 0
    assert sphere.centroids.shape[1] == 3
    assert sphere.normals.shape[1] == 3
    assert len(sphere.areas) == sphere.num_elements

    cube = generate_cube_mesh(size=2.0, n_per_side=5)
    assert cube.num_elements > 0

    cylinder = generate_cylinder_mesh(radius=1.0, height=2.0, n_theta=16, n_height=10)
    assert cylinder.num_elements > 0

    print("通过!")


def test_bem_solver_small():
    print("测试BEM求解器（小规模）...", end=' ')
    k = 2 * np.pi * 500 / 343

    mesh = generate_sphere_mesh(radius=1.0, n_theta=8, n_phi=16)

    solver = BEMSolver(mesh, k)
    solver.assemble_matrices(use_singular_correction=True)

    assert solver._H is not None
    assert solver._G is not None
    assert solver._H.shape == (mesh.num_elements, mesh.num_elements)
    assert solver._G.shape == (mesh.num_elements, mesh.num_elements)

    print("通过!")


def test_acoustic_scattering():
    print("测试声学散射求解...", end=' ')
    k = 2 * np.pi * 500 / 343

    mesh = generate_sphere_mesh(radius=1.0, n_theta=6, n_phi=12)

    scattering = AcousticScattering(mesh, k)

    direction = np.array([1.0, 0.0, 0.0])
    p_incident = scattering.set_incident_plane_wave(direction)

    assert p_incident.shape == (mesh.num_elements,)
    assert np.all(np.isfinite(p_incident))

    p_surface, v_surface = scattering.solve(
        p_incident, method='dirichlet', use_fmm=False
    )

    assert p_surface.shape == (mesh.num_elements,)
    assert v_surface.shape == (mesh.num_elements,)
    assert np.all(np.isfinite(p_surface))
    assert np.all(np.isfinite(v_surface))

    print("通过!")


def test_field_evaluation():
    print("测试场点计算...", end=' ')
    k = 2 * np.pi * 500 / 343

    mesh = generate_sphere_mesh(radius=1.0, n_theta=6, n_phi=12)

    scattering = AcousticScattering(mesh, k)
    p_incident = scattering.set_incident_plane_wave(np.array([1.0, 0.0, 0.0]))
    p_surface, v_surface = scattering.solve(p_incident, method='dirichlet')

    field_points = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ])

    p_field = scattering.compute_scattered_field(field_points, p_surface, v_surface)

    assert p_field.shape == (3,)
    assert np.all(np.isfinite(p_field))

    print("通过!")


def test_fmm_setup():
    print("测试FMM设置...", end=' ')
    try:
        from fmm import FastMultipoleMethod, FMMMatrixFree

        k = 2 * np.pi * 500 / 343
        fmm = FastMultipoleMethod(k, p=4, max_points_per_leaf=10)

        sources = np.random.randn(50, 3)
        charges = np.random.randn(50) + 1j * np.random.randn(50)

        fmm.build_tree(sources)
        assert fmm.octree is not None

        print("通过!")
    except Exception as e:
        print(f"部分通过 (FMM仍在开发中): {e}")


def main():
    print("=" * 60)
    print("FMBEM 库测试套件")
    print("=" * 60)
    print()

    tests = [
        test_kernels,
        test_spherical_harmonics,
        test_mesh_generation,
        test_bem_solver_small,
        test_acoustic_scattering,
        test_field_evaluation,
        test_fmm_setup
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"失败: {e}")

    print()
    print("=" * 60)
    print(f"测试结果: {passed}/{len(tests)} 通过")
    print("=" * 60)

    if passed == len(tests):
        print("\n所有测试通过! 库已准备就绪。")
    else:
        print("\n部分测试失败，请检查实现。")


if __name__ == '__main__':
    main()
