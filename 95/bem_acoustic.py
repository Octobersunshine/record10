import numpy as np
from scipy.sparse.linalg import gmres
from typing import Tuple, Callable, Optional


class TriangleMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices.astype(np.float64)
        self.faces = faces.astype(np.int32)
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        self.face_centers = self._compute_face_centers()
        self.face_normals = self._compute_face_normals()
        self.face_areas = self._compute_face_areas()

    def _compute_face_centers(self) -> np.ndarray:
        centers = np.zeros((self.n_faces, 3), dtype=np.float64)
        for i, face in enumerate(self.faces):
            centers[i] = np.mean(self.vertices[face], axis=0)
        return centers

    def _compute_face_normals(self) -> np.ndarray:
        normals = np.zeros((self.n_faces, 3), dtype=np.float64)
        for i, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normals[i] = normal / norm
        return normals

    def _compute_face_areas(self) -> np.ndarray:
        areas = np.zeros(self.n_faces, dtype=np.float64)
        for i, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            areas[i] = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        return areas


def green_function_3d(x: np.ndarray, y: np.ndarray, k: float) -> complex:
    r = np.linalg.norm(x - y)
    if r < 1e-10:
        return 0.0
    return np.exp(1j * k * r) / (4 * np.pi * r)


def green_function_normal_derivative_3d(x: np.ndarray, y: np.ndarray, ny: np.ndarray, k: float) -> complex:
    r_vec = x - y
    r = np.linalg.norm(r_vec)
    if r < 1e-10:
        return 0.0
    r_hat = r_vec / r
    dot_product = np.dot(r_hat, ny)
    G = np.exp(1j * k * r) / (4 * np.pi * r)
    dG_dn = G * (1j * k - 1.0 / r) * dot_product
    return dG_dn


def gauss_quadrature_triangle(n_points: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    if n_points == 3:
        xi = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        weights = np.array([1/3, 1/3, 1/3])
    elif n_points == 4:
        xi = np.array([[1/3, 1/3], [3/5, 1/5], [1/5, 3/5], [1/5, 1/5]])
        weights = np.array([-9/16, 25/48, 25/48, 25/48])
    elif n_points == 7:
        xi = np.array([
            [1/3, 1/3],
            [0.79742699, 0.10128651],
            [0.10128651, 0.79742699],
            [0.10128651, 0.10128651],
            [0.05971587, 0.47014206],
            [0.47014206, 0.05971587],
            [0.47014206, 0.47014206]
        ])
        weights = np.array([
            0.225,
            0.12593918,
            0.12593918,
            0.12593918,
            0.13239415,
            0.13239415,
            0.13239415
        ])
    else:
        raise ValueError(f"Unsupported number of quadrature points: {n_points}")
    return xi, weights


def gauss_quadrature_square(n_points: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    if n_points == 2:
        xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1.0, 1.0])
    elif n_points == 4:
        xi = np.array([-np.sqrt(3/7 + 2/7*np.sqrt(6/5)),
                       -np.sqrt(3/7 - 2/7*np.sqrt(6/5)),
                       np.sqrt(3/7 - 2/7*np.sqrt(6/5)),
                       np.sqrt(3/7 + 2/7*np.sqrt(6/5))])
        weights = np.array([(18 - np.sqrt(30)) / 36,
                           (18 + np.sqrt(30)) / 36,
                           (18 + np.sqrt(30)) / 36,
                           (18 - np.sqrt(30)) / 36])
    elif n_points == 6:
        t1 = np.sqrt(5/9 - 2/9*np.sqrt(10/7))
        t2 = np.sqrt(5/9 + 2/9*np.sqrt(10/7))
        xi = np.array([-t2, -t1, 0, t1, t2, 0])
        w1 = (322 + 13 * np.sqrt(70)) / 900
        w2 = (322 - 13 * np.sqrt(70)) / 900
        w0 = 128 / 225
        weights = np.array([w2, w1, w0, w1, w2, 0])
    else:
        raise ValueError(f"Unsupported number of quadrature points: {n_points}")
    
    xi_grid1, xi_grid2 = np.meshgrid(xi, xi)
    w_grid1, w_grid2 = np.meshgrid(weights, weights)
    
    xi_2d = np.column_stack([xi_grid1.flatten(), xi_grid2.flatten()])
    w_2d = (w_grid1 * w_grid2).flatten()
    
    xi_2d = (xi_2d + 1) / 2
    w_2d = w_2d / 4
    
    return xi_2d, w_2d


def compute_duffy_integral(
    x: np.ndarray,
    vertices: np.ndarray,
    k: float,
    integral_type: str = "single_layer",
    n_quad: int = 16
) -> complex:
    v0, v1, v2 = vertices
    
    d0 = np.linalg.norm(x - v0)
    d1 = np.linalg.norm(x - v1)
    d2 = np.linalg.norm(x - v2)
    vertex_idx = np.argmin([d0, d1, d2])
    
    if vertex_idx == 1:
        v0, v1, v2 = v1, v0, v2
    elif vertex_idx == 2:
        v0, v1, v2 = v2, v0, v1
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    J = np.column_stack([edge1, edge2])
    detJ = np.linalg.norm(np.cross(edge1, edge2))
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 1e-10 else np.zeros(3)
    
    n_quad_1d = int(np.sqrt(n_quad))
    xi, weights = gauss_quadrature_square(n_quad_1d)
    
    integral = 0.0 + 0.0j
    
    for i in range(len(xi)):
        u, v = xi[i]
        
        if u + v > 1.0:
            continue
        
        y = v0 + u * edge1 + v * edge2
        r_vec = x - y
        r = np.linalg.norm(r_vec)
        
        jacobian_factor = detJ
        
        if r < 1e-10:
            continue
        
        if integral_type == "single_layer":
            G = np.exp(1j * k * r) / (4 * np.pi * r)
            integral += weights[i] * G * jacobian_factor
        elif integral_type == "double_layer":
            r_hat = r_vec / r
            dot_product = np.dot(r_hat, normal)
            G = np.exp(1j * k * r) / (4 * np.pi * r)
            dG_dn = G * (1j * k - 1.0 / r) * dot_product
            integral += weights[i] * dG_dn * jacobian_factor
        else:
            raise ValueError(f"Unknown integral type: {integral_type}")
    
    return integral


def compute_subdivision_integral(
    x: np.ndarray,
    vertices: np.ndarray,
    k: float,
    integral_type: str = "single_layer",
    n_subdivisions: int = 3,
    n_quad: int = 7
) -> complex:
    v0, v1, v2 = vertices
    
    d0 = np.linalg.norm(x - v0)
    d1 = np.linalg.norm(x - v1)
    d2 = np.linalg.norm(x - v2)
    vertex_idx = np.argmin([d0, d1, d2])
    
    if vertex_idx == 1:
        v0, v1, v2 = v1, v0, v2
    elif vertex_idx == 2:
        v0, v1, v2 = v2, v0, v1
    
    triangles = [(v0, v1, v2)]
    
    for _ in range(n_subdivisions):
        new_triangles = []
        for tri in triangles:
            a, b, c = tri
            m_ab = (a + b) / 2
            m_bc = (b + c) / 2
            m_ca = (c + a) / 2
            new_triangles.extend([
                (a, m_ab, m_ca),
                (m_ab, b, m_bc),
                (m_ca, m_bc, c),
                (m_ab, m_bc, m_ca)
            ])
        triangles = new_triangles
    
    integral = 0.0 + 0.0j
    for tri in triangles:
        tri_array = np.array(tri)
        integral += compute_element_integral(x, tri_array, k, integral_type, n_quad)
    
    return integral


def compute_element_integral(
    x: np.ndarray,
    vertices: np.ndarray,
    k: float,
    integral_type: str = "single_layer",
    n_quad: int = 7,
    use_duffy: bool = True,
    use_subdivision: bool = False,
    near_singular_threshold: float = 2.0
) -> complex:
    v0, v1, v2 = vertices
    edge1 = v1 - v0
    edge2 = v2 - v0
    area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
    h = np.sqrt(area)
    
    min_distance = min(
        np.linalg.norm(x - v0),
        np.linalg.norm(x - v1),
        np.linalg.norm(x - v2)
    )
    
    is_near_singular = min_distance < near_singular_threshold * h
    
    if is_near_singular and use_duffy:
        return compute_duffy_integral(x, vertices, k, integral_type, n_quad=16)
    elif is_near_singular and use_subdivision:
        return compute_subdivision_integral(x, vertices, k, integral_type, n_subdivisions=3, n_quad=n_quad)
    else:
        xi, weights = gauss_quadrature_triangle(n_quad)
        
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 1e-10 else np.zeros(3)
        
        integral = 0.0 + 0.0j
        for i in range(len(xi)):
            eta1, eta2 = xi[i]
            y = v0 + eta1 * edge1 + eta2 * edge2
            
            if integral_type == "single_layer":
                G = green_function_3d(x, y, k)
                integral += weights[i] * G * area
            elif integral_type == "double_layer":
                dG_dn = green_function_normal_derivative_3d(x, y, normal, k)
                integral += weights[i] * dG_dn * area
            else:
                raise ValueError(f"Unknown integral type: {integral_type}")
        
        return integral


def assemble_matrices(
    mesh: TriangleMesh,
    k: float,
    use_burton_miller: bool = True,
    eta: float = None,
    use_singular_correction: bool = True,
    near_singular_threshold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    N = mesh.n_faces
    
    if eta is None:
        eta = 1.0 / k if abs(k) > 1e-10 else 1.0
    
    H = np.zeros((N, N), dtype=np.complex128)
    G = np.zeros((N, N), dtype=np.complex128)
    
    n_singular_corrections = 0
    
    for i in range(N):
        x = mesh.face_centers[i]
        nx = mesh.face_normals[i]
        
        for j in range(N):
            vertices = mesh.vertices[mesh.faces[j]]
            
            if i == j:
                H[i, j] = 0.5
                
                if use_singular_correction:
                    edge1 = vertices[1] - vertices[0]
                    edge2 = vertices[2] - vertices[0]
                    area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                    h = np.sqrt(area)
                    
                    min_distance = min(
                        np.linalg.norm(x - vertices[0]),
                        np.linalg.norm(x - vertices[1]),
                        np.linalg.norm(x - vertices[2])
                    )
                    
                    if min_distance < near_singular_threshold * h:
                        n_singular_corrections += 1
                        G[i, j] = compute_duffy_integral(x, vertices, k, "single_layer", n_quad=16)
                    else:
                        G[i, j] = compute_element_integral(x, vertices, k, "single_layer", use_duffy=False)
                else:
                    G[i, j] = compute_element_integral(x, vertices, k, "single_layer", use_duffy=False)
            else:
                if use_singular_correction:
                    H[i, j] = -compute_element_integral(
                        x, vertices, k, "double_layer",
                        near_singular_threshold=near_singular_threshold
                    )
                    G[i, j] = compute_element_integral(
                        x, vertices, k, "single_layer",
                        near_singular_threshold=near_singular_threshold
                    )
                else:
                    H[i, j] = -compute_element_integral(x, vertices, k, "double_layer", use_duffy=False)
                    G[i, j] = compute_element_integral(x, vertices, k, "single_layer", use_duffy=False)
    
    if n_singular_corrections > 0:
        print(f"  应用了 {n_singular_corrections} 次奇异积分修正")
    
    if use_burton_miller:
        A = H + 1j * eta * G
    else:
        A = H
    
    return A, G


def plane_wave(
    points: np.ndarray,
    k: float,
    direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
    amplitude: float = 1.0
) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    return amplitude * np.exp(1j * k * np.dot(points, direction))


def plane_wave_normal_derivative(
    points: np.ndarray,
    normals: np.ndarray,
    k: float,
    direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
    amplitude: float = 1.0
) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    p_inc = amplitude * np.exp(1j * k * np.dot(points, direction))
    dp_inc_dn = 1j * k * np.dot(direction, normals.T) * p_inc
    return dp_inc_dn


def solve_acoustic_scattering(
    mesh: TriangleMesh,
    k: float,
    p_inc: np.ndarray = None,
    dp_inc_dn: np.ndarray = None,
    boundary_condition: str = "neumann",
    use_burton_miller: bool = True,
    eta: float = None,
    solver_tol: float = 1e-5,
    use_singular_correction: bool = True,
    near_singular_threshold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    N = mesh.n_faces
    
    if p_inc is None:
        p_inc = plane_wave(mesh.face_centers, k)
    
    if dp_inc_dn is None:
        dp_inc_dn = plane_wave_normal_derivative(
            mesh.face_centers, mesh.face_normals, k
        )
    
    A, G = assemble_matrices(
        mesh, k, use_burton_miller, eta,
        use_singular_correction=use_singular_correction,
        near_singular_threshold=near_singular_threshold
    )
    
    if boundary_condition == "neumann":
        dp_dn = -dp_inc_dn
        rhs = -np.dot(G, dp_dn)
        if use_burton_miller:
            if eta is None:
                eta = 1.0 / k if abs(k) > 1e-10 else 1.0
            rhs = rhs + 1j * eta * p_inc
        p, info = gmres(A, rhs, tol=solver_tol)
        if info != 0:
            print(f"GMRES did not converge (info={info})")
        p_total = p + p_inc
        dp_total_dn = dp_dn + dp_inc_dn
    elif boundary_condition == "dirichlet":
        p = -p_inc
        rhs = np.dot(A, p)
        dp_dn, info = gmres(G, rhs, tol=solver_tol)
        if info != 0:
            print(f"GMRES did not converge (info={info})")
        p_total = p + p_inc
        dp_total_dn = dp_dn + dp_inc_dn
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_condition}")
    
    return p_total, dp_total_dn


def test_singular_integration_improvement():
    print("=" * 60)
    print("奇异积分改进测试")
    print("=" * 60)
    
    radius = 1.0
    frequency = 500.0
    c = 343.0
    k = 2 * np.pi * frequency / c
    
    print(f"\n参数:")
    print(f"  球体半径: {radius} m")
    print(f"  频率: {frequency} Hz")
    print(f"  波数: {k:.4f} m^-1")
    
    print("\n创建网格...")
    mesh = create_sphere_mesh(radius=radius, refinement_level=2)
    print(f"  顶点数: {mesh.n_vertices}")
    print(f"  单元数: {mesh.n_faces}")
    
    print("\n计算入射波...")
    p_inc = plane_wave(mesh.face_centers, k)
    
    print("\n" + "-" * 60)
    print("测试1: 不使用奇异积分修正")
    print("-" * 60)
    p_no_corr, _ = solve_acoustic_scattering(
        mesh, k, p_inc, None,
        use_singular_correction=False
    )
    
    print("\n" + "-" * 60)
    print("测试2: 使用Duffy变换进行奇异积分修正")
    print("-" * 60)
    p_with_corr, _ = solve_acoustic_scattering(
        mesh, k, p_inc, None,
        use_singular_correction=True
    )
    
    print("\n" + "-" * 60)
    print("结果比较")
    print("-" * 60)
    
    diff = np.abs(p_no_corr - p_with_corr)
    print(f"\n  声压差幅值范围: [{np.min(diff):.2e}, {np.max(diff):.2e}]")
    print(f"  平均声压差幅值: {np.mean(diff):.2e}")
    print(f"  最大声压差幅值: {np.max(diff):.2e}")
    
    rel_diff = diff / (np.abs(p_with_corr) + 1e-10)
    print(f"\n  相对声压差范围: [{np.min(rel_diff):.2e}, {np.max(rel_diff):.2e}]")
    print(f"  平均相对声压差: {np.mean(rel_diff):.2e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return p_no_corr, p_with_corr, mesh, k


def evaluate_field(
    mesh: TriangleMesh,
    p: np.ndarray,
    dp_dn: np.ndarray,
    eval_points: np.ndarray,
    k: float
) -> np.ndarray:
    N_eval = len(eval_points)
    p_scat = np.zeros(N_eval, dtype=np.complex128)
    
    for i in range(N_eval):
        x = eval_points[i]
        for j in range(mesh.n_faces):
            vertices = mesh.vertices[mesh.faces[j]]
            area = mesh.face_areas[j]
            normal = mesh.face_normals[j]
            
            G = compute_element_integral(x, vertices, k, "single_layer")
            dG_dn = compute_element_integral(x, vertices, k, "double_layer")
            
            p_scat[i] += p[j] * dG_dn - dp_dn[j] * G
    
    return p_scat


def create_sphere_mesh(radius: float = 1.0, refinement_level: int = 2) -> TriangleMesh:
    phi = (1 + np.sqrt(5)) / 2
    
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)
    
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    
    for _ in range(refinement_level):
        new_vertices = []
        new_faces = []
        vertex_map = {}
        
        def get_midpoint(v1, v2):
            key = tuple(sorted([v1, v2]))
            if key not in vertex_map:
                mid = (vertices[v1] + vertices[v2]) / 2
                mid = mid / np.linalg.norm(mid)
                vertex_map[key] = len(vertices) + len(new_vertices)
                new_vertices.append(mid)
            return vertex_map[key]
        
        for face in faces:
            v0, v1, v2 = face
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            
            new_faces.append([v0, a, c])
            new_faces.append([v1, b, a])
            new_faces.append([v2, c, b])
            new_faces.append([a, b, c])
        
        vertices = np.vstack([vertices, np.array(new_vertices)])
        faces = np.array(new_faces, dtype=np.int32)
    
    vertices = vertices * radius
    return TriangleMesh(vertices, faces)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_singular_integration_improvement()
    else:
        print("边界元法求解声场散射问题 - 示例程序")
        print("=" * 60)
        
        radius = 1.0
        frequency = 100.0
        c = 343.0
        k = 2 * np.pi * frequency / c
        
        print(f"\n参数:")
        print(f"  球体半径: {radius} m")
        print(f"  频率: {frequency} Hz")
        print(f"  波数: {k:.4f} m^-1")
        
        print("\n创建球体网格...")
        mesh = create_sphere_mesh(radius=radius, refinement_level=2)
        print(f"  顶点数: {mesh.n_vertices}")
        print(f"  单元数: {mesh.n_faces}")
        
        print("\n计算入射波...")
        p_inc = plane_wave(mesh.face_centers, k)
        dp_inc_dn = plane_wave_normal_derivative(mesh.face_centers, mesh.face_normals, k)
        
        print("\n求解表面声压 (Neumann边界条件, 刚性散射体)...")
        print("  使用Duffy变换进行奇异积分修正...")
        p_total, dp_total_dn = solve_acoustic_scattering(
            mesh, k, p_inc, dp_inc_dn,
            boundary_condition="neumann",
            use_singular_correction=True
        )
        
        print(f"\n结果统计:")
        print(f"  表面声压幅值范围: [{np.min(np.abs(p_total)):.4f}, {np.max(np.abs(p_total)):.4f}]")
        print(f"  表面声压平均幅值: {np.mean(np.abs(p_total)):.4f}")
        
        print("\n计算远场声压...")
        theta = np.linspace(0, np.pi, 37)
        r_far = 10.0
        eval_points = np.array([
            [r_far * np.sin(t), 0, r_far * np.cos(t)] for t in theta
        ])
        
        p_scat = evaluate_field(mesh, p_total - p_inc, dp_total_dn - dp_inc_dn, eval_points, k)
        p_far_total = p_scat + plane_wave(eval_points, k)
        
        print(f"  远场声压幅值范围: [{np.min(np.abs(p_far_total)):.4f}, {np.max(np.abs(p_far_total)):.4f}]")
        
        print("\n" + "=" * 60)
        print("提示: 运行 'python bem_acoustic.py test' 进行奇异积分改进对比测试")
        print("=" * 60)
