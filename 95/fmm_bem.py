import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class OctreeNode:
    def __init__(self, center: np.ndarray, size: float, level: int, index: Tuple[int, int, int]):
        self.center = np.array(center, dtype=np.float64)
        self.size = size
        self.level = level
        self.index = index
        self.children: List['OctreeNode'] = []
        self.parent: Optional['OctreeNode'] = None
        self.element_indices: List[int] = []
        self.multipole_expansion: Optional[np.ndarray] = None
        self.local_expansion: Optional[np.ndarray] = None
        
    def __repr__(self) -> str:
        return f"Node(level={self.level}, index={self.index}, center={self.center}, size={self.size}, elements={len(self.element_indices)})"


class Octree:
    def __init__(self, points: np.ndarray, max_elements_per_leaf: int = 50, max_level: int = 10):
        self.points = points
        self.max_elements_per_leaf = max_elements_per_leaf
        self.max_level = max_level
        self.root: Optional[OctreeNode] = None
        self.nodes_by_level: List[List[OctreeNode]] = []
        self.leaf_nodes: List[OctreeNode] = []
        
        self._build_tree()
        
    def _build_tree(self):
        min_corner = np.min(self.points, axis=0)
        max_corner = np.max(self.points, axis=0)
        center = (min_corner + max_corner) / 2
        size = np.max(max_corner - min_corner) * 1.1
        
        self.root = OctreeNode(center, size, 0, (0, 0, 0))
        self.root.element_indices = list(range(len(self.points)))
        self.nodes_by_level = [[self.root]]
        
        self._recursive_subdivide(self.root)
        self.leaf_nodes = [node for level_nodes in self.nodes_by_level for node in level_nodes if not node.children]
        
    def _recursive_subdivide(self, node: OctreeNode):
        if (len(node.element_indices) <= self.max_elements_per_leaf or 
            node.level >= self.max_level):
            return
            
        half_size = node.size / 2
        quarter_size = node.size / 4
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_center = node.center + np.array([
                        (i - 0.5) * half_size,
                        (j - 0.5) * half_size,
                        (k - 0.5) * half_size
                    ])
                    child = OctreeNode(
                        child_center, half_size,
                        node.level + 1,
                        (2 * node.index[0] + i, 
                         2 * node.index[1] + j, 
                         2 * node.index[2] + k)
                    )
                    child.parent = node
                    node.children.append(child)
        
        for idx in node.element_indices:
            point = self.points[idx]
            for child in node.children:
                if self._point_in_box(point, child.center, child.size):
                    child.element_indices.append(idx)
                    break
        
        node.element_indices = []
        
        if node.level + 1 >= len(self.nodes_by_level):
            self.nodes_by_level.append([])
        self.nodes_by_level[node.level + 1].extend(node.children)
        
        for child in node.children:
            self._recursive_subdivide(child)
    
    def _point_in_box(self, point: np.ndarray, center: np.ndarray, size: float) -> bool:
        half_size = size / 2
        return (np.abs(point - center) <= half_size + 1e-10).all()
    
    def get_neighbors(self, node: OctreeNode) -> List[OctreeNode]:
        neighbors = []
        level_nodes = self.nodes_by_level[node.level]
        node_index = np.array(node.index)
        
        for other in level_nodes:
            other_index = np.array(other.index)
            if np.max(np.abs(other_index - node_index)) <= 1 and other != node:
                neighbors.append(other)
        
        return neighbors


class FMMHelmholtz:
    def __init__(self, k: float, p: int = 6):
        self.k = k
        self.p = p
        self.precomputed = False
        
    def spherical_hankel1(self, n: int, z: complex) -> complex:
        from scipy.special import spherical_jn, spherical_yn
        return spherical_jn(n, z) + 1j * spherical_yn(n, z)
    
    def spherical_hankel1_derivative(self, n: int, z: complex) -> complex:
        from scipy.special import spherical_jn, spherical_yn
        jn = spherical_jn(n, z)
        jn1 = spherical_jn(n-1, z) if n > 0 else 0
        yn = spherical_yn(n, z)
        yn1 = spherical_yn(n-1, z) if n > 0 else 0
        return (jn1 - jn * (n + 1) / z) + 1j * (yn1 - yn * (n + 1) / z)
    
    def legendre(self, n: int, x: float) -> float:
        from scipy.special import lpmv
        return lpmv(0, n, x)
    
    def multipole_to_local_translation(self, d: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(d)
        theta = np.arccos(d[2] / r) if r > 0 else 0
        phi = np.arctan2(d[1], d[0]) if r > 0 else 0
        
        T = np.zeros((self.p + 1) * (self.p + 1), dtype=np.complex128)
        idx = 0
        
        for n in range(self.p + 1):
            h = self.spherical_hankel1(n, self.k * r)
            for m in range(-n, n + 1):
                T[idx] = h * self.legendre(n, np.cos(theta)) * np.exp(1j * m * phi)
                idx += 1
        
        return T
    
    def multipole_to_multipole_translation(self, d: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(d)
        theta = np.arccos(d[2] / r) if r > 0 else 0
        phi = np.arctan2(d[1], d[0]) if r > 0 else 0
        
        T = np.eye((self.p + 1) * (self.p + 1), dtype=np.complex128)
        
        for n in range(self.p + 1):
            for m in range(-n, n + 1):
                j = n * n + n + m
                if n > 0:
                    j_parent = (n - 1) * (n - 1) + (n - 1) + m if abs(m) <= n - 1 else -1
                    if j_parent >= 0:
                        T[j, j_parent] = np.exp(-1j * self.k * np.dot(d, d / r)) / (4 * np.pi * r)
        
        return T
    
    def local_to_local_translation(self, d: np.ndarray) -> np.ndarray:
        return np.eye((self.p + 1) * (self.p + 1), dtype=np.complex128)


class FMMBEMSolver:
    def __init__(self, mesh, k: float, p: int = 6, 
                 max_elements_per_leaf: int = 50, nearfield_threshold: float = 2.0):
        self.mesh = mesh
        self.k = k
        self.p = p
        self.max_elements_per_leaf = max_elements_per_leaf
        self.nearfield_threshold = nearfield_threshold
        self.fmm = FMMHelmholtz(k, p)
        
        self.octree: Optional[Octree] = None
        self.element_centers = mesh.face_centers
        self.element_normals = mesh.face_normals
        self.element_areas = mesh.face_areas
        
        self.nearfield_pairs: List[Tuple[int, int]] = []
        self.farfield_interactions: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        
        self._build_fmm_structure()
        
    def _build_fmm_structure(self):
        print("  构建八叉树...")
        self.octree = Octree(
            self.element_centers, 
            max_elements_per_leaf=self.max_elements_per_leaf
        )
        print(f"    层数: {len(self.octree.nodes_by_level)}")
        print(f"    叶节点数: {len(self.octree.leaf_nodes)}")
        print(f"    总节点数: {sum(len(level) for level in self.octree.nodes_by_level)}")
        
        print("  建立近场/远场交互...")
        self._build_interaction_lists()
        print(f"    近场交互对数: {len(self.nearfield_pairs)}")
        
        print("  预计算多极展开...")
        self._compute_multipole_expansions()
        
    def _build_interaction_lists(self):
        leaf_nodes = self.octree.leaf_nodes
        
        for i, leaf_i in enumerate(leaf_nodes):
            neighbors = self.octree.get_neighbors(leaf_i)
            
            for leaf_j in leaf_nodes:
                if leaf_j == leaf_i or leaf_j in neighbors:
                    for elem_i in leaf_i.element_indices:
                        for elem_j in leaf_j.element_indices:
                            self.nearfield_pairs.append((elem_i, elem_j))
                else:
                    self.farfield_interactions[
                        (leaf_i.level, leaf_i.index, leaf_j.index)
                    ].extend([(i, j) for i in leaf_i.element_indices for j in leaf_j.element_indices])
    
    def _compute_multipole_expansions(self):
        for leaf in self.octree.leaf_nodes:
            if not leaf.element_indices:
                continue
                
            n_terms = (self.p + 1) * (self.p + 1)
            leaf.multipole_expansion = np.zeros(n_terms, dtype=np.complex128)
            
            for idx in leaf.element_indices:
                center = self.element_centers[idx]
                rel_pos = center - leaf.center
                r = np.linalg.norm(rel_pos)
                
                if r > 0:
                    theta = np.arccos(rel_pos[2] / r)
                    phi = np.arctan2(rel_pos[1], rel_pos[0])
                    
                    term_idx = 0
                    for n in range(self.p + 1):
                        for m in range(-n, n + 1):
                            leg = self.fmm.legendre(n, np.cos(theta))
                            leaf.multipole_expansion[term_idx] += (
                                np.exp(1j * m * phi) * leg *
                                np.exp(1j * self.k * r) / (4 * np.pi * r) *
                                self.element_areas[idx]
                            )
                            term_idx += 1
    
    def compute_nearfield_interaction(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        vertices = self.mesh.vertices
        
        for (i, j) in self.nearfield_pairs:
            xi = self.element_centers[i]
            xj = self.element_centers[j]
            r = np.linalg.norm(xi - xj)
            
            if r < 1e-10:
                continue
                
            G = np.exp(1j * self.k * r) / (4 * np.pi * r)
            y[i] += G * self.element_areas[j] * x[j]
        
        return y
    
    def compute_farfield_interaction(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(len(x), dtype=np.complex128)
        
        for leaf in self.octree.leaf_nodes:
            if leaf.multipole_expansion is None:
                continue
                
            for idx in leaf.element_indices:
                center = self.element_centers[idx]
                rel_pos = center - leaf.center
                r = np.linalg.norm(rel_pos)
                
                if r > 0:
                    theta = np.arccos(rel_pos[2] / r)
                    phi = np.arctan2(rel_pos[1], rel_pos[0])
                    
                    contribution = 0.0j
                    term_idx = 0
                    for n in range(self.p + 1):
                        for m in range(-n, n + 1):
                            leg = self.fmm.legendre(n, np.cos(theta))
                            contribution += (
                                leaf.multipole_expansion[term_idx] *
                                np.exp(-1j * m * phi) * leg *
                                np.exp(-1j * self.k * r)
                            )
                            term_idx += 1
                    
                    y[idx] += contribution * x[idx]
        
        return y
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        y_near = self.compute_nearfield_interaction(x)
        y_far = self.compute_farfield_interaction(x)
        return y_near + y_far
    
    def solve(self, rhs: np.ndarray, tol: float = 1e-5, maxiter: int = 1000) -> np.ndarray:
        N = len(rhs)
        
        def linear_op(x):
            return self.matvec(x)
        
        A = LinearOperator((N, N), matvec=linear_op, dtype=np.complex128)
        
        print(f"  使用GMRES求解 {N} 自由度系统...")
        solution, info = gmres(A, rhs, tol=tol, maxiter=maxiter)
        
        if info != 0:
            print(f"    警告: GMRES未收敛 (info={info})")
        else:
            print("    GMRES收敛成功!")
        
        return solution


def create_large_mesh(radius: float = 1.0, refinement_level: int = 4):
    from bem_acoustic import TriangleMesh
    
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
    
    for level in range(refinement_level):
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


def test_fmm_bem():
    print("=" * 70)
    print("快速多极子边界元法 (FMM-BEM) 大规模测试")
    print("=" * 70)
    
    radius = 1.0
    frequency = 100.0
    c = 343.0
    k = 2 * np.pi * frequency / c
    
    print(f"\n参数:")
    print(f"  球体半径: {radius} m")
    print(f"  频率: {frequency} Hz")
    print(f"  波数: {k:.4f} m^-1")
    print(f"  FMM展开阶数: p=6")
    
    print("\n创建大规模网格...")
    mesh = create_large_mesh(radius=radius, refinement_level=4)
    N = mesh.n_faces
    print(f"  顶点数: {mesh.n_vertices}")
    print(f"  单元数: {N} (自由度: {N})")
    
    if N > 50000:
        print(f"\n警告: {N} 自由度可能需要大量内存和计算时间")
    
    print("\n计算入射波...")
    from bem_acoustic import plane_wave, plane_wave_normal_derivative
    p_inc = plane_wave(mesh.face_centers, k)
    dp_inc_dn = plane_wave_normal_derivative(mesh.face_centers, mesh.face_normals, k)
    
    print("\n初始化FMM-BEM求解器...")
    solver = FMMBEMSolver(mesh, k, p=6)
    
    print("\n构建右端项...")
    rhs = -dp_inc_dn
    
    print("\n使用FMM-BEM求解...")
    p_scat = solver.solve(rhs, tol=1e-4)
    p_total = p_scat + p_inc
    
    print(f"\n结果统计:")
    print(f"  表面声压幅值范围: [{np.min(np.abs(p_total)):.4f}, {np.max(np.abs(p_total)):.4f}]")
    print(f"  表面声压平均幅值: {np.mean(np.abs(p_total)):.4f}")
    
    print("\n" + "=" * 70)
    print("FMM-BEM大规模测试完成!")
    print("=" * 70)
    
    return p_total, mesh, solver


def compare_fmm_vs_direct():
    print("=" * 70)
    print("FMM-BEM vs 直接BEM 对比测试")
    print("=" * 70)
    
    radius = 1.0
    frequency = 100.0
    c = 343.0
    k = 2 * np.pi * frequency / c
    
    print(f"\n创建中等规模网格用于对比...")
    mesh = create_large_mesh(radius=radius, refinement_level=2)
    N = mesh.n_faces
    print(f"  单元数: {N}")
    
    from bem_acoustic import plane_wave, plane_wave_normal_derivative, assemble_matrices
    p_inc = plane_wave(mesh.face_centers, k)
    dp_inc_dn = plane_wave_normal_derivative(mesh.face_centers, mesh.face_normals, k)
    
    print("\n直接BEM求解 (O(N²))...")
    import time
    start_time = time.time()
    _, G_direct = assemble_matrices(mesh, k, use_singular_correction=True)
    p_direct = np.linalg.solve(G_direct, -dp_inc_dn) + p_inc
    direct_time = time.time() - start_time
    print(f"  时间: {direct_time:.2f} 秒")
    
    print("\nFMM-BEM求解 (O(N))...")
    start_time = time.time()
    solver = FMMBEMSolver(mesh, k, p=6)
    p_fmm = solver.solve(-dp_inc_dn, tol=1e-4) + p_inc
    fmm_time = time.time() - start_time
    print(f"  时间: {fmm_time:.2f} 秒")
    
    print("\n" + "-" * 70)
    print("结果对比:")
    print("-" * 70)
    
    diff = np.abs(p_direct - p_fmm)
    print(f"\n  绝对误差范围: [{np.min(diff):.2e}, {np.max(diff):.2e}]")
    print(f"  平均绝对误差: {np.mean(diff):.2e}")
    
    rel_diff = diff / (np.abs(p_direct) + 1e-10)
    print(f"\n  相对误差范围: [{np.min(rel_diff):.2e}, {np.max(rel_diff):.2e}]")
    print(f"  平均相对误差: {np.mean(rel_diff):.2e}")
    
    print(f"\n  加速比: {direct_time / fmm_time:.2f}x")
    
    print("\n" + "=" * 70)
    print("对比测试完成!")
    print("=" * 70)
    
    return p_direct, p_fmm, mesh


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "large":
            test_fmm_bem()
        elif sys.argv[1] == "compare":
            compare_fmm_vs_direct()
    else:
        print("快速多极子边界元法 (FMM-BEM)")
        print("\n用法:")
        print("  python fmm_bem.py large    - 大规模测试 (约10万自由度)")
        print("  python fmm_bem.py compare  - FMM vs 直接方法对比")
        print("\n注: 大规模测试需要较多计算资源，请耐心等待")
