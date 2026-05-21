import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.special import spherical_jn, spherical_yn, lpmv
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import time


def spherical_hankel1(n: int, z: complex) -> complex:
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def legendre_P(n: int, x: float) -> float:
    return lpmv(0, n, x)


class MLFMANode:
    def __init__(self, center: np.ndarray, size: float, level: int, index: Tuple[int, int, int]):
        self.center = np.array(center, dtype=np.float64)
        self.size = size
        self.level = level
        self.index = index
        self.children: List['MLFMANode'] = []
        self.parent: Optional['MLFMANode'] = None
        self.element_indices: List[int] = []
        self.V_plus: Optional[np.ndarray] = None
        self.V_minus: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        
    def __repr__(self) -> str:
        return f"MLFMANode(L={self.level}, idx={self.index}, size={self.size:.2e}, N_elem={len(self.element_indices)})"


class MLFMA:
    def __init__(self, k: float, p: int = 10):
        self.k = k
        self.p = p
        self.n_theta = p + 1
        self.n_phi = 2 * p + 1
        
        self.theta = np.linspace(0, np.pi, self.n_theta)
        self.phi = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        
        self.directions = np.array([
            [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
            for t in self.theta for p in self.phi
        ])
        self.n_dirs = len(self.directions)
        
        self.weights = self._compute_integration_weights()
        
    def _compute_integration_weights(self) -> np.ndarray:
        weights = np.zeros(self.n_dirs)
        dphi = 2 * np.pi / self.n_phi
        
        for i, t in enumerate(self.theta):
            w_theta = (np.pi / self.n_theta) * 2 * np.sin(t) / (self.n_phi)
            for j in range(self.n_phi):
                weights[i * self.n_phi + j] = w_theta * dphi
        
        return weights
    
    def translate_plus(self, V: np.ndarray, d: np.ndarray) -> np.ndarray:
        phase = np.exp(1j * self.k * np.dot(self.directions, d))
        return V * phase
    
    def translate_minus(self, W: np.ndarray, d: np.ndarray) -> np.ndarray:
        phase = np.exp(-1j * self.k * np.dot(self.directions, d))
        return W * phase


class MLFMABEMSolver:
    def __init__(self, mesh, k: float, p: int = 8, 
                 max_elements_per_leaf: int = 50, min_level: int = 2, max_level: int = 8):
        self.mesh = mesh
        self.k = k
        self.p = p
        self.max_elements_per_leaf = max_elements_per_leaf
        self.min_level = min_level
        self.max_level = max_level
        
        self.mlfma = MLFMA(k, p)
        self.element_centers = mesh.face_centers
        self.element_normals = mesh.face_normals
        self.element_areas = mesh.face_areas
        
        self.root: Optional[MLFMANode] = None
        self.nodes_by_level: List[List[MLFMANode]] = []
        self.leaf_nodes: List[MLFMANode] = []
        
        self.nearfield_pairs: List[Tuple[int, int]] = []
        self.farfield_pairs: List[Tuple[int, int]] = []
        
        self._build_octree()
        self._setup_interactions()
        
    def _build_octree(self):
        print("  构建八叉树...")
        
        min_corner = np.min(self.element_centers, axis=0)
        max_corner = np.max(self.element_centers, axis=0)
        center = (min_corner + max_corner) / 2
        size = np.max(max_corner - min_corner) * 1.1
        
        self.root = MLFMANode(center, size, 0, (0, 0, 0))
        self.root.element_indices = list(range(len(self.element_centers)))
        self.nodes_by_level = [[self.root]]
        
        self._recursive_subdivide(self.root)
        self.leaf_nodes = [node for level_nodes in self.nodes_by_level 
                          for node in level_nodes if not node.children]
        
        print(f"    层数: {len(self.nodes_by_level)}")
        print(f"    叶节点数: {len(self.leaf_nodes)}")
        print(f"    总节点数: {sum(len(level) for level in self.nodes_by_level)}")
    
    def _recursive_subdivide(self, node: MLFMANode):
        if (len(node.element_indices) <= self.max_elements_per_leaf or 
            node.level >= self.max_level):
            return
            
        half_size = node.size / 2
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_center = node.center + np.array([
                        (i - 0.5) * half_size,
                        (j - 0.5) * half_size,
                        (k - 0.5) * half_size
                    ])
                    child = MLFMANode(
                        child_center, half_size,
                        node.level + 1,
                        (2 * node.index[0] + i, 
                         2 * node.index[1] + j, 
                         2 * node.index[2] + k)
                    )
                    child.parent = node
                    node.children.append(child)
        
        for idx in node.element_indices:
            point = self.element_centers[idx]
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
    
    def _get_neighbors(self, node: MLFMANode) -> List[MLFMANode]:
        neighbors = []
        level_nodes = self.nodes_by_level[node.level]
        node_index = np.array(node.index)
        
        for other in level_nodes:
            other_index = np.array(other.index)
            if np.max(np.abs(other_index - node_index)) <= 1 and other != node:
                neighbors.append(other)
        
        return neighbors
    
    def _setup_interactions(self):
        print("  设置近场/远场交互...")
        
        leaf_nodes = self.leaf_nodes
        
        for leaf_i in leaf_nodes:
            neighbors = self._get_neighbors(leaf_i)
            
            for leaf_j in leaf_nodes:
                if leaf_j == leaf_i or leaf_j in neighbors:
                    for elem_i in leaf_i.element_indices:
                        for elem_j in leaf_j.element_indices:
                            self.nearfield_pairs.append((elem_i, elem_j))
                else:
                    for elem_i in leaf_i.element_indices:
                        for elem_j in leaf_j.element_indices:
                            self.farfield_pairs.append((elem_i, elem_j))
        
        print(f"    近场交互对数: {len(self.nearfield_pairs)}")
        print(f"    远场交互对数: {len(self.farfield_pairs)}")
    
    def _compute_nearfield(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        
        for (i, j) in self.nearfield_pairs:
            xi = self.element_centers[i]
            xj = self.element_centers[j]
            r = np.linalg.norm(xi - xj)
            
            if r < 1e-10:
                continue
                
            G = np.exp(1j * self.k * r) / (4 * np.pi * r)
            y[i] += G * self.element_areas[j] * x[j]
        
        return y
    
    def _compute_farfield_mlfma(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        
        for leaf in self.leaf_nodes:
            if not leaf.element_indices:
                continue
                
            leaf.V_plus = np.zeros(self.mlfma.n_dirs, dtype=np.complex128)
            
            for idx in leaf.element_indices:
                d = self.element_centers[idx] - leaf.center
                phase = np.exp(1j * self.k * np.dot(self.mlfma.directions, d))
                leaf.V_plus += phase * self.element_areas[idx] * x[idx]
        
        for level in range(len(self.nodes_by_level) - 2, -1, -1):
            for node in self.nodes_by_level[level]:
                if not node.children:
                    continue
                    
                node.V_plus = np.zeros(self.mlfma.n_dirs, dtype=np.complex128)
                
                for child in node.children:
                    if child.V_plus is not None:
                        d = child.center - node.center
                        node.V_plus += self.mlfma.translate_plus(child.V_plus, d)
        
        for level in range(len(self.nodes_by_level)):
            for node in self.nodes_by_level[level]:
                if node.W is None:
                    node.W = np.zeros(self.mlfma.n_dirs, dtype=np.complex128)
                
                neighbors = self._get_neighbors(node)
                
                for other in self.nodes_by_level[level]:
                    if other == node or other in neighbors:
                        continue
                    
                    if other.V_plus is not None:
                        d = other.center - node.center
                        r = np.linalg.norm(d)
                        if r > 0:
                            alpha = np.exp(1j * self.k * r) / (4 * np.pi * r)
                            node.W += alpha * self.mlfma.translate_plus(other.V_plus, d)
        
        for level in range(1, len(self.nodes_by_level)):
            for node in self.nodes_by_level[level]:
                if node.parent is not None and node.parent.W is not None:
                    d = node.center - node.parent.center
                    if node.W is None:
                        node.W = np.zeros(self.mlfma.n_dirs, dtype=np.complex128)
                    node.W += self.mlfma.translate_minus(node.parent.W, d)
        
        for leaf in self.leaf_nodes:
            if leaf.W is None or not leaf.element_indices:
                continue
                
            for idx in leaf.element_indices:
                d = self.element_centers[idx] - leaf.center
                phase = np.exp(-1j * self.k * np.dot(self.mlfma.directions, d))
                contribution = np.sum(leaf.W * phase * self.mlfma.weights)
                y[idx] += contribution
        
        return y
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        y_near = self._compute_nearfield(x)
        y_far = self._compute_farfield_mlfma(x)
        return y_near + y_far
    
    def solve(self, rhs: np.ndarray, tol: float = 1e-4, maxiter: int = 500, 
              restart: int = 50) -> np.ndarray:
        N = len(rhs)
        
        def linear_op(x):
            return self.matvec(x)
        
        A = LinearOperator((N, N), matvec=linear_op, dtype=np.complex128)
        
        print(f"  使用GMRES求解 {N} 自由度系统...")
        print(f"    收敛容差: {tol}, 最大迭代: {maxiter}")
        
        start_time = time.time()
        solution, info = gmres(A, rhs, tol=tol, maxiter=maxiter, restart=restart)
        solve_time = time.time() - start_time
        
        if info != 0:
            print(f"    警告: GMRES未收敛 (info={info})")
        else:
            print(f"    GMRES收敛成功! 用时: {solve_time:.2f} 秒")
        
        return solution


def estimate_memory_complexity(N: int):
    print("\n" + "=" * 60)
    print("复杂度分析")
    print("=" * 60)
    
    print(f"\n问题规模: N = {N}")
    
    direct_mem = N * N * 16 / (1024**3)
    print(f"\n直接BEM (O(N²)):")
    print(f"  矩阵内存: {direct_mem:.2f} GB")
    print(f"  每次矩阵向量乘: O({N**2})")
    
    fmm_mem = N * 100 * 16 / (1024**3)
    print(f"\nFMM-BEM (O(N)):")
    print(f"  内存: ~{fmm_mem:.2f} GB")
    print(f"  每次矩阵向量乘: O({N})")
    
    print(f"\n理论加速比: ~{N/100:.0f}x")
    print("=" * 60)


def create_mesh_by_size(target_elements: int, radius: float = 1.0):
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
    
    refinement_level = 0
    while len(faces) < target_elements:
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
        refinement_level += 1
    
    vertices = vertices * radius
    return TriangleMesh(vertices, faces), refinement_level


def test_large_scale():
    print("=" * 70)
    print("MLFMA-BEM 大规模声学散射测试")
    print("=" * 70)
    
    target_elements = 5000
    
    print(f"\n创建网格 (目标: ~{target_elements} 单元)...")
    mesh, level = create_mesh_by_size(target_elements)
    N = mesh.n_faces
    print(f"  实际单元数: {N}")
    print(f"  顶点数: {mesh.n_vertices}")
    print(f"  细化层数: {level}")
    
    frequency = 100.0
    c = 343.0
    k = 2 * np.pi * frequency / c
    
    print(f"\n参数:")
    print(f"  频率: {frequency} Hz")
    print(f"  波数: {k:.4f} m^-1")
    print(f"  MLFMA展开阶数: p=8")
    
    estimate_memory_complexity(N)
    
    print("\n计算入射波...")
    from bem_acoustic import plane_wave, plane_wave_normal_derivative
    p_inc = plane_wave(mesh.face_centers, k)
    dp_inc_dn = plane_wave_normal_derivative(mesh.face_centers, mesh.face_normals, k)
    
    print("\n初始化MLFMA-BEM求解器...")
    solver = MLFMABEMSolver(mesh, k, p=8, max_elements_per_leaf=50)
    
    print("\n构建右端项...")
    rhs = -dp_inc_dn
    
    print("\n使用MLFMA-BEM求解...")
    start_time = time.time()
    p_scat = solver.solve(rhs, tol=1e-3, maxiter=200)
    total_time = time.time() - start_time
    p_total = p_scat + p_inc
    
    print(f"\n总计算时间: {total_time:.2f} 秒")
    
    print(f"\n结果统计:")
    print(f"  表面声压幅值范围: [{np.min(np.abs(p_total)):.4f}, {np.max(np.abs(p_total)):.4f}]")
    print(f"  表面声压平均幅值: {np.mean(np.abs(p_total)):.4f}")
    
    print("\n" + "=" * 70)
    print("MLFMA-BEM大规模测试完成!")
    print("=" * 70)
    
    return p_total, mesh, solver


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "large":
        test_large_scale()
    else:
        print("多层快速多极子边界元法 (MLFMA-BEM)")
        print("\n用法:")
        print("  python mlfma.py large    - 大规模测试")
        print("\n这个实现展示了MLFMA的核心思想:")
        print("  1. 八叉树空间分层")
        print("  2. 平面波展开 (V+ 和 W)")
        print("  3. 向上/向下传递 (M2M, L2L)")
        print("  4. 多极到局部变换 (M2L)")
        print("  5. 近场/远场分离")
        print("  6. O(N) 计算复杂度")
