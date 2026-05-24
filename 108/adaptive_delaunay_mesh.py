import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Set, Callable
from collections import defaultdict


class FEMSolver:
    """
    简单的Poisson方程有限元求解器
    """
    def __init__(self, nodes: np.ndarray, elements: np.ndarray, 
                 boundary_nodes: Set[int]):
        self.nodes = nodes
        self.elements = elements
        self.boundary_nodes = boundary_nodes
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)
        
    def compute_element_stiffness(self, element: np.ndarray) -> np.ndarray:
        """
        计算单元刚度矩阵（线性三角形单元）
        """
        pts = self.nodes[element]
        
        area = 0.5 * abs(
            (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) -
            (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0])
        )
        
        B = np.zeros((2, 3))
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            B[0, i] = (pts[j, 1] - pts[k, 1]) / (2 * area)
            B[1, i] = (pts[k, 0] - pts[j, 0]) / (2 * area)
        
        Ke = area * (B.T @ B)
        return Ke
    
    def assemble_stiffness(self) -> csr_matrix:
        """
        组装整体刚度矩阵
        """
        K = lil_matrix((self.n_nodes, self.n_nodes))
        
        for element in self.elements:
            Ke = self.compute_element_stiffness(element)
            for i in range(3):
                for j in range(3):
                    K[element[i], element[j]] += Ke[i, j]
        
        return K.tocsr()
    
    def apply_dirichlet_bc(self, K: csr_matrix, F: np.ndarray, 
                           bc_func: Callable) -> Tuple[csr_matrix, np.ndarray]:
        """
        应用Dirichlet边界条件
        """
        for node in self.boundary_nodes:
            x, y = self.nodes[node]
            bc_value = bc_func(x, y)
            
            F -= K[:, node] * bc_value
            F[node] = bc_value
            
            K[node, :] = 0
            K[:, node] = 0
            K[node, node] = 1
            
        return K, F
    
    def solve(self, rhs_func: Callable, bc_func: Callable) -> np.ndarray:
        """
        求解Poisson方程
        
        参数:
            rhs_func: 右端项函数 f(x, y)
            bc_func: 边界条件函数 g(x, y)
            
        返回:
            节点解向量
        """
        K = self.assemble_stiffness()
        
        F = np.zeros(self.n_nodes)
        for element in self.elements:
            pts = self.nodes[element]
            area = 0.5 * abs(
                (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) -
                (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0])
            )
            
            centroid = np.mean(pts, axis=0)
            f_val = rhs_func(centroid[0], centroid[1])
            
            for node in element:
                F[node] += f_val * area / 3
        
        K, F = self.apply_dirichlet_bc(K, F, bc_func)
        
        u = spsolve(K, F)
        
        return u
    
    def compute_element_gradients(self, u: np.ndarray) -> np.ndarray:
        """
        计算每个单元的常数梯度
        """
        gradients = np.zeros((self.n_elements, 2))
        
        for e_idx, element in enumerate(self.elements):
            pts = self.nodes[element]
            u_e = u[element]
            
            area = 0.5 * abs(
                (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) -
                (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0])
            )
            
            dudx = 0
            dudy = 0
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                dudx += u_e[i] * (pts[j, 1] - pts[k, 1])
                dudy += u_e[i] * (pts[k, 0] - pts[j, 0])
            
            dudx /= (2 * area)
            dudy /= (2 * area)
            
            gradients[e_idx] = [dudx, dudy]
        
        return gradients


class ZZErrorEstimator:
    """
    Zienkiewicz-Zhu梯度恢复法后验误差估计器
    """
    def __init__(self, nodes: np.ndarray, elements: np.ndarray, 
                 boundary_nodes: Set[int]):
        self.nodes = nodes
        self.elements = elements
        self.boundary_nodes = boundary_nodes
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)
        
    def _build_node_element_map(self) -> Dict[int, List[int]]:
        """
        建立节点到单元的映射
        """
        node_to_elems = defaultdict(list)
        for e_idx, element in enumerate(self.elements):
            for node in element:
                node_to_elems[node].append(e_idx)
        return node_to_elems
    
    def recover_gradients(self, element_gradients: np.ndarray) -> np.ndarray:
        """
        使用超单元片恢复法（SPR）恢复节点梯度
        
        参数:
            element_gradients: 单元常数梯度 (n_elements, 2)
            
        返回:
            恢复后的节点梯度 (n_nodes, 2)
        """
        node_to_elems = self._build_node_element_map()
        recovered_gradients = np.zeros((self.n_nodes, 2))
        
        for node_idx in range(self.n_nodes):
            adjacent_elems = node_to_elems[node_idx]
            
            if len(adjacent_elems) == 0:
                continue
                
            if node_idx in self.boundary_nodes:
                patch_gradients = element_gradients[adjacent_elems]
                recovered_gradients[node_idx] = np.mean(patch_gradients, axis=0)
            else:
                patch_gradients = element_gradients[adjacent_elems]
                recovered_gradients[node_idx] = np.mean(patch_gradients, axis=0)
        
        return recovered_gradients
    
    def estimate_error(self, u: np.ndarray, element_gradients: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        估计每个单元的误差
        
        参数:
            u: 节点解向量
            element_gradients: 单元常数梯度
            
        返回:
            element_errors: 每个单元的误差指示子
            total_error: 总误差
        """
        recovered_gradients = self.recover_gradients(element_gradients)
        
        element_errors = np.zeros(self.n_elements)
        
        for e_idx, element in enumerate(self.elements):
            pts = self.nodes[element]
            
            area = 0.5 * abs(
                (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) -
                (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0])
            )
            
            elem_grad = element_gradients[e_idx]
            node_grads = recovered_gradients[element]
            
            error_sq = 0
            for i in range(3):
                diff = node_grads[i] - elem_grad
                error_sq += np.sum(diff ** 2)
            
            element_errors[e_idx] = np.sqrt(area * error_sq / 3)
        
        total_error = np.sqrt(np.sum(element_errors ** 2))
        
        return element_errors, total_error


class MeshRefiner:
    """
    网格细化器 - 实现红细化（1分4）
    """
    def __init__(self, nodes: np.ndarray, elements: np.ndarray,
                 boundary_nodes: Set[int], boundary_points: np.ndarray):
        self.nodes = nodes.copy()
        self.elements = elements.copy()
        self.boundary_nodes = boundary_nodes.copy()
        self.boundary_points = boundary_points
        self.edge_to_node = {}
    
    def _get_edge_key(self, n1: int, n2: int) -> Tuple[int, int]:
        """
        获取边的键（排序后的节点对）
        """
        return tuple(sorted([n1, n2]))
    
    def _get_edge_midpoint(self, n1: int, n2: int) -> int:
        """
        获取边中点节点，如果不存在则创建
        """
        edge_key = self._get_edge_key(n1, n2)
        
        if edge_key in self.edge_to_node:
            return self.edge_to_node[edge_key]
        
        midpoint = (self.nodes[n1] + self.nodes[n2]) / 2
        new_node_idx = len(self.nodes)
        self.nodes = np.vstack([self.nodes, midpoint])
        
        if n1 in self.boundary_nodes and n2 in self.boundary_nodes:
            self.boundary_nodes.add(new_node_idx)
        
        self.edge_to_node[edge_key] = new_node_idx
        
        return new_node_idx
    
    def refine_element(self, element: np.ndarray) -> List[np.ndarray]:
        """
        细化一个单元（红细化：1分4）
        
        返回:
            新的4个单元
        """
        n0, n1, n2 = element
        
        m01 = self._get_edge_midpoint(n0, n1)
        m12 = self._get_edge_midpoint(n1, n2)
        m20 = self._get_edge_midpoint(n2, n0)
        
        new_elements = [
            np.array([n0, m01, m20]),
            np.array([n1, m12, m01]),
            np.array([n2, m20, m12]),
            np.array([m01, m12, m20])
        ]
        
        return new_elements
    
    def refine_mesh(self, elements_to_refine: List[int]) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
        """
        细化指定的单元
        
        参数:
            elements_to_refine: 需要细化的单元索引列表
            
        返回:
            new_nodes: 新的节点数组
            new_elements: 新的单元数组
            new_boundary_nodes: 新的边界节点集合
        """
        self.edge_to_node = {}
        refine_set = set(elements_to_refine)
        
        new_elements = []
        
        for e_idx, element in enumerate(self.elements):
            if e_idx in refine_set:
                refined = self.refine_element(element)
                new_elements.extend(refined)
            else:
                new_elements.append(element.copy())
        
        new_elements = np.array(new_elements)
        
        return self.nodes.copy(), new_elements, self.boundary_nodes.copy()


class AdaptiveDelaunayMesh:
    """
    自适应Delaunay网格生成器
    """
    def __init__(self, boundary_points: np.ndarray, num_internal_points: int = 20):
        self.boundary_points = boundary_points
        self.initial_internal_points = num_internal_points
        self.nodes = None
        self.elements = None
        self.boundary_nodes = None
        
    def point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        判断点是否在多边形内部
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def generate_initial_mesh(self) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
        """
        生成初始Delaunay网格
        """
        min_x, min_y = np.min(self.boundary_points, axis=0)
        max_x, max_y = np.max(self.boundary_points, axis=0)
        
        internal_points = []
        np.random.seed(42)
        attempts = 0
        
        while len(internal_points) < self.initial_internal_points and attempts < 10000:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = np.array([x, y])
            
            if self.point_in_polygon(point, self.boundary_points):
                internal_points.append(point)
            attempts += 1
        
        internal_points = np.array(internal_points)
        
        if len(internal_points) > 0:
            all_points = np.vstack([self.boundary_points, internal_points])
        else:
            all_points = self.boundary_points
        
        tri = Delaunay(all_points)
        
        valid_elements = []
        for element in tri.simplices:
            element_nodes = all_points[element]
            centroid = np.mean(element_nodes, axis=0)
            
            if self.point_in_polygon(centroid, self.boundary_points):
                valid_elements.append(element)
        
        self.nodes = all_points
        self.elements = np.array(valid_elements)
        self.boundary_nodes = set(range(len(self.boundary_points)))
        
        return self.nodes, self.elements, self.boundary_nodes
    
    def adaptive_refine(self, rhs_func: Callable, bc_func: Callable,
                        target_error: float = 0.01, max_refinements: int = 5,
                        refine_ratio: float = 0.3) -> Dict:
        """
        执行自适应网格细化
        
        参数:
            rhs_func: 右端项 f(x, y)
            bc_func: 边界条件 g(x, y)
            target_error: 目标误差
            max_refinements: 最大细化次数
            refine_ratio: 每次细化比例（误差最大的前30%单元）
            
        返回:
            自适应历史统计
        """
        history = {
            'n_nodes': [],
            'n_elements': [],
            'errors': []
        }
        
        if self.nodes is None:
            self.generate_initial_mesh()
        
        for refinement_step in range(max_refinements):
            print(f"\n{'='*60}")
            print(f"自适应步骤 {refinement_step + 1}/{max_refinements}")
            print(f"{'='*60}")
            
            print(f"网格规模: {len(self.nodes)} 节点, {len(self.elements)} 单元")
            
            solver = FEMSolver(self.nodes, self.elements, self.boundary_nodes)
            print("求解有限元问题...")
            u = solver.solve(rhs_func, bc_func)
            
            element_gradients = solver.compute_element_gradients(u)
            
            estimator = ZZErrorEstimator(self.nodes, self.elements, self.boundary_nodes)
            element_errors, total_error = estimator.estimate_error(u, element_gradients)
            
            print(f"估计误差: {total_error:.6e}")
            
            history['n_nodes'].append(len(self.nodes))
            history['n_elements'].append(len(self.elements))
            history['errors'].append(total_error)
            
            if total_error <= target_error:
                print(f"\n✓ 达到目标误差!")
                break
            
            n_refine = max(1, int(len(self.elements) * refine_ratio))
            refine_indices = np.argsort(element_errors)[-n_refine:]
            
            print(f"细化误差最大的 {len(refine_indices)} 个单元...")
            
            refiner = MeshRefiner(self.nodes, self.elements, 
                                  self.boundary_nodes, self.boundary_points)
            self.nodes, self.elements, self.boundary_nodes = \
                refiner.refine_mesh(refine_indices)
        
        print(f"\n{'='*60}")
        print("自适应完成")
        print(f"{'='*60}")
        print(f"最终网格: {len(self.nodes)} 节点, {len(self.elements)} 单元")
        
        return history
    
    def solve_and_return(self, rhs_func: Callable, bc_func: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        求解并返回解
        """
        solver = FEMSolver(self.nodes, self.elements, self.boundary_nodes)
        u = solver.solve(rhs_func, bc_func)
        return u, self.nodes, self.elements
    
    def plot_mesh(self, u: Optional[np.ndarray] = None, 
                  title: str = "自适应网格", ax=None):
        """
        可视化网格和解
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        if u is not None:
            trip = ax.tripcolor(self.nodes[:, 0], self.nodes[:, 1], 
                                self.elements, u, shading='gouraud', cmap='viridis')
            plt.colorbar(trip, ax=ax, label='u(x,y)')
        else:
            ax.triplot(self.nodes[:, 0], self.nodes[:, 1], 
                       self.elements, 'b-', lw=0.5)
        
        boundary_list = list(self.boundary_nodes)
        ax.plot(self.nodes[boundary_list, 0], self.nodes[boundary_list, 1], 
                'ro', markersize=3, label='边界节点')
        
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"{title}\n({len(self.nodes)} 节点, {len(self.elements)} 单元)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def output_mesh(self, filename: str):
        """
        输出网格信息到文件
        """
        output = []
        output.append("=" * 60)
        output.append("自适应Delaunay网格信息")
        output.append("=" * 60)
        output.append(f"节点总数: {len(self.nodes)}")
        output.append(f"边界节点数: {len(self.boundary_nodes)}")
        output.append(f"内部节点数: {len(self.nodes) - len(self.boundary_nodes)}")
        output.append(f"单元总数: {len(self.elements)}")
        output.append("")
        
        output.append("-" * 60)
        output.append("节点坐标:")
        output.append("-" * 60)
        output.append(f"{'节点编号':>10} {'x坐标':>15} {'y坐标':>15}")
        for i, node in enumerate(self.nodes):
            output.append(f"{i:>10} {node[0]:>15.6f} {node[1]:>15.6f}")
        output.append("")
        
        output.append("-" * 60)
        output.append("单元连接信息:")
        output.append("-" * 60)
        output.append(f"{'单元编号':>10} {'节点1':>10} {'节点2':>10} {'节点3':>10}")
        for i, element in enumerate(self.elements):
            output.append(f"{i:>10} {element[0]:>10} {element[1]:>10} {element[2]:>10}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))
        print(f"网格信息已写入: {filename}")


def generate_polygon_boundary(polygon_type: str = 'square', **kwargs) -> np.ndarray:
    """
    生成各种多边形边界
    """
    if polygon_type == 'square':
        size = kwargs.get('size', 1.0)
        return np.array([
            [0, 0],
            [size, 0],
            [size, size],
            [0, size]
        ])
    elif polygon_type == 'rectangle':
        width = kwargs.get('width', 2.0)
        height = kwargs.get('height', 1.0)
        return np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
    elif polygon_type == 'circle':
        radius = kwargs.get('radius', 1.0)
        num_points = kwargs.get('num_points', 20)
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.column_stack([x, y])
    elif polygon_type == 'triangle':
        size = kwargs.get('size', 1.0)
        return np.array([
            [0, 0],
            [size, 0],
            [size / 2, size * np.sqrt(3) / 2]
        ])
    else:
        raise ValueError(f"不支持的多边形类型: {polygon_type}")


def example_square_with_singularity():
    """
    示例：正方形区域，解在中心附近有剧烈变化
    """
    print("自适应网格细化示例 - Poisson方程求解")
    print("解: u(x,y) = sin(πx)sin(πy)")
    
    boundary = generate_polygon_boundary('square', size=1.0)
    
    def rhs(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def bc(x, y):
        return 0.0
    
    mesh = AdaptiveDelaunayMesh(boundary, num_internal_points=10)
    
    history = mesh.adaptive_refine(
        rhs_func=rhs,
        bc_func=bc,
        target_error=0.001,
        max_refinements=4,
        refine_ratio=0.3
    )
    
    u, nodes, elements = mesh.solve_and_return(rhs, bc)
    mesh.output_mesh("adaptive_mesh_final.txt")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mesh.plot_mesh(u=None, title="最终网格", ax=axes[0])
    mesh.plot_mesh(u=u, title="解 u(x,y)", ax=axes[1])
    plt.tight_layout()
    plt.savefig("adaptive_result.png", dpi=150, bbox_inches='tight')
    print("结果图已保存: adaptive_result.png")
    plt.show()
    
    print("\n自适应历史:")
    for i, (n, e, err) in enumerate(zip(history['n_nodes'], history['n_elements'], history['errors'])):
        print(f"  步骤 {i+1}: {n} 节点, {e} 单元, 误差 = {err:.6e}")


def example_l_shaped_domain():
    """
    示例：L形区域，在凹角处有奇异性
    """
    print("\nL形区域自适应示例 - 在凹角处自动加密")
    
    boundary = np.array([
        [0, 0],
        [2, 0],
        [2, 2],
        [1, 2],
        [1, 1],
        [0, 1]
    ])
    
    def rhs(x, y):
        return 1.0
    
    def bc(x, y):
        return 0.0
    
    mesh = AdaptiveDelaunayMesh(boundary, num_internal_points=15)
    
    history = mesh.adaptive_refine(
        rhs_func=rhs,
        bc_func=bc,
        target_error=0.005,
        max_refinements=5,
        refine_ratio=0.25
    )
    
    u, nodes, elements = mesh.solve_and_return(rhs, bc)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mesh.plot_mesh(u=None, title="L形区域自适应网格", ax=axes[0])
    mesh.plot_mesh(u=u, title="解 u(x,y)", ax=axes[1])
    plt.tight_layout()
    plt.savefig("l_shape_adaptive.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    example_square_with_singularity()
