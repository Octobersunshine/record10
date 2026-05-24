import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict


class MeshQualityOptimizer:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray, boundary_nodes: np.ndarray):
        """
        网格质量优化器
        
        参数:
            nodes: 节点坐标 (n_nodes, 2)
            elements: 单元连接 (n_elements, 3)
            boundary_nodes: 边界节点索引
        """
        self.nodes = nodes.copy()
        self.elements = elements.copy()
        self.boundary_nodes = set(boundary_nodes)
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)
        
    def calculate_angles(self, element: np.ndarray) -> np.ndarray:
        """
        计算三角形的三个内角（弧度）
        """
        pts = self.nodes[element]
        angles = np.zeros(3)
        
        for i in range(3):
            p1 = pts[i]
            p2 = pts[(i + 1) % 3]
            p3 = pts[(i + 2) % 3]
            
            v1 = p2 - p1
            v2 = p3 - p1
            
            dot = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product < 1e-10:
                angles[i] = 0
            else:
                cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
                angles[i] = np.arccos(cos_angle)
        
        return angles
    
    def get_element_quality_stats(self) -> Dict:
        """
        获取网格质量统计信息
        """
        min_angles = []
        max_angles = []
        aspect_ratios = []
        
        for element in self.elements:
            angles = self.calculate_angles(element)
            min_angles.append(np.min(angles))
            max_angles.append(np.max(angles))
            
            pts = self.nodes[element]
            edges = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[0] - pts[2])
            ]
            max_edge = max(edges)
            min_edge = min(edges)
            if min_edge > 1e-10:
                aspect_ratios.append(max_edge / min_edge)
        
        min_angles_deg = np.degrees(min_angles)
        max_angles_deg = np.degrees(max_angles)
        
        return {
            'min_angle_min': np.min(min_angles_deg),
            'min_angle_max': np.max(min_angles_deg),
            'min_angle_mean': np.mean(min_angles_deg),
            'max_angle_min': np.min(max_angles_deg),
            'max_angle_max': np.max(max_angles_deg),
            'max_angle_mean': np.mean(max_angles_deg),
            'aspect_ratio_max': max(aspect_ratios) if aspect_ratios else 0,
            'aspect_ratio_mean': np.mean(aspect_ratios) if aspect_ratios else 0,
            'n_bad_large': np.sum(max_angles_deg > 140),
            'n_bad_small': np.sum(min_angles_deg < 20),
            'n_elements': len(self.elements)
        }
    
    def print_quality_stats(self, title: str = "网格质量统计"):
        """
        打印网格质量统计
        """
        stats = self.get_element_quality_stats()
        print("\n" + "=" * 60)
        print(f"{title}")
        print("=" * 60)
        print(f"单元总数: {stats['n_elements']}")
        print(f"最小角范围: {stats['min_angle_min']:.2f}° ~ {stats['min_angle_max']:.2f}° (平均: {stats['min_angle_mean']:.2f}°)")
        print(f"最大角范围: {stats['max_angle_min']:.2f}° ~ {stats['max_angle_max']:.2f}° (平均: {stats['max_angle_mean']:.2f}°)")
        print(f"长宽比最大: {stats['aspect_ratio_max']:.2f} (平均: {stats['aspect_ratio_mean']:.2f})")
        print(f"最大角>140°的单元数: {stats['n_bad_large']} ({stats['n_bad_large']/stats['n_elements']*100:.1f}%)")
        print(f"最小角<20°的单元数: {stats['n_bad_small']} ({stats['n_bad_small']/stats['n_elements']*100:.1f}%)")
        print("=" * 60 + "\n")
    
    def build_edge_to_elements_map(self) -> Dict[Tuple[int, int], List[int]]:
        """
        建立边到单元的映射
        """
        edge_map = defaultdict(list)
        for idx, element in enumerate(self.elements):
            edges = [
                tuple(sorted([element[0], element[1]])),
                tuple(sorted([element[1], element[2]])),
                tuple(sorted([element[2], element[0]]))
            ]
            for edge in edges:
                edge_map[edge].append(idx)
        return edge_map
    
    def find_shared_edge(self, elem1: np.ndarray, elem2: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        找出两个单元的公共边
        """
        set1 = set(elem1)
        set2 = set(elem2)
        common = set1.intersection(set2)
        if len(common) == 2:
            return tuple(sorted(common))
        return None
    
    def is_edge_on_boundary(self, edge: Tuple[int, int]) -> bool:
        """
        判断边是否在边界上
        """
        return (edge[0] in self.boundary_nodes) and (edge[1] in self.boundary_nodes)
    
    def flip_edge(self, elem1_idx: int, elem2_idx: int, edge: Tuple[int, int]) -> bool:
        """
        执行边翻转操作
        
        返回: 是否成功翻转
        """
        elem1 = self.elements[elem1_idx]
        elem2 = self.elements[elem2_idx]
        
        node_a, node_b = edge
        node_c = None
        node_d = None
        
        for node in elem1:
            if node != node_a and node != node_b:
                node_c = node
                break
        
        for node in elem2:
            if node != node_a and node != node_b:
                node_d = node
                break
        
        if node_c is None or node_d is None:
            return False
        
        new_elem1 = np.array([node_a, node_c, node_d])
        new_elem2 = np.array([node_b, node_c, node_d])
        
        if not self._check_valid_triangle(new_elem1) or not self._check_valid_triangle(new_elem2):
            return False
        
        orig_min_angle1 = np.min(self.calculate_angles(elem1))
        orig_min_angle2 = np.min(self.calculate_angles(elem2))
        new_min_angle1 = np.min(self.calculate_angles(new_elem1))
        new_min_angle2 = np.min(self.calculate_angles(new_elem2))
        
        if min(new_min_angle1, new_min_angle2) > min(orig_min_angle1, orig_min_angle2) + 1e-8:
            self.elements[elem1_idx] = new_elem1
            self.elements[elem2_idx] = new_elem2
            return True
        
        return False
    
    def _check_valid_triangle(self, element: np.ndarray) -> bool:
        """
        检查三角形是否有效（非退化）
        """
        pts = self.nodes[element]
        area = 0.5 * abs(
            (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) -
            (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0])
        )
        return area > 1e-12
    
    def edge_flip_optimization(self, max_iterations: int = 10) -> int:
        """
        执行边翻转优化
        
        参数:
            max_iterations: 最大迭代次数
            
        返回:
            成功翻转的边数
        """
        total_flips = 0
        
        for iteration in range(max_iterations):
            edge_map = self.build_edge_to_elements_map()
            flips_this_iter = 0
            
            edges_to_check = list(edge_map.keys())
            
            for edge in edges_to_check:
                elements_sharing = edge_map[edge]
                
                if len(elements_sharing) != 2:
                    continue
                
                if self.is_edge_on_boundary(edge):
                    continue
                
                elem1_idx, elem2_idx = elements_sharing
                
                angles1 = self.calculate_angles(self.elements[elem1_idx])
                angles2 = self.calculate_angles(self.elements[elem2_idx])
                
                if np.max(angles1) <= np.radians(140) and np.max(angles2) <= np.radians(140):
                    if np.min(angles1) >= np.radians(20) and np.min(angles2) >= np.radians(20):
                        continue
                
                if self.flip_edge(elem1_idx, elem2_idx, edge):
                    flips_this_iter += 1
                    total_flips += 1
            
            if flips_this_iter == 0:
                break
            
            print(f"  迭代 {iteration + 1}: 翻转 {flips_this_iter} 条边")
        
        return total_flips
    
    def build_node_neighbors(self) -> Dict[int, Set[int]]:
        """
        建立节点邻居映射
        """
        neighbors = defaultdict(set)
        
        for element in self.elements:
            for i in range(3):
                node_i = element[i]
                for j in range(3):
                    if i != j:
                        neighbors[node_i].add(element[j])
        
        return neighbors
    
    def point_in_polygon(self, point: np.ndarray, boundary_points: np.ndarray) -> bool:
        """
        判断点是否在多边形内部（射线法）
        """
        x, y = point
        n = len(boundary_points)
        inside = False
        
        p1x, p1y = boundary_points[0]
        for i in range(n + 1):
            p2x, p2y = boundary_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def laplace_smooth(self, boundary_points: np.ndarray, 
                       max_iterations: int = 50, 
                       relaxation: float = 0.8,
                       tolerance: float = 1e-6) -> float:
        """
        执行Laplace平滑
        
        参数:
            boundary_points: 边界点坐标，用于约束
            max_iterations: 最大迭代次数
            relaxation: 松弛因子 (0, 1]
            tolerance: 收敛阈值
            
        返回:
            最大位移
        """
        neighbors = self.build_node_neighbors()
        max_displacement = float('inf')
        
        for iteration in range(max_iterations):
            new_nodes = self.nodes.copy()
            current_max_disp = 0
            
            for node_idx in range(self.n_nodes):
                if node_idx in self.boundary_nodes:
                    continue
                
                neighbor_nodes = list(neighbors[node_idx])
                
                if len(neighbor_nodes) == 0:
                    continue
                
                centroid = np.mean(self.nodes[neighbor_nodes], axis=0)
                
                if not self.point_in_polygon(centroid, boundary_points):
                    continue
                
                displacement = relaxation * (centroid - self.nodes[node_idx])
                new_position = self.nodes[node_idx] + displacement
                
                if not self.point_in_polygon(new_position, boundary_points):
                    continue
                
                new_nodes[node_idx] = new_position
                disp_mag = np.linalg.norm(displacement)
                current_max_disp = max(current_max_disp, disp_mag)
            
            max_displacement = current_max_disp
            self.nodes = new_nodes
            
            if max_displacement < tolerance:
                print(f"  Laplace平滑在 {iteration + 1} 次迭代后收敛")
                break
        
        return max_displacement
    
    def optimize_mesh(self, boundary_points: np.ndarray,
                      target_min_angle: float = 20.0,
                      max_outer_iterations: int = 10) -> Dict:
        """
        完整的网格优化流程：边翻转 + Laplace平滑
        
        参数:
            boundary_points: 边界点坐标
            target_min_angle: 目标最小角（度）
            max_outer_iterations: 最大外迭代次数
            
        返回:
            优化统计信息
        """
        print("开始网格质量优化...")
        self.print_quality_stats("优化前质量")
        
        stats = {
            'edge_flips': 0,
            'smooth_iterations': 0,
            'initial_min_angle': 0,
            'final_min_angle': 0
        }
        
        stats['initial_min_angle'] = self.get_element_quality_stats()['min_angle_min']
        
        for outer_iter in range(max_outer_iterations):
            print(f"\n=== 外迭代 {outer_iter + 1} ===")
            
            print("执行边翻转优化...")
            flips = self.edge_flip_optimization(max_iterations=5)
            stats['edge_flips'] += flips
            print(f"  累计翻转 {stats['edge_flips']} 条边")
            
            print("执行Laplace平滑...")
            max_disp = self.laplace_smooth(boundary_points, max_iterations=30)
            stats['smooth_iterations'] += 1
            print(f"  最大位移: {max_disp:.6e}")
            
            current_stats = self.get_element_quality_stats()
            
            if current_stats['min_angle_min'] >= target_min_angle and current_stats['n_bad_large'] == 0:
                print(f"\n✓ 达到目标质量要求！")
                break
        
        stats['final_min_angle'] = self.get_element_quality_stats()['min_angle_min']
        
        self.print_quality_stats("优化后质量")
        
        return stats
    
    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取优化后的网格
        """
        return self.nodes.copy(), self.elements.copy()


class DelaunayMesh:
    def __init__(self, boundary_points: np.ndarray, num_internal_points: int = 50):
        """
        初始化Delaunay网格生成器
        
        参数:
            boundary_points: 多边形边界点，形状为(n, 2)，按顺序排列
            num_internal_points: 内部点数量
        """
        self.boundary_points = boundary_points
        self.num_internal_points = num_internal_points
        self.nodes = None
        self.elements = None
        self.boundary_nodes = None
        self.optimizer = None
        
    def point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        判断点是否在多边形内部（射线法）
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
    
    def generate_internal_points(self) -> np.ndarray:
        """
        在多边形内部生成随机点
        """
        min_x, min_y = np.min(self.boundary_points, axis=0)
        max_x, max_y = np.max(self.boundary_points, axis=0)
        
        internal_points = []
        attempts = 0
        max_attempts = self.num_internal_points * 100
        
        while len(internal_points) < self.num_internal_points and attempts < max_attempts:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = np.array([x, y])
            
            if self.point_in_polygon(point, self.boundary_points):
                internal_points.append(point)
            attempts += 1
            
        return np.array(internal_points)
    
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成Delaunay三角网格
        
        返回:
            nodes: 节点坐标，形状为(n_nodes, 2)
            elements: 单元连接信息，形状为(n_elements, 3)
        """
        internal_points = self.generate_internal_points()
        
        if len(internal_points) > 0:
            all_points = np.vstack([self.boundary_points, internal_points])
        else:
            all_points = self.boundary_points
            
        tri = Delaunay(all_points)
        
        self.nodes = all_points
        self.elements = tri.simplices
        self.boundary_nodes = np.arange(len(self.boundary_points))
        
        return self.nodes, self.elements
    
    def filter_elements(self) -> np.ndarray:
        """
        过滤掉位于多边形外部的单元
        """
        if self.elements is None:
            raise ValueError("请先生成网格")
            
        valid_elements = []
        for element in self.elements:
            element_nodes = self.nodes[element]
            centroid = np.mean(element_nodes, axis=0)
            
            if self.point_in_polygon(centroid, self.boundary_points):
                valid_elements.append(element)
                
        self.elements = np.array(valid_elements)
        return self.elements
    
    def optimize_quality(self, target_min_angle: float = 20.0) -> MeshQualityOptimizer:
        """
        执行网格质量优化
        
        参数:
            target_min_angle: 目标最小角（度）
            
        返回:
            优化器实例
        """
        if self.nodes is None or self.elements is None:
            raise ValueError("请先生成网格")
            
        self.optimizer = MeshQualityOptimizer(
            self.nodes, self.elements, self.boundary_nodes
        )
        
        self.optimizer.optimize_mesh(
            self.boundary_points,
            target_min_angle=target_min_angle
        )
        
        self.nodes, self.elements = self.optimizer.get_mesh()
        
        return self.optimizer
    
    def output_mesh(self, filename: Optional[str] = None):
        """
        输出网格信息
        """
        output = []
        output.append("=" * 60)
        output.append("Delaunay三角网格信息 (已优化)")
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
        
        output_text = "\n".join(output)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"网格信息已写入: {filename}")
        else:
            print(output_text)
    
    def plot_mesh(self, title: str = 'Delaunay三角剖分网格', 
                  show_point_numbers: bool = False):
        """
        可视化网格
        """
        if self.nodes is None or self.elements is None:
            raise ValueError("请先生成网格")
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.triplot(self.nodes[:, 0], self.nodes[:, 1], self.elements, 'b-', lw=0.8)
        ax.plot(self.boundary_points[:, 0], self.boundary_points[:, 1], 
                'ro-', lw=2, markersize=6, label='边界')
        
        internal_indices = np.arange(len(self.boundary_points), len(self.nodes))
        if len(internal_indices) > 0:
            ax.plot(self.nodes[internal_indices, 0], self.nodes[internal_indices, 1], 
                    'go', markersize=4, label='内部点')
            
        if show_point_numbers:
            for i, (x, y) in enumerate(self.nodes):
                ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='bottom')
                
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


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
        
    elif polygon_type == 'hexagon':
        size = kwargs.get('size', 1.0)
        theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        x = size * np.cos(theta)
        y = size * np.sin(theta)
        return np.column_stack([x, y])
        
    else:
        raise ValueError(f"不支持的多边形类型: {polygon_type}")


def main():
    print("Delaunay三角剖分网格生成器 (含质量优化)")
    print("=" * 60)
    
    boundary = generate_polygon_boundary('square', size=1.0)
    
    print(f"\n边界形状: 正方形")
    print(f"边界点数: {len(boundary)}")
    
    np.random.seed(42)
    mesh = DelaunayMesh(boundary, num_internal_points=30)
    
    print("\n正在生成网格...")
    nodes, elements = mesh.generate_mesh()
    mesh.filter_elements()
    
    print(f"初始网格: {len(nodes)} 节点, {len(elements)} 单元")
    
    print("\n正在进行质量优化...")
    mesh.optimize_quality(target_min_angle=20.0)
    
    mesh.output_mesh("mesh_optimized.txt")
    
    print("\n网格生成和优化完成!")


if __name__ == "__main__":
    main()
