import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


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
            elements: 单元连接信息，形状为(n_elements, 3)，每个元素是三个节点的索引
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
    
    def get_boundary_edges(self) -> List[Tuple[int, int]]:
        """
        获取边界边
        """
        if self.elements is None:
            raise ValueError("请先生成网格")
            
        edge_count = {}
        for element in self.elements:
            edges = [
                tuple(sorted([element[0], element[1]])),
                tuple(sorted([element[1], element[2]])),
                tuple(sorted([element[2], element[0]]))
            ]
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
                
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return boundary_edges
    
    def output_mesh(self, filename: Optional[str] = None):
        """
        输出网格信息
        
        参数:
            filename: 如果提供，将信息写入文件
        """
        output = []
        output.append("=" * 60)
        output.append("Delaunay三角网格信息")
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
        output.append("")
        
        boundary_edges = self.get_boundary_edges()
        output.append("-" * 60)
        output.append(f"边界边数: {len(boundary_edges)}")
        output.append("-" * 60)
        output.append(f"{'边编号':>10} {'节点1':>10} {'节点2':>10}")
        for i, (n1, n2) in enumerate(boundary_edges):
            output.append(f"{i:>10} {n1:>10} {n2:>10}")
        
        output_text = "\n".join(output)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"网格信息已写入: {filename}")
        else:
            print(output_text)
            
    def plot_mesh(self, show_point_numbers: bool = True, show_element_numbers: bool = False):
        """
        可视化网格
        """
        if self.nodes is None or self.elements is None:
            raise ValueError("请先生成网格")
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.triplot(self.nodes[:, 0], self.nodes[:, 1], self.elements, 'b-', lw=0.8)
        ax.plot(self.boundary_points[:, 0], self.boundary_points[:, 1], 'ro-', lw=2, markersize=6, label='边界')
        
        internal_indices = np.arange(len(self.boundary_points), len(self.nodes))
        if len(internal_indices) > 0:
            ax.plot(self.nodes[internal_indices, 0], self.nodes[internal_indices, 1], 'go', markersize=4, label='内部点')
            
        if show_point_numbers:
            for i, (x, y) in enumerate(self.nodes):
                ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='bottom')
                
        if show_element_numbers:
            for i, element in enumerate(self.elements):
                centroid = np.mean(self.nodes[element], axis=0)
                ax.annotate(f'E{i}', centroid, fontsize=7, color='red', ha='center', va='center')
                
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Delaunay三角剖分网格')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def generate_polygon_boundary(polygon_type: str = 'square', **kwargs) -> np.ndarray:
    """
    生成各种多边形边界
    
    参数:
        polygon_type: 多边形类型
            'square' - 正方形
            'rectangle' - 矩形 (需要width, height)
            'circle' - 圆形 (需要radius, num_points)
            'triangle' - 三角形
            'hexagon' - 六边形
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
    print("Delaunay三角剖分网格生成器")
    print("=" * 50)
    
    boundary = generate_polygon_boundary('square', size=1.0)
    
    print(f"\n边界形状: 正方形")
    print(f"边界点数: {len(boundary)}")
    
    np.random.seed(42)
    mesh = DelaunayMesh(boundary, num_internal_points=20)
    
    print("\n正在生成网格...")
    nodes, elements = mesh.generate_mesh()
    mesh.filter_elements()
    
    print(f"生成完成!")
    print(f"节点数: {len(nodes)}")
    print(f"单元数: {len(elements)}")
    
    mesh.output_mesh("mesh_output.txt")
    
    print("\n正在显示网格...")
    mesh.plot_mesh(show_point_numbers=True, show_element_numbers=False)


if __name__ == "__main__":
    main()
