import math
import random


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def cross(self, other):
        return self.x * other.y - self.y * other.x
    
    def length_squared(self):
        return self.x * self.x + self.y * self.y
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
    
    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10)))
    
    def __repr__(self):
        return f"Point({self.x:.4f}, {self.y:.4f})"


class Triangle:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.vertices = [a, b, c]
        self._calculate_circumcircle()
    
    def _calculate_circumcircle(self):
        a, b, c = self.vertices
        
        d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
        
        if abs(d) < 1e-10:
            self.center = None
            self.radius_squared = float('inf')
            return
        
        ux = ((a.length_squared() * (b.y - c.y) + 
               b.length_squared() * (c.y - a.y) + 
               c.length_squared() * (a.y - b.y)) / d)
        uy = ((a.length_squared() * (c.x - b.x) + 
               b.length_squared() * (a.x - c.x) + 
               c.length_squared() * (b.x - a.x)) / d)
        
        self.center = Point(ux, uy)
        self.radius_squared = (a - self.center).length_squared()
    
    def contains_point_in_circumcircle(self, point):
        if self.center is None:
            return False
        return (point - self.center).length_squared() < self.radius_squared - 1e-10
    
    def has_vertex(self, point):
        return point in self.vertices
    
    def get_edges(self):
        return [
            (self.a, self.b),
            (self.b, self.c),
            (self.c, self.a)
        ]
    
    def __repr__(self):
        return f"Triangle({self.a}, {self.b}, {self.c})"


def edge_equal(e1, e2):
    return (e1[0] == e2[0] and e1[1] == e2[1]) or \
           (e1[0] == e2[1] and e1[1] == e2[0])


class DelaunayTriangulation:
    def __init__(self, points):
        self.points = [Point(p[0], p[1]) for p in points]
        self.triangles = []
    
    def triangulate(self):
        if len(self.points) < 3:
            return []
        
        min_x = min(p.x for p in self.points)
        max_x = max(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        max_y = max(p.y for p in self.points)
        
        dx = max_x - min_x
        dy = max_y - min_y
        delta_max = max(dx, dy)
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        
        p1 = Point(mid_x - 20 * delta_max, mid_y - delta_max)
        p2 = Point(mid_x, mid_y + 20 * delta_max)
        p3 = Point(mid_x + 20 * delta_max, mid_y - delta_max)
        
        super_triangle = Triangle(p1, p2, p3)
        self.triangles = [super_triangle]
        
        for point in self.points:
            bad_triangles = []
            for triangle in self.triangles:
                if triangle.contains_point_in_circumcircle(point):
                    bad_triangles.append(triangle)
            
            polygon = []
            for triangle in bad_triangles:
                for edge in triangle.get_edges():
                    is_shared = False
                    for other in bad_triangles:
                        if other is triangle:
                            continue
                        for other_edge in other.get_edges():
                            if edge_equal(edge, other_edge):
                                is_shared = True
                                break
                        if is_shared:
                            break
                    if not is_shared:
                        polygon.append(edge)
            
            for triangle in bad_triangles:
                self.triangles.remove(triangle)
            
            for edge in polygon:
                new_triangle = Triangle(edge[0], edge[1], point)
                self.triangles.append(new_triangle)
        
        final_triangles = []
        for triangle in self.triangles:
            has_super = (triangle.has_vertex(p1) or 
                        triangle.has_vertex(p2) or 
                        triangle.has_vertex(p3))
            if not has_super:
                final_triangles.append(triangle)
        
        self.triangles = final_triangles
        return self.triangles
    
    def get_mesh_data(self):
        node_map = {point: i for i, point in enumerate(self.points)}
        
        nodes = [(p.x, p.y) for p in self.points]
        
        elements = []
        for triangle in self.triangles:
            indices = [node_map[v] for v in triangle.vertices]
            elements.append(indices)
        
        return nodes, elements


def point_in_polygon(point, polygon):
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


def generate_internal_points(boundary, num_points):
    polygon_points = [(p.x, p.y) if isinstance(p, Point) else p for p in boundary]
    
    min_x = min(p[0] for p in polygon_points)
    max_x = max(p[0] for p in polygon_points)
    min_y = min(p[1] for p in polygon_points)
    max_y = max(p[1] for p in polygon_points)
    
    internal_points = []
    random.seed(42)
    
    for _ in range(num_points * 100):
        if len(internal_points) >= num_points:
            break
        
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        
        if point_in_polygon((x, y), polygon_points):
            internal_points.append((x, y))
    
    return internal_points


def generate_square_boundary(size=1.0):
    return [
        (0, 0),
        (size, 0),
        (size, size),
        (0, size)
    ]


def generate_circle_boundary(radius=1.0, num_points=20):
    boundary = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        boundary.append((x, y))
    return boundary


def filter_triangles_by_boundary(triangles, boundary):
    boundary_points = [(p.x, p.y) if isinstance(p, Point) else p for p in boundary]
    
    filtered = []
    for triangle in triangles:
        centroid_x = sum(v.x for v in triangle.vertices) / 3
        centroid_y = sum(v.y for v in triangle.vertices) / 3
        
        if point_in_polygon((centroid_x, centroid_y), boundary_points):
            filtered.append(triangle)
    
    return filtered


def output_mesh(nodes, elements, filename=None):
    output = []
    output.append("=" * 60)
    output.append("Delaunay三角剖分 - 纯Python实现")
    output.append("=" * 60)
    output.append(f"节点总数: {len(nodes)}")
    output.append(f"单元总数: {len(elements)}")
    output.append("")
    
    output.append("-" * 60)
    output.append("节点坐标:")
    output.append("-" * 60)
    output.append(f"{'节点编号':>10} {'x坐标':>15} {'y坐标':>15}")
    for i, node in enumerate(nodes):
        output.append(f"{i:>10} {node[0]:>15.6f} {node[1]:>15.6f}")
    output.append("")
    
    output.append("-" * 60)
    output.append("单元连接信息:")
    output.append("-" * 60)
    output.append(f"{'单元编号':>10} {'节点1':>10} {'节点2':>10} {'节点3':>10}")
    for i, element in enumerate(elements):
        output.append(f"{i:>10} {element[0]:>10} {element[1]:>10} {element[2]:>10}")
    
    output_text = "\n".join(output)
    
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"网格信息已写入: {filename}")
    else:
        print(output_text)


def main():
    print("Delaunay三角剖分 - 纯Python实现 (Bowyer-Watson算法)")
    print("=" * 60)
    
    boundary = generate_square_boundary(size=1.0)
    print(f"\n边界形状: 正方形 ({len(boundary)}个边界点)")
    
    num_internal = 10
    internal_points = generate_internal_points(boundary, num_internal)
    print(f"生成内部点: {len(internal_points)}个")
    
    all_points = boundary + internal_points
    print(f"总点数: {len(all_points)}")
    
    print("\n开始三角剖分...")
    dt = DelaunayTriangulation(all_points)
    triangles = dt.triangulate()
    
    boundary_points = [Point(p[0], p[1]) for p in boundary]
    triangles = filter_triangles_by_boundary(triangles, boundary_points)
    
    print(f"生成三角形数量: {len(triangles)}")
    
    nodes, elements = dt.get_mesh_data()
    
    output_mesh(nodes, elements, "delaunay_output.txt")
    
    print("\n计算完成!")
    print("\n三角形单元:")
    for i, tri in enumerate(triangles):
        print(f"  单元 {i}: {tri}")


if __name__ == "__main__":
    main()
