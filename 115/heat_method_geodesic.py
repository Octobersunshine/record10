import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TriangleMesh:
    """三角网格数据结构"""
    
    def __init__(self, vertices, faces):
        """
        参数:
            vertices: (N, 3) 顶点坐标数组
            faces: (M, 3) 三角面顶点索引数组
        """
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int64)
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        
        self._precompute_geometry()
    
    def _precompute_geometry(self):
        """预计算几何量：面积、法向量、半边等"""
        self.face_areas = np.zeros(self.n_faces)
        self.face_normals = np.zeros((self.n_faces, 3))
        self.face_centers = np.zeros((self.n_faces, 3))
        
        for f in range(self.n_faces):
            i, j, k = self.faces[f]
            v0 = self.vertices[i]
            v1 = self.vertices[j]
            v2 = self.vertices[k]
            
            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(cross)
            
            self.face_areas[f] = area
            self.face_normals[f] = cross / (2 * area) if area > 0 else np.zeros(3)
            self.face_centers[f] = (v0 + v1 + v2) / 3
        
        self._build_laplacian()
    
    def _build_laplacian(self):
        """构建余切拉普拉斯矩阵"""
        cot_edges = lil_matrix((self.n_vertices, self.n_vertices))
        mass = lil_matrix((self.n_vertices, self.n_vertices))
        
        for f in range(self.n_faces):
            i, j, k = self.faces[f]
            v0 = self.vertices[i]
            v1 = self.vertices[j]
            v2 = self.vertices[k]
            
            e1 = v1 - v0
            e2 = v2 - v0
            e3 = v2 - v1
            
            l1 = np.dot(e1, e1)
            l2 = np.dot(e2, e2)
            l3 = np.dot(e3, e3)
            
            area = self.face_areas[f]
            
            if area > 1e-12:
                cot_alpha = (l2 + l1 - l3) / (4 * area)
                cot_beta = (l3 + l2 - l1) / (4 * area)
                cot_gamma = (l1 + l3 - l2) / (4 * area)
                
                cot_edges[j, k] += cot_alpha
                cot_edges[k, j] += cot_alpha
                cot_edges[i, k] += cot_beta
                cot_edges[k, i] += cot_beta
                cot_edges[i, j] += cot_gamma
                cot_edges[j, i] += cot_gamma
            
            vertex_area = area / 3
            mass[i, i] += vertex_area
            mass[j, j] += vertex_area
            mass[k, k] += vertex_area
        
        L = lil_matrix((self.n_vertices, self.n_vertices))
        for i in range(self.n_vertices):
            L[i, i] = np.sum(cot_edges[i, :].data) if len(cot_edges[i, :].data) > 0 else 0
            for j in cot_edges[i, :].rows[0] if hasattr(cot_edges[i, :], 'rows') else []:
                if j != i:
                    L[i, j] = -cot_edges[i, j]
        
        self.L = csr_matrix(L)
        self.M = csr_matrix(mass)
        self.M_inv = csr_matrix(np.diag(1.0 / self.M.diagonal()))
    
    def vertex_one_ring(self, v_idx):
        """获取顶点的一环邻域"""
        neighbors = set()
        for f in range(self.n_faces):
            if v_idx in self.faces[f]:
                for v in self.faces[f]:
                    if v != v_idx:
                        neighbors.add(v)
        return list(neighbors)
    
    def get_vertex_mass(self, v_idx):
        """获取顶点的质量（面积）"""
        return self.M[v_idx, v_idx]


def generate_sphere_mesh(radius=1.0, n_lat=30, n_lon=60):
    """生成球面三角网格"""
    vertices = []
    faces = []
    
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    for i in range(n_lat):
        for j in range(n_lon):
            i0 = i * n_lon + j
            i1 = i * n_lon + (j + 1) % n_lon
            i2 = (i + 1) * n_lon + j
            i3 = (i + 1) * n_lon + (j + 1) % n_lon
            
            if i > 0:
                faces.append([i0, i2, i1])
            if i < n_lat - 1:
                faces.append([i1, i2, i3])
    
    return TriangleMesh(vertices, faces)


def generate_torus_mesh(R=2.0, r=1.0, n_u=50, n_v=30):
    """生成环面三角网格"""
    vertices = []
    faces = []
    
    for i in range(n_u):
        u = 2 * np.pi * i / n_u
        for j in range(n_v):
            v = 2 * np.pi * j / n_v
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    for i in range(n_u):
        for j in range(n_v):
            i0 = i * n_v + j
            i1 = i * n_v + (j + 1) % n_v
            i2 = ((i + 1) % n_u) * n_v + j
            i3 = ((i + 1) % n_u) * n_v + (j + 1) % n_v
            
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
    
    return TriangleMesh(vertices, faces)


def generate_cylinder_mesh(radius=1.0, height=3.0, n_theta=40, n_z=20):
    """生成圆柱面三角网格"""
    vertices = []
    faces = []
    
    for i in range(n_z):
        z = -height / 2 + height * i / (n_z - 1)
        for j in range(n_theta):
            theta = 2 * np.pi * j / n_theta
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    for i in range(n_z - 1):
        for j in range(n_theta):
            i0 = i * n_theta + j
            i1 = i * n_theta + (j + 1) % n_theta
            i2 = (i + 1) * n_theta + j
            i3 = (i + 1) * n_theta + (j + 1) % n_theta
            
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
    
    return TriangleMesh(vertices, faces)


class HeatMethodGeodesic:
    """
    基于热方法的测地线距离场计算
    参考: Crane et al. "Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow"
    """
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.n = mesh.n_vertices
        
        h = np.sqrt(mesh.face_areas.mean())
        self.t = h * h
        
        self._build_matrices()
    
    def _build_matrices(self):
        """构建热扩散和泊松求解矩阵"""
        L = self.mesh.L
        M = self.mesh.M
        
        self.A_heat = M + self.t * L
        self.A_poisson = L
    
    def compute_distance_field(self, source_indices):
        """
        计算从源点出发的测地线距离场
        
        参数:
            source_indices: 源点索引列表
            
        返回:
            distances: 各顶点到源点的测地距离
        """
        n = self.n
        
        u0 = np.zeros(n)
        for idx in source_indices:
            u0[idx] = 1.0 / self.mesh.get_vertex_mass(idx)
        
        u = spsolve(self.A_heat, u0)
        
        grad_u = self._compute_gradient(u)
        
        for f in range(self.mesh.n_faces):
            norm = np.linalg.norm(grad_u[f])
            if norm > 1e-12:
                grad_u[f] = -grad_u[f] / norm
            else:
                grad_u[f] = np.zeros(3)
        
        div_X = self._compute_divergence(grad_u)
        
        dist = spsolve(self.A_poisson, div_X)
        
        min_dist = np.min(dist[idx] for idx in source_indices)
        dist = dist - min_dist
        dist = np.maximum(dist, 0)
        
        return dist
    
    def _compute_gradient(self, u):
        """计算标量场在每个三角面上的梯度"""
        grad_u = np.zeros((self.mesh.n_faces, 3))
        
        for f in range(self.mesh.n_faces):
            i, j, k = self.mesh.faces[f]
            v0 = self.mesh.vertices[i]
            v1 = self.mesh.vertices[j]
            v2 = self.mesh.vertices[k]
            n = self.mesh.face_normals[f]
            area = self.mesh.face_areas[f]
            
            if area < 1e-12:
                continue
            
            e1 = v2 - v1
            e2 = v0 - v2
            e3 = v1 - v0
            
            grad = (u[i] * np.cross(n, e1) + u[j] * np.cross(n, e2) + u[k] * np.cross(n, e3)) / (2 * area)
            grad_u[f] = grad
        
        return grad_u
    
    def _compute_divergence(self, X):
        """计算向量场的散度"""
        div_X = np.zeros(self.n)
        
        for f in range(self.mesh.n_faces):
            i, j, k = self.mesh.faces[f]
            v0 = self.mesh.vertices[i]
            v1 = self.mesh.vertices[j]
            v2 = self.mesh.vertices[k]
            n = self.mesh.face_normals[f]
            area = self.mesh.face_areas[f]
            
            if area < 1e-12:
                continue
            
            e1 = v2 - v1
            e2 = v0 - v2
            e3 = v1 - v0
            
            l1_sq = np.dot(e1, e1)
            l2_sq = np.dot(e2, e2)
            l3_sq = np.dot(e3, e3)
            
            cot_alpha = (l2_sq + l3_sq - l1_sq) / (8 * area)
            cot_beta = (l1_sq + l3_sq - l2_sq) / (8 * area)
            cot_gamma = (l1_sq + l2_sq - l3_sq) / (8 * area)
            
            Xf = X[f]
            
            div_X[i] += cot_beta * np.dot(e2, Xf) + cot_gamma * np.dot(-e3, Xf)
            div_X[j] += cot_gamma * np.dot(e3, Xf) + cot_alpha * np.dot(-e1, Xf)
            div_X[k] += cot_alpha * np.dot(e1, Xf) + cot_beta * np.dot(-e2, Xf)
        
        return div_X
    
    def geodesic_distance(self, idx1, idx2):
        """计算两点间的测地距离"""
        dist_field = self.compute_distance_field([idx1])
        return dist_field[idx2]
    
    def shortest_path_indices(self, source_idx, target_idx, dist_field=None, n_steps=100):
        """
        使用梯度下降从目标点回溯到源点，得到近似最短路径
        
        参数:
            source_idx: 源点索引
            target_idx: 目标点索引
            dist_field: 预计算的距离场（可选）
            n_steps: 梯度下降步数
            
        返回:
            path_points: 路径上的点坐标
        """
        if dist_field is None:
            dist_field = self.compute_distance_field([source_idx])
        
        current = self.mesh.vertices[target_idx].copy()
        path = [current.copy()]
        
        for step in range(n_steps):
            f_idx = self._find_closest_face(current)
            if f_idx is None:
                break
            
            i, j, k = self.mesh.faces[f_idx]
            v0 = self.mesh.vertices[i]
            v1 = self.mesh.vertices[j]
            v2 = self.mesh.vertices[k]
            n = self.mesh.face_normals[f_idx]
            area = self.mesh.face_areas[f_idx]
            
            if area < 1e-12:
                break
            
            e1 = v2 - v1
            e2 = v0 - v2
            e3 = v1 - v0
            
            du0, du1, du2 = dist_field[i], dist_field[j], dist_field[k]
            grad = (du0 * np.cross(n, e1) + du1 * np.cross(n, e2) + du2 * np.cross(n, e3)) / (2 * area)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            direction = -grad / grad_norm
            
            h = np.sqrt(self.mesh.face_areas.mean())
            step_size = h * 0.1
            
            current = current + step_size * direction
            path.append(current.copy())
            
            current_dist = self._distance_to_vertex(current, source_idx)
            if current_dist < h:
                break
        
        return np.array(path)
    
    def _find_closest_face(self, point):
        """找到离给定点最近的三角面"""
        min_dist = float('inf')
        closest_face = None
        
        for f in range(self.mesh.n_faces):
            center = self.mesh.face_centers[f]
            dist = np.linalg.norm(point - center)
            if dist < min_dist:
                min_dist = dist
                closest_face = f
        
        return closest_face
    
    def _distance_to_vertex(self, point, vertex_idx):
        return np.linalg.norm(point - self.mesh.vertices[vertex_idx])


def visualize_distance_field(mesh, distance_field, title="测地线距离场"):
    """可视化距离场"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    tris = mesh.faces
    dist = distance_field
    
    max_dist = np.max(dist)
    colors = plt.cm.jet(dist / max_dist)
    
    poly = ax.plot_trisurf(
        mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
        triangles=tris,
        facecolors=colors,
        shade=False,
        alpha=0.9
    )
    
    mappable = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=max_dist))
    mappable.set_array(dist)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('测地距离', rotation=270, labelpad=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    return fig, ax


def visualize_shortest_path(mesh, dist_field, path, source_idx, target_idx, title="最短路径"):
    """可视化最短路径"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    tris = mesh.faces
    max_dist = np.max(dist_field)
    colors = plt.cm.jet(dist_field / max_dist)
    
    poly = ax.plot_trisurf(
        mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
        triangles=tris,
        facecolors=colors,
        shade=False,
        alpha=0.5
    )
    
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=3, label='最短路径')
    
    src = mesh.vertices[source_idx]
    tgt = mesh.vertices[target_idx]
    ax.scatter(src[0], src[1], src[2], color='green', s=200, label='源点', zorder=10)
    ax.scatter(tgt[0], tgt[1], tgt[2], color='blue', s=200, label='目标点', zorder=10)
    
    mappable = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=max_dist))
    mappable.set_array(dist_field)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('测地距离', rotation=270, labelpad=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    return fig, ax
