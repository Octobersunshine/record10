import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize


class MeshConformalMap:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int64)
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        
        self._build_adjacency()
        self._compute_edge_lengths()
        self._compute_cotangent_weights()
        
        self.uv = None
        self.map_type = None
    
    def _build_adjacency(self):
        self.adjacency = [[] for _ in range(self.n_vertices)]
        self.face_vertices = [[] for _ in range(self.n_vertices)]
        
        for fi, face in enumerate(self.faces):
            for i, vi in enumerate(face):
                vj = face[(i + 1) % 3]
                if vj not in self.adjacency[vi]:
                    self.adjacency[vi].append(vj)
                if vi not in self.adjacency[vj]:
                    self.adjacency[vj].append(vi)
                self.face_vertices[vi].append(fi)
    
    def _compute_edge_lengths(self):
        self.edge_lengths = {}
        for face in self.faces:
            for i in range(3):
                vi, vj = face[i], face[(i + 1) % 3]
                key = tuple(sorted((vi, vj)))
                if key not in self.edge_lengths:
                    self.edge_lengths[key] = np.linalg.norm(
                        self.vertices[vi] - self.vertices[vj]
                    )
    
    def _compute_cotangent_weights(self):
        self.cot_weights = lil_matrix((self.n_vertices, self.n_vertices))
        
        for face in self.faces:
            v0, v1, v2 = face
            
            e0 = self.vertices[v1] - self.vertices[v2]
            e1 = self.vertices[v2] - self.vertices[v0]
            e2 = self.vertices[v0] - self.vertices[v1]
            
            l0 = np.linalg.norm(e0)
            l1 = np.linalg.norm(e1)
            l2 = np.linalg.norm(e2)
            
            cos0 = np.dot(e1, e2) / (l1 * l2 + 1e-10)
            cos1 = np.dot(e2, e0) / (l2 * l0 + 1e-10)
            cos2 = np.dot(e0, e1) / (l0 * l1 + 1e-10)
            
            cos0 = np.clip(cos0, -0.999, 0.999)
            cos1 = np.clip(cos1, -0.999, 0.999)
            cos2 = np.clip(cos2, -0.999, 0.999)
            
            cot0 = cos0 / np.sqrt(1 - cos0 * cos0 + 1e-10)
            cot1 = cos1 / np.sqrt(1 - cos1 * cos1 + 1e-10)
            cot2 = cos2 / np.sqrt(1 - cos2 * cos2 + 1e-10)
            
            self.cot_weights[v1, v2] += 0.5 * cot0
            self.cot_weights[v2, v1] += 0.5 * cot0
            self.cot_weights[v2, v0] += 0.5 * cot1
            self.cot_weights[v0, v2] += 0.5 * cot1
            self.cot_weights[v0, v1] += 0.5 * cot2
            self.cot_weights[v1, v0] += 0.5 * cot2
        
        self.cot_weights = self.cot_weights.tocsr()
    
    def _find_boundary_vertices(self):
        boundary_edges = {}
        for face in self.faces:
            for i in range(3):
                vi, vj = face[i], face[(i + 1) % 3]
                key = tuple(sorted((vi, vj)))
                if key in boundary_edges:
                    boundary_edges[key] += 1
                else:
                    boundary_edges[key] = 1
        
        boundary_set = set()
        for (vi, vj), count in boundary_edges.items():
            if count == 1:
                boundary_set.add(vi)
                boundary_set.add(vj)
        
        boundary_list = []
        if boundary_set:
            start_v = next(iter(boundary_set))
            current = start_v
            visited = set()
            
            while current not in visited:
                visited.add(current)
                boundary_list.append(current)
                
                for neighbor in self.adjacency[current]:
                    edge_key = tuple(sorted((current, neighbor)))
                    if (neighbor in boundary_set and 
                        neighbor not in visited and 
                        boundary_edges.get(edge_key, 0) == 1):
                        current = neighbor
                        break
            
            if len(boundary_list) < len(boundary_set):
                boundary_list = list(boundary_set)
        
        return boundary_list
    
    def tutte_parameterization(self, boundary_shape='circle'):
        boundary = self._find_boundary_vertices()
        
        if not boundary:
            raise ValueError("Mesh has no boundary")
        
        n_boundary = len(boundary)
        boundary_uv = np.zeros((n_boundary, 2))
        
        if boundary_shape == 'circle':
            angles = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
            boundary_uv[:, 0] = np.cos(angles)
            boundary_uv[:, 1] = np.sin(angles)
        elif boundary_shape == 'square':
            n_side = (n_boundary + 3) // 4
            for i in range(n_boundary):
                if i < n_side:
                    t = i / n_side
                    boundary_uv[i] = [-1 + 2 * t, -1]
                elif i < 2 * n_side:
                    t = (i - n_side) / n_side
                    boundary_uv[i] = [1, -1 + 2 * t]
                elif i < 3 * n_side:
                    t = (i - 2 * n_side) / n_side
                    boundary_uv[i] = [1 - 2 * t, 1]
                else:
                    t = (i - 3 * n_side) / (n_boundary - 3 * n_side)
                    boundary_uv[i] = [-1, 1 - 2 * t]
        
        L = lil_matrix((self.n_vertices, self.n_vertices))
        b_u = np.zeros(self.n_vertices)
        b_v = np.zeros(self.n_vertices)
        
        for i in range(self.n_vertices):
            if i in boundary:
                L[i, i] = 1.0
                idx = boundary.index(i)
                b_u[i] = boundary_uv[idx, 0]
                b_v[i] = boundary_uv[idx, 1]
            else:
                neighbors = self.adjacency[i]
                n_neighbors = len(neighbors)
                L[i, i] = n_neighbors
                for j in neighbors:
                    L[i, j] = -1.0
        
        L = L.tocsr()
        u = spsolve(L, b_u)
        v = spsolve(L, b_v)
        
        self.uv = np.column_stack([u, v])
        self.map_type = 'tutte'
        
        return self.uv
    
    def harmonic_parameterization(self, boundary_shape='circle'):
        boundary = self._find_boundary_vertices()
        
        if not boundary:
            raise ValueError("Mesh has no boundary")
        
        boundary_set = set(boundary)
        n_boundary = len(boundary)
        boundary_uv = np.zeros((n_boundary, 2))
        
        if boundary_shape == 'circle':
            angles = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
            boundary_uv[:, 0] = np.cos(angles)
            boundary_uv[:, 1] = np.sin(angles)
        
        L = lil_matrix((self.n_vertices, self.n_vertices))
        b_u = np.zeros(self.n_vertices)
        b_v = np.zeros(self.n_vertices)
        
        for i in range(self.n_vertices):
            if i in boundary_set:
                L[i, i] = 1.0
                idx = boundary.index(i)
                b_u[i] = boundary_uv[idx, 0]
                b_v[i] = boundary_uv[idx, 1]
            else:
                total_weight = 0.0
                for j in self.adjacency[i]:
                    w = self.cot_weights[i, j]
                    L[i, j] = -w
                    total_weight += w
                L[i, i] = total_weight
        
        L = L.tocsr()
        u = spsolve(L, b_u)
        v = spsolve(L, b_v)
        
        self.uv = np.column_stack([u, v])
        self.map_type = 'harmonic'
        
        return self.uv
    
    def least_squares_conformal(self, free_boundary=True, max_iter=100, tol=1e-6):
        if self.uv is None:
            self.tutte_parameterization('circle')
        
        if not free_boundary:
            boundary = self._find_boundary_vertices()
            boundary_set = set(boundary)
            fixed_mask = np.array([i in boundary_set for i in range(self.n_vertices)])
        else:
            fixed_mask = np.zeros(self.n_vertices, dtype=bool)
            fixed_mask[0] = True
            fixed_mask[1] = True
        
        def conformal_energy(uv_flat):
            uv = uv_flat.reshape(-1, 2)
            energy = 0.0
            
            for face in self.faces:
                v0, v1, v2 = face
                
                e0 = self.vertices[v1] - self.vertices[v0]
                e1 = self.vertices[v2] - self.vertices[v0]
                n = np.cross(e0, e1)
                area = 0.5 * np.linalg.norm(n)
                
                u0, v0_uv = uv[v0]
                u1, v1_uv = uv[v1]
                u2, v2_uv = uv[v2]
                
                du0 = u1 - u0
                du1 = u2 - u0
                dv0 = v1_uv - v0_uv
                dv1 = v2_uv - v0_uv
                
                J = np.array([[du0, du1], [dv0, dv1]])
                
                if area > 1e-10:
                    E = (np.trace(J.T @ J) / (2 * area + 1e-10)) * area
                    energy += E
            
            return energy
        
        def conformal_gradient(uv_flat):
            uv = uv_flat.reshape(-1, 2)
            grad = np.zeros_like(uv)
            
            for face in self.faces:
                v0, v1, v2 = face
                
                e0 = self.vertices[v1] - self.vertices[v0]
                e1 = self.vertices[v2] - self.vertices[v0]
                n = np.cross(e0, e1)
                area = 0.5 * np.linalg.norm(n)
                
                u0, v0_uv = uv[v0]
                u1, v1_uv = uv[v1]
                u2, v2_uv = uv[v2]
                
                du0 = u1 - u0
                du1 = u2 - u0
                dv0 = v1_uv - v0_uv
                dv1 = v2_uv - v0_uv
                
                if area > 1e-10:
                    factor = 1.0 / (area + 1e-10)
                    
                    grad[v0, 0] += factor * (-du0 - du1)
                    grad[v1, 0] += factor * du0
                    grad[v2, 0] += factor * du1
                    
                    grad[v0, 1] += factor * (-dv0 - dv1)
                    grad[v1, 1] += factor * dv0
                    grad[v2, 1] += factor * dv1
            
            grad[fixed_mask] = 0
            return grad.flatten()
        
        uv_flat = self.uv.flatten().copy()
        
        result = minimize(
            conformal_energy,
            uv_flat,
            jac=conformal_gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol}
        )
        
        self.uv = result.x.reshape(-1, 2)
        self.map_type = 'lscm'
        
        return self.uv
    
    def conformal_energy(self):
        if self.uv is None:
            return None
        
        energy = 0.0
        for face in self.faces:
            v0, v1, v2 = face
            
            e0 = self.vertices[v1] - self.vertices[v0]
            e1 = self.vertices[v2] - self.vertices[v0]
            n = np.cross(e0, e1)
            area = 0.5 * np.linalg.norm(n)
            
            if area < 1e-10:
                continue
            
            u0, v0_uv = self.uv[v0]
            u1, v1_uv = self.uv[v1]
            u2, v2_uv = self.uv[v2]
            
            du0 = u1 - u0
            du1 = u2 - u0
            dv0 = v1_uv - v0_uv
            dv1 = v2_uv - v0_uv
            
            J = np.array([[du0, du1], [dv0, dv1]])
            
            det_J = np.linalg.det(J)
            trace_JtJ = np.trace(J.T @ J)
            
            if det_J < -1e-10:
                energy += 1e10
            else:
                energy += trace_JtJ / (2 * area + 1e-10) * area
        
        return energy
    
    def area_distortion(self):
        if self.uv is None:
            return None
        
        distortions = []
        for face in self.faces:
            v0, v1, v2 = face
            
            e0 = self.vertices[v1] - self.vertices[v0]
            e1 = self.vertices[v2] - self.vertices[v0]
            area_3d = 0.5 * np.linalg.norm(np.cross(e0, e1))
            
            if area_3d < 1e-10:
                continue
            
            u0, v0_uv = self.uv[v0]
            u1, v1_uv = self.uv[v1]
            u2, v2_uv = self.uv[v2]
            
            area_2d = 0.5 * np.abs((u1 - u0) * (v2_uv - v0_uv) - (u2 - u0) * (v1_uv - v0_uv))
            
            distortions.append(area_2d / (area_3d + 1e-10))
        
        return np.array(distortions)
    
    def angle_distortion(self):
        if self.uv is None:
            return None
        
        distortions = []
        for face in self.faces:
            v0, v1, v2 = face
            
            e0_3d = self.vertices[v1] - self.vertices[v0]
            e1_3d = self.vertices[v2] - self.vertices[v0]
            e2_3d = self.vertices[v0] - self.vertices[v1]
            
            e0_uv = self.uv[v1] - self.uv[v0]
            e1_uv = self.uv[v2] - self.uv[v0]
            
            def angle_between(v1, v2):
                dot = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norms < 1e-10:
                    return 0
                return np.arccos(np.clip(dot / norms, -1, 1))
            
            angle_3d = angle_between(e0_3d, e1_3d)
            angle_2d = angle_between(e0_uv, e1_uv)
            
            distortions.append(np.abs(angle_3d - angle_2d))
        
        return np.array(distortions)


class TextureMapper:
    def __init__(self, mesh_map):
        self.mesh_map = mesh_map
    
    def sample_texture(self, texture_image, uv=None):
        if uv is None:
            uv = self.mesh_map.uv
        
        if uv is None:
            raise ValueError("No UV coordinates available")
        
        h, w = texture_image.shape[:2]
        
        u = np.clip(uv[:, 0], -1, 1)
        v = np.clip(uv[:, 1], -1, 1)
        
        x = ((u + 1) / 2 * (w - 1)).astype(np.int64)
        y = ((v + 1) / 2 * (h - 1)).astype(np.int64)
        
        colors = texture_image[y, x]
        
        return colors
    
    def generate_checkerboard(self, size=256, square_size=16):
        texture = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if (i // square_size + j // square_size) % 2 == 0:
                    texture[i, j] = [255, 255, 255]
                else:
                    texture[i, j] = [100, 100, 100]
        return texture
    
    def generate_uv_colors(self):
        if self.mesh_map.uv is None:
            raise ValueError("No UV coordinates available")
        
        uv = self.mesh_map.uv
        colors = np.zeros((len(uv), 3), dtype=np.uint8)
        
        colors[:, 0] = ((uv[:, 0] + 1) / 2 * 255).astype(np.uint8)
        colors[:, 1] = ((uv[:, 1] + 1) / 2 * 255).astype(np.uint8)
        colors[:, 2] = 128
        
        return colors


def create_disk_mesh(n_radial=20, n_angular=60):
    vertices = []
    faces = []
    
    vertices.append([0, 0, 0])
    
    for i in range(n_radial):
        r = (i + 1) / n_radial
        for j in range(n_angular):
            theta = 2 * np.pi * j / n_angular
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append([x, y, 0])
    
    for j in range(n_angular):
        j_next = (j + 1) % n_angular
        faces.append([0, j + 1, j_next + 1])
    
    for i in range(n_radial - 1):
        inner_start = i * n_angular + 1
        outer_start = (i + 1) * n_angular + 1
        
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            
            v0 = inner_start + j
            v1 = inner_start + j_next
            v2 = outer_start + j
            v3 = outer_start + j_next
            
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    
    return np.array(vertices), np.array(faces)


def create_irregular_shape_mesh(shape_func, n_grid=30):
    x = np.linspace(-1, 1, n_grid)
    y = np.linspace(-1, 1, n_grid)
    X, Y = np.meshgrid(x, y)
    
    vertices = []
    vertex_indices = {}
    idx = 0
    
    for i in range(n_grid):
        for j in range(n_grid):
            if shape_func(X[i, j], Y[i, j]):
                vertices.append([X[i, j], Y[i, j], 0])
                vertex_indices[(i, j)] = idx
                idx += 1
    
    faces = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            if (shape_func(X[i, j], Y[i, j]) and
                shape_func(X[i+1, j], Y[i+1, j]) and
                shape_func(X[i, j+1], Y[i, j+1]) and
                shape_func(X[i+1, j+1], Y[i+1, j+1])):
                
                v0 = vertex_indices[(i, j)]
                v1 = vertex_indices[(i+1, j)]
                v2 = vertex_indices[(i, j+1)]
                v3 = vertex_indices[(i+1, j+1)]
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
    
    return np.array(vertices), np.array(faces)


def example_tutte_parameterization():
    print("=" * 60)
    print("Example 1: Tutte Parameterization (Fixed Boundary)")
    print("=" * 60)
    
    vertices, faces = create_disk_mesh(n_radial=10, n_angular=30)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    mapper = MeshConformalMap(vertices, faces)
    
    uv_circle = mapper.tutte_parameterization(boundary_shape='circle')
    print(f"\nTutte (circle boundary):")
    print(f"  UV range: [{uv_circle.min():.3f}, {uv_circle.max():.3f}]")
    
    uv_square = mapper.tutte_parameterization(boundary_shape='square')
    print(f"\nTutte (square boundary):")
    print(f"  UV range: [{uv_square.min():.3f}, {uv_square.max():.3f}]")
    
    area_dist = mapper.area_distortion()
    angle_dist = mapper.angle_distortion()
    print(f"\nDistortion:")
    print(f"  Area distortion mean: {area_dist.mean():.4f}")
    print(f"  Angle distortion mean: {np.degrees(angle_dist.mean()):.2f} deg")
    
    return mapper


def example_harmonic_parameterization():
    print("\n" + "=" * 60)
    print("Example 2: Harmonic Parameterization (Cotangent Weights)")
    print("=" * 60)
    
    vertices, faces = create_disk_mesh(n_radial=10, n_angular=30)
    
    mapper = MeshConformalMap(vertices, faces)
    uv = mapper.harmonic_parameterization(boundary_shape='circle')
    
    print(f"Harmonic parameterization:")
    print(f"  UV range: [{uv.min():.3f}, {uv.max():.3f}]")
    
    area_dist = mapper.area_distortion()
    angle_dist = mapper.angle_distortion()
    print(f"\nDistortion:")
    print(f"  Area distortion mean: {area_dist.mean():.4f}")
    print(f"  Angle distortion mean: {np.degrees(angle_dist.mean()):.2f} deg")
    
    return mapper


def example_lscm():
    print("\n" + "=" * 60)
    print("Example 3: Least Squares Conformal Map (Free Boundary)")
    print("=" * 60)
    
    vertices, faces = create_disk_mesh(n_radial=8, n_angular=24)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    mapper = MeshConformalMap(vertices, faces)
    
    print("\nInitial Tutte parameterization...")
    mapper.tutte_parameterization()
    initial_energy = mapper.conformal_energy()
    print(f"  Initial conformal energy: {initial_energy:.4f}")
    
    print("\nOptimizing LSCM...")
    uv = mapper.least_squares_conformal(free_boundary=True, max_iter=50)
    final_energy = mapper.conformal_energy()
    print(f"  Final conformal energy: {final_energy:.4f}")
    print(f"  Energy reduction: {(1 - final_energy/initial_energy)*100:.1f}%")
    
    area_dist = mapper.area_distortion()
    angle_dist = mapper.angle_distortion()
    print(f"\nDistortion:")
    print(f"  Area distortion mean: {area_dist.mean():.4f}")
    print(f"  Angle distortion mean: {np.degrees(angle_dist.mean()):.2f} deg")
    
    return mapper


def example_irregular_shape():
    print("\n" + "=" * 60)
    print("Example 4: Irregular Shape Parameterization")
    print("=" * 60)
    
    def heart_shape(x, y):
        return (x**2 + y**2 - 1)**3 - x**2 * y**3 < 0
    
    print("Creating heart-shaped mesh...")
    vertices, faces = create_irregular_shape_mesh(heart_shape, n_grid=25)
    print(f"Heart mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    mapper = MeshConformalMap(vertices, faces)
    
    print("\nComputing harmonic parameterization...")
    uv = mapper.harmonic_parameterization(boundary_shape='circle')
    
    print(f"  UV range: [{uv.min():.3f}, {uv.max():.3f}]")
    
    area_dist = mapper.area_distortion()
    angle_dist = mapper.angle_distortion()
    print(f"\nDistortion:")
    print(f"  Area distortion mean: {area_dist.mean():.4f}")
    print(f"  Angle distortion mean: {np.degrees(angle_dist.mean()):.2f} deg")
    
    return mapper


def example_texture_mapping():
    print("\n" + "=" * 60)
    print("Example 5: Texture Mapping")
    print("=" * 60)
    
    vertices, faces = create_disk_mesh(n_radial=10, n_angular=30)
    
    mapper = MeshConformalMap(vertices, faces)
    mapper.tutte_parameterization(boundary_shape='square')
    
    tex_mapper = TextureMapper(mapper)
    
    print("Generating checkerboard texture...")
    checkerboard = tex_mapper.generate_checkerboard(size=256, square_size=16)
    
    print("Sampling texture colors...")
    colors = tex_mapper.sample_texture(checkerboard)
    print(f"  Colored {len(colors)} vertices")
    
    print("\nGenerating UV coordinate colors...")
    uv_colors = tex_mapper.generate_uv_colors()
    print(f"  UV color range: [{uv_colors.min()}, {uv_colors.max()}]")
    
    return tex_mapper


def demo():
    print("\n" + "=" * 60)
    print("Laplacian Conformal Mapping for Arbitrary Shapes")
    print("Free Boundary Methods for Texture Mapping")
    print("=" * 60)
    
    try:
        mapper1 = example_tutte_parameterization()
        mapper2 = example_harmonic_parameterization()
        mapper3 = example_lscm()
        mapper4 = example_irregular_shape()
        tex_mapper = example_texture_mapping()
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Technical Overview")
    print("=" * 60)
    print("""
Parameterization Methods:

1. Tutte Parameterization
   - Maps boundary to fixed shape (circle, square)
   - Interior points: barycentric coordinates
   - Guarantees bijectivity for disk topology
   - Simple but significant angle distortion

2. Harmonic Parameterization
   - Uses cotangent Laplacian weights
   - Minimizes Dirichlet energy
   - Better angle preservation than Tutte
   - Still uses fixed boundary

3. Least Squares Conformal Map (LSCM)
   - Minimizes conformal energy
   - Free boundary optimization
   - Best angle preservation
   - Uses L-BFGS-B optimization

Conformal Energy (LSCM):
    E = Σ_f (|∇u|² + |∇v|²) / (2 * area_f) * area_f
    
    where u, v are UV coordinates.

Distortion Metrics:
    - Area distortion: area_2d / area_3d
    - Angle distortion: |angle_3d - angle_2d|
""")
    
    print("\n" + "=" * 60)
    print("Usage Guide")
    print("=" * 60)
    print("""
Basic Usage:
    from laplacian_conformal import MeshConformalMap, create_disk_mesh
    
    # Create or load mesh
    vertices, faces = create_disk_mesh(n_radial=10, n_angular=30)
    
    # Create mapper
    mapper = MeshConformalMap(vertices, faces)
    
    # Tutte parameterization (fast)
    uv_tutte = mapper.tutte_parameterization(boundary_shape='circle')
    
    # Harmonic parameterization (better quality)
    uv_harmonic = mapper.harmonic_parameterization()
    
    # LSCM (best quality, free boundary)
    uv_lscm = mapper.least_squares_conformal(free_boundary=True)
    
    # Analyze distortion
    area_dist = mapper.area_distortion()
    angle_dist = mapper.angle_distortion()
    energy = mapper.conformal_energy()

Texture Mapping:
    from laplacian_conformal import TextureMapper
    
    tex_mapper = TextureMapper(mapper)
    
    # Generate checkerboard texture
    texture = tex_mapper.generate_checkerboard(size=256)
    
    # Sample texture colors
    colors = tex_mapper.sample_texture(texture)
    
    # Or generate UV coordinate colors
    uv_colors = tex_mapper.generate_uv_colors()
""")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo()
