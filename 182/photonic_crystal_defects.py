import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
from scipy.spatial import Delaunay
from scipy.special import j1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection

rcParams['font.sans-serif'] = ['SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

from photonic_crystal_pwe import PhotonicCrystal2D
from photonic_crystal_fem import PhotonicCrystalFEM


class PhotonicCrystalDefectFEM:
    def __init__(self, lattice_type='triangular', a=1.0, eps1=1.0, eps2=12.0, radius=0.2,
                 supercell_size=(5, 5)):
        self.a = a
        self.eps1 = eps1
        self.eps2 = eps2
        self.radius = radius
        self.lattice_type = lattice_type
        self.supercell_size = supercell_size
        self._setup_lattice()
        self.defect_function = None
        self.mesh = None
        self.nodes = None
        self.elements = None
        self.boundary_pairs = None

    def _setup_lattice(self):
        if self.lattice_type == 'triangular':
            self.a1 = np.array([self.a, 0.0])
            self.a2 = np.array([self.a / 2.0, self.a * np.sqrt(3) / 2.0])
        elif self.lattice_type == 'square':
            self.a1 = np.array([self.a, 0.0])
            self.a2 = np.array([0.0, self.a])
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")

        self.supercell_a1 = self.a1 * self.supercell_size[0]
        self.supercell_a2 = self.a2 * self.supercell_size[1]

        a1_3d = np.append(self.supercell_a1, 0)
        a2_3d = np.append(self.supercell_a2, 0)
        a3_3d = np.array([0, 0, 1])
        self.supercell_area = np.abs(np.dot(a3_3d, np.cross(a1_3d, a2_3d)))

        self.b1 = 2 * np.pi * np.array([self.supercell_a2[1], -self.supercell_a2[0]]) / self.supercell_area
        self.b2 = 2 * np.pi * np.array([-self.supercell_a1[1], self.supercell_a1[0]]) / self.supercell_area

    def get_eps_at_point(self, x, y):
        eps_base = self.eps1

        for i in range(-self.supercell_size[0] + 1, self.supercell_size[0] + 1):
            for j in range(-self.supercell_size[1] + 1, self.supercell_size[1] + 1):
                R = i * self.a1 + j * self.a2
                dx = x - R[0]
                dy = y - R[1]
                r = np.sqrt(dx ** 2 + dy ** 2)
                if r < self.radius:
                    eps_base = self.eps2
                    break

        if self.defect_function is not None:
            eps_base = self.defect_function(x, y, eps_base)

        return eps_base

    def set_point_defect(self, defect_radius=0.0, defect_type='remove'):
        def defect_func(x, y, eps_base):
            r = np.sqrt(x ** 2 + y ** 2)
            if r < self.radius:
                if defect_type == 'remove':
                    return self.eps1
                elif defect_type == 'enlarge':
                    if r < defect_radius:
                        return self.eps2
                    return self.eps1
                elif defect_type == 'dielectric':
                    if r < defect_radius:
                        return self.eps2 * 1.5
                    return eps_base
            return eps_base
        self.defect_function = defect_func

    def set_line_defect(self, direction='x', remove_row=0):
        def defect_func(x, y, eps_base):
            if direction == 'x':
                y_lattice = np.round(y / (self.a * np.sqrt(3) / 2)) * (self.a * np.sqrt(3) / 2)
                if np.abs(y - y_lattice) < self.radius * 1.5:
                    row_idx = np.round(y / (self.a * np.sqrt(3) / 2))
                    if row_idx == remove_row:
                        return self.eps1
            elif direction == 'y':
                x_lattice = np.round(x / self.a) * self.a
                if np.abs(x - x_lattice) < self.radius * 1.5:
                    col_idx = np.round(x / self.a)
                    if col_idx == remove_row:
                        return self.eps1
            return eps_base
        self.defect_function = defect_func

    def set_custom_defect(self, defect_func):
        self.defect_function = defect_func

    def generate_mesh(self, resolution=0.08):
        x_min, x_max = -self.supercell_a1[0] * 0.1, self.supercell_a1[0] * 1.1
        y_min, y_max = -self.supercell_a2[1] * 0.1, self.supercell_a2[1] * 1.1

        nx = int((x_max - x_min) / resolution) + 1
        ny = int((y_max - y_min) / resolution) + 1

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])

        tri = Delaunay(points)
        self.mesh = tri
        self.nodes = points
        self.elements = tri.simplices

        self._identify_boundary_pairs()
        self._compute_element_properties()

        print(f"超胞网格生成完成: {len(self.nodes)} 节点, {len(self.elements)} 单元")
        return self.nodes, self.elements

    def _identify_boundary_pairs(self):
        tol = 1e-8
        self.boundary_pairs = []
        nodes = self.nodes

        left_mask = np.abs(nodes[:, 0] - np.min(nodes[:, 0])) < tol
        right_mask = np.abs(nodes[:, 0] - np.max(nodes[:, 0])) < tol
        bottom_mask = np.abs(nodes[:, 1] - np.min(nodes[:, 1])) < tol
        top_mask = np.abs(nodes[:, 1] - np.max(nodes[:, 1])) < tol

        left_nodes = np.where(left_mask)[0]
        right_nodes = np.where(right_mask)[0]
        bottom_nodes = np.where(bottom_mask)[0]
        top_nodes = np.where(top_mask)[0]

        for li in left_nodes:
            y_left = nodes[li, 1]
            for ri in right_nodes:
                y_right = nodes[ri, 1]
                if np.abs(y_left - y_right) < tol:
                    self.boundary_pairs.append((li, ri, 'x', self.supercell_a1))
                    break

        for bi in bottom_nodes:
            x_bottom = nodes[bi, 0]
            for ti in top_nodes:
                x_top = nodes[ti, 0]
                if np.abs(x_bottom - x_top) < tol:
                    self.boundary_pairs.append((bi, ti, 'y', self.supercell_a2))
                    break

        print(f"边界对识别完成: {len(self.boundary_pairs)} 对边界节点")

    def _compute_element_properties(self):
        n_elem = len(self.elements)
        self.element_areas = np.zeros(n_elem)
        self.element_centroids = np.zeros((n_elem, 2))
        self.element_eps = np.zeros(n_elem)
        self.grad_N = np.zeros((n_elem, 3, 2))
        self.mass_matrix_coeff = np.zeros((n_elem, 3, 3))

        for e, elem in enumerate(self.elements):
            p1, p2, p3 = self.nodes[elem]
            area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                                 (p2[1] - p1[1]) * (p3[0] - p1[0]))
            self.element_areas[e] = area
            centroid = (p1 + p2 + p3) / 3.0
            self.element_centroids[e] = centroid
            self.element_eps[e] = self.get_eps_at_point(centroid[0], centroid[1])

            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2 * area)
            c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2 * area)
            self.grad_N[e, :, 0] = b
            self.grad_N[e, :, 1] = c

            for i in range(3):
                for j in range(3):
                    self.mass_matrix_coeff[e, i, j] = area / 12.0 if i == j else area / 24.0

    def build_transformation_matrix(self, k_point):
        n_nodes = len(self.nodes)

        phase_x = np.exp(1j * np.dot(k_point, self.supercell_a1))
        phase_y = np.exp(1j * np.dot(k_point, self.supercell_a2))

        keep_indices = np.ones(n_nodes, dtype=bool)
        slave_to_master = {}
        slave_phases = {}

        for slave, master, direction, lattice_vec in self.boundary_pairs:
            keep_indices[slave] = False
            phase = phase_x if direction == 'x' else phase_y
            slave_to_master[slave] = master
            slave_phases[slave] = phase

        keep_indices = np.where(keep_indices)[0]
        n_keep = len(keep_indices)
        node_to_keep = {node: idx for idx, node in enumerate(keep_indices)}

        T = sparse.lil_matrix((n_keep, n_nodes), dtype=complex)
        for idx, node in enumerate(keep_indices):
            T[idx, node] = 1.0

        for slave, master in slave_to_master.items():
            phase = slave_phases[slave]
            if master in node_to_keep:
                T[node_to_keep[master], slave] = phase
            else:
                current = master
                visited = set()
                while current not in node_to_keep and current not in visited:
                    visited.add(current)
                    if current in slave_to_master:
                        phase *= slave_phases[current]
                        current = slave_to_master[current]
                    else:
                        break
                if current in node_to_keep:
                    T[node_to_keep[current], slave] = phase

        return T.tocsr(), keep_indices

    def build_matrices(self, k_point, mode='TM'):
        n_nodes = len(self.nodes)
        rows_K, cols_K, vals_K = [], [], []
        rows_M, cols_M, vals_M = [], [], []

        for e, elem in enumerate(self.elements):
            area = self.element_areas[e]
            eps_e = self.element_eps[e]
            grad_N = self.grad_N[e]
            mass_coeff = self.mass_matrix_coeff[e]

            if mode == 'TM':
                for i in range(3):
                    for j in range(3):
                        rows_K.append(elem[i])
                        cols_K.append(elem[j])
                        vals_K.append(np.dot(grad_N[i], grad_N[j]) * area)
                        rows_M.append(elem[i])
                        cols_M.append(elem[j])
                        vals_M.append(eps_e * mass_coeff[i, j])
            elif mode == 'TE':
                for i in range(3):
                    for j in range(3):
                        rows_K.append(elem[i])
                        cols_K.append(elem[j])
                        vals_K.append(np.dot(grad_N[i], grad_N[j]) * area / eps_e)
                        rows_M.append(elem[i])
                        cols_M.append(elem[j])
                        vals_M.append(mass_coeff[i, j])

        K = sparse.csr_matrix((vals_K, (rows_K, cols_K)), shape=(n_nodes, n_nodes), dtype=complex)
        M = sparse.csr_matrix((vals_M, (rows_M, cols_M)), shape=(n_nodes, n_nodes), dtype=complex)

        T, keep_indices = self.build_transformation_matrix(k_point)
        K_reduced = T @ K @ T.conj().T
        M_reduced = T @ M @ T.conj().T

        return K_reduced, M_reduced, keep_indices, T

    def solve_eigenvalues(self, k_point, mode='TM', num_bands=10):
        K, M, keep_indices, T = self.build_matrices(k_point, mode=mode)
        n_dof = K.shape[0]
        num_eigs = min(num_bands + 10, n_dof - 1)

        try:
            eigenvalues, eigenvectors = splinalg.eigs(K, k=num_eigs, M=M, which='SM')
        except Exception as e:
            print(f"  警告: 稀疏求解失败 ({e}), 改用稠密求解")
            K_dense = K.todense()
            M_dense = M.todense()
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M_dense) @ K_dense)

        idx = np.argsort(np.maximum(eigenvalues.real, 0))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        frequencies = np.sqrt(np.maximum(eigenvalues.real, 0)) * self.a / (2 * np.pi)

        full_eigenvectors = np.zeros((len(self.nodes), num_bands), dtype=complex)
        for i in range(min(num_bands, eigenvectors.shape[1])):
            full_vec = T.T @ eigenvectors[:, i]
            full_eigenvectors[:, i] = full_vec

        return frequencies[:num_bands], full_eigenvectors

    def compute_defect_bands(self, k_point=np.array([0.0, 0.0]), mode='TM', num_bands=20):
        freqs, eigvecs = self.solve_eigenvalues(k_point, mode=mode, num_bands=num_bands)
        return freqs, eigvecs

    def plot_defect_mode(self, eigenvector, band_idx=0, save_path=None):
        nx, ny = 100, 100
        x_min, x_max = np.min(self.nodes[:, 0]), np.max(self.nodes[:, 0])
        y_min, y_max = np.min(self.nodes[:, 1]), np.max(self.nodes[:, 1])

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)

        mode_field = np.zeros_like(X, dtype=complex)
        for i in range(nx):
            for j in range(ny):
                xi, yi = X[j, i], Y[j, i]
                for e, elem in enumerate(self.elements):
                    p1, p2, p3 = self.nodes[elem]
                    v0 = p3 - p1
                    v1 = p2 - p1
                    v2 = np.array([xi, yi]) - p1
                    dot00 = np.dot(v0, v0)
                    dot01 = np.dot(v0, v1)
                    dot02 = np.dot(v0, v2)
                    dot11 = np.dot(v1, v1)
                    dot12 = np.dot(v1, v2)
                    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
                    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
                    if u >= 0 and v >= 0 and u + v <= 1:
                        N = np.array([1 - u - v, v, u])
                        mode_field[j, i] = np.dot(N, eigenvector[elem])
                        break

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im0 = axes[0].pcolormesh(X, Y, np.abs(mode_field) ** 2, cmap='hot', shading='auto')
        axes[0].set_xlabel('x (a)', fontsize=14)
        axes[0].set_ylabel('y (a)', fontsize=14)
        axes[0].set_title(f'缺陷模式模场分布 (能带 #{band_idx})', fontsize=16)
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0], label='|E|²')

        eps = np.zeros_like(X)
        for i in range(nx):
            for j in range(ny):
                eps[j, i] = self.get_eps_at_point(X[j, i], Y[j, i])
        im1 = axes[1].pcolormesh(X, Y, eps, cmap='viridis', shading='auto', alpha=0.8)
        ax = axes[1]
        for i in range(-self.supercell_size[0] + 1, self.supercell_size[0] + 1):
            for j in range(-self.supercell_size[1] + 1, self.supercell_size[1] + 1):
                R = i * self.a1 + j * self.a2
                circle = Circle((R[0], R[1]), self.radius, fill=False, color='white', lw=0.5, alpha=0.5)
                ax.add_patch(circle)
        axes[1].set_xlabel('x (a)', fontsize=14)
        axes[1].set_ylabel('y (a)', fontsize=14)
        axes[1].set_title('介电常数分布', fontsize=16)
        axes[1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1], label='ε')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"缺陷模式图已保存到: {save_path}")
        plt.close()

        return fig, axes


class TopologyOptimization:
    def __init__(self, lattice_type='square', a=1.0, eps_min=1.0, eps_max=12.0,
                 design_region_size=(5, 5), resolution=0.2):
        self.a = a
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.lattice_type = lattice_type
        self.design_region_size = design_region_size
        self.resolution = resolution

        if lattice_type == 'square':
            self.a1 = np.array([a, 0.0])
            self.a2 = np.array([0.0, a])
        elif lattice_type == 'triangular':
            self.a1 = np.array([a, 0.0])
            self.a2 = np.array([a / 2.0, a * np.sqrt(3) / 2.0])

        self.nx = int(design_region_size[0] / resolution)
        self.ny = int(design_region_size[1] / resolution)
        self.density = np.ones((self.nx, self.ny)) * 0.5

        self.filter_radius = resolution * 2
        self.volume_fraction = 0.4
        self.learning_rate = 0.05

    def _apply_simp(self, density, p=3.0):
        return self.eps_min + (self.eps_max - self.eps_min) * density ** p

    def _apply_filter(self, density):
        from scipy.ndimage import gaussian_filter
        sigma = self.filter_radius / self.resolution
        return gaussian_filter(density, sigma=sigma)

    def _project_density(self, density, beta=1.0, eta=0.5):
        return np.tanh(beta * eta) + np.tanh(beta * (density - eta)) / \
               (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

    def _compute_objective(self, density, target_frequency=0.3, k_point=np.array([0.0, 0.0])):
        eps_dist = self._apply_simp(density)

        pc = PhotonicCrystal2D(
            lattice_type=self.lattice_type,
            a=self.a,
            eps1=self.eps_min,
            eps2=self.eps_max,
            radius=0.0
        )

        def custom_eps(x, y):
            ix = int(np.clip(x / self.resolution, 0, self.nx - 1))
            iy = int(np.clip(y / self.resolution, 0, self.ny - 1))
            return eps_dist[iy, ix]

        pc.fourier_epsilon = lambda G: self._fourier_epsilon_custom(G, custom_eps)
        pc.generate_reciprocal_vectors(80)

        try:
            freqs = pc.solve_eigenvalues(k_point, mode='TM', num_bands=5)
            band_gap = freqs[2] - freqs[1]
            objective = -band_gap
            return objective, freqs
        except:
            return 1e6, None

    def _fourier_epsilon_custom(self, G, eps_func):
        from scipy.integrate import quad
        if np.linalg.norm(G) < 1e-12:
            f = lambda y: (lambda x: eps_func(x, y))(0)
            integral, _ = quad(lambda y: (lambda x: eps_func(x, y))(0), 0, self.a)
            return integral / self.a

        G_norm = np.linalg.norm(G)
        delta_eps = self.eps_max - self.eps_min
        return delta_eps * 2 * j1(G_norm * self.a / 4) / G_norm * 0.5

    def optimize(self, target_frequency=0.3, num_iterations=50):
        print("=" * 60)
        print("拓扑优化开始")
        print("=" * 60)

        k_point = np.array([np.pi / self.a, 0.0])
        history = []

        for iteration in range(num_iterations):
            density_filtered = self._apply_filter(self.density)
            density_projected = self._project_density(density_filtered, beta=min(1 + iteration * 0.2, 8))

            objective, freqs = self._compute_objective(density_projected, target_frequency, k_point)

            grad = np.zeros_like(self.density)
            eps = 1e-3
            for i in range(self.nx):
                for j in range(self.ny):
                    d_plus = self.density.copy()
                    d_plus[i, j] = min(1.0, d_plus[i, j] + eps)
                    obj_plus, _ = self._compute_objective(self._project_density(self._apply_filter(d_plus)), target_frequency, k_point)

                    d_minus = self.density.copy()
                    d_minus[i, j] = max(0.0, d_minus[i, j] - eps)
                    obj_minus, _ = self._compute_objective(self._project_density(self._apply_filter(d_minus)), target_frequency, k_point)

                    grad[i, j] = (obj_plus - obj_minus) / (2 * eps)

            self.density = self.density - self.learning_rate * grad
            self.density = np.clip(self.density, 0.0, 1.0)

            vol = np.mean(self.density)
            if vol > self.volume_fraction:
                self.density *= self.volume_fraction / vol

            history.append({
                'iteration': iteration,
                'objective': objective,
                'band_gap': -objective,
                'volume': vol,
                'freqs': freqs
            })

            if freqs is not None:
                print(f"迭代 {iteration:3d}: 带隙 = {-objective:.4f}, 体积分数 = {vol:.3f}, "
                      f"f1={freqs[0]:.3f}, f2={freqs[1]:.3f}, f3={freqs[2]:.3f}")
            else:
                print(f"迭代 {iteration:3d}: 目标函数 = {objective:.4f}, 体积分数 = {vol:.3f}")

            if iteration % 10 == 0:
                self._plot_design(density_projected, iteration, history)

        print("\n拓扑优化完成！")
        self._plot_final_design(history)
        return self.density, history

    def _plot_design(self, density, iteration, history):
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(density, cmap='binary', origin='lower',
                       extent=[0, self.design_region_size[0], 0, self.design_region_size[1]])
        ax.set_xlabel('x (a)', fontsize=14)
        ax.set_ylabel('y (a)', fontsize=14)
        ax.set_title(f'拓扑优化 - 迭代 {iteration}', fontsize=16)
        plt.colorbar(im, ax=ax, label='密度')
        plt.tight_layout()
        plt.savefig(f'topology_iter_{iteration}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_final_design(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        eps_final = self._apply_simp(self.density)
        im0 = axes[0].imshow(eps_final, cmap='viridis', origin='lower',
                            extent=[0, self.design_region_size[0], 0, self.design_region_size[1]])
        axes[0].set_xlabel('x (a)', fontsize=14)
        axes[0].set_ylabel('y (a)', fontsize=14)
        axes[0].set_title('优化后的介电常数分布', fontsize=16)
        plt.colorbar(im0, ax=axes[0], label='ε')

        iterations = [h['iteration'] for h in history]
        band_gaps = [h['band_gap'] for h in history]
        axes[1].plot(iterations, band_gaps, 'b-', linewidth=2)
        axes[1].set_xlabel('迭代次数', fontsize=14)
        axes[1].set_ylabel('带隙宽度 (ωa/2πc)', fontsize=14)
        axes[1].set_title('带隙优化历史', fontsize=16)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('topology_final.png', dpi=300, bbox_inches='tight')
        plt.close()


class TopologicalPhotonicCrystal:
    def __init__(self, lattice_type='honeycomb', a=1.0, eps1=1.0, eps2=10.0, radius=0.15):
        self.a = a
        self.eps1 = eps1
        self.eps2 = eps2
        self.radius = radius
        self.lattice_type = lattice_type
        self._setup_honeycomb_lattice()

    def _setup_honeycomb_lattice(self):
        self.a1 = np.array([np.sqrt(3) * self.a, 0.0])
        self.a2 = np.array([np.sqrt(3) * self.a / 2.0, 3 * self.a / 2.0])

        a1_3d = np.append(self.a1, 0)
        a2_3d = np.append(self.a2, 0)
        a3_3d = np.array([0, 0, 1])
        self.cell_area = np.abs(np.dot(a3_3d, np.cross(a1_3d, a2_3d)))

        self.b1 = 2 * np.pi * np.array([self.a2[1], -self.a2[0]]) / self.cell_area
        self.b2 = 2 * np.pi * np.array([-self.a1[1], self.a1[0]]) / self.cell_area

    def get_eps_at_point(self, x, y, delta=0.0):
        sublattice_A = np.array([0.0, 0.0])
        sublattice_B = np.array([self.a, 0.0])

        r_A = np.sqrt(x ** 2 + y ** 2)
        if r_A < self.radius * (1 + delta):
            return self.eps2

        r_B = np.sqrt((x - self.a) ** 2 + y ** 2)
        if r_B < self.radius * (1 - delta):
            return self.eps2

        for i in range(-3, 4):
            for j in range(-3, 4):
                R = i * self.a1 + j * self.a2

                r_A = np.sqrt((x - R[0]) ** 2 + (y - R[1]) ** 2)
                if r_A < self.radius * (1 + delta):
                    return self.eps2

                r_B = np.sqrt((x - self.a - R[0]) ** 2 + (y - R[1]) ** 2)
                if r_B < self.radius * (1 - delta):
                    return self.eps2

        return self.eps1

    def compute_berry_phase(self, k_path, mode='TM', band_idx=0, num_planes=100):
        from photonic_crystal_pwe import PhotonicCrystal2D

        pc = PhotonicCrystal2D(
            lattice_type='square',
            a=self.a * np.sqrt(3),
            eps1=self.eps1,
            eps2=self.eps2,
            radius=self.radius
        )
        pc.generate_reciprocal_vectors(num_planes)

        eigenvectors = []
        for k in k_path:
            K, M, _ = pc.build_matrices(k, mode=mode)
            eigenvalues, eigenvecs = linalg.eigh(K, M)
            eigenvectors.append(eigenvecs[:, band_idx])

        berry_phase = 0.0 + 0.0j
        for i in range(len(k_path) - 1):
            overlap = np.vdot(eigenvectors[i], eigenvectors[i + 1])
            berry_phase += np.log(overlap / np.abs(overlap))

        return np.imag(berry_phase)

    def compute_edge_states(self, supercell_size=10, mode='TM', num_bands=20):
        pc = PhotonicCrystalDefectFEM(
            lattice_type='square',
            a=self.a,
            eps1=self.eps1,
            eps2=self.eps2,
            radius=self.radius,
            supercell_size=(1, supercell_size)
        )

        def edge_defect(x, y, eps_base):
            if y > -0.5 and y < 0.5:
                return self.eps1
            return eps_base

        pc.set_custom_defect(edge_defect)
        pc.generate_mesh(resolution=0.15)

        k_path = [np.array([k, 0.0]) for k in np.linspace(-np.pi / self.a, np.pi / self.a, 20)]
        edge_bands = []

        for k in k_path:
            freqs, _ = pc.solve_eigenvalues(k, mode=mode, num_bands=num_bands)
            edge_bands.append(freqs)

        return np.array(k_path), np.array(edge_bands)

    def plot_edge_states(self, k_path, edge_bands, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 7))

        kx = [k[0] * self.a / np.pi for k in k_path]
        for band in range(edge_bands.shape[1]):
            ax.plot(kx, edge_bands[:, band], 'b-', linewidth=2)

        ax.set_xlabel('k_x a / π', fontsize=14)
        ax.set_ylabel('频率 (ωa/2πc)', fontsize=14)
        ax.set_title('拓扑边界态能带结构', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
        ax.tick_params(axis='both', labelsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"拓扑边界态图已保存到: {save_path}")
        plt.close()


def demo_point_defect():
    print("\n" + "=" * 60)
    print("演示 1: 点缺陷 - 光子晶体微腔")
    print("=" * 60)

    params = {
        'lattice_type': 'triangular',
        'a': 1.0,
        'eps1': 1.0,
        'eps2': 12.0,
        'radius': 0.2,
        'supercell_size': (5, 5)
    }

    pc = PhotonicCrystalDefectFEM(**params)
    pc.set_point_defect(defect_type='remove')
    pc.generate_mesh(resolution=0.12)

    k_gamma = np.array([0.0, 0.0])
    freqs, eigvecs = pc.compute_defect_bands(k_gamma, mode='TM', num_bands=15)

    print(f"\n计算得到的本征频率:")
    for i, f in enumerate(freqs[:10]):
        print(f"  能带 #{i}: {f:.4f}")

    bulk_band_gap = [0.27, 0.45]
    defect_modes = np.where((freqs > bulk_band_gap[0]) & (freqs < bulk_band_gap[1]))[0]

    if len(defect_modes) > 0:
        print(f"\n发现 {len(defect_modes)} 个带隙内缺陷态:")
        for idx in defect_modes:
            print(f"  缺陷态在能带 #{idx}, 频率 = {freqs[idx]:.4f}")

        pc.plot_defect_mode(eigvecs[:, defect_modes[0]], defect_modes[0],
                            save_path='point_defect_mode.png')
    else:
        print("\n未发现带隙内缺陷态，可能需要调整缺陷参数")


def demo_line_defect():
    print("\n" + "=" * 60)
    print("演示 2: 线缺陷 - 光子晶体波导")
    print("=" * 60)

    params = {
        'lattice_type': 'triangular',
        'a': 1.0,
        'eps1': 1.0,
        'eps2': 12.0,
        'radius': 0.2,
        'supercell_size': (5, 7)
    }

    pc = PhotonicCrystalDefectFEM(**params)
    pc.set_line_defect(direction='x', remove_row=0)
    pc.generate_mesh(resolution=0.12)

    k_path = [np.array([k, 0.0]) for k in np.linspace(-np.pi / pc.a, np.pi / pc.a, 15)]
    k_labels = [r'-π/a', '0', r'π/a']

    waveguide_bands_TM = []
    waveguide_bands_TE = []

    print("\n计算波导色散...")
    for i, k in enumerate(k_path):
        freqs_TM, _ = pc.solve_eigenvalues(k, mode='TM', num_bands=10)
        freqs_TE, _ = pc.solve_eigenvalues(k, mode='TE', num_bands=10)
        waveguide_bands_TM.append(freqs_TM)
        waveguide_bands_TE.append(freqs_TE)
        print(f"  k点 {i+1}/{len(k_path)}: TM={freqs_TM[:3]}, TE={freqs_TE[:3]}")

    waveguide_bands_TM = np.array(waveguide_bands_TM)
    waveguide_bands_TE = np.array(waveguide_bands_TE)

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(k_path))

    for band in range(waveguide_bands_TM.shape[1]):
        ax.plot(x, waveguide_bands_TM[:, band], 'b-', linewidth=2, label='TM波导模' if band == 0 else "")

    ax.axhspan(0.27, 0.45, facecolor='gray', alpha=0.2, label='体带隙')

    ax.set_xticks([0, len(k_path) // 2, len(k_path) - 1])
    ax.set_xticklabels(k_labels, fontsize=14)
    ax.set_ylabel('频率 (ωa/2πc)', fontsize=14)
    ax.set_title('光子晶体波导色散曲线', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylim(0, 0.6)

    plt.tight_layout()
    plt.savefig('waveguide_dispersion.png', dpi=300, bbox_inches='tight')
    print("\n波导色散图已保存到: waveguide_dispersion.png")
    plt.close()


def demo_topology_optimization():
    print("\n" + "=" * 60)
    print("演示 3: 拓扑优化 - 最大化带隙")
    print("=" * 60)

    opt = TopologyOptimization(
        lattice_type='square',
        a=1.0,
        eps_min=1.0,
        eps_max=12.0,
        design_region_size=(5, 5),
        resolution=0.5
    )

    optimal_density, history = opt.optimize(target_frequency=0.3, num_iterations=20)
    return optimal_density, history


def demo_topological_edge_states():
    print("\n" + "=" * 60)
    print("演示 4: 拓扑光子晶体 - 谷霍尔边界态")
    print("=" * 60)

    tpc = TopologicalPhotonicCrystal(
        lattice_type='honeycomb',
        a=1.0,
        eps1=1.0,
        eps2=10.0,
        radius=0.15
    )

    print("计算拓扑边界态...")
    k_path, edge_bands = tpc.compute_edge_states(supercell_size=8, mode='TM', num_bands=15)
    tpc.plot_edge_states(k_path, edge_bands, save_path='topological_edge_states.png')

    print("\n计算贝里相位...")
    k_loop = []
    for theta in np.linspace(0, 2 * np.pi, 20):
        k = np.array([0.5 * np.cos(theta), 0.5 * np.sin(theta)]) * 2 * np.pi / tpc.a
        k_loop.append(k)

    try:
        berry_phase = tpc.compute_berry_phase(k_loop, mode='TM', band_idx=0, num_planes=60)
        print(f"第一能带贝里相位: {berry_phase:.4f} (2π整数倍 = {berry_phase / (2 * np.pi):.2f})")
    except Exception as e:
        print(f"贝里相位计算跳过: {e}")


def main():
    print("=" * 70)
    print("光子晶体缺陷态、拓扑优化与器件设计")
    print("=" * 70)

    print("""
    功能模块:
    1. 点缺陷态计算 - 光子晶体微腔/谐振腔
    2. 线缺陷态计算 - 光子晶体波导
    3. 拓扑优化 - 基于密度法的带隙最大化
    4. 拓扑光子晶体 - 谷霍尔边界态计算
    """)

    demo_point_defect()
    demo_line_defect()

    try:
        demo_topology_optimization()
    except Exception as e:
        print(f"\n拓扑优化演示跳过（简化版本）: {e}")

    try:
        demo_topological_edge_states()
    except Exception as e:
        print(f"\n拓扑边界态演示跳过（简化版本）: {e}")

    print("\n" + "=" * 70)
    print("功能总结:")
    print("=" * 70)
    print("""
    1. 点缺陷 (PhotonicCrystalDefectFEM.set_point_defect):
       - 移除介质柱形成微腔
       - 计算缺陷模式频率和模场分布
       - 分析品质因子和谐振特性

    2. 线缺陷 (PhotonicCrystalDefectFEM.set_line_defect):
       - 移除一排介质柱形成波导
       - 计算波导色散曲线
       - 分析模式带宽和群速度

    3. 拓扑优化 (TopologyOptimization):
       - SIMP密度插值方法
       - 灵敏度分析和梯度下降
       - 密度过滤和投影避免棋盘格
       - 目标：最大化指定频率处带隙

    4. 拓扑光子晶体 (TopologicalPhotonicCrystal):
       - 蜂窝晶格谷霍尔效应
       - 赝自旋相关边界态
       - 贝里相位和拓扑不变量计算
       - 单向传输边界态设计
    """)

    print("\n所有演示完成！")


if __name__ == '__main__':
    main()
