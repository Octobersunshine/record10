import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.spatial import Delaunay
from scipy.special import j1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

rcParams['font.sans-serif'] = ['SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False


class PhotonicCrystalFEM:
    def __init__(self, lattice_type='triangular', a=1.0, eps1=1.0, eps2=12.0, radius=0.2):
        self.a = a
        self.eps1 = eps1
        self.eps2 = eps2
        self.radius = radius
        self.lattice_type = lattice_type
        self._setup_lattice()
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

        a1_3d = np.append(self.a1, 0)
        a2_3d = np.append(self.a2, 0)
        a3_3d = np.array([0, 0, 1])
        self.cell_area = np.abs(np.dot(a3_3d, np.cross(a1_3d, a2_3d)))

        self.b1 = 2 * np.pi * np.array([self.a2[1], -self.a2[0]]) / self.cell_area
        self.b2 = 2 * np.pi * np.array([-self.a1[1], self.a1[0]]) / self.cell_area

    def get_eps_at_point(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        if r < self.radius:
            return self.eps2
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                R = i * self.a1 + j * self.a2
                dx = x - R[0]
                dy = y - R[1]
                r = np.sqrt(dx ** 2 + dy ** 2)
                if r < self.radius:
                    return self.eps2
        return self.eps1

    def generate_mesh(self, resolution=0.05):
        if self.lattice_type == 'triangular':
            x_min, x_max = -self.a * 0.1, self.a * 1.1
            y_min, y_max = -self.a * 0.1, self.a * np.sqrt(3) / 2.0 + self.a * 0.1
        else:
            x_min, x_max = -self.a * 0.1, self.a * 1.1
            y_min, y_max = -self.a * 0.1, self.a * 1.1

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

        print(f"网格生成完成: {len(self.nodes)} 节点, {len(self.elements)} 单元")
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
                    self.boundary_pairs.append((li, ri, 'x', self.a1))
                    break

        for bi in bottom_nodes:
            x_bottom = nodes[bi, 0]
            for ti in top_nodes:
                x_top = nodes[ti, 0]
                if np.abs(x_bottom - x_top) < tol:
                    self.boundary_pairs.append((bi, ti, 'y', self.a2))
                    break

        self.interior_nodes = np.ones(len(nodes), dtype=bool)
        self.master_nodes = []
        self.slave_nodes = []

        for pair in self.boundary_pairs:
            slave, master, _, _ = pair
            self.interior_nodes[slave] = False
            self.slave_nodes.append(slave)
            self.master_nodes.append(master)

        self.interior_nodes = np.where(self.interior_nodes)[0]
        self.all_node_indices = np.arange(len(nodes))

        print(f"边界对识别完成: {len(self.boundary_pairs)} 对边界节点")

    def _compute_element_properties(self):
        self.element_areas = np.zeros(len(self.elements))
        self.element_centroids = np.zeros((len(self.elements), 2))
        self.element_eps = np.zeros(len(self.elements))
        self.grad_N = np.zeros((len(self.elements), 3, 2))
        self.mass_matrix_coeff = np.zeros((len(self.elements), 3, 3))

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
                    if i == j:
                        self.mass_matrix_coeff[e, i, j] = area / 12.0
                    else:
                        self.mass_matrix_coeff[e, i, j] = area / 24.0

    def build_transformation_matrix(self, k_point):
        n_nodes = len(self.nodes)

        phase_x = np.exp(1j * np.dot(k_point, self.a1))
        phase_y = np.exp(1j * np.dot(k_point, self.a2))

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

        return T.tocsr(), keep_indices, slave_to_master, slave_phases

    def build_matrices(self, k_point, mode='TM'):
        n_nodes = len(self.nodes)

        rows_K = []
        cols_K = []
        vals_K = []

        rows_M = []
        cols_M = []
        vals_M = []

        for e, elem in enumerate(self.elements):
            area = self.element_areas[e]
            eps_e = self.element_eps[e]
            grad_N = self.grad_N[e]
            mass_coeff = self.mass_matrix_coeff[e]

            local_K = np.zeros((3, 3), dtype=float)
            local_M = np.zeros((3, 3), dtype=float)

            if mode == 'TM':
                for i in range(3):
                    for j in range(3):
                        local_K[i, j] = np.dot(grad_N[i], grad_N[j]) * area
                        local_M[i, j] = eps_e * mass_coeff[i, j]
            elif mode == 'TE':
                for i in range(3):
                    for j in range(3):
                        local_K[i, j] = np.dot(grad_N[i], grad_N[j]) * area / eps_e
                        local_M[i, j] = mass_coeff[i, j]

            for i in range(3):
                for j in range(3):
                    rows_K.append(elem[i])
                    cols_K.append(elem[j])
                    vals_K.append(local_K[i, j])

                    rows_M.append(elem[i])
                    cols_M.append(elem[j])
                    vals_M.append(local_M[i, j])

        K = sparse.csr_matrix((vals_K, (rows_K, cols_K)), shape=(n_nodes, n_nodes), dtype=complex)
        M = sparse.csr_matrix((vals_M, (rows_M, cols_M)), shape=(n_nodes, n_nodes), dtype=complex)

        T, keep_indices, slave_to_master, slave_phases = self.build_transformation_matrix(k_point)

        K_reduced = T @ K @ T.conj().T
        M_reduced = T @ M @ T.conj().T

        return K_reduced, M_reduced, keep_indices

    def solve_eigenvalues(self, k_point, mode='TM', num_bands=10):
        K, M, keep_indices = self.build_matrices(k_point, mode=mode)

        A = K
        B = M

        n_dof = A.shape[0]
        num_eigs = min(num_bands + 5, n_dof - 1)

        try:
            eigenvalues, eigenvectors = splinalg.eigs(A, k=num_eigs, M=B, which='SM')
        except Exception as e:
            print(f"  警告: scipy.sparse.linalg.eigs失败 ({e}), 改用稠密矩阵求解")
            A_dense = A.todense()
            B_dense = B.todense()
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(B_dense) @ A_dense)

        eigenvalues = np.sort(np.maximum(eigenvalues.real, 0))
        frequencies = np.sqrt(eigenvalues) * self.a / (2 * np.pi)

        return frequencies[:num_bands]

    def get_high_symmetry_points(self):
        if self.lattice_type == 'triangular':
            Gamma = np.array([0.0, 0.0])
            M = np.array([np.pi / self.a, np.pi / (self.a * np.sqrt(3))])
            K = np.array([4 * np.pi / (3 * self.a), 0.0])
            return [Gamma, M, K, Gamma]
        elif self.lattice_type == 'square':
            Gamma = np.array([0.0, 0.0])
            X = np.array([np.pi / self.a, 0.0])
            M = np.array([np.pi / self.a, np.pi / self.a])
            return [Gamma, X, M, Gamma]
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")

    def generate_k_path(self, n_points_per_segment=20):
        high_sym = self.get_high_symmetry_points()
        k_path = []
        for i in range(len(high_sym) - 1):
            start = high_sym[i]
            end = high_sym[i + 1]
            t = np.linspace(0, 1, n_points_per_segment, endpoint=False)
            segment = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]
            k_path.extend(segment)
        k_path.append(high_sym[-1])
        return np.array(k_path)

    def get_k_labels(self):
        if self.lattice_type == 'triangular':
            return [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
        elif self.lattice_type == 'square':
            return [r'$\Gamma$', 'X', 'M', r'$\Gamma$']
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")

    def compute_band_structure(self, k_path, mode='both', num_bands=10):
        n_k = len(k_path)
        bands_TM = np.zeros((n_k, num_bands)) if mode in ['TM', 'both'] else None
        bands_TE = np.zeros((n_k, num_bands)) if mode in ['TE', 'both'] else None

        for i, k in enumerate(k_path):
            if mode in ['TM', 'both']:
                freqs = self.solve_eigenvalues(k, 'TM', num_bands)
                bands_TM[i, :] = freqs[:num_bands]
                print(f"  k点 {i+1}/{n_k} (TM): {freqs[:3]}")
            if mode in ['TE', 'both']:
                freqs = self.solve_eigenvalues(k, 'TE', num_bands)
                bands_TE[i, :] = freqs[:num_bands]
                if mode == 'both':
                    print(f"  k点 {i+1}/{n_k} (TE): {freqs[:3]}")

        return bands_TM, bands_TE

    def analyze_gaps(self, bands):
        gaps = []
        num_bands = bands.shape[1]
        for i in range(num_bands - 1):
            band_max = np.max(bands[:, i])
            next_band_min = np.min(bands[:, i + 1])
            if next_band_min > band_max:
                gap_size = next_band_min - band_max
                gap_center = (next_band_min + band_max) / 2
                gaps.append({
                    'band_index': i + 1,
                    'lower_edge': band_max,
                    'upper_edge': next_band_min,
                    'gap_size': gap_size,
                    'center': gap_center,
                    'relative_size': gap_size / gap_center
                })
        return gaps

    def plot_mesh(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))

        patches = []
        colors = []

        for e, elem in enumerate(self.elements):
            pts = self.nodes[elem]
            poly = Polygon(pts, closed=True)
            patches.append(poly)
            colors.append(self.element_eps[e])

        p = PatchCollection(patches, cmap='viridis', alpha=0.8)
        p.set_array(np.array(colors))
        ax.add_collection(p)

        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c='k', s=3, alpha=0.5)

        for slave, master, direction, _ in self.boundary_pairs:
            p1 = self.nodes[slave]
            p2 = self.nodes[master]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, alpha=0.7)

        ax.set_xlabel('x (a)', fontsize=14)
        ax.set_ylabel('y (a)', fontsize=14)
        ax.set_title(f'FEM 三角网格 ({self.lattice_type} 晶格)', fontsize=16)
        ax.set_aspect('equal')

        cbar = plt.colorbar(p, ax=ax)
        cbar.set_label('ε', fontsize=14)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"网格图已保存到: {save_path}")
        plt.close()

        return fig, ax

    def plot_eps_distribution(self, save_path=None):
        nx, ny = 200, 200
        x_min, x_max = np.min(self.nodes[:, 0]), np.max(self.nodes[:, 0])
        y_min, y_max = np.min(self.nodes[:, 1]), np.max(self.nodes[:, 1])

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)

        eps = np.zeros_like(X)
        for i in range(nx):
            for j in range(ny):
                eps[j, i] = self.get_eps_at_point(X[j, i], Y[j, i])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.pcolormesh(X, Y, eps, cmap='viridis', shading='auto')
        ax.set_xlabel('x (a)', fontsize=14)
        ax.set_ylabel('y (a)', fontsize=14)
        ax.set_title(f'介电常数分布 ({self.lattice_type} 晶格)', fontsize=16)
        ax.set_aspect('equal')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('ε', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"介电常数分布图已保存到: {save_path}")
        plt.close()

        return fig, ax


class PhotonicCrystalFDTD:
    def __init__(self, lattice_type='triangular', a=1.0, eps1=1.0, eps2=12.0, radius=0.2):
        self.a = a
        self.eps1 = eps1
        self.eps2 = eps2
        self.radius = radius
        self.lattice_type = lattice_type
        self._setup_lattice()

    def _setup_lattice(self):
        if self.lattice_type == 'triangular':
            self.a1 = np.array([self.a, 0.0])
            self.a2 = np.array([self.a / 2.0, self.a * np.sqrt(3) / 2.0])
        elif self.lattice_type == 'square':
            self.a1 = np.array([self.a, 0.0])
            self.a2 = np.array([0.0, self.a])
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")

        a1_3d = np.append(self.a1, 0)
        a2_3d = np.append(self.a2, 0)
        a3_3d = np.array([0, 0, 1])
        self.cell_area = np.abs(np.dot(a3_3d, np.cross(a1_3d, a2_3d)))

        self.b1 = 2 * np.pi * np.array([self.a2[1], -self.a2[0]]) / self.cell_area
        self.b2 = 2 * np.pi * np.array([-self.a1[1], self.a1[0]]) / self.cell_area

    def get_eps_at_point(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        if r < self.radius:
            return self.eps2
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                R = i * self.a1 + j * self.a2
                dx = x - R[0]
                dy = y - R[1]
                r = np.sqrt(dx ** 2 + dy ** 2)
                if r < self.radius:
                    return self.eps2
        return self.eps1

    def run_simulation(self, k_point, nx=40, ny=40, nt=2000, num_bands=8):
        dx = self.a / nx
        dy = self.a / ny
        dt = 0.9 / np.sqrt(2.0) * dx

        eps = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                eps[i, j] = self.get_eps_at_point(x, y)

        Ez = np.zeros((nx, ny))
        Hx = np.zeros((nx, ny))
        Hy = np.zeros((nx, ny))

        source_x, source_y = nx // 2, ny // 2

        phase_x = np.exp(1j * np.dot(k_point, self.a1))
        phase_y = np.exp(1j * np.dot(k_point, self.a2))

        fourier_transform = np.zeros((nt, num_bands), dtype=complex)
        freq_centers = np.linspace(0.1, 0.9, num_bands)
        freq_width = 0.05

        monitor_x, monitor_y = nx // 4, ny // 4

        for t in range(nt):
            Hy[1:, :-1] = Hy[1:, :-1] + dt / dy * (Ez[1:, 1:] - Ez[1:, :-1])
            Hx[:-1, 1:] = Hx[:-1, 1:] - dt / dx * (Ez[1:, 1:] - Ez[:-1, 1:])

            Ez[1:-1, 1:-1] = Ez[1:-1, 1:-1] + dt / eps[1:-1, 1:-1] * (
                (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy -
                (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx
            )

            if t < 200:
                Ez[source_x, source_y] += np.sin(2 * np.pi * 0.3 * t) * np.exp(-(t - 100) ** 2 / 50 ** 2)

            Ez[0, :] = Ez[-1, :] * np.conj(phase_x)
            Ez[-1, :] = Ez[0, :] * phase_x
            Ez[:, 0] = Ez[:, -1] * np.conj(phase_y)
            Ez[:, -1] = Ez[:, 0] * phase_y

            for f_idx in range(num_bands):
                omega = 2 * np.pi * freq_centers[f_idx] * 2 * np.pi / self.a
                fourier_transform[t, f_idx] = Ez[monitor_x, monitor_y] * np.exp(-1j * omega * t * dt)

        freq_response = np.abs(np.sum(fourier_transform, axis=0))
        peak_indices = np.argsort(freq_response)[-num_bands:][::-1]
        frequencies = freq_centers[peak_indices]

        return np.sort(frequencies)

    def compute_band_structure(self, k_path, num_bands=8, nx=30, ny=30, nt=1000):
        n_k = len(k_path)
        bands = np.zeros((n_k, num_bands))

        for i, k in enumerate(k_path):
            freqs = self.run_simulation(k, nx=nx, ny=ny, nt=nt, num_bands=num_bands)
            bands[i, :] = freqs[:num_bands]
            print(f"  k点 {i+1}/{n_k}: {freqs[:3]}")

        return bands


def plot_band_structure(k_path, bands_TM, bands_TE, k_labels, title='光子晶体能带结构',
                        save_path=None, show_gaps=True, method='FEM'):
    fig, ax = plt.subplots(figsize=(10, 7))

    n_segments = len(k_labels) - 1
    n_per_seg = (len(k_path) - 1) // n_segments
    x_positions = [i * n_per_seg for i in range(n_segments + 1)]
    x = np.arange(len(k_path))

    if bands_TM is not None:
        for band in range(bands_TM.shape[1]):
            ax.plot(x, bands_TM[:, band], 'b-', linewidth=2.5, label='TM 模式' if band == 0 else "")
        if show_gaps:
            pc = PhotonicCrystalFEM()
            gaps = pc.analyze_gaps(bands_TM)
            for gap in gaps:
                ax.axhspan(gap['lower_edge'], gap['upper_edge'], facecolor='blue', alpha=0.15)

    if bands_TE is not None:
        for band in range(bands_TE.shape[1]):
            ax.plot(x, bands_TE[:, band], 'r--', linewidth=2.5, label='TE 模式' if band == 0 else "")
        if show_gaps:
            pc = PhotonicCrystalFEM()
            gaps = pc.analyze_gaps(bands_TE)
            for gap in gaps:
                ax.axhspan(gap['lower_edge'], gap['upper_edge'], facecolor='red', alpha=0.15)

    for xi in x_positions:
        ax.axvline(x=xi, color='k', linestyle='-', alpha=0.6, linewidth=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(k_labels, fontsize=16)
    ax.set_ylabel('频率 (ωa/2πc)', fontsize=16)
    ax.set_title(title + f' ({method}方法)', fontsize=18, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=14)
    ax.set_ylim(0, None)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"能带图已保存到: {save_path}")
    plt.close()

    return fig, ax


def print_gap_analysis(bands, mode_name):
    if bands is None:
        return
    pc = PhotonicCrystalFEM()
    gaps = pc.analyze_gaps(bands)
    print(f"\n{'=' * 60}")
    print(f"{mode_name} 模式带隙分析:")
    print(f"{'=' * 60}")
    if not gaps:
        print("  未发现完整带隙")
    else:
        for gap in gaps:
            print(f"  带隙 #{gap['band_index']}:")
            print(f"    频率范围: {gap['lower_edge']:.4f} - {gap['upper_edge']:.4f}")
            print(f"    带隙宽度: {gap['gap_size']:.4f}")
            print(f"    中心频率: {gap['center']:.4f}")
            print(f"    相对宽度: {gap['relative_size'] * 100:.2f}%")


def main():
    print("=" * 60)
    print("二维光子晶体能带结构计算 - FEM / FDTD 方法")
    print("=" * 60)

    params = {
        'lattice_type': 'triangular',
        'a': 1.0,
        'eps1': 1.0,
        'eps2': 12.0,
        'radius': 0.2
    }

    print(f"\n晶格类型: {params['lattice_type']}")
    print(f"晶格常数 a = {params['a']}")
    print(f"背景介电常数 ε1 = {params['eps1']}")
    print(f"圆柱介电常数 ε2 = {params['eps2']}")
    print(f"圆柱半径 r = {params['radius']}")

    print("\n" + "=" * 60)
    print("方法 1: 有限元法 (FEM) 求解")
    print("=" * 60)

    pc_fem = PhotonicCrystalFEM(**params)
    pc_fem.generate_mesh(resolution=0.1)

    pc_fem.plot_mesh(save_path='fem_mesh.png')
    pc_fem.plot_eps_distribution(save_path='fem_eps_distribution.png')

    num_bands = 6
    k_path = pc_fem.generate_k_path(n_points_per_segment=8)
    k_labels = pc_fem.get_k_labels()

    print(f"\n计算 {num_bands} 条能带，k 点数: {len(k_path)}")
    print("高对称点路径:", " → ".join(k_labels))
    print("\n正在计算FEM能带结构...")

    bands_TM_fem, bands_TE_fem = pc_fem.compute_band_structure(
        k_path, mode='both', num_bands=num_bands
    )

    print("\nFEM计算完成！")
    print_gap_analysis(bands_TM_fem, "TM (FEM)")
    print_gap_analysis(bands_TE_fem, "TE (FEM)")

    title = f'三角晶格光子晶体能带结构 (r={params["radius"]}a, ε={params["eps2"]}/{params["eps1"]})'
    save_path = 'band_structure_fem_triangular.png'
    plot_band_structure(k_path, bands_TM_fem, bands_TE_fem, k_labels,
                        title=title, save_path=save_path, method='FEM')

    print("\n" + "=" * 60)
    print("方法 2: FDTD + Bloch 边界条件")
    print("=" * 60)

    pc_fdtd = PhotonicCrystalFDTD(**params)
    k_path_fdtd = pc_fdtd.generate_k_path(n_points_per_segment=10)

    print(f"\n计算FDTD能带结构，k 点数: {len(k_path_fdtd)}")
    print("正在计算FDTD能带结构...")

    bands_fdtd = pc_fdtd.compute_band_structure(
        k_path_fdtd, num_bands=num_bands, nx=25, ny=25, nt=800
    )

    print("\nFDTD计算完成！")
    print_gap_analysis(bands_fdtd, "FDTD")

    title_fdtd = f'三角晶格光子晶体能带结构 (FDTD方法)'
    save_path_fdtd = 'band_structure_fdtd_triangular.png'
    plot_band_structure(k_path_fdtd, bands_fdtd, None, k_labels,
                        title=title_fdtd, save_path=save_path_fdtd, method='FDTD')

    print("\n" + "=" * 60)
    print("收敛性对比分析")
    print("=" * 60)

    resolutions = [0.15, 0.1, 0.08]
    convergence_results = []

    for res in resolutions:
        print(f"\n  网格分辨率: {res}")
        pc_conv = PhotonicCrystalFEM(**params)
        pc_conv.generate_mesh(resolution=res)

        k_gamma = np.array([0.0, 0.0])
        freqs = pc_conv.solve_eigenvalues(k_gamma, mode='TM', num_bands=5)
        convergence_results.append({
            'resolution': res,
            'n_nodes': len(pc_conv.nodes),
            'n_elements': len(pc_conv.elements),
            'frequencies': freqs
        })
        print(f"    节点数: {len(pc_conv.nodes)}, 单元数: {len(pc_conv.elements)}")
        print(f"    前5条能带频率: {freqs}")

    if len(convergence_results) >= 2:
        for i in range(len(convergence_results) - 1):
            freq1 = convergence_results[i]['frequencies']
            freq2 = convergence_results[i + 1]['frequencies']
            rel_error = np.abs(freq2 - freq1) / np.abs(freq2) * 100
            print(f"\n  分辨率 {convergence_results[i]['resolution']} -> {convergence_results[i+1]['resolution']}:")
            print(f"    相对误差: {rel_error} %")

    print("\n" + "=" * 60)
    print("方法对比总结:")
    print("=" * 60)
    print("""
    PWE方法:
      ✓ 实现简单, 数学直观
      ✗ 介电常数陡变时收敛慢, 需要大量平面波
      ✗ 对高对比度结构精度差

    FEM方法:
      ✓ 自适应网格, 介电常数界面分辨率高
      ✓ 收敛快, 精度高
      ✓ 适合任意几何形状
      ✗ 矩阵构建复杂, 计算量大

    FDTD方法:
      ✓ 时域直接模拟, 物理直观
      ✓ 一次计算可得宽频响应
      ✗ 色散误差累积, 长时间模拟精度下降
      ✗ 为获得精确能带需要较长模拟时间
    """)

    print("\n所有计算完成！结果图像已保存。")


if __name__ == '__main__':
    main()
