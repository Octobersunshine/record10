import numpy as np
from scipy import linalg
from scipy.special import j1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False


class PhotonicCrystal2D:
    def __init__(self, lattice_type='triangular', a=1.0, eps1=1.0, eps2=12.0, radius=0.2):
        self.a = a
        self.eps1 = eps1
        self.eps2 = eps2
        self.radius = radius
        self.lattice_type = lattice_type
        self._setup_lattice()
        self.G_vectors = None
        self.n_G = None

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

    def generate_reciprocal_vectors(self, num_planes):
        n = int(np.ceil(np.sqrt(num_planes)))
        indices = np.arange(-n, n + 1)
        i, j = np.meshgrid(indices, indices, indexing='ij')
        i = i.flatten()
        j = j.flatten()

        Gs = i[:, None] * self.b1 + j[:, None] * self.b2
        norms = np.linalg.norm(Gs, axis=1)

        sorted_indices = np.argsort(norms)
        Gs = Gs[sorted_indices][:num_planes]

        self.G_vectors = Gs
        self.n_G = len(Gs)
        return Gs

    def fourier_epsilon(self, G):
        G_norm = np.linalg.norm(G)
        if G_norm < 1e-12:
            f = np.pi * self.radius ** 2 / self.cell_area
            return self.eps1 * (1 - f) + self.eps2 * f

        x = G_norm * self.radius
        delta_eps = self.eps2 - self.eps1
        return delta_eps * 2 * j1(x) / x * np.pi * self.radius ** 2 / self.cell_area

    def build_epsilon_matrix(self):
        n = self.n_G
        G_diff = self.G_vectors[:, None, :] - self.G_vectors[None, :, :]
        G_norm = np.linalg.norm(G_diff, axis=2)

        f = np.pi * self.radius ** 2 / self.cell_area
        eps_G0 = self.eps1 * (1 - f) + self.eps2 * f

        x = G_norm * self.radius
        delta_eps = self.eps2 - self.eps1

        eps_matrix = np.zeros_like(G_norm, dtype=complex)

        mask = G_norm < 1e-12
        eps_matrix[mask] = eps_G0

        mask = ~mask
        x_safe = x[mask]
        eps_matrix[mask] = delta_eps * 2 * j1(x_safe) / x_safe * np.pi * self.radius ** 2 / self.cell_area

        return eps_matrix.astype(complex)

    def solve_eigenvalues(self, k_point, mode='TM', num_bands=None):
        n = self.n_G
        k_plus_G = k_point + self.G_vectors

        if mode == 'TM':
            K_diag = np.sum(k_plus_G ** 2, axis=1)
            K_matrix = np.diag(K_diag).astype(complex)
            eps_matrix = self.build_epsilon_matrix()
            eigenvalues, _ = linalg.eigh(K_matrix, eps_matrix)
        elif mode == 'TE':
            inv_eps = linalg.inv(self.build_epsilon_matrix())
            kGi_dot_kGj = np.dot(k_plus_G, k_plus_G.T)
            H_TE = inv_eps * kGi_dot_kGj
            eigenvalues, _ = linalg.eigh(H_TE)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        eigenvalues_sorted = np.sort(np.maximum(eigenvalues.real, 0))
        frequencies = np.sqrt(eigenvalues_sorted) * self.a / (2 * np.pi)

        if num_bands is not None:
            return frequencies[:num_bands]
        return frequencies

    def compute_band_structure(self, k_path, mode='both', num_bands=10, num_planes=None):
        if num_planes is None:
            num_planes = num_bands * 4
        self.generate_reciprocal_vectors(num_planes)

        n_k = len(k_path)
        bands_TM = np.zeros((n_k, num_bands)) if mode in ['TM', 'both'] else None
        bands_TE = np.zeros((n_k, num_bands)) if mode in ['TE', 'both'] else None

        for i, k in enumerate(k_path):
            if mode in ['TM', 'both']:
                freqs = self.solve_eigenvalues(k, 'TM')
                bands_TM[i, :] = freqs[:num_bands]
            if mode in ['TE', 'both']:
                freqs = self.solve_eigenvalues(k, 'TE')
                bands_TE[i, :] = freqs[:num_bands]

        return bands_TM, bands_TE

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

    def generate_k_path(self, n_points_per_segment=50):
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


def plot_band_structure(k_path, bands_TM, bands_TE, k_labels, title='光子晶体能带结构',
                        save_path=None, show_gaps=True):
    fig, ax = plt.subplots(figsize=(10, 7))

    n_segments = len(k_labels) - 1
    n_per_seg = (len(k_path) - 1) // n_segments
    x_positions = [i * n_per_seg for i in range(n_segments + 1)]
    x = np.arange(len(k_path))

    if bands_TM is not None:
        for band in range(bands_TM.shape[1]):
            ax.plot(x, bands_TM[:, band], 'b-', linewidth=2.5, label='TM 模式' if band == 0 else "")
        if show_gaps:
            pc = PhotonicCrystal2D()
            gaps = pc.analyze_gaps(bands_TM)
            for gap in gaps:
                ax.axhspan(gap['lower_edge'], gap['upper_edge'], facecolor='blue', alpha=0.15)

    if bands_TE is not None:
        for band in range(bands_TE.shape[1]):
            ax.plot(x, bands_TE[:, band], 'r--', linewidth=2.5, label='TE 模式' if band == 0 else "")
        if show_gaps:
            pc = PhotonicCrystal2D()
            gaps = pc.analyze_gaps(bands_TE)
            for gap in gaps:
                ax.axhspan(gap['lower_edge'], gap['upper_edge'], facecolor='red', alpha=0.15)

    for xi in x_positions:
        ax.axvline(x=xi, color='k', linestyle='-', alpha=0.6, linewidth=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(k_labels, fontsize=16)
    ax.set_ylabel('频率 (ωa/2πc)', fontsize=16)
    ax.set_title(title, fontsize=18, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=14)
    ax.set_ylim(0, None)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"能带图已保存到: {save_path}")

    return fig, ax


def print_gap_analysis(bands, mode_name):
    if bands is None:
        return
    pc = PhotonicCrystal2D()
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


def plot_dielectric_distribution(pc, save_path=None):
    nx, ny = 200, 200
    x = np.linspace(-pc.a, pc.a, nx)
    y = np.linspace(-pc.a, pc.a, ny)
    X, Y = np.meshgrid(x, y)

    eps = np.ones_like(X) * pc.eps1

    for i in range(-1, 2):
        for j in range(-1, 2):
            R = i * pc.a1 + j * pc.a2
            dist = np.sqrt((X - R[0]) ** 2 + (Y - R[1]) ** 2)
            eps[dist < pc.radius] = pc.eps2

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X, Y, eps, cmap='viridis', shading='auto')
    ax.set_xlabel('x (a)', fontsize=14)
    ax.set_ylabel('y (a)', fontsize=14)
    ax.set_title(f'介电常数分布 ({pc.lattice_type} 晶格)', fontsize=16)
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ε', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"介电常数分布图已保存到: {save_path}")

    return fig, ax


def main():
    print("=" * 60)
    print("二维光子晶体能带结构计算 - 平面波展开法 (PWE)")
    print("=" * 60)

    pc = PhotonicCrystal2D(
        lattice_type='triangular',
        a=1.0,
        eps1=1.0,
        eps2=12.0,
        radius=0.2
    )

    print(f"\n晶格类型: {pc.lattice_type}")
    print(f"晶格常数 a = {pc.a}")
    print(f"背景介电常数 ε1 = {pc.eps1}")
    print(f"圆柱介电常数 ε2 = {pc.eps2}")
    print(f"圆柱半径 r = {pc.radius}")
    print(f"填充因子 f = {np.pi * pc.radius**2 / pc.cell_area:.4f}")
    print(f"原胞面积 = {pc.cell_area:.4f}")

    print(f"\n基矢 a1 = {pc.a1}")
    print(f"基矢 a2 = {pc.a2}")
    print(f"倒格矢 b1 = {pc.b1}")
    print(f"倒格矢 b2 = {pc.b2}")

    num_bands = 8
    k_path = pc.generate_k_path(n_points_per_segment=30)
    k_labels = pc.get_k_labels()

    print(f"\n计算 {num_bands} 条能带，k 点数: {len(k_path)}")
    print("高对称点路径:", " → ".join(k_labels))

    print("\n正在计算能带结构...")
    bands_TM, bands_TE = pc.compute_band_structure(
        k_path, mode='both', num_bands=num_bands, num_planes=60
    )

    print("计算完成！")

    print_gap_analysis(bands_TM, "TM")
    print_gap_analysis(bands_TE, "TE")

    title = f'三角晶格光子晶体能带结构 (r={pc.radius}a, ε={pc.eps2}/{pc.eps1})'
    save_path = 'band_structure_triangular.png'
    plot_band_structure(k_path, bands_TM, bands_TE, k_labels, title=title, save_path=save_path)

    plot_dielectric_distribution(pc, save_path='dielectric_distribution.png')

    print("\n" + "=" * 60)
    print("方晶格对比计算...")
    print("=" * 60)

    pc_square = PhotonicCrystal2D(
        lattice_type='square',
        a=1.0,
        eps1=1.0,
        eps2=12.0,
        radius=0.2
    )

    k_path_sq = pc_square.generate_k_path(n_points_per_segment=30)
    k_labels_sq = pc_square.get_k_labels()
    bands_TM_sq, bands_TE_sq = pc_square.compute_band_structure(
        k_path_sq, mode='both', num_bands=num_bands, num_planes=60
    )

    print_gap_analysis(bands_TM_sq, "TM（方晶格）")
    print_gap_analysis(bands_TE_sq, "TE（方晶格）")

    title_sq = f'方晶格光子晶体能带结构 (r={pc_square.radius}a, ε={pc_square.eps2}/{pc_square.eps1})'
    save_path_sq = 'band_structure_square.png'
    plot_band_structure(k_path_sq, bands_TM_sq, bands_TE_sq, k_labels_sq,
                        title=title_sq, save_path=save_path_sq)

    print("\n所有计算完成！结果图像已保存。")


if __name__ == '__main__':
    main()
