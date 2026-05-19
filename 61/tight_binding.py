import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TightBinding:
    def __init__(self, lattice_vectors: np.ndarray, positions: Dict[str, np.ndarray],
                 hopping_params: Dict, on_site_energies: Dict[str, float]):
        self.lattice_vectors = np.array(lattice_vectors, dtype=np.float64)
        self.positions = {atom: np.array(pos, dtype=np.float64) for atom, pos in positions.items()}
        self.hopping_params = hopping_params
        self.on_site_energies = on_site_energies
        self.atoms = list(positions.keys())
        self.n_atoms = len(self.atoms)
        self.reciprocal_vectors = self._compute_reciprocal_vectors()
        self.supercell_vectors = self._generate_supercell_vectors()

    def _compute_reciprocal_vectors(self) -> np.ndarray:
        a1, a2, a3 = self.lattice_vectors
        volume = np.dot(a1, np.cross(a2, a3))
        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume
        b3 = 2 * np.pi * np.cross(a1, a2) / volume
        return np.array([b1, b2, b3])

    def _generate_supercell_vectors(self, max_distance: float = 5.0) -> List[np.ndarray]:
        vectors = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(0, 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    R = i * self.lattice_vectors[0] + j * self.lattice_vectors[1] + k * self.lattice_vectors[2]
                    if np.linalg.norm(R) <= max_distance:
                        vectors.append(R)
        return vectors

    def get_kpath(self, high_symmetry_points: List[Tuple[str, List[float]]],
                  n_points: int = 50, uniform_segment: bool = True) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
        labels_positions = []
        k_coords = []
        
        for label, k in high_symmetry_points:
            labels_positions.append(label)
            k_coords.append(np.array(k, dtype=np.float64))
        
        k_coords = np.array(k_coords)
        n_segments = len(high_symmetry_points) - 1
        
        k_points = []
        labels = []
        current_index = 0
        
        for i in range(n_segments):
            label_start = labels_positions[i]
            label_end = labels_positions[i + 1]
            k_start = k_coords[i]
            k_end = k_coords[i + 1]
            
            if uniform_segment:
                segment_n_points = n_points
            else:
                segment_distance = np.linalg.norm(k_end - k_start)
                total_distance = sum(np.linalg.norm(k_coords[j+1] - k_coords[j]) for j in range(n_segments))
                segment_n_points = max(2, int(np.round(n_points * segment_distance / total_distance * n_segments)))
            
            if i == 0:
                labels.append((current_index, label_start))
            
            if i == n_segments - 1:
                segment_points = np.linspace(k_start, k_end, segment_n_points, endpoint=True)
            else:
                segment_points = np.linspace(k_start, k_end, segment_n_points, endpoint=False)
            
            k_points.extend(segment_points)
            current_index += len(segment_points)
            
            if i < n_segments - 1 or n_segments == 1:
                labels.append((current_index - 1, label_end))
        
        return np.array(k_points), labels

    def _phase_factor(self, k: np.ndarray, dr: np.ndarray) -> complex:
        return np.exp(1j * np.dot(k, dr))

    def build_hamiltonian(self, k: np.ndarray) -> np.ndarray:
        H = np.zeros((self.n_atoms, self.n_atoms), dtype=np.complex128)
        k_cart = k @ self.reciprocal_vectors
        
        for i, atom_i in enumerate(self.atoms):
            H[i, i] = self.on_site_energies.get(atom_i, 0.0)
            
            for j, atom_j in enumerate(self.atoms):
                if isinstance(self.hopping_params, dict) and f"{atom_i}-{atom_j}" in self.hopping_params:
                    if isinstance(self.hopping_params[f"{atom_i}-{atom_j}"], (int, float)):
                        t = self.hopping_params[f"{atom_i}-{atom_j}"]
                        dr = self.positions[atom_j] - self.positions[atom_i]
                        H[i, j] += t * self._phase_factor(k_cart, dr)
                    elif isinstance(self.hopping_params[f"{atom_i}-{atom_j}"], list):
                        for t, R in self.hopping_params[f"{atom_i}-{atom_j}"]:
                            dr = self.positions[atom_j] - self.positions[atom_i] + R
                            H[i, j] += t * self._phase_factor(k_cart, dr)
        
        return H

    def solve(self, k_points: np.ndarray) -> np.ndarray:
        eigenvalues = []
        for k in k_points:
            H = self.build_hamiltonian(k)
            vals = np.linalg.eigvalsh(H)
            eigenvalues.append(vals)
        return np.array(eigenvalues)

    def calculate_bands(self, high_symmetry_points: List[Tuple[str, List[float]]],
                        n_points: int = 50, uniform_segment: bool = True,
                        uniform_axis: bool = True) -> Dict:
        k_points, labels = self.get_kpath(high_symmetry_points, n_points, uniform_segment)
        eigenvalues = self.solve(k_points)
        
        k_distances = np.cumsum(np.linalg.norm(np.diff(k_points, axis=0), axis=1))
        k_distances = np.insert(k_distances, 0, 0)
        
        if uniform_axis:
            label_indices = [idx for idx, _ in labels]
            
            new_k_distances = np.zeros_like(k_distances, dtype=np.float64)
            for i in range(len(label_indices) - 1):
                start_idx = label_indices[i]
                end_idx = label_indices[i + 1]
                segment_length = 1.0
                segment_points = end_idx - start_idx + 1
                new_k_distances[start_idx:end_idx + 1] = np.linspace(
                    i * segment_length, (i + 1) * segment_length, segment_points
                )
            
            adjusted_labels = []
            for i, (idx, label) in enumerate(labels):
                adjusted_labels.append((new_k_distances[idx], label))
            
            k_distances = new_k_distances
        else:
            adjusted_labels = [(k_distances[idx], label) for idx, label in labels]
        
        return {
            'k_distances': k_distances,
            'eigenvalues': eigenvalues,
            'labels': adjusted_labels,
            'k_points': k_points,
            'high_symmetry_points': high_symmetry_points
        }

    def plot_bands(self, bands: Dict, title: str = "能带结构", 
                   save_path: Optional[str] = None, show: bool = True):
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib未安装，无法绘图")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for band in range(bands['eigenvalues'].shape[1]):
            ax.plot(bands['k_distances'], bands['eigenvalues'][:, band], 'b-', linewidth=1.5)
        
        label_positions = [pos for pos, _ in bands['labels']]
        label_names = [label for _, label in bands['labels']]
        
        ax.set_xticks(label_positions)
        ax.set_xticklabels(label_names)
        
        for pos in label_positions:
            ax.axvline(x=pos, color='k', linestyle='--', linewidth=0.5)
        
        ax.set_ylabel('能量 (eV)', fontsize=12)
        ax.set_xlabel('高对称点', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


def example_graphene():
    a = 1.42
    lattice_vectors = np.array([
        [a * np.sqrt(3), 0, 0],
        [a * np.sqrt(3) / 2, 3 * a / 2, 0],
        [0, 0, 10]
    ])
    positions = {
        'A': [0, 0, 0],
        'B': [a * np.sqrt(3) / 2, a / 2, 0]
    }
    
    hopping_params = {}
    t1 = -2.8
    neighbor_vectors = [
        np.array([0, 0, 0]),
        np.array([-a * np.sqrt(3), 0, 0]),
        np.array([-a * np.sqrt(3) / 2, -3 * a / 2, 0]),
    ]
    hopping_params['A-B'] = [(t1, R) for R in neighbor_vectors]
    hopping_params['B-A'] = [(t1, -R) for R in neighbor_vectors]
    
    on_site_energies = {'A': 0.0, 'B': 0.0}
    tb = TightBinding(lattice_vectors, positions, hopping_params, on_site_energies)
    
    k_path = [
        ('Γ', [0, 0, 0]),
        ('M', [1/2, 0, 0]),
        ('K', [1/3, 1/3, 0]),
        ('Γ', [0, 0, 0])
    ]
    bands = tb.calculate_bands(k_path, n_points=100)
    return tb, bands


def example_square_lattice():
    a = 1.0
    lattice_vectors = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, 10]
    ])
    positions = {'A': [0, 0, 0]}
    
    hopping_params = {}
    t1 = -1.0
    neighbor_vectors = [
        np.array([a, 0, 0]),
        np.array([-a, 0, 0]),
        np.array([0, a, 0]),
        np.array([0, -a, 0]),
    ]
    hopping_params['A-A'] = [(t1, R) for R in neighbor_vectors]
    
    on_site_energies = {'A': 0.0}
    tb = TightBinding(lattice_vectors, positions, hopping_params, on_site_energies)
    
    k_path = [
        ('Γ', [0, 0, 0]),
        ('X', [1/2, 0, 0]),
        ('M', [1/2, 1/2, 0]),
        ('Γ', [0, 0, 0])
    ]
    bands = tb.calculate_bands(k_path, n_points=100)
    return tb, bands


if __name__ == "__main__":
    print("=" * 50)
    print("石墨烯能带计算示例")
    print("=" * 50)
    tb_graphene, bands_graphene = example_graphene()
    print(f"k点数: {len(bands_graphene['k_distances'])}")
    print(f"能级数: {bands_graphene['eigenvalues'].shape[1]}")
    print("\n高对称点位置:")
    for pos, label in bands_graphene['labels']:
        print(f"  {label}: x坐标 = {pos:.3f}")
    
    print("\n" + "=" * 50)
    print("简单立方晶格能带计算示例")
    print("=" * 50)
    tb_square, bands_square = example_square_lattice()
    print(f"k点数: {len(bands_square['k_distances'])}")
    print(f"能级数: {bands_square['eigenvalues'].shape[1]}")
    
    print("\n" + "=" * 50)
    print("k点路径修复说明:")
    print("=" * 50)
    print("1. uniform_segment=True: 每段高对称点之间使用相同数量的k点")
    print("2. uniform_axis=True: 横坐标均匀归一化，每段视觉长度相同")
    print("3. 每段内的k点通过linspace插值，确保段内等间距")
    print("4. labels格式更新为 (x坐标位置, 标签名)，便于直接使用")
    
    print("\n能带计算完成！已生成能带图数据。")


class WannierInterpolator:
    """
    瓦尼尔函数插值类，用于从粗k网格的第一性原理计算结果拟合紧束缚模型，
    然后快速插值到任意精细k点网格，大幅加速能带计算。
    
    使用示例:
    --------
    # 1. 准备粗k网格和第一性原理能带
    k_mesh = np.array([[kx1, ky1, kz1], [kx2, ky2, kz2], ...])
    eigenvalues = np.array([[e1_k1, e2_k1, ...], [e1_k2, e2_k2, ...], ...])
    
    # 2. 创建插值器并拟合
    wannier = WannierInterpolator(lattice_vectors, n_bands=4, n_wannier=4)
    wannier.fit(k_mesh, eigenvalues, n_shells=3)
    
    # 3. 插值到精细高对称路径
    k_path = [('Γ', [0,0,0]), ('X', [0.5,0,0]), ...]
    bands = wannier.interpolate_bands(k_path, n_points=200)
    
    # 4. 绘制能带图
    wannier.plot_bands(bands)
    """
    
    def __init__(self, lattice_vectors: np.ndarray, n_bands: int, n_wannier: int = None):
        self.lattice_vectors = np.array(lattice_vectors, dtype=np.float64)
        self.n_bands = n_bands
        self.n_wannier = n_wannier if n_wannier is not None else n_bands
        self.reciprocal_vectors = self._compute_reciprocal_vectors()
        self.hopping_matrix = None
        self.r_vectors = None
        self.is_fitted = False

    def _compute_reciprocal_vectors(self) -> np.ndarray:
        a1, a2, a3 = self.lattice_vectors
        volume = np.dot(a1, np.cross(a2, a3))
        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume
        b3 = 2 * np.pi * np.cross(a1, a2) / volume
        return np.array([b1, b2, b3])

    def generate_r_vectors(self, n_shells: int = 3) -> np.ndarray:
        r_vectors = []
        for i in range(-n_shells, n_shells + 1):
            for j in range(-n_shells, n_shells + 1):
                for k in range(-n_shells, n_shells + 1):
                    R = i * self.lattice_vectors[0] + j * self.lattice_vectors[1] + k * self.lattice_vectors[2]
                    r_vectors.append(R)
        self.r_vectors = np.array(r_vectors)
        return self.r_vectors

    def _construct_hamiltonian_k(self, k: np.ndarray) -> np.ndarray:
        H = np.zeros((self.n_wannier, self.n_wannier), dtype=np.complex128)
        k_cart = k @ self.reciprocal_vectors
        for idx, R in enumerate(self.r_vectors):
            phase = np.exp(1j * np.dot(k_cart, R))
            H += phase * self.hopping_matrix[idx]
        return H

    def fit(self, k_mesh: np.ndarray, eigenvalues: np.ndarray, 
            n_shells: int = 3, method: str = 'least_squares') -> None:
        if self.r_vectors is None:
            self.generate_r_vectors(n_shells)
        
        n_k = len(k_mesh)
        n_r = len(self.r_vectors)
        
        phase_factors = np.zeros((n_k, n_r), dtype=np.complex128)
        for ik, k in enumerate(k_mesh):
            k_cart = k @ self.reciprocal_vectors
            for ir, R in enumerate(self.r_vectors):
                phase_factors[ik, ir] = np.exp(1j * np.dot(k_cart, R))
        
        H_fit = np.zeros((n_r, self.n_wannier, self.n_wannier), dtype=np.complex128)
        
        for n in range(min(self.n_wannier, eigenvalues.shape[1])):
            A = np.zeros((n_k, n_r), dtype=np.float64)
            b = np.zeros(n_k, dtype=np.float64)
            
            for ik in range(n_k):
                A[ik, :] = np.real(phase_factors[ik, :])
                b[ik] = eigenvalues[ik, n]
            
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            for ir in range(n_r):
                H_fit[ir, n, n] = x[ir]
        
        self.hopping_matrix = H_fit
        
        self.is_fitted = True
        
        predicted = self.interpolate(k_mesh)
        error = np.mean(np.abs(predicted - eigenvalues[:, :self.n_wannier]))
        print(f"拟合完成！平均误差: {error:.6e} eV")

    def plot_bands(self, bands: Dict, title: str = "瓦尼尔插值能带结构", 
                   save_path: Optional[str] = None, show: bool = True):
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib未安装，无法绘图")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for band in range(bands['eigenvalues'].shape[1]):
            ax.plot(bands['k_distances'], bands['eigenvalues'][:, band], 'r-', 
                    linewidth=1.5, label='瓦尼尔插值' if band == 0 else "")
        
        label_positions = [pos for pos, _ in bands['labels']]
        label_names = [label for _, label in bands['labels']]
        
        ax.set_xticks(label_positions)
        ax.set_xticklabels(label_names)
        
        for pos in label_positions:
            ax.axvline(x=pos, color='k', linestyle='--', linewidth=0.5)
        
        ax.set_ylabel('能量 (eV)', fontsize=12)
        ax.set_xlabel('高对称点', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig

    @classmethod
    def generate_tb_eigenvalues(cls, tb_model: TightBinding, k_mesh: np.ndarray) -> np.ndarray:
        eigenvalues = []
        for k in k_mesh:
            H = tb_model.build_hamiltonian(k)
            vals = np.linalg.eigvalsh(H)
            eigenvalues.append(vals)
        return np.array(eigenvalues)

    def interpolate(self, k_points: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法进行拟合")
        
        eigenvalues = []
        for k in k_points:
            H = self._construct_hamiltonian_k(k)
            vals = np.linalg.eigvalsh(H)
            eigenvalues.append(vals)
        return np.array(eigenvalues)

    def interpolate_bands(self, high_symmetry_points: List[Tuple[str, List[float]]],
                          n_points: int = 100, uniform_segment: bool = True,
                          uniform_axis: bool = True) -> Dict:
        from copy import deepcopy
        tb_dummy = TightBinding(self.lattice_vectors, {'A': [0,0,0]}, {}, {})
        k_points, labels = tb_dummy.get_kpath(high_symmetry_points, n_points, uniform_segment)
        
        eigenvalues = self.interpolate(k_points)
        
        k_distances = np.cumsum(np.linalg.norm(np.diff(k_points, axis=0), axis=1))
        k_distances = np.insert(k_distances, 0, 0)
        
        if uniform_axis:
            label_indices = []
            current_idx = 0
            for i in range(len(high_symmetry_points) - 1):
                if i == 0:
                    label_indices.append(current_idx)
                if uniform_segment:
                    segment_n = n_points if i < len(high_symmetry_points) - 2 else n_points
                else:
                    segment_n = n_points
                current_idx += segment_n - 1
                label_indices.append(current_idx)
            
            new_k_distances = np.zeros_like(k_distances, dtype=np.float64)
            for i in range(len(label_indices) - 1):
                start_idx = label_indices[i]
                end_idx = label_indices[i + 1]
                segment_points = end_idx - start_idx + 1
                new_k_distances[start_idx:end_idx + 1] = np.linspace(i, i + 1, segment_points)
            
            adjusted_labels = []
            for i, (pos_idx, (_, label)) in enumerate(zip(label_indices, high_symmetry_points)):
                adjusted_labels.append((new_k_distances[pos_idx], label))
            
            k_distances = new_k_distances
        else:
            adjusted_labels = [(k_distances[i], label) for i, (_, label) in enumerate(high_symmetry_points)]
        
        return {
            'k_distances': k_distances,
            'eigenvalues': eigenvalues,
            'labels': adjusted_labels,
            'k_points': k_points
        }


def example_wannier_interpolation():
    print("=" * 60)
    print("瓦尼尔函数插值示例")
    print("=" * 60)
    
    a = 1.0
    lattice_vectors = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, 10]
    ])
    
    print("\n步骤1: 生成粗网格第一性原理能带数据")
    n_kx, n_ky = 6, 6
    k_mesh = []
    for i in range(n_kx):
        for j in range(n_ky):
            kx = i / n_kx - 0.5
            ky = j / n_ky - 0.5
            k_mesh.append([kx, ky, 0])
    k_mesh = np.array(k_mesh)
    
    mock_eigenvalues = np.zeros((len(k_mesh), 1))
    for ik, k in enumerate(k_mesh):
        kx, ky = k[0], k[1]
        e = -2 * (np.cos(2 * np.pi * kx * a) + np.cos(2 * np.pi * ky * a))
        mock_eigenvalues[ik, 0] = e
    
    print(f"  粗网格k点数: {len(k_mesh)}")
    print(f"  k点网格: {n_kx} x {n_ky} = {len(k_mesh)}")
    
    print("\n步骤2: 瓦尼尔插值拟合")
    wannier = WannierInterpolator(lattice_vectors, n_bands=1, n_wannier=1)
    wannier.fit(k_mesh, mock_eigenvalues, n_shells=1)
    
    print("\n步骤3: 沿高对称路径插值计算能带")
    k_path = [
        ('Γ', [0, 0, 0]),
        ('X', [0.5, 0, 0]),
        ('M', [0.5, 0.5, 0]),
        ('Γ', [0, 0, 0])
    ]
    
    bands_fine = wannier.interpolate_bands(k_path, n_points=100)
    print(f"  插值后k点数: {len(bands_fine['k_distances'])}")
    print(f"  能级数: {bands_fine['eigenvalues'].shape[1]}")
    
    print("\n瓦尼尔插值完成！")
    print(f"  加速比: {len(bands_fine['k_distances']) / len(k_mesh):.1f}x")
    
    return wannier, bands_fine


def example_wannier_with_tb():
    print("=" * 60)
    print("瓦尼尔插值 + 紧束缚模型完整示例")
    print("=" * 60)
    
    a = 1.42
    lattice_vectors = np.array([
        [a * np.sqrt(3), 0, 0],
        [a * np.sqrt(3) / 2, 3 * a / 2, 0],
        [0, 0, 10]
    ])
    positions = {
        'A': [0, 0, 0],
        'B': [a * np.sqrt(3) / 2, a / 2, 0]
    }
    hopping_params = {'A-B': -2.8, 'B-A': -2.8}
    on_site_energies = {'A': 0.0, 'B': 0.0}
    
    tb = TightBinding(lattice_vectors, positions, hopping_params, on_site_energies)
    
    print("\n步骤1: 在粗k网格上计算能带（模拟第一性原理计算）")
    n_kx, n_ky = 4, 4
    k_mesh = []
    for i in range(n_kx):
        for j in range(n_ky):
            kx = i / n_kx - 0.5
            ky = j / n_ky - 0.5
            k_mesh.append([kx, ky, 0])
    k_mesh = np.array(k_mesh)
    
    eigenvalues_coarse = tb.solve(k_mesh)
    print(f"  粗网格k点数: {len(k_mesh)} ({n_kx} x {n_ky})")
    print(f"  能级数: {eigenvalues_coarse.shape[1]}")
    
    print("\n步骤2: 用瓦尼尔插值拟合粗网格数据")
    wannier = WannierInterpolator(lattice_vectors, n_bands=2, n_wannier=2)
    wannier.fit(k_mesh, eigenvalues_coarse, n_shells=2)
    
    print("\n步骤3: 沿高对称路径插值计算精细能带")
    k_path = [
        ('Γ', [0, 0, 0]),
        ('M', [0.5, 0, 0]),
        ('K', [1/3, 1/3, 0]),
        ('Γ', [0, 0, 0])
    ]
    
    bands_fine = wannier.interpolate_bands(k_path, n_points=100)
    print(f"  插值后k点数: {len(bands_fine['k_distances'])}")
    
    bands_direct = tb.calculate_bands(k_path, n_points=100)
    error = np.mean(np.abs(bands_fine['eigenvalues'] - bands_direct['eigenvalues']))
    print(f"  插值误差: {error:.6f} eV")
    
    print(f"\n瓦尼尔插值完成！加速比: {len(bands_fine['k_distances']) / len(k_mesh):.1f}x")
    
    return wannier, bands_fine, bands_direct


if __name__ == "__main__":
    example_wannier_with_tb()
