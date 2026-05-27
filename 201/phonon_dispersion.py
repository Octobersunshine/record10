import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class Crystal:
    def __init__(self, lattice: np.ndarray, positions: np.ndarray, 
                 symbols: List[str], masses: np.ndarray):
        self.lattice = np.array(lattice, dtype=np.float64)
        self.positions = np.array(positions, dtype=np.float64)
        self.symbols = symbols
        self.masses = np.array(masses, dtype=np.float64)
        self.n_atoms = len(positions)
        self.reciprocal_lattice = 2 * np.pi * np.linalg.inv(self.lattice.T)

    def get_cartesian_positions(self) -> np.ndarray:
        return self.positions @ self.lattice


class ForceConstants:
    def __init__(self, crystal: Crystal, cutoff: float = 5.0):
        self.crystal = crystal
        self.cutoff = cutoff
        self.fc_matrix = None
        self.fc_with_R = None
    
    def _compute_pairwise_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        cart_pos = self.crystal.get_cartesian_positions()
        n = self.crystal.n_atoms
        distances = np.zeros((n, n, 3, 3, 3, 3))
        lattice = self.crystal.lattice
        
        for i in range(n):
            for j in range(n):
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        for c in range(-1, 2):
                            cell_shift = np.array([a, b, c]) @ lattice
                            dist = cart_pos[j] + cell_shift - cart_pos[i]
                            distances[i, j, a+1, b+1, c+1] = dist
        return distances, cart_pos
    
    def generate_model_fc(self, spring_constant: float = 15.0) -> np.ndarray:
        n = self.crystal.n_atoms
        fc = np.zeros((n, n, 3, 3))
        distances, cart_pos = self._compute_pairwise_distances()
        
        for i in range(n):
            for j in range(n):
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            dist_vec = distances[i, j, a, b, c]
                            dist = np.linalg.norm(dist_vec)
                            if 1e-6 < dist < self.cutoff:
                                unit_vec = dist_vec / dist
                                fc_tensor = spring_constant * np.outer(unit_vec, unit_vec)
                                fc[i, j] += fc_tensor
        
        self.fc_with_R = self._compute_fc_with_R(spring_constant)
        
        self.fc_matrix = fc
        return fc
    
    def _compute_fc_with_R(self, spring_constant: float) -> dict:
        n = self.crystal.n_atoms
        cart_pos = self.crystal.get_cartesian_positions()
        lattice = self.crystal.lattice
        
        fc_with_R = {}
        
        for i in range(n):
            for j in range(n):
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        for c in range(-1, 2):
                            R = np.array([a, b, c])
                            cell_shift = R @ lattice
                            dist_vec = cart_pos[j] + cell_shift - cart_pos[i]
                            dist = np.linalg.norm(dist_vec)
                            
                            if 1e-6 < dist < self.cutoff:
                                unit_vec = dist_vec / dist
                                fc_tensor = spring_constant * np.outer(unit_vec, unit_vec)
                                
                                key = (i, j, a, b, c)
                                fc_with_R[key] = fc_tensor
        
        return fc_with_R
    
    def get_fc_for_dynamical_matrix(self, i: int, j: int, 
                                     a: int, b: int, c: int) -> np.ndarray:
        if self.fc_with_R is not None:
            key = (i, j, a, b, c)
            return self.fc_with_R.get(key, np.zeros((3, 3)))
        else:
            if a == 0 and b == 0 and c == 0:
                return self.fc_matrix[i, j]
            else:
                return np.zeros((3, 3))
    
    def load_fc_from_file(self, filename: str):
        data = np.load(filename, allow_pickle=True)
        self.fc_matrix = data['force_constants']
        if 'fc_with_R' in data.files:
            self.fc_with_R = data['fc_with_R'].item()
        return self.fc_matrix
    
    def save_fc_to_file(self, filename: str):
        if self.fc_with_R is not None:
            np.savez(filename, force_constants=self.fc_matrix, fc_with_R=self.fc_with_R)
        else:
            np.savez(filename, force_constants=self.fc_matrix)


class PhononCalculator:
    def __init__(self, crystal: Crystal, force_constants: ForceConstants,
                 use_asr: bool = True, regularization: str = 'none',
                 reg_strength: float = 1e-4, symmetrize_fc: bool = True):
        self.crystal = crystal
        self.force_constants = force_constants
        self.fc = force_constants.fc_matrix
        self.masses = crystal.masses
        self.n_atoms = crystal.n_atoms
        self.use_asr = use_asr
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.symmetrize_fc = symmetrize_fc
        
        if symmetrize_fc and self.fc is not None:
            self.fc = self._symmetrize_force_constants(self.fc)
        if use_asr and self.fc is not None:
            self.fc = self._enforce_asr(self.fc)
    
    def _symmetrize_force_constants(self, fc: np.ndarray) -> np.ndarray:
        n = self.n_atoms
        fc_sym = np.copy(fc)
        for i in range(n):
            for j in range(n):
                fc_sym[i, j] = 0.5 * (fc[i, j] + fc[j, i].T)
        print(f"Force constants symmetrized: Phi_ij = Phi_ji^T")
        return fc_sym
    
    def _enforce_asr(self, fc: np.ndarray) -> np.ndarray:
        if self.force_constants.fc_with_R is not None:
            print(f"ASR is enforced automatically with fc_with_R structure")
            return fc
        
        n = self.n_atoms
        fc_asr = np.copy(fc)
        for i in range(n):
            fc_asr[i, i] = -np.sum(fc_asr[i, :], axis=0) + fc_asr[i, i]
        print(f"Acoustic Sum Rule enforced: sum_j Phi_ij = 0")
        return fc_asr
    
    def _apply_regularization(self, D: np.ndarray, q: np.ndarray) -> np.ndarray:
        n = self.n_atoms
        D_reg = np.copy(D)
        
        if self.regularization == 'none':
            return D_reg
        
        elif self.regularization == 'simple':
            for i in range(3 * n):
                D_reg[i, i] += self.reg_strength
                
        elif self.regularization == 'near_gamma':
            q_norm = np.linalg.norm(q)
            gamma_threshold = 1e-4
            if q_norm < gamma_threshold:
                alpha = self.reg_strength * (1.0 - q_norm / gamma_threshold)
                for i in range(3 * n):
                    D_reg[i, i] += alpha
                    
        elif self.regularization == 'projected':
            q_cart = q @ self.crystal.reciprocal_lattice
            q_unit = q_cart / (np.linalg.norm(q_cart) + 1e-10)
            
            projection = np.zeros((3 * n, 3 * n), dtype=np.complex128)
            for i in range(n):
                for alpha in range(3):
                    for beta in range(3):
                        projection[3*i+alpha, 3*i+beta] = q_unit[alpha] * q_unit[beta]
            
            q_norm = np.linalg.norm(q)
            gamma_threshold = 1e-4
            if q_norm < gamma_threshold:
                alpha = self.reg_strength * (1.0 - q_norm / gamma_threshold)
                D_reg += alpha * projection
                
        elif self.regularization == 'full':
            q_norm = np.linalg.norm(q)
            gamma_threshold = 1e-4
            if q_norm < gamma_threshold:
                alpha = self.reg_strength * (1.0 - q_norm / gamma_threshold)
                for i in range(3 * n):
                    D_reg[i, i] += alpha
                
                q_cart = q @ self.crystal.reciprocal_lattice
                q_unit = q_cart / (np.linalg.norm(q_cart) + 1e-10)
                projection = np.zeros((3 * n, 3 * n), dtype=np.complex128)
                for i in range(n):
                    for alpha in range(3):
                        for beta in range(3):
                            projection[3*i+alpha, 3*i+beta] = q_unit[alpha] * q_unit[beta]
                D_reg += alpha * projection
            
        return D_reg
    
    def build_dynamical_matrix(self, q: np.ndarray) -> np.ndarray:
        n = self.n_atoms
        D = np.zeros((3 * n, 3 * n), dtype=np.complex128)
        lattice = self.crystal.lattice
        cart_pos = self.crystal.get_cartesian_positions()
        
        for i in range(n):
            for j in range(n):
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        for c in range(-1, 2):
                            cell_shift = np.array([a, b, c]) @ lattice
                            phase = np.exp(1j * np.dot(q, cart_pos[j] + cell_shift - cart_pos[i]))
                            fc_ijR = self.force_constants.get_fc_for_dynamical_matrix(i, j, a, b, c)
                            D[3*i:3*i+3, 3*j:3*j+3] += phase * fc_ijR / np.sqrt(self.masses[i] * self.masses[j])
        
        D = 0.5 * (D + D.conj().T)
        
        if self.regularization != 'none':
            D = self._apply_regularization(D, q)
        
        return D

    def compute_frequencies(self, q: np.ndarray, 
                            convert_to_ghz: bool = True,
                            convert_to_cm: bool = False,
                            zero_negative: bool = True) -> np.ndarray:
        D = self.build_dynamical_matrix(q)
        eigvals = np.linalg.eigvalsh(D)
        
        if zero_negative:
            eigvals = np.where(eigvals < 0, 0, eigvals)
        frequencies = np.sqrt(eigvals)
        
        if convert_to_cm:
            thz_to_cm = 33.356
            eV_to_thz = 241.798926
            ang = 1e-10
            amu = 1.66054e-27
            evang2_to_kg = 1.602176634e-19 / (ang * ang)
            factor = np.sqrt(evang2_to_kg / amu) / (2 * np.pi * 1e12)
            frequencies = frequencies * factor * thz_to_cm
        elif convert_to_ghz:
            ang = 1e-10
            amu = 1.66054e-27
            evang2_to_kg = 1.602176634e-19 / (ang * ang)
            factor = np.sqrt(evang2_to_kg / amu) / (2 * np.pi * 1e9)
            frequencies = frequencies * factor
        
        return frequencies
    
    def check_imaginary_frequencies(self, q_path: List[np.ndarray], 
                                     threshold: float = -1e-10) -> dict:
        imaginary_modes = []
        for i, q in enumerate(q_path):
            D = self.build_dynamical_matrix(q)
            eigvals = np.linalg.eigvalsh(D)
            for mode_idx, ev in enumerate(eigvals):
                if ev < threshold:
                    imaginary_modes.append({
                        'q_index': i,
                        'q_vector': q,
                        'mode': mode_idx,
                        'eigenvalue': ev,
                        'frequency_cm': np.sqrt(abs(ev)) * 33.356
                    })
        
        result = {
            'total_modes': len(q_path) * 3 * self.n_atoms,
            'imaginary_count': len(imaginary_modes),
            'imaginary_modes': imaginary_modes,
            'max_imaginary_freq': max([m['frequency_cm'] for m in imaginary_modes], default=0)
        }
        return result
    
    def print_imaginary_analysis(self, path_points: List[np.ndarray],
                                  n_points_per_segment: int = 50):
        print("\n" + "=" * 60)
        print("Imaginary Frequency Analysis")
        print("=" * 60)
        
        rec_lattice = self.crystal.reciprocal_lattice
        q_path = []
        path_points_cart = [q @ rec_lattice for q in path_points]
        
        for i in range(len(path_points_cart) - 1):
            segment = np.linspace(path_points_cart[i], path_points_cart[i+1], 
                                  n_points_per_segment, endpoint=False)
            q_path.extend(segment)
        q_path.append(path_points_cart[-1])
        
        result = self.check_imaginary_frequencies(q_path)
        
        print(f"Total modes checked: {result['total_modes']}")
        print(f"Imaginary modes found: {result['imaginary_count']}")
        
        if result['imaginary_modes']:
            print(f"\nMax imaginary frequency: {result['max_imaginary_freq']:.4f} cm^-1")
            print("\nDetails of imaginary modes:")
            for mode in result['imaginary_modes'][:10]:
                print(f"  q={mode['q_vector']}, mode {mode['mode']}: "
                      f"{mode['frequency_cm']:.4f} cm^-1 (ev={mode['eigenvalue']:.2e})")
            if len(result['imaginary_modes']) > 10:
                print(f"  ... and {len(result['imaginary_modes']) - 10} more")
        else:
            print("No imaginary frequencies detected!")
        
        return result

    def compute_band_structure(self, path_points: List[np.ndarray], 
                               n_points_per_segment: int = 50,
                               use_reduced_coords: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        q_path = []
        distances = []
        rec_lattice = self.crystal.reciprocal_lattice
        
        path_points_cart = []
        for q in path_points:
            if use_reduced_coords:
                q_cart = q @ rec_lattice
            else:
                q_cart = q
            path_points_cart.append(q_cart)
        
        cumulative_dist = 0.0
        for i in range(len(path_points_cart) - 1):
            start = path_points_cart[i]
            end = path_points_cart[i + 1]
            segment = np.linspace(start, end, n_points_per_segment, endpoint=False)
            q_path.extend(segment)
            
            dist = np.linalg.norm(end - start)
            segment_distances = cumulative_dist + np.linspace(0, dist, n_points_per_segment, endpoint=False)
            distances.extend(segment_distances)
            cumulative_dist += dist
        
        q_path.append(path_points_cart[-1])
        distances.append(cumulative_dist)
        
        frequencies = []
        for q in q_path:
            freqs = self.compute_frequencies(q, convert_to_cm=True)
            frequencies.append(freqs)
        
        return np.array(distances), np.array(frequencies)


class HighSymmetryPaths:
    @staticmethod
    def get_fcc_path() -> Tuple[List[np.ndarray], List[str]]:
        G = np.array([0.0, 0.0, 0.0])
        X = np.array([0.5, 0.0, 0.5])
        W = np.array([0.5, 0.25, 0.75])
        K = np.array([0.375, 0.375, 0.75])
        L = np.array([0.5, 0.5, 0.5])
        U = np.array([0.25, 0.625, 0.625])
        return [G, X, W, K, G, L, U, W, L], ['Γ', 'X', 'W', 'K', 'Γ', 'L', 'U', 'W', 'L']

    @staticmethod
    def get_bcc_path() -> Tuple[List[np.ndarray], List[str]]:
        G = np.array([0.0, 0.0, 0.0])
        H = np.array([0.5, -0.5, 0.5])
        P = np.array([0.25, 0.25, 0.25])
        N = np.array([0.0, 0.0, 0.5])
        return [G, H, P, G, N], ['Γ', 'H', 'P', 'Γ', 'N']

    @staticmethod
    def get_sc_path() -> Tuple[List[np.ndarray], List[str]]:
        G = np.array([0.0, 0.0, 0.0])
        X = np.array([0.5, 0.0, 0.0])
        M = np.array([0.5, 0.5, 0.0])
        R = np.array([0.5, 0.5, 0.5])
        return [G, X, M, G, R, X, M, R], ['Γ', 'X', 'M', 'Γ', 'R', 'X', 'M', 'R']


class PhononVisualizer:
    @staticmethod
    def plot_dispersion(distances: np.ndarray, frequencies: np.ndarray,
                        high_sym_labels: List[str],
                        title: str = 'Phonon Dispersion',
                        ylabel: str = 'Frequency (cm^-1)',
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6)):
        n_bands = frequencies.shape[1]
        fig, ax = plt.subplots(figsize=figsize)
        
        for band in range(n_bands):
            ax.plot(distances, frequencies[:, band], 'b-', linewidth=1.5)
        
        n_points = len(high_sym_labels)
        tick_positions = np.linspace(distances[0], distances[-1], n_points)
        
        for pos in tick_positions:
            ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(high_sym_labels, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(distances[0], distances[-1])
        
        y_min, y_max = np.min(frequencies), np.max(frequencies)
        ax.set_ylim(y_min - 5, y_max * 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()

    @staticmethod
    def plot_dos(frequencies: np.ndarray, dos: np.ndarray,
                 title: str = 'Phonon Density of States',
                 save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(frequencies, dos, 'r-', linewidth=2)
        ax.fill_between(frequencies, dos, alpha=0.3)
        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=14)
        ax.set_ylabel('Density of States', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def compute_dos(phonon_calc: PhononCalculator, n_mesh: int = 20, 
                sigma: float = 5.0, n_freq_bins: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    rec_lat = phonon_calc.crystal.reciprocal_lattice
    qpoints = []
    
    for i in range(n_mesh):
        for j in range(n_mesh):
            for k in range(n_mesh):
                q = np.array([i/n_mesh - 0.5, j/n_mesh - 0.5, k/n_mesh - 0.5]) @ rec_lat
                qpoints.append(q)
    
    all_freqs = []
    for q in qpoints:
        freqs = phonon_calc.compute_frequencies(q, convert_to_cm=True)
        all_freqs.extend(freqs)
    
    all_freqs = np.array(all_freqs)
    freq_min, freq_max = np.min(all_freqs), np.max(all_freqs)
    freq_axis = np.linspace(freq_min - 10, freq_max + 10, n_freq_bins)
    dos = np.zeros_like(freq_axis)
    
    for f in all_freqs:
        dos += np.exp(-(freq_axis - f)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    dos /= len(qpoints)
    return freq_axis, dos


def example_fcc_calculation():
    print("=" * 60)
    print("FCC 晶格声子色散谱计算示例")
    print("=" * 60)
    
    a = 5.43
    lattice = np.array([
        [0.0, a/2, a/2],
        [a/2, 0.0, a/2],
        [a/2, a/2, 0.0]
    ])
    
    positions = np.array([[0.0, 0.0, 0.0]])
    symbols = ['Si']
    masses = np.array([28.0855])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    
    print(f"\n晶格常数: {a} Angstrom")
    print(f"原子数: {crystal.n_atoms}")
    print(f"原子质量: {crystal.masses[0]} u")
    
    fc = ForceConstants(crystal, cutoff=5.0)
    fc.generate_model_fc(spring_constant=20.0)
    print(f"\n力常数矩阵形状: {fc.fc_matrix.shape}")
    
    phonon = PhononCalculator(crystal, fc)
    
    print("\nGamma点频率计算:")
    gamma_point = np.array([0.0, 0.0, 0.0])
    freqs_gamma = phonon.compute_frequencies(gamma_point, convert_to_cm=True)
    print(f"  {freqs_gamma} cm^-1")
    
    print("\n构建高对称路径...")
    path_points, labels = HighSymmetryPaths.get_fcc_path()
    
    print("计算声子色散谱...")
    distances, frequencies = phonon.compute_band_structure(path_points, n_points_per_segment=100)
    
    print(f"频率范围: {np.min(frequencies):.2f} - {np.max(frequencies):.2f} cm^-1")
    
    PhononVisualizer.plot_dispersion(
        distances, frequencies, labels,
        title='FCC Phonon Dispersion (Model Force Constants)',
        ylabel='Frequency (cm^-1)',
        save_path='fcc_phonon_dispersion.png'
    )
    
    print("\n计算声子态密度...")
    freq_axis, dos = compute_dos(phonon, n_mesh=15, sigma=3.0)
    PhononVisualizer.plot_dos(
        freq_axis, dos,
        title='FCC Phonon Density of States',
        save_path='fcc_phonon_dos.png'
    )
    
    print("\n计算完成!")
    return crystal, fc, phonon, distances, frequencies


def example_sc_calculation():
    print("\n" + "=" * 60)
    print("简单立方 (SC) 晶格声子色散谱计算")
    print("=" * 60)
    
    a = 3.0
    lattice = np.eye(3) * a
    positions = np.array([[0.0, 0.0, 0.0]])
    symbols = ['X']
    masses = np.array([50.0])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    
    fc = ForceConstants(crystal, cutoff=4.0)
    fc.generate_model_fc(spring_constant=15.0)
    
    phonon = PhononCalculator(crystal, fc)
    
    path_points, labels = HighSymmetryPaths.get_sc_path()
    distances, frequencies = phonon.compute_band_structure(path_points, n_points_per_segment=80)
    
    PhononVisualizer.plot_dispersion(
        distances, frequencies, labels,
        title='Simple Cubic Phonon Dispersion',
        ylabel='Frequency (cm^-1)',
        save_path='sc_phonon_dispersion.png'
    )
    
    print("\nSC 计算完成!")
    return crystal, fc, phonon


def load_real_fc_and_calculate(fc_file: str, lattice: np.ndarray, 
                                positions: np.ndarray, symbols: List[str],
                                masses: np.ndarray):
    crystal = Crystal(lattice, positions, symbols, masses)
    fc = ForceConstants(crystal)
    fc.load_fc_from_file(fc_file)
    
    phonon = PhononCalculator(crystal, fc)
    path_points, labels = HighSymmetryPaths.get_fcc_path()
    distances, frequencies = phonon.compute_band_structure(path_points)
    
    PhononVisualizer.plot_dispersion(
        distances, frequencies, labels,
        title='Phonon Dispersion from DFT Force Constants'
    )
    
    return phonon


class PhonopyInterface:
    @staticmethod
    def export_to_phonopy(crystal: Crystal, fc_matrix: np.ndarray, 
                          supercell_matrix: Optional[np.ndarray] = None,
                          output_dir: str = 'phonopy_output'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if supercell_matrix is None:
            supercell_matrix = np.eye(3, dtype=int)
        
        poscar_path = os.path.join(output_dir, 'POSCAR')
        with open(poscar_path, 'w') as f:
            f.write(f"{' '.join(crystal.symbols)}\n")
            f.write("1.0\n")
            for row in crystal.lattice:
                f.write(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n")
            f.write(f"{' '.join(crystal.symbols)}\n")
            f.write(f"{' '.join(['1'] * crystal.n_atoms)}\n")
            f.write("Direct\n")
            for pos in crystal.positions:
                f.write(f"  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")
        
        fc_path = os.path.join(output_dir, 'FORCE_CONSTANTS')
        n_atoms = crystal.n_atoms
        with open(fc_path, 'w') as f:
            f.write(f"     {n_atoms}     {n_atoms}\n")
            for i in range(n_atoms):
                for j in range(n_atoms):
                    f.write(f"   {i+1}   {j+1}\n")
                    for alpha in range(3):
                        f.write(f"  {fc_matrix[i, j, alpha, 0]:.15f}  "
                                f"{fc_matrix[i, j, alpha, 1]:.15f}  "
                                f"{fc_matrix[i, j, alpha, 2]:.15f}\n")
        
        print(f"Phonopy文件已导出到: {output_dir}")
        print(f"  - POSCAR: {poscar_path}")
        print(f"  - FORCE_CONSTANTS: {fc_path}")
    
    @staticmethod
    def import_phonopy_force_constants(crystal: Crystal, fc_file: str) -> np.ndarray:
        n_atoms = crystal.n_atoms
        fc = np.zeros((n_atoms, n_atoms, 3, 3))
        
        with open(fc_file, 'r') as f:
            lines = f.readlines()
        
        idx = 0
        n1, n2 = map(int, lines[idx].split())
        idx += 1
        
        for i in range(n1):
            for j in range(n2):
                idx += 1
                for alpha in range(3):
                    vals = list(map(float, lines[idx].split()))
                    fc[i, j, alpha] = vals
                    idx += 1
        
        print(f"已从 {fc_file} 导入力常数矩阵，形状: {fc.shape}")
        return fc


class FiniteDisplacementMethod:
    def __init__(self, crystal: Crystal, displacement: float = 0.01, 
                 supercell_matrix: Optional[np.ndarray] = None):
        self.crystal = crystal
        self.displacement = displacement
        if supercell_matrix is None:
            self.supercell_matrix = np.eye(3, dtype=int)
        else:
            self.supercell_matrix = np.array(supercell_matrix, dtype=int)
        self.displacements = []
        self.forces = []
    
    def generate_displaced_structures(self) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
        n_atoms = self.crystal.n_atoms
        structures = []
        
        for atom_idx in range(n_atoms):
            for direction in range(3):
                for sign in [1, -1]:
                    disp = np.zeros((n_atoms, 3))
                    disp[atom_idx, direction] = sign * self.displacement
                    
                    new_positions = self.crystal.positions + disp
                    structures.append((new_positions, disp, atom_idx, direction))
        
        self.displacements = structures
        print(f"已生成 {len(structures)} 个位移结构")
        return structures
    
    def compute_force_constants_from_forces(self, forces_list: List[np.ndarray]) -> np.ndarray:
        n_atoms = self.crystal.n_atoms
        fc = np.zeros((n_atoms, n_atoms, 3, 3))
        
        struct_idx = 0
        for atom_idx in range(n_atoms):
            for direction in range(3):
                force_plus = forces_list[struct_idx]
                force_minus = forces_list[struct_idx + 1]
                struct_idx += 2
                
                for j in range(n_atoms):
                    for beta in range(3):
                        fc[j, atom_idx, beta, direction] = (force_minus[j, beta] - force_plus[j, beta]) / (2 * self.displacement)
        
        for i in range(n_atoms):
            fc[i, i] = -np.sum(fc[i, :], axis=0)
        
        print(f"力常数矩阵计算完成，形状: {fc.shape}")
        return fc
    
    def write_displaced_poscars(self, output_dir: str = 'displaced_cells'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        structures = self.generate_displaced_structures()
        
        for idx, (positions, disp, atom_idx, direction) in enumerate(structures):
            sign = 'plus' if disp[atom_idx, direction] > 0 else 'minus'
            filename = f"POSCAR-{idx:03d}-atom{atom_idx+1}-{['x','y','z'][direction]}-{sign}"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"Displaced structure: atom {atom_idx+1}, {['x','y','z'][direction]} {sign}\n")
                f.write("1.0\n")
                for row in self.crystal.lattice:
                    f.write(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n")
                f.write(f"{' '.join(self.crystal.symbols)}\n")
                f.write(f"{' '.join(['1'] * self.crystal.n_atoms)}\n")
                f.write("Direct\n")
                for pos in positions:
                    f.write(f"  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")
        
        print(f"位移结构已写入: {output_dir}")


class HighSymmetryPaths:
    @staticmethod
    def get_fcc_path() -> Tuple[List[np.ndarray], List[str]]:
        G = np.array([0.0, 0.0, 0.0])
        X = np.array([0.5, 0.0, 0.5])
        W = np.array([0.5, 0.25, 0.75])
        K = np.array([0.375, 0.375, 0.75])
        L = np.array([0.5, 0.5, 0.5])
        U = np.array([0.25, 0.625, 0.625])
        return [G, X, W, K, G, L, U, W, L], ['Γ', 'X', 'W', 'K', 'Γ', 'L', 'U', 'W', 'L']

    @staticmethod
    def get_bcc_path() -> Tuple[List[np.ndarray], List[str]]:
        G = np.array([0.0, 0.0, 0.0])
        H = np.array([0.5, -0.5, 0.5])
        P = np.array([0.25, 0.25, 0.25])
        N = np.array([0.0, 0.0, 0.5])
        return [G, H, P, G, N], ['Γ', 'H', 'P', 'Γ', 'N']

    @staticmethod
    def get_sc_path() -> Tuple[List[np.ndarray], List[str]]:
        G = np.array([0.0, 0.0, 0.0])
        X = np.array([0.5, 0.0, 0.0])
        M = np.array([0.5, 0.5, 0.0])
        R = np.array([0.5, 0.5, 0.5])
        return [G, X, M, G, R, X, M, R], ['Γ', 'X', 'M', 'Γ', 'R', 'X', 'M', 'R']
    
    @staticmethod
    def get_hexagonal_path() -> Tuple[List[np.ndarray], List[str]]:
        G = np.array([0.0, 0.0, 0.0])
        M = np.array([0.5, 0.0, 0.0])
        K = np.array([1/3, 1/3, 0.0])
        A = np.array([0.0, 0.0, 0.5])
        L = np.array([0.5, 0.0, 0.5])
        H = np.array([1/3, 1/3, 0.5])
        return [G, M, K, G, A, L, H, A], ['Γ', 'M', 'K', 'Γ', 'A', 'L', 'H', 'A']
    
    @staticmethod
    def get_orthorhombic_path() -> Tuple[List[np.ndarray], List[str]]:
        G = np.array([0.0, 0.0, 0.0])
        X = np.array([0.5, 0.0, 0.0])
        Y = np.array([0.0, 0.5, 0.0])
        Z = np.array([0.0, 0.0, 0.5])
        S = np.array([0.5, 0.5, 0.0])
        T = np.array([0.0, 0.5, 0.5])
        U = np.array([0.5, 0.0, 0.5])
        R = np.array([0.5, 0.5, 0.5])
        return [G, X, S, Y, G, Z, U, R, T, Z], ['Γ', 'X', 'S', 'Y', 'Γ', 'Z', 'U', 'R', 'T', 'Z']


def example_finite_displacement():
    print("\n" + "=" * 60)
    print("示例: 有限位移法生成位移结构")
    print("=" * 60)
    
    a = 3.0
    lattice = np.eye(3) * a
    positions = np.array([[0.0, 0.0, 0.0]])
    symbols = ['X']
    masses = np.array([50.0])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    fdm = FiniteDisplacementMethod(crystal, displacement=0.01)
    
    structures = fdm.generate_displaced_structures()
    print(f"生成了 {len(structures)} 个位移结构")
    
    print("\nPhonopy导出示例:")
    fc = ForceConstants(crystal, cutoff=4.0)
    fc.generate_model_fc(spring_constant=15.0)
    PhonopyInterface.export_to_phonopy(crystal, fc.fc_matrix, output_dir='phonopy_example')
    
    return fdm


def example_imaginary_frequency_fix():
    print("\n" + "=" * 60)
    print("示例: 虚频修复方法演示")
    print("=" * 60)
    
    a = 5.43
    lattice = np.array([[0.0, a/2, a/2], [a/2, 0.0, a/2], [a/2, a/2, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0]])
    symbols = ['Si']
    masses = np.array([28.0855])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    
    fc = ForceConstants(crystal, cutoff=5.0)
    fc.generate_model_fc(spring_constant=20.0)
    
    print("\n--- Method 1: Without any fixes ---")
    phonon_raw = PhononCalculator(crystal, fc, use_asr=False, symmetrize_fc=False, regularization='none')
    path_points, labels = HighSymmetryPaths.get_fcc_path()
    phonon_raw.print_imaginary_analysis(path_points, n_points_per_segment=30)
    
    print("\n--- Method 2: With ASR + Symmetrization ---")
    phonon_asr = PhononCalculator(crystal, fc, use_asr=True, symmetrize_fc=True, regularization='none')
    phonon_asr.print_imaginary_analysis(path_points, n_points_per_segment=30)
    
    print("\n--- Method 3: With ASR + Symmetrization + Regularization ---")
    phonon_full = PhononCalculator(crystal, fc, use_asr=True, symmetrize_fc=True, 
                                     regularization='near_gamma', reg_strength=1e-6)
    phonon_full.print_imaginary_analysis(path_points, n_points_per_segment=30)
    
    print("\n--- Method 4: With ASR + Symmetrization + Projected Regularization ---")
    phonon_proj = PhononCalculator(crystal, fc, use_asr=True, symmetrize_fc=True, 
                                   regularization='projected', reg_strength=1e-6)
    phonon_proj.print_imaginary_analysis(path_points, n_points_per_segment=30)
    
    print("\n--- Comparison of Gamma point frequencies ---")
    gamma = np.array([0.0, 0.0, 0.0])
    
    print("Raw:", phonon_raw.compute_frequencies(gamma, convert_to_cm=True))
    print("ASR:", phonon_asr.compute_frequencies(gamma, convert_to_cm=True))
    print("ASR+Reg:", phonon_full.compute_frequencies(gamma, convert_to_cm=True))
    print("ASR+Proj:", phonon_proj.compute_frequencies(gamma, convert_to_cm=True))
    
    print("\n--- Band structure calculation with full fixes ---")
    distances, frequencies = phonon_full.compute_band_structure(path_points, n_points_per_segment=100)
    print(f"Frequency range: {np.min(frequencies):.2f} - {np.max(frequencies):.2f} cm^-1")
    
    PhononVisualizer.plot_dispersion(
        distances, frequencies, labels,
        title='FCC Phonon (Fixed)',
        save_path='fcc_phonon_fixed.png'
    )
    print("Dispersion plot saved as: fcc_phonon_fixed.png")
    
    return phonon_full


def example_regularization_comparison():
    print("\n" + "=" * 60)
    print("示例: 不同正则化方法对比")
    print("=" * 60)
    
    a = 3.0
    lattice = np.eye(3) * a
    positions = np.array([[0.0, 0.0, 0.0]])
    symbols = ['X']
    masses = np.array([50.0])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    
    fc = ForceConstants(crystal, cutoff=4.0)
    fc.generate_model_fc(spring_constant=15.0)
    
    reg_methods = ['none', 'simple', 'near_gamma', 'projected', 'full']
    
    for method in reg_methods:
        print(f"\nRegularization: {method}")
        phonon = PhononCalculator(crystal, fc, use_asr=True, symmetrize_fc=True,
                                      regularization=method, reg_strength=1e-5)
        
        gamma = np.array([0.0, 0.0, 0.0])
        freqs = phonon.compute_frequencies(gamma, convert_to_cm=True)
        print(f"  Gamma point frequencies: {freqs}")
    
    return crystal, fc


class ThirdOrderForceConstants:
    def __init__(self, crystal: Crystal, cutoff: float = 5.0):
        self.crystal = crystal
        self.cutoff = cutoff
        self.n_atoms = crystal.n_atoms
        self.fc3 = None
    
    def generate_model_fc3(self, cubic_coefficient: float = -1000.0) -> np.ndarray:
        n = self.n_atoms
        fc3 = np.zeros((n, n, n, 3, 3, 3))
        cart_pos = self.crystal.get_cartesian_positions()
        lattice = self.crystal.lattice
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            for c in range(-1, 2):
                                cell_shift_j = np.array([a, b, c]) @ lattice
                                cell_shift_k = np.array([a, b, c]) @ lattice
                                
                                r_ij = cart_pos[j] + cell_shift_j - cart_pos[i]
                                r_ik = cart_pos[k] + cell_shift_k - cart_pos[i]
                                
                                dist_ij = np.linalg.norm(r_ij)
                                dist_ik = np.linalg.norm(r_ik)
                                
                                if 1e-6 < dist_ij < self.cutoff and 1e-6 < dist_ik < self.cutoff:
                                    e_ij = r_ij / dist_ij
                                    e_ik = r_ik / dist_ik
                                    
                                    for alpha in range(3):
                                        for beta in range(3):
                                            for gamma in range(3):
                                                fc3[i, j, k, alpha, beta, gamma] += \
                                                    cubic_coefficient * e_ij[alpha] * e_ij[beta] * e_ik[gamma]
        
        if np.max(np.abs(fc3)) < 1e-10:
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        fc3[i, j, k] = cubic_coefficient * np.eye(3).reshape(1, 3, 3)
        
        self.fc3 = fc3
        print(f"三阶力常数矩阵生成完成，形状: {fc3.shape}, 平均幅值: {np.mean(np.abs(fc3)):.4e}")
        return fc3
    
    def save_fc3_to_file(self, filename: str):
        np.savez(filename, fc3=self.fc3)
        print(f"三阶力常数已保存到: {filename}")
    
    def load_fc3_from_file(self, filename: str):
        data = np.load(filename)
        self.fc3 = data['fc3']
        print(f"三阶力常数已从文件加载，形状: {self.fc3.shape}")
        return self.fc3


class ThirdOrderFiniteDisplacement:
    def __init__(self, crystal: Crystal, displacement: float = 0.01):
        self.crystal = crystal
        self.displacement = displacement
        self.n_atoms = crystal.n_atoms
    
    def generate_displaced_structures(self) -> List[Tuple[np.ndarray, int, int, int, int]]:
        structures = []
        n = self.n_atoms
        
        for i in range(n):
            for alpha in range(3):
                for j in range(n):
                    for beta in range(3):
                        disp = np.zeros((n, 3))
                        disp[i, alpha] = self.displacement
                        disp[j, beta] = self.displacement
                        
                        new_positions = self.crystal.positions + disp
                        structures.append((new_positions, i, alpha, j, beta))
        
        print(f"已生成 {len(structures)} 个三阶位移结构")
        return structures
    
    def compute_fc3_from_forces(self, forces_list: List[np.ndarray]) -> np.ndarray:
        n = self.n_atoms
        fc3 = np.zeros((n, n, n, 3, 3, 3))
        
        struct_idx = 0
        for i in range(n):
            for alpha in range(3):
                for j in range(n):
                    for beta in range(3):
                        forces = forces_list[struct_idx]
                        struct_idx += 1
                        
                        for k in range(n):
                            for gamma in range(3):
                                fc3[k, i, j, gamma, alpha, beta] = \
                                    -forces[k, gamma] / (self.displacement ** 2)
        
        print(f"三阶力常数计算完成，形状: {fc3.shape}")
        return fc3


class QuasiHarmonicApproximation:
    def __init__(self, phonon_calculator: PhononCalculator, 
                 volume_range: np.ndarray = None):
        self.phonon = phonon_calculator
        self.crystal = phonon_calculator.crystal
        
        if volume_range is None:
            V0 = np.linalg.det(self.crystal.lattice)
            self.volume_range = np.linspace(0.95 * V0, 1.05 * V0, 11)
        else:
            self.volume_range = volume_range
        
        self.volumes = []
        self.frequencies = []
        self.gamma_values = []
    
    def compute_volume_dependent_frequencies(self, q_mesh: List[np.ndarray], 
                                              n_points: int = 11) -> dict:
        V0 = np.linalg.det(self.crystal.lattice)
        results = {'volumes': [], 'frequencies': []}
        
        for scale in np.linspace(0.95, 1.05, n_points):
            scaled_lattice = self.crystal.lattice * (scale ** (1/3))
            scaled_crystal = Crystal(
                scaled_lattice,
                self.crystal.positions,
                self.crystal.symbols,
                self.crystal.masses
            )
            
            fc = ForceConstants(scaled_crystal, cutoff=5.0)
            fc.generate_model_fc(spring_constant=20.0 * (scale ** -1.5))
            
            phonon = PhononCalculator(scaled_crystal, fc, use_asr=True, symmetrize_fc=True)
            
            freqs_at_q = []
            for q in q_mesh:
                freqs = phonon.compute_frequencies(q, convert_to_cm=False)
                freqs_at_q.append(freqs)
            
            V = np.linalg.det(scaled_lattice)
            results['volumes'].append(V)
            results['frequencies'].append(np.array(freqs_at_q))
        
        self.volumes = np.array(results['volumes'])
        self.frequencies = np.array(results['frequencies'])
        
        print(f"体积依赖频率计算完成: {n_points} 个体积点")
        return results
    
    def compute_gamma(self) -> np.ndarray:
        if len(self.volumes) < 3:
            raise ValueError("需要至少3个体积点来计算Gruneisen参数")
        
        log_V = np.log(self.volumes)
        gamma = -np.gradient(np.log(self.frequencies + 1e-12), log_V, axis=0)
        
        self.gamma_values = gamma
        print(f"Gruneisen参数计算完成，形状: {gamma.shape}")
        return gamma
    
    def compute_free_energy(self, temperatures: np.ndarray, q_mesh_size: int = 10) -> dict:
        rec_lat = self.crystal.reciprocal_lattice
        qpoints = []
        for i in range(q_mesh_size):
            for j in range(q_mesh_size):
                for k in range(q_mesh_size):
                    q = np.array([i/q_mesh_size - 0.5, j/q_mesh_size - 0.5, k/q_mesh_size - 0.5])
                    qpoints.append(q @ rec_lat)
        
        results = {'T': temperatures, 'F': [], 'S': [], 'U': [], 'Cv': []}
        
        hbar = 1.0545718e-34
        kB = 1.380649e-23
        
        ang_to_m = 1e-10
        amu = 1.66054e-27
        eV_to_J = 1.602176634e-19
        evang2_to_kg = eV_to_J / (ang_to_m * ang_to_m)
        conversion = np.sqrt(evang2_to_kg / amu)
        
        for T in temperatures:
            if T < 1e-10:
                beta = np.inf
            else:
                beta = 1 / (kB * T)
            
            total_F = 0.0
            total_S = 0.0
            total_U = 0.0
            total_Cv = 0.0
            
            for q in qpoints:
                freqs = self.phonon.compute_frequencies(q, convert_to_ghz=False)
                omega = freqs * conversion
                
                for w in omega:
                    if w > 1e12:
                        hw = hbar * w
                        if T > 1e-10:
                            x = hw * beta
                            x = min(x, 100)
                            n = 1.0 / (np.exp(x) - 1.0)
                            F = hw / 2 + hw * n
                            U = hw * (n + 0.5)
                            S = kB * ((n + 1) * np.log(n + 1) - n * np.log(n)) if n > 0 else 0
                            Cv = kB * x**2 * np.exp(x) / (np.exp(x) - 1)**2
                        else:
                            F = hw / 2
                            U = hw / 2
                            S = 0.0
                            Cv = 0.0
                        
                        total_F += F
                        total_S += S
                        total_U += U
                        total_Cv += Cv
            
            total_F /= len(qpoints)
            total_S /= len(qpoints)
            total_U /= len(qpoints)
            total_Cv /= len(qpoints)
            
            results['F'].append(total_F)
            results['S'].append(total_S)
            results['U'].append(total_U)
            results['Cv'].append(total_Cv)
        
        for key in ['F', 'S', 'U', 'Cv']:
            results[key] = np.array(results[key])
        
        print(f"热力学量计算完成，温度范围: {temperatures[0]} - {temperatures[-1]} K")
        return results
    
    def compute_thermal_expansion(self, temperatures: np.ndarray, 
                                   B: float = 1e11) -> np.ndarray:
        q_mesh_size = 8
        rec_lat = self.crystal.reciprocal_lattice
        qpoints = []
        for i in range(q_mesh_size):
            for j in range(q_mesh_size):
                for k in range(q_mesh_size):
                    q = np.array([i/q_mesh_size - 0.5, j/q_mesh_size - 0.5, k/q_mesh_size - 0.5])
                    qpoints.append(q @ rec_lat)
        
        self.compute_volume_dependent_frequencies(qpoints[:min(10, len(qpoints))], n_points=5)
        self.compute_gamma()
        
        V0 = np.linalg.det(self.crystal.lattice) * 1e-30
        hbar = 1.0545718e-34
        kB = 1.380649e-23
        
        ang_to_m = 1e-10
        amu = 1.66054e-27
        eV_to_J = 1.602176634e-19
        evang2_to_kg = eV_to_J / (ang_to_m * ang_to_m)
        conversion = np.sqrt(evang2_to_kg / amu)
        
        alpha = np.zeros_like(temperatures)
        
        for idx, T in enumerate(temperatures):
            if T < 1e-10:
                alpha[idx] = 0.0
                continue
            
            beta = 1 / (kB * T)
            gamma_T = 0.0
            Cv_T = 0.0
            
            for q_idx, q in enumerate(qpoints[:min(10, len(qpoints))]):
                freqs = self.phonon.compute_frequencies(q, convert_to_ghz=False)
                omega = freqs * conversion
                
                gamma_q = np.mean(self.gamma_values[:, q_idx, :]) if len(self.gamma_values.shape) > 2 else 1.0
                
                for w in omega:
                    if w > 1e12:
                        x = hbar * w * beta
                        x = min(x, 100)
                        cv_mode = kB * x**2 * np.exp(x) / (np.exp(x) - 1)**2
                        gamma_T += gamma_q * cv_mode
                        Cv_T += cv_mode
            
            if Cv_T > 1e-12:
                alpha[idx] = gamma_T * kB / (3 * B * V0)
            else:
                alpha[idx] = 0.0
        
        print(f"热膨胀系数计算完成，温度范围: {temperatures[0]} - {temperatures[-1]} K")
        return alpha


class PhononBTE:
    def __init__(self, phonon_calculator: PhononCalculator, 
                 fc3: ThirdOrderForceConstants = None):
        self.phonon = phonon_calculator
        self.fc3 = fc3
        self.crystal = phonon_calculator.crystal
    
    def compute_group_velocity(self, q: np.ndarray, dq: float = 1e-5) -> np.ndarray:
        n_modes = 3 * self.crystal.n_atoms
        v_g = np.zeros((n_modes, 3))
        
        q_cart = q
        ang_to_m = 1e-10
        eV_to_J = 1.602176634e-19
        
        for alpha in range(3):
            dq_vec = np.zeros(3)
            dq_vec[alpha] = dq
            
            freqs_plus = self.phonon.compute_frequencies(q_cart + dq_vec, convert_to_ghz=False)
            freqs_minus = self.phonon.compute_frequencies(q_cart - dq_vec, convert_to_ghz=False)
            
            df_dq = (freqs_plus - freqs_minus) / (2 * dq)
            
            amu = 1.66054e-27
            evang2_to_kg = eV_to_J / (ang_to_m * ang_to_m)
            conversion = np.sqrt(evang2_to_kg / amu)
            
            v_g[:, alpha] = df_dq * conversion * ang_to_m
        
        return v_g
    
    def compute_phonon_lifetime(self, q: np.ndarray, T: float = 300.0, 
                                  gamma: float = 1e-12) -> np.ndarray:
        freqs = self.phonon.compute_frequencies(q, convert_to_ghz=False)
        
        ang_to_m = 1e-10
        amu = 1.66054e-27
        eV_to_J = 1.602176634e-19
        evang2_to_kg = eV_to_J / (ang_to_m * ang_to_m)
        conversion = np.sqrt(evang2_to_kg / amu)
        
        omega = freqs * conversion
        
        hbar = 1.0545718e-34
        kB = 1.380649e-23
        
        tau = np.zeros_like(omega)
        
        for i, w in enumerate(omega):
            if w < 1e9:
                tau[i] = 1e-12
                continue
            
            if T < 1e-10:
                tau[i] = 1e-9
                continue
            
            x = hbar * w / (kB * T)
            n = 1.0 / (np.exp(min(x, 50)) - 1.0)
            
            if self.fc3 is not None and self.fc3.fc3 is not None:
                fc3_mean = np.mean(np.abs(self.fc3.fc3))
                gamma_scat = np.abs(gamma * fc3_mean) * (n * (n + 1)) * 1e20
            else:
                gamma_scat = gamma * w**2 * T * 1e-45
            
            tau[i] = 1.0 / (gamma_scat + 1e-12)
        
        return tau
    
    def compute_thermal_conductivity(self, temperatures: np.ndarray, 
                                      q_mesh_size: int = 10,
                                      method: str = 'relaxation_time') -> dict:
        rec_lat = self.crystal.reciprocal_lattice
        qpoints = []
        weights = []
        
        for i in range(q_mesh_size):
            for j in range(q_mesh_size):
                for k in range(q_mesh_size):
                    q_frac = np.array([i/q_mesh_size - 0.5, j/q_mesh_size - 0.5, k/q_mesh_size - 0.5])
                    qpoints.append(q_frac @ rec_lat)
                    weights.append(1.0 / q_mesh_size**3)
        
        hbar = 1.0545718e-34
        kB = 1.380649e-23
        V = np.linalg.det(self.crystal.lattice) * 1e-30
        
        ang_to_m = 1e-10
        amu = 1.66054e-27
        eV_to_J = 1.602176634e-19
        evang2_to_kg = eV_to_J / (ang_to_m * ang_to_m)
        conversion = np.sqrt(evang2_to_kg / amu)
        
        results = {'T': temperatures, 'kappa': [], 'kappa_xyz': []}
        
        for T in temperatures:
            kappa_tensor = np.zeros((3, 3))
            
            for q, weight in zip(qpoints, weights):
                freqs = self.phonon.compute_frequencies(q, convert_to_ghz=False)
                omega = freqs * conversion
                v_g = self.compute_group_velocity(q)
                tau = self.compute_phonon_lifetime(q, T)
                
                for mode_idx in range(len(omega)):
                    w = omega[mode_idx]
                    if w < 1e9:
                        continue
                    
                    x = hbar * w / (kB * T)
                    if x > 100:
                        continue
                    
                    exp_x = np.exp(min(x, 50))
                    n = 1.0 / (exp_x - 1.0)
                    cv_mode = hbar * w * n * (n + 1) / T
                    v = v_g[mode_idx]
                    t = tau[mode_idx]
                    
                    for alpha in range(3):
                        for beta in range(3):
                            kappa_tensor[alpha, beta] += weight * cv_mode * v[alpha] * v[beta] * t / V
            
            kappa_mean = np.trace(kappa_tensor) / 3.0
            results['kappa'].append(kappa_mean)
            results['kappa_xyz'].append(np.diag(kappa_tensor))
        
        results['kappa'] = np.array(results['kappa'])
        results['kappa_xyz'] = np.array(results['kappa_xyz'])
        
        print(f"热导率计算完成，温度范围: {temperatures[0]} - {temperatures[-1]} K")
        return results
    
    def cumulative_kappa(self, T: float = 300.0, q_mesh_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        rec_lat = self.crystal.reciprocal_lattice
        qpoints = []
        
        for i in range(q_mesh_size):
            for j in range(q_mesh_size):
                for k in range(q_mesh_size):
                    q_frac = np.array([i/q_mesh_size - 0.5, j/q_mesh_size - 0.5, k/q_mesh_size - 0.5])
                    qpoints.append(q_frac @ rec_lat)
        
        all_freqs = []
        all_kappa = []
        
        for q in qpoints:
            freqs = self.phonon.compute_frequencies(q, convert_to_cm=True)
            v_g = self.compute_group_velocity(q)
            tau = self.compute_phonon_lifetime(q, T)
            
            for mode_idx in range(len(freqs)):
                all_freqs.append(freqs[mode_idx])
                v_mag = np.linalg.norm(v_g[mode_idx])
                all_kappa.append(v_mag**2 * tau[mode_idx])
        
        all_freqs = np.array(all_freqs)
        all_kappa = np.array(all_kappa)
        
        valid_idx = ~np.isnan(all_kappa) & ~np.isinf(all_kappa) & (all_kappa > 0)
        if np.sum(valid_idx) == 0:
            return np.array([]), np.array([])
        
        all_freqs = all_freqs[valid_idx]
        all_kappa = all_kappa[valid_idx]
        
        sorted_idx = np.argsort(all_freqs)
        sorted_freqs = all_freqs[sorted_idx]
        sorted_kappa = all_kappa[sorted_idx]
        
        total_kappa = np.sum(sorted_kappa)
        if total_kappa > 0:
            cumulative = np.cumsum(sorted_kappa) / total_kappa
        else:
            cumulative = np.zeros_like(sorted_kappa)
        
        return sorted_freqs, cumulative


class AnharmonicVisualizer:
    @staticmethod
    def plot_thermal_expansion(temperatures: np.ndarray, alpha: np.ndarray,
                                save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(temperatures, alpha * 1e6, 'b-', linewidth=2)
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Thermal Expansion Coefficient (10$^{-6}$ K$^{-1}$)', fontsize=12)
        ax.set_title('Thermal Expansion vs Temperature', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"热膨胀图已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_thermal_conductivity(temperatures: np.ndarray, kappa: np.ndarray,
                                   save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(temperatures, kappa, 'r-', linewidth=2)
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Thermal Conductivity (W/m-K)', fontsize=12)
        ax.set_title('Lattice Thermal Conductivity', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"热导率图已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_free_energy(temperatures: np.ndarray, thermo_data: dict,
                          save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(temperatures, thermo_data['F'] * 1e3, 'b-')
        axes[0, 0].set_xlabel('T (K)')
        axes[0, 0].set_ylabel('Free Energy (meV)')
        axes[0, 0].set_title('Helmholtz Free Energy')
        
        axes[0, 1].plot(temperatures, thermo_data['S'], 'r-')
        axes[0, 1].set_xlabel('T (K)')
        axes[0, 1].set_ylabel('Entropy (J/K)')
        axes[0, 1].set_title('Vibrational Entropy')
        
        axes[1, 0].plot(temperatures, thermo_data['U'] * 1e3, 'g-')
        axes[1, 0].set_xlabel('T (K)')
        axes[1, 0].set_ylabel('Internal Energy (meV)')
        axes[1, 0].set_title('Internal Energy')
        
        axes[1, 1].plot(temperatures, thermo_data['Cv'], 'm-')
        axes[1, 1].set_xlabel('T (K)')
        axes[1, 1].set_ylabel('Cv (J/K)')
        axes[1, 1].set_title('Heat Capacity')
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"热力学量图已保存: {save_path}")
        plt.close()


def example_anharmonic_calculations():
    print("\n" + "=" * 60)
    print("示例: 非谐效应计算 - 热膨胀和热导率")
    print("=" * 60)
    
    a = 5.43
    lattice = np.array([[0.0, a/2, a/2], [a/2, 0.0, a/2], [a/2, a/2, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0]])
    symbols = ['Si']
    masses = np.array([28.0855])
    
    crystal = Crystal(lattice, positions, symbols, masses)
    fc = ForceConstants(crystal, cutoff=5.0)
    fc.generate_model_fc(spring_constant=20.0)
    phonon = PhononCalculator(crystal, fc, use_asr=True, symmetrize_fc=True)
    
    print("\n=== 三阶力常数生成 ===")
    fc3 = ThirdOrderForceConstants(crystal, cutoff=4.0)
    fc3.generate_model_fc3(cubic_coefficient=-50.0)
    
    print("\n=== 三阶有限位移法示例 ===")
    fdm3 = ThirdOrderFiniteDisplacement(crystal, displacement=0.01)
    structures = fdm3.generate_displaced_structures()
    print(f"生成位移结构数: {len(structures)}")
    
    print("\n=== 准谐近似: 热膨胀系数 ===")
    qha = QuasiHarmonicApproximation(phonon)
    temperatures = np.array([100, 200, 300, 400, 500, 600, 700, 800])
    alpha = qha.compute_thermal_expansion(temperatures, B=1e11)
    
    for T, a_val in zip(temperatures, alpha):
        print(f"  T={T} K: alpha = {a_val*1e6:.4f} x 10^-6 K^-1")
    
    AnharmonicVisualizer.plot_thermal_expansion(
        temperatures, alpha, save_path='thermal_expansion.png'
    )
    
    print("\n=== 玻尔兹曼输运方程: 热导率 ===")
    bte = PhononBTE(phonon, fc3)
    
    kappa_result = bte.compute_thermal_conductivity(temperatures, q_mesh_size=5)
    
    for T, k in zip(temperatures, kappa_result['kappa']):
        print(f"  T={T} K: kappa = {k:.4f} W/m-K")
    
    AnharmonicVisualizer.plot_thermal_conductivity(
        temperatures, kappa_result['kappa'], save_path='thermal_conductivity.png'
    )
    
    print("\n=== 热力学量计算 ===")
    thermo_data = qha.compute_free_energy(temperatures, q_mesh_size=5)
    AnharmonicVisualizer.plot_free_energy(
        temperatures, thermo_data, save_path='thermodynamic_properties.png'
    )
    
    print("\n=== 群速度计算示例 ===")
    gamma = np.array([0.0, 0.0, 0.0])
    v_g = bte.compute_group_velocity(gamma)
    print(f"Γ点群速度形状: {v_g.shape}")
    print(f"模式1群速度: {v_g[0]} m/s")
    
    print("\n=== 声子寿命计算示例 ===")
    tau = bte.compute_phonon_lifetime(gamma, T=300.0)
    print(f"Γ点声子寿命 (ps): {tau * 1e12}")
    
    print("\n非谐效应计算示例完成!")
    
    return qha, bte


if __name__ == "__main__":
    crystal, fc, phonon, distances, frequencies = example_fcc_calculation()
    example_sc_calculation()
    example_finite_displacement()
    example_imaginary_frequency_fix()
    example_regularization_comparison()
    example_anharmonic_calculations()
