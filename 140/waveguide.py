import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum


class Polarization(Enum):
    X_POLARIZED = 'x'
    Y_POLARIZED = 'y'
    CIRCULAR_RIGHT = 'circular_right'
    CIRCULAR_LEFT = 'circular_left'


class ExcitationType(Enum):
    ELECTRIC_DIPOLE_X = 'ex'
    ELECTRIC_DIPOLE_Y = 'ey'
    ELECTRIC_DIPOLE_Z = 'ez'
    MAGNETIC_DIPOLE_X = 'hx'
    MAGNETIC_DIPOLE_Y = 'hy'
    MAGNETIC_DIPOLE_Z = 'hz'


class RectangularWaveguide:
    def __init__(self, a, b, epsilon_r=1.0, mu_r=1.0):
        self.a = a
        self.b = b
        self.epsilon_0 = 8.854e-12
        self.mu_0 = 4 * np.pi * 1e-7
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r
        self.epsilon = self.epsilon_0 * self.epsilon_r
        self.mu = self.mu_0 * self.mu_r
        self.c = 1 / np.sqrt(self.epsilon_0 * self.mu_0)
        self.eta_0 = np.sqrt(self.mu_0 / self.epsilon_0)

    def cutoff_wavelength(self, m, n, mode='TE'):
        if m == 0 and n == 0:
            raise ValueError("m and n cannot both be zero")
        if mode == 'TM' and (m == 0 or n == 0):
            raise ValueError("For TM modes, m and n must both be at least 1")
        
        lambda_c = 2 / np.sqrt((m / self.a)**2 + (n / self.b)**2)
        return lambda_c

    def cutoff_frequency(self, m, n, mode='TE'):
        lambda_c = self.cutoff_wavelength(m, n, mode)
        f_c = self.c / (lambda_c * np.sqrt(self.epsilon_r * self.mu_r))
        return f_c

    def is_propagating(self, m, n, frequency, mode='TE'):
        f_c = self.cutoff_frequency(m, n, mode)
        return frequency > f_c

    def propagation_constant(self, m, n, frequency, mode='TE'):
        if not self.is_propagating(m, n, frequency, mode):
            return 0.0 + 0.0j
        
        f_c = self.cutoff_frequency(m, n, mode)
        omega = 2 * np.pi * frequency
        k = omega * np.sqrt(self.mu * self.epsilon)
        k_c = 2 * np.pi / self.cutoff_wavelength(m, n, mode)
        beta = np.sqrt(k**2 - k_c**2 + 0j)
        return beta

    def wavelength(self, m, n, frequency, mode='TE'):
        if not self.is_propagating(m, n, frequency, mode):
            return None
        beta = self.propagation_constant(m, n, frequency, mode)
        return 2 * np.pi / np.real(beta)

    def wave_impedance(self, m, n, frequency, mode='TE'):
        if not self.is_propagating(m, n, frequency, mode):
            return None
        
        eta = self.eta_0 / np.sqrt(self.epsilon_r)
        f_c = self.cutoff_frequency(m, n, mode)
        sqrt_term = np.sqrt(1 - (f_c / frequency)**2)
        
        if mode == 'TE':
            return eta / sqrt_term
        elif mode == 'TM':
            return eta * sqrt_term
        else:
            raise ValueError("Mode must be 'TE' or 'TM'")

    def find_degenerate_modes(self, m, n, frequency, tolerance=1e-6):
        """查找与指定模式简并的所有模式"""
        target_f_c = self.cutoff_frequency(m, n, 'TE')
        degenerate_modes = []
        
        for mm in range(0, m + n + 3):
            for nn in range(0, m + n + 3):
                if mm == 0 and nn == 0:
                    continue
                
                for mode_type in ['TE', 'TM']:
                    try:
                        f_c = self.cutoff_frequency(mm, nn, mode_type)
                        if abs(f_c - target_f_c) < tolerance * target_f_c:
                            if not (mm == m and nn == n and mode_type == 'TE'):
                                degenerate_modes.append((mode_type, mm, nn))
                    except ValueError:
                        pass
        
        return degenerate_modes

    def find_all_degenerate_groups(self, max_m=5, max_n=5, tolerance=1e-6):
        """查找所有简并模式组"""
        all_modes = []
        
        for m in range(0, max_m + 1):
            for n in range(0, max_n + 1):
                if m == 0 and n == 0:
                    continue
                
                for mode_type in ['TE', 'TM']:
                    try:
                        f_c = self.cutoff_frequency(m, n, mode_type)
                        all_modes.append((mode_type, m, n, f_c))
                    except ValueError:
                        pass
        
        groups = []
        used_modes = set()
        
        for i, (mode1, m1, n1, f_c1) in enumerate(all_modes):
            if (mode1, m1, n1) in used_modes:
                continue
            
            group = [(mode1, m1, n1, f_c1)]
            used_modes.add((mode1, m1, n1))
            
            for j in range(i + 1, len(all_modes)):
                mode2, m2, n2, f_c2 = all_modes[j]
                if (mode2, m2, n2) in used_modes:
                    continue
                
                if abs(f_c2 - f_c1) < tolerance * f_c1:
                    group.append((mode2, m2, n2, f_c2))
                    used_modes.add((mode2, m2, n2))
            
            groups.append(group)
        
        return groups

    def te_fields_complex(self, m, n, frequency, x, y, z=0, t=0, amplitude=1.0):
        """TE模式的复数场分布（保留相位信息）"""
        if not self.is_propagating(m, n, frequency, 'TE'):
            raise ValueError(f"TE_{m}{n} mode is evanescent at this frequency")
        
        omega = 2 * np.pi * frequency
        beta = self.propagation_constant(m, n, frequency, 'TE')
        k_c = 2 * np.pi / self.cutoff_wavelength(m, n, 'TE')
        
        phase = omega * t - beta * z
        
        E_x = 0j
        E_y = 0j
        E_z = 0j
        
        if m > 0:
            E_y += (amplitude * omega * self.mu * m * np.pi / (k_c**2 * self.a)) * \
                   np.cos(m * np.pi * x / self.a) * np.sin(n * np.pi * y / self.b) * \
                   np.exp(1j * phase)
        
        if n > 0:
            E_x += (-amplitude * omega * self.mu * n * np.pi / (k_c**2 * self.b)) * \
                   np.sin(m * np.pi * x / self.a) * np.cos(n * np.pi * y / self.b) * \
                   np.exp(1j * phase)
        
        H_x = (amplitude * beta * n * np.pi / (k_c**2 * self.b)) * \
              np.sin(m * np.pi * x / self.a) * np.cos(n * np.pi * y / self.b) * \
              np.exp(1j * phase)
        
        H_y = (-amplitude * beta * m * np.pi / (k_c**2 * self.a)) * \
              np.cos(m * np.pi * x / self.a) * np.sin(n * np.pi * y / self.b) * \
              np.exp(1j * phase)
        
        H_z = amplitude * np.cos(m * np.pi * x / self.a) * np.cos(n * np.pi * y / self.b) * \
              np.exp(1j * phase)
        
        return {
            'Ex': E_x, 'Ey': E_y, 'Ez': E_z,
            'Hx': H_x, 'Hy': H_y, 'Hz': H_z
        }

    def tm_fields_complex(self, m, n, frequency, x, y, z=0, t=0, amplitude=1.0):
        """TM模式的复数场分布（保留相位信息）"""
        if not self.is_propagating(m, n, frequency, 'TM'):
            raise ValueError(f"TM_{m}{n} mode is evanescent at this frequency")
        
        omega = 2 * np.pi * frequency
        beta = self.propagation_constant(m, n, frequency, 'TM')
        k_c = 2 * np.pi / self.cutoff_wavelength(m, n, 'TM')
        
        phase = omega * t - beta * z
        
        E_z = amplitude * np.sin(m * np.pi * x / self.a) * np.sin(n * np.pi * y / self.b) * \
              np.exp(1j * phase)
        
        E_x = (-amplitude * beta * m * np.pi / (k_c**2 * self.a)) * \
              np.cos(m * np.pi * x / self.a) * np.sin(n * np.pi * y / self.b) * \
              np.exp(1j * phase)
        
        E_y = (-amplitude * beta * n * np.pi / (k_c**2 * self.b)) * \
              np.sin(m * np.pi * x / self.a) * np.cos(n * np.pi * y / self.b) * \
              np.exp(1j * phase)
        
        H_x = (amplitude * omega * self.epsilon * n * np.pi / (k_c**2 * self.b)) * \
              np.sin(m * np.pi * x / self.a) * np.cos(n * np.pi * y / self.b) * \
              np.exp(1j * phase)
        
        H_y = (-amplitude * omega * self.epsilon * m * np.pi / (k_c**2 * self.a)) * \
              np.cos(m * np.pi * x / self.a) * np.sin(n * np.pi * y / self.b) * \
              np.exp(1j * phase)
        
        H_z = 0j
        
        return {
            'Ex': E_x, 'Ey': E_y, 'Ez': E_z,
            'Hx': H_x, 'Hy': H_y, 'Hz': H_z
        }

    def get_fields_complex(self, m, n, frequency, x, y, z=0, t=0, mode='TE', amplitude=1.0):
        if mode == 'TE':
            return self.te_fields_complex(m, n, frequency, x, y, z, t, amplitude)
        elif mode == 'TM':
            return self.tm_fields_complex(m, n, frequency, x, y, z, t, amplitude)
        else:
            raise ValueError("Mode must be 'TE' or 'TM'")

    def get_fields(self, m, n, frequency, x, y, z=0, t=0, mode='TE', amplitude=1.0):
        fields = self.get_fields_complex(m, n, frequency, x, y, z, t, mode, amplitude)
        return {key: np.real(value) for key, value in fields.items()}

    def compute_mode_norm(self, m, n, frequency, mode='TE', resolution=100):
        """计算模式的归一化常数（用于正交性投影）"""
        x = np.linspace(0, self.a, resolution)
        y = np.linspace(0, self.b, resolution)
        X, Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        fields = self.get_fields_complex(m, n, frequency, X, Y, mode=mode)
        
        E_squared = np.abs(fields['Ex'])**2 + np.abs(fields['Ey'])**2 + np.abs(fields['Ez'])**2
        norm = np.sum(E_squared) * dx * dy
        
        return norm

    def mode_matching(self, source_field, frequency, candidate_modes, resolution=100):
        """
        模式匹配算法：将源场投影到候选模式上
        
        参数:
            source_field: 源场字典，包含Ex, Ey, Ez
            frequency: 工作频率
            candidate_modes: 候选模式列表，格式为[(mode_type, m, n), ...]
            resolution: 采样分辨率
        
        返回:
            模式振幅字典
        """
        x = np.linspace(0, self.a, resolution)
        y = np.linspace(0, self.b, resolution)
        X, Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        amplitudes = {}
        
        for mode_type, m, n in candidate_modes:
            try:
                mode_fields = self.get_fields_complex(m, n, frequency, X, Y, mode=mode_type)
                
                overlap = 0j
                for comp in ['Ex', 'Ey', 'Ez']:
                    overlap += np.sum(np.conj(mode_fields[comp]) * source_field[comp]) * dx * dy
                
                norm = self.compute_mode_norm(m, n, frequency, mode_type, resolution)
                amplitudes[(mode_type, m, n)] = overlap / norm
            except ValueError:
                amplitudes[(mode_type, m, n)] = 0j
        
        return amplitudes

    def symmetry_projection(self, m, n, frequency, excitation_type, mode='TE'):
        """
        基于对称性的模式投影
        
        根据激励源类型，计算模式的激发系数
        """
        if mode == 'TE':
            if n == 0:
                if excitation_type in [ExcitationType.ELECTRIC_DIPOLE_Y, 
                                       ExcitationType.MAGNETIC_DIPOLE_X]:
                    return 1.0
                elif excitation_type in [ExcitationType.ELECTRIC_DIPOLE_X, 
                                         ExcitationType.MAGNETIC_DIPOLE_Y]:
                    return 0.0
            elif m == 0:
                if excitation_type in [ExcitationType.ELECTRIC_DIPOLE_X, 
                                       ExcitationType.MAGNETIC_DIPOLE_Y]:
                    return 1.0
                elif excitation_type in [ExcitationType.ELECTRIC_DIPOLE_Y, 
                                         ExcitationType.MAGNETIC_DIPOLE_X]:
                    return 0.0
        
        if excitation_type in [ExcitationType.ELECTRIC_DIPOLE_Z]:
            if mode == 'TM':
                return 1.0
            else:
                return 0.0
        
        if excitation_type in [ExcitationType.MAGNETIC_DIPOLE_Z]:
            if mode == 'TE':
                return 1.0
            else:
                return 0.0
        
        return 0.707

    def combine_degenerate_modes(self, modes_with_amplitudes, frequency, x, y, z=0, t=0):
        """
        线性组合简并模式
        
        参数:
            modes_with_amplitudes: 列表，格式为[(mode_type, m, n, amplitude), ...]
        """
        combined = {'Ex': 0j, 'Ey': 0j, 'Ez': 0j, 'Hx': 0j, 'Hy': 0j, 'Hz': 0j}
        
        for mode_type, m, n, amplitude in modes_with_amplitudes:
            fields = self.get_fields_complex(m, n, frequency, x, y, z, t, mode_type)
            for key in combined:
                combined[key] += amplitude * fields[key]
        
        return combined

    def generate_polarized_mode(self, m, n, frequency, polarization, x, y, z=0, t=0):
        """
        生成特定极化方向的模式（简并模式的线性组合）
        
        支持: X极化, Y极化, 右旋圆极化, 左旋圆极化
        """
        degenerate = self.find_degenerate_modes(m, n, frequency)
        
        if not degenerate:
            fields = self.get_fields_complex(m, n, frequency, x, y, z, t, 'TE')
            return fields
        
        all_modes = [('TE', m, n)] + [(d[0], d[1], d[2]) for d in degenerate]
        
        if polarization == Polarization.X_POLARIZED:
            amplitudes = []
            for mode_type, mm, nn in all_modes:
                if mm > 0 and nn == 0:
                    amplitudes.append((mode_type, mm, nn, 1.0))
                elif mm == 0 and nn > 0:
                    amplitudes.append((mode_type, mm, nn, 0.0))
                else:
                    coeff = self.symmetry_projection(mm, nn, frequency, 
                                                     ExcitationType.ELECTRIC_DIPOLE_Y, 
                                                     mode_type)
                    amplitudes.append((mode_type, mm, nn, coeff))
        
        elif polarization == Polarization.Y_POLARIZED:
            amplitudes = []
            for mode_type, mm, nn in all_modes:
                if mm > 0 and nn == 0:
                    amplitudes.append((mode_type, mm, nn, 0.0))
                elif mm == 0 and nn > 0:
                    amplitudes.append((mode_type, mm, nn, 1.0))
                else:
                    coeff = self.symmetry_projection(mm, nn, frequency, 
                                                     ExcitationType.ELECTRIC_DIPOLE_X, 
                                                     mode_type)
                    amplitudes.append((mode_type, mm, nn, coeff))
        
        elif polarization == Polarization.CIRCULAR_RIGHT:
            amplitudes = []
            for i, (mode_type, mm, nn) in enumerate(all_modes):
                phase = np.exp(-1j * np.pi / 2 * i)
                amplitudes.append((mode_type, mm, nn, phase / len(all_modes)))
        
        elif polarization == Polarization.CIRCULAR_LEFT:
            amplitudes = []
            for i, (mode_type, mm, nn) in enumerate(all_modes):
                phase = np.exp(1j * np.pi / 2 * i)
                amplitudes.append((mode_type, mm, nn, phase / len(all_modes)))
        
        else:
            raise ValueError("Unknown polarization type")
        
        return self.combine_degenerate_modes(amplitudes, frequency, x, y, z, t)

    def extract_modes_by_symmetry(self, frequency, excitation_type, max_m=3, max_n=3):
        """
        根据激励源对称性，提取所有可能被激发的模式
        """
        extracted = []
        
        for m in range(max_m + 1):
            for n in range(max_n + 1):
                if m == 0 and n == 0:
                    continue
                
                for mode_type in ['TE', 'TM']:
                    try:
                        if self.is_propagating(m, n, frequency, mode_type):
                            coeff = self.symmetry_projection(m, n, frequency, 
                                                             excitation_type, mode_type)
                            if abs(coeff) > 1e-10:
                                extracted.append((mode_type, m, n, coeff))
                    except ValueError:
                        pass
        
        return sorted(extracted, key=lambda x: self.cutoff_frequency(x[1], x[2], x[0]))

    def find_propagating_modes(self, frequency, max_m=5, max_n=5):
        modes = []
        
        for m in range(max_m + 1):
            for n in range(max_n + 1):
                if m == 0 and n == 0:
                    continue
                try:
                    if self.is_propagating(m, n, frequency, 'TE'):
                        modes.append(('TE', m, n))
                except ValueError:
                    pass
                
                try:
                    if self.is_propagating(m, n, frequency, 'TM'):
                        modes.append(('TM', m, n))
                except ValueError:
                    pass
        
        return modes

    def plot_field(self, m, n, frequency, field_component='E', mode='TE', 
                   resolution=100, title=None, polarization=None):
        x = np.linspace(0, self.a, resolution)
        y = np.linspace(0, self.b, resolution)
        X, Y = np.meshgrid(x, y)
        
        if polarization is not None:
            fields = self.generate_polarized_mode(m, n, frequency, polarization, X, Y)
            fields = {key: np.real(value) for key, value in fields.items()}
            mode_label = f"{mode}_{m}{n}_{polarization.value}"
        else:
            fields = self.get_fields(m, n, frequency, X, Y, mode=mode)
            mode_label = f"{mode}_{m}{n}"
        
        if field_component == 'E':
            E_mag = np.sqrt(fields['Ex']**2 + fields['Ey']**2 + fields['Ez']**2)
            data = E_mag
            if title is None:
                title = f'{mode_label} - Electric Field Magnitude'
        elif field_component == 'H':
            H_mag = np.sqrt(fields['Hx']**2 + fields['Hy']**2 + fields['Hz']**2)
            data = H_mag
            if title is None:
                title = f'{mode_label} - Magnetic Field Magnitude'
        elif field_component in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            data = fields[field_component]
            if title is None:
                title = f'{mode_label} - {field_component} Field'
        else:
            raise ValueError("Invalid field component")
        
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(121)
        contour = ax1.contourf(X, Y, data, 50, cmap='viridis')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title(title + ' (Contour)')
        plt.colorbar(contour, ax=ax1)
        
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, data, cmap='viridis', edgecolor='none')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_zlabel('Field Amplitude')
        ax2.set_title(title + ' (3D)')
        plt.colorbar(surf, ax=ax2)
        
        plt.tight_layout()
        return fig

    def plot_polarization_comparison(self, m, n, frequency, mode='TE', resolution=100):
        """绘制不同极化方向的对比图"""
        x = np.linspace(0, self.a, resolution)
        y = np.linspace(0, self.b, resolution)
        X, Y = np.meshgrid(x, y)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        polarizations = [Polarization.X_POLARIZED, Polarization.Y_POLARIZED,
                        Polarization.CIRCULAR_RIGHT, Polarization.CIRCULAR_LEFT]
        components = ['Ex', 'Ey']
        
        for i, pol in enumerate(polarizations):
            fields = self.generate_polarized_mode(m, n, frequency, pol, X, Y)
            for j, comp in enumerate(components):
                ax = axes[j, i]
                data = np.real(fields[comp])
                contour = ax.contourf(X, Y, data, 50, cmap='RdBu_r')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_title(f'{pol.value} - {comp}')
                plt.colorbar(contour, ax=ax)
        
        plt.tight_layout()
        return fig

    def print_mode_info(self, m, n, frequency, mode='TE'):
        lambda_c = self.cutoff_wavelength(m, n, mode)
        f_c = self.cutoff_frequency(m, n, mode)
        is_prop = self.is_propagating(m, n, frequency, mode)
        
        print(f"\n{mode}_{m}{n} Mode Information:")
        print(f"  Cutoff Wavelength: {lambda_c*1000:.2f} mm")
        print(f"  Cutoff Frequency:  {f_c/1e9:.4f} GHz")
        print(f"  Operating Frequency: {frequency/1e9:.4f} GHz")
        print(f"  Status: {'Propagating' if is_prop else 'Evanescent'}")
        
        degenerate = self.find_degenerate_modes(m, n, frequency)
        if degenerate:
            print(f"  Degenerate Modes: {', '.join([f'{d[0]}{d[1]}{d[2]}' for d in degenerate])}")
        else:
            print(f"  Degenerate Modes: None")
        
        if is_prop:
            beta = self.propagation_constant(m, n, frequency, mode)
            lambda_g = self.wavelength(m, n, frequency, mode)
            eta = self.wave_impedance(m, n, frequency, mode)
            print(f"  Propagation Constant (β): {np.real(beta):.4f} rad/m")
            print(f"  Guide Wavelength (λg): {lambda_g*1000:.2f} mm")
            print(f"  Wave Impedance: {eta:.2f} Ω")


def main():
    a = 22.86e-3
    b = 10.16e-3
    frequency = 10e9
    
    wg = RectangularWaveguide(a, b)
    
    print("=" * 70)
    print("Rectangular Waveguide Analysis - with Polarization Control")
    print("=" * 70)
    print(f"Waveguide Dimensions: {a*1000:.2f} mm x {b*1000:.2f} mm")
    print(f"Operating Frequency: {frequency/1e9:.2f} GHz")
    print("=" * 70)
    
    print("\nDominant Mode (TE_10):")
    wg.print_mode_info(1, 0, frequency, 'TE')
    
    print("\n" + "=" * 70)
    print("All Propagating Modes:")
    print("=" * 70)
    propagating_modes = wg.find_propagating_modes(frequency, max_m=3, max_n=3)
    for mode_type, m, n in propagating_modes:
        lambda_c = wg.cutoff_wavelength(m, n, mode_type)
        f_c = wg.cutoff_frequency(m, n, mode_type)
        print(f"  {mode_type}_{m}{n}: f_c = {f_c/1e9:.4f} GHz, λ_c = {lambda_c*1000:.2f} mm")
    
    print("\n" + "=" * 70)
    print("Degenerate Mode Groups:")
    print("=" * 70)
    groups = wg.find_all_degenerate_groups(max_m=3, max_n=3)
    for i, group in enumerate(groups):
        if len(group) > 1:
            print(f"\nGroup {i+1} (f_c = {group[0][3]/1e9:.4f} GHz):")
            for mode_info in group:
                print(f"  {mode_info[0]}_{mode_info[1]}{mode_info[2]}")
    
    print("\n" + "=" * 70)
    print("Symmetry-based Mode Extraction (Y-directed electric dipole):")
    print("=" * 70)
    extracted = wg.extract_modes_by_symmetry(frequency, ExcitationType.ELECTRIC_DIPOLE_Y)
    for mode_type, m, n, coeff in extracted:
        print(f"  {mode_type}_{m}{n}: excitation coefficient = {coeff:.4f}")
    
    print("\n" + "=" * 70)
    print("Mode Matching Example:")
    print("=" * 70)
    resolution = 50
    x = np.linspace(0, a, resolution)
    y = np.linspace(0, b, resolution)
    X, Y = np.meshgrid(x, y)
    
    source_field = {
        'Ex': np.zeros_like(X) + 0j,
        'Ey': np.sin(np.pi * X / a) * np.exp(-((Y - b/2)**2) / (2 * (b/10)**2)) + 0j,
        'Ez': np.zeros_like(X) + 0j
    }
    
    candidates = [('TE', 1, 0), ('TE', 2, 0), ('TE', 0, 1), ('TE', 1, 1), ('TM', 1, 1)]
    amplitudes = wg.mode_matching(source_field, frequency, candidates, resolution)
    
    print("Projection amplitudes:")
    for (mode_type, m, n), amp in amplitudes.items():
        if abs(amp) > 1e-10:
            print(f"  {mode_type}_{m}{n}: |amp| = {abs(amp):.6f}, phase = {np.angle(amp):.4f} rad")
    
    print("\n" + "=" * 70)
    print("Generating field plots...")
    print("=" * 70)
    
    try:
        fig1 = wg.plot_field(1, 0, frequency, field_component='E', mode='TE')
        fig1.savefig('TE10_E_field.png', dpi=150, bbox_inches='tight')
        
        fig2 = wg.plot_field(1, 0, frequency, field_component='H', mode='TE')
        fig2.savefig('TE10_H_field.png', dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"Plotting error: {e}")
    
    print("\nAnalysis complete!")
    plt.show()


if __name__ == "__main__":
    main()
