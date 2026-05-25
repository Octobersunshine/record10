import numpy as np
from scipy.linalg import matrix_balance, svd, lstsq, eig
import warnings


class PolyMAX:
    """
    PolyMAX算法（pLSCF - Polyreference Least-Squares Complex Frequency）
    
    从频响函数（FRF）中提取模态参数：
    - 固有频率
    - 阻尼比
    - 振型向量
    
    参考文献:
    - Guillaume, P., et al. "A poly-reference implementation of the least-squares complex
      frequency-domain modal parameter estimation method." IMAC, 2003.
    """
    
    def __init__(self, frequencies, frf):
        """
        初始化PolyMAX求解器
        
        参数:
            frequencies: 频率点数组 (N,) 单位: Hz
            frf: 频响函数矩阵 (outputs, inputs, N) 或 (outputs, N) 单位: m/N 或 m/s²/N
        """
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.frf = np.asarray(frf, dtype=np.complex128)
        
        if self.frf.ndim == 2:
            self.frf = self.frf[:, np.newaxis, :]
        
        self.N_freq = len(frequencies)
        self.N_out, self.N_in, _ = self.frf.shape
        
        self.omega = 2 * np.pi * frequencies
        self.s = 1j * self.omega
        
        self.poles = None
        self.residues = None
        self.modal_params = None
        self.stable_poles = None
    
    def _build_orthogonal_polynomials(self, order):
        """构建正交多项式基（改进数值稳定性）"""
        s = self.s / np.max(np.abs(self.s))
        
        basis = np.zeros((order + 1, self.N_freq), dtype=np.complex128)
        basis[0] = 1.0
        
        if order >= 1:
            basis[1] = s
        
        for n in range(2, order + 1):
            alpha_n = np.sum(basis[n-1] * np.conj(basis[n-1])) / np.sum(basis[n-2] * np.conj(basis[n-2]))
            basis[n] = s * basis[n-1] - alpha_n * basis[n-2]
        
        return basis
    
    def solve(self, max_order, output_poles=True):
        """
        执行PolyMAX算法求解
        
        参数:
            max_order: 最大模型阶数
            output_poles: 是否返回每个阶数的极点
        
        返回:
            all_poles: 各阶数的极点列表
        """
        print(f"开始PolyMAX分析，最大阶数: {max_order}")
        
        s_norm = self.s / np.max(np.abs(self.s))
        
        all_poles = []
        
        for order in range(1, max_order + 1):
            print(f"  求解阶数 {order}...", end="\r")
            
            poles, residues = self._solve_single_order(order, s_norm)
            all_poles.append(poles)
            
            if order == max_order:
                self.poles = poles
                self.residues = residues
        
        print(f"\nPolyMAX求解完成，计算了 {max_order} 阶模型")
        
        return all_poles
    
    def _solve_single_order(self, order, s_norm):
        """求解单一阶数的模型"""
        N = self.N_freq
        p = order
        
        R = np.zeros((self.N_out, self.N_in, N), dtype=np.complex128)
        for o in range(self.N_out):
            for i in range(self.N_in):
                R[o, i, :] = self.frf[o, i, :]
        
        A = np.zeros((N * self.N_out * self.N_in, 2 * p + 1), dtype=np.complex128)
        b = np.zeros(N * self.N_out * self.N_in, dtype=np.complex128)
        
        idx = 0
        for o in range(self.N_out):
            for i in range(self.N_in):
                for k in range(N):
                    sk = s_norm[k]
                    
                    row = np.zeros(2 * p + 1, dtype=np.complex128)
                    for n in range(p + 1):
                        row[n] = -R[o, i, k] * (sk ** n)
                    for n in range(p):
                        row[p + 1 + n] = sk ** n
                    
                    A[idx] = row
                    b[idx] = R[o, i, k] * (sk ** p)
                    idx += 1
        
        x, residuals, rank, s = lstsq(A, b, rcond=None)
        
        alpha = np.zeros(p + 1, dtype=np.complex128)
        beta = np.zeros(p, dtype=np.complex128)
        alpha[p] = 1.0
        alpha[:p] = x[:p]
        beta[:] = x[p:2*p]
        
        companion = np.zeros((p, p), dtype=np.complex128)
        for i in range(p-1):
            companion[i+1, i] = 1.0
        companion[0, :] = -alpha[:p]
        
        poles_raw, eigvecs = eig(companion)
        
        poles = poles_raw * np.max(np.abs(self.s))
        
        residues = self._compute_residues(poles, order)
        
        return poles, residues
    
    def _compute_residues(self, poles, order):
        """计算留数矩阵"""
        N_poles = len(poles)
        residues = np.zeros((self.N_out, self.N_in, N_poles), dtype=np.complex128)
        
        for o in range(self.N_out):
            for i in range(self.N_in):
                A = np.zeros((self.N_freq, 2 * N_poles), dtype=np.complex128)
                b = self.frf[o, i, :].copy()
                
                for k, s in enumerate(self.s):
                    for p_idx, pole in enumerate(poles):
                        A[k, p_idx] = 1.0 / (s - pole)
                        A[k, N_poles + p_idx] = 1.0 / (s - np.conj(pole))
                
                x, _, _, _ = lstsq(A, b, rcond=None)
                residues[o, i, :] = x[:N_poles]
        
        return residues
    
    def stabilization_diagram(self, max_order, 
                               freq_tol=0.01, 
                               damp_tol=0.05,
                               freq_threshold=1e-6):
        """
        生成稳定图
        
        参数:
            max_order: 最大阶数
            freq_tol: 频率相对容差
            damp_tol: 阻尼相对容差
            freq_threshold: 最小频率阈值
        
        返回:
            stable_poles: 稳定极点列表
            diagram_data: 绘图数据
        """
        print("生成稳定图...")
        
        all_poles = []
        all_damping = []
        all_frequencies = []
        
        for order in range(1, max_order + 1):
            poles, _ = self._solve_single_order(order, self.s / np.max(np.abs(self.s)))
            
            physical_poles = poles[poles.imag >= 0]
            frequencies = np.abs(physical_poles) / (2 * np.pi)
            damping = -np.real(physical_poles) / np.maximum(np.abs(physical_poles), freq_threshold)
            
            valid = (frequencies > freq_threshold) & (damping >= 0) & (damping < 0.5)
            
            all_poles.append(physical_poles[valid])
            all_frequencies.append(frequencies[valid])
            all_damping.append(damping[valid])
        
        stable_poles = []
        stable_frequencies = []
        stable_damping = []
        
        for i in range(1, max_order):
            if len(all_frequencies[i]) == 0 or len(all_frequencies[i-1]) == 0:
                continue
            
            for j, freq in enumerate(all_frequencies[i]):
                for k, freq_prev in enumerate(all_frequencies[i-1]):
                    freq_rel = abs(freq - freq_prev) / max(freq, freq_prev)
                    damp_rel = abs(all_damping[i][j] - all_damping[i-1][k]) / max(all_damping[i][j], all_damping[i-1][k], 1e-6)
                    
                    if freq_rel < freq_tol and damp_rel < damp_tol:
                        stable_poles.append(all_poles[i][j])
                        stable_frequencies.append(freq)
                        stable_damping.append(all_damping[i][j])
                        break
        
        unique_indices = []
        used = set()
        
        for i in range(len(stable_frequencies)):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(stable_frequencies)):
                if j in used:
                    continue
                freq_rel = abs(stable_frequencies[j] - stable_frequencies[i]) / max(stable_frequencies[j], stable_frequencies[i])
                if freq_rel < freq_tol:
                    group.append(j)
                    used.add(j)
            
            avg_idx = group[np.argmin([stable_damping[idx] for idx in group])]
            unique_indices.append(avg_idx)
            used.add(i)
        
        self.stable_poles = {
            'poles': [stable_poles[i] for i in unique_indices],
            'frequencies': [stable_frequencies[i] for i in unique_indices],
            'damping': [stable_damping[i] for i in unique_indices]
        }
        
        print(f"识别到 {len(self.stable_poles['frequencies'])} 个稳定模态")
        
        return self.stable_poles, {
            'all_frequencies': all_frequencies,
            'all_damping': all_damping,
            'stable_frequencies': self.stable_poles['frequencies'],
            'stable_damping': self.stable_poles['damping']
        }
    
    def extract_modal_params(self, poles=None, mode_tol=0.01):
        """
        提取完整的模态参数（频率、阻尼、振型）
        
        参数:
            poles: 指定的极点列表（None则使用稳定图结果）
            mode_tol: 模态频率容差
        
        返回:
            modal_params: 字典包含频率、阻尼、振型
        """
        if poles is None:
            if self.stable_poles is None:
                raise ValueError("请先运行stabilization_diagram()或提供极点")
            poles = self.stable_poles['poles']
        
        N_modes = len(poles)
        frequencies = np.abs(poles) / (2 * np.pi)
        damping = -np.real(poles) / np.maximum(np.abs(poles), 1e-6)
        
        mode_shapes = np.zeros((self.N_out, N_modes), dtype=np.complex128)
        
        for m, pole in enumerate(poles):
            mode_shape = np.zeros(self.N_out, dtype=np.complex128)
            
            for o in range(self.N_out):
                amplitude_sum = 0.0
                for i in range(self.N_in):
                    A = np.zeros((self.N_freq, 2), dtype=np.complex128)
                    for k, s in enumerate(self.s):
                        A[k, 0] = 1.0 / (s - pole)
                        A[k, 1] = 1.0 / (s - np.conj(pole))
                    
                    x, _, _, _ = lstsq(A, self.frf[o, i, :], rcond=None)
                    amplitude_sum += np.abs(x[0])
                
                mode_shape[o] = amplitude_sum / self.N_in
            
            max_idx = np.argmax(np.abs(mode_shape))
            mode_shapes[:, m] = mode_shape / mode_shape[max_idx]
        
        sort_idx = np.argsort(frequencies)
        
        self.modal_params = {
            'frequencies': frequencies[sort_idx],
            'damping': damping[sort_idx],
            'mode_shapes': mode_shapes[:, sort_idx]
        }
        
        return self.modal_params
    
    def print_results(self):
        """打印模态参数结果"""
        if self.modal_params is None:
            print("请先调用 extract_modal_params()")
            return
        
        print("=" * 80)
        print("PolyMAX 实验模态分析结果")
        print("=" * 80)
        print(f"{'模态':<6} {'频率(Hz)':<15} {'阻尼比(%)':<15} {'频率分辨率':<15}")
        print("-" * 80)
        
        for i in range(len(self.modal_params['frequencies'])):
            freq = self.modal_params['frequencies'][i]
            damp = self.modal_params['damping'][i] * 100
            print(f"{i+1:<6} {freq:<15.4f} {damp:<15.4f} {'自动':<15}")
        
        print("-" * 80)
        
        print("\n振型 (取参考点归一化):")
        for i in range(len(self.modal_params['frequencies'])):
            ms = self.modal_params['mode_shapes'][:, i]
            print(f"\n模态 {i+1}:")
            for o in range(self.N_out):
                mag = np.abs(ms[o])
                phase = np.angle(ms[o], deg=True)
                print(f"  测点{o+1}: 幅值={mag:.4f} 相位={phase:.2f}°")
    
    def reconstruct_frf(self, modal_params=None):
        """
        从模态参数重构FRF（用于验证）
        
        参数:
            modal_params: 模态参数字典
        
        返回:
            frf_recon: 重构的FRF
        """
        if modal_params is None:
            modal_params = self.modal_params
        
        freqs = modal_params['frequencies']
        damps = modal_params['damping']
        shapes = modal_params['mode_shapes']
        
        frf_recon = np.zeros_like(self.frf)
        
        for m in range(len(freqs)):
            omega_m = 2 * np.pi * freqs[m]
            sigma_m = damps[m] * omega_m
            
            pole = -sigma_m + 1j * omega_m * np.sqrt(1 - damps[m]**2)
            
            for o in range(self.N_out):
                for i in range(self.N_in):
                    residue = shapes[o, m] * shapes[o, m]
                    for k, s in enumerate(self.s):
                        frf_recon[o, i, k] += residue / (s - pole) + np.conj(residue) / (s - np.conj(pole))
        
        return frf_recon
    
    def compute_fit_quality(self, frf_recon=None):
        """
        计算拟合质量（MAC和相关性）
        
        参数:
            frf_recon: 重构的FRF
        
        返回:
            fit_quality: 拟合质量指标
        """
        if frf_recon is None:
            frf_recon = self.reconstruct_frf()
        
        mac = np.zeros(self.N_out)
        
        for o in range(self.N_out):
            for i in range(self.N_in):
                orig = self.frf[o, i, :]
                recon = frf_recon[o, i, :]
                
                numerator = np.abs(np.sum(orig * np.conj(recon)))**2
                denominator = np.sum(np.abs(orig)**2) * np.sum(np.abs(recon)**2)
                mac[o] = numerator / denominator if denominator > 0 else 0
        
        fit_error = np.mean(np.abs(self.frf - frf_recon) / np.maximum(np.abs(self.frf), 1e-10))
        
        return {
            'MAC': mac,
            'mean_MA_error': fit_error,
            'max_error': np.max(np.abs(self.frf - frf_recon))
        }


def generate_synthetic_frf(frequencies, natural_freqs, damping_ratios, 
                            mode_shapes, sensor_indices=None, ref_index=0,
                            noise_level=0.0):
    """
    生成合成FRF数据（用于测试）
    
    参数:
        frequencies: 频率点数组
        natural_freqs: 各阶固有频率 (N_modes,)
        damping_ratios: 各阶阻尼比 (N_modes,)
        mode_shapes: 振型矩阵 (N_dof, N_modes)
        sensor_indices: 传感器位置索引
        ref_index: 参考点索引
        noise_level: 噪声水平
    
    返回:
        frf: 频响函数矩阵 (N_sensors, N_refs, N_freqs)
    """
    N_freq = len(frequencies)
    N_modes = len(natural_freqs)
    N_dof = mode_shapes.shape[0]
    
    if sensor_indices is None:
        sensor_indices = list(range(N_dof))
    
    N_sensors = len(sensor_indices)
    N_refs = 1
    
    omega = 2 * np.pi * frequencies
    s = 1j * omega
    
    frf = np.zeros((N_sensors, N_refs, N_freq), dtype=np.complex128)
    
    for m in range(N_modes):
        omega_m = 2 * np.pi * natural_freqs[m]
        sigma_m = damping_ratios[m] * omega_m
        omega_d = omega_m * np.sqrt(1 - damping_ratios[m]**2)
        
        pole = -sigma_m + 1j * omega_d
        
        for s_idx, sensor in enumerate(sensor_indices):
            for r_idx in range(N_refs):
                residue = mode_shapes[sensor, m] * mode_shapes[ref_index, m]
                
                for k in range(N_freq):
                    frf[s_idx, r_idx, k] += residue / (s[k] - pole) + \
                                            np.conj(residue) / (s[k] - np.conj(pole))
    
    if noise_level > 0:
        noise = noise_level * np.random.randn(*frf.shape) * np.max(np.abs(frf))
        frf += noise
    
    return frf


def example_simple_system():
    """示例1: 简单3自由度系统"""
    print("=" * 80)
    print("示例1: 3自由度系统模态参数识别")
    print("=" * 80)
    
    frequencies = np.linspace(0, 100, 1025)
    
    natural_freqs_true = np.array([10.0, 25.0, 40.0])
    damping_ratios_true = np.array([0.01, 0.015, 0.02])
    
    mode_shapes_true = np.array([
        [1.0, 1.0, 1.0],
        [0.8, -0.2, -0.9],
        [0.5, -0.8, 0.7]
    ])
    
    print(f"\n真实模态参数:")
    for i in range(3):
        print(f"  模态 {i+1}: f={natural_freqs_true[i]} Hz, ξ={damping_ratios_true[i]*100}%")
    
    frf = generate_synthetic_frf(
        frequencies, natural_freqs_true, damping_ratios_true,
        mode_shapes_true, noise_level=0.005
    )
    
    print(f"\nFRF数据维度: {frf.shape} (传感器×参考点×频率点)")
    
    polymax = PolyMAX(frequencies, frf)
    
    stable_poles, diagram_data = polymax.stabilization_diagram(
        max_order=12,
        freq_tol=0.02,
        damp_tol=0.1
    )
    
    modal_params = polymax.extract_modal_params()
    polymax.print_results()
    
    print("\n与真实值对比:")
    print("-" * 50)
    for i in range(min(3, len(modal_params['frequencies']))):
        f_est = modal_params['frequencies'][i]
        d_est = modal_params['damping'][i]
        f_err = abs(f_est - natural_freqs_true[i]) / natural_freqs_true[i] * 100
        d_err = abs(d_est - damping_ratios_true[i]) / damping_ratios_true[i] * 100
        print(f"模态 {i+1}: 频率误差={f_err:.2f}%, 阻尼误差={d_err:.2f}%")
    
    fit_quality = polymax.compute_fit_quality()
    print(f"\n拟合质量: 平均误差={fit_quality['mean_MA_error']*100:.2f}%")
    
    return polymax


def example_close_modes_ema():
    """示例2: 密集模态识别"""
    print("\n" + "=" * 80)
    print("示例2: 密集模态识别测试")
    print("=" * 80)
    
    frequencies = np.linspace(20, 30, 513)
    
    natural_freqs_true = np.array([24.98, 25.02])
    damping_ratios_true = np.array([0.008, 0.01])
    
    mode_shapes_true = np.array([
        [1.0, 1.0],
        [0.9, -0.85],
        [0.7, 0.75],
        [0.5, -0.6]
    ])
    
    print(f"\n真实密集模态:")
    print(f"  模态1: f={natural_freqs_true[0]} Hz, ξ={damping_ratios_true[0]*100}%")
    print(f"  模态2: f={natural_freqs_true[1]} Hz, ξ={damping_ratios_true[1]*100}%")
    print(f"  频率间隔: {natural_freqs_true[1] - natural_freqs_true[0]} Hz")
    
    frf = generate_synthetic_frf(
        frequencies, natural_freqs_true, damping_ratios_true,
        mode_shapes_true, noise_level=0.002
    )
    
    polymax = PolyMAX(frequencies, frf)
    
    stable_poles, _ = polymax.stabilization_diagram(
        max_order=10,
        freq_tol=0.005,
        damp_tol=0.1
    )
    
    modal_params = polymax.extract_modal_params()
    polymax.print_results()
    
    return polymax


def example_model_correlation():
    """示例3: 模型修正相关（FEM vs EMA对比）"""
    print("\n" + "=" * 80)
    print("示例3: 试验-仿真模型相关分析")
    print("=" * 80)
    
    from modal_analysis_lanczos import LanczosSolver
    
    n = 5
    M = np.eye(n) * 1.0
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0 * 10000
        if i > 0:
            K[i, i-1] = -1.0 * 10000
        if i < n - 1:
            K[i, i+1] = -1.0 * 10000
    
    solver = LanczosSolver(K, M)
    solver.solve(num_eigenvalues=3, tol=1e-9)
    fem_freqs = solver.get_frequencies()
    fem_modes = solver.eigenvectors
    
    print(f"\nFEM计算频率: {fem_freqs}")
    
    frequencies = np.linspace(0, 60, 513)
    damping_ratios = np.array([0.01, 0.012, 0.015])
    
    frf = generate_synthetic_frf(
        frequencies, fem_freqs, damping_ratios,
        fem_modes, noise_level=0.01
    )
    
    polymax = PolyMAX(frequencies, frf)
    stable_poles, _ = polymax.stabilization_diagram(max_order=10)
    modal_params = polymax.extract_modal_params()
    
    print("\nEMA识别结果 vs FEM计算结果:")
    print("-" * 60)
    print(f"{'模态':<6} {'FEM频率(Hz)':<15} {'EMA频率(Hz)':<15} {'误差(%)':<12} {'EMA阻尼(%)':<12}")
    print("-" * 60)
    
    for i in range(min(3, len(modal_params['frequencies']))):
        fem_f = fem_freqs[i]
        ema_f = modal_params['frequencies'][i]
        err = abs(ema_f - fem_f) / fem_f * 100
        damp = modal_params['damping'][i] * 100
        print(f"{i+1:<6} {fem_f:<15.4f} {ema_f:<15.4f} {err:<12.2f} {damp:<12.4f}")
    
    print("\n模型修正建议:")
    print("  - 如果频率误差 > 5%，需要修正刚度或质量分布")
    print("  - 阻尼比由EMA识别，FEM模型通常不包含阻尼")
    print("  - 可通过振型MAC值评估振型相关性")
    
    return polymax, solver


if __name__ == "__main__":
    np.random.seed(42)
    
    example_simple_system()
    example_close_modes_ema()
    example_model_correlation()
    
    print("\n" + "=" * 80)
    print("PolyMAX算法使用指南:")
    print("=" * 80)
    print("1. 准备数据:")
    print("   - frequencies: 频率点数组")
    print("   - frf: 频响函数矩阵 (N_sensors, N_refs, N_freqs)")
    print("2. 创建求解器: polymax = PolyMAX(frequencies, frf)")
    print("3. 生成稳定图: stable_poles, _ = polymax.stabilization_diagram(max_order=20)")
    print("4. 提取模态参数: params = polymax.extract_modal_params()")
    print("5. 打印结果: polymax.print_results()")
    print("6. 验证拟合: fit_quality = polymax.compute_fit_quality()")
    print("=" * 80)
