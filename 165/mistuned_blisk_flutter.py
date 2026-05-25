import numpy as np
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，绘图功能将不可用")


class DryFrictionDamping:
    """干摩擦阻尼模型
    
    包含:
    1. Coulomb摩擦模型
    2. 等效粘性阻尼（谐波平衡法）
    3. 摩擦界面刚度
    """
    
    def __init__(self, **kwargs):
        """
        参数:
            friction_coeff: 摩擦系数 μ
            normal_load: 法向载荷 N
            contact_stiffness: 接触刚度 k_contact
            amplitude: 振动幅值（用于等效线性化）
        """
        self.mu = kwargs.get('friction_coeff', 0.3)
        self.N = kwargs.get('normal_load', 100.0)
        self.k_contact = kwargs.get('contact_stiffness', 1e6)
        self.amplitude = kwargs.get('amplitude', 1e-5)
    
    def equivalent_damping(self, omega, amplitude=None):
        """计算等效粘性阻尼系数（谐波平衡法）
        
        参数:
            omega: 角频率
            amplitude: 振动幅值
            
        返回:
            c_eq: 等效阻尼系数
            k_eq: 等效刚度
        """
        if amplitude is None:
            amplitude = self.amplitude
        
        F_max = self.mu * self.N
        
        if amplitude < 1e-10:
            return 0.0, self.k_contact
        
        slip_ratio = F_max / (self.k_contact * amplitude)
        
        if slip_ratio <= 1.0:
            k_eq = self.k_contact * (1 - 4 * slip_ratio / (3 * np.pi))
            c_eq = 4 * F_max / (np.pi * omega * amplitude)
        else:
            k_eq = 0.0
            c_eq = 4 * F_max / (np.pi * omega * amplitude)
        
        return c_eq, k_eq
    
    def friction_force(self, velocity):
        """Coulomb摩擦力"""
        return -self.mu * self.N * np.tanh(velocity / 1e-6)
    
    def energy_dissipation(self, amplitude, omega):
        """计算一个周期内的能量耗散"""
        F_max = self.mu * self.N
        return 4 * F_max * amplitude


class RogerApproximation:
    """Roger有理函数近似 - Theodorsen函数"""
    
    def __init__(self, order=4):
        self.order = order
        
        if order == 2:
            self.A = np.array([0.5000, 0.5000])
            self.b = np.array([0.0455, 0.3000])
        elif order == 4:
            self.A = np.array([0.1650, 0.3350, 0.3350, 0.1650])
            self.b = np.array([0.0120, 0.0600, 0.2000, 0.6000])
        elif order == 6:
            self.A = np.array([0.0880, 0.1760, 0.2360, 0.2360, 0.1760, 0.0880])
            self.b = np.array([0.0050, 0.0200, 0.0700, 0.2000, 0.5000, 1.2000])
        else:
            raise ValueError("Roger近似阶数只支持 2, 4, 6")
    
    def __call__(self, k):
        if k == 0:
            return 1.0 + 0.0j
        s = 1j * k
        C = 1.0
        for i in range(self.order):
            C -= self.A[i] * s / (1.0 + self.b[i] * s)
        return C


class MistunedBliskAnalysis:
    """失谐叶盘颤振分析
    
    包含:
    1. 叶盘耦合模型（N个叶片 + 圆盘）
    2. 干摩擦阻尼
    3. 频率失谐模型
    4. 模态局部化分析
    5. 颤振风险评估
    """
    
    def __init__(self, **kwargs):
        """
        参数:
            N_blades: 叶片数量
            rho: 空气密度
            b: 叶片半弦长
            m_per_blade: 单个叶片质量
            I_per_blade: 单个叶片转动惯量
            omega_h_nom: 名义弯曲频率 (rad/s)
            omega_alpha_nom: 名义扭转频率 (rad/s)
            zeta_struct: 结构阻尼比
            a: 弹性轴位置
            k_disk: 圆盘刚度
            k_coupling: 叶间耦合刚度
            mistune_std: 失谐标准差（百分比）
            friction_damping: 干摩擦阻尼对象
            num_aero_states: 气动力状态数
        """
        self.N_blades = kwargs.get('N_blades', 24)
        self.rho = kwargs.get('rho', 1.225)
        self.b = kwargs.get('b', 0.15)
        self.m_per_blade = kwargs.get('m_per_blade', 8.0)
        self.I_per_blade = kwargs.get('I_per_blade', 0.08)
        self.omega_h_nom = kwargs.get('omega_h_nom', 2 * np.pi * 8)
        self.omega_alpha_nom = kwargs.get('omega_alpha_nom', 2 * np.pi * 25)
        self.zeta_struct = kwargs.get('zeta_struct', 0.005)
        self.a = kwargs.get('a', -0.3)
        self.k_disk = kwargs.get('k_disk', 1e8)
        self.k_coupling = kwargs.get('k_coupling', 1e6)
        self.mistune_std = kwargs.get('mistune_std', 0.03)
        self.num_aero_states = kwargs.get('num_aero_states', 4)
        
        self.friction = kwargs.get('friction_damping', DryFrictionDamping())
        self.roger = RogerApproximation(order=self.num_aero_states)
        
        self.mistune_h = None
        self.mistune_alpha = None
        self._generate_mistune()
        
        self.k_h_nom = self.m_per_blade * self.omega_h_nom**2
        self.k_alpha_nom = self.I_per_blade * self.omega_alpha_nom**2
    
    def _generate_mistune(self):
        """生成随机失谐（频率偏差）"""
        self.mistune_h = np.random.normal(1.0, self.mistune_std, self.N_blades)
        self.mistune_alpha = np.random.normal(1.0, self.mistune_std, self.N_blades)
    
    def set_mistune(self, mistune_h, mistune_alpha=None):
        """设置特定的失谐模式"""
        self.mistune_h = mistune_h
        if mistune_alpha is None:
            self.mistune_alpha = mistune_h
        else:
            self.mistune_alpha = mistune_alpha
    
    def build_structural_matrices(self, omega=0.0, amplitude=1e-5):
        """构建结构质量、阻尼、刚度矩阵
        
        自由度顺序: [h1, α1, h2, α2, ..., hN, αN, θ_disk]
        """
        N = self.N_blades
        n_dof = 2 * N + 1
        
        M = np.zeros((n_dof, n_dof))
        D = np.zeros((n_dof, n_dof))
        K = np.zeros((n_dof, n_dof))
        
        for i in range(N):
            idx_h = 2 * i
            idx_a = 2 * i + 1
            
            M[idx_h, idx_h] = self.m_per_blade
            M[idx_a, idx_a] = self.I_per_blade
            
            D[idx_h, idx_h] = 2 * self.m_per_blade * self.zeta_struct * self.omega_h_nom
            D[idx_a, idx_a] = 2 * self.I_per_blade * self.zeta_struct * self.omega_alpha_nom
            
            if omega > 0:
                c_friction, k_friction = self.friction.equivalent_damping(omega, amplitude)
                D[idx_a, idx_a] += c_friction
                K[idx_a, idx_a] += k_friction
            
            K[idx_h, idx_h] = self.k_h_nom * self.mistune_h[i]
            K[idx_a, idx_a] = self.k_alpha_nom * self.mistune_alpha[i]
            
            K[idx_h, 2*N] = -self.k_disk
            K[2*N, idx_h] = -self.k_disk
        
        K[2*N, 2*N] = self.k_disk * N
        
        for i in range(N):
            j = (i + 1) % N
            K[2*i, 2*j] = -self.k_coupling
            K[2*j, 2*i] = -self.k_coupling
            K[2*i, 2*i] += self.k_coupling
            K[2*j, 2*j] += self.k_coupling
        
        M[2*N, 2*N] = 100.0
        D[2*N, 2*N] = 2 * M[2*N, 2*N] * self.zeta_struct * self.omega_h_nom * 0.5
        
        return M, D, K
    
    def build_aerodynamic_matrices(self, U, interblade_phase=0.0):
        """构建气动力矩阵（基于Theodorsen理论+行波理论）
        
        参数:
            U: 来流速度
            interblade_phase: 叶间相位角
        """
        N = self.N_blades
        n_dof = 2 * N
        
        A_aero = np.zeros((n_dof, n_dof), dtype=complex)
        b = self.b
        a = self.a
        rho = self.rho
        
        for i in range(N):
            idx_h = 2 * i
            idx_a = 2 * i + 1
            
            k_mean = (self.omega_h_nom + self.omega_alpha_nom) / 2 * b / U
            Ck = self.roger(k_mean)
            
            L_h = 2 * np.pi * rho * U * b * (1j * k_mean + Ck * 1j * k_mean)
            L_a = 2 * np.pi * rho * U * b**2 * (1 + 1j * k_mean * (0.5 - a) + 
                                                 Ck * 1j * k_mean * (0.5 + a))
            M_h = 2 * np.pi * rho * U * b**2 * (1j * k_mean * (0.5 + a) + 
                                                Ck * 1j * k_mean * (0.5 + a))
            M_a = 2 * np.pi * rho * U * b**3 * ((0.5 + a) + 
                                                 1j * k_mean * (1/8 + (0.5 + a)**2) +
                                                 Ck * 1j * k_mean * (0.5 + a)**2)
            
            for j in range(N):
                phase_factor = np.exp(1j * interblade_phase * (j - i))
                
                A_aero[idx_h, 2*j] += L_h * phase_factor / N
                A_aero[idx_h, 2*j+1] += L_a * phase_factor / N
                A_aero[idx_a, 2*j] += M_h * phase_factor / N
                A_aero[idx_a, 2*j+1] += M_a * phase_factor / N
        
        return A_aero
    
    def modal_analysis(self, omega_ref=0.0):
        """模态分析 - 计算固有频率和振型"""
        M, D, K = self.build_structural_matrices(omega_ref)
        
        M_blade = M[:-1, :-1]
        K_blade = K[:-1, :-1]
        
        eigenvalues, eigenvectors = linalg.eig(K_blade, M_blade)
        
        freq_hz = np.sqrt(np.abs(eigenvalues.real)) / (2 * np.pi)
        sorted_idx = np.argsort(freq_hz)
        
        return freq_hz[sorted_idx], eigenvectors[:, sorted_idx]
    
    def analyze_localization(self, mode_idx=None):
        """模态局部化分析
        
        计算局部化因子和振型幅值分布
        """
        freqs, modes = self.modal_analysis()
        
        N = self.N_blades
        
        if mode_idx is None:
            mode_indices = range(len(freqs))
        else:
            mode_indices = [mode_idx] if np.isscalar(mode_idx) else mode_idx
        
        localization_factors = []
        amplitude_distributions = []
        
        for idx in mode_indices:
            mode = modes[:, idx]
            
            bend_amplitudes = np.abs(mode[0::2])
            twist_amplitudes = np.abs(mode[1::2])
            total_amplitudes = bend_amplitudes + twist_amplitudes
            
            if np.sum(total_amplitudes) > 1e-10:
                normalized_amps = total_amplitudes / np.sum(total_amplitudes)
            else:
                normalized_amps = total_amplitudes
            
            LF = np.sum(normalized_amps**2) * N
            
            localization_factors.append({
                'frequency': freqs[idx],
                'localization_factor': LF,
                'amplitude_distribution': normalized_amps
            })
            amplitude_distributions.append(normalized_amps)
        
        return localization_factors, amplitude_distributions
    
    def flutter_analysis(self, U_range):
        """失谐叶盘颤振分析（能量法）"""
        results = []
        
        N = self.N_blades
        freqs, modes = self.modal_analysis()
        
        n_modes = min(2 * N, 20)
        
        for U in U_range:
            mode_dampings = []
            
            for m in range(n_modes):
                mode_shape = modes[:, m]
                mode_freq_omega = 2 * np.pi * freqs[m]
                
                phase_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                max_damping = -np.inf
                
                for phase in phase_angles:
                    A_aero = self.build_aerodynamic_matrices(U, phase)
                    
                    M, D, K = self.build_structural_matrices(mode_freq_omega)
                    M_blade = M[:-1, :-1]
                    D_blade = D[:-1, :-1]
                    K_blade = K[:-1, :-1]
                    
                    modal_mass = mode_shape.conj() @ M_blade @ mode_shape
                    modal_damp = mode_shape.conj() @ D_blade @ mode_shape
                    modal_stiff = mode_shape.conj() @ K_blade @ mode_shape
                    modal_aero = mode_shape.conj() @ A_aero @ mode_shape
                    
                    effective_mass = modal_mass.real
                    effective_damp = modal_damp.real + modal_aero.imag
                    effective_stiff = modal_stiff.real + modal_aero.real
                    
                    if effective_mass > 1e-10:
                        omega_eff = np.sqrt(max(0, effective_stiff / effective_mass))
                        if omega_eff > 0:
                            damping_ratio = effective_damp / (2 * effective_mass * omega_eff)
                        else:
                            damping_ratio = -np.inf
                    else:
                        damping_ratio = -np.inf
                    
                    if damping_ratio > max_damping:
                        max_damping = damping_ratio
                
                mode_dampings.append(max_damping)
            
            results.append({
                'U': U,
                'max_damping': max(mode_dampings),
                'mode_dampings': mode_dampings,
                'critical_mode': np.argmax(mode_dampings)
            })
        
        return results
    
    def monte_carlo_flutter(self, n_samples=50, U_range=None):
        """蒙特卡洛模拟 - 评估失谐叶盘颤振风险
        
        参数:
            n_samples: 样本数
            U_range: 速度范围
        """
        if U_range is None:
            U_range = np.linspace(50, 200, 30)
        
        flutter_speeds = []
        localization_data = []
        
        original_mistune_h = self.mistune_h.copy()
        original_mistune_alpha = self.mistune_alpha.copy()
        
        for sample in range(n_samples):
            self._generate_mistune()
            
            loc_factors, _ = self.analyze_localization()
            avg_loc = np.mean([lf['localization_factor'] for lf in loc_factors])
            max_loc = np.max([lf['localization_factor'] for lf in loc_factors])
            
            flutter_results = self.flutter_analysis(U_range)
            
            flutter_speed = None
            for i, res in enumerate(flutter_results):
                if res['max_damping'] > 0:
                    if i > 0:
                        U0 = flutter_results[i-1]['U']
                        U1 = res['U']
                        d0 = flutter_results[i-1]['max_damping']
                        d1 = res['max_damping']
                        flutter_speed = U0 - d0 * (U1 - U0) / (d1 - d0)
                    else:
                        flutter_speed = U_range[0]
                    break
            
            if flutter_speed is None:
                flutter_speed = U_range[-1]
            
            flutter_speeds.append(flutter_speed)
            localization_data.append({
                'avg_localization': avg_loc,
                'max_localization': max_loc,
                'flutter_speed': flutter_speed
            })
        
        self.mistune_h = original_mistune_h
        self.mistune_alpha = original_mistune_alpha
        
        return {
            'flutter_speeds': np.array(flutter_speeds),
            'mean_flutter': np.mean(flutter_speeds),
            'std_flutter': np.std(flutter_speeds),
            'min_flutter': np.min(flutter_speeds),
            'max_flutter': np.max(flutter_speeds),
            'localization_data': localization_data
        }


def plot_localization(analysis, save_path=None):
    """绘制模态局部化分析结果"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    freqs, modes = analysis.modal_analysis()
    loc_factors, amp_dists = analysis.analyze_localization()
    
    N = analysis.N_blades
    n_modes_plot = min(10, len(freqs))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    lf_values = [lf['localization_factor'] for lf in loc_factors[:n_modes_plot]]
    freq_values = [lf['frequency'] for lf in loc_factors[:n_modes_plot]]
    
    axes[0, 0].bar(range(n_modes_plot), lf_values, alpha=0.7)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='完全和谐')
    axes[0, 0].set_xlabel('模态阶次', fontsize=11)
    axes[0, 0].set_ylabel('局部化因子', fontsize=11)
    axes[0, 0].set_title(f'模态局部化因子 (失谐标准差={analysis.mistune_std*100:.0f}%)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    blades = np.arange(N)
    for i in range(min(4, n_modes_plot)):
        axes[0, 1].plot(blades, amp_dists[i], 'o-', label=f'模态{i+1}', linewidth=1.5, markersize=4)
    axes[0, 1].set_xlabel('叶片编号', fontsize=11)
    axes[0, 1].set_ylabel('归一化振幅', fontsize=11)
    axes[0, 1].set_title('典型振型幅值分布', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)
    
    axes[1, 0].plot(freqs[:n_modes_plot], lf_values, 'bo-', markersize=6)
    axes[1, 0].set_xlabel('固有频率 (Hz)', fontsize=11)
    axes[1, 0].set_ylabel('局部化因子', fontsize=11)
    axes[1, 0].set_title('局部化因子随频率变化', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    tuned_freqs_h = analysis.omega_h_nom / (2 * np.pi) * analysis.mistune_h
    tuned_freqs_a = analysis.omega_alpha_nom / (2 * np.pi) * analysis.mistune_alpha
    
    axes[1, 1].hist(tuned_freqs_h, alpha=0.6, label='弯曲频率', bins=10)
    axes[1, 1].hist(tuned_freqs_a, alpha=0.6, label='扭转频率', bins=10)
    axes[1, 1].set_xlabel('频率 (Hz)', fontsize=11)
    axes[1, 1].set_ylabel('叶片数', fontsize=11)
    axes[1, 1].set_title('失谐频率分布', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"模态局部化分析图已保存至: {save_path}")
    
    plt.show()


def plot_flutter_results(results, save_path=None):
    """绘制颤振分析结果"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    U_values = [r['U'] for r in results]
    max_dampings = [r['max_damping'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(U_values, max_dampings, 'b-', linewidth=2, marker='o', markersize=5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='零阻尼线')
    ax.set_xlabel('来流速度 U (m/s)', fontsize=12)
    ax.set_ylabel('最大阻尼比', fontsize=12)
    ax.set_title('失谐叶盘颤振边界', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"颤振分析结果图已保存至: {save_path}")
    
    plt.show()


def plot_monte_carlo_results(mc_results, save_path=None):
    """绘制蒙特卡洛模拟结果"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(mc_results['flutter_speeds'], bins=15, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=mc_results['mean_flutter'], color='r', linestyle='--', 
                      linewidth=2, label=f'均值={mc_results["mean_flutter"]:.1f} m/s')
    axes[0, 0].set_xlabel('颤振速度 (m/s)', fontsize=11)
    axes[0, 0].set_ylabel('频数', fontsize=11)
    axes[0, 0].set_title('颤振速度概率分布', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    flutter_speeds = mc_results['flutter_speeds']
    sorted_speeds = np.sort(flutter_speeds)
    cdf = np.arange(1, len(sorted_speeds) + 1) / len(sorted_speeds)
    
    axes[0, 1].plot(sorted_speeds, cdf, 'b-', linewidth=2)
    axes[0, 1].axvline(x=mc_results['min_flutter'], color='r', linestyle='--', 
                      label=f'最小值={mc_results["min_flutter"]:.1f}')
    axes[0, 1].axhline(y=0.05, color='g', linestyle='--', label='5%分位')
    axes[0, 1].set_xlabel('颤振速度 (m/s)', fontsize=11)
    axes[0, 1].set_ylabel('累积概率', fontsize=11)
    axes[0, 1].set_title('颤振速度累积分布函数', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    loc_data = mc_results['localization_data']
    avg_locs = [d['avg_localization'] for d in loc_data]
    flut_speeds = [d['flutter_speed'] for d in loc_data]
    
    axes[1, 0].scatter(avg_locs, flut_speeds, alpha=0.6, s=30)
    z = np.polyfit(avg_locs, flut_speeds, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(sorted(avg_locs), p(sorted(avg_locs)), 'r--', linewidth=2)
    axes[1, 0].set_xlabel('平均局部化因子', fontsize=11)
    axes[1, 0].set_ylabel('颤振速度 (m/s)', fontsize=11)
    axes[1, 0].set_title('局部化与颤振速度相关性', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    max_locs = [d['max_localization'] for d in loc_data]
    axes[1, 1].scatter(max_locs, flut_speeds, alpha=0.6, s=30)
    z = np.polyfit(max_locs, flut_speeds, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(sorted(max_locs), p(sorted(max_locs)), 'r--', linewidth=2)
    axes[1, 1].set_xlabel('最大局部化因子', fontsize=11)
    axes[1, 1].set_ylabel('颤振速度 (m/s)', fontsize=11)
    axes[1, 1].set_title('最大局部化与颤振速度相关性', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"蒙特卡洛分析结果图已保存至: {save_path}")
    
    plt.show()


def main():
    print("\n" + "=" * 75)
    print("                失谐叶盘颤振与模态局部化分析")
    print("  (含干摩擦阻尼 + 叶盘耦合 + 频率失谐 + 蒙特卡洛风险评估)")
    print("=" * 75 + "\n")
    
    friction = DryFrictionDamping(
        friction_coeff=0.3,
        normal_load=50.0,
        contact_stiffness=5e5
    )
    
    print("1. 干摩擦阻尼参数:")
    print(f"   摩擦系数 μ = {friction.mu}")
    print(f"   法向载荷 N = {friction.N} N")
    print(f"   接触刚度 k = {friction.k_contact/1e6:.2f} MN/m")
    
    analysis = MistunedBliskAnalysis(
        N_blades=24,
        rho=1.225,
        b=0.15,
        m_per_blade=8.0,
        I_per_blade=0.08,
        omega_h_nom=2 * np.pi * 8,
        omega_alpha_nom=2 * np.pi * 25,
        zeta_struct=0.005,
        a=-0.3,
        k_disk=1e7,
        k_coupling=5e5,
        mistune_std=0.03,
        friction_damping=friction,
        num_aero_states=4
    )
    
    print(f"\n2. 叶盘系统参数:")
    print(f"   叶片数 N = {analysis.N_blades}")
    print(f"   名义弯曲频率 = {analysis.omega_h_nom/(2*np.pi):.1f} Hz")
    print(f"   名义扭转频率 = {analysis.omega_alpha_nom/(2*np.pi):.1f} Hz")
    print(f"   失谐标准差 = {analysis.mistune_std*100:.1f}%")
    print(f"   圆盘刚度 = {analysis.k_disk/1e6:.1f} MN/m")
    print(f"   叶间耦合刚度 = {analysis.k_coupling/1e3:.1f} kN/m")
    
    print("\n3. 模态局部化分析...")
    freqs, modes = analysis.modal_analysis()
    print(f"   计算得到 {len(freqs)} 阶固有频率")
    print(f"   前5阶频率: {[f'{f:.2f}' for f in freqs[:5]]} Hz")
    
    if MATPLOTLIB_AVAILABLE:
        plot_localization(analysis, save_path='modal_localization.png')
    
    print("\n4. 颤振边界分析...")
    U_range = np.linspace(50, 200, 30)
    flutter_results = analysis.flutter_analysis(U_range)
    
    flutter_speed = None
    for i, res in enumerate(flutter_results):
        if res['max_damping'] > 0:
            if i > 0:
                U0 = flutter_results[i-1]['U']
                U1 = res['U']
                d0 = flutter_results[i-1]['max_damping']
                d1 = res['max_damping']
                flutter_speed = U0 - d0 * (U1 - U0) / (d1 - d0)
            else:
                flutter_speed = U_range[0]
            break
    
    if flutter_speed:
        print(f"   预测颤振速度 = {flutter_speed:.1f} m/s")
        print(f"   对应动压 = {0.5 * analysis.rho * flutter_speed**2:.1f} Pa")
    else:
        print("   在分析范围内未发现颤振")
    
    if MATPLOTLIB_AVAILABLE:
        plot_flutter_results(flutter_results, save_path='mistuned_flutter.png')
    
    print("\n5. 蒙特卡洛模拟 (n=30)...")
    mc_results = analysis.monte_carlo_flutter(n_samples=30, U_range=U_range)
    
    print(f"\n   颤振速度统计:")
    print(f"     均值 = {mc_results['mean_flutter']:.1f} m/s")
    print(f"     标准差 = {mc_results['std_flutter']:.1f} m/s")
    print(f"     最小值 = {mc_results['min_flutter']:.1f} m/s")
    print(f"     最大值 = {mc_results['max_flutter']:.1f} m/s")
    print(f"     5%分位降速 = {(1 - mc_results['min_flutter']/mc_results['mean_flutter'])*100:.1f}%")
    
    if MATPLOTLIB_AVAILABLE:
        plot_monte_carlo_results(mc_results, save_path='monte_carlo_results.png')
    
    print("\n" + "=" * 75)
    print("分析完成！")
    print("=" * 75)


if __name__ == "__main__":
    main()
