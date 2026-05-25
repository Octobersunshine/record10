import numpy as np
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，绘图功能将不可用")


class TheodorsenFunction:
    """Theodorsen函数，用于计算非定常气动力
    
    Theodorsen函数 C(k) = F(k) + iG(k)，其中k是约化频率
    """
    
    def __init__(self):
        self.k_values = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                                  0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
        
        self.F_values = np.array([1.0000, 0.9988, 0.9951, 0.9809, 0.9582, 0.9287, 0.8935, 
                                  0.8538, 0.8112, 0.7665, 0.7204, 0.6735, 0.4580, 0.3060, 
                                  0.2170, 0.1620, 0.1000, 0.0680, 0.0490, 0.0290, 0.0190])
        
        self.G_values = np.array([0.0000, 0.0143, 0.0278, 0.0529, 0.0752, 0.0946, 0.1109, 
                                  0.1242, 0.1347, 0.1426, 0.1481, 0.1516, 0.1543, 0.1452, 
                                  0.1305, 0.1159, 0.0908, 0.0728, 0.0600, 0.0425, 0.0320])
        
        self.F_interp = interpolate.interp1d(self.k_values, self.F_values, kind='cubic', 
                                             bounds_error=False, fill_value="extrapolate")
        self.G_interp = interpolate.interp1d(self.k_values, self.G_values, kind='cubic',
                                             bounds_error=False, fill_value="extrapolate")
    
    def __call__(self, k):
        """计算Theodorsen函数值 C(k) = F(k) + iG(k)"""
        k = np.clip(k, 0.0, 100.0)
        if k == 0:
            return 1.0 + 0.0j
        F = self.F_interp(k)
        G = self.G_interp(k)
        return float(F) + 1j * float(G)


class BladeFlutterAnalysis:
    """叶片气动弹性颤振分析（两自由度：弯曲+扭转）
    
    使用Theodorsen非定常气动力理论和p-k法求解颤振边界
    """
    
    def __init__(self, **kwargs):
        """初始化颤振分析参数
        
        参数:
            rho: 空气密度 (kg/m³)
            b: 半弦长 (m)
            m: 单位长度质量 (kg/m)
            I_alpha: 单位长度转动惯量 (kg·m²/m)
            x_alpha: 重心位置相对于弹性轴 (无量纲，以半弦长为单位)
            omega_h: 弯曲固有角频率 (rad/s)
            omega_alpha: 扭转固有角频率 (rad/s)
            zeta_h: 弯曲模态阻尼比
            zeta_alpha: 扭转模态阻尼比
            a: 弹性轴位置相对于弦线中点 (无量纲，以半弦长为单位，前缘为-1，后缘为+1)
        """
        self.rho = kwargs.get('rho', 1.225)
        self.b = kwargs.get('b', 0.1)
        self.m = kwargs.get('m', 10.0)
        self.I_alpha = kwargs.get('I_alpha', 0.1)
        self.x_alpha = kwargs.get('x_alpha', 0.1)
        self.omega_h = kwargs.get('omega_h', 2 * np.pi * 5)
        self.omega_alpha = kwargs.get('omega_alpha', 2 * np.pi * 15)
        self.zeta_h = kwargs.get('zeta_h', 0.01)
        self.zeta_alpha = kwargs.get('zeta_alpha', 0.01)
        self.a = kwargs.get('a', -0.2)
        self.theodorsen = TheodorsenFunction()
    
    def _build_system_matrices(self, U, omega):
        """构建质量、阻尼、刚度矩阵"""
        k = omega * self.b / U
        Ck = self.theodorsen(k)
        
        b = self.b
        a = self.a
        rho = self.rho
        m = self.m
        I_alpha = self.I_alpha
        x_alpha = self.x_alpha
        
        Qhh = np.pi * rho * b**2
        Qha = np.pi * rho * b**3 * (0.5 - a)
        Qah = np.pi * rho * b**3 * (0.5 + a)
        Qaa = np.pi * rho * b**4 * (1/8 + (0.5 + a)**2)
        
        L11_steady = 2 * np.pi * rho * U * b * 1j * k
        L12_steady = 2 * np.pi * rho * U * b**2 * (1 + 1j * k * (0.5 - a))
        M11_steady = 2 * np.pi * rho * U * b**2 * 1j * k * (0.5 + a)
        M12_steady = 2 * np.pi * rho * U * b**3 * ((0.5 + a) + 1j * k * (1/8 + (0.5 + a)**2))
        
        L11_unsteady = 2 * np.pi * rho * U * b * (Ck - 1) * (1j * k + (0.5 - a) * k**2)
        L12_unsteady = 2 * np.pi * rho * U * b**2 * (Ck - 1) * (1j * k + (0.5 - a) * k**2) * (0.5 + a)
        M11_unsteady = 2 * np.pi * rho * U * b**2 * (Ck - 1) * (1j * k + (0.5 - a) * k**2) * (0.5 + a)
        M12_unsteady = 2 * np.pi * rho * U * b**3 * (Ck - 1) * (1j * k + (0.5 - a) * k**2) * (0.5 + a)**2
        
        L11 = L11_steady + L11_unsteady
        L12 = L12_steady + L12_unsteady
        M11 = M11_steady + M11_unsteady
        M12 = M12_steady + M12_unsteady
        
        M_matrix = np.array([[m + Qhh, m * x_alpha * b + Qha],
                            [m * x_alpha * b + Qah, I_alpha + m * x_alpha**2 * b**2 + Qaa]], dtype=complex)
        
        D_matrix = np.array([[2 * m * self.zeta_h * self.omega_h + L11.imag / omega, L12.imag / omega],
                            [M11.imag / omega, 2 * I_alpha * self.zeta_alpha * self.omega_alpha + M12.imag / omega]], dtype=complex)
        
        K_matrix = np.array([[m * self.omega_h**2 + L11.real, L12.real],
                            [M11.real, I_alpha * self.omega_alpha**2 + M12.real]], dtype=complex)
        
        return M_matrix, D_matrix, K_matrix
    
    def state_space_matrix(self, U, omega):
        """构建状态空间矩阵 A = [[0, I], [-M^{-1}K, -M^{-1}D]]"""
        M, D, K = self._build_system_matrices(U, omega)
        
        A = np.zeros((4, 4), dtype=complex)
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 0:2] = -np.linalg.inv(M) @ K
        A[2:4, 2:4] = -np.linalg.inv(M) @ D
        
        return A
    
    def compute_eigenvalues(self, U, omega):
        """计算给定速度和频率下的系统特征值"""
        A = self.state_space_matrix(U, omega)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors
    
    def pk_method(self, U_range):
        """使用p-k法求解颤振边界
        
        参数:
            U_range: 速度范围数组
            
        返回:
            results: 包含每个速度下阻尼和频率的结果列表
        """
        results = []
        omega_min = min(self.omega_h, self.omega_alpha) * 0.3
        omega_max = max(self.omega_h, self.omega_alpha) * 3.0
        
        for U in U_range:
            omega_guess = np.linspace(omega_min, omega_max, 100)
            
            best_damping = -np.inf
            best_omega = 0
            best_eig = None
            
            for omega in omega_guess:
                try:
                    eigenvalues, _ = self.compute_eigenvalues(U, omega)
                    
                    for eig in eigenvalues:
                        damping = eig.real
                        freq = abs(eig.imag)
                        
                        if freq > 0 and damping > best_damping:
                            best_damping = damping
                            best_omega = freq
                            best_eig = eig
                except:
                    continue
            
            results.append({
                'U': U,
                'damping': best_damping,
                'frequency': best_omega / (2 * np.pi),
                'eigenvalue': best_eig
            })
        
        return results
    
    def find_flutter_speed(self, U_start=1.0, U_end=200.0, tol=1e-3):
        """使用二分法寻找颤振速度（阻尼由负变正的点）
        
        参数:
            U_start: 搜索起始速度
            U_end: 搜索结束速度
            tol: 收敛容差
            
        返回:
            flutter_speed: 颤振速度 (m/s)
        """
        omega_min = min(self.omega_h, self.omega_alpha) * 0.3
        omega_max = max(self.omega_h, self.omega_alpha) * 3.0
        
        def get_max_damping(U):
            """获取给定速度下的最大阻尼系数"""
            omega_guess = np.linspace(omega_min, omega_max, 150)
            best_damping = -np.inf
            
            for omega in omega_guess:
                try:
                    eigenvalues, _ = self.compute_eigenvalues(U, omega)
                    for eig in eigenvalues:
                        if eig.imag > 1e-6 and eig.real > best_damping:
                            best_damping = eig.real
                except:
                    continue
            
            return best_damping
        
        damping_start = get_max_damping(U_start)
        damping_end = get_max_damping(U_end)
        
        if damping_start > 0:
            print(f"警告: 在U={U_start}m/s时阻尼已为正 ({damping_start:.4f})")
            return U_start
        if damping_end < 0:
            print(f"警告: 在U={U_end}m/s时阻尼仍为负 ({damping_end:.4f})")
            return None
        
        while (U_end - U_start) > tol:
            U_mid = (U_start + U_end) / 2
            damping_mid = get_max_damping(U_mid)
            
            if damping_mid < 0:
                U_start = U_mid
            else:
                U_end = U_mid
        
        return (U_start + U_end) / 2
    
    def print_parameters(self):
        """打印输入参数"""
        print("=" * 60)
        print("叶片气动弹性颤振分析 - 输入参数")
        print("=" * 60)
        print(f"  空气密度        rho      = {self.rho:.3f} kg/m³")
        print(f"  半弦长          b        = {self.b:.4f} m")
        print(f"  单位长度质量    m        = {self.m:.3f} kg/m")
        print(f"  单位长度转动惯量 I_alpha  = {self.I_alpha:.4f} kg·m²/m")
        print(f"  重心位置        x_alpha  = {self.x_alpha:.3f} (半弦长)")
        print(f"  弯曲频率        f_h      = {self.omega_h/(2*np.pi):.2f} Hz")
        print(f"  扭转频率        f_alpha  = {self.omega_alpha/(2*np.pi):.2f} Hz")
        print(f"  频率比          omega_ratio = {self.omega_alpha/self.omega_h:.2f}")
        print(f"  弯曲阻尼比      zeta_h   = {self.zeta_h:.4f}")
        print(f"  扭转阻尼比      zeta_alpha = {self.zeta_alpha:.4f}")
        print(f"  弹性轴位置      a        = {self.a:.3f} (半弦长)")
        print("=" * 60)


def plot_flutter_results(results, flutter_speed=None, save_path=None):
    """绘制颤振分析结果（V-g图和V-f图）
    
    参数:
        results: p-k法分析结果
        flutter_speed: 颤振速度
        save_path: 图片保存路径
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib未安装，无法绘图")
        return
    
    U_values = [r['U'] for r in results]
    damping_values = [r['damping'] for r in results]
    freq_values = [r['frequency'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(U_values, damping_values, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='零阻尼线')
    if flutter_speed:
        ax1.axvline(x=flutter_speed, color='g', linestyle='--', linewidth=1.5,
                   label=f'颤振速度 = {flutter_speed:.1f} m/s')
    ax1.set_xlabel('来流速度 U (m/s)', fontsize=12)
    ax1.set_ylabel('阻尼系数 g', fontsize=12)
    ax1.set_title('V-g 图：阻尼随速度变化', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.plot(U_values, freq_values, 'b-', linewidth=2, marker='o', markersize=4)
    if flutter_speed:
        ax2.axvline(x=flutter_speed, color='g', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('来流速度 U (m/s)', fontsize=12)
    ax2.set_ylabel('振动频率 f (Hz)', fontsize=12)
    ax2.set_title('V-f 图：频率随速度变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果图已保存至: {save_path}")
    
    plt.show()


def main():
    """主函数：执行颤振分析示例"""
    print("\n" + "=" * 60)
    print("             叶片气动弹性颤振分析程序")
    print("=" * 60 + "\n")
    
    analysis = BladeFlutterAnalysis(
        rho=1.225,
        b=0.15,
        m=8.0,
        I_alpha=0.08,
        x_alpha=0.05,
        omega_h=2 * np.pi * 8,
        omega_alpha=2 * np.pi * 25,
        zeta_h=0.02,
        zeta_alpha=0.02,
        a=-0.3
    )
    
    analysis.print_parameters()
    
    print("\n正在计算颤振速度...")
    flutter_speed = analysis.find_flutter_speed(U_start=10.0, U_end=200.0)
    
    if flutter_speed:
        print("\n" + "=" * 60)
        print("颤振分析结果")
        print("=" * 60)
        print(f"  颤振速度   V_f = {flutter_speed:.2f} m/s")
        print(f"  颤振动压   q_f = {0.5 * analysis.rho * flutter_speed**2:.2f} Pa")
        print(f"  颤振马赫数 Ma_f = {flutter_speed / 340:.3f}")
        print("=" * 60)
    else:
        print("\n在指定速度范围内未找到颤振点")
    
    print("\n正在进行p-k法分析（绘制V-g/V-f图）...")
    U_range = np.linspace(10, 150, 60)
    results = analysis.pk_method(U_range)
    
    print("\n部分计算结果:")
    print("-" * 60)
    print(f"{'速度 (m/s)':>12} {'阻尼系数':>12} {'频率 (Hz)':>12}")
    print("-" * 60)
    for i in range(0, len(results), 10):
        r = results[i]
        print(f"{r['U']:>12.2f} {r['damping']:>12.4f} {r['frequency']:>12.2f}")
    print("-" * 60)
    
    if MATPLOTLIB_AVAILABLE:
        plot_flutter_results(results, flutter_speed, save_path='flutter_analysis_results.png')
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
