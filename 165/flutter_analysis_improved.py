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


class RogerApproximation:
    """Roger有理函数近似 - 用于Theodorsen函数的高精度逼近
    
    C(k) = 1 - A0/(1 + i*k*b1) - A1/(1 + i*k*b2) - ...
    
    其中k = ωb/U 是约化频率
    """
    
    def __init__(self, order=4):
        """
        参数:
            order: Roger近似阶数 (2, 4, 或 6)
        """
        self.order = order
        
        if order == 2:
            self.A = np.array([0.5000, 0.5000])
            self.b = np.array([0.0455, 0.3000])
            self.A0 = 1.0 - np.sum(self.A)
        elif order == 4:
            self.A = np.array([0.1650, 0.3350, 0.3350, 0.1650])
            self.b = np.array([0.0120, 0.0600, 0.2000, 0.6000])
            self.A0 = 1.0 - np.sum(self.A)
        elif order == 6:
            self.A = np.array([0.0880, 0.1760, 0.2360, 0.2360, 0.1760, 0.0880])
            self.b = np.array([0.0050, 0.0200, 0.0700, 0.2000, 0.5000, 1.2000])
            self.A0 = 1.0 - np.sum(self.A)
        else:
            raise ValueError("Roger近似阶数只支持 2, 4, 6")
    
    def __call__(self, k):
        """计算Theodorsen函数值 C(k)"""
        if k == 0:
            return 1.0 + 0.0j
        
        s = 1j * k
        C = 1.0
        
        for i in range(self.order):
            C -= self.A[i] * s / (1.0 + self.b[i] * s)
        
        return C
    
    def get_imaginary(self, k):
        """获取Theodorsen函数虚部 G(k)"""
        return self(k).imag
    
    def get_real(self, k):
        """获取Theodorsen函数实部 F(k)"""
        return self(k).real
    
    def get_state_space_coeffs(self):
        """获取状态空间表示的系数"""
        return self.A, self.b, self.A0


class LeishmanBeddoesStateSpace:
    """Leishman-Beddoes状态空间非定常气动力模型
    
    基于最小状态法（Minimum State Method）构建状态空间气动力模型
    避免频域插值导致的数值发散问题
    """
    
    def __init__(self, num_states=4):
        """
        参数:
            num_states: 气动力状态数
        """
        self.num_states = num_states
        self.roger = RogerApproximation(order=num_states)
        self.A, self.b, self.A0 = self.roger.get_state_space_coeffs()
    
    def build_aero_state_matrix(self, b, U):
        """构建气动力状态矩阵
        
        参数:
            b: 半弦长
            U: 来流速度
            
        返回:
            A_aero: 气动力状态矩阵
            B_aero: 气动力输入矩阵
        """
        n = self.num_states
        
        A_aero = np.zeros((n, n))
        B_aero = np.zeros((n, 4))
        
        for i in range(n):
            tau = self.b[i] * b / U
            A_aero[i, i] = -1.0 / tau
            B_aero[i, 0] = -self.A[i] / tau
            B_aero[i, 1] = -self.A[i] * b / tau
            B_aero[i, 2] = -self.A[i] / (U * tau)
            B_aero[i, 3] = -self.A[i] * b / (U * tau)
        
        return A_aero, B_aero
    
    def compute_aero_forces(self, b, rho, U, states, h_dot, alpha_dot, h_2dot, alpha_2dot, alpha):
        """计算非定常气动力（升力和力矩）
        
        参数:
            b: 半弦长
            rho: 空气密度
            U: 来流速度
            states: 气动力状态变量
            h_dot: 弯曲速度
            alpha_dot: 扭转角速度
            h_2dot: 弯曲加速度
            alpha_2dot: 扭转角加速度
            alpha: 扭转角
            
        返回:
            L: 升力
            M: 俯仰力矩
        """
        Qhh = np.pi * rho * b**2
        Qha = np.pi * rho * b**3 * 0.5
        Qah = Qha
        Qaa = np.pi * rho * b**4 * (1/8 + 0.25)
        
        L_circ = 2 * np.pi * rho * U * b * (h_dot + U * alpha + 0.5 * b * alpha_dot)
        M_circ = 2 * np.pi * rho * U * b**2 * 0.5 * (h_dot + U * alpha + 0.5 * b * alpha_dot)
        
        L_added = Qhh * h_2dot + Qha * alpha_2dot
        M_added = Qah * h_2dot + Qaa * alpha_2dot
        
        L_wake = 0.0
        M_wake = 0.0
        for i in range(self.num_states):
            L_wake += 2 * np.pi * rho * U * b * states[i]
            M_wake += 2 * np.pi * rho * U * b**2 * 0.5 * states[i]
        
        L = L_circ + L_added + L_wake
        M = M_circ + M_added + M_wake
        
        return L, M


class StateSpaceFlutterAnalysis:
    """状态空间颤振分析 - 使用最小状态法
    
    优势:
    1. 避免频域插值误差
    2. 数值稳定性更好
    3. 可直接用于时域仿真
    """
    
    def __init__(self, **kwargs):
        """
        参数:
            rho: 空气密度 (kg/m³)
            b: 半弦长 (m)
            m: 单位长度质量 (kg/m)
            I_alpha: 单位长度转动惯量 (kg·m²/m)
            x_alpha: 重心位置相对于弹性轴
            omega_h: 弯曲固有角频率 (rad/s)
            omega_alpha: 扭转固有角频率 (rad/s)
            zeta_h: 弯曲模态阻尼比
            zeta_alpha: 扭转模态阻尼比
            a: 弹性轴位置
            num_aero_states: 气动力状态数
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
        self.num_aero_states = kwargs.get('num_aero_states', 4)
        
        self.aero_model = LeishmanBeddoesStateSpace(num_states=self.num_aero_states)
        
        self.k_h = self.m * self.omega_h**2
        self.k_alpha = self.I_alpha * self.omega_alpha**2
    
    def build_state_space_matrix(self, U):
        """构建完整的状态空间矩阵 A
        
        状态向量: [h, α, ḣ, α̇, x1, x2, ..., xn]^T
        """
        b = self.b
        a = self.a
        rho = self.rho
        m = self.m
        I_alpha = self.I_alpha
        x_alpha = self.x_alpha
        n_states = self.num_aero_states
        
        Qhh = np.pi * rho * b**2
        Qha = np.pi * rho * b**3 * (0.5 - a)
        Qah = np.pi * rho * b**3 * (0.5 + a)
        Qaa = np.pi * rho * b**4 * (1/8 + (0.5 + a)**2)
        
        M_struct = np.array([[m, m * x_alpha * b],
                             [m * x_alpha * b, I_alpha + m * x_alpha**2 * b**2]])
        
        M_aero = np.array([[Qhh, Qha],
                           [Qah, Qaa]])
        
        M_total = M_struct + M_aero
        
        D_struct = np.array([[2 * m * self.zeta_h * self.omega_h, 0],
                             [0, 2 * I_alpha * self.zeta_alpha * self.omega_alpha]])
        
        D_aero = np.array([[2 * np.pi * rho * b * U, 2 * np.pi * rho * b**2 * U * (0.5 - a)],
                           [2 * np.pi * rho * b**2 * U * (0.5 + a), 2 * np.pi * rho * b**3 * U * (0.5 + a) * (0.5 - a)]])
        
        D_total = D_struct + D_aero
        
        K_struct = np.array([[self.k_h, 0],
                             [0, self.k_alpha]])
        
        K_aero = np.array([[0, 2 * np.pi * rho * b * U**2],
                           [0, 2 * np.pi * rho * b**2 * U**2 * (0.5 + a)]])
        
        K_total = K_struct + K_aero
        
        n_total = 4 + n_states
        A = np.zeros((n_total, n_total))
        
        A[0:2, 2:4] = np.eye(2)
        
        M_inv = np.linalg.inv(M_total)
        
        A[2:4, 0:2] = -M_inv @ K_total
        A[2:4, 2:4] = -M_inv @ D_total
        
        for i in range(n_states):
            tau = self.aero_model.b[i] * b / U
            A[4 + i, 4 + i] = -1.0 / tau
            A[4 + i, 0:2] = np.array([-self.aero_model.A[i] / (U * tau), 
                                       -self.aero_model.A[i] * U / tau])
            A[4 + i, 2:4] = np.array([-self.aero_model.A[i] / tau, 
                                       -self.aero_model.A[i] * b / tau])
        
        C_wake = np.zeros((2, n_states))
        for i in range(n_states):
            C_wake[0, i] = -2 * np.pi * rho * U * b * self.aero_model.A[i]
            C_wake[1, i] = -2 * np.pi * rho * U * b**2 * (0.5 + a) * self.aero_model.A[i]
        
        A[2:4, 4:4+n_states] = M_inv @ C_wake
        
        return A
    
    def compute_eigenvalues(self, U):
        """计算给定速度下的系统特征值"""
        A = self.build_state_space_matrix(U)
        eigenvalues = linalg.eigvals(A)
        return eigenvalues
    
    def pk_method(self, U_range):
        """改进的p-k法 - 使用状态空间模型"""
        results = []
        
        for U in U_range:
            try:
                eigenvalues = self.compute_eigenvalues(U)
                
                positive_freq_indices = eigenvalues.imag > 1e-6
                if np.any(positive_freq_indices):
                    positive_eigs = eigenvalues[positive_freq_indices]
                    max_damping_idx = np.argmax(positive_eigs.real)
                    best_eig = positive_eigs[max_damping_idx]
                    best_damping = best_eig.real
                    best_freq = best_eig.imag / (2 * np.pi)
                else:
                    best_damping = -np.inf
                    best_freq = 0
                    best_eig = None
                    
            except Exception as e:
                best_damping = -np.inf
                best_freq = 0
                best_eig = None
            
            results.append({
                'U': U,
                'damping': best_damping,
                'frequency': best_freq,
                'eigenvalue': best_eig
            })
        
        return results
    
    def find_flutter_speed(self, U_start=1.0, U_end=200.0, tol=1e-3):
        """二分法求解颤振速度 - 使用状态空间模型"""
        
        def get_max_damping(U):
            try:
                eigenvalues = self.compute_eigenvalues(U)
                positive_freq_eigs = eigenvalues[eigenvalues.imag > 1e-6]
                if len(positive_freq_eigs) > 0:
                    return np.max(positive_freq_eigs.real)
                else:
                    return -np.inf
            except:
                return -np.inf
        
        damping_start = get_max_damping(U_start)
        damping_end = get_max_damping(U_end)
        
        if damping_start > 0:
            print(f"警告: 在U={U_start}m/s时阻尼已为正 ({damping_start:.4f})")
            return U_start
        if damping_end < 0:
            print(f"警告: 在U={U_end}m/s时阻尼仍为负 ({damping_end:.4f})")
            return None
        
        iterations = 0
        max_iter = 100
        while (U_end - U_start) > tol and iterations < max_iter:
            U_mid = (U_start + U_end) / 2
            damping_mid = get_max_damping(U_mid)
            
            if damping_mid < 0:
                U_start = U_mid
            else:
                U_end = U_mid
            iterations += 1
        
        return (U_start + U_end) / 2
    
    def time_domain_simulation(self, U, t_end=5.0, dt=0.001):
        """时域仿真验证颤振特性
        
        参数:
            U: 来流速度
            t_end: 仿真结束时间
            dt: 时间步长
            
        返回:
            t: 时间数组
            states: 状态变量时间历程
        """
        A = self.build_state_space_matrix(U)
        n_states = A.shape[0]
        
        t = np.arange(0, t_end, dt)
        n_steps = len(t)
        
        states = np.zeros((n_steps, n_states))
        states[0, 0] = 0.01
        states[0, 1] = 0.001
        
        for i in range(1, n_steps):
            k1 = A @ states[i-1, :]
            k2 = A @ (states[i-1, :] + dt/2 * k1)
            k3 = A @ (states[i-1, :] + dt/2 * k2)
            k4 = A @ (states[i-1, :] + dt * k3)
            states[i, :] = states[i-1, :] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return t, states
    
    def print_parameters(self):
        """打印输入参数"""
        print("=" * 70)
        print("状态空间叶片气动弹性颤振分析 - 输入参数")
        print("=" * 70)
        print(f"  空气密度          rho      = {self.rho:.3f} kg/m³")
        print(f"  半弦长            b        = {self.b:.4f} m")
        print(f"  单位长度质量      m        = {self.m:.3f} kg/m")
        print(f"  单位长度转动惯量  I_alpha  = {self.I_alpha:.4f} kg·m²/m")
        print(f"  重心位置          x_alpha  = {self.x_alpha:.3f} (半弦长)")
        print(f"  弯曲频率          f_h      = {self.omega_h/(2*np.pi):.2f} Hz")
        print(f"  扭转频率          f_alpha  = {self.omega_alpha/(2*np.pi):.2f} Hz")
        print(f"  频率比            omega_ratio = {self.omega_alpha/self.omega_h:.2f}")
        print(f"  弯曲阻尼比        zeta_h   = {self.zeta_h:.4f}")
        print(f"  扭转阻尼比        zeta_alpha = {self.zeta_alpha:.4f}")
        print(f"  弹性轴位置        a        = {self.a:.3f} (半弦长)")
        print(f"  气动力状态数      n_states = {self.num_aero_states}")
        print(f"  气动力模型        Roger({self.num_aero_states}阶有理函数近似)")
        print("=" * 70)


def compare_theodorsen_models(k_values=np.linspace(0, 5, 100)):
    """比较不同Theodorsen函数近似方法"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    k_exact = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                        0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    F_exact = np.array([1.0000, 0.9988, 0.9951, 0.9809, 0.9582, 0.9287, 0.8935, 
                        0.8538, 0.8112, 0.7665, 0.7204, 0.6735, 0.4580, 0.3060, 
                        0.2170, 0.1620, 0.1000, 0.0680])
    G_exact = np.array([0.0000, 0.0143, 0.0278, 0.0529, 0.0752, 0.0946, 0.1109, 
                        0.1242, 0.1347, 0.1426, 0.1481, 0.1516, 0.1543, 0.1452, 
                        0.1305, 0.1159, 0.0908, 0.0728])
    
    from scipy import interpolate
    F_interp = interpolate.interp1d(k_exact, F_exact, kind='cubic')(k_values)
    G_interp = interpolate.interp1d(k_exact, G_exact, kind='cubic')(k_values)
    
    roger2 = RogerApproximation(order=2)
    roger4 = RogerApproximation(order=4)
    roger6 = RogerApproximation(order=6)
    
    F_roger2 = np.array([roger2.get_real(k) for k in k_values])
    G_roger2 = np.array([roger2.get_imaginary(k) for k in k_values])
    F_roger4 = np.array([roger4.get_real(k) for k in k_values])
    G_roger4 = np.array([roger4.get_imaginary(k) for k in k_values])
    F_roger6 = np.array([roger6.get_real(k) for k in k_values])
    G_roger6 = np.array([roger6.get_imaginary(k) for k in k_values])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(k_exact, F_exact, 'ko', label='精确值', markersize=6)
    ax1.plot(k_values, F_interp, 'b--', label='三次插值', linewidth=2)
    ax1.plot(k_values, F_roger2, 'g-', label='Roger 2阶', linewidth=1.5)
    ax1.plot(k_values, F_roger4, 'r-', label='Roger 4阶', linewidth=1.5)
    ax1.plot(k_values, F_roger6, 'm-', label='Roger 6阶', linewidth=1.5)
    ax1.set_xlabel('约化频率 k', fontsize=12)
    ax1.set_ylabel('F(k)', fontsize=12)
    ax1.set_title('Theodorsen函数实部 F(k) 比较', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.plot(k_exact, G_exact, 'ko', label='精确值', markersize=6)
    ax2.plot(k_values, G_interp, 'b--', label='三次插值', linewidth=2)
    ax2.plot(k_values, G_roger2, 'g-', label='Roger 2阶', linewidth=1.5)
    ax2.plot(k_values, G_roger4, 'r-', label='Roger 4阶', linewidth=1.5)
    ax2.plot(k_values, G_roger6, 'm-', label='Roger 6阶', linewidth=1.5)
    ax2.set_xlabel('约化频率 k', fontsize=12)
    ax2.set_ylabel('G(k)', fontsize=12)
    ax2.set_title('Theodorsen函数虚部 G(k) 比较', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('theodorsen_comparison.png', dpi=150, bbox_inches='tight')
    print("Theodorsen函数比较图已保存至: theodorsen_comparison.png")
    plt.show()


def plot_flutter_results(results, flutter_speed=None, save_path=None):
    """绘制颤振分析结果"""
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
    ax1.set_title('V-g 图：阻尼随速度变化 (状态空间法)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.plot(U_values, freq_values, 'b-', linewidth=2, marker='o', markersize=4)
    if flutter_speed:
        ax2.axvline(x=flutter_speed, color='g', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('来流速度 U (m/s)', fontsize=12)
    ax2.set_ylabel('振动频率 f (Hz)', fontsize=12)
    ax2.set_title('V-f 图：频率随速度变化 (状态空间法)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果图已保存至: {save_path}")
    
    plt.show()


def plot_time_response(t, states, U, flutter_speed, save_path=None):
    """绘制时域响应"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(t, states[:, 0], 'b-', linewidth=1.5)
    ax1.set_xlabel('时间 t (s)', fontsize=12)
    ax1.set_ylabel('弯距位移 h (m)', fontsize=12)
    ax1.set_title(f'弯曲位移时域响应 (U={U:.1f} m/s)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, np.degrees(states[:, 1]), 'r-', linewidth=1.5)
    ax2.set_xlabel('时间 t (s)', fontsize=12)
    ax2.set_ylabel('扭转角 α (deg)', fontsize=12)
    ax2.set_title(f'扭转角时域响应 (U={U:.1f} m/s)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"时域响应图已保存至: {save_path}")
    
    plt.show()


def main():
    print("\n" + "=" * 70)
    print("         状态空间叶片气动弹性颤振分析程序")
    print("  (使用Roger有理函数近似 - 消除插值误差和数值发散)")
    print("=" * 70 + "\n")
    
    print("1. 比较不同Theodorsen函数近似方法...")
    compare_theodorsen_models()
    print()
    
    analysis = StateSpaceFlutterAnalysis(
        rho=1.225,
        b=0.15,
        m=8.0,
        I_alpha=0.08,
        x_alpha=0.05,
        omega_h=2 * np.pi * 8,
        omega_alpha=2 * np.pi * 25,
        zeta_h=0.02,
        zeta_alpha=0.02,
        a=-0.3,
        num_aero_states=4
    )
    
    analysis.print_parameters()
    
    print("\n2. 正在计算颤振速度...")
    flutter_speed = analysis.find_flutter_speed(U_start=10.0, U_end=200.0)
    
    if flutter_speed:
        print("\n" + "=" * 70)
        print("颤振分析结果 (状态空间法 / Roger 4阶近似)")
        print("=" * 70)
        print(f"  颤振速度   V_f = {flutter_speed:.2f} m/s")
        print(f"  颤振动压   q_f = {0.5 * analysis.rho * flutter_speed**2:.2f} Pa")
        print(f"  颤振马赫数 Ma_f = {flutter_speed / 340:.3f}")
        print("=" * 70)
    else:
        print("\n在指定速度范围内未找到颤振点")
    
    print("\n3. 正在进行颤振边界分析 (状态空间p-k法)...")
    U_range = np.linspace(10, 150, 60)
    results = analysis.pk_method(U_range)
    
    print("\n部分计算结果:")
    print("-" * 70)
    print(f"{'速度 (m/s)':>12} {'阻尼系数':>15} {'频率 (Hz)':>15}")
    print("-" * 70)
    for i in range(0, len(results), 10):
        r = results[i]
        print(f"{r['U']:>12.2f} {r['damping']:>15.6f} {r['frequency']:>15.3f}")
    print("-" * 70)
    
    if MATPLOTLIB_AVAILABLE:
        plot_flutter_results(results, flutter_speed, save_path='flutter_ss_results.png')
    
    if flutter_speed:
        print("\n4. 时域仿真验证...")
        print(f"   - 亚临界速度 (0.8*Vf): {0.8*flutter_speed:.1f} m/s")
        t1, states1 = analysis.time_domain_simulation(0.8 * flutter_speed, t_end=3.0)
        plot_time_response(t1, states1, 0.8*flutter_speed, flutter_speed, 
                          save_path='time_response_subcritical.png')
        
        print(f"   - 超临界速度 (1.2*Vf): {1.2*flutter_speed:.1f} m/s")
        t2, states2 = analysis.time_domain_simulation(1.2 * flutter_speed, t_end=2.0)
        plot_time_response(t2, states2, 1.2*flutter_speed, flutter_speed,
                          save_path='time_response_supercritical.png')
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
