import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_pml_profile(nt, t, t_max, pml_width, sigma_max=10.0, profile_type='quadratic'):
    """
    创建PML吸收层的sigma分布
    """
    sigma = np.zeros(nt)
    
    t_left = -t_max + pml_width
    t_right = t_max - pml_width
    
    left_mask = t < t_left
    right_mask = t > t_right
    
    if profile_type == 'quadratic':
        sigma[left_mask] = sigma_max * ((t_left - t[left_mask]) / pml_width)**2
        sigma[right_mask] = sigma_max * ((t[right_mask] - t_right) / pml_width)**2
    elif profile_type == 'cubic':
        sigma[left_mask] = sigma_max * ((t_left - t[left_mask]) / pml_width)**3
        sigma[right_mask] = sigma_max * ((t[right_mask] - t_right) / pml_width)**3
    elif profile_type == 'exponential':
        sigma[left_mask] = sigma_max * (1 - np.exp(-((t_left - t[left_mask]) / pml_width)**2))
        sigma[right_mask] = sigma_max * (1 - np.exp(-((t[right_mask] - t_right) / pml_width)**2))
    
    return sigma


def raman_response(t, tau1=12.2, tau2=32.0):
    """
    拉曼响应函数 (归一化)
    
    参数:
        t: 时间数组
        tau1, tau2: 拉曼响应时间参数 (单位: fs)
    
    返回:
        h_R: 拉曼响应函数
    """
    h_R = np.zeros_like(t)
    idx = t >= 0
    tau1_fs = tau1 * 1e-3
    tau2_fs = tau2 * 1e-3
    
    t_rel = t[idx]
    
    h_R[idx] = (tau1_fs**2 + tau2_fs**2) / (tau1_fs * tau2_fs**2) * \
                 np.exp(-t_rel / tau2_fs) * np.sin(t_rel / tau1_fs)
    
    h_R = h_R / np.trapz(h_R, t)
    
    return h_R


def generate_multisoliton(t, positions, amplitudes, phases=None, velocities=None):
    """
    生成多孤子初始条件
    
    参数:
        t: 时间数组
        positions: 各孤子中心位置列表
        amplitudes: 各孤子振幅列表
        phases: 各孤子初始相位列表 (可选)
        velocities: 各孤子速度列表 (可选)
    
    返回:
        u: 多孤子初始波场
    """
    if phases is None:
        phases = [0.0] * len(positions)
    if velocities is None:
        velocities = [0.0] * len(positions)
    
    u = np.zeros_like(t, dtype=complex)
    
    for pos, amp, phi, v in zip(positions, amplitudes, phases, velocities):
        u += amp / np.cosh(t - pos) * np.exp(1j * (phi + v * t))
    
    return u


def generate_bit_pattern(t, bit_sequence, bit_width=1.0, spacing=3.0, start_pos=0.0):
    """
    生成比特序列模式 (用于光纤通信)
    
    参数:
        t: 时间数组
        bit_sequence: 比特序列 (如 [1, 0, 1, 1])
        bit_width: 单个比特宽度
        spacing: 比特间距
        start_pos: 起始位置
    
    返回:
        u: 比特序列波场
    """
    u = np.zeros_like(t, dtype=complex)
    
    for i, bit in enumerate(bit_sequence):
        if bit == 1:
            pos = start_pos + i * spacing
            u += 1.0 / np.cosh((t - pos) / bit_width)
    
    return u


def ssfm_generalized_nlse(
    t, u0, dz, nz,
    beta2=-1.0, gamma=1.0,
    use_raman=False, raman_fraction=0.18,
    use_self_steepening=False, tau_shock=0.0,
    use_pml=False, pml_width=2.0, sigma_max=15.0
):
    """
    广义非线性薛定谔方程 (GNLSE) 求解器
    
    分步傅里叶法求解:
        ∂u/∂z = (iβ2/2) ∂²u/∂t² 
               - iγ (1 + iτ_shock ∂/∂t) [ (1-f_R)|u|²u + f_R u ∫ h_R(τ)|u(t-τ)|²dτ ]
    
    参数:
        t: 时间数组
        u0: 初始波场
        dz: 传播步长
        nz: 传播步数
        beta2: 二阶色散系数
        gamma: 非线性系数
        use_raman: 是否启用拉曼散射
        raman_fraction: 拉曼响应分数 (f_R ≈ 0.18)
        use_self_steepening: 是否启用自陡峭效应
        tau_shock: 自陡峭时间常数 (归一化单位)
        use_pml: 是否使用PML边界
        pml_width: PML层宽度
        sigma_max: PML最大吸收系数
    
    返回:
        z: 传播距离数组
        u: 波场演化矩阵
    """
    nt = len(t)
    dt = t[1] - t[0]
    t_max = np.max(np.abs(t))
    
    omega = 2 * np.pi * np.fft.fftfreq(nt, dt)
    
    if use_pml:
        sigma = create_pml_profile(nt, t, t_max, pml_width, sigma_max)
    else:
        sigma = np.zeros(nt)
    
    if use_raman:
        h_R = raman_response(t)
        H_R_fft = np.fft.fft(h_R) * dt
    
    u = np.zeros((nz, nt), dtype=complex)
    u[0] = u0
    
    linear_op = np.exp(1j * beta2 * omega**2 * dz / 2)
    
    for i in range(1, nz):
        u_current = u[i-1]
        
        u_pml1 = u_current * np.exp(-sigma * dz / 4)
        
        u_linear1 = np.fft.ifft(linear_op * np.fft.fft(u_pml1))
        
        u_pml2 = u_linear1 * np.exp(-sigma * dz / 4)
        
        u_abs2 = np.abs(u_pml2)**2
        
        if use_raman:
            raman_convolution = np.fft.ifft(H_R_fft * np.fft.fft(u_abs2))
            nonlinear_response = (1 - raman_fraction) * u_abs2 + raman_fraction * raman_convolution
        else:
            nonlinear_response = u_abs2
        
        if use_self_steepening and tau_shock > 0:
            u_nl = u_pml2 * nonlinear_response
            u_deriv = np.fft.ifft(1j * omega * np.fft.fft(u_nl))
            nonlinear_term = 1j * gamma * dz * (u_nl + 1j * tau_shock * u_deriv)
        else:
            nonlinear_term = 1j * gamma * dz * u_pml2 * nonlinear_response
        
        u_nonlinear = u_pml2 + nonlinear_term
        
        u_pml3 = u_nonlinear * np.exp(-sigma * dz / 4)
        
        u_linear2 = np.fft.ifft(linear_op * np.fft.fft(u_pml3))
        
        u[i] = u_linear2 * np.exp(-sigma * dz / 4)
    
    z = np.arange(nz) * dz
    
    return z, u


def ssfm_soliton_propagation(
    N=1, dz=0.01, nz=200, nt=1024, t_max=10,
    use_pml=False, pml_width=2.0, sigma_max=15.0,
    use_raman=False, use_self_steepening=False, tau_shock=0.1,
    initial_condition='single', positions=None, amplitudes=None,
    phases=None, velocities=None
):
    """
    分步傅里叶法求解非线性薛定谔方程 (支持高阶非线性效应)
    
    参数:
        N: 孤子阶数
        dz: 传播步长
        nz: 传播步数
        nt: 时间点数
        t_max: 时间窗口半宽
        use_pml: 是否使用PML边界
        pml_width: PML层宽度
        sigma_max: PML最大吸收系数
        use_raman: 是否启用拉曼散射
        use_self_steepening: 是否启用自陡峭
        tau_shock: 自陡峭时间常数
        initial_condition: 初始条件类型 ('single', 'multi', 'bit_pattern')
        positions: 多孤子位置列表
        amplitudes: 多孤子振幅列表
        phases: 多孤子相位列表
        velocities: 多孤子速度列表
    
    返回:
        t: 时间数组
        z: 传播距离数组
        u: 波场演化矩阵
        sigma: PML吸收系数剖面
    """
    t = np.linspace(-t_max, t_max, nt)
    
    if initial_condition == 'single':
        u0 = N / np.cosh(t)
    elif initial_condition == 'multi':
        if positions is None:
            positions = [-3.0, 3.0]
        if amplitudes is None:
            amplitudes = [N, N]
        u0 = generate_multisoliton(t, positions, amplitudes, phases, velocities)
    elif initial_condition == 'bit_pattern':
        bit_sequence = [1, 0, 1, 1, 0, 1]
        u0 = generate_bit_pattern(t, bit_sequence)
    else:
        u0 = N / np.cosh(t)
    
    beta2 = -1.0
    gamma = 1.0
    
    z, u = ssfm_generalized_nlse(
        t, u0, dz, nz,
        beta2=beta2, gamma=gamma,
        use_raman=use_raman, use_self_steepening=use_self_steepening, tau_shock=tau_shock,
        use_pml=use_pml, pml_width=pml_width, sigma_max=sigma_max
    )
    
    sigma = create_pml_profile(nt, t, t_max, pml_width, sigma_max) if use_pml else np.zeros(nt)
    
    return t, z, u, sigma


def plot_soliton_interaction():
    """模拟双孤子相互作用"""
    print("模拟双孤子相互作用...")
    
    nt = 2048
    t_max = 15
    t = np.linspace(-t_max, t_max, nt)
    
    positions = [-4.0, 4.0]
    amplitudes = [1.0, 1.0]
    
    print("  同相双孤子 (吸引相互作用)...")
    u0_inphase = generate_multisoliton(t, positions, amplitudes, phases=[0, 0])
    dz = 0.01
    nz = 800
    z_inphase, u_inphase = ssfm_generalized_nlse(
        t, u0_inphase, dz, nz,
        beta2=-1.0, gamma=1.0,
        use_raman=False, use_self_steepening=False,
        use_pml=True, pml_width=2.0, sigma_max=20.0
    )
    
    print("  反相双孤子 (排斥相互作用)...")
    u0_outphase = generate_multisoliton(t, positions, amplitudes, phases=[0, np.pi])
    z_outphase, u_outphase = ssfm_generalized_nlse(
        t, u0_outphase, dz, nz,
        beta2=-1.0, gamma=1.0,
        use_raman=False, use_self_steepening=False,
        use_pml=True, pml_width=2.0, sigma_max=20.0
    )
    
    print("  不同速度双孤子 (碰撞)...")
    u0_collision = generate_multisoliton(t, positions, amplitudes, 
                                       phases=[0, 0], velocities=[-1.5, 1.5])
    z_collision, u_collision = ssfm_generalized_nlse(
        t, u0_collision, dz, nz,
        beta2=-1.0, gamma=1.0,
        use_raman=False, use_self_steepening=False,
        use_pml=True, pml_width=3.0, sigma_max=20.0
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmax = np.max(np.abs(u0_inphase)**2)
    
    im0 = axes[0].pcolormesh(t, z_inphase, np.abs(u_inphase)**2,
                              cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    axes[0].set_xlabel('时间 t', fontsize=11)
    axes[0].set_ylabel('传播距离 z', fontsize=11)
    axes[0].set_title('同相双孤子 (吸引)', fontsize=12)
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].pcolormesh(t, z_outphase, np.abs(u_outphase)**2,
                              cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    axes[1].set_xlabel('时间 t', fontsize=11)
    axes[1].set_ylabel('传播距离 z', fontsize=11)
    axes[1].set_title('反相双孤子 (排斥)', fontsize=12)
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].pcolormesh(t, z_collision, np.abs(u_collision)**2,
                              cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    axes[2].set_xlabel('时间 t', fontsize=11)
    axes[2].set_ylabel('传播距离 z', fontsize=11)
    axes[2].set_title('不同速度双孤子 (碰撞)', fontsize=12)
    plt.colorbar(im2, ax=axes[2])
    
    fig.suptitle('双孤子相互作用模拟', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig, axes


def plot_higher_order_effects():
    """比较高阶非线性效应的影响"""
    print("比较高阶非线性效应的影响...")
    
    N = 2
    nt = 2048
    t_max = 12
    t = np.linspace(-t_max, t_max, nt)
    u0 = N / np.cosh(t)
    
    dz = 0.005
    nz = 600
    
    print("  标准NLSE (仅克尔非线性)...")
    z_std, u_std = ssfm_generalized_nlse(
        t, u0, dz, nz,
        beta2=-1.0, gamma=1.0,
        use_raman=False, use_self_steepening=False,
        use_pml=True, pml_width=2.0, sigma_max=20.0
    )
    
    print("  NLSE + 拉曼散射...")
    z_raman, u_raman = ssfm_generalized_nlse(
        t, u0, dz, nz,
        beta2=-1.0, gamma=1.0,
        use_raman=True, use_self_steepening=False,
        use_pml=True, pml_width=2.0, sigma_max=20.0
    )
    
    print("  NLSE + 自陡峭...")
    z_ss, u_ss = ssfm_generalized_nlse(
        t, u0, dz, nz,
        beta2=-1.0, gamma=1.0,
        use_raman=False, use_self_steepening=True, tau_shock=0.1,
        use_pml=True, pml_width=2.0, sigma_max=20.0
    )
    
    print("  NLSE + 拉曼 + 自陡峭...")
    z_full, u_full = ssfm_generalized_nlse(
        t, u0, dz, nz,
        beta2=-1.0, gamma=1.0,
        use_raman=True, use_self_steepening=True, tau_shock=0.1,
        use_pml=True, pml_width=2.0, sigma_max=20.0
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    vmax = np.max(np.abs(u0)**2)
    
    im0 = axes[0, 0].pcolormesh(t, z_std, np.abs(u_std)**2,
                                 cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    axes[0, 0].set_title('标准NLSE (克尔非线性)', fontsize=11)
    axes[0, 0].set_ylabel('传播距离 z', fontsize=10)
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].pcolormesh(t, z_raman, np.abs(u_raman)**2,
                                 cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    axes[0, 1].set_title('+ 拉曼散射 (脉冲自频移)', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].pcolormesh(t, z_ss, np.abs(u_ss)**2,
                                 cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    axes[1, 0].set_title('+ 自陡峭 (脉冲不对称)', fontsize=11)
    axes[1, 0].set_xlabel('时间 t', fontsize=10)
    axes[1, 0].set_ylabel('传播距离 z', fontsize=10)
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].pcolormesh(t, z_full, np.abs(u_full)**2,
                                 cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    axes[1, 1].set_title('+ 拉曼 + 自陡峭 (全部效应)', fontsize=11)
    axes[1, 1].set_xlabel('时间 t', fontsize=10)
    plt.colorbar(im3, ax=axes[1, 1])
    
    fig.suptitle('高阶非线性效应对二阶孤子演化的影响', fontsize=14)
    plt.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    idx = -1
    ax2.plot(t, np.abs(u_std[idx])**2, 'b-', label='标准NLSE', linewidth=2)
    ax2.plot(t, np.abs(u_raman[idx])**2, 'r--', label='+拉曼', linewidth=2)
    ax2.plot(t, np.abs(u_ss[idx])**2, 'g-.', label='+自陡峭', linewidth=2)
    ax2.plot(t, np.abs(u_full[idx])**2, 'm:', label='+全部', linewidth=2)
    ax2.set_xlabel('时间 t', fontsize=12)
    ax2.set_ylabel('光强 |u|²', fontsize=12)
    ax2.set_title(f'z = {z_full[idx]:.1f} 处的波形对比', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-8, 8)
    plt.tight_layout()
    
    return fig, fig2


def plot_fiber_communication_demo():
    """光纤通信系统脉冲传输演示"""
    print("光纤通信系统脉冲传输演示...")
    
    nt = 4096
    t_max = 25
    t = np.linspace(-t_max, t_max, nt)
    
    bit_sequence = [1, 0, 1, 1, 0, 1, 0, 1]
    u0 = generate_bit_pattern(t, bit_sequence, bit_width=1.0, spacing=4.0, start_pos=-12.0)
    
    dz = 0.01
    nz = 400
    
    print("  标准光纤传输...")
    z, u = ssfm_generalized_nlse(
        t, u0, dz, nz,
        beta2=-1.0, gamma=1.0,
        use_raman=True, use_self_steepening=True, tau_shock=0.05,
        use_pml=True, pml_width=3.0, sigma_max=25.0
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    im = axes[0].pcolormesh(t, z, np.abs(u)**2, cmap='viridis', shading='auto')
    axes[0].set_xlabel('时间 t', fontsize=12)
    axes[0].set_ylabel('传播距离 z', fontsize=12)
    axes[0].set_title('比特序列在光纤中的传输演化', fontsize=12)
    plt.colorbar(im, ax=axes[0])
    
    z_positions = [0, nz//4, nz//2, 3*nz//4, -1]
    for i, idx in enumerate(z_positions):
        axes[1].plot(t, np.abs(u[idx])**2 + i*0.3, linewidth=2,
                     label=f'z={z[idx]:.1f}')
    axes[1].set_xlabel('时间 t', fontsize=12)
    axes[1].set_ylabel('光强 (偏移显示)', fontsize=12)
    axes[1].set_title('不同传播距离的比特序列波形', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yticks([])
    axes[1].set_xlim(-20, 20)
    
    fig.suptitle('光纤通信系统：比特序列传输模拟', fontsize=14)
    plt.tight_layout()
    
    return fig, axes


def calculate_integrated_intensity(u):
    """计算总光强守恒验证"""
    return np.trapz(np.abs(u)**2, axis=1)


def plot_evolution(t, z, u, cmap='viridis', title=''):
    """绘制时空演化图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im = axes[0].pcolormesh(t, z, np.abs(u)**2, cmap=cmap, shading='auto')
    axes[0].set_xlabel('时间 t', fontsize=12)
    axes[0].set_ylabel('传播距离 z', fontsize=12)
    axes[0].set_title('光强时空演化 |u(t,z)|²', fontsize=12)
    plt.colorbar(im, ax=axes[0])
    
    idx_z0 = 0
    idx_z_mid = len(z) // 2
    idx_z_end = len(z) - 1
    
    axes[1].plot(t, np.abs(u[idx_z0])**2, 'b-', linewidth=2, 
                 label=f'z={z[idx_z0]:.2f}')
    axes[1].plot(t, np.abs(u[idx_z_mid])**2, 'r--', linewidth=2,
                 label=f'z={z[idx_z_mid]:.2f}')
    axes[1].plot(t, np.abs(u[idx_z_end])**2, 'g-.', linewidth=2,
                 label=f'z={z[idx_z_end]:.2f}')
    axes[1].set_xlabel('时间 t', fontsize=12)
    axes[1].set_ylabel('光强 |u|²', fontsize=12)
    axes[1].set_title('不同传播距离的波形', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    return fig, axes


def create_animation(t, z, u, interval=50, save_path=None):
    """创建波传播动画"""
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(t, np.abs(u[0])**2, 'b-', linewidth=2)
    ax.set_xlabel('时间 t', fontsize=12)
    ax.set_ylabel('光强 |u|²', fontsize=12)
    ax.set_title('孤子传播动画', fontsize=14)
    ax.set_ylim(0, 1.1 * np.max(np.abs(u)**2))
    ax.grid(True, alpha=0.3)
    
    z_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                     fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    def update(frame):
        line.set_ydata(np.abs(u[frame])**2)
        z_text.set_text(f'z = {z[frame]:.2f}')
        return line, z_text
    
    ani = FuncAnimation(fig, update, frames=len(z), interval=interval, blit=True)
    
    if save_path:
        ani.save(save_path, writer='ffmpeg', dpi=100)
    
    return fig, ani


def main():
    print("=" * 70)
    print("分步傅里叶法 - 高阶非线性效应与多孤子相互作用")
    print("=" * 70)
    
    print("\n1. 双孤子相互作用模拟...")
    plot_soliton_interaction()
    
    print("\n2. 高阶非线性效应比较...")
    plot_higher_order_effects()
    
    print("\n3. 光纤通信比特序列传输...")
    plot_fiber_communication_demo()
    
    print("\n4. 带PML和高阶效应的一阶孤子...")
    t, z, u, sigma = ssfm_soliton_propagation(
        N=1, dz=0.01, nz=200, nt=1024, t_max=10,
        use_pml=True, pml_width=2.0, sigma_max=20.0,
        use_raman=True, use_self_steepening=True, tau_shock=0.1
    )
    plot_evolution(t, z, u, title='一阶孤子 (PML + 高阶非线性效应)')
    
    intensity = calculate_integrated_intensity(u)
    print(f"   初始光强: {intensity[0]:.4f}, 最终光强: {intensity[-1]:.4f}")
    
    print("\n计算完成！显示图形中...")
    print("\n" + "=" * 70)
    print("功能说明:")
    print("=" * 70)
    print("  1. 自陡峭效应: 脉冲不对称展宽，频谱红移")
    print("  2. 拉曼散射: 脉冲自频移，能量向低频转移")
    print("  3. 双孤子相互作用:")
    print("     - 同相: 吸引，周期性碰撞")
    print("     - 反相: 排斥，相互远离")
    print("     - 不同速度: 碰撞后保持特性不变")
    print("  4. 光纤通信: 比特序列传输模拟")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
