import numpy as np
from scipy.integrate import trapezoid
import warnings


def compute_fourier_coefficients(t, y, T, N, check_nyquist=True):
    """
    计算周期函数的傅里叶级数系数
    
    参数:
        t: 采样时间点数组，长度为M
        y: 对应采样点的函数值数组，长度为M
        T: 周期
        N: 谐波次数（返回前N次谐波的系数）
        check_nyquist: 是否检查奈奎斯特采样定理（默认True）
    
    返回:
        a0: 直流分量系数
        an: 余弦系数数组，长度为N (对应n=1到N)
        bn: 正弦系数数组，长度为N (对应n=1到N)
    """
    omega = 2 * np.pi / T
    M = len(t)
    
    if check_nyquist:
        f_max = N / T
        sample_rate = M / T
        nyquist_rate = 2 * f_max
        
        if sample_rate < nyquist_rate:
            warnings.warn(
                f"采样率不足，可能发生混叠！\n"
                f"  采样率: {sample_rate:.2f} Hz\n"
                f"  奈奎斯特率（2×最高频率）: {nyquist_rate:.2f} Hz\n"
                f"  最高谐波频率 (N={N}): {f_max:.2f} Hz\n"
                f"  建议: 增加采样点数 M 到至少 {int(np.ceil(nyquist_rate * T))}，"
                f"或减少谐波次数 N 到最多 {int(np.floor(sample_rate * T / 2))}",
                UserWarning,
                stacklevel=2
            )
    
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        dt = None
    
    a0 = (2.0 / T) * trapezoid(y, t)
    
    an = np.zeros(N)
    bn = np.zeros(N)
    
    for n in range(1, N + 1):
        cos_term = np.cos(n * omega * t)
        sin_term = np.sin(n * omega * t)
        
        an[n - 1] = (2.0 / T) * trapezoid(y * cos_term, t)
        bn[n - 1] = (2.0 / T) * trapezoid(y * sin_term, t)
    
    return a0, an, bn


def reconstruct_signal(t, a0, an, bn, T):
    """
    根据傅里叶系数重构逼近曲线
    
    参数:
        t: 要重构的时间点数组
        a0: 直流分量系数
        an: 余弦系数数组
        bn: 正弦系数数组
        T: 周期
    
    返回:
        y_reconstructed: 重构的信号值数组
    """
    omega = 2 * np.pi / T
    N = len(an)
    
    y_reconstructed = np.ones_like(t) * (a0 / 2.0)
    
    for n in range(1, N + 1):
        y_reconstructed += an[n - 1] * np.cos(n * omega * t) + bn[n - 1] * np.sin(n * omega * t)
    
    return y_reconstructed


def compute_complex_coefficients(t, y, T, N, check_nyquist=True):
    """
    计算复数形式傅里叶级数系数 cn
    
    复数傅里叶级数: f(t) = sum_{n=-N}^{N} cn * exp(j * n * omega * t)
    
    其中: cn = (1/T) * integral_0^T f(t) * exp(-j * n * omega * t) dt
    
    参数:
        t: 采样时间点数组，长度为M
        y: 对应采样点的函数值数组，长度为M
        T: 周期
        N: 谐波次数（返回 -N 到 N 共 2N+1 个系数）
        check_nyquist: 是否检查奈奎斯特采样定理（默认True）
    
    返回:
        cn: 复数系数数组，长度为 2N+1，索引 0~2N 对应 n=-N~N
            即 cn[0] = c_{-N}, cn[N] = c_0, cn[2N] = c_N
        n_indices: 对应的 n 值数组 [-N, -N+1, ..., N-1, N]
    """
    omega = 2 * np.pi / T
    M = len(t)
    
    if check_nyquist:
        f_max = N / T
        sample_rate = M / T
        nyquist_rate = 2 * f_max
        
        if sample_rate < nyquist_rate:
            warnings.warn(
                f"采样率不足，可能发生混叠！\n"
                f"  采样率: {sample_rate:.2f} Hz\n"
                f"  奈奎斯特率（2×最高频率）: {nyquist_rate:.2f} Hz\n"
                f"  最高谐波频率 (N={N}): {f_max:.2f} Hz\n"
                f"  建议: 增加采样点数 M 到至少 {int(np.ceil(nyquist_rate * T))}，"
                f"或减少谐波次数 N 到最多 {int(np.floor(sample_rate * T / 2))}",
                UserWarning,
                stacklevel=2
            )
    
    n_indices = np.arange(-N, N + 1)
    cn = np.zeros(2 * N + 1, dtype=complex)
    
    for idx, n in enumerate(n_indices):
        basis = np.exp(-1j * n * omega * t)
        cn[idx] = (1.0 / T) * trapezoid(y * basis, t)
    
    return cn, n_indices


def reconstruct_signal_complex(t, cn, n_indices, T):
    """
    根据复数傅里叶系数重构逼近曲线
    
    参数:
        t: 要重构的时间点数组
        cn: 复数系数数组
        n_indices: 对应的 n 值数组
        T: 周期
    
    返回:
        y_reconstructed: 重构的信号值数组（实数）
    """
    omega = 2 * np.pi / T
    y_reconstructed = np.zeros_like(t, dtype=complex)
    
    for idx, n in enumerate(n_indices):
        y_reconstructed += cn[idx] * np.exp(1j * n * omega * t)
    
    return np.real(y_reconstructed)


def compute_spectrum(cn, n_indices):
    """
    从复数傅里叶系数计算幅度谱和相位谱
    
    参数:
        cn: 复数系数数组
        n_indices: 对应的 n 值数组
    
    返回:
        amplitude: 幅度谱数组
        phase: 相位谱数组（弧度，范围 [-pi, pi]）
    """
    amplitude = np.abs(cn)
    phase = np.angle(cn)
    return amplitude, phase


def periodic_extend(t_local, y_local, T, num_periods=3):
    """
    对非周期函数在 [0, T] 上的采样数据进行周期延拓
    
    参数:
        t_local: 原始采样时间点数组（应在 [0, T] 范围内）
        y_local: 对应采样点的函数值数组
        T: 延拓周期
        num_periods: 延拓的周期数（默认3）
    
    返回:
        t_extended: 延拓后的时间点数组
        y_extended: 延拓后的函数值数组
    """
    t_extended = np.concatenate([t_local + k * T for k in range(num_periods)])
    y_extended = np.tile(y_local, num_periods)
    return t_extended, y_extended


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    rcParams['axes.unicode_minus'] = False
    
    T = 2 * np.pi
    N = 10
    M = 1000
    
    t = np.linspace(0, T, M, endpoint=False)
    y = np.sign(np.sin(t))
    
    a0, an, bn = compute_fourier_coefficients(t, y, T, N)
    
    print("=" * 60)
    print("1. 实数形式傅里叶级数系数（方波）")
    print("=" * 60)
    print(f"a0 = {a0:.6f}")
    print(f"a0/2 = {a0/2:.6f}")
    print("\n  n        an            bn")
    print("-" * 30)
    for n in range(N):
        print(f"{n+1:3d}   {an[n]:10.6f}   {bn[n]:10.6f}")
    
    cn, n_indices = compute_complex_coefficients(t, y, T, N)
    
    print("\n" + "=" * 60)
    print("2. 复数形式傅里叶级数系数（方波）")
    print("=" * 60)
    print(f"{'n':>4s}   {'Re(cn)':>12s}   {'Im(cn)':>12s}   {'|cn|':>10s}   {'angle(cn)':>10s}")
    print("-" * 55)
    for idx, n_val in enumerate(n_indices):
        re_part = cn[idx].real
        im_part = cn[idx].imag
        amp = np.abs(cn[idx])
        pha = np.angle(cn[idx])
        print(f"{n_val:4d}   {re_part:12.6f}   {im_part:12.6f}   {amp:10.6f}   {pha:10.6f}")
    
    amplitude, phase = compute_spectrum(cn, n_indices)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].stem(n_indices, amplitude)
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('|cn|')
    axes[0].set_title('幅度谱（复数系数）')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].stem(n_indices, phase)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('phase [rad]')
    axes[1].set_title('相位谱（复数系数）')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complex_spectrum.png', dpi=150, bbox_inches='tight')
    print("\n复数系数幅度谱/相位谱已保存为 complex_spectrum.png")
    
    t_recon = np.linspace(0, T, 500)
    y_recon_real = reconstruct_signal(t_recon, a0, an, bn, T)
    y_recon_complex = reconstruct_signal_complex(t_recon, cn, n_indices, T)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].plot(t, y, 'b-', label='原始信号', linewidth=2)
    axes[0].plot(t_recon, y_recon_real, 'r--', label=f'实数系数逼近 (N={N})', linewidth=1.5)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('y(t)')
    axes[0].set_title('方波 - 实数傅里叶级数逼近')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, y, 'b-', label='原始信号', linewidth=2)
    axes[1].plot(t_recon, y_recon_complex, 'g--', label=f'复数系数逼近 (N={N})', linewidth=1.5)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('y(t)')
    axes[1].set_title('方波 - 复数傅里叶级数逼近')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_vs_complex_reconstruction.png', dpi=150, bbox_inches='tight')
    print("实数/复数逼近对比图已保存为 real_vs_complex_reconstruction.png")
    
    diff_max = np.max(np.abs(y_recon_real - y_recon_complex))
    print(f"\n实数 vs 复数重构最大差异: {diff_max:.2e}")
    
    print("\n" + "=" * 60)
    print("3. 非周期函数的周期延拓逼近")
    print("=" * 60)
    print("原始函数: f(t) = t^2, 定义在 [0, T] 上（非周期）")
    
    M_local = 500
    t_local = np.linspace(0, T, M_local, endpoint=False)
    y_local = t_local ** 2
    
    T_ext = T
    num_periods = 3
    t_extended, y_extended = periodic_extend(t_local, y_local, T_ext, num_periods)
    
    cn_ext, n_indices_ext = compute_complex_coefficients(t_local, y_local, T_ext, N)
    a0_ext, an_ext, bn_ext = compute_fourier_coefficients(t_local, y_local, T_ext, N)
    
    t_plot = np.linspace(0, T_ext * num_periods, 1500)
    y_plot_original = np.zeros_like(t_plot)
    for i, ti in enumerate(t_plot):
        t_mod = ti % T_ext
        y_plot_original[i] = t_mod ** 2
    
    y_plot_recon = reconstruct_signal_complex(t_plot, cn_ext, n_indices_ext, T_ext)
    
    print(f"周期延拓: {num_periods} 个周期, 每周期 T = {T_ext:.4f}")
    print(f"\n复数系数 cn (n = -5 ~ 5):")
    print(f"{'n':>4s}   {'Re(cn)':>12s}   {'Im(cn)':>12s}   {'|cn|':>10s}")
    print("-" * 45)
    for idx, n_val in enumerate(n_indices_ext):
        if abs(n_val) <= 5:
            re_part = cn_ext[idx].real
            im_part = cn_ext[idx].imag
            amp = np.abs(cn_ext[idx])
            print(f"{n_val:4d}   {re_part:12.6f}   {im_part:12.6f}   {amp:10.6f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].plot(t_extended, y_extended, 'b-', label='原始信号 (周期延拓)', linewidth=2)
    axes[0].plot(t_plot, y_plot_recon, 'r--', label=f'傅里叶逼近 (N={N})', linewidth=1.5)
    for k in range(num_periods):
        axes[0].axvline(x=k * T_ext, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('y(t)')
    axes[0].set_title(f'非周期函数 f(t)=t^2 周期延拓后的傅里叶逼近 ({num_periods}个周期)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    amp_ext, phase_ext = compute_spectrum(cn_ext, n_indices_ext)
    axes[1].stem(n_indices_ext, amp_ext)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('|cn|')
    axes[1].set_title('非周期函数延拓后的幅度谱')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nonperiodic_extension.png', dpi=150, bbox_inches='tight')
    print("\n非周期函数延拓图像已保存为 nonperiodic_extension.png")
    
    mse_ext = np.mean((y_local - reconstruct_signal(t_local, a0_ext, an_ext, bn_ext, T_ext))**2)
    print(f"单周期内均方误差 (MSE): {mse_ext:.8f}")
    
    mse_complex_ext = np.mean((y_local - reconstruct_signal_complex(t_local, cn_ext, n_indices_ext, T_ext))**2)
    print(f"复数系数单周期内均方误差 (MSE): {mse_complex_ext:.8f}")
    
    print("\n" + "=" * 60)
    print("4. 系数关系验证: cn 与 an/bn 的对应")
    print("=" * 60)
    print("理论关系: c0 = a0/2, cn = (an - j*bn)/2  (n>0)")
    print(f"\nc0 (复数) = {cn[N]:.6f}")
    print(f"a0/2 (实数) = {a0/2:.6f}")
    print(f"差异: {abs(cn[N] - a0/2):.2e}")
    
    print(f"\n{'n':>3s}   {'cn (re)':>10s}   {'(an-j*bn)/2 (re)':>16s}   {'cn (im)':>10s}   {'(an-j*bn)/2 (im)':>16s}")
    print("-" * 60)
    for n_val in range(1, min(6, N + 1)):
        cn_val = cn[N + n_val]
        expected = (an[n_val - 1] - 1j * bn[n_val - 1]) / 2.0
        print(f"{n_val:3d}   {cn_val.real:10.6f}   {expected.real:16.6f}   {cn_val.imag:10.6f}   {expected.imag:16.6f}")
    
    print("\n" + "=" * 60)
    print("5. 采样率不足演示（触发混叠警告）")
    print("=" * 60)
    
    M_low = 15
    N_high = 10
    t_low = np.linspace(0, T, M_low, endpoint=False)
    y_low = np.sign(np.sin(t_low))
    
    print(f"采样点数 M = {M_low}, 谐波次数 N = {N_high}")
    print(f"采样率 = {M_low/T:.2f} Hz, 奈奎斯特率 = {2*N_high/T:.2f} Hz")
    print("预期会触发混叠警告...\n")
    
    cn_low, n_indices_low = compute_complex_coefficients(t_low, y_low, T, N_high)
    
    print("\n采样不足时的复数系数:")
    print(f"{'n':>4s}   {'|cn|':>10s}")
    print("-" * 20)
    for idx, n_val in enumerate(n_indices_low):
        print(f"{n_val:4d}   {np.abs(cn_low[idx]):10.6f}")
    
    print(f"\n对于 N={N} 次谐波，需要至少 {2*N} 个采样点")
    print(f"当前采样点数 M={M}，满足 M >= 2N 条件: {M >= 2*N}")
    
    print("\n所有测试完成!")
