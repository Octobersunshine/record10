import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tmm_simple(wavelengths, n_list, d_list, n0=1.0, ns=1.5, theta=0.0):
    """简化的传递矩阵法实现"""
    R = np.zeros_like(wavelengths, dtype=float)
    T = np.zeros_like(wavelengths, dtype=float)
    cos_theta0 = np.cos(theta)

    for idx, lambda0 in enumerate(wavelengths):
        k0 = 2 * np.pi / lambda0
        M = np.eye(2, dtype=complex)

        for n, d in zip(n_list, d_list):
            sin_theta = n0 * np.sin(theta) / n
            cos_theta = np.sqrt(1 - sin_theta ** 2 + 0j)
            delta = k0 * n * d * cos_theta
            eta = n * cos_theta

            M = M @ np.array([
                [np.cos(delta), -1j * np.sin(delta) / eta],
                [-1j * eta * np.sin(delta), np.cos(delta)]
            ])

        sin_theta_s = n0 * np.sin(theta) / ns
        cos_theta_s = np.sqrt(1 - sin_theta_s ** 2 + 0j)
        eta_s = ns * cos_theta_s
        eta0 = n0 * cos_theta0

        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]
        denom = A * eta_s + B * eta0 * eta_s + C + D * eta0

        r = (A * eta_s + B * eta0 * eta_s - C - D * eta0) / denom
        t = 2 * eta_s / denom

        R[idx] = np.real(np.abs(r) ** 2)
        T[idx] = np.real(np.abs(t) ** 2 * eta0 / eta_s)

    return R, T


def design_filter_fast(lambda_center=550, bandwidth=50, n_layers=6, max_iter=200):
    """快速滤波器设计"""
    wavelengths = np.linspace(lambda_center - 2 * bandwidth,
                              lambda_center + 2 * bandwidth, 150)

    def target_spec(w, lc, bw):
        sigma = bw / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-((w - lc) ** 2) / (2 * sigma ** 2))

    target_T = target_spec(wavelengths, lambda_center, bandwidth)

    n_low, n_high = 1.38, 2.35
    qw = lambda_center / 4.0

    np.random.seed(123)
    n_init = []
    d_init = []
    for i in range(n_layers):
        base_n = n_high if i % 2 == 0 else n_low
        n_init.append(base_n + np.random.uniform(-0.02, 0.02))
        d_init.append(qw / n_init[-1])

    def loss(params):
        n_opt = np.clip(params[:n_layers], 1.3, 2.5)
        d_opt = np.clip(params[n_layers:], 20, 300)
        _, T = tmm_simple(wavelengths, n_opt, d_opt)

        lb = lambda_center - bandwidth / 2
        ub = lambda_center + bandwidth / 2
        in_band = (wavelengths >= lb) & (wavelengths <= ub)
        out_band = ~in_band

        in_loss = np.mean((1.0 - T[in_band]) ** 2)
        out_loss = np.mean(T[out_band] ** 2)

        return 0.5 * in_loss + 0.5 * out_loss

    x0 = np.concatenate([n_init, d_init])
    print(f"初始损失: {loss(x0):.4f}")

    result = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': max_iter})

    n_opt = np.clip(result.x[:n_layers], 1.3, 2.5).tolist()
    d_opt = np.clip(result.x[n_layers:], 20, 300).tolist()

    print(f"最终损失: {result.fun:.4f}, 迭代次数: {result.nit}")

    return wavelengths, target_T, n_opt, d_opt


def main():
    print("=" * 60)
    print("薄膜光学滤波器 - 传递矩阵法快速演示")
    print("=" * 60)

    lambda_center = 550
    bandwidth = 50
    n_layers = 6

    print(f"\n设计参数: 中心波长={lambda_center}nm, 带宽={bandwidth}nm, 层数={n_layers}")
    print("开始优化...\n")

    wavelengths, target_T, n_opt, d_opt = design_filter_fast(
        lambda_center, bandwidth, n_layers, max_iter=150
    )

    R, T = tmm_simple(wavelengths, n_opt, d_opt)

    print(f"\n{'='*60}")
    print("优化后的膜系参数:")
    print(f"{'层号':<6} {'折射率':<12} {'厚度(nm)':<12} {'光学厚度(nm)':<15}")
    print("-" * 50)
    for i, (n, d) in enumerate(zip(n_opt, d_opt)):
        print(f"{i+1:<6} {n:<12.4f} {d:<12.2f} {n*d:<15.1f}")

    print(f"\n{'='*60}")
    print("性能评估:")
    max_T = np.max(T)
    peak_wl = wavelengths[np.argmax(T)]

    lb = lambda_center - bandwidth / 2
    ub = lambda_center + bandwidth / 2
    in_band = (wavelengths >= lb) & (wavelengths <= ub)
    out_band = ~in_band

    avg_in = np.mean(T[in_band])
    avg_out = np.mean(T[out_band])

    T50 = max_T / 2
    above = np.where(T >= T50)[0]
    fwhm = wavelengths[above[-1]] - wavelengths[above[0]] if len(above) > 1 else 0

    print(f"  峰值透射率: {max_T*100:.2f}% @ {peak_wl:.1f} nm")
    print(f"  通带平均透射率: {avg_in*100:.2f}%")
    print(f"  阻带平均透射率: {avg_out*100:.2f}%")
    print(f"  实际半高宽: {fwhm:.1f} nm (目标: {bandwidth} nm)")
    print(f"  抑制比: {10*np.log10(avg_in/(avg_out+1e-10)):.1f} dB")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(wavelengths, T*100, 'b-', linewidth=2, label='Transmittance')
    axes[0].plot(wavelengths, target_T*100, 'r--', linewidth=1.5, label='Target')
    axes[0].plot(wavelengths, R*100, 'orange', linewidth=1.5, alpha=0.6, label='Reflectance')
    axes[0].axvline(lambda_center, color='k', linestyle=':', alpha=0.7, label='Center')
    axes[0].axvspan(lb, ub, alpha=0.15, color='green', label='Passband')
    axes[0].set_xlabel('Wavelength (nm)', fontsize=12)
    axes[0].set_ylabel('(%)', fontsize=12)
    axes[0].set_title(f'Bandpass Filter Spectrum (λ₀={lambda_center} nm, BW={bandwidth} nm)', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)

    x = np.arange(len(n_opt))
    w = 0.35
    ax1 = axes[1]
    ax1.bar(x - w/2, n_opt, w, color='steelblue', label='Refractive Index')
    ax1.set_xlabel('Layer Number', fontsize=12)
    ax1.set_ylabel('Refractive Index', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(1.0, 2.8)

    ax2 = ax1.twinx()
    ax2.bar(x + w/2, d_opt, w, color='salmon', label='Thickness (nm)')
    ax2.set_ylabel('Thickness (nm)', fontsize=12, color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')

    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc='upper right', fontsize=10)
    axes[1].set_title('Thin Film Stack', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('quick_demo_result.png', dpi=120, bbox_inches='tight')
    print(f"\n结果图已保存为 'quick_demo_result.png'")
    plt.close()

    print("\n程序执行完成！")


if __name__ == "__main__":
    main()
