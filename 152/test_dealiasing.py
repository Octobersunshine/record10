import numpy as np
import matplotlib.pyplot as plt

def test_dealiasing_masks():
    print("=" * 60)
    print("去混叠方法测试")
    print("=" * 60)
    
    N = 64
    L = 10.0
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    k_max = np.max(np.abs(k))
    
    k_cutoff = 2.0/3.0 * k_max
    
    mask_23 = (np.abs(k) <= k_cutoff).astype(float)
    
    k_ratio = k / k_cutoff
    mask_exp = np.exp(-36.0 * k_ratio**36)
    mask_exp = np.where(np.abs(k_ratio) <= 1.0, 1.0, mask_exp)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    k_sorted = np.fft.fftshift(k)
    mask_23_sorted = np.fft.fftshift(mask_23)
    mask_exp_sorted = np.fft.fftshift(mask_exp)
    
    axes[0].plot(k_sorted, mask_23_sorted, 'b-', linewidth=2, label='2/3规则')
    axes[0].plot(k_sorted, mask_exp_sorted, 'r--', linewidth=2, label='指数滤波器')
    axes[0].axvline(x=k_cutoff, color='k', linestyle=':', alpha=0.5, label='k_cutoff = 2/3 k_max')
    axes[0].axvline(x=-k_cutoff, color='k', linestyle=':', alpha=0.5)
    axes[0].set_title('去混叠滤波器对比', fontsize=12)
    axes[0].set_xlabel('波数 k')
    axes[0].set_ylabel('滤波器响应')
    axes[0].set_xlim(-k_max, k_max)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    k_sq = k**2
    
    u = np.exp(-x**2 / 2)
    u_hat = np.fft.fft(u)
    
    u_sq = u**2
    u_sq_hat = np.fft.fft(u_sq)
    
    u_sq_hat_23 = u_sq_hat * mask_23
    u_sq_23 = np.fft.ifft(u_sq_hat_23).real
    
    u_sq_hat_exp = u_sq_hat * mask_exp
    u_sq_exp = np.fft.ifft(u_sq_hat_exp).real
    
    axes[1].plot(x, u_sq, 'k-', linewidth=3, label='精确 u²')
    axes[1].plot(x, u_sq_23, 'b--', linewidth=2, label='2/3规则去混叠')
    axes[1].plot(x, u_sq_exp, 'r:', linewidth=2, label='指数滤波去混叠')
    axes[1].set_title('二次非线性项: 去混叠效果对比', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u²')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dealiasing_comparison.png', dpi=150, bbox_inches='tight')
    print("\n滤波器对比图已保存至: dealiasing_comparison.png")
    
    error_23 = np.mean(np.abs(u_sq - u_sq_23))
    error_exp = np.mean(np.abs(u_sq - u_sq_exp))
    print(f"\n平均绝对误差:")
    print(f"  2/3规则: {error_23:.6f}")
    print(f"  指数滤波: {error_exp:.6f}")
    
    plt.show()

def test_aliasing_demonstration():
    print("\n" + "=" * 60)
    print("混叠现象演示")
    print("=" * 60)
    
    N = 32
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    k_max = np.max(np.abs(k))
    
    k1 = 0.8 * k_max
    k2 = 0.7 * k_max
    
    u1 = np.cos(k1 * x)
    u2 = np.cos(k2 * x)
    
    product = u1 * u2
    product_analytic = 0.5 * np.cos((k1 + k2) * x) + 0.5 * np.cos((k1 - k2) * x)
    
    product_hat = np.fft.fft(product)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(x, u1, 'b-', label=f'k1={k1:.2f}')
    axes[0, 0].plot(x, u2, 'r-', label=f'k2={k2:.2f}')
    axes[0, 0].set_title('原始信号')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    k_sorted = np.fft.fftshift(k)
    prod_spectrum = np.fft.fftshift(np.abs(product_hat))
    axes[0, 1].stem(k_sorted, prod_spectrum, basefmt='b-', use_line_collection=True)
    axes[0, 1].axvline(x=k_max, color='k', linestyle='--', label='±k_max')
    axes[0, 1].axvline(x=-k_max, color='k', linestyle='--')
    axes[0, 1].axvline(x=k1+k2, color='r', linestyle=':', label=f'k1+k2={k1+k2:.2f}')
    axes[0, 1].axvline(x=-(k1+k2), color='r', linestyle=':')
    axes[0, 1].set_title('乘积信号的频谱 (混叠演示)')
    axes[0, 1].set_xlabel('k')
    axes[0, 1].set_xlim(-2*k_max, 2*k_max)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(x, product, 'b-', linewidth=2, label='FFT计算的乘积')
    axes[1, 0].plot(x, product_analytic, 'r--', linewidth=2, label='解析解')
    axes[1, 0].set_title('信号乘积: 混叠误差')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    k_cutoff = 2.0/3.0 * k_max
    mask = (np.abs(k) <= k_cutoff).astype(float)
    
    u1_hat = np.fft.fft(u1) * mask
    u1_filtered = np.fft.ifft(u1_hat).real
    u2_hat = np.fft.fft(u2) * mask
    u2_filtered = np.fft.ifft(u2_hat).real
    
    product_dealiased = u1_filtered * u2_filtered
    product_dealiased_hat = np.fft.fft(product_dealiased) * mask
    product_dealiased = np.fft.ifft(product_dealiased_hat).real
    
    product_analytic_filtered = 0.5 * np.cos((k1 - k2) * x)
    
    axes[1, 1].plot(x, product, 'b-', linewidth=2, alpha=0.3, label='原始(有混叠)')
    axes[1, 1].plot(x, product_dealiased, 'r-', linewidth=2, label='去混叠后')
    axes[1, 1].plot(x, product_analytic_filtered, 'k--', linewidth=2, label='解析解(低频部分)')
    axes[1, 1].set_title('去混叠后的信号乘积')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('aliasing_demo.png', dpi=150, bbox_inches='tight')
    print("\n混叠演示图已保存至: aliasing_demo.png")
    
    error_with_aliasing = np.mean(np.abs(product - product_analytic))
    error_dealiased = np.mean(np.abs(product_dealiased - product_analytic_filtered))
    print(f"\n平均绝对误差:")
    print(f"  有混叠: {error_with_aliasing:.6f}")
    print(f"  去混叠: {error_dealiased:.6f}")
    
    plt.show()

if __name__ == "__main__":
    test_dealiasing_masks()
    test_aliasing_demonstration()
    print("\n" + "=" * 60)
    print("去混叠测试完成!")
    print("=" * 60)
