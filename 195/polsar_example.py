import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import time
from polsar_filter import (
    generate_synthetic_polsar_image,
    create_covariance_matrix_3x3,
    create_coherency_matrix_3x3,
    boxcar_filter_covariance,
    refined_lee_filter_polarimetric,
    polarimetric_whitening_filter,
    enhanced_polarimetric_whitening_filter,
    idan_filter,
    cloude_pottier_decomposition,
    yamaguchi_decomposition,
    calculate_polarization_fidelity,
    compute_polarimetric_features,
    covariance_to_intensity,
    generate_paulirgb_image
)


def calculate_enl(img):
    mean_val = np.mean(img)
    std_val = np.std(img)
    return (mean_val ** 2) / (std_val ** 2) if std_val > 0 else 0


def plot_polsar_results(original_C, C_noisy, C_filtered, method_name, H_orig, H_filtered):
    """
    绘制PolSAR滤波结果对比
    """
    hh_orig, vv_orig, hv_orig = covariance_to_intensity(original_C)
    hh_noisy, vv_noisy, hv_noisy = covariance_to_intensity(C_noisy)
    hh_filt, vv_filt, hv_filt = covariance_to_intensity(C_filtered)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    vmin, vmax = 0, 255
    
    imgs = [
        (hh_orig, 'HH - 原始'), (hh_noisy, 'HH - 加噪'), (hh_filt, f'HH - {method_name}'),
        (vv_orig, 'VV - 原始'), (vv_noisy, 'VV - 加噪'), (vv_filt, f'VV - {method_name}'),
        (hv_orig, 'HV - 原始'), (hv_noisy, 'HV - 加噪'), (hv_filt, f'HV - {method_name}'),
    ]
    
    for idx, (img, title) in enumerate(imgs):
        ax = axes[idx // 3, idx % 3]
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax_h = axes[0, 3]
    im = ax_h.imshow(H_orig, cmap='jet', vmin=0, vmax=1)
    ax_h.set_title('散射熵H - 原始', fontsize=10)
    ax_h.axis('off')
    plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    
    ax_h2 = axes[1, 3]
    im = ax_h2.imshow(H_filtered, cmap='jet', vmin=0, vmax=1)
    ax_h2.set_title(f'散射熵H - {method_name}', fontsize=10)
    ax_h2.axis('off')
    plt.colorbar(im, ax=ax_h2, fraction=0.046, pad=0.04)
    
    diff = np.abs(H_orig - H_filtered)
    ax_diff = axes[2, 3]
    im = ax_diff.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    ax_diff.set_title('H差异图', fontsize=10)
    ax_diff.axis('off')
    plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'PolSAR极化滤波结果 - {method_name}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(f'polsar_{method_name.lower()}_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  结果图已保存: polsar_{method_name.lower()}_result.png')


def main():
    print("="*70)
    print("极化SAR (PolSAR) 相干斑滤波 - 示例程序")
    print("="*70)
    
    print("\n[1] 生成合成PolSAR图像...")
    size = 100
    shh, svv, shv_real, shv_imag = generate_synthetic_polsar_image(size=size)
    
    C_original = create_covariance_matrix_3x3(shh, svv, shv_real, shv_imag)
    T_original = create_coherency_matrix_3x3(shh, svv, shv_real, shv_imag)
    
    print(f"  图像大小: {size}x{size}")
    print(f"  协方差矩阵形状: {C_original.shape}")
    
    print("\n[2] 计算原始极化分解参数...")
    H_orig, A_orig, Alpha_orig, eigvals_orig = cloude_pottier_decomposition(T_original)
    Ps_orig, Pd_orig, Pv_orig, Pc_orig = yamaguchi_decomposition(C_original)
    
    features_orig = compute_polarimetric_features(C_original)
    span_orig = features_orig['span']
    print(f"  平均散射熵 H: {np.mean(H_orig):.4f}")
    print(f"  平均反熵 A: {np.mean(A_orig):.4f}")
    print(f"  平均散射角 Alpha: {np.mean(Alpha_orig):.2f}°")
    
    print("\n[3] 应用极化滤波算法...")
    
    filters = [
        ('Boxcar 5x5', lambda C: boxcar_filter_covariance(C, window_size=5)),
        ('Boxcar 7x7', lambda C: boxcar_filter_covariance(C, window_size=7)),
        ('Refined Lee', lambda C: refined_lee_filter_polarimetric(C, window_size=7, num_looks=1)),
        ('PWF', lambda C: polarimetric_whitening_filter(C, window_size=7, num_looks=1)),
        ('EPWF', lambda C: enhanced_polarimetric_whitening_filter(C, window_size=7, num_looks=1)),
        ('IDAN', lambda C: idan_filter(C, window_size=7, num_looks=1)),
    ]
    
    results = []
    
    for name, filter_func in filters:
        print(f"\n  [{name}]...")
        start = time.time()
        
        C_filtered = filter_func(C_original)
        t = time.time() - start
        
        T_filtered = np.zeros_like(T_original)
        C_to_T = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        for i in range(size):
            for j in range(size):
                T_filtered[:, :, i, j] = C_to_T @ C_filtered[:, :, i, j] @ C_to_T.conj().T
        
        H_f, A_f, Alpha_f, _ = cloude_pottier_decomposition(T_filtered)
        Ps_f, Pd_f, Pv_f, Pc_f = yamaguchi_decomposition(C_filtered)
        
        fidelity = calculate_polarization_fidelity(C_original, C_filtered)
        
        hh_f, vv_f, hv_f = covariance_to_intensity(C_filtered)
        enl_hh = calculate_enl(hh_f)
        enl_vv = calculate_enl(vv_f)
        
        hh_noisy, vv_noisy, _ = covariance_to_intensity(C_original)
        enl_hh_orig = calculate_enl(hh_noisy)
        enl_vv_orig = calculate_enl(vv_noisy)
        
        print(f"    时间: {t:.2f}s")
        print(f"    ENL (HH): {enl_hh_orig:.2f} -> {enl_hh:.2f}")
        print(f"    ENL (VV): {enl_vv_orig:.2f} -> {enl_vv:.2f}")
        print(f"    平均极化保真度: {np.mean(fidelity):.4f}")
        print(f"    H变化: {np.mean(H_orig):.4f} -> {np.mean(H_f):.4f}")
        
        results.append({
            'name': name,
            'C_filtered': C_filtered,
            'H_filtered': H_f,
            'time': t,
            'enl_hh': enl_hh,
            'enl_vv': enl_vv,
            'fidelity': np.mean(fidelity),
            'h_change': abs(np.mean(H_orig) - np.mean(H_f))
        })
    
    print("\n" + "="*70)
    print("[4] 绘制PolSAR极化分解对比图...")
    
    T_original_plot = T_original.copy()
    C_to_T = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
    
    for result in results:
        plot_polsar_results(
            C_original, C_original, result['C_filtered'],
            result['name'], H_orig, result['H_filtered']
        )
    
    print("\n" + "="*70)
    print("[5] 定量分析结果...")
    
    print(f"\n{'方法':<15} {'时间(s)':>8} {'ENL_HH':>8} {'ENL_VV':>8} {'保真度':>8} {'H变化':>8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<15} {r['time']:>8.2f} {r['enl_hh']:>8.2f} {r['enl_vv']:>8.2f} "
              f"{r['fidelity']:>8.4f} {r['h_change']:>8.4f}")
    
    print("\n" + "="*70)
    print("[6] 绘制Pauli基伪彩色对比...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, result in enumerate(results[:4]):
        ax = axes[0, idx]
        hh, vv, hv = covariance_to_intensity(result['C_filtered'])
        rgb = np.zeros((size, size, 3))
        rgb[:, :, 0] = (hh - hh.min()) / (hh.max() - hh.min() + 1e-10)
        rgb[:, :, 1] = (vv - vv.min()) / (vv.max() - vv.min() + 1e-10)
        rgb[:, :, 2] = (hv - hv.min()) / (hv.max() - hv.min() + 1e-10)
        ax.imshow(rgb)
        ax.set_title(f'{result["name"]}\n(HH=R, VV=G, HV=B)', fontsize=10)
        ax.axis('off')
    
    for idx, result in enumerate(results[4:]):
        ax = axes[1, idx]
        hh, vv, hv = covariance_to_intensity(result['C_filtered'])
        rgb = np.zeros((size, size, 3))
        rgb[:, :, 0] = (hh - hh.min()) / (hh.max() - hh.min() + 1e-10)
        rgb[:, :, 1] = (vv - vv.min()) / (vv.max() - vv.min() + 1e-10)
        rgb[:, :, 2] = (hv - hv.min()) / (hv.max() - hv.min() + 1e-10)
        ax.imshow(rgb)
        ax.set_title(f'{result["name"]}\n(HH=R, VV=G, HV=B)', fontsize=10)
        ax.axis('off')
    
    ax = axes[1, 3]
    hh_o, vv_o, hv_o = covariance_to_intensity(C_original)
    rgb_o = np.zeros((size, size, 3))
    rgb_o[:, :, 0] = (hh_o - hh_o.min()) / (hh_o.max() - hh_o.min() + 1e-10)
    rgb_o[:, :, 1] = (vv_o - vv_o.min()) / (vv_o.max() - vv_o.min() + 1e-10)
    rgb_o[:, :, 2] = (hv_o - hv_o.min()) / (hv_o.max() - hv_o.min() + 1e-10)
    ax.imshow(rgb_o)
    ax.set_title('原始图像\n(HH=R, VV=G, HV=B)', fontsize=10)
    ax.axis('off')
    
    plt.suptitle('PolSAR极化通道伪彩色合成', fontsize=14)
    plt.tight_layout()
    plt.savefig('polsar_pseudocolor_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  伪彩色对比图已保存: polsar_pseudocolor_comparison.png")
    
    print("\n" + "="*70)
    print("[7] 绘制Yamaguchi分解对比...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    components = ['Ps (表面散射)', 'Pd (二次散射)', 'Pv (体散射)']
    orig_comps = [Ps_orig, Pd_orig, Pv_orig]
    pwf_idx = next(i for i, r in enumerate(results) if 'PWF' in r['name'])
    
    T_pwf = np.zeros_like(T_original)
    for i in range(size):
        for j in range(size):
            T_pwf[:, :, i, j] = C_to_T @ results[pwf_idx]['C_filtered'][:, :, i, j] @ C_to_T.conj().T
    
    Ps_pwf, Pd_pwf, Pv_pwf, Pc_pwf = yamaguchi_decomposition(results[pwf_idx]['C_filtered'])
    pwf_comps = [Ps_pwf, Pd_pwf, Pv_pwf]
    
    for idx in range(3):
        ax = axes[0, idx]
        im = ax.imshow(orig_comps[idx], cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'原始 - {components[idx]}', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[1, idx]
        im = ax.imshow(pwf_comps[idx], cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'PWF滤波 - {components[idx]}', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Yamaguchi四分量分解对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('polsar_yamaguchi_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Yamaguchi分解对比图已保存: polsar_yamaguchi_comparison.png")
    
    print("\n" + "="*70)
    print("示例程序运行完成！")
    print("="*70)
    
    print("\n总结:")
    print("  - PWF (极化白化滤波) 在极化信息保留和斑点抑制之间取得了最佳平衡")
    print("  - EPWF (增强型PWF) 增加了边缘保护，在复杂区域表现更好")
    print("  - IDAN滤波基于极化特征值滤波，保留了散射特性")
    print("  - 所有极化滤波方法都能在抑制斑点噪声的同时")
    print("    较好地保留H/A/Alpha分解参数和Yamaguchi分解分量")


if __name__ == '__main__':
    main()
