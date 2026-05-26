import numpy as np
import matplotlib.pyplot as plt
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
    covariance_to_intensity,
    compute_polarimetric_features
)


def main():
    print("="*70)
    print("PolSAR 极化滤波 - 使用指南")
    print("="*70)
    
    print("""
1. 生成或加载PolSAR数据
   PolSAR数据通常包含四个极化通道: HH, HV, VH, VV
   对于互易介质: HV = VH
""")
    
    size = 80
    shh, svv, shv_real, shv_imag = generate_synthetic_polsar_image(size=size)
    
    print(f"   生成的合成数据大小: {size}x{size}")
    print(f"   HH幅度范围: [{np.abs(shh).min():.1f}, {np.abs(shh).max():.1f}]")
    print(f"   VV幅度范围: [{np.abs(svv).min():.1f}, {np.abs(svv).max():.1f}]")
    
    print("""
2. 构建极化协方差矩阵和相干矩阵
   C3 (3x3 Lexel基): [[HH*HH*, HH*HV*, HH*VV*], ...]
   T3 (3x3 Pauli基): 基于Pauli散射矢量构建
""")
    
    C = create_covariance_matrix_3x3(shh, svv, shv_real, shv_imag)
    T = create_coherency_matrix_3x3(shh, svv, shv_real, shv_imag)
    
    print(f"   协方差矩阵 C3 形状: {C.shape}")
    print(f"   相干矩阵 T3 形状: {T.shape}")
    
    print("""
3. 计算极化分解参数
   H/A/Alpha分解: 分析散射机制
   Yamaguchi分解: 分离不同散射类型
""")
    
    H, A, Alpha, eigvals = cloude_pottier_decomposition(T)
    Ps, Pd, Pv, Pc = yamaguchi_decomposition(C)
    
    print(f"   散射熵 H: 均值={np.mean(H):.4f}, 范围=[{np.min(H):.4f}, {np.max(H):.4f}]")
    print(f"   反熵 A:   均值={np.mean(A):.4f}")
    print(f"   散射角 Alpha: 均值={np.mean(Alpha):.2f}度")
    print(f"   表面散射 Ps: 均值={np.mean(Ps):.4f}")
    print(f"   二次散射 Pd: 均值={np.mean(Pd):.4f}")
    print(f"   体散射 Pv:   均值={np.mean(Pv):.4f}")
    
    print("""
4. 应用极化滤波算法
   目标: 抑制斑点噪声，同时保留极化信息
""")
    
    print("   [4.1] Boxcar滤波 - 简单滑动平均")
    C_boxcar = boxcar_filter_covariance(C, window_size=7)
    
    print("   [4.2] 极化Refined Lee滤波")
    C_refined = refined_lee_filter_polarimetric(C, window_size=7, num_looks=1)
    
    print("   [4.3] 极化白化滤波(PWF) - 经典算法")
    C_pwf = polarimetric_whitening_filter(C, window_size=7, num_looks=1)
    
    print("   [4.4] 增强型PWF (EPWF)")
    C_epwf = enhanced_polarimetric_whitening_filter(C, window_size=7, num_looks=1)
    
    print("   [4.5] IDAN滤波")
    C_idan = idan_filter(C, window_size=7, num_looks=1)
    
    print("""
5. 评估滤波效果
   使用ENL衡量去噪效果
   使用极化保真度衡量极化信息保留程度
""")
    
    def calculate_enl(img):
        mean_val = np.mean(img)
        std_val = np.std(img)
        return (mean_val ** 2) / (std_val ** 2) if std_val > 0 else 0
    
    hh_orig, vv_orig, hv_orig = covariance_to_intensity(C)
    
    results = {
        'Original': C,
        'Boxcar': C_boxcar,
        'Refined Lee': C_refined,
        'PWF': C_pwf,
        'EPWF': C_epwf,
        'IDAN': C_idan
    }
    
    print(f"   {'Method':<15} {'ENL_HH':>10} {'ENL_VV':>10} {'Fidelity':>10} {'H_change':>10}")
    print("   " + "-"*55)
    
    H_orig_mean = np.mean(H)
    
    for name, C_filt in results.items():
        hh_f, vv_f, hv_f = covariance_to_intensity(C_filt)
        fidelity = calculate_polarization_fidelity(C, C_filt)
        
        T_filt = np.zeros_like(T)
        C_to_T = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        for i in range(size):
            for j in range(size):
                T_filt[:, :, i, j] = C_to_T @ C_filt[:, :, i, j] @ C_to_T.conj().T
        
        H_f, _, _, _ = cloude_pottier_decomposition(T_filt)
        
        line = "   %-15s %10.4f %10.4f %10.4f %10.4f" % (
            name,
            calculate_enl(hh_f),
            calculate_enl(vv_f),
            np.mean(fidelity),
            abs(H_orig_mean - np.mean(H_f))
        )
        print(line)
    
    print("""
6. 可视化极化分解结果
""")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    components = [
        ('Entropy H', H, 'jet'),
        ('Anisotropy A', A, 'jet'),
        ('Alpha Angle', Alpha, 'jet'),
        ('Surface Scatter Ps', Ps, 'jet'),
        ('Double Scatter Pd', Pd, 'jet'),
        ('Volume Scatter Pv', Pv, 'jet'),
        ('Helix Scatter Pc', Pc, 'jet'),
        ('Span', np.real(np.trace(C, axis1=0, axis2=1)), 'gray'),
    ]
    
    for idx, (title, data, cmap) in enumerate(components):
        ax = axes[idx // 4, idx % 4]
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('PolSAR Polarimetric Decomposition Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('polsar_decomposition_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   分解结果可视化图已保存: polsar_decomposition_visualization.png")
    
    print("""
7. 滤波前后极化参数对比
""")
    
    T_boxcar = np.zeros_like(T)
    T_pwf = np.zeros_like(T)
    
    for i in range(size):
        for j in range(size):
            T_boxcar[:, :, i, j] = C_to_T @ C_boxcar[:, :, i, j] @ C_to_T.conj().T
            T_pwf[:, :, i, j] = C_to_T @ C_pwf[:, :, i, j] @ C_to_T.conj().T
    
    H_boxcar, _, _, _ = cloude_pottier_decomposition(T_boxcar)
    H_pwf, _, _, _ = cloude_pottier_decomposition(T_pwf)
    
    Ps_boxcar, Pd_boxcar, Pv_boxcar, Pc_boxcar = yamaguchi_decomposition(C_boxcar)
    Ps_pwf, Pd_pwf, Pv_pwf, Pc_pwf = yamaguchi_decomposition(C_pwf)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    data_to_plot = [
        (H, 'Original H', 'jet'),
        (H_boxcar, 'Boxcar H', 'jet'),
        (H_pwf, 'PWF H', 'jet'),
        (Alpha, 'Original Alpha', 'jet'),
        (Ps, 'Original Ps', 'jet'),
        (Ps_boxcar, 'Boxcar Ps', 'jet'),
        (Ps_pwf, 'PWF Ps', 'jet'),
        (Pd, 'Original Pd', 'jet'),
        (Pv, 'Original Pv', 'jet'),
        (Pc, 'Original Pc', 'jet'),
        (np.abs(H - H_pwf), 'H Diff (Orig-PWF)', 'hot'),
        (np.abs(Ps - Ps_pwf), 'Ps Diff (Orig-PWF)', 'hot'),
    ]
    
    for idx, (data, title, cmap) in enumerate(data_to_plot):
        ax = axes[idx // 4, idx % 4]
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Polarimetric Parameters: Original vs Filtered', fontsize=14)
    plt.tight_layout()
    plt.savefig('polsar_filter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   滤波对比图已保存: polsar_filter_comparison.png")
    
    print("""
8. 实际应用建议
""")
    
    print("""
   选择滤波算法的建议:
   - 数据量小、追求速度: Boxcar滤波 (简单快速)
   - 要求平衡去噪和极化信息保留: PWF (极化白化滤波)
   - 需要保护边缘结构: EPWF (增强型PWF) 或 IDAN
   - 极化信息保真优先: Refined Lee或IDAN
   
   参数选择建议:
   - 窗口大小: 通常选择5-11，根据噪声水平调整
   - 视数(num_looks): 根据SAR系统参数设置，单视数据用1
   - 边缘保护: EPWF提供自适应边缘保护
   
   评估指标:
   - ENL (等效视数): 衡量斑点抑制效果，越大越好
   - 极化保真度: 衡量极化信息保留，越接近1越好
   - H/A/Alpha变化: 分解参数变化越小越好
""")
    
    print("="*70)
    print("PolSAR极化滤波模块使用指南完成！")
    print("="*70)


if __name__ == '__main__':
    main()
