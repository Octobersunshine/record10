"""
高光谱端元提取 - 使用示例（增强版）
"""
import numpy as np
from hyperspectral_endmembers import (
    extract_endmembers, generate_synthetic_data,
    mnf_denoise, pca_denoise, hfc_endmember_count
)


def example_basic():
    """
    示例1: 基础端元提取（手动指定端元数）
    """
    print("=" * 60)
    print("示例1: 基础端元提取（手动指定端元数）")
    print("=" * 60)

    n_bands = 224
    n_pixels = 5000
    n_endmembers = 6

    print(f"\n生成模拟数据: {n_bands} 波段, {n_pixels} 像素, {n_endmembers} 端元")
    data, E_true, _ = generate_synthetic_data(n_pixels, n_bands, n_endmembers, noise_level=0.02)
    print(f"数据形状: {data.shape}")

    print("\n使用 VCA 提取端元...")
    E_vca, idx_vca, info_vca = extract_endmembers(data, n_endmembers, method='vca')
    print(f"VCA 端元矩阵形状: {E_vca.shape}")
    print(f"VCA 端元索引: {idx_vca}")
    print(f"info 字典键: {list(info_vca.keys())}")

    print("\n使用 OSP 提取端元...")
    E_osp, idx_osp, info_osp = extract_endmembers(data, n_endmembers, method='osp')
    print(f"OSP 端元矩阵形状: {E_osp.shape}")
    print(f"OSP 端元索引: {idx_osp}")

    return E_vca, E_osp


def example_denoise():
    """
    示例2: 降噪预处理功能（MNF和PCA）
    """
    print("\n" + "=" * 60)
    print("示例2: 降噪预处理（高噪声数据）")
    print("=" * 60)

    n_bands = 180
    n_pixels = 3000
    n_endmembers = 5

    print(f"\n生成高噪声数据 (噪声水平: 0.15)...")
    data, E_true, _ = generate_synthetic_data(n_pixels, n_bands, n_endmembers, noise_level=0.15, seed=999)
    print(f"数据形状: {data.shape}")
    print(f"真实端元数: {n_endmembers}")

    print("\n无降噪 VCA:")
    E_raw, idx_raw, info_raw = extract_endmembers(data, n_endmembers, method='vca')
    sim_raw = np.mean([np.corrcoef(E_true[:, i], E_raw[:, i])[0, 1] for i in range(n_endmembers)])
    print(f"  平均光谱相似度: {sim_raw:.4f}")

    print("\nPCA 降噪 + VCA (保留99%方差):")
    E_pca, idx_pca, info_pca = extract_endmembers(
        data, n_endmembers, method='vca',
        denoise='pca', denoise_variance_ratio=0.99
    )
    print(f"  保留主成分数: {info_pca['denoise_components']}")
    sim_pca = np.mean([np.corrcoef(E_true[:, i], E_pca[:, i])[0, 1] for i in range(n_endmembers)])
    print(f"  平均光谱相似度: {sim_pca:.4f}")

    print("\nMNF 降噪 + VCA (保留99%方差):")
    E_mnf, idx_mnf, info_mnf = extract_endmembers(
        data, n_endmembers, method='vca',
        denoise='mnf', denoise_variance_ratio=0.99
    )
    print(f"  保留MNF成分数: {info_mnf['denoise_components']}")
    sim_mnf = np.mean([np.corrcoef(E_true[:, i], E_mnf[:, i])[0, 1] for i in range(n_endmembers)])
    print(f"  平均光谱相似度: {sim_mnf:.4f}")

    print("\n直接调用降噪函数示例:")
    denoised_mnf, eigvals_mnf, n_comp_mnf = mnf_denoise(data, variance_ratio=0.95)
    print(f"  MNF 降噪结果: {denoised_mnf.shape}, 保留 {n_comp_mnf} 成分")

    denoised_pca, eigvals_pca, n_comp_pca = pca_denoise(data, variance_ratio=0.95)
    print(f"  PCA 降噪结果: {denoised_pca.shape}, 保留 {n_comp_pca} 成分")

    return info_mnf


def example_auto_estimate():
    """
    示例3: HFC方法自动估计端元数量
    """
    print("\n" + "=" * 60)
    print("示例3: HFC方法自动估计端元数量")
    print("=" * 60)

    n_bands = 200
    n_pixels = 4000
    true_p = 7

    print(f"\n生成模拟数据: 真实端元数 = {true_p}")
    data, E_true, _ = generate_synthetic_data(n_pixels, n_bands, true_p, noise_level=0.03, seed=456)
    print(f"数据形状: {data.shape}")

    print("\n使用不同显著性水平估计端元数:")
    for alpha in [0.90, 0.95, 0.99, 0.999]:
        p_est, eigvals, thresholds = hfc_endmember_count(data, alpha=alpha, max_p=20)
        print(f"  alpha={alpha}: 估计端元数 = {p_est}, 真实 = {true_p}")

    print("\n完整流程: PCA降噪 + HFC自动估计 + VCA提取:")
    E_auto, idx_auto, info_auto = extract_endmembers(
        data, p=None, method='vca',
        denoise='pca', denoise_variance_ratio=0.99,
        hfc_alpha=0.99, hfc_improved=True
    )
    print(f"  PCA保留成分数: {info_auto['denoise_components']}")
    print(f"  HFC估计端元数: {info_auto['p_hfc']}")
    print(f"  实际提取端元数: {info_auto['p_estimated']}")
    print(f"  端元矩阵形状: {E_auto.shape}")

    if info_auto['p_estimated'] > 0:
        n_compare = min(true_p, info_auto['p_estimated'])
        sim_auto = np.mean([np.corrcoef(E_true[:, i], E_auto[:, i])[0, 1] for i in range(n_compare)])
        print(f"  前{n_compare}个端元平均相似度: {sim_auto:.4f}")

    return info_auto


def example_3d_with_denoise():
    """
    示例4: 三维高光谱图像 + 降噪 + 自动估计
    """
    print("\n" + "=" * 60)
    print("示例4: 三维高光谱图像 + 降噪 + 自动估计")
    print("=" * 60)

    n_rows = 80
    n_cols = 80
    n_bands = 150
    n_endmembers = 4

    print(f"\n生成模拟三维数据: {n_rows}x{n_cols}x{n_bands}, 端元数={n_endmembers}")
    data_3d = np.random.rand(n_rows, n_cols, n_bands) * 0.1

    for i in range(n_endmembers):
        peak = np.exp(-(np.arange(n_bands) - (i + 1) * n_bands / (n_endmembers + 1)) ** 2 / 50)
        for r in range(n_rows):
            for c in range(n_cols):
                in_region = False
                if i == 0 and r < n_rows // 2 and c < n_cols // 2:
                    in_region = True
                elif i == 1 and r < n_rows // 2 and c >= n_cols // 2:
                    in_region = True
                elif i == 2 and r >= n_rows // 2 and c < n_cols // 2:
                    in_region = True
                elif i == 3 and r >= n_rows // 2 and c >= n_cols // 2:
                    in_region = True

                if in_region:
                    data_3d[r, c, :] = peak + 0.05 * np.random.randn(n_bands)

    data_3d = np.clip(data_3d, 0, None)
    print(f"三维数据形状: {data_3d.shape}")

    print("\nPCA降噪 + HFC自动估计 + VCA提取:")
    E, idx, info = extract_endmembers(
        data_3d, p=None, method='vca',
        denoise='pca', denoise_variance_ratio=0.98,
        hfc_alpha=0.99
    )
    print(f"  保留主成分数: {info['denoise_components']}")
    print(f"  HFC估计端元数: {info['p_hfc']}")
    print(f"  实际提取端元数: {info['p_estimated']}")
    print(f"  端元矩阵形状: {E.shape}")

    row_idx = idx // n_cols
    col_idx = idx % n_cols
    print(f"\n端元空间位置 (行, 列):")
    for i, (r, c) in enumerate(zip(row_idx, col_idx)):
        region = ""
        if r < n_rows // 2 and c < n_cols // 2:
            region = "区域1 (左上)"
        elif r < n_rows // 2 and c >= n_cols // 2:
            region = "区域2 (右上)"
        elif r >= n_rows // 2 and c < n_cols // 2:
            region = "区域3 (左下)"
        else:
            region = "区域4 (右下)"
        print(f"  端元 {i+1}: ({r}, {c}) - {region}")

    return E, idx, info


def compare_all_methods():
    """
    示例5: 全面对比所有组合
    """
    print("\n" + "=" * 60)
    print("示例5: 各方法组合对比（高噪声数据）")
    print("=" * 60)

    n_bands = 160
    n_pixels = 2500
    n_endmembers = 5

    print(f"\n生成高噪声数据: 噪声水平=0.12, 端元数={n_endmembers}")
    data, E_true, _ = generate_synthetic_data(n_pixels, n_bands, n_endmembers, noise_level=0.12, seed=789)

    methods = [
        ('VCA (无降噪)', 'vca', None),
        ('VCA + PCA', 'vca', 'pca'),
        ('VCA + MNF', 'vca', 'mnf'),
        ('OSP (无降噪)', 'osp', None),
        ('OSP + PCA', 'osp', 'pca'),
        ('OSP + MNF', 'osp', 'mnf'),
    ]

    print("\n" + "-" * 60)
    print(f"{'方法':<20} {'p估计':<8} {'平均相似度':<12} {'成分数':<10}")
    print("-" * 60)

    for name, method, denoise in methods:
        E, idx, info = extract_endmembers(
            data, p=n_endmembers, method=method,
            denoise=denoise, denoise_variance_ratio=0.99
        )
        sim = np.mean([np.corrcoef(E_true[:, i], E[:, i])[0, 1] for i in range(n_endmembers)])
        n_comp = info.get('denoise_components', '-')
        print(f"{name:<20} {info['p_estimated']:<8} {sim:<12.4f} {str(n_comp):<10}")

    print("-" * 60)
    print("\nMNF通常比PCA更适合高光谱数据，因为它考虑了噪声结构")
    print("降噪可以显著提高高噪声情况下的端元提取精度")


if __name__ == "__main__":
    example_basic()
    example_denoise()
    example_auto_estimate()
    example_3d_with_denoise()
    compare_all_methods()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)
    print("\n核心API:")
    print("  extract_endmembers(data, p=None, method='vca',")
    print("                    denoise=None, denoise_variance_ratio=0.99,")
    print("                    hfc_alpha=0.999)")
    print("\n返回值:")
    print("  E: 端元光谱矩阵 (n_bands, n_endmembers)")
    print("     - 每一列对应一个端元的光谱曲线")
    print("  indices: 端元在原始数据中的像素索引")
    print("  info: 额外信息字典（p_estimated, p_hfc, denoise_method等）")
    print("=" * 60)
