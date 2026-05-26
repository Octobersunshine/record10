import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from sar_speckle_filter import (lee_filter, enhanced_lee_filter, 
                                   adaptive_window_lee_filter, sar_nlm_filter, 
                                   fast_sar_nlm_filter, calculate_enl)


def create_synthetic_sar_image(size=200):
    """
    创建带斑点噪声的合成SAR图像用于演示
    """
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    img = np.zeros((size, size))
    
    center1, radius1 = (0.3, 0.3), 0.15
    center2, radius2 = (0.7, 0.6), 0.2
    img[(x - center1[0])**2 + (y - center1[1])**2 < radius1**2] = 150
    img[(x - center2[0])**2 + (y - center2[1])**2 < radius2**2] = 200
    
    img[50:80, 120:160] = 100
    img[120:160, 50:80] = 180
    
    img[img == 0] = 50
    
    speckle = np.random.rayleigh(1, img.shape)
    img_speckled = img * speckle
    
    img_speckled = np.clip(img_speckled, 0, 255)
    
    return img, img_speckled


def calculate_ssim(img1, img2):
    """
    计算结构相似性指数(SSIM) - 用于评估图像质量
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = _gaussian_kernel(11, 1.5)
    
    mu1 = _conv2d(img1, kernel)
    mu2 = _conv2d(img2, kernel)
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = _conv2d(img1**2, kernel) - mu1_sq
    sigma2_sq = _conv2d(img2**2, kernel) - mu2_sq
    sigma12 = _conv2d(img1 * img2, kernel) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def _gaussian_kernel(size, sigma):
    """生成高斯核"""
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    kernel = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _conv2d(img, kernel):
    """简单的二维卷积"""
    from scipy import ndimage
    return ndimage.convolve(img, kernel, mode='reflect')


def demo_comparison():
    """
    演示所有滤波方法的对比
    """
    print("创建合成SAR图像...")
    original, speckled = create_synthetic_sar_image(150)
    
    print(f"原始无噪图像ENL: {calculate_enl(original):.4f}")
    print(f"加噪后图像ENL: {calculate_enl(speckled):.4f}")
    print(f"加噪后SSIM: {calculate_ssim(original, speckled):.4f}")
    
    results = []
    
    print("\n=== 1. Lee滤波 (窗口大小=5) ===")
    start = time.time()
    lee_5 = lee_filter(speckled, window_size=5)
    t_lee = time.time() - start
    enl_lee = calculate_enl(lee_5)
    ssim_lee = calculate_ssim(original, lee_5)
    print(f"ENL: {enl_lee:.4f}, SSIM: {ssim_lee:.4f}, 时间: {t_lee:.2f}s")
    results.append(('Lee (5x5)', lee_5, enl_lee, ssim_lee, t_lee))
    
    print("\n=== 2. Enhanced Lee滤波 (窗口大小=7, 视数=1) ===")
    start = time.time()
    enhanced_lee = enhanced_lee_filter(speckled, window_size=7, num_looks=1)
    t_enhanced = time.time() - start
    enl_enhanced = calculate_enl(enhanced_lee)
    ssim_enhanced = calculate_ssim(original, enhanced_lee)
    print(f"ENL: {enl_enhanced:.4f}, SSIM: {ssim_enhanced:.4f}, 时间: {t_enhanced:.2f}s")
    results.append(('Enhanced Lee (7x7)', enhanced_lee, enl_enhanced, ssim_enhanced, t_enhanced))
    
    print("\n=== 3. 自适应窗口Lee滤波 (3-11, 视数=1) ===")
    start = time.time()
    adaptive_lee = adaptive_window_lee_filter(speckled, min_window=3, max_window=11, num_looks=1)
    t_adaptive = time.time() - start
    enl_adaptive = calculate_enl(adaptive_lee)
    ssim_adaptive = calculate_ssim(original, adaptive_lee)
    print(f"ENL: {enl_adaptive:.4f}, SSIM: {ssim_adaptive:.4f}, 时间: {t_adaptive:.2f}s")
    results.append(('Adaptive Lee (3-11)', adaptive_lee, enl_adaptive, ssim_adaptive, t_adaptive))
    
    print("\n=== 4. 快速NLM滤波 (窗口=5, 搜索=15, h=15) ===")
    start = time.time()
    fast_nlm = fast_sar_nlm_filter(speckled, window_size=5, search_window=15, h=15.0, num_looks=1)
    t_fast_nlm = time.time() - start
    enl_fast_nlm = calculate_enl(fast_nlm)
    ssim_fast_nlm = calculate_ssim(original, fast_nlm)
    print(f"ENL: {enl_fast_nlm:.4f}, SSIM: {ssim_fast_nlm:.4f}, 时间: {t_fast_nlm:.2f}s")
    results.append(('Fast NLM', fast_nlm, enl_fast_nlm, ssim_fast_nlm, t_fast_nlm))
    
    print("\n=== 5. 完整NLM滤波 (窗口=7, 搜索=21, h=10) ===")
    start = time.time()
    full_nlm = sar_nlm_filter(speckled, window_size=7, search_window=21, h=10.0, num_looks=1)
    t_full_nlm = time.time() - start
    enl_full_nlm = calculate_enl(full_nlm)
    ssim_full_nlm = calculate_ssim(original, full_nlm)
    print(f"ENL: {enl_full_nlm:.4f}, SSIM: {ssim_full_nlm:.4f}, 时间: {t_full_nlm:.2f}s")
    results.append(('Full NLM', full_nlm, enl_full_nlm, ssim_full_nlm, t_full_nlm))
    
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title('原始无噪图像', fontsize=10)
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(speckled, cmap='gray')
    plt.title(f'加噪图像\nENL={calculate_enl(speckled):.2f}', fontsize=10)
    plt.axis('off')
    
    for idx, (name, img, enl, ssim, t) in enumerate(results):
        plt.subplot(2, 4, idx + 3)
        plt.imshow(img, cmap='gray')
        plt.title(f'{name}\nENL={enl:.2f}, SSIM={ssim:.3f}', fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('all_filter_comparison.png', dpi=150)
    print("\n完整对比图已保存为: all_filter_comparison.png")
    plt.show()
    
    plot_enhanced_comparison(original, speckled, lee_5, adaptive_lee, fast_nlm)
    
    print_quantitative_comparison(results, original, speckled)


def plot_enhanced_comparison(original, speckled, lee, adaptive, nlm):
    """
    绘制增强对比图，突出显示均匀区域的改进
    """
    plt.figure(figsize=(15, 10))
    
    uniform_region = (slice(10, 40), slice(10, 40))
    
    plt.subplot(2, 3, 1)
    plt.imshow(speckled[uniform_region], cmap='gray', vmin=0, vmax=255)
    plt.title('加噪图像均匀区域', fontsize=11)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(lee[uniform_region], cmap='gray', vmin=0, vmax=255)
    plt.title(f'Lee滤波均匀区域\nENL={calculate_enl(lee[uniform_region]):.2f}', fontsize=11)
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(adaptive[uniform_region], cmap='gray', vmin=0, vmax=255)
    plt.title(f'自适应Lee均匀区域\nENL={calculate_enl(adaptive[uniform_region]):.2f}', fontsize=11)
    plt.axis('off')
    
    edge_region = (slice(30, 80), slice(120, 170))
    
    plt.subplot(2, 3, 4)
    plt.imshow(speckled[edge_region], cmap='gray', vmin=0, vmax=255)
    plt.title('加噪图像边缘区域', fontsize=11)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(adaptive[edge_region], cmap='gray', vmin=0, vmax=255)
    plt.title('自适应Lee边缘区域\n(保持边缘)', fontsize=11)
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(nlm[edge_region], cmap='gray', vmin=0, vmax=255)
    plt.title('NLM边缘区域\n(保持细节)', fontsize=11)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('region_comparison.png', dpi=150)
    print("区域对比图已保存为: region_comparison.png")
    plt.show()


def print_quantitative_comparison(results, original, speckled):
    """
    打印定量对比分析
    """
    print("\n" + "="*80)
    print("定量对比分析")
    print("="*80)
    print(f"{'方法':<25} {'ENL':>10} {'SSIM':>10} {'时间(s)':>10} {'ENL提升':>10}")
    print("-" * 65)
    
    base_enl = calculate_enl(speckled)
    print(f"{'原始加噪图像':<25} {base_enl:>10.4f} {calculate_ssim(original, speckled):>10.4f} {'-':>10} {'-':>10}")
    
    for name, img, enl, ssim, t in results:
        improvement = ((enl - base_enl) / base_enl) * 100
        print(f"{name:<25} {enl:>10.4f} {ssim:>10.4f} {t:>10.2f} {improvement:>9.1f}%")
    
    print("="*80)
    print("\n算法说明：")
    print("  - Lee滤波: 固定窗口，均匀区域欠平滑")
    print("  - Enhanced Lee: 三区域模型，优于基础Lee")
    print("  - Adaptive Lee: 自适应窗口，均匀区用大窗口，边缘区用小窗口")
    print("  - Fast NLM: 预分类+步长采样加速的非局部均值")
    print("  - Full NLM: 完整非局部均值，效果最佳但计算较慢")


def adaptive_window_visualization():
    """
    可视化自适应窗口的选择
    """
    print("\n\n=== 自适应窗口选择可视化 ===")
    original, speckled = create_synthetic_sar_image(150)
    
    window_map = np.zeros_like(speckled)
    min_window, max_window = 3, 11
    num_looks = 1
    
    window_sizes = list(range(min_window, max_window + 1, 2))
    pad = max_window // 2
    img_padded = np.pad(speckled, pad, mode='reflect')
    cu = np.sqrt(2.0 / num_looks)
    
    for i in range(speckled.shape[0]):
        for j in range(speckled.shape[1]):
            adaptive_ws = min_window
            for ws in window_sizes:
                half = ws // 2
                window = img_padded[i+pad-half:i+pad+half+1, 
                                   j+pad-half:j+pad+half+1]
                mean_win = np.mean(window)
                std_win = np.std(window)
                ci = std_win / (mean_win + 1e-10)
                if ci <= cu:
                    adaptive_ws = ws
                else:
                    break
            window_map[i, j] = adaptive_ws
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(speckled, cmap='gray')
    plt.title('原始SAR图像', fontsize=11)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    cmap = plt.cm.get_cmap('viridis', len(window_sizes))
    im = plt.imshow(window_map, cmap=cmap, vmin=min_window, vmax=max_window)
    plt.title('自适应窗口大小分布', fontsize=11)
    plt.axis('off')
    cbar = plt.colorbar(im, ticks=window_sizes)
    cbar.set_label('窗口大小')
    
    plt.subplot(1, 3, 3)
    filtered = adaptive_window_lee_filter(speckled, min_window, max_window, num_looks)
    plt.imshow(filtered, cmap='gray')
    plt.title('自适应Lee滤波结果', fontsize=11)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('adaptive_window_map.png', dpi=150)
    print("自适应窗口分布图已保存为: adaptive_window_map.png")
    plt.show()


if __name__ == '__main__':
    print("="*80)
    print("SAR图像相干斑抑制 - 增强版示例程序")
    print("="*80)
    print("包含算法：Lee滤波、Enhanced Lee、自适应窗口Lee、NLM、快速NLM")
    print("="*80)
    
    try:
        demo_comparison()
    except Exception as e:
        print(f"完整对比演示出错: {e}")
    
    try:
        adaptive_window_visualization()
    except Exception as e:
        print(f"自适应窗口可视化出错: {e}")
    
    print("\n" + "="*80)
    print("示例运行完成！")
    print("="*80)
