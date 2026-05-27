"""
基线校正功能测试 - 验证基线漂移和荧光背景问题的解决
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing import (
    remove_baseline, remove_fluorescence_background,
    smooth_spectrum, normalize_spectrum, preprocess_pipeline
)
from main import RamanIdentifier, generate_example_spectra


def generate_spectrum_with_fluorescence(true_material: str = '葡萄糖',
                                       fluorescence_strength: float = 0.5,
                                       noise_level: float = 0.05):
    """
    生成带有荧光背景的光谱
    
    Args:
        true_material: 真实物质名称
        fluorescence_strength: 荧光背景强度
        noise_level: 噪声水平
    
    Returns:
        (波长数组, 原始强度, 带荧光的强度)
    """
    spectra = generate_example_spectra()
    
    if true_material not in spectra:
        raise ValueError(f"未知物质: {true_material}")
    
    wl, inten = spectra[true_material]
    
    x = np.linspace(0, 1, len(wl))
    fluorescence = fluorescence_strength * (
        0.3 + 0.5 * x + 0.2 * np.sin(3 * np.pi * x)
    )
    
    noise = np.random.normal(0, noise_level, len(wl))
    
    inten_with_fluorescence = inten + fluorescence + noise
    
    return wl, inten, inten_with_fluorescence


def test_baseline_methods():
    """测试不同基线校正方法的效果"""
    print("=" * 70)
    print("测试不同基线校正方法对荧光背景的处理效果")
    print("=" * 70)
    
    wl, clean_inten, noisy_inten = generate_spectrum_with_fluorescence(
        fluorescence_strength=0.8, noise_level=0.08
    )
    
    methods = ['polyfit', 'als', 'airpls', 'iarpls', 'median']
    
    results = {}
    for method in methods:
        try:
            if method == 'polyfit':
                corrected, baseline = remove_baseline(noisy_inten, method, degree=4)
            elif method == 'median':
                corrected, baseline = remove_baseline(noisy_inten, method, window_size=31)
            else:
                corrected, baseline = remove_baseline(noisy_inten, method, lam=1e5)
            
            correlation = np.corrcoef(corrected, clean_inten)[0, 1]
            rmse = np.sqrt(np.mean((corrected - clean_inten) ** 2))
            
            results[method] = {
                'corrected': corrected,
                'baseline': baseline,
                'correlation': correlation,
                'rmse': rmse
            }
            
            print(f"\n{method:12s}: 相关系数={correlation:.4f}, RMSE={rmse:.4f}")
        except Exception as e:
            print(f"\n{method:12s}: 失败 - {e}")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    axes[0, 0].plot(wl, clean_inten, 'b-', label='原始纯净光谱')
    axes[0, 0].set_title('原始纯净光谱')
    axes[0, 0].set_xlabel('拉曼位移 (cm⁻¹)')
    axes[0, 0].set_ylabel('强度')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(wl, noisy_inten, 'r-', label='带荧光背景的光谱')
    axes[0, 1].set_title('带荧光背景的光谱')
    axes[0, 1].set_xlabel('拉曼位移 (cm⁻¹)')
    axes[0, 1].set_ylabel('强度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for i, (method, data) in enumerate(results.items()):
        idx = i + 2
        row = idx // 2
        col = idx % 2
        if row >= axes.shape[0] or col >= axes.shape[1]:
            continue
        axes[row, col].plot(wl, clean_inten, 'b--', alpha=0.5, label='原始')
        axes[row, col].plot(wl, data['corrected'], 'r-', label='校正后')
        axes[row, col].plot(wl, data['baseline'], 'g-', alpha=0.7, label='估计基线')
        axes[row, col].set_title(f'{method} (相关系数={data["correlation"]:.3f})')
        axes[row, col].set_xlabel('拉曼位移 (cm⁻¹)')
        axes[row, col].set_ylabel('强度')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_correction_comparison.png', dpi=150)
    print(f"\n结果图已保存到: baseline_correction_comparison.png")
    plt.close()
    
    return results


def test_matching_with_fluorescence():
    """测试荧光背景下的物质识别"""
    print("\n" + "=" * 70)
    print("测试荧光背景下的物质识别性能")
    print("=" * 70)
    
    identifier = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    true_materials = ['葡萄糖', '乙醇', '蔗糖']
    fluorescence_levels = [0.2, 0.5, 0.8, 1.2]
    
    results = []
    
    for material in true_materials:
        for fluo_level in fluorescence_levels:
            wl, _, noisy_inten = generate_spectrum_with_fluorescence(
                true_material=material,
                fluorescence_strength=fluo_level,
                noise_level=0.05
            )
            
            result = identifier.identify(
                wl, noisy_inten,
                method='combined',
                similarity_method='cosine',
                threshold=0.7
            )
            
            correct = result['identified_material'] == material
            results.append({
                'material': material,
                'fluorescence': fluo_level,
                'identified': result['identified_material'],
                'score': result['best_score'],
                'correct': correct,
                'reliable': result['reliable']
            })
            
            status = "✓" if correct else "✗"
            print(f"  {material:8s} (荧光={fluo_level:4.1f}): "
                  f"识别={result['identified_material']:8s} "
                  f"分数={result['best_score']:.4f} {status}")
    
    print("\n" + "-" * 70)
    accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"总体准确率: {accuracy:.2%}")
    
    for material in true_materials:
        mat_results = [r for r in results if r['material'] == material]
        mat_acc = sum(r['correct'] for r in mat_results) / len(mat_results)
        print(f"  {material:8s}: {mat_acc:.2%}")
    
    return results


def test_normalization_methods():
    """测试不同归一化方法的效果"""
    print("\n" + "=" * 70)
    print("测试不同归一化方法对匹配的影响")
    print("=" * 70)
    
    wl, clean_inten, noisy_inten = generate_spectrum_with_fluorescence(
        fluorescence_strength=0.5, noise_level=0.05
    )
    
    corrected, _ = remove_baseline(noisy_inten, 'airpls', lam=1e5)
    
    norm_methods = ['minmax', 'zscore', 'l2']
    
    for method in norm_methods:
        normed = normalize_spectrum(corrected, method)
        correlation = np.corrcoef(normed, normalize_spectrum(clean_inten, method))[0, 1]
        print(f"  {method:8s}: 归一化后相关系数={correlation:.4f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, method in enumerate(norm_methods):
        axes[i].plot(wl, normalize_spectrum(clean_inten, method), 'b-', label='原始', alpha=0.7)
        axes[i].plot(wl, normalize_spectrum(corrected, method), 'r--', label='校正后', alpha=0.7)
        axes[i].set_title(f'{method} 归一化')
        axes[i].set_xlabel('拉曼位移 (cm⁻¹)')
        axes[i].set_ylabel('归一化强度')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150)
    print(f"\n结果图已保存到: normalization_comparison.png")
    plt.close()


def test_full_pipeline():
    """测试完整预处理流水线"""
    print("\n" + "=" * 70)
    print("完整预处理流水线测试")
    print("=" * 70)
    
    wl, clean_inten, noisy_inten = generate_spectrum_with_fluorescence(
        fluorescence_strength=0.8, noise_level=0.08
    )
    
    pipelines = [
        {
            'name': '平滑 + AirPLS + MinMax',
            'smooth': True,
            'smooth_method': 'savgol',
            'smooth_kwargs': {'window_length': 7, 'polyorder': 2},
            'baseline': True,
            'baseline_method': 'airpls',
            'baseline_kwargs': {'lam': 1e5},
            'normalize': True,
            'normalize_method': 'minmax'
        },
        {
            'name': '平滑 + IarPLS + MinMax',
            'smooth': True,
            'smooth_method': 'savgol',
            'smooth_kwargs': {'window_length': 7, 'polyorder': 2},
            'baseline': True,
            'baseline_method': 'iarpls',
            'baseline_kwargs': {'lam': 1e5},
            'normalize': True,
            'normalize_method': 'minmax'
        },
        {
            'name': '平滑 + 多项式(4阶) + MinMax',
            'smooth': True,
            'smooth_method': 'savgol',
            'smooth_kwargs': {'window_length': 7, 'polyorder': 2},
            'baseline': True,
            'baseline_method': 'polyfit',
            'baseline_kwargs': {'degree': 4},
            'normalize': True,
            'normalize_method': 'minmax'
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(wl, clean_inten, 'b-', label='原始纯净光谱')
    axes[0, 0].plot(wl, noisy_inten, 'r-', label='带荧光的光谱', alpha=0.7)
    axes[0, 0].set_title('原始光谱 vs 带荧光的光谱')
    axes[0, 0].set_xlabel('拉曼位移 (cm⁻¹)')
    axes[0, 0].set_ylabel('强度')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for i, pipeline in enumerate(pipelines):
        row = (i + 1) // 2
        col = (i + 1) % 2
        
        pipeline_copy = {k: v for k, v in pipeline.items() if k != 'name'}
        _, processed, info = preprocess_pipeline(wl, noisy_inten, **pipeline_copy)
        clean_processed, _, _ = preprocess_pipeline(wl, clean_inten, **pipeline_copy)
        
        correlation = np.corrcoef(processed, clean_processed)[0, 1]
        
        axes[row, col].plot(wl, clean_processed, 'b-', label='原始(处理后)', alpha=0.7)
        axes[row, col].plot(wl, processed, 'r--', label='校正后', alpha=0.7)
        axes[row, col].set_title(f'{pipeline["name"]}\n相关系数={correlation:.4f}')
        axes[row, col].set_xlabel('拉曼位移 (cm⁻¹)')
        axes[row, col].set_ylabel('归一化强度')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        
        print(f"  {pipeline['name']}: 相关系数={correlation:.4f}")
    
    plt.tight_layout()
    plt.savefig('pipeline_comparison.png', dpi=150)
    print(f"\n结果图已保存到: pipeline_comparison.png")
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("拉曼光谱基线漂移和荧光背景校正测试")
    print("=" * 70)
    
    test_baseline_methods()
    test_matching_with_fluorescence()
    test_normalization_methods()
    test_full_pipeline()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
