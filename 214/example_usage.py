"""
拉曼光谱识别系统 - 使用示例
"""

import numpy as np
from main import RamanIdentifier, generate_example_spectra, generate_unknown_spectrum


def example_basic():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用流程")
    print("=" * 60)
    
    identifier = RamanIdentifier()
    
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    unknown_wl, unknown_inten = generate_unknown_spectrum(
        true_material='葡萄糖', noise_level=0.1
    )
    
    result = identifier.identify(
        unknown_wl, unknown_inten,
        method='combined',
        similarity_method='cosine',
        threshold=0.7
    )
    
    print(f"\n识别结果: {result['identified_material']}")
    print(f"相似度: {result['best_score']:.4f}")
    print(f"可靠: {'是' if result['reliable'] else '否'}")
    
    print("\nTop 3 匹配:")
    for name, score in result['top_matches'][:3]:
        print(f"  {name}: {score:.4f}")


def example_custom_library():
    """构建自定义光谱库示例"""
    print("\n" + "=" * 60)
    print("示例2: 构建自定义光谱库")
    print("=" * 60)
    
    identifier = RamanIdentifier()
    
    wavelengths = np.linspace(200, 1800, 500)
    
    custom_spectra = {
        '物质A': (wavelengths, np.sin(wavelengths / 100) + 1),
        '物质B': (wavelengths, np.cos(wavelengths / 80) + 1),
        '物质C': (wavelengths, np.sin(wavelengths / 120) * np.cos(wavelengths / 200) + 1),
    }
    
    identifier.build_library_from_data(custom_spectra)
    
    print(f"\n库中物质: {identifier.library.list_materials()}")
    
    unknown_wl = wavelengths
    unknown_inten = np.sin(wavelengths / 100) + 1 + np.random.normal(0, 0.1, len(wavelengths))
    
    result = identifier.identify(
        unknown_wl, unknown_inten,
        method='intensity',
        similarity_method='pearson'
    )
    
    print(f"\n识别结果: {result['identified_material']}")
    print(f"相似度: {result['best_score']:.4f}")


def example_peak_only():
    """仅使用峰匹配示例"""
    print("\n" + "=" * 60)
    print("示例3: 仅使用谱峰匹配")
    print("=" * 60)
    
    identifier = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    unknown_wl, unknown_inten = generate_unknown_spectrum(
        true_material='乙醇', noise_level=0.05
    )
    
    result = identifier.identify(
        unknown_wl, unknown_inten,
        method='peak',
        similarity_method='cosine',
        threshold=0.6
    )
    
    print(f"\n识别结果: {result['identified_material']}")
    print(f"相似度: {result['best_score']:.4f}")
    print(f"检测到的峰数: {len(result['peak_analysis']['positions'])}")


def example_save_load_library():
    """保存和加载光谱库示例"""
    print("\n" + "=" * 60)
    print("示例4: 保存和加载光谱库")
    print("=" * 60)
    
    identifier1 = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier1.build_library_from_data(spectra)
    
    library_dir = 'raman_library'
    identifier1.library.save_library(library_dir)
    print(f"\n光谱库已保存到: {library_dir}/")
    
    identifier2 = RamanIdentifier()
    identifier2.library.load_library(library_dir)
    print(f"加载的光谱库包含 {len(identifier2.library)} 种物质")
    print(f"物质列表: {identifier2.library.list_materials()}")


def example_different_similarity_methods():
    """不同相似度方法对比"""
    print("\n" + "=" * 60)
    print("示例5: 不同相似度方法对比")
    print("=" * 60)
    
    identifier = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    unknown_wl, unknown_inten = generate_unknown_spectrum(
        true_material='蔗糖', noise_level=0.08
    )
    
    methods = ['cosine', 'pearson', 'spearman', 'euclidean', 'manhattan']
    
    print("\n全谱匹配结果:")
    for method in methods:
        result = identifier.identify(
            unknown_wl, unknown_inten,
            method='intensity',
            similarity_method=method
        )
        print(f"  {method:12s}: {result['identified_material']:8s} (score={result['best_score']:.4f})")


if __name__ == '__main__':
    example_basic()
    example_custom_library()
    example_peak_only()
    example_save_load_library()
    example_different_similarity_methods()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
