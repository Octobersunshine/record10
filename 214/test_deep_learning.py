"""
深度学习光谱识别测试 - 验证1D-CNN、ResNet和混合光谱分析
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import RamanIdentifier, generate_example_spectra, generate_unknown_spectrum
from deep_learning import (
    generate_mixture_spectra, DeepLearningAnalyzer,
    Raman1DCNN, RamanResNet, RamanMixupModel
)


def test_model_architectures():
    """测试模型架构"""
    print("=" * 70)
    print("测试模型架构")
    print("=" * 70)
    
    input_length = 1000
    num_classes = 6
    
    models = {
        '1D-CNN': Raman1DCNN(num_classes, input_length),
        'ResNet': RamanResNet(num_classes, input_length),
        'MixupModel': RamanMixupModel(num_classes, input_length)
    }
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, 1, input_length)
            output = model(dummy_input)
            print(f"{name:12s}: 输入 {dummy_input.shape} -> 输出 {output.shape}")
            
            features = model.extract_features(dummy_input)
            print(f"             特征向量维度: {features.shape}")
    
    print()


def test_spectrum_augmentation():
    """测试光谱数据增强"""
    print("=" * 70)
    print("测试光谱数据增强")
    print("=" * 70)
    
    from deep_learning import SpectraAugmentation
    
    wl = np.linspace(200, 1800, 1000)
    original = np.sin(wl / 100) * 0.5 + 0.5
    
    augment = SpectraAugmentation(
        noise_level=0.05,
        intensity_shift_range=0.1,
        wavelength_shift_max=5,
        scaling_range=0.1
    )
    
    original_tensor = torch.FloatTensor(original).unsqueeze(0).unsqueeze(0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(wl, original)
    axes[0, 0].set_title('原始光谱')
    
    for i in range(5):
        row = (i + 1) // 3
        col = (i + 1) % 3
        augmented = augment(original_tensor)
        axes[row, col].plot(wl, augmented.squeeze().numpy())
        axes[row, col].set_title(f'增强样本 {i+1}')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=100)
    print("数据增强示例图已保存到: augmentation_examples.png")
    plt.close()


def test_1dcnn_classification():
    """测试1D-CNN分类"""
    print("\n" + "=" * 70)
    print("测试1D-CNN光谱分类")
    print("=" * 70)
    
    identifier = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    print("训练1D-CNN模型...")
    history = identifier.train_deep_learning_model(
        model_type='cnn',
        epochs=15,
        batch_size=32,
        lr=0.001
    )
    
    print("\n评估模型...")
    test_materials = ['葡萄糖', '乙醇', '蔗糖']
    correct = 0
    total = 0
    
    for material in test_materials:
        for _ in range(10):
            wl, inten = generate_unknown_spectrum(
                true_material=material, noise_level=0.1
            )
            result = identifier.identify_with_deep_learning(wl, inten)
            if result['identified_material'] == material:
                correct += 1
            total += 1
            print(f"  真实: {material:8s} -> 预测: {result['identified_material']:8s} "
                  f"(置信度: {result['confidence']:.4f})")
    
    print(f"\n准确率: {correct}/{total} = {correct/total:.2%}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='训练损失')
    axes[0].plot(history['val_loss'], label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练过程 - Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], label='训练准确率')
    axes[1].plot(history['val_acc'], label='验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('训练过程 - Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png', dpi=100)
    print("训练曲线图已保存到: cnn_training_history.png")
    plt.close()
    
    return history


def test_mixture_analysis():
    """测试混合光谱分析"""
    print("\n" + "=" * 70)
    print("测试混合光谱分析（药物真伪鉴别）")
    print("=" * 70)
    
    identifier = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    print("训练混合光谱分析模型...")
    history = identifier.train_mixture_model(
        num_mixtures=2000,
        max_components=3,
        epochs=20,
        batch_size=32
    )
    
    print("\n药物真伪鉴别测试:")
    print("-" * 70)
    
    class_names = identifier.library.list_materials()
    pure_spectra = {}
    for name in class_names:
        _, processed_inten = identifier.library.get_processed_spectrum(name)
        pure_spectra[name] = processed_inten
    
    test_cases = [
        {
            'name': '纯葡萄糖（真药）',
            'components': ['葡萄糖'],
            'weights': [1.0]
        },
        {
            'name': '葡萄糖+蔗糖（假药）',
            'components': ['葡萄糖', '蔗糖'],
            'weights': [0.7, 0.3]
        },
        {
            'name': '葡萄糖+乙醇（假药）',
            'components': ['葡萄糖', '乙醇'],
            'weights': [0.8, 0.2]
        },
        {
            'name': '葡萄糖+蔗糖+乙醇（假药）',
            'components': ['葡萄糖', '蔗糖', '乙醇'],
            'weights': [0.5, 0.3, 0.2]
        }
    ]
    
    for case in test_cases:
        mixed = np.zeros(1000)
        for comp, weight in zip(case['components'], case['weights']):
            mixed += weight * pure_spectra[comp]
        
        mixed += np.random.normal(0, 0.03, 1000)
        mixed = np.clip((mixed - mixed.min()) / (mixed.max() - mixed.min()), 0, 1)
        
        result = identifier.analyze_mixture_with_deep_learning(
            identifier.library.common_wavelengths, mixed
        )
        
        detected = [c['name'] for c in result['detected_components']]
        probs = {c['name']: c['probability'] for c in result['detected_components']}
        
        print(f"\n  {case['name']}:")
        print(f"    真实成分: {case['components']}")
        print(f"    检测成分: {detected}")
        for name in class_names:
            prob = result['probabilities'][class_names.index(name)]
            status = "✓" if name in case['components'] else " "
            print(f"      {status} {name:8s}: {prob:.4f}")
        
        is_authentic = len(detected) == 1 and detected[0] == '葡萄糖'
        print(f"    真伪判断: {'真药 ✓' if is_authentic else '假药 ✗'}")
    
    print("\n混合光谱分析模型训练完成!")
    
    return history


def test_feature_extraction():
    """测试特征提取"""
    print("\n" + "=" * 70)
    print("测试光谱特征提取")
    print("=" * 70)
    
    identifier = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    identifier.train_deep_learning_model(
        model_type='cnn',
        epochs=10,
        batch_size=32
    )
    
    class_names = identifier.library.list_materials()
    
    features = []
    for name in class_names:
        _, processed_inten = identifier.library.get_processed_spectrum(name)
        feature = identifier.dl_analyzer.extract_features(processed_inten)
        features.append(feature)
    
    features = np.array(features)
    
    print("特征向量维度:", features.shape[1])
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    for i, name in enumerate(class_names):
        axes[0].scatter(features_pca[i, 0], features_pca[i, 1], s=100, label=name)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PCA特征可视化')
    axes[0].legend()
    
    if len(features) >= 4:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_tsne = tsne.fit_transform(features)
        
        for i, name in enumerate(class_names):
            axes[1].scatter(features_tsne[i, 0], features_tsne[i, 1], s=100, label=name)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('t-SNE特征可视化')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('feature_visualization.png', dpi=100)
    print("特征可视化图已保存到: feature_visualization.png")
    plt.close()


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n" + "=" * 70)
    print("测试模型保存和加载")
    print("=" * 70)
    
    identifier = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    identifier.train_deep_learning_model(
        model_type='cnn',
        epochs=5,
        batch_size=32
    )
    
    model_path = 'test_model.pth'
    identifier.dl_analyzer.save_model(model_path)
    
    wl, inten = generate_unknown_spectrum(true_material='葡萄糖')
    result_before = identifier.identify_with_deep_learning(wl, inten)
    
    new_analyzer = DeepLearningAnalyzer(model_type='cnn', num_classes=6)
    new_analyzer.load_model(model_path)
    
    result_after = new_analyzer.predict(
        np.interp(
            np.linspace(200, 1800, 1000),
            wl,
            inten
        )
    )
    
    print(f"保存前预测: {result_before['identified_material']}")
    print(f"加载后预测: {result_after['class']}")
    print(f"结果一致: {result_before['identified_material'] == result_after['class']}")
    
    import os
    if os.path.exists(model_path):
        os.remove(model_path)


def main():
    print("=" * 70)
    print("深度学习光谱识别系统测试")
    print("=" * 70)
    
    test_model_architectures()
    test_spectrum_augmentation()
    test_1dcnn_classification()
    test_mixture_analysis()
    test_feature_extraction()
    test_model_save_load()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
