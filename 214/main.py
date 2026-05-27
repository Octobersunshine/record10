import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from spectrum_loader import load_spectrum
from preprocessing import preprocess_pipeline
from peak_detection import analyze_peaks, create_peak_vector
from matching import match_spectrum, peak_based_match, combined_match, identify_material
from spectrum_library import SpectrumLibrary
from utils import resample_spectrum, interpolate_spectrum

try:
    import torch
    from deep_learning import DeepLearningAnalyzer, generate_mixture_spectra
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("警告: PyTorch未安装，深度学习功能不可用")


class RamanIdentifier:
    """拉曼光谱识别系统"""
    
    def __init__(self):
        self.library = SpectrumLibrary()
        self.preprocess_kwargs = {
            'smooth': True,
            'smooth_method': 'savgol',
            'smooth_kwargs': {'window_length': 7, 'polyorder': 2},
            'baseline': True,
            'baseline_method': 'airpls',
            'baseline_kwargs': {'lam': 1e5, 'niter': 15},
            'normalize': True,
            'normalize_method': 'minmax'
        }
        self.peak_kwargs = {
            'height': 0.03,
            'distance': 5,
            'prominence': 0.015
        }
    
    def build_library_from_files(self, file_dict: Dict[str, str]):
        """
        从文件构建光谱库
        
        Args:
            file_dict: {物质名称: 文件路径} 字典
        """
        for name, file_path in file_dict.items():
            self.library.add_from_file(
                name, file_path,
                preprocess_kwargs=self.preprocess_kwargs,
                peak_kwargs=self.peak_kwargs
            )
        print(f"光谱库构建完成，共 {len(self.library)} 种物质")
    
    def build_library_from_data(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        从数据构建光谱库
        
        Args:
            data_dict: {物质名称: (波长数组, 强度数组)} 字典
        """
        for name, (wl, inten) in data_dict.items():
            self.library.add_spectrum(
                name, wl, inten,
                preprocess_kwargs=self.preprocess_kwargs,
                peak_kwargs=self.peak_kwargs
            )
        print(f"光谱库构建完成，共 {len(self.library)} 种物质")
    
    def preprocess_unknown(self, wavelengths: np.ndarray,
                           intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
        """
        预处理未知光谱
        
        Args:
            wavelengths: 波长数组
            intensities: 强度数组
        
        Returns:
            (处理后的波长, 处理后的强度, 谱峰分析, 峰向量)
        """
        wl, inten = resample_spectrum(wavelengths, intensities, self.library.num_points)
        
        _, processed_inten, _ = preprocess_pipeline(
            wl, inten, **self.preprocess_kwargs
        )
        
        if self.library.common_wavelengths is not None:
            processed_inten = interpolate_spectrum(
                wl, processed_inten, self.library.common_wavelengths
            )
            wl = self.library.common_wavelengths
        
        peak_analysis = analyze_peaks(wl, processed_inten, **self.peak_kwargs)
        
        wl_range = (wl.min(), wl.max())
        peak_vector = create_peak_vector(
            wl, processed_inten, peak_analysis,
            num_bins=self.library.peak_vector_bins,
            wl_range=wl_range
        )
        
        return wl, processed_inten, peak_analysis, peak_vector
    
    def identify(self, wavelengths: np.ndarray, intensities: np.ndarray,
                 method: str = 'combined',
                 intensity_weight: float = 0.5,
                 peak_weight: float = 0.5,
                 similarity_method: str = 'cosine',
                 threshold: float = 0.7,
                 top_k: int = 5) -> Dict:
        """
        识别未知光谱
        
        Args:
            wavelengths: 波长数组
            intensities: 强度数组
            method: 匹配方法，'intensity'（全谱）, 'peak'（峰向量）, 'combined'（组合）
            intensity_weight: 全谱权重（组合模式）
            peak_weight: 峰向量权重（组合模式）
            similarity_method: 相似度计算方法
            threshold: 识别阈值
            top_k: 返回前K个结果
        
        Returns:
            识别结果字典
        """
        if len(self.library) == 0:
            raise ValueError("光谱库为空，请先构建光谱库")
        
        wl, processed_inten, peak_analysis, peak_vector = self.preprocess_unknown(
            wavelengths, intensities
        )
        
        if method == 'intensity':
            matches = match_spectrum(
                processed_inten,
                self.library.processed_intensities,
                self.library.names,
                method=similarity_method,
                top_k=top_k
            )
        elif method == 'peak':
            matches = peak_based_match(
                peak_vector,
                self.library.peak_vectors,
                self.library.names,
                method=similarity_method,
                top_k=top_k
            )
        elif method == 'combined':
            matches = combined_match(
                processed_inten,
                peak_vector,
                self.library.processed_intensities,
                self.library.peak_vectors,
                self.library.names,
                intensity_weight=intensity_weight,
                peak_weight=peak_weight,
                method=similarity_method,
                top_k=top_k
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        material_name, best_score, reliable = identify_material(matches, threshold)
        
        return {
            'identified_material': material_name,
            'best_score': best_score,
            'reliable': reliable,
            'top_matches': matches,
            'method': method,
            'similarity_method': similarity_method,
            'threshold': threshold,
            'processed_wavelengths': wl,
            'processed_intensities': processed_inten,
            'peak_analysis': peak_analysis,
            'peak_vector': peak_vector
        }
    
    def identify_from_file(self, file_path: str, **kwargs) -> Dict:
        """
        从文件识别光谱
        
        Args:
            file_path: 文件路径
            **kwargs: 传递给identify的参数
        
        Returns:
            识别结果字典
        """
        wl, inten = load_spectrum(file_path)
        return self.identify(wl, inten, **kwargs)
    
    def prepare_deep_learning_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备深度学习训练数据
        
        Returns:
            (光谱数组, 标签数组, 类别名称列表)
        """
        if len(self.library) == 0:
            raise ValueError("光谱库为空，请先构建光谱库")
        
        spectra = []
        labels = []
        class_names = self.library.list_materials()
        
        for i, name in enumerate(class_names):
            _, processed_inten = self.library.get_processed_spectrum(name)
            for _ in range(50):
                noise = np.random.normal(0, 0.05, len(processed_inten))
                intensity_shift = np.random.uniform(-0.1, 0.1)
                augmented = processed_inten + noise + intensity_shift
                augmented = np.clip(augmented, 0, 1)
                spectra.append(augmented)
                labels.append(i)
        
        return np.array(spectra), np.array(labels), class_names
    
    def train_deep_learning_model(self, model_type: str = 'cnn',
                                  epochs: int = 30,
                                  batch_size: int = 32,
                                  lr: float = 0.001) -> Dict:
        """
        训练深度学习模型
        
        Args:
            model_type: 模型类型 ('cnn', 'resnet')
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
        
        Returns:
            训练历史字典
        """
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("PyTorch未安装，深度学习功能不可用")
        
        spectra, labels, class_names = self.prepare_deep_learning_data()
        
        self.dl_analyzer = DeepLearningAnalyzer(
            model_type=model_type,
            num_classes=len(class_names),
            input_length=self.library.num_points
        )
        self.dl_analyzer.set_class_names(class_names)
        
        train_loader, val_loader = self.dl_analyzer.prepare_data(
            spectra, labels,
            batch_size=batch_size,
            use_augmentation=True
        )
        
        history = self.dl_analyzer.train(
            train_loader, val_loader,
            epochs=epochs,
            lr=lr
        )
        
        return history
    
    def identify_with_deep_learning(self, wavelengths: np.ndarray,
                                     intensities: np.ndarray) -> Dict:
        """
        使用深度学习模型识别光谱
        
        Args:
            wavelengths: 波长数组
            intensities: 强度数组
        
        Returns:
            识别结果字典
        """
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("PyTorch未安装，深度学习功能不可用")
        
        if not hasattr(self, 'dl_analyzer') or not self.dl_analyzer.is_trained:
            raise RuntimeError("请先训练深度学习模型")
        
        wl, inten = resample_spectrum(wavelengths, intensities, self.library.num_points)
        _, processed_inten, _ = preprocess_pipeline(wl, inten, **self.preprocess_kwargs)
        
        if self.library.common_wavelengths is not None:
            processed_inten = interpolate_spectrum(
                wl, processed_inten, self.library.common_wavelengths
            )
        
        result = self.dl_analyzer.predict(processed_inten)
        
        return {
            'identified_material': result['class'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'class_names': self.dl_analyzer.class_names
        }
    
    def analyze_mixture_with_deep_learning(self, wavelengths: np.ndarray,
                                           intensities: np.ndarray,
                                           threshold: float = 0.5) -> Dict:
        """
        使用深度学习模型分析混合光谱
        
        Args:
            wavelengths: 波长数组
            intensities: 强度数组
            threshold: 成分检测阈值
        
        Returns:
            分析结果字典
        """
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("PyTorch未安装，深度学习功能不可用")
        
        if not hasattr(self, 'dl_analyzer') or not self.dl_analyzer.is_trained:
            raise RuntimeError("请先训练深度学习模型")
        
        if self.dl_analyzer.model_type != 'mixup':
            raise RuntimeError("混合分析需要使用mixup模型")
        
        wl, inten = resample_spectrum(wavelengths, intensities, self.library.num_points)
        _, processed_inten, _ = preprocess_pipeline(wl, inten, **self.preprocess_kwargs)
        
        if self.library.common_wavelengths is not None:
            processed_inten = interpolate_spectrum(
                wl, processed_inten, self.library.common_wavelengths
            )
        
        result = self.dl_analyzer.predict(processed_inten)
        
        return {
            'detected_components': result['detected_components'],
            'probabilities': result['probabilities'],
            'class_names': self.dl_analyzer.class_names
        }
    
    def train_mixture_model(self, num_mixtures: int = 2000,
                            max_components: int = 3,
                            epochs: int = 50,
                            batch_size: int = 32) -> Dict:
        """
        训练混合光谱分析模型
        
        Args:
            num_mixtures: 生成的混合光谱数量
            max_components: 混合光谱中的最大组分数
            epochs: 训练轮数
            batch_size: 批大小
        
        Returns:
            训练历史字典
        """
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("PyTorch未安装，深度学习功能不可用")
        
        class_names = self.library.list_materials()
        
        pure_spectra = {}
        for name in class_names:
            _, processed_inten = self.library.get_processed_spectrum(name)
            pure_spectra[name] = processed_inten
        
        mixtures, labels, _ = generate_mixture_spectra(
            pure_spectra,
            num_mixtures=num_mixtures,
            max_components=max_components
        )
        
        self.dl_analyzer = DeepLearningAnalyzer(
            model_type='mixup',
            num_classes=len(class_names),
            input_length=self.library.num_points
        )
        self.dl_analyzer.set_class_names(class_names)
        
        train_loader, val_loader = self.dl_analyzer.prepare_data(
            mixtures, labels,
            batch_size=batch_size,
            use_augmentation=True,
            train_ratio=0.8
        )
        
        history = self.dl_analyzer.train(
            train_loader, val_loader,
            epochs=epochs,
            lr=0.001
        )
        
        return history
    
    def plot_result(self, result: Dict, save_path: Optional[str] = None):
        """
        绘制识别结果
        
        Args:
            result: 识别结果字典
            save_path: 保存路径，可选
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        wl = result['processed_wavelengths']
        inten = result['processed_intensities']
        peak_analysis = result['peak_analysis']
        
        axes[0, 0].plot(wl, inten, 'b-', label='未知光谱')
        if len(peak_analysis['positions']) > 0:
            axes[0, 0].plot(peak_analysis['positions'], peak_analysis['heights'],
                           'ro', markersize=6, label=f'检测到 {len(peak_analysis["positions"])} 个峰')
        axes[0, 0].set_xlabel('拉曼位移 (cm⁻¹)')
        axes[0, 0].set_ylabel('归一化强度')
        axes[0, 0].set_title('未知光谱及谱峰检测')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        if result['identified_material'] is not None:
            ref_wl, ref_inten = self.library.get_processed_spectrum(
                result['identified_material']
            )
            axes[0, 1].plot(wl, inten, 'b-', label='未知光谱', alpha=0.7)
            axes[0, 1].plot(ref_wl, ref_inten, 'r--', label=result['identified_material'], alpha=0.7)
            axes[0, 1].set_xlabel('拉曼位移 (cm⁻¹)')
            axes[0, 1].set_ylabel('归一化强度')
            title = f"最佳匹配: {result['identified_material']}\n相似度: {result['best_score']:.4f}"
            if not result['reliable']:
                title += ' (不可靠)'
            axes[0, 1].set_title(title)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, '未识别到匹配物质',
                           ha='center', va='center', fontsize=14,
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('匹配结果')
        
        top_matches = result['top_matches']
        names = [m[0] for m in top_matches]
        scores = [m[1] for m in top_matches]
        colors = ['green' if i == 0 and result['reliable'] else 'orange' if i == 0 else 'skyblue'
                 for i in range(len(names))]
        bars = axes[1, 0].barh(range(len(names)), scores, color=colors)
        axes[1, 0].set_yticks(range(len(names)))
        axes[1, 0].set_yticklabels(names)
        axes[1, 0].set_xlabel('相似度')
        axes[1, 0].set_title(f'Top {len(names)} 匹配结果 ({result["method"]})')
        axes[1, 0].set_xlim(0, 1.1)
        for bar, score in zip(bars, scores):
            axes[1, 0].text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                           f'{score:.4f}', va='center')
        axes[1, 0].axvline(x=result['threshold'], color='red', linestyle='--',
                          label=f'阈值={result["threshold"]}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        peak_vector = result['peak_vector']
        axes[1, 1].stem(np.linspace(wl.min(), wl.max(), len(peak_vector)),
                       peak_vector, basefmt='b-', linefmt='g-', markerfmt='ro')
        axes[1, 1].set_xlabel('拉曼位移 (cm⁻¹)')
        axes[1, 1].set_ylabel('峰强度')
        axes[1, 1].set_title('峰向量表示')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果图已保存到: {save_path}")
        
        plt.show()


def generate_example_spectra() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    生成示例光谱数据
    
    Returns:
        {物质名称: (波长数组, 强度数组)} 字典
    """
    np.random.seed(42)
    
    wavelengths = np.linspace(200, 1800, 500)
    
    materials = {
        '葡萄糖': [420, 520, 790, 850, 920, 1060, 1120, 1280, 1360, 1460],
        '果糖': [430, 540, 780, 870, 940, 1080, 1150, 1260, 1380, 1480],
        '蔗糖': [410, 510, 770, 860, 910, 1040, 1130, 1270, 1350, 1450],
        '乙醇': [450, 550, 810, 880, 960, 1090, 1170, 1290, 1400, 1500],
        '丙酮': [460, 560, 795, 890, 970, 1070, 1180, 1310, 1410, 1520],
        '甲醇': [440, 530, 800, 875, 950, 1085, 1160, 1285, 1390, 1490]
    }
    
    spectra = {}
    for name, peaks in materials.items():
        intensities = np.zeros_like(wavelengths)
        for peak in peaks:
            sigma = 15 + np.random.randn() * 3
            amplitude = 0.5 + np.random.rand() * 0.5
            intensities += amplitude * np.exp(-(wavelengths - peak) ** 2 / (2 * sigma ** 2))
        
        baseline = 0.1 + 0.05 * np.sin(wavelengths / 200)
        noise = np.random.normal(0, 0.02, len(wavelengths))
        intensities = intensities + baseline + noise
        
        spectra[name] = (wavelengths, intensities)
    
    return spectra


def generate_unknown_spectrum(true_material: str = '葡萄糖',
                              noise_level: float = 0.05,
                              shift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成未知光谱（带噪声和偏移）
    
    Args:
        true_material: 真实物质名称
        noise_level: 噪声水平
        shift: 强度偏移
    
    Returns:
        (波长数组, 强度数组)
    """
    spectra = generate_example_spectra()
    
    if true_material not in spectra:
        raise ValueError(f"未知物质: {true_material}")
    
    wl, inten = spectra[true_material]
    
    noise = np.random.normal(0, noise_level, len(inten))
    inten = inten + noise + shift
    
    return wl, inten


def main():
    print("=" * 70)
    print("拉曼光谱识别系统 - 含深度学习功能")
    print("=" * 70)
    
    print("\n1. 构建光谱库...")
    identifier = RamanIdentifier()
    spectra = generate_example_spectra()
    identifier.build_library_from_data(spectra)
    
    print("\n库中物质列表:")
    for name in identifier.library.list_materials():
        print(f"  - {name}")
    
    print("\n2. 传统方法识别...")
    unknown_wl, unknown_inten = generate_unknown_spectrum(
        true_material='葡萄糖', noise_level=0.08, shift=0.1
    )
    
    result = identifier.identify(
        unknown_wl, unknown_inten,
        method='combined',
        intensity_weight=0.4,
        peak_weight=0.6,
        similarity_method='cosine',
        threshold=0.7,
        top_k=5
    )
    
    print("\n传统方法识别结果:")
    print(f"  识别物质: {result['identified_material']}")
    print(f"  最佳相似度: {result['best_score']:.4f}")
    print(f"  识别可靠: {'是' if result['reliable'] else '否'}")
    
    if DEEP_LEARNING_AVAILABLE:
        print("\n3. 深度学习模型训练 (1D-CNN)...")
        history = identifier.train_deep_learning_model(
            model_type='cnn',
            epochs=20,
            batch_size=32,
            lr=0.001
        )
        
        print("\n4. 深度学习模型识别...")
        dl_result = identifier.identify_with_deep_learning(unknown_wl, unknown_inten)
        print(f"\n深度学习识别结果:")
        print(f"  识别物质: {dl_result['identified_material']}")
        print(f"  置信度: {dl_result['confidence']:.4f}")
        
        print("\n5. 混合光谱分析模型训练 (药物真伪鉴别)...")
        mix_history = identifier.train_mixture_model(
            num_mixtures=3000,
            max_components=3,
            epochs=30,
            batch_size=32
        )
        
        print("\n6. 药物真伪鉴别测试...")
        pure_spectra = {}
        for name in identifier.library.list_materials():
            _, processed_inten = identifier.library.get_processed_spectrum(name)
            pure_spectra[name] = processed_inten
        
        print("\n  测试1: 纯物质葡萄糖（真药）")
        test_inten = pure_spectra['葡萄糖'] + np.random.normal(0, 0.03, 1000)
        test_inten = np.clip((test_inten - test_inten.min()) / (test_inten.max() - test_inten.min()), 0, 1)
        mix_result = identifier.analyze_mixture_with_deep_learning(
            identifier.library.common_wavelengths, test_inten
        )
        print(f"    检测成分: {[c['name'] for c in mix_result['detected_components']]}")
        
        print("\n  测试2: 葡萄糖+蔗糖混合物（假药）")
        mixed = 0.7 * pure_spectra['葡萄糖'] + 0.3 * pure_spectra['蔗糖']
        mixed = mixed + np.random.normal(0, 0.03, 1000)
        mixed = np.clip((mixed - mixed.min()) / (mixed.max() - mixed.min()), 0, 1)
        mix_result = identifier.analyze_mixture_with_deep_learning(
            identifier.library.common_wavelengths, mixed
        )
        print(f"    检测成分: {[c['name'] for c in mix_result['detected_components']]}")
        
        print("\n  测试3: 葡萄糖+乙醇混合物（假药）")
        mixed = 0.8 * pure_spectra['葡萄糖'] + 0.2 * pure_spectra['乙醇']
        mixed = mixed + np.random.normal(0, 0.03, 1000)
        mixed = np.clip((mixed - mixed.min()) / (mixed.max() - mixed.min()), 0, 1)
        mix_result = identifier.analyze_mixture_with_deep_learning(
            identifier.library.common_wavelengths, mixed
        )
        print(f"    检测成分: {[c['name'] for c in mix_result['detected_components']]}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
