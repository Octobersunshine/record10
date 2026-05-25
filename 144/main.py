import numpy as np
import cv2
import matplotlib.pyplot as plt
from orientation import compute_orientation_field, visualize_orientation_field
from singularity import (
    detect_singularities_robust, 
    multiscale_singularity_detection,
    visualize_singularities,
    visualize_poincare_map,
    visualize_confidence_map
)
from gabor_enhance import (
    fast_enhance_fingerprint,
    multiscale_enhance,
    estimate_quality_map
)

try:
    from deep_learning import SingularityDetector
    from data_generator import (
        generate_synthetic_fingerprint_with_keypoints,
        apply_wet_effect,
        apply_dry_effect,
        apply_blur_effect,
        apply_noise_effect
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def generate_synthetic_fingerprint(size=(256, 256), fingerprint_type='whorl'):
    if TORCH_AVAILABLE:
        image, cores, deltas = generate_synthetic_fingerprint_with_keypoints(size, fingerprint_type)
        return image
    else:
        h, w = size
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w // 2, h // 2
        
        if fingerprint_type == 'loop_right':
            dy = y - cy
            dx = x - cx
            angle = np.arctan2(dy, dx)
            r = np.sqrt(dx**2 + dy**2)
            ridge_pattern = np.sin(r / 4 + angle)
        elif fingerprint_type == 'loop_left':
            dy = y - cy
            dx = x - cx
            angle = np.arctan2(dy, -dx)
            r = np.sqrt(dx**2 + dy**2)
            ridge_pattern = np.sin(r / 4 + angle)
        elif fingerprint_type == 'whorl':
            dy = y - cy
            dx = x - cx
            angle = np.arctan2(dy, dx)
            r = np.sqrt(dx**2 + dy**2)
            ridge_pattern = np.sin(r / 3 + 2 * angle)
        elif fingerprint_type == 'arch':
            ridge_freq = 1 / 8.0
            ridge_pattern = np.sin(2 * np.pi * ridge_freq * (y + 0.3 * x))
        else:
            ridge_freq = 1 / 8.0
            ridge_pattern = np.sin(2 * np.pi * ridge_freq * y)
        
        ridge_pattern = (ridge_pattern + 1) / 2
        noise = np.random.normal(0, 0.1, (h, w))
        ridge_pattern = np.clip(ridge_pattern + noise, 0, 1)
        pattern = (ridge_pattern * 255).astype(np.uint8)
        
        mask = np.zeros((h, w), dtype=np.float64)
        center = (w // 2, h // 2)
        axes = (int(w * 0.4), int(h * 0.4))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 5)
        
        pattern = (pattern * mask + 128 * (1 - mask)).astype(np.uint8)
        return pattern


def create_low_quality_fingerprint(image, blur_kernel=5, noise_level=0.05):
    h, w = image.shape
    
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    noise = np.random.normal(0, noise_level * 255, (h, w))
    noisy = blurred.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy


def process_fingerprint_robust(image, use_multiscale=False, use_enhancement=False, 
                               use_deep_learning=False, dl_model_path=None):
    print("  计算方向场...")
    theta_field, coherence = compute_orientation_field(
        image, 
        block_size=16,
        gradient_sigma=1.0,
        orientation_sigma=3.0
    )
    
    enhanced_image = None
    if use_enhancement:
        print("  Gabor滤波增强...")
        enhanced_image = fast_enhance_fingerprint(image, theta_field)
        theta_enhanced, coherence_enhanced = compute_orientation_field(
            enhanced_image,
            block_size=16,
            gradient_sigma=1.0,
            orientation_sigma=3.0
        )
        theta_field = theta_enhanced
        coherence = coherence_enhanced
    
    if use_deep_learning and TORCH_AVAILABLE:
        print("  深度学习检测...")
        try:
            detector = SingularityDetector(
                model_path=dl_model_path,
                num_cores=2,
                num_deltas=2,
                img_size=256,
                use_heatmap=False
            )
            cores, deltas = detector.detect(
                enhanced_image if enhanced_image is not None else image,
                confidence_threshold=0.3
            )
        except Exception as e:
            print(f"  DL检测失败，使用传统方法: {e}")
            cores, deltas, poincare_map, confidence_map = detect_singularities_robust(
                theta_field,
                coherence_map=coherence,
                min_distance=20,
                poincare_threshold=0.6,
                confidence_threshold=0.3,
                coherence_threshold=0.2
            )
        poincare_map = None
        confidence_map = None
    elif use_multiscale:
        print("  多分辨率检测...")
        def theta_func(img):
            return compute_orientation_field(img, block_size=16, gradient_sigma=1.0, orientation_sigma=3.0)
        
        cores, deltas = multiscale_singularity_detection(
            image if enhanced_image is None else enhanced_image,
            theta_func,
            scales=[1.0, 0.75, 0.5],
            min_distance=20,
            poincare_threshold=0.6,
            confidence_threshold=0.3,
            coherence_threshold=0.2
        )
        poincare_map = None
        confidence_map = None
    else:
        print("  鲁棒奇异点检测...")
        cores, deltas, poincare_map, confidence_map = detect_singularities_robust(
            theta_field,
            coherence_map=coherence,
            min_distance=20,
            poincare_threshold=0.6,
            confidence_threshold=0.3,
            coherence_threshold=0.2
        )
    
    print(f"  检测到 {len(cores)} 个核心点, {len(deltas)} 个三角点")
    
    return theta_field, coherence, cores, deltas, poincare_map, confidence_map, enhanced_image


def compare_all_methods(original_image, low_quality_image, fp_type, dl_model_path=None):
    print("\n" + "=" * 70)
    print(f"指纹类型: {fp_type} - 所有方法对比")
    print("=" * 70)
    
    methods = [
        ('原始 - 基础方法', {'use_multiscale': False, 'use_enhancement': False, 'use_deep_learning': False}),
        ('低质量 - 基础方法', {'use_multiscale': False, 'use_enhancement': False, 'use_deep_learning': False}),
        ('低质量 - 置信度过滤', {'use_multiscale': False, 'use_enhancement': False, 'use_deep_learning': False}),
        ('低质量 - Gabor增强', {'use_multiscale': False, 'use_enhancement': True, 'use_deep_learning': False}),
        ('低质量 - Gabor+多分辨率', {'use_multiscale': True, 'use_enhancement': True, 'use_deep_learning': False}),
    ]
    
    if TORCH_AVAILABLE:
        methods.append(('低质量 - 深度学习', {'use_multiscale': False, 'use_enhancement': True, 'use_deep_learning': True, 'dl_model_path': dl_model_path}))
    
    results = {}
    all_data = {}
    
    for i, (method_name, params) in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] {method_name}:")
        
        if '原始' in method_name:
            img = original_image
        else:
            img = low_quality_image
        
        theta, coh, cores, deltas, poincare, conf, enhanced = process_fingerprint_robust(img, **params)
        
        results[method_name] = (cores, deltas)
        all_data[method_name] = {
            'theta': theta,
            'coh': coh,
            'cores': cores,
            'deltas': deltas,
            'poincare': poincare,
            'conf': conf,
            'enhanced': enhanced,
            'image': img
        }
    
    num_cols = len(methods)
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 12))
    
    for i, (method_name, data) in enumerate(all_data.items()):
        if num_cols == 1:
            ax1, ax2, ax3 = axes[0], axes[1], axes[2]
        else:
            ax1, ax2, ax3 = axes[0, i], axes[1, i], axes[2, i]
        
        if data['enhanced'] is not None:
            ax1.imshow(data['enhanced'], cmap='gray')
        else:
            ax1.imshow(data['image'], cmap='gray')
        ax1.set_title(method_name, fontsize=9)
        ax1.axis('off')
        
        vis = visualize_singularities(data['image'], data['cores'], data['deltas'], show_confidence=True)
        ax2.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"C:{len(data['cores'])}, D:{len(data['deltas'])}", fontsize=9)
        ax2.axis('off')
        
        if data['poincare'] is not None:
            poincare_vis = visualize_poincare_map(data['poincare'])
            ax3.imshow(cv2.cvtColor(poincare_vis, cv2.COLOR_BGR2RGB))
            ax3.set_title('Poincare指数', fontsize=9)
        else:
            ax3.imshow(data['coh'], cmap='hot')
            ax3.set_title('方向一致性', fontsize=9)
        ax3.axis('off')
    
    plt.tight_layout()
    
    output_path = f'all_methods_{fp_type}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n对比结果已保存至: {output_path}")
    plt.close()
    
    return results


def compare_wet_dry_conditions(original_image, fp_type, dl_model_path=None):
    print("\n" + "=" * 70)
    print(f"指纹类型: {fp_type} - 湿润/干燥条件测试")
    print("=" * 70)
    
    conditions = [
        ('原始', original_image),
        ('湿润', apply_wet_effect(original_image, severity=0.7)),
        ('干燥', apply_dry_effect(original_image, severity=0.7)),
        ('模糊', apply_blur_effect(original_image, severity=0.6)),
        ('噪声', apply_noise_effect(original_image, severity=0.6)),
    ]
    
    results = {}
    
    fig, axes = plt.subplots(2, len(conditions), figsize=(4 * len(conditions), 8))
    
    for i, (cond_name, img) in enumerate(conditions):
        print(f"\n处理: {cond_name}")
        
        theta, coh, cores, deltas, poincare, conf, enhanced = process_fingerprint_robust(
            img, use_multiscale=True, use_enhancement=True, 
            use_deep_learning=TORCH_AVAILABLE, dl_model_path=dl_model_path
        )
        
        results[cond_name] = (cores, deltas)
        
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'{cond_name}')
        axes[0, i].axis('off')
        
        vis = visualize_singularities(img, cores, deltas, show_confidence=True)
        axes[1, i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f'C:{len(cores)}, D:{len(deltas)}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    output_path = f'conditions_{fp_type}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n条件测试结果已保存至: {output_path}")
    plt.close()
    
    return results


def main():
    print("=" * 70)
    print("指纹奇异点检测 - 完整方法对比")
    print("=" * 70)
    
    if TORCH_AVAILABLE:
        print("PyTorch可用: 深度学习方法已启用")
    else:
        print("PyTorch不可用: 深度学习方法已禁用")
        print("  安装PyTorch: pip install torch torchvision")
    
    fingerprint_types = ['loop_right', 'loop_left', 'whorl', 'arch']
    
    print("\n生成测试指纹并运行方法对比...")
    
    all_results = {}
    
    for fp_type in fingerprint_types:
        print(f"\n{'='*70}")
        print(f"处理指纹类型: {fp_type}")
        print(f"{'='*70}")
        
        original = generate_synthetic_fingerprint(size=(256, 256), fingerprint_type=fp_type)
        low_quality = create_low_quality_fingerprint(original, blur_kernel=7, noise_level=0.08)
        
        results = compare_all_methods(original, low_quality, fp_type)
        all_results[fp_type] = results
        
        compare_wet_dry_conditions(original, fp_type)
    
    print("\n" + "=" * 70)
    print("检测结果统计:")
    print("=" * 70)
    
    method_names = list(all_results[fingerprint_types[0]].keys())
    
    print(f"{'指纹类型':<15} {'方法':<25} {'Core':<6} {'Delta':<6}")
    print("-" * 60)
    
    for fp_type in fingerprint_types:
        print(f"\n{fp_type:<15}")
        for method_name in method_names:
            cores, deltas = all_results[fp_type][method_name]
            print(f"{'':<15} {method_name:<25} {len(cores):<6} {len(deltas):<6}")
    
    print("\n" + "=" * 70)
    print("处理完成! 请查看生成的 PNG 图像文件")
    print("=" * 70)


def process_custom_image(image_path, dl_model_path=None):
    print(f"加载图像: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("无法加载图像!")
        return
    
    print(f"图像尺寸: {image.shape}")
    
    low_quality = create_low_quality_fingerprint(image, blur_kernel=5, noise_level=0.05)
    
    results = compare_all_methods(image, low_quality, 'custom', dl_model_path)
    
    print("\n检测结果:")
    for method_name, (cores, deltas) in results.items():
        print(f"  {method_name}: {len(cores)}个Core, {len(deltas)}个Delta")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        dl_model_path = sys.argv[2] if len(sys.argv) > 2 else None
        process_custom_image(image_path, dl_model_path)
    else:
        main()
