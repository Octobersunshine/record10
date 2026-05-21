import numpy as np
import matplotlib.pyplot as plt
from dtw import DTW, FastDTW, OnlineDTW
from gesture_processor import HandKeypointProcessor, GestureDataAugmenter, smooth_sequence
from gesture_recognizer import GestureRecognizer, RealTimeGestureRecognizer, StreamingGestureRecognizer, create_sample_gestures


def demo_dimension_imbalance_problem():
    print("=" * 60)
    print("问题演示：维度不均衡导致的距离计算偏差")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_dim = 10
    n_samples = 100
    
    seq1 = np.random.randn(n_samples, n_dim)
    seq2 = np.random.randn(n_samples, n_dim)
    
    scales = np.array([100.0, 50.0, 10.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    seq1_scaled = seq1 * scales
    seq2_scaled = seq2 * scales
    
    dtw_unscaled = DTW(distance_func='euclidean', use_normalization=False)
    dist_unscaled, _, _ = dtw_unscaled.compute(seq1, seq2)
    print(f"\n无尺度差异时的 DTW 距离: {dist_unscaled:.4f}")
    
    dtw_scaled_no_norm = DTW(distance_func='euclidean', use_normalization=False)
    dist_scaled_no_norm, _, _ = dtw_scaled_no_norm.compute(seq1_scaled, seq2_scaled)
    print(f"有尺度差异但无归一化时的 DTW 距离: {dist_scaled_no_norm:.4f}")
    
    dtw_scaled_with_norm = DTW(distance_func='euclidean', use_normalization=True)
    dist_scaled_with_norm, _, _ = dtw_scaled_with_norm.compute(seq1_scaled, seq2_scaled)
    print(f"有尺度差异且有归一化时的 DTW 距离: {dist_scaled_with_norm:.4f}")
    
    print("\n各维度的标准差:")
    for i, scale in enumerate(scales):
        print(f"  维度 {i}: 尺度因子 = {scale:.1f}")
    
    print("\n结论：归一化有效解决了大尺度维度主导距离计算的问题！")
    print()


def demo_weighted_distance():
    print("=" * 60)
    print("加权距离演示：根据维度重要性调整权重")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_samples = 50
    n_dim = 6
    
    seq1 = np.random.randn(n_samples, n_dim)
    seq2 = np.random.randn(n_samples, n_dim)
    
    for i in range(3):
        seq2[:, i] = seq1[:, i] + np.random.randn(n_samples) * 0.1
    
    for i in range(3, 6):
        seq2[:, i] = seq1[:, i] + np.random.randn(n_samples) * 2.0
    
    dtw_unweighted = DTW(distance_func='euclidean', use_normalization=True, weights=None)
    dist_unweighted, _, _ = dtw_unweighted.compute(seq1, seq2)
    print(f"\n等权重 DTW 距离: {dist_unweighted:.4f}")
    
    weights = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
    dtw_weighted = DTW(distance_func='euclidean', use_normalization=True, weights=weights)
    dist_weighted, _, _ = dtw_weighted.compute(seq1, seq2)
    print(f"加权 DTW 距离 (前3维权重更高): {dist_weighted:.4f}")
    
    print("\n维度差异分析:")
    for i in range(n_dim):
        diff = np.mean(np.abs(seq1[:, i] - seq2[:, i]))
        print(f"  维度 {i}: 平均差异 = {diff:.4f}")
    
    print("\n结论：通过加权可以让重要维度对距离计算产生更大影响！")
    print()


def demo_gesture_recognition_with_normalization():
    print("=" * 60)
    print("手势识别中的归一化效果演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    recognizer_vanilla = GestureRecognizer(
        distance_func='euclidean',
        use_dim_normalization=False,
        use_dim_weights=False
    )
    
    recognizer_improved = GestureRecognizer(
        distance_func='euclidean',
        use_dim_normalization=True,
        use_dim_weights=True,
        weight_type='inverse_variance'
    )
    
    sample_gestures = create_sample_gestures()
    
    for gesture_name, sequences in sample_gestures.items():
        recognizer_vanilla.add_template(gesture_name, sequences, augment=False)
        recognizer_improved.add_template(gesture_name, sequences, augment=False)
    
    print("\n模板已添加完成")
    print(f"已注册手势: {recognizer_vanilla.list_gestures()}")
    
    dim_stats = recognizer_improved.analyze_dimension_importance()
    print(f"\n特征维度统计:")
    print(f"  维度数量: {len(dim_stats.get('variance', []))}")
    print(f"  方差范围: [{np.min(dim_stats['variance']):.6f}, {np.max(dim_stats['variance']):.6f}]")
    print(f"  方差最大的前5个维度: {np.argsort(dim_stats['variance'])[-5:][::-1]}")
    
    if 'weights' in recognizer_improved.get_dimension_stats():
        weights = recognizer_improved.get_dimension_stats()['weights']
        if weights is not None:
            print(f"\n维度权重统计:")
            print(f"  权重范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
            print(f"  权重最大的前5个维度: {np.argsort(weights)[-5:][::-1]}")
    
    test_gesture = sample_gestures['swipe_right'][0].copy()
    
    n_dims = test_gesture.shape[1] * test_gesture.shape[2]
    scale_factors = np.ones(n_dims)
    scale_factors[:10] = 100.0
    
    test_gesture_flat = test_gesture.reshape(len(test_gesture), -1)
    test_gesture_scaled = test_gesture_flat * scale_factors
    test_gesture_scaled = test_gesture_scaled.reshape(test_gesture.shape)
    
    print("\n测试1：原始手势识别")
    results_vanilla = recognizer_vanilla.recognize(test_gesture)
    results_improved = recognizer_improved.recognize(test_gesture)
    print(f"  传统方法识别结果: {results_vanilla[0][0]} (距离: {results_vanilla[0][1]:.4f})")
    print(f"  改进方法识别结果: {results_improved[0][0]} (距离: {results_improved[0][1]:.4f})")
    
    print("\n测试2：尺度缩放后的手势识别")
    results_vanilla_scaled = recognizer_vanilla.recognize(test_gesture_scaled)
    results_improved_scaled = recognizer_improved.recognize(test_gesture_scaled)
    print(f"  传统方法识别结果: {results_vanilla_scaled[0][0]} (距离: {results_vanilla_scaled[0][1]:.4f})")
    print(f"  改进方法识别结果: {results_improved_scaled[0][0]} (距离: {results_improved_scaled[0][1]:.4f})")
    
    if results_vanilla_scaled[0][0] != 'swipe_right' and results_improved_scaled[0][0] == 'swipe_right':
        print("\n✓ 改进方法在尺度变化下仍能正确识别手势！")
    elif results_vanilla_scaled[0][0] == 'swipe_right' and results_improved_scaled[0][0] == 'swipe_right':
        print("\n✓ 两种方法都能正确识别，但改进方法的距离更稳定！")
    
    vanilla_distance_change = abs(results_vanilla_scaled[0][1] - results_vanilla[0][1]) / results_vanilla[0][1]
    improved_distance_change = abs(results_improved_scaled[0][1] - results_improved[0][1]) / results_improved[0][1]
    print(f"\n距离变化率:")
    print(f"  传统方法: {vanilla_distance_change*100:.1f}%")
    print(f"  改进方法: {improved_distance_change*100:.1f}%")
    
    if improved_distance_change < vanilla_distance_change:
        print("✓ 改进方法的距离计算更加稳定！")
    
    print()


def visualize_dimension_importance():
    print("=" * 60)
    print("维度重要性可视化")
    print("=" * 60)
    
    recognizer = GestureRecognizer(
        use_dim_normalization=True,
        use_dim_weights=True
    )
    
    sample_gestures = create_sample_gestures()
    for gesture_name, sequences in sample_gestures.items():
        recognizer.add_template(gesture_name, sequences, augment=False)
    
    dim_stats = recognizer.analyze_dimension_importance()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    n_dims = min(30, len(dim_stats['variance']))
    
    axes[0, 0].bar(range(n_dims), dim_stats['variance'][:n_dims])
    axes[0, 0].set_title('各维度方差')
    axes[0, 0].set_xlabel('维度索引')
    axes[0, 0].set_ylabel('方差')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(range(n_dims), dim_stats['range'][:n_dims])
    axes[0, 1].set_title('各维度数值范围')
    axes[0, 1].set_xlabel('维度索引')
    axes[0, 1].set_ylabel('范围 (max-min)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].bar(range(n_dims), dim_stats['importance'][:n_dims])
    axes[1, 0].set_title('各维度重要性 (方差占比)')
    axes[1, 0].set_xlabel('维度索引')
    axes[1, 0].set_ylabel('重要性')
    axes[1, 0].grid(True, alpha=0.3)
    
    weights = recognizer.get_dimension_stats().get('weights')
    if weights is not None:
        axes[1, 1].bar(range(n_dims), weights[:n_dims])
        axes[1, 1].set_title('各维度权重 (逆方差)')
        axes[1, 1].set_xlabel('维度索引')
        axes[1, 1].set_ylabel('权重')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dimension_importance.png', dpi=150, bbox_inches='tight')
    print("\n可视化图像已保存到: dimension_importance.png")
    plt.close()
    
    print()


def benchmark_normalization_effect():
    print("=" * 60)
    print("归一化效果基准测试")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_tests = 10
    n_gestures = 3
    n_samples_per_gesture = 5
    seq_length = 30
    n_dims = 63
    
    vanilla_correct = 0
    improved_correct = 0
    vanilla_distances = []
    improved_distances = []
    
    for test_idx in range(n_tests):
        recognizer_vanilla = GestureRecognizer(
            use_dim_normalization=False,
            use_dim_weights=False
        )
        recognizer_improved = GestureRecognizer(
            use_dim_normalization=True,
            use_dim_weights=True
        )
        
        train_gestures = {}
        test_gestures = {}
        
        for g in range(n_gestures):
            train_seqs = []
            for i in range(n_samples_per_gesture):
                base_seq = np.random.randn(seq_length, n_dims) * 0.1 + g * 0.5
                train_seqs.append(base_seq)
            train_gestures[f'gesture_{g}'] = train_seqs
            
            test_seq = np.random.randn(seq_length, n_dims) * 0.1 + g * 0.5
            scale_factors = np.random.rand(n_dims) * 100 + 1
            test_seq_scaled = test_seq * scale_factors
            test_gestures[f'gesture_{g}'] = test_seq_scaled
        
        for g_name, seqs in train_gestures.items():
            recognizer_vanilla.add_template(g_name, [s.reshape(len(s), 21, 3) for s in seqs], augment=False)
            recognizer_improved.add_template(g_name, [s.reshape(len(s), 21, 3) for s in seqs], augment=False)
        
        for true_label, test_seq in test_gestures.items():
            test_seq_reshaped = test_seq.reshape(len(test_seq), 21, 3)
            
            result_vanilla = recognizer_vanilla.recognize(test_seq_reshaped, top_k=1)
            result_improved = recognizer_improved.recognize(test_seq_reshaped, top_k=1)
            
            if result_vanilla[0][0] == true_label:
                vanilla_correct += 1
            if result_improved[0][0] == true_label:
                improved_correct += 1
            
            vanilla_distances.append(result_vanilla[0][1])
            improved_distances.append(result_improved[0][1])
    
    total_tests = n_tests * n_gestures
    
    print(f"\n测试结果 (共 {total_tests} 次测试):")
    print(f"  传统方法正确率: {vanilla_correct}/{total_tests} ({vanilla_correct/total_tests*100:.1f}%)")
    print(f"  改进方法正确率: {improved_correct}/{total_tests} ({improved_correct/total_tests*100:.1f}%)")
    print(f"\n平均距离:")
    print(f"  传统方法: {np.mean(vanilla_distances):.4f} ± {np.std(vanilla_distances):.4f}")
    print(f"  改进方法: {np.mean(improved_distances):.4f} ± {np.std(improved_distances):.4f}")
    print(f"\n距离变异系数:")
    print(f"  传统方法: {np.std(vanilla_distances)/np.mean(vanilla_distances)*100:.1f}%")
    print(f"  改进方法: {np.std(improved_distances)/np.mean(improved_distances)*100:.1f}%")
    
    if improved_correct > vanilla_correct:
        print("\n✓ 改进方法显著提升了识别准确率！")
    if np.std(improved_distances) < np.std(vanilla_distances):
        print("✓ 改进方法显著降低了距离计算的方差！")
    
    print()


def demo_online_dtw_basic():
    print("=" * 60)
    print("在线 DTW 基础演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    template_length = 30
    n_dims = 10
    
    template = np.random.randn(template_length, n_dims) * 0.1
    test_sequence = template + np.random.randn(template_length, n_dims) * 0.05
    
    online_dtw = OnlineDTW(
        template=template,
        distance_func='euclidean',
        use_normalization=False,
        early_stop_threshold=10.0,
        min_frames_for_early_stop=10
    )
    
    print(f"\n模板长度: {template_length}")
    print(f"特征维度: {n_dims}")
    print("\n逐帧处理进度:")
    
    distances = []
    progress_history = []
    
    for i, frame in enumerate(test_sequence):
        dist, completed, status = online_dtw.update(frame)
        distances.append(dist)
        progress_history.append(status['progress'])
        
        print(f"  帧 {i+1}/{len(test_sequence)}: 距离={dist:.4f}, 进度={status['progress']:.1f}%", end='')
        if completed:
            print(f" [匹配完成! 最终距离={dist:.4f}]")
        else:
            print()
    
    alignment_path = online_dtw.get_alignment_path()
    print(f"\n对齐路径长度: {len(alignment_path)}")
    print(f"路径前5个点: {alignment_path[:5]}")
    print(f"路径后5个点: {alignment_path[-5:]}")
    
    print()


def demo_streaming_gesture_recognition():
    print("=" * 60)
    print("流式手势识别演示（实时检测 + 早停）")
    print("=" * 60)
    
    np.random.seed(42)
    
    recognizer = GestureRecognizer(
        use_dim_normalization=True,
        use_dim_weights=True
    )
    
    sample_gestures = create_sample_gestures()
    for gesture_name, sequences in sample_gestures.items():
        recognizer.add_template(gesture_name, sequences, augment=False)
    
    print(f"\n已注册手势: {recognizer.list_gestures()}")
    
    streaming_recognizer = StreamingGestureRecognizer(
        base_recognizer=recognizer,
        confidence_threshold=0.7,
        min_frames_for_detection=10,
        max_frames_for_detection=60,
        smoothing_window=3
    )
    
    test_gesture_name = 'swipe_right'
    test_sequence = sample_gestures[test_gesture_name][0]
    print(f"\n测试手势: {test_gesture_name}")
    print(f"序列长度: {len(test_sequence)}")
    
    print("\n实时检测进度:")
    print("-" * 80)
    print(f"{'帧':<6} {'检测手势':<15} {'置信度':<12} {'距离':<12} {'状态':<20}")
    print("-" * 80)
    
    detection_history = []
    first_detection_frame = None
    
    for i, frame in enumerate(test_sequence):
        detected_gesture, confidence, result = streaming_recognizer.update(frame)
        
        status = "检测中..."
        if detected_gesture and first_detection_frame is None:
            first_detection_frame = i + 1
            status = f"✓ 检测到: {detected_gesture}"
        elif detected_gesture:
            status = f"✓ 已确认: {detected_gesture}"
        
        print(f"{i+1:<6} {result['best_gesture']:<15} {confidence:<12.3f} {result['best_distance']:<12.4f} {status}")
        
        detection_history.append({
            'frame': i + 1,
            'detected': detected_gesture,
            'confidence': confidence,
            'best_gesture': result['best_gesture'],
            'best_distance': result['best_distance']
        })
        
        if detected_gesture and i > len(test_sequence) // 2:
            break
    
    print("-" * 80)
    
    if first_detection_frame:
        print(f"\n✓ 成功检测到手势: {test_gesture_name}")
        print(f"  首次检测帧: {first_detection_frame}")
        print(f"  序列完成度: {first_detection_frame / len(test_sequence) * 100:.1f}%")
        print(f"  最终置信度: {detection_history[-1]['confidence']:.3f}")
        
        if first_detection_frame < len(test_sequence):
            print(f"  ✓ 早停成功: 提前 {len(test_sequence) - first_detection_frame} 帧完成检测!")
    else:
        print(f"\n✗ 未检测到手势，需要更多帧或降低置信度阈值")
    
    progress = streaming_recognizer.get_progress()
    print(f"\n各模板匹配进度:")
    for gesture_name, prog in progress.items():
        print(f"  {gesture_name}: {prog:.1f}%")
    
    print()


def visualize_streaming_detection():
    print("=" * 60)
    print("流式手势检测可视化")
    print("=" * 60)
    
    np.random.seed(42)
    
    recognizer = GestureRecognizer(
        use_dim_normalization=True,
        use_dim_weights=True
    )
    
    sample_gestures = create_sample_gestures()
    for gesture_name, sequences in sample_gestures.items():
        recognizer.add_template(gesture_name, sequences, augment=False)
    
    streaming_recognizer = StreamingGestureRecognizer(
        base_recognizer=recognizer,
        confidence_threshold=0.7,
        min_frames_for_detection=5,
        max_frames_for_detection=60
    )
    
    test_gesture_name = 'click'
    test_sequence = sample_gestures[test_gesture_name][0]
    
    history = {
        'frames': [],
        'confidences': {},
        'distances': {}
    }
    
    for frame in test_sequence:
        detected, confidence, result = streaming_recognizer.update(frame)
        
        history['frames'].append(len(history['frames']) + 1)
        for gesture_name, dist in result['all_distances'].items():
            if gesture_name not in history['distances']:
                history['distances'][gesture_name] = []
            history['distances'][gesture_name].append(dist)
        
        smoothed = streaming_recognizer.get_smoothed_confidence()
        for gesture_name, conf in smoothed.items():
            if gesture_name not in history['confidences']:
                history['confidences'][gesture_name] = []
            history['confidences'][gesture_name].append(conf)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    ax1 = axes[0, 0]
    for i, (gesture_name, distances) in enumerate(history['distances'].items()):
        ax1.plot(history['frames'], distances, label=gesture_name, color=colors[i % len(colors)], linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='距离阈值')
    ax1.set_xlabel('帧')
    ax1.set_ylabel('DTW 距离')
    ax1.set_title('各手势模板的实时匹配距离')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for i, (gesture_name, confidences) in enumerate(history['confidences'].items()):
        ax2.plot(history['frames'][-len(confidences):], confidences, label=gesture_name, color=colors[i % len(colors)], linewidth=2)
    ax2.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='置信度阈值')
    ax2.set_xlabel('帧')
    ax2.set_ylabel('置信度')
    ax2.set_title('各手势的实时检测置信度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    progress = streaming_recognizer.get_progress()
    gestures = list(progress.keys())
    y_pos = np.arange(len(gestures))
    progress_values = [progress[g] for g in gestures]
    bars = ax3.barh(y_pos, progress_values, color=[colors[i % len(colors)] for i in range(len(gestures))])
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(gestures)
    ax3.set_xlabel('匹配进度 (%)')
    ax3.set_title('各手势模板匹配进度')
    ax3.set_xlim(0, 100)
    for i, v in enumerate(progress_values):
        ax3.text(v + 1, i, f'{v:.1f}%', va='center')
    ax3.grid(True, alpha=0.3, axis='x')
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    info_text = f"""
    流式手势检测结果
    
    测试手势: {test_gesture_name}
    序列长度: {len(test_sequence)} 帧
    模板数量: {len(recognizer.list_gestures())} 个手势
    
    关键特性:
    ✓ 实时逐帧处理
    ✓ 早停机制 (距离阈值)
    ✓ 置信度平滑
    ✓ 多模板并行匹配
    
    使用说明:
    1. 每帧更新所有模板的匹配状态
    2. 置信度连续超过阈值则触发检测
    3. 距离过大时自动早停不匹配模板
    """
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig('streaming_detection.png', dpi=150, bbox_inches='tight')
    print("\n可视化图像已保存到: streaming_detection.png")
    plt.close()
    
    print()


def demo_early_stop_mechanism():
    print("=" * 60)
    print("早停机制演示 (Early Stop)")
    print("=" * 60)
    
    np.random.seed(42)
    
    template = np.random.randn(30, 10) * 0.1
    
    good_match = template + np.random.randn(30, 10) * 0.05
    bad_match = np.random.randn(30, 10) * 1.0
    
    print(f"\n模板长度: {len(template)}")
    print(f"\n测试1: 良好匹配序列 (早停不会触发)")
    print("-" * 60)
    
    online_dtw_good = OnlineDTW(
        template=template,
        early_stop_threshold=1.0,
        min_frames_for_early_stop=10
    )
    
    for i, frame in enumerate(good_match):
        dist, completed, status = online_dtw_good.update(frame)
        early_stop_triggered = completed and i < len(template) - 1
        if i < 30:
            print(f"  帧 {i+1}: 距离={dist:.4f}, 早停={'✓' if early_stop_triggered else '✗'}")
    
    print(f"\n  最终状态: {'完成匹配' if online_dtw_good.is_complete else '继续匹配'}")
    print(f"  最终距离: {online_dtw_good.final_distance:.4f}")
    
    print(f"\n测试2: 不匹配序列 (早停会触发)")
    print("-" * 60)
    
    online_dtw_bad = OnlineDTW(
        template=template,
        early_stop_threshold=0.5,
        min_frames_for_early_stop=10
    )
    
    early_stop_frame = None
    for i, frame in enumerate(bad_match):
        dist, completed, status = online_dtw_bad.update(frame)
        if completed and i < len(template) - 1:
            early_stop_frame = i + 1
        print(f"  帧 {i+1}: 距离={dist:.4f}, 早停={'✓ 触发!' if completed and i < len(template)-1 else '✗'}")
        if completed:
            break
    
    if early_stop_frame:
        print(f"\n  ✓ 早停成功! 在第 {early_stop_frame} 帧终止匹配")
        print(f"    节省了 {len(template) - early_stop_frame} 帧的计算量")
    else:
        print(f"\n  未触发早停，完成了所有帧的匹配")
    
    print()


def main():
    print("\n" + "=" * 60)
    print("基于 DTW 的手势识别系统 - 完整功能演示")
    print("=" * 60 + "\n")
    
    demo_dimension_imbalance_problem()
    
    demo_weighted_distance()
    
    demo_gesture_recognition_with_normalization()
    
    demo_online_dtw_basic()
    
    demo_streaming_gesture_recognition()
    
    demo_early_stop_mechanism()
    
    visualize_dimension_importance()
    
    visualize_streaming_detection()
    
    benchmark_normalization_effect()
    
    print("=" * 60)
    print("演示完成!")
    print("=" * 60)
    print("\n核心功能总结:")
    print("\n【第一部分 - 距离计算优化】")
    print("1. Z-score 归一化 - 消除不同维度的尺度差异")
    print("2. 逆方差加权 - 降低高方差维度的主导作用")
    print("3. 维度重要性分析 - 自动计算各维度权重")
    print("4. 模板级归一化参数 - 基于所有模板数据计算统计量")
    print("\n【第二部分 - 流式/在线识别】")
    print("5. OnlineDTW - 逐帧增量计算，支持流式处理")
    print("6. StreamingGestureRecognizer - 实时手势检测")
    print("7. 早停机制 - 距离过大时自动终止匹配，节省计算")
    print("8. 置信度平滑 - 基于历史帧平滑输出，减少抖动")
    print("9. 多模板并行匹配 - 同时与所有手势模板匹配")
    print("\n使用建议:")
    print("- use_dim_normalization=True (推荐)")
    print("- use_dim_weights=True (推荐)")
    print("- confidence_threshold=0.7 (默认)")
    print("- min_frames_for_detection=10 (防止误检)")


if __name__ == "__main__":
    main()
