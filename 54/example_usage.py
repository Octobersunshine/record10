import numpy as np
from tensor_decomposition import TensorDecomposer


def example_manual_rank_problems():
    print("=" * 70)
    print("示例1: 手动选择秩的问题")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    shape = (5, 6, 7)
    tensor = decomposer.create_random_tensor(shape)
    
    print(f"\n张量形状: {shape}")
    print("\n不同秩的效果对比:")
    print("-" * 70)
    print(f"{'秩':>4} | {'RMSE':>10} | {'过拟合分数':>12} | {'状态':>10}")
    print("-" * 70)
    
    for rank in [1, 2, 3, 4, 5, 6, 7]:
        factors = decomposer.cp_decomposition(tensor, rank)
        reconstructed = decomposer.reconstruct_from_cp(factors)
        rmse = decomposer.calculate_rmse(np.array(tensor), reconstructed)
        overfit_info = decomposer._detect_overfitting(factors[1])
        
        status = "过拟合!" if overfit_info['is_overfitting'] else "正常"
        print(f"{rank:4d} | {rmse:10.6f} | {overfit_info['avg_correlation_score']:14.3f} | {status:>10}")
    
    print("\n问题分析:")
    print("  ✗ 秩=1~2: 误差太大，欠拟合")
    print("  ✓ 秩=3~4: 误差适中，拟合良好")
    print("  ✗ 秩=5~7: 过拟合，因子矩阵列相关")
    print()


def example_auto_rank_selection():
    print("=" * 70)
    print("示例2: 自动选择最佳秩 (肘部法则)")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    shape = (5, 6, 7)
    tensor = decomposer.create_random_tensor(shape)
    
    print(f"\n张量形状: {shape}")
    
    optimal_rank, results = decomposer.select_optimal_rank(
        tensor, min_rank=1, max_rank=7, method='elbow', decomposition_method='cp', verbose=True
    )
    
    print(f"\n使用最佳秩 {optimal_rank} 进行分解:")
    factors = decomposer.cp_decomposition(tensor, optimal_rank)
    reconstructed = decomposer.reconstruct_from_cp(factors)
    rmse = decomposer.calculate_rmse(np.array(tensor), reconstructed)
    
    print(f"  重构误差 RMSE: {rmse:.6f}")
    
    weights, factor_matrices = factors
    print(f"\n  权重向量: {weights}")
    for i, fm in enumerate(factor_matrices):
        print(f"  模式 {i} 因子矩阵形状: {fm.shape}")
    print()


def example_cp_decomposition_auto():
    print("=" * 70)
    print("示例3: 一键自动CP分解")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    shape = (4, 5, 6)
    tensor = decomposer.create_random_tensor(shape)
    
    print(f"\n张量形状: {shape}")
    print("调用 cp_decomposition_auto() 自动完成秩选择和分解...\n")
    
    factors, optimal_rank, results = decomposer.cp_decomposition_auto(
        tensor, min_rank=1, max_rank=5, method='elbow', verbose=True
    )
    
    print(f"\n分解完成! 最佳秩 = {optimal_rank}")
    reconstructed = decomposer.reconstruct_from_cp(factors)
    rmse = decomposer.calculate_rmse(np.array(tensor), reconstructed)
    print(f"最终重构误差 RMSE: {rmse:.6f}")
    print()


def example_tensor_ring_basic():
    print("=" * 70)
    print("示例4: 张量环分解基础")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    shape = (4, 5, 6)
    tensor = decomposer.create_random_tensor(shape)
    
    print(f"\n张量形状: {shape}")
    
    tr_rank = 3
    tr_factors = decomposer.tensor_ring_decomposition(tensor, tr_rank)
    
    print(f"\n张量环分解秩: {tr_rank}")
    for i, f in enumerate(tr_factors):
        print(f"  模式 {i} 因子形状: {f.shape}")
    
    tr_recon = decomposer.reconstruct_from_tensor_ring(tr_factors)
    tr_rmse = decomposer.calculate_rmse(np.array(tensor), tr_recon)
    print(f"\n重构误差 RMSE: {tr_rmse:.6f}")
    print()


def example_tensor_ring_high_order():
    print("=" * 70)
    print("示例5: 高阶张量环分解 (5维)")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    shape = (3, 4, 5, 6, 7)
    tensor = decomposer.create_random_tensor(shape)
    
    print(f"\n5维张量形状: {shape}")
    print(f"原始张量元素数量: {np.prod(shape)}")
    
    tr_rank = 3
    tr_factors = decomposer.tensor_ring_decomposition(tensor, tr_rank)
    
    print(f"\n张量环分解秩: {tr_rank}")
    for i, f in enumerate(tr_factors):
        print(f"  模式 {i} 因子形状: {f.shape}")
    
    tr_params = sum(f.size for f in tr_factors)
    print(f"\n张量环参数数量: {tr_params}")
    print(f"压缩率: {(1 - tr_params / np.prod(shape)) * 100:.2f}%")
    
    tr_recon = decomposer.reconstruct_from_tensor_ring(tr_factors)
    tr_rmse = decomposer.calculate_rmse(np.array(tensor), tr_recon)
    print(f"重构误差 RMSE: {tr_rmse:.6f}")
    print()


def example_tensor_ring_auto():
    print("=" * 70)
    print("示例6: 张量环自动秩选择")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    shape = (4, 5, 6)
    tensor = decomposer.create_random_tensor(shape)
    
    print(f"\n张量形状: {shape}")
    
    factors, optimal_rank, results = decomposer.tensor_ring_decomposition_auto(
        tensor, min_rank=2, max_rank=6, method='elbow', verbose=True
    )
    
    print(f"\n张量环分解完成! 最佳秩 = {optimal_rank}")
    reconstructed = decomposer.reconstruct_from_tensor_ring(factors)
    rmse = decomposer.calculate_rmse(np.array(tensor), reconstructed)
    print(f"最终重构误差 RMSE: {rmse:.6f}")
    print()


def example_decomposition_comparison():
    print("=" * 70)
    print("示例7: 三种分解方法比较")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    
    print("\n3维张量比较:")
    shape_3d = (5, 6, 7)
    tensor_3d = decomposer.create_random_tensor(shape_3d)
    print(f"张量形状: {shape_3d}")
    decomposer.compare_decompositions(tensor_3d, cp_rank=4, tr_rank=4, tucker_ranks=[4, 4, 4])
    
    print("\n5维张量比较 (TR优势场景):")
    shape_5d = (3, 4, 5, 6, 7)
    tensor_5d = decomposer.create_random_tensor(shape_5d)
    print(f"张量形状: {shape_5d}")
    decomposer.compare_decompositions(tensor_5d, cp_rank=3, tr_rank=3, tucker_ranks=[3, 3, 3, 3, 3])
    print()


def example_robustness_test():
    print("=" * 70)
    print("示例8: 鲁棒性检验 - 噪声敏感性分析")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    shape = (4, 5, 6)
    tensor = decomposer.create_random_tensor(shape)
    
    print(f"\n张量形状: {shape}")
    
    print("\n1. CP分解鲁棒性检验:")
    cp_robustness = decomposer.robustness_test(
        tensor, 
        decomposition_method='cp',
        ranks=[3, 5],
        noise_levels=[0.0, 0.05, 0.1, 0.15],
        n_trials=3,
        verbose=True
    )
    
    print("\n2. 张量环分解鲁棒性检验:")
    tr_robustness = decomposer.robustness_test(
        tensor, 
        decomposition_method='tr',
        ranks=[3, 5],
        noise_levels=[0.0, 0.05, 0.1, 0.15],
        n_trials=3,
        verbose=True
    )
    print()


def example_noise_addition():
    print("=" * 70)
    print("示例9: 带噪声张量的分解")
    print("=" * 70)
    
    decomposer = TensorDecomposer()
    shape = (4, 5, 6)
    
    clean_tensor = decomposer.create_random_tensor(shape)
    noisy_tensor = decomposer.add_noise(clean_tensor, noise_level=0.1)
    
    print(f"\n张量形状: {shape}")
    print(f"噪声水平: 10%")
    
    noise_rmse = decomposer.calculate_rmse(np.array(clean_tensor), np.array(noisy_tensor))
    print(f"添加噪声后RMSE: {noise_rmse:.6f}")
    
    tr_rank = 4
    tr_factors = decomposer.tensor_ring_decomposition(noisy_tensor, tr_rank)
    tr_recon = decomposer.reconstruct_from_tensor_ring(tr_factors)
    tr_rmse = decomposer.calculate_rmse(np.array(clean_tensor), tr_recon)
    
    print(f"\n张量环分解去噪效果:")
    print(f"  分解秩: {tr_rank}")
    print(f"  去噪后RMSE: {tr_rmse:.6f}")
    print(f"  去噪效果: {(1 - tr_rmse / noise_rmse) * 100:.2f}% 噪声抑制")
    print()


def summary():
    print("=" * 70)
    print("功能总结")
    print("=" * 70)
    
    print("\n【分解方法】")
    print("  1. CP分解 (CANDECOMP/PARAFAC)")
    print("     - 适合低维张量，解释性强")
    print("     - cp_decomposition(), cp_decomposition_auto()")
    
    print("\n  2. Tucker分解")
    print("     - 核心张量 + 因子矩阵")
    print("     - tucker_decomposition()")
    
    print("\n  3. 张量环分解 (Tensor Ring) ✨")
    print("     - 适合高阶张量 (5维以上)")
    print("     - 参数数量随阶数线性增长")
    print("     - 更好的数值稳定性")
    print("     - tensor_ring_decomposition(), tensor_ring_decomposition_auto()")
    
    print("\n【秩选择方法】")
    print("  1. 肘部法则 - 误差曲线拐点")
    print("  2. 交叉验证 - 泛化能力评估")
    
    print("\n【鲁棒性检验】")
    print("  - 噪声敏感性分析")
    print("  - 多组噪声水平测试")
    print("  - 不同秩的稳定性比较")
    print("  - robustness_test()")
    
    print("\n【方法选择指南】")
    print("  3维及以下: CP/Tucker")
    print("  4-5维: Tucker/TR")
    print("  5维以上: 张量环分解 (TR) ✨")
    print("  有噪声数据: 秩保守选择 + TR")
    print("=" * 70)


if __name__ == "__main__":
    example_manual_rank_problems()
    example_auto_rank_selection()
    example_cp_decomposition_auto()
    example_tensor_ring_basic()
    example_tensor_ring_high_order()
    example_tensor_ring_auto()
    example_decomposition_comparison()
    example_robustness_test()
    example_noise_addition()
    summary()
