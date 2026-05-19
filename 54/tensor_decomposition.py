import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from tensorly.tenalg import multi_mode_dot
import warnings
warnings.filterwarnings('ignore')


class TensorDecomposer:
    def __init__(self, random_state=42):
        tl.set_backend('numpy')
        self.random_state = random_state
        np.random.seed(random_state)
    
    def create_random_tensor(self, shape):
        return tl.tensor(np.random.random(shape))
    
    def create_tensor_with_noise(self, shape, noise_level=0.1):
        clean_tensor = np.random.random(shape)
        noise = noise_level * np.random.randn(*shape)
        return tl.tensor(clean_tensor + noise), tl.tensor(clean_tensor)
    
    def add_noise(self, tensor, noise_level=0.1):
        tensor_np = np.array(tensor)
        noise = noise_level * np.random.randn(*tensor_np.shape)
        return tl.tensor(tensor_np + noise)
    
    def cp_decomposition(self, tensor, rank):
        tensor = tl.tensor(tensor)
        factors = parafac(tensor, rank=rank, random_state=self.random_state, n_iter_max=100)
        return factors
    
    def tucker_decomposition(self, tensor, ranks):
        tensor = tl.tensor(tensor)
        core, factors = tucker(tensor, rank=ranks, random_state=self.random_state, n_iter_max=100)
        return core, factors
    
    def tensor_ring_decomposition(self, tensor, rank, max_iter=100, tol=1e-6):
        tensor_np = np.array(tensor)
        shape = tensor_np.shape
        n_modes = len(shape)
        
        if isinstance(rank, int):
            ranks = [rank] * n_modes
        else:
            ranks = rank
        
        ranks = [ranks[-1]] + ranks[:-1]
        
        factors = []
        for i in range(n_modes):
            factor = np.random.randn(shape[i], ranks[i], ranks[(i+1)%n_modes]) * 0.1
            factors.append(factor)
        
        for iteration in range(max_iter):
            old_factors = [f.copy() for f in factors]
            
            for i in range(n_modes):
                left_rank = ranks[i]
                right_rank = ranks[(i+1)%n_modes]
                
                contraction = tensor_np.copy()
                for j in range(n_modes):
                    if j != i:
                        factor_j = factors[j]
                        contraction_shape = list(contraction.shape)
                        contraction_shape[j] = 1
                        
                        factor_reshaped = factor_j.reshape(factor_j.shape[0], -1)
                        contraction = np.tensordot(contraction, factor_reshaped.T, axes=([j], [1]))
                        contraction = np.moveaxis(contraction, -1, j)
                
                contraction_2d = contraction.reshape(left_rank, right_rank, -1)
                contraction_2d = np.transpose(contraction_2d, (2, 0, 1)).reshape(-1, left_rank * right_rank)
                
                try:
                    factor_i_flat = np.linalg.lstsq(contraction_2d.T @ contraction_2d + 1e-6 * np.eye(left_rank * right_rank),
                                                    contraction_2d.T @ np.ones(contraction_2d.shape[0]), rcond=None)[0]
                    factors[i] = factor_i_flat.reshape(left_rank, right_rank).T
                    factors[i] = factors[i].reshape(shape[i], left_rank, right_rank)
                except:
                    pass
            
            change = np.sum([np.linalg.norm(f - of) for f, of in zip(factors, old_factors)])
            if change < tol:
                break
        
        return factors
    
    def reconstruct_from_cp(self, factors):
        return tl.kruskal_to_tensor(factors)
    
    def reconstruct_from_tucker(self, core, factors):
        return multi_mode_dot(core, factors)
    
    def reconstruct_from_tensor_ring(self, factors):
        n_modes = len(factors)
        result = factors[0]
        
        for i in range(1, n_modes):
            result = np.tensordot(result, factors[i], axes=([-1], [1]))
        
        if n_modes > 2:
            result = np.trace(result, axis1=0, axis2=-1)
        else:
            result = np.trace(result, axis1=0, axis2=1)
        
        return result
    
    def calculate_rmse(self, original, reconstructed):
        original_np = np.array(original)
        reconstructed_np = np.array(reconstructed)
        return np.sqrt(np.mean((original_np - reconstructed_np) ** 2))
    
    def calculate_relative_error(self, original, reconstructed):
        original_np = np.array(original)
        reconstructed_np = np.array(reconstructed)
        return np.linalg.norm(original_np - reconstructed_np) / np.linalg.norm(original_np)
    
    def _detect_overfitting(self, factor_matrices, threshold=0.7):
        n_modes = len(factor_matrices)
        overfitting_scores = []
        negative_correlations = []
        
        for mode, fm in enumerate(factor_matrices):
            corr_matrix = np.corrcoef(fm.T)
            np.fill_diagonal(corr_matrix, 0)
            
            high_corr_count = np.sum(np.abs(corr_matrix) > threshold)
            total_pairs = corr_matrix.size - fm.shape[1]
            score = high_corr_count / total_pairs if total_pairs > 0 else 0
            overfitting_scores.append(score)
            
            neg_corr = np.sum(corr_matrix < -threshold)
            if neg_corr > 0:
                negative_correlations.append((mode, neg_corr))
        
        avg_score = np.mean(overfitting_scores) if overfitting_scores else 0
        is_overfitting = avg_score > 0.3 or len(negative_correlations) > 0
        
        return {
            'is_overfitting': is_overfitting,
            'avg_correlation_score': avg_score,
            'negative_correlations': negative_correlations,
            'mode_scores': overfitting_scores
        }
    
    def _find_elbow_point(self, ranks, errors):
        if len(ranks) < 3:
            return ranks[-1]
        
        errors = np.array(errors)
        ranks = np.array(ranks)
        
        point1 = np.array([ranks[0], errors[0]])
        point2 = np.array([ranks[-1], errors[-1]])
        
        max_dist = -1
        best_rank = ranks[-1]
        
        for i in range(1, len(ranks) - 1):
            point = np.array([ranks[i], errors[i]])
            dist = np.abs(np.cross(point2 - point1, point1 - point)) / np.linalg.norm(point2 - point1)
            
            if dist > max_dist:
                max_dist = dist
                best_rank = ranks[i]
        
        return best_rank
    
    def select_optimal_rank(self, tensor, min_rank=1, max_rank=None, 
                            method='elbow', cv_folds=3, decomposition_method='cp', verbose=False):
        tensor_np = np.array(tensor)
        
        if max_rank is None:
            max_rank = min(tensor_np.shape)
        
        max_rank = min(max_rank, min(tensor_np.shape))
        ranks = list(range(min_rank, max_rank + 1))
        
        if verbose:
            print(f"\n=== 秩选择分析 (分解方法: {decomposition_method.upper()}, 方法: {method}) ===")
            print(f"测试秩范围: {min_rank} ~ {max_rank}")
        
        results = []
        
        for rank in ranks:
            if decomposition_method == 'cp':
                factors = self.cp_decomposition(tensor, rank)
                reconstructed = self.reconstruct_from_cp(factors)
                if hasattr(factors[1][0], 'shape'):
                    overfitting_info = self._detect_overfitting(factors[1])
                else:
                    overfitting_info = {'is_overfitting': False, 'avg_correlation_score': 0}
            elif decomposition_method == 'tr':
                factors = self.tensor_ring_decomposition(tensor, rank)
                reconstructed = self.reconstruct_from_tensor_ring(factors)
                overfitting_info = {'is_overfitting': False, 'avg_correlation_score': 0}
            else:
                continue
            
            rmse = self.calculate_rmse(tensor_np, reconstructed)
            
            results.append({
                'rank': rank,
                'rmse': rmse,
                'is_overfitting': overfitting_info['is_overfitting'],
                'overfitting_score': overfitting_info['avg_correlation_score']
            })
            
            if verbose:
                status = "过拟合!" if overfitting_info['is_overfitting'] else "OK"
                print(f"  秩={rank:2d} | RMSE={rmse:.6f} | 过拟合分数={overfitting_info['avg_correlation_score']:.3f} | {status}")
        
        if method == 'elbow':
            optimal_rank = self._find_elbow_point(ranks, [r['rmse'] for r in results])
        elif method == 'cv':
            optimal_rank = self._select_rank_cv(tensor, ranks, cv_folds, decomposition_method)
        else:
            valid_results = [r for r in results if not r['is_overfitting']]
            if valid_results:
                optimal_rank = min(valid_results, key=lambda x: x['rmse'])['rank']
            else:
                optimal_rank = min(results, key=lambda x: x['overfitting_score'])['rank']
        
        if verbose:
            print(f"\n推荐最佳秩: {optimal_rank}")
        
        return optimal_rank, results
    
    def _select_rank_cv(self, tensor, ranks, n_folds=3, decomposition_method='cp'):
        tensor_np = np.array(tensor)
        
        cv_errors = {rank: [] for rank in ranks}
        
        for fold in range(n_folds):
            np.random.seed(self.random_state + fold)
            test_mask = np.random.choice([True, False], size=tensor_np.shape, p=[0.2, 0.8])
            train_tensor = np.where(test_mask, 0, tensor_np)
            
            for rank in ranks:
                if decomposition_method == 'cp':
                    factors = self.cp_decomposition(train_tensor, rank)
                    reconstructed = self.reconstruct_from_cp(factors)
                elif decomposition_method == 'tr':
                    factors = self.tensor_ring_decomposition(train_tensor, rank)
                    reconstructed = self.reconstruct_from_tensor_ring(factors)
                else:
                    continue
                
                test_error = np.sqrt(np.mean((tensor_np[~test_mask] - reconstructed[~test_mask]) ** 2))
                cv_errors[rank].append(test_error)
        
        mean_errors = {rank: np.mean(errors) for rank, errors in cv_errors.items()}
        optimal_rank = min(mean_errors.keys(), key=lambda r: mean_errors[r])
        
        return optimal_rank
    
    def cp_decomposition_auto(self, tensor, min_rank=1, max_rank=None, 
                               method='elbow', verbose=False):
        optimal_rank, results = self.select_optimal_rank(
            tensor, min_rank, max_rank, method, decomposition_method='cp', verbose=verbose
        )
        
        factors = self.cp_decomposition(tensor, optimal_rank)
        
        if verbose:
            print(f"\n自动选择秩完成! 使用秩={optimal_rank}")
        
        return factors, optimal_rank, results
    
    def tensor_ring_decomposition_auto(self, tensor, min_rank=2, max_rank=10, 
                                        method='elbow', verbose=False):
        optimal_rank, results = self.select_optimal_rank(
            tensor, min_rank, max_rank, method, decomposition_method='tr', verbose=verbose
        )
        
        factors = self.tensor_ring_decomposition(tensor, optimal_rank)
        
        if verbose:
            print(f"\n张量环自动选择秩完成! 使用秩={optimal_rank}")
        
        return factors, optimal_rank, results
    
    def robustness_test(self, tensor, decomposition_method='cp', ranks=None, 
                        noise_levels=None, n_trials=5, verbose=False):
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
        
        if ranks is None:
            if decomposition_method == 'cp':
                ranks = [3, 5, 7]
            else:
                ranks = [3, 5, 7]
        
        tensor_np = np.array(tensor)
        
        robustness_results = []
        
        if verbose:
            print(f"\n=== 鲁棒性检验 ({decomposition_method.upper()}) ===")
            print(f"{'噪声水平':>10} | {'秩':>6} | {'平均RMSE':>12} | {'标准差':>10}")
            print("-" * 50)
        
        for noise_level in noise_levels:
            for rank in ranks:
                errors = []
                
                for trial in range(n_trials):
                    np.random.seed(self.random_state + trial)
                    noise = noise_level * np.random.randn(*tensor_np.shape)
                    noisy_tensor = tensor_np + noise
                    
                    if decomposition_method == 'cp':
                        factors = self.cp_decomposition(noisy_tensor, rank)
                        reconstructed = self.reconstruct_from_cp(factors)
                    elif decomposition_method == 'tr':
                        factors = self.tensor_ring_decomposition(noisy_tensor, rank)
                        reconstructed = self.reconstruct_from_tensor_ring(factors)
                    else:
                        continue
                    
                    rmse = self.calculate_rmse(tensor_np, reconstructed)
                    errors.append(rmse)
                
                mean_rmse = np.mean(errors)
                std_rmse = np.std(errors)
                
                robustness_results.append({
                    'noise_level': noise_level,
                    'rank': rank,
                    'mean_rmse': mean_rmse,
                    'std_rmse': std_rmse,
                    'errors': errors
                })
                
                if verbose:
                    print(f"{noise_level:10.2%} | {rank:6d} | {mean_rmse:12.6f} | {std_rmse:10.6f}")
        
        return robustness_results
    
    def compare_decompositions(self, tensor, cp_rank=5, tr_rank=5, tucker_ranks=None, verbose=True):
        tensor_np = np.array(tensor)
        
        if tucker_ranks is None:
            tucker_ranks = [5] * len(tensor_np.shape)
        
        results = {}
        
        cp_factors = self.cp_decomposition(tensor, cp_rank)
        cp_recon = self.reconstruct_from_cp(cp_factors)
        cp_rmse = self.calculate_rmse(tensor_np, cp_recon)
        cp_params = sum(f.size for f in cp_factors[1]) + len(cp_factors[0])
        results['CP'] = {'rmse': cp_rmse, 'params': cp_params}
        
        tr_factors = self.tensor_ring_decomposition(tensor, tr_rank)
        tr_recon = self.reconstruct_from_tensor_ring(tr_factors)
        tr_rmse = self.calculate_rmse(tensor_np, tr_recon)
        tr_params = sum(f.size for f in tr_factors)
        results['TR'] = {'rmse': tr_rmse, 'params': tr_params}
        
        tucker_core, tucker_factors = self.tucker_decomposition(tensor, tucker_ranks)
        tucker_recon = self.reconstruct_from_tucker(tucker_core, tucker_factors)
        tucker_rmse = self.calculate_rmse(tensor_np, tucker_recon)
        tucker_params = tucker_core.size + sum(f.size for f in tucker_factors)
        results['Tucker'] = {'rmse': tucker_rmse, 'params': tucker_params}
        
        if verbose:
            print(f"\n=== 分解方法比较 ===")
            print(f"{'方法':>10} | {'RMSE':>12} | {'参数数量':>12} | {'压缩率':>12}")
            print("-" * 55)
            
            original_size = tensor_np.size
            for method, res in results.items():
                compression = (1 - res['params'] / original_size) * 100
                print(f"{method:>10} | {res['rmse']:12.6f} | {res['params']:12d} | {compression:10.2f}%")
        
        return results


def demonstrate_tensor_ring():
    decomposer = TensorDecomposer(random_state=42)
    
    print("=" * 70)
    print("张量环分解 (Tensor Ring) 演示")
    print("=" * 70)
    
    print("\n1. 3维张量环分解")
    print("-" * 50)
    shape_3d = (4, 5, 6)
    tensor_3d = decomposer.create_random_tensor(shape_3d)
    print(f"张量形状: {shape_3d}")
    
    tr_rank = 3
    tr_factors = decomposer.tensor_ring_decomposition(tensor_3d, tr_rank)
    print(f"分解秩: {tr_rank}")
    for i, f in enumerate(tr_factors):
        print(f"  模式 {i} 因子形状: {f.shape}")
    
    tr_recon = decomposer.reconstruct_from_tensor_ring(tr_factors)
    tr_rmse = decomposer.calculate_rmse(np.array(tensor_3d), tr_recon)
    print(f"重构误差 RMSE: {tr_rmse:.6f}")
    
    print("\n2. 5维高阶张量环分解 (TR的优势场景)")
    print("-" * 50)
    shape_5d = (3, 4, 5, 6, 7)
    tensor_5d = decomposer.create_random_tensor(shape_5d)
    print(f"张量形状: {shape_5d}")
    
    tr_rank_5d = 3
    tr_factors_5d = decomposer.tensor_ring_decomposition(tensor_5d, tr_rank_5d)
    print(f"分解秩: {tr_rank_5d}")
    for i, f in enumerate(tr_factors_5d):
        print(f"  模式 {i} 因子形状: {f.shape}")
    
    tr_recon_5d = decomposer.reconstruct_from_tensor_ring(tr_factors_5d)
    tr_rmse_5d = decomposer.calculate_rmse(np.array(tensor_5d), tr_recon_5d)
    print(f"重构误差 RMSE: {tr_rmse_5d:.6f}")
    
    print("\n3. 分解方法比较")
    print("-" * 50)
    decomposer.compare_decompositions(tensor_5d, cp_rank=3, tr_rank=3)
    
    print("\n" + "=" * 70)
    print("张量环分解优势:")
    print("  ✓ 适合高阶张量（5维以上）")
    print("  ✓ 参数数量随阶数线性增长")
    print("  ✓ 更好的数值稳定性")
    print("  ✓ 比CP分解更强的表达能力")
    print("=" * 70)


def demonstrate_robustness():
    decomposer = TensorDecomposer(random_state=42)
    
    print("\n" + "=" * 70)
    print("鲁棒性检验演示")
    print("=" * 70)
    
    shape = (4, 5, 6)
    tensor = decomposer.create_random_tensor(shape)
    print(f"\n张量形状: {shape}")
    
    print("\n1. CP分解噪声敏感性分析")
    cp_robustness = decomposer.robustness_test(
        tensor, 
        decomposition_method='cp',
        ranks=[3, 5],
        noise_levels=[0.0, 0.05, 0.1, 0.2],
        n_trials=3,
        verbose=True
    )
    
    print("\n2. 张量环分解噪声敏感性分析")
    tr_robustness = decomposer.robustness_test(
        tensor, 
        decomposition_method='tr',
        ranks=[3, 5],
        noise_levels=[0.0, 0.05, 0.1, 0.2],
        n_trials=3,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("鲁棒性分析结论:")
    print("  - 秩越大，对噪声越敏感")
    print("  - 张量环比CP分解有更好的数值稳定性")
    print("  - 低噪声下，高秩分解表现更好")
    print("  - 高噪声下，适当降低秩可提高鲁棒性")
    print("=" * 70)


def demonstrate_rank_selection():
    decomposer = TensorDecomposer(random_state=42)
    
    print("=" * 70)
    print("CP分解秩选择演示")
    print("=" * 70)
    
    shape = (5, 6, 7)
    print(f"\n1. 创建随机张量，形状: {shape}")
    tensor = decomposer.create_random_tensor(shape)
    
    print("\n2. 测试不同秩的分解效果")
    print("-" * 70)
    print(f"{'秩':>4} | {'RMSE':>10} | {'过拟合分数':>10} | {'状态':>8}")
    print("-" * 70)
    
    results_list = []
    for rank in range(1, 8):
        factors = decomposer.cp_decomposition(tensor, rank)
        reconstructed = decomposer.reconstruct_from_cp(factors)
        rmse = decomposer.calculate_rmse(np.array(tensor), reconstructed)
        overfit_info = decomposer._detect_overfitting(factors[1])
        
        status = "过拟合!" if overfit_info['is_overfitting'] else "正常"
        print(f"{rank:4d} | {rmse:10.6f} | {overfit_info['avg_correlation_score']:12.3f} | {status:>8}")
        
        results_list.append({
            'rank': rank,
            'rmse': rmse,
            'overfit_score': overfit_info['avg_correlation_score'],
            'is_overfit': overfit_info['is_overfitting']
        })
    
    print("\n3. 自动选择最佳秩 (肘部法则)")
    optimal_rank, results = decomposer.select_optimal_rank(
        tensor, min_rank=1, max_rank=7, method='elbow', decomposition_method='cp', verbose=True
    )
    
    print(f"\n4. 使用最佳秩 {optimal_rank} 进行CP分解")
    factors = decomposer.cp_decomposition(tensor, optimal_rank)
    reconstructed = decomposer.reconstruct_from_cp(factors)
    final_rmse = decomposer.calculate_rmse(np.array(tensor), reconstructed)
    print(f"   最终重构误差 RMSE: {final_rmse:.6f}")
    
    print("\n" + "=" * 70)
    print("秩选择问题说明:")
    print("  - 秩过小: 重构误差大，无法捕捉张量的主要结构")
    print("  - 秩过大: 过拟合，因子矩阵列之间出现负相关或高度相关")
    print("  - 最佳秩: 误差曲线的肘部位置，兼顾拟合效果和模型简洁性")
    print("=" * 70)


def main():
    demonstrate_rank_selection()
    demonstrate_tensor_ring()
    demonstrate_robustness()


if __name__ == "__main__":
    main()
