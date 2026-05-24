import numpy as np
from tt_decomposition import TTTensor, tt_svd, tt_add, tt_multiply, tt_round, tt_cross, tt_cross_greedy, TensorTrainCross


def test_basic_decomposition():
    print("=" * 60)
    print("Test 1: Basic TT Decomposition and Reconstruction")
    print("=" * 60)
    
    tensor = np.random.rand(4, 5, 6)
    tt = tt_svd(tensor, eps=1e-10)
    
    print(f"Original shape: {tensor.shape}")
    print(f"TT representation: {tt}")
    
    reconstructed = tt.full()
    error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
    print(f"Relative reconstruction error: {error:.2e}")
    
    assert error < 1e-9, f"Reconstruction error too high: {error}"
    print("PASSED\n")


def test_rank_truncation():
    print("=" * 60)
    print("Test 2: Rank Truncation")
    print("=" * 60)
    
    tensor = np.random.rand(8, 8, 8, 8)
    
    tt_full = tt_svd(tensor, eps=0)
    print(f"Full TT ranks: {tt_full.ranks}")
    
    for eps in [1e-2, 1e-4, 1e-6]:
        tt_trunc = tt_svd(tensor, eps=eps)
        reconstructed = tt_trunc.full()
        error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
        print(f"eps={eps:.0e}, ranks={tt_trunc.ranks}, error={error:.2e}")
    
    print("PASSED\n")


def test_tt_addition():
    print("=" * 60)
    print("Test 3: TT Addition")
    print("=" * 60)
    
    tensor1 = np.random.rand(5, 6, 7)
    tensor2 = np.random.rand(5, 6, 7)
    
    tt1 = tt_svd(tensor1, eps=1e-10)
    tt2 = tt_svd(tensor2, eps=1e-10)
    
    tt_sum = tt_add(tt1, tt2)
    sum_reconstructed = tt_sum.full()
    true_sum = tensor1 + tensor2
    
    error = np.linalg.norm(true_sum - sum_reconstructed) / np.linalg.norm(true_sum)
    print(f"Addition relative error: {error:.2e}")
    
    assert error < 1e-9, f"Addition error too high: {error}"
    print("PASSED\n")


def test_tt_elementwise_multiply():
    print("=" * 60)
    print("Test 4: TT Element-wise Multiplication")
    print("=" * 60)
    
    tensor1 = np.random.rand(4, 4, 4)
    tensor2 = np.random.rand(4, 4, 4)
    
    tt1 = tt_svd(tensor1, eps=1e-10)
    tt2 = tt_svd(tensor2, eps=1e-10)
    
    tt_prod = tt_multiply(tt1, tt2)
    prod_reconstructed = tt_prod.full()
    true_prod = tensor1 * tensor2
    
    error = np.linalg.norm(true_prod - prod_reconstructed) / np.linalg.norm(true_prod)
    print(f"Multiplication relative error: {error:.2e}")
    
    assert error < 1e-9, f"Multiplication error too high: {error}"
    print("PASSED\n")


def test_low_rank_tensor():
    print("=" * 60)
    print("Test 5: Low-Rank Tensor (Tensor Train Format)")
    print("=" * 60)
    
    core1 = np.random.rand(1, 5, 3)
    core2 = np.random.rand(3, 6, 4)
    core3 = np.random.rand(4, 7, 1)
    
    tt_gt = TTTensor([core1, core2, core3])
    tensor = tt_gt.full()
    
    tt_reconstructed = tt_svd(tensor, eps=1e-12)
    
    print(f"Ground truth TT: {tt_gt}")
    print(f"Reconstructed TT: {tt_reconstructed}")
    
    error = np.linalg.norm(tensor - tt_reconstructed.full()) / np.linalg.norm(tensor)
    print(f"Reconstruction error: {error:.2e}")
    
    assert error < 1e-10, f"Low-rank reconstruction error too high: {error}"
    print("PASSED\n")


def test_high_dimensional_tensor():
    print("=" * 60)
    print("Test 6: High-Dimensional Tensor (Storage Comparison)")
    print("=" * 60)
    
    dims = [8] * 6
    tensor = np.random.rand(*dims)
    
    tt = tt_svd(tensor, eps=1e-6, max_rank=10)
    
    original_storage = np.prod(dims)
    tt_storage = sum(core.size for core in tt.cores)
    
    print(f"Original shape: {dims}")
    print(f"Original storage: {original_storage} elements")
    print(f"TT storage: {tt_storage} elements")
    print(f"Compression ratio: {original_storage / tt_storage:.2f}x")
    print(f"TT ranks: {tt.ranks}")
    
    error = np.linalg.norm(tensor - tt.full()) / np.linalg.norm(tensor)
    print(f"Relative error: {error:.2e}")
    
    print("PASSED\n")


def test_separable_function():
    print("=" * 60)
    print("Test 7: Separable Function (Should have rank 1)")
    print("=" * 60)
    
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 11)
    z = np.linspace(0, 1, 12)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    tensor = np.sin(X) * np.cos(Y) * np.exp(Z)
    
    tt = tt_svd(tensor, eps=1e-12)
    
    print(f"Separable function TT ranks: {tt.ranks}")
    print(f"Expected ranks: all 1")
    
    error = np.linalg.norm(tensor - tt.full()) / np.linalg.norm(tensor)
    print(f"Reconstruction error: {error:.2e}")
    
    assert all(r == 1 for r in tt.ranks), "Separable function should have rank 1"
    print("PASSED\n")


def test_adaptive_truncation():
    print("=" * 60)
    print("Test 8: Adaptive Truncation Strategy Verification")
    print("=" * 60)
    
    np.random.seed(1234)
    tensor = np.random.rand(10, 10, 10, 10)
    
    u, s, vh = np.linalg.svd(tensor.reshape(100, 100), full_matrices=False)
    s[10:] *= 1e-4
    tensor = (u @ np.diag(s) @ vh).reshape(10, 10, 10, 10)
    
    norm_tensor = np.linalg.norm(tensor)
    print(f"Tensor norm: {norm_tensor:.4f}")
    print()
    
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]
    errors = []
    ranks_list = []
    
    for eps in eps_list:
        tt = tt_svd(tensor, eps=eps)
        reconstructed = tt.full()
        rel_error = np.linalg.norm(tensor - reconstructed) / norm_tensor
        
        errors.append(rel_error)
        ranks_list.append(tt.ranks)
        
        print(f"eps={eps:.0e}: ranks={tt.ranks}, actual_error={rel_error:.2e}, error_bound_satisfied={rel_error <= eps}")
    
    print()
    
    for eps, error in zip(eps_list, errors):
        assert error <= eps * 1.1, f"Error {error:.2e} exceeds bound {eps:.0e} (too much information loss)"
    
    for i in range(len(errors) - 1):
        assert ranks_list[i] <= ranks_list[i+1] or errors[i] >= errors[i+1], \
            f"Rank should decrease or error should increase as eps increases"
    
    print("Adaptive truncation correctly balances rank and error!")
    print("PASSED\n")


def test_truncation_with_max_rank():
    print("=" * 60)
    print("Test 9: Truncation with Max Rank Constraint")
    print("=" * 60)
    
    tensor = np.random.rand(15, 15, 15)
    
    tt_unconstrained = tt_svd(tensor, eps=1e-10)
    print(f"Unconstrained ranks: {tt_unconstrained.ranks}")
    
    max_rank = 3
    tt_constrained = tt_svd(tensor, eps=1e-10, max_rank=max_rank)
    
    print(f"Constrained (max_rank={max_rank}) ranks: {tt_constrained.ranks}")
    
    for r in tt_constrained.ranks:
        assert r <= max_rank, f"Rank {r} exceeds max_rank {max_rank}"
    
    error_unconstrained = np.linalg.norm(tensor - tt_unconstrained.full()) / np.linalg.norm(tensor)
    error_constrained = np.linalg.norm(tensor - tt_constrained.full()) / np.linalg.norm(tensor)
    
    print(f"Unconstrained error: {error_unconstrained:.2e}")
    print(f"Constrained error: {error_constrained:.2e}")
    
    assert error_constrained >= error_unconstrained - 1e-12, "Constrained error should be >= unconstrained error"
    
    print("PASSED\n")


def test_tt_cross_basic():
    print("=" * 60)
    print("Test 10: TT-Cross Basic Functionality")
    print("=" * 60)
    
    shape = (8, 8, 8)
    
    def tensor_func(idx):
        x1, x2, x3 = idx
        return np.sin(x1 * 0.5) * np.cos(x2 * 0.3) * np.exp(-0.1 * x3)
    
    full_tensor = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                full_tensor[i, j, k] = tensor_func((i, j, k))
    
    tt_svd_result = tt_svd(full_tensor, eps=1e-6)
    print(f"TT-SVD ranks: {tt_svd_result.ranks}")
    print(f"TT-SVD storage: {sum(c.size for c in tt_svd_result.cores)}")
    
    tt_cross_result, n_evals = tt_cross(tensor_func, shape, max_rank=5, eps=1e-4, verbose=True)
    
    reconstructed = tt_cross_result.full()
    error = np.linalg.norm(full_tensor - reconstructed) / np.linalg.norm(full_tensor)
    
    print(f"TT-Cross ranks: {tt_cross_result.ranks}")
    print(f"TT-Cross function evaluations: {n_evals}")
    print(f"Full tensor elements: {np.prod(shape)}")
    print(f"Evaluation ratio: {n_evals / np.prod(shape):.2%}")
    print(f"Reconstruction relative error: {error:.2e}")
    
    assert error < 1e-2, f"TT-Cross error too high: {error}"
    print("PASSED\n")


def test_tt_cross_greedy():
    print("=" * 60)
    print("Test 11: TT-Cross Greedy Algorithm")
    print("=" * 60)
    
    shape = (6, 7, 8)
    
    def separable_func(idx):
        x1, x2, x3 = idx
        return (x1 + 1) * (x2 + 1) * (x3 + 1)
    
    full_tensor = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                full_tensor[i, j, k] = separable_func((i, j, k))
    
    tt_cross_result, n_evals = tt_cross_greedy(separable_func, shape, rank=3, verbose=True)
    
    reconstructed = tt_cross_result.full()
    error = np.linalg.norm(full_tensor - reconstructed) / np.linalg.norm(full_tensor)
    
    print(f"TT-Cross Greedy ranks: {tt_cross_result.ranks}")
    print(f"Function evaluations: {n_evals}")
    print(f"Full tensor elements: {np.prod(shape)}")
    print(f"Evaluation ratio: {n_evals / np.prod(shape):.2%}")
    print(f"Reconstruction relative error: {error:.2e}")
    
    assert error < 1e-10, f"Separable function should be exact with rank 1"
    print("PASSED\n")


def test_tt_cross_class():
    print("=" * 60)
    print("Test 12: TensorTrainCross Class")
    print("=" * 60)
    
    shape = (5, 6, 7, 8)
    
    def func_4d(idx):
        x1, x2, x3, x4 = idx
        return np.sin(x1 * 0.2 + x2 * 0.3) * np.cos(x3 * 0.1 + x4 * 0.4)
    
    full_tensor = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    full_tensor[i, j, k, l] = func_4d((i, j, k, l))
    
    tt_cross_solver = TensorTrainCross(func_4d, shape, max_rank=8, eps=1e-3, verbose=True)
    tt_result = tt_cross_solver.compute(n_sweeps=2)
    
    reconstructed = tt_result.full()
    error = np.linalg.norm(full_tensor - reconstructed) / np.linalg.norm(full_tensor)
    
    print(f"TT-Cross Class ranks: {tt_result.ranks}")
    print(f"Function evaluations: {tt_cross_solver.n_evaluations}")
    print(f"Full tensor elements: {np.prod(shape)}")
    print(f"Evaluation ratio: {tt_cross_solver.n_evaluations / np.prod(shape):.2%}")
    print(f"Reconstruction relative error: {error:.2e}")
    
    assert error < 1e-1, f"TT-Cross Class error too high: {error}"
    print("PASSED\n")


def test_tt_cross_high_dimensional():
    print("=" * 60)
    print("Test 13: TT-Cross for High-Dimensional Tensors")
    print("=" * 60)
    
    ndim = 5
    shape = (5,) * ndim
    
    def high_dim_func(idx):
        result = 1.0
        for i, x in enumerate(idx):
            result *= np.sin(0.5 * (x + 1))
        return result
    
    full_tensor = np.zeros(shape)
    it = np.nditer(full_tensor, flags=['multi_index'])
    for _ in it:
        full_tensor[it.multi_index] = high_dim_func(it.multi_index)
    
    tt_cross_result, n_evals = tt_cross(high_dim_func, shape, max_rank=5, eps=1e-4, verbose=True)
    
    reconstructed = tt_cross_result.full()
    error = np.linalg.norm(full_tensor - reconstructed) / np.linalg.norm(full_tensor)
    
    print(f"\n{ndim}D Tensor Summary:")
    print(f"Shape: {shape}")
    print(f"Full tensor elements: {np.prod(shape)}")
    print(f"TT-Cross function evaluations: {n_evals}")
    print(f"Compression in function calls: {np.prod(shape) / n_evals:.1f}x")
    print(f"TT ranks: {tt_cross_result.ranks}")
    print(f"Reconstruction error: {error:.2e}")
    
    assert error < 1e-2, f"High-dimensional TT-Cross error too high: {error}"
    print("PASSED\n")


if __name__ == "__main__":
    np.random.seed(42)
    
    test_basic_decomposition()
    test_rank_truncation()
    test_tt_addition()
    test_tt_elementwise_multiply()
    test_low_rank_tensor()
    test_high_dimensional_tensor()
    test_separable_function()
    test_adaptive_truncation()
    test_truncation_with_max_rank()
    test_tt_cross_basic()
    test_tt_cross_greedy()
    test_tt_cross_class()
    test_tt_cross_high_dimensional()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
