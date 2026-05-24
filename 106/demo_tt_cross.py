import numpy as np
from tt_decomposition import tt_svd, tt_cross, tt_cross_greedy, TensorTrainCross


def demo_separable_function():
    print("=" * 70)
    print("Demo 1: TT-Cross for Separable Functions")
    print("=" * 70)
    
    shape = (10, 12, 14)
    
    def separable_func(idx):
        x1, x2, x3 = idx
        return np.sin(x1 * 0.5) * np.cos(x2 * 0.3) * np.exp(-0.1 * x3)
    
    full_tensor = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                full_tensor[i, j, k] = separable_func((i, j, k))
    
    print(f"\nTensor shape: {shape}")
    print(f"Full tensor elements: {np.prod(shape)}")
    
    tt_svd_result = tt_svd(full_tensor, eps=1e-8)
    print(f"\nTT-SVD:")
    print(f"  Ranks: {tt_svd_result.ranks}")
    print(f"  Storage: {sum(c.size for c in tt_svd_result.cores)} elements")
    print(f"  Compression: {np.prod(shape) / sum(c.size for c in tt_svd_result.cores):.1f}x")
    
    tt_cross_result, n_evals = tt_cross(separable_func, shape, max_rank=10, eps=1e-6, verbose=True)
    
    print(f"\nTT-Cross:")
    print(f"  Ranks: {tt_cross_result.ranks}")
    print(f"  Function evaluations: {n_evals}")
    print(f"  Evaluation ratio: {n_evals / np.prod(shape):.2%}")
    
    error = np.linalg.norm(full_tensor - tt_cross_result.full()) / np.linalg.norm(full_tensor)
    print(f"  Reconstruction error: {error:.2e}")


def demo_high_dimensional():
    print("\n" + "=" * 70)
    print("Demo 2: TT-Cross for High-Dimensional Tensors")
    print("=" * 70)
    
    ndim = 6
    shape = (8,) * ndim
    
    def multi_dimensional_func(idx):
        result = 1.0
        for i, x in enumerate(idx):
            result *= np.sin(0.3 * (x + 1))
        return result
    
    full_tensor = np.zeros(shape)
    it = np.nditer(full_tensor, flags=['multi_index'])
    for _ in it:
        full_tensor[it.multi_index] = multi_dimensional_func(it.multi_index)
    
    print(f"\n{ndim}D Tensor shape: {shape}")
    print(f"Full tensor elements: {np.prod(shape):,}")
    
    tt_cross_result, n_evals = tt_cross(multi_dimensional_func, shape, max_rank=8, eps=1e-4, verbose=True)
    
    print(f"\nTT-Cross Summary:")
    print(f"  Ranks: {tt_cross_result.ranks}")
    print(f"  Function evaluations: {n_evals:,}")
    print(f"  Elements saved: {np.prod(shape) - n_evals:,}")
    print(f"  Compression factor: {np.prod(shape) / n_evals:.1f}x")
    
    error = np.linalg.norm(full_tensor - tt_cross_result.full()) / np.linalg.norm(full_tensor)
    print(f"  Reconstruction error: {error:.2e}")


def demo_class_interface():
    print("\n" + "=" * 70)
    print("Demo 3: TensorTrainCross Class Interface")
    print("=" * 70)
    
    shape = (6, 7, 8, 9)
    
    def func_4d(idx):
        x1, x2, x3, x4 = idx
        return np.sin(x1 * 0.2 + x2 * 0.15) * np.cos(x3 * 0.25 + x4 * 0.1)
    
    full_tensor = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    full_tensor[i, j, k, l] = func_4d((i, j, k, l))
    
    print(f"\n4D Tensor shape: {shape}")
    print(f"Full tensor elements: {np.prod(shape):,}")
    
    solver = TensorTrainCross(func_4d, shape, max_rank=10, eps=1e-5, verbose=True)
    tt_result = solver.compute(n_sweeps=3)
    
    print(f"\nResults:")
    print(f"  Final ranks: {tt_result.ranks}")
    print(f"  Total function evaluations: {solver.n_evaluations:,}")
    print(f"  Evaluation ratio: {solver.n_evaluations / np.prod(shape):.2%}")
    
    error = np.linalg.norm(full_tensor - tt_result.full()) / np.linalg.norm(full_tensor)
    print(f"  Reconstruction error: {error:.2e}")


def demo_greedy_algorithm():
    print("\n" + "=" * 70)
    print("Demo 4: TT-Cross Greedy Algorithm")
    print("=" * 70)
    
    shape = (10, 10, 10)
    
    def rank_one_func(idx):
        x1, x2, x3 = idx
        return (x1 + 1) * (x2 + 2) * (x3 + 3)
    
    full_tensor = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                full_tensor[i, j, k] = rank_one_func((i, j, k))
    
    print(f"\nTensor shape: {shape}")
    print(f"Full tensor elements: {np.prod(shape)}")
    
    tt_result, n_evals = tt_cross_greedy(rank_one_func, shape, rank=5, verbose=True)
    
    print(f"\nGreedy TT-Cross:")
    print(f"  Ranks: {tt_result.ranks}")
    print(f"  Function evaluations: {n_evals}")
    print(f"  Evaluation ratio: {n_evals / np.prod(shape):.2%}")
    
    error = np.linalg.norm(full_tensor - tt_result.full()) / np.linalg.norm(full_tensor)
    print(f"  Reconstruction error: {error:.2e}")
    
    if error < 1e-10:
        print("  -> Separable function reconstructed exactly!")


def demo_pde_application():
    print("\n" + "=" * 70)
    print("Demo 5: TT-Cross for PDE Solution Evaluation")
    print("=" * 70)
    
    ndim = 4
    n = 10
    shape = (n,) * ndim
    
    def pde_solution(idx):
        x = np.array(idx) / (n - 1)
        r = np.sqrt(np.sum((x - 0.5)**2))
        return np.exp(-10 * r)
    
    full_tensor = np.zeros(shape)
    it = np.nditer(full_tensor, flags=['multi_index'])
    for _ in it:
        full_tensor[it.multi_index] = pde_solution(it.multi_index)
    
    print(f"\n{ndim}D PDE solution tensor")
    print(f"Shape: {shape}")
    print(f"Full tensor elements: {np.prod(shape):,}")
    
    tt_result, n_evals = tt_cross(pde_solution, shape, max_rank=15, eps=1e-5, verbose=True)
    
    print(f"\nTT-Cross Results:")
    print(f"  Ranks: {tt_result.ranks}")
    print(f"  Function evaluations: {n_evals:,}")
    print(f"  Compression: {np.prod(shape) / n_evals:.1f}x fewer evaluations")
    
    error = np.linalg.norm(full_tensor - tt_result.full()) / np.linalg.norm(full_tensor)
    print(f"  Reconstruction error: {error:.2e}")
    
    print(f"\nThis demonstrates how TT-Cross can be used to:")
    print(f"  1. Sample PDE solution at strategic points only")
    print(f"  2. Build low-rank TT representation without full tensor")
    print(f"  3. Achieve significant computational savings")


if __name__ == "__main__":
    np.random.seed(42)
    
    demo_separable_function()
    demo_high_dimensional()
    demo_class_interface()
    demo_greedy_algorithm()
    demo_pde_application()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
