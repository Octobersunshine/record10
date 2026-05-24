import numpy as np
from scipy.sparse import csr_matrix, spdiags, lil_matrix
from scipy.sparse.linalg import spsolve, gmres, cg, bicgstab, LinearOperator
import time

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse.linalg import gmres as cp_gmres
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def generate_poisson_matrix(n):
    N = n * n
    A = lil_matrix((N, N))
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = 4.0
            
            if i > 0:
                A[idx, (i-1)*n + j] = -1.0
            if i < n-1:
                A[idx, (i+1)*n + j] = -1.0
            if j > 0:
                A[idx, i*n + (j-1)] = -1.0
            if j < n-1:
                A[idx, i*n + (j+1)] = -1.0
    
    return csr_matrix(A)


def compute_diagonal_dominance(A):
    n = A.shape[0]
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    
    diag_dominance = np.zeros(n)
    
    for i in range(n):
        row_start = A_indptr[i]
        row_end = A_indptr[i+1]
        
        diag = 0.0
        sum_off_diag = 0.0
        
        for j in range(row_start, row_end):
            col = A_indices[j]
            val = abs(A_data[j])
            if col == i:
                diag = val
            else:
                sum_off_diag += val
        
        if diag > 0:
            diag_dominance[i] = diag / (diag + sum_off_diag)
        else:
            diag_dominance[i] = 0.0
    
    return diag_dominance


def adaptive_theta(A, base_theta=0.25):
    diag_dominance = compute_diagonal_dominance(A)
    avg_dominance = np.mean(diag_dominance)
    
    if avg_dominance < 0.3:
        theta = 0.5
    elif avg_dominance < 0.5:
        theta = 0.35
    elif avg_dominance < 0.7:
        theta = 0.25
    else:
        theta = 0.15
    
    return min(max(theta, 0.05), 0.7)


def gauss_seidel_smooth(A, x, b, iterations=1):
    n = A.shape[0]
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    
    for _ in range(iterations):
        for i in range(n):
            row_start = A_indptr[i]
            row_end = A_indptr[i+1]
            
            diag = 0.0
            sum_val = b[i]
            
            for j in range(row_start, row_end):
                col = A_indices[j]
                val = A_data[j]
                if col == i:
                    diag = val
                else:
                    sum_val -= val * x[col]
            
            x[i] = sum_val / diag
    
    return x


def jacobi_smooth(A, x, b, iterations=1, omega=0.8):
    D = A.diagonal()
    D_inv = 1.0 / D
    
    for _ in range(iterations):
        r = b - A.dot(x)
        x = x + omega * D_inv * r
    
    return x


def compute_strong_connections(A, theta):
    n = A.shape[0]
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    
    strong_in = [[] for _ in range(n)]
    strong_out = [[] for _ in range(n)]
    
    for i in range(n):
        row_start = A_indptr[i]
        row_end = A_indptr[i+1]
        
        max_off_diag_neg = 0.0
        for j in range(row_start, row_end):
            col = A_indices[j]
            val = A_data[j]
            if col != i and val < 0:
                max_off_diag_neg = max(max_off_diag_neg, abs(val))
        
        if max_off_diag_neg > 0:
            threshold = theta * max_off_diag_neg
            
            for j in range(row_start, row_end):
                col = A_indices[j]
                val = A_data[j]
                if col != i and val < 0 and abs(val) >= threshold:
                    strong_in[col].append(i)
                    strong_out[i].append(col)
    
    return strong_in, strong_out


def ruge_stuben_coarsening(A, theta=None):
    if theta is None:
        theta = adaptive_theta(A)
    
    n = A.shape[0]
    strong_in, strong_out = compute_strong_connections(A, theta)
    
    influence = np.zeros(n)
    for i in range(n):
        influence[i] = len(strong_in[i])
    
    nodes = list(range(n))
    nodes.sort(key=lambda x: (-influence[x], x))
    
    status = np.zeros(n, dtype=int)
    
    for u in nodes:
        if status[u] == 0:
            status[u] = 1
            for v in strong_out[u]:
                if status[v] == 0:
                    status[v] = -1
    
    for u in range(n):
        if status[u] == -1:
            needs_coarse = False
            for v in strong_out[u]:
                if status[v] == -1:
                    has_common_c = False
                    for w in strong_out[u]:
                        if status[w] == 1 and w in strong_out[v]:
                            has_common_c = True
                            break
                    if not has_common_c:
                        needs_coarse = True
                        break
            
            if needs_coarse:
                status[u] = 1
                for v in strong_out[u]:
                    if status[v] == 0:
                        status[v] = -1
    
    for u in range(n):
        if status[u] == 0:
            status[u] = 1
    
    C = np.where(status == 1)[0]
    F = np.where(status == -1)[0]
    
    return C, F, status, theta, strong_out


def build_interpolation(A, C, F, status, strong_out=None):
    n = A.shape[0]
    n_coarse = len(C)
    
    c_to_idx = {c: i for i, c in enumerate(C)}
    
    P = lil_matrix((n, n_coarse))
    
    for c in C:
        P[c, c_to_idx[c]] = 1.0
    
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    
    for f in F:
        row_start = A_indptr[f]
        row_end = A_indptr[f+1]
        
        strong_c_neighbors = []
        weak_neighbors = []
        
        if strong_out is not None:
            strong_set = set(strong_out[f])
        else:
            strong_set = set()
        
        diag = 0.0
        for j in range(row_start, row_end):
            col = A_indices[j]
            val = A_data[j]
            if col == f:
                diag = val
            elif col != f:
                if val < 0:
                    if col in strong_set or status[col] == 1:
                        if status[col] == 1:
                            strong_c_neighbors.append((col, val))
                        else:
                            weak_neighbors.append((col, val))
        
        if len(strong_c_neighbors) > 0:
            sum_strong = sum(val for _, val in strong_c_neighbors)
            
            if len(weak_neighbors) > 0:
                sum_weak = sum(val for _, val in weak_neighbors)
                alpha = diag / (diag - sum_weak)
                scaled_sum = sum_strong / alpha
            else:
                scaled_sum = sum_strong
            
            if abs(scaled_sum) > 1e-15:
                for c, val in strong_c_neighbors:
                    P[f, c_to_idx[c]] = -val / scaled_sum
        else:
            all_c_neighbors = []
            for j in range(row_start, row_end):
                col = A_indices[j]
                val = A_data[j]
                if col != f and val < 0 and status[col] == 1:
                    all_c_neighbors.append((col, val))
            
            if len(all_c_neighbors) > 0:
                sum_c = sum(val for _, val in all_c_neighbors)
                for c, val in all_c_neighbors:
                    P[f, c_to_idx[c]] = -val / sum_c
            else:
                P[f, :] = 1.0 / n_coarse
    
    return csr_matrix(P)


def build_restriction(P):
    return P.T.tocsr()


def build_coarse_matrix(A, P, R):
    return R.dot(A).dot(P).tocsr()


def v_cycle(A, x, b, levels, smooth_iter=2, theta=None):
    if levels == 0 or A.shape[0] < 100:
        x[:] = spsolve(A, b)
        return x
    
    x = gauss_seidel_smooth(A, x, b, smooth_iter)
    
    r = b - A.dot(x)
    
    C, F, status, used_theta, strong_out = ruge_stuben_coarsening(A, theta)
    P = build_interpolation(A, C, F, status, strong_out)
    R = build_restriction(P)
    A_coarse = build_coarse_matrix(A, P, R)
    
    r_coarse = R.dot(r)
    e_coarse = np.zeros(A_coarse.shape[0])
    
    v_cycle(A_coarse, e_coarse, r_coarse, levels - 1, smooth_iter, theta)
    
    x = x + P.dot(e_coarse)
    
    x = gauss_seidel_smooth(A, x, b, smooth_iter)
    
    return x


def amg_solve(A, b, max_levels=5, max_iter=20, tol=1e-8, smooth_iter=2, 
               theta=None, verbose=True):
    n = A.shape[0]
    x = np.zeros(n)
    
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        return x
    
    if theta is None:
        auto_theta = adaptive_theta(A)
        if verbose:
            diag_dominance = compute_diagonal_dominance(A)
            print(f"Average diagonal dominance: {np.mean(diag_dominance):.3f}")
            print(f"Adaptive theta chosen: {auto_theta:.3f}")
    else:
        auto_theta = theta
        if verbose:
            print(f"Using fixed theta: {auto_theta:.3f}")
    
    C, F, _, _, _ = ruge_stuben_coarsening(A, auto_theta)
    if verbose:
        print(f"Coarsest level preview: C={len(C)}, F={len(F)}, ratio={len(C)/n:.3f}")
    
    for iteration in range(max_iter):
        r = b - A.dot(x)
        r_norm = np.linalg.norm(r) / b_norm
        
        if verbose:
            print(f"Iteration {iteration + 1}: Relative residual = {r_norm:.2e}")
        
        if r_norm < tol:
            if verbose:
                print(f"Converged in {iteration + 1} iterations!")
            return x
        
        x = v_cycle(A, x, b, max_levels, smooth_iter, auto_theta)
    
    if verbose:
        print(f"Reached maximum iterations ({max_iter})")
    
    return x


def test_different_matrices():
    print("="*60)
    print("Testing AMG with different matrix types")
    print("="*60)
    
    n = 16
    N = n * n
    
    print(f"\n1. Standard Poisson matrix (5-point stencil)")
    print("-"*50)
    A_poisson = generate_poisson_matrix(n)
    x_exact = np.ones(N)
    b = A_poisson.dot(x_exact)
    
    x_amg = amg_solve(A_poisson, b, max_levels=4, max_iter=15, tol=1e-8)
    error = np.linalg.norm(x_amg - x_exact) / np.linalg.norm(x_exact)
    print(f"Final relative error: {error:.2e}")
    
    print(f"\n2. Anisotropic diffusion (test adaptive theta)")
    print("-"*50)
    A_aniso = lil_matrix((N, N))
    eps = 0.1
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A_aniso[idx, idx] = 2 * eps + 2
            
            if i > 0:
                A_aniso[idx, (i-1)*n + j] = -eps
            if i < n-1:
                A_aniso[idx, (i+1)*n + j] = -eps
            if j > 0:
                A_aniso[idx, i*n + (j-1)] = -1.0
            if j < n-1:
                A_aniso[idx, i*n + (j+1)] = -1.0
    
    A_aniso = csr_matrix(A_aniso)
    b_aniso = A_aniso.dot(x_exact)
    
    x_amg_aniso = amg_solve(A_aniso, b_aniso, max_levels=4, max_iter=15, tol=1e-8)
    error_aniso = np.linalg.norm(x_amg_aniso - x_exact) / np.linalg.norm(x_exact)
    print(f"Final relative error: {error_aniso:.2e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


def main():
    test_different_matrices()
    
    print("\n" + "="*60)
    print("Detailed Poisson solve with comparison")
    print("="*60)
    
    n = 32
    N = n * n
    
    print(f"\nGenerating {n}x{n} Poisson problem (N={N})...")
    A = generate_poisson_matrix(n)
    
    x_exact = np.ones(N)
    b = A.dot(x_exact)
    
    print("\nSolving with AMG (adaptive theta)...")
    x_amg = amg_solve(A, b, max_levels=5, max_iter=15, tol=1e-8, smooth_iter=2)
    
    error = np.linalg.norm(x_amg - x_exact) / np.linalg.norm(x_exact)
    print(f"\nAMG relative error to exact solution: {error:.2e}")
    
    print("\nSolving with direct solver for comparison...")
    x_direct = spsolve(A, b)
    error_direct = np.linalg.norm(x_direct - x_exact) / np.linalg.norm(x_exact)
    print(f"Direct solver relative error: {error_direct:.2e}")
    
    print("\nAMG vs direct solver difference:")
    diff = np.linalg.norm(x_amg - x_direct) / np.linalg.norm(x_direct)
    print(f"Relative difference: {diff:.2e}")


class AMGPreconditioner:
    def __init__(self, A, max_levels=4, smooth_iter=2, theta=None, verbose=False):
        self.A = A
        self.max_levels = max_levels
        self.smooth_iter = smooth_iter
        self.verbose = verbose
        self.n = A.shape[0]
        
        if theta is None:
            self.theta = adaptive_theta(A)
        else:
            self.theta = theta
        
        self._setup_hierarchy()
        
        if verbose:
            print(f"AMG Preconditioner setup complete:")
            print(f"  Levels: {len(self.hierarchy)}")
            for i, level in enumerate(self.hierarchy):
                print(f"  Level {i}: N={level['A'].shape[0]}")
    
    def _setup_hierarchy(self):
        self.hierarchy = []
        current_A = self.A
        
        for level in range(self.max_levels):
            if current_A.shape[0] < 100:
                break
            
            C, F, status, _, strong_out = ruge_stuben_coarsening(current_A, self.theta)
            P = build_interpolation(current_A, C, F, status, strong_out)
            R = build_restriction(P)
            A_coarse = build_coarse_matrix(current_A, P, R)
            
            self.hierarchy.append({
                'A': current_A,
                'P': P,
                'R': R,
                'A_coarse': A_coarse,
                'C': C,
                'F': F,
                'status': status,
                'strong_out': strong_out
            })
            
            current_A = A_coarse
        
        self.coarse_A = current_A
        self.coarse_LU = None
    
    def _solve_coarse(self, b):
        if self.coarse_A.shape[0] <= 1:
            return b / (self.coarse_A[0, 0] + 1e-15)
        
        if self.coarse_LU is None:
            try:
                from scipy.sparse.linalg import splu
                self.coarse_LU = splu(self.coarse_A.tocsc())
                return self.coarse_LU.solve(b)
            except:
                return spsolve(self.coarse_A, b)
        
        return self.coarse_LU.solve(b)
    
    def _v_cycle_once(self, x, b):
        b_stack = []
        current_x = x.copy()
        current_b = b.copy()
        
        for level in self.hierarchy:
            b_stack.append(current_b.copy())
            current_x = gauss_seidel_smooth(level['A'], current_x, current_b, self.smooth_iter)
            r = current_b - level['A'].dot(current_x)
            current_b = level['R'].dot(r)
            current_x = np.zeros_like(current_b)
        
        current_x = self._solve_coarse(current_b)
        
        for level in reversed(self.hierarchy):
            current_x = level['P'].dot(current_x)
            original_b = b_stack.pop()
            current_x = gauss_seidel_smooth(level['A'], current_x, original_b, self.smooth_iter)
        
        return current_x
    
    def apply(self, x):
        b = x.copy()
        x_out = np.zeros_like(b)
        x_out = self._v_cycle_once(x_out, b)
        return x_out
    
    def aslinearoperator(self):
        return LinearOperator((self.n, self.n), matvec=self.apply, dtype=self.A.dtype)


class GPUAMGPreconditioner:
    def __init__(self, A, max_levels=4, smooth_iter=2, theta=None, verbose=False):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is not installed. GPU support unavailable.")
        
        self.A_cpu = A
        self.A = cp_csr_matrix(A)
        self.max_levels = max_levels
        self.smooth_iter = smooth_iter
        self.verbose = verbose
        self.n = A.shape[0]
        
        if theta is None:
            diag_dominance = _cp_compute_diagonal_dominance(self.A)
            avg_dominance = cp.mean(diag_dominance).get()
            
            if avg_dominance < 0.3:
                self.theta = 0.5
            elif avg_dominance < 0.5:
                self.theta = 0.35
            elif avg_dominance < 0.7:
                self.theta = 0.25
            else:
                self.theta = 0.15
        else:
            self.theta = theta
        
        self._setup_hierarchy()
        
        if verbose:
            print(f"GPU AMG Preconditioner setup complete:")
            print(f"  Levels: {len(self.hierarchy)}")
            for i, level in enumerate(self.hierarchy):
                print(f"  Level {i}: N={level['A'].shape[0]}")
    
    def _setup_hierarchy(self):
        self.hierarchy = []
        current_A = self.A
        
        for level in range(self.max_levels):
            if current_A.shape[0] < 100:
                break
            
            C, F, status, strong_out = _cp_ruge_stuben_coarsening(current_A, self.theta)
            P = _cp_build_interpolation(current_A, C, F, status, strong_out)
            R = P.T.tocsr()
            A_coarse = R.dot(current_A).dot(P).tocsr()
            
            self.hierarchy.append({
                'A': current_A,
                'P': P,
                'R': R,
                'A_coarse': A_coarse,
            })
            
            current_A = A_coarse
        
        self.coarse_A = current_A
    
    def _cp_gauss_seidel(self, A, x, b, iterations=1):
        A_cpu = A.get()
        x_cpu = x.get()
        b_cpu = b.get()
        
        x_cpu = gauss_seidel_smooth(A_cpu, x_cpu, b_cpu, iterations)
        
        return cp.array(x_cpu)
    
    def _solve_coarse(self, b):
        coarse_A_cpu = self.coarse_A.get()
        b_cpu = b.get()
        
        if coarse_A_cpu.shape[0] <= 1:
            return cp.array(b_cpu / (coarse_A_cpu[0, 0] + 1e-15))
        
        x_cpu = spsolve(coarse_A_cpu, b_cpu)
        return cp.array(x_cpu)
    
    def apply(self, x):
        if not isinstance(x, cp.ndarray):
            x = cp.array(x)
        
        b_stack = []
        current_x = cp.zeros_like(x)
        current_b = x.copy()
        
        for level in self.hierarchy:
            b_stack.append(current_b.copy())
            current_x = self._cp_gauss_seidel(level['A'], current_x, current_b, self.smooth_iter)
            r = current_b - level['A'].dot(current_x)
            current_b = level['R'].dot(r)
            current_x = cp.zeros_like(current_b)
        
        current_x = self._solve_coarse(current_b)
        
        for level in reversed(self.hierarchy):
            current_x = level['P'].dot(current_x)
            original_b = b_stack.pop()
            current_x = self._cp_gauss_seidel(level['A'], current_x, original_b, self.smooth_iter)
        
        return current_x
    
    def aslinearoperator(self):
        def matvec(x):
            return self.apply(x)
        return LinearOperator((self.n, self.n), matvec=matvec, dtype=self.A.dtype)


def _cp_compute_diagonal_dominance(A):
    n = A.shape[0]
    diag_dominance = cp.zeros(n)
    
    diag = A.diagonal()
    row_sums = cp.abs(A).sum(axis=1).A1
    
    mask = diag > 0
    diag_dominance[mask] = cp.abs(diag[mask]) / row_sums[mask]
    
    return diag_dominance


def _cp_compute_strong_connections(A, theta):
    n = A.shape[0]
    strong_out = [[] for _ in range(n)]
    
    A_cpu = A.get()
    
    for i in range(n):
        row = A_cpu.getrow(i)
        max_off_diag_neg = 0.0
        
        for j in range(row.nnz):
            col = row.indices[j]
            val = row.data[j]
            if col != i and val < 0:
                max_off_diag_neg = max(max_off_diag_neg, abs(val))
        
        if max_off_diag_neg > 0:
            threshold = theta * max_off_diag_neg
            
            for j in range(row.nnz):
                col = row.indices[j]
                val = row.data[j]
                if col != i and val < 0 and abs(val) >= threshold:
                    strong_out[i].append(col)
    
    return strong_out


def _cp_ruge_stuben_coarsening(A, theta):
    n = A.shape[0]
    strong_out = _cp_compute_strong_connections(A, theta)
    
    strong_in = [[] for _ in range(n)]
    for i in range(n):
        for j in strong_out[i]:
            strong_in[j].append(i)
    
    influence = np.zeros(n)
    for i in range(n):
        influence[i] = len(strong_in[i])
    
    nodes = list(range(n))
    nodes.sort(key=lambda x: (-influence[x], x))
    
    status = np.zeros(n, dtype=int)
    
    for u in nodes:
        if status[u] == 0:
            status[u] = 1
            for v in strong_out[u]:
                if status[v] == 0:
                    status[v] = -1
    
    for u in range(n):
        if status[u] == -1:
            needs_coarse = False
            for v in strong_out[u]:
                if status[v] == -1:
                    has_common_c = False
                    for w in strong_out[u]:
                        if status[w] == 1 and w in strong_out[v]:
                            has_common_c = True
                            break
                    if not has_common_c:
                        needs_coarse = True
                        break
            
            if needs_coarse:
                status[u] = 1
                for v in strong_out[u]:
                    if status[v] == 0:
                        status[v] = -1
    
    for u in range(n):
        if status[u] == 0:
            status[u] = 1
    
    C = np.where(status == 1)[0]
    F = np.where(status == -1)[0]
    
    return C, F, status, strong_out


def _cp_build_interpolation(A, C, F, status, strong_out):
    A_cpu = A.get()
    P_cpu = build_interpolation(A_cpu, C, F, status, strong_out)
    return cp_csr_matrix(P_cpu)


def solve_with_krylov(A, b, method='gmres', preconditioner='amg', 
                      max_iter=1000, tol=1e-8, verbose=True, use_gpu=False,
                      amg_levels=4, amg_smooth=2):
    if verbose:
        print(f"\nSolving with {method.upper()}", end="")
        if preconditioner:
            print(f" + {preconditioner.upper()} preconditioner")
        else:
            print(" (no preconditioner)")
        if use_gpu:
            print("Using GPU acceleration")
    
    start_time = time.time()
    residual_history = []
    
    def callback(xk):
        r = b - A.dot(xk)
        residual_history.append(np.linalg.norm(r))
    
    M = None
    if preconditioner == 'amg':
        if use_gpu and GPU_AVAILABLE:
            amg = GPUAMGPreconditioner(A, max_levels=amg_levels, smooth_iter=amg_smooth, verbose=verbose)
            M = amg.aslinearoperator()
        else:
            if use_gpu and not GPU_AVAILABLE and verbose:
                print("Warning: GPU requested but CuPy not available, using CPU instead")
            amg = AMGPreconditioner(A, max_levels=amg_levels, smooth_iter=amg_smooth, verbose=verbose)
            M = amg.aslinearoperator()
    
    setup_time = time.time() - start_time
    solve_start = time.time()
    
    method = method.lower()
    if method == 'gmres':
        x, info = gmres(A, b, M=M, maxiter=max_iter, tol=tol, callback=callback)
    elif method == 'cg':
        x, info = cg(A, b, M=M, maxiter=max_iter, tol=tol, callback=callback)
    elif method == 'bicgstab':
        x, info = bicgstab(A, b, M=M, maxiter=max_iter, tol=tol, callback=callback)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gmres', 'cg', or 'bicgstab'")
    
    solve_time = time.time() - solve_start
    total_time = time.time() - start_time
    
    if verbose:
        print(f"  Setup time: {setup_time:.3f}s")
        print(f"  Solve time: {solve_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Iterations: {len(residual_history)}")
        if info == 0:
            print(f"  Converged successfully!")
        else:
            print(f"  Did not converge (info={info})")
    
    return x, {
        'iterations': len(residual_history),
        'setup_time': setup_time,
        'solve_time': solve_time,
        'total_time': total_time,
        'residuals': residual_history,
        'converged': info == 0
    }


def generate_anisotropic_matrix(n, eps=0.01):
    N = n * n
    A = lil_matrix((N, N))
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = eps + 1 + eps + 1
            
            if i > 0:
                A[idx, (i-1)*n + j] = -eps
            if i < n-1:
                A[idx, (i+1)*n + j] = -eps
            if j > 0:
                A[idx, i*n + (j-1)] = -1.0
            if j < n-1:
                A[idx, i*n + (j+1)] = -1.0
    
    return csr_matrix(A)


def generate_hyperbolic_like_matrix(n):
    N = n * n
    A = lil_matrix((N, N))
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = 3.0
            
            if i > 0:
                A[idx, (i-1)*n + j] = -0.5
            if i < n-1:
                A[idx, (i+1)*n + j] = -1.5
            if j > 0:
                A[idx, i*n + (j-1)] = -0.3
            if j < n-1:
                A[idx, i*n + (j+1)] = -0.7
    
    return csr_matrix(A)


def benchmark_study():
    print("="*70)
    print("AMG Preconditioner Benchmark for Ill-conditioned Problems")
    print("="*70)
    
    n = 32
    N = n * n
    
    test_cases = [
        ("Standard Poisson", generate_poisson_matrix(n), 0.5),
        ("Moderate Anisotropy (ε=0.1)", generate_anisotropic_matrix(n, eps=0.1), 0.2),
        ("Strong Anisotropy (ε=0.01)", generate_anisotropic_matrix(n, eps=0.01), 0.05),
        ("Hyperbolic-like", generate_hyperbolic_like_matrix(n), 0.3),
    ]
    
    methods = ['gmres']
    preconditioners = [None, 'amg']
    
    for name, A, expected_cond in test_cases:
        print(f"\n{'='*70}")
        print(f"Test Case: {name}")
        print(f"Size: {N}x{N}")
        print(f"Approximate condition number ~ 1/{expected_cond:.3f}")
        print('='*70)
        
        x_exact = np.ones(N)
        b = A.dot(x_exact)
        
        results = {}
        for method in methods:
            for precond in preconditioners:
                label = f"{method.upper()}" + (" + AMG" if precond else "")
                print(f"\n  --- {label} ---")
                try:
                    x, stats = solve_with_krylov(
                        A, b, method=method, preconditioner=precond,
                        max_iter=500, tol=1e-8, verbose=True
                    )
                    error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
                    print(f"  Relative error: {error:.2e}")
                    results[label] = stats
                except Exception as e:
                    print(f"  Failed: {e}")
        
        if len(results) > 1:
            print(f"\n  --- Speedup Summary ---")
            base_key = [k for k in results.keys() if 'AMG' not in k][0]
            base_iter = results[base_key]['iterations']
            base_time = results[base_key]['total_time']
            
            for key, stats in results.items():
                if key != base_key:
                    iter_speedup = base_iter / stats['iterations'] if stats['iterations'] > 0 else float('inf')
                    time_speedup = base_time / stats['total_time'] if stats['total_time'] > 0 else float('inf')
                    print(f"  {key}:")
                    print(f"    Iteration speedup: {iter_speedup:.2f}x")
                    print(f"    Time speedup: {time_speedup:.2f}x")
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


def main():
    benchmark_study()
    
    print("\n" + "="*70)
    print("Detailed AMG+GMRES Solve Example")
    print("="*70)
    
    n = 32
    N = n * n
    
    print(f"\nGenerating strongly anisotropic problem (N={N})...")
    A = generate_anisotropic_matrix(n, eps=0.01)
    
    x_exact = np.ones(N)
    b = A.dot(x_exact)
    
    print("\n1. Solving with unpreconditioned GMRES...")
    x_gmres, stats_gmres = solve_with_krylov(
        A, b, method='gmres', preconditioner=None,
        max_iter=500, tol=1e-8, verbose=True
    )
    
    print("\n2. Solving with AMG-preconditioned GMRES...")
    x_amg, stats_amg = solve_with_krylov(
        A, b, method='gmres', preconditioner='amg',
        max_iter=500, tol=1e-8, verbose=True
    )
    
    print("\n" + "="*70)
    print("Comparison Summary:")
    print("="*70)
    print(f"Unpreconditioned GMRES:")
    print(f"  Iterations: {stats_gmres['iterations']}")
    print(f"  Total time: {stats_gmres['total_time']:.3f}s")
    print(f"  Error: {np.linalg.norm(x_gmres - x_exact) / np.linalg.norm(x_exact):.2e}")
    print(f"\nAMG-preconditioned GMRES:")
    print(f"  Iterations: {stats_amg['iterations']}")
    print(f"  Total time: {stats_amg['total_time']:.3f}s")
    print(f"  Error: {np.linalg.norm(x_amg - x_exact) / np.linalg.norm(x_exact):.2e}")
    
    if stats_gmres['iterations'] > 0 and stats_amg['iterations'] > 0:
        print(f"\nSpeedup:")
        print(f"  Iterations: {stats_gmres['iterations'] / stats_amg['iterations']:.1f}x")
        print(f"  Time: {stats_gmres['total_time'] / stats_amg['total_time']:.1f}x")


if __name__ == "__main__":
    main()
