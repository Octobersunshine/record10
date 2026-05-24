import numpy as np
from tt_decomposition import TTTensor, tt_svd, tt_add, tt_multiply, tt_round, tt_dot, tt_norm


def create_laplacian_1d(n, h):
    laplacian = np.diag(-2 * np.ones(n)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
    return laplacian / h**2


def create_identity_1d(n):
    return np.eye(n)


def kron_list(matrices):
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result


def create_laplacian_nd(dims, h):
    ndim = len(dims)
    matrices = []
    
    for i in range(ndim):
        mats = [create_identity_1d(dims[j]) for j in range(ndim)]
        mats[i] = create_laplacian_1d(dims[i], h[i])
        matrices.append(kron_list(mats))
    
    return sum(matrices)


def tt_from_1d_operators(ops_list):
    cores = []
    for op in ops_list:
        n_in, n_out = op.shape
        core = op.reshape(1, n_out, n_in, 1)
        cores.append(core)
    return TTTensor(cores)


def create_laplacian_tt(dims, h):
    cores = []
    ndim = len(dims)
    
    for i in range(ndim):
        n = dims[i]
        lapl_1d = create_laplacian_1d(n, h[i])
        ident_1d = create_identity_1d(n)
        
        core = np.zeros((ndim, n, n, ndim))
        
        core[0, :, :, 0] = ident_1d
        
        core[i, :, :, ndim-1] = lapl_1d
        
        if i > 0:
            core[i, :, :, i] = ident_1d
        
        if i < ndim - 1:
            core[0, :, :, i+1] = ident_1d
        
        if i == 0:
            core = core[:, :, :, -1:]
            core = core[:1, :, :, :]
        
        cores.append(core)
    
    return cores


def solve_heat_eq_tt(dims, h, dt, t_end, u0_tt):
    ndim = len(dims)
    
    ident_cores = []
    for n in dims:
        core = np.eye(n).reshape(1, n, n, 1)
        ident_cores.append(core)
    ident_tt = TTTensor(ident_cores)
    
    lapl_cores = []
    for i, n in enumerate(dims):
        lapl_1d = create_laplacian_1d(n, h[i])
        core = np.zeros((2, n, n, 2))
        core[0, :, :, 0] = np.eye(n)
        core[0, :, :, 1] = lapl_1d
        core[1, :, :, 1] = np.eye(n)
        lapl_cores.append(core)
    
    lapl_cores[0] = lapl_cores[0][:1, :, :, :]
    lapl_cores[-1] = lapl_cores[-1][:, :, :, -1:]
    
    u_tt = u0_tt
    t = 0.0
    n_steps = int(t_end / dt)
    
    for step in range(n_steps):
        u_flat = u_tt.full()
        
        lapl_u = np.zeros_like(u_flat)
        for i in range(ndim):
            lapl_1d = create_laplacian_1d(dims[i], h[i])
            axes = list(range(ndim))
            axes.remove(i)
            axes.insert(0, i)
            u_transposed = np.transpose(u_flat, axes)
            shape_orig = u_transposed.shape
            u_reshaped = u_transposed.reshape(dims[i], -1)
            lapl_u_reshaped = lapl_1d @ u_reshaped
            lapl_u += np.transpose(lapl_u_reshaped.reshape(shape_orig), axes)
        
        u_new_flat = u_flat + dt * lapl_u
        u_tt = tt_svd(u_new_flat, eps=1e-8, max_rank=20)
        t += dt
    
    return u_tt


def create_initial_condition(dims, x_grids):
    ndim = len(dims)
    
    grids = np.meshgrid(*x_grids, indexing='ij')
    
    u0 = np.ones(dims)
    for i, grid in enumerate(grids):
        u0 *= np.exp(-5 * (grid - 0.5)**2)
    
    return u0


def main():
    print("=" * 70)
    print("TT-based High-Dimensional PDE Solver")
    print("=" * 70)
    
    ndim = 4
    n_per_dim = 20
    dims = [n_per_dim] * ndim
    
    print(f"\nProblem Setup:")
    print(f"  Dimensions: {ndim}D")
    print(f"  Grid size per dimension: {n_per_dim}")
    print(f"  Total unknowns: {np.prod(dims):,}")
    
    x_grids = [np.linspace(0, 1, n) for n in dims]
    h = [x[1] - x[0] for x in x_grids]
    
    print(f"\nCreating initial condition...")
    u0 = create_initial_condition(dims, x_grids)
    
    print(f"Performing TT decomposition of initial condition...")
    u0_tt = tt_svd(u0, eps=1e-10)
    print(f"  Initial TT ranks: {u0_tt.ranks}")
    print(f"  TT storage: {sum(c.size for c in u0_tt.cores):,} elements")
    print(f"  Compression ratio: {np.prod(dims) / sum(c.size for c in u0_tt.cores):.2f}x")
    
    recon_error = np.linalg.norm(u0 - u0_tt.full()) / np.linalg.norm(u0)
    print(f"  Reconstruction error: {recon_error:.2e}")
    
    print(f"\nSolving heat equation using TT format...")
    dt = 0.001
    t_end = 0.05
    
    u_final_tt = solve_heat_eq_tt(dims, h, dt, t_end, u0_tt)
    
    print(f"\nFinal Solution:")
    print(f"  Final TT ranks: {u_final_tt.ranks}")
    print(f"  Final TT storage: {sum(c.size for c in u_final_tt.cores):,} elements")
    
    u_final_full = u_final_tt.full()
    
    print(f"\nSolution Statistics:")
    print(f"  Initial max: {u0.max():.6f}")
    print(f"  Final max: {u_final_full.max():.6f}")
    print(f"  Mass conservation error: {abs(u0.sum() - u_final_full.sum()) / u0.sum():.2e}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(42)
    main()
