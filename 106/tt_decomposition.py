import numpy as np


class TTTensor:
    def __init__(self, cores):
        self.cores = cores
        self.ndim = len(cores)
        self.shape = tuple(core.shape[1] for core in cores)
        self.ranks = self._compute_ranks()

    def _compute_ranks(self):
        ranks = []
        for i in range(self.ndim - 1):
            ranks.append(self.cores[i].shape[2])
        return tuple(ranks)

    def __repr__(self):
        return f"TTTensor(shape={self.shape}, ranks={self.ranks})"

    def full(self):
        result = self.cores[0]
        for i in range(1, self.ndim):
            result = np.tensordot(result, self.cores[i], axes=([-1], [0]))
        return result[0, ..., 0]


def tt_svd(tensor, eps=1e-10, max_rank=None):
    tensor = np.asarray(tensor)
    ndim = tensor.ndim
    shape = tensor.shape

    if ndim < 2:
        raise ValueError("Tensor must have at least 2 dimensions")

    cores = []
    C = tensor.copy()
    prev_rank = 1

    norm_tensor = np.linalg.norm(tensor)
    
    if ndim > 2:
        delta = eps * norm_tensor / np.sqrt(ndim - 1)
    else:
        delta = eps * norm_tensor
    delta_sq = delta * delta

    for i in range(ndim - 1):
        C = C.reshape(prev_rank * shape[i], -1)
        
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        
        if eps > 0 and norm_tensor > 0:
            s_sq = S ** 2
            
            cumsum_rev = np.cumsum(s_sq[::-1])[::-1]
            
            ranks_keep = np.sum(cumsum_rev > delta_sq)
            
            if ranks_keep == 0:
                ranks_keep = 1
        else:
            ranks_keep = len(S)
        
        if max_rank is not None:
            ranks_keep = min(ranks_keep, max_rank)
        
        ranks_keep = max(1, ranks_keep)
        ranks_keep = min(ranks_keep, len(S))

        U = U[:, :ranks_keep]
        S = S[:ranks_keep]
        Vh = Vh[:ranks_keep, :]

        core = U.reshape(prev_rank, shape[i], ranks_keep)
        cores.append(core)

        C = np.dot(np.diag(S), Vh)
        prev_rank = ranks_keep

    cores.append(C.reshape(prev_rank, shape[-1], 1))
    
    return TTTensor(cores)


def tt_add(tt1, tt2):
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    ndim = tt1.ndim
    cores = []
    
    for i in range(ndim):
        r1_prev, n, r1_next = tt1.cores[i].shape
        r2_prev, n, r2_next = tt2.cores[i].shape
        
        if i == 0:
            core = np.zeros((1, n, r1_next + r2_next))
            core[0, :, :r1_next] = tt1.cores[i][0, :, :]
            core[0, :, r1_next:] = tt2.cores[i][0, :, :]
        elif i == ndim - 1:
            core = np.zeros((r1_prev + r2_prev, n, 1))
            core[:r1_prev, :, 0] = tt1.cores[i][:, :, 0]
            core[r1_prev:, :, 0] = tt2.cores[i][:, :, 0]
        else:
            core = np.zeros((r1_prev + r2_prev, n, r1_next + r2_next))
            core[:r1_prev, :, :r1_next] = tt1.cores[i]
            core[r1_prev:, :, r1_next:] = tt2.cores[i]
        
        cores.append(core)
    
    return TTTensor(cores)


def tt_multiply(tt1, tt2):
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    cores = []
    
    for i in range(tt1.ndim):
        r1_prev, n, r1_next = tt1.cores[i].shape
        r2_prev, n, r2_next = tt2.cores[i].shape
        
        core = np.zeros((r1_prev * r2_prev, n, r1_next * r2_next))
        
        for j in range(n):
            mat1 = tt1.cores[i][:, j, :]
            mat2 = tt2.cores[i][:, j, :]
            core[:, j, :] = np.kron(mat1, mat2)
        
        cores.append(core)
    
    return TTTensor(cores)


def tt_round(tt, eps=1e-10, max_rank=None):
    ndim = tt.ndim
    cores = [core.copy() for core in tt.cores]
    
    for i in range(ndim - 1, 0, -1):
        r_prev, n, r_next = cores[i].shape
        mat = cores[i].reshape(r_prev, n * r_next)
        
        Q, R = np.linalg.qr(mat.T, mode='reduced')
        Q = Q.T
        R = R.T
        
        cores[i] = Q.reshape(Q.shape[0], n, r_next)
        cores[i - 1] = np.tensordot(cores[i - 1], R, axes=([2], [0]))
    
    tensor = TTTensor(cores)
    return tt_svd(tensor.full(), eps=eps, max_rank=max_rank)


def tt_matvec(tt_A, tt_x):
    ndim = tt_A.ndim
    if tt_x.ndim != ndim:
        raise ValueError(f"Dimensions must match: {tt_A.ndim} vs {tt_x.ndim}")
    
    cores = []
    for i in range(ndim):
        rA_prev, n_row, n_col, rA_next = tt_A.cores[i].shape
        rx_prev, n, rx_next = tt_x.cores[i].shape
        
        if n_col != n:
            raise ValueError(f"Column dimension mismatch at mode {i}: {n_col} vs {n}")
        
        core = np.tensordot(tt_A.cores[i], tt_x.cores[i], axes=([2], [1]))
        core = core.transpose(0, 3, 1, 4, 2)
        core = core.reshape(rA_prev * rx_prev, n_row, rA_next * rx_next)
        cores.append(core)
    
    return TTTensor(cores)


def tt_dot(tt1, tt2):
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    result = np.ones((1, 1))
    for i in range(tt1.ndim):
        core1 = tt1.cores[i]
        core2 = tt2.cores[i]
        
        temp = np.tensordot(core1, core2, axes=([1], [1]))
        result = np.tensordot(result, temp, axes=([0, 1], [0, 2]))
    
    return result[0, 0]


def tt_norm(tt):
    return np.sqrt(tt_dot(tt, tt))


def tt_cross(f, shape, max_rank, eps=1e-6, max_sweeps=5, verbose=False):
    ndim = len(shape)
    
    if isinstance(max_rank, int):
        ranks = [1] + [max_rank] * (ndim - 1) + [1]
    else:
        ranks = [1] + list(max_rank) + [1]
    
    for i in range(1, ndim):
        ranks[i] = min(ranks[i], np.prod(shape[:i]), np.prod(shape[i:]))
    
    cores = None
    n_evals = 0
    
    def _eval(idx):
        nonlocal n_evals
        n_evals += 1
        return f(idx)
    
    def _build_index(mode, row_flat, col_flat):
        idx = [0] * ndim
        idx[mode] = row_flat % shape[mode]
        rest = row_flat // shape[mode]
        for i in range(mode - 1, -1, -1):
            idx[i] = rest % shape[i]
            rest = rest // shape[i]
        rest = col_flat
        for i in range(ndim - 1, mode, -1):
            idx[i] = rest % shape[i]
            rest = rest // shape[i]
        return tuple(idx)
    
    for sweep in range(max_sweeps):
        new_cores = []
        n_evals_this_sweep = n_evals
        
        for i in range(ndim - 1):
            r_left = ranks[i]
            n = shape[i]
            r_right = ranks[i + 1]
            
            n_rows = min(r_left * n, r_right * 2)
            n_cols = min(r_right, max(1, np.prod(shape[i+1:]) // 2))
            
            n_rows = min(n_rows, r_left * n)
            n_cols = min(n_cols, np.prod(shape[i+1:]))
            
            if n_cols == 0:
                n_cols = 1
            
            mat = np.zeros((r_left * n, n_cols))
            
            for row in range(min(r_left * n, n_rows)):
                for col in range(n_cols):
                    idx = _build_index(i, row, col)
                    mat[row, col] = _eval(idx)
            
            U, S, Vh = np.linalg.svd(mat[:, :n_cols], full_matrices=False)
            
            if eps > 0:
                s_sq = S ** 2
                total_energy = np.sum(s_sq)
                if total_energy > 0:
                    cumsum_rev = np.cumsum(s_sq[::-1])[::-1]
                    keep = np.sum(cumsum_rev > (eps ** 2) * total_energy / (ndim - 1))
                    keep = max(1, keep)
                else:
                    keep = 1
            else:
                keep = len(S)
            
            keep = min(keep, r_right)
            
            U = U[:, :keep]
            core = U.reshape(r_left, n, keep)
            new_cores.append(core)
            
            ranks[i + 1] = keep
        
        last_core = np.zeros((ranks[-2], shape[-1], 1))
        for r in range(ranks[-2]):
            for k in range(shape[-1]):
                idx = list(np.unravel_index(r, shape[:-1])) + [k]
                last_core[r, k, 0] = _eval(tuple(idx))
        new_cores.append(last_core)
        
        cores = new_cores
        
        if verbose:
            print(f"Sweep {sweep + 1}/{max_sweeps}: ranks = {tuple(ranks[1:-1])}, "
                  f"evaluations = {n_evals - n_evals_this_sweep}")
    
    return TTTensor(cores), n_evals


def tt_cross_greedy(f, shape, rank, verbose=False):
    ndim = len(shape)
    n_evals = 0
    
    def _eval(idx):
        nonlocal n_evals
        n_evals += 1
        return f(idx)
    
    cores = []
    
    for i in range(ndim - 1):
        if i == 0:
            n = shape[i]
            r_next = min(rank, np.prod(shape[i+1:]))
            
            mat = np.zeros((n, r_next))
            for row in range(n):
                for col in range(r_next):
                    idx = [0] * ndim
                    idx[i] = row
                    rest = col
                    for j in range(ndim - 1, i, -1):
                        idx[j] = rest % shape[j]
                        rest = rest // shape[j]
                    mat[row, col] = _eval(tuple(idx))
            
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = min(rank, len(S))
            U = U[:, :keep]
            core = U.reshape(1, n, keep)
            cores.append(core)
            r_prev = keep
        else:
            n = shape[i]
            r_next = min(rank, np.prod(shape[i+1:]))
            
            mat = np.zeros((r_prev * n, r_next))
            for row in range(r_prev * n):
                for col in range(r_next):
                    idx = [0] * ndim
                    
                    idx[i] = row % n
                    rest_left = row // n
                    for j in range(i - 1, -1, -1):
                        idx[j] = rest_left % shape[j]
                        rest_left = rest_left // shape[j]
                    
                    rest_right = col
                    for j in range(ndim - 1, i, -1):
                        idx[j] = rest_right % shape[j]
                        rest_right = rest_right // shape[j]
                    
                    mat[row, col] = _eval(tuple(idx))
            
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = min(rank, len(S))
            U = U[:, :keep]
            core = U.reshape(r_prev, n, keep)
            cores.append(core)
            r_prev = keep
    
    last_core = np.zeros((r_prev, shape[-1], 1))
    for r in range(r_prev):
        for k in range(shape[-1]):
            idx = [0] * ndim
            idx[-1] = k
            rest = r
            for j in range(ndim - 2, -1, -1):
                idx[j] = rest % shape[j]
                rest = rest // shape[j]
            last_core[r, k, 0] = _eval(tuple(idx))
    cores.append(last_core)
    
    if verbose:
        print(f"Total function evaluations: {n_evals}")
    
    return TTTensor(cores), n_evals


class TensorTrainCross:
    def __init__(self, f, shape, max_rank, eps=1e-8, verbose=False):
        self.f = f
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.max_rank = max_rank
        self.eps = eps
        self.verbose = verbose
        self.n_evaluations = 0
        self.cores = None
    
    def _eval(self, idx):
        self.n_evaluations += 1
        return self.f(tuple(idx))
    
    def _get_left_indices(self, mode, r):
        if mode == 0:
            return [tuple()]
        
        shape_left = self.shape[:mode]
        indices = []
        total = np.prod(shape_left)
        r = min(r, total)
        
        for idx_flat in range(r):
            idx = np.unravel_index(idx_flat, shape_left)
            indices.append(tuple(idx))
        
        return indices
    
    def _get_right_indices(self, mode, r):
        if mode == self.ndim - 1:
            return [tuple()]
        
        shape_right = self.shape[mode + 1:]
        indices = []
        total = np.prod(shape_right)
        r = min(r, total)
        
        for idx_flat in range(r):
            idx = np.unravel_index(idx_flat, shape_right)
            indices.append(tuple(idx))
        
        return indices
    
    def compute(self, n_sweeps=3):
        ndim = self.ndim
        shape = self.shape
        
        if isinstance(self.max_rank, int):
            ranks = [1] + [self.max_rank] * (ndim - 1) + [1]
        else:
            ranks = [1] + list(self.max_rank) + [1]
        
        for i in range(1, ndim):
            ranks[i] = min(ranks[i], np.prod(shape[:i]), np.prod(shape[i:]))
        
        for sweep in range(n_sweeps):
            new_cores = []
            
            for i in range(ndim - 1):
                r_left = ranks[i]
                n = shape[i]
                r_right = ranks[i + 1]
                
                left_indices = self._get_left_indices(i, r_left)
                right_indices = self._get_right_indices(i, r_right)
                
                n_left = len(left_indices)
                n_right = len(right_indices)
                
                interface = np.zeros((n_left, n, n_right))
                
                for il, left_idx in enumerate(left_indices):
                    for k in range(n):
                        for ir, right_idx in enumerate(right_indices):
                            full_idx = list(left_idx) + [k] + list(right_idx)
                            interface[il, k, ir] = self._eval(tuple(full_idx))
                
                mat = interface.reshape(n_left * n, n_right)
                
                U, S, Vh = np.linalg.svd(mat, full_matrices=False)
                
                if self.eps > 0:
                    s_sq = S ** 2
                    total_energy = np.sum(s_sq)
                    if total_energy > 0:
                        cumsum_rev = np.cumsum(s_sq[::-1])[::-1]
                        keep = np.sum(cumsum_rev > (self.eps ** 2) * total_energy / np.sqrt(ndim - 1))
                        keep = max(1, keep)
                    else:
                        keep = 1
                else:
                    keep = len(S)
                
                keep = min(keep, r_right)
                
                U = U[:, :keep]
                core = U.reshape(n_left, n, keep)
                new_cores.append(core)
                ranks[i + 1] = keep
            
            last_core = np.zeros((ranks[-2], shape[-1], 1))
            left_indices_last = self._get_left_indices(ndim - 1, ranks[-2])
            
            for il, left_idx in enumerate(left_indices_last):
                for k in range(shape[-1]):
                    full_idx = list(left_idx) + [k]
                    last_core[il, k, 0] = self._eval(tuple(full_idx))
            new_cores.append(last_core)
            
            self.cores = new_cores
            
            if self.verbose:
                print(f"Sweep {sweep + 1}/{n_sweeps}: ranks={tuple(ranks[1:-1])}, "
                      f"total_evals={self.n_evaluations}")
        
        return TTTensor(self.cores)

