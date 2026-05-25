import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter, sobel
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')


def create_quality_map(phase_wrapped, amplitude=None):
    rows, cols = phase_wrapped.shape
    
    dx = np.diff(phase_wrapped, axis=1)
    dy = np.diff(phase_wrapped, axis=0)
    
    phase_deriv_x = np.zeros((rows, cols))
    phase_deriv_y = np.zeros((rows, cols))
    phase_deriv_x[:, :-1] = np.abs(dx)
    phase_deriv_y[:-1, :] = np.abs(dy)
    
    max_deriv = np.max([phase_deriv_x.max(), phase_deriv_y.max()])
    if max_deriv > 0:
        phase_deriv_x /= max_deriv
        phase_deriv_y /= max_deriv
    
    quality_deriv = 1 - np.maximum(phase_deriv_x, phase_deriv_y)
    
    if amplitude is not None:
        amp_norm = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-10)
        quality_amp = amp_norm
    else:
        quality_amp = np.ones_like(phase_wrapped)
    
    laplacian = np.zeros_like(phase_wrapped)
    laplacian[1:-1, 1:-1] = (np.abs(phase_wrapped[2:, 1:-1] - phase_wrapped[:-2, 1:-1]) +
                              np.abs(phase_wrapped[1:-1, 2:] - phase_wrapped[1:-1, :-2]))
    quality_lap = 1 - (laplacian / (laplacian.max() + 1e-10))
    
    quality_map = quality_deriv * 0.5 + quality_amp * 0.3 + quality_lap * 0.2
    quality_map = (quality_map - quality_map.min()) / (quality_map.max() - quality_map.min() + 1e-10)
    
    return quality_map


def quality_guided_unwrapping(phase_wrapped, quality_map):
    rows, cols = phase_wrapped.shape
    unwrapped = np.copy(phase_wrapped)
    
    border_quality = np.zeros((rows, cols))
    border_quality[0, :] = -1
    border_quality[-1, :] = -1
    border_quality[:, 0] = -1
    border_quality[:, -1] = -1
    
    inner_quality = quality_map.copy()
    inner_quality *= (1 - np.abs(border_quality))
    
    start_y, start_x = np.unravel_index(np.argmax(inner_quality), inner_quality.shape)
    
    visited = np.zeros((rows, cols), dtype=bool)
    unwrapped = np.zeros((rows, cols))
    
    from heapq import heappush, heappop
    heap = []
    
    def add_neighbors(y, x):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and not visited[ny, nx]:
                edge_quality = (quality_map[y, x] + quality_map[ny, nx]) / 2
                heappush(heap, (-edge_quality, ny, nx, y, x))
    
    visited[start_y, start_x] = True
    unwrapped[start_y, start_x] = phase_wrapped[start_y, start_x]
    add_neighbors(start_y, start_x)
    
    count = 0
    while heap:
        neg_qual, ny, nx, py, px = heappop(heap)
        
        if visited[ny, nx]:
            continue
        
        visited[ny, nx] = True
        
        diff = phase_wrapped[ny, nx] - phase_wrapped[py, px]
        k = np.round((unwrapped[py, px] - phase_wrapped[ny, nx]) / (2 * np.pi))
        unwrapped[ny, nx] = phase_wrapped[ny, nx] + 2 * np.pi * k
        
        add_neighbors(ny, nx)
        
        count += 1
        if count % 10000 == 0:
            pass
    
    return unwrapped


def least_squares_unwrapping(phase_wrapped, mask=None):
    rows, cols = phase_wrapped.shape
    N = rows * cols
    
    if mask is None:
        mask = np.ones((rows, cols), dtype=bool)
    
    dx = np.zeros((rows, cols))
    dy = np.zeros((rows, cols))
    
    dx[:, :-1] = phase_wrapped[:, 1:] - phase_wrapped[:, :-1]
    dy[:-1, :] = phase_wrapped[1:, :] - phase_wrapped[:-1, :]
    
    dx = dx - 2 * np.pi * np.round(dx / (2 * np.pi))
    dy = dy - 2 * np.pi * np.round(dy / (2 * np.pi))
    
    A = lil_matrix((2 * N, N))
    b = np.zeros(2 * N)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1 and mask[i, j] and mask[i, j + 1]:
                A[idx, i * cols + j] = -1
                A[idx, i * cols + j + 1] = 1
                b[idx] = dx[i, j]
                idx += 1
            
            if i < rows - 1 and mask[i, j] and mask[i + 1, j]:
                A[idx, i * cols + j] = -1
                A[idx, (i + 1) * cols + j] = 1
                b[idx] = dy[i, j]
                idx += 1
    
    A = A.tocsr()[:idx]
    b = b[:idx]
    
    ATA = A.T @ A
    ATb = A.T @ b
    
    x = spsolve(ATA, ATb)
    
    unwrapped = x.reshape((rows, cols))
    
    return unwrapped


def fft_based_unwrapping(phase_wrapped):
    rows, cols = phase_wrapped.shape
    
    dx = np.zeros((rows, cols))
    dy = np.zeros((rows, cols))
    
    dx[:, :-1] = phase_wrapped[:, 1:] - phase_wrapped[:, :-1]
    dy[:-1, :] = phase_wrapped[1:, :] - phase_wrapped[:-1, :]
    
    dx = dx - 2 * np.pi * np.round(dx / (2 * np.pi))
    dy = dy - 2 * np.pi * np.round(dy / (2 * np.pi))
    
    dxx = np.zeros((rows, cols))
    dyy = np.zeros((rows, cols))
    
    dxx[:, 1:] = dx[:, 1:] - dx[:, :-1]
    dyy[1:, :] = dy[1:, :] - dy[:-1, :]
    
    laplacian = dxx + dyy
    
    u = np.arange(cols)
    v = np.arange(rows)
    U, V = np.meshgrid(u, v)
    
    denom = (2 * np.cos(2 * np.pi * U / cols) - 2) + (2 * np.cos(2 * np.pi * V / rows) - 2)
    denom[0, 0] = 1
    
    Laplacian_ft = fft2(laplacian)
    Phi_ft = Laplacian_ft / denom
    Phi_ft[0, 0] = 0
    
    unwrapped = np.real(ifft2(Phi_ft))
    
    unwrapped = unwrapped - unwrapped[0, 0] + phase_wrapped[0, 0]
    
    return unwrapped


def zernike_radial(n, m, rho):
    m_abs = np.abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho)
    
    R = np.zeros_like(rho)
    max_k = (n - m_abs) // 2
    for k in range(max_k + 1):
        coeff = ((-1)**k * np.math.factorial(n - k) /
                 (np.math.factorial(k) * 
                  np.math.factorial((n + m_abs) // 2 - k) *
                  np.math.factorial((n - m_abs) // 2 - k)))
        R += coeff * rho**(n - 2 * k)
    
    return R


def zernike_polynomial(n, m, rho, theta):
    R = zernike_radial(n, m, rho)
    if m >= 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(-m * theta)


def generate_zernike_basis(rows, cols, max_order=6):
    y, x = np.mgrid[0:rows, 0:cols]
    x_norm = (2 * x - cols + 1) / (cols - 1) if cols > 1 else 0
    y_norm = (2 * y - rows + 1) / (rows - 1) if rows > 1 else 0
    
    rho = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)
    
    valid_mask = rho <= 1.0
    
    basis_functions = []
    noll_indices = []
    
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            Z = zernike_polynomial(n, m, rho, theta)
            Z[~valid_mask] = 0
            
            norm_factor = np.sqrt((n + 1) / np.pi) if m == 0 else np.sqrt(2 * (n + 1) / np.pi)
            Z = Z / norm_factor
            
            basis_functions.append(Z)
            noll_indices.append((n, m))
    
    return np.array(basis_functions), valid_mask, noll_indices


def fit_zernike(phase_map, max_order=6, mask=None):
    rows, cols = phase_map.shape
    
    if mask is None:
        mask = np.ones((rows, cols), dtype=bool)
    
    basis, valid_mask, indices = generate_zernike_basis(rows, cols, max_order)
    
    combined_mask = mask & valid_mask
    
    N = len(basis)
    A = np.zeros((combined_mask.sum(), N))
    b = phase_map[combined_mask]
    
    for i in range(N):
        A[:, i] = basis[i][combined_mask]
    
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    reconstructed = np.zeros_like(phase_map)
    for i in range(N):
        reconstructed += coeffs[i] * basis[i]
    
    residual = phase_map - reconstructed
    
    return coeffs, reconstructed, residual, indices


def correct_aberration(phase_map, max_order=4, remove_orders=None):
    if remove_orders is None:
        remove_orders = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2)]
    
    coeffs, aberration, residual, indices = fit_zernike(phase_map, max_order=max_order)
    
    aberration_to_remove = np.zeros_like(phase_map)
    basis, valid_mask, _ = generate_zernike_basis(phase_map.shape[0], phase_map.shape[1], max_order)
    
    for (n, m) in remove_orders:
        if (n, m) in indices:
            idx = indices.index((n, m))
            aberration_to_remove += coeffs[idx] * basis[idx]
    
    corrected = phase_map - aberration_to_remove
    
    return corrected, aberration_to_remove, coeffs, indices


def remove_phase_tilt(phase_map):
    rows, cols = phase_map.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    mask = np.ones_like(phase_map, dtype=bool)
    center = (rows // 2, cols // 2)
    radius = min(rows, cols) // 3
    y_grid, x_grid = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x_grid - center[1])**2 + (y_grid - center[0])**2)
    mask[dist_from_center > radius] = False
    
    A = np.column_stack([X[mask].flatten(), Y[mask].flatten(), np.ones(mask.sum())])
    b = phase_map[mask].flatten()
    
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    tilt_phase = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    phase_corrected = phase_map - tilt_phase
    
    return phase_corrected, tilt_phase, coeffs


class PhaseUnwrapper:
    def __init__(self, method='quality_guided'):
        self.method = method
        
    def unwrap(self, phase_wrapped, amplitude=None, mask=None):
        if self.method == 'quality_guided':
            quality_map = create_quality_map(phase_wrapped, amplitude)
            return quality_guided_unwrapping(phase_wrapped, quality_map)
        elif self.method == 'least_squares':
            return least_squares_unwrapping(phase_wrapped, mask)
        elif self.method == 'fft':
            return fft_based_unwrapping(phase_wrapped)
        elif self.method == 'simple':
            return np.unwrap(np.unwrap(phase_wrapped, axis=0), axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class AberrationCorrector:
    def __init__(self, max_order=6):
        self.max_order = max_order
        self.coeffs = None
        self.indices = None
        
    def correct(self, phase_map, remove_orders=None, mask=None):
        corrected, aberration, self.coeffs, self.indices = correct_aberration(
            phase_map, self.max_order, remove_orders
        )
        return corrected, aberration
    
    def get_aberration_terms(self):
        terms = {}
        if self.coeffs is not None and self.indices is not None:
            for i, (n, m) in enumerate(self.indices):
                terms[(n, m)] = self.coeffs[i]
        return terms


def temporal_unwrapping(phases_sequence, reference_idx=0):
    unwrapped_sequence = []
    reference = phases_sequence[reference_idx]
    
    for i, phase in enumerate(phases_sequence):
        diff = phase - reference
        k = np.round(diff / (2 * np.pi))
        unwrapped = phase - 2 * np.pi * k
        unwrapped_sequence.append(unwrapped)
    
    return np.array(unwrapped_sequence)
