import numpy as np
from typing import Tuple, List, Dict, Optional, Union


class ZernikeAnalyzer:
    def __init__(self, max_order: int = 6, fit_method: str = 'svd'):
        self.max_order = max_order
        self.noll_indices = self._generate_noll_indices(max_order)
        self.fit_method = fit_method
        self._gs_basis = None
        self._gs_transform = None
        self._last_A = None
    
    def _generate_noll_indices(self, max_order: int) -> List[Tuple[int, int]]:
        indices = []
        for n in range(max_order + 1):
            for m in range(-n, n + 1, 2):
                indices.append((n, m))
        return indices
    
    def zernike_radial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        m_abs = abs(m)
        if (n - m_abs) % 2 != 0:
            return np.zeros_like(rho)
        
        result = np.zeros_like(rho)
        max_k = (n - m_abs) // 2
        for k in range(max_k + 1):
            sign = (-1) ** k
            numerator = np.math.factorial(n - k)
            denominator = (np.math.factorial(k) * 
                          np.math.factorial((n + m_abs) // 2 - k) * 
                          np.math.factorial((n - m_abs) // 2 - k))
            coeff = sign * numerator / denominator
            result += coeff * rho ** (n - 2 * k)
        return result
    
    def zernike(self, n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        if rho.shape != theta.shape:
            raise ValueError("rho and theta must have the same shape")
        
        R = self.zernike_radial(n, m, rho)
        
        if m >= 0:
            return R * np.cos(m * theta)
        else:
            return R * np.sin(abs(m) * theta)
    
    def zernike_noll(self, j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        if j < 0 or j >= len(self.noll_indices):
            raise ValueError(f"Invalid Noll index {j}")
        n, m = self.noll_indices[j]
        return self.zernike(n, m, rho, theta)
    
    def get_zernike_name(self, j: int) -> str:
        names = {
            0: "Piston",
            1: "Tilt X",
            2: "Tilt Y",
            3: "Defocus",
            4: "Astigmatism 45°",
            5: "Astigmatism 0°",
            6: "Coma Y",
            7: "Coma X",
            8: "Trefoil Y",
            9: "Trefoil X",
            10: "Spherical Aberration",
            11: "Secondary Astigmatism 45°",
            12: "Secondary Astigmatism 0°",
            13: "Tetrafoil 22.5°",
            14: "Tetrafoil 0°",
            15: "Secondary Coma Y",
            16: "Secondary Coma X",
            17: "Secondary Trefoil Y",
            18: "Secondary Trefoil X",
            19: "Pentafoil Y",
            20: "Pentafoil X",
            21: "Secondary Spherical Aberration",
        }
        return names.get(j, f"Z{j} (n={self.noll_indices[j][0]}, m={self.noll_indices[j][1]})")
    
    def get_zernike_type(self, j: int) -> str:
        types = {
            0: "Piston",
            1: "Tilt",
            2: "Tilt",
            3: "Defocus",
            4: "Astigmatism",
            5: "Astigmatism",
            6: "Coma",
            7: "Coma",
            8: "Trefoil",
            9: "Trefoil",
            10: "Spherical",
            11: "Secondary Astigmatism",
            12: "Secondary Astigmatism",
            13: "Tetrafoil",
            14: "Tetrafoil",
            15: "Secondary Coma",
            16: "Secondary Coma",
            17: "Secondary Trefoil",
            18: "Secondary Trefoil",
            19: "Pentafoil",
            20: "Pentafoil",
            21: "Secondary Spherical",
        }
        return types.get(j, "High Order")
    
    def gram_schmidt_orthogonalize(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_points, n_terms = A.shape
        Q = np.zeros((n_points, n_terms))
        R = np.zeros((n_terms, n_terms))
        
        for j in range(n_terms):
            v = A[:, j].copy()
            
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], v)
                v -= R[i, j] * Q[:, i]
            
            R[j, j] = np.linalg.norm(v)
            if R[j, j] > 1e-10:
                Q[:, j] = v / R[j, j]
            else:
                Q[:, j] = v
        
        return Q, R
    
    def check_orthogonality(self, A: np.ndarray) -> Dict:
        n_terms = A.shape[1]
        norm_A = np.sqrt(np.sum(A ** 2, axis=0))
        norm_A[norm_A < 1e-10] = 1.0
        A_normalized = A / norm_A
        
        gram_matrix = A_normalized.T @ A_normalized
        
        diag = np.diag(gram_matrix)
        off_diag = gram_matrix - np.diag(diag)
        
        max_off_diag = np.max(np.abs(off_diag))
        mean_off_diag = np.mean(np.abs(off_diag))
        
        orthogonality_score = 1.0 - (max_off_diag + mean_off_diag) / 2
        
        return {
            'gram_matrix': gram_matrix,
            'max_off_diagonal': max_off_diag,
            'mean_off_diagonal': mean_off_diag,
            'orthogonality_score': orthogonality_score,
            'norms': norm_A
        }
    
    def fit_svd(self, A: np.ndarray, z: np.ndarray, rcond: float = 1e-10) -> Tuple[np.ndarray, Dict]:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        
        s_inv = np.zeros_like(s)
        mask = s > (rcond * s[0])
        s_inv[mask] = 1.0 / s[mask]
        
        Sigma_inv = np.diag(s_inv)
        
        A_pinv = Vt.T @ Sigma_inv @ U.T
        coeffs = A_pinv @ z
        
        z_fit = A @ coeffs
        residuals = np.sum((z - z_fit) ** 2)
        
        metrics = {
            'singular_values': s,
            'effective_rank': np.sum(mask),
            'condition_number': s[0] / s[-1] if s[-1] > 0 else np.inf,
            'residuals': residuals
        }
        
        return coeffs, metrics
    
    def fit_gram_schmidt(self, A: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, Dict]:
        Q, R = self.gram_schmidt_orthogonalize(A)
        
        y = Q.T @ z
        
        n_terms = A.shape[1]
        coeffs = np.zeros(n_terms)
        for j in range(n_terms - 1, -1, -1):
            coeffs[j] = y[j]
            for i in range(j + 1, n_terms):
                coeffs[j] -= R[j, i] * coeffs[i]
            if R[j, j] > 1e-10:
                coeffs[j] /= R[j, j]
        
        z_fit = A @ coeffs
        residuals = np.sum((z - z_fit) ** 2)
        
        ortho_check = self.check_orthogonality(Q)
        
        metrics = {
            'residuals': residuals,
            'R_matrix': R,
            'Q_orthogonality': ortho_check
        }
        
        return coeffs, metrics
    
    def fit_wavefront(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                     mask: Optional[np.ndarray] = None,
                     method: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        if method is None:
            method = self.fit_method
        
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        z = np.asarray(z).flatten()
        
        if mask is not None:
            mask = np.asarray(mask).flatten()
            x = x[mask]
            y = y[mask]
            z = z[mask]
        
        rho = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        
        valid = rho <= 1.0
        x = x[valid]
        y = y[valid]
        z = z[valid]
        rho = rho[valid]
        theta = theta[valid]
        
        if len(z) < len(self.noll_indices):
            raise ValueError(f"Not enough data points. Need at least {len(self.noll_indices)} points.")
        
        n_terms = len(self.noll_indices)
        A = np.zeros((len(z), n_terms))
        
        for j in range(n_terms):
            A[:, j] = self.zernike_noll(j, rho, theta)
        
        self._last_A = A
        
        ortho_check_before = self.check_orthogonality(A)
        
        if method == 'svd':
            coeffs, fit_metrics = self.fit_svd(A, z)
        elif method == 'gram_schmidt':
            coeffs, fit_metrics = self.fit_gram_schmidt(A, z)
        elif method == 'lstsq':
            coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
            fit_metrics = {
                'residuals': residuals[0] if len(residuals) > 0 else 0,
                'rank': rank,
                'singular_values': s
            }
        else:
            raise ValueError(f"Unknown method: {method}. Use 'svd', 'gram_schmidt', or 'lstsq'.")
        
        z_fit = A @ coeffs
        rms_residual = np.sqrt(np.mean((z - z_fit) ** 2))
        rms_total = np.sqrt(np.mean((z - np.mean(z)) ** 2))
        ss_res = np.sum((z - z_fit) ** 2)
        ss_tot = np.sum((z - np.mean(z)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        metrics = {
            'method': method,
            'rms_residual': rms_residual,
            'rms_total': rms_total,
            'r2': r2,
            'n_points': len(z),
            'n_terms': n_terms,
            'orthogonality_before': ortho_check_before,
            **fit_metrics
        }
        
        return coeffs, metrics
    
    def analyze_aberrations(self, coeffs: np.ndarray) -> Dict:
        results = {
            'coefficients': {},
            'rms_by_type': {},
            'total_rms': 0.0,
            'peak_to_valley': 0.0,
            'summary': []
        }
        
        for j in range(min(len(coeffs), len(self.noll_indices))):
            name = self.get_zernike_name(j)
            ab_type = self.get_zernike_type(j)
            results['coefficients'][j] = {
                'value': coeffs[j],
                'name': name,
                'type': ab_type,
                'n': self.noll_indices[j][0],
                'm': self.noll_indices[j][1]
            }
        
        type_rms = {}
        for j, info in results['coefficients'].items():
            ab_type = info['type']
            if ab_type not in type_rms:
                type_rms[ab_type] = 0.0
            type_rms[ab_type] += info['value'] ** 2
        
        for ab_type, sum_sq in type_rms.items():
            results['rms_by_type'][ab_type] = np.sqrt(sum_sq)
        
        if len(coeffs) > 1:
            results['total_rms'] = np.sqrt(np.sum(coeffs[1:] ** 2))
        
        if len(coeffs) > 0:
            grid_size = 100
            x = np.linspace(-1, 1, grid_size)
            y = np.linspace(-1, 1, grid_size)
            X, Y = np.meshgrid(x, y)
            rho = np.sqrt(X ** 2 + Y ** 2)
            theta = np.arctan2(Y, X)
            mask = rho <= 1.0
            
            wf = np.zeros_like(X)
            for j in range(len(coeffs)):
                wf += coeffs[j] * self.zernike_noll(j, rho, theta)
            
            valid_wf = wf[mask]
            results['peak_to_valley'] = np.max(valid_wf) - np.min(valid_wf)
        
        summary = []
        for j, info in sorted(results['coefficients'].items()):
            if abs(info['value']) > 1e-10:
                summary.append({
                    'index': j,
                    'name': info['name'],
                    'coefficient': info['value'],
                    'contribution': (info['value'] ** 2) / (results['total_rms'] ** 2) if results['total_rms'] > 0 else 0
                })
        
        summary.sort(key=lambda x: abs(x['coefficient']), reverse=True)
        results['summary'] = summary
        
        return results
    
    def reconstruct_wavefront(self, coeffs: np.ndarray, 
                             grid_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        
        W = np.zeros_like(X)
        for j in range(min(len(coeffs), len(self.noll_indices))):
            W += coeffs[j] * self.zernike_noll(j, rho, theta)
        
        return X, Y, W
    
    def print_orthogonality_report(self, metrics: Dict) -> None:
        if 'orthogonality_before' not in metrics:
            return
        
        ortho = metrics['orthogonality_before']
        print(f"\nZernike Basis Orthogonality Check (Discrete Sampling):")
        print(f"  Max off-diagonal:      {ortho['max_off_diagonal']:.6f}")
        print(f"  Mean off-diagonal:     {ortho['mean_off_diagonal']:.6f}")
        print(f"  Orthogonality score:   {ortho['orthogonality_score']:.6f}")
        
        if ortho['max_off_diagonal'] > 0.1:
            print(f"  WARNING: Significant non-orthogonality detected!")
            print(f"           Consider using 'svd' or 'gram_schmidt' method.")
    
    def print_analysis(self, coeffs: np.ndarray, metrics: Optional[Dict] = None) -> None:
        analysis = self.analyze_aberrations(coeffs)
        
        print("=" * 70)
        print("ZERNIKE WAVEFRONT ANALYSIS")
        print("=" * 70)
        
        if metrics:
            print(f"\nFit Quality Metrics:")
            print(f"  Method:                {metrics.get('method', 'unknown')}")
            print(f"  Number of points:      {metrics['n_points']}")
            print(f"  Number of terms:       {metrics['n_terms']}")
            print(f"  RMS residual error:    {metrics['rms_residual']:.6f}")
            print(f"  RMS wavefront error:   {metrics['rms_total']:.6f}")
            print(f"  R-squared:             {metrics['r2']:.6f}")
            
            if 'condition_number' in metrics:
                print(f"  Condition number:      {metrics['condition_number']:.2f}")
            if 'effective_rank' in metrics:
                print(f"  Effective rank:        {metrics['effective_rank']}")
            
            self.print_orthogonality_report(metrics)
        
        print(f"\nWavefront Statistics:")
        print(f"  Total RMS (w/o piston): {analysis['total_rms']:.6f}")
        print(f"  Peak-to-Valley:         {analysis['peak_to_valley']:.6f}")
        
        print(f"\nZernike Coefficients by Type:")
        for ab_type, rms_val in sorted(analysis['rms_by_type'].items()):
            if rms_val > 1e-10:
                print(f"  {ab_type:25s}: {rms_val:.6f} RMS")
        
        print(f"\nDominant Aberrations:")
        print(f"  {'Index':<6} {'Name':<30} {'Coeff':>12} {'Contrib':>10}")
        print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*10}")
        for item in analysis['summary'][:10]:
            contrib_pct = item['contribution'] * 100 if item['contribution'] > 0 else 0
            print(f"  Z{item['index']:<5} {item['name']:<30} {item['coefficient']:>+12.6f} {contrib_pct:>9.1f}%")
        
        print("=" * 70)
    
    def compute_complex_amplitude(self, coeffs: np.ndarray, 
                                 grid_size: int = 256,
                                 wavelength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        
        W = np.zeros_like(X)
        for j in range(min(len(coeffs), len(self.noll_indices))):
            W += coeffs[j] * self.zernike_noll(j, rho, theta)
        
        pupil_mask = rho <= 1.0
        
        phase = 2 * np.pi * W / wavelength
        amplitude = np.where(pupil_mask, 1.0, 0.0)
        complex_amp = amplitude * np.exp(1j * phase)
        
        return X, Y, complex_amp
    
    def compute_psf(self, coeffs: np.ndarray, 
                   grid_size: int = 256,
                   wavelength: float = 1.0,
                   normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, complex_amp = self.compute_complex_amplitude(coeffs, grid_size, wavelength)
        
        psf = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.ifftshift(complex_amp))) ** 2)
        
        if normalize:
            psf_max = psf.max()
            if psf_max > 0:
                psf = psf / psf_max
        
        dx = X[0, 1] - X[0, 0]
        f = np.fft.fftshift(np.fft.fftfreq(grid_size, dx))
        fx, fy = np.meshgrid(f, f)
        
        return fx, fy, psf
    
    def compute_strehl_ratio(self, coeffs: np.ndarray, 
                            wavelength: float = 1.0,
                            method: str = 'approximate') -> float:
        if method == 'exact':
            grid_size = 256
            fx, fy, psf_aberrated = self.compute_psf(coeffs, grid_size, wavelength, normalize=False)
            coeffs_ideal = np.zeros_like(coeffs)
            _, _, psf_ideal = self.compute_psf(coeffs_ideal, grid_size, wavelength, normalize=False)
            return psf_aberrated.max() / psf_ideal.max() if psf_ideal.max() > 0 else 0.0
        
        elif method == 'approximate':
            wf_rms = np.sqrt(np.sum(coeffs[1:] ** 2)) if len(coeffs) > 1 else 0.0
            strehl = np.exp(-(2 * np.pi * wf_rms / wavelength) ** 2)
            return min(strehl, 1.0)
        
        elif method == 'marechal':
            wf_rms = np.sqrt(np.sum(coeffs[1:] ** 2)) if len(coeffs) > 1 else 0.0
            strehl = 1.0 - (2 * np.pi * wf_rms / wavelength) ** 2
            return max(strehl, 0.0)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'exact', 'approximate', or 'marechal'.")
    
    def analyze_psf(self, fx: np.ndarray, fy: np.ndarray, psf: np.ndarray) -> Dict:
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        peak_value = psf.max()
        peak_pos = np.unravel_index(psf.argmax(), psf.shape)
        
        half_max = peak_value / 2.0
        above_half = psf >= half_max
        if np.any(above_half):
            y_idx, x_idx = np.where(above_half)
            fwhm_x = (x_idx.max() - x_idx.min()) * (fx[0, 1] - fx[0, 0])
            fwhm_y = (y_idx.max() - y_idx.min()) * (fy[1, 0] - fy[0, 0])
        else:
            fwhm_x = fwhm_y = 0.0
        
        total_energy = psf.sum()
        cumulative_energy = np.zeros_like(psf)
        
        center_y, center_x = psf.shape[0] // 2, psf.shape[1] // 2
        y_grid, x_grid = np.ogrid[:psf.shape[0], :psf.shape[1]]
        r = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        
        energy_in_r = []
        radii = np.arange(1, min(psf.shape) // 2, 1)
        for radius in radii:
            mask = r <= radius
            energy_in_r.append(psf[mask].sum() / total_energy if total_energy > 0 else 0)
        
        ee50_radius = 0
        ee80_radius = 0
        for i, energy in enumerate(energy_in_r):
            if ee50_radius == 0 and energy >= 0.5:
                ee50_radius = radii[i] * (fx[0, 1] - fx[0, 0])
            if ee80_radius == 0 and energy >= 0.8:
                ee80_radius = radii[i] * (fx[0, 1] - fx[0, 0])
                break
        
        dx = fx[0, 1] - fx[0, 0]
        dy = fy[1, 0] - fy[0, 0]
        
        return {
            'peak_value': peak_value,
            'peak_position': (fx[peak_pos], fy[peak_pos]),
            'fwhm_x': fwhm_x,
            'fwhm_y': fwhm_y,
            'fwhm_avg': (fwhm_x + fwhm_y) / 2,
            'ee50_radius': ee50_radius,
            'ee80_radius': ee80_radius,
            'total_energy': total_energy,
            'pixel_size': (dx, dy),
            'energy_curve': (radii * dx, energy_in_r)
        }
    
    def compute_mtf(self, coeffs: np.ndarray, 
                   grid_size: int = 256,
                   wavelength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fx, fy, psf = self.compute_psf(coeffs, grid_size, wavelength, normalize=True)
        
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        mtf = np.abs(otf)
        mtf = mtf / mtf[grid_size // 2, grid_size // 2] if mtf[grid_size // 2, grid_size // 2] > 0 else mtf
        
        return fx, fy, mtf
    
    def print_psf_analysis(self, coeffs: np.ndarray, 
                          wavelength: float = 1.0,
                          grid_size: int = 256) -> None:
        print("=" * 70)
        print("PSF AND STREHL RATIO ANALYSIS")
        print("=" * 70)
        
        strehl_exact = self.compute_strehl_ratio(coeffs, wavelength, method='exact')
        strehl_approx = self.compute_strehl_ratio(coeffs, wavelength, method='approximate')
        strehl_marechal = self.compute_strehl_ratio(coeffs, wavelength, method='marechal')
        
        print(f"\nStrehl Ratio:")
        print(f"  Exact (PSF peak ratio):  {strehl_exact:.6f}")
        print(f"  Approximate (exp(-σ²)):  {strehl_approx:.6f}")
        print(f"  Marechal (1 - σ²):       {strehl_marechal:.6f}")
        
        if strehl_exact >= 0.8:
            print(f"  Status: ★ DIFFRACTION LIMITED (Strehl ≥ 0.8)")
        elif strehl_exact >= 0.5:
            print(f"  Status: ● GOOD QUALITY")
        elif strehl_exact >= 0.2:
            print(f"  Status: ○ MODERATE QUALITY")
        else:
            print(f"  Status: ✕ POOR QUALITY")
        
        fx, fy, psf = self.compute_psf(coeffs, grid_size, wavelength)
        psf_analysis = self.analyze_psf(fx, fy, psf)
        
        print(f"\nPoint Spread Function (PSF):")
        print(f"  Peak intensity:          {psf_analysis['peak_value']:.4f}")
        print(f"  FWHM (X):                {psf_analysis['fwhm_x']:.6f} λ/D")
        print(f"  FWHM (Y):                {psf_analysis['fwhm_y']:.6f} λ/D")
        print(f"  FWHM (Avg):              {psf_analysis['fwhm_avg']:.6f} λ/D")
        print(f"  EE50 radius:             {psf_analysis['ee50_radius']:.6f} λ/D")
        print(f"  EE80 radius:             {psf_analysis['ee80_radius']:.6f} λ/D")
        
        wf_rms = np.sqrt(np.sum(coeffs[1:] ** 2)) if len(coeffs) > 1 else 0.0
        print(f"\nWavefront Error:")
        print(f"  RMS WFE:                 {wf_rms:.4f} waves")
        print(f"  RMS WFE:                 {wf_rms * wavelength:.4f} (λ = {wavelength})")
        print(f"  Marechal criterion:      {'✓' if wf_rms < wavelength/14 else '✗'} "
              f"(λ/14 = {wavelength/14:.4f} waves)")
        
        print("=" * 70)
    
    def compare_aberrations_psf(self, aberration_coeffs_list: List[np.ndarray],
                               labels: List[str],
                               wavelength: float = 1.0,
                               grid_size: int = 256) -> Dict:
        results = []
        
        for i, (coeffs, label) in enumerate(zip(aberration_coeffs_list, labels)):
            strehl = self.compute_strehl_ratio(coeffs, wavelength, method='exact')
            fx, fy, psf = self.compute_psf(coeffs, grid_size, wavelength)
            psf_info = self.analyze_psf(fx, fy, psf)
            
            results.append({
                'label': label,
                'coeffs': coeffs,
                'strehl_ratio': strehl,
                'psf_analysis': psf_info,
                'psf': psf,
                'fx': fx,
                'fy': fy
            })
        
        return {'comparisons': results}


def generate_test_wavefront(aberration_type: str = 'mixed', grid_size: int = 100, 
                           noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    
    analyzer = ZernikeAnalyzer(max_order=6)
    W = np.zeros_like(X)
    
    coeffs_dict = {
        'defocus': {3: 1.0},
        'astigmatism': {4: 0.8, 5: 0.5},
        'coma': {6: 0.7, 7: 0.4},
        'spherical': {10: 0.6},
        'trefoil': {8: 0.5, 9: 0.3},
        'mixed': {
            3: 1.0,    # Defocus
            4: 0.5,    # Astigmatism 45
            5: 0.3,    # Astigmatism 0
            6: 0.4,    # Coma Y
            7: 0.2,    # Coma X
            10: 0.3,   # Spherical
        }
    }
    
    if aberration_type not in coeffs_dict:
        aberration_type = 'mixed'
    
    for j, val in coeffs_dict[aberration_type].items():
        W += val * analyzer.zernike_noll(j, rho, theta)
    
    if noise > 0:
        W += np.random.normal(0, noise, W.shape)
    
    return X, Y, W


def main():
    print("Testing Zernike Wavefront Analysis...")
    print()
    
    analyzer = ZernikeAnalyzer(max_order=6)
    
    X, Y, W = generate_test_wavefront('mixed', grid_size=50, noise=0.05)
    
    print("Input wavefront with mixed aberrations + noise")
    print(f"Grid size: {X.shape}")
    print()
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = W.flatten()
    mask = (x_flat ** 2 + y_flat ** 2) <= 1.0
    
    coeffs, metrics = analyzer.fit_wavefront(x_flat, y_flat, z_flat, mask=mask)
    
    analyzer.print_analysis(coeffs, metrics)
    
    print("\nReconstructing wavefront from coefficients...")
    X_rec, Y_rec, W_rec = analyzer.reconstruct_wavefront(coeffs, grid_size=50)
    print(f"Reconstructed wavefront shape: {W_rec.shape}")
    print(f"Reconstruction complete!")


if __name__ == "__main__":
    main()
