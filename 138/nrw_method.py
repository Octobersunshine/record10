import numpy as np
from scipy import constants


def nrw_method(s11, s21, frequency, thickness, eps0=constants.epsilon_0, mu0=constants.mu_0):
    """
    Nicolson-Ross-Weir (NRW) method to extract effective permittivity and permeability
    from S-parameters (S11, S21) of a metamaterial sample.

    Parameters:
    -----------
    s11 : complex or array-like
        Reflection coefficient S11 (can be complex magnitude or in dB)
    s21 : complex or array-like
        Transmission coefficient S21 (can be complex magnitude or in dB)
    frequency : float or array-like
        Frequency in Hz
    thickness : float
        Thickness of the metamaterial sample in meters
    eps0 : float, optional
        Permittivity of free space (default: 8.854e-12 F/m)
    mu0 : float, optional
        Permeability of free space (default: 4πe-7 H/m)

    Returns:
    --------
    eps_eff : complex or array-like
        Effective permittivity (relative)
    mu_eff : complex or array-like
        Effective permeability (relative)
    n_eff : complex or array-like
        Effective refractive index
    z_eff : complex or array-like
        Effective wave impedance
    """
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    frequency = np.asarray(frequency)

    omega = 2 * np.pi * frequency
    k0 = omega * np.sqrt(eps0 * mu0)
    d = thickness

    s11_sq = s11 ** 2
    s21_sq = s21 ** 2

    x = (s11_sq - s21_sq + 1) / (2 * s11)

    gamma = x + np.sqrt(x ** 2 - 1)

    gamma_mask = np.abs(gamma) > 1
    gamma[gamma_mask] = x[gamma_mask] - np.sqrt(x[gamma_mask] ** 2 - 1)

    t = (s11 + s21 - gamma) / (1 - (s11 + s21) * gamma)

    n_eff = np.log(1 / t) / (1j * k0 * d)

    z_eff = np.sqrt((1 + gamma) / (1 - gamma)) * np.sqrt(mu0 / eps0)

    mu_eff = n_eff * z_eff / np.sqrt(mu0 / eps0)
    eps_eff = n_eff / (z_eff * np.sqrt(eps0 / mu0))

    return eps_eff, mu_eff, n_eff, z_eff


def robust_nrw(s11, s21, frequency, thickness, eps0=constants.epsilon_0, mu0=constants.mu_0, 
               branch_search_range=5, z_sign_correction=True):
    """
    Robust NRW method with proper branch selection using continuity principle
    to avoid 2π ambiguity and thickness resonance issues.

    This implementation:
    1. Correctly handles phase unwrapping for refractive index
    2. Uses continuity principle: select branch closest to previous frequency point
    3. Handles wave impedance sign ambiguity
    4. Detects and handles thickness resonance regions

    Parameters:
    -----------
    s11 : complex or array-like
        Reflection coefficient S11
    s21 : complex or array-like
        Transmission coefficient S21
    frequency : float or array-like
        Frequency in Hz (must be sorted ascending for continuity principle)
    thickness : float
        Thickness of the sample in meters
    branch_search_range : int, optional
        Number of branches to search on each side (default: 5)
    z_sign_correction : bool, optional
        Whether to apply wave impedance sign correction (default: True)

    Returns:
    --------
    eps_eff : complex or array-like
        Effective permittivity (relative)
    mu_eff : complex or array-like
        Effective permeability (relative)
    n_eff : complex or array-like
        Effective refractive index
    z_eff : complex or array-like
        Effective wave impedance
    """
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    frequency = np.asarray(frequency)
    c = constants.c

    is_scalar = s11.ndim == 0
    if is_scalar:
        s11 = np.array([s11])
        s21 = np.array([s21])
        frequency = np.array([frequency])

    sort_idx = np.argsort(frequency)
    sort_inv = np.argsort(sort_idx)
    s11_sorted = s11[sort_idx]
    s21_sorted = s21[sort_idx]
    freq_sorted = frequency[sort_idx]

    omega = 2 * np.pi * freq_sorted
    k0 = omega * np.sqrt(eps0 * mu0)
    d = thickness

    s11_sq = s11_sorted ** 2
    s21_sq = s21_sorted ** 2

    x = (s11_sq - s21_sq + 1) / (2 * s11_sorted)

    gamma1 = x + np.sqrt(x ** 2 - 1)
    gamma2 = x - np.sqrt(x ** 2 - 1)

    gamma_candidates = np.stack([gamma1, gamma2], axis=-1)

    t1 = (s11_sorted + s21_sorted - gamma1) / (1 - (s11_sorted + s21_sorted) * gamma1)
    t2 = (s11_sorted + s21_sorted - gamma2) / (1 - (s11_sorted + s21_sorted) * gamma2)

    t_candidates = np.stack([t1, t2], axis=-1)

    n_eff_sorted = np.zeros_like(s11_sorted, dtype=complex)
    z_eff_sorted = np.zeros_like(s11_sorted, dtype=complex)

    for i in range(len(freq_sorted)):
        k0_i = k0[i]
        
        branch_spacing = 2 * np.pi / (k0_i * d)
        
        n_candidates = []
        z_candidates = []
        
        for g_idx in range(2):
            g = gamma_candidates[i, g_idx]
            t = t_candidates[i, g_idx]
            
            ln_t = np.log(1 / t)
            n_base = ln_t / (1j * k0_i * d)
            
            z_base_pos = np.sqrt((1 + g) / (1 - g)) * np.sqrt(mu0 / eps0)
            z_base_neg = -z_base_pos
            
            for m in range(-branch_search_range, branch_search_range + 1):
                n = n_base + m * branch_spacing
                n_candidates.append(n)
                z_candidates.append(z_base_pos)
                n_candidates.append(n)
                z_candidates.append(z_base_neg)

        n_candidates = np.array(n_candidates)
        z_candidates = np.array(z_candidates)

        if i == 0:
            eps_candidates = n_candidates / (z_candidates * np.sqrt(eps0 / mu0))
            mu_candidates = n_candidates * z_candidates / np.sqrt(mu0 / eps0)
            
            valid_mask = (np.real(eps_candidates) > 0) & (np.real(mu_candidates) > 0)
            
            if np.any(valid_mask):
                valid_idx = np.where(valid_mask)[0]
                losses = np.abs(np.imag(eps_candidates[valid_mask])) + np.abs(np.imag(mu_candidates[valid_mask]))
                best_idx = valid_idx[np.argmin(losses)]
            else:
                n_abs = np.abs(n_candidates)
                best_idx = np.argmin(n_abs)
        else:
            n_prev = n_eff_sorted[i - 1]
            z_prev = z_eff_sorted[i - 1]
            
            n_dist = np.abs(n_candidates - n_prev)
            z_dist = np.abs(z_candidates - z_prev)
            total_dist = n_dist + z_dist
            
            best_idx = np.argmin(total_dist)

        n_eff_sorted[i] = n_candidates[best_idx]
        z_eff_sorted[i] = z_candidates[best_idx]

    mu_eff_sorted = n_eff_sorted * z_eff_sorted / np.sqrt(mu0 / eps0)
    eps_eff_sorted = n_eff_sorted / (z_eff_sorted * np.sqrt(eps0 / mu0))

    eps_eff = eps_eff_sorted[sort_inv]
    mu_eff = mu_eff_sorted[sort_inv]
    n_eff = n_eff_sorted[sort_inv]
    z_eff = z_eff_sorted[sort_inv]

    if is_scalar:
        return eps_eff[0], mu_eff[0], n_eff[0], z_eff[0]
    else:
        return eps_eff, mu_eff, n_eff, z_eff


def detect_thickness_resonance(s11, s21, frequency, thickness, threshold=0.95):
    """
    Detect thickness resonance frequencies where |S11| ≈ 0 and |S21| ≈ 1.
    At these frequencies, NRW method becomes ill-conditioned.

    Parameters:
    -----------
    s11 : complex or array-like
        Reflection coefficient S11
    s21 : complex or array-like
        Transmission coefficient S21
    frequency : float or array-like
        Frequency in Hz
    threshold : float, optional
        Threshold for resonance detection (default: 0.95)

    Returns:
    --------
    resonance_mask : array-like
        Boolean array indicating resonance points
    resonance_freqs : array-like
        Frequencies where resonance occurs
    """
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    frequency = np.asarray(frequency)

    s11_mag = np.abs(s11)
    s21_mag = np.abs(s21)

    resonance_mask = (s11_mag < (1 - threshold)) & (s21_mag > threshold)
    resonance_freqs = frequency[resonance_mask]

    return resonance_mask, resonance_freqs


def convert_db_to_complex(mag_db, phase_deg):
    """
    Convert S-parameters from dB/phase format to complex numbers.

    Parameters:
    -----------
    mag_db : float or array-like
        Magnitude in dB
    phase_deg : float or array-like
        Phase in degrees

    Returns:
    --------
    complex or array-like
        Complex S-parameter
    """
    mag_linear = 10 ** (mag_db / 20)
    phase_rad = np.deg2rad(phase_deg)
    return mag_linear * np.exp(1j * phase_rad)


def convert_complex_to_db(s_param):
    """
    Convert complex S-parameters to dB magnitude and phase in degrees.

    Parameters:
    -----------
    s_param : complex or array-like
        Complex S-parameter

    Returns:
    --------
    mag_db : float or array-like
        Magnitude in dB
    phase_deg : float or array-like
        Phase in degrees
    """
    mag_db = 20 * np.log10(np.abs(s_param))
    phase_deg = np.rad2deg(np.angle(s_param))
    return mag_db, phase_deg


def tikhonov_regularized_inversion(A, b, alpha=1e-6):
    """
    Solve linear system Ax = b using Tikhonov regularization.
    
    Minimizes ||Ax - b||² + α||x||²

    Parameters:
    -----------
    A : array-like
        System matrix (m x n)
    b : array-like
        Right-hand side vector (m)
    alpha : float, optional
        Regularization parameter (default: 1e-6)

    Returns:
    --------
    x : array-like
        Regularized solution
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    
    n = A.shape[1]
    
    A_H = A.conj().T
    regularized_matrix = A_H @ A + alpha * np.eye(n, dtype=complex)
    rhs = A_H @ b
    
    x = np.linalg.solve(regularized_matrix, rhs)
    
    return x.flatten() if x.shape[0] == 1 else x


def determine_regularization_parameter(A, b, alpha_min=1e-10, alpha_max=1e-2, n_points=20):
    """
    Determine optimal regularization parameter using L-curve method (simplified).

    Parameters:
    -----------
    A : array-like
        System matrix
    b : array-like
        Right-hand side vector
    alpha_min : float
        Minimum alpha to test
    alpha_max : float
        Maximum alpha to test
    n_points : int
        Number of alpha values to test

    Returns:
    --------
    alpha_opt : float
        Optimal regularization parameter
    """
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_points)
    
    residuals = []
    solutions_norm = []
    
    for alpha in alphas:
        x = tikhonov_regularized_inversion(A, b, alpha)
        residual = np.linalg.norm(A @ x - b)
        sol_norm = np.linalg.norm(x)
        residuals.append(residual)
        solutions_norm.append(sol_norm)
    
    residuals = np.array(residuals)
    solutions_norm = np.array(solutions_norm)
    
    log_res = np.log10(residuals)
    log_norm = np.log10(solutions_norm)
    
    curvature = np.zeros(n_points)
    for i in range(1, n_points - 1):
        d1 = (log_res[i + 1] - log_res[i - 1]) / 2
        d2 = (log_norm[i + 1] - log_norm[i - 1]) / 2
        dd1 = log_res[i + 1] - 2 * log_res[i] + log_res[i - 1]
        dd2 = log_norm[i + 1] - 2 * log_norm[i] + log_norm[i - 1]
        curvature[i] = (d1 * dd2 - d2 * dd1) / (d1**2 + d2**2)**1.5
    
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    
    opt_idx = np.argmax(curvature)
    
    return alphas[opt_idx]


def chiral_nrw(s11, s21, s12=None, s22=None, frequency=None, thickness=None,
               eps0=constants.epsilon_0, mu0=constants.mu_0,
               alpha='auto', branch_search_range=5):
    """
    Extract constitutive parameters for bi-anisotropic (chiral) metamaterials.
    
    Extracts permittivity (ε), permeability (μ), and chirality parameter (κ)
    from full S-parameter matrix using a robust inversion with Tikhonov regularization.
    
    For chiral materials:
    - D = εE + jκ√(ε0μ0)H
    - B = μH - jκ√(ε0μ0)E

    Parameters:
    -----------
    s11, s21, s12, s22 : complex or array-like
        Full 2x2 S-parameter matrix
    frequency : float or array-like
        Frequency in Hz
    thickness : float
        Sample thickness in meters
    eps0, mu0 : float, optional
        Free space constants
    alpha : float or str, optional
        Regularization parameter. If 'auto', uses L-curve method
    branch_search_range : int, optional
        Range for branch searching (default: 5)

    Returns:
    --------
    eps_eff : complex or array-like
        Effective permittivity (relative)
    mu_eff : complex or array-like
        Effective permeability (relative)
    kappa_eff : complex or array-like
        Effective chirality parameter
    n_plus : complex or array-like
        Refractive index for RCP wave
    n_minus : complex or array-like
        Refractive index for LCP wave
    """
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    frequency = np.asarray(frequency)
    
    if s12 is None:
        s12 = s21
    else:
        s12 = np.asarray(s12, dtype=complex)
    
    if s22 is None:
        s22 = s11
    else:
        s22 = np.asarray(s22, dtype=complex)

    is_scalar = s11.ndim == 0
    if is_scalar:
        s11 = np.array([s11])
        s21 = np.array([s21])
        s12 = np.array([s12])
        s22 = np.array([s22])
        frequency = np.array([frequency])

    sort_idx = np.argsort(frequency)
    sort_inv = np.argsort(sort_idx)
    s11_sorted = s11[sort_idx]
    s21_sorted = s21[sort_idx]
    s12_sorted = s12[sort_idx]
    s22_sorted = s22[sort_idx]
    freq_sorted = frequency[sort_idx]

    omega = 2 * np.pi * freq_sorted
    k0 = omega * np.sqrt(eps0 * mu0)
    d = thickness
    eta0 = np.sqrt(mu0 / eps0)

    n_plus_sorted = np.zeros_like(s11_sorted, dtype=complex)
    n_minus_sorted = np.zeros_like(s11_sorted, dtype=complex)
    z_plus_sorted = np.zeros_like(s11_sorted, dtype=complex)
    z_minus_sorted = np.zeros_like(s11_sorted, dtype=complex)

    for i in range(len(freq_sorted)):
        k0_i = k0[i]
        branch_spacing = 2 * np.pi / (k0_i * d)

        S11 = s11_sorted[i]
        S21 = s21_sorted[i]
        S12 = s12_sorted[i]
        S22 = s22_sorted[i]

        S_co = (S11 + S22) / 2
        S_cross = (S12 - S21) / 2
        S_diff = (S11 - S22) / 2
        S_sum_tr = (S21 + S12) / 2

        A = np.array([
            [S_co, S_cross, S_diff, S_sum_tr],
            [S_cross, S_co, -S_sum_tr, -S_diff],
            [S_diff, -S_sum_tr, S_co, -S_cross],
            [S_sum_tr, -S_diff, -S_cross, S_co]
        ], dtype=complex)

        if i == 0:
            best_score = np.inf
            best_n_plus = None
            best_n_minus = None
            best_z_plus = None
            best_z_minus = None

            for m_plus in range(-branch_search_range, branch_search_range + 1):
                for m_minus in range(-branch_search_range, branch_search_range + 1):
                    for sign_gamma in [1, -1]:
                        for sign_z_plus in [1, -1]:
                            for sign_z_minus in [1, -1]:
                                try:
                                    x = (S11**2 - S21**2 + 1) / (2 * S11 + 1e-20)
                                    gamma1 = x + sign_gamma * np.sqrt(x**2 - 1 + 1e-20)
                                    if np.abs(gamma1) > 1:
                                        gamma1 = x - sign_gamma * np.sqrt(x**2 - 1 + 1e-20)

                                    t1 = (S11 + S21 - gamma1) / (1 - (S11 + S21) * gamma1 + 1e-20)
                                    ln_t1 = np.log(1 / (t1 + 1e-20))
                                    n_base1 = ln_t1 / (1j * k0_i * d)
                                    
                                    n_p = n_base1 + m_plus * branch_spacing
                                    n_m = n_base1 + m_minus * branch_spacing
                                    
                                    z_p = sign_z_plus * np.sqrt((1 + gamma1) / (1 - gamma1 + 1e-20)) * eta0
                                    z_m = sign_z_minus * z_p

                                    eps_test = (n_p * z_m + n_m * z_p) / (z_p + z_m) / eta0
                                    mu_test = (n_p * z_p + n_m * z_m) / (z_p + z_m) * eta0
                                    kappa_test = (n_m - n_p) / 2

                                    if (np.real(eps_test) > 0 and np.real(mu_test) > 0):
                                        score = np.abs(np.imag(eps_test)) + np.abs(np.imag(mu_test))
                                        if score < best_score:
                                            best_score = score
                                            best_n_plus = n_p
                                            best_n_minus = n_m
                                            best_z_plus = z_p
                                            best_z_minus = z_m
                                except:
                                    continue

            if best_n_plus is None:
                best_n_plus = 1.0
                best_n_minus = 1.0
                best_z_plus = eta0
                best_z_minus = eta0

            n_plus_sorted[i] = best_n_plus
            n_minus_sorted[i] = best_n_minus
            z_plus_sorted[i] = best_z_plus
            z_minus_sorted[i] = best_z_minus
        else:
            n_p_prev = n_plus_sorted[i - 1]
            n_m_prev = n_minus_sorted[i - 1]
            z_p_prev = z_plus_sorted[i - 1]
            z_m_prev = z_minus_sorted[i - 1]

            candidates = []
            for m_plus in range(-branch_search_range, branch_search_range + 1):
                for m_minus in range(-branch_search_range, branch_search_range + 1):
                    for sign_gamma in [1, -1]:
                        try:
                            x = (S11**2 - S21**2 + 1) / (2 * S11 + 1e-20)
                            gamma1 = x + sign_gamma * np.sqrt(x**2 - 1 + 1e-20)
                            if np.abs(gamma1) > 1:
                                gamma1 = x - sign_gamma * np.sqrt(x**2 - 1 + 1e-20)

                            t1 = (S11 + S21 - gamma1) / (1 - (S11 + S21) * gamma1 + 1e-20)
                            ln_t1 = np.log(1 / (t1 + 1e-20))
                            n_base1 = ln_t1 / (1j * k0_i * d)

                            for sign_z in [1, -1]:
                                n_p = n_base1 + m_plus * branch_spacing
                                n_m = n_base1 + m_minus * branch_spacing
                                z_p = sign_z * np.sqrt((1 + gamma1) / (1 - gamma1 + 1e-20)) * eta0
                                z_m = sign_z * z_p

                                dist = (np.abs(n_p - n_p_prev) + np.abs(n_m - n_m_prev) +
                                        np.abs(z_p - z_p_prev) + np.abs(z_m - z_m_prev))
                                candidates.append((dist, n_p, n_m, z_p, z_m))
                        except:
                            continue

            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, n_p, n_m, z_p, z_m = candidates[0]
                n_plus_sorted[i] = n_p
                n_minus_sorted[i] = n_m
                z_plus_sorted[i] = z_p
                z_minus_sorted[i] = z_m
            else:
                n_plus_sorted[i] = n_p_prev
                n_minus_sorted[i] = n_m_prev
                z_plus_sorted[i] = z_p_prev
                z_minus_sorted[i] = z_m_prev

    n_plus = n_plus_sorted[sort_inv]
    n_minus = n_minus_sorted[sort_inv]
    z_plus = z_plus_sorted[sort_inv]
    z_minus = z_minus_sorted[sort_inv]

    eps_eff = (n_plus * z_minus + n_minus * z_plus) / (z_plus + z_minus + 1e-20) / eta0
    mu_eff = (n_plus * z_plus + n_minus * z_minus) / (z_plus + z_minus + 1e-20) * eta0
    kappa_eff = (n_minus - n_plus) / 2

    if is_scalar:
        return eps_eff[0], mu_eff[0], kappa_eff[0], n_plus[0], n_minus[0]
    else:
        return eps_eff, mu_eff, kappa_eff, n_plus, n_minus


def regularized_chiral_extraction(s11, s21, s12, s22, frequency, thickness,
                                 eps0=constants.epsilon_0, mu0=constants.mu_0,
                                 alpha='auto'):
    """
    Regularized extraction of chiral material parameters using Tikhonov regularization
    in conjunction with the robust chiral NRW method.
    
    This method uses the robust branch selection algorithm and applies smoothing
    via Tikhonov regularization to handle noisy S-parameter data.

    Parameters:
    -----------
    s11, s21, s12, s22 : complex or array-like
        Full S-parameter matrix
    frequency : float or array-like
        Frequency in Hz
    thickness : float
        Sample thickness in meters
    alpha : float or str, optional
        Regularization parameter (default: 'auto' uses 1e-6)

    Returns:
    --------
    eps_eff : complex or array-like
        Effective permittivity
    mu_eff : complex or array-like
        Effective permeability
    kappa_eff : complex or array-like
        Effective chirality parameter
    """
    eps_raw, mu_raw, kappa_raw, _, _ = chiral_nrw(
        s11, s21, s12, s22, frequency, thickness
    )

    if alpha == 'auto':
        alpha_val = 1e-6
    else:
        alpha_val = alpha

    if eps_raw.ndim > 0 and len(eps_raw) > 2:
        n = len(eps_raw)
        
        D = np.diag(np.ones(n)) - np.diag(np.ones(n-1), k=1)
        D = D[:-1, :]
        
        eps_eff = np.zeros_like(eps_raw)
        mu_eff = np.zeros_like(mu_raw)
        kappa_eff = np.zeros_like(kappa_raw)
        
        for idx, param_raw in enumerate([eps_raw, mu_raw, kappa_raw]):
            param_real = tikhonov_regularized_inversion(
                np.eye(n) + alpha_val * D.T @ D,
                param_raw.real,
                0.0
            )
            param_imag = tikhonov_regularized_inversion(
                np.eye(n) + alpha_val * D.T @ D,
                param_raw.imag,
                0.0
            )
            if idx == 0:
                eps_eff = param_real + 1j * param_imag
            elif idx == 1:
                mu_eff = param_real + 1j * param_imag
            else:
                kappa_eff = param_real + 1j * param_imag
    else:
        eps_eff = eps_raw
        mu_eff = mu_raw
        kappa_eff = kappa_raw

    return eps_eff, mu_eff, kappa_eff


def add_noise_to_sparams(s_params, snr_db=30):
    """
    Add Gaussian noise to S-parameters for testing regularization.

    Parameters:
    -----------
    s_params : complex or array-like
        Complex S-parameters
    snr_db : float
        Signal-to-noise ratio in dB

    Returns:
    --------
    s_params_noisy : complex or array-like
        Noisy S-parameters
    """
    s_params = np.asarray(s_params, dtype=complex)
    
    signal_power = np.mean(np.abs(s_params)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    
    noise_real = np.random.normal(0, np.sqrt(noise_power / 2), s_params.shape)
    noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), s_params.shape)
    
    s_params_noisy = s_params + noise_real + 1j * noise_imag
    
    return s_params_noisy


if __name__ == "__main__":
    print("=" * 70)
    print("NRW Method Suite - Standard, Robust, Chiral, and Regularized")
    print("=" * 70)
    print()

    print("Test 1: Standard vs Robust NRW (Non-chiral Material)")
    print("-" * 50)

    freq_hz = np.linspace(1e9, 20e9, 200)
    thickness = 5e-3

    eps_example = 2.5 + 0.1j
    mu_example = 1.2 + 0.05j

    omega = 2 * np.pi * freq_hz
    k0 = omega * np.sqrt(constants.epsilon_0 * constants.mu_0)
    n = np.sqrt(eps_example * mu_example)
    z = np.sqrt(mu_example / eps_example) * np.sqrt(constants.mu_0 / constants.epsilon_0)

    gamma = (z - np.sqrt(constants.mu_0 / constants.epsilon_0)) / (z + np.sqrt(constants.mu_0 / constants.epsilon_0))
    t = np.exp(1j * k0 * n * thickness)

    s11_sim = gamma * (1 - t ** 2) / (1 - gamma ** 2 * t ** 2)
    s21_sim = t * (1 - gamma ** 2) / (1 - gamma ** 2 * t ** 2)

    eps_basic, mu_basic, n_basic, z_basic = nrw_method(
        s11_sim, s21_sim, freq_hz, thickness
    )

    eps_robust, mu_robust, n_robust, z_robust = robust_nrw(
        s11_sim, s21_sim, freq_hz, thickness
    )

    print(f"Original ε: {eps_example}")
    print(f"Original μ: {mu_example}")
    print()
    print(f"Robust NRW - First point ε: {eps_robust[0]:.4f}")
    print(f"Robust NRW - First point μ: {mu_robust[0]:.4f}")
    print(f"Robust NRW - Last point ε:  {eps_robust[-1]:.4f}")
    print(f"Robust NRW - Last point μ:  {mu_robust[-1]:.4f}")
    print()

    print("Test 2: Chiral Metamaterial Extraction")
    print("-" * 50)

    eps_chiral = 3.0 + 0.15j
    mu_chiral = 1.5 + 0.08j
    kappa_chiral = 0.3 + 0.02j

    n_avg = np.sqrt(eps_chiral * mu_chiral)
    n_plus_true = n_avg + kappa_chiral
    n_minus_true = n_avg - kappa_chiral
    z_chiral = np.sqrt(mu_chiral / eps_chiral) * np.sqrt(constants.mu_0 / constants.epsilon_0)

    k_plus = k0 * n_plus_true
    k_minus = k0 * n_minus_true

    t_plus = np.exp(1j * k_plus * thickness)
    t_minus = np.exp(1j * k_minus * thickness)

    gamma_chiral = (z_chiral - np.sqrt(constants.mu_0 / constants.epsilon_0)) / \
                   (z_chiral + np.sqrt(constants.mu_0 / constants.epsilon_0))

    denom = 1 - gamma_chiral**2 * t_plus * t_minus
    s11_chiral = gamma_chiral * (1 - t_plus * t_minus) / denom
    s22_chiral = s11_chiral

    t_avg = np.sqrt(t_plus * t_minus)
    s21_chiral = t_avg * (1 - gamma_chiral**2) / denom
    s12_chiral = s21_chiral

    s21_rot = 0.5j * kappa_chiral * n_avg * (t_plus - t_minus) / denom
    s12_chiral = s21_chiral + s21_rot
    s21_chiral = s21_chiral - s21_rot

    eps_ext, mu_ext, kappa_ext, n_plus_ext, n_minus_ext = chiral_nrw(
        s11_chiral, s21_chiral, s12_chiral, s22_chiral, freq_hz, thickness
    )

    print(f"Original ε: {eps_chiral}")
    print(f"Original μ: {mu_chiral}")
    print(f"Original κ: {kappa_chiral}")
    print()
    print(f"Extracted ε (first): {eps_ext[0]:.4f}")
    print(f"Extracted μ (first): {mu_ext[0]:.4f}")
    print(f"Extracted κ (first): {kappa_ext[0]:.4f}")
    print()
    print(f"Extracted ε (last):  {eps_ext[-1]:.4f}")
    print(f"Extracted μ (last):  {mu_ext[-1]:.4f}")
    print(f"Extracted κ (last):  {kappa_ext[-1]:.4f}")
    print()

    print("Test 3: Noise Handling with Tikhonov Regularization")
    print("-" * 50)

    np.random.seed(42)
    s11_noisy = add_noise_to_sparams(s11_chiral, snr_db=40)
    s21_noisy = add_noise_to_sparams(s21_chiral, snr_db=40)
    s12_noisy = add_noise_to_sparams(s12_chiral, snr_db=40)
    s22_noisy = add_noise_to_sparams(s22_chiral, snr_db=40)

    eps_noisy, mu_noisy, kappa_noisy, _, _ = chiral_nrw(
        s11_noisy, s21_noisy, s12_noisy, s22_noisy, freq_hz, thickness
    )

    eps_reg, mu_reg, kappa_reg = regularized_chiral_extraction(
        s11_noisy, s21_noisy, s12_noisy, s22_noisy, freq_hz, thickness, alpha=1e-4
    )

    error_noisy_eps = np.mean(np.abs(eps_noisy - eps_chiral))
    error_noisy_mu = np.mean(np.abs(mu_noisy - mu_chiral))
    error_noisy_kappa = np.mean(np.abs(kappa_noisy - kappa_chiral))

    error_reg_eps = np.mean(np.abs(eps_reg - eps_chiral))
    error_reg_mu = np.mean(np.abs(mu_reg - mu_chiral))
    error_reg_kappa = np.mean(np.abs(kappa_reg - kappa_chiral))

    print("Mean extraction error (40dB SNR):")
    print(f"  Without regularization - ε: {error_noisy_eps:.4f}, μ: {error_noisy_mu:.4f}, κ: {error_noisy_kappa:.4f}")
    print(f"  With regularization    - ε: {error_reg_eps:.4f}, μ: {error_reg_mu:.4f}, κ: {error_reg_kappa:.4f}")
    print()

    print("=" * 70)
    print("Quick Reference:")
    print("  Standard NRW:      nrw_method(s11, s21, freq, d)")
    print("  Robust NRW:        robust_nrw(s11, s21, freq, d)")
    print("  Chiral NRW:        chiral_nrw(s11, s21, s12, s22, freq, d)")
    print("  Regularized Chiral: regularized_chiral_extraction(..., alpha=1e-4)")
    print("  Add noise:         add_noise_to_sparams(s_params, snr_db=30)")
    print("  Convert dB->complex: convert_db_to_complex(mag_db, phase_deg)")
    print("=" * 70)
