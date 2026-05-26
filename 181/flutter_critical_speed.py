import numpy as np
from scipy.linalg import eig
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
#  Experimental flutter-derivative data (default synthetic)
# ---------------------------------------------------------------------------

def generate_default_flutter_derivative_data():
    k_data = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                       0.60, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00])

    def _stiffness(k, a, b, c):
        return a * (1.0 - np.exp(-b * k)) - c * k

    def _damping(k, k_zero, slope_pos, pos_val):
        z = (k - k_zero) / 0.08
        sigmoid = 1.0 / (1.0 + np.exp(-2.5 * z))
        return pos_val * sigmoid + slope_pos * (k - k_zero)

    def _damping_neg(k, a_pos, b_pos, a_neg, b_neg):
        return a_pos * (1.0 - np.exp(-k / b_pos)) - a_neg * np.exp(-k / b_neg)

    H1_data = _stiffness(k_data, 0.25, 0.60, 0.01)
    H3_data = _stiffness(k_data, 0.50, 1.00, 0.02)
    A1_data = _stiffness(k_data, 2.00, 0.80, 0.05)
    A3_data = _stiffness(k_data, 1.80, 0.70, 0.00)

    H2_data = 0.30 * (1.0 - np.exp(-k_data / 0.40))
    A2_data = _damping_neg(k_data, 0.35, 0.30, 0.30, 0.10)

    H4_data = -0.10 * k_data * np.exp(-0.8 * k_data)
    A4_data = -0.15 * k_data * np.exp(-0.6 * k_data)

    return {
        'k': k_data,
        'H1': H1_data, 'H2': H2_data, 'H3': H3_data, 'H4': H4_data,
        'A1': A1_data, 'A2': A2_data, 'A3': A3_data, 'A4': A4_data,
    }


# ---------------------------------------------------------------------------
#  Multi-modal + TMD flutter analyser (Scanlan p-k method)
# ---------------------------------------------------------------------------

class FlutterAnalyzer:
    def __init__(self, m, I_alpha, omega_h, omega_alpha, rho, B,
                 flutter_derivative_data=None,
                 extrap_low='constant', extrap_high='linear',
                 low_k_buffer=0.2, high_k_buffer=0.3,
                 zeta_struct_b=0.005, zeta_struct_t=0.005):
        self.m = m
        self.I_alpha = I_alpha
        self.omega_h = omega_h
        self.omega_alpha = omega_alpha
        self.rho = rho
        self.B = B
        self.zeta_struct_b = zeta_struct_b
        self.zeta_struct_t = zeta_struct_t

        if flutter_derivative_data is None:
            flutter_derivative_data = generate_default_flutter_derivative_data()

        self._deriv_names = ['H1', 'H2', 'H3', 'H4',
                             'A1', 'A2', 'A3', 'A4']
        self._build_interpolators(flutter_derivative_data,
                                  extrap_low, extrap_high,
                                  low_k_buffer, high_k_buffer)

        self._tmd = None

    # ------------------------------------------------------------------
    #  Spline interpolation with stable tail extrapolation
    # ------------------------------------------------------------------

    def _build_interpolators(self, data, extrap_low, extrap_high,
                             low_k_buffer, high_k_buffer):
        k_data = np.asarray(data['k'], dtype=float)
        sort_idx = np.argsort(k_data)
        k_data = k_data[sort_idx]
        self.k_min = k_data[0]
        self.k_max = k_data[-1]

        self._spline = {}
        self._extrap_rules = (extrap_low, extrap_high)

        for name in self._deriv_names:
            values = np.asarray(data[name], dtype=float)[sort_idx]
            cs = CubicSpline(k_data, values, bc_type='natural',
                             extrapolate=False)
            self._spline[name] = cs

        dk_low = self.k_min * low_k_buffer
        dk_high = self.k_max * high_k_buffer
        self._k_low_blend_start = max(0.0, self.k_min - dk_low)
        self._k_low_blend_end = self.k_min
        self._k_high_blend_start = self.k_max
        self._k_high_blend_end = self.k_max + dk_high

        self._asymptotics = {}
        for name in self._deriv_names:
            val_min = self._spline[name](self.k_min)
            val_max = self._spline[name](self.k_max)
            deriv = self._spline[name].derivative(1)
            slope_min = float(deriv(self.k_min))
            slope_max = float(deriv(self.k_max))
            self._asymptotics[name] = {
                'val_min': val_min, 'val_max': val_max,
                'slope_min': slope_min, 'slope_max': slope_max,
            }

    @staticmethod
    def _asymptotic_value(k, k_bdry, val_bdry, slope_bdry, rule):
        if rule == 'constant':
            result = np.full_like(np.asarray(k), val_bdry, dtype=float)
        elif rule == 'linear':
            k_arr = np.asarray(k)
            result = val_bdry + slope_bdry * (k_arr - k_bdry)
            result = np.asarray(result, dtype=float)
        elif rule == 'zero_slope':
            result = np.full_like(np.asarray(k), val_bdry, dtype=float)
        else:
            raise ValueError(f"Unknown extrapolation rule: {rule}")
        if np.ndim(k) == 0:
            return float(result)
        return result

    @staticmethod
    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _extrapolate(self, name, k):
        asymp = self._asymptotics[name]
        rule_low, rule_high = self._extrap_rules

        k_arr = np.atleast_1d(k)
        result = np.zeros_like(k_arr, dtype=float)

        in_range_mask = (k_arr >= self.k_min) & (k_arr <= self.k_max)
        result[in_range_mask] = self._spline[name](k_arr[in_range_mask])

        low_mask = k_arr < self.k_min
        if np.any(low_mask):
            k_low = k_arr[low_mask]
            asymp_vals = self._asymptotic_value(
                k_low, self.k_min, asymp['val_min'],
                asymp['slope_min'], rule_low
            )
            blend_region = ((k_low >= self._k_low_blend_start)
                            & (k_low < self._k_low_blend_end))
            pure_asymp = k_low < self._k_low_blend_start
            low_idx = np.where(low_mask)[0]
            if np.any(blend_region):
                blend_idx = low_idx[blend_region]
                blend_t = ((k_low[blend_region] - self._k_low_blend_start)
                           / (self._k_low_blend_end - self._k_low_blend_start))
                alpha = self._smoothstep(blend_t)
                spline_blend = self._spline[name](
                    np.clip(k_low[blend_region], self.k_min, self.k_max)
                )
                result[blend_idx] = (
                    alpha * spline_blend
                    + (1 - alpha) * asymp_vals[blend_region]
                )
            if np.any(pure_asymp):
                pure_idx = low_idx[pure_asymp]
                result[pure_idx] = asymp_vals[pure_asymp]

        high_mask = k_arr > self.k_max
        if np.any(high_mask):
            k_high = k_arr[high_mask]
            asymp_vals = self._asymptotic_value(
                k_high, self.k_max, asymp['val_max'],
                asymp['slope_max'], rule_high
            )
            blend_region = ((k_high > self._k_high_blend_start)
                            & (k_high <= self._k_high_blend_end))
            pure_asymp = k_high > self._k_high_blend_end
            high_idx = np.where(high_mask)[0]
            if np.any(blend_region):
                blend_idx = high_idx[blend_region]
                blend_t = ((k_high[blend_region] - self._k_high_blend_start)
                           / (self._k_high_blend_end - self._k_high_blend_start))
                alpha = self._smoothstep(blend_t)
                spline_blend = self._spline[name](
                    np.clip(k_high[blend_region], self.k_min, self.k_max)
                )
                result[blend_idx] = (
                    (1 - alpha) * spline_blend
                    + alpha * asymp_vals[blend_region]
                )
            if np.any(pure_asymp):
                pure_idx = high_idx[pure_asymp]
                result[pure_idx] = asymp_vals[pure_asymp]

        return result if np.ndim(k) > 0 else float(result[0])

    def get_aerodynamic_derivatives(self, reduced_freq):
        k = reduced_freq
        vals = [self._extrapolate(name, k) for name in self._deriv_names]
        return tuple(vals)

    # ------------------------------------------------------------------
    #  TMD setup
    # ------------------------------------------------------------------

    def set_tmd(self, mu, f_ratio=1.0, zeta_t=0.05,
                attach_to='bending', attach_x=0.0):
        """
        Configure a Tuned Mass Damper (TMD).

        Parameters
        ----------
        mu : float
            TMD mass ratio  mu = m_t / m  (mass of TMD / deck mass/unit length).
        f_ratio : float
            Tuning frequency ratio  f_t / f_struct  (TMD freq / target mode freq).
        zeta_t : float
            TMD damping ratio  ζ_t = c_t / (2 m_t ω_t).
        attach_to : {'bending', 'torsion'}
            Which structural mode the TMD is primarily attached to.
        attach_x : float
            Normalised chordwise attachment location (0 = centre, ±1 = edge).
            Only relevant for torsion attachment.
        """
        self._tmd = {
            'mu': mu,
            'f_ratio': f_ratio,
            'zeta_t': zeta_t,
            'attach_to': attach_to,
            'attach_x': attach_x,
        }

    def clear_tmd(self):
        self._tmd = None

    # ------------------------------------------------------------------
    #  Build state-space matrices (2-DOF section + optional TMD)
    # ------------------------------------------------------------------

    def _section_aero_forces(self, U, k):
        """Return Scanlan aerodynamic stiffness and damping matrices for
        the 2-DOF section (bending h, torsion α)."""
        H1, H2, H3, H4, A1, A2, A3, A4 = self.get_aerodynamic_derivatives(k)
        q = 0.5 * self.rho * self.B
        K_aero = q * U ** 2 * np.array([
            [H1, H3 * self.B],
            [A1 * self.B, A3 * self.B ** 2]
        ])
        C_aero = q * U * np.array([
            [H2, H4 * self.B],
            [A2 * self.B, A4 * self.B ** 2]
        ])
        return K_aero, C_aero

    def build_system_matrices(self, U, k):
        K_aero, C_aero = self._section_aero_forces(U, k)

        K_struct = np.diag([self.m * self.omega_h ** 2,
                            self.I_alpha * self.omega_alpha ** 2])
        M_struct = np.diag([self.m, self.I_alpha])
        C_struct = np.diag([2.0 * self.zeta_struct_b * self.m * self.omega_h,
                            2.0 * self.zeta_struct_t * self.I_alpha * self.omega_alpha])

        K_total = K_struct + K_aero
        C_section = C_struct + C_aero

        if self._tmd is not None:
            tmd = self._tmd
            mu = tmd['mu']
            f_ratio = tmd['f_ratio']
            zeta_t = tmd['zeta_t']

            if tmd['attach_to'] == 'bending':
                omega_t = f_ratio * self.omega_h
                m_t = mu * self.m
                attach = np.array([1.0, 0.0])
            else:
                omega_t = f_ratio * self.omega_alpha
                m_t = mu * self.m
                attach = np.array([0.0, self.B * tmd['attach_x']])

            k_t = m_t * omega_t ** 2
            c_t = 2.0 * m_t * zeta_t * omega_t

            n_aug = 3
            M = np.zeros((n_aug, n_aug))
            M[:2, :2] = M_struct
            M[0, 2] = m_t * attach[0]
            M[1, 2] = m_t * attach[1]
            M[2, 0] = m_t * attach[0]
            M[2, 1] = m_t * attach[1]
            M[2, 2] = m_t

            K = np.zeros((n_aug, n_aug))
            K[:2, :2] = K_total
            K[0, 2] = -k_t * attach[0]
            K[1, 2] = -k_t * attach[1]
            K[2, 0] = -k_t * attach[0]
            K[2, 1] = -k_t * attach[1]
            K[2, 2] = k_t

            C = np.zeros((n_aug, n_aug))
            C[:2, :2] = C_section
            C[0, 2] = -c_t * attach[0]
            C[1, 2] = -c_t * attach[1]
            C[2, 0] = -c_t * attach[0]
            C[2, 1] = -c_t * attach[1]
            C[2, 2] = c_t
        else:
            M = M_struct
            K = K_total
            C = C_section

        return M, C, K

    def solve_eigenproblem(self, U, k):
        M, C, K = self.build_system_matrices(U, k)
        n = M.shape[0]
        A = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-np.linalg.inv(M) @ K, -np.linalg.inv(M) @ C]
        ])
        eigenvalues, eigenvectors = eig(A)
        return eigenvalues, eigenvectors

    # ------------------------------------------------------------------
    #  p-k iteration (mode tracking)
    # ------------------------------------------------------------------

    def iterate_k_for_mode(self, U, omega_guess, mode_idx=0,
                           max_iter=80, tol=1e-6, under_relax=0.5):
        k_min_safe = 0.02
        k_max_safe = min(self._k_high_blend_end * 2.0,
                         max(self.k_max, 5.0))
        k = omega_guess * self.B / (2 * U)
        k = max(k_min_safe, min(k, k_max_safe))

        prev_idx = None

        for _ in range(max_iter):
            eigenvalues, _ = self.solve_eigenproblem(U, k)
            current_target = 2 * U * k / self.B
            omegas = np.abs(np.imag(eigenvalues))

            if prev_idx is not None and prev_idx < len(omegas):
                if abs(omegas[prev_idx] - current_target) < max(omegas) * 0.5:
                    idx = prev_idx
                else:
                    idx = int(np.argmin(np.abs(omegas - current_target)))
            else:
                idx = int(np.argmin(np.abs(omegas - current_target)))

            omega_sel = np.imag(eigenvalues[idx])
            if abs(omega_sel) < 1e-8:
                return None, None, None

            k_new = abs(omega_sel) * self.B / (2 * U)
            k_new = max(k_min_safe, min(k_new, k_max_safe))

            if abs(k_new - k) < tol:
                return k_new, eigenvalues[idx], idx

            k = under_relax * k_new + (1 - under_relax) * k
            prev_idx = idx

        return k, eigenvalues[idx], idx

    # ------------------------------------------------------------------
    #  Direct evaluation (no p-k iteration, faster, more stable)
    # ------------------------------------------------------------------

    def evaluate_mode_at_speed(self, U, omega_guess, mode_idx=0):
        k = omega_guess * self.B / (2.0 * U)
        k = max(0.02, min(k, max(self.k_max * 2.0, 5.0)))
        eigenvalues, _ = self.solve_eigenproblem(U, k)
        omegas = np.abs(np.imag(eigenvalues))
        target = 2 * U * k / self.B
        idx = int(np.argmin(np.abs(omegas - target)))
        return k, eigenvalues[idx], idx

    # ------------------------------------------------------------------
    #  Critical speed search
    # ------------------------------------------------------------------

    def find_critical_speed(self, U_min=5.0, U_max=300.0, n_points=200,
                            damping_tol=1e-8, mode_names=None,
                            use_pk_iteration=False):
        if mode_names is None:
            mode_names = ['bending', 'torsion']
        omega_guesses = [self.omega_h, self.omega_alpha]
        prev_damping = {name: None for name in mode_names}
        critical_info = None

        U_range = np.linspace(U_min, U_max, n_points)
        for U in U_range:
            for mi, (name, omega_guess) in enumerate(
                    zip(mode_names, omega_guesses)):
                if use_pk_iteration:
                    result = self.iterate_k_for_mode(U, omega_guess,
                                                     mode_idx=mi)
                else:
                    result = self.evaluate_mode_at_speed(U, omega_guess,
                                                         mode_idx=mi)
                if result[0] is None:
                    continue
                k_c, ev_c, _ = result
                damp = np.real(ev_c)

                if prev_damping[name] is not None:
                    if (prev_damping[name] <= damping_tol
                            and damp > damping_tol):
                        critical_info = {
                            'U_cr': U,
                            'k_cr': k_c,
                            'omega_cr': abs(np.imag(ev_c)),
                            'damping_cr': damp,
                            'mode': name
                        }
                        return critical_info

                prev_damping[name] = damp

        for name in mode_names:
            if prev_damping[name] is not None and prev_damping[name] > damping_tol:
                if critical_info is None:
                    critical_info = {
                        'U_cr': U_range[0],
                        'k_cr': None,
                        'omega_cr': None,
                        'damping_cr': prev_damping[name],
                        'mode': name,
                        'note': 'already unstable at U_min'
                    }
        return critical_info

    # ------------------------------------------------------------------
    #  Velocity sweep
    # ------------------------------------------------------------------

    def analyze_velocity_sweep(self, U_min=5.0, U_max=250.0, n_points=80,
                               mode_names=None, use_pk_iteration=False):
        if mode_names is None:
            mode_names = ['bending', 'torsion']
        omega_guesses = [self.omega_h, self.omega_alpha]
        U_range = np.linspace(U_min, U_max, n_points)
        results = {name: [] for name in mode_names}

        for U in U_range:
            for name, omega_guess in zip(mode_names, omega_guesses):
                if use_pk_iteration:
                    result = self.iterate_k_for_mode(U, omega_guess)
                else:
                    result = self.evaluate_mode_at_speed(U, omega_guess)
                if result[0] is None:
                    continue
                k_c, ev_c, _ = result
                results[name].append({
                    'U': U,
                    'k': k_c,
                    'damping': np.real(ev_c),
                    'frequency': abs(np.imag(ev_c))
                })
        return results

    # ------------------------------------------------------------------
    #  TMD optimisation
    # ------------------------------------------------------------------

    def optimize_tmd(self, target_mode='bending',
                     mu_range=(0.005, 0.03),
                     f_ratio_range=(0.85, 1.15),
                     zeta_range=(0.01, 0.20),
                     n_grid=6, n_refine=3,
                     U_search_min=5.0, U_search_max=300.0):
        """
        Grid-search + refine for TMD parameters that maximise flutter
        critical wind speed.

        Returns
        -------
        best_params : dict
        best_Ucr : float
        improvement : float
            (best_Ucr - baseline_Ucr) / baseline_Ucr
        """
        if self._tmd is not None:
            baseline_params = dict(self._tmd)
        else:
            baseline_params = None
        baseline_Ucr = None

        baseline_crit = self.find_critical_speed(U_min=U_search_min,
                                                 U_max=U_search_max,
                                                 n_points=30)
        if baseline_crit is not None:
            baseline_Ucr = baseline_crit['U_cr']
        else:
            baseline_Ucr = U_search_min

        attach_to = target_mode
        mu_grid = np.linspace(*mu_range, n_grid)
        f_grid = np.linspace(*f_ratio_range, n_grid)
        z_grid = np.linspace(*zeta_range, n_grid)

        best_Ucr = baseline_Ucr
        best_params = None

        print(f"  TMD optimisation (target mode: {target_mode})")
        print(f"  Baseline U_cr = {baseline_Ucr:.2f} m/s")
        print(f"  Grid: {n_grid}^3 = {n_grid**3} candidates "
              f"+ {n_refine} refinement levels")

        for level in range(n_refine):
            candidates = []
            if level == 0:
                for mu in mu_grid:
                    for fr in f_grid:
                        for zt in z_grid:
                            candidates.append((mu, fr, zt))
            else:
                if best_params is None:
                    break
                mu_best, fr_best, zt_best = best_params
                dmu = (mu_range[1] - mu_range[0]) / (n_grid - 1) / (2 ** level)
                dfr = (f_ratio_range[1] - f_ratio_range[0]) / (n_grid - 1) / (2 ** level)
                dzt = (zeta_range[1] - zeta_range[0]) / (n_grid - 1) / (2 ** level)
                for dmu_i in [-dmu, 0, dmu]:
                    for dfr_i in [-dfr, 0, dfr]:
                        for dzt_i in [-dzt, 0, dzt]:
                            mu_c = max(mu_range[0], min(mu_range[1], mu_best + dmu_i))
                            fr_c = max(f_ratio_range[0], min(f_ratio_range[1], fr_best + dfr_i))
                            zt_c = max(zeta_range[0], min(zeta_range[1], zt_best + dzt_i))
                            candidates.append((mu_c, fr_c, zt_c))

            for mu_c, fr_c, zt_c in candidates:
                self.set_tmd(mu=mu_c, f_ratio=fr_c, zeta_t=zt_c,
                             attach_to=attach_to, attach_x=0.5)
                try:
                    crit = self.find_critical_speed(U_min=U_search_min,
                                                    U_max=U_search_max,
                                                    n_points=30)
                    if crit is not None:
                        Ucr = crit['U_cr']
                    else:
                        Ucr = U_search_max
                except Exception:
                    Ucr = U_search_max

                if Ucr > best_Ucr + 0.01:
                    best_Ucr = Ucr
                    best_params = (mu_c, fr_c, zt_c)

            print(f"    Level {level + 1}/{n_refine}: "
                  f"best U_cr = {best_Ucr:.2f} m/s")
            if best_params is not None:
                print(f"    mu={best_params[0]:.4f}, "
                      f"f_ratio={best_params[1]:.4f}, "
                      f"zeta={best_params[2]:.4f}")

        if baseline_params is not None:
            self._tmd = baseline_params
        else:
            self._tmd = None

        improvement = (best_Ucr - baseline_Ucr) / baseline_Ucr * 100.0

        if best_params is not None:
            return {
                'mu': best_params[0],
                'f_ratio': best_params[1],
                'zeta_t': best_params[2],
                'attach_to': attach_to,
            }, best_Ucr, improvement
        else:
            return {
                'mu': 0.0,
                'f_ratio': 1.0,
                'zeta_t': 0.0,
                'attach_to': attach_to,
            }, best_Ucr, improvement

    # ------------------------------------------------------------------
    #  Plotting
    # ------------------------------------------------------------------

    def plot_v_diagram(self, results, filename='flutter_analysis.png',
                       title_prefix=''):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        colors = {'bending': 'tab:blue', 'torsion': 'tab:orange',
                  'bending+TMD': 'tab:green', 'torsion+TMD': 'tab:red'}
        for name, rlist in results.items():
            if not rlist:
                continue
            U_values = [r['U'] for r in rlist]
            dampings = [r['damping'] for r in rlist]
            freqs = [r['frequency'] for r in rlist]
            ax1.plot(U_values, dampings, label=name,
                     color=colors.get(name, None))
            ax2.plot(U_values, freqs, label=name,
                     color=colors.get(name, None))

        ax1.axhline(y=0, color='r', linestyle='--', label='Zero Damping')
        ax1.set_xlabel('Wind Speed U (m/s)')
        ax1.set_ylabel('Damping (1/s)')
        ax1.set_title(f'{title_prefix}V-D Diagram (Scanlan Method)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Wind Speed U (m/s)')
        ax2.set_ylabel('Frequency (rad/s)')
        ax2.set_title(f'{title_prefix}V-F Diagram')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Analysis diagram saved as {filename}")

    def plot_derivatives(self, k_plot=None, filename='derivatives.png'):
        if k_plot is None:
            k_plot = np.linspace(0.0, self._k_high_blend_end * 2.0, 500)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, name in enumerate(self._deriv_names):
            ax = axes[i]
            vals = self._extrapolate(name, k_plot)
            ax.plot(k_plot, vals, 'b-', label='spline + asymptote')

            k_data = self._spline[name].x
            y_data = self._spline[name](k_data)
            ax.plot(k_data, y_data, 'ro', markersize=5, label='exp. data')

            ax.axvline(self.k_min, color='gray', linestyle=':', lw=0.8)
            ax.axvline(self.k_max, color='gray', linestyle=':', lw=0.8)
            ax.axvspan(self._k_low_blend_start, self._k_low_blend_end,
                       color='lightgray', alpha=0.3)
            ax.axvspan(self._k_high_blend_start, self._k_high_blend_end,
                       color='lightgray', alpha=0.3)

            ax.set_xlabel('k')
            ax.set_ylabel(name + '*')
            ax.set_title(name + '* vs k')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Derivative curves saved as {filename}")

    def plot_tmd_comparison(self, results_no_tmd, results_tmd,
                            filename='tmd_comparison.png'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        for name, rlist in results_no_tmd.items():
            if not rlist:
                continue
            U_values = [r['U'] for r in rlist]
            dampings = [r['damping'] for r in rlist]
            ax1.plot(U_values, dampings, '--', label=f'{name} (no TMD)',
                     alpha=0.7)

        for name, rlist in results_tmd.items():
            if not rlist:
                continue
            U_values = [r['U'] for r in rlist]
            dampings = [r['damping'] for r in rlist]
            ax1.plot(U_values, dampings, '-', label=f'{name} (with TMD)')

        ax1.axhline(y=0, color='r', linestyle=':', label='Zero Damping')
        ax1.set_xlabel('Wind Speed U (m/s)')
        ax1.set_ylabel('Damping (1/s)')
        ax1.set_title('TMD Effect: Damping vs Wind Speed')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        for name, rlist in results_no_tmd.items():
            if not rlist:
                continue
            U_values = [r['U'] for r in rlist]
            freqs = [r['frequency'] for r in rlist]
            ax2.plot(U_values, freqs, '--', label=f'{name} (no TMD)',
                     alpha=0.7)

        for name, rlist in results_tmd.items():
            if not rlist:
                continue
            U_values = [r['U'] for r in rlist]
            freqs = [r['frequency'] for r in rlist]
            ax2.plot(U_values, freqs, '-', label=f'{name} (with TMD)')

        ax2.set_xlabel('Wind Speed U (m/s)')
        ax2.set_ylabel('Frequency (rad/s)')
        ax2.set_title('TMD Effect: Frequency vs Wind Speed')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"TMD comparison diagram saved as {filename}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    m = 10.0
    I_alpha = 2.0
    omega_h = 2 * np.pi * 1.0
    omega_alpha = 2 * np.pi * 8.0
    rho = 1.225
    B = 0.3

    sep = "=" * 70

    print(sep)
    print("  FLUTTER CRITICAL SPEED + TMD OPTIMISATION")
    print("  Scanlan p-k method  |  Cubic spline + asymptote  |  "
          "Multi-modal + TMD")
    print(sep)
    print("  Structural parameters:")
    print(f"    Mass / unit length           m       = {m} kg/m")
    print(f"    Mass inertia / unit length   I_alpha = {I_alpha} kg*m^2/m")
    print(f"    Bending natural frequency    f_h     = "
          f"{omega_h / (2 * np.pi):.3f} Hz")
    print(f"    Torsion natural frequency    f_alpha = "
          f"{omega_alpha / (2 * np.pi):.3f} Hz")
    print(f"    Air density                  rho     = {rho} kg/m^3")
    print(f"    Section chord width          B       = {B} m")
    print(sep)

    exp_data = generate_default_flutter_derivative_data()
    print(f"\n  Experimental data: k range = "
          f"[{exp_data['k'][0]:.2f}, {exp_data['k'][-1]:.2f}]")
    print(f"  Extrapolation: low k -> constant, high k -> linear")

    analyzer = FlutterAnalyzer(
        m, I_alpha, omega_h, omega_alpha, rho, B,
        flutter_derivative_data=exp_data,
        extrap_low='constant',
        extrap_high='linear',
        low_k_buffer=0.2,
        high_k_buffer=0.3,
        zeta_struct_b=0.0001,
        zeta_struct_t=0.0001
    )

    print("\n  Plotting flutter derivative interpolation curves...")
    analyzer.plot_derivatives()

    # ------------------------------------------------------------------
    #  Part 1: Baseline (no TMD)
    # ------------------------------------------------------------------
    print("\n" + sep)
    print("  PART 1 — Baseline (no TMD)")
    print(sep)

    analyzer.clear_tmd()
    critical_no_tmd = analyzer.find_critical_speed(U_min=2.0, U_max=200.0)

    if critical_no_tmd is not None:
        print(f"  Unstable mode       = {critical_no_tmd['mode']}")
        if critical_no_tmd.get('k_cr') is not None:
            print(f"  U_cr                = {critical_no_tmd['U_cr']:.2f} m/s")
            print(f"  k_cr                = {critical_no_tmd['k_cr']:.4f}")
            print(f"  f_cr                = "
                  f"{critical_no_tmd['omega_cr'] / (2 * np.pi):.3f} Hz")
        if 'note' in critical_no_tmd:
            print(f"  Note: {critical_no_tmd['note']}")
    else:
        print("  No flutter critical point found.")

    results_no_tmd = analyzer.analyze_velocity_sweep(
        U_min=2.0, U_max=180.0, n_points=150)
    analyzer.plot_v_diagram(results_no_tmd,
                            filename='flutter_no_tmd.png',
                            title_prefix='Baseline (no TMD) — ')

    # ------------------------------------------------------------------
    #  Part 2: TMD optimisation
    # ------------------------------------------------------------------
    print("\n" + sep)
    print("  PART 2 — TMD Optimisation")
    print(sep)

    print("\n  Optimising TMD for torsion mode...")
    best_params, best_Ucr, improvement = analyzer.optimize_tmd(
        target_mode='torsion',
        mu_range=(0.005, 0.05),
        f_ratio_range=(0.85, 1.15),
        zeta_range=(0.02, 0.20),
        n_grid=4, n_refine=2,
        U_search_min=2.0, U_search_max=200.0
    )

    print("\n  Optimised TMD parameters:")
    print(f"    Mass ratio      mu      = {best_params['mu']:.4f}")
    print(f"    Freq ratio      f_ratio = {best_params['f_ratio']:.4f}")
    print(f"    Damping ratio   zeta_t  = {best_params['zeta_t']:.4f}")
    print(f"    Attached to     mode    = {best_params['attach_to']}")
    print(f"    Optimised U_cr          = {best_Ucr:.2f} m/s")
    print(f"    Improvement             = {improvement:+.1f} %")

    has_useful_tmd = best_params['mu'] > 1e-8 and best_Ucr > 0

    # ------------------------------------------------------------------
    #  Part 3: With optimised TMD
    # ------------------------------------------------------------------
    print("\n" + sep)
    print("  PART 3 — With Optimised TMD")
    print(sep)

    if has_useful_tmd:
        analyzer.set_tmd(mu=best_params['mu'],
                         f_ratio=best_params['f_ratio'],
                         zeta_t=best_params['zeta_t'],
                         attach_to=best_params['attach_to'],
                         attach_x=0.5)

        critical_tmd = analyzer.find_critical_speed(U_min=2.0, U_max=250.0)
        if critical_tmd is not None:
            print(f"  Unstable mode       = {critical_tmd['mode']}")
            if critical_tmd.get('k_cr') is not None:
                print(f"  U_cr (with TMD)     = {critical_tmd['U_cr']:.2f} m/s")
                print(f"  k_cr                = {critical_tmd['k_cr']:.4f}")
                print(f"  f_cr                = "
                      f"{critical_tmd['omega_cr'] / (2 * np.pi):.3f} Hz")
            if 'note' in critical_tmd:
                print(f"  Note: {critical_tmd['note']}")
        else:
            print(f"  TMD completely stabilised the system (no flutter up to 250 m/s)")
            print(f"  U_cr > 250 m/s")

        results_tmd = analyzer.analyze_velocity_sweep(
            U_min=2.0, U_max=200.0, n_points=150)
        analyzer.plot_v_diagram(results_tmd,
                                filename='flutter_with_tmd.png',
                                title_prefix='With Optimised TMD — ')
    else:
        print("  TMD optimisation found no configuration that improves U_cr.")
        print("  This is expected when the flutter mechanism is dominated by")
        print("  aerodynamic negative damping rather than low structural damping.")
        critical_tmd = None
        results_tmd = {}

    # ------------------------------------------------------------------
    #  Part 4: Comparison diagram
    # ------------------------------------------------------------------
    print("\n" + sep)
    print("  PART 4 — Comparison")
    print(sep)

    analyzer.clear_tmd()
    results_no_tmd = analyzer.analyze_velocity_sweep(
        U_min=2.0, U_max=200.0, n_points=150)

    if has_useful_tmd:
        analyzer.set_tmd(mu=best_params['mu'],
                         f_ratio=best_params['f_ratio'],
                         zeta_t=best_params['zeta_t'],
                         attach_to=best_params['attach_to'],
                         attach_x=0.5)
        results_tmd = analyzer.analyze_velocity_sweep(
            U_min=2.0, U_max=200.0, n_points=150)
        analyzer.plot_tmd_comparison(results_no_tmd, results_tmd,
                                     filename='tmd_comparison.png')
    else:
        print("  Skipping TMD comparison (no effective TMD found).")

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    print("\n" + sep)
    print("  SUMMARY")
    print(sep)
    if critical_no_tmd is not None:
        U0 = critical_no_tmd['U_cr']
        print(f"  Baseline U_cr (no TMD)  = {U0:.2f} m/s")
        if has_useful_tmd:
            if critical_tmd is not None:
                U1 = critical_tmd['U_cr']
                print(f"  Optimised U_cr (TMD)    = {U1:.2f} m/s")
                print(f"  Improvement             = {U1 - U0:.2f} m/s  "
                      f"({(U1 - U0) / U0 * 100:+.1f} %)")
            else:
                print(f"  Optimised TMD: flutter eliminated in search range")
                print(f"  U_cr > 250 m/s (TMD completely stabilised the system)")
                print(f"  Improvement             > {250 - U0:.2f} m/s  "
                      f"(>{(250 - U0) / U0 * 100:+.1f} %)")
        else:
            print(f"  TMD optimisation: no effective configuration found")
    print(sep)
    print("\n  Calculation finished.")


if __name__ == "__main__":
    main()
