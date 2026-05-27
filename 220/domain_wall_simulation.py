import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class DomainWallSimulator:
    def __init__(self, params):
        self.N = params.get('N', 200)
        self.dx = params.get('dx', 1e-9)
        self.alpha = params.get('alpha', 0.1)
        self.A = params.get('A', 1e-11)
        self.K = params.get('K', 1e5)
        self.Ms = params.get('Ms', 8e5)
        self.gamma = params.get('gamma', 1.76e11)
        self.H_ext_Apm = params.get('H_ext', 1000)
        self.dt = params.get('dt', 1e-14)
        self.total_steps = params.get('total_steps', 5000)
        self.save_interval = params.get('save_interval', 50)
        self.method = params.get('method', 'heun_projected')
        self.gs_max_iter = params.get('gs_max_iter', 10)
        self.gs_tol = params.get('gs_tol', 1e-8)

        self.mu0 = 4 * np.pi * 1e-7
        self.H_ext = self.mu0 * self.H_ext_Apm
        self.x = np.linspace(-self.N * self.dx / 2, self.N * self.dx / 2, self.N)

        self.M = np.zeros((self.N, 3))
        self.H_eff = np.zeros((self.N, 3))

        self.delta0 = np.sqrt(self.A / self.K)

        self.time_history = []
        self.position_history = []
        self.width_history = []
        self.M_history = []
        self.norm_history = []
        self.norm_max_history = []
        self.norm_min_history = []
        self.gs_iter_history = []

        self.step_function = self._get_step_function()

    def _get_step_function(self):
        methods = {
            'heun_original': self.heun_original_step,
            'heun_projected': self.heun_projected_step,
            'euler_projected': self.euler_projected_step,
            'gauss_seidel_projected': self.gauss_seidel_projected_step,
            'midpoint_projected': self.midpoint_projected_step
        }
        if self.method not in methods:
            raise ValueError(f"Unknown method: {self.method}. "
                           f"Available methods: {list(methods.keys())}")
        return methods[self.method]

    def initialize_domain_wall(self, wall_type='bloch', initial_position=0.0):
        for i in range(self.N):
            x_shifted = self.x[i] - initial_position
            theta = 2 * np.arctan(np.exp(x_shifted / self.delta0))
            if wall_type == 'bloch':
                self.M[i, 0] = np.cos(theta)
                self.M[i, 1] = np.sin(theta)
                self.M[i, 2] = 0.0
            else:
                self.M[i, 0] = np.cos(theta)
                self.M[i, 1] = 0.0
                self.M[i, 2] = np.sin(theta)
        self.normalize_magnetization()

    def normalize_magnetization(self, M=None):
        if M is None:
            norm = np.linalg.norm(self.M, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            self.M = self.M / norm
            return self.M
        else:
            norm = np.linalg.norm(M, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            return M / norm

    def project_to_unit_sphere(self, M):
        norm = np.linalg.norm(M, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        return M / norm

    def compute_effective_field(self, M=None):
        if M is None:
            M = self.M

        H_exchange = np.zeros_like(M)
        for i in range(1, self.N - 1):
            H_exchange[i] = (2 * self.A / (self.mu0 * self.Ms * self.dx**2)) * \
                           (M[i+1] + M[i-1] - 2 * M[i])
        H_exchange[0] = H_exchange[1]
        H_exchange[-1] = H_exchange[-2]

        H_anisotropy = np.zeros_like(M)
        H_anisotropy[:, 0] = (2 * self.K / (self.mu0 * self.Ms)) * M[:, 0]

        H_external = np.zeros_like(M)
        H_external[:, 0] = self.H_ext_Apm

        return H_exchange + H_anisotropy + H_external

    def llg_rhs(self, M):
        H_eff = self.compute_effective_field(M)

        M_cross_H = np.cross(M, H_eff)
        M_cross_McrossH = np.cross(M, M_cross_H)

        precession = -self.gamma * self.mu0 * M_cross_H
        damping = -self.alpha * self.gamma * self.mu0 * M_cross_McrossH

        rhs = (precession + damping) / (1 + self.alpha**2)
        return rhs

    def llg_rhs_single(self, m, H_eff):
        m_cross_H = np.cross(m, H_eff)
        m_cross_mcrossH = np.cross(m, m_cross_H)

        precession = -self.gamma * self.mu0 * m_cross_H
        damping = -self.alpha * self.gamma * self.mu0 * m_cross_mcrossH

        return (precession + damping) / (1 + self.alpha**2)

    def heun_original_step(self):
        k1 = self.llg_rhs(self.M)
        M_pred = self.M + self.dt * k1

        norm_pred = np.linalg.norm(M_pred, axis=1, keepdims=True)
        norm_pred[norm_pred == 0] = 1.0
        M_pred = M_pred / norm_pred

        k2 = self.llg_rhs(M_pred)
        self.M = self.M + 0.5 * self.dt * (k1 + k2)

        self.normalize_magnetization()

    def heun_projected_step(self):
        k1 = self.llg_rhs(self.M)
        M_pred = self.M + self.dt * k1
        M_pred = self.project_to_unit_sphere(M_pred)

        k2 = self.llg_rhs(M_pred)
        M_new = self.M + 0.5 * self.dt * (k1 + k2)
        self.M = self.project_to_unit_sphere(M_new)

    def euler_projected_step(self):
        dMdt = self.llg_rhs(self.M)
        M_new = self.M + self.dt * dMdt
        self.M = self.project_to_unit_sphere(M_new)

    def midpoint_projected_step(self):
        k1 = self.llg_rhs(self.M)
        M_half = self.M + 0.5 * self.dt * k1
        M_half = self.project_to_unit_sphere(M_half)

        k2 = self.llg_rhs(M_half)
        M_new = self.M + self.dt * k2
        self.M = self.project_to_unit_sphere(M_new)

    def gauss_seidel_projected_step(self):
        H_eff = self.compute_effective_field(self.M)
        M_old = self.M.copy()
        M_new = self.M.copy()
        total_iter = 0

        for iter_num in range(self.gs_max_iter):
            max_delta = 0.0
            for i in range(self.N):
                if i > 0 and i < self.N - 1:
                    H_ex_i = (2 * self.A / (self.mu0 * self.Ms * self.dx**2)) * \
                            (M_new[i+1] + M_new[i-1] - 2 * M_new[i])
                elif i == 0:
                    H_ex_i = (2 * self.A / (self.mu0 * self.Ms * self.dx**2)) * \
                            (M_new[i+1] - M_new[i])
                else:
                    H_ex_i = (2 * self.A / (self.mu0 * self.Ms * self.dx**2)) * \
                            (M_new[i-1] - M_new[i])

                H_an_i = (2 * self.K / (self.mu0 * self.Ms)) * M_new[i, 0]
                H_ext_i = self.H_ext_Apm

                H_eff_i = np.array([H_ex_i[0] + H_an_i + H_ext_i, H_ex_i[1], H_ex_i[2]])

                rhs = self.llg_rhs_single(M_new[i], H_eff_i)

                m_new = M_old[i] + self.dt * rhs
                m_new = self.project_to_unit_sphere(m_new.reshape(1, 3)).reshape(3)

                delta = np.max(np.abs(m_new - M_new[i]))
                max_delta = max(max_delta, delta)

                M_new[i] = m_new

            total_iter = iter_num + 1
            if max_delta < self.gs_tol:
                break

        self.M = M_new

        if hasattr(self, 'gs_iter_history'):
            self.gs_iter_history.append(total_iter)

        return total_iter

    def compute_magnetization_norm_stats(self):
        norms = np.linalg.norm(self.M, axis=1)
        return {
            'mean': np.mean(norms),
            'max': np.max(norms),
            'min': np.min(norms),
            'std': np.std(norms),
            'max_error': np.max(np.abs(norms - 1.0))
        }

    def compute_domain_wall_position(self):
        m_x = self.M[:, 0]
        center_idx = np.argmin(np.abs(m_x))
        if 0 < center_idx < self.N - 1:
            m1, m2 = m_x[center_idx - 1], m_x[center_idx + 1]
            x1, x2 = self.x[center_idx - 1], self.x[center_idx + 1]
            if abs(m2 - m1) > 1e-10:
                t = -m1 / (m2 - m1)
                return x1 + t * (x2 - x1)
        return self.x[center_idx]

    def compute_domain_wall_width(self):
        m_x = self.M[:, 0]
        m_x_abs = np.abs(m_x)

        dw_position = self.compute_domain_wall_position()

        idx_left = np.where((m_x_abs >= 0.75) & (self.x < dw_position))[0]
        idx_right = np.where((m_x_abs >= 0.75) & (self.x > dw_position))[0]

        if len(idx_left) == 0 or len(idx_right) == 0:
            return self.delta0

        idx_outer_left = idx_left[-1]
        idx_outer_right = idx_right[0]

        idx_inner_left = idx_outer_left + 1
        idx_inner_right = idx_outer_right - 1

        if idx_inner_left >= self.N or idx_inner_right < 0:
            return self.delta0

        m_outer_left = m_x[idx_outer_left]
        m_inner_left = m_x[idx_inner_left]
        m_outer_right = m_x[idx_outer_right]
        m_inner_right = m_x[idx_inner_right]

        if abs(m_inner_left - m_outer_left) < 1e-10 or abs(m_inner_right - m_outer_right) < 1e-10:
            return self.delta0

        sign_left = np.sign(m_outer_left)
        t_left = (sign_left * 0.75 - m_outer_left) / (m_inner_left - m_outer_left)
        x075_left = self.x[idx_outer_left] + t_left * (self.x[idx_inner_left] - self.x[idx_outer_left])

        sign_right = np.sign(m_outer_right)
        t_right = (sign_right * 0.75 - m_outer_right) / (m_inner_right - m_outer_right)
        x075_right = self.x[idx_outer_right] + t_right * (self.x[idx_inner_right] - self.x[idx_outer_right])

        if x075_right <= x075_left:
            return self.delta0

        width = (x075_right - x075_left) / np.log(3)
        return abs(width)

    def run(self):
        self.initialize_domain_wall(wall_type='bloch', initial_position=0.0)

        norm_stats = self.compute_magnetization_norm_stats()
        print(f"\nInitial norm stats: max_error = {norm_stats['max_error']:.2e}")

        for step in tqdm(range(self.total_steps), desc=f"Simulating ({self.method})"):
            self.step_function()

            if step % self.save_interval == 0:
                current_time = step * self.dt
                position = self.compute_domain_wall_position()
                width = self.compute_domain_wall_width()
                norm_stats = self.compute_magnetization_norm_stats()

                self.time_history.append(current_time)
                self.position_history.append(position)
                self.width_history.append(width)
                self.M_history.append(self.M.copy())
                self.norm_history.append(norm_stats['mean'])
                self.norm_max_history.append(norm_stats['max'])
                self.norm_min_history.append(norm_stats['min'])

        return {
            'time': np.array(self.time_history),
            'position': np.array(self.position_history),
            'width': np.array(self.width_history),
            'M_history': self.M_history,
            'norm_mean': np.array(self.norm_history),
            'norm_max': np.array(self.norm_max_history),
            'norm_min': np.array(self.norm_min_history),
            'gs_iter': np.array(self.gs_iter_history) if self.gs_iter_history else None
        }

    def compute_velocity(self):
        if len(self.position_history) < 2:
            return 0.0

        times = np.array(self.time_history)
        positions = np.array(self.position_history)

        n_steady = len(times) // 2
        v, _ = np.polyfit(times[n_steady:], positions[n_steady:], 1)

        return v

    def analytical_velocity(self):
        return (self.gamma * self.alpha * self.H_ext * self.delta0) / (1 + self.alpha**2)

    def plot_results(self, results):
        has_gs = results.get('gs_iter') is not None
        nrows = 3 if has_gs else 2
        ncols = 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))

        if nrows == 1:
            axes = axes.reshape(1, -1)

        ax1 = axes[0, 0]
        ax1.plot(results['time'] * 1e9, results['position'] * 1e9, 'b-', linewidth=2, label='Numerical')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Domain Wall Position (nm)')
        ax1.set_title(f'Domain Wall Position vs Time ({self.method})')
        ax1.grid(True, alpha=0.3)

        velocity = self.compute_velocity()
        v_analytical = self.analytical_velocity()

        t_fit = results['time'][len(results['time'])//2:]
        pos0 = results['position'][len(results['time'])//2]
        ax1.plot(t_fit * 1e9, (pos0 + velocity * (t_fit - t_fit[0])) * 1e9,
                 'r--', linewidth=2, label=f'Linear fit')
        ax1.legend()

        ax1.text(0.05, 0.95, f'v = {velocity:.2e} m/s\nv_analytical = {v_analytical:.2e} m/s',
                 transform=ax1.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2 = axes[0, 1]
        ax2.plot(results['time'] * 1e9, np.array(results['width']) * 1e9, 'r-', linewidth=2)
        ax2.axhline(y=self.delta0 * 1e9, color='k', linestyle='--', label=f'Δ₀ = {self.delta0*1e9:.1f} nm')
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Domain Wall Width (nm)')
        ax2.set_title('Domain Wall Width vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        M_final = results['M_history'][-1]
        ax3.plot(self.x * 1e9, M_final[:, 0], 'b-', label='m_x', linewidth=2)
        ax3.plot(self.x * 1e9, M_final[:, 1], 'r-', label='m_y', linewidth=2)
        ax3.plot(self.x * 1e9, M_final[:, 2], 'g-', label='m_z', linewidth=2)
        ax3.set_xlabel('Position x (nm)')
        ax3.set_ylabel('Normalized Magnetization')
        ax3.set_title('Final Magnetization Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-1.1, 1.1)

        ax4 = axes[1, 1]
        times = results['time']
        norm_max = results['norm_max']
        norm_min = results['norm_min']
        norm_mean = results['norm_mean']

        ax4.fill_between(times * 1e9, norm_min, norm_max, alpha=0.3, label='Min-Max range')
        ax4.plot(times * 1e9, norm_mean, 'b-', linewidth=2, label='Mean |m|')
        ax4.axhline(y=1.0, color='k', linestyle='--', label='|m| = 1.0')
        ax4.set_xlabel('Time (ns)')
        ax4.set_ylabel('Magnetization Norm |m|')
        ax4.set_title('Magnetization Norm Conservation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        max_error = np.max(np.abs(norm_max - 1.0))
        ax4.text(0.05, 0.95, f'Max | |m| - 1 | = {max_error:.2e}',
                 transform=ax4.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if has_gs:
            ax5 = axes[2, 0]
            ax5.plot(times * 1e9, results['gs_iter'], 'g.-', linewidth=1)
            ax5.set_xlabel('Time (ns)')
            ax5.set_ylabel('Number of Iterations')
            ax5.set_title('Gauss-Seidel Iterations per Step')
            ax5.grid(True, alpha=0.3)

            ax6 = axes[2, 1]
            pos = results['position']
            if len(pos) > 2:
                velocity_inst = np.gradient(pos, times)
                ax6.plot(times * 1e9, velocity_inst, 'g-', linewidth=2, label='Instantaneous')
                ax6.axhline(y=velocity, color='r', linestyle='--', label=f'Steady-state = {velocity:.2e}')
                ax6.axhline(y=v_analytical, color='k', linestyle='--', label=f'Analytical = {v_analytical:.2e}')
            ax6.set_xlabel('Time (ns)')
            ax6.set_ylabel('Velocity (m/s)')
            ax6.set_title('Domain Wall Velocity')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'domain_wall_results_{self.method}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

    def animate_domain_wall(self, results):
        fig, ax = plt.subplots(figsize=(10, 6))

        line_x, = ax.plot([], [], 'b-', label='m_x', linewidth=2)
        line_y, = ax.plot([], [], 'r-', label='m_y', linewidth=2)
        line_z, = ax.plot([], [], 'g-', label='m_z', linewidth=2)
        pos_line = ax.axvline(x=0, color='k', linestyle='--', label='DW Position')

        ax.set_xlim(self.x[0] * 1e9, self.x[-1] * 1e9)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Position x (nm)')
        ax.set_ylabel('Normalized Magnetization')
        ax.set_title(f'Domain Wall Motion ({self.method})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        def init():
            line_x.set_data([], [])
            line_y.set_data([], [])
            line_z.set_data([], [])
            time_text.set_text('')
            return line_x, line_y, line_z, time_text, pos_line

        def update(frame):
            M = results['M_history'][frame]
            x_nm = self.x * 1e9

            line_x.set_data(x_nm, M[:, 0])
            line_y.set_data(x_nm, M[:, 1])
            line_z.set_data(x_nm, M[:, 2])

            pos = results['position'][frame] * 1e9
            pos_line.set_xdata([pos, pos])

            current_time = results['time'][frame] * 1e9
            time_text.set_text(f'Time = {current_time:.2f} ns')

            return line_x, line_y, line_z, time_text, pos_line

        anim = FuncAnimation(fig, update, frames=len(results['M_history']),
                             init_func=init, blit=True, interval=50)

        filename = f'domain_wall_motion_{self.method}.gif'
        anim.save(filename, writer='pillow', fps=20, dpi=100)
        plt.close()
        print(f"  Saved: {filename}")


def run_method_comparison(params_high_field):
    methods = ['heun_original', 'heun_projected', 'euler_projected',
               'midpoint_projected', 'gauss_seidel_projected']

    results_all = {}

    print("\n" + "=" * 70)
    print("NUMERICAL METHOD COMPARISON (High Field Stability Test)")
    print("=" * 70)

    for method in methods:
        print(f"\n{'─' * 70}")
        print(f"Testing method: {method}")
        print(f"{'─' * 70}")

        params = params_high_field.copy()
        params['method'] = method

        sim = DomainWallSimulator(params)

        try:
            results = sim.run()

            velocity = sim.compute_velocity()
            v_analytical = sim.analytical_velocity()

            max_norm_error = np.max(np.abs(results['norm_max'] - 1.0))
            avg_width = np.mean(results['width'][-len(results['width'])//2:])

            results_all[method] = {
                'velocity': velocity,
                'v_error': abs(velocity - v_analytical) / v_analytical * 100,
                'max_norm_error': max_norm_error,
                'avg_width': avg_width,
                'results': results,
                'sim': sim
            }

            print(f"\n  Results for {method}:")
            print(f"    Velocity: {velocity:.4e} m/s (error: {results_all[method]['v_error']:.2f}%)")
            print(f"    Max | |m| - 1 |: {max_norm_error:.2e}")
            print(f"    Average width: {avg_width*1e9:.2f} nm")

            if method == 'gauss_seidel_projected' and results.get('gs_iter') is not None:
                avg_iter = np.mean(results['gs_iter'])
                max_iter = np.max(results['gs_iter'])
                print(f"    Avg GS iterations: {avg_iter:.1f}, Max: {max_iter}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results_all[method] = None

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n{'Method':<25} {'v (m/s)':<15} {'Error (%)':<12} {'Max | |m|-1 |':<15} {'Status':<10}")
    print("-" * 77)

    for method in methods:
        if results_all[method] is not None:
            r = results_all[method]
            status = "✓ Stable" if r['max_norm_error'] < 1e-6 else "⚠ Drift"
            if r['v_error'] > 10:
                status = "✗ Inaccurate"
            print(f"{method:<25} {r['velocity']:<15.4e} {r['v_error']:<12.2f} "
                  f"{r['max_norm_error']:<15.2e} {status:<10}")
        else:
            print(f"{method:<25} {'N/A':<15} {'N/A':<12} {'N/A':<15} {'✗ Failed':<10}")

    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS")
    print("=" * 70)

    if results_all.get('heun_original') and results_all.get('heun_projected'):
        improvement = (results_all['heun_original']['max_norm_error'] /
                      max(results_all['heun_projected']['max_norm_error'], 1e-16))
        print(f"\n  Projection Heun vs Original Heun:")
        print(f"    Norm error improvement factor: {improvement:.1f}×")

    if results_all.get('gauss_seidel_projected'):
        gs_err = results_all['gauss_seidel_projected']['max_norm_error']
        heun_err = results_all.get('heun_projected', {}).get('max_norm_error', 1)
        if gs_err < heun_err:
            print(f"\n  Gauss-Seidel implicit method:")
            print(f"    Most stable, norm error = {gs_err:.2e}")

    print("\nGenerating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for method in methods:
        if results_all.get(method) and results_all[method] is not None:
            r = results_all[method]['results']
            ax1.plot(r['time'] * 1e9, r['position'] * 1e9, label=method, linewidth=2)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Position (nm)')
    ax1.set_title('Domain Wall Position vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for method in methods:
        if results_all.get(method) and results_all[method] is not None:
            r = results_all[method]['results']
            norm_error = np.abs(r['norm_max'] - 1.0)
            ax2.semilogy(r['time'] * 1e9, norm_error, label=method, linewidth=2)
    ax2.axhline(y=1e-12, color='k', linestyle='--', label='Machine precision')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('| |m| - 1 |')
    ax2.set_title('Magnetization Norm Error (log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    ax3 = axes[1, 0]
    for method in methods:
        if results_all.get(method) and results_all[method] is not None:
            r = results_all[method]['results']
            velocities = np.gradient(r['position'], r['time'])
            ax3.plot(r['time'] * 1e9, velocities, label=method, linewidth=2)
    if results_all.get('heun_projected'):
        v_analytical = results_all['heun_projected']['sim'].analytical_velocity()
        ax3.axhline(y=v_analytical, color='k', linestyle='--', label=f'Analytical = {v_analytical:.2e}')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Domain Wall Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    method_names = []
    v_errors = []
    norm_errors = []
    for method in methods:
        if results_all.get(method) and results_all[method] is not None:
            method_names.append(method)
            v_errors.append(results_all[method]['v_error'])
            norm_errors.append(results_all[method]['max_norm_error'])

    x = np.arange(len(method_names))
    width = 0.35
    ax4.bar(x - width/2, v_errors, width, label='Velocity error (%)', color='b', alpha=0.7)
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x + width/2, norm_errors, width, label='Norm error', color='r', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(method_names, rotation=45, ha='right')
    ax4.set_ylabel('Velocity Error (%)', color='b')
    ax4_twin.set_ylabel('Max Norm Error', color='r')
    ax4.set_title('Error Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150)
    plt.close()
    print("  Saved: method_comparison.png")

    return results_all


def main():
    params_normal = {
        'N': 200,
        'dx': 1e-9,
        'alpha': 0.1,
        'A': 1e-11,
        'K': 1e5,
        'Ms': 8e5,
        'gamma': 1.76e11,
        'H_ext': 15000,
        'dt': 1e-14,
        'total_steps': 10000,
        'save_interval': 100,
        'method': 'heun_projected'
    }

    params_high_field = {
        'N': 200,
        'dx': 1e-9,
        'alpha': 0.05,
        'A': 1e-11,
        'K': 1e5,
        'Ms': 8e5,
        'gamma': 1.76e11,
        'H_ext': 100000,
        'dt': 5e-15,
        'total_steps': 5000,
        'save_interval': 50,
        'gs_max_iter': 15,
        'gs_tol': 1e-10
    }

    print("=" * 70)
    print("Domain Wall Simulation with Improved Numerical Stability")
    print("=" * 70)

    run_mode = 'compare'

    if run_mode == 'compare':
        results_all = run_method_comparison(params_high_field)

        if results_all.get('heun_projected'):
            print(f"\n\n{'Generating detailed plots for heun_projected':^70}")
            print("-" * 70)
            sim = results_all['heun_projected']['sim']
            results = results_all['heun_projected']['results']
            sim.plot_results(results)
            sim.animate_domain_wall(results)

    else:
        print(f"\n{'Material Parameters':^70}")
        print("-" * 70)
        print(f"  Damping coefficient (α): {params_normal['alpha']:.3f}")
        print(f"  Exchange constant (A): {params_normal['A']:.2e} J/m")
        print(f"  Anisotropy constant (K): {params_normal['K']:.2e} J/m³")
        print(f"  Saturation magnetization (Ms): {params_normal['Ms']:.2e} A/m")
        print(f"  Gyromagnetic ratio (γ): {params_normal['gamma']:.2e} rad/(s·T)")
        print(f"  External field (H): {params_normal['H_ext']:.0f} A/m = {params_normal['H_ext']*4e-7*np.pi:.4f} T")

        print(f"\n{'Numerical Parameters':^70}")
        print("-" * 70)
        print(f"  Method: {params_normal['method']}")
        print(f"  Number of grid points: {params_normal['N']}")
        print(f"  Grid spacing (dx): {params_normal['dx']*1e9:.1f} nm")
        print(f"  Time step (dt): {params_normal['dt']*1e15:.1f} fs")
        print(f"  Total steps: {params_normal['total_steps']}")
        print(f"  Simulation time: {params_normal['total_steps']*params_normal['dt']*1e9:.1f} ns")

        simulator = DomainWallSimulator(params_normal)

        print(f"\n{'Analytical Results':^70}")
        print("-" * 70)
        print(f"  Domain wall width (Δ₀ = √(A/K)): {simulator.delta0*1e9:.2f} nm")
        print(f"  μ₀Ms: {simulator.mu0 * simulator.Ms:.2f} T")
        v_analytical = simulator.analytical_velocity()
        print(f"  Walker velocity: {v_analytical:.4e} m/s")

        print(f"\n{'Running Simulation':^70}")
        print("-" * 70)
        results = simulator.run()

        velocity = simulator.compute_velocity()
        avg_width = np.mean(results['width'][-len(results['width'])//2:])

        print(f"\n{'Simulation Results':^70}")
        print("=" * 70)
        print(f"  Method: {simulator.method}")
        print(f"  Steady-state velocity: {velocity:.4e} m/s")
        print(f"  Analytical velocity:   {v_analytical:.4e} m/s")
        print(f"  Relative error:        {abs(velocity - v_analytical)/v_analytical*100:.2f}%")
        print(f"\n  Average domain wall width: {avg_width*1e9:.2f} nm")
        print(f"  Analytical width (Δ₀):     {simulator.delta0*1e9:.2f} nm")
        print(f"\n  Initial position: {results['position'][0]*1e9:.2f} nm")
        print(f"  Final position:   {results['position'][-1]*1e9:.2f} nm")
        print(f"  Total displacement: {(results['position'][-1]-results['position'][0])*1e9:.2f} nm")

        max_norm_error = np.max(np.abs(results['norm_max'] - 1.0))
        print(f"\n  Max | |m| - 1 |: {max_norm_error:.2e}")
        print(f"  Norm conservation: {'✓ Excellent' if max_norm_error < 1e-10 else '⚠ Good' if max_norm_error < 1e-6 else '✗ Poor'}")

        print(f"\n{'Generating Visualizations':^70}")
        print("-" * 70)
        simulator.plot_results(results)
        simulator.animate_domain_wall(results)

    print(f"\n{'Done!':^70}")
    print("=" * 70)


if __name__ == "__main__":
    main()
