import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time


def tmm_optimized(wavelengths, n_list, d_list, n0=1.0, ns=1.5):
    """优化的TMM实现"""
    R = np.zeros_like(wavelengths, dtype=float)
    T = np.zeros_like(wavelengths, dtype=float)

    for idx, lambda0 in enumerate(wavelengths):
        k0 = 2 * np.pi / lambda0
        M = np.eye(2, dtype=complex)

        for n, d in zip(n_list, d_list):
            delta = k0 * n * d
            eta = n
            M = M @ np.array([
                [np.cos(delta), -1j * np.sin(delta) / eta],
                [-1j * eta * np.sin(delta), np.cos(delta)]
            ])

        eta_s = ns
        eta0 = n0
        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]
        denom = A * eta_s + B * eta0 * eta_s + C + D * eta0

        r = (A * eta_s + B * eta0 * eta_s - C - D * eta0) / denom
        t = 2 * eta_s / denom

        R[idx] = np.real(np.abs(r) ** 2)
        T[idx] = np.real(np.abs(t) ** 2 * eta0 / eta_s)

    return R, T


class FastGlobalFilter:
    """快速全局优化滤波器 - 精简版"""

    def __init__(self):
        self.wavelengths = None
        self.target_T = None
        self.lambda_center = None
        self.bandwidth = None

    def compute_loss(self, params, n_layers):
        """多目标损失函数"""
        n_opt = np.clip(params[:n_layers], 1.3, 2.6)
        d_opt = np.clip(params[n_layers:], 20, 400)

        _, T = tmm_optimized(self.wavelengths, n_opt.tolist(), d_opt.tolist())

        lb = self.lambda_center - self.bandwidth / 2
        ub = self.lambda_center + self.bandwidth / 2
        in_band = (self.wavelengths >= lb) & (self.wavelengths <= ub)
        out_band = ~in_band

        in_loss = np.mean((1.0 - T[in_band]) ** 2)
        out_loss = np.mean(T[out_band] ** 2)

        ew = self.bandwidth * 0.1
        le = (self.wavelengths >= lb - ew) & (self.wavelengths <= lb + ew)
        re = (self.wavelengths >= ub - ew) & (self.wavelengths <= ub + ew)
        edge_loss = np.mean(T[le | re] ** 2) if np.any(le | re) else 0.0

        return 0.35 * in_loss + 0.40 * out_loss + 0.25 * edge_loss

    def pso_fast(self, n_layers, n_particles=20, n_iter=80):
        """快速PSO优化"""
        n_dim = 2 * n_layers
        qw = self.lambda_center / 4.0

        particles = np.zeros((n_particles, n_dim))
        for p in range(n_particles):
            for i in range(n_layers):
                base = 2.35 if i % 2 == 0 else 1.38
                particles[p, i] = base + np.random.uniform(-0.2, 0.2)
                particles[p, n_layers + i] = qw / particles[p, i] + np.random.uniform(-15, 15)

        particles[:, :n_layers] = np.clip(particles[:, :n_layers], 1.3, 2.6)
        particles[:, n_layers:] = np.clip(particles[:, n_layers:], 20, 400)

        velocities = np.random.uniform(-0.5, 0.5, (n_particles, n_dim))
        pbest_pos = particles.copy()
        pbest_val = np.array([self.compute_loss(p, n_layers) for p in particles])

        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]

        history = [gbest_val]
        w, c1, c2 = 0.6, 1.8, 1.8

        for it in range(n_iter):
            r1, r2 = np.random.random((n_particles, n_dim)), np.random.random((n_particles, n_dim))
            velocities = w * velocities + c1 * r1 * (pbest_pos - particles) + c2 * r2 * (gbest_pos - particles)
            velocities = np.clip(velocities, -0.8, 0.8)

            particles = particles + velocities
            particles[:, :n_layers] = np.clip(particles[:, :n_layers], 1.3, 2.6)
            particles[:, n_layers:] = np.clip(particles[:, n_layers:], 20, 400)

            for p in range(n_particles):
                val = self.compute_loss(particles[p], n_layers)
                if val < pbest_val[p]:
                    pbest_val[p], pbest_pos[p] = val, particles[p].copy()
                    if val < gbest_val:
                        gbest_val, gbest_pos = val, particles[p].copy()

            history.append(gbest_val)

        return gbest_pos, history

    def sa_fast(self, n_layers, n_iter=300):
        """快速模拟退火"""
        n_dim = 2 * n_layers
        qw = self.lambda_center / 4.0

        current = np.zeros(n_dim)
        for i in range(n_layers):
            current[i] = 2.35 if i % 2 == 0 else 1.38
            current[n_layers + i] = qw / current[i]

        cur_loss = self.compute_loss(current, n_layers)
        best, best_loss = current.copy(), cur_loss
        history = [best_loss]
        T0, Tf = 50.0, 0.01

        for it in range(n_iter):
            temp = T0 * (Tf / T0) ** (it / n_iter)
            step = 0.15 * temp / T0
            candidate = current + np.random.normal(0, step, n_dim)
            candidate[:n_layers] = np.clip(candidate[:n_layers], 1.3, 2.6)
            candidate[n_layers:] = np.clip(candidate[n_layers:], 20, 400)

            cand_loss = self.compute_loss(candidate, n_layers)
            delta = cand_loss - cur_loss

            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                current, cur_loss = candidate, cand_loss
                if cur_loss < best_loss:
                    best, best_loss = current.copy(), cur_loss

            history.append(best_loss)

        return best, history

    def design(self, lambda_center, bandwidth, n_layers=6, verbose=True):
        """设计滤波器"""
        self.lambda_center = lambda_center
        self.bandwidth = bandwidth
        self.wavelengths = np.linspace(lambda_center - 2.5 * bandwidth,
                                       lambda_center + 2.5 * bandwidth, 120)

        def target_spec(w, lc, bw):
            sigma = bw / (2 * np.sqrt(2 * np.log(2)))
            return np.exp(-((w - lc) ** 2) / (2 * sigma ** 2))

        self.target_T = target_spec(self.wavelengths, lambda_center, bandwidth)

        if verbose:
            print(f"Design: λ₀={lambda_center}nm, BW={bandwidth}nm, Layers={n_layers}")
            print("-" * 50)

        start = time.time()

        if verbose:
            print("[1/3] PSO global search...", end=" ", flush=True)
        pso_best, pso_hist = self.pso_fast(n_layers, n_particles=15, n_iter=60)
        pso_loss = self.compute_loss(pso_best, n_layers)
        if verbose:
            print(f"done (loss={pso_loss:.4f})")

        if verbose:
            print("[2/3] SA exploration...", end=" ", flush=True)
        sa_best, sa_hist = self.sa_fast(n_layers, n_iter=200)
        sa_loss = self.compute_loss(sa_best, n_layers)
        if verbose:
            print(f"done (loss={sa_loss:.4f})")

        best = pso_best if pso_loss < sa_loss else sa_best
        best_loss = min(pso_loss, sa_loss)

        if verbose:
            print("[3/3] L-BFGS-B refinement...", end=" ", flush=True)
        result = minimize(self.compute_loss, best, args=(n_layers,),
                         method='L-BFGS-B', options={'maxiter': 300})
        if result.fun < best_loss:
            best, best_loss = result.x, result.fun
        if verbose:
            print(f"done (final loss={best_loss:.4f})")

        elapsed = time.time() - start

        n_opt = np.clip(best[:n_layers], 1.3, 2.6).tolist()
        d_opt = np.clip(best[n_layers:], 20, 400).tolist()

        R, T = tmm_optimized(self.wavelengths, n_opt, d_opt)

        lb = lambda_center - bandwidth / 2
        ub = lambda_center + bandwidth / 2
        in_band = (self.wavelengths >= lb) & (self.wavelengths <= ub)
        out_band = ~in_band

        max_T = np.max(T)
        peak_wl = self.wavelengths[np.argmax(T)]
        avg_in = np.mean(T[in_band])
        avg_out = np.mean(T[out_band])

        T50 = max_T / 2
        above = np.where(T >= T50)[0]
        fwhm = self.wavelengths[above[-1]] - self.wavelengths[above[0]] if len(above) > 1 else 0

        rejection = 10 * np.log10(avg_in / (avg_out + 1e-12))
        history = pso_hist + sa_hist + [best_loss]

        if verbose:
            print(f"\nTime: {elapsed:.1f}s")
            print(f"\n{'='*60}")
            print(f"Optimized Stack:")
            print(f"{'Layer':<8} {'n':<12} {'d(nm)':<12} {'n*d(nm)':<15}")
            print("-" * 50)
            for i, (n, d) in enumerate(zip(n_opt, d_opt)):
                print(f"{i+1:<8} {n:<12.4f} {d:<12.2f} {n*d:<15.1f}")

            print(f"\nPerformance:")
            print(f"  Peak T: {max_T*100:.2f}% @ {peak_wl:.1f}nm")
            print(f"  Avg in-band T: {avg_in*100:.2f}%")
            print(f"  Avg out-of-band T: {avg_out*100:.2f}%")
            print(f"  FWHM: {fwhm:.1f}nm (target: {bandwidth}nm)")
            print(f"  Rejection: {rejection:.1f}dB")
            print(f"{'='*60}")

        return {
            'n_list': n_opt, 'd_list': d_opt,
            'max_T': max_T, 'peak_wl': peak_wl,
            'avg_in': avg_in, 'avg_out': avg_out,
            'fwhm': fwhm, 'rejection_db': rejection,
            'T': T, 'R': R, 'history': history
        }


def plot_comparison(result, lambda_center, bandwidth, save_path='fast_global_result.png'):
    """绘制对比图"""
    wavelengths = result.get('wavelengths', None)
    if wavelengths is None:
        wavelengths = np.linspace(lambda_center - 2.5 * bandwidth,
                                  lambda_center + 2.5 * bandwidth, 120)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    T, R = result['T'], result['R']
    target_T = np.exp(-((wavelengths - lambda_center) ** 2) /
                      (2 * (bandwidth / (2 * np.sqrt(2 * np.log(2)))) ** 2))

    axes[0].plot(wavelengths, T * 100, 'b-', linewidth=2, label='Transmittance')
    axes[0].plot(wavelengths, target_T * 100, 'r--', linewidth=1.5, label='Target')
    axes[0].plot(wavelengths, R * 100, 'orange', linewidth=1.5, alpha=0.6, label='Reflectance')
    axes[0].axvline(lambda_center, color='k', linestyle=':', alpha=0.7)
    lb, ub = lambda_center - bandwidth / 2, lambda_center + bandwidth / 2
    axes[0].axvspan(lb, ub, alpha=0.15, color='green', label='Passband')
    axes[0].set_xlabel('Wavelength (nm)', fontsize=12)
    axes[0].set_ylabel('(%)', fontsize=12)
    axes[0].set_title(f'Bandpass Filter (λ₀={lambda_center}nm, BW={bandwidth}nm)', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)

    x = np.arange(len(result['n_list']))
    w = 0.35
    ax1 = axes[1]
    ax1.bar(x - w / 2, result['n_list'], w, color='steelblue', label='Refractive Index')
    ax1.set_xlabel('Layer Number', fontsize=12)
    ax1.set_ylabel('Refractive Index', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(1.0, 2.8)
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, result['d_list'], w, color='salmon', label='Thickness (nm)')
    ax2.set_ylabel('Thickness (nm)', fontsize=12, color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc='upper right', fontsize=10)
    axes[1].set_title('Optimized Thin Film Stack', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"\nPlot saved to '{save_path}'")
    plt.close()


def main():
    print("=" * 60)
    print("Global Optimization Filter - Fast Demo")
    print("PSO + SA + L-BFGS-B Hybrid Strategy")
    print("=" * 60)

    filter = FastGlobalFilter()

    lambda_center = 550.0
    bandwidth = 40.0

    result = filter.design(
        lambda_center=lambda_center,
        bandwidth=bandwidth,
        n_layers=8,
        verbose=True
    )

    result['wavelengths'] = filter.wavelengths

    plot_comparison(result, lambda_center, bandwidth)

    print("\nComplete!")


if __name__ == "__main__":
    main()
