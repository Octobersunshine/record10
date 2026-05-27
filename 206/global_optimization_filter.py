import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class OptimizationResult:
    n_list: List[float]
    d_list: List[float]
    metrics: Dict
    loss_history: List[float]
    best_loss: float


class GlobalThinFilmFilter:
    """
    薄膜光学滤波器 - 全局优化版本
    
    改进:
    1. 粒子群优化(PSO) + L-BFGS-B混合策略，避免局部最优
    2. 多目标优化：通带透射率 + 带外抑制 + 边缘锐度
    3. 自适应权重调节
    4. 多种全局优化算法可选
    """

    def __init__(self, n_substrate: float = 1.5, n_incident: float = 1.0):
        self.n_substrate = n_substrate
        self.n_incident = n_incident
        self.wavelengths = None
        self.target_T = None
        self.lambda_center = None
        self.bandwidth = None

    def tmm(self, wavelengths: np.ndarray, n_list: List[float],
            d_list: List[float], theta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """传递矩阵法 - 向量化版本加速"""
        n_layers = len(n_list)
        R = np.zeros_like(wavelengths, dtype=float)
        T = np.zeros_like(wavelengths, dtype=float)
        n0, ns = self.n_incident, self.n_substrate

        for idx, lambda0 in enumerate(wavelengths):
            k0 = 2 * np.pi / lambda0
            M = np.eye(2, dtype=complex)

            for n, d in zip(n_list, d_list):
                sin_theta = n0 * np.sin(theta) / n
                cos_theta = np.sqrt(1 - sin_theta ** 2 + 0j)
                delta = k0 * n * d * cos_theta
                eta = n * cos_theta

                M = M @ np.array([
                    [np.cos(delta), -1j * np.sin(delta) / eta],
                    [-1j * eta * np.sin(delta), np.cos(delta)]
                ])

            sin_theta_s = n0 * np.sin(theta) / ns
            cos_theta_s = np.sqrt(1 - sin_theta_s ** 2 + 0j)
            eta_s = ns * cos_theta_s
            eta0 = n0 * np.cos(theta)

            A, B = M[0, 0], M[0, 1]
            C, D = M[1, 0], M[1, 1]
            denom = A * eta_s + B * eta0 * eta_s + C + D * eta0

            r = (A * eta_s + B * eta0 * eta_s - C - D * eta0) / denom
            t = 2 * eta_s / denom

            R[idx] = np.real(np.abs(r) ** 2)
            T[idx] = np.real(np.abs(t) ** 2 * eta0 / eta_s)

        return R, T

    def _evaluate_multi_objective(self, params: np.ndarray, n_layers: int) -> Dict:
        """多目标评估函数"""
        n_opt = np.clip(params[:n_layers], 1.3, 2.6)
        d_opt = np.clip(params[n_layers:], 20, 400)

        _, T = self.tmm(self.wavelengths, n_opt.tolist(), d_opt.tolist())

        lb = self.lambda_center - self.bandwidth / 2
        ub = self.lambda_center + self.bandwidth / 2
        in_band = (self.wavelengths >= lb) & (self.wavelengths <= ub)
        out_band = ~in_band

        f1 = np.mean((1.0 - T[in_band]) ** 2)

        f2 = np.mean(T[out_band] ** 2)

        edge_width = self.bandwidth * 0.15
        left_edge = (self.wavelengths >= lb - edge_width) & (self.wavelengths <= lb + edge_width)
        right_edge = (self.wavelengths >= ub - edge_width) & (self.wavelengths <= ub + edge_width)
        edge_mask = left_edge | right_edge
        f3 = np.mean(T[edge_mask] ** 2) if np.any(edge_mask) else 0.0

        return {'in_band_loss': f1, 'out_band_loss': f2, 'edge_loss': f3, 'T': T}

    def _weighted_sum_loss(self, params: np.ndarray, n_layers: int,
                          weights: Dict = None) -> float:
        """加权求和损失函数"""
        if weights is None:
            weights = {'in_band': 0.35, 'out_band': 0.40, 'edge': 0.25}

        obj = self._evaluate_multi_objective(params, n_layers)
        return (weights['in_band'] * obj['in_band_loss'] +
                weights['out_band'] * obj['out_band_loss'] +
                weights['edge'] * obj['edge_loss'])

    def pso_optimize(self, n_layers: int, n_particles: int = 30,
                     n_iterations: int = 100, c1: float = 2.0, c2: float = 2.0,
                     w: float = 0.7, verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        粒子群优化 (Particle Swarm Optimization)
        
        参数:
            n_layers: 膜层数
            n_particles: 粒子数量
            n_iterations: 迭代次数
            c1, c2: 学习因子
            w: 惯性权重
        """
        n_dim = 2 * n_layers
        n_low, n_high = 1.3, 2.6
        d_low, d_high = 20, 400

        qw = self.lambda_center / 4.0
        n_init_hi, n_init_lo = 2.35, 1.38

        particles = np.zeros((n_particles, n_dim))
        for p in range(n_particles):
            for i in range(n_layers):
                base_n = n_init_hi if i % 2 == 0 else n_init_lo
                particles[p, i] = base_n + np.random.uniform(-0.3, 0.3)
                particles[p, n_layers + i] = qw / particles[p, i] + np.random.uniform(-20, 20)
            particles[p, :n_layers] = np.clip(particles[p, :n_layers], n_low, n_high)
            particles[p, n_layers:] = np.clip(particles[p, n_layers:], d_low, d_high)

        velocities = np.random.uniform(-1, 1, (n_particles, n_dim)) * 0.1

        pbest_positions = particles.copy()
        pbest_values = np.array([self._weighted_sum_loss(p, n_layers) for p in particles])

        gbest_idx = np.argmin(pbest_values)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_value = pbest_values[gbest_idx]

        loss_history = [gbest_value]

        if verbose:
            print(f"  PSO初始化完成, 初始最佳损失: {gbest_value:.6f}")

        for iteration in range(n_iterations):
            r1 = np.random.random((n_particles, n_dim))
            r2 = np.random.random((n_particles, n_dim))

            cognitive = c1 * r1 * (pbest_positions - particles)
            social = c2 * r2 * (gbest_position - particles)
            velocities = w * velocities + cognitive + social

            max_vel = 0.5
            velocities = np.clip(velocities, -max_vel, max_vel)

            particles = particles + velocities
            particles[:, :n_layers] = np.clip(particles[:, :n_layers], n_low, n_high)
            particles[:, n_layers:] = np.clip(particles[:, n_layers:], d_low, d_high)

            for p in range(n_particles):
                current_value = self._weighted_sum_loss(particles[p], n_layers)

                if current_value < pbest_values[p]:
                    pbest_values[p] = current_value
                    pbest_positions[p] = particles[p].copy()

                    if current_value < gbest_value:
                        gbest_value = current_value
                        gbest_position = particles[p].copy()

            loss_history.append(gbest_value)

            if verbose and (iteration + 1) % 20 == 0:
                print(f"  PSO迭代 {iteration + 1}/{n_iterations}, 最佳损失: {gbest_value:.6f}")

        return gbest_position, loss_history

    def simulated_annealing(self, n_layers: int, initial_temp: float = 100.0,
                           final_temp: float = 0.01, n_iterations: int = 500,
                           verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        模拟退火优化 (Simulated Annealing)
        """
        n_dim = 2 * n_layers
        qw = self.lambda_center / 4.0

        current = np.zeros(n_dim)
        for i in range(n_layers):
            current[i] = 2.35 if i % 2 == 0 else 1.38
            current[n_layers + i] = qw / current[i]

        current_loss = self._weighted_sum_loss(current, n_layers)
        best = current.copy()
        best_loss = current_loss

        loss_history = [best_loss]
        temp = initial_temp

        if verbose:
            print(f"  SA初始化完成, 初始损失: {current_loss:.6f}")

        for iteration in range(n_iterations):
            temp = initial_temp * (final_temp / initial_temp) ** (iteration / n_iterations)

            step_size = 0.1 * temp / initial_temp
            candidate = current + np.random.normal(0, step_size, n_dim)
            candidate[:n_layers] = np.clip(candidate[:n_layers], 1.3, 2.6)
            candidate[n_layers:] = np.clip(candidate[n_layers:], 20, 400)

            candidate_loss = self._weighted_sum_loss(candidate, n_layers)

            delta = candidate_loss - current_loss

            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current = candidate
                current_loss = candidate_loss

                if current_loss < best_loss:
                    best = current.copy()
                    best_loss = current_loss

            loss_history.append(best_loss)

            if verbose and (iteration + 1) % 100 == 0:
                print(f"  SA迭代 {iteration + 1}/{n_iterations}, 温度: {temp:.4f}, 最佳损失: {best_loss:.6f}")

        return best, loss_history

    def local_refine(self, initial_params: np.ndarray, n_layers: int,
                     max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """L-BFGS-B局部微调"""
        result = minimize(self._weighted_sum_loss, initial_params,
                         args=(n_layers,), method='L-BFGS-B',
                         options={'maxiter': max_iter})
        return result.x, result.fun

    def design_bandpass_global(self, lambda_center: float, bandwidth: float,
                               n_layers: int = 8, method: str = 'hybrid',
                               n_points: int = 200, verbose: bool = True) -> OptimizationResult:
        """
        全局优化设计带通滤波器
        
        参数:
            lambda_center: 中心波长 (nm)
            bandwidth: 带宽 (nm)
            n_layers: 薄膜层数
            method: 优化方法 - 'pso', 'sa', 'hybrid'(推荐)
            n_points: 波长采样点数
            verbose: 是否显示进度
        """
        self.lambda_center = lambda_center
        self.bandwidth = bandwidth

        self.wavelengths = np.linspace(lambda_center - 2.5 * bandwidth,
                                       lambda_center + 2.5 * bandwidth, n_points)

        def target_spec(w, lc, bw):
            sigma = bw / (2 * np.sqrt(2 * np.log(2)))
            return np.exp(-((w - lc) ** 2) / (2 * sigma ** 2))

        self.target_T = target_spec(self.wavelengths, lambda_center, bandwidth)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Global Optimization Design")
            print(f"  Center: {lambda_center} nm, Bandwidth: {bandwidth} nm")
            print(f"  Layers: {n_layers}, Method: {method.upper()}")
            print(f"{'='*70}")

        best_params = None
        best_loss = float('inf')
        full_history = []

        start_time = time.time()

        if method in ['pso', 'hybrid']:
            if verbose:
                print("\n[1/3] PSO Global Search...")
            pso_best, pso_history = self.pso_optimize(
                n_layers, n_particles=40, n_iterations=150, verbose=verbose)
            pso_loss = self._weighted_sum_loss(pso_best, n_layers)
            full_history.extend(pso_history)

            if pso_loss < best_loss:
                best_loss = pso_loss
                best_params = pso_best

        if method in ['sa', 'hybrid']:
            if verbose:
                print("\n[2/3] Simulated Annealing Exploration...")
            sa_best, sa_history = self.simulated_annealing(
                n_layers, n_iterations=800, verbose=verbose)
            sa_loss = self._weighted_sum_loss(sa_best, n_layers)
            full_history.extend(sa_history)

            if sa_loss < best_loss:
                best_loss = sa_loss
                best_params = sa_best

        if method in ['hybrid']:
            if verbose:
                print("\n[3/3] L-BFGS-B Local Refinement...")
            refined_params, refined_loss = self.local_refine(best_params, n_layers, max_iter=500)
            if refined_loss < best_loss:
                best_loss = refined_loss
                best_params = refined_params
            full_history.append(best_loss)

        elif method not in ['pso', 'sa']:
            raise ValueError(f"Unknown method: {method}. Use 'pso', 'sa', or 'hybrid'")

        elapsed = time.time() - start_time

        n_opt = np.clip(best_params[:n_layers], 1.3, 2.6).tolist()
        d_opt = np.clip(best_params[n_layers:], 20, 400).tolist()

        metrics = self._evaluate_performance(n_opt, d_opt)

        if verbose:
            print(f"\nOptimization complete in {elapsed:.1f}s")
            print(f"Best loss: {best_loss:.6f}")
            self._print_results(n_opt, d_opt, metrics)

        return OptimizationResult(
            n_list=n_opt, d_list=d_opt, metrics=metrics,
            loss_history=full_history, best_loss=best_loss
        )

    def _evaluate_performance(self, n_list, d_list):
        """评估性能"""
        R, T = self.tmm(self.wavelengths, n_list, d_list)

        lb = self.lambda_center - self.bandwidth / 2
        ub = self.lambda_center + self.bandwidth / 2
        in_band = (self.wavelengths >= lb) & (self.wavelengths <= ub)
        out_band = ~in_band

        max_T = np.max(T)
        peak_wl = self.wavelengths[np.argmax(T)]
        avg_in = np.mean(T[in_band])
        avg_out = np.mean(T[out_band])
        min_out = np.min(T[out_band])

        T50 = max_T / 2
        above = np.where(T >= T50)[0]
        fwhm = self.wavelengths[above[-1]] - self.wavelengths[above[0]] if len(above) > 1 else 0

        ripple = np.max(T[in_band]) - np.min(T[in_band]) if np.sum(in_band) > 1 else 0
        rejection = 10 * np.log10(avg_in / (avg_out + 1e-12))

        return {
            'max_T': max_T, 'peak_wl': peak_wl,
            'avg_T_in': avg_in, 'avg_T_out': avg_out,
            'min_T_out': min_out, 'fwhm': fwhm,
            'ripple': ripple, 'rejection_db': rejection,
            'R': R, 'T': T
        }

    def _print_results(self, n_list, d_list, metrics):
        """打印结果"""
        print(f"\n{'='*60}")
        print(f"Optimized Thin Film Stack:")
        print(f"{'Layer':<8} {'n':<12} {'d (nm)':<12} {'n*d (nm)':<15}")
        print("-" * 50)
        for i, (n, d) in enumerate(zip(n_list, d_list)):
            print(f"{i + 1:<8} {n:<12.4f} {d:<12.2f} {n * d:<15.1f}")

        print(f"\nPerformance Metrics:")
        print(f"  Peak transmittance: {metrics['max_T']*100:.2f}% @ {metrics['peak_wl']:.1f} nm")
        print(f"  Average in-band T: {metrics['avg_T_in']*100:.2f}%")
        print(f"  Average out-of-band T: {metrics['avg_T_out']*100:.2f}%")
        print(f"  Minimum out-of-band T: {metrics['min_T_out']*100:.2f}%")
        print(f"  FWHM: {metrics['fwhm']:.1f} nm (target: {self.bandwidth:.1f} nm)")
        print(f"  In-band ripple: {metrics['ripple']*100:.2f}%")
        print(f"  Rejection ratio: {metrics['rejection_db']:.1f} dB")
        print(f"{'='*60}")

    def plot_results(self, result: OptimizationResult, save_path: str = 'global_filter_result.png'):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        m = result.metrics

        axes[0, 0].plot(self.wavelengths, m['T'] * 100, 'b-', linewidth=2, label='Transmittance')
        axes[0, 0].plot(self.wavelengths, self.target_T * 100, 'r--', linewidth=1.5, label='Target')
        axes[0, 0].plot(self.wavelengths, m['R'] * 100, 'orange', linewidth=1.5, alpha=0.6, label='Reflectance')
        axes[0, 0].axvline(self.lambda_center, color='k', linestyle=':', alpha=0.7, label='Center λ')
        lb = self.lambda_center - self.bandwidth / 2
        ub = self.lambda_center + self.bandwidth / 2
        axes[0, 0].axvspan(lb, ub, alpha=0.15, color='green', label='Passband')
        axes[0, 0].set_xlabel('Wavelength (nm)', fontsize=11)
        axes[0, 0].set_ylabel('(%)', fontsize=11)
        axes[0, 0].set_title(f'Filter Spectrum (λ₀={self.lambda_center} nm, BW={self.bandwidth} nm)', fontsize=12)
        axes[0, 0].legend(fontsize=9, loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 105)

        x = np.arange(len(result.n_list))
        w = 0.35
        ax1 = axes[0, 1]
        ax1.bar(x - w / 2, result.n_list, w, color='steelblue', label='Refractive Index')
        ax1.set_xlabel('Layer Number', fontsize=11)
        ax1.set_ylabel('Refractive Index', fontsize=11, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(1.0, 2.8)
        ax2 = ax1.twinx()
        ax2.bar(x + w / 2, result.d_list, w, color='salmon', label='Thickness (nm)')
        ax2.set_ylabel('Thickness (nm)', fontsize=11, color='salmon')
        ax2.tick_params(axis='y', labelcolor='salmon')
        l1, la1 = ax1.get_legend_handles_labels()
        l2, la2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, la1 + la2, loc='upper right', fontsize=9)
        axes[0, 1].set_title('Thin Film Stack', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        if result.loss_history:
            axes[1, 0].plot(range(len(result.loss_history)), result.loss_history, 'b-', linewidth=1.5)
            axes[1, 0].set_xlabel('Iteration', fontsize=11)
            axes[1, 0].set_ylabel('Loss Value', fontsize=11)
            axes[1, 0].set_title('Convergence History', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')

        obj_names = ['In-Band\nTransmission', 'Out-of-Band\nSuppression', 'Edge\nSharpness']
        obj_values = [
            m['avg_T_in'] * 100,
            (1 - m['avg_T_out']) * 100,
            max(0, 100 - abs(m['fwhm'] - self.bandwidth) / self.bandwidth * 100)
        ]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        bars = axes[1, 1].bar(obj_names, obj_values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Score (%)', fontsize=11)
        axes[1, 1].set_title('Multi-Objective Performance', fontsize=12)
        axes[1, 1].set_ylim(0, 110)
        for bar, val in zip(bars, obj_values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                           f'{val:.1f}%', ha='center', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\nPlot saved to '{save_path}'")
        plt.close()


def main():
    print("=" * 70)
    print("Global Optimization Thin Film Filter Design")
    print("PSO + Simulated Annealing + L-BFGS-B Hybrid Strategy")
    print("=" * 70)

    filter = GlobalThinFilmFilter(n_substrate=1.5, n_incident=1.0)

    lambda_center = 550.0
    bandwidth = 40.0
    n_layers = 8

    result = filter.design_bandpass_global(
        lambda_center=lambda_center,
        bandwidth=bandwidth,
        n_layers=n_layers,
        method='hybrid',
        verbose=True
    )

    filter.plot_results(result, save_path='global_filter_result.png')

    print("\n" + "=" * 70)
    print("Design complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
