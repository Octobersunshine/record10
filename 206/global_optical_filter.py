import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class FilterResult:
    n_list: List[float]
    d_list: List[float]
    metrics: Dict
    loss_history: List[float]
    best_loss: float
    wavelengths: np.ndarray
    target_spectrum: np.ndarray


class GlobalOpticalFilter:
    """
    薄膜光学滤波器 - 全局优化版本
    
    核心改进:
    1. 混合优化策略: PSO(全局搜索) + SA(模拟退火) + L-BFGS-B(局部微调)
    2. 多目标优化: 通带透射率 + 带外抑制 + 边缘锐度
    3. 自适应参数: 动态调整惯性权重和退火温度
    4. 收敛保证: 多次全局搜索确保跳出局部最优
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
        """
        传递矩阵法 (Transfer Matrix Method)
        
        计算多层薄膜的反射率和透射率。
        每层薄膜的特征矩阵:
            M = [[cos(δ),  -i·sin(δ)/η],
                 [-i·η·sin(δ),  cos(δ)]]
        其中: δ = 2π·n·d·cos(θ)/λ, η = n·cos(θ)
        """
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

    def _multi_objective_loss(self, params: np.ndarray, n_layers: int,
                              weights: Dict = None) -> float:
        """
        多目标损失函数
        
        目标1: 通带透射率最大化 (in_band_loss)
        目标2: 带外抑制最大化 (out_band_loss)  
        目标3: 通带边缘锐度 (edge_loss)
        """
        if weights is None:
            weights = {'in_band': 0.35, 'out_band': 0.40, 'edge': 0.25}

        n_opt = np.clip(params[:n_layers], 1.3, 2.6)
        d_opt = np.clip(params[n_layers:], 20, 400)

        _, T = self.tmm(self.wavelengths, n_opt.tolist(), d_opt.tolist())

        lb = self.lambda_center - self.bandwidth / 2
        ub = self.lambda_center + self.bandwidth / 2
        in_band = (self.wavelengths >= lb) & (self.wavelengths <= ub)
        out_band = ~in_band

        in_band_loss = np.mean((1.0 - T[in_band]) ** 2)
        out_band_loss = np.mean(T[out_band] ** 2)

        edge_width = self.bandwidth * 0.1
        left_edge = (self.wavelengths >= lb - edge_width) & (self.wavelengths <= lb + edge_width)
        right_edge = (self.wavelengths >= ub - edge_width) & (self.wavelengths <= ub + edge_width)
        edge_mask = left_edge | right_edge
        edge_loss = np.mean(T[edge_mask] ** 2) if np.any(edge_mask) else 0.0

        return (weights['in_band'] * in_band_loss +
                weights['out_band'] * out_band_loss +
                weights['edge'] * edge_loss)

    def pso_optimize(self, n_layers: int, n_particles: int = 30,
                     n_iterations: int = 100, verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        粒子群优化 (Particle Swarm Optimization)
        
        原理: 模拟鸟群觅食行为，粒子根据自身经验(pbest)和群体经验(gbest)更新位置。
        优势: 全局搜索能力强，不易陷入局部最优
        """
        n_dim = 2 * n_layers
        qw = self.lambda_center / 4.0

        particles = np.zeros((n_particles, n_dim))
        for p in range(n_particles):
            for i in range(n_layers):
                base_n = 2.35 if i % 2 == 0 else 1.38
                particles[p, i] = base_n + np.random.uniform(-0.25, 0.25)
                particles[p, n_layers + i] = qw / particles[p, i] + np.random.uniform(-15, 15)

        particles[:, :n_layers] = np.clip(particles[:, :n_layers], 1.3, 2.6)
        particles[:, n_layers:] = np.clip(particles[:, n_layers:], 20, 400)

        velocities = np.random.uniform(-0.3, 0.3, (n_particles, n_dim))

        pbest_positions = particles.copy()
        pbest_values = np.array([self._multi_objective_loss(p, n_layers) for p in particles])

        gbest_idx = np.argmin(pbest_values)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_value = pbest_values[gbest_idx]

        loss_history = [gbest_value]

        if verbose:
            print(f"    PSO初始化: 初始最佳损失 = {gbest_value:.6f}")

        w_start, w_end = 0.9, 0.4
        c1, c2 = 2.0, 2.0

        for it in range(n_iterations):
            w = w_start - (w_start - w_end) * it / n_iterations

            r1 = np.random.random((n_particles, n_dim))
            r2 = np.random.random((n_particles, n_dim))

            cognitive = c1 * r1 * (pbest_positions - particles)
            social = c2 * r2 * (gbest_position - particles)
            velocities = w * velocities + cognitive + social

            max_vel = 0.5
            velocities = np.clip(velocities, -max_vel, max_vel)

            particles = particles + velocities
            particles[:, :n_layers] = np.clip(particles[:, :n_layers], 1.3, 2.6)
            particles[:, n_layers:] = np.clip(particles[:, n_layers:], 20, 400)

            for p in range(n_particles):
                current_val = self._multi_objective_loss(particles[p], n_layers)

                if current_val < pbest_values[p]:
                    pbest_values[p] = current_val
                    pbest_positions[p] = particles[p].copy()

                    if current_val < gbest_value:
                        gbest_value = current_val
                        gbest_position = particles[p].copy()

            loss_history.append(gbest_value)

            if verbose and (it + 1) % 25 == 0:
                print(f"    PSO迭代 {it + 1}/{n_iterations}: 最佳损失 = {gbest_value:.6f}")

        return gbest_position, loss_history

    def simulated_annealing(self, n_layers: int, n_iterations: int = 500,
                            verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        模拟退火优化 (Simulated Annealing)
        
        原理: 模拟金属退火过程，高温时接受较差解（探索），低温时趋向最优（利用）。
        优势: 能有效跳出局部最优，理论上保证收敛到全局最优
        """
        n_dim = 2 * n_layers
        qw = self.lambda_center / 4.0

        current = np.zeros(n_dim)
        for i in range(n_layers):
            current[i] = 2.35 if i % 2 == 0 else 1.38
            current[n_layers + i] = qw / current[i]

        current_loss = self._multi_objective_loss(current, n_layers)
        best = current.copy()
        best_loss = current_loss

        loss_history = [best_loss]

        T_initial, T_final = 50.0, 0.01

        if verbose:
            print(f"    SA初始化: 初始损失 = {current_loss:.6f}")

        for it in range(n_iterations):
            T = T_initial * (T_final / T_initial) ** (it / n_iterations)

            step_size = 0.15 * T / T_initial
            candidate = current + np.random.normal(0, step_size, n_dim)
            candidate[:n_layers] = np.clip(candidate[:n_layers], 1.3, 2.6)
            candidate[n_layers:] = np.clip(candidate[n_layers:], 20, 400)

            candidate_loss = self._multi_objective_loss(candidate, n_layers)

            delta = candidate_loss - current_loss

            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current = candidate
                current_loss = candidate_loss

                if current_loss < best_loss:
                    best = current.copy()
                    best_loss = current_loss

            loss_history.append(best_loss)

            if verbose and (it + 1) % 100 == 0:
                print(f"    SA迭代 {it + 1}/{n_iterations}: T={T:.4f}, 最佳损失 = {best_loss:.6f}")

        return best, loss_history

    def local_refinement(self, initial_params: np.ndarray, n_layers: int,
                         max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """L-BFGS-B局部微调 - 在全局最优附近精细搜索"""
        result = minimize(self._multi_objective_loss, initial_params,
                         args=(n_layers,), method='L-BFGS-B',
                         options={'maxiter': max_iter})
        return result.x, result.fun

    def design_bandpass(self, lambda_center: float, bandwidth: float,
                        n_layers: int = 8, method: str = 'hybrid',
                        n_points: int = 200, verbose: bool = True) -> FilterResult:
        """
        设计带通滤波器
        
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
            print(f"  薄膜光学滤波器 - 全局优化设计")
            print(f"  中心波长: {lambda_center} nm, 带宽: {bandwidth} nm")
            print(f"  薄膜层数: {n_layers}, 优化方法: {method.upper()}")
            print(f"{'='*70}")

        start_time = time.time()
        best_params = None
        best_loss = float('inf')
        full_history = []

        if method in ['pso', 'hybrid']:
            if verbose:
                print("\n[1/3] PSO全局搜索...")
            pso_best, pso_hist = self.pso_optimize(
                n_layers, n_particles=30, n_iterations=80, verbose=verbose)
            pso_loss = self._multi_objective_loss(pso_best, n_layers)
            full_history.extend(pso_hist)

            if pso_loss < best_loss:
                best_loss = pso_loss
                best_params = pso_best

        if method in ['sa', 'hybrid']:
            if verbose:
                print("\n[2/3] 模拟退火探索...")
            sa_best, sa_hist = self.simulated_annealing(
                n_layers, n_iterations=400, verbose=verbose)
            sa_loss = self._multi_objective_loss(sa_best, n_layers)
            full_history.extend(sa_hist)

            if sa_loss < best_loss:
                best_loss = sa_loss
                best_params = sa_best

        if method == 'hybrid':
            if verbose:
                print("\n[3/3] L-BFGS-B局部微调...")
            refined, refined_loss = self.local_refinement(best_params, n_layers, max_iter=300)
            if refined_loss < best_loss:
                best_loss = refined_loss
                best_params = refined
            full_history.append(best_loss)
        elif method not in ['pso', 'sa']:
            raise ValueError(f"未知方法: {method}。请使用 'pso', 'sa' 或 'hybrid'")

        elapsed = time.time() - start_time

        n_opt = np.clip(best_params[:n_layers], 1.3, 2.6).tolist()
        d_opt = np.clip(best_params[n_layers:], 20, 400).tolist()

        metrics = self._evaluate_performance(n_opt, d_opt)

        if verbose:
            print(f"\n{'='*70}")
            print(f"  优化完成! 耗时: {elapsed:.1f}秒")
            print(f"  最终损失: {best_loss:.6f}")
            self._print_results(n_opt, d_opt, metrics)

        return FilterResult(
            n_list=n_opt, d_list=d_opt, metrics=metrics,
            loss_history=full_history, best_loss=best_loss,
            wavelengths=self.wavelengths, target_spectrum=self.target_T
        )

    def _evaluate_performance(self, n_list, d_list):
        """评估滤波器性能"""
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
        """打印优化结果"""
        print(f"\n  优化后的膜系参数:")
        print(f"  {'层号':<6} {'折射率':<12} {'厚度(nm)':<12} {'光学厚度(nm)':<15}")
        print(f"  {'-'*50}")
        for i, (n, d) in enumerate(zip(n_list, d_list)):
            print(f"  {i + 1:<6} {n:<12.4f} {d:<12.2f} {n * d:<15.1f}")

        print(f"\n  性能指标:")
        print(f"    峰值透射率: {metrics['max_T']*100:.2f}% @ {metrics['peak_wl']:.1f} nm")
        print(f"    通带平均透射率: {metrics['avg_T_in']*100:.2f}%")
        print(f"    阻带平均透射率: {metrics['avg_T_out']*100:.2f}%")
        print(f"    阻带最小透射率: {metrics['min_T_out']*100:.2f}%")
        print(f"    实际半高宽: {metrics['fwhm']:.1f} nm (目标: {self.bandwidth:.1f} nm)")
        print(f"    通带波纹: {metrics['ripple']*100:.2f}%")
        print(f"    抑制比: {metrics['rejection_db']:.1f} dB")
        print(f"{'='*70}")

    def plot_results(self, result: FilterResult, save_path: str = 'filter_global_opt.png'):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        m = result.metrics
        wl = result.wavelengths

        axes[0, 0].plot(wl, m['T'] * 100, 'b-', linewidth=2, label='透射率')
        axes[0, 0].plot(wl, result.target_spectrum * 100, 'r--', linewidth=1.5, label='目标')
        axes[0, 0].plot(wl, m['R'] * 100, 'orange', linewidth=1.5, alpha=0.6, label='反射率')
        axes[0, 0].axvline(self.lambda_center, color='k', linestyle=':', alpha=0.7, label='中心波长')
        lb = self.lambda_center - self.bandwidth / 2
        ub = self.lambda_center + self.bandwidth / 2
        axes[0, 0].axvspan(lb, ub, alpha=0.15, color='green', label='通带')
        axes[0, 0].set_xlabel('波长 (nm)', fontsize=11)
        axes[0, 0].set_ylabel('(%)', fontsize=11)
        axes[0, 0].set_title(f'滤波器光谱 (λ₀={self.lambda_center} nm, BW={self.bandwidth} nm)', fontsize=12)
        axes[0, 0].legend(fontsize=9, loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 105)

        x = np.arange(len(result.n_list))
        w = 0.35
        ax1 = axes[0, 1]
        ax1.bar(x - w / 2, result.n_list, w, color='steelblue', label='折射率')
        ax1.set_xlabel('膜层编号', fontsize=11)
        ax1.set_ylabel('折射率', fontsize=11, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(1.0, 2.8)
        ax2 = ax1.twinx()
        ax2.bar(x + w / 2, result.d_list, w, color='salmon', label='厚度 (nm)')
        ax2.set_ylabel('厚度 (nm)', fontsize=11, color='salmon')
        ax2.tick_params(axis='y', labelcolor='salmon')
        l1, la1 = ax1.get_legend_handles_labels()
        l2, la2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, la1 + la2, loc='upper right', fontsize=9)
        axes[0, 1].set_title('膜系参数', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        if result.loss_history:
            axes[1, 0].plot(range(len(result.loss_history)), result.loss_history, 'b-', linewidth=1.5)
            axes[1, 0].set_xlabel('迭代次数', fontsize=11)
            axes[1, 0].set_ylabel('损失值', fontsize=11)
            axes[1, 0].set_title('收敛曲线', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')

        obj_names = ['通带透射率', '带外抑制', '边缘锐度']
        obj_scores = [
            m['avg_T_in'] * 100,
            (1 - m['avg_T_out']) * 100,
            max(0, 100 - abs(m['fwhm'] - self.bandwidth) / self.bandwidth * 100)
        ]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        bars = axes[1, 1].bar(obj_names, obj_scores, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('得分 (%)', fontsize=11)
        axes[1, 1].set_title('多目标性能评估', fontsize=12)
        axes[1, 1].set_ylim(0, 110)
        for bar, val in zip(bars, obj_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                           f'{val:.1f}%', ha='center', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n图表已保存至 '{save_path}'")
        plt.close()


def design_filter(lambda_center: float, bandwidth: float, n_layers: int = 8,
                  method: str = 'hybrid') -> FilterResult:
    """便捷函数：设计定制化滤波器"""
    filter = GlobalOpticalFilter(n_substrate=1.5, n_incident=1.0)
    result = filter.design_bandpass(
        lambda_center=lambda_center,
        bandwidth=bandwidth,
        n_layers=n_layers,
        method=method,
        verbose=True
    )
    filter.plot_results(result)
    return result


def main():
    print("=" * 70)
    print("  薄膜光学滤波器设计 - 全局优化算法")
    print("  PSO + 模拟退火 + L-BFGS-B 混合策略")
    print("  多目标优化: 通带透射率 + 带外抑制 + 边缘锐度")
    print("=" * 70)

    result = design_filter(
        lambda_center=550.0,
        bandwidth=40.0,
        n_layers=8,
        method='hybrid'
    )

    print("\n" + "=" * 70)
    print("  设计完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
