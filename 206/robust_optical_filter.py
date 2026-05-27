import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import time


@dataclass
class RobustnessResult:
    yield_rate: float
    yield_confidence_interval: Tuple[float, float]
    sensitivity_per_layer: np.ndarray
    performance_distribution: Dict
    tolerance_recommendation: Dict
    monte_carlo_samples: int
    error_level: float


@dataclass
class FilterResult:
    n_list: List[float]
    d_list: List[float]
    metrics: Dict
    loss_history: List[float]
    best_loss: float
    wavelengths: np.ndarray
    target_spectrum: np.ndarray
    robustness: Optional[RobustnessResult] = None


class RobustOpticalFilter:
    """
    薄膜光学滤波器 - 鲁棒性增强版本
    
    新增功能:
    1. 蒙特卡洛模拟分析膜厚误差影响
    2. 成品率预测
    3. 灵敏度分析
    4. 公差设计建议
    """

    def __init__(self, n_substrate: float = 1.5, n_incident: float = 1.0):
        self.n_substrate = n_substrate
        self.n_incident = n_incident
        self.wavelengths = None
        self.target_T = None
        self.lambda_center = None
        self.bandwidth = None
        self.specs = None

    def tmm(self, wavelengths: np.ndarray, n_list: List[float],
            d_list: List[float], theta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """传递矩阵法"""
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

    def set_specifications(self, min_transmittance: float = 0.8,
                           max_out_of_band: float = 0.1,
                           fwhm_tolerance: float = 0.15):
        """
        设置滤波器规格
        
        参数:
            min_transmittance: 通带最小透射率 (默认: 80%)
            max_out_of_band: 带外最大透射率 (默认: 10%)
            fwhm_tolerance: 半高宽相对误差容忍度 (默认: ±15%)
        """
        self.specs = {
            'min_T': min_transmittance,
            'max_out_T': max_out_of_band,
            'fwhm_tol': fwhm_tolerance
        }

    def check_specifications(self, metrics: Dict) -> bool:
        """检查滤波器是否满足规格"""
        if self.specs is None:
            self.set_specifications()

        lb = self.lambda_center - self.bandwidth / 2
        ub = self.lambda_center + self.bandwidth / 2

        fwhm_ok = (abs(metrics['fwhm'] - self.bandwidth) / self.bandwidth <= self.specs['fwhm_tol'])
        trans_ok = (metrics['avg_T_in'] >= self.specs['min_T'])
        reject_ok = (metrics['avg_T_out'] <= self.specs['max_out_T'])

        return fwhm_ok and trans_ok and reject_ok

    def monte_carlo_analysis(self, n_list: List[float], d_list: List[float],
                             thickness_error: float = 0.02,
                             n_samples: int = 2000,
                             verbose: bool = True) -> RobustnessResult:
        """
        蒙特卡洛模拟分析膜厚误差的鲁棒性
        
        参数:
            n_list: 折射率列表
            d_list: 标称厚度列表
            thickness_error: 膜厚相对误差 (默认: ±2%)
            n_samples: 蒙特卡洛样本数
            verbose: 是否显示进度
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"  鲁棒性分析 - 蒙特卡洛模拟")
            print(f"  膜厚误差: ±{thickness_error*100:.0f}%")
            print(f"  样本数: {n_samples}")
            print(f"{'='*70}")

        n_layers = len(d_list)
        d_nominal = np.array(d_list)

        all_metrics = []
        passes = 0
        all_d_variations = []

        if verbose:
            print("\n  运行蒙特卡洛模拟...")

        for i in range(n_samples):
            d_variation = d_nominal * np.random.normal(1.0, thickness_error / 3, n_layers)
            d_variation = np.clip(d_variation, d_nominal * (1 - 3 * thickness_error),
                                   d_nominal * (1 + 3 * thickness_error))
            all_d_variations.append(d_variation)

            metrics = self._evaluate_performance(n_list, d_variation.tolist())
            all_metrics.append(metrics)

            if self.check_specifications(metrics):
                passes += 1

            if verbose and (i + 1) % 500 == 0:
                print(f"    已完成 {i + 1}/{n_samples} 样本...")

        all_d_variations = np.array(all_d_variations)

        yield_rate = passes / n_samples
        se = np.sqrt(yield_rate * (1 - yield_rate) / n_samples)
        ci_low = max(0, yield_rate - 1.96 * se)
        ci_high = min(1, yield_rate + 1.96 * se)

        max_T_values = np.array([m['max_T'] for m in all_metrics])
        avg_T_in_values = np.array([m['avg_T_in'] for m in all_metrics])
        avg_T_out_values = np.array([m['avg_T_out'] for m in all_metrics])
        fwhm_values = np.array([m['fwhm'] for m in all_metrics])
        rejection_values = np.array([m['rejection_db'] for m in all_metrics])

        performance_dist = {
            'max_T': {'mean': np.mean(max_T_values), 'std': np.std(max_T_values),
                      'min': np.min(max_T_values), 'max': np.max(max_T_values),
                      'values': max_T_values},
            'avg_T_in': {'mean': np.mean(avg_T_in_values), 'std': np.std(avg_T_in_values),
                         'min': np.min(avg_T_in_values), 'max': np.max(avg_T_in_values),
                         'values': avg_T_in_values},
            'avg_T_out': {'mean': np.mean(avg_T_out_values), 'std': np.std(avg_T_out_values),
                          'min': np.min(avg_T_out_values), 'max': np.max(avg_T_out_values),
                          'values': avg_T_out_values},
            'fwhm': {'mean': np.mean(fwhm_values), 'std': np.std(fwhm_values),
                     'min': np.min(fwhm_values), 'max': np.max(fwhm_values),
                     'values': fwhm_values},
            'rejection': {'mean': np.mean(rejection_values), 'std': np.std(rejection_values),
                          'min': np.min(rejection_values), 'max': np.max(rejection_values),
                          'values': rejection_values},
        }

        sensitivity = self._calculate_sensitivity(n_list, d_list, thickness_error)

        tolerance_rec = self._recommend_tolerance(n_list, d_list, sensitivity,
                                                   target_yield=0.90)

        if verbose:
            self._print_robustness_results(yield_rate, (ci_low, ci_high),
                                          performance_dist, sensitivity, tolerance_rec)

        return RobustnessResult(
            yield_rate=yield_rate,
            yield_confidence_interval=(ci_low, ci_high),
            sensitivity_per_layer=sensitivity,
            performance_distribution=performance_dist,
            tolerance_recommendation=tolerance_rec,
            monte_carlo_samples=n_samples,
            error_level=thickness_error
        )

    def _calculate_sensitivity(self, n_list: List[float], d_list: List[float],
                               error_level: float) -> np.ndarray:
        """计算每层的灵敏度"""
        n_layers = len(d_list)
        d_nominal = np.array(d_list)
        sensitivity = np.zeros(n_layers)

        base_metrics = self._evaluate_performance(n_list, d_list)
        base_loss = (self.specs['min_T'] - base_metrics['avg_T_in'] if base_metrics['avg_T_in'] < self.specs['min_T'] else 0) + \
                    (base_metrics['avg_T_out'] - self.specs['max_out_T'] if base_metrics['avg_T_out'] > self.specs['max_out_T'] else 0)

        delta = error_level / 3
        for i in range(n_layers):
            d_plus = d_nominal.copy()
            d_plus[i] *= (1 + delta)
            metrics_plus = self._evaluate_performance(n_list, d_plus.tolist())

            d_minus = d_nominal.copy()
            d_minus[i] *= (1 - delta)
            metrics_minus = self._evaluate_performance(n_list, d_minus.tolist())

            loss_plus = (self.specs['min_T'] - metrics_plus['avg_T_in'] if metrics_plus['avg_T_in'] < self.specs['min_T'] else 0) + \
                        (metrics_plus['avg_T_out'] - self.specs['max_out_T'] if metrics_plus['avg_T_out'] > self.specs['max_out_T'] else 0)
            loss_minus = (self.specs['min_T'] - metrics_minus['avg_T_in'] if metrics_minus['avg_T_in'] < self.specs['min_T'] else 0) + \
                         (metrics_minus['avg_T_out'] - self.specs['max_out_T'] if metrics_minus['avg_T_out'] > self.specs['max_out_T'] else 0)

            sensitivity[i] = max(abs(loss_plus - base_loss), abs(loss_minus - base_loss)) / (delta * d_nominal[i])

        return sensitivity / np.max(sensitivity) if np.max(sensitivity) > 0 else sensitivity

    def _recommend_tolerance(self, n_list: List[float], d_list: List[float],
                             sensitivity: np.ndarray, target_yield: float = 0.90) -> Dict:
        """推荐公差设计"""
        n_layers = len(d_list)
        total_sensitivity = np.sum(sensitivity)

        if total_sensitivity == 0:
            base_tol = 0.03
            return {
                'per_layer': np.full(n_layers, base_tol),
                'tight_layers': [],
                'loose_layers': list(range(n_layers)),
                'expected_yield': target_yield
            }

        tight_factor = 0.7
        loose_factor = 1.5
        base_tol = 0.025

        tol_rec = np.zeros(n_layers)
        for i in range(n_layers):
            if sensitivity[i] > np.mean(sensitivity) + 0.3 * np.std(sensitivity):
                tol_rec[i] = base_tol * tight_factor
            elif sensitivity[i] < np.mean(sensitivity) - 0.3 * np.std(sensitivity):
                tol_rec[i] = base_tol * loose_factor
            else:
                tol_rec[i] = base_tol

        tight_layers = [i for i in range(n_layers) if tol_rec[i] < base_tol]
        loose_layers = [i for i in range(n_layers) if tol_rec[i] > base_tol]

        return {
            'per_layer': tol_rec,
            'tight_layers': tight_layers,
            'loose_layers': loose_layers,
            'target_yield': target_yield,
            'base_tolerance': base_tol
        }

    def _print_robustness_results(self, yield_rate, ci, perf_dist, sensitivity, tolerance):
        """打印鲁棒性分析结果"""
        print(f"\n  蒙特卡洛分析结果:")
        print(f"    成品率: {yield_rate*100:.1f}% (95% CI: {ci[0]*100:.1f}% - {ci[1]*100:.1f}%)")
        print(f"\n  性能分布:")
        print(f"    峰值透射率: {perf_dist['max_T']['mean']*100:.1f} ± {perf_dist['max_T']['std']*100:.1f}%")
        print(f"    通带平均透射率: {perf_dist['avg_T_in']['mean']*100:.1f} ± {perf_dist['avg_T_in']['std']*100:.1f}%")
        print(f"    带外平均透射率: {perf_dist['avg_T_out']['mean']*100:.1f} ± {perf_dist['avg_T_out']['std']*100:.1f}%")
        print(f"    半高宽: {perf_dist['fwhm']['mean']:.1f} ± {perf_dist['fwhm']['std']:.1f} nm")
        print(f"    抑制比: {perf_dist['rejection']['mean']:.1f} ± {perf_dist['rejection']['std']:.1f} dB")

        print(f"\n  层灵敏度分析 (归一化):")
        sorted_idx = np.argsort(sensitivity)[::-1]
        for idx in sorted_idx[:5]:
            print(f"    层 {idx+1}: {sensitivity[idx]:.3f}")

        print(f"\n  公差设计建议:")
        print(f"    基础公差: ±{tolerance['base_tolerance']*100:.0f}%")
        if tolerance['tight_layers']:
            tight_str = ', '.join([str(i+1) for i in tolerance['tight_layers']])
            print(f"    需严格控制的层: {tight_str} (建议: ±{tolerance['base_tolerance']*0.7*100:.0f}%)")
        if tolerance['loose_layers']:
            loose_str = ', '.join([str(i+1) for i in tolerance['loose_layers']])
            print(f"    可放宽控制的层: {loose_str} (建议: ±{tolerance['base_tolerance']*1.5*100:.0f}%)")
        print(f"{'='*70}")

    def _multi_objective_loss(self, params: np.ndarray, n_layers: int,
                              weights: Dict = None) -> float:
        """多目标损失函数"""
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

    def pso_optimize(self, n_layers: int, n_particles: int = 25,
                     n_iterations: int = 60, verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """粒子群优化"""
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
            velocities = np.clip(velocities, -0.5, 0.5)

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

            if verbose and (it + 1) % 20 == 0:
                print(f"    PSO迭代 {it + 1}/{n_iterations}: 最佳损失 = {gbest_value:.6f}")

        return gbest_position, loss_history

    def simulated_annealing(self, n_layers: int, n_iterations: int = 300,
                            verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """模拟退火优化"""
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

        T_initial, T_final = 40.0, 0.01

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
                         max_iter: int = 300) -> Tuple[np.ndarray, float]:
        """L-BFGS-B局部微调"""
        result = minimize(self._multi_objective_loss, initial_params,
                         args=(n_layers,), method='L-BFGS-B',
                         options={'maxiter': max_iter})
        return result.x, result.fun

    def design_bandpass(self, lambda_center: float, bandwidth: float,
                        n_layers: int = 8, method: str = 'hybrid',
                        n_points: int = 150, verbose: bool = True,
                        specs: Dict = None) -> FilterResult:
        """设计带通滤波器"""
        self.lambda_center = lambda_center
        self.bandwidth = bandwidth

        if specs:
            self.specs = specs
        else:
            self.set_specifications()

        self.wavelengths = np.linspace(lambda_center - 2.5 * bandwidth,
                                       lambda_center + 2.5 * bandwidth, n_points)

        def target_spec(w, lc, bw):
            sigma = bw / (2 * np.sqrt(2 * np.log(2)))
            return np.exp(-((w - lc) ** 2) / (2 * sigma ** 2))

        self.target_T = target_spec(self.wavelengths, lambda_center, bandwidth)

        if verbose:
            print(f"\n{'='*70}")
            print(f"  薄膜光学滤波器 - 全局优化 + 鲁棒性分析")
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
                n_layers, n_particles=25, n_iterations=60, verbose=verbose)
            pso_loss = self._multi_objective_loss(pso_best, n_layers)
            full_history.extend(pso_hist)
            if pso_loss < best_loss:
                best_loss = pso_loss
                best_params = pso_best

        if method in ['sa', 'hybrid']:
            if verbose:
                print("\n[2/3] 模拟退火探索...")
            sa_best, sa_hist = self.simulated_annealing(
                n_layers, n_iterations=300, verbose=verbose)
            sa_loss = self._multi_objective_loss(sa_best, n_layers)
            full_history.extend(sa_hist)
            if sa_loss < best_loss:
                best_loss = sa_loss
                best_params = sa_best

        if method == 'hybrid':
            if verbose:
                print("\n[3/3] L-BFGS-B局部微调...")
            refined, refined_loss = self.local_refinement(best_params, n_layers, max_iter=250)
            if refined_loss < best_loss:
                best_loss = refined_loss
                best_params = refined
            full_history.append(best_loss)
        elif method not in ['pso', 'sa']:
            raise ValueError(f"未知方法: {method}")

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

    def plot_comprehensive(self, result: FilterResult, save_path: str = 'robust_filter_result.png'):
        """绘制综合结果图表"""
        n_plots = 2
        if result.robustness is not None:
            n_plots = 4

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

        m = result.metrics
        wl = result.wavelengths

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(wl, m['T'] * 100, 'b-', linewidth=2, label='透射率')
        ax1.plot(wl, result.target_spectrum * 100, 'r--', linewidth=1.5, label='目标')
        ax1.plot(wl, m['R'] * 100, 'orange', linewidth=1.5, alpha=0.6, label='反射率')
        ax1.axvline(self.lambda_center, color='k', linestyle=':', alpha=0.7, label='中心波长')
        lb = self.lambda_center - self.bandwidth / 2
        ub = self.lambda_center + self.bandwidth / 2
        ax1.axvspan(lb, ub, alpha=0.15, color='green', label='通带')
        ax1.axhline(self.specs['min_T'] * 100, color='red', linestyle='--', alpha=0.5, label='规格下限')
        ax1.set_xlabel('波长 (nm)', fontsize=11)
        ax1.set_ylabel('(%)', fontsize=11)
        ax1.set_title(f'滤波器光谱 (λ₀={self.lambda_center} nm, BW={self.bandwidth} nm)', fontsize=12)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(result.n_list))
        w = 0.35
        ax2.bar(x - w / 2, result.n_list, w, color='steelblue', label='折射率')
        ax2.set_xlabel('膜层编号', fontsize=11)
        ax2.set_ylabel('折射率', fontsize=11, color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax2.set_ylim(1.0, 2.8)
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x + w / 2, result.d_list, w, color='salmon', label='厚度 (nm)')
        ax2_twin.set_ylabel('厚度 (nm)', fontsize=11, color='salmon')
        ax2_twin.tick_params(axis='y', labelcolor='salmon')
        l1, la1 = ax2.get_legend_handles_labels()
        l2, la2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(l1 + l2, la1 + la2, loc='upper right', fontsize=9)
        ax2.set_title('膜系参数', fontsize=12)
        ax2.set_xticks(x)
        ax2.grid(True, alpha=0.3, axis='y')

        if result.robustness is not None:
            rob = result.robustness
            perf = rob.performance_distribution

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.hist(perf['avg_T_in']['values'] * 100, bins=30, alpha=0.7, color='green', label='通带透射率')
            ax3.hist(perf['avg_T_out']['values'] * 100, bins=30, alpha=0.7, color='red', label='带外透射率')
            ax3.axvline(self.specs['min_T'] * 100, color='darkgreen', linestyle='--', linewidth=2, label='通带规格')
            ax3.axvline(self.specs['max_out_T'] * 100, color='darkred', linestyle='--', linewidth=2, label='带外规格')
            ax3.set_xlabel('透射率 (%)', fontsize=11)
            ax3.set_ylabel('频数', fontsize=11)
            ax3.set_title(f'性能分布 (成品率={rob.yield_rate*100:.1f}%)', fontsize=12)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')

            ax4 = fig.add_subplot(gs[1, 1])
            layers = np.arange(len(result.n_list))
            colors = ['red' if s > np.mean(rob.sensitivity_per_layer) else 'steelblue'
                      for s in rob.sensitivity_per_layer]
            ax4.bar(layers, rob.sensitivity_per_layer, color=colors, alpha=0.8)
            ax4.axhline(np.mean(rob.sensitivity_per_layer), color='k', linestyle='--', alpha=0.7, label='平均值')
            ax4.set_xlabel('膜层编号', fontsize=11)
            ax4.set_ylabel('归一化灵敏度', fontsize=11)
            ax4.set_title('层灵敏度分析', fontsize=12)
            ax4.set_xticks(layers)
            ax4.set_xticklabels([str(i + 1) for i in layers])
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')

            for i, tol in enumerate(rob.tolerance_recommendation['per_layer']):
                ax4.text(i, rob.sensitivity_per_layer[i] + 0.02, f'±{tol*100:.0f}%',
                        ha='center', fontsize=8, rotation=90)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n图表已保存至 '{save_path}'")
        plt.close()


def design_robust_filter(lambda_center: float, bandwidth: float,
                         n_layers: int = 8, thickness_error: float = 0.02,
                         mc_samples: int = 1500, specs: Dict = None) -> FilterResult:
    """便捷函数：设计鲁棒滤波器并进行完整分析"""
    filter = RobustOpticalFilter()

    if specs is None:
        specs = {
            'min_T': 0.80,
            'max_out_T': 0.10,
            'fwhm_tol': 0.15
        }

    result = filter.design_bandpass(
        lambda_center=lambda_center,
        bandwidth=bandwidth,
        n_layers=n_layers,
        method='hybrid',
        specs=specs,
        verbose=True
    )

    result.robustness = filter.monte_carlo_analysis(
        n_list=result.n_list,
        d_list=result.d_list,
        thickness_error=thickness_error,
        n_samples=mc_samples,
        verbose=True
    )

    filter.plot_comprehensive(result)

    return result


def main():
    print("=" * 70)
    print("  薄膜光学滤波器设计 - 全局优化 + 鲁棒性分析")
    print("  蒙特卡洛模拟 + 灵敏度分析 + 公差设计")
    print("=" * 70)

    specs = {
        'min_T': 0.80,
        'max_out_T': 0.10,
        'fwhm_tol': 0.15
    }

    result = design_robust_filter(
        lambda_center=550.0,
        bandwidth=40.0,
        n_layers=6,
        thickness_error=0.02,
        mc_samples=1000,
        specs=specs
    )

    print("\n" + "=" * 70)
    print("  设计完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
