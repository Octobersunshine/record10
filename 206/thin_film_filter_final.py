import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class ThinFilmFilter:
    """
    薄膜光学滤波器设计类 - 基于传递矩阵法(TMM)
    
    实现多层薄膜的反射率和透射率计算，并通过数值优化
    设计带通、带阻等光学滤波器。
    """

    def __init__(self, n_substrate: float = 1.5, n_incident: float = 1.0):
        """
        初始化滤波器
        
        参数:
            n_substrate: 基底折射率 (默认: 1.5, 玻璃)
            n_incident: 入射介质折射率 (默认: 1.0, 空气)
        """
        self.n_substrate = n_substrate
        self.n_incident = n_incident

    def tmm(self, wavelengths: np.ndarray, n_list: List[float],
            d_list: List[float], theta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        传递矩阵法 (Transfer Matrix Method) 计算多层薄膜的光学特性
        
        原理:
            对于每层薄膜，构建2x2特征矩阵：
            M = [[cos(delta), -i*sin(delta)/eta],
                 [-i*eta*sin(delta), cos(delta)]]
            其中 delta = 2*pi*n*d*cos(theta)/lambda
            eta = n*cos(theta) (TE波)
        
        参数:
            wavelengths: 波长数组 (nm)
            n_list: 每层折射率列表
            d_list: 每层厚度列表 (nm)
            theta: 入射角 (弧度，默认: 0 正入射)
            
        返回:
            R: 反射率数组
            T: 透射率数组
        """
        n_layers = len(n_list)
        if n_layers != len(d_list):
            raise ValueError("折射率列表和厚度列表长度必须相同")

        R = np.zeros_like(wavelengths, dtype=float)
        T = np.zeros_like(wavelengths, dtype=float)

        n0 = self.n_incident
        ns = self.n_substrate
        cos_theta0 = np.cos(theta)

        for idx, lambda0 in enumerate(wavelengths):
            k0 = 2 * np.pi / lambda0
            M = np.eye(2, dtype=complex)

            for n, d in zip(n_list, d_list):
                sin_theta = n0 * np.sin(theta) / n
                cos_theta = np.sqrt(1 - sin_theta ** 2 + 0j)
                delta = k0 * n * d * cos_theta
                eta = n * cos_theta

                cos_d = np.cos(delta)
                sin_d = np.sin(delta)

                layer_matrix = np.array([
                    [cos_d, -1j * sin_d / eta],
                    [-1j * eta * sin_d, cos_d]
                ])
                M = M @ layer_matrix

            sin_theta_s = n0 * np.sin(theta) / ns
            cos_theta_s = np.sqrt(1 - sin_theta_s ** 2 + 0j)
            eta_s = ns * cos_theta_s
            eta0 = n0 * cos_theta0

            A, B = M[0, 0], M[0, 1]
            C, D = M[1, 0], M[1, 1]

            denom = A * eta_s + B * eta0 * eta_s + C + D * eta0
            r = (A * eta_s + B * eta0 * eta_s - C - D * eta0) / denom
            t = 2 * eta_s / denom

            R[idx] = np.real(np.abs(r) ** 2)
            T[idx] = np.real(np.abs(t) ** 2 * eta0 / eta_s)

        return R, T

    def design_bandpass(self, lambda_center: float, bandwidth: float,
                        n_layers: int = 10, max_iter: int = 1000,
                        lambda_min: Optional[float] = None,
                        lambda_max: Optional[float] = None,
                        n_points: int = 300,
                        n_low: float = 1.38,
                        n_high: float = 2.35) -> Tuple[List[float], List[float]]:
        """
        设计带通滤波器
        
        使用L-BFGS-B优化算法最小化损失函数，优化每层的折射率和厚度。
        
        参数:
            lambda_center: 中心波长 (nm)
            bandwidth: 带宽 (nm)
            n_layers: 薄膜层数
            max_iter: 最大迭代次数
            lambda_min: 最小波长 (nm)
            lambda_max: 最大波长 (nm)
            n_points: 波长采样点数
            n_low: 低折射率材料参考值 (如SiO2: 1.38)
            n_high: 高折射率材料参考值 (如TiO2: 2.35)
            
        返回:
            n_list: 优化后的折射率列表
            d_list: 优化后的厚度列表 (nm)
        """
        if lambda_min is None:
            lambda_min = max(300, lambda_center - 2.5 * bandwidth)
        if lambda_max is None:
            lambda_max = lambda_center + 2.5 * bandwidth

        self.lambda_center = lambda_center
        self.bandwidth = bandwidth
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.wavelengths = np.linspace(lambda_min, lambda_max, n_points)

        def target_spectrum(wavelengths, lambda_c, bw):
            sigma = bw / (2 * np.sqrt(2 * np.log(2)))
            return np.exp(-((wavelengths - lambda_c) ** 2) / (2 * sigma ** 2))

        self.target_T = target_spectrum(self.wavelengths, lambda_center, bandwidth)

        quarter_wave = lambda_center / 4.0
        n_initial = []
        d_initial = []
        np.random.seed(42)

        for i in range(n_layers):
            base_n = n_high if i % 2 == 0 else n_low
            perturbation = np.random.uniform(-0.03, 0.03)
            n_initial.append(base_n + perturbation)
            d_initial.append(quarter_wave / n_initial[-1])

        def loss(params):
            n_opt = np.clip(params[:n_layers], 1.3, 2.6)
            d_opt = np.clip(params[n_layers:], 20, 400)

            _, T = self.tmm(self.wavelengths, n_opt.tolist(), d_opt.tolist())

            lambda_low = lambda_center - bandwidth / 2
            lambda_high = lambda_center + bandwidth / 2
            in_band = (self.wavelengths >= lambda_low) & (self.wavelengths <= lambda_high)
            out_band = ~in_band

            in_band_loss = np.mean((1.0 - T[in_band]) ** 2)
            out_band_loss = np.mean(T[out_band] ** 2)
            edge_penalty = 0.0

            if np.sum(in_band) > 2:
                edge_mask = np.zeros_like(self.wavelengths, dtype=bool)
                edge_width = bandwidth * 0.2
                left_edge = (self.wavelengths >= lambda_low - edge_width) & (self.wavelengths <= lambda_low + edge_width)
                right_edge = (self.wavelengths >= lambda_high - edge_width) & (self.wavelengths <= lambda_high + edge_width)
                edge_mask = left_edge | right_edge
                if np.any(edge_mask):
                    edge_penalty = np.mean(T[edge_mask] ** 2)

            return 0.4 * in_band_loss + 0.4 * out_band_loss + 0.2 * edge_penalty

        x0 = np.concatenate([n_initial, d_initial])

        print(f"{'='*60}")
        print(f"Starting optimization:")
        print(f"  Center wavelength: {lambda_center} nm")
        print(f"  Bandwidth: {bandwidth} nm")
        print(f"  Number of layers: {n_layers}")
        print(f"  Initial loss: {loss(x0):.6f}")
        print(f"{'='*60}")

        result = minimize(loss, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter})

        n_opt = np.clip(result.x[:n_layers], 1.3, 2.6).tolist()
        d_opt = np.clip(result.x[n_layers:], 20, 400).tolist()

        print(f"\nOptimization complete:")
        print(f"  Final loss: {result.fun:.6f}")
        print(f"  Iterations: {result.nit}")
        print(f"  Success: {result.success}")

        return n_opt, d_opt

    def evaluate(self, n_list: List[float], d_list: List[float],
                 plot: bool = True, save_path: str = 'filter_result.png') -> dict:
        """
        评估滤波器性能并生成报告
        
        参数:
            n_list: 折射率列表
            d_list: 厚度列表 (nm)
            plot: 是否绘制图表
            save_path: 图表保存路径
            
        返回:
            metrics: 性能指标字典
        """
        R, T = self.tmm(self.wavelengths, n_list, d_list)

        lambda_low = self.lambda_center - self.bandwidth / 2
        lambda_high = self.lambda_center + self.bandwidth / 2
        in_band = (self.wavelengths >= lambda_low) & (self.wavelengths <= lambda_high)
        out_band = ~in_band

        max_T = np.max(T)
        max_T_wavelength = self.wavelengths[np.argmax(T)]
        avg_T_in = np.mean(T[in_band])
        avg_T_out = np.mean(T[out_band])
        min_T_out = np.min(T[out_band])

        T_50 = max_T / 2
        above_50 = np.where(T >= T_50)[0]
        fwhm = self.wavelengths[above_50[-1]] - self.wavelengths[above_50[0]] if len(above_50) > 1 else 0

        ripple = np.max(T[in_band]) - np.min(T[in_band]) if np.sum(in_band) > 1 else 0
        rejection = 10 * np.log10(avg_T_in / (avg_T_out + 1e-12))

        metrics = {
            'max_transmittance': max_T,
            'peak_wavelength': max_T_wavelength,
            'avg_transmittance_in_band': avg_T_in,
            'avg_transmittance_out_band': avg_T_out,
            'min_transmittance_out_band': min_T_out,
            'fwhm': fwhm,
            'ripple': ripple,
            'rejection_ratio_db': rejection,
            'center_wavelength': self.lambda_center,
            'target_bandwidth': self.bandwidth,
            'wavelengths': self.wavelengths,
            'R': R,
            'T': T
        }

        print(f"\n{'='*60}")
        print(f"Filter Performance Metrics:")
        print(f"{'='*60}")
        print(f"  Peak transmittance: {max_T * 100:.2f}% at {max_T_wavelength:.1f} nm")
        print(f"  Average in-band transmittance: {avg_T_in * 100:.2f}%")
        print(f"  Average out-of-band transmittance: {avg_T_out * 100:.2f}%")
        print(f"  Minimum out-of-band transmittance: {min_T_out * 100:.2f}%")
        print(f"  FWHM: {fwhm:.1f} nm (target: {self.bandwidth:.1f} nm)")
        print(f"  In-band ripple: {ripple * 100:.2f}%")
        print(f"  Rejection ratio: {rejection:.1f} dB")
        print(f"{'='*60}")

        if plot:
            self._plot_spectrum(R, T, n_list, d_list, save_path)

        return metrics

    def _plot_spectrum(self, R, T, n_list, d_list, save_path):
        """绘制光谱图和膜系参数"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].plot(self.wavelengths, T * 100, 'b-', linewidth=2, label='Transmittance')
        axes[0].plot(self.wavelengths, self.target_T * 100, 'r--', linewidth=1.5, label='Target')
        axes[0].plot(self.wavelengths, R * 100, 'orange', linewidth=1.5, alpha=0.6, label='Reflectance')
        axes[0].axvline(self.lambda_center, color='k', linestyle=':', alpha=0.7, label='Center λ')
        axes[0].axvspan(self.lambda_center - self.bandwidth / 2,
                        self.lambda_center + self.bandwidth / 2,
                        alpha=0.15, color='green', label='Passband')
        axes[0].set_xlabel('Wavelength (nm)', fontsize=12)
        axes[0].set_ylabel('(%)', fontsize=12)
        axes[0].set_title(f'Bandpass Filter Spectrum (λ₀={self.lambda_center} nm, BW={self.bandwidth} nm)',
                          fontsize=13)
        axes[0].legend(fontsize=10, loc='best')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)

        layer_indices = np.arange(len(n_list))
        bar_width = 0.35

        ax1 = axes[1]
        ax1.bar(layer_indices - bar_width / 2, n_list, bar_width,
                color='steelblue', label='Refractive Index')
        ax1.set_xlabel('Layer Number', fontsize=12)
        ax1.set_ylabel('Refractive Index', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(1.0, 2.8)

        ax2 = ax1.twinx()
        ax2.bar(layer_indices + bar_width / 2, d_list, bar_width,
                color='salmon', label='Thickness (nm)')
        ax2.set_ylabel('Thickness (nm)', fontsize=12, color='salmon')
        ax2.tick_params(axis='y', labelcolor='salmon')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        axes[1].set_title('Thin Film Stack Parameters', fontsize=13)
        axes[1].set_xticks(layer_indices)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to '{save_path}'")
        plt.close()


def design_custom_filter(lambda_center: float, bandwidth: float, n_layers: int = 10,
                         max_iter: int = 800) -> dict:
    """
    便捷函数：设计定制化的带通滤波器
    
    参数:
        lambda_center: 中心波长 (nm)
        bandwidth: 带宽 (nm)
        n_layers: 薄膜层数
        max_iter: 最大迭代次数
        
    返回:
        result: 包含膜系参数和性能指标的字典
    """
    filter = ThinFilmFilter(n_substrate=1.5, n_incident=1.0)

    n_list, d_list = filter.design_bandpass(
        lambda_center=lambda_center,
        bandwidth=bandwidth,
        n_layers=n_layers,
        max_iter=max_iter
    )

    print(f"\nOptimized thin film stack:")
    print(f"{'Layer':<8} {'n':<12} {'d (nm)':<12} {'n*d (nm)':<15}")
    print("-" * 50)
    for i, (n, d) in enumerate(zip(n_list, d_list)):
        print(f"{i + 1:<8} {n:<12.4f} {d:<12.2f} {n * d:<15.1f}")

    metrics = filter.evaluate(n_list, d_list)

    return {
        'n_list': n_list,
        'd_list': d_list,
        'metrics': metrics
    }


def main():
    print("=" * 70)
    print("Thin Film Optical Filter Design using Transfer Matrix Method (TMM)")
    print("=" * 70)

    result = design_custom_filter(
        lambda_center=550.0,
        bandwidth=40.0,
        n_layers=10,
        max_iter=600
    )

    print("\n" + "=" * 70)
    print("Design complete!")
    print("=" * 70)
    print("\nExample usage to design another filter:")
    print("  result = design_custom_filter(lambda_center=850, bandwidth=80, n_layers=12)")
    print("\nOr use the class directly:")
    print("  filter = ThinFilmFilter()")
    print("  n_list, d_list = filter.design_bandpass(633, 50, n_layers=8)")
    print("  metrics = filter.evaluate(n_list, d_list)")


if __name__ == "__main__":
    main()
